#!/usr/bin/env python3
"""Select an idle ROCm render node for the local notebook CI.

The script runs on the self-hosted runner host before the Docker container is
started. It chooses one AMD render node that is currently idle, writes GitHub
Actions outputs when requested, and creates a small CI lock file so concurrent
CI jobs using this selector do not pick the same card.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AMD_VENDOR_ID = "0x1002"
DEFAULT_LOCK_DIR = Path("/tmp/radeon-ci-gpu-locks")


@dataclass
class RenderNode:
    name: str
    path: str
    pci_bus: str | None


@dataclass
class GpuStats:
    card_index: int | None = None
    pci_bus: str | None = None
    vram_used_gb: float | None = None
    gpu_util_pct: float | None = None


@dataclass
class Candidate:
    node: RenderNode
    stats: GpuStats
    pids: list[int]
    lock_file: Path
    locked: bool
    idle: bool
    reason: str


def normalize_bus(value: Any) -> str | None:
    text = str(value or "").strip()
    m = re.search(r"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", text)
    return m.group(1).lower() if m else None


def parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace(",", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def parse_bytes(value: Any) -> int | None:
    num = parse_float(value)
    if num is None:
        return None
    text = str(value or "").lower()
    if "tib" in text:
        return int(num * 1024**4)
    if "tb" in text:
        return int(num * 1000**4)
    if "gib" in text:
        return int(num * 1024**3)
    if "gb" in text:
        return int(num * 1000**3)
    if "mib" in text:
        return int(num * 1024**2)
    if "mb" in text:
        return int(num * 1000**2)
    return int(num)


def card_index_from_key(key: str) -> int | None:
    m = re.search(r"(?:card|gpu)?\s*(\d+)", key.lower())
    return int(m.group(1)) if m else None


def discover_render_nodes(allowed: str = "") -> list[RenderNode]:
    allowed_names = {
        Path(item.strip()).name
        for item in allowed.split(",")
        if item.strip()
    }
    nodes: list[RenderNode] = []
    for entry in sorted(Path("/sys/class/drm").glob("renderD*")):
        if allowed_names and entry.name not in allowed_names:
            continue
        device = entry / "device"
        vendor_path = device / "vendor"
        try:
            vendor = vendor_path.read_text().strip().lower()
        except OSError:
            continue
        if vendor != AMD_VENDOR_ID:
            continue
        pci_bus = normalize_bus(os.path.realpath(device))
        nodes.append(RenderNode(entry.name, f"/dev/dri/{entry.name}", pci_bus))
    return nodes


def run_rocm_smi_json() -> dict[str, Any]:
    cmd = [
        "rocm-smi",
        "--showmeminfo",
        "vram",
        "--showuse",
        "--showbus",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(
            f"rocm-smi failed with exit {result.returncode}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return json.loads(result.stdout)


def parse_rocm_smi(data: dict[str, Any]) -> tuple[dict[str, GpuStats], dict[int, GpuStats]]:
    by_bus: dict[str, GpuStats] = {}
    by_index: dict[int, GpuStats] = {}
    for key, raw in data.items():
        if not isinstance(raw, dict):
            continue
        stats = GpuStats(card_index=card_index_from_key(key))
        for field, value in raw.items():
            field_l = field.lower()
            bus = normalize_bus(value) if ("pci" in field_l or "bdf" in field_l or "bus" in field_l) else None
            if bus:
                stats.pci_bus = bus
            if "vram" in field_l and "used" in field_l:
                used = parse_bytes(value)
                if used is not None:
                    stats.vram_used_gb = round(used / 1e9, 3)
            if (
                "gpu" in field_l
                and ("use" in field_l or "util" in field_l or "busy" in field_l)
                and "memory" not in field_l
                and "vram" not in field_l
            ):
                util = parse_float(value)
                if util is not None:
                    stats.gpu_util_pct = max(0.0, min(100.0, util))
        if stats.pci_bus:
            by_bus[stats.pci_bus] = stats
        if stats.card_index is not None:
            by_index[stats.card_index] = stats
    return by_bus, by_index


def pids_using_node(render_node: str) -> list[int]:
    try:
        result = subprocess.run(
            ["fuser", render_node],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    pids: set[int] = set()
    for token in re.findall(r"\d+", result.stdout):
        try:
            pids.add(int(token))
        except ValueError:
            pass
    return sorted(pids)


def lock_is_stale(lock_file: Path, stale_seconds: int) -> bool:
    if stale_seconds <= 0:
        return False
    try:
        age = time.time() - lock_file.stat().st_mtime
    except OSError:
        return False
    return age > stale_seconds


def lock_file_for(lock_dir: Path, node: RenderNode) -> Path:
    return lock_dir / f"{node.name}.lock"


def try_lock(lock_file: Path, stale_seconds: int, owner: dict[str, Any]) -> bool:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    if lock_file.exists() and lock_is_stale(lock_file, stale_seconds):
        try:
            lock_file.unlink()
        except OSError:
            pass
    try:
        fd = os.open(str(lock_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(owner, f, indent=2)
        f.write("\n")
    return True


def release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


def state_reason(
    stats: GpuStats,
    pids: list[int],
    locked: bool,
    max_used_gb: float,
    max_util_pct: float,
    require_no_pids: bool,
) -> tuple[bool, str]:
    if locked:
        return False, "locked by another CI run"
    if stats.vram_used_gb is None:
        return False, "missing VRAM usage from rocm-smi"
    if stats.vram_used_gb > max_used_gb:
        return False, f"VRAM used {stats.vram_used_gb:.1f} GB > {max_used_gb:.1f} GB"
    if stats.gpu_util_pct is not None and stats.gpu_util_pct > max_util_pct:
        return False, f"GPU util {stats.gpu_util_pct:.0f}% > {max_util_pct:.0f}%"
    if require_no_pids and pids:
        return False, f"render node has process(es): {','.join(map(str, pids))}"
    return True, "idle"


def collect_candidates(
    nodes: list[RenderNode],
    by_bus: dict[str, GpuStats],
    by_index: dict[int, GpuStats],
    args: argparse.Namespace,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for fallback_index, node in enumerate(nodes):
        stats = by_bus.get(node.pci_bus or "") or by_index.get(fallback_index) or GpuStats()
        lock_file = lock_file_for(Path(args.lock_dir), node)
        locked = lock_file.exists() and not lock_is_stale(lock_file, args.stale_lock_seconds)
        pids = pids_using_node(node.path)
        idle, reason = state_reason(
            stats,
            pids,
            locked,
            args.max_used_gb,
            args.max_util_pct,
            args.require_no_pids,
        )
        candidates.append(Candidate(node, stats, pids, lock_file, locked, idle, reason))
    return candidates


def candidate_sort_key(candidate: Candidate) -> tuple[float, float, int]:
    used = candidate.stats.vram_used_gb if candidate.stats.vram_used_gb is not None else float("inf")
    util = candidate.stats.gpu_util_pct if candidate.stats.gpu_util_pct is not None else 0.0
    card = candidate.stats.card_index if candidate.stats.card_index is not None else 999
    return (used, util, card)


def print_candidates(candidates: list[Candidate]) -> None:
    print("Detected Radeon render nodes:")
    for c in candidates:
        stats = c.stats
        used = "n/a" if stats.vram_used_gb is None else f"{stats.vram_used_gb:.1f} GB"
        util = "n/a" if stats.gpu_util_pct is None else f"{stats.gpu_util_pct:.0f}%"
        pids = "-" if not c.pids else ",".join(map(str, c.pids))
        card = "?" if stats.card_index is None else str(stats.card_index)
        print(
            f"- card={card} render={c.node.path} pci={c.node.pci_bus or '?'} "
            f"used={used} util={util} pids={pids} idle={c.idle} reason={c.reason}",
            flush=True,
        )


def write_github_output(path: str, selected: Candidate) -> None:
    if not path:
        return
    stats = selected.stats
    values = {
        "render_node": selected.node.path,
        "render_node_name": selected.node.name,
        "card_index": "" if stats.card_index is None else str(stats.card_index),
        "pci_bus": selected.node.pci_bus or stats.pci_bus or "",
        "vram_used_gb": "" if stats.vram_used_gb is None else f"{stats.vram_used_gb:.3f}",
        "gpu_util_pct": "" if stats.gpu_util_pct is None else f"{stats.gpu_util_pct:.1f}",
        "lock_file": str(selected.lock_file),
    }
    with open(path, "a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def select_once(args: argparse.Namespace, owner: dict[str, Any]) -> Candidate | None:
    nodes = discover_render_nodes(args.allowed_render_nodes)
    if not nodes:
        allowed = f" from {args.allowed_render_nodes}" if args.allowed_render_nodes else ""
        raise RuntimeError(f"no AMD render nodes found{allowed}")
    by_bus, by_index = parse_rocm_smi(run_rocm_smi_json())
    candidates = collect_candidates(nodes, by_bus, by_index, args)
    print_candidates(candidates)
    for candidate in sorted((c for c in candidates if c.idle), key=candidate_sort_key):
        if args.no_lock or try_lock(candidate.lock_file, args.stale_lock_seconds, owner):
            return candidate
        candidate.locked = True
        candidate.idle = False
        candidate.reason = "lost lock race"
    return None


def run_selector(args: argparse.Namespace) -> int:
    owner = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "github_job": os.environ.get("GITHUB_JOB", ""),
        "render_selector": "tools/select_idle_rocm_gpu.py",
    }
    deadline = time.time() + args.wait_seconds
    attempt = 0
    last_error = ""
    while True:
        attempt += 1
        print(f"Selecting idle Radeon GPU (attempt {attempt})...", flush=True)
        try:
            selected = select_once(args, owner)
        except Exception as exc:
            selected = None
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[select-gpu] {last_error}", flush=True)
        if selected is not None:
            print(
                f"Selected Radeon GPU: render={selected.node.path} "
                f"card={selected.stats.card_index if selected.stats.card_index is not None else '?'} "
                f"pci={selected.node.pci_bus or selected.stats.pci_bus or '?'} "
                f"vram_used={selected.stats.vram_used_gb}GB "
                f"util={selected.stats.gpu_util_pct}%",
                flush=True,
            )
            write_github_output(args.github_output, selected)
            return 0
        if time.time() >= deadline or args.wait_seconds <= 0:
            print("No idle Radeon GPU available.", flush=True)
            if last_error:
                print(f"Last error: {last_error}", flush=True)
            return 1
        sleep_s = min(args.poll_seconds, max(1, int(deadline - time.time())))
        print(f"No idle GPU yet; retrying in {sleep_s}s.", flush=True)
        time.sleep(sleep_s)


def self_test() -> int:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as td:
        args = parser.parse_args([
            "--lock-dir",
            td,
            "--max-used-gb",
            "2",
            "--max-util-pct",
            "5",
            "--no-lock",
        ])
        nodes = [
            RenderNode("renderD128", "/dev/dri/renderD128", "0000:01:00.0"),
            RenderNode("renderD129", "/dev/dri/renderD129", "0000:02:00.0"),
            RenderNode("renderD130", "/dev/dri/renderD130", "0000:03:00.0"),
        ]
        by_bus = {
            "0000:01:00.0": GpuStats(0, "0000:01:00.0", 14.2, 0.0),
            "0000:02:00.0": GpuStats(1, "0000:02:00.0", 0.4, 55.0),
            "0000:03:00.0": GpuStats(2, "0000:03:00.0", 0.3, 0.0),
        }
        candidates = collect_candidates(nodes, by_bus, {}, args)
        idle = sorted((c for c in candidates if c.idle), key=candidate_sort_key)
        if not idle or idle[0].node.name != "renderD130":
            print("self-test failed: expected renderD130", file=sys.stderr)
            return 1
        print_candidates(candidates)
    print("self-test passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-used-gb", type=float, default=2.0)
    ap.add_argument("--max-util-pct", type=float, default=5.0)
    ap.add_argument("--wait-seconds", type=int, default=3600)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--allowed-render-nodes", default=os.environ.get("RADEON_CI_RENDER_NODES", ""))
    ap.add_argument("--lock-dir", default=str(DEFAULT_LOCK_DIR))
    ap.add_argument("--stale-lock-seconds", type=int, default=72 * 3600)
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    ap.add_argument("--require-no-pids", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return self_test()
    return run_selector(args)


if __name__ == "__main__":
    raise SystemExit(main())
