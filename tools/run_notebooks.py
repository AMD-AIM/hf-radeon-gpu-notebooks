#!/usr/bin/env python3
"""Run Hugging Face Oneclick notebooks for the local W7900 CI."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
TARGET_CSV = REPO / "doc" / "ci_target_models.csv"
ORIGINAL_NOTEBOOK_DIR = REPO / "original_notebooks"
SOURCE_DATA_DIR = REPO / "source_data"
SOURCE_MAP_JSON = SOURCE_DATA_DIR / "resource_map.json"
DEFAULT_ENV_FILE = os.environ.get(
    "RADEON_CI_ENV_FILE", "/disk/ssd1/zihaomu_amd/ci_secrets.env"
)
HF_CANONICAL = "https://huggingface.co"
VRAM_BUDGET_GB = float(os.environ.get("RADEON_CI_VRAM_PER_GPU_GB", "48"))
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SECRET_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
LOADED_ENV: dict[str, str] = {}
SECRET_VALUES: set[str] = set()


GPU_PREAMBLE = '''# [ci-normalize] prefer GPU placement for HF helpers.
import functools as _ci_functools
import os as _ci_os
import urllib.request as _ci_urllib_request
import transformers as _ci_transformers

_ci_original_pipeline = _ci_transformers.pipeline
def _ci_pipeline(*args, **kwargs):
    kwargs.setdefault("device_map", "auto")
    return _ci_original_pipeline(*args, **kwargs)
_ci_transformers.pipeline = _ci_pipeline

for _ci_name in (
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSpeechSeq2Seq",
    "AutoModelForTokenClassification",
):
    _ci_cls = getattr(_ci_transformers, _ci_name, None)
    if _ci_cls is None or not hasattr(_ci_cls, "from_pretrained"):
        continue
    _ci_original = _ci_cls.from_pretrained.__func__
    def _ci_make_from_pretrained(original):
        @classmethod
        @_ci_functools.wraps(original)
        def _ci_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("device_map", "auto")
            kwargs.setdefault("torch_dtype", "auto")
            return original(cls, *args, **kwargs)
        return _ci_from_pretrained
    _ci_cls.from_pretrained = _ci_make_from_pretrained(_ci_original)

_ci_original_urlopen = _ci_urllib_request.urlopen
def _ci_urlopen(url, *args, **kwargs):
    candidate = getattr(url, "full_url", url)
    if isinstance(candidate, str) and _ci_os.path.isfile(candidate):
        return open(candidate, "rb")
    return _ci_original_urlopen(url, *args, **kwargs)
_ci_urllib_request.urlopen = _ci_urlopen

try:
    import requests as _ci_requests
    _ci_original_requests_get = _ci_requests.get
    class _ci_local_file_response:
        def __init__(self, path):
            self.path = path
            self.status_code = 200
            self.ok = True
            with open(path, "rb") as _ci_f:
                self.content = _ci_f.read()
        def raise_for_status(self):
            return None
    def _ci_requests_get(url, *args, **kwargs):
        if isinstance(url, str) and _ci_os.path.isfile(url):
            return _ci_local_file_response(url)
        return _ci_original_requests_get(url, *args, **kwargs)
    _ci_requests.get = _ci_requests_get
except Exception:
    pass

print("[ci-normalize] GPU placement defaults applied")
'''

PIP_INSTALL_RE = re.compile(
    r"(?is)(?:^|\b|[%!])(?:python3?\s+-m\s+)?pip\s+install\b|['\"]pip\s+install\b"
)
REMOTE_ONLY_RE = re.compile(
    r"router\.huggingface\.co|from openai import|OpenAI\(|"
    r"from google\.colab|InferenceClient"
)
REMOTE_INFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*##\s+Remote Inference via Inference Providers\b"
)
MARKDOWN_SECTION_HEADING_RE = re.compile(r"(?im)^\s*##\s+")
LOGIN_CALL_RE = re.compile(r"\blogin\s*\(")
ENV_ASSIGN_RE = re.compile(
    r"""^\s*(?:os\.)?environ\s*\[\s*['"]"""
    r"""(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN|HF_ENDPOINT|"""
    r"""HF_HUB_DISABLE_XET|HTTP_PROXY|"""
    r"""HTTPS_PROXY|http_proxy|https_proxy)['"]\s*\]\s*="""
)
HF_TOKEN_PLACEHOLDER_RE = re.compile(
    r"""(?m)^(\s*(?:os\.)?environ\s*\[\s*['"]HF_TOKEN['"]\s*\]\s*=\s*)"""
    r"""['"]YOUR_TOKEN_HERE['"]\s*$"""
)
GRANITE_SAMPLE_RE = re.compile(
    r"""hf_hub_download\(\s*repo_id\s*=\s*model_path\s*,\s*"""
    r"""filename\s*=\s*['"]multilingual_sample\.wav['"]\s*\)"""
)


@dataclass(frozen=True)
class Target:
    model_id: str
    notebook: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


RUN_STARTED_AT = time.time()
RUN_STARTED_UTC = utc_now()


def load_source_data_rewrites() -> dict[str, str]:
    if not SOURCE_MAP_JSON.is_file():
        return {}
    try:
        raw_map = json.loads(SOURCE_MAP_JSON.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[source-data] could not load {SOURCE_MAP_JSON}: {exc}", flush=True)
        return {}

    rewrites: dict[str, str] = {}
    for url, filename in raw_map.items():
        local_path = SOURCE_DATA_DIR / str(filename)
        if local_path.is_file():
            rewrites[str(url)] = str(local_path.resolve())
        else:
            print(f"[source-data] missing local asset for {url}: {local_path}", flush=True)
    return rewrites


SOURCE_DATA_REWRITES = load_source_data_rewrites()


def mask_env(name: str, value: str) -> str:
    return "******" if any(h in name.upper() for h in SECRET_HINTS) else value


def redact_secrets(text: str) -> str:
    redacted = str(text)
    for secret in SECRET_VALUES:
        if secret and len(secret) >= 8:
            redacted = redacted.replace(secret, "******")
    return redacted


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.is_file():
        print(f"[env] no host env file at {env_path}", flush=True)
        return

    try:
        env_lines = env_path.read_text().splitlines()
    except OSError as exc:
        print(f"[env] could not read {env_path}: {type(exc).__name__}: {exc}", flush=True)
        return

    loaded: dict[str, str] = {}
    for line in env_lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            LOADED_ENV[key] = value
            if any(h in key.upper() for h in SECRET_HINTS):
                SECRET_VALUES.add(value)
            if key not in TOKEN_ENV_KEYS:
                os.environ[key] = value
            loaded[key] = value

    shown = ", ".join(f"{k}={mask_env(k, v)}" for k, v in sorted(loaded.items()))
    print(f"[env] loaded {len(loaded)} var(s) from {env_path}: {shown}", flush=True)


def load_targets(path: Path, text_filter: str = "") -> list[Target]:
    targets: list[Target] = []
    needle = text_filter.lower().strip()
    with path.open(newline="") as f:
        for index, row in enumerate(csv.DictReader(f), 2):
            enabled = (row.get("enabled") or "yes").strip().lower()
            if enabled in {"0", "false", "no", "n"}:
                continue

            model_id = (row.get("model_id") or "").strip()
            notebook = (row.get("notebook") or "").strip()
            if not model_id or not notebook:
                raise ValueError(f"{path}:{index}: model_id and notebook are required")
            if needle and needle not in f"{model_id} {notebook}".lower():
                continue
            targets.append(Target(model_id=model_id, notebook=notebook))

    return targets


def hf_endpoint_url(path: str) -> str | None:
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        return None
    return f"{endpoint}/{path.lstrip('/')}"


def native_notebook_urls(model_id: str) -> list[str]:
    canonical = f"{HF_CANONICAL}/{model_id}.ipynb"
    urls = []
    endpoint = hf_endpoint_url(f"{model_id}.ipynb")
    if endpoint:
        urls.append(endpoint)
    urls.append(canonical)
    return list(dict.fromkeys(urls))


def read_text_url(url: str, timeout: int = 120) -> str:
    headers = {"User-Agent": "huggingface-oneclick-notebook-ci"}
    token = runtime_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_notebook_json(raw_text: str, source: str) -> dict[str, Any]:
    notebook = json.loads(raw_text)
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{source} is not a valid notebook JSON document")
    return notebook


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def canonical_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    canonical = copy.deepcopy(notebook)
    for cell in canonical.get("cells", []):
        source = cell.get("source")
        if isinstance(source, list):
            cell["source"] = "".join(str(item) for item in source)
    return canonical


def notebook_has_effective_update(snapshot_path: Path, downloaded: dict[str, Any]) -> bool:
    if not snapshot_path.is_file():
        return True

    try:
        current = parse_notebook_json(snapshot_path.read_text(), str(snapshot_path))
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return True

    return canonical_notebook(current) != canonical_notebook(downloaded)


def write_original_notebook_snapshot(
    target: Target,
    raw_text: str,
    notebook: dict[str, Any],
) -> str:
    ORIGINAL_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = ORIGINAL_NOTEBOOK_DIR / target.notebook
    if not notebook_has_effective_update(snapshot_path, notebook):
        return display_path(snapshot_path)

    snapshot_path.write_text(raw_text)
    print(
        f"[oneclick-sync] updated {display_path(snapshot_path)} from downloaded notebook",
        flush=True,
    )
    return display_path(snapshot_path)


def load_notebook(
    target: Target,
    sync_native_snapshot: bool = True,
) -> tuple[dict[str, Any], dict[str, str | None]]:
    errors: list[str] = []
    for url in native_notebook_urls(target.model_id):
        for attempt in range(1, 4):
            try:
                raw_text = read_text_url(url)
                notebook = parse_notebook_json(raw_text, url)
                snapshot = (
                    write_original_notebook_snapshot(target, raw_text, notebook)
                    if sync_native_snapshot
                    else display_path(ORIGINAL_NOTEBOOK_DIR / target.notebook)
                )
                return notebook, {
                    "source": f"{HF_CANONICAL}/{target.model_id}.ipynb",
                    "fetched_url": url,
                    "snapshot": snapshot,
                }
            except (
                OSError,
                TimeoutError,
                ValueError,
                urllib.error.URLError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as exc:
                errors.append(f"{url} attempt {attempt}: {type(exc).__name__}: {exc}")
                time.sleep(min(attempt * 2, 5))

    fallback_path = ORIGINAL_NOTEBOOK_DIR / target.notebook
    if fallback_path.is_file():
        try:
            notebook = parse_notebook_json(fallback_path.read_text(), str(fallback_path))
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(
                f"{display_path(fallback_path)} fallback: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"[oneclick-fallback] using {display_path(fallback_path)} after "
                "download failure",
                flush=True,
            )
            return notebook, {
                "source": display_path(fallback_path),
                "fetched_url": None,
                "snapshot": display_path(fallback_path),
            }

    raise RuntimeError("could not download Hugging Face notebook: " + " | ".join(errors))


def assert_jpeg_support() -> None:
    try:
        import torch
        import torchvision
        import torchvision.io as tvio
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"required CI dependency is missing ({exc.name}); run inside "
            "huaggingface_for_amd_radeon:latest"
        ) from exc

    try:
        jpg = Path(tempfile.gettempdir()) / "radeon_ci_jpeg_smoke.jpg"
        Image.new("RGB", (8, 8), color=(32, 96, 160)).save(jpg, format="JPEG")
        encoded = tvio.read_file(str(jpg))
        checks = {
            "decode_image": tvio.decode_image(encoded),
            "decode_jpeg": tvio.decode_jpeg(encoded),
            "read_image": tvio.read_image(str(jpg)),
        }
        for name, tensor in checks.items():
            if tuple(getattr(tensor, "shape", ())) != (3, 8, 8):
                raise RuntimeError(f"{name} returned unexpected shape {tuple(tensor.shape)}")
    except Exception as exc:
        raise RuntimeError(
            "native torchvision JPEG decode failed; this CI no longer injects "
            "a PIL fallback, so the image must provide libjpeg support"
        ) from exc

    print(
        "[preflight] native torchvision JPEG decode OK "
        f"(torch={torch.__version__}, torchvision={torchvision.__version__})",
        flush=True,
    )


def comment_cell(source: str, reason: str) -> str:
    return "\n".join(f"# [ci-skip {reason}] {line}" for line in source.splitlines())


def protect_env_assignments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if (
            ENV_ASSIGN_RE.match(line)
            and "[ci-injected-hf-token]" not in line
        ):
            line = "# [ci-protect env] " + line
        lines.append(line)
    return "\n".join(lines)


def runtime_hf_token() -> str:
    for key in TOKEN_ENV_KEYS:
        token = LOADED_ENV.get(key) or os.environ.get(key, "")
        if token and token != "YOUR_TOKEN_HERE":
            SECRET_VALUES.add(token)
            return token
    return ""


def should_skip_remote_or_auth(source: str) -> bool:
    if REMOTE_ONLY_RE.search(source):
        return True
    if HF_TOKEN_PLACEHOLDER_RE.search(source):
        return True
    if "huggingface_hub import login" in source or LOGIN_CALL_RE.search(source):
        return True
    return False


def remote_section_decision(cell: dict[str, Any], in_remote_section: bool) -> tuple[bool, bool]:
    if cell.get("cell_type") != "markdown":
        return in_remote_section, in_remote_section

    source = "".join(cell.get("source", []))
    if REMOTE_INFERENCE_HEADING_RE.search(source):
        return True, True
    if in_remote_section and MARKDOWN_SECTION_HEADING_RE.search(source):
        return False, False
    return in_remote_section, in_remote_section


def rewrite_hf_urls_to_endpoint(source: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        return source
    return source.replace(HF_CANONICAL, endpoint)


def rewrite_source_data_urls(source: str) -> str:
    for url, local_path in SOURCE_DATA_REWRITES.items():
        source = source.replace(url, local_path)

    granite_sample = SOURCE_DATA_DIR / "multilingual_sample.wav"
    if granite_sample.is_file():
        source = GRANITE_SAMPLE_RE.sub(repr(str(granite_sample.resolve())), source)
    return source


def normalize_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(notebook)
    in_remote_section = False
    cells: list[dict[str, Any]] = [
        {
            "cell_type": "code",
            "metadata": {"ci_preamble": True},
            "execution_count": None,
            "outputs": [],
            "source": GPU_PREAMBLE,
        }
    ]

    for original_index, cell in enumerate(normalized.get("cells", [])):
        drop_cell, in_remote_section = remote_section_decision(cell, in_remote_section)
        if drop_cell:
            continue

        if cell.get("cell_type") != "code":
            cells.append(cell)
            continue

        source = "".join(cell.get("source", []))
        if should_skip_remote_or_auth(source):
            source = comment_cell(source, "remote-or-auth")
        elif PIP_INSTALL_RE.search(source):
            source = comment_cell(source, "pip-cell")
        else:
            source = protect_env_assignments(source)
            source = rewrite_source_data_urls(source)
            source = rewrite_hf_urls_to_endpoint(source)

        clean_cell = dict(cell)
        metadata = dict(clean_cell.get("metadata") or {})
        metadata["ci_original_cell_index"] = original_index
        clean_cell["metadata"] = metadata
        clean_cell["source"] = source
        clean_cell["outputs"] = []
        clean_cell["execution_count"] = None
        cells.append(clean_cell)

    normalized["cells"] = cells
    normalized.get("metadata", {}).pop("kernelspec", None)
    return normalized


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def compact_error(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    text = " ".join(strip_ansi(str(text)).replace("\n", " ").replace("\r", " ").split())
    for marker in (
        " See documentation",
        " If reserved",
        " You can update",
        " If this does not work",
        " This could be because",
    ):
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos].rstrip()
    text = text.replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def metric_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group(0)) if match else None


def metric_bytes(value: Any) -> int | None:
    number = metric_number(value)
    if number is None:
        return None
    text = str(value or "").lower()
    if "tib" in text:
        return int(number * 1024**4)
    if "tb" in text:
        return int(number * 1000**4)
    if "gib" in text:
        return int(number * 1024**3)
    if "gb" in text:
        return int(number * 1000**3)
    if "mib" in text:
        return int(number * 1024**2)
    if "mb" in text:
        return int(number * 1000**2)
    return int(number)


def sample_gpu() -> tuple[int, float | None]:
    result = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    data = json.loads(result.stdout)
    used_bytes = 0
    util_values: list[float] = []
    for card in data.values():
        if not isinstance(card, dict):
            continue
        for key, value in card.items():
            lowered = key.lower()
            if "used" in lowered and "vram" in lowered:
                parsed = metric_bytes(value)
                if parsed is not None:
                    used_bytes += parsed
            if (
                "gpu" in lowered
                and ("use" in lowered or "util" in lowered or "busy" in lowered)
                and "memory" not in lowered
                and "vram" not in lowered
            ):
                util = metric_number(value)
                if util is not None:
                    util_values.append(max(0.0, min(100.0, util)))
    return used_bytes, (max(util_values) if util_values else None)


def poll_gpu(stop: threading.Event, stats: dict[str, Any]) -> None:
    while not stop.is_set():
        try:
            used, util = sample_gpu()
            stats["vram_peak_bytes"] = max(stats["vram_peak_bytes"], used)
            if util is not None:
                stats["gpu_util_peak_pct"] = max(stats["gpu_util_peak_pct"], util)
                stats["gpu_util_sum_pct"] += util
                stats["gpu_util_samples"] += 1
        except Exception:
            pass
        time.sleep(2)


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def format_duration(seconds: Any) -> str:
    value = int(round(float(seconds or 0)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def core_error(report: dict[str, Any]) -> str:
    if report.get("run_error"):
        return compact_error(report["run_error"])
    for cell in report.get("cells", []):
        if cell.get("status") != "PASSED" and cell.get("error"):
            return compact_error(cell["error"])
    return ""


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_progress(
    results_dir: Path,
    reports: list[dict[str, Any]],
    pending: list[str],
    running: str | None = None,
) -> None:
    passed = sum(r["overall_status"] == "PASSED" for r in reports)
    failed = sum(r["overall_status"] == "FAILED" for r in reports)
    errored = sum(r["overall_status"] == "ERROR" for r in reports)
    lines = [
        "# Hugging Face Oneclick Notebook CI Progress",
        "",
        f"_updated {utc_now()}_",
        "",
        f"done {len(reports)} · PASS {passed} · FAIL {failed} · ERROR {errored} · pending {len(pending)}",
        "",
    ]
    if running:
        lines += [f"Running: `{running}`", ""]

    if reports:
        lines += [
            "| # | Mode | Status | Model | Peak VRAM | GPU util avg/peak | Time | Log |",
            "|--:|:----:|:------:|:------|:---------:|:-----------------:|-----:|:----|",
        ]
        for index, report in enumerate(reports, 1):
            lines.append(
                f"| {index} | {report['mode']} | {report['overall_status']} | "
                f"`{report['model_id']}` | {report['vram_peak_gb']} GB | "
                f"{format_pct(report.get('gpu_util_avg_pct'))}/"
                f"{format_pct(report.get('gpu_util_peak_pct'))} | "
                f"{report['elapsed_seconds']:.0f}s | {report['log_file']} |"
            )

    if pending:
        lines += ["", "Pending:", ""]
        lines += [f"- `{name}`" for name in pending[:50]]
        if len(pending) > 50:
            lines.append(f"- ... and {len(pending) - 50} more")

    (results_dir / "progress.md").write_text("\n".join(lines) + "\n")


def write_summary(
    results_dir: Path,
    reports: list[dict[str, Any]],
    policy: str,
) -> None:
    passed = sum(r["overall_status"] == "PASSED" for r in reports)
    failed = sum(r["overall_status"] == "FAILED" for r in reports)
    errored = sum(r["overall_status"] == "ERROR" for r in reports)
    total = len(reports)
    rate = (100.0 * passed / total) if total else 0.0
    icon = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERR"}
    total_elapsed = sum(float(r.get("elapsed_seconds") or 0.0) for r in reports)
    wall_elapsed = time.time() - RUN_STARTED_AT

    def timing_row(name: str, rows: list[dict[str, Any]]) -> str:
        elapsed_values = [float(r.get("elapsed_seconds") or 0.0) for r in rows]
        elapsed_total = sum(elapsed_values)
        average = elapsed_total / len(elapsed_values) if elapsed_values else 0.0
        shortest = min(elapsed_values) if elapsed_values else 0.0
        longest = max(elapsed_values) if elapsed_values else 0.0
        return (
            f"| {name} | {len(rows)} | {format_duration(elapsed_total)} | "
            f"{format_duration(average)} | {format_duration(shortest)} | "
            f"{format_duration(longest)} |"
        )

    lines = [
        "# Hugging Face Oneclick Notebook CI - Results",
        "",
        f"**{total} notebook jobs · {passed} PASS · {failed} FAIL · {errored} ERROR · "
        f"{rate:.1f}% pass · CI wall time {format_duration(wall_elapsed)} · "
        f"notebook time {format_duration(total_elapsed)}**",
        "",
        f"Started: {RUN_STARTED_UTC}",
        f"Generated: {utc_now()}",
        "",
        f"Policy: {policy}",
        "",
        "## Timing",
        "",
        "| Scope | Jobs | Total Time | Avg / Job | Fastest | Slowest |",
        "|:------|-----:|-----------:|----------:|--------:|--------:|",
        timing_row("All", reports),
        timing_row("HF Oneclick Notebooks", reports),
    ]

    if reports:
        lines += [
            "",
            "## HF Oneclick Notebooks",
            "",
            f"**{len(reports)} job(s) - "
            f"{sum(r['overall_status'] == 'PASSED' for r in reports)} PASS / "
            f"{sum(r['overall_status'] == 'FAILED' for r in reports)} FAIL / "
            f"{sum(r['overall_status'] == 'ERROR' for r in reports)} ERROR**",
            "",
            "| # | Status | Model | Cells P/F/T | Peak VRAM | GPU util avg/peak | Time | Core error |",
            "|--:|:------:|:------|:-----------:|:---------:|:-----------------:|-----:|:-----------|",
        ]
        for index, report in enumerate(reports, 1):
            peak = report["vram_peak_gb"]
            vram = f"{peak} / {VRAM_BUDGET_GB:.0f} GB ({100.0 * peak / VRAM_BUDGET_GB:.0f}%)"
            lines.append(
                f"| {index} | {icon[report['overall_status']]} | "
                f"`{report['model_id']}` | "
                f"{report['cells_passed']}/{report['cells_failed']}/{report['cells_total']} | "
                f"{vram} | "
                f"{format_pct(report.get('gpu_util_avg_pct'))}/"
                f"{format_pct(report.get('gpu_util_peak_pct'))} | "
                f"{format_duration(report['elapsed_seconds'])} | {core_error(report)} |"
            )

    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {passed}/{total} passed ({rate:.1f}%)", flush=True)


def build_artifact_notebook(original: dict[str, Any], executed: dict[str, Any]) -> dict[str, Any]:
    artifact = copy.deepcopy(original)
    executed_by_original_index: dict[int, dict[str, Any]] = {}
    fallback_cells: list[dict[str, Any]] = []
    for cell in executed.get("cells", []):
        if cell.get("cell_type") != "code" or cell.get("metadata", {}).get("ci_preamble"):
            continue
        original_index = cell.get("metadata", {}).get("ci_original_cell_index")
        if isinstance(original_index, int):
            executed_by_original_index[original_index] = cell
        else:
            fallback_cells.append(cell)

    artifact_cells: list[dict[str, Any]] = []
    in_remote_section = False
    fallback_index = 0
    for original_index, cell in enumerate(artifact.get("cells", [])):
        drop_cell, in_remote_section = remote_section_decision(cell, in_remote_section)
        if drop_cell:
            continue

        if cell.get("cell_type") != "code":
            artifact_cells.append(cell)
            continue

        executed_cell = executed_by_original_index.get(original_index)
        if executed_cell is None and fallback_index < len(fallback_cells):
            executed_cell = fallback_cells[fallback_index]
            fallback_index += 1

        cell["outputs"] = []
        cell["execution_count"] = None
        if executed_cell is not None:
            cell["outputs"] = executed_cell.get("outputs", [])
            cell["execution_count"] = executed_cell.get("execution_count")
        artifact_cells.append(cell)

    artifact["cells"] = artifact_cells
    return artifact


def run_one(
    target: Target,
    args: argparse.Namespace,
    results_dir: Path,
) -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient

    mode = "oneclick"
    artifact_name = f"{mode}__{target.notebook}"
    log_path = results_dir / artifact_name.replace(".ipynb", ".log")

    def emit(handle: Any, line: str, stdout: bool = False) -> None:
        line = redact_secrets(line)
        handle.write(line + "\n")
        if stdout:
            print(line, flush=True)

    with log_path.open("w", buffering=1) as log:
        started = time.time()
        source_info: dict[str, str | None] = {"source": None, "fetched_url": None}
        try:
            original, source_info = load_notebook(target)
            normalized = normalize_notebook(original)
            notebook = nbformat.from_dict(normalized)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            emit(log, f"# {artifact_name} ({target.model_id}) mode={mode}")
            emit(log, f"# started {utc_now()}")
            emit(log, f"[SOURCE-ERROR] {error}")
            print(f"[SOURCE-ERROR] {mode} {target.model_id}: {compact_error(error, 300)}", flush=True)
            report = make_report(target, mode, artifact_name, source_info, 0.0, {}, error, [])
            save_json(results_dir / artifact_name.replace(".ipynb", ".json"), report)
            return report

        emit(log, f"# {artifact_name} ({target.model_id}) mode={mode}")
        emit(log, f"# started {utc_now()}")
        emit(log, f"# source={source_info.get('source')}")
        emit(log, f"# fetched_url={source_info.get('fetched_url')}")
        emit(log, "# visible_gpus=0\n")

        def on_cell_start(cell: dict[str, Any], cell_index: int, **_: Any) -> None:
            if cell.get("metadata", {}).get("ci_preamble"):
                return
            emit(log, f"\n----- cell {cell_index} -----")
            emit(log, "".join(cell.get("source", [])).rstrip())
            emit(log, "----- output -----")

        def on_cell_executed(cell: dict[str, Any], cell_index: int, **_: Any) -> None:
            if cell.get("metadata", {}).get("ci_preamble"):
                return
            for output in cell.get("outputs", []):
                kind = output.get("output_type")
                if kind == "stream":
                    emit(log, "".join(output.get("text", "")).rstrip(), args.echo_output)
                elif kind == "execute_result":
                    text = "".join(output.get("data", {}).get("text/plain", "")).rstrip()
                    emit(log, text, args.echo_output)
                elif kind == "error":
                    core = compact_error(f"{output.get('ename')}: {output.get('evalue')}", 260)
                    emit(log, f"[ERROR] cell {cell_index}: {core}", True)
                    for traceback_line in output.get("traceback", []):
                        emit(log, strip_ansi(traceback_line).rstrip(), args.echo_traceback)

        gpu_stats = {
            "vram_peak_bytes": 0,
            "gpu_util_sum_pct": 0.0,
            "gpu_util_peak_pct": 0.0,
            "gpu_util_samples": 0,
        }
        stop = threading.Event()
        poller = threading.Thread(target=poll_gpu, args=(stop, gpu_stats), daemon=True)
        poller.start()

        saved_gpu_env = {
            key: os.environ.get(key) for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        }
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        run_error = None
        exec_error: dict[str, str] = {}
        client = NotebookClient(
            notebook,
            timeout=args.cell_timeout,
            allow_errors=True,
            kernel_name="python3",
            on_cell_start=on_cell_start,
            on_cell_executed=on_cell_executed,
        )

        def execute() -> None:
            try:
                client.execute()
            except Exception as exc:
                exec_error["message"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=execute, daemon=True)
        worker.start()
        worker.join(args.notebook_timeout)
        if worker.is_alive():
            run_error = f"notebook timeout > {args.notebook_timeout}s"
            emit(log, f"[TIMEOUT] {run_error}", True)
            try:
                if getattr(client, "km", None):
                    client.km.shutdown_kernel(now=True)
            except Exception:
                pass
            worker.join(10)
        elif exec_error:
            run_error = exec_error["message"]
            emit(log, f"[KERNEL-ERROR] {run_error}", True)

        elapsed = round(time.time() - started, 1)
        stop.set()
        poller.join(timeout=5)
        for key, old_value in saved_gpu_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

        cells, passed, failed = collect_cell_results(notebook)
        overall = "ERROR" if run_error else ("PASSED" if failed == 0 else "FAILED")
        gpu_avg = None
        if gpu_stats["gpu_util_samples"]:
            gpu_avg = round(gpu_stats["gpu_util_sum_pct"] / gpu_stats["gpu_util_samples"], 1)
        gpu_peak = (
            round(gpu_stats["gpu_util_peak_pct"], 1)
            if gpu_stats["gpu_util_samples"]
            else None
        )
        vram_peak = round(gpu_stats["vram_peak_bytes"] / 1e9, 1)

        emit(
            log,
            f"\n# RESULT {overall} passed={passed} failed={failed} elapsed={elapsed}s "
            f"peak_vram={vram_peak}GB gpu_util_avg={format_pct(gpu_avg)} "
            f"gpu_util_peak={format_pct(gpu_peak)}",
        )

    report = make_report(
        target,
        mode,
        artifact_name,
        source_info,
        elapsed,
        {
            "vram_peak_gb": vram_peak,
            "gpu_util_avg_pct": gpu_avg,
            "gpu_util_peak_pct": gpu_peak,
            "gpu_util_samples": gpu_stats["gpu_util_samples"],
        },
        run_error,
        cells,
        overall=overall,
    )
    save_json(results_dir / artifact_name.replace(".ipynb", ".json"), report)

    output_notebook = results_dir / artifact_name
    nbformat.write(nbformat.from_dict(build_artifact_notebook(original, notebook)), str(output_notebook))
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "html", str(output_notebook)],
        capture_output=True,
        text=True,
        check=False,
    )

    print(
        f"[{overall:6}] {mode:6} {target.notebook:45} "
        f"cells {passed}/{passed + failed}  {vram_peak:>5} GB  "
        f"gpu {format_pct(gpu_avg)}/{format_pct(gpu_peak)}  {elapsed:>6}s",
        flush=True,
    )
    return report


def collect_cell_results(notebook: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    cells: list[dict[str, Any]] = []
    passed = failed = 0
    index = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code" or cell.get("metadata", {}).get("ci_preamble"):
            continue
        index += 1
        error = next((o for o in cell.get("outputs", []) if o.get("output_type") == "error"), None)
        if error:
            failed += 1
            cells.append(
                {
                    "index": index,
                    "status": "FAILED",
                    "error": f"{error.get('ename')}: {error.get('evalue')}",
                }
            )
        else:
            passed += 1
            cells.append({"index": index, "status": "PASSED", "error": None})
    return cells, passed, failed


def make_report(
    target: Target,
    mode: str,
    artifact_name: str,
    source_info: dict[str, str | None],
    elapsed: float,
    gpu: dict[str, Any],
    run_error: str | None,
    cells: list[dict[str, Any]],
    overall: str | None = None,
) -> dict[str, Any]:
    passed = sum(cell["status"] == "PASSED" for cell in cells)
    failed = sum(cell["status"] == "FAILED" for cell in cells)
    if overall is None:
        overall = "ERROR" if run_error else ("PASSED" if failed == 0 else "FAILED")
    return {
        "mode": mode,
        "model_id": target.model_id,
        "notebook": target.notebook,
        "artifact_notebook": artifact_name,
        "source": source_info.get("source"),
        "fetched_url": source_info.get("fetched_url"),
        "snapshot": source_info.get("snapshot"),
        "overall_status": overall,
        "cells_passed": passed,
        "cells_failed": failed,
        "cells_total": passed + failed,
        "elapsed_seconds": elapsed,
        "vram_peak_gb": gpu.get("vram_peak_gb", 0.0),
        "gpu_util_avg_pct": gpu.get("gpu_util_avg_pct"),
        "gpu_util_peak_pct": gpu.get("gpu_util_peak_pct"),
        "gpu_util_samples": gpu.get("gpu_util_samples", 0),
        "vram_budget_gb": VRAM_BUDGET_GB,
        "log_file": artifact_name.replace(".ipynb", ".log"),
        "run_error": run_error,
        "cells": cells,
        "finished_at": utc_now(),
    }


def validate_plan(targets: list[Target]) -> list[str]:
    errors: list[str] = []
    print(f"Plan contains {len(targets)} notebook job(s).", flush=True)
    for target in targets:
        try:
            notebook, source = load_notebook(target, sync_native_snapshot=False)
            normalized = normalize_notebook(notebook)
            code_cells = sum(
                cell.get("cell_type") == "code"
                and not cell.get("metadata", {}).get("ci_preamble")
                for cell in normalized.get("cells", [])
            )
            fetched = source.get("fetched_url") or source.get("source")
            print(
                f"[PLAN OK] oneclick {target.model_id:45} "
                f"{target.notebook:45} code_cells={code_cells:02d} source={fetched}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            print(
                f"[PLAN ERR] oneclick {target.model_id:45} "
                f"{target.notebook:45} {compact_error(error, 240)}",
                flush=True,
            )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--target-file", default=str(TARGET_CSV))
    parser.add_argument(
        "--fail-on",
        choices=["all", "none"],
        default="none",
        help="which failures should make the process exit non-zero",
    )
    parser.add_argument("--filter", default="")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--cell-timeout", type=int, default=900)
    parser.add_argument("--notebook-timeout", type=int, default=1800)
    parser.add_argument("--echo-output", action="store_true")
    parser.add_argument("--echo-traceback", action="store_true")
    parser.add_argument(
        "--env-file",
        default="",
        help=f"optional KEY=VALUE env file for HF_TOKEN/proxy settings; default {DEFAULT_ENV_FILE}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(args.env_file or DEFAULT_ENV_FILE)
    if not args.plan_only:
        assert_jpeg_support()

    targets = load_targets(Path(args.target_file), args.filter)
    if not targets:
        raise SystemExit(f"no enabled targets matched filter {args.filter!r}")

    if args.plan_only:
        errors = validate_plan(targets)
        raise SystemExit(1 if errors else 0)

    policy = (
        f"source=hf-oneclick; fail_on={args.fail_on}; "
        "single Radeon GPU; model ids load through mounted Hugging Face cache"
    )
    reports: list[dict[str, Any]] = []
    pending = [f"oneclick__{target.notebook}" for target in targets]
    write_progress(results_dir, reports, pending)

    print(
        f"Running Hugging Face Oneclick notebook CI: targets={len(targets)}, "
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')}",
        flush=True,
    )

    print(f"\n==> PHASE HF Oneclick notebooks: {len(targets)} job(s)", flush=True)
    for target in targets:
        run_name = f"oneclick__{target.notebook}"
        if run_name in pending:
            pending.remove(run_name)
        print(f"==> START {run_name} ({target.model_id})", flush=True)
        write_progress(results_dir, reports, pending, running=run_name)
        report = run_one(target, args, results_dir)
        reports.append(report)
        write_summary(results_dir, reports, policy)
        write_progress(results_dir, reports, pending)

    write_summary(results_dir, reports, policy)

    if args.fail_on == "none":
        return
    failing = [
        report
        for report in reports
        if report["overall_status"] != "PASSED"
    ]
    if failing:
        print(f"\nFailing CI because {len(failing)} {args.fail_on} job(s) did not pass:", flush=True)
        for report in failing:
            print(
                f"- {report['mode']} {report['model_id']} "
                f"({report['overall_status']}, log={report['log_file']})",
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
