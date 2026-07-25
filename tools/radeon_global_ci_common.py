"""Small shared helpers used by the Radeon Global notebook controller."""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
TARGET_CSV = REPO / "doc" / "ci_target_models.csv"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SECRET_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
TOKEN_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
DISABLED_TARGET_VALUES = {"0", "false", "no", "n"}
SECRET_VALUES: set[str] = set()


@dataclass(frozen=True)
class Target:
    model_id: str
    notebook: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


RUN_STARTED_AT = time.time()
RUN_STARTED_UTC = utc_now()


def redact_secrets(text: str) -> str:
    redacted = str(text)
    for secret in SECRET_VALUES:
        if secret and len(secret) >= 8:
            redacted = redacted.replace(secret, "******")
    return redacted


def runtime_hf_token() -> str:
    for key in TOKEN_ENV_KEYS:
        token = os.environ.get(key, "")
        if token and token != "YOUR_TOKEN_HERE":
            SECRET_VALUES.add(token)
            return token
    return ""


def load_targets(path: Path, text_filter: str = "") -> list[Target]:
    targets: list[Target] = []
    needle = text_filter.lower().strip()
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            enabled = (row.get("enabled") or "yes").strip().lower()
            if enabled in DISABLED_TARGET_VALUES:
                continue

            model_id = (row.get("model_id") or "").strip()
            notebook = (row.get("notebook") or "").strip()
            if not model_id or not notebook:
                raise ValueError(
                    f"{path}:{line_number}: model_id and notebook are required"
                )
            if needle and needle not in f"{model_id} {notebook}".lower():
                continue
            targets.append(Target(model_id=model_id, notebook=notebook))
    return targets


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def compact_error(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    compact = " ".join(
        strip_ansi(str(text)).replace("\n", " ").replace("\r", " ").split()
    )
    compact = compact.replace("|", "\\|")
    if len(compact) > limit:
        return compact[: limit - 1].rstrip() + "…"
    return compact


def format_duration(seconds: Any) -> str:
    value = int(round(float(seconds or 0)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_download_tries(report: dict[str, Any]) -> str:
    if report.get("download_status") == "IN_NOTEBOOK":
        return "\\"
    return str(report.get("download_attempts", 0))


def format_download_duration(report: dict[str, Any]) -> str:
    if report.get("download_status") == "IN_NOTEBOOK":
        return "in notebook"
    return format_duration(report.get("download_elapsed_seconds"))


def core_error(report: dict[str, Any]) -> str:
    if report.get("run_error"):
        return compact_error(report["run_error"])
    for cell in report.get("cells", []):
        if cell.get("status") != "PASSED" and cell.get("error"):
            return compact_error(cell["error"])
    return ""


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_report(
    target: Target,
    artifact_name: str,
    elapsed: float,
    run_error: str | None,
    cells: list[dict[str, Any]],
    started_at: str | None,
) -> dict[str, Any]:
    passed = sum(cell["status"] == "PASSED" for cell in cells)
    failed = sum(cell["status"] == "FAILED" for cell in cells)
    overall = "ERROR" if run_error else ("PASSED" if failed == 0 else "FAILED")
    return {
        "mode": "radeon-pod",
        "model_id": target.model_id,
        "notebook": target.notebook,
        "artifact_notebook": artifact_name,
        "source": "radeon-global-managed",
        "overall_status": overall,
        "cells_passed": passed,
        "cells_failed": failed,
        "cells_total": passed + failed,
        "download_status": "IN_NOTEBOOK",
        "download_attempts": 0,
        "download_elapsed_seconds": 0.0,
        "elapsed_seconds": elapsed,
        "log_file": artifact_name.replace(".ipynb", ".log"),
        "run_error": redact_secrets(run_error) if run_error else None,
        "cells": cells,
        "started_at": started_at,
        "finished_at": utc_now(),
    }


def write_progress(
    results_dir: Path,
    reports: list[dict[str, Any]],
    pending: list[str],
    running: str | None = None,
) -> None:
    passed = sum(report["overall_status"] == "PASSED" for report in reports)
    failed = sum(report["overall_status"] == "FAILED" for report in reports)
    errored = sum(report["overall_status"] == "ERROR" for report in reports)
    lines = [
        "# HF One-Click Radeon Global CI Progress",
        "",
        f"_updated {utc_now()}_",
        "",
        f"done {len(reports)} · PASS {passed} · FAIL {failed} · "
        f"ERROR {errored} · pending {len(pending)}",
        "",
    ]
    if running:
        lines += [f"Running: `{running}`", ""]

    if reports:
        lines += [
            "| # | Status | Model | Download | Cell retries | Total | Log |",
            "|--:|:------:|:------|---------:|-------------:|------:|:----|",
        ]
        for index, report in enumerate(reports, 1):
            lines.append(
                f"| {index} | {report['overall_status']} | "
                f"`{report['model_id']}` | {format_download_duration(report)} | "
                f"{report.get('cell_execution_retries', 0)} | "
                f"{format_duration(report['elapsed_seconds'])} | "
                f"{report['log_file']} |"
            )

    if pending:
        lines += ["", "Pending:", ""]
        lines += [f"- `{name}`" for name in pending]

    (results_dir / "progress.md").write_text("\n".join(lines) + "\n")


def write_summary(
    results_dir: Path,
    reports: list[dict[str, Any]],
    policy: str,
) -> None:
    passed = sum(report["overall_status"] == "PASSED" for report in reports)
    failed = sum(report["overall_status"] == "FAILED" for report in reports)
    errored = sum(report["overall_status"] == "ERROR" for report in reports)
    total = len(reports)
    rate = (100.0 * passed / total) if total else 0.0
    total_elapsed = sum(float(report.get("elapsed_seconds") or 0.0) for report in reports)
    wall_elapsed = time.time() - RUN_STARTED_AT
    icons = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERR"}

    elapsed_values = [float(report.get("elapsed_seconds") or 0.0) for report in reports]
    average = total_elapsed / total if total else 0.0
    shortest = min(elapsed_values) if elapsed_values else 0.0
    longest = max(elapsed_values) if elapsed_values else 0.0
    lines = [
        "# HF One-Click Radeon Global CI - Results",
        "",
        f"**{total} notebook jobs · {passed} PASS · {failed} FAIL · "
        f"{errored} ERROR · {rate:.1f}% pass · "
        f"CI wall time {format_duration(wall_elapsed)} · "
        f"model job time {format_duration(total_elapsed)}**",
        "",
        f"Started: {RUN_STARTED_UTC}",
        f"Generated: {utc_now()}",
        "",
        f"Policy: {policy}",
        "",
        "## Timing",
        "",
        "| Jobs | Total Time | Avg / Job | Fastest | Slowest |",
        "|-----:|-----------:|----------:|--------:|--------:|",
        f"| {total} | {format_duration(total_elapsed)} | "
        f"{format_duration(average)} | {format_duration(shortest)} | "
        f"{format_duration(longest)} |",
    ]

    if reports:
        lines += [
            "",
            "## Radeon Global Notebooks",
            "",
            "| # | Status | Model | Download | Model Download Tries | "
            "Cell Retries | Cells P/F/T | Total | Core error |",
            "|--:|:------:|:------|---------:|------:|-------------:|"
            ":-----------:|------:|:-----------|",
        ]
        for index, report in enumerate(reports, 1):
            lines.append(
                f"| {index} | {icons[report['overall_status']]} | "
                f"`{report['model_id']}` | {format_download_duration(report)} | "
                f"{format_download_tries(report)} | "
                f"{report.get('cell_execution_retries', 0)} | "
                f"{report['cells_passed']}/{report['cells_failed']}/"
                f"{report['cells_total']} | "
                f"{format_duration(report['elapsed_seconds'])} | "
                f"{core_error(report)} |"
            )

    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {passed}/{total} passed ({rate:.1f}%)", flush=True)
