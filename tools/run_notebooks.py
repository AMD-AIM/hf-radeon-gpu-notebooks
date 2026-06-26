#!/usr/bin/env python3
"""Execute radeon_notebooks/*.ipynb against local model weights on GPU 2,3.

Run inside huaggingface_for_amd_radeon:latest. Patches each notebook in memory
(HF id -> local /disk/ssdN path), executes with nbconvert offline, records
per-cell PASS/FAIL + peak VRAM + elapsed, and emits JSON / HTML / summary.md.
"""
import argparse, copy, csv, json, os, re, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "radeon_notebooks"
MAP_CSV = REPO / "doc" / "model_path_mapping.csv"


def load_mapping():
    with open(MAP_CSV, newline="") as f:
        return {r["notebook"]: r for r in csv.DictReader(f)}


def patch_notebook(nb, model_id, local_path, keep_pip):
    pip_re = re.compile(r"^\s*[%!]?\s*pip\s+install", re.I)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not keep_pip:
            src = "\n".join(
                ("# [ci-skipped pip] " + ln) if pip_re.match(ln) else ln
                for ln in src.splitlines()
            )
        src = src.replace(model_id, local_path)          # the core path rewrite
        cell["source"] = src
        cell["outputs"] = []
        cell["execution_count"] = None
    nb.get("metadata", {}).pop("kernelspec", None)
    return nb


def poll_vram(stop, peak):
    """Track peak summed VRAM (bytes) across the visible GPUs via rocm-smi."""
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=10).stdout
            data = json.loads(out)
            used = 0
            for card in data.values():
                for k, v in card.items():
                    if "used" in k.lower() and "vram" in k.lower():
                        try:
                            used += int(v)
                        except (TypeError, ValueError):
                            pass
            peak[0] = max(peak[0], used)
        except Exception:
            pass
        time.sleep(2)


def run_one(nb_file, row, args, results_dir):
    model_id, local_path = row["model_id"], row["local_path"]
    original = json.loads((NB_DIR / nb_file).read_text())

    # --- TEMPORARY, IN-MEMORY ONLY -------------------------------------------
    # Rewrite the HF id -> local /disk/ssdN path on a deep copy that lives only
    # in RAM and is handed straight to the kernel. It is NEVER written to disk.
    patched = patch_notebook(copy.deepcopy(original), model_id, local_path,
                             args.keep_pip)
    nb_node = nbformat.from_dict(patched)
    # -------------------------------------------------------------------------

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    peak, stop = [0], threading.Event()
    th = threading.Thread(target=poll_vram, args=(stop, peak), daemon=True)
    th.start()

    pod_error, exec_err = None, {}
    client = NotebookClient(nb_node, timeout=args.cell_timeout,
                            allow_errors=True, kernel_name="python3")

    def _exec():
        try:
            client.execute()
        except Exception as e:                     # kernel crash / cell timeout
            exec_err["msg"] = f"{type(e).__name__}: {e}"

    start = time.time()
    worker = threading.Thread(target=_exec, daemon=True)
    worker.start()
    worker.join(args.nb_timeout)                   # whole-notebook watchdog
    if worker.is_alive():
        pod_error = f"notebook timeout > {args.nb_timeout}s"
        try:                                       # force-free the GPU
            if getattr(client, "km", None) is not None:
                client.km.shutdown_kernel(now=True)
        except Exception:
            pass
        worker.join(10)
    elif "msg" in exec_err:
        pod_error = exec_err["msg"]
    elapsed = round(time.time() - start, 1)
    stop.set(); th.join(timeout=5)

    # Tally per-cell results from the executed in-memory node.
    cells, passed, failed = [], 0, 0
    idx = 0
    for cell in nb_node.cells:
        if cell.get("cell_type") != "code":
            continue
        idx += 1
        err = next((o for o in cell.get("outputs", [])
                    if o.get("output_type") == "error"), None)
        if err:
            failed += 1
            cells.append({"index": idx, "status": "FAILED",
                          "error": f'{err.get("ename")}: {err.get("evalue")}'})
        else:
            passed += 1
            cells.append({"index": idx, "status": "PASSED", "error": None})

    overall = "ERROR" if pod_error else ("PASSED" if failed == 0 else "FAILED")

    # --- Persist the RESULT, not the rewritten script ------------------------
    # Saved notebook = ORIGINAL source (HF ids preserved, no /disk path) with
    # only the freshly produced outputs merged in. The temporary local-path
    # rewrite is discarded, so no host-specific path ever lands on disk.
    artifact = nbformat.from_dict(original)
    executed_code = [c for c in nb_node.cells if c.get("cell_type") == "code"]
    j = 0
    for cell in artifact.cells:
        if cell.get("cell_type") != "code":
            continue
        if j < len(executed_code):
            cell["outputs"] = executed_code[j].get("outputs", [])
            cell["execution_count"] = executed_code[j].get("execution_count")
        j += 1
    out_nb = results_dir / nb_file
    nbformat.write(artifact, str(out_nb))
    # Rendered HTML (original paths, real outputs) so reviewers can read output.
    subprocess.run(["jupyter", "nbconvert", "--to", "html", str(out_nb)],
                   capture_output=True, text=True)
    # -------------------------------------------------------------------------

    report = {
        "notebook": nb_file, "model_id": model_id, "local_path": local_path,
        "storage": row["storage"], "overall_status": overall,
        "cells_passed": passed, "cells_failed": failed,
        "cells_total": passed + failed, "elapsed_seconds": elapsed,
        "vram_peak_gb": round(peak[0] / 1e9, 1), "vram_total_gb": 96.0,
        "pod_error": pod_error, "cells": cells,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (results_dir / nb_file.replace(".ipynb", ".json")).write_text(
        json.dumps(report, indent=2))
    print(f"[{overall:6}] {nb_file:55} "
          f"cells {passed}/{passed + failed}  "
          f"{report['vram_peak_gb']:>5} GB  {elapsed:>6}s"
          + (f"  ({pod_error})" if pod_error else ""))
    return report


def write_summary(reports, results_dir):
    n = len(reports)
    npass = sum(r["overall_status"] == "PASSED" for r in reports)
    nfail = sum(r["overall_status"] == "FAILED" for r in reports)
    nerr = sum(r["overall_status"] == "ERROR" for r in reports)
    rate = (100.0 * npass / n) if n else 0.0
    icon = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERR "}
    lines = [
        "# Radeon Local Notebook CI — Results",
        "",
        f"**{n} notebooks · {npass} passed · {nfail} failed · {nerr} errored "
        f"· {rate:.1f}% pass** (GPU 2+3, 96 GB budget)",
        "",
        "| # | Status | Model | Notebook | Cells P/F/T | Peak VRAM | Time | Notes |",
        "|--:|:------:|:------|:---------|:-----------:|:---------:|-----:|:------|",
    ]
    for i, r in enumerate(sorted(reports, key=lambda x: x["notebook"]), 1):
        note = r["pod_error"] or (
            "" if r["overall_status"] == "PASSED"
            else "; ".join(c["error"] for c in r["cells"]
                           if c["status"] == "FAILED")[:120])
        lines.append(
            f"| {i} | {icon[r['overall_status']]} | `{r['model_id']}` | "
            f"{r['notebook']} | {r['cells_passed']}/{r['cells_failed']}/"
            f"{r['cells_total']} | {r['vram_peak_gb']} GB | "
            f"{r['elapsed_seconds']:.0f}s | {note} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {npass}/{n} passed ({rate:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--filter", default="",
                    help="substring match on notebook name or model id")
    ap.add_argument("--cell-timeout", type=int, default=900)
    ap.add_argument("--nb-timeout", type=int, default=1800)
    ap.add_argument("--keep-pip", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping()

    todo = []
    for nb_file in sorted(mapping):
        row = mapping[nb_file]
        if args.filter and args.filter.lower() not in (
                nb_file + " " + row["model_id"]).lower():
            continue
        if row.get("model_dir_exists") != "yes":
            print(f"[SKIP  ] {nb_file:55} weights missing: {row['local_path']}")
            continue
        todo.append((nb_file, row))

    print(f"Running {len(todo)} notebook(s) on GPU 2+3 ...\n")
    reports = [run_one(nb, row, args, results_dir) for nb, row in todo]
    write_summary(reports, results_dir)


if __name__ == "__main__":
    main()
