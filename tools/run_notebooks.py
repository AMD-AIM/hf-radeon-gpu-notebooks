#!/usr/bin/env python3
"""Execute original_notebooks/*.ipynb against local model weights on GPU 2,3.

Run inside huaggingface_for_amd_radeon:latest. For each eligible notebook the
runner builds a throwaway IN-MEMORY copy, applies normalization patches (force
GPU placement, neutralize offline-incompatible demo cells, rewrite the HF id to
the local /disk/ssdN path, fix the text-model AutoModel class), executes it with
nbclient offline, records per-cell PASS/FAIL + peak VRAM + elapsed, and writes
JSON / HTML / summary.md. The path-rewritten script is NEVER written to disk:
only the original notebook source plus fresh outputs is persisted.
"""
import argparse, copy, csv, json, os, re, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "original_notebooks"
MAP_CSV = REPO / "doc" / "model_path_mapping.csv"

# Injected as the first cell of every notebook. Globally forces GPU placement
# (device_map/torch_dtype default to "auto") and offline mode, so raw HF
# notebooks that lack device_map="auto" still load on the GPU instead of CPU.
NORM_PREAMBLE = '''# [ci-normalize] force GPU placement + offline (injected by CI)
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import functools as _ft, transformers as _tf
_orig_pipeline = _tf.pipeline
def _ci_pipeline(*a, **k):
    k.setdefault("device_map", "auto")
    return _orig_pipeline(*a, **k)
_tf.pipeline = _ci_pipeline
for _n in ("AutoModelForCausalLM", "AutoModelForMultimodalLM",
           "AutoModelForImageTextToText", "AutoModelForSeq2SeqLM",
           "AutoModelForSpeechSeq2Seq", "AutoModel"):
    _c = getattr(_tf, _n, None)
    if _c is None:
        continue
    _orig_fp = _c.from_pretrained.__func__
    def _mk(orig):
        @classmethod
        @_ft.wraps(orig)
        def _fp(cls, *a, **k):
            k.setdefault("device_map", "auto")
            k.setdefault("torch_dtype", "auto")
            return orig(cls, *a, **k)
        return _fp
    _c.from_pretrained = _mk(_orig_fp)
print("[ci-normalize] applied")
'''

# Cells matching this are pure online API / interactive-login demos that cannot
# run in offline CI; they are commented out wholesale.
OFFLINE_BAD = re.compile(
    r"router\.huggingface\.co|from openai import|OpenAI\(|YOUR_TOKEN_HERE"
    r"|huggingface_hub import login|login\(\)|from google\.colab")
PIP_RE = re.compile(r"^\s*[%!]?\s*pip\s+install", re.I)
VL_KEYS = ("vl", "vision", "image", "multimodal", "omni", "clip", "siglip",
           "janus", "ocr", "mllama", "idefics", "internvl", "gemma3",
           "minicpm_v", "minicpmo")


def load_mapping():
    with open(MAP_CSV, newline="") as f:
        return {r["notebook"]: r for r in csv.DictReader(f)}


def is_vl_model(local_path):
    """Decide whether a model is vision/multimodal from its local config.json."""
    try:
        cfg = json.load(open(os.path.join(local_path, "config.json")))
    except Exception:
        return False
    if "vision_config" in cfg:
        return True
    blob = (cfg.get("model_type", "") + " " +
            " ".join(cfg.get("architectures", []) or [])).lower()
    return any(k in blob for k in VL_KEYS)


def patch_notebook(nb, model_id, local_path, keep_pip):
    vl = is_vl_model(local_path)
    preamble = {"cell_type": "code", "metadata": {"ci_preamble": True},
                "execution_count": None, "outputs": [], "source": NORM_PREAMBLE}
    out_cells = [preamble]
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            out_cells.append(cell)
            continue
        src = "".join(cell.get("source", []))
        if OFFLINE_BAD.search(src):
            src = "\n".join("# [ci-skip offline] " + ln
                            for ln in src.splitlines())
        else:
            lines = []
            for ln in src.splitlines():
                if not keep_pip and PIP_RE.match(ln):
                    ln = "# [ci-skip pip] " + ln
                lines.append(ln)
            src = "\n".join(lines)
            src = src.replace(model_id, local_path)          # core path rewrite
            if not vl:                                       # text model fix
                src = src.replace("AutoModelForMultimodalLM",
                                  "AutoModelForCausalLM")
        cell = dict(cell)
        cell["source"] = src
        cell["outputs"] = []
        cell["execution_count"] = None
        out_cells.append(cell)
    nb = dict(nb)
    nb["cells"] = out_cells
    nb.get("metadata", {}).pop("kernelspec", None)
    return nb, vl


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
    patched, vl = patch_notebook(copy.deepcopy(original), model_id,
                                 local_path, args.keep_pip)
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

    # Tally per-cell results, skipping the injected normalization preamble.
    cells, passed, failed = [], 0, 0
    idx = 0
    for cell in nb_node.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("metadata", {}).get("ci_preamble"):
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
    # Saved notebook = ORIGINAL source (HF ids preserved, no /disk path, no
    # injected preamble) with only the fresh outputs merged in.
    artifact = nbformat.from_dict(original)
    executed_code = [c for c in nb_node.cells
                     if c.get("cell_type") == "code"
                     and not c.get("metadata", {}).get("ci_preamble")]
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
    subprocess.run(["jupyter", "nbconvert", "--to", "html", str(out_nb)],
                   capture_output=True, text=True)
    # -------------------------------------------------------------------------

    report = {
        "notebook": nb_file, "model_id": model_id, "local_path": local_path,
        "storage": row.get("storage"), "gpus": row.get("gpus"),
        "is_vl": vl, "overall_status": overall,
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
          + (f"  ({pod_error})" if pod_error else ""), flush=True)
    return report


def write_summary(reports, results_dir):
    n = len(reports)
    npass = sum(r["overall_status"] == "PASSED" for r in reports)
    nfail = sum(r["overall_status"] == "FAILED" for r in reports)
    nerr = sum(r["overall_status"] == "ERROR" for r in reports)
    rate = (100.0 * npass / n) if n else 0.0
    icon = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERR "}
    lines = [
        "# Radeon Local Notebook CI — Results (original_notebooks)",
        "",
        f"**{n} notebooks · {npass} passed · {nfail} failed · {nerr} errored "
        f"· {rate:.1f}% pass** (GPU 2+3, 96 GB budget)",
        "",
        "| # | Status | Model | Notebook | VL | Cells P/F/T | Peak VRAM | Time | Notes |",
        "|--:|:------:|:------|:---------|:--:|:-----------:|:---------:|-----:|:------|",
    ]
    for i, r in enumerate(sorted(reports, key=lambda x: x["notebook"]), 1):
        note = r["pod_error"] or (
            "" if r["overall_status"] == "PASSED"
            else "; ".join(c["error"] for c in r["cells"]
                           if c["status"] == "FAILED")[:120])
        lines.append(
            f"| {i} | {icon[r['overall_status']]} | `{r['model_id']}` | "
            f"{r['notebook']} | {'Y' if r.get('is_vl') else ''} | "
            f"{r['cells_passed']}/{r['cells_failed']}/{r['cells_total']} | "
            f"{r['vram_peak_gb']} GB | {r['elapsed_seconds']:.0f}s | {note} |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary: {npass}/{n} passed ({rate:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--filter", default="",
                    help="substring match on notebook name or model id")
    ap.add_argument("--cell-timeout", type=int, default=900)
    ap.add_argument("--nb-timeout", type=int, default=1800)
    ap.add_argument("--keep-pip", action="store_true")
    ap.add_argument("--include-ineligible", action="store_true",
                    help="also run notebooks marked eligible=no (OOM/incomplete)")
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
        if not args.include_ineligible and row.get("eligible") != "yes":
            reason = row.get("download_status") or "ineligible"
            print(f"[SKIP  ] {nb_file:55} ({reason}, gpus={row.get('gpus')})")
            continue
        todo.append((nb_file, row))

    total = len(todo)
    print(f"Running {total} notebook(s) on GPU 2+3 ...\n", flush=True)
    reports = []
    for i, (nb_file, row) in enumerate(todo, 1):
        print(f"==> [{i}/{total}] START {nb_file}  ({row['model_id']})",
              flush=True)
        reports.append(run_one(nb_file, row, args, results_dir))
        npass = sum(r["overall_status"] == "PASSED" for r in reports)
        nfail = sum(r["overall_status"] == "FAILED" for r in reports)
        nerr = sum(r["overall_status"] == "ERROR" for r in reports)
        print(f"    PROGRESS {i}/{total}  ok={npass} fail={nfail} err={nerr}",
              flush=True)
        # incremental summary so progress is visible mid-run via artifacts/logs
        write_summary(reports, results_dir)
    write_summary(reports, results_dir)


if __name__ == "__main__":
    main()
