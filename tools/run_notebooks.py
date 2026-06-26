#!/usr/bin/env python3
"""Execute original_notebooks/*.ipynb against local model weights on GPU 2,3.

Run inside huaggingface_for_amd_radeon:latest. For each eligible notebook the
runner builds a throwaway IN-MEMORY copy, applies normalization patches (force
GPU placement, neutralize offline-incompatible demo cells, rewrite the HF id to
the local /disk/ssdN path, fix the text-model AutoModel class), executes it with
nbclient offline, records per-cell PASS/FAIL + peak VRAM + elapsed, and writes
JSON / HTML / summary.md. The path-rewritten script is NEVER written to disk:
only the original notebook source plus fresh outputs is persisted.

Live observability (for debugging user-reported errors):
  * results/<model>.log  — per-notebook EXECUTION LOG written cell-by-cell in
    real time: every cell's stdout/stream output, results, and full error
    tracebacks. Tail it live, or download it as a CI artifact.
  * error tracebacks are also echoed to stdout (the GitHub Actions step log) as
    they happen; pass --echo-output to also echo successful cell output.
  * results/progress.md + progress.json — live dashboard (running/done/pending/
    skipped, per-model VRAM + elapsed), refreshed every ~2 s.
"""
import argparse, copy, csv, json, os, re, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "original_notebooks"
MAP_CSV = REPO / "doc" / "model_path_mapping.csv"
SOURCE_DATA = REPO / "source_data"        # locally-cached notebook web resources
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_resource_map():
    """url -> absolute local path, for web images/audio pre-downloaded to
    source_data/ (so notebooks never fetch them at run time)."""
    rm = SOURCE_DATA / "resource_map.json"
    out = {}
    try:
        for url, name in json.loads(rm.read_text()).items():
            if name:
                out[url] = str(SOURCE_DATA / name)
    except Exception:
        pass
    return out


RESOURCE_MAP = load_resource_map()

# Host-local secrets / proxy config (HF_TOKEN, HF_ENDPOINT, ...). Lives OUTSIDE
# the repo so it is never committed, never uploaded, and survives
# actions/checkout's clean step. Override with --env-file or RADEON_CI_ENV_FILE.
DEFAULT_ENV_FILE = os.environ.get(
    "RADEON_CI_ENV_FILE", "/disk/ssd1/zihaomu_amd/ci_secrets.env")
SECRET_KEY_HINT = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def load_local_env(path):
    """Parse a KEY=VALUE file (comments / blank lines / optional 'export ')."""
    env = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                env[k] = v
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[warn] could not read env file {path}: {e}", flush=True)
    return env


def _mask(k, v):
    return "******" if any(h in k.upper() for h in SECRET_KEY_HINT) else v

# Injected as the first cell of every notebook. Globally forces GPU placement
# (device_map/torch_dtype default to "auto") and offline mode, so raw HF
# notebooks that lack device_map="auto" still load on the GPU instead of CPU.
NORM_PREAMBLE = '''# [ci-normalize] force GPU placement (injected by CI)
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

OFFLINE_BAD = re.compile(
    r"router\.huggingface\.co|from openai import|OpenAI\(|YOUR_TOKEN_HERE"
    r"|huggingface_hub import login|login\(\)|from google\.colab"
    r"|InferenceClient")  # HF cloud remote-inference: needs network/token
PIP_RE = re.compile(r"^\s*[%!]?\s*pip\s+install", re.I)
# Defensive: notebooks often reset HF_TOKEN to a placeholder
# (e.g. os.environ['HF_TOKEN'] = 'YOUR_TOKEN_HERE') or override the endpoint /
# offline flags. Such a line would clobber the host-local env we injected. We
# comment out ONLY those lines so the rest of the cell still runs.
ENV_GUARD = re.compile(
    r"""^\s*(?:os\.)?environ\s*\[\s*['"]"""
    r"""(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN|HF_ENDPOINT"""
    r"""|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|HF_HUB_DISABLE_XET|HTTP_PROXY"""
    r"""|HTTPS_PROXY|http_proxy|https_proxy)['"]\s*\]\s*=""")
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
                elif ENV_GUARD.match(ln):
                    ln = "# [ci-protect env] " + ln
                lines.append(ln)
            src = "\n".join(lines)
            src = src.replace(model_id, local_path)          # core path rewrite
            for _url, _local in RESOURCE_MAP.items():        # web asset -> local
                if _url in src:
                    src = src.replace(_url, _local)
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


def _fmt_hms(sec):
    sec = int(sec)
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


class Progress:
    """Thread-safe live dashboard written to results/progress.{json,md}."""

    def __init__(self, results_dir, total, skipped, max_gpus):
        self.results_dir = results_dir
        self.total = total
        self.skipped = skipped
        self.max_gpus = max_gpus
        self.pending = []
        self.running = None
        self.results = []
        self.started_at = time.time()
        self.lock = threading.Lock()

    def set_pending(self, names):
        with self.lock:
            self.pending = list(names)
        self.flush()

    def start_model(self, idx, nb, model, gpus):
        with self.lock:
            self.running = {"index": idx, "notebook": nb, "model": model,
                            "gpus": gpus, "started_at": time.time(),
                            "elapsed_s": 0.0, "vram_gb": 0.0, "vram_peak_gb": 0.0,
                            "cell": 0}
            if nb in self.pending:
                self.pending.remove(nb)
        self.flush()

    def set_cell(self, n):
        with self.lock:
            if self.running:
                self.running["cell"] = n
        self.flush()

    def update_vram(self, cur_gb, peak_gb):
        with self.lock:
            if self.running:
                self.running["vram_gb"] = round(cur_gb, 1)
                self.running["vram_peak_gb"] = round(peak_gb, 1)
                self.running["elapsed_s"] = round(
                    time.time() - self.running["started_at"], 1)
        self.flush()

    def finish_model(self, report):
        with self.lock:
            self.results.append(report)
            self.running = None
        self.flush()

    def flush(self):
        try:
            self._write_json(); self._write_md()
        except Exception:
            pass

    def _write_json(self):
        data = {"updated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed": _fmt_hms(time.time() - self.started_at),
                "max_gpus": self.max_gpus, "total": self.total,
                "done": len(self.results), "running": self.running,
                "pending": self.pending, "skipped": self.skipped,
                "results": self.results}
        tmp = self.results_dir / "progress.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.results_dir / "progress.json")

    def _write_md(self):
        npass = sum(r["overall_status"] == "PASSED" for r in self.results)
        nfail = sum(r["overall_status"] == "FAILED" for r in self.results)
        nerr = sum(r["overall_status"] == "ERROR" for r in self.results)
        L = ["# Radeon Local Notebook CI — Live Progress", "",
             f"_updated {datetime.now(timezone.utc).isoformat()} · "
             f"elapsed {_fmt_hms(time.time() - self.started_at)}_", "",
             f"**filter: max_gpus={self.max_gpus}** · "
             f"total {self.total} · done {len(self.results)} "
             f"(ok {npass} / fail {nfail} / err {nerr}) · "
             f"pending {len(self.pending)} · skipped {len(self.skipped)}", "",
             "## \u25b6 Running now"]
        if self.running:
            r = self.running
            L += ["",
                  f"**[{r['index']}/{self.total}] {r['notebook']}** "
                  f"(`{r['model']}`, {r['gpus']}-GPU) — cell {r['cell']}, "
                  f"running **{r['elapsed_s']:.0f}s**, VRAM **{r['vram_gb']} GB** "
                  f"(peak {r['vram_peak_gb']})  ·  live log: "
                  f"`results/{r['notebook'].replace('.ipynb', '.log')}`", ""]
        else:
            L += ["", "_idle_", ""]
        L += ["## Done (latest first)", "",
              "| # | Status | Model | Cells P/F/T | Peak VRAM | Time | Log |",
              "|--:|:------:|:------|:-----------:|:---------:|-----:|:----|"]
        icon = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERR "}
        for i, r in enumerate(reversed(self.results), 1):
            log = r["notebook"].replace(".ipynb", ".log")
            L.append(
                f"| {len(self.results) - i + 1} | {icon[r['overall_status']]} | "
                f"`{r['model_id']}` | {r['cells_passed']}/{r['cells_failed']}/"
                f"{r['cells_total']} | {r['vram_peak_gb']} GB | "
                f"{r['elapsed_seconds']:.0f}s | {log} |")
        L += ["", f"## Pending ({len(self.pending)})", ""]
        for nb in self.pending[:40]:
            L.append(f"- {nb}")
        if len(self.pending) > 40:
            L.append(f"- … and {len(self.pending) - 40} more")
        L += ["", f"## Skipped ({len(self.skipped)})", "",
              "| Notebook | Reason | gpus |", "|:---------|:-------|:----:|"]
        for s in self.skipped:
            L.append(f"| {s['notebook']} | {s['reason']} | {s.get('gpus', '')} |")
        (self.results_dir / "progress.md").write_text("\n".join(L) + "\n")


def poll_vram(stop, peak, progress):
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
            if progress is not None:
                progress.update_vram(used / 1e9, peak[0] / 1e9)
        except Exception:
            pass
        time.sleep(2)


def run_one(nb_file, row, args, results_dir, progress=None):
    model_id, local_path = row["model_id"], row["local_path"]
    original = json.loads((NB_DIR / nb_file).read_text())
    patched, vl = patch_notebook(copy.deepcopy(original), model_id,
                                 local_path, args.keep_pip)
    nb_node = nbformat.from_dict(patched)

    # Per-notebook execution log, written cell-by-cell in real time.
    log_path = results_dir / nb_file.replace(".ipynb", ".log")
    logf = open(log_path, "w", buffering=1)
    logf.write(f"# {nb_file}  ({model_id})  VL={vl}\n"
               f"# started {datetime.now(timezone.utc).isoformat()}\n"
               f"# local_path={local_path}\n\n")

    def emit(line, to_stdout=False):
        logf.write(line + "\n")
        if to_stdout:
            print(line, flush=True)

    def on_cell_start(cell, cell_index, **kw):
        if cell.get("metadata", {}).get("ci_preamble"):
            return
        if progress is not None:
            progress.set_cell(cell_index)
        src = "".join(cell.get("source", []))
        emit(f"\n----- cell {cell_index} -----")
        emit(src.rstrip())
        emit("----- output -----")

    def on_cell_executed(cell, cell_index, execute_reply=None, **kw):
        if cell.get("metadata", {}).get("ci_preamble"):
            return
        for o in cell.get("outputs", []):
            t = o.get("output_type")
            if t == "stream":
                emit("".join(o.get("text", "")).rstrip(),
                     to_stdout=args.echo_output)
            elif t == "execute_result":
                emit("".join(o.get("data", {}).get("text/plain", "")).rstrip(),
                     to_stdout=args.echo_output)
            elif t == "error":
                # Errors ALWAYS echo to the CI step log (the debugging signal).
                emit(f"[ERROR] {o.get('ename')}: {o.get('evalue')}",
                     to_stdout=True)
                for tb in o.get("traceback", []):
                    emit(ANSI.sub("", tb).rstrip(), to_stdout=True)

    peak, stop = [0], threading.Event()
    th = threading.Thread(target=poll_vram, args=(stop, peak, progress),
                          daemon=True)
    th.start()

    pod_error, exec_err = None, {}
    client = NotebookClient(nb_node, timeout=args.cell_timeout,
                            allow_errors=True, kernel_name="python3",
                            on_cell_start=on_cell_start,
                            on_cell_executed=on_cell_executed)

    def _exec():
        try:
            client.execute()
        except Exception as e:
            exec_err["msg"] = f"{type(e).__name__}: {e}"

    start = time.time()
    worker = threading.Thread(target=_exec, daemon=True)
    worker.start()
    worker.join(args.nb_timeout)
    if worker.is_alive():
        pod_error = f"notebook timeout > {args.nb_timeout}s"
        emit(f"[TIMEOUT] {pod_error}", to_stdout=True)
        try:
            if getattr(client, "km", None) is not None:
                client.km.shutdown_kernel(now=True)
        except Exception:
            pass
        worker.join(10)
    elif "msg" in exec_err:
        pod_error = exec_err["msg"]
        emit(f"[KERNEL-ERROR] {pod_error}", to_stdout=True)
    elapsed = round(time.time() - start, 1)
    stop.set(); th.join(timeout=5)

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
    emit(f"\n# RESULT {overall}  passed={passed} failed={failed} "
         f"elapsed={elapsed}s peak_vram={round(peak[0] / 1e9, 1)}GB")
    logf.close()

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

    report = {
        "notebook": nb_file, "model_id": model_id, "local_path": local_path,
        "storage": row.get("storage"), "gpus": row.get("gpus"),
        "is_vl": vl, "overall_status": overall,
        "cells_passed": passed, "cells_failed": failed,
        "cells_total": passed + failed, "elapsed_seconds": elapsed,
        "vram_peak_gb": round(peak[0] / 1e9, 1), "vram_total_gb": 96.0,
        "log_file": log_path.name, "pod_error": pod_error, "cells": cells,
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
        "# Radeon Local Notebook CI — Results (original_notebooks)", "",
        f"**{n} notebooks · {npass} passed · {nfail} failed · {nerr} errored "
        f"· {rate:.1f}% pass** (GPU 2+3, 96 GB budget)", "",
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
    ap.add_argument("--max-gpus", type=int, default=1,
                    help="only run models needing <= this many GPUs "
                         "(default 1 = single-card only)")
    ap.add_argument("--order", choices=["size", "name"], default="size",
                    help="run order: 'size' = smallest model first (default)")
    ap.add_argument("--echo-output", action="store_true",
                    help="also echo successful cell output to stdout (errors are "
                         "always echoed); per-notebook .log always has full output")
    ap.add_argument("--env-file", default="",
                    help="KEY=VALUE file of extra env (HF_TOKEN, HF_ENDPOINT, "
                         "proxies) injected before each notebook; host-local, "
                         "never committed. Defaults to RADEON_CI_ENV_FILE or "
                         f"{DEFAULT_ENV_FILE}")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping()

    # Inject host-local env (token / proxy / endpoint) so every kernel inherits
    # it. The token never enters a notebook cell, an artifact, or git.
    env_path = args.env_file or DEFAULT_ENV_FILE
    extra_env = load_local_env(env_path)
    if extra_env:
        os.environ.update(extra_env)
        shown = ", ".join(f"{k}={_mask(k, v)}" for k, v in extra_env.items())
        print(f"[env] loaded {len(extra_env)} var(s) from {env_path}: {shown}",
              flush=True)
        # Network intentionally allowed (proxy/token provided): do NOT force
        # offline, so notebooks can fetch images etc. Local weights still load
        # from /disk because ids are rewritten to absolute paths.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        print("[env] network ENABLED (offline flags cleared)", flush=True)
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print(f"[env] no env file at {env_path}; running fully OFFLINE",
              flush=True)

    def gpus_of(row):
        try:
            return int(row.get("gpus") or 99)
        except (TypeError, ValueError):
            return 99

    todo, skipped = [], []
    for nb_file in sorted(mapping):
        row = mapping[nb_file]
        if args.filter and args.filter.lower() not in (
                nb_file + " " + row["model_id"]).lower():
            continue
        if not args.include_ineligible and row.get("eligible") != "yes":
            reason = row.get("download_status") or "ineligible"
            skipped.append({"notebook": nb_file, "model": row["model_id"],
                            "reason": reason, "gpus": row.get("gpus", "")})
            print(f"[SKIP  ] {nb_file:55} ({reason}, gpus={row.get('gpus')})",
                  flush=True)
            continue
        if gpus_of(row) > args.max_gpus:
            skipped.append({"notebook": nb_file, "model": row["model_id"],
                            "reason": f"needs {row.get('gpus')} GPUs "
                                      f"(> max_gpus={args.max_gpus})",
                            "gpus": row.get("gpus", "")})
            print(f"[SKIP  ] {nb_file:55} (needs {row.get('gpus')} GPUs "
                  f"> max {args.max_gpus})", flush=True)
            continue
        todo.append((nb_file, row))

    if args.order == "size":
        def _sz(item):
            lp = item[1].get("local_path", "")
            try:
                return sum(f.stat().st_size for f in Path(lp).rglob("*")
                           if f.is_file())
            except Exception:
                return float("inf")
        todo.sort(key=_sz)

    total = len(todo)
    progress = Progress(results_dir, total, skipped, args.max_gpus)
    progress.set_pending([nb for nb, _ in todo])

    print(f"Running {total} notebook(s) on GPU 2+3 "
          f"(max_gpus={args.max_gpus}, order={args.order}); "
          f"{len(skipped)} skipped.\n", flush=True)

    reports = []
    for i, (nb_file, row) in enumerate(todo, 1):
        print(f"==> [{i}/{total}] START {nb_file}  ({row['model_id']})  "
              f"log: results/{nb_file.replace('.ipynb', '.log')}", flush=True)
        progress.start_model(i, nb_file, row["model_id"], row.get("gpus"))
        report = run_one(nb_file, row, args, results_dir, progress)
        progress.finish_model(report)
        reports.append(report)
        npass = sum(r["overall_status"] == "PASSED" for r in reports)
        nfail = sum(r["overall_status"] == "FAILED" for r in reports)
        nerr = sum(r["overall_status"] == "ERROR" for r in reports)
        print(f"    PROGRESS {i}/{total}  ok={npass} fail={nfail} err={nerr}",
              flush=True)
        write_summary(reports, results_dir)
    write_summary(reports, results_dir)


if __name__ == "__main__":
    main()
