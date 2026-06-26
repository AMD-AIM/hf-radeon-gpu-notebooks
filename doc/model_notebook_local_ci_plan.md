# Radeon Local GPU CI — Model Notebook Test Plan

Self-hosted GitHub Actions plan to execute **every notebook in
`radeon_notebooks/`** on local AMD Radeon W7900 hardware, inside the
`huaggingface_for_amd_radeon:latest` container, using **only GPU id 2 and 3**,
with the huge model weights served from the local SSDs (`/disk/ssd1`,
`/disk/ssd2`).

The final goal: a GitHub CI run whose **job summary shows, per model, the
notebook result** (pass/fail, per-cell status, peak VRAM, runtime), plus the
**rendered executed notebooks** (HTML) as downloadable artifacts so the actual
model output can be inspected.

> This is the local/on-prem variant. The cluster/Kubernetes version lives in
> [model_notebook_ci_plan.md](model_notebook_ci_plan.md). We reuse its
> **pass/fail evaluation idea** (per-cell error inspection + peak VRAM +
> elapsed time) but the infrastructure, GPU pinning, offline path rewriting,
> and summary are specific to this host.

---

> **Status — validated on this host (2026-06-26).** A live dry-run of the
> runner in §6.1 inside the container on GPU 2+3 executed
> `85_Qwen__Qwen3-0.6B.ipynb` **in memory** against
> `/disk/ssd1/zihaomu_amd/models/Qwen__Qwen3-0.6B` → **PASSED 3/3 cells,
> 3.6 GB peak VRAM, ~19 s**. Confirmed the path rewrite stays off disk: the
> saved `.ipynb`/`.html` contain the original `Qwen/Qwen3-0.6B` id and the real
> outputs, while `grep /disk/ssd1 results/*.ipynb` returns **0 hits**.

## 0. TL;DR — what runs

```text
GitHub self-hosted runner (this host: wx-ms-w7900d-0033)
  └─ docker run  huaggingface_for_amd_radeon:latest   (GPU 2 + 3 only)
       └─ tools/run_notebooks.py
            ├─ for each radeon_notebooks/*.ipynb:
            │    1. look up local model path in doc/model_path_mapping.csv
            │    2. rewrite the HF model id  ->  local /disk/ssdN path  (in-memory)
            │    3. neutralize "%pip install" cells (image is pre-provisioned)
            │    4. execute IN MEMORY with nbclient (offline, allow-errors)
            │    5. record per-cell PASS/FAIL, peak VRAM, elapsed time
            │    6. save ORIGINAL notebook + outputs as .ipynb + .html
            └─ write results/summary.md  ->  GitHub Step Summary + artifacts
```

Nothing on HuggingFace is mutated, and **the path-rewritten script is never
written to disk** (not the source repo, not `results/`, not a temp file). The
HF-id -> local-path swap is applied to an in-memory copy that is handed straight
to the kernel (`nbclient`); only the execution **outputs** are persisted, merged
back onto the *original* notebook source. So the saved `.ipynb`/`.html` keep the
original `org/model` ids — no host-specific `/disk/ssdN` path ever lands on disk,
and the notebooks under `radeon_notebooks/` are untouched.

---

## 1. Hardware & GPU pinning (verified)

Host `wx-ms-w7900d-0033`: 8 × AMD Radeon PRO W7900 (`gfx1100`, 48 GB each).
The CI is restricted to **GPU 2 and GPU 3** → a **96 GB** VRAM budget.

The `amd-smi` GPU index maps to the kernel render node as follows (verified on
this host — needed because Docker isolates GPUs by `/dev/dri/renderD*`):

| amd-smi GPU | PCI BDF        | KFD node | Render node               | card    |
|:-----------:|:---------------|:--------:|:--------------------------|:--------|
| 0           | 0000:03:00.0   | 2        | `/dev/dri/renderD128`     | card1   |
| 1           | 0000:23:00.0   | 3        | `/dev/dri/renderD129`     | card2   |
| **2**       | **0000:43:00.0** | **4**  | **`/dev/dri/renderD130`** | card3   |
| **3**       | **0000:63:00.0** | **5**  | **`/dev/dri/renderD131`** | card4   |
| 4           | 0000:83:00.0   | 6        | `/dev/dri/renderD132`     | card5   |
| 5           | 0000:a3:00.0   | 7        | `/dev/dri/renderD133`     | card6   |
| 6           | 0000:c3:00.0   | 8        | `/dev/dri/renderD134`     | card7   |
| 7           | 0000:e3:00.0   | 9        | `/dev/dri/renderD135`     | card8   |

So **GPU 2 → `renderD130`** and **GPU 3 → `renderD131`**.

Required group IDs on this host: `video=44`, `render=993`
(`getent group video render`).

### Why render-node isolation (recommended)

Passing only the two render nodes gives **true device isolation**: the container
physically cannot touch GPUs 0,1,4–7. Inside the container the two GPUs
re-index to `0` and `1`. Verified:

```text
$ docker run ... --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 ...
torch.cuda.is_available() = True
torch.cuda.device_count() = 2          # exactly GPU 2 and 3, nothing else
```

Alternative (looser) approach — expose all GPUs and filter by env:
`--device=/dev/dri -e HIP_VISIBLE_DEVICES=2,3`. This still lets `amd-smi`
*see* all 8 cards, so we prefer the render-node method above.

---

## 2. Container image (verified)

`huaggingface_for_amd_radeon:latest` (image id `ecf646ce6101`, ~41 GB, already
present locally). Verified contents:

| Component   | Value                                   |
|-------------|-----------------------------------------|
| Python      | 3.12.3 (venv at `/opt/venv`)            |
| PyTorch     | `2.10.0+rocm7.2.4`                       |
| transformers| `5.12.1`                                |
| jupyter     | `/opt/venv/bin/jupyter`                 |
| nbconvert   | `7.17.1` (+ nbclient `0.11.0`)          |
| ROCm tools  | `/opt/rocm/bin/rocm-smi`, `amd-smi`     |

**Implication:** jupyter, nbconvert and a working ROCm PyTorch are already
baked in — **no per-run `pip install` is required**, and the notebooks'
`%pip install` cells are intentionally skipped (see §4). Missing-package
failures are fixed by rebuilding the image, not at run time.

---

## 3. Storage & the model-path mapping

### 3.1 Naming convention (verified)

HuggingFace ids are stored on the SSDs with `/` replaced by `__`:

```text
<org>/<model>   ->   /disk/ssd{1,2}/zihaomu_amd/models/<org>__<model>
```

Example (the one from the request):

```text
openai/gpt-oss-20b  ->  /disk/ssd1/zihaomu_amd/models/openai__gpt-oss-20b
```

Whether a model lives on **SSD1 or SSD2** is taken from
[hf_model_summary_path.csv](hf_model_summary_path.csv) (column *Storage
Location*) and then **verified against the actual directory on disk**.

### 3.2 `doc/model_path_mapping.csv` (generated)

This file is the single source of truth the runner consumes. One row per
notebook in `radeon_notebooks/`. Schema:

| Column             | Meaning                                                    |
|--------------------|------------------------------------------------------------|
| `notebook`         | filename in `radeon_notebooks/` (join key)                 |
| `model_id`         | HF id to search-and-replace in the notebook code cells     |
| `slug`             | `model_id` with `/`→`__`                                    |
| `storage`          | `SSD1` / `SSD2`                                             |
| `local_path`       | absolute path the runner substitutes in                    |
| `model_dir_exists` | `yes`/`no` — preflight check that weights are really there  |

Current status: **87 notebooks mapped, all `model_dir_exists = yes`**
(86 on SSD1, 1 on SSD2).

The CSV is regenerated by [tools/build_model_path_mapping.py](#tool-mapping)
(§9.1) any time models or notebooks change.

---

## 4. In-memory notebook patching (the "temporary path replace")

For each notebook the runner builds a throwaway **in-memory** copy and applies
these patches **before execution**. This copy lives only in RAM — it is handed
directly to the kernel and is **never written to disk** (not the source repo,
not `results/`, not a temp file):

1. **Model-id → local path.** Every occurrence of the exact `model_id` string
   inside **code cells** is replaced with `local_path`. This rewrites the load
   calls, e.g.

   ```python
   pipe = pipeline("text-generation", model="openai/gpt-oss-20b", device_map="auto")
   tok  = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
   # becomes
   pipe = pipeline("text-generation", model="/disk/ssd1/zihaomu_amd/models/openai__gpt-oss-20b", device_map="auto")
   tok  = AutoTokenizer.from_pretrained("/disk/ssd1/zihaomu_amd/models/openai__gpt-oss-20b")
   ```

   Replacement is scoped to that notebook's own id, so there is no risk of one
   id being a prefix of another. **This rewrite is temporary:** after execution
   the saved artifact is rebuilt from the *original* notebook source (original
   ids) with only the freshly produced outputs merged in, so the local path is
   discarded and never persisted (verified: `grep /disk/ssd1 results/*.ipynb`
   returns nothing).

2. **Neutralize `%pip install` cells.** The image is pre-provisioned and CI
   runs **offline**, so pip lines are commented out (`# [ci-skipped pip] ...`).
   Use `--keep-pip` to override for a model that genuinely needs an extra wheel.

3. **Strip `kernelspec`** from notebook metadata to avoid nbconvert
   "no such kernel" errors.

4. **Force offline loading** via env: `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`. Combined with absolute local paths this guarantees
   weights load from `/disk/ssdN` and the run never reaches the network.

`device_map="auto"` is already present in these notebooks, so a model that
needs both cards is automatically sharded across GPU 2 and 3.

---

## 5. Canonical `docker run` (GPU 2 + 3, SSDs mounted)

```bash
docker run --rm \
  --name hf-ci-gpu23 \
  --device=/dev/kfd \
  --device=/dev/dri/renderD130 \      # GPU 2
  --device=/dev/dri/renderD131 \      # GPU 3
  --group-add 44 --group-add 993 \    # video, render
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32g \
  -v /disk/ssd1:/disk/ssd1 \          # SSD1 weights (read path preserved)
  -v /disk/ssd2:/disk/ssd2 \          # SSD2 weights
  -v "$PWD":/workspace -w /workspace \ # the checked-out repo
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /bin/bash \
  huaggingface_for_amd_radeon:latest \
  -lc 'python3 tools/run_notebooks.py --results-dir results'
```

The SSD mounts use **identical in/out paths** so the `local_path` values in the
CSV resolve unchanged inside the container.

### 5.1 Persistent dev container (`scripts/start_ci_container.sh`)

The CI workflow (§6.3) uses a throwaway `docker run --rm`, but for interactive
development against the model and script folders there is a **long-lived**
container created once and reused. It bind-mounts the same paths, so it holds no
state and is safe to remove/recreate:

```bash
./scripts/start_ci_container.sh              # create (or re-start if it exists)
./scripts/start_ci_container.sh --recreate   # force a clean rebuild
docker exec -it hf_radeon_ci bash            # enter it
```

| Setting          | Value                                                        |
|------------------|--------------------------------------------------------------|
| Container name   | `hf_radeon_ci`                                               |
| GPUs             | 2 + 3 only (`renderD130`, `renderD131`) — verified 2 visible |
| Restart policy   | `unless-stopped` (survives reboot / docker restart)          |
| `/disk/ssd1`     | → `/disk/ssd1` (SSD1 model weights, 98 dirs)                 |
| `/disk/ssd2`     | → `/disk/ssd2` (SSD2 model weights, 31 dirs)                 |
| repo             | → `/workspace` (87 notebooks + `tools/` + `doc/`)            |
| Env              | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`                 |

Run the batch inside the running container exactly as CI does:

```bash
docker exec -w /workspace hf_radeon_ci \
  python3 tools/run_notebooks.py --results-dir results --filter Qwen3-0.6B
```

**Status — created & verified on this host (2026-06-26):** `hf_radeon_ci` is up,
`torch.cuda.device_count() == 2` (GPU 2 GUID 60148 + GPU 3 GUID 13037), both SSD
model trees and the repo (`radeon_notebooks/`, `doc/model_path_mapping.csv`) are
visible and `/workspace` is writable.

---

## 6. Files to add to the repo

Two new files turn this plan into a runnable pipeline. Both are listed here in
full so the document is self-contained.

<a name="tool-runner"></a>
### 6.1 `tools/run_notebooks.py`

```python
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
```

<a name="tool-mapping"></a>
### 6.2 `tools/build_model_path_mapping.py`

Regenerates `doc/model_path_mapping.csv` from the summary CSV + on-disk probe.
(This is the exact generator already used to produce the current file.)

```python
#!/usr/bin/env python3
"""Rebuild doc/model_path_mapping.csv: join radeon_notebooks with the summary
CSV and verify each model directory on /disk/ssd{1,2}."""
import csv, os, re, glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "doc" / "hf_model_summary_path.csv"
NB_DIR = ROOT / "radeon_notebooks"
OUT = ROOT / "doc" / "model_path_mapping.csv"
BASE = {"SSD1": "/disk/ssd1/zihaomu_amd/models",
        "SSD2": "/disk/ssd2/zihaomu_amd/models"}

summary = {}
with open(SUMMARY, newline="") as f:
    for row in csv.reader(f):
        if not row or not row[0].strip() or row[0].strip() == "Model":
            continue
        nb = row[5].strip() if len(row) > 5 else ""
        if nb:
            summary[nb] = (row[0].strip(),
                           row[2].strip().upper() if len(row) > 2 else "")


def model_id_from_nb(path):
    try:
        for cell in json.load(open(path)).get("cells", []):
            if cell.get("cell_type") == "markdown":
                m = re.search(r"Run test:\s*([^\s\\]+)",
                              "".join(cell.get("source", [])))
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return None


def nb_num(p):
    m = re.match(r"^(\d+)_", os.path.basename(p))
    return int(m.group(1)) if m else 0


rows = []
for path in sorted(glob.glob(str(NB_DIR / "*.ipynb")), key=nb_num):
    fname = os.path.basename(path)
    m = re.match(r"^\d+_(.+)\.ipynb$", fname)
    slug_fname = m.group(1) if m else fname[:-6]
    model_id, storage = summary.get(fname, (None, ""))
    if not model_id:
        model_id = model_id_from_nb(path) or slug_fname.replace("__", "/", 1)
    slug = model_id.replace("/", "__")
    order = ([storage] if storage in BASE else []) + \
            [s for s in ("SSD1", "SSD2") if s != storage]
    local_path, chosen, exists = "", storage, False
    for s in order:
        p = os.path.join(BASE[s], slug)
        if os.path.isdir(p):
            local_path, chosen, exists = p, s, True
            break
    if not local_path:
        chosen = storage if storage in BASE else "SSD1"
        local_path = os.path.join(BASE[chosen], slug)
    rows.append((fname, model_id, slug, chosen, local_path,
                 "yes" if exists else "no"))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["notebook", "model_id", "slug", "storage",
                "local_path", "model_dir_exists"])
    w.writerows(rows)
print(f"wrote {OUT}  ({len(rows)} rows, "
      f"{sum(1 for r in rows if r[5] == 'yes')} present)")
```

<a name="workflow"></a>
### 6.3 `.github/workflows/radeon-local-notebook-ci.yml`

```yaml
name: radeon-local-notebook-ci

on:
  workflow_dispatch:
    inputs:
      filter:
        description: "Substring filter on notebook name / model id (blank = all)"
        required: false
        default: ""
      keep_pip:
        description: "Keep %pip install cells (default: skip)"
        type: boolean
        default: false
  push:
    branches: [ feature/local-ci ]
  schedule:
    - cron: "0 18 * * *"   # nightly 18:00 UTC

concurrency:
  group: radeon-local-ci          # only one CI on the 2 GPUs at a time
  cancel-in-progress: false

jobs:
  run-notebooks:
    runs-on: [self-hosted, rocm, w7900]   # label your registered runner this way
    timeout-minutes: 1440
    steps:
      - uses: actions/checkout@v4

      - name: Refresh model path mapping
        run: python3 tools/build_model_path_mapping.py

      - name: Execute notebooks on GPU 2 + 3
        run: |
          mkdir -p results
          KEEP_PIP=""
          if [ "${{ github.event.inputs.keep_pip }}" = "true" ]; then KEEP_PIP="--keep-pip"; fi
          docker run --rm \
            --name hf-ci-gpu23-${{ github.run_id }} \
            --device=/dev/kfd \
            --device=/dev/dri/renderD130 \
            --device=/dev/dri/renderD131 \
            --group-add 44 --group-add 993 \
            --security-opt seccomp=unconfined \
            --ipc=host --shm-size 32g \
            -v /disk/ssd1:/disk/ssd1 \
            -v /disk/ssd2:/disk/ssd2 \
            -v "$PWD":/workspace -w /workspace \
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
            --entrypoint /bin/bash \
            huaggingface_for_amd_radeon:latest \
            -lc "python3 tools/run_notebooks.py \
                   --results-dir results \
                   --filter '${{ github.event.inputs.filter }}' $KEEP_PIP"

      - name: Publish summary to job page
        if: always()
        run: cat results/summary.md >> "$GITHUB_STEP_SUMMARY"

      - name: Upload results (JSON + executed notebooks + HTML)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: radeon-notebook-results-${{ github.run_id }}
          path: |
            results/*.json
            results/*.ipynb
            results/*.html
            results/summary.md
          retention-days: 30
```

Notes:
- `runs-on` must match the labels of the **self-hosted runner registered on
  this host** (see §10). Because only GPU 2+3 are dedicated to CI, the
  `concurrency` group serialises runs so two CI jobs never fight over the cards.
- The whole batch runs **serially inside one container** (one launch, lowest
  overhead). The per-notebook process isolation comes from `nbconvert` starting
  a fresh kernel for every notebook.
- `build_model_path_mapping.py` runs on the host (plain Python, no GPU) so the
  mapping is always current before the container starts.

---

## 7. Pass / fail evaluation

Adapted from the reference plan's per-cell scheme, extended with our
offline/local fields.

**Per code cell**
- `FAILED` — the executed cell contains an `output_type == "error"`.
- `PASSED` — otherwise (including cells with no output). Markdown cells are
  ignored.

**Per notebook (`overall_status`)**
- `PASSED` — every code cell passed (no error outputs).
- `FAILED` — at least one code cell produced an error output. `nbclient` runs
  with `allow_errors=True`, so execution continues and every failing cell is
  captured.
- `ERROR` — infrastructure problem: kernel crash, per-cell timeout
  (`--cell-timeout`), or whole-notebook watchdog timeout (`--nb-timeout`). On a
  watchdog timeout the kernel is force-shut to free VRAM. Tracked in `pod_error`.

**Recorded metrics per notebook**

| Field                                   | Source                              |
|-----------------------------------------|-------------------------------------|
| `cells_passed / cells_failed / cells_total` | parsed executed notebook        |
| `elapsed_seconds`                       | wall-clock around in-memory execute |
| `vram_peak_gb` (budget `vram_total_gb=96`) | 2 s `rocm-smi` poll, peak summed |
| `pod_error`                             | timeout / missing weights / parse   |
| `cells[]`                               | index, status, `ename: evalue`      |

Timeouts: `--cell-timeout 900` (per cell), `--nb-timeout 1800` (per notebook).
Raise both for the largest models (e.g. anything that shards across both cards).

---

## 8. Outputs — "model + ipynb result"

Every run produces, under `results/` (uploaded as a CI artifact):

| Artifact                         | Purpose                                        |
|----------------------------------|------------------------------------------------|
| `<NN_model>.json`                | machine-readable per-notebook report           |
| `<NN_model>.ipynb`               | **original** notebook source + execution outputs (the temporary local-path rewrite is **not** persisted) |
| `<NN_model>.html`                | rendered notebook — read the real model output |
| `summary.md`                     | the aggregate table (also posted to the job page) |

The **GitHub Step Summary** of the run shows the headline table directly on the
Actions run page:

```text
# Radeon Local Notebook CI — Results

**87 notebooks · 79 passed · 6 failed · 2 errored · 90.8% pass** (GPU 2+3, 96 GB budget)

| # | Status | Model | Notebook | Cells P/F/T | Peak VRAM | Time | Notes |
|--:|:------:|:------|:---------|:-----------:|:---------:|-----:|:------|
| 1 | PASS | `openai/gpt-oss-20b` | 27_openai__gpt-oss-20b.ipynb | 3/0/3 | 41.2 GB | 132s | |
| 2 | PASS | `Qwen/Qwen3-14B`      | 28_Qwen__Qwen3-14B.ipynb     | 3/0/3 | 28.7 GB |  78s | |
| ...                                                                                       |
```

(numbers above are illustrative). To inspect what a model actually produced,
download the artifact and open the matching `.html`.

---

## 9. VRAM budget & known limits

- **Budget = 96 GB** (2 × 48 GB). `device_map="auto"` shards across both cards.
- Models whose **weights alone exceed ~90 GB** will likely `FAILED` with an OOM
  during load. In the current `radeon_notebooks/` set the largest single entry
  is `Wan-AI/Wan2.2-T2V-A14B` (~117 GB on disk) — expect OOM/`FAILED`; it is
  kept in the matrix so the failure is visible rather than silently skipped.
- Notebooks needing a wheel not baked into the image will `FAILED` on import.
  Fix by adding the package to the image and re-running (do **not** rely on
  per-run pip — CI is offline). This mirrors the "missing pip packages" /
  "transformers too old" categories in the reference plan.
- Gated/incomplete downloads are excluded automatically: the runner skips any
  row whose `model_dir_exists != yes`.

---

## 10. One-time host setup

1. **Register a self-hosted runner** on `wx-ms-w7900d-0033`, labelled to match
   the workflow:

   ```bash
   # from repo Settings ▸ Actions ▸ Runners ▸ New self-hosted runner (Linux x64)
   ./config.sh --url https://github.com/AMD-AIM/hf-radeon-gpu-notebooks \
               --token <RUNNER_TOKEN> \
               --labels self-hosted,rocm,w7900 \
               --name w7900-gpu23
   ./run.sh        # or install as a service: sudo ./svc.sh install && sudo ./svc.sh start
   ```

   The runner's Linux user must be in the `docker`, `video`, and `render`
   groups (already true for `zihaomu` on this host).

2. **Confirm the image is present** (it is): `docker images | grep huagging`.

3. **Add the three files** from §6 (`tools/run_notebooks.py`,
   `tools/build_model_path_mapping.py`, the workflow YAML) and push to the
   `feature/local-ci` branch to trigger the first run, or use
   *Run workflow* (workflow_dispatch) with an optional `filter`.

---

## 11. Local dry-run (no GitHub, smoke test)

Validate the whole flow on one small model before wiring CI:

```bash
cd /home/zihaomu/big_card/notebook_polish/hf-radeon-gpu-notebooks
python3 tools/build_model_path_mapping.py     # refresh the CSV

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 \
  --group-add 44 --group-add 993 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32g \
  -v /disk/ssd1:/disk/ssd1 -v /disk/ssd2:/disk/ssd2 \
  -v "$PWD":/workspace -w /workspace \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /bin/bash \
  huaggingface_for_amd_radeon:latest \
  -lc 'python3 tools/run_notebooks.py --results-dir results --filter Qwen3-0.6B'

cat results/summary.md            # inspect the table
xdg-open results/85_Qwen__Qwen3-0.6B.html 2>/dev/null   # inspect model output
```

A green `PASSED` row here means the path rewrite, offline load, GPU pinning and
result capture all work end-to-end.

---

## 12. Source-of-truth files

| File                                  | Role                                        |
|---------------------------------------|---------------------------------------------|
| `radeon_notebooks/*.ipynb`            | the notebooks under test (not modified)     |
| `doc/hf_model_summary_path.csv`       | which SSD each model lives on (input)       |
| `doc/model_path_mapping.csv`          | generated id→path map the runner consumes   |
| `tools/build_model_path_mapping.py`   | regenerates the map (§6.2)                   |
| `tools/run_notebooks.py`              | the executor (§6.1)                          |
| `.github/workflows/radeon-local-notebook-ci.yml` | the CI entrypoint (§6.3)         |
| `scripts/start_ci_container.sh`       | persistent dev container launcher (§5.1)    |
