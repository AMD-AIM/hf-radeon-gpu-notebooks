#!/usr/bin/env python3
"""Rebuild doc/model_path_mapping.csv: join original_notebooks with the summary
CSV, resolve each model directory on /disk/ssd{1,2}, and mark CI eligibility.

A notebook is ``eligible`` for the GPU 2+3 (96 GB) CI when ALL hold:
  * its weights are fully downloaded   (summary "Download Status" == Completed*)
  * it needs <= 2 GPUs                  (summary "GPUs (48GB AMD W7900)" <= 2)
  * the model directory really exists   (on-disk probe)
"""
import csv, os, re, glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "doc" / "hf_model_summary_path.csv"
NB_DIR = ROOT / "original_notebooks"
OUT = ROOT / "doc" / "model_path_mapping.csv"
BASE = {"SSD1": "/disk/ssd1/zihaomu_amd/models",
        "SSD2": "/disk/ssd2/zihaomu_amd/models"}
MAX_GPUS = 2

# summary: notebook filename -> (model_id, gpus, storage, download_status)
summary = {}
with open(SUMMARY, newline="") as f:
    for row in csv.reader(f):
        if not row or not row[0].strip() or row[0].strip() == "Model":
            continue
        nb = row[5].strip() if len(row) > 5 else ""
        if not nb:
            continue
        summary[nb] = (
            row[0].strip(),                                  # model_id
            row[1].strip() if len(row) > 1 else "",          # gpus
            row[2].strip().upper() if len(row) > 2 else "",  # storage
            row[4].strip() if len(row) > 4 else "",          # download_status
        )


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
    model_id, gpus, storage, dl_status = summary.get(fname, (None, "", "", ""))
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

    try:
        gpu_n = int(gpus)
    except (TypeError, ValueError):
        gpu_n = 99
    completed = dl_status.lower().startswith("completed")
    eligible = exists and completed and gpu_n <= MAX_GPUS

    rows.append((fname, model_id, slug, chosen, gpus, dl_status, local_path,
                 "yes" if exists else "no", "yes" if eligible else "no"))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["notebook", "model_id", "slug", "storage", "gpus",
                "download_status", "local_path", "model_dir_exists", "eligible"])
    w.writerows(rows)

elig = sum(1 for r in rows if r[8] == "yes")
print(f"wrote {OUT}  ({len(rows)} notebooks, {elig} eligible, "
      f"{len(rows) - elig} skipped)")
