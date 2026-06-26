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
