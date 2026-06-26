#!/usr/bin/env python3
"""Pre-download every web resource (images/audio/...) referenced by
original_notebooks/*.ipynb into source_data/, and write source_data/
resource_map.json (url -> local filename).

The runner rewrites those URLs to the local files so VL/vision notebooks never
fetch anything at run time. Re-run this whenever notebooks add new resources.

Routes (this host cannot reach huggingface.co/github directly):
  * huggingface.co  -> HF_RESOURCE_PROXY (default http://134.199.133.77)
  * raw.githubusercontent.com -> https://ghproxy.net/<url>
  * everything else -> direct
  * fallback for hf.co -> https://hf-mirror.com
"""
import glob, hashlib, json, os, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "original_notebooks"
OUT = ROOT / "source_data"
PROXY = os.environ.get("HF_RESOURCE_PROXY", "http://134.199.133.77")

URL = re.compile(r'https?://[^\s"\'\)\]]+')
RES_EXT = re.compile(r'\.(png|jpe?g|gif|webp|bmp|mp3|wav|flac|m4a|mp4|ogg|pdf|wms)(\?|$)', re.I)


def collect_urls():
    urls = set()
    for f in sorted(glob.glob(str(NB_DIR / "*.ipynb"))):
        nb = json.load(open(f))
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            for u in URL.findall(src):
                u = u.rstrip(".,")
                if RES_EXT.search(u):
                    urls.add(u)
    return sorted(urls)


def local_name(u):
    base = os.path.basename(u.split("?")[0])
    stem, ext = os.path.splitext(base)
    return f"{stem}_{hashlib.md5(u.encode()).hexdigest()[:6]}{ext}"


def route(u):
    if "huggingface.co" in u:
        return u.replace("https://huggingface.co", PROXY).replace(
            "http://huggingface.co", PROXY)
    if "raw.githubusercontent.com" in u:
        return "https://ghproxy.net/" + u
    return u


def fetch(url, dest):
    subprocess.run(["curl", "-sSL", "-m", "60", "-o", dest, url])
    return os.path.getsize(dest) if os.path.exists(dest) else 0


def main():
    OUT.mkdir(exist_ok=True)
    mapping = {}
    for u in collect_urls():
        name = local_name(u)
        dest = str(OUT / name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            mapping[u] = name
            print(f"[skip] {name}")
            continue
        print(f"[get ] {name} <- {route(u)[:70]}")
        sz = fetch(route(u), dest)
        if sz <= 1000 and "huggingface.co" in u:        # fallback mirror
            sz = fetch(u.replace("https://huggingface.co", "https://hf-mirror.com"), dest)
        if sz > 1000:
            os.chmod(dest, 0o644)
            mapping[u] = name
            print(f"       OK {sz}B")
        else:
            mapping[u] = None
            print(f"       FAIL ({sz}B)")
    (OUT / "resource_map.json").write_text(json.dumps(mapping, indent=2))
    ok = sum(1 for v in mapping.values() if v)
    print(f"\nwrote {OUT/'resource_map.json'} ({ok}/{len(mapping)} downloaded)")


if __name__ == "__main__":
    main()
