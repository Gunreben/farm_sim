"""Download the CC0 Poly Haven assets listed in config/polyhaven_assets.yaml.

    python3 tools/fetch_polyhaven.py [--dry-run]

Layout: assets/polyhaven/{hdris,textures,models}/<name>/... Existing files with the right size
are skipped, so re-running only fetches what is missing. Uses the public API (api.polyhaven.com).
"""

import argparse
import json
import os
import urllib.request

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "polyhaven")
UA = {"User-Agent": "farm_sim-asset-fetch/0.1 (research simulation)"}


def _get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def _fetch(url, dest, size, dry):
    if os.path.exists(dest) and (not size or os.path.getsize(dest) == size):
        return 0
    if dry:
        print(f"  would fetch {url} ({size / 1e6:.1f} MB)")
        return size or 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "polyhaven_assets.yaml")))
    total, credits = 0, []

    for name in cfg["hdris"]["names"]:
        files = _get_json(f"https://api.polyhaven.com/files/{name}")
        f = files["hdri"][cfg["hdris"]["resolution"]]["hdr"]
        total += _fetch(f["url"], os.path.join(OUT, "hdris", f"{name}.hdr"), f["size"], args.dry_run)
        credits.append(("hdri", name))

    res = cfg["textures"]["resolution"]
    for name in cfg["textures"]["names"]:
        files = _get_json(f"https://api.polyhaven.com/files/{name}")
        for m in cfg["textures"]["maps"]:
            if m not in files:
                continue
            f = files[m][res]["jpg"]
            total += _fetch(f["url"], os.path.join(OUT, "textures", name, f"{name}_{m}.jpg"), f["size"], args.dry_run)
        credits.append(("texture", name))

    res = cfg["models"]["resolution"]
    for name in cfg["models"]["names"]:
        files = _get_json(f"https://api.polyhaven.com/files/{name}")
        entry = files["usd"][res]["usd"]
        base = os.path.join(OUT, "models", name)
        total += _fetch(entry["url"], os.path.join(base, os.path.basename(entry["url"])), entry["size"], args.dry_run)
        for rel, inc in entry.get("include", {}).items():
            total += _fetch(inc["url"], os.path.join(base, rel), inc["size"], args.dry_run)
        credits.append(("model", name))

    if not args.dry_run:
        with open(os.path.join(OUT, "CREDITS.md"), "w") as f:
            f.write("# Poly Haven assets (CC0)\n\nAll assets below are CC0 from https://polyhaven.com.\n\n")
            for kind, name in credits:
                f.write(f"- {kind}: [{name}](https://polyhaven.com/a/{name})\n")
    print(f"{'would download' if args.dry_run else 'downloaded'} {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
