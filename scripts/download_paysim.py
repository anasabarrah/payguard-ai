"""Download the PaySim dataset (~470 MB) and verify its checksum.

PaySim (Lopez-Rojas et al., 2016) is published on Kaggle as ealaxi/paysim1.
This uses a public Hugging Face mirror so no Kaggle account is needed; the
SHA-256 matches the mirror's Git LFS object.

Usage: python scripts/download_paysim.py [--out data/raw/paysim.csv]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

URL = ("https://huggingface.co/datasets/vitaliy-sharandin/synthetic-fraud-detection/"
       "resolve/main/PS_20174392719_1491204439457_log.csv")
SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "paysim.csv")
    args = ap.parse_args()
    out: Path = args.out
    if out.exists() and sha256(out) == SHA256:
        print(f"[ok] {out} already present and verified")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    print(f"downloading PaySim to {out} ...")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if sha256(tmp) != SHA256:
        tmp.unlink()
        print("[error] checksum mismatch, file discarded", file=sys.stderr)
        return 1
    tmp.replace(out)
    print("[ok] downloaded and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
