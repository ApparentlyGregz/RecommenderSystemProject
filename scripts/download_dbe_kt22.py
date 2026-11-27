"""Download the DBE-KT22 CSV files into data/raw/dbe-kt22/ (about 17 MB).

The official copy is on the Australian Data Archive (doi:10.26193/6DZWOH), which
needs a browser. This script uses the public mirror of the same raw files on
Hugging Face (datasets/Unggi/dbe-kt22_raw_data). If it is unavailable,
download the files from the archive by hand into data/raw/dbe-kt22/.

Usage:
    python scripts/download_dbe_kt22.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "dbe-kt22"
MIRROR = "https://huggingface.co/datasets/Unggi/dbe-kt22_raw_data/resolve/main/"
FILES = ["Transaction.csv", "Questions.csv", "Question_KC_Relationships.csv", "KCs.csv", "KC_Relationships.csv",
         "Question_Choices.csv", "Specialization.csv", "Student_Specialization.csv"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUT / name
        if target.exists() and target.stat().st_size > 0:
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading")
        urllib.request.urlretrieve(MIRROR + name, target)
    sizes = {n: (OUT / n).stat().st_size for n in FILES}
    if sizes["Transaction.csv"] < 10_000_000:
        sys.exit("Transaction.csv looks truncated; download it from doi:10.26193/6DZWOH instead.")
    print(f"Done: {sum(sizes.values()) / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
