"""
Download the third-party raw datasets used by this benchmark.

These are NOT redistributed in this repository. They belong to their
original authors and are fetched directly from their public sources.

  Santa Fe laser  - Hubner, Abraham & Weiss, Phys. Rev. A 40, 6354 (1989)
  Bonn EEG        - Andrzejak et al., Phys. Rev. E 64, 061907 (2001)

Run this once before build_final_dataset.py.
"""
import urllib.request, pathlib, sys

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

SOURCES = {
    "laser_raw.txt": "https://raw.githubusercontent.com/MaterialMan/CHARC/master/Support%20files/other/Datasets/laser.txt",
    "eeg_raw.csv":   "https://raw.githubusercontent.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/master/A%26B%26C%26D%26E.csv",
}

for name, url in SOURCES.items():
    dest = DATA / name
    if dest.exists():
        print(f"{name} already present, skipping")
        continue
    print(f"downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  saved to {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        print(f"  fetch manually from: {url}", file=sys.stderr)
