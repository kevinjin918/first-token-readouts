"""Download NIH ChestX-ray14 into the TraceCXR data root.

Pulls from the public HuggingFace mirror ``alkzar90/NIH-Chest-X-ray-dataset`` (not gated)
and lays the files out exactly where ``tracecxr.data.chestxray14`` expects them:

    <data_root>/chestxray14/
        Data_Entry_2017.csv      (mirror ships it as Data_Entry_2017_v2020.csv)
        BBox_List_2017.csv
        images/*.png

``data_root`` is ``tracecxr.core.config.data_root()`` (override with ``TRACECXR_DATA_DIR``).

Modes:
    --sample   CSVs + images_001.zip (~2 GB, ~5k images incl. some bbox images). For local
               dev and the M3 smoke test.
    --full     CSVs + all 12 image zips (~42 GB). Run this on the GPU/cloud box.

Usage:
    python scripts/download_nih.py --sample
    python scripts/download_nih.py --full
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Allow running as `python scripts/download_nih.py` from the repo root, where the script's
# own directory (not the repo root) is what lands on sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download  # noqa: E402
from tracecxr.core.config import data_root  # noqa: E402

REPO_ID = "alkzar90/NIH-Chest-X-ray-dataset"
ALL_ZIPS = [f"data/images/images_{i:03d}.zip" for i in range(1, 13)]
CSVS = {
    "data/BBox_List_2017.csv": "BBox_List_2017.csv",
    "data/Data_Entry_2017_v2020.csv": "Data_Entry_2017.csv",
}


def _fetch(repo_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
    dest.write_bytes(Path(cached).read_bytes())
    print(f"  {repo_path} -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="CSVs + images_001.zip (~2 GB)")
    g.add_argument("--full", action="store_true", help="CSVs + all 12 zips (~42 GB)")
    args = ap.parse_args()

    out = data_root() / "chestxray14"
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {out}")
    print("Downloading metadata CSVs...")
    for repo_path, name in CSVS.items():
        _fetch(repo_path, out / name)

    zips = ALL_ZIPS if args.full else ALL_ZIPS[:1]
    print(f"Downloading {len(zips)} image zip(s)...")
    for repo_path in zips:
        print(f"  fetching {repo_path} ...")
        cached = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
        with zipfile.ZipFile(cached) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".png")]
            for m in members:
                target = images_dir / Path(m).name
                if not target.exists():
                    with zf.open(m) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        print(f"    extracted {len(members)} images")

    n_images = sum(1 for _ in images_dir.glob("*.png"))
    print(f"Done. {n_images} images under {images_dir}")


if __name__ == "__main__":
    main()
