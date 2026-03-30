"""
download_data.py – Download HAM10000 from Kaggle and organise into data/raw/.

Prerequisites:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
      OR place ~/.kaggle/kaggle.json

Usage:
    python download_data.py
"""

import os
import zipfile
from pathlib import Path
import shutil

import config


def download_from_kaggle():
    """Download and extract HAM10000 via the Kaggle API."""
    raw_dir = config.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[download] Downloading HAM10000 from Kaggle …")
    os.system(
        f'kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 '
        f'--path "{raw_dir}" --unzip'
    )
    print("[download] Done.")
    _post_process(raw_dir)


def _post_process(raw_dir: Path):
    """Flatten extracted folders if needed and verify metadata CSV."""

    # Some Kaggle zips put images in a nested folder – flatten.
    for sub in raw_dir.rglob("*.jpg"):
        parent_name = sub.parent.name
        if "part_1" in parent_name or "part_2" in parent_name:
            # Already in correct sub-folder
            continue
        # Move into part_1 by default
        dest_dir = raw_dir / "HAM10000_images_part_1"
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / sub.name
        if not dest.exists():
            shutil.move(str(sub), str(dest))

    # Rename metadata CSV if needed
    candidates = list(raw_dir.glob("*metadata*.csv")) + list(raw_dir.glob("HAM10000_metadata.csv"))
    if candidates and not config.METADATA_CSV.exists():
        shutil.copy(str(candidates[0]), str(config.METADATA_CSV))
        print(f"[download] Metadata → {config.METADATA_CSV}")

    # Verify
    if config.METADATA_CSV.exists():
        import pandas as pd
        meta = pd.read_csv(config.METADATA_CSV)
        print(f"[download] Metadata loaded: {len(meta)} rows, columns: {list(meta.columns)}")
    else:
        print("[download] WARNING: metadata CSV not found. Check data/raw/ manually.")

    jpg_count = len(list(raw_dir.rglob("*.jpg")))
    print(f"[download] Total .jpg files found: {jpg_count}")


def download_from_isic():
    """
    Fallback: Download subset via ISIC API v2 (no authentication required).
    Downloads a mix of images for quick testing.
    """
    import urllib.request
    import json
    import time
    import pandas as pd

    # Updated mapping for various ISIC diagnosis strings
    label_map = {
        "actinic keratosis": "akiec",
        "basal cell carcinoma": "bcc",
        "benign keratosis": "bkl",
        "dermatofibroma": "df",
        "melanoma": "mel",
        "nevus": "nv",
        "vascular lesion": "vasc",
        "pigmented benign keratosis": "bkl",
        "seborrheic keratosis": "bkl",
        "squamous cell carcinoma": "akiec"
    }

    # Use a simpler endpoint that is less likely to fail with 400
    # Search for images with any diagnosis info
    search_url = "https://api.isic-archive.com/api/v2/images/?limit=200&has_tabular=true"

    raw_dir = config.RAW_DIR / "HAM10000_images_part_1"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[download] Fetching image list from ISIC API v2 …")
    try:
        req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            items = data.get("results", [])
    except Exception as e:
        print(f"[download] ISIC API error: {e}")
        return

    if not items:
        print("[download] No images found via ISIC API.")
        return

    print(f"[download] Found {len(items)} images metadata. Downloading images …")
    meta_rows = []
    total = 0

    for item in items:
        isic_id  = item["isic_id"]
        clinical = item.get("metadata", {}).get("clinical", {})
        
        # In v2, diagnosis is hierarchical. Try to find the most specific one.
        diag_raw = clinical.get("diagnosis_5") or clinical.get("diagnosis_4") or \
                   clinical.get("diagnosis_3") or clinical.get("diagnosis_2") or \
                   clinical.get("diagnosis_1") or "nevus"
        
        dx = label_map.get(str(diag_raw).lower(), "nv")
        
        # Download image - use thumbnail_256
        img_url = item.get("files", {}).get("thumbnail_256", {}).get("url")
        if not img_url:
             continue

        dest = raw_dir / f"{isic_id}.jpg"
        if not dest.exists():
            try:
                time.sleep(0.05)
                req_img = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_img, timeout=20) as r_img:
                    with open(dest, 'wb') as f:
                        f.write(r_img.read())
            except Exception:
                continue

        meta_rows.append({
            "image_id": isic_id,
            "dx": dx,
            "dx_type": "histo",
            "age": clinical.get("age_approx", 50),
            "sex": clinical.get("sex", "male"),
            "localization": clinical.get("anatom_site_general", "unknown")
        })
        total += 1
        if total % 20 == 0:
            print(f"  downloaded {total}/{len(items)} …")

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(config.METADATA_CSV, index=False)
        print(f"[download] ISIC fallback done. {len(meta_df)} images -> {config.METADATA_CSV}")
    else:
        print("[download] FAILED: No metadata rows could be processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["kaggle", "isic"], default="kaggle",
        help="Where to download from (default: kaggle)"
    )
    args = parser.parse_args()

    if args.source == "kaggle":
        download_from_kaggle()
    else:
        download_from_isic()
