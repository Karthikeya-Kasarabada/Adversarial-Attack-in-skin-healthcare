import os, shutil
import pandas as pd
from pathlib import Path
import config

def main():
    print("[demo] Building tiny demo dataset for Streamlit Cloud...")
    meta_path = Path('data/raw/HAM10000_metadata.csv')
    if not meta_path.exists():
        print("Metadata not found!")
        return

    meta = pd.read_csv(meta_path)
    demo_dir = Path('data/demo_images')
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    demo_rows = []
    for cls in config.CLASS_NAMES:
        # Get first image of this class
        matches = meta[meta['dx'] == cls]
        if matches.empty: continue
        row = matches.iloc[0]
        img_id = row['image_id']
        
        # Find the image
        src1 = Path('data/raw/HAM10000_images_part_1') / f'{img_id}.jpg'
        src2 = Path('data/raw/HAM10000_images_part_2') / f'{img_id}.jpg'
        src = src1 if src1.exists() else src2
        
        if src.exists():
            dest = demo_dir / f'{img_id}.jpg'
            shutil.copy(src, dest)
            demo_rows.append(row)
            print(f"  Copied {img_id}.jpg ({cls})")
            
    # Save the mini metadata
    pd.DataFrame(demo_rows).to_csv(demo_dir / 'demo_metadata.csv', index=False)
    print(f"[demo] Wrote demo metadata with {len(demo_rows)} images to {demo_dir}/demo_metadata.csv")

if __name__ == '__main__':
    main()
