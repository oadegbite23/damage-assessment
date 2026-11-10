"""
Data storage pre-processing: convert raw xBD files (images + label JSON)
into parquet files, once per split (train/test/holdout).

Run this once after downloading the datasets from xview2.org. The parquet files 
(XBDParquetDataset) produced from this script will be used in data_loading.py as 
opposed to the raw image files. The raw PNGs/JSONS can then be deleted to reclaim 
disk space as everything needed for training is captured in the parquet file. 

Usage:
    python src/build_parquet.py --data-dir data --split train


Design notes:
- Images are stored as their ORIGINAL ENCODED BYTES (read directly off
  disk, not decoded-then-reencoded) — this is lossless and avoids
  unnecessary recompression. Masks don't exist as files, so those ARE
  encoded (PNG) from the rasterized numpy arrays.
- Decoding happens later, in XBDParquetDataset.__getitem__ — this
  script only handles the one-time raw-files -> parquet conversion.
"""

import argparse
from pathlib import Path
import cv2
import pandas as pd
import pyarrow
from data_loading import TARGET_EVENTS, parse_label_json, polygons_to_masks
import glob

def build_parquet(data_dir: Path, split: str):
    split_dir = data_dir/Path(split)
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'

    post_labels = sorted(labels_dir.glob("*_post_disaster.json")) # find all the ground truth labels 
    
    #post_labels = [p for p in post_labels if any(e in p.name for e in events)] # event + location pairing

    rows = []
    for post_label_path in post_labels:
        sample_id = post_label_path.name.replace("_post_disaster.json","") # grab event

        pre_image_path = images_dir / f"{sample_id}_pre_disaster.png" # pre disaster image
        post_image_path = images_dir / f"{sample_id}_post_disaster.png" #post disaster image
        pre_label_path = labels_dir / f"{sample_id}_pre_disaster.json" #ground truth label

        # Read original image bytes as-is (no decode/reencode)
        pre_image_bytes = pre_image_path.read_bytes()
        post_image_bytes = post_image_path.read_bytes()

        pre_buildings, pre_meta = parse_label_json(pre_label_path)
        post_buildings, post_meta = parse_label_json(post_label_path)
        height, width = pre_meta["height"], pre_meta["width"]

        localization_mask, _ = polygons_to_masks(pre_buildings, height, width)
        _, damage_mask = polygons_to_masks(post_buildings, height, width)

        _, loc_encoded = cv2.imencode(".png", localization_mask) #compresses image matrix and stores in RAM instead of hard drive 
        _, dmg_encoded = cv2.imencode(".png", damage_mask)


        # info to be stored in parquet
        rows.append(
                        {
                            "sample_id": sample_id,
                            "disaster": post_meta.get("disaster", ""),
                            "split": split,
                            "height": height,
                            "width": width,
                            "pre_image_bytes": pre_image_bytes,
                            "post_image_bytes": post_image_bytes,
                            "localization_mask_bytes": loc_encoded.tobytes(),
                            "damage_mask_bytes": dmg_encoded.tobytes(),
                        }
                    )
    
    df = pd.DataFrame(rows)
    out_dir = data_dir / "parquet"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{split}.parquet"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy") # export to parquet

    print(f"Wrote {len(df)} samples to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path('/Users/oadegbite/Documents/AI_Projects/damage-assessment/data'))
    parser.add_argument("--split", required= True, choices= ['train','test','hold'])
    args=parser.parse_args()
    build_parquet(args.data_dir, args.split)
