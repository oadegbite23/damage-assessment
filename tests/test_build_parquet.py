
import argparse
from pathlib import Path
import cv2
import pandas as pd
import pyarrow
from data_loading import TARGET_EVENTS, parse_label_json, polygons_to_masks
import glob
import pytest

class build_parquet_wrapper():
    def __init__(self, data_dir:Path, split:str):
        self.data_dir = Path(data_dir)
        self.split = Path(split)


    def find_gt_labels(self):
        split_dir = self.data_dir / self.split
        images_dir = split_dir / Path('images')
        labels_dir = split_dir / Path('labels')

        post_labels = sorted(labels_dir.glob("*_post_disaster.json")) # find all the ground truth labels 
        #post_labels = [p for p in post_labels if any(e in p.name for e in events)] # event + location pairing
        self.post_label_path = post_labels[0]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.split_dir = split_dir
        #return post_labels[0], images_dir, labels_dir

    def load_images(self): #, post_label_path: str, images_dir: str, labels_dir: str):
        #for post_label_path in post_labels:
        sample_id = self.post_label_path.name.replace("_post_disaster.json","") # grab event
        self.sample_id = sample_id

        pre_image_path = self.images_dir / f"{sample_id}_pre_disaster.png" # pre disaster image
        self.pre_image_path = pre_image_path
        post_image_path = self.images_dir / f"{sample_id}_post_disaster.png" #post disaster image
        self.post_image_path = post_image_path
        pre_label_path = self.labels_dir / f"{sample_id}_pre_disaster.json" #ground truth label
        self.pre_label_path = pre_label_path


        # Read original image bytes as-is (no decode/reencode)
        pre_image_bytes = pre_image_path.read_bytes()
        self.pre_image_bytes = pre_image_bytes
        post_image_bytes = post_image_path.read_bytes()
        self.post_image_bytes = post_image_bytes
        #return pre_image_bytes, post_image_bytes, pre_label_path, sample_id

    def image_masks(self):
        pre_buildings, pre_meta = parse_label_json(self.pre_label_path)
        post_buildings, post_meta = parse_label_json(self.post_label_path)
        height, width = pre_meta["height"], pre_meta["width"]

        localization_mask, _ = polygons_to_masks(pre_buildings, height, width)
        _, damage_mask = polygons_to_masks(post_buildings, height, width)

        success_loc, loc_encoded = cv2.imencode(".png", localization_mask)
        success_dmg, dmg_encoded = cv2.imencode(".png", damage_mask)
        self.success_loc = success_loc
        self.success_dmg = success_dmg
        #return success_loc, loc_encoded, success_dmg, dmg_encoded

    def image_masks_store_in_parquet(self):
        pre_buildings, pre_meta = parse_label_json(self.pre_label_path)
        post_buildings, post_meta = parse_label_json(self.post_label_path)
        height, width = pre_meta["height"], pre_meta["width"]

        localization_mask, _ = polygons_to_masks(pre_buildings, height, width)
        _, damage_mask = polygons_to_masks(post_buildings, height, width)

        success_loc, loc_encoded = cv2.imencode(".png", localization_mask)
        success_dmg, dmg_encoded = cv2.imencode(".png", damage_mask)

        rows = []
        rows.append(
                            {
                                "sample_id": self.sample_id,
                                "disaster": str(post_meta.get("disaster", "")),
                                "split": str(self.split),
                                "height": height,
                                "width": width,
                                "pre_image_bytes": self.pre_image_bytes,
                                "post_image_bytes": self.post_image_bytes,
                                "localization_mask_bytes": loc_encoded.tobytes(),
                                "damage_mask_bytes": dmg_encoded.tobytes(),
                            }
                        )
        
        df = pd.DataFrame(rows)
        out_dir = self.data_dir / "parquet"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{self.split}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy") # export to parquet

        print(f"Wrote {len(df)} samples to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path




def test_find_gt_labels():
    # find_gt_labels
    #post_labels, images_dir, labels_dir = bpw.find_gt_labels()
    bpw.find_gt_labels()
    assert isinstance(bpw.post_label_path,Path), 'post disaster labels are not being extracted correctly'
    assert bpw.images_dir == Path('/Users/oadegbite/Documents/AI_Projects/damage-assessment/data/train/images'), 'issue with string concatenation'
    assert bpw.labels_dir == Path('/Users/oadegbite/Documents/AI_Projects/damage-assessment/data/train/labels'), 'issue with string conccatenation'
    assert bpw.images_dir.exists()
    assert bpw.labels_dir.exists()
    #return post_labels, images_dir, labels_dir


def test_load_images():
    # load_images
    #pre_image_bytes, post_image_bytes, pre_label_path, sample_id = bpw.load_images(post_label_path, images_dir, labels_dir)
    bpw.load_images()
    assert isinstance(bpw.pre_image_bytes, bytes), 'incorrect loading, issue with cv2.imencode()'
    assert isinstance(bpw.post_image_bytes, bytes), 'incorrect loading, issue with cv2.imencode()'
    #return pre_image_bytes, post_image_bytes, pre_label_path, sample_id

def test_image_masks():
    #image_masks #compresses image matrix and stores in RAM instead of hard drive
    #store all necessary info in parquet  
    #success_loc, loc_encoded, success_dmg, dmg_encoded = bpw.image_masks(pre_label_path, post_label_path)
    bpw.image_masks()
    assert bpw.success_loc, 'Failed to encode location image mask with .png extension'
    assert bpw.success_dmg, 'Failed to encode damage image mask with .png extension'
    #return loc_encoded, dmg_encoded

def test_image_masks_store_in_parquet():
    #image_masks #compresses image matrix and stores in RAM instead of hard drive
    #store all necessary info in parquet  
    #out_path = bpw.image_masks_store_in_parquet(sample_id, pre_label_path, post_label_path, pre_image_bytes, post_image_bytes)
    #success_loc, loc_encoded, success_dmg, dmg_encoded = image_masks(pre_label_path, post_label_path)
    output_path = bpw.image_masks_store_in_parquet()
    assert output_path.exists(), 'Parquet file not correctly constructed or saved'
    #return out_path

bpw = build_parquet_wrapper('/Users/oadegbite/Documents/AI_Projects/damage-assessment/data', 'train')

#post_label_path, images_dir, labels_dir = test_find_gt_labels()

#print('POST LABEL PATH:',post_label_path[0])
#print("IMAGES DIR:", images_dir)
#print("LABELS DIR:",labels_dir)
#pre_image_bytes, post_image_bytes, pre_label_path, sample_id = test_load_images(post_label_path,images_dir,labels_dir)
#_,_ = test_image_masks(pre_label_path, post_label_path)
#output_path = test_image_masks_store_in_parquet(sample_id, pre_label_path, post_label_path, pre_image_bytes, post_image_bytes)

test_find_gt_labels()
test_load_images()
test_image_masks()
test_image_masks_store_in_parquet()





