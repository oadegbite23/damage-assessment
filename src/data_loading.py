"""
Data loading utilities for the xBD dataset (xView2 Challenge).

This project uses a 3-event subset (Hurricane Michael, Santa Rosa
Wildfire, Mexico City Earthquake) extracted from the official xBD
download (via xview2.org, Aspera Connect), rather than the full
19-event, 30.3GB dataset — see README.md for rationale (local storage
constraints).

Expected directory structure under `data/` (per the xView2 challenge
data format):

    data/
      train/
        images/
          <disaster>_<id>_pre_disaster.png
          <disaster>_<id>_post_disaster.png
          ...
        labels/
          <disaster>_<id>_pre_disaster.json   # building polygons
          <disaster>_<id>_post_disaster.json  # + damage labels
      test/
        images/
        labels/
      holdout/
        images/
        labels/

Each pre/post image pair shares an <id>. Label JSON files follow the
xBD schema: a "features" dict with "xy" (pixel-space polygons, what we
rasterize against) and "lng_lat" (geo-space, unused here) keys, plus a
"metadata" dict with the disaster name, image dimensions, etc. Each
feature's "properties" includes "feature_type" ("building") and, in
post-disaster labels only, "subtype" — the damage label:
    no-damage | minor-damage | major-damage | destroyed | un-classified
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from torch.utils.data import Dataset

# Initialize damage subtype -> integer class. 
# - Background/ no building =  0.
# "un-classified" (buildings visible but damage is unknown) — mapped to 0 (so they don't corrupt the loss)
# rest of damage subtypes (described in README.md) 0 mapped 1 -> 3

DAMAGE_CLASSES = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0,
}

TARGET_EVENTS = ["hurricane-michael", "santa-rosa-wildfire", "mexico-city-earthquake"]


@dataclass
class BuildingPolygon:
    polygon: Polygon
    #subtype: Optional[str] = None  # syntax for python < 3.10 
    subtype: str | None # None for pre-disaster labels (no damage info yet)


def parse_label_json(label_path: Path) -> tuple[list[BuildingPolygon], dict]:
    """Parse one xBD label JSON file into building polygons + metadata.

    Uses the "xy" (pixel-space) polygons, not "lng_lat" (geo-space),
    since we're rasterizing directly against the image grid rather
    than doing any geospatial reprojection.
    """
    with open(label_path) as f:
        data = json.load(f)

    buildings = []
    for feature in data["features"]["xy"]:
        if feature["properties"].get("feature_type") != "building":
            continue
        polygon = wkt.loads(feature["wkt"])
        subtype = feature["properties"].get("subtype")  # absent in pre-disaster labels
        buildings.append(BuildingPolygon(polygon=polygon, subtype=subtype))

    return buildings, data["metadata"]


def polygons_to_masks(
    buildings: list[BuildingPolygon], height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize building polygons into two masks:

    - localization_mask: binary (0/1), building vs. not — used for
      the pre-disaster image and as the "is there a building here"
      signal regardless of damage state.
    - damage_mask: 0-4 (background, no-damage, minor, major,
      destroyed) — only meaningful for post-disaster labels; will be
      all zeros if `buildings` came from a pre-disaster label file
      (subtype is None for all of them).

    NOTE: overlapping polygons will overwrite each other in whichever
    order they're drawn — this is rare in xBD but worth knowing about
    if you see unexpected boundary artifacts later.
    """
    localization_mask = np.zeros((height, width), dtype=np.uint8)
    damage_mask = np.zeros((height, width), dtype=np.uint8)

    for b in buildings:
        coords = np.array(b.polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(localization_mask, [coords], 1)

        if b.subtype is not None:
            class_id = DAMAGE_CLASSES.get(b.subtype, 0)
            cv2.fillPoly(damage_mask, [coords], class_id)

    return localization_mask, damage_mask


class XBDDataset(Dataset):
    """Paired pre/post-disaster image + mask dataset for one xBD split
    (train/test/holdout), optionally filtered to specific disaster
    events.

    Returns, per item: pre_image, post_image, localization_mask,
    damage_mask (all as numpy arrays; wrap with transforms/ToTensor
    in train.py once the transform pipeline is defined).
    """

    def __init__(self, split_dir: Path, events: list[str] | None = None, transform=None):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.transform = transform

        # Each pair is identified by the shared <disaster>_<id> prefix
        # of its filenames; we key off the post-disaster label file
        # since that's the one with damage subtypes.
        post_labels = sorted(self.labels_dir.glob("*_post_disaster.json"))
        if events is not None:
            post_labels = [p for p in post_labels if any(e in p.name for e in events)]
        self.sample_ids = [p.name.replace("_post_disaster.json", "") for p in post_labels]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]

        pre_image = cv2.imread(str(self.images_dir / f"{sample_id}_pre_disaster.png"))
        post_image = cv2.imread(str(self.images_dir / f"{sample_id}_post_disaster.png"))
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        # Localization mask comes from the pre-disaster label (buildings
        # as they existed before the disaster); damage mask comes from
        # the post-disaster label (same footprints + damage subtype).
        pre_buildings, pre_meta = parse_label_json(
            self.labels_dir / f"{sample_id}_pre_disaster.json"
        )
        post_buildings, post_meta = parse_label_json(
            self.labels_dir / f"{sample_id}_post_disaster.json"
        )
        height, width = pre_meta["height"], pre_meta["width"]

        localization_mask, _ = polygons_to_masks(pre_buildings, height, width)
        _, damage_mask = polygons_to_masks(post_buildings, height, width)

        sample = {
            "pre_image": pre_image,
            "post_image": post_image,
            "localization_mask": localization_mask,
            "damage_mask": damage_mask,
            "disaster": post_meta.get("disaster", ""),
            "sample_id": sample_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def event_split(
    split_dir: Path,
    val_events: list[str],
    test_events: list[str],
    train_events: list[str] = TARGET_EVENTS,
) -> tuple[XBDDataset, XBDDataset, XBDDataset]:
    """Build train/val/test datasets split BY DISASTER EVENT rather
    than randomly.

    This is the leakage-aware approach: a random split would put
    images from the same disaster (often the same geographic area,
    similar building types and imagery conditions) into both train
    and val/test, inflating apparent performance. Splitting by event
    tests actual generalization to unseen disasters instead.

    With only 3 target events, this means picking e.g. 1 event for
    val, 1 for test, and training on the remaining — a small-N
    limitation worth stating plainly in the README's Results section
    once you have numbers, not something to gloss over.
    """
    train = [e for e in train_events if e not in val_events and e not in test_events]
    return (
        XBDDataset(split_dir, events=train),
        XBDDataset(split_dir, events=val_events),
        XBDDataset(split_dir, events=test_events),
    )
