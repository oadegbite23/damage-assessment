"""
Data loading utilities for the xBD dataset (xView2 Challenge).

Works with either the full xBD dataset (19 events, ~30GB) or a subset
such as the 3 events in TARGET_EVENTS (Hurricane Michael, Santa Rosa
Wildfire, Mexico City Earthquake). For the full dataset, convert the
raw files once with build_parquet.py and train from
XBDParquetDataset — it reads lazily, so memory stays flat as the
dataset grows. The raw-file XBDDataset is fine for small subsets and
for inspecting individual samples.

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
      hold/
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
import random
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from torch.utils.data import Dataset, Sampler

# Damage subtype -> integer class. Background (no building) is 0.
# "un-classified" buildings exist in the data (visible but damage
# unknown) — mapped to 0 here so they don't corrupt the loss; revisit
# this choice once you've checked how common they are in your 3 events.
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
    subtype: str | None  # None for pre-disaster labels (no damage info yet)


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
    (train/hold/test), optionally filtered to specific disaster
    events.

    Returns, per item: pre_image, post_image, localization_mask,
    damage_mask (all as numpy arrays; wrap with transforms/ToTensor
    in train.py once the transform pipeline is defined).
    """

    def __init__(
        self,
        split_dir: Path,
        events: list[str] | None = None,
        transform=None,
        exclude_events: list[str] | None = None,
    ):
        """events=None means ALL events in the split; pass a list (e.g.
        TARGET_EVENTS) to restrict. exclude_events drops events after
        that. Matching is by substring of the filename."""
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
        if exclude_events:
            post_labels = [p for p in post_labels if not any(e in p.name for e in exclude_events)]
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


# The xView2 challenge ships three fixed partitions. We use them as-is
# (no re-splitting) so results are comparable with other teams' scores.
OFFICIAL_SPLITS = ("train", "hold", "test")


def official_splits(
    data_dir: Path,
    train_events: list[str] | None = None,
    transform=None,
) -> tuple[XBDDataset, XBDDataset, XBDDataset]:
    """Raw-file datasets for the challenge's own partitions, returned as
    (train, hold, test) from data_dir/train, data_dir/hold, data_dir/test.

    Nothing is re-split here. Use train to fit, hold to tune and pick
    checkpoints, and test only for the final reported number.

    train_events restricts ONLY the training partition (None = all
    events). hold and test always keep every event, so models trained
    on different subsets are all scored on the same data and stay
    roughly comparable with published results. For a per-event score,
    break the metric down by event (see evaluate.py) rather than
    filtering the evaluation sets.
    """
    data_dir = Path(data_dir)
    return (
        XBDDataset(data_dir / "train", events=train_events, transform=transform),
        XBDDataset(data_dir / "hold", transform=transform),
        XBDDataset(data_dir / "test", transform=transform),
    )


def build_row_index(group_sizes: list[int], keep) -> list[tuple[int, int]]:
    """Map kept rows of ONE parquet file to (row_group, offset_in_group).

    Args:
        group_sizes: number of rows in each row group, in file order
            (from parquet metadata — no data is read).
        keep: per-row booleans in file order, same total length as
            sum(group_sizes) (True = row passes the event filter).

    Pure function (no I/O) so it can be unit-tested without pyarrow.
    """
    if sum(group_sizes) != len(keep):
        raise ValueError(f"keep has {len(keep)} flags but row groups hold {sum(group_sizes)} rows")
    index = []
    flat = 0
    for group, n_rows in enumerate(group_sizes):
        for offset in range(n_rows):
            if keep[flat + offset]:
                index.append((group, offset))
        flat += n_rows
    return index


# Parquet reads whole row groups. A group bigger than this is loaded into
# memory in full (once per DataLoader worker) every time it's touched.
LARGE_ROW_GROUP_BYTES = 256 * 1024 * 1024


def find_oversized_row_groups(byte_sizes: list[int], limit_bytes: int = LARGE_ROW_GROUP_BYTES) -> list[int]:
    """Indices of row groups whose (uncompressed) size exceeds limit_bytes.

    Pure function (no I/O) so it can be unit-tested without pyarrow.
    """
    return [g for g, size in enumerate(byte_sizes) if size > limit_bytes]


class XBDParquetDataset(Dataset):
    """Like XBDDataset, but reads from parquet files produced by
    build_parquet.py — built to scale to the full xBD dataset.

    Memory behavior:
    - __init__ reads parquet METADATA plus the small `disaster` column
      only. No image bytes are loaded up front, so init cost and RAM
      stay small regardless of dataset size.
    - Files are opened lazily, once per process (so each DataLoader
      worker gets its own handle), and are dropped on pickling.
    - __getitem__ reads one row group (only the payload columns) and
      keeps the last `cache_row_groups` of them in a small LRU cache.

    Random-access caveat: parquet can only read WHOLE row groups. With
    fully random shuffling, every sample costs a full row-group read
    (build_parquet's default is 16 rows -> ~16x read amplification;
    the cache rarely helps). Use RowGroupShuffleSampler below to shuffle
    at the row-group level instead, and set `cache_row_groups` >= its
    `groups_per_buffer`.

    Args:
        parquet_paths: a parquet file, a directory of *.parquet files,
            or a list of either. A directory loads every partition in it
            together, so for train/hold/test use official_splits_parquet
            below instead.
        events: keep only these disasters (exact match on the
            `disaster` column, e.g. "hurricane-michael"). None = all.
        exclude_events: drop these disasters. None = drop nothing.
    """

    _PAYLOAD_COLUMNS = [
        "pre_image_bytes",
        "post_image_bytes",
        "localization_mask_bytes",
        "damage_mask_bytes",
    ]

    def __init__(
        self,
        parquet_paths,
        events: list[str] | None = None,
        exclude_events: list[str] | None = None,
        transform=None,
        cache_row_groups: int = 2,
    ):
        import pyarrow.parquet as pq  # lazy: keeps module import light

        if isinstance(parquet_paths, (str, Path)):
            parquet_paths = [parquet_paths]
        self.paths: list[Path] = []
        for p in map(Path, parquet_paths):
            self.paths.extend(sorted(p.glob("*.parquet")) if p.is_dir() else [p])
        if not self.paths:
            raise FileNotFoundError(f"No parquet files found in {parquet_paths}")

        self.transform = transform
        self.cache_row_groups = cache_row_groups
        self._files: dict[int, "pq.ParquetFile"] = {}
        self._group_cache: OrderedDict = OrderedDict()

        keep_events = set(events) if events is not None else None
        drop_events = set(exclude_events) if exclude_events is not None else set()

        # _index[i] = (file_idx, row_group, offset_in_group) for dataset item i
        self._index: list[tuple[int, int, int]] = []
        self.disasters: list[str] = []  # disaster name per dataset item
        self.sample_ids: list[str] = []  # sample_id per dataset item
        for file_idx, path in enumerate(self.paths):
            pf = pq.ParquetFile(path)
            meta = pf.metadata
            group_sizes = [meta.row_group(g).num_rows for g in range(meta.num_row_groups)]
            byte_sizes = [meta.row_group(g).total_byte_size for g in range(meta.num_row_groups)]
            oversized = find_oversized_row_groups(byte_sizes)
            if oversized:
                biggest = byte_sizes.index(max(byte_sizes))
                warnings.warn(
                    f"{path.name}: {len(oversized)} of {meta.num_row_groups} row group(s) are "
                    f"larger than {LARGE_ROW_GROUP_BYTES // 2**20} MB (largest ≈ "
                    f"{byte_sizes[biggest] / 1e9:.1f} GB, {group_sizes[biggest]} rows). Parquet "
                    "reads whole row groups, so touching any sample loads that entire group "
                    "into memory, separately in each DataLoader worker. Fine if the file fits "
                    "comfortably in RAM (use num_workers=0 or 1); otherwise rewrite it with "
                    "smaller row groups, e.g. df.to_parquet(..., row_group_size=16).",
                    stacklevel=2,
                )
            # Columnar read of two small string columns — doesn't touch image bytes
            labels = pf.read(columns=["sample_id", "disaster"])
            file_ids = labels.column("sample_id").to_pylist()
            file_disasters = labels.column("disaster").to_pylist()
            keep = [
                (keep_events is None or d in keep_events) and d not in drop_events
                for d in file_disasters
            ]
            for group, offset in build_row_index(group_sizes, keep):
                self._index.append((file_idx, group, offset))
            self.disasters.extend(d for d, k in zip(file_disasters, keep) if k)
            self.sample_ids.extend(s for s, k in zip(file_ids, keep) if k)

    # Lets samplers group items by row group without touching the data
    @property
    def group_ids(self) -> list[tuple[int, int]]:
        return [(f, g) for f, g, _ in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def __getstate__(self):
        # Open file handles / cached tables shouldn't cross process
        # boundaries (DataLoader workers); each process re-opens lazily.
        state = self.__dict__.copy()
        state["_files"] = {}
        state["_group_cache"] = OrderedDict()
        return state

    def _read_row(self, file_idx: int, group: int, offset: int) -> dict:
        import pyarrow.parquet as pq

        key = (file_idx, group)
        table = self._group_cache.get(key)
        if table is None:
            pf = self._files.get(file_idx)
            if pf is None:
                pf = self._files[file_idx] = pq.ParquetFile(self.paths[file_idx])
            table = pf.read_row_group(group, columns=self._PAYLOAD_COLUMNS)
            self._group_cache[key] = table
            while len(self._group_cache) > self.cache_row_groups:
                self._group_cache.popitem(last=False)  # evict least recently used
        else:
            self._group_cache.move_to_end(key)
        return table.slice(offset, 1).to_pylist()[0]

    def __getitem__(self, idx: int):
        file_idx, group, offset = self._index[idx]
        row = self._read_row(file_idx, group, offset)

        pre_image = cv2.imdecode(
            np.frombuffer(row["pre_image_bytes"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        post_image = cv2.imdecode(
            np.frombuffer(row["post_image_bytes"], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        localization_mask = cv2.imdecode(
            np.frombuffer(row["localization_mask_bytes"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        damage_mask = cv2.imdecode(
            np.frombuffer(row["damage_mask_bytes"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )

        sample = {
            "pre_image": pre_image,
            "post_image": post_image,
            "localization_mask": localization_mask,
            "damage_mask": damage_mask,
            "disaster": self.disasters[idx],
            "sample_id": self.sample_ids[idx],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class RowGroupShuffleSampler(Sampler):
    """Shuffle for XBDParquetDataset that respects parquet's row groups.

    Each epoch: shuffle the order of row groups, take `groups_per_buffer`
    of them at a time, and shuffle the samples within that buffer. Every
    sample still appears exactly once per epoch, but reads stay
    localized (each row group is read ~once per epoch instead of once
    per sample). Set the dataset's `cache_row_groups` >= groups_per_buffer.

    Tradeoff: batches are drawn from a window of groups_per_buffer *
    rows_per_group samples rather than the whole dataset, so shuffling
    is slightly less thorough than a true global shuffle. With several
    DataLoader workers each worker has its own cache, so the I/O savings
    are partial.

    Call set_epoch(epoch) each epoch for a different order.
    """

    def __init__(self, dataset: XBDParquetDataset, groups_per_buffer: int = 8, seed: int = 0):
        by_group: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, gid in enumerate(dataset.group_ids):
            by_group[gid].append(i)
        self._groups = list(by_group.values())
        self.groups_per_buffer = groups_per_buffer
        self.seed = seed
        self.epoch = 0
        self._n = len(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = self._groups[:]
        rng.shuffle(order)
        for start in range(0, len(order), self.groups_per_buffer):
            buffer = [i for group in order[start : start + self.groups_per_buffer] for i in group]
            rng.shuffle(buffer)
            yield from buffer


def official_splits_parquet(
    parquet_dir,
    train_events: list[str] | None = None,
    transform=None,
    cache_row_groups: int = 8,
) -> tuple[XBDParquetDataset, XBDParquetDataset, XBDParquetDataset]:
    """Parquet datasets for the challenge's own partitions, returned as
    (train, hold, test) from parquet_dir/train.parquet, hold.parquet and
    test.parquet (the files build_parquet.py writes).

    Nothing is re-split. Each partition comes from its own file, so
    images from one partition can never appear in another. Do NOT point
    XBDParquetDataset at the whole directory for training: that would
    mix all three partitions together.

    train_events restricts ONLY the training partition (None = all
    events). hold and test always keep every event, so models trained
    on different subsets are all scored on the same data and stay
    roughly comparable with published results. For a per-event score,
    break the metric down by event (see evaluate.py) rather than
    filtering the evaluation sets.
    """
    parquet_dir = Path(parquet_dir)
    paths = {name: parquet_dir / f"{name}.parquet" for name in OFFICIAL_SPLITS}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {parquet_dir}. Run build_parquet.py once per "
            f"partition ({', '.join(OFFICIAL_SPLITS)})."
        )
    kwargs = dict(transform=transform, cache_row_groups=cache_row_groups)
    return (
        XBDParquetDataset(paths["train"], events=train_events, **kwargs),
        XBDParquetDataset(paths["hold"], **kwargs),
        XBDParquetDataset(paths["test"], **kwargs),
    )
