# damage-assessment

## Overview

This project uses deep learning to perform building localization and damage classification from satellite imagery using the publicly available xBD Dataset [https://xview2.org/dataset]. Provided with pre-disaster and post-disaster satellite images, the goal is to:
1. **Localize** buildings through building footprints as a binary mask.
2. **Classify damage severity** per building using a 4-level ordinal scale defined as: 
    - 0: Background 
    - 1: No damage (Undisturbed. No sign of water, structural damage, shingle damage, or burn marks)
    - 2: minor damage (Building partially burnt, water surrounding the structure, volcanic flow nearby, roof elements missing, or visible cracks)
    - 3: major damage (Partial wall or roof collapse, encroaching volcanic flow, or the structure is surrounded by water or mud)
    - 4: destroyed (Structure is scorched, completely collapsed, partially or completely covered with water or mud, or no longer present); 
    by comparing the pre- and post-disaster imagery.

**Status:** 🚧 In progress. See [Results](#results) for what's done so far.

## Motivation

With the onset of a super El Niño event, I was inspired to expand my machine learning expertise using data  that is both outside of my healthcare domain and also particularly relevant today as the effects of climate change increase in severity. This project applies a computer vision/deep learning pipeline (Siamese U-Net-style segmentation and classification) to the xBD satellite image dataset, building on my prior experience with image segmentation, feature engineering, and ensemble modeling from medical image research, while developing my deep learning architecture experience (CNNs, U-Net variants). 

Similar to my background with medical imaging, traditional assessments of satellite imagery is typically performed manually by an experienced professional, a labor-intensive process. This project aims to show that leveraging computer vision in this way could reduce labor time to hours.

## Setup

### Requirements
See `requirements.txt`. Key dependencies: PyTorch, segmentation-models-pytorch,
rasterio, shapely, scikit-learn.

```bash
pip install -r requirements.txt
```

### Data
This project uses the full **xBD dataset** (22,068 images, 19 disaster events, 30.3GB), available via download following the required free registration from the official source at xview2.org. The challenge's own `train`, `hold` and `test` partitions are used as-is, with no re-splitting, so results are comparable with other teams'.

The experiment is to compare models trained on different subsets of the **training** events, for example a single disaster type versus all of them. Three events provide distinct damage mechanisms:
   - **Hurricane Michael** (wind/structural damage)
   - **Santa Rosa Wildfire** (fire damage)
   - **Mexico City Earthquake** (seismic damage)

Every model is evaluated on the same full `hold` and `test` partitions, with scores broken down by event to show how well a model trained on some disaster types generalizes to the others. Only the training partition is ever filtered.

Data is **not redistributed here** (see `.gitignore` — `data/` is excluded from version control).

1. Registration at [xview2.org/dataset](https://xview2.org/dataset) —
   already done.
2. Download the Challenge training/test/holdout sets from
   [xview2.org/download-links](https://xview2.org/download-links)
   (via Aspera Connect).
3. Extract the archives. Keep every event in all three partitions,
   since `hold` and `test` must stay complete; which events a model
   trains on is chosen later, when the data is loaded.
4. Place the extracted data under `data/` following the structure
   documented in `src/data_loading.py`.
5. Run the one-time preprocessing step to convert the raw files into
   parquet (see `src/build_parquet.py`), once per split folder:
   ```bash
   python src/build_parquet.py --data-dir data --split train
   python src/build_parquet.py --data-dir data --split test
   python src/build_parquet.py --data-dir data --split hold
   ```
   This rasterizes label polygons into masks once (instead of on
   every training step) and writes one parquet file per split under
   `data/parquet/`, containing every event in that split. The raw
   extracted PNGs/JSONs can be deleted afterward to reclaim disk space.

6. Train from parquet using the challenge's **official partitions**
   (no re-splitting, so results are comparable with other teams'):
   ```python
   from data_loading import official_splits_parquet

   train_ds, hold_ds, test_ds = official_splits_parquet("data/parquet")
   ```
   Each dataset comes from its own file (`train.parquet`, `hold.parquet`,
   `test.parquet`). Fit on `train`, tune and pick checkpoints on `hold`,
   and report the final number on `test` only. Don't point
   `XBDParquetDataset` at the whole `data/parquet` directory for
   training; that would mix all three partitions.

   To train on a subset of events, restrict **only** the training
   partition; `hold` and `test` always keep every event:
   ```python
   from data_loading import TARGET_EVENTS

   train_ds, hold_ds, test_ds = official_splits_parquet(
       "data/parquet", train_events=["hurricane-michael"]   # or TARGET_EVENTS
   )
   ```
   Use `collections.Counter(ds.disasters)` to see which events a
   dataset contains.

7. Score each model with `src/evaluate.py`: the xView2 metric
   (0.3 × localization F1 + 0.7 × damage F1) overall and per event, so
   models trained on different subsets can be compared on identical
   evaluation data. Keep the architecture, hyperparameters and training
   budget the same across subsets, and note each subset's size, since a
   model trained on a bigger event can win just by seeing more images.

**Note on disk space:** the Aspera packages from xview2.org are
typically bundled as full train/test/hold archives, and the parquet
files are roughly as large as the raw images, so plan for the full
dataset. Convert one partition at a time and delete its raw files
afterward if space is tight.


## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/            # xBD data (gitignored — see Setup)
├── src/             # Core pipeline code
│   ├── build_parquet.py
│   ├── data_loading.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── tests/           # pytest (run `pytest` from the repo root)
├── notebooks/        # Exploratory analysis, kept separate from src/
└── results/          # Metrics, figures (gitignored except summaries)
```


## Results

_To be filled in as the project progresses — localization F1,
damage-classification F1, and the combined xView2 challenge metric will
go here._

## Citation

Gupta, R., et al. (2019). *xBD: A
Dataset for Assessing Building Damage from Satellite Imagery.*
