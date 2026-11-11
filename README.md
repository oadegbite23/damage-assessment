# damage-assessment

## Overview

This project uses deep learning to perform building localization and damage classification from satellite imagery using the publicly available xBD Dataset [https://xview2.org/dataset]. Provided with pre-disaster and post-disaster satellite images, the goal is to:
1. **Localize** buildings through building footprints as a binary mask.
2. **Classify damage severity** per building using a 4-level ordinal scale defined as: 
    - 0: no damage (Undisturbed. No sign of water, structural damage, shingle damage, or burn marks)
    - 1: minor damage (Building partially burnt, water surrounding the structure, volcanic flow nearby, roof elements missing, or visible cracks)
    - 2: major damage (Partial wall or roof collapse, encroaching volcanic flow, or the structure is surrounded by water or mud)
    - 3: destroyed (Structure is scorched, completely collapsed, partially or completely covered with water or mud, or no longer present); 
    by comparing the pre- and post-disaster imagery.

**Status:** 🚧 In progress. See [Results](#results) for what's done so far.

## Motivation

With the onset of a super El Niño event, I was inspired to expand my maching learning expertise using data  that is both outside of my healthcare domain and also particulary relevant today as the affects of climate change increase in severity. This project applies a computer vision/deep learning pipeline (Siamese U-Net-style segmentation and classification) to the xBD satellite image dataset, building on my prior experience with image segmentation, feature engineering, and ensemble modeling from medical image research, while developing my deep learning architecture experience (CNNs, U-Net variants). 

Similar to my background with medical imaging, traditional assesments of satellite imagery is typically performed manually by an experienced professional, a labor-intensive process. This project aims to show that leveraging computer vision in this way could reduce labor time to hours.

## Setup

### Requirements
See `requirements.txt`. Key dependencies: PyTorch, segmentation-models-pytorch,
rasterio, shapely, scikit-learn.

```bash
pip install -r requirements.txt
```

### Data
This project uses a subset of the **xBD dataset**, scoped down due to local storage/compute constraints.The full dataset (22,068 images, 30.3GB) is available via download following the required free registration from the offical source at xview2.org. Here I've used 3 of the 19 disaster events:
    - **Huricane Michael** (wind/structural damage)
    - **Santa Rosa Wildfire** (fire damage)
    - **Mexico City Earthquake** (seismic damage)

These three events provide three distinct damage mechanisms to increase generalization capabilities, using a fraction of the full dataset. 
Data is **not redistributed here** (see `.gitignore` — `data/` is excluded from version control).

1. Registration at [xview2.org/dataset](https://xview2.org/dataset) —
   already done.
2. Download the Challenge training/test/holdout sets from
   [xview2.org/download-links](https://xview2.org/download-links)
   (via Aspera Connect).
3. From the downloaded archives, extract only the files matching the
   3 target event names (`hurricane-michael`, `santa-rosa-wildfire`,
   `mexico-city-earthquake`) rather than unpacking everything — see
   `src/data_loading.py` for the extraction approach.
4. Place the extracted data under `data/` following the structure
   documented in `src/data_loading.py`.

   ## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/            # xBD data (gitignored — see Setup)
├── src/             # Core pipeline code
│   ├── build_paraquet.py
    ├── data_loading.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
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
