# CSIRO Biomass Prediction

Competition-grade repository for the Kaggle **CSIRO Biomass Prediction** challenge. The goal is to predict five biomass targets from top-view pasture imagery and metadata:

- `Dry_Green_g` weight `0.1`
- `Dry_Dead_g` weight `0.1`
- `Dry_Clover_g` weight `0.1`
- `GDM_g` weight `0.2`
- `Dry_Total_g` weight `0.5`

## Current Status

The requested Kaggle command was run:

```python
import kagglehub
path = kagglehub.competition_download("csiro-biomass")
```

Kaggle returned `401 Unauthorized`, which means the local account must be authenticated and must accept the competition rules at the Kaggle competition page before data can be downloaded. The repository is ready to run once that is fixed.

## Project Layout

```text
CSIRO-Biomass/
├── data/
├── notebooks/
├── src/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── ensemble/
│   └── utils/
├── configs/
├── reports/
├── submissions/
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

## Quickstart

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download data:

```bash
python -m src.datasets.download --out-dir data
```

Run deep EDA:

```bash
python -m src.datasets.eda --data-dir data --report-dir reports
```

Create folds:

```bash
python -m src.training.make_folds --data-dir data --out data/folds.csv --strategy stratified
```

Train metadata baselines:

```bash
python -m src.training.train_metadata --folds-csv data/folds.csv --out-dir models/metadata
```

Generate a submission:

```bash
python -m src.inference.make_submission --data-dir data --model-dir models/metadata --out submissions/submission.csv
```

## Phase Plan

### 1. Deep EDA

Implemented in `src.datasets.eda`.

Outputs:

- Missing values for train/test
- Numeric summaries
- Target distributions
- Correlation matrix
- State/species/date count plots where columns exist
- Image width/height/mode profile
- Perceptual hash duplicate report
- Biomass constraint report:
  - `Dry_Total_g - (Dry_Green_g + Dry_Dead_g + Dry_Clover_g)`
  - `GDM_g - (Dry_Green_g + Dry_Clover_g)`

### 2. Validation Strategy

Implemented in `src.training.make_folds`.

Strategies:

- `random`
- `stratified`
- `group`
- `date`

The recommended default before seeing the files is stratified folds on `Dry_Total_g`, then compare against date and group folds to detect leakage.

### 3. Metadata Models

Implemented in `src.training.train_metadata`.

Features:

- NDVI, height
- month, season, day of year
- cyclic month encoding
- species count and species frequency
- NDVI-height interactions
- state/species categorical encoding

Models:

- LightGBM, CatBoost, XGBoost when installed
- ExtraTrees, RandomForest, HistGradientBoosting fallback

### 4. Image Embeddings

Implemented in `src.features.extract_embeddings`.

Planned encoders:

- DINOv2
- SigLIP
- CLIP
- ConvNeXt V2
- EVA-02

The first high-ROI run should be frozen embeddings plus CatBoost/LightGBM, then blend across encoders.

### 5-9. Vision, Multimodal, Multi-task, Constraints, SSL

Config scaffolding is in `configs/vision_multimodal.yaml`. The intended implementation order is:

1. ConvNeXtV2/EVA-02 supervised multi-target model
2. Metadata fusion branch
3. Separate vs hierarchical heads
4. Soft biological losses
5. DINO/MAE/SimCLR self-supervised pretraining if the unlabeled image pool is large enough

### 10. External Data

Research candidates:

- Historical weather: rainfall, temperature, humidity, solar radiation
- Satellite products: Sentinel-2, Landsat, MODIS vegetation indices
- Soil moisture
- Australian pasture/agricultural statistics

Every external source must document license, date/location join keys, and leakage risk.

### 11-13. Pseudo-labeling, TTA, Ensembling

Implemented first piece:

- `src.ensemble.optimize_weights` optimizes OOF ensemble weights.

Planned:

- confidence-filtered pseudo labels
- flip/rotation/multi-crop/scale TTA
- mean, weighted, rank, stacking, and blending ensembles

### 14. Explainability

Planned reports:

- SHAP for metadata and embedding models
- feature importance
- embedding UMAP/t-SNE
- state/species/date error slices
- hard-sample gallery

## Top 5 Highest-ROI Approaches

See `reports/top_roi_strategy.md`.

## Notes

This repository uses a weighted log-R² helper aligned with the target weights supplied in the project brief. If the competition page defines a slightly different transform or aggregation, update `src.utils.metrics` before trusting CV scores.
