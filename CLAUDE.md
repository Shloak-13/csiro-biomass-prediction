# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Lint and format
ruff check src/
ruff format src/

# Tests (no test suite exists yet, but pytest is configured)
pytest tests/
pytest tests/test_foo.py::test_bar  # single test
```

### Pipeline execution order

```bash
# 1. Download competition data
python src/datasets/download.py --out-dir data

# 2. Create cross-validation folds
python src/training/make_folds.py --data-dir data --out data/folds.csv --strategy stratified --n-splits 5 --seed 42

# 3. Extract vision embeddings (optional, for multimodal runs)
python src/features/extract_embeddings.py --model dinov2 --batch-size 16

# 4. Train metadata models
python src/training/train_metadata.py --folds-csv data/folds.csv --seed 42

# 5. Generate submission
python src/inference/make_submission.py --data-dir data --model-dir models/metadata --out submissions/submission.csv

# 6. Optimize ensemble weights (when blending multiple prediction files)
python src/ensemble/optimize_weights.py --truth data/folds.csv --preds pred1.csv pred2.csv --out reports/weights.csv

# 7. Submit to Kaggle
kaggle competitions submit -c csiro-pasture-biomass-prediction -f submissions/final.csv -m "message"
```

## Architecture

### Pipeline layers (`src/`)

| Layer | Module | Role |
|-------|--------|------|
| Data | `datasets/` | Download, normalize, EDA, investigation scripts |
| Features | `features/` | Tabular feature engineering; vision embedding extraction |
| Models | `models/` | Model specs — vision backbones (timm/transformers) and metadata models (sklearn/boosting) |
| Training | `training/` | Fold generation and k-fold metadata model training |
| Inference | `inference/` | OOF/test prediction generation and submission formatting |
| Ensemble | `ensemble/` | SLSQP-based weight optimization over prediction files |
| Utils | `utils/` | Constants, I/O normalization, weighted log-R2 metric |

### Key data flow

Raw Kaggle CSVs are in **long format** (`target_name`, `target` columns). `src/utils/io.py:normalize_train_frame()` pivots these to **wide format** (one column per target) before any processing. The inverse happens at submission time via `normalize_submission_frame()`.

The five prediction targets and their competition weights are defined in `src/utils/constants.py`:
- `Dry_Green_g` (0.1), `Dry_Dead_g` (0.1), `Dry_Clover_g` (0.1), `GDM_g` (0.2), `Dry_Total_g` (0.5)

The primary metric is **weighted log-R2** (not standard R2) — predictions are log1p-transformed before computing per-target R2, then weighted. See `src/utils/metrics.py`.

### Cross-validation and reproducibility

`make_folds.py` adds a `fold` column to the normalized train CSV. It supports four strategies: `stratified` (default, bins target by quantile), `group` (by image_id/State), `date` (temporal), and `random`. The full experiment runs 3 seeds × 5 folds × 3 architectures = 45 models; OOF predictions from each are used as inputs to `optimize_weights.py`.

### Feature engineering and train/test leakage prevention

`src/features/tabular.py:add_tabular_features()` returns both the enriched DataFrame and a `stats` dict. Pass `fit_stats=None` during training (fits and returns stats), then pass `fit_stats=<saved_stats>` during inference to apply the same transformations without refitting. The stats dict is serialized to `feature_stats.json` alongside trained models.

### Metadata model pipeline

`src/models/metadata_models.py:build_preprocessor()` returns a `ColumnTransformer` (numeric: impute → scale; categorical: impute → one-hot). `make_pipeline(spec)` composes this with the model. All boosters (LightGBM, CatBoost, XGBoost) are wrapped in `MultiOutputRegressor` to predict all five targets independently.

### Vision models

`src/models/vision_backbones.py` defines `VISION_BACKBONES` (timm fine-tuning: EfficientNetV2-S, ConvNeXtV2-Base, Swin-Base, ViT-Base, EVA02-Base) and `EMBEDDING_MODELS` (frozen feature extraction: DINOv2, SigLIP, CLIP, ConvNeXtV2, EVA02 via HuggingFace transformers or timm). Embeddings are stored as `.npy` arrays with a path-mapping CSV in `data/embeddings/`.

### Configuration

YAML configs in `configs/` (`base.yaml`, `metadata_baseline.yaml`, `vision_multimodal.yaml`) set seeds, fold counts, target weights, and paths. `base.yaml` is the source of truth for competition-wide constants. Every pipeline script also accepts argparse flags that override config defaults.

### Biomass constraints

`src/inference/make_submission.py` supports optional physical constraint enforcement (`--constraint derive_total` or `derive_gdm_total`) to ensure predictions satisfy `Dry_Total = Dry_Green + Dry_Dead + Dry_Clover`.
