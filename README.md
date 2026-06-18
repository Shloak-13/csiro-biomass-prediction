# CSIRO Biomass Prediction

**Kaggle Competition:** Predict five pasture biomass targets from top-view drone imagery and metadata.

---

## Results

| Split | Score |
|---|---|
| **Public LB** | **0.21417** |
| **Private LB** | **0.24522** |
| Best local CV (LightGBM, post-competition) | 0.711 |

Competition: [csiro-biomass](https://www.kaggle.com/competitions/csiro-biomass) · Deadline: January 28, 2026 · Metric: weighted log-R² (targets weighted 0.1 / 0.1 / 0.1 / 0.2 / 0.5)

---

## Architecture

The pipeline combines two modalities:

- **Metadata models** — LightGBM / CatBoost / XGBoost on tabular features derived from `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, `Sampling_Date`. Best local CV: **0.711** (post-competition, with correct column names and log-R² metric).
- **Vision models** — ConvNeXtV2-Base multitask network predicting biomass + auxiliary NDVI/Height heads, enabling test-time inference without metadata (test.csv has no metadata columns).
- **Ensemble** — SLSQP-optimized weighted blend of OOF predictions.

Key constraint: test data contains only `sample_id`, `image_path`, `target_name` — no environmental metadata. All deployable approaches must use imagery.

---

## Tech Stack

Python · PyTorch · timm · scikit-learn · LightGBM · CatBoost · XGBoost · albumentations · wandb

---

## Project Structure

```
src/
├── datasets/        # Download, EDA, canopy crop benchmarks
├── features/        # Tabular feature engineering; vision embedding extraction
├── models/          # Vision backbones (timm/transformers) and metadata model specs
├── training/        # Fold generation (make_folds.py) and k-fold training (train_metadata.py)
├── inference/       # Submission formatting (make_submission.py)
├── ensemble/        # SLSQP weight optimisation (optimize_weights.py)
└── utils/           # Constants, I/O normalisation, weighted log-R² metric
kaggle_kernels/      # Packaged Kaggle notebook kernels (push via scripts/kaggle/)
scripts/kaggle/      # Orchestration: create → push → monitor → download
configs/             # YAML experiment configs (base, metadata_baseline, vision_multimodal)
experiments/         # registry.csv — all submitted and local experiments with scores
reports/             # Analysis outputs: CV scores, SHAP importance, strategy docs
```

---

## How to Run

```bash
pip install -r requirements.txt

# Download competition data
python src/datasets/download.py --out-dir data

# Create 5-fold CV splits
python src/training/make_folds.py --data-dir data --out data/folds.csv --strategy stratified

# Train metadata models (LightGBM, CatBoost, XGBoost, ExtraTrees)
python src/training/train_metadata.py --folds-csv data/folds.csv --seed 42

# Generate submission (metadata models)
python src/inference/make_submission.py --data-dir data --model-dir models/metadata --out submissions/submission.csv

# Push vision training kernel to Kaggle GPU
python scripts/kaggle/push_kernel.py kaggle_kernels/csiro-multitask-ndvi-height-v1/
python scripts/kaggle/monitor_kernel.py shloakshetty/csiro-multitask-ndvi-height-v1
python scripts/kaggle/download_outputs.py shloakshetty/csiro-multitask-ndvi-height-v1
```

---

## Key Findings

- **Metadata dominates** — Environmental features (NDVI × Height interaction, day-of-year, species) drive predictions. Local CV 0.711 vs. 0.556 for best image embeddings.
- **Test metadata unavailable** — test.csv has no `Pre_GSHH_NDVI` / `Height_Ave_cm`, requiring image-based inference. The multitask network predicts NDVI/Height as auxiliary targets to bridge this gap.
- **Embeddings hurt when fused** — Adding DINOv2/SigLIP/ConvNeXtV2 embeddings to metadata decreased CV by 0.039. Canopy-aware preprocessing is likely needed before fusion helps.
- **Biological constraints improve reliability** — `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g` holds in train data (median residual ≈ 0); enforcing it post-prediction is justified.
- **Critical bugs fixed** — Original code used wrong column names (`NDVI`/`Height` instead of `Pre_GSHH_NDVI`/`Height_Ave_cm`) and standard R² instead of log-R², making all pre-fix CV scores unreliable.

---

## Training Details

- **CV strategy:** Stratified 5-fold (bins target quantiles)
- **Metadata models:** 45 total — 3 seeds × 5 folds × 3 model families
- **Vision training:** 5-fold GroupKFold, 12 epochs, ConvNeXtV2-Base backbone, 384×384 images, multitask loss (biomass + 0.25 × aux NDVI/Height smooth-L1)
- **Ensemble:** OOF-optimised weights via SLSQP (non-negative, sum to 1)
