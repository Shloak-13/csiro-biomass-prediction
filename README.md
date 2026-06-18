# CSIRO Biomass Prediction

**Kaggle Competition:** Predict five pasture biomass targets from top-view drone imagery and metadata.

---

## Results

I entered the [CSIRO Image2Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass) competition (3,805 teams), placed **3,469th**, with a Public LB score of **0.21417** and a Private LB score of **0.24522**.

---

## Architecture

The pipeline combines two modalities:

- **Metadata models** — LightGBM / CatBoost / XGBoost on tabular features derived from `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, `Sampling_Date`.
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

### Local metadata pipeline (working end-to-end)

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
```

### Vision/Kaggle pipeline (incomplete)

```bash
# Push vision training kernel to Kaggle GPU
python scripts/kaggle/push_kernel.py kaggle_kernels/csiro-multitask-ndvi-height-v1/
python scripts/kaggle/monitor_kernel.py shloakshetty/csiro-multitask-ndvi-height-v1
python scripts/kaggle/download_outputs.py shloakshetty/csiro-multitask-ndvi-height-v1
```

This never produced a scored submission: the training dataset (`shloakshetty/csiro-biomass-data`) was never uploaded to Kaggle, and the competition closed before the vision pipeline was fully wired up.

---

## Key Findings

- **Metadata dominates** — Environmental features (NDVI × Height interaction, day-of-year, species) are the strongest predictors; image embeddings alone lagged substantially (local CV 0.556).
- **Test metadata unavailable** — test.csv has no `Pre_GSHH_NDVI` / `Height_Ave_cm`, requiring image-based inference. The multitask network predicts NDVI/Height as auxiliary targets to bridge this gap.
- **Embeddings hurt when fused** — Adding DINOv2/SigLIP/ConvNeXtV2 embeddings to metadata decreased CV by 0.039. Canopy-aware preprocessing is likely needed before fusion helps.
- **Biological constraints improve reliability** — `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g` holds in train data (median residual ≈ 0); enforcing it post-prediction is justified.

---

## Post-Competition Debugging

After the competition closed I went back through the codebase and found two bugs that had been silently invalidating my local validation scores throughout the competition.

**Bug 1 — Wrong column names in feature engineering (`src/features/tabular.py`).** The code checked for columns named `NDVI` and `Height`, but the actual competition data uses `Pre_GSHH_NDVI` and `Height_Ave_cm`. The `if` condition silently evaluated to `False` on every run, meaning the two highest-ranked features by SHAP importance — `ndvi_x_height` (SHAP 0.278) and `height_per_ndvi` (SHAP 0.160) — were never computed. Every CV score logged during the competition was produced without these features.

**Bug 2 — Wrong metric (`src/datasets/full_benchmark.py`).** The benchmark pipeline computed standard R² per target. The competition metric is weighted log-space R²: targets are log1p-transformed before R² is calculated, then weighted (0.1 / 0.1 / 0.1 / 0.2 / 0.5). Standard and log-space R² diverge significantly on biomass targets, which span several orders of magnitude. This made every benchmark comparison computed before the fix unreliable.

After fixing both bugs, local cross-validation reached a weighted log-R² of **0.711** on the training data. This score was never submitted to the leaderboard and has no corresponding LB entry. It also cannot be directly submitted: `test.csv` contains only `sample_id`, `image_path`, and `target_name` — it has no `Pre_GSHH_NDVI` or `Height_Ave_cm` columns. A model that depends on those features cannot run on the real test data without first predicting them from imagery, which is what the incomplete vision pipeline is intended to address.

---

## Training Details

- **CV strategy:** Stratified 5-fold (bins target quantiles)
- **Metadata models:** 45 total — 3 seeds × 5 folds × 3 model families
- **Vision training:** 5-fold GroupKFold, 12 epochs, ConvNeXtV2-Base backbone, 384×384 images, multitask loss (biomass + 0.25 × aux NDVI/Height smooth-L1)
- **Ensemble:** OOF-optimised weights via SLSQP (non-negative, sum to 1)
