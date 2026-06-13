# CSIRO Biomass Prediction

**Kaggle Competition:** Predict five pasture biomass targets from top-view imagery and metadata.

---

## Overview

This project predicts pasture biomass components (dry green, dry dead, clover, growth deficiency, total dry matter) from high-resolution top-view drone imagery and environmental metadata. The approach combines deep learning vision models with tabular metadata using an ensemble framework.

---

## Results

- **Competition:** Kaggle CSIRO Pasture Biomass Prediction
- **Best Score:** Competitive ensemble achieving strong performance
- **Approach:** Multi-backbone vision ensemble (ConvNeXt, EfficientNet-B3/B4) with tabular feature fusion
- **Model Artifacts:** 45 trained models (3 seeds × 5 folds × 3 architectures) with ensemble weight optimization

---

## Architecture

The pipeline integrates:
- **Vision Models:** ConvNeXt and EfficientNet backbones trained on pasture imagery
- **Metadata Models:** Tabular feature processing from environmental data
- **Ensemble:** Weighted voting across folds and seeds optimized for biomass prediction
- **Inference:** Efficient batch processing with optional test-time augmentation

---

## Tech Stack

Python · PyTorch · Scikit-learn · NumPy · Pandas · ConvNeXt · EfficientNet

---

## Project Structure

```
.
├── src/
│   ├── datasets.py              # Data loading and augmentation
│   ├── features.py              # Feature engineering
│   ├── models.py                # Model architecture definitions
│   ├── train.py                 # Training loop
│   ├── inference.py             # Batch inference
│   └── ensemble.py              # Ensemble weight optimization
├── notebooks/                   # Analysis and experimentation
├── models/                      # Trained model weights (.pth files)
├── experiments/                 # Experiment logs and metrics
├── data/                        # Raw imagery and metadata
├── submissions/                 # Kaggle submission CSVs
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download competition data:**
   ```bash
   kaggle competitions download -c csiro-pasture-biomass-prediction
   unzip csiro-pasture-biomass-prediction.zip -d data/
   ```

3. **Run the pipeline:**
   ```bash
   python src/features.py              # Feature engineering
   python src/train.py --fold 0        # Train fold 0
   python src/inference.py             # Generate predictions
   python src/ensemble.py              # Optimize ensemble weights
   ```

4. **Submit to Kaggle:**
   ```bash
   kaggle competitions submit -c csiro-pasture-biomass-prediction -f submissions/final.csv -m "Ensemble submission"
   ```

---

## Key Insights

- **Vision features dominate** — Learned image representations outperform hand-engineered features
- **Metadata contributes** — Environmental data (soil type, rainfall) improves predictions
- **Ensemble wins** — Combining multiple backbones (ConvNeXt + EfficientNet) reduces variance
- **Fold stratification matters** — Consistent cross-validation prevents overfitting

---

## Training & Validation

Models are trained with stratified k-fold cross-validation (5 folds) across 3 random seeds:
- **Total models trained:** 45 (3 seeds × 5 folds × 3 architectures)
- **Validation strategy:** Out-of-fold (OOF) predictions for ensemble weight optimization
- **Inference:** Weighted average across all 45 models with learned ensemble weights

