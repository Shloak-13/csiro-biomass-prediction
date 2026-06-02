from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.tabular import add_tabular_features, split_xy
from src.models.metadata_models import default_model_specs, make_pipeline
from src.utils.constants import TARGET_COLS
from src.utils.io import normalize_train_frame, read_csv
from src.utils.metrics import weighted_log_r2


def train_metadata(folds_csv: Path, out_dir: Path, seed: int) -> None:
    df = pd.read_csv(folds_csv)
    targets = [c for c in TARGET_COLS if c in df.columns]
    if len(targets) != len(TARGET_COLS):
        raise ValueError(f"Expected all targets in folds CSV. Found: {targets}")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_df, stats = add_tabular_features(df)
    x_all, y_all = split_xy(feature_df, targets)
    x_all = x_all.drop(columns=["fold"], errors="ignore")
    folds = df["fold"].astype(int).values

    results = []
    oof_store = {}
    for spec in default_model_specs(seed):
        oof = np.zeros((len(df), len(targets)), dtype=float)
        fold_scores = []
        for fold in sorted(np.unique(folds)):
            trn = folds != fold
            val = folds == fold
            x_trn, fit_stats = add_tabular_features(df.loc[trn])
            x_val, _ = add_tabular_features(df.loc[val], fit_stats)
            x_trn, y_trn = split_xy(x_trn, targets)
            x_val, y_val = split_xy(x_val, targets)
            x_trn = x_trn.drop(columns=["fold"], errors="ignore")
            x_val = x_val.drop(columns=["fold"], errors="ignore")
            model = make_pipeline(x_trn, spec.estimator)
            model.fit(x_trn, np.log1p(y_trn.clip(lower=0)))
            pred = np.expm1(model.predict(x_val))
            pred = np.clip(pred, 0, None)
            oof[val] = pred
            score = weighted_log_r2(y_val.values, pred)
            fold_scores.append(score)
            joblib.dump(model, out_dir / f"{spec.name}_fold{fold}.joblib")

        overall = weighted_log_r2(y_all.values, oof)
        results.append({"model": spec.name, "cv": overall, "fold_scores": fold_scores})
        oof_store[spec.name] = oof
        pd.DataFrame(oof, columns=targets).to_csv(out_dir / f"{spec.name}_oof.csv", index=False)

    with (out_dir / "feature_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    pd.DataFrame(results).sort_values("cv", ascending=False).to_csv(
        out_dir / "metadata_cv_results.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-csv", type=Path, default=Path("data/folds.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/metadata"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_metadata(args.folds_csv, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
