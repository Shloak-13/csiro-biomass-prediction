from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.tabular import add_tabular_features
from src.utils.constants import TARGET_COLS
from src.utils.io import read_csv
from src.utils.metrics import enforce_biomass_constraints


def predict_metadata(test: pd.DataFrame, model_dir: Path) -> np.ndarray:
    model_paths = sorted(model_dir.glob("*_fold*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No fold models found under {model_dir}")
    x_test, _ = add_tabular_features(test)
    preds = []
    for path in model_paths:
        model = joblib.load(path)
        pred = np.expm1(model.predict(x_test))
        preds.append(np.clip(pred, 0, None))
    return np.mean(preds, axis=0)


def format_submission(sample_submission: pd.DataFrame, test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    if {"target_name", "target"}.issubset(sample_submission.columns):
        id_col = next(c for c in sample_submission.columns if c not in {"target_name", "target"})
        test_id_col = id_col if id_col in test.columns else next(c for c in test.columns if "id" in c.lower())
        wide = pd.DataFrame(pred, columns=TARGET_COLS)
        wide.insert(0, id_col, test[test_id_col].values)
        long = wide.melt(id_vars=id_col, value_vars=TARGET_COLS, var_name="target_name", value_name="target")
        return sample_submission[[id_col, "target_name"]].merge(long, on=[id_col, "target_name"], how="left")

    out = sample_submission.copy()
    for idx, target in enumerate(TARGET_COLS):
        if target in out.columns:
            out[target] = pred[:, idx]
    if "target" in out.columns and len(out) == pred.size:
        out["target"] = pred.reshape(-1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/metadata"))
    parser.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--constraint", choices=["none", "derive_total", "derive_gdm_total"], default="none")
    args = parser.parse_args()

    test = read_csv(args.data_dir, "test.csv")
    sample = read_csv(args.data_dir, "sample_submission.csv")
    pred = predict_metadata(test, args.model_dir)
    pred = enforce_biomass_constraints(pred, args.constraint)
    sub = format_submission(sample, test, pred)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
