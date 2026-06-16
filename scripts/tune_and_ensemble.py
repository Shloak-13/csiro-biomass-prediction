"""
Optuna hyperparameter tuning + OOF ensemble for CSIRO biomass prediction.
Runs per-target LightGBM tuning then blends with XGBoost OOFs.
Usage: python -m scripts.tune_and_ensemble --data-dir data/csiro-biomass
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.features.tabular import add_tabular_features, split_xy
from src.training.make_folds import add_folds
from src.utils.constants import TARGET_COLS, TARGET_WEIGHTS
from src.utils.io import normalize_train_frame, read_csv
from src.utils.metrics import weighted_log_r2


def _cv_score(X, y_col, folds, params: dict) -> float:
    preds = np.zeros(len(X))
    for fold in np.unique(folds):
        trn, val = folds != fold, folds == fold
        m = LGBMRegressor(**params, verbose=-1)
        m.fit(X[trn], np.log1p(y_col[trn].clip(lower=0)))
        preds[val] = np.expm1(m.predict(X[val])).clip(0)
    y_arr = np.column_stack([y_col.values] * len(TARGET_COLS))
    p_arr = np.column_stack([preds] * len(TARGET_COLS))
    return preds, y_col.values


def tune_lgbm_per_target(X: np.ndarray, y: pd.DataFrame, folds: np.ndarray, n_trials: int, seed: int) -> dict:
    best_params = {}
    for target in TARGET_COLS:
        y_col = y[target]

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
                "random_state": seed,
            }
            oof = np.zeros(len(y_col))
            for fold in np.unique(folds):
                trn, val = folds != fold, folds == fold
                m = LGBMRegressor(**params, verbose=-1)
                m.fit(X[trn], np.log1p(y_col.values[trn].clip(0)))
                oof[val] = np.expm1(m.predict(X[val])).clip(0)
            yt = y_col.values
            denom = np.sum((np.log1p(yt.clip(0)) - np.log1p(yt.clip(0)).mean()) ** 2)
            r2 = 1.0 - np.sum((np.log1p(yt.clip(0)) - np.log1p(oof.clip(0))) ** 2) / (denom + 1e-9)
            return -r2

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params[target] = study.best_params
        print(f"  {target}: best R2={-study.best_value:.4f}  params={study.best_params}")

    return best_params


def train_with_params(X: np.ndarray, y: pd.DataFrame, folds: np.ndarray, params_per_target: dict, seed: int) -> np.ndarray:
    oof = np.zeros((len(y), len(TARGET_COLS)))
    for i, target in enumerate(TARGET_COLS):
        p = {**params_per_target[target], "random_state": seed, "verbose": -1}
        y_col = y[target].values
        for fold in np.unique(folds):
            trn, val = folds != fold, folds == fold
            m = LGBMRegressor(**p)
            m.fit(X[trn], np.log1p(y_col[trn].clip(0)))
            oof[val, i] = np.expm1(m.predict(X[val])).clip(0)
    return oof


def train_xgb_oof(X: np.ndarray, y: pd.DataFrame, folds: np.ndarray, seed: int) -> np.ndarray:
    oof = np.zeros((len(y), len(TARGET_COLS)))
    for i, target in enumerate(TARGET_COLS):
        y_col = y[target].values
        for fold in np.unique(folds):
            trn, val = folds != fold, folds == fold
            m = XGBRegressor(
                n_estimators=1500, learning_rate=0.02, max_depth=5,
                min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror",
                random_state=seed, verbosity=0,
            )
            m.fit(X[trn], np.log1p(y_col[trn].clip(0)))
            oof[val, i] = np.expm1(m.predict(X[val])).clip(0)
    return oof


def find_blend_weights(lgbm_oof: np.ndarray, xgb_oof: np.ndarray, y: pd.DataFrame) -> np.ndarray:
    y_arr = y[TARGET_COLS].values

    def neg_score(w):
        w = np.clip(w, 0, 1)
        blend = w[0] * lgbm_oof + (1 - w[0]) * xgb_oof
        return -weighted_log_r2(y_arr, blend)

    result = minimize(neg_score, x0=[0.6], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    return float(np.clip(result.x[0], 0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/csiro-biomass"))
    parser.add_argument("--folds-csv", type=Path, default=Path("data/folds.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/tuned"))
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data — always pivot on image_path (not sample_id which is per-row unique)
    if args.folds_csv.exists():
        df = pd.read_csv(args.folds_csv)
        if set(TARGET_COLS).issubset(df.columns) and len(df) < 500:
            print(f"Loaded {len(df)}-row wide folds CSV")
        else:
            print("Stale folds CSV detected (wrong format), recreating...")
            df = None

    if not args.folds_csv.exists() or df is None:
        print("Creating folds from train.csv...")
        train_raw = read_csv(args.data_dir, "train.csv")
        meta_cols = [c for c in train_raw.columns if c not in {"target_name", "target", "sample_id", "image_path"}]
        meta = train_raw.groupby("image_path")[meta_cols].first().reset_index()
        targets_wide = train_raw.pivot_table(
            index="image_path", columns="target_name", values="target", aggfunc="first"
        ).reset_index()
        train_wide = meta.merge(targets_wide, on="image_path")
        df = add_folds(train_wide, strategy="random", n_splits=5, seed=args.seed)
        args.folds_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.folds_csv, index=False)
        print(f"Saved {len(df)}-row folds CSV to {args.folds_csv}")

    feature_df, stats = add_tabular_features(df)
    x_df, y = split_xy(feature_df, TARGET_COLS)
    x_df = x_df.drop(columns=["fold"], errors="ignore")

    # Drop rows where any target is NaN (incomplete pivot rows)
    valid_mask = y[TARGET_COLS].notna().all(axis=1)
    x_df = x_df[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    folds = df["fold"].astype(int).values[valid_mask]
    print(f"Using {valid_mask.sum()} / {len(valid_mask)} rows after dropping NaN targets")

    # Drop non-numeric/id columns before converting to numpy
    drop_cols = [c for c in x_df.columns if x_df[c].dtype == object or c in ("sample_id", "image_path", "Sampling_Date")]
    x_df = x_df.drop(columns=drop_cols, errors="ignore")

    # Simple impute
    x_df = x_df.fillna(x_df.median(numeric_only=True))

    # One-hot encode remaining categoricals
    cat_cols = x_df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        x_df = pd.get_dummies(x_df, columns=cat_cols, drop_first=False)

    X = x_df.values.astype(float)
    folds = df["fold"].astype(int).values

    # --- Optuna tuning ---
    print(f"\nTuning LightGBM ({args.n_trials} trials per target)...")
    best_params = tune_lgbm_per_target(X, y, folds, args.n_trials, args.seed)
    with open(args.out_dir / "best_lgbm_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # --- OOF predictions ---
    print("\nGenerating tuned LightGBM OOFs...")
    lgbm_oof = train_with_params(X, y, folds, best_params, args.seed)
    lgbm_cv = weighted_log_r2(y[TARGET_COLS].values, lgbm_oof)
    print(f"Tuned LightGBM CV: {lgbm_cv:.6f}")

    print("\nGenerating XGBoost OOFs...")
    xgb_oof = train_xgb_oof(X, y, folds, args.seed)
    xgb_cv = weighted_log_r2(y[TARGET_COLS].values, xgb_oof)
    print(f"XGBoost CV: {xgb_cv:.6f}")

    # --- Blend ---
    w = find_blend_weights(lgbm_oof, xgb_oof, y)
    blend_oof = w * lgbm_oof + (1 - w) * xgb_oof
    blend_cv = weighted_log_r2(y[TARGET_COLS].values, blend_oof)
    print(f"\nBlend (LightGBM={w:.2f}, XGBoost={1-w:.2f}) CV: {blend_cv:.6f}")

    # Save per-target breakdown
    results = {"lgbm_cv": lgbm_cv, "xgb_cv": xgb_cv, "blend_cv": blend_cv, "lgbm_weight": w}
    for i, t in enumerate(TARGET_COLS):
        for name, oof in [("lgbm", lgbm_oof), ("xgb", xgb_oof), ("blend", blend_oof)]:
            yt = np.log1p(y[t].values.clip(0))
            yp = np.log1p(oof[:, i].clip(0))
            denom = np.sum((yt - yt.mean()) ** 2)
            r2 = 0.0 if denom == 0 else 1.0 - np.sum((yt - yp) ** 2) / denom
            results[f"{name}_{t}"] = round(r2, 6)

    with open(args.out_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_dir}/cv_results.json")

    # --- Train final models on full data for submission ---
    print("\nTraining final models on full data...")
    final_lgbm = []
    final_xgb = []
    for i, target in enumerate(TARGET_COLS):
        p = {**best_params[target], "random_state": args.seed, "verbose": -1}
        y_col = y[target].values

        lgbm = LGBMRegressor(**p)
        lgbm.fit(X, np.log1p(y_col.clip(0)))
        final_lgbm.append(lgbm)
        joblib.dump(lgbm, args.out_dir / f"lgbm_{target}.joblib")

        xgb = XGBRegressor(
            n_estimators=1500, learning_rate=0.02, max_depth=5,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror",
            random_state=args.seed, verbosity=0,
        )
        xgb.fit(X, np.log1p(y_col.clip(0)))
        final_xgb.append(xgb)
        joblib.dump(xgb, args.out_dir / f"xgb_{target}.joblib")

    joblib.dump({"lgbm_weight": w, "feature_cols": list(x_df.columns), "fit_stats": stats}, args.out_dir / "ensemble_meta.joblib")
    print(f"Models saved to {args.out_dir}/")

    print("\n=== SUMMARY ===")
    print(f"Baseline LightGBM CV (from reports): 0.716104")
    print(f"Tuned LightGBM CV:   {lgbm_cv:.6f}")
    print(f"XGBoost CV:          {xgb_cv:.6f}")
    print(f"Blend CV:            {blend_cv:.6f}")
    print(f"Gain vs baseline:   +{blend_cv - 0.716104:.6f}")


if __name__ == "__main__":
    main()
