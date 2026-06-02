from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.utils.constants import TARGET_COLS
from src.utils.io import infer_id_column, normalize_train_frame, read_csv


def add_folds(df: pd.DataFrame, strategy: str, n_splits: int, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1
    y_target = "Dry_Total_g" if "Dry_Total_g" in out.columns else out.select_dtypes("number").columns[-1]

    if strategy == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(out)
    elif strategy == "stratified":
        bins = pd.qcut(out[y_target].rank(method="first"), q=min(10, len(out) // n_splits), labels=False)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(out, bins)
    elif strategy == "group":
        group_col = next((c for c in ["image_id", "sample_id", "State", "Sampling_Date"] if c in out.columns), None)
        if group_col is None:
            group_col = infer_id_column(out)
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(out, groups=out[group_col].astype(str))
    elif strategy == "date":
        if "Sampling_Date" not in out.columns:
            raise ValueError("Date-aware folds require Sampling_Date.")
        dates = pd.to_datetime(out["Sampling_Date"], errors="coerce")
        order = np.argsort(dates.fillna(dates.min()).values)
        fold_ids = np.arange(len(out)) % n_splits
        out.iloc[order, out.columns.get_loc("fold")] = fold_ids
        return out
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for fold, (_, val_idx) in enumerate(splits):
        out.loc[out.index[val_idx], "fold"] = fold
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/folds.csv"))
    parser.add_argument("--strategy", choices=["random", "stratified", "group", "date"], default="stratified")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = normalize_train_frame(read_csv(args.data_dir, "train.csv"))
    missing = [c for c in TARGET_COLS if c not in train.columns]
    if missing:
        print(f"Warning: missing expected target columns after normalization: {missing}")
    folded = add_folds(train, args.strategy, args.n_splits, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    folded.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
