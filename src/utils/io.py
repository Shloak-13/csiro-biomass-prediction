from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.constants import ID_CANDIDATES, IMAGE_EXTENSIONS, TARGET_COLS


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_competition_file(data_dir: str | Path, name: str) -> Path:
    data_dir = Path(data_dir)
    matches = list(data_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {data_dir}")
    return matches[0]


def list_images(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return [p for p in data_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def read_csv(data_dir: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(find_competition_file(data_dir, name))


def infer_id_column(df: pd.DataFrame) -> str:
    for col in ID_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer id column. Columns: {list(df.columns)}")


def normalize_train_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Return one row per image/sample with five target columns when possible."""
    if set(TARGET_COLS).issubset(train.columns):
        return train.copy()

    if {"target_name", "target"}.issubset(train.columns):
        id_col = infer_id_column(train)
        meta_cols = [
            c for c in train.columns if c not in {"target_name", "target"} and c != id_col
        ]
        first_meta = train.groupby(id_col, as_index=False)[meta_cols].first() if meta_cols else train[[id_col]].drop_duplicates()
        wide = train.pivot_table(index=id_col, columns="target_name", values="target", aggfunc="first")
        wide = wide.reset_index()
        return first_meta.merge(wide, on=id_col, how="left")

    return train.copy()


def normalize_submission_frame(sample_submission: pd.DataFrame) -> pd.DataFrame:
    return sample_submission.copy()
