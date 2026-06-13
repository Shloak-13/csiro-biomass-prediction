from __future__ import annotations

import numpy as np
import pandas as pd


def season_from_month(month: float) -> str:
    if pd.isna(month):
        return "unknown"
    month = int(month)
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def add_tabular_features(df: pd.DataFrame, fit_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    stats = {} if fit_stats is None else dict(fit_stats)

    if "Sampling_Date" in out.columns:
        dates = pd.to_datetime(out["Sampling_Date"], errors="coerce")
        out["sampling_month"] = dates.dt.month
        out["sampling_dayofyear"] = dates.dt.dayofyear
        out["sampling_year"] = dates.dt.year
        out["sampling_season"] = out["sampling_month"].map(season_from_month)
        out["sampling_month_sin"] = np.sin(2 * np.pi * out["sampling_month"] / 12)
        out["sampling_month_cos"] = np.cos(2 * np.pi * out["sampling_month"] / 12)

    if "Species" in out.columns:
        species_text = out["Species"].fillna("unknown").astype(str)
        out["species_count"] = species_text.str.split(r"[,;/|+]").map(len)
        if fit_stats is None:
            stats["species_freq"] = species_text.value_counts(normalize=True).to_dict()
        out["species_freq"] = species_text.map(stats.get("species_freq", {})).fillna(0)

    if {"Pre_GSHH_NDVI", "Height_Ave_cm"}.issubset(out.columns):
        out["ndvi_x_height"] = out["Pre_GSHH_NDVI"] * out["Height_Ave_cm"]
        out["height_per_ndvi"] = out["Height_Ave_cm"] / out["Pre_GSHH_NDVI"].replace(0, np.nan)

    if {"Pre_GSHH_NDVI", "sampling_month"}.issubset(out.columns):
        out["ndvi_x_month"] = out["Pre_GSHH_NDVI"] * out["sampling_month"]

    return out, stats


def split_xy(df: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[targets].copy()
    drop_cols = set(targets) | {"target", "target_name"}
    x = df[[c for c in df.columns if c not in drop_cols]].copy()
    return x, y
