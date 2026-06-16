from __future__ import annotations

import numpy as np
import pandas as pd

_CLOVER_SPECIES = {"Clover", "Ryegrass_Clover", "WhiteClover", "Phalaris_Clover",
                   "SubcloverLosa", "SubcloverDalkeith", "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass",
                   "Phalaris_Ryegrass_Clover", "Ryegrass_Clover"}


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
        # Quadratic day-of-year captures peak/trough nonlinearity
        out["dayofyear_sin"] = np.sin(2 * np.pi * out["sampling_dayofyear"] / 365)
        out["dayofyear_cos"] = np.cos(2 * np.pi * out["sampling_dayofyear"] / 365)

    if "Species" in out.columns:
        species_text = out["Species"].fillna("unknown").astype(str)
        out["species_count"] = species_text.str.split(r"[,;/|+_]").map(len)
        if fit_stats is None:
            stats["species_freq"] = species_text.value_counts(normalize=True).to_dict()
        out["species_freq"] = species_text.map(stats.get("species_freq", {})).fillna(0)
        out["has_clover"] = species_text.isin(_CLOVER_SPECIES).astype(int)
        out["has_ryegrass"] = species_text.str.contains("Ryegrass", na=False).astype(int)
        out["has_phalaris"] = species_text.str.contains("Phalaris", na=False).astype(int)
        out["has_lucerne"] = species_text.str.contains("Lucerne", na=False).astype(int)
        out["has_fescue"] = species_text.str.contains("Fescue", na=False).astype(int)

    if "Pre_GSHH_NDVI" in out.columns:
        ndvi = out["Pre_GSHH_NDVI"].clip(lower=0)
        out["ndvi_sq"] = ndvi ** 2
        out["ndvi_sqrt"] = np.sqrt(ndvi)
        out["ndvi_log1p"] = np.log1p(ndvi)

    if "Height_Ave_cm" in out.columns:
        h = out["Height_Ave_cm"].clip(lower=0)
        out["height_sq"] = h ** 2
        out["height_sqrt"] = np.sqrt(h)
        out["height_log1p"] = np.log1p(h)

    if {"Pre_GSHH_NDVI", "Height_Ave_cm"}.issubset(out.columns):
        ndvi = out["Pre_GSHH_NDVI"].clip(lower=0)
        h = out["Height_Ave_cm"].clip(lower=0)
        out["ndvi_x_height"] = ndvi * h
        out["height_per_ndvi"] = h / ndvi.replace(0, np.nan)
        out["ndvi_x_height_sq"] = ndvi * h ** 2
        out["ndvi_sq_x_height"] = ndvi ** 2 * h

    if {"Pre_GSHH_NDVI", "sampling_month"}.issubset(out.columns):
        out["ndvi_x_month"] = out["Pre_GSHH_NDVI"] * out["sampling_month"]
        out["ndvi_x_dayofyear_sin"] = out["Pre_GSHH_NDVI"] * out["dayofyear_sin"]
        out["ndvi_x_dayofyear_cos"] = out["Pre_GSHH_NDVI"] * out["dayofyear_cos"]

    if {"Height_Ave_cm", "sampling_dayofyear"}.issubset(out.columns):
        out["height_x_dayofyear"] = out["Height_Ave_cm"] * out["sampling_dayofyear"]
        out["height_x_dayofyear_sin"] = out["Height_Ave_cm"] * out["dayofyear_sin"]

    if {"Pre_GSHH_NDVI", "Height_Ave_cm", "sampling_dayofyear"}.issubset(out.columns):
        out["ndvi_x_height_x_dayofyear"] = (
            out["Pre_GSHH_NDVI"] * out["Height_Ave_cm"] * out["sampling_dayofyear"] / 365
        )

    return out, stats


def split_xy(df: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[targets].copy()
    drop_cols = set(targets) | {"target", "target_name"}
    x = df[[c for c in df.columns if c not in drop_cols]].copy()
    return x, y
