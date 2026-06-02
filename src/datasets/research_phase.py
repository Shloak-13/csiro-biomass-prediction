from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=float)
META_COLS = ["Pre_GSHH_NDVI", "Height_Ave_cm", "State", "Species", "Sampling_Date"]
IMAGE_MODEL_IDS = {
    "dinov2": "facebook/dinov2-small",
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    return float(np.sum(WEIGHTS * np.array(scores)))


def make_wide(train: pd.DataFrame) -> pd.DataFrame:
    meta = (
        train[["image_path", *META_COLS]]
        .drop_duplicates("image_path")
        .sort_values("image_path")
        .reset_index(drop=True)
    )
    wide = train.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="first")
    wide = wide.reset_index()
    out = meta.merge(wide, on="image_path", how="left")
    out["image_id"] = out["image_path"].map(lambda x: Path(str(x)).stem)
    dates = pd.to_datetime(out["Sampling_Date"], errors="coerce")
    out["month"] = dates.dt.month
    out["dayofyear"] = dates.dt.dayofyear
    out["year"] = dates.dt.year
    out["season"] = out["month"].map(season)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["species_count"] = out["Species"].fillna("").astype(str).str.split(r"[,;/|+]").map(len)
    out["ndvi_x_height"] = out["Pre_GSHH_NDVI"] * out["Height_Ave_cm"]
    return out


def season(month: float) -> str:
    if pd.isna(month):
        return "unknown"
    month = int(month)
    if month in [12, 1, 2]:
        return "summer"
    if month in [3, 4, 5]:
        return "autumn"
    if month in [6, 7, 8]:
        return "winter"
    return "spring"


def stratified_folds(y: pd.Series, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    bins = pd.qcut(y.rank(method="first"), q=min(10, len(y) // n_splits), labels=False)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, bins))


def image_features(image_path: Path) -> dict[str, float | int | str]:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb.resize((384, 384)), dtype=np.float32) / 255.0
        full_w, full_h = img.size
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-6)
    exg = 2 * g - r - b
    green_ratio = (g > r * 1.05) & (g > b * 1.05)
    veg_mask = (exg > 0.05) & (g > 0.18)
    lap = ndimage.laplace(gray)
    sx = ndimage.sobel(gray, axis=0)
    sy = ndimage.sobel(gray, axis=1)
    hist_features: dict[str, float] = {}
    for channel_name, channel in [("r", r), ("g", g), ("b", b), ("gray", gray), ("exg", exg)]:
        hist, _ = np.histogram(channel, bins=8, range=(-1 if channel_name == "exg" else 0, 1), density=True)
        for idx, value in enumerate(hist):
            hist_features[f"hist_{channel_name}_{idx}"] = float(value)
    with Image.open(image_path) as raw:
        phash = str(imagehash.phash(raw))
        ahash = str(imagehash.average_hash(raw))
    return {
        "image_path": image_path.name,
        "width": full_w,
        "height": full_h,
        "aspect_ratio": full_w / full_h,
        "phash": phash,
        "ahash": ahash,
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "green_pixel_ratio": float(green_ratio.mean()),
        "vegetation_coverage_exg": float(veg_mask.mean()),
        "exg_mean": float(exg.mean()),
        "exg_std": float(exg.std()),
        "texture_laplacian_var": float(lap.var()),
        "texture_sobel_mean": float(np.sqrt(sx**2 + sy**2).mean()),
        "gray_entropy": float(stats.entropy(np.histogram(gray, bins=64, range=(0, 1))[0] + 1)),
        **hist_features,
    }


def compute_image_features(data_dir: Path, wide: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv)
    rows = []
    for rel in tqdm(wide["image_path"].tolist(), desc="Image features"):
        rows.append(image_features(data_dir / rel))
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def duplicate_report(image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dupes = image_df[image_df.duplicated("phash", keep=False)].sort_values("phash")
    near_rows = []
    hashes = [(row.image_path, imagehash.hex_to_hash(row.phash)) for row in image_df.itertuples()]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hashes[i][1] - hashes[j][1]
            if 0 < dist <= 4:
                near_rows.append({"image_a": hashes[i][0], "image_b": hashes[j][0], "phash_distance": dist})
    return dupes, pd.DataFrame(near_rows).sort_values("phash_distance") if near_rows else pd.DataFrame()


def feature_target_correlations(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue
        for target in TARGETS:
            corr = df[[feature, target]].corr(method="spearman").iloc[0, 1]
            rows.append({"feature": feature, "target": target, "spearman": corr, "abs_spearman": abs(corr)})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)


def metadata_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    features = [
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "State",
        "Species",
        "month",
        "dayofyear",
        "season",
        "month_sin",
        "month_cos",
        "species_count",
        "ndvi_x_height",
    ]
    x = df[features].copy()
    cat_cols = ["State", "Species", "season"]
    num_cols = [c for c in x.columns if c not in cat_cols]
    return x, num_cols, cat_cols


def cv_metadata_models(df: pd.DataFrame) -> pd.DataFrame:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    x, num_cols, cat_cols = metadata_matrix(df)
    y = df[TARGETS].values
    folds = stratified_folds(df["Dry_Total_g"])
    rows = []

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols)]
    )
    model_defs = [
        (
            "LightGBM_metadata",
            Pipeline(
                [
                    ("pre", pre),
                    (
                        "model",
                        MultiOutputRegressor(
                            LGBMRegressor(
                                n_estimators=450,
                                learning_rate=0.035,
                                num_leaves=15,
                                min_child_samples=12,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                random_state=42,
                                verbose=-1,
                            )
                        ),
                    ),
                ]
            ),
            None,
        ),
        (
            "CatBoost_metadata",
            CatBoostRegressor(
                loss_function="MultiRMSE",
                iterations=700,
                learning_rate=0.035,
                depth=5,
                l2_leaf_reg=6,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
            [x.columns.get_loc(c) for c in cat_cols],
        ),
    ]
    for name, model, cat_idx in model_defs:
        oof = np.zeros_like(y)
        fold_scores = []
        for fold, (trn, val) in enumerate(folds):
            if cat_idx is None:
                model.fit(x.iloc[trn], np.log1p(y[trn]))
                pred = np.expm1(model.predict(x.iloc[val]))
            else:
                train_pool_x = x.iloc[trn].copy()
                val_pool_x = x.iloc[val].copy()
                for col in cat_cols:
                    train_pool_x[col] = train_pool_x[col].astype(str)
                    val_pool_x[col] = val_pool_x[col].astype(str)
                model.fit(train_pool_x, np.log1p(y[trn]), cat_features=cat_idx)
                pred = np.expm1(model.predict(val_pool_x))
            pred = np.clip(pred, 0, None)
            oof[val] = pred
            fold_scores.append(weighted_r2(y[val], pred))
        rows.append(
            {
                "model": name,
                "cv_weighted_r2": weighted_r2(y, oof),
                **{f"fold_{i}": score for i, score in enumerate(fold_scores)},
            }
        )
    return pd.DataFrame(rows).sort_values("cv_weighted_r2", ascending=False)


def cv_mixed_metadata_numeric(df: pd.DataFrame, numeric_extra_cols: list[str], prefix: str) -> pd.DataFrame:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    meta_x, meta_num_cols, cat_cols = metadata_matrix(df)
    x = pd.concat(
        [meta_x.reset_index(drop=True), df[numeric_extra_cols].reset_index(drop=True)],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    num_cols = meta_num_cols + numeric_extra_cols
    for col in num_cols:
        x[col] = x[col].fillna(x[col].median())
    y = df[TARGETS].values
    folds = stratified_folds(df["Dry_Total_g"])
    rows = []
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols)]
    )
    model_defs = [
        (
            f"LightGBM_{prefix}",
            Pipeline(
                [
                    ("pre", pre),
                    (
                        "model",
                        MultiOutputRegressor(
                            LGBMRegressor(
                                n_estimators=500,
                                learning_rate=0.03,
                                num_leaves=15,
                                min_child_samples=10,
                                subsample=0.9,
                                colsample_bytree=0.85,
                                random_state=42,
                                verbose=-1,
                            )
                        ),
                    ),
                ]
            ),
            None,
        ),
        (
            f"CatBoost_{prefix}",
            CatBoostRegressor(
                loss_function="MultiRMSE",
                iterations=650,
                learning_rate=0.035,
                depth=5,
                l2_leaf_reg=6,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
            [x.columns.get_loc(c) for c in cat_cols],
        ),
    ]
    for name, model, cat_idx in model_defs:
        oof = np.zeros_like(y)
        fold_scores = []
        for trn, val in folds:
            if cat_idx is None:
                model.fit(x.iloc[trn], np.log1p(y[trn]))
                pred = np.expm1(model.predict(x.iloc[val]))
            else:
                x_trn = x.iloc[trn].copy()
                x_val = x.iloc[val].copy()
                for col in cat_cols:
                    x_trn[col] = x_trn[col].astype(str)
                    x_val[col] = x_val[col].astype(str)
                model.fit(x_trn, np.log1p(y[trn]), cat_features=cat_idx)
                pred = np.expm1(model.predict(x_val))
            pred = np.clip(pred, 0, None)
            oof[val] = pred
            fold_scores.append(weighted_r2(y[val], pred))
        rows.append({"model": name, "cv_weighted_r2": weighted_r2(y, oof), "fold_scores": fold_scores})
    return pd.DataFrame(rows).sort_values("cv_weighted_r2", ascending=False)


def cv_numeric_models(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[TARGETS].values
    folds = stratified_folds(df["Dry_Total_g"])
    models = [
        (
            f"LightGBM_{prefix}",
            MultiOutputRegressor(
                LGBMRegressor(
                    n_estimators=350,
                    learning_rate=0.04,
                    num_leaves=15,
                    min_child_samples=10,
                    random_state=42,
                    verbose=-1,
                )
            ),
        ),
        (
            f"CatBoost_{prefix}",
            CatBoostRegressor(
                loss_function="MultiRMSE",
                iterations=450,
                learning_rate=0.04,
                depth=4,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
        ),
    ]
    rows = []
    for name, model in models:
        oof = np.zeros_like(y)
        fold_scores = []
        for trn, val in folds:
            model.fit(x.iloc[trn], np.log1p(y[trn]))
            pred = np.clip(np.expm1(model.predict(x.iloc[val])), 0, None)
            oof[val] = pred
            fold_scores.append(weighted_r2(y[val], pred))
        rows.append({"model": name, "cv_weighted_r2": weighted_r2(y, oof), "fold_scores": fold_scores})
    return pd.DataFrame(rows).sort_values("cv_weighted_r2", ascending=False)


def extract_embedding_matrix(
    data_dir: Path, df: pd.DataFrame, model_key: str, subset_n: int, cache_dir: Path
) -> tuple[pd.DataFrame, np.ndarray | None, str | None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = df.sort_values("image_path").head(subset_n).copy()
    out_npy = cache_dir / f"{model_key}_{subset_n}.npy"
    out_csv = cache_dir / f"{model_key}_{subset_n}_rows.csv"
    if out_npy.exists() and out_csv.exists():
        return pd.read_csv(out_csv), np.load(out_npy), None
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = IMAGE_MODEL_IDS[model_key]
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        vectors = []
        batch_size = 8 if device == "cuda" else 4
        for start in tqdm(range(0, len(rows), batch_size), desc=f"{model_key} embeddings"):
            batch = rows.iloc[start : start + batch_size]
            images = [Image.open(data_dir / p).convert("RGB") for p in batch["image_path"]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    emb = model.get_image_features(pixel_values=inputs["pixel_values"])
                    if not torch.is_tensor(emb):
                        if hasattr(emb, "image_embeds") and emb.image_embeds is not None:
                            emb = emb.image_embeds
                        elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                            emb = emb.pooler_output
                        elif hasattr(emb, "last_hidden_state"):
                            emb = emb.last_hidden_state.mean(dim=1)
                        else:
                            raise TypeError(f"Unsupported image feature return type: {type(emb)!r}")
                else:
                    output = model(**inputs)
                    if hasattr(output, "pooler_output") and output.pooler_output is not None:
                        emb = output.pooler_output
                    else:
                        emb = output.last_hidden_state.mean(dim=1)
            vectors.append(emb.float().cpu().numpy())
        mat = np.concatenate(vectors, axis=0)
        np.save(out_npy, mat)
        rows.to_csv(out_csv, index=False)
        return rows, mat, None
    except Exception as exc:
        return rows, None, repr(exc)


def embedding_experiment(data_dir: Path, df: pd.DataFrame, reports_dir: Path, subset_n: int) -> pd.DataFrame:
    all_rows = []
    failures = []
    cache_dir = data_dir.parent / "embedding_cache"
    for model_key in IMAGE_MODEL_IDS:
        rows, emb, error = extract_embedding_matrix(data_dir, df, model_key, subset_n, cache_dir)
        if error:
            failures.append({"embedding": model_key, "error": error})
            continue
        emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
        emb_df = pd.concat(
            [rows[["image_path", *TARGETS]].reset_index(drop=True), pd.DataFrame(emb, columns=emb_cols)],
            axis=1,
        )
        result = cv_numeric_models(emb_df, emb_cols, model_key)
        result.insert(0, "embedding", model_key)
        result.insert(1, "subset_n", len(rows))
        all_rows.append(result)
    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if failures:
        pd.DataFrame(failures).to_csv(reports_dir / "embedding_failures.csv", index=False)
    else:
        failure_path = reports_dir / "embedding_failures.csv"
        if failure_path.exists():
            failure_path.unlink()
    return out


def top_image_feature_importance(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[TARGETS].values
    model = ExtraTreesRegressor(n_estimators=600, random_state=42, min_samples_leaf=2, n_jobs=-1)
    model.fit(x, y)
    return pd.DataFrame({"feature": feature_cols, "extra_trees_importance": model.feature_importances_}).sort_values(
        "extra_trees_importance", ascending=False
    )


def write_dataset_audit(
    reports_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    wide: pd.DataFrame,
    image_df: pd.DataFrame,
    duplicates: pd.DataFrame,
    near_duplicates: pd.DataFrame,
    target_corr: pd.DataFrame,
) -> None:
    missing = train.isna().sum().to_frame("missing_train")
    test_missing = test.isna().sum().to_frame("missing_test")
    target_desc = wide[TARGETS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    lines = [
        "# Dataset Audit",
        "",
        "## Dimensions",
        f"- `train.csv`: {train.shape[0]} rows x {train.shape[1]} columns",
        f"- `test.csv`: {test.shape[0]} rows x {test.shape[1]} columns",
        f"- `sample_submission.csv`: {sample.shape[0]} rows x {sample.shape[1]} columns",
        f"- Unique train images: {wide['image_path'].nunique()}",
        f"- Unique test images: {test['image_path'].nunique() if 'image_path' in test else 'unknown'}",
        f"- Number of targets: {train['target_name'].nunique()} ({', '.join(sorted(train['target_name'].unique()))})",
        "",
        "## Missing Values",
        missing.to_markdown(),
        "",
        "### Test Missing Values",
        test_missing.to_markdown(),
        "",
        "## Duplicate Rows and Images",
        f"- Full duplicate train rows: {int(train.duplicated().sum())}",
        f"- Duplicate perceptual hashes: {len(duplicates)} rows",
        f"- Near-duplicate pairs with pHash distance <= 4: {len(near_duplicates)}",
        "",
        "## Image Resolutions and Aspect Ratios",
        image_df[["width", "height", "aspect_ratio"]].describe().to_markdown(),
        "",
        "## Date Distribution",
        wide["Sampling_Date"].value_counts().sort_index().to_markdown(),
        "",
        "## State Distribution",
        wide["State"].value_counts().to_markdown(),
        "",
        "## Species Distribution",
        wide["Species"].value_counts().head(40).to_markdown(),
        "",
        "## Target Distributions",
        target_desc.to_markdown(),
        "",
        "## Target Correlations",
        target_corr.to_markdown(),
    ]
    (reports_dir / "dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_winning_strategy(
    reports_dir: Path,
    metadata_scores: pd.DataFrame,
    image_scores: pd.DataFrame,
    embedding_scores: pd.DataFrame,
    constraint: dict,
) -> None:
    best_meta = float(metadata_scores["cv_weighted_r2"].max()) if not metadata_scores.empty else math.nan
    best_img = float(image_scores["cv_weighted_r2"].max()) if not image_scores.empty else math.nan
    best_emb = float(embedding_scores["cv_weighted_r2"].max()) if not embedding_scores.empty else math.nan
    constraint_help = constraint["median_abs_error"] < 1e-3
    rows = [
        ["Metadata + image features GBDT", max(best_meta, best_img), "Low", "5-15 min", "No", "+baseline to strong CV"],
        ["DINOv2/CLIP/SigLIP embeddings + GBDT", best_emb, "Medium", "30-90 min CPU/GPU", "Helpful", "+large if embeddings beat metadata"],
        ["Multimodal GBDT stack: metadata + embeddings + image stats", max(best_meta, best_emb), "Medium", "1-2 h", "Optional", "+most likely immediate gain"],
        ["OOF weighted ensemble across metadata, image stats, embeddings", max(best_meta, best_img, best_emb), "Low", "minutes", "No", "+robust private LB"],
        ["Constraint-aware post-processing for total/GDM", best_meta, "Low", "minutes", "No", "+high if train identity holds"],
        ["Crop/resize audit and canopy-focused crops", best_emb, "Medium", "1-3 h", "Helpful", "+likely if plot frame/background dominates"],
        ["Fine-tune ConvNeXtV2/EVA after embedding proof", best_emb + 0.03 if not math.isnan(best_emb) else math.nan, "High", "4-12 h", "Yes", "+possible but later"],
        ["Pseudo-labeling test after stable validation", best_emb, "Medium", "1-3 h", "No/Optional", "+only after public/test size known"],
        ["Date/state calibrated residual model", best_meta, "Medium", "30-60 min", "No", "+private robustness"],
        ["External weather/satellite joins", best_meta, "High", "4-10 h", "No", "+uncertain; depends on location granularity"],
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "Approach",
            "Expected CV score anchor",
            "Complexity",
            "Training time",
            "GPU requirements",
            "Expected leaderboard gain",
        ],
    )
    text = [
        "# Winning Strategy",
        "",
        "## Core Findings",
        f"- Best metadata-only CV weighted R2 observed: `{best_meta:.5f}`",
        f"- Best hand-crafted image-feature CV weighted R2 observed: `{best_img:.5f}`",
        f"- Best subset embedding CV weighted R2 observed: `{best_emb:.5f}`" if not math.isnan(best_emb) else "- Embedding CV could not complete; see embedding report.",
        f"- Biological total constraint is {'strongly justified' if constraint_help else 'not exact enough to hard-enforce blindly'} based on median absolute residual.",
        "",
        "## Ranked Top 10 Approaches",
        table.to_markdown(index=False),
        "",
        "## Recommended Next Experiment",
        "",
        "Run the multimodal frozen-embedding experiment on all train images, then blend with metadata:",
        "",
        "```powershell",
        "cd D:\\csiro\\CSIRO-Biomass",
        "& 'C:\\Users\\sapna shetty\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' -m src.datasets.research_phase --data-dir data\\csiro-biomass --embedding-subset 357",
        "```",
    ]
    (reports_dir / "winning_strategy.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/csiro-biomass"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--embedding-subset", type=int, default=96)
    args = parser.parse_args()

    start = time.time()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    wide = make_wide(train)
    wide.to_csv(args.reports_dir / "train_wide.csv", index=False)

    image_df = compute_image_features(args.data_dir, wide, args.reports_dir / "image_features.csv")
    merged = wide.merge(image_df, left_on=wide["image_path"].map(lambda x: Path(str(x)).name), right_on="image_path", suffixes=("", "_img"))
    duplicates, near_duplicates = duplicate_report(image_df)
    duplicates.to_csv(args.reports_dir / "duplicate_images.csv", index=False)
    near_duplicates.to_csv(args.reports_dir / "near_duplicate_images.csv", index=False)

    target_corr = wide[TARGETS].corr()
    target_corr.to_csv(args.reports_dir / "target_correlations.csv")
    write_dataset_audit(args.reports_dir, train, test, sample, wide, image_df, duplicates, near_duplicates, target_corr)

    comp_sum = wide["Dry_Green_g"] + wide["Dry_Dead_g"] + wide["Dry_Clover_g"]
    residual = wide["Dry_Total_g"] - comp_sum
    constraint = {
        "mean_error": float(residual.mean()),
        "median_error": float(residual.median()),
        "median_abs_error": float(residual.abs().median()),
        "max_abs_error": float(residual.abs().max()),
        "correlation_total_vs_component_sum": float(wide["Dry_Total_g"].corr(comp_sum)),
        "pct_abs_error_le_1e_6": float((residual.abs() <= 1e-6).mean()),
        "pct_abs_error_le_1g": float((residual.abs() <= 1.0).mean()),
    }
    gdm_residual = wide["GDM_g"] - (wide["Dry_Green_g"] + wide["Dry_Clover_g"])
    constraint["gdm_median_abs_error"] = float(gdm_residual.abs().median())
    constraint["gdm_pct_abs_error_le_1g"] = float((gdm_residual.abs() <= 1.0).mean())
    pd.Series(constraint).to_json(args.reports_dir / "biomass_relationships.json", indent=2)
    residual.to_frame("dry_total_minus_components").to_csv(args.reports_dir / "biomass_residuals.csv", index=False)

    image_feature_cols = [
        c
        for c in merged.columns
        if c.startswith(("brightness", "saturation", "green_", "vegetation_", "exg_", "texture_", "gray_", "hist_"))
        or c in ["width", "height", "aspect_ratio"]
    ]
    img_corr = feature_target_correlations(merged, image_feature_cols)
    img_corr.to_csv(args.reports_dir / "image_feature_target_correlations.csv", index=False)
    img_importance = top_image_feature_importance(merged, image_feature_cols)
    img_importance.to_csv(args.reports_dir / "image_feature_importance.csv", index=False)
    image_scores = cv_numeric_models(merged, image_feature_cols, "image_features")
    image_scores.to_csv(args.reports_dir / "image_feature_cv_scores.csv", index=False)
    mixed_scores = cv_mixed_metadata_numeric(merged, image_feature_cols, "metadata_image_features")
    mixed_scores.to_csv(args.reports_dir / "metadata_image_feature_cv_scores.csv", index=False)

    metadata_scores = cv_metadata_models(wide)
    metadata_scores.to_csv(args.reports_dir / "metadata_cv_scores.csv", index=False)

    embedding_scores = embedding_experiment(args.data_dir, wide, args.reports_dir, args.embedding_subset)
    if not embedding_scores.empty:
        embedding_scores.to_csv(args.reports_dir / "embedding_cv_scores.csv", index=False)

    embedding_lines = [
        "# Embedding Comparison",
        "",
        f"Subset size requested: {args.embedding_subset}",
        "",
        "## CV Scores",
        embedding_scores.to_markdown(index=False) if not embedding_scores.empty else "No embedding model completed.",
    ]
    failures_path = args.reports_dir / "embedding_failures.csv"
    if failures_path.exists():
        embedding_lines.extend(["", "## Failures", pd.read_csv(failures_path).to_markdown(index=False)])
    (args.reports_dir / "embedding_comparison.md").write_text("\n".join(embedding_lines), encoding="utf-8")

    biomass_lines = [
        "# Biomass Relationships",
        "",
        "## Dry_Total_g vs component sum",
        pd.Series(constraint).to_frame("value").to_markdown(),
        "",
        "Interpretation: "
        + (
            "The total is effectively an identity of dry green, dead, and clover biomass. Constraint-aware modeling/post-processing is justified."
            if constraint["median_abs_error"] < 1e-3
            else "The relationship is strong but not exact. Use a soft constraint or validation-tested post-processing rather than forcing every prediction."
        ),
    ]
    (args.reports_dir / "biomass_relationships.md").write_text("\n".join(biomass_lines), encoding="utf-8")

    image_lines = [
        "# Image Analysis",
        "",
        "## Top Predictive Image Features by Spearman Correlation",
        img_corr.head(30).to_markdown(index=False),
        "",
        "## Top Predictive Image Features by ExtraTrees Importance",
        img_importance.head(30).to_markdown(index=False),
        "",
        "## Hand-crafted Image Feature CV Scores",
        image_scores.to_markdown(index=False),
        "",
        "## Metadata + Hand-crafted Image Feature CV Scores",
        mixed_scores.to_markdown(index=False),
    ]
    (args.reports_dir / "image_analysis.md").write_text("\n".join(image_lines), encoding="utf-8")

    metadata_lines = [
        "# Metadata Importance",
        "",
        "## Metadata-only CV Scores",
        metadata_scores.to_markdown(index=False),
        "",
        "## Metadata + Image Feature CV Scores",
        mixed_scores.to_markdown(index=False),
        "",
        "## Interpretation",
        "Metadata dominates the quick research phase if its CV exceeds image-only and subset embedding baselines. A metadata+image lift would justify lightweight multimodal fusion before end-to-end CNN training.",
    ]
    (args.reports_dir / "metadata_importance.md").write_text("\n".join(metadata_lines), encoding="utf-8")

    winning_image_scores = pd.concat([image_scores, mixed_scores], ignore_index=True)
    write_winning_strategy(args.reports_dir, metadata_scores, winning_image_scores, embedding_scores, constraint)
    manifest = {
        "elapsed_seconds": round(time.time() - start, 2),
        "n_train_rows": len(train),
        "n_unique_train_images": int(wide["image_path"].nunique()),
        "embedding_subset": args.embedding_subset,
    }
    (args.reports_dir / "research_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
