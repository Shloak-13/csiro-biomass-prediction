from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from src.utils.metrics import weighted_log_r2

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
COMPONENT_TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g"]
WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=float)
META_BASE = ["Pre_GSHH_NDVI", "Height_Ave_cm", "State", "Species", "Sampling_Date"]


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    backend: str
    model_id: str
    batch_size: int
    feasible_cpu: bool = True


EMBEDDINGS = [
    EmbeddingSpec("dinov2_large", "transformers", "facebook/dinov2-large", 2),
    EmbeddingSpec("dinov2_giant", "transformers", "facebook/dinov2-giant", 1, feasible_cpu=False),
    EmbeddingSpec("siglip_base", "transformers", "google/siglip-base-patch16-224", 2),
    EmbeddingSpec("convnextv2_base", "timm", "convnextv2_base.fcmae_ft_in22k_in1k", 2),
    EmbeddingSpec("eva02_base", "timm", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", 1, feasible_cpu=False),
]


def memory_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Deprecated: use weighted_log_r2 from src.utils.metrics instead (competition uses log-space R2)."""
    return weighted_log_r2(y_true, y_pred)


def target_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {target: float(r2_score(y_true[:, i], y_pred[:, i])) for i, target in enumerate(TARGETS)}


def make_wide(train: pd.DataFrame) -> pd.DataFrame:
    meta = train[["image_path", *META_BASE]].drop_duplicates("image_path").sort_values("image_path")
    y = train.pivot_table(index="image_path", columns="target_name", values="target", aggfunc="first")
    out = meta.merge(y.reset_index(), on="image_path", how="left").reset_index(drop=True)
    dates = pd.to_datetime(out["Sampling_Date"], errors="coerce")
    out["month"] = dates.dt.month
    out["dayofyear"] = dates.dt.dayofyear
    out["year"] = dates.dt.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["season"] = out["month"].map(season)
    out["species_count"] = out["Species"].fillna("").astype(str).str.split(r"[,;/|+]").map(len)
    out["ndvi_x_height"] = out["Pre_GSHH_NDVI"] * out["Height_Ave_cm"]
    out["height_per_ndvi"] = out["Height_Ave_cm"] / out["Pre_GSHH_NDVI"].replace(0, np.nan)
    return out


def season(month: float) -> str:
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


def folds(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    bins = pd.qcut(df["Dry_Total_g"].rank(method="first"), q=10, labels=False)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(df, bins))


def metadata_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    cols = [
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "State",
        "Species",
        "month",
        "dayofyear",
        "year",
        "month_sin",
        "month_cos",
        "season",
        "species_count",
        "ndvi_x_height",
        "height_per_ndvi",
    ]
    x = df[cols].copy()
    cat_cols = ["State", "Species", "season"]
    num_cols = [c for c in x.columns if c not in cat_cols]
    for col in num_cols:
        x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(x[col].median())
    return x, num_cols, cat_cols


def image_transform(size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def tensor_from_images(paths: list[Path], transform):
    import torch

    images = [transform(Image.open(path).convert("RGB")) for path in paths]
    return torch.stack(images)


def extract_transformers(spec: EmbeddingSpec, data_dir: Path, image_paths: list[str], out: Path) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    start = time.time()
    mem0 = memory_mb()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(spec.model_id, use_fast=True)
    model = AutoModel.from_pretrained(spec.model_id).to(device).eval()
    vectors = []
    for i in tqdm(range(0, len(image_paths), spec.batch_size), desc=spec.name):
        batch_paths = [data_dir / p for p in image_paths[i : i + spec.batch_size]]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                emb = model.get_image_features(pixel_values=inputs["pixel_values"])
                if not torch.is_tensor(emb):
                    if hasattr(emb, "image_embeds") and emb.image_embeds is not None:
                        emb = emb.image_embeds
                    elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                        emb = emb.pooler_output
                    else:
                        emb = emb.last_hidden_state.mean(dim=1)
            else:
                y = model(**inputs)
                emb = y.pooler_output if getattr(y, "pooler_output", None) is not None else y.last_hidden_state.mean(dim=1)
        vectors.append(emb.float().cpu().numpy())
    mat = np.concatenate(vectors, axis=0).astype("float32")
    np.save(out, mat)
    dim = int(mat.shape[1])
    del model
    gc.collect()
    return {"embedding": spec.name, "dim": dim, "runtime_sec": time.time() - start, "memory_delta_mb": memory_mb() - mem0, "status": "ok"}


def extract_timm(spec: EmbeddingSpec, data_dir: Path, image_paths: list[str], out: Path) -> dict:
    import timm
    import torch

    start = time.time()
    mem0 = memory_mb()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(spec.model_id, pretrained=True, num_classes=0).to(device).eval()
    transform = image_transform(224)
    vectors = []
    for i in tqdm(range(0, len(image_paths), spec.batch_size), desc=spec.name):
        batch_paths = [data_dir / p for p in image_paths[i : i + spec.batch_size]]
        x = tensor_from_images(batch_paths, transform).to(device)
        with torch.no_grad():
            emb = model(x)
            if emb.ndim > 2:
                emb = emb.mean(dim=tuple(range(2, emb.ndim)))
        vectors.append(emb.float().cpu().numpy())
    mat = np.concatenate(vectors, axis=0).astype("float32")
    np.save(out, mat)
    dim = int(mat.shape[1])
    del model
    gc.collect()
    return {"embedding": spec.name, "dim": dim, "runtime_sec": time.time() - start, "memory_delta_mb": memory_mb() - mem0, "status": "ok"}


def extract_embeddings(data_dir: Path, df: pd.DataFrame, cache_dir: Path, allow_huge: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    image_paths = df["image_path"].tolist()
    pd.DataFrame({"image_path": image_paths}).to_csv(cache_dir / "embedding_rows.csv", index=False)
    for spec in EMBEDDINGS:
        out = cache_dir / f"{spec.name}.npy"
        meta = {"embedding": spec.name, "model_id": spec.model_id}
        if out.exists():
            mat = np.load(out, mmap_mode="r")
            rows.append({**meta, "dim": int(mat.shape[1]), "runtime_sec": 0.0, "memory_delta_mb": 0.0, "status": "cached"})
            continue
        if (not spec.feasible_cpu) and (not allow_huge):
            rows.append({**meta, "dim": np.nan, "runtime_sec": 0.0, "memory_delta_mb": 0.0, "status": "skipped_cpu_memory_risk"})
            continue
        try:
            if spec.backend == "transformers":
                result = extract_transformers(spec, data_dir, image_paths, out)
            else:
                result = extract_timm(spec, data_dir, image_paths, out)
            rows.append({**meta, **result})
        except Exception as exc:
            rows.append({**meta, "dim": np.nan, "runtime_sec": np.nan, "memory_delta_mb": np.nan, "status": f"failed: {repr(exc)}"})
            if out.exists():
                out.unlink()
    return pd.DataFrame(rows)


def build_regressor(name: str):
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=260,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                random_state=42,
                verbose=-1,
            )
        )
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=280,
            learning_rate=0.045,
            depth=5,
            l2_leaf_reg=8,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=220,
                learning_rate=0.045,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
        )
    raise ValueError(name)


def cv_tabular(
    df: pd.DataFrame,
    x: pd.DataFrame,
    model_name: str,
    cat_cols: list[str] | None,
    targets: list[str] = TARGETS,
    derive_total: bool = False,
    force_total_constraint: bool = False,
) -> dict:
    y_full = df[TARGETS].values
    y_train = df[targets].values
    splits = folds(df)
    oof = np.zeros_like(y_full)
    fold_scores = []
    start = time.time()
    mem0 = memory_mb()
    for trn, val in splits:
        model = build_regressor(model_name)
        if cat_cols:
            x_trn = x.iloc[trn].copy()
            x_val = x.iloc[val].copy()
            for col in cat_cols:
                x_trn[col] = x_trn[col].astype(str)
                x_val[col] = x_val[col].astype(str)
            cat_idx = [x.columns.get_loc(c) for c in cat_cols]
            if model_name == "catboost":
                model.fit(x_trn, np.log1p(y_train[trn]), cat_features=cat_idx)
                pred_small = np.expm1(model.predict(x_val))
            else:
                num_cols = [c for c in x.columns if c not in cat_cols]
                pre = ColumnTransformer(
                    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", StandardScaler(), num_cols)]
                )
                pipe = Pipeline([("pre", pre), ("model", model)])
                pipe.fit(x_trn, np.log1p(y_train[trn]))
                pred_small = np.expm1(pipe.predict(x_val))
        else:
            x_num = x.replace([np.inf, -np.inf], np.nan).fillna(0)
            model.fit(x_num.iloc[trn], np.log1p(y_train[trn]))
            pred_small = np.expm1(model.predict(x_num.iloc[val]))
        pred_small = np.clip(pred_small, 0, None)
        if targets == TARGETS:
            pred = pred_small
        else:
            pred = np.zeros((len(val), len(TARGETS)), dtype=float)
            for i, target in enumerate(targets):
                pred[:, TARGETS.index(target)] = pred_small[:, i]
            if derive_total:
                pred[:, TARGETS.index("Dry_Total_g")] = (
                    pred[:, TARGETS.index("Dry_Green_g")]
                    + pred[:, TARGETS.index("Dry_Dead_g")]
                    + pred[:, TARGETS.index("Dry_Clover_g")]
                )
        if force_total_constraint:
            pred[:, TARGETS.index("Dry_Total_g")] = (
                pred[:, TARGETS.index("Dry_Green_g")]
                + pred[:, TARGETS.index("Dry_Dead_g")]
                + pred[:, TARGETS.index("Dry_Clover_g")]
            )
        oof[val] = pred
        fold_scores.append(weighted_r2(y_full[val], pred))
    return {
        "model": model_name,
        "cv_weighted_r2": weighted_r2(y_full, oof),
        **target_scores(y_full, oof),
        "fold_scores": json.dumps(fold_scores),
        "runtime_sec": time.time() - start,
        "memory_delta_mb": memory_mb() - mem0,
    }


def embedding_frame(cache_dir: Path, name: str) -> pd.DataFrame:
    mat = np.load(cache_dir / f"{name}.npy")
    return pd.DataFrame(mat, columns=[f"{name}_{i}" for i in range(mat.shape[1])])


def run_benchmarks(df: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_x, _, meta_cat = metadata_frame(df)
    ok_embeddings = [p.stem for p in cache_dir.glob("*.npy")]
    models = ["catboost", "lightgbm", "xgboost"]
    emb_rows = []
    fusion_rows = []

    for model_name in models:
        fusion_rows.append({"feature_set": "A_metadata_only", "embedding": "none", **cv_tabular(df, meta_x, model_name, meta_cat)})

    for emb in ok_embeddings:
        emb_x = embedding_frame(cache_dir, emb)
        for model_name in models:
            score = cv_tabular(df, emb_x, model_name, None)
            emb_rows.append({"embedding": emb, "feature_set": "B_embeddings_only", **score})
            fusion_rows.append({"feature_set": "B_embeddings_only", "embedding": emb, **score})
        fused = pd.concat([meta_x.reset_index(drop=True), emb_x.reset_index(drop=True)], axis=1)
        for model_name in models:
            fusion_rows.append({"feature_set": "C_metadata_plus_embeddings", "embedding": emb, **cv_tabular(df, fused, model_name, meta_cat)})

    target_rows = []
    for model_name in models:
        target_rows.append({"approach": "predict_all_5_independently", **cv_tabular(df, meta_x, model_name, meta_cat)})
        target_rows.append(
            {
                "approach": "predict_5_posthoc_total_constraint",
                **cv_tabular(df, meta_x, model_name, meta_cat, force_total_constraint=True),
            }
        )
        target_rows.append(
            {
                "approach": "predict_4_derive_total",
                **cv_tabular(df, meta_x, model_name, meta_cat, targets=COMPONENT_TARGETS, derive_total=True),
            }
        )
    return pd.DataFrame(emb_rows), pd.DataFrame(fusion_rows), pd.DataFrame(target_rows)


def leakage_and_shap(df: pd.DataFrame, reports_dir: Path) -> pd.DataFrame:
    import shap
    from lightgbm import LGBMRegressor

    meta_x, num_cols, cat_cols = metadata_frame(df)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols), ("num", "passthrough", num_cols)],
        verbose_feature_names_out=False,
    )
    x_enc = pd.DataFrame(pre.fit_transform(meta_x), columns=pre.get_feature_names_out())
    rows = []
    reports_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
        )
        model.fit(x_enc, np.log1p(df[target].values))
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_enc)
        mean_abs = np.abs(values).mean(axis=0)
        imp = pd.DataFrame({"feature": x_enc.columns, "mean_abs_shap": mean_abs, "target": target})
        rows.append(imp)
        shap.summary_plot(values, x_enc, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(reports_dir / f"shap_{target}.png", dpi=160, bbox_inches="tight")
        plt.close()
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(reports_dir / "metadata_shap_importance.csv", index=False)
    return out


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    show = df if max_rows is None else df.head(max_rows)
    return show.to_markdown(index=False)


def write_reports(
    reports_dir: Path,
    extract_meta: pd.DataFrame,
    emb_scores: pd.DataFrame,
    fusion_scores: pd.DataFrame,
    target_scores_df: pd.DataFrame,
    shap_imp: pd.DataFrame,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    extract_meta.to_csv(reports_dir / "full_embedding_extraction.csv", index=False)
    emb_scores.to_csv(reports_dir / "full_embedding_cv_scores.csv", index=False)
    fusion_scores.to_csv(reports_dir / "multimodal_cv_scores.csv", index=False)
    target_scores_df.to_csv(reports_dir / "target_engineering_cv_scores.csv", index=False)

    full = emb_scores.merge(
        extract_meta[["embedding", "dim", "runtime_sec", "memory_delta_mb", "status"]],
        on="embedding",
        how="left",
        suffixes=("", "_extract"),
    )
    (reports_dir / "full_embedding_benchmark.md").write_text(
        "\n".join(
            [
                "# Full Embedding Benchmark",
                "",
                "All feasible embeddings were extracted on all 357 unique training images. DINOv2 Giant and EVA-02 are marked according to runtime feasibility/status.",
                "",
                "## Extraction Metadata",
                md_table(extract_meta),
                "",
                "## CV Scores",
                md_table(full.sort_values("cv_weighted_r2", ascending=False)),
            ]
        ),
        encoding="utf-8",
    )

    best_meta = fusion_scores[fusion_scores["feature_set"] == "A_metadata_only"]["cv_weighted_r2"].max()
    best_fused = fusion_scores[fusion_scores["feature_set"] == "C_metadata_plus_embeddings"]["cv_weighted_r2"].max()
    gain = best_fused - best_meta if not math.isnan(best_fused) else math.nan
    (reports_dir / "multimodal_benchmark.md").write_text(
        "\n".join(
            [
                "# Multimodal Benchmark",
                "",
                f"- Best metadata-only CV weighted R2: `{best_meta:.6f}`",
                f"- Best metadata + embedding CV weighted R2: `{best_fused:.6f}`" if not math.isnan(best_fused) else "- No fused embedding model completed.",
                f"- Incremental gain from embeddings: `{gain:.6f}`" if not math.isnan(gain) else "- Incremental gain unavailable.",
                "",
                "## All Scores",
                md_table(fusion_scores.sort_values("cv_weighted_r2", ascending=False)),
            ]
        ),
        encoding="utf-8",
    )

    shap_top = shap_imp.groupby("feature", as_index=False)["mean_abs_shap"].mean().sort_values("mean_abs_shap", ascending=False)
    (reports_dir / "target_engineering.md").write_text(
        "\n".join(["# Target Engineering", "", md_table(target_scores_df.sort_values("cv_weighted_r2", ascending=False))]),
        encoding="utf-8",
    )
    best_row = fusion_scores.sort_values("cv_weighted_r2", ascending=False).iloc[0].to_dict()
    answers = [
        "# Decision Report",
        "",
        "## Answers",
        f"1. Image information adds value beyond metadata: {'yes' if gain > 0 else 'not yet in this benchmark'}.",
        f"2. CV gain from the best fused model: `{gain:.6f}`." if not math.isnan(gain) else "2. CV gain could not be measured.",
        f"3. Multimodal fusion is {'justified' if gain > 0.005 else 'not justified yet without better embeddings/crops'} based on this benchmark.",
        f"4. Highest ROI model now: `{best_row['model']}` with `{best_row['feature_set']}` / `{best_row['embedding']}` at CV `{best_row['cv_weighted_r2']:.6f}`.",
        "5. Train next: tune the best metadata/constraint model first, then retry full embeddings with canopy-aware crops if fusion is not positive.",
        "",
        "## Metadata Importance / Leakage Check",
        "High NDVI/height SHAP importance means they are strong proxies, but not proof of leakage. They are provided in train metadata and absent from the sample test metadata here, so deployment/test schema must be verified before relying on them.",
        "",
        md_table(shap_top.head(30)),
        "",
        "## Top 5 Next Experiments",
        "| Rank | Experiment | Rationale |",
        "|---:|:---|:---|",
        "| 1 | Hyperparameter tune metadata LightGBM/CatBoost/XGBoost with target constraints | Highest current CV and fastest iteration |",
        "| 2 | Validate `predict_4_derive_total` plus GDM consistency across folds | Biomass identity is almost exact and high weight sits on total |",
        "| 3 | Full embedding fusion with canopy crop/resize variants | Current raw embeddings may underuse 2000x1000 plot detail |",
        "| 4 | OOF weighted ensemble of metadata, constrained, and best embedding/fusion models | Low effort, usually robust |",
        "| 5 | Group/date/state validation stress test and residual calibration | Fold instability implies distribution shift risk |",
    ]
    (reports_dir / "decision_report.md").write_text("\n".join(answers), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/csiro-biomass"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/embedding_cache_full"))
    parser.add_argument("--allow-huge", action="store_true", help="Attempt DINOv2 Giant/EVA-02 even on CPU.")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    wide = make_wide(train)
    extract_meta = extract_embeddings(args.data_dir, wide, args.cache_dir, args.allow_huge)
    emb_scores, fusion_scores, target_scores_df = run_benchmarks(wide, args.cache_dir)
    shap_imp = leakage_and_shap(wide, args.reports_dir)
    write_reports(args.reports_dir, extract_meta, emb_scores, fusion_scores, target_scores_df, shap_imp)


if __name__ == "__main__":
    main()
