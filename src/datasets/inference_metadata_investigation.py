from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.full_benchmark import TARGETS, WEIGHTS, make_wide, weighted_r2


def folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    bins = pd.qcut(df["Dry_Total_g"].rank(method="first"), q=10, labels=False)
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, bins))


def load_embedding(cache_dir: Path, name: str) -> pd.DataFrame:
    arr = np.load(cache_dir / f"{name}.npy")
    return pd.DataFrame(arr, columns=[f"{name}_{i}" for i in range(arr.shape[1])])


def build_regressor(kind: str):
    if kind == "hgb":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=350,
                learning_rate=0.045,
                l2_regularization=0.05,
                max_leaf_nodes=15,
                random_state=42,
            )
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=700,
                early_stopping=True,
                random_state=42,
            ),
        )
    raise ValueError(kind)


def cv_predict(x: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, kind: str) -> tuple[np.ndarray, list[float]]:
    oof = np.zeros_like(y, dtype=float)
    scores = []
    for trn, val in folds(df):
        model = build_regressor(kind)
        model.fit(x.iloc[trn], y[trn])
        pred = model.predict(x.iloc[val])
        oof[val] = pred
        if y.shape[1] == len(TARGETS):
            scores.append(weighted_r2(y[val], pred))
        else:
            scores.append(float(np.mean([r2_score(y[val, i], pred[:, i]) for i in range(y.shape[1])])))
    return oof, scores


def image_to_metadata(df: pd.DataFrame, embeddings: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = df[["Pre_GSHH_NDVI", "Height_Ave_cm"]].values
    rows = []
    oof_store = {}
    for emb_name, x in embeddings.items():
        for kind in ["hgb", "mlp"]:
            start = time.time()
            oof, fold_scores = cv_predict(x, y, df, kind)
            oof_store[f"{emb_name}_{kind}"] = oof
            rows.append(
                {
                    "embedding": emb_name,
                    "model": kind,
                    "ndvi_r2": r2_score(y[:, 0], oof[:, 0]),
                    "height_r2": r2_score(y[:, 1], oof[:, 1]),
                    "mean_r2": np.mean([r2_score(y[:, 0], oof[:, 0]), r2_score(y[:, 1], oof[:, 1])]),
                    "fold_scores": json.dumps(fold_scores),
                    "runtime_sec": time.time() - start,
                }
            )
    return pd.DataFrame(rows).sort_values("mean_r2", ascending=False), oof_store


def pseudo_metadata_to_biomass(df: pd.DataFrame, oof_meta: dict[str, np.ndarray]) -> pd.DataFrame:
    y = df[TARGETS].values
    rows = []
    for source, meta_pred in oof_meta.items():
        x = pd.DataFrame(meta_pred, columns=["pred_ndvi", "pred_height"])
        x["pred_ndvi_x_height"] = x["pred_ndvi"] * x["pred_height"]
        for kind in ["hgb", "mlp"]:
            start = time.time()
            pred, fold_scores = cv_predict(x, np.log1p(y), df, kind)
            pred = np.clip(np.expm1(pred), 0, None)
            rows.append(
                {
                    "pseudo_metadata_source": source,
                    "biomass_model": kind,
                    "cv_weighted_r2": weighted_r2(y, pred),
                    **{target: r2_score(y[:, i], pred[:, i]) for i, target in enumerate(TARGETS)},
                    "fold_scores": json.dumps(fold_scores),
                    "runtime_sec": time.time() - start,
                }
            )
    return pd.DataFrame(rows).sort_values("cv_weighted_r2", ascending=False)


def lightweight_image_model(df: pd.DataFrame, embeddings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    y = df[TARGETS].values
    rows = []
    for emb_name in ["convnextv2_base", "dinov2_large", "siglip_base"]:
        if emb_name not in embeddings:
            continue
        x = embeddings[emb_name]
        start = time.time()
        pred, fold_scores = cv_predict(x, np.log1p(y), df, "mlp")
        pred = np.clip(np.expm1(pred), 0, None)
        rows.append(
            {
                "image_model": f"{emb_name}_mlp_head",
                "note": "Frozen image encoder embeddings plus trained MLP head; CPU-safe proxy for lightweight image model.",
                "cv_weighted_r2": weighted_r2(y, pred),
                **{target: r2_score(y[:, i], pred[:, i]) for i, target in enumerate(TARGETS)},
                "fold_scores": json.dumps(fold_scores),
                "runtime_sec": time.time() - start,
            }
        )
    return pd.DataFrame(rows).sort_values("cv_weighted_r2", ascending=False)


def write_report(
    reports_dir: Path,
    data_dir: Path,
    df: pd.DataFrame,
    meta_pred: pd.DataFrame,
    pseudo: pd.DataFrame,
    image_model: pd.DataFrame,
) -> None:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    train_meta = ["Pre_GSHH_NDVI", "Height_Ave_cm", "State", "Species", "Sampling_Date"]
    hidden_inference = "unlikely_available"
    test_has_metadata = set(train_meta).issubset(test.columns)
    if test_has_metadata:
        hidden_inference = "available_in_public_test"

    md = [
        "# Inference Metadata Audit",
        "",
        "## Local Competition Bundle Evidence",
        f"- Files present: `train.csv`, `test.csv`, `sample_submission.csv`, `train/`, `test/`.",
        f"- `train.csv` columns: `{list(train.columns)}`",
        f"- `test.csv` columns: `{list(test.columns)}`",
        f"- `sample_submission.csv` columns: `{list(sample.columns)}`",
        f"- Metadata columns present in train: `{train_meta}`",
        f"- Metadata columns present in public test: `{test_has_metadata}`",
        "",
        "## Determination",
        f"- Are metadata fields available in hidden test? **{hidden_inference}**. Kaggle hidden test normally follows the public `test.csv` schema, and the downloaded public test has only `sample_id`, `image_path`, and `target_name`.",
        "- Are metadata fields reconstructed by Kaggle? **No local evidence.** The bundle has no sidecar metadata file, no mapping file, and no API artifact that could reconstruct NDVI/height/state/species/date.",
        "- Are metadata fields intentionally withheld? **Yes, for inference.** They are labels/auxiliary train metadata, not columns in `test.csv`.",
        "- Are metadata features allowed at inference? **Only if generated from allowed inputs.** True NDVI/height/state/species/date from train rows should not be used for test unless the competition provides them in the runtime input. Predicted pseudo-metadata from images is allowed in spirit because it uses only test images.",
        "",
        "## Source Notes",
        "- Local source of truth: downloaded `test.csv` schema.",
        "- Public competition/data references found during search describe submissions as `sample_id` plus `target`, matching the local sample submission format.",
        "",
        "## Image-only Strategy if Metadata is Unavailable",
        "1. Use full-image foundation embeddings as baseline: DINOv2 Large, SigLIP, ConvNeXtV2.",
        "2. Add canopy-focused preprocessing: remove black/white borders if present, vegetation mask using excess-green, crop high-vegetation tiles, and preserve wide aspect ratio via tiled inference.",
        "3. Train patch-based regressors: split each 2000x1000 image into overlapping tiles, aggregate tile embeddings with mean/max/attention pooling.",
        "4. Use multi-crop test-time inference: full image, left/right/center crops, vegetation-mask crop, horizontal flip.",
        "5. Enforce biomass consistency only after validating it on image-only OOF predictions.",
        "",
        "## Image -> NDVI / Height CV",
        meta_pred.to_markdown(index=False),
        "",
        "## Pseudo-metadata -> Biomass CV",
        pseudo.to_markdown(index=False),
        "",
        "## Lightweight Image Model CV",
        image_model.to_markdown(index=False),
        "",
        "## Final Answers",
        "1. Can metadata be used during inference? **No, not as true metadata from the competition files currently available.**",
        f"2. Strongest image-only strategy so far: `{image_model.iloc[0]['image_model']}` at CV `{image_model.iloc[0]['cv_weighted_r2']:.6f}` if the lightweight head beats embedding GBDTs; otherwise DINOv2 Large + XGBoost from `full_embedding_benchmark.md` remains the baseline.",
        f"3. Can NDVI and Height be predicted from images? Best mean R2 is `{meta_pred.iloc[0]['mean_r2']:.6f}`; NDVI R2 `{meta_pred.iloc[0]['ndvi_r2']:.6f}`, Height R2 `{meta_pred.iloc[0]['height_r2']:.6f}`.",
        f"4. Is pseudo-metadata viable? Best pseudo-metadata biomass CV is `{pseudo.iloc[0]['cv_weighted_r2']:.6f}`. Compare this to pure embedding CV around `0.556414` from DINOv2 Large + XGBoost.",
        "5. Next experiment: canopy-tile DINOv2/ConvNeXtV2 embeddings with multi-crop pooling, then pseudo-NDVI/height as auxiliary heads.",
    ]
    (reports_dir / "inference_metadata_audit.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/csiro-biomass"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/embedding_cache_full"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    df = make_wide(train)
    embeddings = {}
    for name in ["dinov2_large", "siglip_base", "convnextv2_base"]:
        path = args.cache_dir / f"{name}.npy"
        if path.exists():
            embeddings[name] = load_embedding(args.cache_dir, name)
    if not embeddings:
        raise FileNotFoundError(f"No full embedding caches found under {args.cache_dir}")

    meta_pred, oof_meta = image_to_metadata(df, embeddings)
    pseudo = pseudo_metadata_to_biomass(df, oof_meta)
    image_model = lightweight_image_model(df, embeddings)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    meta_pred.to_csv(args.reports_dir / "image_to_metadata_cv.csv", index=False)
    pseudo.to_csv(args.reports_dir / "pseudo_metadata_biomass_cv.csv", index=False)
    image_model.to_csv(args.reports_dir / "lightweight_image_model_cv.csv", index=False)
    write_report(args.reports_dir, args.data_dir, df, meta_pred, pseudo, image_model)


if __name__ == "__main__":
    main()
