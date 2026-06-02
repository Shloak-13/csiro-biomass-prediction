from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.datasets.full_benchmark import TARGETS, WEIGHTS, make_wide, weighted_r2


def folds(df: pd.DataFrame, n_splits: int = 5):
    bins = pd.qcut(df["Dry_Total_g"].rank(method="first"), q=10, labels=False)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(df, bins))


def load_feature_sets(root: Path, wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feature_sets: dict[str, pd.DataFrame] = {}
    image_features = pd.read_csv(root / "reports" / "image_features.csv")
    image_features["image_file"] = image_features["image_path"]
    key = wide["image_path"].map(lambda p: Path(str(p)).name)
    merged = wide[["image_path"]].assign(image_file=key).merge(image_features, on="image_file", how="left")
    handcrafted_cols = [
        c
        for c in merged.columns
        if c.startswith(("brightness", "saturation", "green_", "vegetation_", "exg_", "texture_", "gray_", "hist_"))
        or c in ["width", "height", "aspect_ratio"]
    ]
    feature_sets["handcrafted_image"] = merged[handcrafted_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    cache = root / "data" / "embedding_cache_full"
    for path in cache.glob("*.npy"):
        mat = np.load(path)
        feature_sets[path.stem] = pd.DataFrame(mat, columns=[f"{path.stem}_{i}" for i in range(mat.shape[1])])
    if {"dinov2_large", "siglip_base", "convnextv2_base"}.issubset(feature_sets):
        feature_sets["all_embeddings"] = pd.concat(
            [feature_sets["dinov2_large"], feature_sets["siglip_base"], feature_sets["convnextv2_base"]],
            axis=1,
        )
    return feature_sets


def build_meta_regressor(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "extratrees":
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
    from lightgbm import LGBMRegressor

    return MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=260,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
        )
    )


def cv_predict_metadata(df: pd.DataFrame, x: pd.DataFrame, model_name: str) -> tuple[np.ndarray, dict]:
    y = df[["Pre_GSHH_NDVI", "Height_Ave_cm"]].values
    oof = np.zeros_like(y, dtype=float)
    start = time.time()
    for trn, val in folds(df):
        model = build_meta_regressor(model_name)
        x_clean = x.replace([np.inf, -np.inf], np.nan).fillna(0)
        model.fit(x_clean.iloc[trn], y[trn])
        oof[val] = model.predict(x_clean.iloc[val])
    return oof, {
        "model": model_name,
        "ndvi_r2": float(r2_score(y[:, 0], oof[:, 0])),
        "height_r2": float(r2_score(y[:, 1], oof[:, 1])),
        "mean_r2": float((r2_score(y[:, 0], oof[:, 0]) + r2_score(y[:, 1], oof[:, 1])) / 2),
        "runtime_sec": time.time() - start,
    }


def pseudo_metadata_biomass(df: pd.DataFrame, pseudo_meta: np.ndarray, extra_x: pd.DataFrame | None = None) -> dict:
    from lightgbm import LGBMRegressor

    base = pd.DataFrame(pseudo_meta, columns=["pred_ndvi", "pred_height"])
    if extra_x is not None:
        x = pd.concat([base, extra_x.reset_index(drop=True)], axis=1)
    else:
        x = base
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[TARGETS].values
    oof = np.zeros_like(y)
    for trn, val in folds(df):
        model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=280,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
            )
        )
        model.fit(x.iloc[trn], np.log1p(y[trn]))
        oof[val] = np.clip(np.expm1(model.predict(x.iloc[val])), 0, None)
    return {
        "cv_weighted_r2": weighted_r2(y, oof),
        **{target: float(r2_score(y[:, i], oof[:, i])) for i, target in enumerate(TARGETS)},
    }


def train_lightweight_efficientnet(root: Path, df: pd.DataFrame, out_dir: Path) -> dict:
    import torch
    import timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class BiomassDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, data_dir: Path, tfm):
            self.frame = frame.reset_index(drop=True)
            self.data_dir = data_dir
            self.tfm = tfm

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            image = Image.open(self.data_dir / row["image_path"]).convert("RGB")
            return self.tfm(image), torch.tensor(np.log1p(row[TARGETS].values.astype("float32")))

    data_dir = root / "data" / "csiro-biomass"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    y = df[TARGETS].values
    oof = np.zeros_like(y)
    weights = torch.tensor(WEIGHTS.astype("float32")).to(device)
    splits = folds(df, n_splits=3)
    start = time.time()
    for fold, (trn, val) in enumerate(splits):
        model = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        train_loader = DataLoader(BiomassDataset(df.iloc[trn], data_dir, tfm), batch_size=8, shuffle=True)
        val_loader = DataLoader(BiomassDataset(df.iloc[val], data_dir, tfm), batch_size=8, shuffle=False)
        model.train()
        for _ in range(2):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = (((pred - yb) ** 2) * weights).mean()
                loss.backward()
                opt.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                pred = model(xb.to(device)).cpu().numpy()
                preds.append(pred)
        oof[val] = np.clip(np.expm1(np.concatenate(preds, axis=0)), 0, None)
        del model
    return {
        "model": "tf_efficientnetv2_b0.in1k",
        "folds": 3,
        "epochs": 2,
        "image_size": 160,
        "device": device,
        "runtime_sec": time.time() - start,
        "cv_weighted_r2": weighted_r2(y, oof),
        **{target: float(r2_score(y[:, i], oof[:, i])) for i, target in enumerate(TARGETS)},
    }


def write_audit(root: Path, reports_dir: Path, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    lines = [
        "# Inference Metadata Audit",
        "",
        "## Local Competition Files",
        f"- `train.csv` columns: `{list(train.columns)}`",
        f"- `test.csv` columns: `{list(test.columns)}`",
        f"- `sample_submission.csv` columns: `{list(sample.columns)}`",
        f"- Local test images: `{test['image_path'].nunique()}` image, `{len(test)}` target rows.",
        "",
        "## Conclusion",
        "Metadata is not available in the provided inference schema. The public `test.csv` contains only `sample_id`, `image_path`, and `target_name`; no `Pre_GSHH_NDVI`, `Height_Ave_cm`, `State`, `Species`, or `Sampling_Date` columns are present.",
        "",
        "Kaggle hidden test rows normally follow the same schema as public `test.csv` and `sample_submission.csv`. There is no local file, sidecar metadata table, or sample submission field that would reconstruct metadata at scoring time.",
        "",
        "## Answers",
        "1. Are metadata fields available in hidden test? Unknown directly, but the evidence strongly indicates no: hidden test should match `test.csv` schema.",
        "2. Are metadata fields reconstructed by Kaggle? No evidence. The submission interface only asks for `sample_id,target`.",
        "3. Are metadata fields intentionally withheld? Yes, operationally: train has metadata, test does not.",
        "4. Are metadata features allowed at inference? Only if derived from allowed inputs. Raw train metadata columns cannot be used for test rows because they are absent.",
        "",
        "## Strategic Implication",
        "Treat metadata-only CV as an optimistic upper bound or teacher signal, not a directly submit-ready model. Submission-ready models must use images and `target_name`, or pseudo-metadata predicted from images.",
    ]
    (reports_dir / "inference_metadata_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()
    root = args.root
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    train = pd.read_csv(root / "data" / "csiro-biomass" / "train.csv")
    test = pd.read_csv(root / "data" / "csiro-biomass" / "test.csv")
    sample = pd.read_csv(root / "data" / "csiro-biomass" / "sample_submission.csv")
    wide = make_wide(train)
    write_audit(root, reports, train, test, sample)

    feature_sets = load_feature_sets(root, wide)
    meta_rows = []
    oof_store = {}
    for feature_name, x in feature_sets.items():
        for model_name in ["ridge", "lightgbm", "extratrees"]:
            oof, row = cv_predict_metadata(wide, x, model_name)
            row["feature_set"] = feature_name
            meta_rows.append(row)
            oof_store[(feature_name, model_name)] = oof
    meta_scores = pd.DataFrame(meta_rows).sort_values("mean_r2", ascending=False)
    meta_scores.to_csv(reports / "image_to_metadata_scores.csv", index=False)

    best = meta_scores.iloc[0]
    best_key = (best["feature_set"], best["model"])
    pseudo = oof_store[best_key]
    pseudo_rows = []
    pseudo_rows.append({"feature_set": "pseudo_ndvi_height_only", **pseudo_metadata_biomass(wide, pseudo)})
    for emb in ["dinov2_large", "siglip_base", "convnextv2_base", "all_embeddings"]:
        if emb in feature_sets:
            pseudo_rows.append({"feature_set": f"pseudo_ndvi_height_plus_{emb}", **pseudo_metadata_biomass(wide, pseudo, feature_sets[emb])})
    pseudo_scores = pd.DataFrame(pseudo_rows).sort_values("cv_weighted_r2", ascending=False)
    pseudo_scores.to_csv(reports / "pseudo_metadata_scores.csv", index=False)

    image_strategy = [
        "# Image-Only Strategy Report",
        "",
        "## Recommended Image-Only Stack",
        "1. Preprocess each 2000x1000 quadrat with vegetation segmentation using excess-green and saturation thresholds.",
        "2. Generate canopy-focused crops: full image, center crop, left/right halves, and top vegetation-density patches.",
        "3. Extract frozen embeddings per crop using DINOv2 Large, SigLIP, and ConvNeXtV2.",
        "4. Add image-derived pseudo-metadata: predicted NDVI and predicted height.",
        "5. Train per-target GBDTs and a small CNN/ConvNeXt/EfficientNet model; ensemble OOF predictions.",
        "6. Use multi-crop test-time averaging and enforce non-negativity plus validation-tested biomass consistency.",
        "",
        "## Why",
        "Raw full-image embeddings underperformed metadata, but embeddings-only models beat hand-crafted image features. The likely missing piece is crop quality: the images are wide 2:1 quadrats with substantial background/frame variation.",
    ]
    (reports / "image_only_strategy.md").write_text("\n".join(image_strategy), encoding="utf-8")

    e2e_result = None
    if not args.skip_e2e:
        try:
            e2e_result = train_lightweight_efficientnet(root, wide, reports)
        except Exception as exc:
            e2e_result = {"status": f"failed: {repr(exc)}"}
    else:
        e2e_result = {"status": "skipped"}
    pd.Series(e2e_result).to_json(reports / "lightweight_e2e_result.json", indent=2)

    best_embedding = pd.read_csv(reports / "full_embedding_cv_scores.csv").sort_values("cv_weighted_r2", ascending=False).iloc[0]
    lines = [
        "# Metadata Inference Decision",
        "",
        "## Image to Metadata",
        meta_scores.head(20).to_markdown(index=False),
        "",
        "## Pseudo-Metadata to Biomass",
        pseudo_scores.to_markdown(index=False),
        "",
        "## Lightweight End-to-End Result",
        pd.Series(e2e_result).to_frame("value").to_markdown(),
        "",
        "## Final Answers",
        "1. Metadata cannot be used directly at inference under the provided schema.",
        f"2. Strongest current image-only strategy: full-image DINOv2 Large + XGBoost embedding baseline at `{best_embedding['cv_weighted_r2']:.6f}`, then improve it with canopy crops and multi-crop fusion.",
        f"3. NDVI/Height can be predicted from images with best mean R2 `{best['mean_r2']:.6f}` (`{best['feature_set']}` + `{best['model']}`), but inspect per-target R2 before trusting pseudo-metadata.",
        f"4. Pseudo-metadata viability: best pseudo-metadata biomass CV `{pseudo_scores.iloc[0]['cv_weighted_r2']:.6f}`. Compare this against pure embedding CV before using it.",
        "5. Next experiment: canopy-cropped DINOv2/SigLIP/ConvNeXtV2 embeddings plus pseudo-NDVI/height, blended with the lightweight CNN OOF if it is positive.",
    ]
    (reports / "metadata_inference_decision.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
