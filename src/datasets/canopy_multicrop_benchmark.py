from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.datasets.full_benchmark import TARGETS, make_wide, weighted_r2


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    model_id: str
    batch_size: int


MODELS = [
    ModelSpec("dinov2_large", "transformers", "facebook/dinov2-large", 2),
    ModelSpec("siglip_base", "transformers", "google/siglip-base-patch16-224", 2),
    ModelSpec("convnextv2_base", "timm", "convnextv2_base.fcmae_ft_in22k_in1k", 2),
]
VARIANTS = ["original", "canopy", "center", "highveg"]


def folds(df: pd.DataFrame, n_splits: int = 5):
    bins = pd.qcut(df["Dry_Total_g"].rank(method="first"), q=10, labels=False)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(df, bins))


def exg_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = arr.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b
    # Adaptive but conservative: keep pixels clearly greener than local background.
    threshold = max(0.03, float(np.percentile(exg, 62)))
    mask = (exg > threshold) & (g > 0.16) & (g > r * 1.03) & (g > b * 1.03)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    return exg, mask


def bbox_from_mask(mask: np.ndarray, pad_frac: float = 0.06) -> tuple[int, int, int, int]:
    labels, n = ndimage.label(mask)
    h, w = mask.shape
    if n == 0:
        return 0, 0, w, h
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    label = int(np.argmax(sizes) + 1)
    ys, xs = np.where(labels == label)
    if len(xs) == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    pad = int(max(x2 - x1, y2 - y1) * pad_frac)
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def square_center_crop_box(w: int, h: int) -> tuple[int, int, int, int]:
    side = min(w, h)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return x1, y1, x1 + side, y1 + side


def high_veg_crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    side = min(h, w)
    step = max(32, side // 6)
    integral = mask.astype(np.float32)
    best = (-1.0, 0, 0)
    for y in range(0, h - side + 1, step):
        for x in range(0, w - side + 1, step):
            density = float(integral[y : y + side, x : x + side].mean())
            if density > best[0]:
                best = (density, x, y)
    _, x, y = best
    return x, y, x + side, y + side


def prepare_crops(data_dir: Path, wide: pd.DataFrame, crop_dir: Path, stats_path: Path) -> pd.DataFrame:
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ["canopy", "center", "highveg"]:
        (crop_dir / variant).mkdir(parents=True, exist_ok=True)
    for rel in tqdm(wide["image_path"], desc="Canopy crops"):
        src = data_dir / rel
        name = Path(rel).name
        img = Image.open(src).convert("RGB")
        arr = np.asarray(img)
        _, mask = exg_mask(arr)
        veg_ratio = float(mask.mean())
        boxes = {
            "canopy": bbox_from_mask(mask),
            "center": square_center_crop_box(img.width, img.height),
            "highveg": high_veg_crop_box(mask),
        }
        row = {"image_path": rel, "image_file": name, "vegetation_ratio": veg_ratio}
        for variant, box in boxes.items():
            out = crop_dir / variant / name
            crop = img.crop(box)
            crop.save(out, quality=95)
            x1, y1, x2, y2 = box
            row[f"{variant}_x1"] = x1
            row[f"{variant}_y1"] = y1
            row[f"{variant}_x2"] = x2
            row[f"{variant}_y2"] = y2
            row[f"{variant}_width"] = x2 - x1
            row[f"{variant}_height"] = y2 - y1
            row[f"{variant}_area_frac"] = ((x2 - x1) * (y2 - y1)) / (img.width * img.height)
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats.to_csv(stats_path, index=False)
    return stats


def variant_paths(data_dir: Path, crop_dir: Path, wide: pd.DataFrame, variant: str) -> list[Path]:
    if variant == "original":
        return [data_dir / p for p in wide["image_path"].tolist()]
    return [crop_dir / variant / Path(p).name for p in wide["image_path"].tolist()]


def extract_transformers(spec: ModelSpec, paths: list[Path], out: Path) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(spec.model_id, use_fast=True)
    model = AutoModel.from_pretrained(spec.model_id).to(device).eval()
    vectors = []
    for i in tqdm(range(0, len(paths), spec.batch_size), desc=out.stem):
        images = [Image.open(p).convert("RGB") for p in paths[i : i + spec.batch_size]]
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
    mat = np.concatenate(vectors).astype("float32")
    np.save(out, mat)
    del model
    gc.collect()
    return {"dim": mat.shape[1], "runtime_sec": time.time() - start}


def extract_timm(spec: ModelSpec, paths: list[Path], out: Path) -> dict:
    import timm
    import torch
    from torchvision import transforms

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(spec.model_id, pretrained=True, num_classes=0).to(device).eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    vectors = []
    for i in tqdm(range(0, len(paths), spec.batch_size), desc=out.stem):
        batch = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[i : i + spec.batch_size]]).to(device)
        with torch.no_grad():
            emb = model(batch)
            if emb.ndim > 2:
                emb = emb.mean(dim=tuple(range(2, emb.ndim)))
        vectors.append(emb.float().cpu().numpy())
    mat = np.concatenate(vectors).astype("float32")
    np.save(out, mat)
    del model
    gc.collect()
    return {"dim": mat.shape[1], "runtime_sec": time.time() - start}


def ensure_embeddings(data_dir: Path, crop_dir: Path, wide: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    full_cache = data_dir.parent / "embedding_cache_full"
    for spec in MODELS:
        for variant in VARIANTS:
            out = cache_dir / f"{spec.name}_{variant}.npy"
            if variant == "original" and not out.exists():
                src = full_cache / f"{spec.name}.npy"
                if src.exists():
                    arr = np.load(src)
                    np.save(out, arr)
            if out.exists():
                arr = np.load(out, mmap_mode="r")
                rows.append({"model": spec.name, "variant": variant, "dim": arr.shape[1], "runtime_sec": 0.0, "status": "cached"})
                continue
            paths = variant_paths(data_dir, crop_dir, wide, variant)
            try:
                result = extract_transformers(spec, paths, out) if spec.backend == "transformers" else extract_timm(spec, paths, out)
                rows.append({"model": spec.name, "variant": variant, "status": "ok", **result})
            except Exception as exc:
                rows.append({"model": spec.name, "variant": variant, "dim": np.nan, "runtime_sec": np.nan, "status": f"failed: {repr(exc)}"})
    return pd.DataFrame(rows)


def load_embedding(cache_dir: Path, model: str, variant: str) -> pd.DataFrame:
    arr = np.load(cache_dir / f"{model}_{variant}.npy")
    return pd.DataFrame(arr, columns=[f"{model}_{variant}_{i}" for i in range(arr.shape[1])])


def feature_sets(cache_dir: Path) -> dict[str, pd.DataFrame]:
    raw = {}
    for model in [m.name for m in MODELS]:
        available = []
        for variant in VARIANTS:
            path = cache_dir / f"{model}_{variant}.npy"
            if path.exists():
                raw[f"{model}_{variant}"] = load_embedding(cache_dir, model, variant)
                available.append(raw[f"{model}_{variant}"])
        if available:
            raw[f"{model}_all_crops"] = pd.concat(available, axis=1)
    for variant in VARIANTS:
        parts = [raw[f"{m.name}_{variant}"] for m in MODELS if f"{m.name}_{variant}" in raw]
        if parts:
            raw[f"all_models_{variant}"] = pd.concat(parts, axis=1)
    all_parts = [raw[f"{m.name}_all_crops"] for m in MODELS if f"{m.name}_all_crops" in raw]
    if all_parts:
        raw["all_models_all_crops"] = pd.concat(all_parts, axis=1)

    keep = [
        "all_models_original",
        "all_models_canopy",
        "all_models_center",
        "all_models_highveg",
        "dinov2_large_all_crops",
        "siglip_base_all_crops",
        "convnextv2_base_all_crops",
        "all_models_all_crops",
    ]
    sets = {}
    for name in keep:
        if name not in raw:
            continue
        frame = raw[name]
        if frame.shape[1] > 512:
            n_components = min(256, frame.shape[0] - 1, frame.shape[1])
            x = StandardScaler().fit_transform(frame)
            reduced = PCA(n_components=n_components, random_state=42).fit_transform(x)
            sets[f"{name}_pca{n_components}"] = pd.DataFrame(
                reduced, columns=[f"{name}_pca_{i}" for i in range(n_components)]
            )
        else:
            sets[name] = frame
    return sets


def cv_predict_meta(df: pd.DataFrame, x: pd.DataFrame) -> tuple[np.ndarray, dict]:
    y = df[["Pre_GSHH_NDVI", "Height_Ave_cm"]].values
    oof = np.zeros_like(y)
    for trn, val in folds(df):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x.iloc[trn], y[trn])
        oof[val] = model.predict(x.iloc[val])
    return oof, {
        "ndvi_r2": r2_score(y[:, 0], oof[:, 0]),
        "height_r2": r2_score(y[:, 1], oof[:, 1]),
        "mean_r2": (r2_score(y[:, 0], oof[:, 0]) + r2_score(y[:, 1], oof[:, 1])) / 2,
    }


def build_regressor(name: str, n_outputs: int):
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
                random_state=42,
                verbose=-1,
            )
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
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
        )
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MultiRMSE",
        iterations=260,
        learning_rate=0.045,
        depth=5,
        l2_leaf_reg=8,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def cv_biomass(df: pd.DataFrame, x: pd.DataFrame, model_name: str, multitask: bool = False) -> dict:
    biomass = df[TARGETS].values
    if multitask:
        y = df[["Pre_GSHH_NDVI", "Height_Ave_cm", *TARGETS]].values
    else:
        y = biomass
    oof_biomass = np.zeros_like(biomass)
    for trn, val in folds(df):
        model = build_regressor(model_name, y.shape[1])
        model.fit(x.iloc[trn], np.log1p(np.clip(y[trn], 0, None)))
        pred = np.clip(np.expm1(model.predict(x.iloc[val])), 0, None)
        oof_biomass[val] = pred[:, -5:] if multitask else pred
    return {
        "model": model_name,
        "cv_weighted_r2": weighted_r2(biomass, oof_biomass),
        **{t: r2_score(biomass[:, i], oof_biomass[:, i]) for i, t in enumerate(TARGETS)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    data_dir = root / "data" / "csiro-biomass"
    crop_dir = root / "data" / "canopy_crops"
    cache_dir = root / "data" / "embedding_cache_multicrop"
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    train = pd.read_csv(data_dir / "train.csv")
    wide = make_wide(train)

    stats_path = reports / "canopy_crop_stats.csv"
    crop_stats = prepare_crops(data_dir, wide, crop_dir, stats_path)
    emb_meta = ensure_embeddings(data_dir, crop_dir, wide, cache_dir)
    emb_meta.to_csv(reports / "multicrop_embedding_extraction.csv", index=False)
    sets = feature_sets(cache_dir)

    meta_rows = []
    pseudo_store = {}
    for name, x in sets.items():
        pred_meta, scores = cv_predict_meta(wide, x)
        meta_rows.append({"feature_set": name, "dim": x.shape[1], **scores})
        pseudo_store[name] = pred_meta
    meta_scores = pd.DataFrame(meta_rows).sort_values("mean_r2", ascending=False)
    meta_scores.to_csv(reports / "multicrop_image_to_metadata_scores.csv", index=False)

    rows = []
    for name, x in sets.items():
        for model in ["xgboost", "lightgbm", "catboost"]:
            rows.append({"feature_set": name, "dim": x.shape[1], "task": "biomass_only", **cv_biomass(wide, x, model)})
            rows.append({"feature_set": name, "dim": x.shape[1], "task": "joint_meta_biomass", **cv_biomass(wide, x, model, multitask=True)})
    biomass_scores = pd.DataFrame(rows).sort_values("cv_weighted_r2", ascending=False)
    biomass_scores.to_csv(reports / "multicrop_biomass_scores.csv", index=False)

    fusion_rows = []
    best_meta_name = meta_scores.iloc[0]["feature_set"]
    for name, x in sets.items():
        pseudo = pd.DataFrame(pseudo_store[best_meta_name], columns=["pred_ndvi", "pred_height"])
        fused = pd.concat([x.reset_index(drop=True), pseudo], axis=1)
        for model in ["xgboost", "lightgbm", "catboost"]:
            fusion_rows.append({"feature_set": name, "metadata_source": best_meta_name, "dim": fused.shape[1], **cv_biomass(wide, fused, model)})
    fusion_scores = pd.DataFrame(fusion_rows).sort_values("cv_weighted_r2", ascending=False)
    fusion_scores.to_csv(reports / "multicrop_fusion_scores.csv", index=False)

    canopy_md = [
        "# Canopy Cropping",
        "",
        "## Crop Statistics",
        crop_stats.describe().T.to_markdown(),
        "",
        "## Extraction",
        emb_meta.to_markdown(index=False),
    ]
    (reports / "canopy_cropping.md").write_text("\n".join(canopy_md), encoding="utf-8")

    summary = [
        "# Multi-Crop Representation Benchmark",
        "",
        "## Image -> Metadata",
        meta_scores.head(20).to_markdown(index=False),
        "",
        "## Multi-Task Embedding Models",
        biomass_scores.head(30).to_markdown(index=False),
        "",
        "## Fusion With Predicted NDVI/Height",
        fusion_scores.head(30).to_markdown(index=False),
        "",
        "## Success Criteria",
        f"- Beat baseline `0.596703`: {'yes' if fusion_scores.iloc[0]['cv_weighted_r2'] > 0.596703 else 'no'}",
        f"- Reach `0.63+`: {'yes' if fusion_scores.iloc[0]['cv_weighted_r2'] >= 0.63 else 'no'}",
        f"- Best CV: `{fusion_scores.iloc[0]['cv_weighted_r2']:.6f}`",
    ]
    (reports / "multicrop_representation_benchmark.md").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
