from __future__ import annotations

import argparse
import json
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from src.utils.constants import METADATA_COLS, TARGET_COLS
from src.utils.io import list_images, normalize_train_frame, read_csv


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def profile_images(image_paths: list[Path], max_images: int) -> pd.DataFrame:
    rows = []
    for path in tqdm(image_paths[:max_images], desc="Hashing images"):
        try:
            with Image.open(path) as img:
                rows.append(
                    {
                        "path": str(path),
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "phash": str(imagehash.phash(img)),
                        "ahash": str(imagehash.average_hash(img)),
                    }
                )
        except Exception as exc:
            rows.append({"path": str(path), "error": repr(exc)})
    return pd.DataFrame(rows)


def target_constraint_report(df: pd.DataFrame) -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {
        "has_all_component_columns": set(TARGET_COLS).issubset(df.columns)
    }
    if not result["has_all_component_columns"]:
        return result
    component_sum = df["Dry_Green_g"] + df["Dry_Dead_g"] + df["Dry_Clover_g"]
    diff = df["Dry_Total_g"] - component_sum
    result.update(
        {
            "n": int(diff.notna().sum()),
            "mean_diff_total_minus_components": float(diff.mean()),
            "median_abs_diff": float(diff.abs().median()),
            "max_abs_diff": float(diff.abs().max()),
            "pct_exact_within_1e_6": float((diff.abs() <= 1e-6).mean()),
            "pct_close_within_1g": float((diff.abs() <= 1.0).mean()),
        }
    )
    if "GDM_g" in df:
        gdm_expected = df["Dry_Green_g"] + df["Dry_Clover_g"]
        gdm_diff = df["GDM_g"] - gdm_expected
        result.update(
            {
                "gdm_median_abs_diff": float(gdm_diff.abs().median()),
                "gdm_pct_close_within_1g": float((gdm_diff.abs() <= 1.0).mean()),
            }
        )
    return result


def run_eda(data_dir: Path, report_dir: Path, max_hash_images: int = 5000) -> None:
    train_raw = read_csv(data_dir, "train.csv")
    test = read_csv(data_dir, "test.csv")
    sample_submission = read_csv(data_dir, "sample_submission.csv")
    train = normalize_train_frame(train_raw)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"

    summary = {
        "train_raw_shape": train_raw.shape,
        "train_wide_shape": train.shape,
        "test_shape": test.shape,
        "sample_submission_shape": sample_submission.shape,
        "train_columns": list(train_raw.columns),
        "test_columns": list(test.columns),
        "missing_train": train.isna().sum().to_dict(),
        "missing_test": test.isna().sum().to_dict(),
        "target_constraint": target_constraint_report(train),
    }

    numeric = train.select_dtypes("number")
    if not numeric.empty:
        numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(
            report_dir / "numeric_summary.csv"
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric.corr(), cmap="vlag", center=0)
        save_plot(figure_dir / "correlation_matrix.png")

    available_targets = [c for c in TARGET_COLS if c in train.columns]
    if available_targets:
        train[available_targets].describe().T.to_csv(report_dir / "target_summary.csv")
        fig, axes = plt.subplots(len(available_targets), 1, figsize=(9, 3 * len(available_targets)))
        if len(available_targets) == 1:
            axes = [axes]
        for ax, col in zip(axes, available_targets, strict=False):
            sns.histplot(train[col], kde=True, ax=ax)
            ax.set_title(col)
        save_plot(figure_dir / "target_distributions.png")

        sns.pairplot(train[available_targets].dropna(), corner=True)
        save_plot(figure_dir / "target_pairplot.png")

    for col in METADATA_COLS:
        if col not in train.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            plt.figure(figsize=(8, 4))
            sns.histplot(train[col], kde=True)
            save_plot(figure_dir / f"{col.lower()}_distribution.png")
        else:
            counts = train[col].astype(str).value_counts().head(30)
            counts.to_csv(report_dir / f"{col.lower()}_counts.csv")
            plt.figure(figsize=(10, 5))
            sns.barplot(x=counts.values, y=counts.index)
            save_plot(figure_dir / f"{col.lower()}_counts.png")

    if "Sampling_Date" in train.columns:
        dates = pd.to_datetime(train["Sampling_Date"], errors="coerce")
        date_frame = pd.DataFrame({"date": dates})
        date_frame["month"] = dates.dt.month
        date_frame["dayofyear"] = dates.dt.dayofyear
        date_frame.to_csv(report_dir / "date_features_profile.csv", index=False)
        plt.figure(figsize=(8, 4))
        sns.countplot(x=date_frame["month"])
        save_plot(figure_dir / "sampling_month_distribution.png")

    image_df = profile_images(list_images(data_dir), max_hash_images)
    image_df.to_csv(report_dir / "image_profile.csv", index=False)
    if "phash" in image_df:
        exact_dupes = image_df[image_df.duplicated("phash", keep=False)].sort_values("phash")
        exact_dupes.to_csv(report_dir / "duplicate_image_hashes.csv", index=False)
        summary["n_profiled_images"] = int(len(image_df))
        summary["n_duplicate_phash_rows"] = int(len(exact_dupes))

    with (report_dir / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--max-hash-images", type=int, default=5000)
    args = parser.parse_args()
    run_eda(args.data_dir, args.report_dir, args.max_hash_images)


if __name__ == "__main__":
    main()
