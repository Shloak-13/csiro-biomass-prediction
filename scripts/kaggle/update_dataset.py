from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "csiro-biomass"


def write_metadata(source: Path, dataset: str, title: str) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")

    metadata = {
        "title": title,
        "id": dataset,
        "licenses": [{"name": "unknown"}],
    }
    path = source / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def update_dataset(source: Path, dataset: str, title: str, message: str, dir_mode: str) -> str:
    write_metadata(source, dataset, title)
    cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(source.resolve()),
        "--dir-mode",
        dir_mode,
        "-m",
        message,
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a new version of the user-owned CSIRO Biomass Kaggle dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", default=os.environ.get("CSIRO_KAGGLE_DATASET"))
    parser.add_argument("--title", default="CSIRO Biomass Local Dataset")
    parser.add_argument("--message", default=f"Dataset refresh {datetime.now(timezone.utc).isoformat()}")
    parser.add_argument("--dir-mode", choices=["skip", "zip", "tar"], default="zip")
    args = parser.parse_args()

    if not args.dataset:
        raise SystemExit("Provide --dataset or set CSIRO_KAGGLE_DATASET.")

    dataset = update_dataset(args.source, args.dataset, args.title, args.message, args.dir_mode)
    print(f"dataset={dataset}")


if __name__ == "__main__":
    main()
