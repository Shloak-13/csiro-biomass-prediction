from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "csiro-biomass"


def slugify(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        elif char in {" ", "-", "_"}:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug


def write_metadata(source: Path, owner: str, slug: str, title: str) -> Path:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset source not found: {source}")

    metadata = {
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "unknown"}],
    }
    path = source / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def create_dataset(source: Path, owner: str, slug: str, title: str, public: bool, dir_mode: str) -> str:
    write_metadata(source, owner, slug, title)
    cmd = ["kaggle", "datasets", "create", "-p", str(source.resolve()), "--dir-mode", dir_mode]
    if public:
        cmd.append("--public")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return f"{owner}/{slug}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a user-owned Kaggle dataset for the CSIRO Biomass files.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"))
    parser.add_argument("--slug", default="csiro-biomass")
    parser.add_argument("--title", default="CSIRO Biomass Local Dataset")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dir-mode", choices=["skip", "zip", "tar"], default="zip")
    args = parser.parse_args()

    if not args.owner:
        raise SystemExit("Provide --owner or set KAGGLE_USERNAME.")

    slug = slugify(args.slug)
    dataset = create_dataset(args.source, args.owner, slug, args.title, args.public, args.dir_mode)
    print(f"dataset={dataset}")


if __name__ == "__main__":
    main()
