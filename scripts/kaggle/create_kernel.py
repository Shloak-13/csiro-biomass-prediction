from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "notebooks" / "kaggle_templates"
PACKAGE_ROOT = ROOT / "kaggle_kernels"
REGISTRY = ROOT / "experiments" / "registry.csv"


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


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "NO_GIT_REPO"


def ensure_registry() -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY.exists():
        REGISTRY.write_text(
            "experiment_name,kaggle_kernel,git_commit,runtime,CV,LB score,notes\n",
            encoding="utf-8",
        )


def create_kernel_package(
    experiment_name: str,
    template: str,
    owner: str,
    title: str | None,
    kernel_slug: str | None,
    enable_gpu: bool,
    enable_internet: bool,
    force: bool,
) -> Path:
    template_path = TEMPLATE_DIR / template
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    kernel_slug = slugify(kernel_slug or experiment_name)
    title = title or experiment_name.replace("_", " ").replace("-", " ").title()
    kernel_id = f"{owner}/{kernel_slug}"
    package_dir = PACKAGE_ROOT / kernel_slug

    if package_dir.exists():
        if not force:
            raise FileExistsError(f"{package_dir} already exists. Use --force to overwrite.")
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    code_file = template_path.name
    shutil.copy2(template_path, package_dir / code_file)

    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "competition_sources": ["csiro-biomass"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (package_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    config = {
        "experiment_name": experiment_name,
        "kernel": kernel_id,
        "template": template,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "expected_outputs": ["submission.csv", "oof_predictions.csv", "training_log.csv", "metrics.json"],
    }
    (package_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    ensure_registry()
    return package_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Kaggle kernel package.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--template", required=True, help="Template notebook filename under notebooks/kaggle_templates.")
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"))
    parser.add_argument("--kernel-slug")
    parser.add_argument("--title")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--internet", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.owner:
        raise SystemExit("Provide --owner or set KAGGLE_USERNAME.")

    package = create_kernel_package(
        experiment_name=args.experiment_name,
        template=args.template,
        owner=args.owner,
        title=args.title,
        kernel_slug=args.kernel_slug,
        enable_gpu=not args.no_gpu,
        enable_internet=args.internet,
        force=args.force,
    )
    print(package)


if __name__ == "__main__":
    main()
