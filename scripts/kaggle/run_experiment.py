from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.create_kernel import REGISTRY, create_kernel_package, ensure_registry, git_commit
from scripts.kaggle.download_outputs import download_outputs
from scripts.kaggle.monitor_kernel import monitor
from scripts.kaggle.push_kernel import push_kernel


def append_registry(
    experiment_name: str,
    kernel: str,
    runtime: float | str = "",
    cv: str = "",
    lb_score: str = "",
    notes: str = "",
) -> None:
    ensure_registry()
    with REGISTRY.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([experiment_name, kernel, git_commit(), runtime, cv, lb_score, notes])


def extract_cv(output_dir: Path) -> str:
    metrics = output_dir / "metrics.json"
    if metrics.exists():
        data = json.loads(metrics.read_text(encoding="utf-8"))
        for key in ["cv_weighted_r2", "best_cv", "cv"]:
            if key in data:
                return str(data[key])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create, push, monitor, and download one Kaggle experiment.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--kernel-slug")
    parser.add_argument("--title")
    parser.add_argument("--internet", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--timeout-minutes", type=int, default=720)
    args = parser.parse_args()

    start = time.time()
    package = create_kernel_package(
        experiment_name=args.experiment_name,
        template=args.template,
        owner=args.owner,
        title=args.title,
        kernel_slug=args.kernel_slug,
        enable_gpu=True,
        enable_internet=args.internet,
        force=args.force,
    )
    kernel = push_kernel(package)
    append_registry(args.experiment_name, kernel, notes=f"submitted package={package}")

    final_status = "submitted"
    if args.monitor:
        final_status = monitor(kernel, args.interval, args.timeout_minutes)

    output_dir = None
    cv = ""
    if args.download:
        output_dir = download_outputs(kernel, args.experiment_name, force=True)
        cv = extract_cv(output_dir)

    runtime = round(time.time() - start, 1)
    append_registry(
        args.experiment_name,
        kernel,
        runtime=runtime,
        cv=cv,
        notes=f"status={final_status}; outputs={output_dir or ''}",
    )


if __name__ == "__main__":
    main()
