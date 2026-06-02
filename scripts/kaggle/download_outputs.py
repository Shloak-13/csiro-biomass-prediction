from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "kaggle_outputs"


def download_outputs(kernel: str, experiment_name: str | None, force: bool) -> Path:
    slug = experiment_name or kernel.replace("/", "__")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / slug / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", kernel, "-p", str(out_dir)]
    if force:
        cmd.append("-o")
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download latest Kaggle kernel outputs.")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_outputs(args.kernel, args.experiment_name, args.force)


if __name__ == "__main__":
    main()
