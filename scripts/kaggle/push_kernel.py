from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_kernel_id(package_dir: Path) -> str:
    metadata = json.loads((package_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    return metadata["id"]


def push_kernel(package_dir: Path) -> str:
    package_dir = package_dir.resolve()
    if not (package_dir / "kernel-metadata.json").exists():
        raise FileNotFoundError(f"Missing kernel-metadata.json in {package_dir}")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(package_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    output = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0 or "kernel push error" in output:
        raise SystemExit(result.returncode)
    return load_kernel_id(package_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a Kaggle kernel package and start a run.")
    parser.add_argument("--package-dir", required=True, type=Path)
    args = parser.parse_args()
    print(push_kernel(args.package_dir))


if __name__ == "__main__":
    main()
