from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def download_competition(out_dir: Path, competition: str = "csiro-biomass") -> Path:
    import kagglehub

    cache_path = Path(kagglehub.competition_download(competition))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / competition
    if target.exists():
        return target
    if cache_path.is_dir():
        shutil.copytree(cache_path, target)
    else:
        shutil.copy2(cache_path, out_dir / cache_path.name)
        target = out_dir / cache_path.name
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--competition", default="csiro-biomass")
    args = parser.parse_args()
    path = download_competition(args.out_dir, args.competition)
    print(path)


if __name__ == "__main__":
    main()
