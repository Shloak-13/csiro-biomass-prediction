from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.constants import TARGET_COLS
from src.utils.metrics import weighted_log_r2


def load_prediction(path: Path, targets: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    return df[targets].values


def optimize_weights(y_true: np.ndarray, preds: list[np.ndarray]) -> np.ndarray:
    n = len(preds)

    def objective(w: np.ndarray) -> float:
        w = np.clip(w, 0, None)
        w = w / max(w.sum(), 1e-12)
        blended = sum(weight * pred for weight, pred in zip(w, preds, strict=False))
        return -weighted_log_r2(y_true, blended)

    result = minimize(
        objective,
        x0=np.ones(n) / n,
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        method="SLSQP",
    )
    weights = np.clip(result.x, 0, None)
    return weights / max(weights.sum(), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--preds", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/ensemble_weights.csv"))
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    targets = [c for c in TARGET_COLS if c in truth.columns]
    y_true = truth[targets].values
    preds = [load_prediction(path, targets) for path in args.preds]
    weights = optimize_weights(y_true, preds)
    result = pd.DataFrame({"prediction_file": [str(p) for p in args.preds], "weight": weights})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
