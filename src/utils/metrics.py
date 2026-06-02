from __future__ import annotations

import numpy as np

from src.utils.constants import TARGET_COLS, TARGET_WEIGHTS


def weighted_log_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.log1p(np.clip(y_true, 0, None))
    y_pred = np.log1p(np.clip(y_pred, 0, None))
    scores = []
    for idx, target in enumerate(TARGET_COLS):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = 0.0 if denom == 0 else 1.0 - np.sum((yt - yp) ** 2) / denom
        scores.append(TARGET_WEIGHTS[target] * r2)
    return float(np.sum(scores))


def enforce_biomass_constraints(pred: np.ndarray, mode: str = "derive_total") -> np.ndarray:
    pred = np.clip(np.asarray(pred, dtype=float).copy(), 0, None)
    col = {name: i for i, name in enumerate(TARGET_COLS)}
    if mode == "derive_gdm_total":
        pred[:, col["GDM_g"]] = pred[:, col["Dry_Green_g"]] + pred[:, col["Dry_Clover_g"]]
        pred[:, col["Dry_Total_g"]] = pred[:, col["GDM_g"]] + pred[:, col["Dry_Dead_g"]]
    elif mode == "derive_total":
        pred[:, col["Dry_Total_g"]] = (
            pred[:, col["Dry_Green_g"]]
            + pred[:, col["Dry_Dead_g"]]
            + pred[:, col["Dry_Clover_g"]]
        )
    elif mode != "none":
        raise ValueError(f"Unknown constraint mode: {mode}")
    return pred
