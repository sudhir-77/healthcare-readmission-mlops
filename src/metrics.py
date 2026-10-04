"""Ranking and calibration metrics.

Bin edges follow equal-width partitions of [0, 1]. A score of 1.0 falls in
the last bin. Empty bins are skipped and do not contribute to ECE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    if table.empty:
        raise ValueError("no observations to score")
    n = int(table["count"].sum())
    weighted = (table["count"] / n) * (table["mean_predicted"] - table["fraction_positive"]).abs()
    return float(weighted.sum())


def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if len(y) == 0:
        raise ValueError("empty inputs")
    if np.any(~np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against interior edges so bin i is [edges[i], edges[i+1]).
    # The rightmost edge is inclusive via a clip.
    bin_id = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": count,
                "mean_predicted": float(p[mask].mean()),
                "fraction_positive": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def percentile(values: np.ndarray, q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("empty values")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    return float(np.percentile(arr, q))
