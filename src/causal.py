"""Propensity-score IPW and AIPW for a medication-change indicator.

The estimand is the encounter-level average treatment effect of a recorded
diabetes-medication change on 30-day readmission. Nuisance models are fit on
the analysis sample. Confidence intervals resample patients and recompute the
estimator from the fixed scores, so they do not include uncertainty from
selecting the propensity specification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def standardized_mean_difference(
    values: np.ndarray,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Austin SMD: weighted mean difference over the unweighted pooled SD."""
    x = np.asarray(values, dtype=float)
    group = np.asarray(treated, dtype=bool)
    w = np.ones(len(x), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if len(x) != len(group) or len(x) != len(w):
        raise ValueError("values, treated, and weights must have the same length")

    def weighted_mean(mask: np.ndarray) -> float:
        ww = w[mask]
        total = float(ww.sum())
        if total <= 0:
            return float("nan")
        return float(np.sum(ww * x[mask]) / total)

    def sample_variance(mask: np.ndarray) -> float:
        sample = x[mask]
        if len(sample) < 2:
            return float("nan")
        return float(np.var(sample, ddof=1))

    mean_t = weighted_mean(group)
    mean_c = weighted_mean(~group)
    var_t = sample_variance(group)
    var_c = sample_variance(~group)
    if not all(np.isfinite(item) for item in (mean_t, mean_c, var_t, var_c)):
        return float("nan")
    denom = float(np.sqrt((var_t + var_c) / 2.0))
    gap = float(mean_t - mean_c)
    if denom < 1e-12:
        return 0.0 if abs(gap) < 1e-8 else float("inf")
    return gap / denom


def naive_risk_difference(outcome: np.ndarray, treated: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated, dtype=bool)
    if t.sum() == 0 or (~t).sum() == 0:
        raise ValueError("both treatment groups are required")
    return float(y[t].mean() - y[~t].mean())


def ipw_risk_difference(outcome: np.ndarray, treated: np.ndarray, propensity: np.ndarray) -> float:
    """Hajek (normalized) inverse-probability weighted risk difference."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), 1e-6, 1.0 - 1e-6)
    w1 = t / e
    w0 = (1.0 - t) / (1.0 - e)
    if w1.sum() <= 0 or w0.sum() <= 0:
        raise ValueError("IPW weights sum to zero")
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def aipw_scores(
    outcome: np.ndarray,
    treated: np.ndarray,
    propensity: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
) -> np.ndarray:
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), 1e-6, 1.0 - 1e-6)
    m1 = np.asarray(mu1, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    return (m1 - m0) + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)


def aipw_risk_difference(
    outcome: np.ndarray,
    treated: np.ndarray,
    propensity: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
) -> float:
    return float(np.mean(aipw_scores(outcome, treated, propensity, mu1, mu0)))


def percentile_interval(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def cluster_bootstrap_mean(
    patient_ids: np.ndarray,
    scores: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Encounter-mean of ``scores``, resampling patients with replacement."""
    _patients, inverse = np.unique(patient_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(scores, dtype=float))
    counts = np.bincount(inverse).astype(float)
    rng = np.random.RandomState(seed)
    drawn = rng.randint(0, len(sums), size=(n_boot, len(sums)))
    return sums[drawn].sum(axis=1) / counts[drawn].sum(axis=1)


def cluster_bootstrap_ipw(
    patient_ids: np.ndarray,
    outcome: np.ndarray,
    treated: np.ndarray,
    propensity: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), 1e-6, 1.0 - 1e-6)
    w1 = t / e
    w0 = (1.0 - t) / (1.0 - e)
    _patients, inverse = np.unique(patient_ids, return_inverse=True)
    sum_w1 = np.bincount(inverse, weights=w1)
    sum_w1y = np.bincount(inverse, weights=w1 * y)
    sum_w0 = np.bincount(inverse, weights=w0)
    sum_w0y = np.bincount(inverse, weights=w0 * y)
    rng = np.random.RandomState(seed)
    drawn = rng.randint(0, len(sum_w1), size=(n_boot, len(sum_w1)))
    mu1 = sum_w1y[drawn].sum(axis=1) / sum_w1[drawn].sum(axis=1)
    mu0 = sum_w0y[drawn].sum(axis=1) / sum_w0[drawn].sum(axis=1)
    return mu1 - mu0


def _fit_propensity(
    features: pd.DataFrame,
    treated: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=400, random_state=seed)
    model.fit(scaled, treated)
    return model.predict_proba(scaled)[:, 1]


def _outcome_predictions(
    features: pd.DataFrame,
    outcome: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    y_arm = outcome[mask]
    if int(mask.sum()) < 20 or len(np.unique(y_arm)) < 2:
        fill = float(y_arm.mean()) if int(mask.sum()) else 0.0
        return np.full(len(outcome), fill, dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.loc[mask])
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400, random_state=seed)
    model.fit(scaled, y_arm)
    return model.predict_proba(scaler.transform(features))[:, 1]


def _max_abs_smd(features: pd.DataFrame, treated: np.ndarray, weights: np.ndarray) -> float:
    values = [
        abs(standardized_mean_difference(features[col].to_numpy(), treated, weights))
        for col in features.columns
    ]
    finite = [item for item in values if np.isfinite(item)]
    if not finite:
        return float("inf")
    return float(max(finite))


def _smd_table(features: pd.DataFrame, treated: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        unweighted = standardized_mean_difference(features[col].to_numpy(), treated, None)
        weighted = standardized_mean_difference(features[col].to_numpy(), treated, weights)
        rows.append(
            {
                "covariate": col,
                "smd_unweighted": unweighted,
                "smd_weighted": weighted,
            }
        )
    return pd.DataFrame(rows)


def _continuous_columns(features: pd.DataFrame) -> list[str]:
    chosen = []
    for col in features.columns:
        if int(features[col].nunique()) > 3:
            chosen.append(col)
    return chosen


def estimate_effects(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_boot: int = 200,
    seed: int = SEED,
) -> dict:
    """Fit propensity and outcome models and return effect estimates.

    The propensity penalty and clipping threshold are chosen to shrink the
    maximum absolute weighted SMD. That search uses treatment and covariates
    only. Outcome regression is fit after the specification is locked.
    """
    if len(frame) != len(features):
        raise ValueError("frame and features must have the same number of rows")
    outcome = frame["readmit_30"].to_numpy(dtype=float)
    treated = frame["med_change"].to_numpy(dtype=int)
    patient_ids = frame["patient_nbr"].to_numpy()
    design = features.reset_index(drop=True).astype(float)
    treated_bool = treated.astype(bool)

    best: dict | None = None
    for use_squares in (False, True):
        current = design
        if use_squares:
            worst = best["table"].assign(abs_smd=lambda df: df["smd_weighted"].abs())
            ranked = worst.sort_values("abs_smd", ascending=False)
            to_square = [
                row.covariate
                for row in ranked.itertuples()
                if row.covariate in design.columns and row.covariate in _continuous_columns(design)
            ][:4]
            current = design.copy()
            for col in to_square:
                current[f"{col}__sq"] = current[col] ** 2
        for c_value in (1.0, 50.0):
            raw = _fit_propensity(current, treated, c_value, seed)
            for clip in (0.01, 0.05):
                clipped = np.clip(raw, clip, 1.0 - clip)
                weights = np.where(treated_bool, 1.0 / clipped, 1.0 / (1.0 - clipped))
                # Balance is judged on the pre-specified covariates, not on
                # square terms added only to help the propensity model fit.
                table = _smd_table(design, treated_bool, weights)
                max_smd = float(np.nanmax(np.abs(table["smd_weighted"].to_numpy())))
                candidate = {
                    "raw": raw,
                    "clipped": clipped,
                    "weights": weights,
                    "table": table,
                    "max_smd": max_smd,
                    "c_value": c_value,
                    "clip": clip,
                    "design": current,
                    "used_squares": use_squares,
                }
                if best is None or max_smd < best["max_smd"]:
                    best = candidate
        assert best is not None
        if best["max_smd"] < 0.1:
            break

    assert best is not None
    locked = best["design"]
    mu1 = _outcome_predictions(locked, outcome, treated_bool, seed)
    mu0 = _outcome_predictions(locked, outcome, ~treated_bool, seed + 1)
    ipw = ipw_risk_difference(outcome, treated, best["clipped"])
    scores = aipw_scores(outcome, treated, best["clipped"], mu1, mu0)
    aipw = float(np.mean(scores))
    naive = naive_risk_difference(outcome, treated_bool)
    ipw_boots = cluster_bootstrap_ipw(
        patient_ids, outcome, treated, best["clipped"], n_boot, seed
    )
    aipw_boots = cluster_bootstrap_mean(patient_ids, scores, n_boot, seed + 1)
    ipw_ci = percentile_interval(ipw_boots)
    aipw_ci = percentile_interval(aipw_boots)
    unweighted_max = float(np.nanmax(np.abs(best["table"]["smd_unweighted"].to_numpy())))
    propensity = best["raw"]
    detail = pd.DataFrame(
        {
            "encounter_id": frame["encounter_id"].to_numpy(),
            "patient_nbr": patient_ids,
            "treatment": treated,
            "outcome": outcome,
            "propensity": propensity,
            "propensity_clipped": best["clipped"],
            "weight": best["weights"],
            "mu0": mu0,
            "mu1": mu1,
            "aipw_score": scores,
        }
    )
    return {
        "ipw": ipw,
        "aipw": aipw,
        "naive": naive,
        "ipw_ci": ipw_ci,
        "aipw_ci": aipw_ci,
        "max_smd_weighted": best["max_smd"],
        "max_smd_unweighted": unweighted_max,
        "c_value": best["c_value"],
        "clip": best["clip"],
        "used_squares": best["used_squares"],
        "propensity_min": float(np.min(propensity)),
        "propensity_max": float(np.max(propensity)),
        "overlap_frac": float(np.mean((propensity >= 0.1) & (propensity <= 0.9))),
        "n": int(len(frame)),
        "n_treated": int(treated_bool.sum()),
        "smd_table": best["table"],
        "detail": detail,
        "propensity": propensity,
        "treated": treated_bool,
    }
