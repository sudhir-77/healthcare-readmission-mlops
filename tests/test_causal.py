"""Propensity-score formulas on hand-computed samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal import (
    aipw_risk_difference,
    cluster_bootstrap_mean,
    estimate_effects,
    ipw_risk_difference,
    naive_risk_difference,
    percentile_interval,
    standardized_mean_difference,
)


def test_smd_of_a_known_shift() -> None:
    values = np.array([1.0, 2.0, 3.0, 0.0, 1.0, 2.0])
    treated = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    # Means 2 and 1. Sample variance of each group is 1. SMD = 1.
    assert standardized_mean_difference(values, treated) == pytest.approx(1.0)


def test_ipw_hajek_on_four_rows() -> None:
    outcome = np.array([1.0, 1.0, 0.0, 0.0])
    treated = np.array([1, 1, 0, 0])
    propensity = np.array([0.5, 0.5, 0.5, 0.5])
    assert ipw_risk_difference(outcome, treated, propensity) == pytest.approx(1.0)
    assert naive_risk_difference(outcome, treated.astype(bool)) == pytest.approx(1.0)


def test_aipw_two_rows() -> None:
    outcome = np.array([1.0, 0.0])
    treated = np.array([1.0, 0.0])
    propensity = np.array([0.5, 0.5])
    mu1 = np.array([0.4, 0.4])
    mu0 = np.array([0.2, 0.2])
    # Scores are 1.4 and 0.6. The mean is 1.0.
    assert aipw_risk_difference(outcome, treated, propensity, mu1, mu0) == pytest.approx(1.0)


def test_percentile_interval_endpoints() -> None:
    samples = np.arange(100, dtype=float)
    lo, hi = percentile_interval(samples)
    # 2.5th percentile: 0.025 * 99 = 2.475
    # 97.5th percentile: 0.975 * 99 = 96.525
    assert lo == pytest.approx(2.475)
    assert hi == pytest.approx(96.525)


def test_cluster_bootstrap_mean_of_a_constant() -> None:
    patients = np.array([1, 1, 2, 3])
    scores = np.array([0.2, 0.2, 0.2, 0.2])
    draws = cluster_bootstrap_mean(patients, scores, n_boot=15, seed=0)
    assert np.allclose(draws, 0.2)


def test_estimate_effects_returns_finite_balance() -> None:
    rng = np.random.RandomState(0)
    n = 500
    features = pd.DataFrame(
        {
            "age_mid": rng.normal(70, 10, size=n),
            "number_inpatient": rng.poisson(1, size=n).astype(float),
            "gender_male": rng.binomial(1, 0.45, size=n).astype(float),
        }
    )
    treated = rng.binomial(1, 0.4, size=n)
    outcome = rng.binomial(1, 0.15, size=n)
    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": np.arange(n),
            "readmit_30": outcome,
            "med_change": treated,
        }
    )
    result = estimate_effects(frame, features, n_boot=20, seed=0)
    assert np.isfinite(result["ipw"])
    assert np.isfinite(result["aipw"])
    assert result["smd_table"]["covariate"].tolist() == ["age_mid", "number_inpatient", "gender_male"]
    assert result["max_smd_weighted"] < 0.1
    assert len(result["detail"]) == n
