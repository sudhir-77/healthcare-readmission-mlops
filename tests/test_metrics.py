"""Hand-checked ranking, calibration, and percentile values."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import auroc, brier_score, expected_calibration_error, percentile


def test_auroc_matches_pairwise_count() -> None:
    # Positives at 0.2 and 0.4, negatives at 0.1 and 0.3.
    # Three of the four positive-negative pairs are correctly ordered: 3/4.
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert auroc(y_true, scores) == pytest.approx(0.75)


def test_ece_two_bins_hand_computed() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=float)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    # Bin [0, 0.5): conf 0.15, acc 0, weight 0.5 -> 0.075
    # Bin [0.5, 1]: conf 0.85, acc 1, weight 0.5 -> 0.075
    assert expected_calibration_error(y_true, scores, n_bins=2) == pytest.approx(0.15)


def test_ece_puts_probability_one_in_the_last_bin() -> None:
    y_true = np.array([0.0, 1.0])
    scores = np.array([0.0, 1.0])
    # Each point sits in its own half. |0-0|*0.5 + |1-1|*0.5 = 0.
    assert expected_calibration_error(y_true, scores, n_bins=2) == pytest.approx(0.0)


def test_brier_score() -> None:
    assert brier_score(np.array([0, 1]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert brier_score(np.array([0, 1]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_percentile_linear_interpolation() -> None:
    # index = 0.95 * (4 - 1) = 2.85; value = 3 + 0.85 * 1 = 3.85
    assert percentile(np.array([1.0, 2.0, 3.0, 4.0]), 95) == pytest.approx(3.85)


def test_auroc_is_nan_for_one_class() -> None:
    assert np.isnan(auroc(np.array([0, 0, 0]), np.array([0.2, 0.3, 0.4])))
