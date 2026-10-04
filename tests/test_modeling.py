"""Encoding, training, and the rule that test labels are not used."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.encode import encode_splits
from src.modeling import train_and_score
from src.plots import save_balance_plot, save_calibration_plot, save_overlap_plot
from src.metrics import calibration_table


def test_unseen_category_does_not_add_a_column() -> None:
    train = pd.DataFrame(
        {
            "age_mid": [50.0, 60.0, 70.0],
            "race_group": ["Caucasian", "Asian", "Caucasian"],
        }
    )
    other = pd.DataFrame(
        {
            "age_mid": [55.0, None],
            "race_group": ["Hispanic", "Asian"],
        }
    )
    train_x, other_x, _ignored, columns = encode_splits(
        train, other, other, ["age_mid"], ["race_group"]
    )
    assert columns == list(train_x.columns)
    assert list(other_x.columns) == columns
    assert "race_group_Hispanic" not in other_x.columns
    assert other_x.iloc[0].sum() == other_x.iloc[0]["age_mid"]


def test_changing_test_labels_does_not_change_scores() -> None:
    features, labels = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        random_state=0,
        weights=[0.75, 0.25],
    )
    frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(6)])
    train, val, test = frame.iloc[:180], frame.iloc[180:240], frame.iloc[240:]
    y_train, y_val, y_test = labels[:180], labels[180:240], labels[240:]
    config = [
        {
            "n_estimators": 25,
            "num_leaves": 7,
            "min_child_samples": 10,
            "learning_rate": 0.1,
            "early_stopping_rounds": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }
    ]
    original = train_and_score(
        train, y_train, val, y_val, test, y_test, config, seed=0, n_jobs=1
    )
    flipped = train_and_score(
        train, y_train, val, y_val, test, 1 - y_test, config, seed=0, n_jobs=1
    )
    np.testing.assert_allclose(original["p_test"], flipped["p_test"])
    assert np.all((original["p_test"] >= 0) & (original["p_test"] <= 1))


def test_plots_write_pngs(tmp_path) -> None:
    table = calibration_table(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_bins=2)
    save_calibration_plot(table, tmp_path / "cal.png")
    balance = pd.DataFrame(
        {
            "covariate": ["age_mid", "gender_male"],
            "smd_unweighted": [0.4, 0.2],
            "smd_weighted": [0.05, 0.01],
        }
    )
    save_balance_plot(balance, tmp_path / "smd.png")
    save_overlap_plot(
        np.array([0.2, 0.8, 0.4, 0.6]),
        np.array([False, True, False, True]),
        tmp_path / "overlap.png",
    )
    for name in ("cal.png", "smd.png", "overlap.png"):
        assert (tmp_path / name).stat().st_size > 0
