"""Train-only encoding of numeric and categorical features."""

from __future__ import annotations

import pandas as pd

from src.config import (
    CAUSAL_CATEGORICAL,
    CAUSAL_NUMERIC,
    PREDICTION_CATEGORICAL,
    PREDICTION_NUMERIC,
)


def prediction_frames(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    return encode_splits(train, val, test, PREDICTION_NUMERIC, PREDICTION_CATEGORICAL)


def causal_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoded, columns = _encode(frame, frame, CAUSAL_NUMERIC, CAUSAL_CATEGORICAL)
    return encoded, columns


def encode_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit medians and dummy columns on train. Align val and test to them."""
    train_x, columns = _encode(train, train, numeric, categorical)
    val_x, _ = _encode(train, val, numeric, categorical, columns=columns)
    test_x, _ = _encode(train, test, numeric, categorical, columns=columns)
    return train_x, val_x, test_x, columns


def _encode(
    fit_frame: pd.DataFrame,
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    medians = fit_frame[numeric].median(numeric_only=True)
    nums = frame[numeric].astype(float).copy()
    nums = nums.fillna(medians)
    nums = nums.fillna(0.0)
    cats = frame[categorical].copy()
    for col in categorical:
        cats[col] = cats[col].astype("object").where(cats[col].notna(), "missing").astype(str)
    dummies = pd.get_dummies(cats, columns=categorical, dtype=float)
    encoded = pd.concat([nums.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    if columns is None:
        # Drop train columns with no variation. They do not help either model.
        nunique = encoded.nunique(dropna=False)
        columns = [col for col in encoded.columns if int(nunique[col]) > 1]
    encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded, list(columns)
