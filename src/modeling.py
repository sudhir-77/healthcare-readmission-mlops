"""LightGBM, isotonic calibration, logistic baseline, and MLflow registry."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.metrics import auroc


class CalibratedBooster:
    """LightGBM classifier with an isotonic map on the positive-class score."""

    def __init__(
        self,
        model: lgb.LGBMClassifier,
        calibrator: IsotonicRegression,
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.feature_names = list(feature_names)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = _as_feature_frame(features, self.feature_names)
        raw = self.model.predict_proba(frame)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


class LogisticBaseline:
    def __init__(self, scaler: StandardScaler, model: LogisticRegression, feature_names: list[str]) -> None:
        self.scaler = scaler
        self.model = model
        self.feature_names = list(feature_names)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = _as_feature_frame(features, self.feature_names)
        scaled = self.scaler.transform(frame)
        return self.model.predict_proba(scaled)


def _as_feature_frame(features: pd.DataFrame | np.ndarray, names: list[str]) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        missing = [name for name in names if name not in features.columns]
        if missing:
            raise ValueError(f"missing features: {missing[:5]}")
        return features.loc[:, names]
    array = np.asarray(features, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} features, got {array.shape[1]}")
    return pd.DataFrame(array, columns=names)


def fit_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    *,
    seed: int = SEED,
    n_jobs: int = -1,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(params.get("n_estimators", 400)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        num_leaves=int(params.get("num_leaves", 31)),
        min_child_samples=int(params.get("min_child_samples", 50)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_X=x_val,
        eval_y=y_val,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(int(params.get("early_stopping_rounds", 30)), verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def fit_isotonic(raw_scores: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(raw_scores, dtype=float), np.asarray(y_val, dtype=float))
    return calibrator


def fit_logistic_baseline(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = SEED,
) -> LogisticBaseline:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=seed,
    )
    model.fit(scaled, y_train)
    return LogisticBaseline(scaler, model, feature_names)


def train_and_score(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray | None,
    configs: list[dict],
    *,
    seed: int = SEED,
    n_jobs: int = -1,
) -> dict:
    """Pick a LightGBM config by validation AUROC, then calibrate on validation.

    ``y_test`` is length-checked and otherwise ignored. Model selection uses
    validation scores only.
    """
    if y_test is not None and len(y_test) != len(x_test):
        raise ValueError("y_test length does not match x_test")
    if not configs:
        raise ValueError("configs must not be empty")

    best: dict | None = None
    for params in configs:
        model = fit_lightgbm(x_train, y_train, x_val, y_val, params, seed=seed, n_jobs=n_jobs)
        raw_val = model.predict_proba(x_val)[:, 1]
        score = auroc(y_val, raw_val)
        if best is None or score > best["val_auroc_raw"]:
            best = {"model": model, "params": dict(params), "val_auroc_raw": score, "raw_val": raw_val}
    assert best is not None
    calibrator = fit_isotonic(best["raw_val"], y_val)
    calibrated = CalibratedBooster(best["model"], calibrator, list(x_train.columns))
    baseline = fit_logistic_baseline(x_train, y_train, list(x_train.columns), seed=seed)
    return {
        "model": calibrated,
        "baseline": baseline,
        "params": best["params"],
        "val_auroc_raw": float(best["val_auroc_raw"]),
        "p_val": calibrated.predict_proba(x_val)[:, 1],
        "p_test": calibrated.predict_proba(x_test)[:, 1],
        "p_test_baseline": baseline.predict_proba(x_test)[:, 1],
        "p_val_baseline": baseline.predict_proba(x_val)[:, 1],
    }


class BoosterPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper so the calibrated booster can be registered."""

    def __init__(self, booster: CalibratedBooster | None = None) -> None:
        self.booster = booster

    def predict(
        self,
        context: object,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        del context, params
        if self.booster is None:
            raise RuntimeError("booster is not loaded")
        return self.booster.predict_proba(model_input)[:, 1]


def register_model(
    model: CalibratedBooster,
    params: dict,
    metrics: dict[str, float],
    tracking_dir: Path,
) -> str:
    """Log the calibrated model and register it in a local sqlite MLflow store."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = tracking_dir / "mlflow.db"
    artifact_root = tracking_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve().as_posix()}")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("readmission-30day")
    if experiment is None:
        client.create_experiment("readmission-30day", artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment("readmission-30day")
    with mlflow.start_run(run_name="calibrated-lightgbm"):
        mlflow.log_params({key: _param_value(value) for key, value in params.items()})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        mlflow.pyfunc.log_model(
            name="readmission-model",
            python_model=BoosterPyfunc(model),
            registered_model_name="readmission_30day",
        )
    versions = client.search_model_versions("name='readmission_30day'")
    version = max(int(item.version) for item in versions)
    return f"models:/readmission_30day/{version}"


def _param_value(value: object) -> str | int | float:
    if isinstance(value, (int, float, str)):
        return value
    return str(value)
