"""Plots for calibration, SHAP, overlap, and covariate balance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling import CalibratedBooster


def save_calibration_plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="perfect")
    if not table.empty:
        ax.plot(
            table["mean_predicted"],
            table["fraction_positive"],
            marker="o",
            color="#2c7fb8",
            label="calibrated LightGBM",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed 30-day readmission rate")
    ax.set_title("Test calibration (10 equal-width bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_shap_plot(
    model: CalibratedBooster,
    features: pd.DataFrame,
    path: Path,
    seed: int = 0,
    max_rows: int = 250,
) -> None:
    import shap

    rng = np.random.RandomState(seed)
    n = min(max_rows, len(features))
    if n == 0:
        return
    sample = features.iloc[rng.choice(len(features), size=n, replace=False)]
    explainer = shap.TreeExplainer(model.model.booster_)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    mean_abs = np.mean(np.abs(array), axis=0)
    order = np.argsort(mean_abs)[-15:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.asarray(sample.columns)[order], mean_abs[order], color="#2c7fb8")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Test-sample feature contributions")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_overlap_plot(propensity: np.ndarray, treated: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(propensity[~treated], bins=bins, alpha=0.55, label="No medication change", color="#9ecae1")
    ax.hist(propensity[treated], bins=bins, alpha=0.55, label="Medication change", color="#e6550d")
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Encounters")
    ax.set_title("Overlap of medication-change propensity")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_balance_plot(table: pd.DataFrame, path: Path) -> None:
    ordered = table.assign(abs_smd=lambda df: df["smd_weighted"].abs()).sort_values("abs_smd")
    # Show the covariates with the largest weighted imbalance, up to 20.
    show = ordered.tail(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(show))
    ax.scatter(show["smd_unweighted"].abs(), y, label="Before weighting", color="#bdbdbd")
    ax.scatter(show["smd_weighted"].abs(), y, label="After IPW", color="#2c7fb8")
    ax.axvline(0.1, color="#e6550d", linestyle="--", label="0.1 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(show["covariate"])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title("Covariate balance")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
