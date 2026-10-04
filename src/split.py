"""Patient-grouped train, validation, and test splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SEED, TEST_FRACTION, VAL_FRACTION


def assign_patient_splits(
    patient_ids: np.ndarray,
    *,
    seed: int = SEED,
    test_fraction: float = TEST_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> dict[int, str]:
    """Map each patient to train, val, or test.

    The shuffle is seeded. Patients are sorted before the shuffle so the
    result depends only on the id set and the seed. Test patients are a
    prefix of the shuffled list and are never revisited for tuning.
    """
    if test_fraction <= 0 or val_fraction <= 0:
        raise ValueError("fractions must be positive")
    if test_fraction + val_fraction >= 1:
        raise ValueError("test_fraction + val_fraction must be < 1")
    patients = np.unique(np.asarray(patient_ids))
    if len(patients) < 3:
        raise ValueError("need at least 3 patients to form train, val, and test")
    rng = np.random.RandomState(seed)
    order = patients.copy()
    rng.shuffle(order)
    n = len(order)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    n_test = min(max(n_test, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_test - 1)
    labels: dict[int, str] = {}
    for patient in order[:n_test]:
        labels[int(patient)] = "test"
    for patient in order[n_test : n_test + n_val]:
        labels[int(patient)] = "val"
    for patient in order[n_test + n_val :]:
        labels[int(patient)] = "train"
    return labels


def apply_split(frame: pd.DataFrame, *, seed: int = SEED) -> pd.DataFrame:
    labels = assign_patient_splits(frame["patient_nbr"].to_numpy(), seed=seed)
    out = frame.copy()
    out["split"] = out["patient_nbr"].map(lambda pid: labels[int(pid)])
    if out["split"].isna().any():
        raise RuntimeError("some encounters were not assigned a split")
    _assert_group_integrity(out)
    return out


def _assert_group_integrity(frame: pd.DataFrame) -> None:
    counts = frame.groupby("patient_nbr")["split"].nunique()
    if int(counts.max()) != 1:
        raise RuntimeError("a patient appears in more than one split")
    present = set(frame["split"].unique())
    if present != {"train", "val", "test"}:
        raise RuntimeError(f"expected train/val/test, found {present}")


def split_frames(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"] == "train"].copy()
    val = frame.loc[frame["split"] == "val"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    return train, val, test


def save_split(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["encounter_id", "patient_nbr", "split"]
    frame[cols].to_csv(path, index=False)


def subset_patients(frame: pd.DataFrame, n_patients: int, seed: int = SEED) -> pd.DataFrame:
    patients = np.unique(frame["patient_nbr"].to_numpy())
    if n_patients >= len(patients):
        return frame.copy()
    rng = np.random.RandomState(seed)
    chosen = set(rng.choice(patients, size=n_patients, replace=False).tolist())
    return frame.loc[frame["patient_nbr"].isin(chosen)].copy()
