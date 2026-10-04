"""Patient-level split integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.split import apply_split, assign_patient_splits, save_split, subset_patients


def test_splits_are_disjoint_and_cover_every_patient() -> None:
    labels = assign_patient_splits(np.arange(200), seed=0)
    groups = {"train": set(), "val": set(), "test": set()}
    for patient, split in labels.items():
        groups[split].add(patient)
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert set().union(*groups.values()) == set(range(200))


def test_split_fractions_are_near_the_requested_sizes() -> None:
    labels = assign_patient_splits(np.arange(1000), seed=0, test_fraction=0.15, val_fraction=0.15)
    counts = pd.Series(labels).value_counts()
    assert counts["test"] == pytest.approx(150, abs=1)
    assert counts["val"] == pytest.approx(150, abs=1)
    assert counts["train"] == 1000 - counts["test"] - counts["val"]


def test_same_seed_is_stable_and_a_new_seed_moves_patients() -> None:
    first = assign_patient_splits(np.arange(50), seed=0)
    second = assign_patient_splits(np.arange(50), seed=0)
    third = assign_patient_splits(np.arange(50), seed=1)
    assert first == second
    assert first != third


def test_encounters_from_one_patient_stay_together() -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(40),
            "patient_nbr": np.repeat(np.arange(20), 2),
        }
    )
    assigned = apply_split(frame, seed=0)
    mixed = assigned.groupby("patient_nbr")["split"].nunique()
    assert int(mixed.max()) == 1
    patients = {
        split: set(assigned.loc[assigned["split"] == split, "patient_nbr"])
        for split in ("train", "val", "test")
    }
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])


def test_save_split_and_subset(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(30),
            "patient_nbr": np.arange(30),
        }
    )
    assigned = apply_split(frame, seed=0)
    path = tmp_path / "splits" / "patients.csv"
    save_split(assigned, path)
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["encounter_id", "patient_nbr", "split"]
    subset = subset_patients(assigned, 8, seed=0)
    assert subset["patient_nbr"].nunique() == 8
    assert len(subset_patients(assigned, 1000, seed=0)) == len(assigned)


def test_reject_tiny_inputs() -> None:
    with pytest.raises(ValueError):
        assign_patient_splits(np.array([1, 1, 2]), seed=0)
    with pytest.raises(ValueError):
        assign_patient_splits(np.arange(10), seed=0, test_fraction=0.0, val_fraction=0.2)
    with pytest.raises(ValueError):
        assign_patient_splits(np.arange(10), seed=0, test_fraction=0.6, val_fraction=0.5)
