"""End-to-end smoke test of `python run.py --quick`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run import main


def test_quick_pipeline_writes_recomputable_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    model_dir = tmp_path / "models"
    main(
        [
            "--quick",
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
        ]
    )
    document = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert document["project"] == "healthcare-readmission-mlops"
    assert document["seeds"] == [0]
    assert {"train", "val", "test"} <= set(document["splits"]["encounters"])
    for item in document["success_criteria"]:
        assert {"name", "value", "op", "threshold", "passed"} <= set(item)
    assert document["runtime_seconds"] < 180

    assignments = pd.read_csv(output_dir / "split_assignments.csv")
    mixed = assignments.groupby("patient_nbr")["split"].nunique()
    assert int(mixed.max()) == 1
    test_patients = set(assignments.loc[assignments["split"] == "test", "patient_nbr"])
    train_patients = set(assignments.loc[assignments["split"] == "train", "patient_nbr"])
    assert test_patients.isdisjoint(train_patients)

    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    assert {"encounter_id", "patient_nbr", "y_true", "p_lightgbm", "p_baseline"} <= set(predictions.columns)
    assert predictions["encounter_id"].is_unique
    assert set(predictions["patient_nbr"]) <= test_patients
    assert (output_dir / "causal_analysis.md").exists()
    assert (output_dir / "smd_table.csv").exists()
    assert (model_dir / "calibrated_lgbm.joblib").exists()
