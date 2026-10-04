"""SQL feature view: history, diagnosis groups, and cohort filters."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.config import MED_COLUMNS
from src.data import DataQualityError
from src.features import build_feature_frame, load_raw_frame, quality_report


def _row(**overrides) -> dict:
    row = {column: "No" for column in MED_COLUMNS}
    row.update(
        {
            "encounter_id": 1,
            "patient_nbr": 1,
            "race": "Caucasian",
            "gender": "Female",
            "age": "[70-80)",
            "admission_type_id": 1,
            "discharge_disposition_id": 1,
            "admission_source_id": 7,
            "time_in_hospital": 4,
            "payer_code": "MC",
            "medical_specialty": "Cardiology",
            "num_lab_procedures": 40,
            "num_procedures": 1,
            "num_medications": 12,
            "number_outpatient": 1,
            "number_emergency": 0,
            "number_inpatient": 2,
            "diag_1": "428",
            "diag_2": "250.01",
            "diag_3": "V58",
            "number_diagnoses": 6,
            "max_glu_serum": None,
            "A1Cresult": ">8",
            "change": "No",
            "diabetesMed": "Yes",
            "readmitted": "NO",
        }
    )
    row.update(overrides)
    return row


def _frame() -> pd.DataFrame:
    rows = [
        _row(encounter_id=1, patient_nbr=10, readmitted="NO", time_in_hospital=4, change="No"),
        _row(
            encounter_id=2,
            patient_nbr=10,
            readmitted="<30",
            time_in_hospital=2,
            change="Ch",
            metformin="Up",
            diag_1="250.13",
        ),
        _row(
            encounter_id=3,
            patient_nbr=10,
            readmitted="NO",
            time_in_hospital=6,
            change="No",
            diag_1="V57",
        ),
        _row(encounter_id=4, patient_nbr=20, discharge_disposition_id=11, readmitted="NO"),
        _row(encounter_id=5, patient_nbr=30, diag_1="486", readmitted=">30", gender="Male"),
    ]
    return pd.DataFrame(rows)


def test_history_excludes_current_and_future_encounters() -> None:
    con = duckdb.connect()
    load_raw_frame(con, _frame())
    features, report = build_feature_frame(con, expect_published_rows=False)
    con.close()
    patient = features.loc[features["patient_nbr"] == 10].sort_values("encounter_id")
    assert list(patient["encounter_id"]) == [1, 2, 3]
    assert list(patient["prior_encounters"]) == [0, 1, 2]
    # Encounter 1 is not a 30-day readmission. Encounter 2 is. Encounter 3 must
    # see only that one prior event, not its own label.
    assert list(patient["prior_readmit_30"]) == [0, 0, 1]
    assert list(patient["prior_hospital_days"]) == [0, 4, 6]
    assert report["n_excluded"] == 1
    assert 4 not in set(features["encounter_id"])


def test_diagnosis_and_medication_flags() -> None:
    con = duckdb.connect()
    load_raw_frame(con, _frame())
    features, _report = build_feature_frame(con, expect_published_rows=False)
    con.close()
    first = features.loc[features["encounter_id"] == 1].iloc[0]
    second = features.loc[features["encounter_id"] == 2].iloc[0]
    third = features.loc[features["encounter_id"] == 3].iloc[0]
    fifth = features.loc[features["encounter_id"] == 5].iloc[0]
    assert first["diag1_group"] == "circulatory"
    assert first["diag2_group"] == "diabetes"
    assert first["diag3_group"] == "vcode"
    assert second["diag1_group"] == "diabetes"
    assert second["n_med_up"] == 1
    assert second["med_change"] == 1
    assert third["diag1_group"] == "vcode"
    assert fifth["diag1_group"] == "respiratory"
    assert fifth["gender_male"] == 1
    assert first["age_mid"] == 75
    assert first["total_prior_util"] == 3
    assert first["a1c_group"] == "gt8"


def test_duplicate_encounter_id_fails_quality() -> None:
    frame = _frame()
    frame.loc[0, "encounter_id"] = 2
    con = duckdb.connect()
    load_raw_frame(con, frame)
    with pytest.raises(DataQualityError, match="duplicate"):
        quality_report(con)
    con.close()


def test_bad_readmission_label_fails_quality() -> None:
    frame = _frame()
    frame.loc[0, "readmitted"] = "MAYBE"
    con = duckdb.connect()
    load_raw_frame(con, frame)
    with pytest.raises(DataQualityError, match="readmitted"):
        quality_report(con)
    con.close()
