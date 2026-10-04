"""DuckDB SQL feature views for the diabetes encounter cohort."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.config import EXCLUDED_DISCHARGE_IDS, EXPECTED_RAW_ROWS, MED_COLUMNS
from src.data import DataQualityError, assert_row_count

ICD_GROUP_MACRO = """
CREATE OR REPLACE MACRO icd9_group(code) AS
CASE
  WHEN code IS NULL OR code = '' THEN 'missing'
  WHEN upper(substr(code, 1, 1)) = 'V' THEN 'vcode'
  WHEN upper(substr(code, 1, 1)) = 'E' THEN 'ecode'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) = 250 THEN 'diabetes'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 390 AND 459 THEN 'circulatory'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 460 AND 519 THEN 'respiratory'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 520 AND 579 THEN 'digestive'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 580 AND 629 THEN 'genitourinary'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 140 AND 239 THEN 'neoplasm'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 710 AND 739 THEN 'musculoskeletal'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 800 AND 999 THEN 'injury'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 240 AND 279 THEN 'endocrine'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 680 AND 709 THEN 'skin'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 780 AND 799 THEN 'symptoms'
  WHEN try_cast(regexp_extract(code, '^([0-9]+)', 1) AS INTEGER) BETWEEN 290 AND 319 THEN 'mental'
  ELSE 'other'
END
"""


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _med_sum(predicate: str) -> str:
    """Count medication columns after casting.

    DuckDB reads all-No drug columns as BOOLEAN, so compare on the lowercase
    text form. 'false' and 'no' mean the drug was not given.
    """
    parts = [
        f"CASE WHEN lower(CAST({_quoted(col)} AS VARCHAR)) {predicate} THEN 1 ELSE 0 END"
        for col in MED_COLUMNS
    ]
    return " + ".join(parts)


def feature_sql(excluded_ids: tuple[int, ...] = EXCLUDED_DISCHARGE_IDS) -> str:
    excluded = ", ".join(str(i) for i in excluded_ids)
    med_up = _med_sum("= 'up'")
    med_down = _med_sum("= 'down'")
    med_active = _med_sum("IN ('up', 'down', 'steady')")
    return f"""
    CREATE OR REPLACE VIEW encounter_features AS
    WITH base AS (
      SELECT
        CAST(encounter_id AS BIGINT) AS encounter_id,
        CAST(patient_nbr AS BIGINT) AS patient_nbr,
        race,
        gender,
        age,
        admission_type_id,
        discharge_disposition_id,
        admission_source_id,
        time_in_hospital,
        payer_code,
        medical_specialty,
        num_lab_procedures,
        num_procedures,
        num_medications,
        number_outpatient,
        number_emergency,
        number_inpatient,
        diag_1,
        diag_2,
        diag_3,
        number_diagnoses,
        max_glu_serum,
        A1Cresult,
        {", ".join(_quoted(col) for col in MED_COLUMNS)},
        change,
        diabetesMed,
        readmitted,
        CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END AS readmit_30
      FROM raw
    ),
    ordered AS (
      SELECT
        *,
        COALESCE(COUNT(*) OVER (
          PARTITION BY patient_nbr
          ORDER BY encounter_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS prior_encounters,
        COALESCE(SUM(readmit_30) OVER (
          PARTITION BY patient_nbr
          ORDER BY encounter_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS prior_readmit_30,
        COALESCE(SUM(time_in_hospital) OVER (
          PARTITION BY patient_nbr
          ORDER BY encounter_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS prior_hospital_days
      FROM base
    )
    SELECT
      encounter_id,
      patient_nbr,
      readmit_30,
      CASE WHEN change = 'Ch' THEN 1 WHEN change = 'No' THEN 0 END AS med_change,
      CASE
        WHEN age = '[0-10)' THEN 5
        WHEN age = '[10-20)' THEN 15
        WHEN age = '[20-30)' THEN 25
        WHEN age = '[30-40)' THEN 35
        WHEN age = '[40-50)' THEN 45
        WHEN age = '[50-60)' THEN 55
        WHEN age = '[60-70)' THEN 65
        WHEN age = '[70-80)' THEN 75
        WHEN age = '[80-90)' THEN 85
        WHEN age = '[90-100)' THEN 95
      END AS age_mid,
      CASE WHEN gender = 'Male' THEN 1 ELSE 0 END AS gender_male,
      CASE WHEN gender IN ('Male', 'Female') THEN 1 ELSE 0 END AS gender_known,
      time_in_hospital,
      num_lab_procedures,
      num_procedures,
      num_medications,
      number_outpatient,
      number_emergency,
      number_inpatient,
      number_diagnoses,
      prior_encounters,
      prior_readmit_30,
      prior_hospital_days,
      (number_outpatient + number_emergency + number_inpatient) AS total_prior_util,
      ({med_up}) AS n_med_up,
      ({med_down}) AS n_med_down,
      ({med_active}) AS n_med_active,
      CASE
        WHEN lower(CAST(diabetesMed AS VARCHAR)) IN ('yes', 'true') THEN 1
        ELSE 0
      END AS on_diabetes_med,
      CASE
        WHEN race IS NULL THEN 'missing'
        WHEN race = 'Caucasian' THEN 'Caucasian'
        WHEN race = 'AfricanAmerican' THEN 'AfricanAmerican'
        WHEN race = 'Hispanic' THEN 'Hispanic'
        WHEN race = 'Asian' THEN 'Asian'
        ELSE 'Other'
      END AS race_group,
      COALESCE(CAST(admission_type_id AS VARCHAR), 'missing') AS admission_type,
      CASE
        WHEN discharge_disposition_id IN (1, 6, 8) THEN 'home'
        WHEN discharge_disposition_id IN (3, 4, 22, 24, 27) THEN 'facility'
        WHEN discharge_disposition_id IN (2, 5, 9, 10, 15, 23, 28, 29) THEN 'transfer'
        WHEN discharge_disposition_id = 7 THEN 'ama'
        ELSE 'other'
      END AS discharge_group,
      CASE
        WHEN admission_source_id IN (7) THEN 'emergency'
        WHEN admission_source_id IN (1, 2, 3) THEN 'referral'
        WHEN admission_source_id IN (4, 5, 6) THEN 'transfer'
        ELSE 'other'
      END AS admission_source_group,
      icd9_group(diag_1) AS diag1_group,
      icd9_group(diag_2) AS diag2_group,
      icd9_group(diag_3) AS diag3_group,
      CASE
        WHEN A1Cresult IS NULL THEN 'none'
        WHEN A1Cresult = 'Norm' THEN 'norm'
        WHEN A1Cresult = '>7' THEN 'gt7'
        WHEN A1Cresult = '>8' THEN 'gt8'
        ELSE 'other'
      END AS a1c_group,
      CASE
        WHEN max_glu_serum IS NULL THEN 'none'
        WHEN max_glu_serum = 'Norm' THEN 'norm'
        WHEN max_glu_serum = '>200' THEN 'gt200'
        WHEN max_glu_serum = '>300' THEN 'gt300'
        ELSE 'other'
      END AS glucose_group,
      CASE
        WHEN medical_specialty IS NULL THEN 'missing'
        WHEN medical_specialty IN (
          'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
          'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
          'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology'
        ) THEN medical_specialty
        ELSE 'other'
      END AS specialty_group,
      CASE
        WHEN insulin IS NULL THEN 'missing'
        WHEN insulin IN ('No', 'Steady', 'Up', 'Down') THEN insulin
        ELSE 'other'
      END AS insulin_group,
      CASE
        WHEN payer_code IS NULL THEN 'missing'
        WHEN payer_code IN ('MC', 'HM', 'SP', 'BC', 'MD', 'CP', 'UN', 'OG', 'DM', 'CM')
          THEN payer_code
        ELSE 'other'
      END AS payer_group,
      discharge_disposition_id
    FROM ordered
    WHERE discharge_disposition_id NOT IN ({excluded})
      AND discharge_disposition_id IS NOT NULL
    """


def quality_report(con: duckdb.DuckDBPyConnection) -> dict:
    """Run data-quality checks against the ``raw`` table.

    Returns a dict with counts. Raises DataQualityError on hard failures.
    """
    columns = {row[0] for row in con.execute("DESCRIBE raw").fetchall()}
    required = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "change",
        "discharge_disposition_id",
        "diag_1",
        "age",
        "gender",
        "diabetesMed",
        "time_in_hospital",
    }
    missing = sorted(required - columns)
    n_rows = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    n_enc = con.execute("SELECT COUNT(DISTINCT encounter_id) FROM raw").fetchone()[0]
    n_patients = con.execute("SELECT COUNT(DISTINCT patient_nbr) FROM raw").fetchone()[0]
    n_null_id = con.execute(
        "SELECT COUNT(*) FROM raw WHERE encounter_id IS NULL OR patient_nbr IS NULL"
    ).fetchone()[0]
    bad_label = con.execute(
        "SELECT COUNT(*) FROM raw WHERE readmitted NOT IN ('<30', '>30', 'NO') OR readmitted IS NULL"
    ).fetchone()[0]
    bad_change = con.execute(
        "SELECT COUNT(*) FROM raw WHERE change NOT IN ('Ch', 'No') OR change IS NULL"
    ).fetchone()[0]
    bad_stay = con.execute(
        "SELECT COUNT(*) FROM raw WHERE time_in_hospital IS NULL OR time_in_hospital < 1 OR time_in_hospital > 14"
    ).fetchone()[0]
    failures: list[str] = []
    if missing:
        failures.append(f"missing columns: {missing}")
    if n_null_id:
        failures.append(f"{n_null_id} rows with null encounter_id or patient_nbr")
    if n_enc != n_rows:
        failures.append(f"duplicate encounter_id values: rows={n_rows} distinct={n_enc}")
    if bad_label:
        failures.append(f"{bad_label} rows with unexpected readmitted values")
    if bad_change:
        failures.append(f"{bad_change} rows with unexpected change values")
    if bad_stay:
        failures.append(f"{bad_stay} rows with time_in_hospital outside 1-14")
    report = {
        "n_rows": int(n_rows),
        "n_distinct_encounters": int(n_enc),
        "n_patients": int(n_patients),
        "failures": failures,
    }
    if failures:
        raise DataQualityError("; ".join(failures))
    return report


def load_raw_csv(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE raw AS
        SELECT * FROM read_csv(?, header=true, nullstr='?', auto_detect=true, sample_size=-1)
        """,
        [str(csv_path)],
    )


def load_raw_frame(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> None:
    con.register("_incoming", frame)
    con.execute("CREATE OR REPLACE TABLE raw AS SELECT * FROM _incoming")
    con.unregister("_incoming")


def build_feature_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    expect_published_rows: bool = False,
) -> tuple[pd.DataFrame, dict]:
    report = quality_report(con)
    if expect_published_rows:
        assert_row_count(report["n_rows"], EXPECTED_RAW_ROWS)
    con.execute(ICD_GROUP_MACRO)
    con.execute(feature_sql())
    frame = con.execute("SELECT * FROM encounter_features").fetchdf()
    if frame["encounter_id"].duplicated().any():
        raise DataQualityError("feature view produced duplicate encounter_id values")
    if frame["med_change"].isna().any():
        raise DataQualityError("med_change has nulls after the SQL view")
    if frame["readmit_30"].isna().any():
        raise DataQualityError("readmit_30 has nulls after the SQL view")
    report["n_cohort"] = int(len(frame))
    report["n_excluded"] = int(report["n_rows"] - report["n_cohort"])
    report["readmit_30_rate"] = float(frame["readmit_30"].mean()) if len(frame) else 0.0
    report["med_change_rate"] = float(frame["med_change"].mean()) if len(frame) else 0.0
    return frame, report
