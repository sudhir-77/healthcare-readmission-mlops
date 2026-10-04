"""Paths, seeds, and cohort constants."""

from __future__ import annotations

from pathlib import Path

SEED = 0
SEEDS = [SEED]

PROJECT_SLUG = "healthcare-readmission-mlops"
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
SPLIT_DIR = ROOT / "data" / "splits"

EXPECTED_RAW_ROWS = 101_766

# UCI ML Repository, CC BY 4.0.
DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
]

# Expired or hospice discharges are not eligible for hospital readmission.
EXCLUDED_DISCHARGE_IDS = (11, 13, 14, 19, 20, 21)

TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

# Full-run latency benchmark. Quick mode uses a smaller count.
BENCHMARK_REQUESTS = 500
BENCHMARK_WARMUP = 20
QUICK_BENCHMARK_REQUESTS = 40
QUICK_BENCHMARK_WARMUP = 5
QUICK_PATIENTS = 1800
QUICK_BOOTSTRAPS = 30
FULL_BOOTSTRAPS = 200

MED_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

PREDICTION_NUMERIC = [
    "age_mid",
    "gender_male",
    "gender_known",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "prior_encounters",
    "prior_readmit_30",
    "prior_hospital_days",
    "n_med_up",
    "n_med_down",
    "n_med_active",
    "on_diabetes_med",
    "med_change",
    "total_prior_util",
]

PREDICTION_CATEGORICAL = [
    "race_group",
    "admission_type",
    "discharge_group",
    "admission_source_group",
    "diag1_group",
    "diag2_group",
    "diag3_group",
    "a1c_group",
    "glucose_group",
    "specialty_group",
    "insulin_group",
    "payer_group",
]

# Pre-treatment covariates for the medication-change effect.
# Discharge disposition, in-hospital medication counts, and length of stay
# are left out because they can be affected by the treatment decision.
CAUSAL_NUMERIC = [
    "age_mid",
    "gender_male",
    "gender_known",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "prior_encounters",
    "prior_readmit_30",
    "prior_hospital_days",
    "total_prior_util",
]

CAUSAL_CATEGORICAL = [
    "race_group",
    "admission_type",
    "admission_source_group",
    "diag1_group",
    "a1c_group",
]
