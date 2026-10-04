"""End-to-end readmission pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.benchmark import run_latency_benchmark
from src.causal import estimate_effects
from src.config import (
    BENCHMARK_REQUESTS,
    BENCHMARK_WARMUP,
    FULL_BOOTSTRAPS,
    QUICK_BENCHMARK_REQUESTS,
    QUICK_BENCHMARK_WARMUP,
    QUICK_BOOTSTRAPS,
    QUICK_PATIENTS,
    ROOT,
    SEED,
)
from src.data import download_dataset
from src.encode import causal_matrix, prediction_frames
from src.features import build_feature_frame, load_raw_csv
from src.hardware import hardware_info
from src.metrics import auroc, brier_score, calibration_table, expected_calibration_error
from src.modeling import register_model, train_and_score
from src.plots import save_balance_plot, save_calibration_plot, save_overlap_plot, save_shap_plot
from src.reporting import (
    build_document,
    criterion,
    update_readme,
    write_causal_report,
    write_json,
    write_resume_notes,
)
from src.split import apply_split, save_split, split_frames, subset_patients

FULL_CONFIGS = [
    {
        "num_leaves": 31,
        "min_child_samples": 40,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "early_stopping_rounds": 30,
    },
    {
        "num_leaves": 63,
        "min_child_samples": 80,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "early_stopping_rounds": 30,
    },
    {
        "num_leaves": 31,
        "min_child_samples": 100,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "early_stopping_rounds": 40,
    },
]
QUICK_CONFIGS = [
    {
        "num_leaves": 15,
        "min_child_samples": 20,
        "learning_rate": 0.1,
        "n_estimators": 40,
        "early_stopping_rounds": 8,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }
]


def run_pipeline(
    *,
    quick: bool = False,
    device: str = "cpu",
    output_dir: Path | None = None,
    model_dir: Path | None = None,
    measure_coverage: bool = True,
) -> dict:
    started = time.perf_counter()
    if device == "cuda":
        print("CUDA was requested. This pipeline trains LightGBM on CPU.", flush=True)
    output_dir = Path(output_dir or ROOT / "results")
    model_dir = Path(model_dir or ROOT / "models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading encounter file if needed...", flush=True)
    csv_path = download_dataset()
    con = duckdb.connect()
    try:
        load_raw_csv(con, csv_path)
        frame, quality = build_feature_frame(con, expect_published_rows=True)
    finally:
        con.close()
    print(
        f"Raw rows {quality['n_rows']}, cohort {quality['n_cohort']}, "
        f"excluded {quality['n_excluded']}",
        flush=True,
    )
    if quick:
        frame = subset_patients(frame, QUICK_PATIENTS, SEED)
        print(f"Quick mode: {frame['patient_nbr'].nunique()} patients", flush=True)

    frame = apply_split(frame, seed=SEED)
    save_split(frame, output_dir / "split_assignments.csv")
    train, val, test = split_frames(frame)
    x_train, x_val, x_test, columns = prediction_frames(train, val, test)
    y_train = train["readmit_30"].to_numpy(dtype=int)
    y_val = val["readmit_30"].to_numpy(dtype=int)
    y_test = test["readmit_30"].to_numpy(dtype=int)

    print("Training LightGBM and the logistic baseline...", flush=True)
    trained = train_and_score(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        QUICK_CONFIGS if quick else FULL_CONFIGS,
        seed=SEED,
        n_jobs=1 if quick else -1,
    )
    test_auroc = auroc(y_test, trained["p_test"])
    base_auroc = auroc(y_test, trained["p_test_baseline"])
    test_ece = expected_calibration_error(y_test, trained["p_test"], n_bins=10)
    base_ece = expected_calibration_error(y_test, trained["p_test_baseline"], n_bins=10)
    cal_table = calibration_table(y_test, trained["p_test"], n_bins=10)
    cal_table.to_csv(output_dir / "calibration_bins.csv", index=False)
    save_calibration_plot(cal_table, output_dir / "calibration_curve.png")

    predictions = pd.DataFrame(
        {
            "encounter_id": test["encounter_id"].to_numpy(),
            "patient_nbr": test["patient_nbr"].to_numpy(),
            "y_true": y_test,
            "p_lightgbm": trained["p_test"],
            "p_baseline": trained["p_test_baseline"],
        }
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "encounter_id": val["encounter_id"].to_numpy(),
            "patient_nbr": val["patient_nbr"].to_numpy(),
            "y_true": y_val,
            "p_lightgbm": trained["p_val"],
            "p_baseline": trained["p_val_baseline"],
        }
    ).to_csv(output_dir / "val_predictions.csv", index=False)

    print("Fitting the propensity-score analysis on train and validation...", flush=True)
    analysis = pd.concat([train, val], axis=0).reset_index(drop=True)
    causal_x, _causal_columns = causal_matrix(analysis)
    effects = estimate_effects(
        analysis,
        causal_x,
        n_boot=QUICK_BOOTSTRAPS if quick else FULL_BOOTSTRAPS,
        seed=SEED,
    )
    effects["smd_table"].to_csv(output_dir / "smd_table.csv", index=False)
    effects["detail"].to_csv(output_dir / "causal_scores.csv", index=False)
    covariates = causal_x.copy()
    covariates.insert(0, "encounter_id", analysis["encounter_id"].to_numpy())
    covariates.insert(1, "patient_nbr", analysis["patient_nbr"].to_numpy())
    covariates["treatment"] = analysis["med_change"].to_numpy()
    covariates["weight"] = effects["detail"]["weight"].to_numpy()
    covariates.to_csv(output_dir / "causal_covariates.csv", index=False)
    save_overlap_plot(effects["propensity"], effects["treated"], output_dir / "propensity_overlap.png")
    save_balance_plot(effects["smd_table"], output_dir / "smd_balance.png")
    save_shap_plot(
        trained["model"],
        x_test,
        output_dir / "shap_summary.png",
        seed=SEED,
        max_rows=40 if quick else 250,
    )

    model_path = model_dir / "calibrated_lgbm.joblib"
    meta_path = model_dir / "metadata.json"
    joblib.dump(trained["model"], model_path)
    print("Registering the model in the local MLflow store...", flush=True)
    model_uri = register_model(
        trained["model"],
        trained["params"],
        {"val_auroc_raw": trained["val_auroc_raw"], "test_auroc": test_auroc, "test_ece": test_ece},
        ROOT / "mlruns",
    )
    payload = {name: float(x_test.iloc[0][name]) for name in columns}
    (model_dir / "benchmark_payload.json").write_text(json.dumps(payload), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "feature_names": columns,
                "model_version": model_uri,
                "mlflow_registry": "readmission_30day",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_requests = QUICK_BENCHMARK_REQUESTS if quick else BENCHMARK_REQUESTS
    warmup = QUICK_BENCHMARK_WARMUP if quick else BENCHMARK_WARMUP
    print(f"Benchmarking {n_requests} scoring requests...", flush=True)
    latency = run_latency_benchmark(
        model_path=model_path,
        meta_path=meta_path,
        payload=payload,
        n_requests=n_requests,
        warmup=warmup,
        log_path=output_dir / "benchmark_server.log",
    )

    coverage_percent = 0.0
    if (not quick) and measure_coverage:
        print("Measuring test coverage...", flush=True)
        coverage_percent = _measure_coverage(output_dir)

    document = _document(
        quality=quality,
        frame=frame,
        test_auroc=test_auroc,
        base_auroc=base_auroc,
        test_ece=test_ece,
        base_ece=base_ece,
        trained=trained,
        y_test=y_test,
        effects=effects,
        latency=latency,
        coverage_percent=coverage_percent,
        hardware=hardware_info(device),
        runtime_seconds=time.perf_counter() - started,
        quick=quick,
    )
    write_json(output_dir / "metrics.json", document)
    write_causal_report(output_dir / "causal_analysis.md", document)
    if not quick:
        update_readme(ROOT / "README.md", document)
        write_resume_notes(ROOT / "RESUME_NOTES.md", document)
    else:
        write_resume_notes(output_dir / "RESUME_NOTES.md", document)
    print(f"Wrote {output_dir / 'metrics.json'}", flush=True)
    return document


def _document(
    *,
    quality: dict,
    frame: pd.DataFrame,
    test_auroc: float,
    base_auroc: float,
    test_ece: float,
    base_ece: float,
    trained: dict,
    y_test: np.ndarray,
    effects: dict,
    latency: dict,
    coverage_percent: float,
    hardware: dict,
    runtime_seconds: float,
    quick: bool,
) -> dict:
    lift = float(test_auroc - base_auroc)
    criteria = [
        criterion("Test AUROC (30-day readmission)", test_auroc, ">=", 0.66),
        criterion("Test AUROC (30-day readmission) lift over baseline", lift, ">=", 0.01),
        criterion("Expected calibration error", test_ece, "<=", 0.03),
        criterion("Covariate balance after weighting", effects["max_smd_weighted"], "<", 0.1),
        criterion("Scoring API p95 latency", latency["p95_ms"], "<", 100),
        criterion("Test coverage of src/", coverage_percent if not quick else 0.0, ">=", 75),
    ]
    # Quick runs do not execute the coverage suite. Leave that criterion failed
    # rather than copying a stale number.
    if quick:
        criteria[-1]["passed"] = False
        criteria[-1]["value"] = 0.0

    display = {
        "n_raw": int(quality["n_rows"]),
        "n_cohort": int(quality["n_cohort"]),
        "test_auroc": round(float(test_auroc), 3),
        "test_ece": round(float(test_ece), 3),
        "baseline_auroc": round(float(base_auroc), 3),
        "p95_ms": round(float(latency["p95_ms"]), 1),
        "n_requests": int(latency["n_requests"]),
        "ipw_pp": round(float(effects["ipw"]) * 100, 2),
        "aipw_pp": round(float(effects["aipw"]) * 100, 2),
        "aipw_ci_low_pp": round(float(effects["aipw_ci"][0]) * 100, 2),
        "aipw_ci_high_pp": round(float(effects["aipw_ci"][1]) * 100, 2),
        "max_smd": round(float(effects["max_smd_weighted"]), 3),
    }
    metrics = {
        "data": {
            "n_raw": int(quality["n_rows"]),
            "n_cohort": int(quality["n_cohort"]),
            "n_excluded_discharge": int(quality["n_excluded"]),
            "n_patients_raw": int(quality["n_patients"]),
            "cohort_readmit_30_rate": float(quality["readmit_30_rate"]),
            "cohort_med_change_rate": float(quality["med_change_rate"]),
            "quick_mode": bool(quick),
        },
        "test": {
            "lightgbm_calibrated": {
                "auroc": float(test_auroc),
                "ece": float(test_ece),
                "brier": brier_score(y_test, trained["p_test"]),
                "prevalence": float(np.mean(y_test)),
            },
            "logistic_regression": {
                "auroc": float(base_auroc),
                "ece": float(base_ece),
                "brier": brier_score(y_test, trained["p_test_baseline"]),
            },
        },
        "val": {
            "lightgbm_raw_auroc": float(trained["val_auroc_raw"]),
            "lightgbm_calibrated_auroc": float(auroc(frame.loc[frame["split"] == "val", "readmit_30"], trained["p_val"])),
        },
        "model": {
            "selected_params": trained["params"],
            "mlflow_model_name": "readmission_30day",
        },
        "causal": {
            "estimand": "encounter-level ATE of diabetes medication change on 30-day readmission",
            "n": effects["n"],
            "n_treated": effects["n_treated"],
            "ipw_risk_difference": float(effects["ipw"]),
            "aipw_risk_difference": float(effects["aipw"]),
            "naive_risk_difference": float(effects["naive"]),
            "ipw_ci95": [float(effects["ipw_ci"][0]), float(effects["ipw_ci"][1])],
            "aipw_ci95": [float(effects["aipw_ci"][0]), float(effects["aipw_ci"][1])],
            "max_smd_weighted": float(effects["max_smd_weighted"]),
            "max_smd_unweighted": float(effects["max_smd_unweighted"]),
            "propensity_min": float(effects["propensity_min"]),
            "propensity_max": float(effects["propensity_max"]),
            "overlap_frac": float(effects["overlap_frac"]),
            "clip": float(effects["clip"]),
            "c_value": float(effects["c_value"]),
            "used_squares": bool(effects["used_squares"]),
            "n_bootstrap": int(QUICK_BOOTSTRAPS if quick else FULL_BOOTSTRAPS),
        },
        "service": {
            "p50_ms": float(latency["p50_ms"]),
            "p95_ms": float(latency["p95_ms"]),
            "max_ms": float(latency["max_ms"]),
            "mean_ms": float(latency["mean_ms"]),
            "n_requests": int(latency["n_requests"]),
        },
        "coverage": {"line_percent": float(coverage_percent if not quick else 0.0)},
        "display": display,
    }
    baseline = {
        "model": "l2_logistic_regression",
        "test_auroc": float(base_auroc),
        "test_ece": float(base_ece),
        "causal_naive_risk_difference": float(effects["naive"]),
    }
    splits = {
        "encounters": _counts(frame, "encounter_id"),
        "patients": _counts(frame.drop_duplicates("patient_nbr"), "patient_nbr"),
    }
    return build_document(
        criteria=criteria,
        metrics=metrics,
        baseline=baseline,
        splits=splits,
        hardware=hardware,
        runtime_seconds=runtime_seconds,
    )


def _counts(frame: pd.DataFrame, id_col: str) -> dict:
    del id_col
    return {
        "train": int((frame["split"] == "train").sum()),
        "val": int((frame["split"] == "val").sum()),
        "test": int((frame["split"] == "test").sum()),
    }


def _measure_coverage(output_dir: Path) -> float:
    cov_json = output_dir / "coverage.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src",
        f"--cov-report=json:{cov_json}",
        "--cov-report=term-missing",
        "-q",
    ]
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    log_path = output_dir / "pytest_coverage.log"
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if not cov_json.exists():
        raise RuntimeError(f"coverage report missing\n{completed.stdout}\n{completed.stderr}")
    payload = json.loads(cov_json.read_text(encoding="utf-8"))
    percent = float(payload["totals"]["percent_covered"])
    if completed.returncode != 0:
        raise RuntimeError(
            f"pytest failed while measuring coverage (line percent {percent})\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-2000:]}"
        )
    return percent
