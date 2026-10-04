"""Write metrics.json, the causal report, resume notes, and the README table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import PROJECT_SLUG, SEEDS

RESULTS_START = "<!-- RESULTS_TABLE_START -->"
RESULTS_END = "<!-- RESULTS_TABLE_END -->"


def criterion(name: str, value: float, op: str, threshold: float, threshold_high: float | None = None) -> dict:
    return {
        "name": name,
        "value": _num(value),
        "op": op,
        "threshold": _num(threshold),
        "threshold_high": None if threshold_high is None else _num(threshold_high),
        "passed": _passes(value, op, threshold, threshold_high),
    }


def _passes(value: float, op: str, threshold: float, threshold_high: float | None) -> bool:
    if value is None or not np.isfinite(value):
        return False
    if op == ">=":
        return bool(value >= threshold)
    if op == ">":
        return bool(value > threshold)
    if op == "<=":
        return bool(value <= threshold)
    if op == "<":
        return bool(value < threshold)
    if op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high")
        return bool(threshold <= value <= threshold_high)
    raise ValueError(f"unknown op {op}")


def _num(value: float) -> float:
    return float(value)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def build_document(
    *,
    criteria: list[dict],
    metrics: dict,
    baseline: dict,
    splits: dict,
    hardware: dict,
    runtime_seconds: float,
) -> dict:
    return {
        "project": PROJECT_SLUG,
        "success_criteria": criteria,
        "metrics": json_ready(metrics),
        "baseline": json_ready(baseline),
        "splits": json_ready(splits),
        "seeds": list(SEEDS),
        "hardware": json_ready(hardware),
        "runtime_seconds": float(runtime_seconds),
    }


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(document), indent=2) + "\n", encoding="utf-8")


def render_results_section(document: dict) -> str:
    lines = [
        "| Criterion | Value | Rule | Passed |",
        "|---|---:|---|---|",
    ]
    failed = []
    for item in document["success_criteria"]:
        high = item.get("threshold_high")
        if item["op"] == "between" and high is not None:
            rule = f"{item['threshold']} to {high}"
        else:
            rule = f"{item['op']} {item['threshold']}"
        lines.append(
            f"| {item['name']} | {item['value']} | {rule} | {'yes' if item['passed'] else 'no'} |"
        )
        if not item["passed"]:
            failed.append(item["name"])
    metrics = document["metrics"]
    test = metrics["test"]["lightgbm_calibrated"]
    base = document["baseline"]
    causal = metrics["causal"]
    lines.extend(
        [
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Test AUROC, calibrated LightGBM | {test['auroc']} |",
            f"| Test AUROC, L2 logistic baseline | {base['test_auroc']} |",
            f"| Test ECE, calibrated LightGBM | {test['ece']} |",
            f"| AIPW risk difference | {causal['aipw_risk_difference']} |",
            f"| IPW risk difference | {causal['ipw_risk_difference']} |",
            f"| Naive risk difference | {causal['naive_risk_difference']} |",
            f"| Max weighted absolute SMD | {causal['max_smd_weighted']} |",
            f"| Scoring p95 latency (ms) | {metrics['service']['p95_ms']} |",
            f"| Runtime (seconds) | {document['runtime_seconds']} |",
            "",
        ]
    )
    if failed:
        lines.append("Failed criteria: " + ", ".join(failed) + ".")
    else:
        lines.append("All success criteria passed on this run.")
    lines.append("")
    lines.append("Plots: `results/calibration_curve.png`, `results/shap_summary.png`, `results/propensity_overlap.png`, `results/smd_balance.png`.")
    return "\n".join(lines)


def update_readme(readme_path: Path, document: dict) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        raise RuntimeError("README.md is missing results markers")
    before, rest = text.split(RESULTS_START, 1)
    _, after = rest.split(RESULTS_END, 1)
    updated = before + RESULTS_START + "\n" + render_results_section(document) + "\n" + RESULTS_END + after
    readme_path.write_text(updated, encoding="utf-8")


def write_resume_notes(path: Path, document: dict) -> None:
    display = document["metrics"]["display"]
    summary = (
        "This project trains a 30-day readmission classifier on the UCI Diabetes 130-US Hospitals "
        f"encounters ({display['n_raw']} rows) and serves the calibrated scores from a local FastAPI app. "
        "It also estimates the observational association between an in-hospital diabetes medication "
        "change and readmission with propensity-score weighting."
    )
    bullets = [
        (
            f"Built a 30-day readmission model on {display['n_cohort']} at-risk encounters from "
            f"{display['n_raw']} inpatient records, with SQL features in DuckDB and calibrated LightGBM, "
            f"reaching {display['test_auroc']} AUROC on held-out patients ({display['test_ece']} ECE)."
        ),
        (
            f"Served the model through a FastAPI endpoint at {display['p95_ms']} ms p95 latency "
            f"over {display['n_requests']} local single-record requests, with the model version stored "
            "in an MLflow registry."
        ),
        (
            f"Estimated the effect of medication changes on readmission with IPW "
            f"({display['ipw_pp']} pts) and AIPW ({display['aipw_pp']} pts, 95% CI "
            f"{display['aipw_ci_low_pp']} to {display['aipw_ci_high_pp']}), checking overlap and "
            f"covariate balance (max SMD {display['max_smd']})."
        ),
        (
            f"Compared the boosted model with an L2 logistic regression baseline "
            f"({display['baseline_auroc']} test AUROC) and kept every test patient out of fitting, "
            f"calibration, and model selection."
        ),
    ]
    skills = (
        "Python, SQL, DuckDB, pandas, scikit-learn, LightGBM, MLflow, FastAPI, "
        "Docker, SHAP, pytest, propensity scores, IPW, AIPW"
    )
    description = (
        "30-day hospital readmission scoring with DuckDB features, calibrated LightGBM, "
        "a FastAPI service, and a propensity-score effect estimate."
    )
    text = "\n".join(
        [
            "# Resume notes",
            "",
            "## Draft bullets",
            "",
            *(f"- {bullet}" for bullet in bullets),
            "",
            "## Skills and tools",
            "",
            skills,
            "",
            "## Project summary",
            "",
            summary,
            "",
            "## GitHub description",
            "",
            description,
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def write_causal_report(path: Path, document: dict) -> None:
    causal = document["metrics"]["causal"]
    lines = [
        "# Medication change and 30-day readmission",
        "",
        "This is an observational analysis. It is not evidence that changing a diabetes "
        "medication causes a change in readmission. The estimate is an association that "
        "would equal an average treatment effect only if the assumptions below hold.",
        "",
        "## Estimand",
        "",
        "The treatment is the dataset field `change` (Ch versus No): whether any diabetes "
        "medication dose changed during the encounter. The outcome is readmission within 30 days. "
        "Each encounter is one row. Patients in the test split are not used here, so the same "
        "held-out patients stay untouched by both the predictive model and this analysis.",
        "",
        "## Estimates",
        "",
        f"- Naive risk difference (treated minus untreated): {causal['naive_risk_difference']}",
        f"- IPW (Hajek) risk difference: {causal['ipw_risk_difference']}",
        f"- IPW 95% patient-cluster bootstrap interval: {causal['ipw_ci95']}",
        f"- AIPW risk difference: {causal['aipw_risk_difference']}",
        f"- AIPW 95% patient-cluster bootstrap interval: {causal['aipw_ci95']}",
        f"- Analysis encounters: {causal['n']}",
        f"- Treated encounters: {causal['n_treated']}",
        f"- Propensity range before clipping: {causal['propensity_min']} to {causal['propensity_max']}",
        f"- Share of propensities inside [0.1, 0.9]: {causal['overlap_frac']}",
        f"- Propensity clipping used for weights: {causal['clip']}",
        f"- Logistic penalty C chosen for balance: {causal['c_value']}",
        f"- Square terms added to the propensity model: {causal['used_squares']}",
        f"- Max absolute SMD before weighting: {causal['max_smd_unweighted']}",
        f"- Max absolute SMD after weighting: {causal['max_smd_weighted']}",
        "",
        "Positive values mean a higher readmission rate in the medication-change group. "
        "Intervals are percentile intervals from a patient-cluster bootstrap of the fixed "
        "IPW and AIPW scores. They do not include uncertainty from choosing the propensity specification.",
        "",
        "## Balance and overlap",
        "",
        "Standardized mean differences use inverse-probability weights and the unweighted "
        "pooled standard deviation (Austin's definition). The success rule is that every "
        "pre-specified covariate has absolute weighted SMD below 0.1. The penalty and the "
        "propensity clip are chosen to reduce that maximum. The outcome is not used in that choice. "
        "The covariate table is `results/smd_table.csv`. Overlap is plotted in "
        "`results/propensity_overlap.png`.",
        "",
        "## Assumptions",
        "",
        "1. Conditional exchangeability: given the pre-treatment covariates, medication change "
        "is independent of the potential readmission outcomes. Unmeasured severity can violate this.",
        "2. Positivity: both decisions have non-zero probability at every covariate pattern we keep. "
        "Extreme scores are clipped, which changes the estimand slightly toward the trimmed population.",
        "3. Consistency: the recorded change flag matches the treatment contrast we describe.",
        "4. No interference across encounters, and the encounter order in `encounter_id` is a usable time order.",
        "5. Discharge disposition, length of stay, and in-hospital medication counts are excluded "
        "from the covariate set because the medication decision can affect them.",
        "",
        "A1C result group is included as a severity marker. The lab can be drawn during the stay, "
        "so it is not a pure pre-treatment covariate. That is a limitation.",
        "",
        "## What this does not say",
        "",
        "Do not read the point estimate as proof that a medication-change program would reduce "
        "or increase readmissions. A randomized or stepped-wedge evaluation would be required "
        "for that claim. The predictive model is a separate tool and was not used to choose this specification.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
