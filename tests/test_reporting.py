"""Metrics document, resume notes, and the README results block."""

from __future__ import annotations

import json
from pathlib import Path

from src.reporting import (
    RESULTS_END,
    RESULTS_START,
    criterion,
    render_results_section,
    update_readme,
    write_causal_report,
    write_json,
    write_resume_notes,
)


def _document() -> dict:
    criteria = [
        criterion("Test AUROC (30-day readmission)", 0.671234, ">=", 0.66),
        criterion("Expected calibration error", 0.012345, "<=", 0.03),
    ]
    return {
        "project": "healthcare-readmission-mlops",
        "success_criteria": criteria,
        "metrics": {
            "test": {"lightgbm_calibrated": {"auroc": 0.671234, "ece": 0.012345}},
            "causal": {
                "aipw_risk_difference": -0.012,
                "ipw_risk_difference": -0.01,
                "naive_risk_difference": 0.02,
                "max_smd_weighted": 0.04,
            },
            "service": {"p95_ms": 8.2},
            "display": {
                "n_raw": 101766,
                "n_cohort": 99000,
                "test_auroc": 0.671,
                "test_ece": 0.012,
                "baseline_auroc": 0.64,
                "p95_ms": 8.2,
                "n_requests": 500,
                "ipw_pp": -1.0,
                "aipw_pp": -1.2,
                "aipw_ci_low_pp": -2.0,
                "aipw_ci_high_pp": -0.4,
                "max_smd": 0.04,
            },
        },
        "baseline": {"test_auroc": 0.64},
        "splits": {"encounters": {"train": 1, "val": 1, "test": 1}},
        "seeds": [0],
        "hardware": {"cpu": "test", "ram_gb": 8, "gpu": None, "python": "3.12", "os": "test"},
        "runtime_seconds": 1.5,
    }


def test_criterion_boundaries() -> None:
    assert criterion("Covariate balance after weighting", 0.099, "<", 0.1)["passed"] is True
    assert criterion("Covariate balance after weighting", 0.1, "<", 0.1)["passed"] is False
    assert criterion("Scoring API p95 latency", 100, "<", 100)["passed"] is False
    assert criterion("Expected calibration error", 0.03, "<=", 0.03)["passed"] is True


def test_resume_notes_use_display_numbers(tmp_path: Path) -> None:
    document = _document()
    path = tmp_path / "RESUME_NOTES.md"
    write_resume_notes(path, document)
    text = path.read_text(encoding="utf-8")
    assert "0.671" in text
    assert "0.012" in text
    assert "8.2" in text
    assert "-1.2" in text
    assert "0.04" in text
    assert "—" not in text


def test_readme_results_block_is_replaced(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{RESULTS_START}\nold\n{RESULTS_END}\noutro\n", encoding="utf-8")
    update_readme(readme, _document())
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text
    assert "Test AUROC (30-day readmission)" in text
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    assert "yes" in render_results_section(_document())


def test_causal_report_states_observational_limits(tmp_path: Path) -> None:
    document = _document()
    document["metrics"]["causal"].update(
        {
            "ipw_ci95": [-0.02, 0.01],
            "aipw_ci95": [-0.03, 0.0],
            "n": 100,
            "n_treated": 40,
            "propensity_min": 0.05,
            "propensity_max": 0.9,
            "overlap_frac": 0.8,
            "clip": 0.01,
            "c_value": 1.0,
            "used_squares": False,
            "max_smd_unweighted": 0.2,
        }
    )
    path = tmp_path / "causal_analysis.md"
    write_causal_report(path, document)
    text = path.read_text(encoding="utf-8")
    assert "observational" in text
    assert "not evidence" in text


def test_write_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    write_json(path, _document())
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["seeds"] == [0]
    assert loaded["success_criteria"][0]["passed"] is True
