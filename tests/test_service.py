"""Scoring API validation and a stubbed probability."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.service import ScoreRequest, create_app, score_feature_row


class _StubModel:
    def predict_proba(self, features):
        rows = np.asarray(features)
        probability = np.full(len(rows), 0.25)
        return np.column_stack([1.0 - probability, probability])


def test_score_row_rejects_missing_and_extra_fields() -> None:
    with pytest.raises(ValueError, match="feature mismatch"):
        score_feature_row(_StubModel(), ["age_mid", "gender_male"], {"age_mid": 70.0})


def test_request_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({"features": {"age_mid": float("inf")}})
    with pytest.raises(ValidationError):
        ScoreRequest.model_validate({"features": {}})


def test_health_and_score_round_trip() -> None:
    names = ["age_mid", "gender_male"]
    with TestClient(create_app(_StubModel(), names, model_version="stub-1")) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        response = client.post("/score", json={"features": {"age_mid": 70.0, "gender_male": 1.0}})
        assert response.status_code == 200
        body = response.json()
        assert body["readmission_probability"] == pytest.approx(0.25)
        assert body["model_version"] == "stub-1"
        bad = client.post("/score", json={"features": {"age_mid": 70.0}})
        assert bad.status_code == 422


def test_service_loads_a_saved_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "calibrated_lgbm.joblib"
    meta_path = tmp_path / "metadata.json"
    joblib.dump(_StubModel(), model_path)
    meta_path.write_text(
        json.dumps(
            {
                "feature_names": ["age_mid", "gender_male"],
                "model_version": "from-disk",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("READMISSION_MODEL_PATH", str(model_path))
    monkeypatch.setenv("READMISSION_META_PATH", str(meta_path))
    with TestClient(create_app()) as client:
        response = client.post(
            "/score",
            json={"features": {"age_mid": 60.0, "gender_male": 0.0}},
        )
    assert response.status_code == 200
    assert response.json()["model_version"] == "from-disk"
