"""FastAPI scoring service.

The process loads the calibrated model once. Each request is one engineered
feature row. Feature engineering stays in the DuckDB views; the service checks
that the payload matches the training columns and returns a probability.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import ROOT

DEFAULT_MODEL_PATH = ROOT / "models" / "calibrated_lgbm.joblib"
DEFAULT_META_PATH = ROOT / "models" / "metadata.json"


class ScoreRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Engineered feature values")

    @field_validator("features")
    @classmethod
    def finite_values(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("features must not be empty")
        for key, item in value.items():
            if not math.isfinite(float(item)):
                raise ValueError(f"non-finite value for {key}")
        return value


class ScoreResponse(BaseModel):
    readmission_probability: float
    model_version: str


def score_feature_row(model: object, feature_names: list[str], payload: dict[str, float]) -> float:
    missing = [name for name in feature_names if name not in payload]
    extra = [name for name in payload if name not in feature_names]
    if missing or extra:
        raise ValueError(
            "feature mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}, expected={len(feature_names)}"
        )
    row = np.array([[float(payload[name]) for name in feature_names]], dtype=float)
    probability = float(model.predict_proba(row)[0, 1])
    if not math.isfinite(probability):
        raise ValueError("model returned a non-finite probability")
    return probability


def load_saved_model(model_path: Path | None = None, meta_path: Path | None = None) -> dict:
    model_path = model_path or DEFAULT_MODEL_PATH
    meta_path = meta_path or model_path.with_name("metadata.json")
    if not model_path.exists():
        raise FileNotFoundError(f"trained model not found at {model_path}")
    model = joblib.load(model_path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"feature_names": list(model.feature_names), "model_version": "local"}
    return {
        "model": model,
        "feature_names": list(meta["feature_names"]),
        "model_version": str(meta.get("model_version", "local")),
    }


def _resolve_model_paths(
    model_path: Path | None,
    meta_path: Path | None,
) -> tuple[Path | None, Path | None]:
    import os

    if model_path is None and os.environ.get("READMISSION_MODEL_PATH"):
        model_path = Path(os.environ["READMISSION_MODEL_PATH"])
    if meta_path is None and os.environ.get("READMISSION_META_PATH"):
        meta_path = Path(os.environ["READMISSION_META_PATH"])
    return model_path, meta_path


def create_app(
    model: object | None = None,
    feature_names: list[str] | None = None,
    model_version: str = "test",
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if model is not None:
            app.state.model = model
            app.state.feature_names = list(feature_names or [])
            app.state.model_version = model_version
        else:
            resolved_model, resolved_meta = _resolve_model_paths(model_path, meta_path)
            loaded = load_saved_model(resolved_model, resolved_meta)
            app.state.model = loaded["model"]
            app.state.feature_names = loaded["feature_names"]
            app.state.model_version = loaded["model_version"]
        yield

    app = FastAPI(title="30-day readmission risk", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        loaded = getattr(app.state, "model", None) is not None
        return {"status": "ok" if loaded else "loading", "model_loaded": loaded}

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        try:
            probability = score_feature_row(
                app.state.model,
                list(app.state.feature_names),
                request.features,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ScoreResponse(
            readmission_probability=probability,
            model_version=str(app.state.model_version),
        )

    return app


app = create_app()
