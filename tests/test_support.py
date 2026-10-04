"""Latency helper and cached-download checks."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.benchmark import collect_latencies, summarize_latencies
from src.data import DataQualityError, _check_digest, _extract_diabetes_csv, assert_row_count, download_dataset
from src.hardware import hardware_info


def test_collect_latencies_counts_measured_calls_only() -> None:
    calls = {"n": 0}

    def send() -> None:
        calls["n"] += 1

    times = collect_latencies(send, n_requests=4, warmup=2)
    assert calls["n"] == 6
    assert len(times) == 4
    summary = summarize_latencies(times, 4)
    assert summary["n_requests"] == 4
    assert summary["p95_ms"] >= summary["p50_ms"]


def test_cached_csv_skips_the_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.data.KNOWN_CSV_SHA256", "")
    csv_path = tmp_path / "diabetic_data.csv"
    csv_path.write_text("encounter_id\n1\n", encoding="utf-8")
    assert download_dataset(tmp_path) == csv_path
    # Second call still hits the cache and accepts the sidecar written above.
    assert download_dataset(tmp_path) == csv_path


def test_sidecar_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.data.KNOWN_CSV_SHA256", "")
    csv_path = tmp_path / "diabetic_data.csv"
    csv_path.write_text("x", encoding="utf-8")
    csv_path.with_suffix(".sha256").write_text("ab" * 32 + "\n", encoding="utf-8")
    with pytest.raises(DataQualityError, match="SHA-256"):
        _check_digest(csv_path)


def test_known_digest_rejects_a_different_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "diabetic_data.csv"
    csv_path.write_text("not the uci file", encoding="utf-8")
    with pytest.raises(DataQualityError, match="SHA-256"):
        _check_digest(csv_path)


def test_row_count_guard() -> None:
    with pytest.raises(DataQualityError, match="101766"):
        assert_row_count(10)


def test_zip_extracts_the_encounter_file(tmp_path: Path) -> None:
    payload_dir = tmp_path / "pack"
    payload_dir.mkdir()
    dest = tmp_path / "out" / "diabetic_data.csv"
    buffer_path = tmp_path / "data.zip"
    with zipfile.ZipFile(buffer_path, "w") as archive:
        archive.writestr("nested/diabetic_data.csv", "encounter_id,patient_nbr\n1,2\n")
        archive.writestr("nested/IDs_mapping.csv", "id,desc\n1,home\n")
    _extract_diabetes_csv(buffer_path.read_bytes(), dest)
    assert "encounter_id" in dest.read_text(encoding="utf-8")
    assert (dest.parent / "IDs_mapping.csv").exists()


def test_hardware_report_has_the_expected_fields() -> None:
    info = hardware_info("cpu")
    assert info["gpu"] is None
    assert info["requested_device"] == "cpu"
    assert info["ram_gb"] > 0
    assert info["python"]
