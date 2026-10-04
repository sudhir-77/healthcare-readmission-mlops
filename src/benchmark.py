"""Local latency benchmark for the scoring service."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

from src.metrics import percentile


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def collect_latencies(send, n_requests: int, warmup: int) -> list[float]:
    if n_requests < 1:
        raise ValueError("n_requests must be positive")
    for _ in range(warmup):
        send()
    times: list[float] = []
    for _ in range(n_requests):
        start = time.perf_counter()
        send()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarize_latencies(times: list[float], n_requests: int) -> dict:
    arr = np.asarray(times, dtype=float)
    return {
        "p50_ms": percentile(arr, 50),
        "p95_ms": percentile(arr, 95),
        "max_ms": float(arr.max()),
        "mean_ms": float(arr.mean()),
        "n_requests": int(n_requests),
    }


def _wait_until_healthy(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            response = requests.get(base_url + "/health", timeout=1)
            if response.status_code == 200 and response.json().get("model_loaded"):
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"scoring service did not become healthy: {last_error}")


def run_latency_benchmark(
    *,
    model_path: Path,
    meta_path: Path,
    payload: dict,
    n_requests: int,
    warmup: int,
    log_path: Path,
) -> dict:
    """Start uvicorn, time single-record POSTs, then stop the server."""
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.service:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    # The service reads the model path from the environment so tests can point
    # uvicorn at a temporary file without editing the package default.
    env = _env_with_model(model_path, meta_path)
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(_project_root()),
    )
    try:
        _wait_until_healthy(base_url)
        session = requests.Session()

        def send() -> None:
            response = session.post(base_url + "/score", json={"features": payload}, timeout=5)
            response.raise_for_status()

        times = collect_latencies(send, n_requests, warmup)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        log_handle.close()
    return summarize_latencies(times, n_requests)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_with_model(model_path: Path, meta_path: Path) -> dict:
    import os

    env = os.environ.copy()
    env["READMISSION_MODEL_PATH"] = str(model_path)
    env["READMISSION_META_PATH"] = str(meta_path)
    return env
