"""Download and verify the UCI diabetes encounter file."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import requests

from src.config import DATA_URLS, EXPECTED_RAW_ROWS, RAW_DIR

# SHA-256 of diabetic_data.csv inside the UCI zip (CC BY 4.0 extract).
KNOWN_CSV_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"


class DataQualityError(RuntimeError):
    """Raised when the raw extract fails a published integrity check."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _extract_diabetes_csv(payload: bytes, dest: Path) -> Path:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith("diabetic_data.csv")]
        if not names:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and "id" not in Path(name).name.lower()
            ]
        if not names:
            raise DataQualityError("zip did not contain diabetic_data.csv")
        member = names[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, dest.open("wb") as out:
            out.write(src.read())
        mapping = [name for name in archive.namelist() if name.lower().endswith("ids_mapping.csv")]
        if mapping:
            map_dest = dest.parent / "IDs_mapping.csv"
            with archive.open(mapping[0]) as src, map_dest.open("wb") as out:
                out.write(src.read())
    return dest


def download_dataset(dest_dir: Path | None = None, timeout: int = 120) -> Path:
    """Download the UCI zip if the CSV is not already cached.

    Integrity is the published row count (checked later in SQL) plus a
    SHA-256 of the cached CSV. The digest is computed on every run and
    compared with ``KNOWN_CSV_SHA256`` when that constant matches a real
    64-character hex digest. A placeholder constant is ignored so the first
    local run can record the digest in ``data/raw/diabetic_data.sha256``.
    """
    dest_dir = dest_dir or RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / "diabetic_data.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        _check_digest(csv_path)
        return csv_path

    last_error: Exception | None = None
    for url in DATA_URLS:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            _extract_diabetes_csv(response.content, csv_path)
            _check_digest(csv_path)
            return csv_path
        except (requests.RequestException, zipfile.BadZipFile, DataQualityError, OSError) as exc:
            last_error = exc
            if csv_path.exists():
                csv_path.unlink()
            continue
    raise DataQualityError(f"could not download the UCI diabetes dataset: {last_error}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


def _check_digest(csv_path: Path) -> str:
    digest = sha256_file(csv_path)
    sidecar = csv_path.with_suffix(".sha256")
    recorded = KNOWN_CSV_SHA256.strip().lower()
    if _is_sha256(recorded) and digest != recorded:
        raise DataQualityError(
            f"SHA-256 mismatch for {csv_path.name}: got {digest}, expected {recorded}"
        )
    if sidecar.exists():
        previous = sidecar.read_text(encoding="utf-8").strip().lower()
        if previous and previous != digest:
            raise DataQualityError(
                f"cached SHA-256 changed for {csv_path.name}: got {digest}, file had {previous}"
            )
    else:
        sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


def assert_row_count(n_rows: int, expected: int = EXPECTED_RAW_ROWS) -> None:
    if n_rows != expected:
        raise DataQualityError(f"expected {expected} encounters, found {n_rows}")
