"""Reproduce the readmission analysis from scratch.

Examples
--------
python run.py
python run.py --quick
python run.py --device cpu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the readmission pipeline")
    parser.add_argument("--quick", action="store_true", help="Run a small subset for tests and CI")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Skip the pytest coverage measurement (full runs measure it by default)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    document = run_pipeline(
        quick=args.quick,
        device=args.device,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        measure_coverage=not args.skip_coverage,
    )
    for item in document["success_criteria"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']}: {item['value']} {item['op']} {item['threshold']}")
    print(f"runtime_seconds={document['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
