#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline.run_validators import (  # noqa: E402
    validate_stage09_candidate_run,
    validate_stage10_infer_eval_run,
    validate_stage10_rank_model_run,
    validate_stage11_dataset_run,
)


VALIDATORS = {
    "stage09_candidate": validate_stage09_candidate_run,
    "stage10_infer_eval": validate_stage10_infer_eval_run,
    "stage10_rank_model": validate_stage10_rank_model_run,
    "stage11_dataset": validate_stage11_dataset_run,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a local Stage09-Stage11 run directory.")
    parser.add_argument("--kind", choices=sorted(VALIDATORS), required=True, help="Artifact kind to validate.")
    parser.add_argument("--run-dir", required=True, help="Run directory to validate.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser()
    errors = VALIDATORS[args.kind](run_dir)
    if errors:
        print(f"FAIL {args.kind} {run_dir}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PASS {args.kind} {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
