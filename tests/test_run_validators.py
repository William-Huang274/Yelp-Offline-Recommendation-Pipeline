from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.run_validators import (
    missing_required_fields,
    validate_stage09_candidate_run,
    validate_stage11_dataset_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_missing_required_fields_flags_blank_and_empty_values() -> None:
    missing = missing_required_fields({"run_id": "x", "run_tag": "", "summary": []}, ("run_id", "run_tag", "summary"))
    assert missing == ["run_tag", "summary"]


def test_stage09_validator_accepts_valid_fixture(fake_stage09_run: Path) -> None:
    assert validate_stage09_candidate_run(fake_stage09_run) == []


def test_stage11_validator_accepts_valid_fixture(fake_stage11_dataset_run: Path) -> None:
    assert validate_stage11_dataset_run(fake_stage11_dataset_run) == []


def _require_local_artifact(run_dir: Path) -> None:
    if not run_dir.exists():
        pytest.skip(f"local frozen artifact not available: {run_dir}")


def test_stage09_validator_smoke_checks_frozen_local_run() -> None:
    run_dir = REPO_ROOT / "data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion"
    _require_local_artifact(run_dir)
    assert validate_stage09_candidate_run(run_dir) == []


def test_stage11_validator_smoke_checks_frozen_local_run() -> None:
    run_dir = REPO_ROOT / "data/output/11_qlora_data/20260311_011112_stage11_1_qlora_build_dataset"
    _require_local_artifact(run_dir)
    assert validate_stage11_dataset_run(run_dir) == []
