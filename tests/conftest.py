from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _touch_dir_file(path: Path, filename: str = "part-00000.json") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text("{}", encoding="utf-8")


@pytest.fixture
def fake_stage09_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "20260313_000000_full_stage09_candidate_fusion"
    _write_json(
        run_dir / "run_meta.json",
        {
            "run_id": "20260313_000000",
            "run_tag": "stage09_candidate_fusion",
            "recall_profile": "coverage_stage2",
            "output_dir": str(run_dir),
        },
    )
    bucket_dir = run_dir / "bucket_10"
    _write_json(
        bucket_dir / "bucket_meta.json",
        {
            "n_users": 12,
            "n_test": 12,
            "pretrim_top_k_used": 150,
        },
    )
    for dirname in (
        "truth.parquet",
        "train_history.parquet",
        "candidates_all.parquet",
        "candidates_pretrim150.parquet",
    ):
        _touch_dir_file(bucket_dir / dirname)
    return run_dir


@pytest.fixture
def fake_stage11_dataset_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "20260313_000001_stage11_1_qlora_build_dataset"
    bucket_dir = run_dir / "bucket_10"
    _write_json(
        run_dir / "run_meta.json",
        {
            "run_id": "20260313_000001",
            "run_tag": "stage11_1_qlora_build_dataset",
            "source_stage09_run": "D:/5006_BDA_project/data/output/09_candidate_fusion/example",
            "summary": [
                {
                    "bucket": 10,
                    "candidate_file": "candidates_pretrim150.parquet",
                    "output_bucket_dir": "D:/5006_BDA_project/data/output/11_qlora_data/example/bucket_10",
                }
            ],
        },
    )
    for dirname in (
        "all_parquet",
        "train_json",
        "eval_json",
        "user_evidence_table",
        "item_evidence_table",
        "pair_evidence_audit",
    ):
        _touch_dir_file(bucket_dir / dirname)
    return run_dir
