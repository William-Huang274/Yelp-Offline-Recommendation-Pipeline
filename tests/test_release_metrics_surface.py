from __future__ import annotations

from pathlib import Path

from conftest import read_csv_rows, read_json


def test_stage10_current_mainline_snapshot_has_all_cohorts(repo_root: Path) -> None:
    summary_path = repo_root / "data" / "output" / "current_release" / "stage10" / "stage10_current_mainline_summary.json"
    payload = read_json(summary_path)
    rows = payload["bucket_snapshots"]
    buckets = {row["bucket"] for row in rows}
    assert buckets == {"bucket2", "bucket5", "bucket10"}
    for row in rows:
        assert row["learned_recall_at_10"] > row["prescore_recall_at_10"]


def test_stage11_current_snapshot_has_expected_experts_and_eval_lines(repo_root: Path) -> None:
    expert_summary_path = repo_root / "data" / "output" / "current_release" / "stage11" / "experts" / "expert_training_summary.json"
    expert_payload = read_json(expert_summary_path)
    experts = {row["expert_band"] for row in expert_payload["experts"]}
    assert experts == {"11-30", "31-60", "61-100"}

    expert_rows = {row["expert_band"]: row for row in expert_payload["experts"]}
    assert expert_rows["11-30"]["true_win_11_30"] > 0.95

    eval_rows = read_csv_rows(
        repo_root / "data" / "metrics" / "current_release" / "stage11" / "stage11_bucket5_eval_reference_lines.csv"
    )
    names = {row["line"] for row in eval_rows}
    assert {"two_band_best_known_v120", "tri_band_freeze_v124"} <= names
