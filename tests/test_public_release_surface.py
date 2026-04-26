from __future__ import annotations

from pathlib import Path


def test_public_release_surface_has_expected_files(repo_root: Path) -> None:
    expected = [
        repo_root / "README.md",
        repo_root / "README.zh-CN.md",
        repo_root / "docs" / "README.md",
        repo_root / "docs" / "README.zh-CN.md",
        repo_root / "docs" / "architecture.md",
        repo_root / "docs" / "architecture.zh-CN.md",
        repo_root / "docs" / "evaluation.md",
        repo_root / "docs" / "evaluation.zh-CN.md",
        repo_root / "docs" / "recruiter_pitch.zh-CN.md",
        repo_root / "docs" / "release_notes.md",
        repo_root / "docs" / "eval_protocol.md",
        repo_root / "docs" / "badcase_taxonomy.md",
        repo_root / "docs" / "model_card.md",
        repo_root / "docs" / "serving_release.md",
        repo_root / "docs" / "serving_release.zh-CN.md",
        repo_root / "docs" / "project" / "README.md",
        repo_root / "docs" / "project" / "README.zh-CN.md",
        repo_root / "docs" / "project" / "current_frozen_line.md",
        repo_root / "docs" / "project" / "current_frozen_line.zh-CN.md",
        repo_root / "docs" / "project" / "design_choices.md",
        repo_root / "docs" / "project" / "design_choices.zh-CN.md",
        repo_root / "docs" / "project" / "repository_map.md",
        repo_root / "docs" / "project" / "repository_map.zh-CN.md",
        repo_root / "docs" / "contracts" / "launcher_env_conventions.md",
        repo_root / "docs" / "contracts" / "launcher_env_conventions.zh-CN.md",
        repo_root / "docs" / "stage11" / "stage11_31_60_only_and_segmented_fusion_20260408.md",
        repo_root / "docs" / "stage11" / "stage11_case_notes_20260409.md",
        repo_root / "config" / "serving.yaml",
        repo_root / "config" / "demo" / "replay_request_input.json",
        repo_root / "config" / "demo" / "batch_infer_demo_input.json",
        repo_root / "config" / "demo" / "full_chain_minimal_input.json",
        repo_root / "config" / "demo" / "stage11_model_prompt_smoke_case.json",
        repo_root / "tools" / "export_serving_validation_report.py",
        repo_root / "tools" / "load_test_mock_serving.py",
        repo_root / "tools" / "run_stage01_11_minidemo.py",
        repo_root / "tools" / "run_full_chain_smoke.py",
        repo_root / "tools" / "run_stage11_model_prompt_smoke.py",
        repo_root / "data" / "output" / "current_release" / "manifest.json",
        repo_root / "data" / "output" / "current_release" / "stage10" / "stage10_current_mainline_summary.json",
        repo_root / "data" / "output" / "current_release" / "stage11" / "experts" / "bucket5_11_30_v101_run_meta.json",
        repo_root / "data" / "output" / "current_release" / "stage11" / "experts" / "expert_training_summary.json",
        repo_root / "data" / "metrics" / "current_release" / "stage10" / "stage10_current_mainline_snapshot.csv",
        repo_root / "data" / "metrics" / "current_release" / "stage11" / "stage11_bucket5_eval_reference_lines.csv",
    ]
    missing = [path for path in expected if not path.exists()]
    assert not missing, f"missing public release files: {missing}"


def test_public_showcase_surface_has_expected_history_runs(repo_root: Path) -> None:
    expected = [
        repo_root / "data" / "output" / "showcase_history" / "stage11" / "v117_segmented_11_3" / "run_meta.json",
        repo_root / "data" / "output" / "showcase_history" / "stage11" / "v120_joint12_default" / "run_meta.json",
        repo_root / "data" / "output" / "showcase_history" / "stage11" / "v121_joint12_gate" / "run_meta.json",
        repo_root / "data" / "metrics" / "showcase_history" / "stage11" / "bucket5_v117_segmented_metrics.csv",
    ]
    missing = [path for path in expected if not path.exists()]
    assert not missing, f"missing showcase files: {missing}"
