from __future__ import annotations

from pathlib import Path


def test_public_launcher_surface_has_current_entrypoints(repo_root: Path) -> None:
    launchers = [
        "stage09_bucket5_mainline.sh",
        "stage09_bucket5_typed_intent_assets.sh",
        "stage09_bucket5_stage11_assets.sh",
        "stage10_bucket2_mainline.sh",
        "stage10_bucket5_mainline.sh",
        "stage10_bucket10_mainline.sh",
        "stage11_bucket5_11_1.sh",
        "stage11_bucket5_export_only.sh",
        "stage11_bucket5_train.sh",
        "stage11_bucket5_eval.sh",
        "stage11_bucket5_watch.sh",
    ]
    missing = [
        repo_root / "scripts" / "launchers" / name
        for name in launchers
        if not (repo_root / "scripts" / "launchers" / name).exists()
    ]
    assert not missing, f"missing launcher wrappers: {missing}"
    assert (repo_root / "scripts" / "launchers" / "_path_contract.sh").exists()


def test_root_stage_scripts_still_exist_for_runtime_compatibility(repo_root: Path) -> None:
    expected = [
        repo_root / "scripts" / "run_stage09_bucket5_structural_v5_sourceparity.sh",
        repo_root / "scripts" / "run_stage10_bucket5_structural_v5_joint_min_cls_typed_intent_phase3_slicefix.sh",
        repo_root / "scripts" / "run_stage11_bucket5_top250_semantic_compact_rm_train.sh",
        repo_root / "scripts" / "run_stage11_bucket5_top250_semantic_compact_rm_eval_only.sh",
    ]
    missing = [path for path in expected if not path.exists()]
    assert not missing, f"missing runtime-compatible stage scripts: {missing}"
