from __future__ import annotations

from pathlib import Path

import pipeline.project_paths as project_paths


def test_latest_pointer_roundtrip(tmp_path: Path, monkeypatch) -> None:
    latest_dir = tmp_path / "_latest_runs"
    prod_dir = tmp_path / "_prod_runs"
    run_dir = tmp_path / "20260313_stage10_eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(project_paths, "LATEST_RUN_DIR", latest_dir)
    monkeypatch.setattr(project_paths, "PROD_RUN_DIR", prod_dir)

    pointer_path = project_paths.write_latest_run_pointer("stage10_eval", run_dir, {"stage": 10})
    assert pointer_path.exists()
    assert project_paths.read_latest_run_pointer("stage10_eval")["stage"] == 10
    assert project_paths.resolve_latest_run_pointer("stage10_eval") == run_dir


def test_production_pointer_roundtrip(tmp_path: Path, monkeypatch) -> None:
    latest_dir = tmp_path / "_latest_runs"
    prod_dir = tmp_path / "_prod_runs"
    run_dir = tmp_path / "20260313_stage11_eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(project_paths, "LATEST_RUN_DIR", latest_dir)
    monkeypatch.setattr(project_paths, "PROD_RUN_DIR", prod_dir)

    pointer_path = project_paths.write_production_run_pointer("stage11_release", run_dir, {"stage": 11})
    assert pointer_path.exists()
    assert project_paths.read_production_run_pointer("stage11_release")["stage"] == 11
    assert project_paths.resolve_production_run_pointer("stage11_release") == run_dir
