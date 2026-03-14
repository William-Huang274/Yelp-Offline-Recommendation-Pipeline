from __future__ import annotations

import importlib
from pathlib import Path

import pipeline.project_paths as project_paths


def test_project_root_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BDA_PROJECT_ROOT", str(tmp_path))
    try:
        module = importlib.reload(project_paths)
        assert module.PROJECT_ROOT == tmp_path.resolve()
        assert module.project_path("data").as_posix().endswith("/data")
    finally:
        monkeypatch.delenv("BDA_PROJECT_ROOT", raising=False)
        importlib.reload(project_paths)


def test_env_or_project_path_normalizes_legacy_repo_root(monkeypatch) -> None:
    monkeypatch.setenv("GL10_TEST_PATH", r"D:/5006 BDA project/data/output/09_candidate_fusion")
    resolved = project_paths.env_or_project_path("GL10_TEST_PATH", "data/output/09_candidate_fusion")
    assert resolved.as_posix().endswith("/data/output/09_candidate_fusion")
