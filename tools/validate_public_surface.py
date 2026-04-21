#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


PUBLIC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.zh-CN.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "README.zh-CN.md",
    REPO_ROOT / "docs" / "architecture.md",
    REPO_ROOT / "docs" / "architecture.zh-CN.md",
    REPO_ROOT / "docs" / "evaluation.md",
    REPO_ROOT / "docs" / "evaluation.zh-CN.md",
    REPO_ROOT / "docs" / "serving_release.md",
    REPO_ROOT / "docs" / "serving_release.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "README.md",
    REPO_ROOT / "docs" / "project" / "README.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "current_frozen_line.md",
    REPO_ROOT / "docs" / "project" / "current_frozen_line.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "design_choices.md",
    REPO_ROOT / "docs" / "project" / "design_choices.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "repository_map.md",
    REPO_ROOT / "docs" / "project" / "repository_map.zh-CN.md",
    REPO_ROOT / "docs" / "contracts" / "launcher_env_conventions.md",
    REPO_ROOT / "docs" / "stage11" / "stage11_case_notes_20260409.md",
    REPO_ROOT / "config" / "README.md",
    REPO_ROOT / "data" / "output" / "current_release" / "manifest.json",
    REPO_ROOT / "data" / "output" / "current_release" / "stage10" / "stage10_current_mainline_summary.json",
    REPO_ROOT / "data" / "output" / "current_release" / "stage11" / "experts" / "bucket5_11_30_v101_run_meta.json",
    REPO_ROOT / "data" / "output" / "current_release" / "stage11" / "experts" / "expert_training_summary.json",
]

LINKED_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.zh-CN.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "README.zh-CN.md",
]


def assert_exists(path: Path) -> list[str]:
    return [] if path.exists() else [f"missing file: {path}"]


def assert_links(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for target in LINK_RE.findall(text):
        if not (path.parent / target).resolve().exists():
            errors.append(f"{path.name} unresolved link: {target}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in PUBLIC_FILES:
        errors.extend(assert_exists(path))
    for path in LINKED_FILES:
        if path.exists():
            errors.extend(assert_links(path))

    manifest_path = REPO_ROOT / "data" / "output" / "current_release" / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("current_output_surface", "")).strip() != "data/output/current_release":
            errors.append("current_release manifest has unexpected current_output_surface field")

    if errors:
        print("FAIL public_surface")
        for item in errors:
            print(f"- {item}")
        return 1

    print("PASS public_surface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
