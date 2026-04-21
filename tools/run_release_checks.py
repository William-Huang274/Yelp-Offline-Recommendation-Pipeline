#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

REQUIRED_DOCS = [
    REPO_ROOT / "docs" / "project" / "README.md",
    REPO_ROOT / "docs" / "project" / "README.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "guide_index.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "teacher_requirement_alignment.md",
    REPO_ROOT / "docs" / "project" / "environment_setup.md",
    REPO_ROOT / "docs" / "project" / "data_lineage_and_storage.md",
    REPO_ROOT / "docs" / "project" / "reproduce_mainline.md",
    REPO_ROOT / "docs" / "project" / "challenges_and_tradeoffs.md",
    REPO_ROOT / "docs" / "project" / "evaluation_and_casebook.md",
    REPO_ROOT / "docs" / "project" / "demo_runbook.md",
    REPO_ROOT / "docs" / "project" / "cloud_and_local_demo_runbook.zh-CN.md",
    REPO_ROOT / "docs" / "project" / "acceptance_checklist.md",
    REPO_ROOT / "docs" / "project" / "repo_navigation.md",
    REPO_ROOT / "docs" / "project" / "proposal_template_content.md",
    REPO_ROOT / "docs" / "project" / "final_report_outline.md",
]

PYTEST_TARGETS = [
    "tests/test_demo_tools.py",
    "tests/test_public_readme_links.py",
    "tests/test_public_release_surface.py",
    "tests/test_release_metrics_surface.py",
    "tests/test_launcher_surface.py",
]


def print_step_result(name: str, ok: bool, output: str) -> None:
    label = "PASS" if ok else "FAIL"
    print(f"{label} {name}")
    if output.strip():
        print(output.rstrip())


def run_command(name: str, command: list[str]) -> bool:
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = proc.stdout
    if proc.stderr.strip():
        if output.strip():
            output += "\n"
        output += proc.stderr
    ok = proc.returncode == 0
    print_step_result(name, ok, output)
    return ok


def check_required_docs() -> bool:
    missing = [str(path.relative_to(REPO_ROOT)) for path in REQUIRED_DOCS if not path.exists()]
    if missing:
        print_step_result("required_docs", False, "\n".join(f"- missing: {item}" for item in missing))
        return False
    print_step_result("required_docs", True, "\n".join(f"- {path.relative_to(REPO_ROOT)}" for path in REQUIRED_DOCS))
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reviewer-facing checks for the current frozen repository line.")
    parser.add_argument("--skip-pytest", action="store_true", help="skip the core pytest suite")
    parser.add_argument("--skip-demo", action="store_true", help="skip the demo CLI smoke step")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ok = True

    ok &= check_required_docs()
    ok &= run_command("public_surface", [PYTHON, "tools/validate_public_surface.py"])
    ok &= run_command("current_release", [PYTHON, "tools/validate_current_release.py"])

    if not args.skip_demo:
        ok &= run_command("demo_cli", [PYTHON, "tools/demo_recommend.py", "summary"])
        ok &= run_command("batch_infer_demo", [PYTHON, "tools/batch_infer_demo.py"])
        ok &= run_command("mock_serving_self_test", [PYTHON, "tools/mock_serving_api.py", "--self-test"])

    if not args.skip_pytest:
        ok &= run_command("pytest_core", [PYTHON, "-m", "pytest", "-q", *PYTEST_TARGETS])

    if ok:
        print("PASS release_checks")
        return 0

    print("FAIL release_checks")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
