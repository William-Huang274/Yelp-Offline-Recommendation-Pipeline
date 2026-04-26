#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def print_step(name: str, status: str, output: str = "") -> None:
    print(f"{status} {name}")
    if output.strip():
        print(output.rstrip())


def run_step(name: str, command: list[str], *, optional: bool = False, optional_hint: str = "") -> bool:
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

    if proc.returncode == 0:
        print_step(name, "PASS", output)
        return True

    if optional:
        suffix = output.strip()
        if optional_hint:
            suffix = f"{suffix}\n[HINT] {optional_hint}".strip()
        print_step(name, "WARN", suffix)
        return True

    print_step(name, "FAIL", output)
    return False


def powershell_command() -> list[str] | None:
    for candidate in ("pwsh", "powershell"):
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public stage01->stage11 smoke surface without requiring large local training artifacts."
    )
    parser.add_argument(
        "--strict-local-artifacts",
        action="store_true",
        help="fail when Stage09/Stage10 local wrapper prerequisites are missing",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ok = True
    stage_help_steps = [
        ("stage01_ingest_help", [PYTHON, "scripts/stage01_to_stage08/01_data prep.py", "--help"]),
        ("stage02_analysis_help", [PYTHON, "scripts/stage01_to_stage08/02_data_analysis.py", "--help"]),
        ("stage03_recsys_help", [PYTHON, "scripts/stage01_to_stage08/03_la_recsys.py", "--help"]),
        ("stage04_validate_help", [PYTHON, "scripts/stage01_to_stage08/04_la_recsys_valid.py", "--help"]),
        ("stage05_freeze_help", [PYTHON, "scripts/stage01_to_stage08/05_freeze_recsys_results.py", "--help"]),
        ("stage06_insight_help", [PYTHON, "scripts/stage01_to_stage08/06_insight_text.py", "--help"]),
        ("stage07_cluster_help", [PYTHON, "scripts/stage01_to_stage08/07_embedding_cluster.py", "--help"]),
        ("stage08_profile_merge_help", [PYTHON, "scripts/stage01_to_stage08/08_merge_cluster_profile.py", "--help"]),
    ]
    for name, command in stage_help_steps:
        ok &= run_step(name, command)

    ok &= run_step("stage01_11_minidemo", [PYTHON, "tools/run_stage01_11_minidemo.py"])

    pwsh = powershell_command()
    if pwsh is None:
        print_step("windows_local_wrappers", "WARN", "No PowerShell executable found; skipping stage09/stage10 local wrapper checks.")
    else:
        local_optional = not args.strict_local_artifacts
        local_hint = (
            "Large local Stage09/Stage10 artifacts are intentionally not part of the public clone; "
            "pull them from cloud or rerun with --strict-local-artifacts when validating a full local replay."
        )
        ok &= run_step(
            "stage09_local_check",
            [*pwsh, "tools/run_stage09_local.ps1", "-CheckOnly"],
            optional=local_optional,
            optional_hint=local_hint,
        )
        ok &= run_step(
            "stage10_bucket5_local_check",
            [*pwsh, "tools/run_stage10_bucket5_local.ps1", "-CheckOnly"],
            optional=local_optional,
            optional_hint=local_hint,
        )
        ok &= run_step(
            "stage10_bucket2_local_check",
            [*pwsh, "tools/run_stage10_bucket2_local.ps1", "-CheckOnly"],
            optional=True,
            optional_hint="Pull stage09_bucket2_sourceparity from cloud if you need full bucket2 local replay.",
        )

    ok &= run_step("stage11_model_prompt_smoke", [PYTHON, "tools/run_stage11_model_prompt_smoke.py"])
    ok &= run_step("stage11_demo_summary", [PYTHON, "tools/demo_recommend.py", "summary"])
    sample_request_id = "stage11_b5_u000097"
    ok &= run_step(
        "serving_replay_baseline",
        [PYTHON, "tools/batch_infer_demo.py", "--request-id", sample_request_id, "--strategy", "baseline"],
    )
    ok &= run_step(
        "serving_replay_xgboost_live",
        [
            PYTHON,
            "tools/batch_infer_demo.py",
            "--request-id",
            sample_request_id,
            "--strategy",
            "xgboost",
            "--stage09-mode",
            "lookup_live",
            "--stage10-mode",
            "xgb_live",
            "--stage11-mode",
            "replay",
        ],
    )
    ok &= run_step(
        "serving_replay_reward_cache_miss",
        [
            PYTHON,
            "tools/batch_infer_demo.py",
            "--request-id",
            sample_request_id,
            "--strategy",
            "reward_rerank",
            "--simulate-stage11-cache-miss",
        ],
    )
    ok &= run_step("stage11_mock_serving_self_test", [PYTHON, "tools/mock_serving_api.py", "--self-test"])
    ok &= run_step(
        "mock_serving_load_test",
        [PYTHON, "tools/load_test_mock_serving.py", "--requests", "6", "--concurrency", "2", "--simulate-fallback-every", "3"],
    )

    if ok:
        print("PASS full_chain_smoke")
        return 0
    print("FAIL full_chain_smoke")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
