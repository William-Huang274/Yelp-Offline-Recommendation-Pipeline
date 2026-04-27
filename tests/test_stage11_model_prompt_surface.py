from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage11_model_prompt_smoke_tool(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tools/stage/run_stage11_model_prompt_smoke.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + ("\n" + result.stderr if result.stderr else "")
    assert "PASS stage11_model_prompt_smoke" in result.stdout


def test_stage11_model_prompt_case_contract(repo_root: Path) -> None:
    case_path = repo_root / "config" / "demo" / "stage11_model_prompt_smoke_case.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))

    assert payload["reward_model_mainline"]["base_model"] == "Qwen3.5-9B"
    assert "prompt_only_probe_surface" not in payload
