from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_stage11_model_prompt_smoke_tool(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tools/run_stage11_model_prompt_smoke.py"],
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
    assert payload["prompt_only_probe_surface"]["probe_launchers"] == [
        {
            "path": "scripts/runtime_sh/run_stage11_persona_sft_v3_qwen35_35b_a3b_probe.sh",
            "model_marker": "Qwen3.5-35B-A3B-Base",
        },
        {
            "path": "scripts/runtime_sh/run_stage11_persona_sft_v3_qwen3_30b_a3b_probe.sh",
            "model_marker": "Qwen3-30B-A3B-Base",
        },
    ]
    assert payload["prompt_only_probe_surface"]["prompt_templates"]
