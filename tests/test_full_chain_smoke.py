from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_full_chain_smoke_tool(repo_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "tools/release/run_full_chain_smoke.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + ("\n" + result.stderr if result.stderr else "")
    assert "PASS full_chain_smoke" in result.stdout
