from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from batch_infer_demo import DEFAULT_INPUT_PATH, load_release_reference, rank_payload, read_json
from mock_serving_api import health_payload


def test_batch_infer_demo_contract() -> None:
    payload = read_json(DEFAULT_INPUT_PATH)
    result = rank_payload(payload)

    assert result["service"] == "batch_infer_demo"
    assert result["summary_metrics"]["input_candidates"] == len(payload["candidates"])
    assert result["summary_metrics"]["stage11_rescued_candidates"] >= 1
    assert result["summary_metrics"]["stage11_rescued_into_top_k"] >= 1
    assert len(result["top_k"]) == payload["top_k"]
    assert result["top_k"][0]["rank"] == 1


def test_mock_serving_health_contract() -> None:
    payload = health_payload()
    release_ref = load_release_reference()

    assert payload["status"] == "ok"
    assert payload["service"] == "mock_serving_api"
    assert payload["release_id"] == release_ref["release_id"]
