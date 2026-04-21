from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from batch_infer_demo import DEFAULT_INPUT_PATH, SERVING_CONFIG_PATH, load_release_reference, load_serving_config, rank_payload, read_json
from mock_serving_api import health_payload
from run_stage01_11_minidemo import DEFAULT_INPUT_PATH as MINI_INPUT_PATH, run_minidemo


def test_batch_infer_demo_contract() -> None:
    payload = read_json(DEFAULT_INPUT_PATH)
    result = rank_payload(payload)

    assert result["service"] == "batch_infer_demo"
    assert result["summary_metrics"]["input_candidates"] == len(payload["candidates"])
    assert result["summary_metrics"]["stage11_rescued_candidates"] >= 1
    assert result["summary_metrics"]["stage11_rescued_into_top_k"] >= 1
    assert len(result["top_k"]) == payload["top_k"]
    assert result["top_k"][0]["rank"] == 1


def test_batch_infer_demo_strategy_switch_and_fallback() -> None:
    payload = read_json(DEFAULT_INPUT_PATH)
    baseline = rank_payload({**payload, "strategy": "baseline"})
    xgboost = rank_payload({**payload, "strategy": "xgboost"})
    fallback = rank_payload({**payload, "strategy": "reward_rerank", "simulate_failure_for": "reward_rerank"})

    assert baseline["strategy_used"] == "baseline"
    assert xgboost["strategy_used"] == "xgboost"
    assert fallback["strategy_requested"] == "reward_rerank"
    assert fallback["strategy_used"] == "xgboost"
    assert fallback["fallback_used"] is True
    assert fallback["serving_metrics"]["fallback_count"] == 1


def test_serving_config_contract() -> None:
    config = load_serving_config()

    assert SERVING_CONFIG_PATH.exists()
    assert config["default_strategy"] == "reward_rerank"
    assert {"baseline", "xgboost", "reward_rerank"} <= set(config["allowed_strategies"])
    assert config["latency_budget_ms"] > 0


def test_mock_serving_health_contract() -> None:
    payload = health_payload()
    release_ref = load_release_reference()

    assert payload["status"] == "ok"
    assert payload["service"] == "mock_serving_api"
    assert payload["release_id"] == release_ref["release_id"]
    assert payload["default_strategy"] == "reward_rerank"


def test_stage01_11_minidemo_contract() -> None:
    payload = read_json(MINI_INPUT_PATH)
    result = run_minidemo(payload)

    assert result["status"] == "pass"
    assert len(result["stages"]) == 11
    assert result["stages"][0]["stage"] == "stage01_ingest"
    assert result["stages"][-1]["stage"] == "stage11_reward_rerank"
    assert result["rank_result"]["strategy_used"] == "reward_rerank"
