from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for tool_dir in (REPO_ROOT / "tools" / "serving", REPO_ROOT / "tools" / "demo"):
    if str(tool_dir) not in sys.path:
        sys.path.insert(0, str(tool_dir))

from batch_infer_demo import DEFAULT_INPUT_PATH, SERVING_CONFIG_PATH, load_release_reference, load_serving_config, rank_payload, read_json
from export_serving_validation_report import build_report
from load_test_mock_serving import strip_to_replay_payload
from mock_serving_api import health_payload
from replay_store import load_replay_store
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


def test_batch_infer_demo_replay_contract() -> None:
    replay_store = load_replay_store()
    result = rank_payload(
        {
            "request_id": replay_store.sample_request_id,
            "strategy": "reward_rerank",
            "debug": True,
        }
    )

    assert result["mode"] == "replay_request"
    assert result["request_id"] == replay_store.sample_request_id
    assert result["request_context"]["bucket"] == "bucket5"
    assert result["request_context"]["top_city"]
    assert result["summary_metrics"]["replay_window_candidates"] == 100
    assert result["stage11_audit"]["enabled"] is True
    assert result["stage11_policy"]["mode"] == "cache_first_bounded_rescue"
    assert result["stage11_policy"]["cache_status"] == "hit"
    assert result["stage11_policy"]["applied_to_serving"] is True
    assert result["stage11_audit"]["rescued_candidates"] >= 1
    assert result["user_profile"]["user_id"]
    assert result["user_profile"]["n_train"] is not None
    assert result["user_profile"]["profile_confidence"] is not None
    assert len(result["user_profile"]["preferred_cuisines"]) >= 1
    assert result["user_state_snapshot"]["top_city"]
    assert result["user_state_snapshot"]["long_term_pref_text"]
    assert len(result["user_state_snapshot"]["positive_tags"]) >= 1
    assert result["offline_truth_audit"]["available"] is True
    assert result["offline_truth_audit"]["served_online"] is False
    assert result["offline_truth_audit"]["truth_item"]["stage10_rank"] is not None
    assert result["fallback_demo"]["available"] is True
    assert result["fallback_demo"]["fallback_used"] is True
    assert result["fallback_demo"]["strategy_used"] == "xgboost"
    assert len(result["top_k"]) == 5
    assert result["top_k"][0]["rank"] == 1
    assert result["top_k"][0]["why_recommended"]["summary"]
    assert result["stage11_live"]["status"] == "skipped"
    assert result["serving_metrics"]["latency_ms"] <= result["serving_metrics"]["wall_latency_ms"]


def test_batch_infer_demo_replay_dry_run_live_contract() -> None:
    replay_store = load_replay_store()
    result = rank_payload(
        {
            "request_id": replay_store.sample_request_id,
            "strategy": "reward_rerank",
            "stage11_live_mode": "remote_dry_run",
        }
    )

    assert result["mode"] == "replay_request"
    assert result["stage11_live"]["status"] == "dry_run"
    assert result["stage11_live"]["mode"] == "remote_dry_run"
    assert result["stage11_live"]["remote_payload"]["request_id"] == replay_store.sample_request_id


def test_batch_infer_demo_replay_xgb_live_contract() -> None:
    replay_store = load_replay_store()
    result = rank_payload(
        {
            "request_id": replay_store.sample_request_id,
            "strategy": "xgboost",
            "stage09_mode": "lookup_live",
            "stage10_mode": "xgb_live",
            "stage11_mode": "replay",
        }
    )

    assert result["mode"] == "replay_request"
    assert result["execution_modes"]["stage09_used"] == "lookup_live"
    assert result["execution_modes"]["stage10_used"] == "xgb_live"
    assert result["stage09_trace"]["mode"] == "lookup_live"
    assert result["stage10_summary"]["mode"] == "xgb_live"
    assert result["summary_metrics"]["input_candidates"] >= 100
    assert result["summary_metrics"]["stage10_live_window_size"] >= 100
    assert result["latency_breakdown_ms"]["stage10"] > 0
    assert result["offline_truth_audit"]["requested_strategy_live_probe"]["available"] is True
    assert len(result["top_k"]) == 5


def test_batch_infer_demo_stage11_cache_miss_falls_back_to_stage10() -> None:
    replay_store = load_replay_store()
    result = rank_payload(
        {
            "request_id": replay_store.sample_request_id,
            "strategy": "reward_rerank",
            "simulate_stage11_cache_miss": True,
        }
    )

    assert result["stage11_policy"]["cache_status"] == "miss_simulated"
    assert result["stage11_policy"]["applied_to_serving"] is False
    assert result["stage11_policy"]["backfill_event"]["enqueued"] is True
    assert result["execution_modes"]["stage11_used"] == "stage10_fallback"
    assert result["summary_metrics"]["stage11_rescued_into_top_k"] == 0
    assert result["top_k"][0]["why_recommended"]["ranking_reason"].startswith("stage10 backbone")


def test_load_test_replay_payload_strips_handwritten_candidates() -> None:
    payload = {
        "top_k": 5,
        "stage09_mode": "lookup_live",
        "stage10_mode": "xgb_live",
        "stage11_mode": "replay",
        "candidates": [{"business_id": "b1"}],
    }

    replay_payload = strip_to_replay_payload(payload)

    assert replay_payload["top_k"] == 5
    assert replay_payload["stage09_mode"] == "lookup_live"
    assert replay_payload["stage10_mode"] == "xgb_live"
    assert replay_payload["stage11_mode"] == "replay"
    assert "candidates" not in replay_payload


def test_serving_validation_report_contains_stage_scope() -> None:
    summary = {
        "mode": "in_process",
        "requests": 2,
        "warmup_requests": 1,
        "concurrency": 1,
        "unique_request_ids": 2,
        "traffic_profile": "mixed",
        "strategy_requested": "reward_rerank",
        "success_rate": 1.0,
        "fallback_rate": 0.5,
        "cache_hit_rate": 0.5,
        "cache_miss_count": 1,
        "backfill_count": 1,
        "strategy_requested_counts": {"reward_rerank": 2},
        "strategy_used_counts": {"reward_rerank": 1, "xgboost": 1},
        "cache_status_counts": {"hit": 1, "miss_simulated": 1},
        "latency_ms": {"p50": 80.0, "p95": 90.0, "p99": 95.0, "max": 100.0},
        "wall_latency_ms": {"p50": 81.0, "p95": 91.0, "p99": 96.0, "max": 101.0},
        "audit_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
        "stage_latency_ms": {
            "request_lookup": {"p50": 1.0, "p95": 2.0, "p99": 3.0, "avg": 1.5},
            "stage09": {"p50": 2.0, "p95": 3.0, "p99": 4.0, "avg": 2.5},
            "stage10": {"p50": 30.0, "p95": 40.0, "p99": 45.0, "avg": 32.0},
            "stage11": {"p50": 3.0, "p95": 4.0, "p99": 5.0, "avg": 3.5},
            "offline_truth_audit": {"p50": 1.0, "p95": 2.0, "p99": 3.0, "avg": 1.5},
            "fallback_demo": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0},
        },
        "audit_breakdown_ms": {
            "stage11_live": {"p50": 0.0, "p95": 0.0, "avg": 0.0},
            "fallback_demo": {"p50": 0.0, "p95": 0.0, "avg": 0.0},
        },
        "stage_mode_counts": {
            "stage09": {"lookup_live": 2},
            "stage10": {"xgb_live": 2},
            "stage11": {"cache_first": 1, "stage10_fallback": 1},
        },
        "source_alignment_counts": {
            "stage09": {"fallback_local_candidate_pack": 2},
            "stage10": {"fallback_local_candidate_pack": 2},
        },
    }

    report = build_report(
        summary,
        input_path=REPO_ROOT / "summary.json",
        serving_p95_budget_ms=250.0,
        serving_p99_budget_ms=300.0,
        success_rate_floor=0.99,
    )

    assert "Stage09 modes: `lookup_live=2`" in report
    assert "Stage10 modes: `xgb_live=2`" in report
    assert "Stage11 modes: `cache_first=1, stage10_fallback=1`" in report
    assert "serving_latency_p95" in report


def test_serving_config_contract() -> None:
    config = load_serving_config()

    assert SERVING_CONFIG_PATH.exists()
    assert config["default_strategy"] == "reward_rerank"
    assert {"baseline", "xgboost", "reward_rerank"} <= set(config["allowed_strategies"])
    assert config["latency_budget_ms"] > 0
    assert config["stage09_default_mode"] == "replay"
    assert config["stage10_default_mode"] == "replay"
    assert config["stage11_default_mode"] == "replay"
    assert config["stage11_live_default_mode"] == "off"


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
