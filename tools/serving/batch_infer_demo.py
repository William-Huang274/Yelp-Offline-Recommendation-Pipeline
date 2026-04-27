#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from replay_store import load_replay_store
from stage10_live_local import load_stage09_live_candidates, score_stage10_live
from stage11_live_remote import maybe_run_stage11_live_verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "config" / "demo" / "batch_infer_demo_input.json"
SERVING_CONFIG_PATH = REPO_ROOT / "config" / "serving.yaml"
CURRENT_RELEASE = REPO_ROOT / "data" / "output" / "current_release"

DEFAULT_SERVING_CONFIG: dict[str, Any] = {
    "release_id": "release_closeout_20260409",
    "release_surface": "data/output/current_release",
    "model_version": "stage09_route_v5__stage10_xgb_mainline__stage11_qwen35_9b_rm_v124",
    "default_strategy": "reward_rerank",
    "allowed_strategies": ["baseline", "xgboost", "reward_rerank"],
    "fallback_order": ["reward_rerank", "xgboost", "baseline"],
    "top_k": 5,
    "stage09_topn": 12,
    "stage11_topn": 30,
    "latency_budget_ms": 250,
    "stage09_default_mode": "replay",
    "stage10_default_mode": "replay",
    "stage11_default_mode": "replay",
    "champion_pointer": "stage11_release",
    "aligned_fallback_pointer": "stage10_release",
    "emergency_baseline_pointer": "stage09_release",
    "stage11_live_default_mode": "off",
    "stage11_live_topn": 100,
    "stage11_live_max_rivals": 11,
    "stage11_live_batch_size": 8,
    "stage11_live_max_seq_len": 1280,
    "stage11_live_timeout_s": 1800,
    "stage11_worker_port": 18080,
    "stage11_worker_startup_timeout_s": 300,
    "stage11_worker_poll_interval_s": 5.0,
    "stage11_worker_http_timeout_s": 180,
    "stage11_worker_ssh_timeout_s": 240,
    "stage11_policy_mode": "cache_first_bounded_rescue",
    "stage11_policy_latency_budget_ms": 50,
    "stage11_cache_ttl_hours": 24,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_config_scalar(value: str) -> Any:
    raw = value.strip()
    if not raw:
        return ""
    if raw.startswith("[") and raw.endswith("]"):
        body = raw[1:-1].strip()
        if not body:
            return []
        return [item.strip().strip("\"'") for item in body.split(",")]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip("\"'")


def load_serving_config(path: Path = SERVING_CONFIG_PATH) -> dict[str, Any]:
    config = {
        key: list(value) if isinstance(value, list) else value
        for key, value in DEFAULT_SERVING_CONFIG.items()
    }
    if not path.exists():
        return config

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key:
            config[key] = parse_config_scalar(value)
    return config


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def overlap_ratio(a: list[str], b: list[str]) -> float:
    left = {str(item).strip().lower() for item in a if str(item).strip()}
    right = {str(item).strip().lower() for item in b if str(item).strip()}
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
VALID_STAGE09_MODES = {"replay", "lookup_live"}
VALID_STAGE10_MODES = {"replay", "xgb_live"}
VALID_STAGE11_MODES = {"replay", "remote_dry_run", "remote_verify", "remote_worker"}


def _normalize_phrase(value: Any) -> str:
    return NORMALIZE_RE.sub(" ", str(value or "").strip().lower()).strip()


@lru_cache(maxsize=1)
def load_release_reference() -> dict[str, Any]:
    manifest = read_json(CURRENT_RELEASE / "manifest.json")
    stage09 = read_json(CURRENT_RELEASE / "stage09" / "bucket5_route_aware_sourceparity" / "summary.json")
    stage10 = read_json(CURRENT_RELEASE / "stage10" / "stage10_current_mainline_summary.json")
    stage11 = read_json(CURRENT_RELEASE / "stage11" / "eval" / "bucket5_tri_band_freeze_v124_alpha036" / "summary.json")
    bucket5 = next(row for row in stage10["bucket_snapshots"] if row["bucket"] == "bucket5")
    return {
        "release_id": manifest["release_id"],
        "current_output_surface": manifest["current_output_surface"],
        "stage09_truth_in_pretrim150": float(stage09["current_metrics"]["truth_in_pretrim150"]),
        "stage09_hard_miss": float(stage09["current_metrics"]["hard_miss"]),
        "stage10_bucket5_learned_recall_at_10": float(bucket5["learned_recall_at_10"]),
        "stage10_bucket5_learned_ndcg_at_10": float(bucket5["learned_ndcg_at_10"]),
        "stage11_tri_band_run": str(stage11["run"]),
        "stage11_tri_band_alpha": float(stage11["alpha"]),
        "stage11_tri_band_recall_at_10": float(stage11["recall_at_10"]),
        "stage11_tri_band_ndcg_at_10": float(stage11["ndcg_at_10"]),
    }


def warm_demo_assets(include_replay: bool = True) -> dict[str, Any]:
    release_ref = load_release_reference()
    if include_replay:
        load_replay_store()
    return release_ref


def _debug_enabled(payload: dict[str, Any]) -> bool:
    marker = payload.get("debug", False)
    return _is_truthy(marker)


def _is_truthy(marker: Any) -> bool:
    if isinstance(marker, bool):
        return marker
    if isinstance(marker, (int, float)):
        return bool(marker)
    return str(marker).strip().lower() in {"1", "true", "yes", "on"}


def _json_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _json_float(value: Any, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _short_text(value: Any, limit: int = 72) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _split_pipe_tags(value: Any, limit: int = 8) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    tags = [part.strip() for part in text.split("|") if part.strip()]
    return tags[:limit]


def _split_keywords(value: Any, limit: int = 10) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    tags = [part.strip() for part in text.split(",") if part.strip()]
    return tags[:limit]


def _parse_json_string_list(value: Any, limit: int = 12) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()][:limit]


def _candidate_tag_matches(candidate_text_norm: str, tags: list[str], limit: int = 4) -> list[str]:
    hits: list[str] = []
    for tag in tags:
        norm_tag = _normalize_phrase(tag)
        if not norm_tag:
            continue
        if norm_tag in candidate_text_norm:
            hits.append(str(tag))
        if len(hits) >= limit:
            break
    return hits


def _build_replay_candidate_explanation(
    row: pd.Series,
    *,
    strategy: str,
    user_positive_tags: list[str],
    user_negative_tags: list[str],
) -> dict[str, Any]:
    candidate_text_norm = _normalize_phrase(
        f"{row.get('name', '')} {row.get('categories', '')} {row.get('city', '')}"
    )
    matched_positive = _candidate_tag_matches(candidate_text_norm, user_positive_tags, limit=4)
    matched_negative = _candidate_tag_matches(candidate_text_norm, user_negative_tags, limit=3)

    stage10_rank = _json_int(row.get("learned_rank"))
    final_rank = _json_int(row.get("stage11_final_rank"))
    route_band = str(row.get("route_band", "") or row.get("stage11_band_label", "") or "")
    moved_up = stage10_rank is not None and final_rank is not None and final_rank < stage10_rank

    if strategy == "reward_rerank" and moved_up:
        ranking_reason = (
            f"stage11 promoted from rank {stage10_rank} to {final_rank} "
            f"via {route_band}; reward_score={_json_float(row.get('reward_score'))}"
        )
    elif strategy == "reward_rerank":
        ranking_reason = (
            f"stage11 kept this item at rank {final_rank}; "
            f"gate_pass={bool(row.get('joint_gate_pass', False))}"
        )
    elif strategy == "xgboost":
        ranking_reason = f"stage10 backbone placed this item at rank {stage10_rank}"
    else:
        ranking_reason = f"baseline pre-rank placed this item at rank {_json_int(row.get('pre_rank'))}"

    summary_parts: list[str] = []
    if matched_positive:
        summary_parts.append("matches user interests: " + ", ".join(matched_positive))
    if matched_negative:
        summary_parts.append("also overlaps avoid signals: " + ", ".join(matched_negative))
    summary_parts.append(ranking_reason)

    return {
        "matched_positive_tags": matched_positive,
        "matched_negative_tags": matched_negative,
        "ranking_reason": ranking_reason,
        "summary": "; ".join(summary_parts),
    }


def _safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _request_is_replay(payload: dict[str, Any]) -> bool:
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        return False
    request_id = str(payload.get("request_id", "")).strip()
    return bool(request_id)


def _resolve_mode(
    raw: Any,
    *,
    allowed: set[str],
    default: str,
    aliases: dict[str, str] | None = None,
) -> str:
    mode = str(raw or "").strip().lower() or str(default).strip().lower()
    aliases = aliases or {}
    mode = aliases.get(mode, mode)
    if mode not in allowed:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def resolve_stage09_mode(payload: dict[str, Any], serving_config: dict[str, Any]) -> str:
    raw = payload.get("stage09_mode", serving_config.get("stage09_default_mode", "replay"))
    return _resolve_mode(raw, allowed=VALID_STAGE09_MODES, default="replay")


def resolve_stage10_mode(payload: dict[str, Any], serving_config: dict[str, Any]) -> str:
    raw = payload.get("stage10_mode", serving_config.get("stage10_default_mode", "replay"))
    return _resolve_mode(raw, allowed=VALID_STAGE10_MODES, default="replay")


def resolve_stage11_mode(payload: dict[str, Any], serving_config: dict[str, Any]) -> str:
    raw = payload.get(
        "stage11_mode",
        payload.get(
            "stage11_live_mode",
            serving_config.get("stage11_default_mode", serving_config.get("stage11_live_default_mode", "replay")),
        ),
    )
    return _resolve_mode(
        raw,
        allowed=VALID_STAGE11_MODES,
        default="replay",
        aliases={"off": "replay"},
    )


def _should_include_fallback_demo(payload: dict[str, Any]) -> bool:
    marker = payload.get("include_fallback_demo", False)
    if isinstance(marker, bool):
        return marker
    if isinstance(marker, (int, float)):
        return bool(marker)
    return str(marker).strip().lower() in {"1", "true", "yes", "on"}


def derive_features(user_profile: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    preferred_cuisines = [str(x) for x in user_profile.get("preferred_cuisines", [])]
    cuisine_tags = [str(x) for x in candidate.get("cuisine_tags", [])]
    cuisine_match = overlap_ratio(preferred_cuisines, cuisine_tags)
    preferred_price = int(user_profile.get("preferred_price_tier", 2))
    price_tier = int(candidate.get("price_tier", preferred_price))
    price_match = clamp(1.0 - abs(price_tier - preferred_price) / 3.0)
    preferred_zone = str(user_profile.get("preferred_zone", "")).strip().lower()
    candidate_zone = str(candidate.get("zone", "")).strip().lower()
    zone_match = 1.0 if preferred_zone and preferred_zone == candidate_zone else 0.35
    novelty = clamp(candidate.get("novelty", 0.0))
    popularity = clamp(candidate.get("popularity", 0.0))
    quality = clamp(candidate.get("quality", 0.0))
    prescore = clamp(candidate.get("prescore", 0.0))
    rm_score = clamp(candidate.get("rm_score", 0.0))
    route_hints = candidate.get("route_hints", {}) if isinstance(candidate.get("route_hints", {}), dict) else {}
    als = clamp(route_hints.get("als", 0.0))
    profile = clamp(route_hints.get("profile", 0.0))
    context = clamp(route_hints.get("context", 0.0))
    popular = clamp(route_hints.get("popular", 0.0))
    return {
        "cuisine_match": cuisine_match,
        "price_match": price_match,
        "zone_match": zone_match,
        "novelty": novelty,
        "popularity": popularity,
        "quality": quality,
        "prescore": prescore,
        "rm_score": rm_score,
        "route_als": als,
        "route_profile": profile,
        "route_context": context,
        "route_popular": popular,
    }


def score_stage09(features: dict[str, float]) -> tuple[float, str]:
    route_breakdown = {
        "profile": 0.34 * features["route_profile"],
        "als": 0.24 * features["route_als"],
        "context": 0.18 * features["route_context"],
        "cuisine": 0.12 * features["cuisine_match"],
        "price": 0.06 * features["price_match"],
        "popular": 0.06 * features["route_popular"],
    }
    best_route = max(route_breakdown, key=route_breakdown.get)
    return sum(route_breakdown.values()), best_route


def score_stage10(features: dict[str, float]) -> float:
    return (
        0.42 * features["prescore"]
        + 0.18 * features["cuisine_match"]
        + 0.12 * features["price_match"]
        + 0.10 * features["zone_match"]
        + 0.10 * features["quality"]
        + 0.07 * features["novelty"]
        - 0.04 * features["popularity"]
    )


def stage11_band_for_rank(rank: int) -> str | None:
    if 11 <= rank <= 30:
        return "11-30"
    if 31 <= rank <= 60:
        return "31-60"
    if 61 <= rank <= 100:
        return "61-100"
    return None


def stage11_bonus(features: dict[str, float], stage10_rank: int, alpha: float, stage11_topn: int) -> tuple[float, str | None]:
    if stage10_rank > stage11_topn:
        return 0.0, None
    band = stage11_band_for_rank(stage10_rank)
    if band is None:
        return 0.0, None
    band_multiplier = {"11-30": 0.36, "31-60": 0.24, "61-100": 0.08}[band]
    bonus = alpha * features["rm_score"] * band_multiplier * (0.8 + 0.2 * features["route_profile"])
    return bonus, band


def strategy_attempt_order(requested_strategy: str, serving_config: dict[str, Any]) -> list[str]:
    allowed = [str(item) for item in serving_config.get("allowed_strategies", [])]
    fallback_order = [str(item) for item in serving_config.get("fallback_order", []) if str(item) in allowed]
    if requested_strategy not in allowed:
        raise ValueError(f"unsupported strategy: {requested_strategy}")
    if requested_strategy in fallback_order:
        return fallback_order[fallback_order.index(requested_strategy) :]
    return [requested_strategy]


def should_simulate_failure(payload: dict[str, Any], strategy: str) -> bool:
    marker = payload.get("simulate_failure_for", payload.get("force_fail_strategy"))
    if marker is None:
        return False
    if isinstance(marker, list):
        return strategy in {str(item) for item in marker}
    return str(marker).strip() == strategy


def rank_payload_for_strategy(
    payload: dict[str, Any],
    strategy: str,
    release_ref: dict[str, Any],
    serving_config: dict[str, Any],
) -> dict[str, Any]:
    if should_simulate_failure(payload, strategy):
        raise RuntimeError(f"simulated failure for {strategy}")

    user_profile = payload.get("user_profile", {}) if isinstance(payload.get("user_profile", {}), dict) else {}
    candidates_raw = payload.get("candidates", [])
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("payload requires a non-empty candidates list")

    top_k = max(1, int(payload.get("top_k", serving_config.get("top_k", 5))))
    stage09_topn = max(top_k, int(payload.get("stage09_topn", serving_config.get("stage09_topn", 12))))
    stage11_topn = max(top_k, int(payload.get("stage11_topn", serving_config.get("stage11_topn", 30))))

    enriched: list[dict[str, Any]] = []
    for row in candidates_raw:
        if not isinstance(row, dict):
            continue
        features = derive_features(user_profile, row)
        route_score, route_name = score_stage09(features)
        stage10_score = score_stage10(features)
        enriched.append(
            {
                "business_id": str(row.get("business_id", "")).strip(),
                "name": str(row.get("name", "")).strip() or str(row.get("business_id", "")).strip(),
                "zone": str(row.get("zone", "")).strip(),
                "cuisine_tags": [str(item) for item in row.get("cuisine_tags", [])],
                "features": features,
                "baseline_score": features["prescore"],
                "stage09_route_score": route_score,
                "stage09_primary_route": route_name,
                "stage10_score": stage10_score,
            }
        )

    enriched.sort(key=lambda row: (row["stage09_route_score"], row["stage10_score"]), reverse=True)
    retained = enriched[:stage09_topn]

    for index, row in enumerate(sorted(retained, key=lambda item: item["baseline_score"], reverse=True), start=1):
        row["baseline_rank"] = index
    for index, row in enumerate(sorted(retained, key=lambda item: item["stage10_score"], reverse=True), start=1):
        row["stage10_rank"] = index

    alpha = float(release_ref["stage11_tri_band_alpha"]) if strategy == "reward_rerank" else 0.0
    for row in retained:
        bonus, band = (
            stage11_bonus(row["features"], int(row["stage10_rank"]), alpha=alpha, stage11_topn=stage11_topn)
            if strategy == "reward_rerank"
            else (0.0, None)
        )
        row["stage11_bonus"] = bonus
        row["stage11_band"] = band
        if strategy == "baseline":
            row["final_score"] = row["baseline_score"]
        elif strategy == "xgboost":
            row["final_score"] = row["stage10_score"]
        else:
            row["final_score"] = row["stage10_score"] + bonus

    final_sorted = sorted(retained, key=lambda row: row["final_score"], reverse=True)
    for index, row in enumerate(final_sorted, start=1):
        row["final_rank"] = index

    rescued = [
        row for row in final_sorted if strategy == "reward_rerank" and row["stage11_band"] and row["final_rank"] < row["stage10_rank"]
    ]
    top_candidates = []
    for row in final_sorted[:top_k]:
        top_candidates.append(
            {
                "rank": int(row["final_rank"]),
                "business_id": row["business_id"],
                "name": row["name"],
                "zone": row["zone"],
                "baseline_rank": int(row["baseline_rank"]),
                "stage10_rank": int(row["stage10_rank"]),
                "stage09_primary_route": row["stage09_primary_route"],
                "stage11_band": row["stage11_band"],
                "baseline_score": round(float(row["baseline_score"]), 4),
                "stage10_score": round(float(row["stage10_score"]), 4),
                "stage11_bonus": round(float(row["stage11_bonus"]), 4),
                "final_score": round(float(row["final_score"]), 4),
            }
        )

    candidate_trace = []
    for row in final_sorted:
        candidate_trace.append(
            {
                "business_id": row["business_id"],
                "name": row["name"],
                "stage09_route_score": round(float(row["stage09_route_score"]), 4),
                "stage09_primary_route": row["stage09_primary_route"],
                "baseline_score": round(float(row["baseline_score"]), 4),
                "baseline_rank": int(row["baseline_rank"]),
                "stage10_score": round(float(row["stage10_score"]), 4),
                "stage10_rank": int(row["stage10_rank"]),
                "stage11_band": row["stage11_band"],
                "stage11_bonus": round(float(row["stage11_bonus"]), 4),
                "final_rank": int(row["final_rank"]),
            }
        )

    return {
        "request_id": str(payload.get("request_id", "")).strip() or "demo_request",
        "service": "batch_infer_demo",
        "mode": "mock_batch_inference",
        "strategy_used": strategy,
        "release_reference": release_ref,
        "serving_config": {
            "config_path": str(SERVING_CONFIG_PATH.relative_to(REPO_ROOT)),
            "model_version": str(serving_config.get("model_version", "")),
            "latency_budget_ms": int(serving_config.get("latency_budget_ms", 0)),
        },
        "user_profile": {
            "user_id": str(user_profile.get("user_id", "")).strip(),
            "activity_bucket": str(user_profile.get("activity_bucket", "")).strip(),
            "preferred_cuisines": [str(x) for x in user_profile.get("preferred_cuisines", [])],
            "preferred_price_tier": int(user_profile.get("preferred_price_tier", 2)),
            "preferred_zone": str(user_profile.get("preferred_zone", "")).strip(),
        },
        "summary_metrics": {
            "input_candidates": len(candidates_raw),
            "stage09_retained_candidates": len(retained),
            "stage09_retention_ratio": round(len(retained) / max(1, len(candidates_raw)), 4),
            "stage10_window_size": len(retained),
            "stage11_rescued_candidates": len(rescued),
            "stage11_rescued_into_top_k": sum(1 for row in rescued if row["final_rank"] <= top_k),
            "top_k": top_k,
        },
        "top_k": top_candidates,
        "rescued_candidates": [
            {
                "business_id": row["business_id"],
                "name": row["name"],
                "stage10_rank": int(row["stage10_rank"]),
                "final_rank": int(row["final_rank"]),
                "stage11_band": row["stage11_band"],
                "stage11_bonus": round(float(row["stage11_bonus"]), 4),
            }
            for row in rescued
        ],
        "candidate_trace": candidate_trace,
    }


def _replay_strategy_fields(strategy: str) -> tuple[str, str]:
    if strategy == "baseline":
        return "pre_rank", "pre_score"
    if strategy == "xgboost":
        return "learned_rank", "learned_blend_score"
    return "stage11_final_rank", "final_score"


def _replay_candidate_view(
    row: pd.Series,
    *,
    rank_value: int,
    active_score_col: str,
    stage09_primary_route: str | None,
) -> dict[str, Any]:
    route_value = str(stage09_primary_route or row.get("stage09_primary_route", "") or "").strip()
    if not route_value:
        source_raw = row.get("source_set")
        if hasattr(source_raw, "tolist") and not isinstance(source_raw, str):
            source_items = [str(item).strip() for item in source_raw.tolist() if str(item).strip()]
        elif isinstance(source_raw, (list, tuple, set)):
            source_items = [str(item).strip() for item in source_raw if str(item).strip()]
        else:
            source_items = []
        for preferred in ("profile", "als", "cluster", "popular"):
            if preferred in source_items:
                route_value = preferred
                break
        if not route_value and source_items:
            route_value = source_items[0]
    return {
        "rank": rank_value,
        "business_id": str(row.get("business_id", "") or ""),
        "name": str(row.get("name", "") or ""),
        "city": str(row.get("city", "") or ""),
        "baseline_rank": _json_int(row.get("pre_rank")),
        "stage10_rank": _json_int(row.get("learned_rank")),
        "stage11_rank": _json_int(row.get("stage11_final_rank")),
        "stage09_primary_route": route_value,
        "stage11_band": str(row.get("route_band", "") or row.get("stage11_band_label", "") or ""),
        "baseline_score": _json_float(row.get("pre_score")),
        "stage10_score": _json_float(row.get("learned_blend_score")),
        "reward_score": _json_float(row.get("reward_score")),
        "stage11_bonus": _json_float(row.get("rescue_bonus")),
        "joint_gate_pass": bool(row.get("joint_gate_pass", False)),
        "final_score": _json_float(row.get(active_score_col)),
        "movement_vs_stage10": None
        if pd.isna(row.get("learned_rank")) or pd.isna(row.get("stage11_final_rank"))
        else int(row.get("learned_rank")) - int(row.get("stage11_final_rank")),
    }


def _build_ranked_candidate_outputs(
    sorted_active: pd.DataFrame,
    *,
    top_k: int,
    trace_limit: int,
    strategy: str,
    active_score_col: str,
    stage09_primary_route: str | None,
    user_positive_tags: list[str],
    user_negative_tags: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_candidates: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(sorted_active.head(top_k).iterrows(), start=1):
        candidate_view = _replay_candidate_view(
            row,
            rank_value=idx,
            active_score_col=active_score_col,
            stage09_primary_route=stage09_primary_route,
        )
        candidate_view["why_recommended"] = _build_replay_candidate_explanation(
            row,
            strategy=strategy,
            user_positive_tags=user_positive_tags,
            user_negative_tags=user_negative_tags,
        )
        top_candidates.append(candidate_view)

    candidate_trace: list[dict[str, Any]] = []
    for _, row in sorted_active.head(trace_limit).iterrows():
        explanation = _build_replay_candidate_explanation(
            row,
            strategy=strategy,
            user_positive_tags=user_positive_tags,
            user_negative_tags=user_negative_tags,
        )
        candidate_trace.append(
            {
                "business_id": str(row.get("business_id", "") or ""),
                "name": str(row.get("name", "") or ""),
                "city": str(row.get("city", "") or ""),
                "categories_preview": _short_text(row.get("categories", ""), limit=84),
                "pre_rank": _json_int(row.get("pre_rank")),
                "pre_score": _json_float(row.get("pre_score")),
                "stage10_rank": _json_int(row.get("learned_rank")),
                "stage10_score": _json_float(row.get("learned_blend_score")),
                "stage11_rank": _json_int(row.get("stage11_final_rank")),
                "reward_score": _json_float(row.get("reward_score")),
                "rescue_bonus": _json_float(row.get("rescue_bonus")),
                "joint_gate_pass": bool(row.get("joint_gate_pass", False)),
                "stage11_band": str(row.get("route_band", "") or row.get("stage11_band_label", "") or ""),
                "active_rank": _json_int(row.get("active_rank")),
                "active_score": _json_float(row.get(active_score_col)),
                "movement_vs_stage10": None
                if pd.isna(row.get("learned_rank")) or pd.isna(row.get("stage11_final_rank"))
                else int(row.get("learned_rank")) - int(row.get("stage11_final_rank")),
                "why_recommended": explanation["summary"],
            }
        )
    return top_candidates, candidate_trace


def _attach_live_truth_probe(
    truth_audit: dict[str, Any],
    live_pdf: pd.DataFrame | None,
    *,
    rank_col: str,
    score_col: str,
    reference_label: str,
    top_k: int,
) -> dict[str, Any]:
    if not truth_audit.get("available") or live_pdf is None or live_pdf.empty:
        return truth_audit
    truth_item = truth_audit.get("truth_item", {})
    business_id = str(truth_item.get("business_id", "") or "").strip()
    if not business_id:
        return truth_audit
    probe_pdf = live_pdf[live_pdf["business_id"].astype(str).eq(business_id)].copy()
    if probe_pdf.empty:
        truth_audit["requested_strategy_live_probe"] = {
            "available": False,
            "reference": reference_label,
            "reason": "truth item not found in live candidate pack",
        }
        return truth_audit
    probe_row = probe_pdf.sort_values([rank_col, "item_idx"], kind="stable").iloc[0]
    truth_audit["requested_strategy_live_probe"] = {
        "available": True,
        "reference": reference_label,
        "rank": _json_int(probe_row.get(rank_col)),
        "score": _json_float(probe_row.get(score_col)),
        "in_top_k": bool(int(probe_row.get(rank_col)) <= top_k),
    }
    return truth_audit


def _build_offline_truth_audit(working: pd.DataFrame, *, requested_strategy: str, top_k: int) -> dict[str, Any]:
    truth_pdf = working[working["label_true"].eq(1)].sort_values(["item_idx"], kind="stable").copy()
    if truth_pdf.empty:
        return {
            "available": False,
            "served_online": False,
            "note": "offline eval truth is not present for this replay request",
            "truth_count": 0,
        }

    truth_row = truth_pdf.iloc[0]
    strategy_rank_col, _ = _replay_strategy_fields(requested_strategy)
    truth_strategy_rank = _json_int(truth_row.get(strategy_rank_col))
    return {
        "available": True,
        "served_online": False,
        "note": "offline eval truth is attached for replay audit only; a real online request would not see this field",
        "truth_count": int(truth_pdf.shape[0]),
        "truth_item": {
            "business_id": str(truth_row.get("business_id", "") or ""),
            "name": str(truth_row.get("name", "") or ""),
            "city": str(truth_row.get("city", "") or ""),
            "baseline_rank": _json_int(truth_row.get("pre_rank")),
            "stage10_rank": _json_int(truth_row.get("learned_rank")),
            "stage11_rank": _json_int(truth_row.get("stage11_final_rank")),
            "requested_strategy_rank": truth_strategy_rank,
            "baseline_score": _json_float(truth_row.get("pre_score")),
            "stage10_score": _json_float(truth_row.get("learned_blend_score")),
            "stage11_score": _json_float(truth_row.get("final_score")),
            "joint_gate_pass": bool(truth_row.get("joint_gate_pass", False)),
            "route_band": str(truth_row.get("route_band", "") or truth_row.get("stage11_band_label", "") or ""),
            "in_top_k_by_baseline": bool(int(truth_row.get("pre_rank")) <= top_k),
            "in_top_k_by_stage10": bool(int(truth_row.get("learned_rank")) <= top_k),
            "in_top_k_by_stage11": bool(int(truth_row.get("stage11_final_rank")) <= top_k),
            "in_top_k_by_requested_strategy": bool((truth_strategy_rank or 999999) <= top_k),
            "movement_vs_stage10": None
            if pd.isna(truth_row.get("learned_rank")) or pd.isna(truth_row.get("stage11_final_rank"))
            else int(truth_row.get("learned_rank")) - int(truth_row.get("stage11_final_rank")),
        },
    }


def _build_fallback_demo(
    payload: dict[str, Any],
    serving_config: dict[str, Any],
    requested_strategy: str,
) -> dict[str, Any]:
    if should_simulate_failure(payload, requested_strategy):
        return {
            "available": False,
            "status": "skipped",
            "reason": "current request is already simulating a failure",
        }
    if requested_strategy not in {"reward_rerank", "xgboost"}:
        return {
            "available": False,
            "status": "skipped",
            "reason": "fallback demo is only meaningful when the requested strategy has a lower fallback tier",
        }

    probe_payload = dict(payload)
    probe_payload["simulate_failure_for"] = requested_strategy
    probe_payload["stage11_live_mode"] = "off"
    probe_payload["include_fallback_demo"] = False
    probe_payload["debug"] = False
    probe_result = rank_payload(probe_payload, serving_config=serving_config)
    return {
        "available": True,
        "status": "demonstrated",
        "requested_strategy": requested_strategy,
        "simulate_failure_for": requested_strategy,
        "strategy_used": probe_result["strategy_used"],
        "fallback_used": bool(probe_result["fallback_used"]),
        "fallback_reason": str(probe_result.get("fallback_reason", "") or ""),
        "latency_ms": probe_result["serving_metrics"]["latency_ms"],
        "top_k_preview": [
            {
                "rank": row["rank"],
                "business_id": row["business_id"],
                "name": row["name"],
            }
            for row in probe_result.get("top_k", [])[:3]
        ],
    }


def rank_payload_for_replay_strategy(
    payload: dict[str, Any],
    strategy: str,
    release_ref: dict[str, Any],
    serving_config: dict[str, Any],
    replay_store: Any,
    request_pdf: pd.DataFrame,
) -> dict[str, Any]:
    if should_simulate_failure(payload, strategy):
        raise RuntimeError(f"simulated failure for {strategy}")

    request_id = str(payload.get("request_id", "")).strip()
    if not request_id:
        raise ValueError("replay mode requires request_id")

    top_k = max(1, int(payload.get("top_k", serving_config.get("top_k", 5))))
    stage09_topn = max(top_k, int(payload.get("stage09_topn", serving_config.get("stage09_topn", 12))))
    requested_stage09_mode = resolve_stage09_mode(payload, serving_config)
    requested_stage10_mode = resolve_stage10_mode(payload, serving_config)
    requested_stage11_mode = resolve_stage11_mode(payload, serving_config)
    user_positive_tags = _parse_json_string_list(request_pdf.iloc[0].get("user_pos_tags_json", ""), limit=18)
    user_negative_tags = _parse_json_string_list(request_pdf.iloc[0].get("user_neg_tags_json", ""), limit=12)

    working = request_pdf.copy()
    request_user_idx = int(working.iloc[0]["user_idx"])
    request_user_id = str(working.iloc[0].get("user_id", "") or "")
    user_top_city = str(working.iloc[0].get("top_city", "") or "")
    user_top_geo = str(working.iloc[0].get("top_geo_cell_3dp", "") or "")

    latency_breakdown = {
        "request_lookup": 0.0,
        "stage09": 0.0,
        "stage10": 0.0,
        "stage11": 0.0,
        "offline_truth_audit": 0.0,
        "fallback_demo": 0.0,
    }

    stage09_effective_mode = requested_stage09_mode
    stage10_effective_mode = requested_stage10_mode
    stage09_live_pdf: pd.DataFrame | None = None
    stage09_live_meta: dict[str, Any] = {}
    stage10_live_pdf: pd.DataFrame | None = None
    stage10_live_meta: dict[str, Any] = {}
    stage09_error = ""
    stage10_error = ""

    need_stage09_live = requested_stage09_mode == "lookup_live" or requested_stage10_mode == "xgb_live"
    if need_stage09_live:
        stage09_started = time.perf_counter()
        try:
            stage09_live_pdf, stage09_live_meta = load_stage09_live_candidates(request_user_idx)
        except Exception as exc:
            stage09_effective_mode = "replay"
            stage09_error = str(exc)
        latency_breakdown["stage09"] = round((time.perf_counter() - stage09_started) * 1000.0, 3)

    if requested_stage10_mode == "xgb_live":
        stage10_started = time.perf_counter()
        try:
            stage10_live_pdf, stage10_live_meta = score_stage10_live(request_user_idx, top_k=max(top_k, 10))
        except Exception as exc:
            stage10_effective_mode = "replay"
            stage10_error = str(exc)
        latency_breakdown["stage10"] = round((time.perf_counter() - stage10_started) * 1000.0, 3)

    stage09_pre_head = working.sort_values(["pre_rank", "item_idx"], kind="stable").head(stage09_topn)
    if stage09_effective_mode == "lookup_live" and stage09_live_pdf is not None and not stage09_live_pdf.empty:
        live_pre_head = stage09_live_pdf.sort_values(["pre_rank", "item_idx"], kind="stable").head(stage09_topn)
        replay_pre10 = set(stage09_pre_head.head(10)["business_id"].astype(str).tolist())
        live_pre10 = set(live_pre_head.head(10)["business_id"].astype(str).tolist())
        stage09_trace = {
            "mode": "lookup_live",
            "requested_mode": requested_stage09_mode,
            "effective_mode": stage09_effective_mode,
            "window_size": int(stage09_live_pdf.shape[0]),
            "visible_topn": stage09_topn,
            "data_source": str(stage09_live_meta.get("data_source", "")),
            "source_alignment": str(stage09_live_meta.get("source_alignment", "")),
            "requested_release_stage09_bucket_dir": str(stage09_live_meta.get("requested_release_stage09_bucket_dir", "")),
            "effective_stage09_bucket_dir": str(stage09_live_meta.get("effective_stage09_bucket_dir", "")),
            "note": (
                "per-user candidate lookup from local stage09 parquet"
                if stage09_live_meta.get("source_alignment") == "release_sourceparity"
                else "per-user candidate lookup from local fallback candidate pack; current_release sourceparity parquet is not synced on this host"
            ),
            "timings_ms": dict(stage09_live_meta.get("timings_ms", {})),
            "head_overlap_replay_pre10_vs_lookup_live_10": int(len(replay_pre10 & live_pre10)),
            "top_pre_candidates": [
                {
                    "pre_rank": _json_int(row.get("pre_rank")),
                    "business_id": str(row.get("business_id", "") or ""),
                    "name": str(row.get("name", "") or ""),
                    "city": str(row.get("city", "") or ""),
                    "pre_score": _json_float(row.get("pre_score")),
                    "route_preview": _replay_candidate_view(
                        row,
                        rank_value=int(row.get("pre_rank")),
                        active_score_col="pre_score",
                        stage09_primary_route=None,
                    ).get("stage09_primary_route"),
                }
                for _, row in live_pre_head.iterrows()
            ],
        }
    else:
        stage09_trace = {
            "mode": "frozen_replay",
            "requested_mode": requested_stage09_mode,
            "effective_mode": stage09_effective_mode,
            "window_size": int(working.shape[0]),
            "visible_topn": stage09_topn,
            "note": (
                "freeze pack does not retain per-route recall attribution; this replay shows the frozen pre-ranked shortlist only"
                if not stage09_error
                else f"lookup_live unavailable, fell back to replay: {stage09_error}"
            ),
            "top_pre_candidates": [
                {
                    "pre_rank": _json_int(row.get("pre_rank")),
                    "business_id": str(row.get("business_id", "") or ""),
                    "name": str(row.get("name", "") or ""),
                    "city": str(row.get("city", "") or ""),
                    "pre_score": _json_float(row.get("pre_score")),
                }
                for _, row in stage09_pre_head.iterrows()
            ],
        }

    rescued_mask = working["stage11_final_rank"].lt(working["learned_rank"])
    rescued_pdf = working[rescued_mask].sort_values(
        ["stage11_final_rank", "learned_rank", "item_idx"],
        kind="stable",
    )
    gate_scored_pdf = working[working["reward_score"].notna()].copy()
    gate_rejected_pdf = gate_scored_pdf[~gate_scored_pdf["joint_gate_pass"]].sort_values(
        ["learned_rank", "item_idx"],
        kind="stable",
    )
    simulate_stage11_cache_miss = _is_truthy(payload.get("simulate_stage11_cache_miss", False))
    stage11_requested_enabled = strategy == "reward_rerank"
    stage11_cache_candidate_count = int(gate_scored_pdf.shape[0])
    stage11_cache_status = (
        "miss_simulated"
        if stage11_requested_enabled and simulate_stage11_cache_miss
        else "hit"
        if stage11_requested_enabled and stage11_cache_candidate_count > 0
        else "miss"
        if stage11_requested_enabled
        else "available_not_requested"
    )
    stage11_policy_applied = stage11_requested_enabled and stage11_cache_status == "hit"
    stage10_head = working.sort_values(["learned_rank", "item_idx"], kind="stable").head(max(10, top_k))
    if stage10_effective_mode == "xgb_live" and stage10_live_pdf is not None and not stage10_live_pdf.empty:
        live_stage10_head = stage10_live_pdf.sort_values(["learned_rank", "item_idx"], kind="stable").head(max(10, top_k))
        replay_stage10_10 = set(stage10_head.head(10)["business_id"].astype(str).tolist())
        live_stage10_10 = set(live_stage10_head.head(10)["business_id"].astype(str).tolist())
        stage10_summary = {
            "mode": "xgb_live",
            "requested_mode": requested_stage10_mode,
            "effective_mode": stage10_effective_mode,
            "probe_only": strategy != "xgboost",
            "window_size": int(stage10_live_pdf.shape[0]),
            "data_source": str(stage10_live_meta.get("data_source", "")),
            "history_source": str(stage10_live_meta.get("history_source", "")),
            "model_json": str(stage10_live_meta.get("model_json", "")),
            "model_file": str(stage10_live_meta.get("model_file", "")),
            "source_alignment": str(stage10_live_meta.get("source_alignment", "")),
            "blend_alpha": _json_float(stage10_live_meta.get("blend_alpha")),
            "blend_mode": str(stage10_live_meta.get("blend_mode", "")),
            "feature_count": int(stage10_live_meta.get("feature_count", 0) or 0),
            "warm_model_cache": bool(stage10_live_meta.get("warm_model_cache", False)),
            "timings_ms": dict(stage10_live_meta.get("timings_ms", {})),
            "head_overlap_replay_stage10_10": int(len(replay_stage10_10 & live_stage10_10)),
            "learned_top_candidates": [
                {
                    "learned_rank": _json_int(row.get("learned_rank")),
                    "business_id": str(row.get("business_id", "") or ""),
                    "name": str(row.get("name", "") or ""),
                    "stage10_score": _json_float(row.get("learned_blend_score")),
                    "pre_rank": _json_int(row.get("pre_rank")),
                    "learned_score": _json_float(row.get("learned_score")),
                }
                for _, row in live_stage10_head.iterrows()
            ],
        }
    else:
        stage10_summary = {
            "mode": "frozen_replay",
            "requested_mode": requested_stage10_mode,
            "effective_mode": stage10_effective_mode,
            "window_size": int(working.shape[0]),
            "note": (
                ""
                if not stage10_error
                else f"xgb_live unavailable, fell back to replay: {stage10_error}"
            ),
            "learned_top_candidates": [
                {
                    "learned_rank": _json_int(row.get("learned_rank")),
                    "business_id": str(row.get("business_id", "") or ""),
                    "name": str(row.get("name", "") or ""),
                    "stage10_score": _json_float(row.get("learned_blend_score")),
                    "pre_rank": _json_int(row.get("pre_rank")),
                }
                for _, row in stage10_head.iterrows()
            ],
            "head_overlap_pre10_vs_stage10_10": int(
                len(
                    set(working[working["pre_rank"].le(10)]["business_id"].astype(str).tolist())
                    & set(working[working["learned_rank"].le(10)]["business_id"].astype(str).tolist())
                )
            ),
        }

    display_df = working.copy()
    rank_col, active_score_col = _replay_strategy_fields(strategy)
    active_strategy_for_explanation = strategy
    stage09_primary_route = "frozen_pre_rank_window"
    if strategy == "baseline" and stage09_effective_mode == "lookup_live" and stage09_live_pdf is not None and not stage09_live_pdf.empty:
        display_df = stage09_live_pdf.copy()
        display_df["learned_rank"] = pd.NA
        display_df["learned_blend_score"] = pd.NA
        display_df["stage11_final_rank"] = pd.NA
        display_df["reward_score"] = pd.NA
        display_df["rescue_bonus"] = 0.0
        display_df["joint_gate_pass"] = False
        display_df["route_band"] = ""
        display_df["stage11_band_label"] = ""
        rank_col, active_score_col = "pre_rank", "pre_score"
        stage09_primary_route = None
    elif strategy == "xgboost" and stage10_effective_mode == "xgb_live" and stage10_live_pdf is not None and not stage10_live_pdf.empty:
        display_df = stage10_live_pdf.copy()
        display_df["stage11_final_rank"] = pd.NA
        display_df["reward_score"] = pd.NA
        display_df["rescue_bonus"] = 0.0
        display_df["joint_gate_pass"] = False
        display_df["route_band"] = ""
        display_df["stage11_band_label"] = ""
        rank_col, active_score_col = "learned_rank", "learned_blend_score"
        stage09_primary_route = None
    elif strategy == "reward_rerank" and not stage11_policy_applied:
        rank_col, active_score_col = "learned_rank", "learned_blend_score"
        active_strategy_for_explanation = "xgboost"

    display_df["active_rank"] = pd.to_numeric(display_df.get(rank_col), errors="coerce").fillna(999999).astype(int)
    sorted_active = display_df.sort_values(["active_rank", "item_idx"], kind="stable").reset_index(drop=True)
    trace_limit = len(sorted_active) if _debug_enabled(payload) else max(top_k, 25)
    top_candidates, candidate_trace = _build_ranked_candidate_outputs(
        sorted_active,
        top_k=top_k,
        trace_limit=trace_limit,
        strategy=active_strategy_for_explanation,
        active_score_col=active_score_col,
        stage09_primary_route=stage09_primary_route,
        user_positive_tags=user_positive_tags,
        user_negative_tags=user_negative_tags,
    )
    replay_source = _safe_relative(replay_store.paths.score_csv)
    rescued_into_top_k = int(
        rescued_pdf[rescued_pdf["stage11_final_rank"].le(top_k)].shape[0]
    ) if stage11_policy_applied else 0
    stage11_started = time.perf_counter()
    stage11_audit = {
        "mode": "frozen_replay",
        "requested_mode": requested_stage11_mode,
        "effective_mode": requested_stage11_mode,
        "enabled": stage11_requested_enabled,
        "applied_to_serving": stage11_policy_applied,
        "cache_status": stage11_cache_status,
        "note": (
            "frozen v124 tri-band rescue cache over the stored top100 shortlist"
            if stage11_policy_applied
            else "stage11 cache was not applied to the synchronous serving result"
            if stage11_requested_enabled
            else "stage11 replay is available in the freeze pack but not applied for this strategy"
        ),
        "reward_scored_candidates": int(gate_scored_pdf.shape[0]),
        "joint_gate_pass_candidates": int(gate_scored_pdf[gate_scored_pdf["joint_gate_pass"]].shape[0]),
        "joint_gate_rejected_candidates": int(gate_rejected_pdf.shape[0]),
        "rescued_candidates": int(rescued_pdf.shape[0]) if stage11_policy_applied else 0,
        "rescued_into_top_k": rescued_into_top_k,
        "top_promotions": [
            {
                "business_id": str(row.get("business_id", "") or ""),
                "name": str(row.get("name", "") or ""),
                "stage10_rank": _json_int(row.get("learned_rank")),
                "final_rank": _json_int(row.get("stage11_final_rank")),
                "promotion_gain": None
                if pd.isna(row.get("learned_rank")) or pd.isna(row.get("stage11_final_rank"))
                else int(row.get("learned_rank")) - int(row.get("stage11_final_rank")),
                "route_band": str(row.get("route_band", "") or row.get("stage11_band_label", "") or ""),
                "reward_score": _json_float(row.get("reward_score")),
                "rescue_bonus": _json_float(row.get("rescue_bonus")),
                "joint_gate_pass": bool(row.get("joint_gate_pass", False)),
            }
            for _, row in rescued_pdf.head(10).iterrows()
        ],
        "gate_rejected_examples": [
            {
                "business_id": str(row.get("business_id", "") or ""),
                "name": str(row.get("name", "") or ""),
                "stage10_rank": _json_int(row.get("learned_rank")),
                "route_band": str(row.get("route_band", "") or row.get("stage11_band_label", "") or ""),
                "reward_score": _json_float(row.get("reward_score")),
                "rescue_bonus": _json_float(row.get("rescue_bonus")),
                "joint_gate_pass": bool(row.get("joint_gate_pass", False)),
            }
            for _, row in gate_rejected_pdf.head(6).iterrows()
        ],
    }
    cache_hit_ratio = round(stage11_cache_candidate_count / max(1, int(working.shape[0])), 4)
    stage11_policy = {
        "mode": str(serving_config.get("stage11_policy_mode", "cache_first_bounded_rescue")),
        "serving_role": "synchronous_cache_policy",
        "live_rm_role": "audit_only" if requested_stage11_mode != "replay" else "disabled",
        "requested_live_mode": requested_stage11_mode,
        "cache_key_fields": ["user_idx", "item_idx", "context_bucket", "rm_version"],
        "context_bucket": "homepage_recommend_replay:bucket5",
        "rm_version": str(release_ref.get("stage11_tri_band_run", "v124")),
        "cache_source": replay_source,
        "cache_status": stage11_cache_status,
        "cache_hit": stage11_cache_status == "hit",
        "cache_hit_ratio": cache_hit_ratio if stage11_requested_enabled and not simulate_stage11_cache_miss else 0.0,
        "cached_reward_candidates_available": stage11_cache_candidate_count,
        "applied_to_serving": stage11_policy_applied,
        "fallback_strategy_on_miss": "stage10_xgboost",
        "on_cache_hit": "apply_bounded_rescue_with_gate",
        "on_cache_miss": "return_stage10_and_enqueue_rm_backfill",
        "protect_top10": True,
        "max_promotions": int(min(5, max(0, rescued_into_top_k))) if stage11_policy_applied else 0,
        "latency_budget_ms": int(serving_config.get("stage11_policy_latency_budget_ms", 50) or 50),
        "cache_ttl_hours": int(serving_config.get("stage11_cache_ttl_hours", 24) or 24),
        "backfill_event": {
            "enqueued": bool(stage11_requested_enabled and not stage11_policy_applied),
            "queue": "stage11_rm_backfill",
            "reason": stage11_cache_status,
            "user_idx": request_user_idx,
            "candidate_window_size": int(working.shape[0]),
            "live_mode_for_audit": requested_stage11_mode,
        },
    }
    latency_breakdown["stage11"] = round((time.perf_counter() - stage11_started) * 1000.0, 3)

    truth_started = time.perf_counter()
    offline_truth_audit = _build_offline_truth_audit(
        working,
        requested_strategy=strategy,
        top_k=top_k,
    )
    if strategy == "baseline" and stage09_effective_mode == "lookup_live" and stage09_live_pdf is not None:
        offline_truth_audit = _attach_live_truth_probe(
            offline_truth_audit,
            stage09_live_pdf,
            rank_col="pre_rank",
            score_col="pre_score",
            reference_label="stage09_lookup_live",
            top_k=top_k,
        )
    if strategy == "xgboost" and stage10_effective_mode == "xgb_live" and stage10_live_pdf is not None:
        offline_truth_audit = _attach_live_truth_probe(
            offline_truth_audit,
            stage10_live_pdf,
            rank_col="learned_rank",
            score_col="learned_blend_score",
            reference_label="stage10_xgb_live",
            top_k=top_k,
        )
    latency_breakdown["offline_truth_audit"] = round((time.perf_counter() - truth_started) * 1000.0, 3)

    summary_input_candidates = int(display_df.shape[0])

    return {
        "request_id": request_id,
        "service": "batch_infer_demo",
        "mode": "replay_request",
        "strategy_used": strategy,
        "release_reference": release_ref,
        "serving_config": {
            "config_path": str(SERVING_CONFIG_PATH.relative_to(REPO_ROOT)),
            "model_version": str(serving_config.get("model_version", "")),
            "latency_budget_ms": int(serving_config.get("latency_budget_ms", 0)),
        },
        "user_profile": {
            "user_id": request_user_id or f"bucket5_user_idx_{request_user_idx}",
            "activity_bucket": "bucket5",
            "preferred_cuisines": _split_keywords(working.iloc[0].get("profile_keywords", "")),
            "preferred_price_tier": None,
            "preferred_zone": "",
            "n_train": _json_int(working.iloc[0].get("n_train")),
            "tier": str(working.iloc[0].get("tier", "") or ""),
            "profile_confidence": _json_float(working.iloc[0].get("profile_confidence")),
            "profile_top_pos_tags": _split_pipe_tags(working.iloc[0].get("profile_top_pos_tags", "")),
            "profile_top_neg_tags": _split_pipe_tags(working.iloc[0].get("profile_top_neg_tags", "")),
            "profile_text_short": _short_text(working.iloc[0].get("profile_text_short", ""), limit=220),
            "primary_city": user_top_city,
            "primary_geo_cell_3dp": user_top_geo,
            "profile_source": _safe_relative(replay_store.paths.user_profile_csv),
        },
        "user_state_snapshot": {
            "quality_tier": str(working.iloc[0].get("user_view_quality_tier", "") or ""),
            "quality_signal_count": _json_int(working.iloc[0].get("user_quality_signal_count")),
            "top_city": user_top_city,
            "top_geo_cell_3dp": user_top_geo,
            "long_term_top_cuisine": str(working.iloc[0].get("long_term_top_cuisine", "") or ""),
            "recent_top_cuisine": str(working.iloc[0].get("recent_top_cuisine", "") or ""),
            "negative_top_cuisine": str(working.iloc[0].get("negative_top_cuisine", "") or ""),
            "negative_pressure": _json_float(working.iloc[0].get("negative_pressure")),
            "positive_tags": _parse_json_string_list(working.iloc[0].get("user_pos_tags_json", ""), limit=12),
            "negative_tags": _parse_json_string_list(working.iloc[0].get("user_neg_tags_json", ""), limit=8),
            "long_term_pref_text": _short_text(working.iloc[0].get("user_long_pref_text", ""), limit=260),
            "recent_intent_text": _short_text(working.iloc[0].get("user_recent_intent_text", ""), limit=220),
            "negative_avoid_text": _short_text(working.iloc[0].get("user_negative_avoid_text", ""), limit=220),
            "context_text": _short_text(working.iloc[0].get("user_context_text", ""), limit=220),
            "source": _safe_relative(replay_store.paths.user_text_views_parquet),
            "nonllm_source": str(working.iloc[0].get("nonllm_source", "") or ""),
        },
        "request_context": {
            "request_id": request_id,
            "user_idx": request_user_idx,
            "user_id": request_user_id,
            "bucket": "bucket5",
            "scene": "homepage_recommend_replay",
            "top_city": user_top_city,
            "top_geo_cell_3dp": user_top_geo,
            "replay_source": replay_source,
            "candidate_window_size": int(working.shape[0]),
            "cohort_note": replay_store.cohort_note,
            "sample_request_id": replay_store.sample_request_id,
        },
        "execution_modes": {
            "stage09_requested": requested_stage09_mode,
            "stage09_used": stage09_effective_mode,
            "stage10_requested": requested_stage10_mode,
            "stage10_used": stage10_effective_mode,
            "stage11_requested": requested_stage11_mode,
            "stage11_used": "cache_first" if stage11_policy_applied else "stage10_fallback",
        },
        "summary_metrics": {
            "input_candidates": summary_input_candidates,
            "stage09_retained_candidates": int(stage09_live_pdf.shape[0]) if stage09_live_pdf is not None else int(working.shape[0]),
            "stage09_retention_ratio": 1.0,
            "stage10_window_size": int(stage10_live_pdf.shape[0]) if stage10_live_pdf is not None else int(working.shape[0]),
            "stage11_rescued_candidates": int(rescued_pdf.shape[0]) if stage11_policy_applied else 0,
            "stage11_rescued_into_top_k": rescued_into_top_k,
            "top_k": top_k,
            "replay_window_candidates": int(working.shape[0]),
            "stage09_live_window_size": int(stage09_live_pdf.shape[0]) if stage09_live_pdf is not None else 0,
            "stage10_live_window_size": int(stage10_live_pdf.shape[0]) if stage10_live_pdf is not None else 0,
        },
        "top_k": top_candidates,
        "rescued_candidates": [
            {
                "business_id": str(row.get("business_id", "") or ""),
                "name": str(row.get("name", "") or ""),
                "stage10_rank": _json_int(row.get("learned_rank")),
                "final_rank": _json_int(row.get("stage11_final_rank")),
                "stage11_band": str(row.get("route_band", "") or row.get("stage11_band_label", "") or ""),
                "stage11_bonus": _json_float(row.get("rescue_bonus")),
                "reward_score": _json_float(row.get("reward_score")),
                "joint_gate_pass": bool(row.get("joint_gate_pass", False)),
            }
            for _, row in rescued_pdf.head(20).iterrows()
        ]
        if stage11_policy_applied
        else [],
        "stage09_trace": stage09_trace,
        "stage10_summary": stage10_summary,
        "stage11_policy": stage11_policy,
        "stage11_audit": stage11_audit,
        "offline_truth_audit": offline_truth_audit,
        "candidate_trace": candidate_trace,
        "_latency_breakdown_ms": latency_breakdown,
        "_stage10_model_warm": bool(stage10_live_meta.get("warm_model_cache", False)),
    }


def rank_payload(payload: dict[str, Any], serving_config: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config = serving_config or load_serving_config()
    release_ref = load_release_reference()
    requested_strategy = str(
        payload.get("strategy", payload.get("ranking_strategy", config.get("default_strategy", "reward_rerank")))
    ).strip()

    replay_mode = _request_is_replay(payload)
    replay_store = None
    request_pdf = None
    request_lookup_ms = 0.0
    replay_store_warm = False
    if replay_mode:
        replay_hits_before = load_replay_store.cache_info().hits
        request_lookup_started = time.perf_counter()
        replay_store = load_replay_store()
        request_id = str(payload.get("request_id", "")).strip()
        request_pdf = replay_store.get_request_frame(request_id)
        request_lookup_ms = round((time.perf_counter() - request_lookup_started) * 1000.0, 3)
        replay_store_warm = load_replay_store.cache_info().hits > replay_hits_before

    errors: list[str] = []
    result: dict[str, Any] | None = None
    strategy_used = ""
    for strategy in strategy_attempt_order(requested_strategy, config):
        try:
            if replay_mode:
                result = rank_payload_for_replay_strategy(
                    payload,
                    strategy,
                    release_ref,
                    config,
                    replay_store=replay_store,
                    request_pdf=request_pdf,
                )
            else:
                result = rank_payload_for_strategy(payload, strategy, release_ref, config)
            strategy_used = strategy
            break
        except Exception as exc:
            errors.append(f"{strategy}: {exc}")

    if result is None:
        raise ValueError("; ".join(errors) or "ranking failed")

    stage11_live_ms = 0.0
    fallback_demo_ms = 0.0
    if replay_mode and replay_store is not None:
        stage11_mode = resolve_stage11_mode(payload, config)
        if stage11_mode == "replay":
            result["stage11_live"] = {
                "enabled": False,
                "mode": "replay",
                "status": "skipped",
                "reason": "stage11_mode=replay",
            }
        else:
            live_payload = dict(payload)
            live_payload["stage11_live_mode"] = stage11_mode
            live_started = time.perf_counter()
            try:
                result["stage11_live"] = maybe_run_stage11_live_verify(
                    replay_result=result,
                    payload=live_payload,
                    serving_config=config,
                    replay_store=replay_store,
                )
            except Exception as exc:
                result["stage11_live"] = {
                    "enabled": True,
                    "mode": stage11_mode,
                    "status": "error",
                    "detail": str(exc),
                }
            stage11_live_ms = round((time.perf_counter() - live_started) * 1000.0, 3)
        if _debug_enabled(payload) or _should_include_fallback_demo(payload):
            fallback_started = time.perf_counter()
            try:
                result["fallback_demo"] = _build_fallback_demo(
                    payload,
                    serving_config=config,
                    requested_strategy=requested_strategy,
                )
            except Exception as exc:
                result["fallback_demo"] = {
                    "available": False,
                    "status": "error",
                    "detail": str(exc),
                }
            fallback_demo_ms = round((time.perf_counter() - fallback_started) * 1000.0, 3)
        else:
            result["fallback_demo"] = {
                "available": False,
                "status": "skipped",
                "reason": "set debug=true or include_fallback_demo=true to attach a fallback demonstration",
            }
    else:
        result["stage11_live"] = {
            "enabled": False,
            "mode": "off",
            "status": "skipped",
            "reason": "legacy_payload_mode",
        }
        result["fallback_demo"] = {
            "available": False,
            "status": "skipped",
            "reason": "fallback demo is only attached in replay mode",
        }

    wall_latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    audit_latency_ms = round(stage11_live_ms + fallback_demo_ms, 3)
    serving_latency_ms = round(max(0.0, wall_latency_ms - audit_latency_ms), 3)
    fallback_used = strategy_used != requested_strategy
    fallback_count = 1 if fallback_used else 0
    latency_breakdown = dict(result.pop("_latency_breakdown_ms", {}))
    latency_breakdown["request_lookup"] = round(request_lookup_ms, 3)
    latency_breakdown["fallback_demo"] = 0.0
    latency_breakdown["replay_store_warm"] = bool(replay_store_warm)
    latency_breakdown["stage10_model_warm"] = bool(result.pop("_stage10_model_warm", False))
    latency_breakdown["total"] = serving_latency_ms
    audit_latency_breakdown = {
        "stage11_live": round(stage11_live_ms, 3),
        "fallback_demo": round(fallback_demo_ms, 3),
        "total": audit_latency_ms,
    }
    result["strategy_requested"] = requested_strategy
    result["strategy_used"] = strategy_used
    result["fallback_used"] = fallback_used
    result["fallback_reason"] = errors[0] if fallback_used and errors else ""
    result["serving_latency_breakdown_ms"] = latency_breakdown
    result["audit_latency_breakdown_ms"] = audit_latency_breakdown
    result["wall_latency_ms"] = wall_latency_ms
    result["latency_breakdown_ms"] = latency_breakdown
    result["serving_metrics"] = {
        "success": True,
        "latency_ms": serving_latency_ms,
        "wall_latency_ms": wall_latency_ms,
        "audit_latency_ms": audit_latency_ms,
        "latency_budget_ms": int(config.get("latency_budget_ms", 0)),
        "strategy_requested": requested_strategy,
        "strategy_used": strategy_used,
        "fallback_count": fallback_count,
        "stage11_live_status": str(result.get("stage11_live", {}).get("status", "")),
    }
    result["summary_metrics"]["fallback_count"] = fallback_count
    result["summary_metrics"]["latency_ms"] = serving_latency_ms
    result["summary_metrics"]["wall_latency_ms"] = wall_latency_ms
    result["summary_metrics"]["audit_latency_ms"] = audit_latency_ms
    return result


def print_text_report(result: dict[str, Any], input_label: str) -> None:
    print("Batch Inference Demo")
    print(f"- request_id: {result['request_id']}")
    print(f"- input: {input_label}")
    print(f"- release_id: {result['release_reference']['release_id']}")
    print(f"- mode: {result.get('mode', '')}")
    print(f"- strategy: requested={result['strategy_requested']} used={result['strategy_used']}")
    print(
        f"- latency_ms: serving={result['serving_metrics']['latency_ms']} "
        f"wall={result['serving_metrics'].get('wall_latency_ms')} "
        f"audit={result['serving_metrics'].get('audit_latency_ms')}"
    )
    execution_modes = result.get("execution_modes", {})
    if execution_modes:
        print(
            "- execution_modes: "
            f"stage09={execution_modes.get('stage09_used')} "
            f"stage10={execution_modes.get('stage10_used')} "
            f"stage11={execution_modes.get('stage11_used')}"
        )
    latency_breakdown = result.get("latency_breakdown_ms", {})
    if latency_breakdown:
        print(
            "- latency_breakdown_ms: "
            f"lookup={latency_breakdown.get('request_lookup')} "
            f"stage09={latency_breakdown.get('stage09')} "
            f"stage10={latency_breakdown.get('stage10')} "
            f"stage11={latency_breakdown.get('stage11')} "
            f"fallback_demo={latency_breakdown.get('fallback_demo')} "
            f"total={latency_breakdown.get('total')}"
        )
    request_context = result.get("request_context", {})
    if request_context:
        print(f"- replay_user_idx: {request_context.get('user_idx')}")
        print(f"- replay_note: {request_context.get('cohort_note')}")
        if request_context.get("top_city"):
            print(f"- replay_city: {request_context.get('top_city')}")
    user_state_snapshot = result.get("user_state_snapshot", {})
    if user_state_snapshot:
        if user_state_snapshot.get("long_term_top_cuisine"):
            print(f"- long_term_top_cuisine: {user_state_snapshot.get('long_term_top_cuisine')}")
        if user_state_snapshot.get("recent_intent_text"):
            print(f"- recent_intent: {user_state_snapshot.get('recent_intent_text')}")
    print("")
    print("Summary Metrics")
    for key, value in result["summary_metrics"].items():
        print(f"- {key}: {value}")
    print("")
    print("Top-K")
    for row in result["top_k"]:
        band = row.get("stage11_band") or "-"
        route = row.get("stage09_primary_route", "-")
        print(
            f"- rank={row['rank']} {row['name']} "
            f"(stage10_rank={row.get('stage10_rank')}, route={route}, "
            f"band={band}, final_score={row.get('final_score')})"
        )
        if row.get("why_recommended", {}).get("summary"):
            print(f"  reason: {row['why_recommended']['summary']}")
    print("")
    if result["rescued_candidates"]:
        print("Rescued Candidates")
        for row in result["rescued_candidates"]:
            print(
                f"- {row['name']}: stage10_rank={row['stage10_rank']} -> "
                f"final_rank={row['final_rank']} ({row['stage11_band']}, bonus={row['stage11_bonus']})"
            )
    else:
        print("Rescued Candidates")
        print("- none")
    print("")
    stage11_policy = result.get("stage11_policy", {})
    if stage11_policy:
        print("Stage11 Online Policy")
        print(
            f"- mode={stage11_policy.get('mode')} cache_status={stage11_policy.get('cache_status')} "
            f"applied={stage11_policy.get('applied_to_serving')} "
            f"live_rm_role={stage11_policy.get('live_rm_role')}"
        )
        backfill = stage11_policy.get("backfill_event", {})
        if backfill.get("enqueued"):
            print(f"- backfill: queue={backfill.get('queue')} reason={backfill.get('reason')}")
        print("")
    stage11_live = result.get("stage11_live", {})
    print("Stage11 Live")
    print(
        f"- enabled={stage11_live.get('enabled', False)} mode={stage11_live.get('mode', 'off')} "
        f"status={stage11_live.get('status', 'skipped')}"
    )
    if stage11_live.get("command_preview"):
        print(f"- command_preview: {stage11_live['command_preview']}")
    if stage11_live.get("reason"):
        print(f"- reason: {stage11_live['reason']}")
    if stage11_live.get("detail"):
        print(f"- detail: {stage11_live['detail']}")
    print("")
    truth_audit = result.get("offline_truth_audit", {})
    if truth_audit.get("available"):
        truth_item = truth_audit.get("truth_item", {})
        print("Offline Truth Audit")
        print(
            f"- truth={truth_item.get('name')} "
            f"(baseline_rank={truth_item.get('baseline_rank')}, "
            f"stage10_rank={truth_item.get('stage10_rank')}, "
            f"stage11_rank={truth_item.get('stage11_rank')})"
        )
        print(
            f"- top_k_hits: baseline={truth_item.get('in_top_k_by_baseline')} "
            f"stage10={truth_item.get('in_top_k_by_stage10')} "
            f"stage11={truth_item.get('in_top_k_by_stage11')}"
        )
        print("")
    fallback_demo = result.get("fallback_demo", {})
    if fallback_demo.get("available"):
        print("Fallback Demo")
        print(
            f"- requested={fallback_demo.get('requested_strategy')} "
            f"simulate_failure_for={fallback_demo.get('simulate_failure_for')} "
            f"used={fallback_demo.get('strategy_used')} "
            f"fallback_used={fallback_demo.get('fallback_used')}"
        )
        if fallback_demo.get("fallback_reason"):
            print(f"- reason: {fallback_demo.get('fallback_reason')}")
        print("")
    print("Release Reference")
    print(
        "- "
        f"stage09 truth_in_pretrim150={result['release_reference']['stage09_truth_in_pretrim150']:.4f}, "
        f"hard_miss={result['release_reference']['stage09_hard_miss']:.4f}"
    )
    print(
        "- "
        f"stage10 bucket5 learned recall/ndcg="
        f"{result['release_reference']['stage10_bucket5_learned_recall_at_10']:.4f} / "
        f"{result['release_reference']['stage10_bucket5_learned_ndcg_at_10']:.4f}"
    )
    print(
        "- "
        f"stage11 {result['release_reference']['stage11_tri_band_run']} "
        f"alpha={result['release_reference']['stage11_tri_band_alpha']:.2f}, "
        f"recall/ndcg={result['release_reference']['stage11_tri_band_recall_at_10']:.4f} / "
        f"{result['release_reference']['stage11_tri_band_ndcg_at_10']:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mock batch inference entry point for the current Yelp ranking stack."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="path to a small JSON request payload; ignored when --request-id is set",
    )
    parser.add_argument(
        "--request-id",
        default="",
        help="replay-first request id from the frozen Stage11 pack, e.g. stage11_b5_u000097",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format",
    )
    parser.add_argument(
        "--strategy",
        choices=("baseline", "xgboost", "reward_rerank"),
        default=None,
        help="override the serving strategy from config/serving.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="return a longer candidate_trace window in replay mode",
    )
    parser.add_argument(
        "--stage09-mode",
        choices=("replay", "lookup_live"),
        default=None,
        help="control Stage09 replay vs local per-user candidate lookup in replay mode",
    )
    parser.add_argument(
        "--stage10-mode",
        choices=("replay", "xgb_live"),
        default=None,
        help="control Stage10 replay vs local XGBoost scoring in replay mode",
    )
    parser.add_argument(
        "--stage11-mode",
        choices=("replay", "remote_dry_run", "remote_verify", "remote_worker"),
        default=None,
        help="control Stage11 replay vs optional remote verification in replay mode",
    )
    parser.add_argument(
        "--simulate-failure-for",
        choices=("baseline", "xgboost", "reward_rerank"),
        default=None,
        help="force a strategy failure to demonstrate fallback behavior",
    )
    parser.add_argument(
        "--simulate-stage11-cache-miss",
        action="store_true",
        help="force the Stage11 cache-first policy to serve Stage10 and enqueue RM backfill",
    )
    parser.add_argument(
        "--include-fallback-demo",
        action="store_true",
        help="attach a nested fallback demonstration in replay mode",
    )
    parser.add_argument(
        "--stage11-live-mode",
        choices=("off", "remote_dry_run", "remote_verify", "remote_worker"),
        default=None,
        help="control optional remote Stage11 live verification",
    )
    parser.add_argument(
        "--stage11-live-topn",
        type=int,
        default=None,
        help="max pre-rank window sent to remote Stage11 live verification",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.request_id:
        payload: dict[str, Any] = {
            "request_id": str(args.request_id).strip(),
            "debug": bool(args.debug),
        }
        input_label = f"replay:{payload['request_id']}"
    else:
        input_path = Path(args.input).expanduser().resolve()
        payload = read_json(input_path)
        input_label = str(input_path)
        if args.debug:
            payload["debug"] = True
    if args.strategy:
        payload["strategy"] = args.strategy
    if args.simulate_failure_for:
        payload["simulate_failure_for"] = args.simulate_failure_for
    if args.simulate_stage11_cache_miss:
        payload["simulate_stage11_cache_miss"] = True
    if args.include_fallback_demo:
        payload["include_fallback_demo"] = True
    if args.stage09_mode:
        payload["stage09_mode"] = args.stage09_mode
    if args.stage10_mode:
        payload["stage10_mode"] = args.stage10_mode
    if args.stage11_mode:
        payload["stage11_mode"] = args.stage11_mode
    if args.stage11_live_mode:
        payload["stage11_live_mode"] = args.stage11_live_mode
    if args.stage11_live_topn is not None:
        payload["stage11_live_topn"] = int(args.stage11_live_topn)

    result = rank_payload(payload)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print_text_report(result, input_label=input_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
