#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    "champion_pointer": "stage11_release",
    "aligned_fallback_pointer": "stage10_release",
    "emergency_baseline_pointer": "stage09_release",
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


def rank_payload(payload: dict[str, Any], serving_config: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config = serving_config or load_serving_config()
    release_ref = load_release_reference()
    requested_strategy = str(
        payload.get("strategy", payload.get("ranking_strategy", config.get("default_strategy", "reward_rerank")))
    ).strip()

    errors: list[str] = []
    result: dict[str, Any] | None = None
    strategy_used = ""
    for strategy in strategy_attempt_order(requested_strategy, config):
        try:
            result = rank_payload_for_strategy(payload, strategy, release_ref, config)
            strategy_used = strategy
            break
        except Exception as exc:
            errors.append(f"{strategy}: {exc}")

    if result is None:
        raise ValueError("; ".join(errors) or "ranking failed")

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    fallback_used = strategy_used != requested_strategy
    fallback_count = 1 if fallback_used else 0
    result["strategy_requested"] = requested_strategy
    result["strategy_used"] = strategy_used
    result["fallback_used"] = fallback_used
    result["fallback_reason"] = errors[0] if fallback_used and errors else ""
    result["serving_metrics"] = {
        "success": True,
        "latency_ms": latency_ms,
        "latency_budget_ms": int(config.get("latency_budget_ms", 0)),
        "strategy_requested": requested_strategy,
        "strategy_used": strategy_used,
        "fallback_count": fallback_count,
    }
    result["summary_metrics"]["fallback_count"] = fallback_count
    result["summary_metrics"]["latency_ms"] = latency_ms
    return result


def print_text_report(result: dict[str, Any], input_path: Path) -> None:
    print("Batch Inference Demo")
    print(f"- request_id: {result['request_id']}")
    print(f"- input: {input_path}")
    print(f"- release_id: {result['release_reference']['release_id']}")
    print(f"- strategy: requested={result['strategy_requested']} used={result['strategy_used']}")
    print(f"- latency_ms: {result['serving_metrics']['latency_ms']}")
    print("")
    print("Summary Metrics")
    for key, value in result["summary_metrics"].items():
        print(f"- {key}: {value}")
    print("")
    print("Top-K")
    for row in result["top_k"]:
        band = row["stage11_band"] or "-"
        print(
            f"- rank={row['rank']} {row['name']} "
            f"(stage10_rank={row['stage10_rank']}, route={row['stage09_primary_route']}, "
            f"band={band}, final_score={row['final_score']})"
        )
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
        help="path to a small JSON request payload",
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = read_json(input_path)
    if args.strategy:
        payload["strategy"] = args.strategy
    result = rank_payload(payload)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print_text_report(result, input_path=input_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
