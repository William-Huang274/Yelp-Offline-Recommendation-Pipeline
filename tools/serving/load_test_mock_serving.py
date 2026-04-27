#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from batch_infer_demo import DEFAULT_INPUT_PATH, rank_payload, read_json
from replay_store import load_replay_store


STAGE_LATENCY_KEYS = (
    "request_lookup",
    "stage09",
    "stage10",
    "stage11",
    "offline_truth_audit",
    "fallback_demo",
)
REPLAY_PAYLOAD_KEYS = (
    "top_k",
    "debug",
    "stage09_mode",
    "stage10_mode",
    "stage11_mode",
    "stage11_live_mode",
    "stage11_live_topn",
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def post_rank(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def clamp_rate(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def select_request_ids(
    *,
    explicit_request_id: str,
    base_payload: dict[str, Any],
    request_sample_size: int,
    seed: int,
) -> list[str]:
    if explicit_request_id.strip():
        return [explicit_request_id.strip()]
    payload_request_id = str(base_payload.get("request_id", "") or "").strip()
    if request_sample_size <= 0 and payload_request_id:
        return [payload_request_id]
    store = load_replay_store()
    request_ids = list(store.request_ids)
    rng = random.Random(int(seed))
    rng.shuffle(request_ids)
    if request_sample_size > 0:
        request_ids = request_ids[: min(int(request_sample_size), len(request_ids))]
    return request_ids or [store.sample_request_id]


def strip_to_replay_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep serving controls while removing hand-written candidate lists."""
    return {key: payload[key] for key in REPLAY_PAYLOAD_KEYS if key in payload}


def apply_traffic_profile(
    *,
    payload: dict[str, Any],
    index: int,
    strategy: str,
    traffic_profile: str,
    cache_miss_rate: float,
    strategy_failure_rate: float,
    xgboost_rate: float,
    baseline_rate: float,
    rng: random.Random,
) -> str:
    selected_strategy = strategy
    if traffic_profile == "mixed":
        draw = rng.random()
        baseline_cut = clamp_rate(baseline_rate)
        xgboost_cut = baseline_cut + clamp_rate(xgboost_rate)
        failure_cut = xgboost_cut + clamp_rate(strategy_failure_rate)
        miss_cut = failure_cut + clamp_rate(cache_miss_rate)
        if draw < baseline_cut:
            selected_strategy = "baseline"
        elif draw < xgboost_cut:
            selected_strategy = "xgboost"
        elif draw < failure_cut:
            selected_strategy = "reward_rerank"
            payload["simulate_failure_for"] = "reward_rerank"
        elif draw < miss_cut:
            selected_strategy = "reward_rerank"
            payload["simulate_stage11_cache_miss"] = True
        else:
            selected_strategy = "reward_rerank"
    payload["strategy"] = selected_strategy
    payload["traffic_profile"] = traffic_profile
    payload["traffic_index"] = int(index)
    return selected_strategy


def run_one(
    index: int,
    base_payload: dict[str, Any],
    strategy: str,
    simulate_fallback_every: int,
    url: str | None,
    preserve_request_id: bool,
    request_ids: list[str],
    traffic_profile: str,
    cache_miss_rate: float,
    strategy_failure_rate: float,
    xgboost_rate: float,
    baseline_rate: float,
    seed: int,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    request_index = abs(index) if index < 0 else index
    if request_ids:
        payload["request_id"] = request_ids[(max(1, request_index) - 1) % len(request_ids)]
    elif not preserve_request_id or not payload.get("request_id"):
        payload["request_id"] = f"load_test_{index:05d}"
    traffic_rng = random.Random(int(seed) + int(index) * 1009)
    requested_strategy = apply_traffic_profile(
        payload=payload,
        index=index,
        strategy=strategy,
        traffic_profile=traffic_profile,
        cache_miss_rate=cache_miss_rate,
        strategy_failure_rate=strategy_failure_rate,
        xgboost_rate=xgboost_rate,
        baseline_rate=baseline_rate,
        rng=traffic_rng,
    )
    if simulate_fallback_every > 0 and index % simulate_fallback_every == 0:
        payload["simulate_failure_for"] = requested_strategy

    started = time.perf_counter()
    try:
        if url:
            result = post_rank(url, payload)
        else:
            result = rank_payload(payload)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        serving_metrics = result.get("serving_metrics", {})
        stage_breakdown = result.get("serving_latency_breakdown_ms", result.get("latency_breakdown_ms", {}))
        audit_breakdown = result.get("audit_latency_breakdown_ms", {})
        execution_modes = result.get("execution_modes", {})
        stage11_policy = result.get("stage11_policy", {})
        stage09_trace = result.get("stage09_trace", {})
        stage10_summary = result.get("stage10_summary", {})
        return {
            "ok": True,
            "request_id": str(payload.get("request_id", "")),
            "strategy_requested": str(result.get("strategy_requested", requested_strategy)),
            "latency_ms": float(serving_metrics.get("latency_ms", elapsed_ms)),
            "wall_latency_ms": elapsed_ms,
            "audit_latency_ms": float(serving_metrics.get("audit_latency_ms", 0.0) or 0.0),
            "stage_latencies_ms": {
                key: float(stage_breakdown.get(key, 0.0) or 0.0) for key in STAGE_LATENCY_KEYS
            },
            "audit_breakdown_ms": {
                "stage11_live": float(audit_breakdown.get("stage11_live", 0.0) or 0.0),
                "fallback_demo": float(audit_breakdown.get("fallback_demo", 0.0) or 0.0),
            },
            "fallback_count": int(serving_metrics.get("fallback_count", 0)),
            "strategy_used": str(result.get("strategy_used", "")),
            "fallback_used": bool(result.get("fallback_used", False)),
            "stage09_mode": str(execution_modes.get("stage09_used", "")),
            "stage10_mode": str(execution_modes.get("stage10_used", "")),
            "stage11_mode": str(execution_modes.get("stage11_used", "")),
            "stage09_source_alignment": str(stage09_trace.get("source_alignment", "")),
            "stage10_source_alignment": str(stage10_summary.get("source_alignment", "")),
            "stage11_cache_status": str(stage11_policy.get("cache_status", "")),
            "stage11_cache_hit": bool(stage11_policy.get("cache_hit", False)),
            "stage11_applied_to_serving": bool(stage11_policy.get("applied_to_serving", False)),
            "backfill_enqueued": bool(stage11_policy.get("backfill_event", {}).get("enqueued", False)),
            "stage11_live_status": str(result.get("stage11_live", {}).get("status", "")),
            "error": "",
        }
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return {
            "ok": False,
            "request_id": str(payload.get("request_id", "")),
            "strategy_requested": requested_strategy,
            "latency_ms": elapsed_ms,
            "wall_latency_ms": elapsed_ms,
            "audit_latency_ms": 0.0,
            "stage_latencies_ms": {key: 0.0 for key in STAGE_LATENCY_KEYS},
            "audit_breakdown_ms": {"stage11_live": 0.0, "fallback_demo": 0.0},
            "fallback_count": 0,
            "strategy_used": "",
            "fallback_used": False,
            "stage09_mode": "",
            "stage10_mode": "",
            "stage11_mode": "",
            "stage09_source_alignment": "",
            "stage10_source_alignment": "",
            "stage11_cache_status": "",
            "stage11_cache_hit": False,
            "stage11_applied_to_serving": False,
            "backfill_enqueued": False,
            "stage11_live_status": "",
            "error": str(exc),
        }


def summarize(
    results: list[dict[str, Any]],
    requests: int,
    concurrency: int,
    strategy: str,
    url: str | None,
    warmup_requests: int,
    request_ids: list[str],
    traffic_profile: str,
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in results if row["ok"]]
    wall_latencies = [float(row["wall_latency_ms"]) for row in results if row["ok"]]
    audit_latencies = [float(row["audit_latency_ms"]) for row in results if row["ok"]]
    success_count = sum(1 for row in results if row["ok"])
    fallback_count = sum(int(row["fallback_count"]) for row in results)
    cache_hit_count = sum(1 for row in results if row.get("stage11_cache_hit"))
    cache_miss_count = sum(1 for row in results if str(row.get("stage11_cache_status", "")).startswith("miss"))
    backfill_count = sum(1 for row in results if row.get("backfill_enqueued"))
    strategy_counts: dict[str, int] = {}
    requested_strategy_counts: dict[str, int] = {}
    stage_mode_counts: dict[str, dict[str, int]] = {"stage09": {}, "stage10": {}, "stage11": {}}
    source_alignment_counts: dict[str, dict[str, int]] = {"stage09": {}, "stage10": {}}
    cache_status_counts: dict[str, int] = {}
    for row in results:
        used = str(row["strategy_used"] or "failed")
        strategy_counts[used] = strategy_counts.get(used, 0) + 1
        requested = str(row.get("strategy_requested", "") or "unknown")
        requested_strategy_counts[requested] = requested_strategy_counts.get(requested, 0) + 1
        for stage_name, field in (("stage09", "stage09_mode"), ("stage10", "stage10_mode"), ("stage11", "stage11_mode")):
            mode = str(row.get(field, "") or "unknown")
            stage_mode_counts[stage_name][mode] = stage_mode_counts[stage_name].get(mode, 0) + 1
        for stage_name, field in (("stage09", "stage09_source_alignment"), ("stage10", "stage10_source_alignment")):
            value = str(row.get(field, "") or "unknown")
            source_alignment_counts[stage_name][value] = source_alignment_counts[stage_name].get(value, 0) + 1
        cache_status = str(row.get("stage11_cache_status", "") or "unknown")
        cache_status_counts[cache_status] = cache_status_counts.get(cache_status, 0) + 1
    first_errors = [str(row["error"]) for row in results if not row["ok"]][:3]

    def summarize_values(values: list[float]) -> dict[str, float]:
        return {
            "min": round(min(values), 3) if values else 0.0,
            "p50": round(statistics.median(values), 3) if values else 0.0,
            "p95": round(percentile(values, 0.95), 3),
            "p99": round(percentile(values, 0.99), 3),
            "max": round(max(values), 3) if values else 0.0,
            "avg": round(statistics.mean(values), 3) if values else 0.0,
        }

    stage_latency_ms = {
        key: summarize_values([float(row.get("stage_latencies_ms", {}).get(key, 0.0)) for row in results if row["ok"]])
        for key in STAGE_LATENCY_KEYS
    }
    audit_breakdown_ms = {
        key: summarize_values([float(row.get("audit_breakdown_ms", {}).get(key, 0.0)) for row in results if row["ok"]])
        for key in ("stage11_live", "fallback_demo")
    }
    return {
        "mode": "http" if url else "in_process",
        "url": url or "",
        "requests": requests,
        "warmup_requests": warmup_requests,
        "concurrency": concurrency,
        "unique_request_ids": len(set(request_ids)) if request_ids else 0,
        "request_id_preview": request_ids[:10],
        "traffic_profile": traffic_profile,
        "strategy_requested": strategy,
        "success_count": success_count,
        "success_rate": round(success_count / max(1, requests), 4),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / max(1, success_count), 4),
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_hit_rate": round(cache_hit_count / max(1, success_count), 4),
        "backfill_count": backfill_count,
        "strategy_used_counts": strategy_counts,
        "strategy_requested_counts": requested_strategy_counts,
        "stage_mode_counts": stage_mode_counts,
        "source_alignment_counts": source_alignment_counts,
        "cache_status_counts": cache_status_counts,
        "latency_ms": summarize_values(latencies),
        "wall_latency_ms": summarize_values(wall_latencies),
        "audit_latency_ms": summarize_values(audit_latencies),
        "stage_latency_ms": stage_latency_ms,
        "audit_breakdown_ms": audit_breakdown_ms,
        "first_errors": first_errors,
    }


def print_text(summary: dict[str, Any]) -> None:
    print("Mock Serving Load Test")
    print(f"- mode: {summary['mode']}")
    print(f"- warmup_requests: {summary['warmup_requests']}")
    print(f"- requests: {summary['requests']}")
    print(f"- unique_request_ids: {summary['unique_request_ids']}")
    print(f"- traffic_profile: {summary['traffic_profile']}")
    print(f"- concurrency: {summary['concurrency']}")
    print(f"- strategy_requested: {summary['strategy_requested']}")
    print(f"- success_rate: {summary['success_rate']}")
    print(f"- fallback_count: {summary['fallback_count']}")
    print(f"- cache_hit_rate: {summary['cache_hit_rate']}")
    print(f"- backfill_count: {summary['backfill_count']}")
    print(f"- strategy_used_counts: {summary['strategy_used_counts']}")
    print(f"- stage_mode_counts: {summary['stage_mode_counts']}")
    print(
        "- latency_ms: "
        f"p50={summary['latency_ms']['p50']} "
        f"p95={summary['latency_ms']['p95']} "
        f"p99={summary['latency_ms']['p99']} "
        f"max={summary['latency_ms']['max']}"
    )
    print(
        "- audit_latency_ms: "
        f"p50={summary['audit_latency_ms']['p50']} "
        f"p95={summary['audit_latency_ms']['p95']}"
    )
    print(
        "- per_stage_p95_ms: "
        + " ".join(f"{key}={summary['stage_latency_ms'][key]['p95']}" for key in STAGE_LATENCY_KEYS)
    )
    if summary["first_errors"]:
        print(f"- first_errors: {summary['first_errors']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small local load test for the mock ranking service.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="JSON request payload")
    parser.add_argument("--request-id", default="", help="override input with a replay request_id")
    parser.add_argument("--request-sample-size", type=int, default=0, help="sample N replay request_ids")
    parser.add_argument("--request-seed", type=int, default=20260426, help="deterministic request sampling seed")
    parser.add_argument(
        "--preserve-request-id",
        action="store_true",
        help="keep the same request_id for every request, useful for replay/remote-worker tests",
    )
    parser.add_argument("--requests", type=int, default=20, help="number of requests")
    parser.add_argument("--warmup-requests", type=int, default=0, help="serial warmup requests excluded from metrics")
    parser.add_argument("--concurrency", type=int, default=4, help="worker threads")
    parser.add_argument(
        "--strategy",
        choices=("baseline", "xgboost", "reward_rerank"),
        default="reward_rerank",
        help="serving strategy to request",
    )
    parser.add_argument(
        "--simulate-fallback-every",
        type=int,
        default=0,
        help="simulate a requested-strategy failure every N requests to verify fallback counters",
    )
    parser.add_argument("--traffic-profile", choices=("fixed", "mixed"), default="fixed")
    parser.add_argument("--cache-miss-rate", type=float, default=0.1, help="mixed profile Stage11 cache miss rate")
    parser.add_argument(
        "--strategy-failure-rate",
        type=float,
        default=0.05,
        help="mixed profile reward_rerank failure fallback rate",
    )
    parser.add_argument("--xgboost-rate", type=float, default=0.05, help="mixed profile xgboost-only traffic rate")
    parser.add_argument("--baseline-rate", type=float, default=0.0, help="mixed profile baseline-only traffic rate")
    parser.add_argument("--stage09-mode", choices=("replay", "lookup_live"), default=None)
    parser.add_argument("--stage10-mode", choices=("replay", "xgb_live"), default=None)
    parser.add_argument("--stage11-mode", choices=("replay", "remote_dry_run", "remote_verify", "remote_worker"), default=None)
    parser.add_argument("--stage11-live-topn", type=int, default=None)
    parser.add_argument("--url", default="", help="optional HTTP /rank URL, e.g. http://127.0.0.1:8000/rank")
    parser.add_argument("--output", default="", help="optional JSON summary output path")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.request_id.strip():
        base_payload = {"request_id": args.request_id.strip(), "top_k": 10}
        args.preserve_request_id = True
    else:
        base_payload = read_json(Path(args.input).expanduser().resolve())
    if args.stage09_mode:
        base_payload["stage09_mode"] = args.stage09_mode
    if args.stage10_mode:
        base_payload["stage10_mode"] = args.stage10_mode
    if args.stage11_mode:
        base_payload["stage11_mode"] = args.stage11_mode
    if args.stage11_live_topn is not None:
        base_payload["stage11_live_topn"] = int(args.stage11_live_topn)
    if int(args.request_sample_size) > 0 and isinstance(base_payload.get("candidates"), list):
        base_payload = strip_to_replay_payload(base_payload)
    requests = max(1, int(args.requests))
    warmup_requests = max(0, int(args.warmup_requests))
    concurrency = max(1, int(args.concurrency))
    url = args.url.strip() or None
    request_ids = select_request_ids(
        explicit_request_id=args.request_id,
        base_payload=base_payload,
        request_sample_size=max(0, int(args.request_sample_size)),
        seed=int(args.request_seed),
    )

    for index in range(1, warmup_requests + 1):
        run_one(
            -index,
            base_payload,
            args.strategy,
            0,
            url,
            bool(args.preserve_request_id),
            request_ids,
            str(args.traffic_profile),
            float(args.cache_miss_rate),
            float(args.strategy_failure_rate),
            float(args.xgboost_rate),
            float(args.baseline_rate),
            int(args.request_seed),
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                index,
                base_payload,
                args.strategy,
                args.simulate_fallback_every,
                url,
                bool(args.preserve_request_id),
                request_ids,
                str(args.traffic_profile),
                float(args.cache_miss_rate),
                float(args.strategy_failure_rate),
                float(args.xgboost_rate),
                float(args.baseline_rate),
                int(args.request_seed),
            )
            for index in range(1, requests + 1)
        ]
        results = [future.result() for future in as_completed(futures)]

    summary = summarize(
        results,
        requests=requests,
        concurrency=concurrency,
        strategy=args.strategy,
        url=url,
        warmup_requests=warmup_requests,
        request_ids=request_ids,
        traffic_profile=str(args.traffic_profile),
    )
    if args.output.strip():
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print_text(summary)
    return 0 if summary["success_count"] == requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
