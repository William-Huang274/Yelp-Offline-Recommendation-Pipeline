#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from batch_infer_demo import DEFAULT_INPUT_PATH, read_json  # noqa: E402
from load_test_mock_serving import (  # noqa: E402
    apply_traffic_profile,
    post_rank,
    run_one,
    select_request_ids,
    strip_to_replay_payload,
    summarize,
)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def value_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "min": round(min(values), 3) if values else 0.0,
        "p50": round(statistics.median(values), 3) if values else 0.0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3) if values else 0.0,
        "avg": round(statistics.mean(values), 3) if values else 0.0,
    }


def run_warmup(
    *,
    warmup_requests: int,
    base_payload: dict[str, Any],
    args: argparse.Namespace,
    request_ids: list[str],
) -> None:
    for index in range(1, max(0, int(warmup_requests)) + 1):
        run_one(
            -index,
            base_payload,
            args.strategy,
            0,
            args.url.strip() or None,
            bool(args.preserve_request_id),
            request_ids,
            str(args.traffic_profile),
            float(args.cache_miss_rate),
            float(args.strategy_failure_rate),
            float(args.xgboost_rate),
            float(args.baseline_rate),
            int(args.request_seed),
        )


def run_sustained(
    *,
    base_payload: dict[str, Any],
    args: argparse.Namespace,
    request_ids: list[str],
) -> tuple[list[dict[str, Any]], float]:
    duration_s = max(1.0, float(args.duration_s))
    concurrency = max(1, int(args.concurrency))
    max_requests = max(0, int(args.max_requests))
    url = args.url.strip() or None
    started = time.perf_counter()
    deadline = started + duration_s
    results: list[dict[str, Any]] = []
    next_index = 1

    def submit_one(executor: ThreadPoolExecutor, index: int):
        return executor.submit(
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

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = set()
        while len(futures) < concurrency and time.perf_counter() < deadline:
            if max_requests and next_index > max_requests:
                break
            futures.add(submit_one(executor, next_index))
            next_index += 1

        while futures:
            done, futures = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            while len(futures) < concurrency and time.perf_counter() < deadline:
                if max_requests and next_index > max_requests:
                    break
                futures.add(submit_one(executor, next_index))
                next_index += 1
            if max_requests and next_index > max_requests and not futures:
                break

    elapsed_s = time.perf_counter() - started
    return results, elapsed_s


def summarize_by_filter(results: list[dict[str, Any]], predicate) -> dict[str, float]:
    return value_summary([float(row.get("latency_ms", 0.0) or 0.0) for row in results if row.get("ok") and predicate(row)])


def build_engineering_summary(
    *,
    results: list[dict[str, Any]],
    elapsed_s: float,
    args: argparse.Namespace,
    request_ids: list[str],
) -> dict[str, Any]:
    base = summarize(
        results,
        requests=len(results),
        concurrency=max(1, int(args.concurrency)),
        strategy=args.strategy,
        url=args.url.strip() or None,
        warmup_requests=max(0, int(args.warmup_requests)),
        request_ids=request_ids,
        traffic_profile=str(args.traffic_profile),
    )
    ok_rows = [row for row in results if row.get("ok")]
    failed_rows = [row for row in results if not row.get("ok")]
    policy_fallback_rows = [
        row
        for row in ok_rows
        if str(row.get("stage11_mode", "")) == "stage10_fallback"
        or str(row.get("stage11_cache_status", "")).startswith("miss")
    ]
    cache_hit_rows = [
        row
        for row in ok_rows
        if str(row.get("strategy_requested", "")) == "reward_rerank"
        and str(row.get("strategy_used", "")) == "reward_rerank"
        and str(row.get("stage11_cache_status", "")) == "hit"
        and str(row.get("stage11_mode", "")) == "cache_first"
    ]
    strategy_fallback_rows = [row for row in ok_rows if row.get("fallback_used")]
    non_reward_rows = [row for row in ok_rows if str(row.get("strategy_requested", "")) != "reward_rerank"]

    cache_hit_latency = summarize_by_filter(results, lambda row: row in cache_hit_rows)
    policy_fallback_latency = summarize_by_filter(results, lambda row: row in policy_fallback_rows)
    strategy_fallback_latency = summarize_by_filter(results, lambda row: row in strategy_fallback_rows)
    non_reward_latency = summarize_by_filter(results, lambda row: row in non_reward_rows)

    baseline_p50 = cache_hit_latency["p50"]
    fallback_denominator = len(policy_fallback_rows) + len(strategy_fallback_rows) + len(failed_rows)
    fallback_recovery_rate = (
        round((len(policy_fallback_rows) + len(strategy_fallback_rows)) / fallback_denominator, 4)
        if fallback_denominator
        else 1.0
    )
    error_counts = Counter(str(row.get("error", "") or "unknown") for row in failed_rows)
    stage11_fallback_extra_p50 = (
        round(policy_fallback_latency["p50"] - baseline_p50, 3)
        if policy_fallback_latency["count"] and cache_hit_latency["count"]
        else 0.0
    )
    strategy_fallback_extra_p50 = (
        round(strategy_fallback_latency["p50"] - baseline_p50, 3)
        if strategy_fallback_latency["count"] and cache_hit_latency["count"]
        else 0.0
    )

    base["engineering"] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "duration_target_s": float(args.duration_s),
        "duration_observed_s": round(elapsed_s, 3),
        "throughput_rps": round(len(results) / max(elapsed_s, 1e-9), 3),
        "stage11_policy_fallback_count": len(policy_fallback_rows),
        "stage11_policy_fallback_rate": round(len(policy_fallback_rows) / max(1, len(ok_rows)), 4),
        "strategy_fallback_count": len(strategy_fallback_rows),
        "strategy_fallback_rate": round(len(strategy_fallback_rows) / max(1, len(ok_rows)), 4),
        "combined_fallback_recovery_rate": fallback_recovery_rate,
        "error_counts": dict(error_counts),
        "latency_groups_ms": {
            "reward_rerank_cache_hit": cache_hit_latency,
            "stage11_policy_fallback_to_stage10": policy_fallback_latency,
            "strategy_fallback_to_xgboost": strategy_fallback_latency,
            "non_reward_requested": non_reward_latency,
        },
        "fallback_extra_latency_p50_ms": {
            "stage11_policy_fallback_vs_cache_hit": stage11_fallback_extra_p50,
            "strategy_fallback_vs_cache_hit": strategy_fallback_extra_p50,
        },
        "scope": {
            "bucket": "bucket5",
            "stage09_mode": "lookup_live per-user parquet lookup when requested",
            "stage10_mode": "xgb_live CPU scoring when requested",
            "stage11_mode": "cache-first frozen reward-rerank scores; no GPU online inference",
            "cache_miss_semantics": "cache miss or uncovered Stage11 policy falls back to Stage10 and may enqueue backfill",
        },
    }
    return base


def render_report(summary: dict[str, Any]) -> str:
    eng = summary["engineering"]
    latency = summary["latency_ms"]
    wall = summary["wall_latency_ms"]
    groups = eng["latency_groups_ms"]
    stage = summary["stage_latency_ms"]

    def fmt_counts(counts: dict[str, Any]) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "-"

    lines = [
        "# Mock Serving Engineering Load Report",
        "",
        f"- Generated at: `{eng['generated_at_utc']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Duration target / observed: `{eng['duration_target_s']}s` / `{eng['duration_observed_s']}s`",
        f"- Requests: `{summary['requests']}` after `{summary['warmup_requests']}` warmup",
        f"- Concurrency: `{summary['concurrency']}`",
        f"- Throughput: `{eng['throughput_rps']}` req/s",
        f"- Unique replay request_ids: `{summary['unique_request_ids']}`",
        f"- Traffic profile: `{summary['traffic_profile']}`",
        f"- Requested strategy default: `{summary['strategy_requested']}`",
        "",
        "## SLA Signals",
        "",
        "| Signal | Observed |",
        "| --- | --- |",
        f"| success_rate | {summary['success_rate']:.2%} |",
        f"| serving_latency_p50 | {latency['p50']} ms |",
        f"| serving_latency_p95 | {latency['p95']} ms |",
        f"| serving_latency_p99 | {latency['p99']} ms |",
        f"| serving_latency_max | {latency['max']} ms |",
        f"| wall_latency_p95 | {wall['p95']} ms |",
        f"| combined_fallback_recovery_rate | {eng['combined_fallback_recovery_rate']:.2%} |",
        "",
        "## Fallback And Cache",
        "",
        f"- Strategy requested counts: `{fmt_counts(summary['strategy_requested_counts'])}`",
        f"- Strategy used counts: `{fmt_counts(summary['strategy_used_counts'])}`",
        f"- Stage mode counts: `stage09={summary['stage_mode_counts']['stage09']}`, `stage10={summary['stage_mode_counts']['stage10']}`, `stage11={summary['stage_mode_counts']['stage11']}`",
        f"- Source alignment: `stage09={summary['source_alignment_counts']['stage09']}`, `stage10={summary['source_alignment_counts']['stage10']}`",
        f"- Stage11 cache status counts: `{fmt_counts(summary['cache_status_counts'])}`",
        f"- Stage11 policy fallback to Stage10: `{eng['stage11_policy_fallback_count']}` ({eng['stage11_policy_fallback_rate']:.2%})",
        f"- Strategy fallback to XGBoost: `{eng['strategy_fallback_count']}` ({eng['strategy_fallback_rate']:.2%})",
        f"- Backfill count: `{summary['backfill_count']}`",
        "",
        "## Latency Groups",
        "",
        "| Group | Count | p50 | p95 | p99 | avg |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in groups.items():
        lines.append(f"| {name} | {item['count']} | {item['p50']} | {item['p95']} | {item['p99']} | {item['avg']} |")
    lines.extend(
        [
            "",
            "## Per-Stage P95",
            "",
            "| Stage | p95 | avg |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in ("request_lookup", "stage09", "stage10", "stage11", "offline_truth_audit", "fallback_demo"):
        item = stage[name]
        lines.append(f"| {name} | {item['p95']} | {item['avg']} |")
    lines.extend(
        [
            "",
            "## Error Distribution",
            "",
            f"- Error counts: `{fmt_counts(eng['error_counts'])}`",
            f"- First errors: `{summary['first_errors']}`",
            "",
            "## Known Limits",
            "",
            "- This is a CPU-only local replay/mock-serving validation, not production traffic or online A/B.",
            "- Stage09 is represented by per-user parquet lookup from a materialized candidate pack; it does not run full Spark recall recomputation per request.",
            "- Stage10 uses CPU XGBoost live scoring for bucket5 only in the current demo surface.",
            "- Stage11 uses frozen reward-rerank cache policy only; no GPU reward-model online inference is executed.",
            "- Cache misses simulate uncovered Stage11 policy and fall back to Stage10; broader bucket2/bucket10 live recompute is not wired in this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sustained engineering load report for the mock serving demo.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="JSON request payload")
    parser.add_argument("--request-id", default="", help="override input with a replay request_id")
    parser.add_argument("--request-sample-size", type=int, default=517, help="sample N replay request_ids")
    parser.add_argument("--request-seed", type=int, default=20260426)
    parser.add_argument("--preserve-request-id", action="store_true")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--max-requests", type=int, default=0, help="optional hard cap for debugging")
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--strategy", choices=("baseline", "xgboost", "reward_rerank"), default="reward_rerank")
    parser.add_argument("--simulate-fallback-every", type=int, default=0)
    parser.add_argument("--traffic-profile", choices=("fixed", "mixed"), default="mixed")
    parser.add_argument("--cache-miss-rate", type=float, default=0.15)
    parser.add_argument("--strategy-failure-rate", type=float, default=0.05)
    parser.add_argument("--xgboost-rate", type=float, default=0.10)
    parser.add_argument("--baseline-rate", type=float, default=0.00)
    parser.add_argument("--stage09-mode", choices=("replay", "lookup_live"), default="lookup_live")
    parser.add_argument("--stage10-mode", choices=("replay", "xgb_live"), default="xgb_live")
    parser.add_argument(
        "--stage11-mode",
        choices=("replay", "remote_dry_run", "remote_verify", "remote_worker"),
        default="replay",
    )
    parser.add_argument("--stage11-live-topn", type=int, default=None)
    parser.add_argument("--url", default="", help="optional HTTP /rank URL")
    parser.add_argument("--output-json", default="data/output/serving_validation/engineering_load_summary.json")
    parser.add_argument("--output-md", default="docs/serving_engineering_load_report.md")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.request_id.strip():
        base_payload = {"request_id": args.request_id.strip(), "top_k": 10}
        args.preserve_request_id = True
    else:
        base_payload = read_json(Path(args.input))
    base_payload = copy.deepcopy(base_payload)
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

    request_ids = select_request_ids(
        explicit_request_id=args.request_id,
        base_payload=base_payload,
        request_sample_size=max(0, int(args.request_sample_size)),
        seed=int(args.request_seed),
    )

    run_warmup(warmup_requests=args.warmup_requests, base_payload=base_payload, args=args, request_ids=request_ids)
    results, elapsed_s = run_sustained(base_payload=base_payload, args=args, request_ids=request_ids)
    summary = build_engineering_summary(results=results, elapsed_s=elapsed_s, args=args, request_ids=request_ids)

    json_path = Path(args.output_json).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path = Path(args.output_md).expanduser().resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_report(summary), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print("Mock Serving Engineering Load")
        print(f"- output_json: {json_path}")
        print(f"- output_md: {md_path}")
        print(f"- duration_observed_s: {summary['engineering']['duration_observed_s']}")
        print(f"- requests: {summary['requests']}")
        print(f"- throughput_rps: {summary['engineering']['throughput_rps']}")
        print(f"- success_rate: {summary['success_rate']}")
        print(
            "- latency_ms: "
            f"p50={summary['latency_ms']['p50']} "
            f"p95={summary['latency_ms']['p95']} "
            f"p99={summary['latency_ms']['p99']} "
            f"max={summary['latency_ms']['max']}"
        )
        print(
            "- fallback: "
            f"stage11_policy={summary['engineering']['stage11_policy_fallback_count']} "
            f"strategy={summary['engineering']['strategy_fallback_count']} "
            f"recovery={summary['engineering']['combined_fallback_recovery_rate']}"
        )
        print(f"- stage_mode_counts: {summary['stage_mode_counts']}")
        if summary["first_errors"]:
            print(f"- first_errors: {summary['first_errors']}")
    return 0 if summary["success_count"] == summary["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
