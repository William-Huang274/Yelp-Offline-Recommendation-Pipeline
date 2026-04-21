#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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


def run_one(
    index: int,
    base_payload: dict[str, Any],
    strategy: str,
    simulate_fallback_every: int,
    url: str | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["request_id"] = f"load_test_{index:05d}"
    payload["strategy"] = strategy
    if simulate_fallback_every > 0 and index % simulate_fallback_every == 0:
        payload["simulate_failure_for"] = strategy

    started = time.perf_counter()
    try:
        if url:
            result = post_rank(url, payload)
        else:
            result = rank_payload(payload)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        serving_metrics = result.get("serving_metrics", {})
        return {
            "ok": True,
            "latency_ms": float(serving_metrics.get("latency_ms", elapsed_ms)),
            "wall_latency_ms": elapsed_ms,
            "fallback_count": int(serving_metrics.get("fallback_count", 0)),
            "strategy_used": str(result.get("strategy_used", "")),
            "error": "",
        }
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return {
            "ok": False,
            "latency_ms": elapsed_ms,
            "wall_latency_ms": elapsed_ms,
            "fallback_count": 0,
            "strategy_used": "",
            "error": str(exc),
        }


def summarize(results: list[dict[str, Any]], requests: int, concurrency: int, strategy: str, url: str | None) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in results if row["ok"]]
    wall_latencies = [float(row["wall_latency_ms"]) for row in results if row["ok"]]
    success_count = sum(1 for row in results if row["ok"])
    fallback_count = sum(int(row["fallback_count"]) for row in results)
    strategy_counts: dict[str, int] = {}
    for row in results:
        used = str(row["strategy_used"] or "failed")
        strategy_counts[used] = strategy_counts.get(used, 0) + 1
    first_errors = [str(row["error"]) for row in results if not row["ok"]][:3]
    return {
        "mode": "http" if url else "in_process",
        "url": url or "",
        "requests": requests,
        "concurrency": concurrency,
        "strategy_requested": strategy,
        "success_count": success_count,
        "success_rate": round(success_count / max(1, requests), 4),
        "fallback_count": fallback_count,
        "strategy_used_counts": strategy_counts,
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "wall_latency_ms": {
            "p50": round(statistics.median(wall_latencies), 3) if wall_latencies else 0.0,
            "p95": round(percentile(wall_latencies, 0.95), 3),
        },
        "first_errors": first_errors,
    }


def print_text(summary: dict[str, Any]) -> None:
    print("Mock Serving Load Test")
    print(f"- mode: {summary['mode']}")
    print(f"- requests: {summary['requests']}")
    print(f"- concurrency: {summary['concurrency']}")
    print(f"- strategy_requested: {summary['strategy_requested']}")
    print(f"- success_rate: {summary['success_rate']}")
    print(f"- fallback_count: {summary['fallback_count']}")
    print(f"- strategy_used_counts: {summary['strategy_used_counts']}")
    print(
        "- latency_ms: "
        f"p50={summary['latency_ms']['p50']} "
        f"p95={summary['latency_ms']['p95']} "
        f"p99={summary['latency_ms']['p99']} "
        f"max={summary['latency_ms']['max']}"
    )
    if summary["first_errors"]:
        print(f"- first_errors: {summary['first_errors']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small local load test for the mock ranking service.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="JSON request payload")
    parser.add_argument("--requests", type=int, default=20, help="number of requests")
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
    parser.add_argument("--url", default="", help="optional HTTP /rank URL, e.g. http://127.0.0.1:8000/rank")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_payload = read_json(Path(args.input).expanduser().resolve())
    requests = max(1, int(args.requests))
    concurrency = max(1, int(args.concurrency))
    url = args.url.strip() or None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_one, index, base_payload, args.strategy, args.simulate_fallback_every, url)
            for index in range(1, requests + 1)
        ]
        results = [future.result() for future in as_completed(futures)]

    summary = summarize(results, requests=requests, concurrency=concurrency, strategy=args.strategy, url=url)
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print_text(summary)
    return 0 if summary["success_count"] == requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
