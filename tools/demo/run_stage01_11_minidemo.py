#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
TOOLS_ROOT = TOOLS_DIR.parents[0]
REPO_ROOT = TOOLS_DIR.parents[1]
SERVING_TOOLS_DIR = TOOLS_ROOT / "serving"
if str(SERVING_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(SERVING_TOOLS_DIR))

from batch_infer_demo import rank_payload, read_json


DEFAULT_INPUT_PATH = REPO_ROOT / "config" / "demo" / "full_chain_minimal_input.json"


def stage_record(stage: str, status: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {"stage": stage, "status": status, "summary": summary}


def activity_bucket(review_count: int) -> str:
    if review_count <= 3:
        return "bucket2_0_3_cold_start"
    if review_count <= 6:
        return "bucket2_4_6_light_user"
    if review_count <= 10:
        return "bucket5"
    return "bucket10"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.replace("/", " ").replace("-", " ").split() if len(token) > 2]


def dominant_cluster(cuisine_tags: list[str]) -> str:
    tags = {tag.lower() for tag in cuisine_tags}
    if tags & {"ramen", "japanese", "sushi", "bento", "izakaya"}:
        return "japanese_noodle"
    if tags & {"mexican", "tacos"}:
        return "mexican_quick_service"
    if tags & {"indian", "curry"}:
        return "south_asian_curry"
    return "general_restaurant"


def build_stage09_payload(payload: dict[str, Any], merged_businesses: list[dict[str, Any]]) -> dict[str, Any]:
    user = dict(payload["user"])
    reviews = payload.get("reviews", [])
    user["activity_bucket"] = activity_bucket(len(reviews))
    return {
        "request_id": payload.get("demo_id", "stage01_11_minimal_demo"),
        "strategy": "reward_rerank",
        "top_k": int(payload.get("top_k", 3)),
        "stage09_topn": max(int(payload.get("top_k", 3)), len(merged_businesses)),
        "stage11_topn": 30,
        "user_profile": user,
        "candidates": merged_businesses,
    }


def run_minidemo(payload: dict[str, Any]) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []

    user = payload["user"]
    businesses = payload.get("businesses", [])
    reviews = payload.get("reviews", [])
    truth = payload.get("holdout_truth", {})

    stages.append(
        stage_record(
            "stage01_ingest",
            "pass",
            {
                "users": 1,
                "businesses": len(businesses),
                "reviews": len(reviews),
                "contract": "raw Yelp-like rows normalized into in-memory demo tables",
            },
        )
    )

    positive_reviews = [row for row in reviews if int(row.get("stars", 0)) >= 4]
    review_count = len(reviews)
    stages.append(
        stage_record(
            "stage02_profile",
            "pass",
            {
                "user_id": user["user_id"],
                "activity_bucket": activity_bucket(review_count),
                "positive_reviews": len(positive_reviews),
            },
        )
    )

    baseline_candidates = sorted(businesses, key=lambda row: float(row.get("prescore", 0.0)), reverse=True)
    stages.append(
        stage_record(
            "stage03_baseline_recsys",
            "pass",
            {
                "candidate_rows": len(baseline_candidates),
                "top_prescore_business": baseline_candidates[0]["business_id"] if baseline_candidates else "",
            },
        )
    )

    truth_business = str(truth.get("business_id", ""))
    truth_in_pool = truth_business in {str(row.get("business_id", "")) for row in baseline_candidates}
    stages.append(
        stage_record(
            "stage04_validation",
            "pass" if truth_in_pool else "fail",
            {"truth_business": truth_business, "truth_in_candidate_pool": truth_in_pool},
        )
    )

    frozen_ids = [row["business_id"] for row in baseline_candidates]
    stages.append(
        stage_record(
            "stage05_freeze",
            "pass",
            {
                "frozen_candidate_ids": frozen_ids,
                "freeze_scope": "minimal demo fixture, not a production metric artifact",
            },
        )
    )

    token_counts: Counter[str] = Counter()
    for review in reviews:
        token_counts.update(tokenize(str(review.get("text", ""))))
    stages.append(
        stage_record(
            "stage06_text_insight",
            "pass",
            {"top_terms": [term for term, _ in token_counts.most_common(5)]},
        )
    )

    clustered_businesses = []
    for row in baseline_candidates:
        enriched = dict(row)
        enriched["semantic_cluster"] = dominant_cluster([str(tag) for tag in row.get("cuisine_tags", [])])
        clustered_businesses.append(enriched)
    stages.append(
        stage_record(
            "stage07_embedding_cluster",
            "pass",
            {
                "cluster_count": len({row["semantic_cluster"] for row in clustered_businesses}),
                "method": "deterministic tag-cluster stand-in for small demo",
            },
        )
    )

    merged_businesses = []
    for row in clustered_businesses:
        enriched = dict(row)
        enriched["profile_cluster_match"] = 1 if row["semantic_cluster"] == "japanese_noodle" else 0
        merged_businesses.append(enriched)
    stages.append(
        stage_record(
            "stage08_profile_merge",
            "pass",
            {
                "merged_candidates": len(merged_businesses),
                "profile_cluster_matches": sum(int(row["profile_cluster_match"]) for row in merged_businesses),
            },
        )
    )

    rank_request = build_stage09_payload(payload, merged_businesses)
    rank_result = rank_payload(rank_request)
    metrics = rank_result["summary_metrics"]
    stages.append(
        stage_record(
            "stage09_recall_routing",
            "pass",
            {
                "retained_candidates": metrics["stage09_retained_candidates"],
                "retention_ratio": metrics["stage09_retention_ratio"],
            },
        )
    )
    stages.append(
        stage_record(
            "stage10_structured_rerank",
            "pass",
            {
                "strategy_available": "xgboost",
                "demo_scoring": "lightweight structured formula aligned to Stage10 feature meaning",
                "top_business_after_stage10": min(rank_result["candidate_trace"], key=lambda row: row["stage10_rank"])["business_id"],
            },
        )
    )
    stages.append(
        stage_record(
            "stage11_reward_rerank",
            "pass",
            {
                "strategy_used": rank_result["strategy_used"],
                "rescued_candidates": metrics["stage11_rescued_candidates"],
                "top_k": [row["business_id"] for row in rank_result["top_k"]],
            },
        )
    )

    return {
        "demo_id": payload.get("demo_id", "stage01_11_minimal_demo"),
        "status": "pass" if all(stage["status"] == "pass" for stage in stages) else "fail",
        "scope": "contract-level minimal sample; full Spark/GPU training is intentionally not run",
        "stages": stages,
        "rank_result": {
            "request_id": rank_result["request_id"],
            "strategy_used": rank_result["strategy_used"],
            "serving_metrics": rank_result["serving_metrics"],
            "top_k": rank_result["top_k"],
        },
    }


def print_text(result: dict[str, Any]) -> None:
    print("Stage01-Stage11 Minimal Demo")
    print(f"- demo_id: {result['demo_id']}")
    print(f"- status: {result['status']}")
    print(f"- scope: {result['scope']}")
    for stage in result["stages"]:
        print(f"- {stage['stage']}: {stage['status']} {stage['summary']}")
    print(f"- final_top_k: {[row['business_id'] for row in result['rank_result']['top_k']]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a contract-level Stage01 -> Stage11 minimal demo.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="minimal JSON fixture")
    parser.add_argument("--output", default="", help="optional JSON output path")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = read_json(Path(args.input).expanduser().resolve())
    result = run_minidemo(payload)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print_text(result)
    if result["status"] == "pass":
        print("PASS stage01_11_minidemo")
        return 0
    print("FAIL stage01_11_minidemo")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
