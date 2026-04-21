#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RELEASE = REPO_ROOT / "data" / "output" / "current_release"
CURRENT_METRICS = REPO_ROOT / "data" / "metrics" / "current_release"


CASEBOOK: dict[str, dict[str, object]] = {
    "boundary_11_30": {
        "title": "Boundary rescue into top10",
        "user_idx": 1072,
        "item_idx": 58,
        "learned_rank": 17,
        "blend_rank": 4,
        "final_rank": 8,
        "route_band": "boundary_11_30",
        "reward_score": 13.4375,
        "rescue_bonus": 0.7579,
        "why": [
            "Stage10 already placed the truth near the front boundary.",
            "Stage11 resolved a difficult local comparison among nearby rivals.",
            "This case explains why the boundary lane is a good fit for local rescue rather than full-list reranking.",
        ],
        "doc": "docs/stage11/stage11_case_notes_20260409.md",
    },
    "mid_31_40": {
        "title": "Mid-band rescue promoted to the front rank",
        "user_idx": 1940,
        "item_idx": 92,
        "learned_rank": 36,
        "blend_rank": 1,
        "final_rank": 2,
        "route_band": "rescue_31_40",
        "reward_score": 11.3125,
        "rescue_bonus": 1.0350,
        "why": [
            "The truth candidate started in the mid rescue band rather than near the top.",
            "Stage11 evaluated it against local competitors instead of reranking the whole list.",
            "Controlled release rules allowed a strong local winner to enter the final front rank.",
        ],
        "doc": "docs/stage11/stage11_case_notes_20260409.md",
    },
    "deep_61_100_policy": {
        "title": "Conservative deep-band policy",
        "user_idx": None,
        "item_idx": None,
        "learned_rank": None,
        "blend_rank": None,
        "final_rank": None,
        "route_band": "deep_61_100",
        "reward_score": None,
        "rescue_bonus": None,
        "why": [
            "The 61-100 expert is trained and validated, but the current freeze keeps it conservative.",
            "Its value is rank uplift rather than aggressive top-rank takeover in the current line.",
            "This preserves stability while keeping room for future policy expansion.",
        ],
        "doc": "docs/stage11/stage11_case_notes_20260409.md",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_stage09_summary() -> dict[str, object]:
    payload = read_json(CURRENT_RELEASE / "stage09" / "bucket5_route_aware_sourceparity" / "summary.json")
    audit = read_json(CURRENT_RELEASE / "stage09" / "bucket5_route_aware_sourceparity" / "stage09_recall_audit.json")
    row = audit["rows"][0]
    return {
        "line": payload["line"],
        "truth_in_pretrim150": float(payload["current_metrics"]["truth_in_pretrim150"]),
        "hard_miss": float(payload["current_metrics"]["hard_miss"]),
        "n_users": int(row["n_users"]),
        "cand_pretrim_avg": float(row["cand_pretrim_avg"]),
        "profile_routes_enabled": str(row["profile_routes_enabled"]),
    }


def load_stage10_summary() -> list[dict[str, object]]:
    payload = read_json(CURRENT_RELEASE / "stage10" / "stage10_current_mainline_summary.json")
    return payload["bucket_snapshots"]


def load_stage11_summary() -> dict[str, object]:
    two_band = read_json(CURRENT_RELEASE / "stage11" / "eval" / "bucket5_two_band_best_known_v120_alpha080.json")
    tri_band = read_json(CURRENT_RELEASE / "stage11" / "eval" / "bucket5_tri_band_freeze_v124_alpha036" / "summary.json")
    experts = read_json(CURRENT_RELEASE / "stage11" / "experts" / "expert_training_summary.json")
    return {
        "two_band": two_band,
        "tri_band": tri_band,
        "experts": experts["experts"],
    }


def print_summary() -> None:
    stage09 = load_stage09_summary()
    stage10 = load_stage10_summary()
    stage11 = load_stage11_summary()

    print("Current Frozen Yelp Ranking Review Line")
    print("")
    print("Stage09 - Route-aware recall")
    print(f"- line: {stage09['line']}")
    print(f"- truth_in_pretrim150: {stage09['truth_in_pretrim150']:.4f}")
    print(f"- hard_miss: {stage09['hard_miss']:.4f}")
    print(f"- eval users: {stage09['n_users']}")
    print(f"- avg pretrim candidates per user: {stage09['cand_pretrim_avg']:.1f}")
    print(f"- enabled profile routes: {stage09['profile_routes_enabled']}")
    print("")
    print("Stage10 - Structured rerank")
    for row in stage10:
        print(
            "- "
            f"{row['bucket']}: pre={float(row['prescore_recall_at_10']):.4f} / {float(row['prescore_ndcg_at_10']):.4f}, "
            f"learned={float(row['learned_recall_at_10']):.4f} / {float(row['learned_ndcg_at_10']):.4f}"
        )
    print("")
    print("Stage11 - Reward-model rescue rerank")
    for expert in stage11["experts"]:
        band = str(expert["expert_band"])
        if band == "11-30":
            signal = f"true_win_11_30={float(expert['true_win_11_30']):.4f}"
        elif band == "31-60":
            signal = (
                f"true_win_31_40={float(expert['true_win_31_40']):.4f}, "
                f"true_win_41_60={float(expert['true_win_41_60']):.4f}"
            )
        else:
            signal = f"true_win_61_100={float(expert['true_win_61_100']):.4f}"
        print(f"- expert {band}: {signal}")
    print(
        "- two-band best-known line: "
        f"alpha={float(stage11['two_band']['alpha']):.2f}, "
        f"recall@10={float(stage11['two_band']['recall_at_10']):.4f}, "
        f"ndcg@10={float(stage11['two_band']['ndcg_at_10']):.4f}"
    )
    print(
        "- tri-band freeze line: "
        f"alpha={float(stage11['tri_band']['alpha']):.2f}, "
        f"recall@10={float(stage11['tri_band']['recall_at_10']):.4f}, "
        f"ndcg@10={float(stage11['tri_band']['ndcg_at_10']):.4f}"
    )
    print("")
    print("Next demo commands")
    print("- python tools/demo_recommend.py list-cases")
    print("- python tools/demo_recommend.py show-case --case boundary_11_30")


def list_cases() -> None:
    print("Available demo cases")
    for case_id, payload in CASEBOOK.items():
        user_idx = payload["user_idx"]
        user_text = f"user_idx={user_idx}" if user_idx is not None else "policy-only"
        print(f"- {case_id}: {payload['title']} ({user_text})")


def resolve_case(case_id: str | None, user_idx: int | None) -> tuple[str, dict[str, object]]:
    if case_id:
        if case_id not in CASEBOOK:
            raise KeyError(f"unknown case: {case_id}")
        return case_id, CASEBOOK[case_id]

    if user_idx is not None:
        for candidate_id, payload in CASEBOOK.items():
            if payload.get("user_idx") == user_idx:
                return candidate_id, payload
        raise KeyError(f"no demo case found for user_idx={user_idx}")

    raise KeyError("either --case or --user-idx must be provided")


def show_case(case_id: str | None, user_idx: int | None) -> None:
    resolved_id, payload = resolve_case(case_id=case_id, user_idx=user_idx)
    print(f"Case: {resolved_id}")
    print(f"Title: {payload['title']}")
    if payload["user_idx"] is not None:
        print(f"- user_idx: {payload['user_idx']}")
        print(f"- item_idx: {payload['item_idx']}")
        print(f"- learned_rank: {payload['learned_rank']}")
        print(f"- blend_rank: {payload['blend_rank']}")
        print(f"- final_rank: {payload['final_rank']}")
        print(f"- route_band: {payload['route_band']}")
        print(f"- reward_score: {payload['reward_score']}")
        print(f"- rescue_bonus: {payload['rescue_bonus']}")
    else:
        print(f"- route_band: {payload['route_band']}")
    print("")
    print("Why it matters")
    for bullet in payload["why"]:
        print(f"- {bullet}")
    print("")
    print(f"Reference doc: {payload['doc']}")


def print_walkthrough() -> None:
    print("Suggested live walkthrough")
    print("- python tools/run_release_checks.py --skip-pytest")
    print("- python tools/demo_recommend.py summary")
    print("- python tools/demo_recommend.py list-cases")
    print("- python tools/demo_recommend.py show-case --case boundary_11_30")
    print("- python tools/demo_recommend.py show-case --case mid_31_40")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Review and demo helper for the current frozen Yelp ranking line. "
            "When no command is provided, it prints the summary."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("summary", help="print the current Stage09/10/11 frozen summary")
    sub.add_parser("list-cases", help="list canonical demo cases")
    show_parser = sub.add_parser("show-case", help="print one canonical case")
    show_parser.add_argument("--case", default="", help="case id such as boundary_11_30")
    show_parser.add_argument("--user-idx", type=int, default=None, help="optional user_idx lookup")
    sub.add_parser("walkthrough", help="print a minimal live-demo command sequence")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args or ["summary"])

    if args.command == "summary":
        print_summary()
        return 0
    if args.command == "list-cases":
        list_cases()
        return 0
    if args.command == "show-case":
        show_case(case_id=str(args.case or "").strip() or None, user_idx=args.user_idx)
        return 0
    if args.command == "walkthrough":
        print_walkthrough()
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
