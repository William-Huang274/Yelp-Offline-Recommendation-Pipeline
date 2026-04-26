#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/output/serving_validation/latest_summary.json")
DEFAULT_OUTPUT = Path("docs/serving_validation_report.md")
DEFAULT_SERVING_P95_BUDGET_MS = 250.0
DEFAULT_SERVING_P99_BUDGET_MS = 300.0
DEFAULT_SUCCESS_RATE_FLOOR = 0.99


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_ms(value: Any) -> str:
    return f"{as_float(value):.3f} ms"


def format_rate(value: Any) -> str:
    return f"{as_float(value) * 100.0:.2f}%"


def format_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def stat(summary: dict[str, Any], path: tuple[str, ...], key: str, default: float = 0.0) -> float:
    node: Any = summary
    for name in path:
        if not isinstance(node, dict):
            return default
        node = node.get(name, {})
    if not isinstance(node, dict):
        return default
    return as_float(node.get(key), default)


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def make_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build_report(
    summary: dict[str, Any],
    *,
    input_path: Path,
    serving_p95_budget_ms: float,
    serving_p99_budget_ms: float,
    success_rate_floor: float,
) -> str:
    success_rate = as_float(summary.get("success_rate"))
    serving_p95 = stat(summary, ("latency_ms",), "p95")
    serving_p99 = stat(summary, ("latency_ms",), "p99")
    audit_p95 = stat(summary, ("audit_latency_ms",), "p95")
    stage_latency = summary.get("stage_latency_ms", {})
    audit_breakdown = summary.get("audit_breakdown_ms", {})

    status_rows = [
        [
            "success_rate",
            format_rate(success_rate),
            f">= {format_rate(success_rate_floor)}",
            pass_fail(success_rate >= success_rate_floor),
        ],
        [
            "serving_latency_p95",
            format_ms(serving_p95),
            f"<= {format_ms(serving_p95_budget_ms)}",
            pass_fail(serving_p95 <= serving_p95_budget_ms),
        ],
        [
            "serving_latency_p99",
            format_ms(serving_p99),
            f"<= {format_ms(serving_p99_budget_ms)}",
            pass_fail(serving_p99 <= serving_p99_budget_ms),
        ],
    ]

    latency_rows = []
    for name in ("latency_ms", "wall_latency_ms", "audit_latency_ms"):
        values = summary.get(name, {})
        if isinstance(values, dict):
            latency_rows.append(
                [
                    name,
                    format_ms(values.get("p50")),
                    format_ms(values.get("p95")),
                    format_ms(values.get("p99", 0.0)),
                    format_ms(values.get("max", 0.0)),
                ]
            )

    stage_rows = []
    if isinstance(stage_latency, dict):
        for name in ("request_lookup", "stage09", "stage10", "stage11", "offline_truth_audit", "fallback_demo"):
            values = stage_latency.get(name, {})
            if isinstance(values, dict):
                stage_rows.append(
                    [
                        name,
                        format_ms(values.get("p50")),
                        format_ms(values.get("p95")),
                        format_ms(values.get("p99")),
                        format_ms(values.get("avg")),
                    ]
                )

    audit_rows = []
    if isinstance(audit_breakdown, dict):
        for name in ("stage11_live", "fallback_demo"):
            values = audit_breakdown.get(name, {})
            if isinstance(values, dict):
                audit_rows.append([name, format_ms(values.get("p50")), format_ms(values.get("p95")), format_ms(values.get("avg"))])

    stage_mode_counts = summary.get("stage_mode_counts", {})
    source_alignment_counts = summary.get("source_alignment_counts", {})
    stage09_modes = stage_mode_counts.get("stage09", {}) if isinstance(stage_mode_counts, dict) else {}
    stage10_modes = stage_mode_counts.get("stage10", {}) if isinstance(stage_mode_counts, dict) else {}
    stage11_modes = stage_mode_counts.get("stage11", {}) if isinstance(stage_mode_counts, dict) else {}
    stage09_alignment = source_alignment_counts.get("stage09", {}) if isinstance(source_alignment_counts, dict) else {}
    stage10_alignment = source_alignment_counts.get("stage10", {}) if isinstance(source_alignment_counts, dict) else {}

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Serving Validation Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Source JSON: `{input_path.as_posix()}`",
        f"- Mode: `{summary.get('mode', '-')}`",
        f"- Requests: `{summary.get('requests', 0)}` after `{summary.get('warmup_requests', 0)}` warmup",
        f"- Concurrency: `{summary.get('concurrency', 0)}`",
        f"- Unique replay request_ids: `{summary.get('unique_request_ids', 0)}`",
        f"- Traffic profile: `{summary.get('traffic_profile', '-')}`",
        f"- Requested strategy: `{summary.get('strategy_requested', '-')}`",
        "",
        "## SLA Gate",
        "",
        *make_table(["Signal", "Observed", "Gate", "Status"], status_rows),
        "",
        "## Traffic Mix",
        "",
        f"- Requested strategy counts: `{format_counts(summary.get('strategy_requested_counts'))}`",
        f"- Used strategy counts: `{format_counts(summary.get('strategy_used_counts'))}`",
        f"- Fallback rate: `{format_rate(summary.get('fallback_rate'))}`",
        f"- Cache hit rate: `{format_rate(summary.get('cache_hit_rate'))}`",
        f"- Cache miss count: `{summary.get('cache_miss_count', 0)}`",
        f"- Backfill count: `{summary.get('backfill_count', 0)}`",
        f"- Cache status counts: `{format_counts(summary.get('cache_status_counts'))}`",
        "",
        "## Latency Summary",
        "",
        *make_table(["Metric", "p50", "p95", "p99", "max"], latency_rows),
        "",
        "## Per-Stage Latency",
        "",
        *make_table(["Stage", "p50", "p95", "p99", "avg"], stage_rows),
        "",
        "## Stage09/10/11 Serving Scope",
        "",
        f"- Stage09 modes: `{format_counts(stage09_modes)}`",
        f"- Stage09 source alignment: `{format_counts(stage09_alignment)}`",
        f"- Stage10 modes: `{format_counts(stage10_modes)}`",
        f"- Stage10 source alignment: `{format_counts(stage10_alignment)}`",
        f"- Stage11 modes: `{format_counts(stage11_modes)}`",
        "",
        "## Audit Path",
        "",
        f"- Audit latency p95: `{format_ms(audit_p95)}`",
        *make_table(["Audit block", "p50", "p95", "avg"], audit_rows),
        "",
        "## Interpretation",
        "",
        "- `serving_latency_ms` excludes offline-only audit work and is the number to compare with the mock online budget.",
        "- `wall_latency_ms` includes local Python overhead and optional audit hooks, so it is useful for demo experience but not the serving SLA gate.",
        "- `stage11_live` is treated as audit/backfill unless the serving policy reports `applied_to_serving=true`; cache miss should fall back to Stage10 and enqueue backfill.",
        "- Stage09 is represented by replay lookup or live lookup mode in this validation. It does not retrain or redefine recall candidates.",
        "",
        "## Known Limits",
        "",
        "- This is a local replay/mock-serving validation report, not a production traffic A/B report.",
        "- The request stream is sampled from frozen replay request_ids, so it verifies policy behavior, latency accounting, and fallback paths rather than new user distribution quality.",
        "- Metric definitions, labels, candidate boundaries, and split logic are not changed by this report.",
    ]
    first_errors = summary.get("first_errors")
    if first_errors:
        lines.extend(["", "## First Errors", ""])
        for error in first_errors:
            lines.append(f"- `{error}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export load_test_mock_serving JSON into a Markdown validation report.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="load_test_mock_serving JSON summary")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown report path")
    parser.add_argument("--serving-p95-budget-ms", type=float, default=DEFAULT_SERVING_P95_BUDGET_MS)
    parser.add_argument("--serving-p99-budget-ms", type=float, default=DEFAULT_SERVING_P99_BUDGET_MS)
    parser.add_argument("--success-rate-floor", type=float, default=DEFAULT_SUCCESS_RATE_FLOOR)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when an SLA gate fails")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary = read_json(input_path)
    report = build_report(
        summary,
        input_path=input_path,
        serving_p95_budget_ms=float(args.serving_p95_budget_ms),
        serving_p99_budget_ms=float(args.serving_p99_budget_ms),
        success_rate_floor=float(args.success_rate_floor),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    success_rate = as_float(summary.get("success_rate"))
    serving_p95 = stat(summary, ("latency_ms",), "p95")
    serving_p99 = stat(summary, ("latency_ms",), "p99")
    gates_pass = (
        success_rate >= float(args.success_rate_floor)
        and serving_p95 <= float(args.serving_p95_budget_ms)
        and serving_p99 <= float(args.serving_p99_budget_ms)
    )
    print(f"wrote {output_path}")
    return 1 if args.strict and not gates_pass else 0


if __name__ == "__main__":
    raise SystemExit(main())
