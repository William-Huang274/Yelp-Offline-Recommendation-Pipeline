from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline.project_paths import read_production_run_pointer


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    level: str
    area: str
    message: str


def sanitize_name(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip()).strip("._")
    return safe or "release"


def add_result(results: list[CheckResult], level: str, area: str, message: str) -> None:
    results.append(CheckResult(level=level, area=area, message=message))


def markdown_link(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return f"[{path.name}](./{relative})"
    except Exception:
        return f"[{path.name}]({path.as_posix()})"


def to_float(raw: Any) -> float | None:
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def to_int(raw: Any) -> int | None:
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def metric_row_by_model(rows: list[dict[str, str]], model_name: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("model", "")).strip() == model_name:
            return row
    return None


def overall_status(results: list[CheckResult]) -> str:
    levels = {item.level for item in results}
    if FAIL in levels:
        return FAIL
    if WARN in levels:
        return WARN
    return PASS


def run_monitor() -> tuple[str, list[CheckResult], dict[str, Any], str]:
    results: list[CheckResult] = []
    policy = read_production_run_pointer("release_policy")
    stage09 = read_production_run_pointer("stage09_release")
    stage10 = read_production_run_pointer("stage10_release")
    stage11 = read_production_run_pointer("stage11_release")
    if not all((policy, stage09, stage10, stage11)):
        add_result(results, FAIL, "pointers", "one or more production pointers are missing")
        return FAIL, results, {}, "release"

    release_label = str(policy.get("release_label", "")).strip() or "release"
    contract = policy.get("release_contract", {}) if isinstance(policy.get("release_contract"), dict) else {}
    expected_users = to_int(contract.get("eval_users"))
    expected_topn = to_int(contract.get("candidate_topn"))
    expected_candidates = expected_users * expected_topn if expected_users and expected_topn else None

    truth_in_pretrim = to_float(stage09.get("truth_in_pretrim"))
    hard_miss = to_float(stage09.get("hard_miss"))
    if truth_in_pretrim is None:
        add_result(results, FAIL, "stage09", "stage09_release missing truth_in_pretrim")
    elif truth_in_pretrim >= 0.82:
        add_result(results, PASS, "stage09", f"truth_in_pretrim healthy: {truth_in_pretrim:.6f}")
    else:
        add_result(results, FAIL, "stage09", f"truth_in_pretrim below floor: {truth_in_pretrim:.6f}")
    if hard_miss is None:
        add_result(results, FAIL, "stage09", "stage09_release missing hard_miss")
    elif hard_miss <= 0.10:
        add_result(results, PASS, "stage09", f"hard_miss within bound: {hard_miss:.6f}")
    else:
        add_result(results, FAIL, "stage09", f"hard_miss above bound: {hard_miss:.6f}")

    stage10_metrics_path = Path(str(stage10.get("metrics_file", REPO_ROOT / "data/metrics/recsys_stage10_results_gl07.csv")).strip())
    stage11_metrics_path = Path(str(stage11.get("metrics_file", "")).strip())
    if not stage10_metrics_path.exists():
        add_result(results, FAIL, "stage10", f"stage10 metrics missing: {stage10_metrics_path}")
        return FAIL, results, {}, release_label
    if not stage11_metrics_path.exists():
        add_result(results, FAIL, "stage11", f"stage11 metrics missing: {stage11_metrics_path}")
        return FAIL, results, {}, release_label

    stage10_rows = read_csv_rows(stage10_metrics_path)
    stage11_rows = read_csv_rows(stage11_metrics_path)
    stage10_selected = metric_row_by_model(stage10_rows, str(stage10.get("selected_model", "")).strip())
    stage10_prescore = metric_row_by_model(stage10_rows, "PreScore@10")
    stage11_selected = metric_row_by_model(stage11_rows, str(stage11.get("selected_model", "")).strip())
    stage11_prescore = metric_row_by_model(stage11_rows, "PreScore@10")
    if stage10_selected is None or stage10_prescore is None:
        add_result(results, FAIL, "stage10", "stage10 metrics missing required model rows")
        return FAIL, results, {}, release_label
    if stage11_selected is None or stage11_prescore is None:
        add_result(results, FAIL, "stage11", "stage11 metrics missing required model rows")
        return FAIL, results, {}, release_label

    stage10_users = to_int(stage10_selected.get("n_users"))
    stage11_users = to_int(stage11_selected.get("n_users"))
    stage10_candidates = to_int(stage10_selected.get("n_candidates"))
    stage11_candidates = to_int(stage11_selected.get("n_candidates"))
    if expected_users is not None and stage10_users == expected_users and stage11_users == expected_users:
        add_result(results, PASS, "traffic", f"users evaluated stable at {expected_users}")
    else:
        add_result(results, FAIL, "traffic", f"user count drift detected: stage10={stage10_users} stage11={stage11_users} expected={expected_users}")

    if expected_candidates is not None and stage10_candidates == expected_candidates and stage11_candidates == expected_candidates:
        add_result(results, PASS, "traffic", f"candidate row count stable at {expected_candidates}")
    elif (stage10_candidates or 0) > 0 and (stage11_candidates or 0) > 0:
        add_result(results, WARN, "traffic", f"candidate row count differs from release contract: stage10={stage10_candidates} stage11={stage11_candidates} expected={expected_candidates}")
    else:
        add_result(results, FAIL, "traffic", f"candidate row count invalid: stage10={stage10_candidates} stage11={stage11_candidates}")

    prescore_recall = to_float(stage10_prescore.get("recall_at_k"))
    stage10_recall = to_float(stage10_selected.get("recall_at_k"))
    stage11_recall = to_float(stage11_selected.get("recall_at_k"))
    if stage11_recall is None or stage10_recall is None or prescore_recall is None:
        add_result(results, FAIL, "ranking", "missing recall metrics for drift check")
    elif stage11_recall > stage10_recall > prescore_recall:
        add_result(results, PASS, "ranking", f"champion drift healthy: stage11={stage11_recall:.6f} > stage10={stage10_recall:.6f} > prescore={prescore_recall:.6f}")
    elif stage11_recall > prescore_recall:
        add_result(results, WARN, "ranking", f"champion no longer beats fallback cleanly: stage11={stage11_recall:.6f}, stage10={stage10_recall:.6f}, prescore={prescore_recall:.6f}")
    else:
        add_result(results, FAIL, "ranking", f"champion regressed to baseline or worse: stage11={stage11_recall:.6f}, prescore={prescore_recall:.6f}")

    stage10_model_path = Path(str(stage10.get("rank_model_json", "")).strip())
    if stage10_model_path.exists():
        rank_model = read_json(stage10_model_path)
        if str(rank_model.get("run_tag", "")).strip():
            add_result(results, PASS, "training", "stage10 rank_model.json is readable")
        else:
            add_result(results, FAIL, "training", "stage10 rank_model.json missing run_tag")
    else:
        add_result(results, FAIL, "training", f"stage10 rank_model.json missing: {stage10_model_path}")

    stage11_model_dir = Path(str(stage11.get("source_run_11_2", "")).strip())
    stage11_adapter_dir = Path(str(stage11.get("adapter_dir", "")).strip())
    stage11_model_meta_path = stage11_model_dir / "run_meta.json"
    if not stage11_model_meta_path.exists():
        add_result(results, FAIL, "training", f"stage11 model run_meta missing: {stage11_model_meta_path}")
    else:
        stage11_model_meta = read_json(stage11_model_meta_path)
        train_runtime = to_float(stage11_model_meta.get("train_runtime_sec"))
        train_pairs = to_int(stage11_model_meta.get("train_pairs"))
        eval_pairs = to_int(stage11_model_meta.get("eval_pairs"))
        if (train_runtime or 0.0) > 0 and (train_pairs or 0) > 0 and (eval_pairs or 0) > 0 and stage11_adapter_dir.exists():
            add_result(results, PASS, "training", f"stage11 training artifact healthy: runtime={train_runtime:.1f}s train_pairs={train_pairs} eval_pairs={eval_pairs}")
        else:
            add_result(results, FAIL, "training", f"stage11 training artifact incomplete: runtime={train_runtime} train_pairs={train_pairs} eval_pairs={eval_pairs} adapter_exists={stage11_adapter_dir.exists()}")
        if stage11_model_meta.get("enforce_stage09_gate") is False:
            add_result(results, WARN, "training", "stage11 model run still records enforce_stage09_gate=false")

    summary = {
        "release_label": release_label,
        "truth_in_pretrim": truth_in_pretrim,
        "hard_miss": hard_miss,
        "users_evaluated": expected_users,
        "candidate_rows_expected": expected_candidates,
        "stage10_recall": stage10_recall,
        "stage11_recall": stage11_recall,
        "prescore_recall": prescore_recall,
    }
    return overall_status(results), results, summary, release_label


def build_report(status: str, results: list[CheckResult], summary: dict[str, Any], release_label: str, report_path: Path) -> str:
    counts = {
        PASS: sum(1 for item in results if item.level == PASS),
        WARN: sum(1 for item in results if item.level == WARN),
        FAIL: sum(1 for item in results if item.level == FAIL),
    }
    lines = [
        f"# Release Monitor Report: {release_label}",
        "",
        f"- generated_at: `{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- overall_status: `{status}`",
        f"- report_path: {markdown_link(report_path)}",
        "",
        "## Summary",
        "",
        f"- truth_in_pretrim: `{summary.get('truth_in_pretrim')}`",
        f"- hard_miss: `{summary.get('hard_miss')}`",
        f"- users_evaluated: `{summary.get('users_evaluated')}`",
        f"- candidate_rows_expected: `{summary.get('candidate_rows_expected')}`",
        f"- stage11_recall_at_10: `{summary.get('stage11_recall')}`",
        f"- stage10_recall_at_10: `{summary.get('stage10_recall')}`",
        f"- prescore_recall_at_10: `{summary.get('prescore_recall')}`",
        f"- check_counts: `PASS={counts[PASS]} WARN={counts[WARN]} FAIL={counts[FAIL]}`",
        "",
        "## Checks",
        "",
        "| Level | Area | Message |",
        "| --- | --- | --- |",
    ]
    for item in results:
        lines.append(f"| `{item.level}` | `{item.area}` | {item.message} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    status, results, summary, release_label = run_monitor()
    report_path = REPO_ROOT / "docs" / f"release_monitor_report_{sanitize_name(release_label)}.md"
    report_path.write_text(build_report(status, results, summary, release_label, report_path), encoding="utf-8")
    print(f"{status} {report_path}")
    return 1 if status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
