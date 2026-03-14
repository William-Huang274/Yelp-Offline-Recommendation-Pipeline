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

from pipeline.project_paths import production_run_pointer_path, read_production_run_pointer, resolve_production_run_pointer
from pipeline.run_validators import load_json_object, validate_stage09_candidate_run, validate_stage11_dataset_run


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
RELEASE_REPORT_PREFIX = "release_readiness_report_"


@dataclass
class CheckResult:
    level: str
    area: str
    message: str


def sanitize_name(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip()).strip("._")
    return safe or "release"


def markdown_link(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return f"[{path.name}](./{relative})"
    except Exception:
        return f"[{path.name}]({path.as_posix()})"


def canonical_run_id(raw: str | Path | None) -> str:
    text = str(raw or "").strip().replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def same_run_id(left: str | Path | None, right: str | Path | None) -> bool:
    return canonical_run_id(left) == canonical_run_id(right)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


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


def add_result(results: list[CheckResult], level: str, area: str, message: str) -> None:
    results.append(CheckResult(level=level, area=area, message=message))


def pointer_payload(pointer_name: str, results: list[CheckResult]) -> dict[str, Any] | None:
    pointer_path = production_run_pointer_path(pointer_name)
    payload = read_production_run_pointer(pointer_name)
    if payload is None:
        add_result(results, FAIL, "pointers", f"missing or unreadable production pointer: {pointer_path}")
        return None
    add_result(results, PASS, "pointers", f"loaded production pointer: {pointer_path.name}")
    return payload


def require_path(path: Path, results: list[CheckResult], area: str, label: str, expect_dir: bool | None = None) -> bool:
    if not path.exists():
        add_result(results, FAIL, area, f"missing path: {label} -> {path}")
        return False
    if expect_dir is True and not path.is_dir():
        add_result(results, FAIL, area, f"expected directory: {label} -> {path}")
        return False
    if expect_dir is False and not path.is_file():
        add_result(results, FAIL, area, f"expected file: {label} -> {path}")
        return False
    add_result(results, PASS, area, f"found path: {label}")
    return True


def find_release_manifest(release_label: str) -> Path | None:
    sanitized = sanitize_name(release_label)
    prod_manifest = REPO_ROOT / "data/output/_prod_runs" / f"release_manifest_{sanitized}.json"
    if prod_manifest.exists():
        return prod_manifest
    output_root = REPO_ROOT / "data/output"
    for candidate in sorted(output_root.rglob("*manifest.json")):
        try:
            payload = load_json_object(candidate)
        except Exception:
            continue
        if str(payload.get("version_label", "")).strip() == release_label:
            return candidate
        if str(payload.get("release_label", "")).strip() == release_label:
            return candidate
    return None


def metric_row_by_model(rows: list[dict[str, str]], model_name: str) -> dict[str, str] | None:
    for row in rows:
        if str(row.get("model", "")).strip() == model_name:
            return row
    return None


def required_docs() -> list[Path]:
    return [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/go_live_readiness_20260312.md",
        REPO_ROOT / "docs/v1_freeze_20260313.md",
        REPO_ROOT / "docs/first_champion_closeout_20260313.md",
        REPO_ROOT / "docs/stage09_reaudit_20260313.md",
        REPO_ROOT / "docs/gl07_stage10_stage11_alignment_20260313.md",
        REPO_ROOT / "docs/gl08_prod_pointers_20260313.md",
        REPO_ROOT / "docs/gl09_path_unification_20260313.md",
        REPO_ROOT / "docs/gl10_smoke_tests_20260313.md",
        REPO_ROOT / "docs/gl12_batch_runner_20260313.md",
        REPO_ROOT / "docs/rollback_and_monitoring.md",
        REPO_ROOT / "docs/config_reference.md",
        REPO_ROOT / "docs/data_contract.md",
        REPO_ROOT / "docs/stage11_cloud_run_profile_20260309.md",
    ]


def overall_status(results: list[CheckResult]) -> str:
    levels = {item.level for item in results}
    if FAIL in levels:
        return FAIL
    if WARN in levels:
        return WARN
    return PASS


def build_report_markdown(
    release_label: str,
    status: str,
    results: list[CheckResult],
    summary: dict[str, Any],
    report_path: Path,
) -> str:
    counts = {
        PASS: sum(1 for item in results if item.level == PASS),
        WARN: sum(1 for item in results if item.level == WARN),
        FAIL: sum(1 for item in results if item.level == FAIL),
    }
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines: list[str] = [
        f"# Release Readiness Report: {release_label}",
        "",
        f"- generated_at: `{generated_at}`",
        f"- overall_status: `{status}`",
        f"- report_path: {markdown_link(report_path)}",
        "",
        "## Summary",
        "",
        f"- champion_pointer: `{summary['champion_pointer']}`",
        f"- aligned_fallback_pointer: `{summary['fallback_pointer']}`",
        f"- emergency_baseline_pointer: `{summary['baseline_pointer']}`",
        f"- source_run_09: `{summary['source_run_09']}`",
        f"- release_contract: `bucket={summary['bucket']}, eval_users={summary['eval_users']}, candidate_topn={summary['candidate_topn']}, top_k={summary['top_k']}`",
        f"- check_counts: `PASS={counts[PASS]} WARN={counts[WARN]} FAIL={counts[FAIL]}`",
        "",
        "## Key Metrics",
        "",
        f"- stage09 audit: `truth_in_all={summary['stage09_truth_in_all']}` `truth_in_pretrim={summary['stage09_truth_in_pretrim']}` `hard_miss={summary['stage09_hard_miss']}`",
        f"- stage10 fallback: `{summary['stage10_model']}` `recall@10={summary['stage10_recall']}` `ndcg@10={summary['stage10_ndcg']}`",
        f"- stage10 baseline: `PreScore@10` `recall@10={summary['prescore_recall']}` `ndcg@10={summary['prescore_ndcg']}`",
        f"- stage11 champion: `{summary['stage11_model']}` `recall@10={summary['stage11_recall']}` `ndcg@10={summary['stage11_ndcg']}`",
        "",
        "## Checks",
        "",
        "| Level | Area | Message |",
        "| --- | --- | --- |",
    ]
    for item in results:
        lines.append(f"| `{item.level}` | `{item.area}` | {item.message} |")
    return "\n".join(lines) + "\n"


def run_checks() -> tuple[str, list[CheckResult], dict[str, Any], str]:
    results: list[CheckResult] = []

    policy = pointer_payload("release_policy", results)
    if policy is None:
        return FAIL, results, {}, "release"

    release_label = str(policy.get("release_label", "")).strip() or "release"
    champion_pointer_name = str(policy.get("champion_pointer", "")).strip()
    fallback_pointer_name = str(policy.get("aligned_fallback_pointer", "")).strip()
    baseline_pointer_name = str(policy.get("emergency_baseline_pointer", "")).strip()
    contract = policy.get("release_contract", {}) if isinstance(policy.get("release_contract"), dict) else {}
    expected_source_run = str(contract.get("source_run_09", "")).strip()
    expected_bucket = to_int(contract.get("bucket"))
    expected_eval_users = to_int(contract.get("eval_users"))
    expected_candidate_topn = to_int(contract.get("candidate_topn"))
    expected_top_k = to_int(contract.get("top_k"))

    stage09_pointer = pointer_payload(baseline_pointer_name, results) if baseline_pointer_name else None
    stage10_pointer = pointer_payload(fallback_pointer_name, results) if fallback_pointer_name else None
    stage11_pointer = pointer_payload(champion_pointer_name, results) if champion_pointer_name else None
    if not all((stage09_pointer, stage10_pointer, stage11_pointer)):
        return FAIL, results, {}, release_label

    for pointer_name, payload in (
        (baseline_pointer_name, stage09_pointer),
        (fallback_pointer_name, stage10_pointer),
        (champion_pointer_name, stage11_pointer),
    ):
        if str(payload.get("release_label", "")).strip() != release_label:
            add_result(results, FAIL, "pointers", f"{pointer_name} release_label does not match policy release_label")
        else:
            add_result(results, PASS, "pointers", f"{pointer_name} release_label matches policy")

    stage09_run = resolve_production_run_pointer(baseline_pointer_name)
    stage10_run = resolve_production_run_pointer(fallback_pointer_name)
    stage11_run = resolve_production_run_pointer(champion_pointer_name)
    if stage09_run is None:
        add_result(results, FAIL, "stage09", f"could not resolve {baseline_pointer_name} run_dir")
    else:
        require_path(stage09_run, results, "stage09", f"{baseline_pointer_name} run_dir", expect_dir=True)
    if stage10_run is None:
        add_result(results, FAIL, "stage10", f"could not resolve {fallback_pointer_name} run_dir")
    else:
        require_path(stage10_run, results, "stage10", f"{fallback_pointer_name} run_dir", expect_dir=True)
    if stage11_run is None:
        add_result(results, FAIL, "stage11", f"could not resolve {champion_pointer_name} run_dir")
    else:
        require_path(stage11_run, results, "stage11", f"{champion_pointer_name} run_dir", expect_dir=True)

    if stage09_run is not None:
        stage09_errors = validate_stage09_candidate_run(stage09_run)
        if stage09_errors:
            for error in stage09_errors:
                add_result(results, FAIL, "stage09", error)
        else:
            add_result(results, PASS, "stage09", "stage09 candidate run validator passed")
    stage11_data_path = Path(str(stage11_pointer.get("source_run_11_1_data", "")).strip())
    if require_path(stage11_data_path, results, "stage11", "stage11 dataset run", expect_dir=True):
        stage11_data_errors = validate_stage11_dataset_run(stage11_data_path)
        if stage11_data_errors:
            for error in stage11_data_errors:
                add_result(results, FAIL, "stage11", error)
        else:
            add_result(results, PASS, "stage11", "stage11 dataset run validator passed")

    stage09_metrics_path = Path(str(stage09_pointer.get("audit_metrics_file", "")).strip())
    stage10_metrics_path = Path(str(stage10_pointer.get("metrics_file", REPO_ROOT / "data/metrics/recsys_stage10_results_gl07.csv")).strip())
    stage11_metrics_path = Path(str(stage11_pointer.get("metrics_file", "")).strip())
    stage10_run_meta_path = stage10_run / "run_meta.json" if stage10_run is not None else Path()
    stage11_run_meta_path = stage11_run / "run_meta.json" if stage11_run is not None else Path()
    stage10_rank_model_path = Path(str(stage10_pointer.get("rank_model_json", "")).strip())
    stage10_cohort_path = Path(str(stage10_pointer.get("eval_user_cohort_path", "")).strip())
    stage11_alpha_path = Path(str(stage11_pointer.get("alpha_sweep_best_file", "")).strip())
    stage11_model_dir = Path(str(stage11_pointer.get("source_run_11_2", "")).strip())
    stage11_adapter_dir = Path(str(stage11_pointer.get("adapter_dir", "")).strip())

    require_path(stage09_metrics_path, results, "stage09", "stage09 audit metrics file", expect_dir=False)
    require_path(stage10_metrics_path, results, "stage10", "stage10 metrics file", expect_dir=False)
    require_path(stage10_run_meta_path, results, "stage10", "stage10 run_meta.json", expect_dir=False)
    require_path(stage10_rank_model_path, results, "stage10", "stage10 rank_model.json", expect_dir=False)
    require_path(stage10_cohort_path, results, "stage10", "stage10 eval user cohort", expect_dir=False)
    require_path(stage11_metrics_path, results, "stage11", "stage11 metrics file", expect_dir=False)
    require_path(stage11_run_meta_path, results, "stage11", "stage11 run_meta.json", expect_dir=False)
    require_path(stage11_alpha_path, results, "stage11", "stage11 alpha sweep best file", expect_dir=False)
    require_path(stage11_model_dir, results, "stage11", "stage11 model run dir", expect_dir=True)
    require_path(stage11_adapter_dir, results, "stage11", "stage11 adapter dir", expect_dir=True)

    stage09_truth_in_all = None
    stage09_truth_in_pretrim = None
    stage09_hard_miss = None
    if stage09_metrics_path.exists():
        rows = read_csv_rows(stage09_metrics_path)
        target_row = None
        for row in rows:
            if same_run_id(row.get("source_run_09", ""), expected_source_run) and to_int(row.get("bucket")) == expected_bucket:
                target_row = row
                break
        if target_row is None:
            add_result(results, FAIL, "stage09", "stage09 audit metrics file missing expected release source row")
        else:
            stage09_truth_in_all = to_float(target_row.get("truth_in_all"))
            stage09_truth_in_pretrim = to_float(target_row.get("truth_in_pretrim"))
            stage09_hard_miss = to_float(target_row.get("hard_miss"))
            add_result(results, PASS, "stage09", "stage09 audit metrics match release source and bucket")

    stage10_model_name = str(stage10_pointer.get("selected_model", "")).strip()
    stage10_recall = None
    stage10_ndcg = None
    prescore_recall = None
    prescore_ndcg = None
    if stage10_run_meta_path.exists():
        stage10_meta = load_json_object(stage10_run_meta_path)
        if not same_run_id(stage10_meta.get("source_stage09_run", ""), expected_source_run):
            add_result(results, FAIL, "stage10", "stage10 run_meta source_stage09_run does not match release contract")
        else:
            add_result(results, PASS, "stage10", "stage10 run_meta source matches release contract")
        if to_int(stage10_meta.get("eval_candidate_topn")) != expected_candidate_topn:
            add_result(results, FAIL, "stage10", "stage10 run_meta eval_candidate_topn does not match release contract")
        else:
            add_result(results, PASS, "stage10", "stage10 run_meta candidate topn matches release contract")
        if to_int(stage10_meta.get("top_k")) != expected_top_k:
            add_result(results, FAIL, "stage10", "stage10 run_meta top_k does not match release contract")
        else:
            add_result(results, PASS, "stage10", "stage10 run_meta top_k matches release contract")
    if stage10_metrics_path.exists():
        stage10_rows = read_csv_rows(stage10_metrics_path)
        if not stage10_rows:
            add_result(results, FAIL, "stage10", "stage10 metrics file is empty")
        else:
            if all(same_run_id(row.get("source_run_09", ""), expected_source_run) for row in stage10_rows):
                add_result(results, PASS, "stage10", "stage10 metrics source_run_09 matches release contract")
            else:
                add_result(results, FAIL, "stage10", "stage10 metrics source_run_09 does not match release contract")
            model_row = metric_row_by_model(stage10_rows, stage10_model_name)
            prescore_row = metric_row_by_model(stage10_rows, "PreScore@10")
            if model_row is None:
                add_result(results, FAIL, "stage10", f"stage10 metrics missing selected model row: {stage10_model_name}")
            else:
                stage10_recall = to_float(model_row.get("recall_at_k"))
                stage10_ndcg = to_float(model_row.get("ndcg_at_k"))
                if to_int(model_row.get("n_users")) != expected_eval_users:
                    add_result(results, FAIL, "stage10", "stage10 selected model row n_users does not match release contract")
                else:
                    add_result(results, PASS, "stage10", "stage10 selected model row matches release user count")
            if prescore_row is None:
                add_result(results, FAIL, "stage10", "stage10 metrics missing PreScore@10 row")
            else:
                prescore_recall = to_float(prescore_row.get("recall_at_k"))
                prescore_ndcg = to_float(prescore_row.get("ndcg_at_k"))
                add_result(results, PASS, "stage10", "stage10 metrics include PreScore@10 row")
            pointer_recall = to_float(stage10_pointer.get("recall_at_k"))
            if pointer_recall is not None and stage10_recall is not None and abs(pointer_recall - stage10_recall) > 1e-9:
                add_result(results, FAIL, "stage10", "stage10 pointer recall does not match metrics row")
            elif pointer_recall is not None and stage10_recall is not None:
                add_result(results, PASS, "stage10", "stage10 pointer recall matches metrics row")

    stage11_model_name = str(stage11_pointer.get("selected_model", "")).strip()
    stage11_recall = None
    stage11_ndcg = None
    if stage11_run_meta_path.exists():
        stage11_meta = load_json_object(stage11_run_meta_path)
        if not same_run_id(stage11_meta.get("source_run_09", ""), expected_source_run):
            add_result(results, FAIL, "stage11", "stage11 run_meta source_run_09 does not match release contract")
        else:
            add_result(results, PASS, "stage11", "stage11 run_meta source matches release contract")
        if to_int(stage11_meta.get("rerank_topn")) != expected_candidate_topn:
            add_result(results, FAIL, "stage11", "stage11 run_meta rerank_topn does not match release contract")
        else:
            add_result(results, PASS, "stage11", "stage11 run_meta rerank_topn matches release contract")
        if to_int(stage11_meta.get("top_k")) != expected_top_k:
            add_result(results, FAIL, "stage11", "stage11 run_meta top_k does not match release contract")
        else:
            add_result(results, PASS, "stage11", "stage11 run_meta top_k matches release contract")
        if stage11_meta.get("enforce_stage09_gate") is False:
            add_result(results, WARN, "stage11", "stage11 champion run_meta still records enforce_stage09_gate=false")
    if stage11_metrics_path.exists():
        stage11_rows = read_csv_rows(stage11_metrics_path)
        if not stage11_rows:
            add_result(results, FAIL, "stage11", "stage11 metrics file is empty")
        else:
            if all(same_run_id(row.get("source_run_09", ""), expected_source_run) for row in stage11_rows):
                add_result(results, PASS, "stage11", "stage11 metrics source_run_09 matches release contract")
            else:
                add_result(results, FAIL, "stage11", "stage11 metrics source_run_09 does not match release contract")
            model_row = metric_row_by_model(stage11_rows, stage11_model_name)
            prescore_row = metric_row_by_model(stage11_rows, "PreScore@10")
            if model_row is None:
                add_result(results, FAIL, "stage11", f"stage11 metrics missing selected model row: {stage11_model_name}")
            else:
                stage11_recall = to_float(model_row.get("recall_at_k"))
                stage11_ndcg = to_float(model_row.get("ndcg_at_k"))
                if to_int(model_row.get("n_users")) != expected_eval_users:
                    add_result(results, FAIL, "stage11", "stage11 selected model row n_users does not match release contract")
                else:
                    add_result(results, PASS, "stage11", "stage11 selected model row matches release user count")
            if prescore_row is None:
                add_result(results, FAIL, "stage11", "stage11 metrics missing PreScore@10 row")
            else:
                add_result(results, PASS, "stage11", "stage11 metrics include PreScore@10 row")
            pointer_recall = to_float(stage11_pointer.get("recall_at_k"))
            if pointer_recall is not None and stage11_recall is not None and abs(pointer_recall - stage11_recall) > 1e-12:
                add_result(results, FAIL, "stage11", "stage11 pointer recall does not match metrics row")
            elif pointer_recall is not None and stage11_recall is not None:
                add_result(results, PASS, "stage11", "stage11 pointer recall matches metrics row")

    if stage11_alpha_path.exists():
        alpha_payload = load_json_object(stage11_alpha_path)
        alpha_best = to_float(alpha_payload.get("alpha"))
        if alpha_best is None:
            add_result(results, FAIL, "stage11", "alpha_sweep_best.json missing numeric alpha")
        else:
            add_result(results, PASS, "stage11", "alpha_sweep_best.json is readable")

    if stage11_recall is not None and stage10_recall is not None and prescore_recall is not None:
        if stage11_recall > stage10_recall > prescore_recall:
            add_result(results, PASS, "ranking", "champion > fallback > PreScore ordering holds on frozen release metrics")
        else:
            add_result(results, FAIL, "ranking", "champion ordering does not hold on frozen release metrics")

    manifest_path = find_release_manifest(release_label)
    if manifest_path is None:
        add_result(results, WARN, "manifest", "could not locate a manifest file matching the current release_label")
    else:
        add_result(results, PASS, "manifest", f"located release manifest: {manifest_path.name}")
        manifest_payload = load_json_object(manifest_path)
        production_ready = bool(manifest_payload.get("status", {}).get("production_ready"))
        if not production_ready:
            add_result(results, WARN, "manifest", "release manifest still marks production_ready=false")

    rollback_snapshots = sorted((REPO_ROOT / "data/output/_prod_runs").glob("rollback_snapshot_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if rollback_snapshots:
        add_result(results, PASS, "ops", f"rollback snapshot exists: {rollback_snapshots[0].name}")
    else:
        add_result(results, WARN, "ops", "no rollback snapshot found under data/output/_prod_runs")

    monitor_report_path = REPO_ROOT / "docs" / f"release_monitor_report_{sanitize_name(release_label)}.md"
    if monitor_report_path.exists() and monitor_report_path.stat().st_size > 0:
        add_result(results, PASS, "ops", f"release monitor report present: {monitor_report_path.name}")
    else:
        add_result(results, WARN, "ops", f"release monitor report missing: {monitor_report_path.name}")

    for doc_path in required_docs():
        if not doc_path.exists():
            add_result(results, FAIL, "docs", f"missing required doc: {doc_path.name}")
            continue
        if doc_path.stat().st_size == 0:
            add_result(results, FAIL, "docs", f"zero-byte required doc: {doc_path.name}")
            continue
        add_result(results, PASS, "docs", f"required doc present and readable: {doc_path.name}")

    summary = {
        "champion_pointer": champion_pointer_name,
        "fallback_pointer": fallback_pointer_name,
        "baseline_pointer": baseline_pointer_name,
        "source_run_09": expected_source_run,
        "bucket": expected_bucket,
        "eval_users": expected_eval_users,
        "candidate_topn": expected_candidate_topn,
        "top_k": expected_top_k,
        "stage09_truth_in_all": stage09_truth_in_all,
        "stage09_truth_in_pretrim": stage09_truth_in_pretrim,
        "stage09_hard_miss": stage09_hard_miss,
        "stage10_model": stage10_model_name,
        "stage10_recall": stage10_recall,
        "stage10_ndcg": stage10_ndcg,
        "prescore_recall": prescore_recall,
        "prescore_ndcg": prescore_ndcg,
        "stage11_model": stage11_model_name,
        "stage11_recall": stage11_recall,
        "stage11_ndcg": stage11_ndcg,
    }
    return overall_status(results), results, summary, release_label


def main() -> int:
    status, results, summary, release_label = run_checks()
    report_name = f"{RELEASE_REPORT_PREFIX}{sanitize_name(release_label)}.md"
    report_path = REPO_ROOT / "docs" / report_name
    report_text = build_report_markdown(release_label, status, results, summary, report_path)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"{status} {report_path}")
    return 1 if status == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
