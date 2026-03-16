from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.project_paths import (
    PROD_RUN_DIR,
    production_run_pointer_path,
    read_production_run_pointer,
    resolve_production_run_pointer,
    write_production_run_pointer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIRST_CHAMPION_MANIFEST = (
    REPO_ROOT / "data/output/_first_champion_freeze_20260313/first_champion_manifest.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def sanitize_name(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(raw or "").strip()).strip("._")
    return safe or "release"


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def conservative_spark_defaults() -> dict[str, str]:
    return {
        "SPARK_MASTER": "local[2]",
        "SPARK_DRIVER_MEMORY": "6g",
        "SPARK_EXECUTOR_MEMORY": "6g",
        "SPARK_SQL_SHUFFLE_PARTITIONS": "12",
        "SPARK_DEFAULT_PARALLELISM": "12",
    }


def base_env(overrides: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in conservative_spark_defaults().items():
        env.setdefault(key, value)
    env.setdefault("BDA_PROJECT_ROOT", str(REPO_ROOT))
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            env[key] = text
    return env


def run_python_script(script_path: Path, env_updates: dict[str, Any] | None = None, dry_run: bool = False) -> int:
    command = [sys.executable, str(script_path)]
    env = base_env(env_updates)
    print(f"[RUN] {' '.join(shlex.quote(part) for part in command)}")
    if env_updates:
        print("[ENV]")
        for key in sorted(env_updates):
            value = env_updates[key]
            if value is None or not str(value).strip():
                continue
            print(f"  {key}={value}")
    if dry_run:
        print("[DRY-RUN] command not executed")
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def load_release_policy() -> dict[str, Any]:
    payload = read_production_run_pointer("release_policy")
    if payload is None:
        raise FileNotFoundError(f"missing production pointer: {production_run_pointer_path('release_policy')}")
    return payload


def load_release_pointer(pointer_name: str) -> dict[str, Any]:
    payload = read_production_run_pointer(pointer_name)
    if payload is None:
        raise FileNotFoundError(f"missing production pointer: {production_run_pointer_path(pointer_name)}")
    return payload


def resolve_release_run(pointer_name: str) -> Path:
    path = resolve_production_run_pointer(pointer_name)
    if path is None:
        raise FileNotFoundError(f"could not resolve production run pointer: {pointer_name}")
    return path


def current_release_context() -> dict[str, Any]:
    policy = load_release_policy()
    champion_pointer = str(policy.get("champion_pointer", "")).strip()
    fallback_pointer = str(policy.get("aligned_fallback_pointer", "")).strip()
    baseline_pointer = str(policy.get("emergency_baseline_pointer", "")).strip()
    contract = policy.get("release_contract", {}) if isinstance(policy.get("release_contract"), dict) else {}
    return {
        "policy": policy,
        "contract": contract,
        "stage09_pointer_name": baseline_pointer,
        "stage10_pointer_name": fallback_pointer,
        "stage11_pointer_name": champion_pointer,
        "stage09_pointer": load_release_pointer(baseline_pointer),
        "stage10_pointer": load_release_pointer(fallback_pointer),
        "stage11_pointer": load_release_pointer(champion_pointer),
        "stage09_run": resolve_release_run(baseline_pointer),
        "stage10_run": resolve_release_run(fallback_pointer),
        "stage11_run": resolve_release_run(champion_pointer),
    }


def current_prod_state_snapshot(reason: str) -> dict[str, Any]:
    release_policy = read_production_run_pointer("release_policy") or {}
    current_label = str(release_policy.get("release_label", "")).strip()
    pointers: dict[str, Any] = {}
    for pointer_name in ("release_policy", "stage09_release", "stage10_release", "stage11_release"):
        payload = read_production_run_pointer(pointer_name)
        if payload is not None:
            pointers[pointer_name] = payload
    return {
        "snapshot_id": timestamp_id(),
        "created_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "reason": reason,
        "release_label": current_label,
        "pointers": pointers,
    }


def snapshot_file_path(snapshot_payload: dict[str, Any]) -> Path:
    release_label = sanitize_name(snapshot_payload.get("release_label", "") or "unlabeled")
    snapshot_id = sanitize_name(snapshot_payload.get("snapshot_id", "") or timestamp_id())
    return PROD_RUN_DIR / f"rollback_snapshot_{snapshot_id}_{release_label}.json"


def latest_rollback_snapshot_path() -> Path | None:
    candidates = sorted(PROD_RUN_DIR.glob("rollback_snapshot_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def write_current_snapshot(reason: str, dry_run: bool) -> Path:
    payload = current_prod_state_snapshot(reason=reason)
    snapshot_path = snapshot_file_path(payload)
    print(f"[SNAPSHOT] reason={reason}")
    print(f"[SNAPSHOT] target={snapshot_path}")
    if not dry_run:
        write_json(snapshot_path, payload)
    return snapshot_path


def run_validate(dry_run: bool) -> int:
    return run_python_script(TOOLS_DIR / "check_release_readiness.py", dry_run=dry_run)


def run_monitor(dry_run: bool) -> int:
    return run_python_script(TOOLS_DIR / "check_release_monitoring.py", dry_run=dry_run)


def run_recall(dry_run: bool) -> int:
    ctx = current_release_context()
    bucket = ctx["contract"].get("bucket", "")
    env_updates = {
        "INPUT_09_RUN_DIR": ctx["stage09_run"],
        "MIN_TRAIN_BUCKETS_OVERRIDE": bucket,
        "OUTPUT_09_AUDIT_ROOT_DIR": REPO_ROOT / "data/output/09_recall_audit",
        "METRICS_DIR": REPO_ROOT / "data/metrics",
    }
    return run_python_script(SCRIPTS_DIR / "09_1_recall_audit.py", env_updates=env_updates, dry_run=dry_run)


def run_rank(dry_run: bool) -> int:
    ctx = current_release_context()
    contract = ctx["contract"]
    stage10_pointer = ctx["stage10_pointer"]
    env_updates = {
        "INPUT_09_RUN_DIR": ctx["stage09_run"],
        "RANK_MODEL_JSON": stage10_pointer.get("rank_model_json", ""),
        "RANK_EVAL_USER_COHORT_PATH": stage10_pointer.get("eval_user_cohort_path", ""),
        "RANK_EVAL_CANDIDATE_TOPN": contract.get("candidate_topn", ""),
        "RANK_RERANK_TOPN": contract.get("candidate_topn", ""),
        "RANK_EVAL_TOP_K": contract.get("top_k", ""),
        "RANK_BUCKETS_OVERRIDE": contract.get("bucket", ""),
        "RANK_DIAGNOSTICS_ENABLE": "false",
        "RANK_AUDIT_ENABLE": "false",
        "OUTPUT_10_2_ROOT_DIR": REPO_ROOT / "data/output/10_2_rank_infer_eval",
        "STAGE10_RESULTS_METRICS_PATH": REPO_ROOT / "data/metrics/recsys_stage10_results.csv",
    }
    return run_python_script(SCRIPTS_DIR / "10_2_rank_infer_eval.py", env_updates=env_updates, dry_run=dry_run)


def _stage11_runtime_env(ctx: dict[str, Any]) -> dict[str, Any]:
    contract = ctx["contract"]
    stage11_pointer = ctx["stage11_pointer"]
    stage11_run_meta = read_json(ctx["stage11_run"] / "run_meta.json")
    env_updates: dict[str, Any] = {
        "INPUT_09_RUN_DIR": ctx["stage09_run"],
        "INPUT_11_2_RUN_DIR": stage11_pointer.get("source_run_11_2", ""),
        "INPUT_11_DATA_RUN_DIR": stage11_pointer.get("source_run_11_1_data", ""),
        "BUCKETS_OVERRIDE": contract.get("bucket", ""),
        "RANK_EVAL_TOP_K": contract.get("top_k", ""),
        "QLORA_RERANK_TOPN": contract.get("candidate_topn", ""),
        "QLORA_BLEND_ALPHA": stage11_pointer.get("recorded_blend_alpha", ""),
        "QLORA_ENFORCE_STAGE09_GATE": str(bool(stage11_run_meta.get("enforce_stage09_gate", True))).lower(),
        "QLORA_EVAL_USE_STAGE11_SPLIT": str(bool(stage11_run_meta.get("use_stage11_eval_split", True))).lower(),
        "QLORA_EVAL_PROFILE": stage11_run_meta.get("eval_profile", ""),
        "QLORA_EVAL_MAX_USERS_PER_BUCKET": stage11_run_meta.get("max_users_per_bucket", ""),
        "QLORA_EVAL_MAX_ROWS_PER_BUCKET": stage11_run_meta.get("max_rows_per_bucket", ""),
        "QLORA_EVAL_ROW_CAP_ORDERED": str(bool(stage11_run_meta.get("row_cap_ordered", False))).lower(),
        "QLORA_PROMPT_MODE": stage11_run_meta.get("prompt_mode", ""),
        "QLORA_EVAL_PROMPT_BUILD_MODE": stage11_run_meta.get("prompt_build_mode", ""),
        "QLORA_EVAL_ATTN_IMPLEMENTATION": stage11_run_meta.get("attn_implementation", ""),
        "QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED": str(bool(stage11_run_meta.get("review_base_cache_enabled", False))).lower(),
        "QLORA_EVAL_REVIEW_BASE_CACHE_ROOT": stage11_run_meta.get("review_base_cache_root", ""),
        "OUTPUT_11_SIDECAR_ROOT_DIR": REPO_ROOT / "data/output/11_qlora_sidecar_eval",
        "METRICS_STAGE11_SIDECAR_PATH": REPO_ROOT / "data/metrics/recsys_stage11_qlora_sidecar_results.csv",
    }
    eval_user_cohort_path = str(stage11_run_meta.get("eval_user_cohort_path", "")).strip()
    if eval_user_cohort_path:
        env_updates["QLORA_EVAL_USER_COHORT_PATH"] = eval_user_cohort_path
    return env_updates


def run_eval(dry_run: bool) -> int:
    ctx = current_release_context()
    env_updates = _stage11_runtime_env(ctx)
    return run_python_script(SCRIPTS_DIR / "11_3_qlora_sidecar_eval.py", env_updates=env_updates, dry_run=dry_run)


def current_release_manifest_path() -> Path:
    if FIRST_CHAMPION_MANIFEST.exists():
        return FIRST_CHAMPION_MANIFEST
    raise FileNotFoundError("default frozen champion manifest not found; pass --manifest-path explicitly")


def build_publish_payload(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    release_label = str(manifest.get("version_label", "")).strip()
    if not release_label:
        raise ValueError(f"manifest missing version_label: {manifest_path}")
    contract = manifest.get("alignment_contract", {}) if isinstance(manifest.get("alignment_contract"), dict) else {}
    policy_block = manifest.get("release_policy", {}) if isinstance(manifest.get("release_policy"), dict) else {}
    champion_lineage = manifest.get("champion_lineage", {}) if isinstance(manifest.get("champion_lineage"), dict) else {}
    fallback_lineage = (
        manifest.get("aligned_fallback_lineage", {})
        if isinstance(manifest.get("aligned_fallback_lineage"), dict)
        else {}
    )

    source_run_09 = str(contract.get("source_run_09", "")).strip()
    stage09_run_dir = REPO_ROOT / "data/output/09_candidate_fusion" / source_run_09
    stage09_audit = champion_lineage.get("stage09_audit", {}) if isinstance(champion_lineage.get("stage09_audit"), dict) else {}
    stage10_model_run = (
        fallback_lineage.get("stage10_model_run", {})
        if isinstance(fallback_lineage.get("stage10_model_run"), dict)
        else {}
    )
    stage10_eval = (
        fallback_lineage.get("stage10_aligned_eval", {})
        if isinstance(fallback_lineage.get("stage10_aligned_eval"), dict)
        else {}
    )
    stage11_data = (
        champion_lineage.get("stage11_1_dataset", {})
        if isinstance(champion_lineage.get("stage11_1_dataset"), dict)
        else {}
    )
    stage11_model = (
        champion_lineage.get("stage11_2_dpo_eval_run", {})
        if isinstance(champion_lineage.get("stage11_2_dpo_eval_run"), dict)
        else {}
    )
    stage11_eval = (
        champion_lineage.get("stage11_3_eval", {})
        if isinstance(champion_lineage.get("stage11_3_eval"), dict)
        else {}
    )
    metrics_block = manifest.get("metrics", {}) if isinstance(manifest.get("metrics"), dict) else {}
    stage09_metrics = metrics_block.get("stage09_audit", {}) if isinstance(metrics_block.get("stage09_audit"), dict) else {}
    stage10_metrics = metrics_block.get("stage10_aligned", {}) if isinstance(metrics_block.get("stage10_aligned"), dict) else {}
    stage11_metrics = metrics_block.get("stage11_champion", {}) if isinstance(metrics_block.get("stage11_champion"), dict) else {}

    stage10_eval_meta_path = Path(str(stage10_eval.get("local_run_meta", "")).strip())
    if not stage10_eval_meta_path.exists():
        raise FileNotFoundError(f"stage10 aligned run_meta missing: {stage10_eval_meta_path}")
    stage10_run_dir = stage10_eval_meta_path.parent

    stage11_run_dir = Path(str(stage11_eval.get("local_run_dir", "")).strip())
    if not stage11_run_dir.exists():
        raise FileNotFoundError(f"stage11 eval run_dir missing: {stage11_run_dir}")

    existing_policy = read_production_run_pointer("release_policy") or {}
    approved_at = str(existing_policy.get("approved_at", "")).strip()
    if not approved_at:
        approved_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    policy_payload = {
        "pointer_name": "release_policy",
        "release_label": release_label,
        "approved_at": approved_at,
        "champion_pointer": "stage11_release",
        "aligned_fallback_pointer": "stage10_release",
        "emergency_baseline_pointer": "stage09_release",
        "release_contract": {
            "source_run_09": source_run_09,
            "bucket": contract.get("bucket", ""),
            "eval_users": contract.get("eval_users", ""),
            "candidate_topn": contract.get("candidate_topn", ""),
            "top_k": contract.get("top_k", ""),
        },
        "notes": [
            "Keep _latest_runs for experiments only.",
            "Do not let smoke or partial runs overwrite _prod_runs.",
            "Publish from a frozen champion manifest, not from an ad hoc run.",
            f"release_policy_source={manifest_path}",
        ],
    }

    stage09_payload = {
        "pointer_name": "stage09_release",
        "run_dir": str(stage09_run_dir),
        "release_label": release_label,
        "release_role": "emergency_baseline_source",
        "source_run_09": source_run_09,
        "audit_run_id": stage09_audit.get("audit_run_id", ""),
        "audit_metrics_file": stage09_audit.get("local_metrics_file", ""),
        "bucket": contract.get("bucket", ""),
        "truth_in_all": stage09_metrics.get("truth_in_all", ""),
        "truth_in_pretrim": stage09_metrics.get("truth_in_pretrim", ""),
        "pretrim_cut_loss": stage09_metrics.get("pretrim_cut_loss", ""),
        "hard_miss": stage09_metrics.get("hard_miss", ""),
    }

    stage10_payload = {
        "pointer_name": "stage10_release",
        "run_dir": str(stage10_run_dir),
        "release_label": release_label,
        "release_role": "aligned_fallback",
        "source_run_09": source_run_09,
        "rank_model_json": str(Path(str(stage10_model_run.get("local_model_dir", "")).strip()) / "rank_model.json"),
        "eval_user_cohort_path": stage10_eval.get("eval_user_cohort_path", ""),
        "metrics_file": stage10_eval.get("metrics_file", ""),
        "eval_candidate_topn": contract.get("candidate_topn", ""),
        "top_k": contract.get("top_k", ""),
        "selected_model": "LearnedBlendXGBCls@10",
        "recall_at_k": stage10_metrics.get("learned_blend_xgb_cls_at_10", {}).get("recall", ""),
        "ndcg_at_k": stage10_metrics.get("learned_blend_xgb_cls_at_10", {}).get("ndcg", ""),
        "fallback_from_release_policy": True,
    }

    stage11_payload = {
        "pointer_name": "stage11_release",
        "run_dir": str(stage11_run_dir),
        "release_label": release_label,
        "release_role": "champion",
        "source_run_09": source_run_09,
        "source_run_11_1_data": stage11_data.get("local_run_dir", ""),
        "source_run_11_2": stage11_model.get("local_run_dir", ""),
        "adapter_dir": str(Path(str(stage11_model.get("local_run_dir", "")).strip()) / "adapter"),
        "metrics_file": stage11_eval.get("metrics_file", ""),
        "alpha_sweep_best_file": stage11_eval.get("alpha_sweep_best_file", ""),
        "top_k": contract.get("top_k", ""),
        "rerank_topn": contract.get("candidate_topn", ""),
        "recorded_blend_alpha": stage11_eval.get("recorded_blend_alpha", ""),
        "alpha_sweep_best": stage11_eval.get("alpha_sweep_best", ""),
        "selected_model": "QLoRASidecar@10",
        "recall_at_k": stage11_metrics.get("qlora_sidecar_at_10", {}).get("recall", ""),
        "ndcg_at_k": stage11_metrics.get("qlora_sidecar_at_10", {}).get("ndcg", ""),
        "champion_from_release_policy": True,
    }

    prod_manifest_payload = {
        "release_label": release_label,
        "published_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_manifest_path": str(manifest_path),
        "status": {
            "internal_pilot_ready": True,
            "production_ready": False,
        },
        "policy_source": policy_block,
        "release_contract": policy_payload["release_contract"],
        "prod_pointers": {
            "stage09_release": stage09_payload["run_dir"],
            "stage10_release": stage10_payload["run_dir"],
            "stage11_release": stage11_payload["run_dir"],
        },
    }
    return {
        "release_label": release_label,
        "policy": policy_payload,
        "stage09": stage09_payload,
        "stage10": stage10_payload,
        "stage11": stage11_payload,
        "prod_manifest": prod_manifest_payload,
    }


def publish_release(manifest_path: Path, dry_run: bool) -> int:
    payload = build_publish_payload(manifest_path)
    release_label = payload["release_label"]
    prod_manifest_path = PROD_RUN_DIR / f"release_manifest_{sanitize_name(release_label)}.json"

    print(f"[PUBLISH] manifest_source={manifest_path}")
    print(f"[PUBLISH] release_label={release_label}")
    print(f"[PUBLISH] prod_manifest={prod_manifest_path}")
    existing_policy = read_production_run_pointer("release_policy")
    if existing_policy is not None:
        write_current_snapshot(reason=f"before_publish:{release_label}", dry_run=dry_run)
    if dry_run:
        print("[DRY-RUN] prod pointers not updated")
        return 0

    write_production_run_pointer("stage09_release", payload["stage09"]["run_dir"], payload["stage09"])
    write_production_run_pointer("stage10_release", payload["stage10"]["run_dir"], payload["stage10"])
    write_production_run_pointer("stage11_release", payload["stage11"]["run_dir"], payload["stage11"])
    write_production_run_pointer("release_policy", REPO_ROOT, payload["policy"])
    prod_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(prod_manifest_path, payload["prod_manifest"])
    print("[PUBLISH] prod pointers updated")
    return 0


def rollback_release(snapshot_path: Path, dry_run: bool) -> int:
    snapshot = read_json(snapshot_path)
    pointers = snapshot.get("pointers", {}) if isinstance(snapshot.get("pointers"), dict) else {}
    if not pointers:
        raise ValueError(f"snapshot has no pointers: {snapshot_path}")
    print(f"[ROLLBACK] snapshot_source={snapshot_path}")
    print(f"[ROLLBACK] snapshot_release_label={snapshot.get('release_label', '')}")
    write_current_snapshot(reason=f"before_rollback:{snapshot_path.name}", dry_run=dry_run)
    if dry_run:
        print("[DRY-RUN] rollback not applied")
        return 0

    for pointer_name in ("stage09_release", "stage10_release", "stage11_release", "release_policy"):
        payload = pointers.get(pointer_name)
        if not isinstance(payload, dict):
            continue
        run_dir = payload.get("run_dir", REPO_ROOT)
        write_production_run_pointer(pointer_name, run_dir, payload)

    applied_path = PROD_RUN_DIR / f"rollback_applied_{timestamp_id()}_{sanitize_name(snapshot.get('release_label', 'rollback'))}.json"
    write_json(
        applied_path,
        {
            "applied_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
            "snapshot_source": str(snapshot_path),
            "restored_release_label": str(snapshot.get("release_label", "")).strip(),
        },
    )
    print(f"[ROLLBACK] restored pointers from {snapshot_path.name}")
    print(f"[ROLLBACK] audit_record={applied_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal pilot batch runner for the frozen release path.")
    parser.add_argument(
        "--mode",
        choices=("validate", "monitor", "recall", "rank", "eval", "publish", "rollback"),
        required=True,
        help="Runner mode.",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Frozen champion manifest used by publish mode. Defaults to the current first champion manifest.",
    )
    parser.add_argument(
        "--snapshot-path",
        default="",
        help="Rollback snapshot used by rollback mode. Defaults to the latest rollback_snapshot_*.json under _prod_runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command or publish target without executing it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "validate":
        return run_validate(dry_run=args.dry_run)
    if args.mode == "monitor":
        return run_monitor(dry_run=args.dry_run)
    if args.mode == "recall":
        return run_recall(dry_run=args.dry_run)
    if args.mode == "rank":
        return run_rank(dry_run=args.dry_run)
    if args.mode == "eval":
        return run_eval(dry_run=args.dry_run)
    if args.mode == "rollback":
        snapshot_path = Path(args.snapshot_path).expanduser() if args.snapshot_path else latest_rollback_snapshot_path()
        if snapshot_path is None:
            raise FileNotFoundError(f"no rollback snapshot found under {PROD_RUN_DIR}")
        return rollback_release(snapshot_path=snapshot_path, dry_run=args.dry_run)
    manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else current_release_manifest_path()
    return publish_release(manifest_path=manifest_path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
