from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline.project_paths import normalize_legacy_project_path, project_path
from pipeline.run_validators import (
    validate_stage09_candidate_run,
    validate_stage10_infer_eval_run,
    validate_stage10_rank_model_run,
)


RUNNER_NAME = "bucket_stage10_gate_runner.py"
STAGE09_SUFFIX = "_stage09_candidate_fusion"
STAGE09_AUDIT_SUFFIX = "_stage09_recall_audit"
STAGE10_TRAIN_SUFFIX = "_stage10_1_rank_train"
STAGE10_EVAL_SUFFIX = "_stage10_2_rank_infer_eval"
VALIDATORS = {
    "stage09_candidate": validate_stage09_candidate_run,
    "stage10_rank_model": validate_stage10_rank_model_run,
    "stage10_infer_eval": validate_stage10_infer_eval_run,
}

DEFAULT_STAGE08_CLUSTER_ROOT = "data/output/08_cluster_labels/full"
DEFAULT_STAGE09_USER_PROFILE_RUN = "data/output/09_user_profiles/20260304_234037_full_stage09_user_profile_build"
DEFAULT_STAGE09_ITEM_SEMANTIC_RUN = "data/output/09_item_semantics/20260305_000408_full_stage09_item_semantic_build"
DEFAULT_STAGE10_GATE_OUTPUT_ROOT = "data/output/stage10_gate"
DEFAULT_STAGE10_GATE_METRICS_ROOT = "data/metrics/stage10_gate"
DEFAULT_TEMP_ROOT = "data/spark-tmp/stage10_gate"


def timestamp_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json object required: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def sanitize_label(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(raw or "").strip()).strip("._")
    return safe or "bucket_stage10_gate"


def abs_path(raw: str | Path) -> Path:
    return normalize_legacy_project_path(project_path(raw)).resolve()


def path_or_none(raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    return normalize_legacy_project_path(text).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_run_dirs(root: Path, suffix: str) -> dict[str, float]:
    if not root.exists():
        return {}
    out: dict[str, float] = {}
    for child in root.iterdir():
        if child.is_dir() and child.name.endswith(suffix):
            out[child.name] = child.stat().st_mtime
    return out


def detect_new_run_dir(root: Path, suffix: str, before: dict[str, float], started_at: float) -> Path:
    candidates: list[Path] = []
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.endswith(suffix):
                continue
            if child.name not in before:
                candidates.append(child)
    if candidates:
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return candidates[0]

    fallback: list[Path] = []
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir() or not child.name.endswith(suffix):
                continue
            if child.stat().st_mtime >= started_at - 1.0:
                fallback.append(child)
    if fallback:
        fallback.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return fallback[0]
    raise FileNotFoundError(f"could not detect new run dir in {root} suffix={suffix}")


def format_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("BDA_PROJECT_ROOT", str(REPO_ROOT))
    return env


def print_env_updates(env_updates: dict[str, Any]) -> None:
    if not env_updates:
        return
    print("[ENV]")
    for key in sorted(env_updates):
        value = env_updates[key]
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        print(f"  {key}={text}")


def run_python_script(script_relative: str, env_updates: dict[str, Any], dry_run: bool) -> int:
    command = [sys.executable, str(REPO_ROOT / script_relative)]
    env = base_env()
    for key, value in env_updates.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        env[key] = text
    print(f"[RUN] {format_command(command)}")
    print_env_updates(env_updates)
    if dry_run:
        print("[DRY-RUN] command not executed")
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return int(completed.returncode)


def validator_errors(kind: str, run_dir: Path) -> list[str]:
    return VALIDATORS[kind](run_dir)


def validator_passes(kind: str, run_dir: Path) -> tuple[bool, list[str]]:
    errors = validator_errors(kind, run_dir)
    return (not errors, errors)


def manifest_template(args: argparse.Namespace, manifest_path: Path, roots: dict[str, Path]) -> dict[str, Any]:
    bucket_name = f"bucket_{args.bucket}"
    metrics_dir = roots["metrics_root"] / bucket_name
    return {
        "manifest_version": 1,
        "runner": str(REPO_ROOT / "scripts/pipeline" / RUNNER_NAME),
        "label": sanitize_label(args.label),
        "bucket": int(args.bucket),
        "scope": "stage09_to_stage10",
        "stage11_in_scope": False,
        "release_pointers_untouched": True,
        "created_at": timestamp_now(),
        "updated_at": timestamp_now(),
        "manifest_path": str(manifest_path),
        "shared_contracts": {
            "stage08_cluster_profile_root": str(abs_path(args.stage08_cluster_root)),
            "stage09_user_profile_run_dir": str(abs_path(args.user_profile_run_dir)),
            "stage09_item_semantic_run_dir": str(abs_path(args.item_semantic_run_dir)),
        },
        "roots": {
            "stage10_gate_output_root": str(roots["stage10_gate_output_root"]),
            "metrics_root": str(roots["metrics_root"]),
            "temp_root": str(roots["temp_root"]),
            "stage09_output_root": str(roots["stage09_output_root"]),
            "stage09_recall_audit_output_root": str(roots["stage09_recall_audit_output_root"]),
            "stage10_train_output_root": str(roots["stage10_train_output_root"]),
            "stage10_eval_output_root": str(roots["stage10_eval_output_root"]),
            "metrics_dir": str(metrics_dir),
        },
        "policy": {
            "recall_profile": str(args.recall_profile),
            "train_model_backend": "xgboost_cls",
            "use_explicit_eval_cohort": False,
            "bucket_stage_order": [
                "stage09_candidate",
                "stage09_recall_audit",
                "stage10_rank_model",
                "stage10_infer_eval",
            ],
        },
        "artifacts": {
            "stage09_run_dir": "",
            "stage09_recall_audit_run_dir": "",
            "stage10_rank_model_run_dir": "",
            "stage10_rank_model_json": "",
            "stage10_infer_eval_run_dir": "",
        },
        "steps": {
            "stage09_candidate": {"status": "pending"},
            "stage09_recall_audit": {"status": "pending"},
            "stage10_rank_model": {"status": "pending"},
            "stage10_infer_eval": {"status": "pending"},
            "validate": {"status": "pending"},
        },
        "validators": {},
        "summaries": {},
        "commands": [],
    }


def load_manifest(args: argparse.Namespace, manifest_path: Path, roots: dict[str, Path]) -> dict[str, Any]:
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = manifest_template(args, manifest_path, roots)
    manifest.setdefault("artifacts", {})
    manifest.setdefault("steps", {})
    manifest.setdefault("validators", {})
    manifest.setdefault("summaries", {})
    manifest.setdefault("commands", [])
    manifest.setdefault("shared_contracts", {})
    manifest.setdefault("roots", {})
    manifest.setdefault("policy", {})
    manifest["manifest_version"] = 1
    manifest["runner"] = str(REPO_ROOT / "scripts/pipeline" / RUNNER_NAME)
    manifest["label"] = sanitize_label(args.label)
    manifest["bucket"] = int(args.bucket)
    manifest["scope"] = "stage09_to_stage10"
    manifest["stage11_in_scope"] = False
    manifest["release_pointers_untouched"] = True
    manifest["manifest_path"] = str(manifest_path)
    manifest["updated_at"] = timestamp_now()
    manifest["shared_contracts"]["stage08_cluster_profile_root"] = str(abs_path(args.stage08_cluster_root))
    manifest["shared_contracts"]["stage09_user_profile_run_dir"] = str(abs_path(args.user_profile_run_dir))
    manifest["shared_contracts"]["stage09_item_semantic_run_dir"] = str(abs_path(args.item_semantic_run_dir))
    manifest["roots"].update(
        {
            "stage10_gate_output_root": str(roots["stage10_gate_output_root"]),
            "metrics_root": str(roots["metrics_root"]),
            "temp_root": str(roots["temp_root"]),
            "stage09_output_root": str(roots["stage09_output_root"]),
            "stage09_recall_audit_output_root": str(roots["stage09_recall_audit_output_root"]),
            "stage10_train_output_root": str(roots["stage10_train_output_root"]),
            "stage10_eval_output_root": str(roots["stage10_eval_output_root"]),
            "metrics_dir": str(roots["metrics_root"] / f"bucket_{args.bucket}"),
        }
    )
    manifest["policy"]["recall_profile"] = str(args.recall_profile)
    manifest["policy"]["train_model_backend"] = "xgboost_cls"
    manifest["policy"]["use_explicit_eval_cohort"] = False
    for step in ("stage09_candidate", "stage09_recall_audit", "stage10_rank_model", "stage10_infer_eval", "validate"):
        manifest["steps"].setdefault(step, {"status": "pending"})
    return manifest


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = timestamp_now()
    write_json(manifest_path, manifest)


def mark_step(manifest: dict[str, Any], step: str, status: str, **extra: Any) -> None:
    payload = manifest.setdefault("steps", {}).setdefault(step, {})
    payload["status"] = status
    if status == "running":
        payload["started_at"] = timestamp_now()
        payload.pop("completed_at", None)
        payload.pop("return_code", None)
    if status in {"completed", "failed", "reused"}:
        payload["completed_at"] = timestamp_now()
    if status in {"completed", "reused"}:
        payload.pop("return_code", None)
    for key, value in extra.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value


def append_command(manifest: dict[str, Any], stage: str, script_relative: str, env_updates: dict[str, Any]) -> None:
    manifest.setdefault("commands", []).append(
        {
            "recorded_at": timestamp_now(),
            "stage": stage,
            "script": script_relative,
            "env": {k: str(v) for k, v in env_updates.items() if v is not None and str(v).strip()},
        }
    )


def require_existing(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"missing required path: {label}")
    if not path.exists():
        raise FileNotFoundError(f"path not found for {label}: {path}")
    return path


def artifact_from_manifest(manifest: dict[str, Any], key: str) -> Path | None:
    return path_or_none(manifest.get("artifacts", {}).get(key))


def require_known_path(path: Path | None, label: str, dry_run: bool = False) -> Path:
    if path is None:
        raise FileNotFoundError(f"missing required path: {label}")
    if dry_run or path.exists():
        return path
    raise FileNotFoundError(f"path not found for {label}: {path}")


def explicit_or_manifest(arg_value: str | None, manifest: dict[str, Any], key: str) -> Path | None:
    text = str(arg_value or "").strip()
    if text:
        return require_existing(abs_path(text), key)
    manifest_path = artifact_from_manifest(manifest, key)
    if manifest_path is not None and manifest_path.exists():
        return manifest_path
    return None


def shared_temp_env(temp_root: Path, bucket: int) -> dict[str, Any]:
    bucket_root = ensure_dir(temp_root / f"bucket_{bucket}")
    return {
        "SPARK_LOCAL_DIR": str(ensure_dir(bucket_root / "spark_local")),
        "PY_TEMP_DIR": str(ensure_dir(bucket_root / "pytemp")),
        "SPARK_TMP_SESSION_ISOLATION": "true",
        "SPARK_TMP_AUTOCLEAN_ENABLED": "true",
        "SPARK_TMP_CLEAN_ON_EXIT": "true",
        "SPARK_TMP_RETENTION_HOURS": "8",
        "SPARK_TMP_CLEAN_MAX_ENTRIES": "3000",
        "SPARK_NETWORK_TIMEOUT": "600s",
        "SPARK_EXECUTOR_HEARTBEAT_INTERVAL": "60s",
    }


def stage09_env(args: argparse.Namespace, roots: dict[str, Path]) -> dict[str, Any]:
    env = {
        "RUN_PROFILE_OVERRIDE": "full",
        "RECALL_PROFILE_OVERRIDE": str(args.recall_profile),
        "MIN_TRAIN_BUCKETS_OVERRIDE": str(args.bucket),
        "OUTPUT_ROOT_DIR": str(roots["stage09_output_root"]),
        "CLUSTER_PROFILE_ROOT_DIR": str(abs_path(args.stage08_cluster_root)),
        "USER_PROFILE_RUN_DIR": str(abs_path(args.user_profile_run_dir)),
        "ITEM_SEMANTIC_RUN_DIR": str(abs_path(args.item_semantic_run_dir)),
        "ENABLE_PROFILE_RECALL": "true",
        "ENABLE_ITEM_SEMANTIC": "true",
        "SPARK_MASTER": str(args.stage09_spark_master),
        "SPARK_DRIVER_MEMORY": str(args.spark_driver_memory),
        "SPARK_EXECUTOR_MEMORY": str(args.spark_executor_memory),
        "SPARK_SQL_SHUFFLE_PARTITIONS": str(args.stage09_shuffle_partitions),
        "SPARK_DEFAULT_PARALLELISM": str(args.stage09_default_parallelism),
        "SPARK_SQL_ADAPTIVE_ENABLED": "true",
        "LOCAL_PARQUET_WRITE_MODE": "driver_parquet",
        "LOCAL_PARQUET_WRITE_CHUNK_ROWS": "50000",
    }
    env.update(shared_temp_env(roots["temp_root"], args.bucket))
    return env


def audit_env(args: argparse.Namespace, roots: dict[str, Path], stage09_run_dir: Path) -> dict[str, Any]:
    metrics_dir = ensure_dir(roots["metrics_root"] / f"bucket_{args.bucket}")
    env = {
        "INPUT_09_RUN_DIR": str(stage09_run_dir),
        "OUTPUT_09_AUDIT_ROOT_DIR": str(roots["stage09_recall_audit_output_root"]),
        "METRICS_DIR": str(metrics_dir),
        "MIN_TRAIN_BUCKETS_OVERRIDE": str(args.bucket),
        "SPARK_MASTER": str(args.stage10_spark_master),
        "SPARK_DRIVER_MEMORY": str(args.spark_driver_memory),
        "SPARK_EXECUTOR_MEMORY": str(args.spark_executor_memory),
        "SPARK_SQL_SHUFFLE_PARTITIONS": str(args.audit_shuffle_partitions),
        "SPARK_DEFAULT_PARALLELISM": str(args.audit_default_parallelism),
    }
    env.update(shared_temp_env(roots["temp_root"], args.bucket))
    return env


def stage10_train_env(args: argparse.Namespace, roots: dict[str, Path], stage09_run_dir: Path) -> dict[str, Any]:
    env = {
        "INPUT_09_RUN_DIR": str(stage09_run_dir),
        "OUTPUT_10_1_ROOT_DIR": str(roots["stage10_train_output_root"]),
        "TRAIN_BUCKETS_OVERRIDE": str(args.bucket),
        "TRAIN_MODEL_BACKEND": "xgboost_cls",
        "SPARK_MASTER": str(args.stage10_spark_master),
        "SPARK_DRIVER_MEMORY": str(args.spark_driver_memory),
        "SPARK_EXECUTOR_MEMORY": str(args.spark_executor_memory),
        "SPARK_SQL_SHUFFLE_PARTITIONS": str(args.stage10_shuffle_partitions),
        "SPARK_DEFAULT_PARALLELISM": str(args.stage10_default_parallelism),
        "SPARK_SQL_ADAPTIVE_ENABLED": "false",
        "PY_TEMP_SESSION_PREFIX": f"b{int(args.bucket)}_stage10_1",
    }
    env.update(shared_temp_env(roots["temp_root"], args.bucket))
    return env


def stage10_eval_env(args: argparse.Namespace, roots: dict[str, Path], stage09_run_dir: Path, rank_model_run_dir: Path) -> dict[str, Any]:
    metrics_dir = ensure_dir(roots["metrics_root"] / f"bucket_{args.bucket}")
    env = {
        "INPUT_09_RUN_DIR": str(stage09_run_dir),
        "RANK_MODEL_JSON": str(rank_model_run_dir / "rank_model.json"),
        "OUTPUT_10_2_ROOT_DIR": str(roots["stage10_eval_output_root"]),
        "STAGE10_RESULTS_METRICS_PATH": str(metrics_dir / f"recsys_stage10_bucket_{int(args.bucket)}_results.csv"),
        "RANK_BUCKETS_OVERRIDE": str(args.bucket),
        "RANK_EVAL_TOP_K": "10",
        "RANK_DIAGNOSTICS_ENABLE": "false",
        "RANK_AUDIT_ENABLE": "false",
        "SPARK_MASTER": str(args.stage10_spark_master),
        "SPARK_DRIVER_MEMORY": str(args.spark_driver_memory),
        "SPARK_EXECUTOR_MEMORY": str(args.spark_executor_memory),
        "SPARK_SQL_SHUFFLE_PARTITIONS": str(args.stage10_shuffle_partitions),
        "SPARK_DEFAULT_PARALLELISM": str(args.stage10_default_parallelism),
        "SPARK_SQL_ADAPTIVE_ENABLED": "false",
        "XGB_BATCH_MODE": "hash_partition_memory",
    }
    env.update(shared_temp_env(roots["temp_root"], args.bucket))
    return env


def pick_bucket_row(rows: list[dict[str, Any]], bucket: int, field: str = "bucket") -> dict[str, Any]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get(field)) == int(bucket):
                return row
        except Exception:
            continue
    return {}


def pick_bucket_eval_row(rows: list[dict[str, Any]], bucket: int) -> dict[str, Any]:
    bucket_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            if int(row.get("bucket_min_train_reviews")) != int(bucket):
                continue
        except Exception:
            continue
        bucket_rows.append(row)
    if not bucket_rows:
        return {}

    learned = [row for row in bucket_rows if str(row.get("model", "")).startswith("LearnedBlend")]
    chosen_pool = learned or bucket_rows
    chosen_pool.sort(
        key=lambda row: (
            float(row.get("ndcg_at_k", 0.0)),
            float(row.get("recall_at_k", 0.0)),
            str(row.get("model", "")),
        ),
        reverse=True,
    )
    return chosen_pool[0]


def dry_run_run_dir(root: Path, bucket: int, suffix: str) -> Path:
    return root / f"DRYRUN_bucket_{int(bucket)}{suffix}"

def summarize_stage09(run_dir: Path, bucket: int) -> dict[str, Any]:
    run_meta = read_json(run_dir / "run_meta.json")
    bucket_meta = read_json(run_dir / f"bucket_{bucket}" / "bucket_meta.json")
    return {
        "run_dir": str(run_dir),
        "run_id": run_meta.get("run_id", ""),
        "run_profile": run_meta.get("run_profile", ""),
        "recall_profile": run_meta.get("recall_profile", ""),
        "bucket": int(bucket),
        "n_users": bucket_meta.get("n_users", 0),
        "n_test": bucket_meta.get("n_test", 0),
        "pretrim_top_k_used": bucket_meta.get("pretrim_top_k_used", 0),
        "profile_recall_status": bucket_meta.get("profile_recall_status", ""),
        "item_semantic_enabled": bucket_meta.get("item_semantic_enabled", False),
        "tower_seq_enabled": bucket_meta.get("tower_seq_enabled", False),
        "tower_seq_status": bucket_meta.get("tower_seq_status", ""),
    }


def summarize_stage09_audit(run_dir: Path, bucket: int) -> dict[str, Any]:
    payload = read_json(run_dir / "stage09_recall_audit.json")
    row = pick_bucket_row(payload.get("rows", []), bucket, field="bucket")
    if not row:
        raise KeyError(f"bucket {bucket} not found in {run_dir / 'stage09_recall_audit.json'}")
    return {
        "run_dir": str(run_dir),
        "run_id": payload.get("run_id", ""),
        "bucket": int(bucket),
        "truth_in_all": row.get("truth_in_all", 0.0),
        "truth_in_pretrim": row.get("truth_in_pretrim", 0.0),
        "pretrim_cut_loss": row.get("pretrim_cut_loss", 0.0),
        "hard_miss": row.get("hard_miss", 0.0),
        "n_users": row.get("n_users", 0),
        "pretrim_file": row.get("pretrim_file", ""),
    }


def summarize_stage10_train(run_dir: Path, bucket: int) -> dict[str, Any]:
    payload = read_json(run_dir / "rank_model.json")
    model_payload = payload.get("models_by_bucket", {}).get(str(bucket), {})
    metrics = model_payload.get("metrics", {}) if isinstance(model_payload, dict) else {}
    split_users = model_payload.get("split_users", {}) if isinstance(model_payload, dict) else {}
    return {
        "run_dir": str(run_dir),
        "run_id": payload.get("run_id", ""),
        "bucket": int(bucket),
        "model_backend": model_payload.get("model_backend", ""),
        "model_file": model_payload.get("model_file", ""),
        "blend_alpha": metrics.get("blend_alpha", ""),
        "blend_ndcg_at_k": metrics.get("blend_ndcg_at_k", ""),
        "blend_recall_at_k": metrics.get("blend_recall_at_k", ""),
        "train_rows": metrics.get("train_rows", 0),
        "train_pos": metrics.get("train_pos", 0),
        "train_users": metrics.get("train_users", 0),
        "valid_users": metrics.get("valid_users", 0),
        "test_users": metrics.get("test_users", 0),
        "test_users_file": split_users.get("test_users_file", ""),
    }


def summarize_stage10_eval(run_dir: Path, bucket: int) -> dict[str, Any]:
    payload = read_json(run_dir / "run_meta.json")
    row = pick_bucket_eval_row(payload.get("rows", []), bucket)
    if not row:
        raise KeyError(f"bucket {bucket} not found in {run_dir / 'run_meta.json'} rows")
    return {
        "run_dir": str(run_dir),
        "run_id": payload.get("run_id_10", ""),
        "bucket": int(bucket),
        "selected_model": row.get("model", ""),
        "recall_at_k": row.get("recall_at_k", 0.0),
        "ndcg_at_k": row.get("ndcg_at_k", 0.0),
        "n_users": row.get("n_users", 0),
        "n_items": row.get("n_items", 0),
        "n_candidates": row.get("n_candidates", 0),
        "source_stage09_run": row.get("source_run_09", ""),
    }


def record_validator(manifest: dict[str, Any], kind: str, run_dir: Path) -> None:
    passed, errors = validator_passes(kind, run_dir)
    manifest.setdefault("validators", {})[kind] = {
        "checked_at": timestamp_now(),
        "run_dir": str(run_dir),
        "status": "pass" if passed else "fail",
        "errors": errors,
    }
    if not passed:
        raise RuntimeError(f"validator failed for {kind}: {errors}")


def maybe_reuse(step: str, artifact_path: Path | None, force: bool) -> Path | None:
    if force:
        return None
    if artifact_path is None:
        return None
    if artifact_path.exists():
        print(f"[SKIP] reuse {step}: {artifact_path}")
        return artifact_path
    return None


def run_stage09(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path, roots: dict[str, Path]) -> Path:
    reused = maybe_reuse("stage09_candidate", explicit_or_manifest(args.stage09_run_dir, manifest, "stage09_run_dir"), args.force)
    if reused is not None:
        mark_step(manifest, "stage09_candidate", "reused", run_dir=reused)
        record_validator(manifest, "stage09_candidate", reused)
        manifest.setdefault("summaries", {})["stage09_candidate"] = summarize_stage09(reused, args.bucket)
        save_manifest(manifest_path, manifest)
        return reused

    env_updates = stage09_env(args, roots)
    before = list_run_dirs(roots["stage09_output_root"], STAGE09_SUFFIX)
    append_command(manifest, "stage09_candidate", "scripts/09_candidate_fusion.py", env_updates)
    mark_step(manifest, "stage09_candidate", "running")
    save_manifest(manifest_path, manifest)
    started_at = time.time()
    code = run_python_script("scripts/09_candidate_fusion.py", env_updates, dry_run=args.dry_run)
    if code != 0:
        mark_step(manifest, "stage09_candidate", "failed", return_code=code)
        save_manifest(manifest_path, manifest)
        raise RuntimeError(f"stage09 candidate fusion failed with code={code}")
    if args.dry_run:
        mark_step(manifest, "stage09_candidate", "completed", dry_run=True)
        save_manifest(manifest_path, manifest)
        return dry_run_run_dir(roots["stage09_output_root"], args.bucket, STAGE09_SUFFIX)

    run_dir = detect_new_run_dir(roots["stage09_output_root"], STAGE09_SUFFIX, before, started_at)
    manifest["artifacts"]["stage09_run_dir"] = str(run_dir)
    record_validator(manifest, "stage09_candidate", run_dir)
    manifest.setdefault("summaries", {})["stage09_candidate"] = summarize_stage09(run_dir, args.bucket)
    mark_step(manifest, "stage09_candidate", "completed", run_dir=run_dir)
    save_manifest(manifest_path, manifest)
    return run_dir


def run_stage09_audit(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path, roots: dict[str, Path], stage09_run_dir: Path) -> Path:
    reused = maybe_reuse(
        "stage09_recall_audit",
        explicit_or_manifest(args.stage09_audit_run_dir, manifest, "stage09_recall_audit_run_dir"),
        args.force,
    )
    if reused is not None:
        mark_step(manifest, "stage09_recall_audit", "reused", run_dir=reused)
        manifest.setdefault("summaries", {})["stage09_recall_audit"] = summarize_stage09_audit(reused, args.bucket)
        save_manifest(manifest_path, manifest)
        return reused

    env_updates = audit_env(args, roots, stage09_run_dir)
    before = list_run_dirs(roots["stage09_recall_audit_output_root"], STAGE09_AUDIT_SUFFIX)
    append_command(manifest, "stage09_recall_audit", "scripts/09_1_recall_audit.py", env_updates)
    mark_step(manifest, "stage09_recall_audit", "running")
    save_manifest(manifest_path, manifest)
    started_at = time.time()
    code = run_python_script("scripts/09_1_recall_audit.py", env_updates, dry_run=args.dry_run)
    if code != 0:
        mark_step(manifest, "stage09_recall_audit", "failed", return_code=code)
        save_manifest(manifest_path, manifest)
        raise RuntimeError(f"stage09 recall audit failed with code={code}")
    if args.dry_run:
        mark_step(manifest, "stage09_recall_audit", "completed", dry_run=True)
        save_manifest(manifest_path, manifest)
        return dry_run_run_dir(roots["stage09_recall_audit_output_root"], args.bucket, STAGE09_AUDIT_SUFFIX)

    run_dir = detect_new_run_dir(roots["stage09_recall_audit_output_root"], STAGE09_AUDIT_SUFFIX, before, started_at)
    manifest["artifacts"]["stage09_recall_audit_run_dir"] = str(run_dir)
    manifest.setdefault("summaries", {})["stage09_recall_audit"] = summarize_stage09_audit(run_dir, args.bucket)
    mark_step(manifest, "stage09_recall_audit", "completed", run_dir=run_dir)
    save_manifest(manifest_path, manifest)
    return run_dir


def run_stage10_train(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path, roots: dict[str, Path], stage09_run_dir: Path) -> Path:
    reused = maybe_reuse(
        "stage10_rank_model",
        explicit_or_manifest(args.stage10_rank_model_run_dir, manifest, "stage10_rank_model_run_dir"),
        args.force,
    )
    if reused is not None:
        manifest["artifacts"]["stage10_rank_model_json"] = str(reused / "rank_model.json")
        mark_step(manifest, "stage10_rank_model", "reused", run_dir=reused)
        record_validator(manifest, "stage10_rank_model", reused)
        manifest.setdefault("summaries", {})["stage10_rank_model"] = summarize_stage10_train(reused, args.bucket)
        save_manifest(manifest_path, manifest)
        return reused

    env_updates = stage10_train_env(args, roots, stage09_run_dir)
    before = list_run_dirs(roots["stage10_train_output_root"], STAGE10_TRAIN_SUFFIX)
    append_command(manifest, "stage10_rank_model", "scripts/10_1_rank_train.py", env_updates)
    mark_step(manifest, "stage10_rank_model", "running")
    save_manifest(manifest_path, manifest)
    started_at = time.time()
    code = run_python_script("scripts/10_1_rank_train.py", env_updates, dry_run=args.dry_run)
    if code != 0:
        mark_step(manifest, "stage10_rank_model", "failed", return_code=code)
        save_manifest(manifest_path, manifest)
        raise RuntimeError(f"stage10 rank train failed with code={code}")
    if args.dry_run:
        mark_step(manifest, "stage10_rank_model", "completed", dry_run=True)
        save_manifest(manifest_path, manifest)
        return dry_run_run_dir(roots["stage10_train_output_root"], args.bucket, STAGE10_TRAIN_SUFFIX)

    run_dir = detect_new_run_dir(roots["stage10_train_output_root"], STAGE10_TRAIN_SUFFIX, before, started_at)
    manifest["artifacts"]["stage10_rank_model_run_dir"] = str(run_dir)
    manifest["artifacts"]["stage10_rank_model_json"] = str(run_dir / "rank_model.json")
    record_validator(manifest, "stage10_rank_model", run_dir)
    manifest.setdefault("summaries", {})["stage10_rank_model"] = summarize_stage10_train(run_dir, args.bucket)
    mark_step(manifest, "stage10_rank_model", "completed", run_dir=run_dir)
    save_manifest(manifest_path, manifest)
    return run_dir

def run_stage10_eval(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    roots: dict[str, Path],
    stage09_run_dir: Path,
    rank_model_run_dir: Path,
) -> Path:
    reused = maybe_reuse(
        "stage10_infer_eval",
        explicit_or_manifest(args.stage10_infer_eval_run_dir, manifest, "stage10_infer_eval_run_dir"),
        args.force,
    )
    if reused is not None:
        mark_step(manifest, "stage10_infer_eval", "reused", run_dir=reused)
        record_validator(manifest, "stage10_infer_eval", reused)
        manifest.setdefault("summaries", {})["stage10_infer_eval"] = summarize_stage10_eval(reused, args.bucket)
        save_manifest(manifest_path, manifest)
        return reused

    env_updates = stage10_eval_env(args, roots, stage09_run_dir, rank_model_run_dir)
    before = list_run_dirs(roots["stage10_eval_output_root"], STAGE10_EVAL_SUFFIX)
    append_command(manifest, "stage10_infer_eval", "scripts/10_2_rank_infer_eval.py", env_updates)
    mark_step(manifest, "stage10_infer_eval", "running")
    save_manifest(manifest_path, manifest)
    started_at = time.time()
    code = run_python_script("scripts/10_2_rank_infer_eval.py", env_updates, dry_run=args.dry_run)
    if code != 0:
        mark_step(manifest, "stage10_infer_eval", "failed", return_code=code)
        save_manifest(manifest_path, manifest)
        raise RuntimeError(f"stage10 infer/eval failed with code={code}")
    if args.dry_run:
        mark_step(manifest, "stage10_infer_eval", "completed", dry_run=True)
        save_manifest(manifest_path, manifest)
        return dry_run_run_dir(roots["stage10_eval_output_root"], args.bucket, STAGE10_EVAL_SUFFIX)

    run_dir = detect_new_run_dir(roots["stage10_eval_output_root"], STAGE10_EVAL_SUFFIX, before, started_at)
    manifest["artifacts"]["stage10_infer_eval_run_dir"] = str(run_dir)
    record_validator(manifest, "stage10_infer_eval", run_dir)
    manifest.setdefault("summaries", {})["stage10_infer_eval"] = summarize_stage10_eval(run_dir, args.bucket)
    mark_step(manifest, "stage10_infer_eval", "completed", run_dir=run_dir)
    save_manifest(manifest_path, manifest)
    return run_dir


def run_validate(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    mark_step(manifest, "validate", "running")
    save_manifest(manifest_path, manifest)
    if args.dry_run:
        mark_step(manifest, "validate", "completed", dry_run=True)
        save_manifest(manifest_path, manifest)
        return

    stage09_run_dir = require_existing(explicit_or_manifest(args.stage09_run_dir, manifest, "stage09_run_dir"), "stage09_run_dir")
    stage10_train_run_dir = require_existing(
        explicit_or_manifest(args.stage10_rank_model_run_dir, manifest, "stage10_rank_model_run_dir"),
        "stage10_rank_model_run_dir",
    )
    stage10_eval_run_dir = require_existing(
        explicit_or_manifest(args.stage10_infer_eval_run_dir, manifest, "stage10_infer_eval_run_dir"),
        "stage10_infer_eval_run_dir",
    )
    record_validator(manifest, "stage09_candidate", stage09_run_dir)
    record_validator(manifest, "stage10_rank_model", stage10_train_run_dir)
    record_validator(manifest, "stage10_infer_eval", stage10_eval_run_dir)
    mark_step(manifest, "validate", "completed")
    save_manifest(manifest_path, manifest)


def execution_order(mode: str) -> list[str]:
    if mode == "stage09":
        return ["stage09_candidate"]
    if mode == "audit":
        return ["stage09_recall_audit"]
    if mode == "train":
        return ["stage10_rank_model"]
    if mode == "eval":
        return ["stage10_infer_eval"]
    if mode == "validate":
        return ["validate"]
    return [
        "stage09_candidate",
        "stage09_recall_audit",
        "stage10_rank_model",
        "stage10_infer_eval",
        "validate",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated bucket2/5 stage09->stage10 gate completion on local artifacts.")
    parser.add_argument("--bucket", type=int, choices=[2, 5], required=True, help="Bucket to run through the stage10 gate chain.")
    parser.add_argument(
        "--mode",
        choices=["stage09", "audit", "train", "eval", "validate", "full"],
        default="full",
        help="Gate stage to run. full = stage09 -> audit -> train -> eval -> validate.",
    )
    parser.add_argument("--label", default="", help="Manifest label. Defaults to gl13_bucketX_stage10_gate.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and env without executing the stage scripts.")
    parser.add_argument("--force", action="store_true", help="Rerun the requested stages even if the manifest already records artifacts.")
    parser.add_argument(
        "--gate-output-root",
        "--stage10-gate-output-root",
        "--backfill-output-root",
        dest="gate_output_root",
        default=DEFAULT_STAGE10_GATE_OUTPUT_ROOT,
        help="Isolated output root for bucket stage10 gate runs. Legacy alias: --backfill-output-root.",
    )
    parser.add_argument(
        "--gate-metrics-root",
        "--stage10-gate-metrics-root",
        "--backfill-metrics-root",
        dest="gate_metrics_root",
        default=DEFAULT_STAGE10_GATE_METRICS_ROOT,
        help="Tracked metrics root for bucket stage10 gate runs. Legacy alias: --backfill-metrics-root.",
    )
    parser.add_argument("--temp-root", default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--stage08-cluster-root", default=DEFAULT_STAGE08_CLUSTER_ROOT)
    parser.add_argument("--user-profile-run-dir", default=DEFAULT_STAGE09_USER_PROFILE_RUN)
    parser.add_argument("--item-semantic-run-dir", default=DEFAULT_STAGE09_ITEM_SEMANTIC_RUN)
    parser.add_argument("--recall-profile", default="coverage_stage2")
    parser.add_argument("--spark-driver-memory", default="6g")
    parser.add_argument("--spark-executor-memory", default="6g")
    parser.add_argument("--stage09-spark-master", default="local[4]")
    parser.add_argument("--stage10-spark-master", default="local[2]")
    parser.add_argument("--stage09-shuffle-partitions", type=int, default=12)
    parser.add_argument("--stage09-default-parallelism", type=int, default=12)
    parser.add_argument("--audit-shuffle-partitions", type=int, default=8)
    parser.add_argument("--audit-default-parallelism", type=int, default=8)
    parser.add_argument("--stage10-shuffle-partitions", type=int, default=8)
    parser.add_argument("--stage10-default-parallelism", type=int, default=8)
    parser.add_argument("--stage09-run-dir", default="", help="Optional explicit stage09 run dir for downstream-only modes.")
    parser.add_argument("--stage09-audit-run-dir", default="", help="Optional explicit stage09 recall-audit run dir.")
    parser.add_argument("--stage10-rank-model-run-dir", default="", help="Optional explicit stage10 train run dir.")
    parser.add_argument("--stage10-infer-eval-run-dir", default="", help="Optional explicit stage10 eval run dir.")
    return parser


def build_roots(args: argparse.Namespace) -> dict[str, Path]:
    stage10_gate_output_root = abs_path(args.gate_output_root)
    metrics_root = abs_path(args.gate_metrics_root)
    temp_root = abs_path(args.temp_root)
    return {
        "stage10_gate_output_root": stage10_gate_output_root,
        "metrics_root": metrics_root,
        "temp_root": temp_root,
        "stage09_output_root": stage10_gate_output_root / "09_candidate_fusion",
        "stage09_recall_audit_output_root": stage10_gate_output_root / "09_recall_audit",
        "stage10_train_output_root": stage10_gate_output_root / "10_rank_models",
        "stage10_eval_output_root": stage10_gate_output_root / "10_2_rank_infer_eval",
        "manifest_root": metrics_root / "manifests",
    }


def main() -> int:
    args = build_parser().parse_args()
    if not str(args.label or "").strip():
        args.label = f"gl13_bucket{int(args.bucket)}_stage10_gate"

    roots = build_roots(args)
    for key in (
        "stage10_gate_output_root",
        "metrics_root",
        "temp_root",
        "stage09_output_root",
        "stage09_recall_audit_output_root",
        "stage10_train_output_root",
        "stage10_eval_output_root",
        "manifest_root",
    ):
        ensure_dir(roots[key])

    manifest_path = roots["manifest_root"] / f"{sanitize_label(args.label)}.json"
    manifest = load_manifest(args, manifest_path, roots)
    save_manifest(manifest_path, manifest)

    stage09_run_dir: Path | None = explicit_or_manifest(args.stage09_run_dir, manifest, "stage09_run_dir")
    stage10_train_run_dir: Path | None = explicit_or_manifest(args.stage10_rank_model_run_dir, manifest, "stage10_rank_model_run_dir")

    try:
        for step in execution_order(args.mode):
            if step == "stage09_candidate":
                stage09_run_dir = run_stage09(args, manifest, manifest_path, roots)
            elif step == "stage09_recall_audit":
                stage09_run_dir = require_known_path(stage09_run_dir or explicit_or_manifest(args.stage09_run_dir, manifest, "stage09_run_dir"), "stage09_run_dir", dry_run=args.dry_run)
                run_stage09_audit(args, manifest, manifest_path, roots, stage09_run_dir)
            elif step == "stage10_rank_model":
                stage09_run_dir = require_known_path(stage09_run_dir or explicit_or_manifest(args.stage09_run_dir, manifest, "stage09_run_dir"), "stage09_run_dir", dry_run=args.dry_run)
                stage10_train_run_dir = run_stage10_train(args, manifest, manifest_path, roots, stage09_run_dir)
            elif step == "stage10_infer_eval":
                stage09_run_dir = require_known_path(stage09_run_dir or explicit_or_manifest(args.stage09_run_dir, manifest, "stage09_run_dir"), "stage09_run_dir", dry_run=args.dry_run)
                stage10_train_run_dir = require_known_path(
                    stage10_train_run_dir or explicit_or_manifest(args.stage10_rank_model_run_dir, manifest, "stage10_rank_model_run_dir"),
                    "stage10_rank_model_run_dir",
                    dry_run=args.dry_run,
                )
                run_stage10_eval(args, manifest, manifest_path, roots, stage09_run_dir, stage10_train_run_dir)
            elif step == "validate":
                run_validate(args, manifest, manifest_path)
    except Exception as exc:
        manifest["last_error"] = str(exc)
        save_manifest(manifest_path, manifest)
        print(f"[FAIL] {exc}")
        return 1

    print(f"[OK] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
