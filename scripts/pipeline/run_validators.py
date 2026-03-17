from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_BUCKET_DIR_PATTERN = re.compile(r"^bucket_(\d+)$")


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json object required: {path}")
    return payload


def missing_required_fields(payload: dict[str, Any], required_fields: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for field in required_fields:
        value = payload.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
            continue
        if isinstance(value, (list, dict)) and not value:
            missing.append(field)
    return missing


def _bucket_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    buckets: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = _BUCKET_DIR_PATTERN.match(child.name)
        if not match:
            continue
        buckets.append((int(match.group(1)), child))
    buckets.sort(key=lambda item: item[0])
    return buckets


def _require_file(path: Path, errors: list[str], label: str) -> None:
    if not path.exists() or not path.is_file():
        errors.append(f"missing file: {label} -> {path}")


def _require_dir(path: Path, errors: list[str], label: str, require_children: bool = False) -> None:
    if not path.exists() or not path.is_dir():
        errors.append(f"missing dir: {label} -> {path}")
        return
    if require_children:
        try:
            next(path.iterdir())
        except StopIteration:
            errors.append(f"empty dir: {label} -> {path}")


def _path_from_reference(base_dir: Path, raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def validate_stage09_candidate_run(run_dir: str | Path) -> list[str]:
    run_path = Path(run_dir)
    errors: list[str] = []
    if not run_path.exists() or not run_path.is_dir():
        return [f"run dir not found: {run_path}"]

    run_meta_path = run_path / "run_meta.json"
    _require_file(run_meta_path, errors, "stage09 run_meta")
    run_meta: dict[str, Any] = {}
    if run_meta_path.exists():
        try:
            run_meta = load_json_object(run_meta_path)
        except Exception as exc:
            errors.append(f"invalid json: {run_meta_path} -> {exc}")
    if run_meta:
        for field in missing_required_fields(run_meta, ("run_id", "run_tag", "recall_profile", "output_dir")):
            errors.append(f"missing stage09 run_meta field: {field}")

    buckets = _bucket_dirs(run_path)
    if not buckets:
        errors.append(f"no bucket directories found under {run_path}")
        return errors

    for bucket_id, bucket_dir in buckets:
        bucket_meta_path = bucket_dir / "bucket_meta.json"
        _require_file(bucket_meta_path, errors, f"bucket_{bucket_id} meta")
        if bucket_meta_path.exists():
            try:
                bucket_meta = load_json_object(bucket_meta_path)
            except Exception as exc:
                errors.append(f"invalid json: {bucket_meta_path} -> {exc}")
            else:
                for field in missing_required_fields(bucket_meta, ("n_users", "n_test", "pretrim_top_k_used")):
                    errors.append(f"missing bucket_{bucket_id} meta field: {field}")

        for dir_name in ("truth.parquet", "train_history.parquet", "candidates_all.parquet"):
            _require_dir(bucket_dir / dir_name, errors, f"bucket_{bucket_id}/{dir_name}", require_children=True)

        pretrim_dirs = [
            child
            for child in bucket_dir.iterdir()
            if child.is_dir() and child.name.startswith("candidates_pretrim") and child.name.endswith(".parquet")
        ]
        if not pretrim_dirs:
            errors.append(f"missing pretrim parquet dir under {bucket_dir}")
        else:
            for pretrim_dir in pretrim_dirs:
                _require_dir(pretrim_dir, errors, f"bucket_{bucket_id}/{pretrim_dir.name}", require_children=True)

    return errors



def validate_stage10_rank_model_run(run_dir: str | Path) -> list[str]:
    run_path = Path(run_dir)
    errors: list[str] = []
    if not run_path.exists() or not run_path.is_dir():
        return [f"run dir not found: {run_path}"]

    rank_model_path = run_path / "rank_model.json"
    _require_file(rank_model_path, errors, "stage10 rank_model.json")
    _require_dir(run_path / "models", errors, "stage10 models", require_children=True)
    _require_dir(run_path / "splits", errors, "stage10 splits", require_children=True)

    payload: dict[str, Any] = {}
    if rank_model_path.exists():
        try:
            payload = load_json_object(rank_model_path)
        except Exception as exc:
            errors.append(f"invalid json: {rank_model_path} -> {exc}")

    if not payload:
        return errors

    for field in missing_required_fields(
        payload,
        ("run_id", "run_tag", "source_stage09_run", "feature_columns", "models_by_bucket"),
    ):
        errors.append(f"missing stage10 rank_model field: {field}")

    models_by_bucket = payload.get("models_by_bucket", {})
    if not isinstance(models_by_bucket, dict) or not models_by_bucket:
        errors.append(f"invalid or empty models_by_bucket in {rank_model_path}")
        return errors

    for bucket_key, bucket_payload in sorted(models_by_bucket.items(), key=lambda item: str(item[0])):
        bucket_label = f"bucket_{bucket_key}"
        if not isinstance(bucket_payload, dict):
            errors.append(f"invalid model payload for {bucket_label}")
            continue

        for field in missing_required_fields(
            bucket_payload,
            ("model_backend", "feature_columns", "model_file", "split_users"),
        ):
            errors.append(f"missing {bucket_label} rank_model field: {field}")

        model_file_path = _path_from_reference(run_path, bucket_payload.get("model_file"))
        if model_file_path is None:
            errors.append(f"missing {bucket_label} model_file path")
        else:
            _require_file(model_file_path, errors, f"{bucket_label} model_file")

        split_users = bucket_payload.get("split_users", {})
        if not isinstance(split_users, dict):
            errors.append(f"invalid split_users payload for {bucket_label}")
            continue

        test_users_path = _path_from_reference(run_path, split_users.get("test_users_file"))
        if test_users_path is None:
            errors.append(f"missing {bucket_label} split_users.test_users_file")
        else:
            _require_file(test_users_path, errors, f"{bucket_label} test_users_file")

    return errors


def validate_stage10_infer_eval_run(run_dir: str | Path) -> list[str]:
    run_path = Path(run_dir)
    errors: list[str] = []
    if not run_path.exists() or not run_path.is_dir():
        return [f"run dir not found: {run_path}"]

    run_meta_path = run_path / "run_meta.json"
    _require_file(run_meta_path, errors, "stage10 infer_eval run_meta")

    payload: dict[str, Any] = {}
    if run_meta_path.exists():
        try:
            payload = load_json_object(run_meta_path)
        except Exception as exc:
            errors.append(f"invalid json: {run_meta_path} -> {exc}")

    if not payload:
        return errors

    for field in missing_required_fields(
        payload,
        ("run_id_10", "run_tag", "source_stage09_run", "rank_model_json", "top_k", "rows"),
    ):
        errors.append(f"missing stage10 infer_eval field: {field}")

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        errors.append(f"missing or empty rows in {run_meta_path}")
    else:
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"invalid row payload at rows[{idx}] in {run_meta_path}")
                continue
            for field in missing_required_fields(
                row,
                (
                    "run_id_10",
                    "source_run_09",
                    "bucket_min_train_reviews",
                    "model",
                    "recall_at_k",
                    "ndcg_at_k",
                    "n_users",
                    "n_items",
                    "n_candidates",
                ),
            ):
                errors.append(f"missing stage10 infer_eval row[{idx}] field: {field}")

    diagnostics_path = _path_from_reference(run_path, payload.get("diagnostics_summary_json"))
    if diagnostics_path is not None:
        _require_file(diagnostics_path, errors, "stage10 diagnostics summary")

    return errors


def validate_stage11_dataset_run(run_dir: str | Path) -> list[str]:
    run_path = Path(run_dir)
    errors: list[str] = []
    if not run_path.exists() or not run_path.is_dir():
        return [f"run dir not found: {run_path}"]

    run_meta_path = run_path / "run_meta.json"
    _require_file(run_meta_path, errors, "stage11 dataset run_meta")
    run_meta: dict[str, Any] = {}
    if run_meta_path.exists():
        try:
            run_meta = load_json_object(run_meta_path)
        except Exception as exc:
            errors.append(f"invalid json: {run_meta_path} -> {exc}")
    if run_meta:
        for field in missing_required_fields(run_meta, ("run_id", "run_tag", "source_stage09_run", "summary")):
            errors.append(f"missing stage11 dataset run_meta field: {field}")

    summary_entries = run_meta.get("summary", []) if isinstance(run_meta.get("summary"), list) else []
    summary_by_bucket = {
        int(entry["bucket"]): entry
        for entry in summary_entries
        if isinstance(entry, dict) and str(entry.get("bucket", "")).isdigit()
    }

    buckets = _bucket_dirs(run_path)
    if not buckets:
        errors.append(f"no bucket directories found under {run_path}")
        return errors

    for bucket_id, bucket_dir in buckets:
        if bucket_id not in summary_by_bucket:
            errors.append(f"missing summary entry for bucket_{bucket_id}")
        else:
            for field in missing_required_fields(summary_by_bucket[bucket_id], ("candidate_file", "output_bucket_dir")):
                errors.append(f"missing stage11 summary field for bucket_{bucket_id}: {field}")

        for dir_name in ("all_parquet", "user_evidence_table", "item_evidence_table", "pair_evidence_audit"):
            _require_dir(bucket_dir / dir_name, errors, f"bucket_{bucket_id}/{dir_name}", require_children=True)

        if not any((bucket_dir / name).is_dir() for name in ("train_json", "rich_sft_train_json")):
            errors.append(f"missing train json export under {bucket_dir}")
        if not any((bucket_dir / name).is_dir() for name in ("eval_json", "rich_sft_eval_json")):
            errors.append(f"missing eval json export under {bucket_dir}")

    return errors
