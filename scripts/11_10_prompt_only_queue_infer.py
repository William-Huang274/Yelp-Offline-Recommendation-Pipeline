from __future__ import annotations

import argparse
import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script} [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]")
    print("This stage script starts prompt-only queue inference and may load tokenizer/model dependencies.")
    print("Set the required environment variables or pass explicit paths, then run without --help.")
    sys.exit(0)

import importlib.util
import json
import math
import os
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "11_4_prompt_only_user_state_audit.py"
DEFAULT_INPUT_ROOT_DIR = PROJECT_ROOT / "data" / "output" / "11_prompt_only_user_state_audit"
DEFAULT_OUTPUT_ROOT_DIR = PROJECT_ROOT / "data" / "output" / "11_prompt_only_queue_infer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=None, help="Explicit stage11 prompt-only input run dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit output run dir for queue inference.")
    return parser.parse_args()


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


def env_optional_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def resolve_latest_run(root: Path) -> Path:
    runs = [path for path in root.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run directory found under {root}")
    return runs[0]


def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage11_prompt_only_audit", AUDIT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {AUDIT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list payload in {path}")
    out: list[dict[str, Any]] = []
    for row in payload:
        if isinstance(row, dict):
            out.append(row)
    return out


def build_prompt_records(module: Any, rows: list[dict[str, Any]]) -> list[Any]:
    records: list[Any] = []
    for idx, row in enumerate(rows):
        prompt_text = str(row.get("prompt_text", "") or "")
        if not prompt_text:
            raise ValueError("prompt_text missing from prompt_only_inputs.json; rerun 11_4 with prompt text included")
        input_index = int(row.get("input_index", idx))
        selection_score = float(row.get("selection_score", 0.0) or 0.0)
        record = module.PromptRecord(
            input_index=input_index,
            user_id=str(row.get("user_id", "") or ""),
            density_band=str(row.get("density_band", "") or ""),
            selected_rank_in_band=int(row.get("selected_rank_in_band", 0) or 0),
            user_meta=row.get("user_meta") or {},
            user_evidence_stream=row.get("user_evidence_stream") or [],
            recent_event_sequence=row.get("recent_event_sequence") or [],
            anchor_positive_events=row.get("anchor_positive_events") or [],
            anchor_negative_events=row.get("anchor_negative_events") or [],
            anchor_conflict_events=row.get("anchor_conflict_events") or [],
            context_notes=row.get("context_notes") or {},
            prompt_text=prompt_text,
            selection_score=selection_score,
        )
        records.append(record)
    records.sort(key=lambda rec: rec.input_index)
    return records


def percentile_from_sorted(values: list[int], pct: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    rank = pct * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return int(values[lower])
    weight = rank - lower
    value = values[lower] * (1.0 - weight) + values[upper] * weight
    return int(round(value))


def summarize_lengths(values: list[int]) -> dict[str, int]:
    ordered = sorted(int(v) for v in values)
    if not ordered:
        return {
            "rows": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "sum": 0,
        }
    return {
        "rows": len(ordered),
        "min": int(ordered[0]),
        "p50": percentile_from_sorted(ordered, 0.50),
        "p90": percentile_from_sorted(ordered, 0.90),
        "p95": percentile_from_sorted(ordered, 0.95),
        "p99": percentile_from_sorted(ordered, 0.99),
        "max": int(ordered[-1]),
        "sum": int(sum(ordered)),
    }


def queue_label_for_tokens(prompt_tokens: int, regular_max: int, high_max: int) -> str:
    if prompt_tokens <= regular_max:
        return "regular"
    if prompt_tokens <= high_max:
        return "high_budget"
    return "overflow"


def select_rows_for_run(
    audit_df: pd.DataFrame,
    sample_max_rows: int,
    sample_strategy: str,
) -> pd.DataFrame:
    if sample_max_rows <= 0 or len(audit_df) <= sample_max_rows:
        out = audit_df.copy()
        out["sample_selected"] = True
        out["sample_rank"] = range(1, len(out) + 1)
        return out

    ordered = audit_df.copy()
    queue_priority = {"overflow": 0, "high_budget": 1, "regular": 2}
    if sample_strategy == "longest_first":
        ordered = ordered.sort_values(
            by=["prompt_tokens", "selection_score", "selected_rank_in_band"],
            ascending=[False, False, True],
        )
    else:
        ordered["queue_priority"] = ordered["queue_label"].map(queue_priority).fillna(9).astype(int)
        ordered = ordered.sort_values(
            by=["queue_priority", "prompt_tokens", "selection_score", "selected_rank_in_band"],
            ascending=[True, False, False, True],
        )
        ordered = ordered.drop(columns=["queue_priority"])
    selected = ordered.head(sample_max_rows).copy()
    selected["sample_selected"] = True
    selected["sample_rank"] = range(1, len(selected) + 1)
    return selected.sort_values(by=["input_index"]).reset_index(drop=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_token_audit_df(
    records: list[Any],
    prompt_token_lengths: dict[int, int],
    regular_max: int,
    high_max: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        prompt_tokens = int(prompt_token_lengths.get(rec.input_index, 0))
        rows.append(
            {
                "user_id": rec.user_id,
                "density_band": rec.density_band,
                "selected_rank_in_band": rec.selected_rank_in_band,
                "input_index": rec.input_index,
                "selection_score": rec.selection_score,
                "prompt_tokens": prompt_tokens,
                "prompt_chars": len(rec.prompt_text),
                "review_evidence_count": len(rec.user_evidence_stream),
                "recent_event_count": len(rec.recent_event_sequence),
                "positive_anchor_count": len(rec.anchor_positive_events),
                "negative_anchor_count": len(rec.anchor_negative_events),
                "queue_label": queue_label_for_tokens(prompt_tokens, regular_max, high_max),
            }
        )
    audit_df = pd.DataFrame(rows).sort_values(by=["input_index"]).reset_index(drop=True)
    return audit_df


def summarize_queue_counts(audit_df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for queue_name, queue_df in audit_df.groupby("queue_label", sort=False):
        length_summary = summarize_lengths(queue_df["prompt_tokens"].tolist())
        band_counts = {
            str(key): int(value)
            for key, value in queue_df["density_band"].value_counts(dropna=False).to_dict().items()
        }
        rows.append(
            {
                "queue_label": str(queue_name),
                "count": int(len(queue_df)),
                "token_summary": length_summary,
                "density_band_counts": band_counts,
            }
        )
    return {
        "rows": int(len(audit_df)),
        "queue_counts": {str(k): int(v) for k, v in audit_df["queue_label"].value_counts().to_dict().items()},
        "density_band_counts": {str(k): int(v) for k, v in audit_df["density_band"].value_counts().to_dict().items()},
        "token_summary": summarize_lengths(audit_df["prompt_tokens"].tolist()),
        "queues": rows,
    }


def choose_prompt_sample(audit_df: pd.DataFrame) -> dict[str, Any]:
    if audit_df.empty:
        return {}
    ordered = audit_df.sort_values(
        by=["queue_label", "prompt_tokens", "selection_score"],
        ascending=[True, False, False],
    )
    sample_row = ordered.iloc[0].to_dict()
    return {str(k): v for k, v in sample_row.items()}


def main() -> None:
    args = parse_args()

    input_run_dir = args.input_dir or env_optional_path("INPUT_11_PROMPT_ONLY_USER_STATE_AUDIT_RUN_DIR")
    if input_run_dir is None:
        input_root_dir = env_path("INPUT_11_PROMPT_ONLY_USER_STATE_AUDIT_ROOT_DIR", DEFAULT_INPUT_ROOT_DIR)
        input_run_dir = resolve_latest_run(input_root_dir)
    prompt_inputs_path = input_run_dir / "prompt_only_inputs.json"
    if not prompt_inputs_path.exists():
        raise FileNotFoundError(f"prompt_only_inputs.json not found under {input_run_dir}")

    output_root_dir = env_path("OUTPUT_11_PROMPT_ONLY_QUEUE_INFER_ROOT_DIR", DEFAULT_OUTPUT_ROOT_DIR)
    output_run_dir = args.output_dir or env_optional_path("OUTPUT_11_PROMPT_ONLY_QUEUE_INFER_RUN_DIR")
    if output_run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_stage11_prompt_only_queue_infer"
        output_run_dir = output_root_dir / run_id
    output_run_dir.mkdir(parents=True, exist_ok=True)

    regular_max = env_int("PROMPT_ONLY_QUEUE_REGULAR_MAX_TOKENS", 16000)
    high_max = env_int("PROMPT_ONLY_QUEUE_HIGH_MAX_TOKENS", 32000)
    sample_max_rows = env_int("PROMPT_ONLY_QUEUE_SAMPLE_MAX_ROWS", 0)
    sample_strategy = os.environ.get("PROMPT_ONLY_QUEUE_SAMPLE_STRATEGY", "longtail_priority").strip().lower()
    resume_from_existing = env_bool("PROMPT_ONLY_QUEUE_RESUME_FROM_EXISTING", True)
    run_inference = env_bool("PROMPT_ONLY_QUEUE_RUN_INFERENCE", True)
    checkpoint_every_batches = env_int("PROMPT_ONLY_CHECKPOINT_EVERY_BATCHES", 4)
    max_new_tokens = env_int("PROMPT_ONLY_GEN_MAX_NEW_TOKENS", 4096)
    prompt_sample_user_id = os.environ.get("PROMPT_ONLY_QUEUE_SAMPLE_USER_ID", "").strip()
    infer_backend = os.environ.get("PROMPT_ONLY_INFER_BACKEND", "hf").strip().lower() or "hf"
    guardrails_hard_fail = env_bool("PROMPT_ONLY_GUARDRAILS_HARD_FAIL", True)
    write_structured_output_schema = env_bool("PROMPT_ONLY_WRITE_STRUCTURED_OUTPUT_SCHEMA", True)
    vllm_runtime = {
        "gpu_memory_utilization": os.environ.get("PROMPT_ONLY_VLLM_GPU_MEMORY_UTILIZATION", "").strip(),
        "max_model_len": os.environ.get("PROMPT_ONLY_VLLM_MAX_MODEL_LEN", "").strip(),
        "max_num_seqs": os.environ.get("PROMPT_ONLY_VLLM_MAX_NUM_SEQS", "").strip(),
        "max_num_batched_tokens": os.environ.get("PROMPT_ONLY_VLLM_MAX_NUM_BATCHED_TOKENS", "").strip(),
        "tensor_parallel_size": os.environ.get("PROMPT_ONLY_VLLM_TENSOR_PARALLEL_SIZE", "").strip(),
        "enforce_eager": os.environ.get("PROMPT_ONLY_VLLM_ENFORCE_EAGER", "").strip(),
        "require_guided_decoding": os.environ.get("PROMPT_ONLY_VLLM_REQUIRE_GUIDED_DECODING", "").strip(),
    }

    queue_configs = {
        "regular": {
            "gen_batch_size": env_int("PROMPT_ONLY_QUEUE_REGULAR_GEN_BATCH_SIZE", 6),
            "max_batch_prompt_tokens": env_int("PROMPT_ONLY_QUEUE_REGULAR_MAX_BATCH_PROMPT_TOKENS", 90000),
            "max_new_tokens": env_int("PROMPT_ONLY_QUEUE_REGULAR_MAX_NEW_TOKENS", max_new_tokens),
        },
        "high_budget": {
            "gen_batch_size": env_int("PROMPT_ONLY_QUEUE_HIGH_GEN_BATCH_SIZE", 2),
            "max_batch_prompt_tokens": env_int("PROMPT_ONLY_QUEUE_HIGH_MAX_BATCH_PROMPT_TOKENS", 64000),
            "max_new_tokens": env_int("PROMPT_ONLY_QUEUE_HIGH_MAX_NEW_TOKENS", max(max_new_tokens, 6144)),
        },
        "overflow": {
            "gen_batch_size": env_int("PROMPT_ONLY_QUEUE_OVERFLOW_GEN_BATCH_SIZE", 1),
            "max_batch_prompt_tokens": env_int("PROMPT_ONLY_QUEUE_OVERFLOW_MAX_BATCH_PROMPT_TOKENS", 50000),
            "max_new_tokens": env_int("PROMPT_ONLY_QUEUE_OVERFLOW_MAX_NEW_TOKENS", max(max_new_tokens, 8192)),
        },
    }

    module = load_audit_module()
    module.CHECKPOINT_EVERY_BATCHES = checkpoint_every_batches
    module.SORT_BY_PROMPT_LENGTH = True

    prompt_rows = load_prompt_rows(prompt_inputs_path)
    all_records = build_prompt_records(module, prompt_rows)

    tokenizer = AutoTokenizer.from_pretrained(module.BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_token_lengths = module.estimate_prompt_token_lengths(all_records, tokenizer)
    audit_df = build_token_audit_df(all_records, prompt_token_lengths, regular_max, high_max)
    queue_summary = summarize_queue_counts(audit_df)
    audit_df.to_parquet(output_run_dir / "prompt_token_audit.parquet", index=False)
    write_json(output_run_dir / "prompt_token_audit_summary.json", queue_summary)

    selected_df = select_rows_for_run(audit_df, sample_max_rows, sample_strategy)
    selected_user_ids = set(selected_df["user_id"].astype(str).tolist())
    selected_records = [rec for rec in all_records if rec.user_id in selected_user_ids]
    selected_df = selected_df.copy()
    selected_df["sample_selected"] = True
    selected_df.to_parquet(output_run_dir / "selected_prompt_token_audit.parquet", index=False)
    selected_queue_names = [
        queue_name
        for queue_name in ["regular", "high_budget", "overflow"]
        if any(
            queue_label_for_tokens(prompt_token_lengths.get(rec.input_index, 0), regular_max, high_max) == queue_name
            for rec in selected_records
        )
    ]
    shared_backend_info: dict[str, Any] = {
        "enabled": False,
        "infer_backend": infer_backend,
    }

    queue_plan: dict[str, Any] = {
        "input_run_dir": str(input_run_dir),
        "prompt_inputs_path": str(prompt_inputs_path),
        "output_run_dir": str(output_run_dir),
        "base_model": module.BASE_MODEL,
        "sample_max_rows": sample_max_rows,
        "sample_strategy": sample_strategy,
        "selected_rows": int(len(selected_df)),
        "selected_queue_counts": {
            str(k): int(v) for k, v in selected_df["queue_label"].value_counts().to_dict().items()
        },
        "selected_density_band_counts": {
            str(k): int(v) for k, v in selected_df["density_band"].value_counts().to_dict().items()
        },
        "queue_configs": deepcopy(queue_configs),
        "checkpoint_every_batches": checkpoint_every_batches,
        "max_new_tokens": max_new_tokens,
        "infer_backend": infer_backend,
        "guardrails_hard_fail": guardrails_hard_fail,
        "write_structured_output_schema": write_structured_output_schema,
        "vllm_runtime": vllm_runtime,
        "shared_backend": shared_backend_info,
    }
    write_json(output_run_dir / "queue_plan.json", queue_plan)

    prompt_sample_row: dict[str, Any]
    if prompt_sample_user_id:
        sample_matches = selected_df[selected_df["user_id"].astype(str) == prompt_sample_user_id]
        prompt_sample_row = sample_matches.iloc[0].to_dict() if not sample_matches.empty else {}
    else:
        prompt_sample_row = choose_prompt_sample(selected_df)
    if prompt_sample_row:
        prompt_sample_user_id = str(prompt_sample_row.get("user_id", "") or "")
        sample_record = next((rec for rec in selected_records if rec.user_id == prompt_sample_user_id), None)
        if sample_record is not None:
            write_json(output_run_dir / "prompt_sample_meta.json", prompt_sample_row)
            (output_run_dir / "prompt_sample.txt").write_text(sample_record.prompt_text, encoding="utf-8")

    queue_results: dict[str, Any] = {}
    combined_raw_rows: list[dict[str, Any]] = []
    combined_parsed_rows: list[dict[str, Any]] = []
    combined_batch_metrics: list[dict[str, Any]] = []

    if run_inference:
        shared_backend_state = None
        try:
            if infer_backend == "vllm" and selected_queue_names:
                module.GEN_BATCH_SIZE = max(int(queue_configs[name]["gen_batch_size"]) for name in selected_queue_names)
                module.MAX_BATCH_PROMPT_TOKENS = max(
                    int(queue_configs[name]["max_batch_prompt_tokens"]) for name in selected_queue_names
                )
                shared_backend_state = module.create_generation_backend(
                    max(int(queue_configs[name]["max_new_tokens"]) for name in selected_queue_names)
                )
                shared_backend_info.update(
                    {
                        "enabled": True,
                        "guided_decoding_mode": shared_backend_state.guided_decoding_mode,
                        "tokenizer_load_seconds": shared_backend_state.tokenizer_load_seconds,
                        "model_load_seconds": shared_backend_state.model_load_seconds,
                        "engine_batch_size_hint": int(module.GEN_BATCH_SIZE),
                        "engine_max_batch_prompt_tokens_hint": int(module.MAX_BATCH_PROMPT_TOKENS),
                        "engine_max_new_tokens_hint": max(int(queue_configs[name]["max_new_tokens"]) for name in selected_queue_names),
                    }
                )
                queue_plan["shared_backend"] = shared_backend_info
                write_json(output_run_dir / "queue_plan.json", queue_plan)

            for queue_name in ["regular", "high_budget", "overflow"]:
                queue_records = [
                    rec
                    for rec in selected_records
                    if queue_label_for_tokens(prompt_token_lengths.get(rec.input_index, 0), regular_max, high_max)
                    == queue_name
                ]
                if not queue_records:
                    continue
                config = queue_configs[queue_name]
                module.GEN_BATCH_SIZE = int(config["gen_batch_size"])
                module.MAX_BATCH_PROMPT_TOKENS = int(config["max_batch_prompt_tokens"])
                artifact_prefix = f"prompt_only_generation_{queue_name}"
                existing_state = (
                    module.load_generation_result_from_dir(output_run_dir, artifact_prefix, queue_records)
                    if resume_from_existing
                    else None
                )
                print(
                    f"[QUEUE] queue={queue_name} rows={len(queue_records)} batch_size={module.GEN_BATCH_SIZE} "
                    f"max_batch_prompt_tokens={module.MAX_BATCH_PROMPT_TOKENS} "
                    f"max_new_tokens={int(config['max_new_tokens'])}",
                    flush=True,
                )
                result = module.run_generation(
                    queue_records,
                    output_run_dir,
                    artifact_prefix=artifact_prefix,
                    max_new_tokens=int(config["max_new_tokens"]),
                    existing_state=existing_state,
                    backend_state=shared_backend_state,
                )
                queue_summary_row = {
                    "queue_label": queue_name,
                    "rows": int(len(queue_records)),
                    "requested_gen_batch_size": int(module.GEN_BATCH_SIZE),
                    "requested_max_batch_prompt_tokens": int(module.MAX_BATCH_PROMPT_TOKENS),
                    "requested_max_new_tokens": int(config["max_new_tokens"]),
                    "token_summary": summarize_lengths(
                        [int(prompt_token_lengths.get(rec.input_index, 0)) for rec in queue_records]
                    ),
                    "result_summary": result.summary,
                }
                queue_results[queue_name] = queue_summary_row
                for row in result.raw_rows:
                    row_copy = dict(row)
                    row_copy["queue_label"] = queue_name
                    combined_raw_rows.append(row_copy)
                for row in result.parsed_rows:
                    row_copy = dict(row)
                    row_copy["queue_label"] = queue_name
                    combined_parsed_rows.append(row_copy)
                for metric in result.batch_metrics:
                    metric_copy = dict(metric)
                    metric_copy["queue_label"] = queue_name
                    combined_batch_metrics.append(metric_copy)
                write_json(output_run_dir / f"{artifact_prefix}_queue_summary.json", queue_summary_row)
        finally:
            if shared_backend_state is not None:
                module.close_generation_backend(shared_backend_state)

        combined_raw_rows.sort(key=lambda row: int(row.get("input_index", 0)))
        combined_parsed_rows.sort(key=lambda row: int(row.get("input_index", 0)))
        write_json(output_run_dir / "prompt_only_generation_raw.json", combined_raw_rows)
        write_json(output_run_dir / "prompt_only_generation_parsed.json", combined_parsed_rows)
        write_json(output_run_dir / "prompt_only_generation_batch_metrics.json", combined_batch_metrics)

    final_summary = {
        "input_run_dir": str(input_run_dir),
        "prompt_inputs_path": str(prompt_inputs_path),
        "output_run_dir": str(output_run_dir),
        "base_model": module.BASE_MODEL,
        "rows_total": int(len(all_records)),
        "rows_selected": int(len(selected_records)),
        "sample_max_rows": sample_max_rows,
        "sample_strategy": sample_strategy,
        "resume_from_existing": resume_from_existing,
        "run_inference": run_inference,
        "token_budget_thresholds": {
            "regular_max_tokens": regular_max,
            "high_budget_max_tokens": high_max,
        },
        "all_queue_summary": queue_summary,
        "selected_queue_summary": summarize_queue_counts(selected_df),
        "queue_results": queue_results,
        "queue_configs": queue_configs,
        "checkpoint_every_batches": checkpoint_every_batches,
        "max_new_tokens": max_new_tokens,
        "infer_backend": infer_backend,
        "guardrails_hard_fail": guardrails_hard_fail,
        "write_structured_output_schema": write_structured_output_schema,
        "vllm_runtime": vllm_runtime,
        "shared_backend": shared_backend_info,
        "completed_queue_labels": list(queue_results.keys()),
        "selected_user_ids": selected_df["user_id"].astype(str).tolist(),
    }
    write_json(output_run_dir / "prompt_only_generation_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
