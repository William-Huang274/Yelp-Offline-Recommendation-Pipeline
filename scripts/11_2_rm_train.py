from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from pipeline.project_paths import env_or_project_path, write_latest_run_pointer
from pipeline.stage11_pairwise import (
    DPO_PAIR_POLICY,
    build_rich_sft_dpo_pairs,
    pair_records_for_reward_training,
)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


RUN_TAG = os.getenv("RUN_TAG", "stage11_2_rm_train").strip() or "stage11_2_rm_train"
QLORA_RESUME_RUN_DIR = os.getenv("QLORA_RESUME_RUN_DIR", "").strip()
INPUT_11_RUN_DIR = os.getenv("INPUT_11_RUN_DIR", "").strip()
INPUT_11_ROOT = env_or_project_path("INPUT_11_ROOT_DIR", "data/output/11_qlora_data")
INPUT_11_SUFFIX = "_stage11_1_qlora_build_dataset"
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_MODELS_ROOT_DIR", "data/output/11_qlora_models")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
BASE_MODEL = os.getenv("QLORA_BASE_MODEL", "/root/hf_models/Qwen3.5-9B").strip()
REQUIRED_BASE_MODEL = os.getenv("QLORA_REQUIRED_BASE_MODEL", "/root/hf_models/Qwen3.5-9B").strip()
ENFORCE_REQUIRED_BASE_MODEL = os.getenv("QLORA_ENFORCE_REQUIRED_BASE_MODEL", "true").strip().lower() == "true"
TRUST_REMOTE_CODE = os.getenv("QLORA_TRUST_REMOTE_CODE", "true").strip().lower() == "true"
USE_4BIT = os.getenv("QLORA_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("QLORA_USE_BF16", "true").strip().lower() == "true"
LORA_R = int(os.getenv("QLORA_LORA_R", "16").strip() or 16)
LORA_ALPHA = int(os.getenv("QLORA_LORA_ALPHA", "32").strip() or 32)
LORA_DROPOUT = float(os.getenv("QLORA_LORA_DROPOUT", "0.05").strip() or 0.05)
LORA_TARGET_MODULES = os.getenv(
    "QLORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
).strip()

MAX_SEQ_LEN = int(os.getenv("QLORA_MAX_SEQ_LEN", "512").strip() or 512)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_PAD_TO_MULTIPLE_OF", "0").strip() or 0)
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
SAVE_EVAL_SCORE_DUMPS = os.getenv("QLORA_SAVE_EVAL_SCORE_DUMPS", "false").strip().lower() == "true"
EVAL_SCORE_DUMPS_DIRNAME = os.getenv("QLORA_EVAL_SCORE_DUMPS_DIRNAME", "eval_score_dumps").strip() or "eval_score_dumps"
ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "false").strip().lower() == "true"
QWEN35_SAFE_KBIT_PREP = os.getenv("QLORA_QWEN35_SAFE_KBIT_PREP", "true").strip().lower() == "true"
QWEN35_FORCE_FLOAT_PARAMS_BF16 = os.getenv("QLORA_QWEN35_FORCE_FLOAT_PARAMS_BF16", "true").strip().lower() == "true"
QWEN35_MAMBA_SSM_DTYPE = os.getenv("QLORA_QWEN35_MAMBA_SSM_DTYPE", "auto").strip()
ATTN_IMPLEMENTATION = os.getenv("QLORA_ATTN_IMPLEMENTATION", "").strip().lower()
PAIR_PROMPT_STYLE = os.getenv("QLORA_PAIR_PROMPT_STYLE", "local_listwise_compare").strip().lower() or "local_listwise_compare"
BOUNDARY_PROBE_SCORE_MODE = os.getenv("QLORA_BOUNDARY_PROBE_SCORE_MODE", "boundary_head_reason_weighted").strip().lower() or "boundary_head_reason_weighted"
BOUNDARY_PROBE_HEAD_WEIGHT = float(os.getenv("QLORA_BOUNDARY_PROBE_HEAD_WEIGHT", "0.3").strip() or 0.3)
BOUNDARY_PROBE_REASON_WEIGHT = float(os.getenv("QLORA_BOUNDARY_PROBE_REASON_WEIGHT", "0.15").strip() or 0.15)
BOUNDARY_PROBE_REASON_TARGETS = [
    str(x).strip()
    for x in (os.getenv(
        "QLORA_BOUNDARY_PROBE_REASON_TARGETS",
        "semantic_edge_xgb_ambiguous,semantic_edge_underweighted,semantic_edge_xgb_ambiguous_weak,semantic_edge_underweighted_weak,channel_context_underweighted,channel_context_underweighted_weak,multi_route_underweighted,multi_route_underweighted_weak,head_prior_blocked,head_prior_blocked_weak",
    ) or "").split(",")
    if str(x).strip()
]
BOUNDARY_TOO_EASY_WIN_RATE = float(os.getenv("QLORA_BOUNDARY_TOO_EASY_WIN_RATE", "0.99").strip() or 0.99)
BOUNDARY_TOO_EASY_MIN_COUNT = max(1, int(os.getenv("QLORA_BOUNDARY_TOO_EASY_MIN_COUNT", "50").strip() or 50))
BEST_MODEL_METRIC_NAME = os.getenv("QLORA_BEST_MODEL_METRIC", "eval_boundary_probe_score").strip() or "eval_boundary_probe_score"

EPOCHS = float(os.getenv("QLORA_EPOCHS", "1.0").strip() or 1.0)
LR = float(os.getenv("QLORA_LR", "5e-5").strip() or 5e-5)
WEIGHT_DECAY = float(os.getenv("QLORA_WEIGHT_DECAY", "0.01").strip() or 0.01)
WARMUP_RATIO = float(os.getenv("QLORA_WARMUP_RATIO", "0.05").strip() or 0.05)
BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "1").strip() or 1)
EVAL_BATCH_SIZE = int(os.getenv("QLORA_EVAL_BATCH_SIZE", str(max(1, BATCH_SIZE))).strip() or max(1, BATCH_SIZE))
GRAD_ACC = int(os.getenv("QLORA_GRAD_ACC", "8").strip() or 8)
EVAL_STEPS = int(os.getenv("QLORA_EVAL_STEPS", "1000").strip() or 1000)
SAVE_STEPS = int(os.getenv("QLORA_SAVE_STEPS", "1000").strip() or 1000)
SAVE_TOTAL_LIMIT = int(os.getenv("QLORA_SAVE_TOTAL_LIMIT", "2").strip() or 2)
LOGGING_STEPS = int(os.getenv("QLORA_LOGGING_STEPS", "10").strip() or 10)
GRADIENT_CHECKPOINTING = os.getenv("QLORA_GRADIENT_CHECKPOINTING", "true").strip().lower() == "true"

DPO_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_MAX_PAIRS", "8").strip() or 8)
DPO_TRUE_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_TRUE_MAX_PAIRS", "2").strip() or 2)
DPO_VALID_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_VALID_MAX_PAIRS", "3").strip() or 3)
DPO_HIST_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_HIST_MAX_PAIRS", "1").strip() or 1)
DPO_ALLOW_MID_NEG = os.getenv("QLORA_DPO_ALLOW_MID_NEG", "true").strip().lower() == "true"

RM_TRUE_PAIR_WEIGHT = float(os.getenv("QLORA_RM_TRUE_PAIR_WEIGHT", "1.0").strip() or 1.0)
RM_VALID_PAIR_WEIGHT = float(os.getenv("QLORA_RM_VALID_PAIR_WEIGHT", "0.35").strip() or 0.35)
RM_HIST_PAIR_WEIGHT = float(os.getenv("QLORA_RM_HIST_PAIR_WEIGHT", "0.0").strip() or 0.0)
RM_BOUNDARY_PAIR_WEIGHT = float(os.getenv("QLORA_RM_BOUNDARY_PAIR_WEIGHT", "1.35").strip() or 1.35)
RM_STRUCTURED_PAIR_WEIGHT = float(os.getenv("QLORA_RM_STRUCTURED_PAIR_WEIGHT", "1.2").strip() or 1.2)
RM_SLOT_DECAY = float(os.getenv("QLORA_RM_SLOT_DECAY", "0.85").strip() or 0.85)
RM_SAMPLE_WEIGHT_MODE = os.getenv("QLORA_RM_SAMPLE_WEIGHT_MODE", "capped_max").strip().lower() or "capped_max"
RM_SAMPLE_WEIGHT_CAP = float(os.getenv("QLORA_RM_SAMPLE_WEIGHT_CAP", "1.25").strip() or 1.25)
RM_OBJECTIVE = os.getenv("QLORA_RM_OBJECTIVE", "listwise").strip().lower() or "listwise"
PAIRWISE_SOURCE_MODE = os.getenv("QLORA_PAIRWISE_SOURCE_MODE", "pool").strip().lower() or "pool"
TARGET_TRUE_BANDS_RAW = os.getenv("QLORA_TARGET_TRUE_BANDS", "").strip()
RM_LISTWISE_MAX_NEGATIVES = max(1, int(os.getenv("QLORA_RM_LISTWISE_MAX_NEGATIVES", "4").strip() or 4))
_RM_LISTWISE_MIN_NEGATIVES_RAW = os.getenv("QLORA_RM_LISTWISE_MIN_NEGATIVES", "2").strip() or "2"
RM_LISTWISE_TRAIN_MIN_NEGATIVES = max(
    1,
    int((os.getenv("QLORA_RM_LISTWISE_TRAIN_MIN_NEGATIVES", _RM_LISTWISE_MIN_NEGATIVES_RAW).strip() or _RM_LISTWISE_MIN_NEGATIVES_RAW)),
)
RM_LISTWISE_EVAL_MIN_NEGATIVES = max(
    1,
    int((os.getenv("QLORA_RM_LISTWISE_EVAL_MIN_NEGATIVES", _RM_LISTWISE_MIN_NEGATIVES_RAW).strip() or _RM_LISTWISE_MIN_NEGATIVES_RAW)),
)
RM_LISTWISE_TEMPERATURE = float(os.getenv("QLORA_RM_LISTWISE_TEMPERATURE", "1.0").strip() or 1.0)
RM_SCORE_CHUNK_SIZE = max(0, int(os.getenv("QLORA_RM_SCORE_CHUNK_SIZE", "0").strip() or 0))
RM_REQUIRE_SCORING_PROMPT = os.getenv("QLORA_RM_REQUIRE_SCORING_PROMPT", "true").strip().lower() == "true"
RM_EXPECTED_PROMPT_MODE = os.getenv(
    "QLORA_RM_EXPECTED_PROMPT_MODE",
    "semantic_compact_boundary_rm",
).strip().lower() or "semantic_compact_boundary_rm"
RM_EVAL_MAX_PAIRS_PER_USER = max(
    2,
    int((os.getenv("QLORA_RM_EVAL_MAX_PAIRS", str(max(2, DPO_MAX_PAIRS_PER_USER // 2))).strip() or max(2, DPO_MAX_PAIRS_PER_USER // 2))),
)
RM_EVAL_TRUE_MAX_PAIRS_PER_USER = max(
    0,
    int((os.getenv("QLORA_RM_EVAL_TRUE_MAX_PAIRS", str(DPO_TRUE_MAX_PAIRS_PER_USER)).strip() or DPO_TRUE_MAX_PAIRS_PER_USER)),
)
RM_EVAL_VALID_MAX_PAIRS_PER_USER = max(
    0,
    int((os.getenv("QLORA_RM_EVAL_VALID_MAX_PAIRS", str(DPO_VALID_MAX_PAIRS_PER_USER)).strip() or DPO_VALID_MAX_PAIRS_PER_USER)),
)
RM_EVAL_HIST_MAX_PAIRS_PER_USER = max(
    0,
    int((os.getenv("QLORA_RM_EVAL_HIST_MAX_PAIRS", str(DPO_HIST_MAX_PAIRS_PER_USER)).strip() or DPO_HIST_MAX_PAIRS_PER_USER)),
)
if RM_OBJECTIVE not in {"pairwise", "listwise"}:
    raise RuntimeError(f"unsupported QLORA_RM_OBJECTIVE={RM_OBJECTIVE}; expected pairwise|listwise")
if RM_SAMPLE_WEIGHT_MODE not in {"max", "top2_mean", "mean", "capped_max"}:
    raise RuntimeError(
        f"unsupported QLORA_RM_SAMPLE_WEIGHT_MODE={RM_SAMPLE_WEIGHT_MODE}; "
        "expected max|top2_mean|mean|capped_max"
    )
if BOUNDARY_PROBE_SCORE_MODE not in {"boundary_only", "boundary_head_weighted", "boundary_head_reason_weighted"}:
    raise RuntimeError(
        f"unsupported QLORA_BOUNDARY_PROBE_SCORE_MODE={BOUNDARY_PROBE_SCORE_MODE}; "
        "expected boundary_only|boundary_head_weighted|boundary_head_reason_weighted"
    )


def _metric_token(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


def parse_bucket_override(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return sorted(list(set(out)))


def parse_target_true_bands(raw: str) -> set[str]:
    out: set[str] = set()
    for part in str(raw or "").split(","):
        token = str(part or "").strip().lower()
        if token:
            out.add(token)
    return out


TARGET_TRUE_BANDS = parse_target_true_bands(TARGET_TRUE_BANDS_RAW)


def _boundary_rank_band(rank: int) -> str:
    r = int(rank or 0)
    if r <= 10:
        return "head_guard"
    if r <= 30:
        return "boundary_11_30"
    if r <= 60:
        return "rescue_31_60"
    if r <= 100:
        return "rescue_61_100"
    return "outside_100"


def _row_learned_rank_band(row: dict[str, Any]) -> str:
    learned_rank = int(
        row.get("chosen_learned_rank", row.get("learned_rank", row.get("pre_rank", 999999))) or 999999
    )
    if learned_rank < 999999:
        return _boundary_rank_band(learned_rank)
    direct = str(row.get("chosen_learned_rank_band", "") or "").strip().lower()
    if direct:
        return direct
    direct = str(row.get("learned_rank_band", "") or "").strip().lower()
    if direct:
        return direct
    return _boundary_rank_band(int(row.get("pre_rank", 999999) or 999999))


def filter_rows_for_target_true_bands(
    rows: list[dict[str, Any]],
    target_true_bands: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not target_true_bands:
        return list(rows), {
            "enabled": False,
            "target_true_bands": [],
            "input_rows": int(len(rows)),
            "output_rows": int(len(rows)),
        }
    eligible_users: set[int] = set()
    positive_band_counts_before: dict[str, int] = defaultdict(int)
    input_positive_rows = 0
    input_negative_rows = 0
    for row in rows:
        uid = int(row.get("user_idx", -1) or -1)
        if uid < 0:
            continue
        label = int(row.get("label", 0) or 0)
        if label == 1:
            input_positive_rows += 1
            band = _row_learned_rank_band(row)
            positive_band_counts_before[str(band)] += 1
            if band in target_true_bands:
                eligible_users.add(uid)
        else:
            input_negative_rows += 1
    filtered_rows: list[dict[str, Any]] = []
    dropped_positive_rows = 0
    dropped_negative_rows = 0
    output_positive_rows = 0
    output_negative_rows = 0
    output_users: set[int] = set()
    positive_band_counts_after: dict[str, int] = defaultdict(int)
    for row in rows:
        uid = int(row.get("user_idx", -1) or -1)
        if uid < 0 or uid not in eligible_users:
            if int(row.get("label", 0) or 0) == 1:
                dropped_positive_rows += 1
            else:
                dropped_negative_rows += 1
            continue
        label = int(row.get("label", 0) or 0)
        if label == 1:
            band = _row_learned_rank_band(row)
            if band not in target_true_bands:
                dropped_positive_rows += 1
                continue
            output_positive_rows += 1
            positive_band_counts_after[str(band)] += 1
        else:
            output_negative_rows += 1
        output_users.add(uid)
        filtered_rows.append(row)
    return filtered_rows, {
        "enabled": True,
        "target_true_bands": sorted(list(target_true_bands)),
        "input_rows": int(len(rows)),
        "output_rows": int(len(filtered_rows)),
        "input_positive_rows": int(input_positive_rows),
        "input_negative_rows": int(input_negative_rows),
        "output_positive_rows": int(output_positive_rows),
        "output_negative_rows": int(output_negative_rows),
        "dropped_positive_rows": int(dropped_positive_rows),
        "dropped_negative_rows": int(dropped_negative_rows),
        "eligible_users": int(len(eligible_users)),
        "output_users": int(len(output_users)),
        "positive_band_counts_before": dict(sorted(positive_band_counts_before.items())),
        "positive_band_counts_after": dict(sorted(positive_band_counts_after.items())),
    }


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage11_dataset_run() -> Path:
    if INPUT_11_RUN_DIR:
        run_dir = Path(INPUT_11_RUN_DIR)
        if not run_dir.exists():
            raise FileNotFoundError(f"INPUT_11_RUN_DIR not found: {run_dir}")
        return run_dir
    return pick_latest_run(INPUT_11_ROOT, INPUT_11_SUFFIX)


def is_qwen35_model_type(model_type: str) -> bool:
    return str(model_type or "").strip().lower().startswith("qwen3_5")


def prepare_model_for_kbit_training_qwen35_safe(model: Any, enable_input_grads: bool) -> Any:
    for _, param in model.named_parameters():
        param.requires_grad = False
    if enable_input_grads:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(module: Any, _inputs: Any, output: Any) -> None:
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    return model


def recast_float_params(model: Any, target_dtype: Any) -> int:
    n_cast = 0
    for param in model.parameters():
        if param.__class__.__name__ == "Params4bit":
            continue
        if getattr(param, "dtype", None) == torch.float32:
            param.data = param.data.to(target_dtype)
            n_cast += 1
    return int(n_cast)


def load_json_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        pdf = pd.read_json(path, lines=True)
        rows.extend(pdf.to_dict(orient="records"))
    return rows


def collect_stage11_rows(
    source_11: Path,
    buckets: list[int],
    source_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mode = str(source_mode or "rich_sft").strip().lower() or "rich_sft"
    if mode == "pool":
        train_dir_name = "pairwise_pool_train_json"
        eval_dir_name = "pairwise_pool_eval_json"
        parquet_dir_candidates = ["pairwise_pool_all_parquet"]
    else:
        train_dir_name = "rich_sft_train_json"
        eval_dir_name = "rich_sft_eval_json"
        parquet_dir_candidates = ["rich_sft_all_parquet", "rich_sft_parquet"]

    train_json: list[str] = []
    eval_json: list[str] = []
    parquet_dirs: list[Path] = []
    summary: dict[str, Any] = {"source_mode": mode, "buckets": []}
    for bucket in buckets:
        bdir = source_11 / f"bucket_{bucket}"
        train_dir = bdir / train_dir_name
        eval_dir = bdir / eval_dir_name
        parquet_dir: Path | None = None
        for parquet_name in parquet_dir_candidates:
            candidate = bdir / parquet_name
            if candidate.exists():
                parquet_dir = candidate
                break
        bucket_train = sorted([p.as_posix() for p in train_dir.glob("*.json")]) if train_dir.exists() else []
        bucket_eval = sorted([p.as_posix() for p in eval_dir.glob("*.json")]) if eval_dir.exists() else []
        train_json.extend(bucket_train)
        eval_json.extend(bucket_eval)
        if not bucket_train and parquet_dir is not None:
            parquet_dirs.append(parquet_dir)
        summary["buckets"].append(
            {
                "bucket": int(bucket),
                "train_json_files": len(bucket_train),
                "eval_json_files": len(bucket_eval),
                "train_dir_name": train_dir_name,
                "eval_dir_name": eval_dir_name,
                "parquet_dir": str(parquet_dir or ""),
            }
        )
    if train_json:
        return load_json_rows(train_json), load_json_rows(eval_json) if eval_json else [], summary
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for parquet_dir in parquet_dirs:
        pdf = pd.read_parquet(parquet_dir.as_posix())
        if "split" not in pdf.columns:
            raise RuntimeError(f"split column missing in parquet fallback dir: {parquet_dir}")
        train_rows.extend(pdf.loc[pdf["split"].astype(str).str.lower() == "train"].to_dict(orient="records"))
        eval_rows.extend(pdf.loc[pdf["split"].astype(str).str.lower() == "eval"].to_dict(orient="records"))
    if not train_rows:
        raise RuntimeError(f"no {mode} rows found under {source_11} for buckets={buckets}")
    summary["storage"] = "parquet"
    return train_rows, eval_rows, summary


def pair_source_weight(source_name: str) -> float:
    src = str(source_name or "").strip().lower()
    if src == "true":
        return float(RM_TRUE_PAIR_WEIGHT)
    if src == "valid":
        return float(RM_VALID_PAIR_WEIGHT)
    if src == "hist":
        return float(RM_HIST_PAIR_WEIGHT)
    return 1.0


def pair_bucket_weight(selection_bucket: str) -> float:
    bucket = str(selection_bucket or "").strip().lower()
    if "boundary_kickout" in bucket:
        return float(RM_BOUNDARY_PAIR_WEIGHT)
    if "structured_lift" in bucket:
        return float(RM_STRUCTURED_PAIR_WEIGHT)
    return 1.0


def pair_slot_weight(selection_slot: int) -> float:
    slot = max(0, int(selection_slot))
    return float(float(RM_SLOT_DECAY) ** slot)


def aggregate_sample_weight(neg_weights: list[float]) -> float:
    if not neg_weights:
        return 1.0
    weights = [max(0.0, float(w)) for w in neg_weights]
    max_w = max(weights)
    if RM_SAMPLE_WEIGHT_MODE == "max":
        return float(max_w)
    if RM_SAMPLE_WEIGHT_MODE == "capped_max":
        return float(min(max_w, max(0.0, float(RM_SAMPLE_WEIGHT_CAP))))
    if RM_SAMPLE_WEIGHT_MODE == "mean":
        return float(sum(weights) / max(len(weights), 1))
    if RM_SAMPLE_WEIGHT_MODE == "top2_mean":
        top2 = sorted(weights, reverse=True)[:2]
        return float(sum(top2) / max(len(top2), 1))
    return float(max_w)


def build_rm_examples(pair_records: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    weights: list[float] = []
    for rec in pair_records:
        source_name = str(rec.get("chosen_label_source", "") or "unknown")
        selection_bucket = str(rec.get("selection_bucket", "") or "")
        selection_slot = int(rec.get("selection_slot", 0) or 0)
        total_weight = float(
            pair_source_weight(source_name)
            * pair_bucket_weight(selection_bucket)
            * pair_slot_weight(selection_slot)
        )
        if total_weight <= 0.0:
            continue
        text_chosen = str(rec.get("text_chosen", "") or "").strip()
        text_rejected = str(rec.get("text_rejected", "") or "").strip()
        if not text_chosen or not text_rejected:
            continue
        rows.append(
            {
                "text_chosen": text_chosen,
                "text_rejected": text_rejected,
                "pair_weight": total_weight,
                "selection_slot": int(selection_slot),
                "selection_bucket": str(selection_bucket),
                "chosen_label_source": str(source_name),
                "user_idx": int(rec.get("user_idx", -1) or -1),
                "group_key": str(rec.get("group_key", "") or ""),
                "chosen_item_idx": int(rec.get("chosen_item_idx", -1) or -1),
                "rejected_item_idx": int(rec.get("rejected_item_idx", -1) or -1),
                "chosen_pre_rank": int(rec.get("chosen_pre_rank", -1) or -1),
                "rejected_pre_rank": int(rec.get("rejected_pre_rank", -1) or -1),
                "chosen_learned_rank": int(rec.get("chosen_learned_rank", -1) or -1),
                "rejected_learned_rank": int(rec.get("rejected_learned_rank", -1) or -1),
                "chosen_learned_rank_band": str(rec.get("chosen_learned_rank_band", "") or ""),
                "rejected_learned_rank_band": str(rec.get("rejected_learned_rank_band", "") or ""),
                "chosen_primary_reason": str(rec.get("chosen_primary_reason", "") or ""),
                "chosen_easy_but_useful": bool(rec.get("chosen_easy_but_useful", False)),
                "chosen_hard_but_learnable": bool(rec.get("chosen_hard_but_learnable", False)),
                "chosen_non_actionable": bool(rec.get("chosen_non_actionable", False)),
            }
        )
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        bucket_counts[selection_bucket] = bucket_counts.get(selection_bucket, 0) + 1
        weights.append(total_weight)
    if not rows:
        raise RuntimeError(f"no RM rows remained after weighting for split={split_name}")
    xs = sorted(weights)
    mid = xs[int(round((len(xs) - 1) * 0.5))]
    p95 = xs[int(round((len(xs) - 1) * 0.95))]
    return rows, {
        "split": split_name,
        "rows": int(len(rows)),
        "chosen_label_source_counts": dict(sorted(source_counts.items())),
        "selection_bucket_counts": dict(sorted(bucket_counts.items())),
        "pair_weight": {
            "min": float(xs[0]),
            "p50": float(mid),
            "p95": float(p95),
            "mean": float(sum(xs) / max(1, len(xs))),
            "max": float(xs[-1]),
        },
        "selection_slot": {
            "min": int(min(int(r.get("selection_slot", 0) or 0) for r in rows)),
            "max": int(max(int(r.get("selection_slot", 0) or 0) for r in rows)),
        },
    }


def build_listwise_rm_examples(
    pair_records: list[dict[str, Any]],
    split_name: str,
    min_negatives: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_negatives = max(1, int(min_negatives))
    grouped: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    pair_prompt_style_counts: dict[str, int] = {}
    group_prompt_style_counts: dict[str, int] = {}
    neg_counts: list[int] = []
    sample_weights: list[float] = []
    dropped_small_groups = 0
    dropped_empty_groups = 0
    rows: list[dict[str, Any]] = []

    for rec in pair_records:
        source_name = str(rec.get("chosen_label_source", "") or "unknown")
        selection_bucket = str(rec.get("selection_bucket", "") or "")
        selection_slot = int(rec.get("selection_slot", 0) or 0)
        total_weight = float(
            pair_source_weight(source_name)
            * pair_bucket_weight(selection_bucket)
            * pair_slot_weight(selection_slot)
        )
        if total_weight <= 0.0:
            continue
        text_chosen = str(rec.get("text_chosen", "") or "").strip()
        text_rejected = str(rec.get("text_rejected", "") or "").strip()
        if not text_chosen or not text_rejected:
            continue
        enriched = dict(rec)
        enriched["pair_weight"] = float(total_weight)
        grouped.setdefault(str(rec.get("group_key", "") or ""), []).append(enriched)

    for group_key, group_rows in grouped.items():
        if not group_key:
            continue
        group_rows.sort(
            key=lambda r: (
                int(r.get("selection_slot", 0) or 0),
                -float(r.get("pair_weight", 0.0) or 0.0),
                int(r.get("rejected_pre_rank", 999999) or 999999),
            )
        )
        chosen_text = str(group_rows[0].get("text_chosen", "") or "").strip()
        prompt_styles = {str(r.get("prompt_style", "") or "unknown") for r in group_rows}
        if prompt_styles == {"local_listwise_compare"}:
            group_prompt_style = "local_only"
        elif prompt_styles == {"blocker_compare"}:
            group_prompt_style = "blocker_only"
        elif "local_listwise_compare" in prompt_styles and "blocker_compare" in prompt_styles:
            group_prompt_style = "mixed_local_blocker"
        else:
            group_prompt_style = "mixed_other"
        chosen_label_source = str(group_rows[0].get("chosen_label_source", "") or "unknown")
        chosen_item_idx = int(group_rows[0].get("chosen_item_idx", -1) or -1)
        chosen_pre_rank = int(group_rows[0].get("chosen_pre_rank", -1) or -1)
        chosen_learned_rank = int(group_rows[0].get("chosen_learned_rank", -1) or -1)
        chosen_learned_rank_band = str(group_rows[0].get("chosen_learned_rank_band", "") or "")
        chosen_primary_reason = str(group_rows[0].get("chosen_primary_reason", "") or "")
        user_idx = int(group_rows[0].get("user_idx", -1) or -1)

        neg_texts: list[str] = []
        neg_weights: list[float] = []
        selection_buckets: list[str] = []
        selection_slots: list[int] = []
        neg_item_idxs: list[int] = []
        neg_pre_ranks: list[int] = []
        neg_learned_ranks: list[int] = []
        seen_neg_items: set[int] = set()
        for rec in group_rows:
            neg_text = str(rec.get("text_rejected", "") or "").strip()
            neg_item_idx = int(rec.get("rejected_item_idx", -1) or -1)
            neg_pre_rank = int(rec.get("rejected_pre_rank", -1) or -1)
            neg_learned_rank = int(rec.get("rejected_learned_rank", -1) or -1)
            if not neg_text:
                continue
            if neg_item_idx >= 0 and neg_item_idx in seen_neg_items:
                continue
            if neg_item_idx >= 0:
                seen_neg_items.add(neg_item_idx)
            neg_texts.append(neg_text)
            neg_weights.append(float(rec.get("pair_weight", 1.0) or 1.0))
            selection_buckets.append(str(rec.get("selection_bucket", "") or ""))
            selection_slots.append(int(rec.get("selection_slot", 0) or 0))
            neg_item_idxs.append(int(neg_item_idx))
            neg_pre_ranks.append(int(neg_pre_rank))
            neg_learned_ranks.append(int(neg_learned_rank))
            if len(neg_texts) >= int(RM_LISTWISE_MAX_NEGATIVES):
                break

        if not neg_texts:
            dropped_empty_groups += 1
            continue
        if len(neg_texts) < int(min_negatives):
            dropped_small_groups += 1
            continue

        sample_weight = aggregate_sample_weight(neg_weights)
        rel_neg_weights = [float(w) / max(sample_weight, 1e-6) for w in neg_weights]
        rows.append(
            {
                "text_chosen": chosen_text,
                "text_rejected_list": neg_texts,
                "neg_weights": rel_neg_weights,
                "sample_weight": float(sample_weight),
                "selection_buckets": selection_buckets,
                "selection_slots": selection_slots,
                "chosen_label_source": chosen_label_source,
                "group_key": group_key,
                "user_idx": int(user_idx),
                "chosen_item_idx": int(chosen_item_idx),
                "chosen_pre_rank": int(chosen_pre_rank),
                "chosen_learned_rank": int(chosen_learned_rank),
                "chosen_learned_rank_band": str(chosen_learned_rank_band),
                "chosen_primary_reason": str(chosen_primary_reason),
                "chosen_easy_but_useful": bool(group_rows[0].get("chosen_easy_but_useful", False)),
                "chosen_hard_but_learnable": bool(group_rows[0].get("chosen_hard_but_learnable", False)),
                "chosen_non_actionable": bool(group_rows[0].get("chosen_non_actionable", False)),
                "group_prompt_style": str(group_prompt_style),
                "neg_item_idxs": neg_item_idxs,
                "neg_pre_ranks": neg_pre_ranks,
                "neg_learned_ranks": neg_learned_ranks,
            }
        )
        source_counts[chosen_label_source] = source_counts.get(chosen_label_source, 0) + 1
        group_prompt_style_counts[group_prompt_style] = group_prompt_style_counts.get(group_prompt_style, 0) + 1
        for rec in group_rows:
            style_name = str(rec.get("prompt_style", "") or "unknown")
            pair_prompt_style_counts[style_name] = pair_prompt_style_counts.get(style_name, 0) + 1
        for bucket_name in selection_buckets:
            bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
        neg_counts.append(len(neg_texts))
        sample_weights.append(float(sample_weight))

    if not rows:
        raise RuntimeError(f"no listwise RM rows remained for split={split_name}")

    neg_counts.sort()
    sample_weights.sort()
    return rows, {
        "split": split_name,
        "rows": int(len(rows)),
        "chosen_label_source_counts": dict(sorted(source_counts.items())),
        "pair_prompt_style_counts": dict(sorted(pair_prompt_style_counts.items())),
        "group_prompt_style_counts": dict(sorted(group_prompt_style_counts.items())),
        "selection_bucket_counts": dict(sorted(bucket_counts.items())),
        "negatives_per_slate": {
            "min": int(neg_counts[0]),
            "p50": int(neg_counts[int(round((len(neg_counts) - 1) * 0.5))]),
            "p95": int(neg_counts[int(round((len(neg_counts) - 1) * 0.95))]),
            "max": int(neg_counts[-1]),
        },
        "sample_weight": {
            "min": float(sample_weights[0]),
            "p50": float(sample_weights[int(round((len(sample_weights) - 1) * 0.5))]),
            "p95": float(sample_weights[int(round((len(sample_weights) - 1) * 0.95))]),
            "mean": float(sum(sample_weights) / max(1, len(sample_weights))),
            "max": float(sample_weights[-1]),
        },
        "dropped_groups": {
            "empty": int(dropped_empty_groups),
            "below_min_negatives": int(dropped_small_groups),
        },
        "listwise_config": {
            "max_negatives": int(RM_LISTWISE_MAX_NEGATIVES),
            "min_negatives": int(min_negatives),
        },
    }


def build_tokenized_rm_records(rows: list[dict[str, Any]], tokenizer: Any, dataset_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    chosen_at_max = 0
    rejected_at_max = 0
    for row in rows:
        chosen = tokenizer(str(row["text_chosen"]), truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)
        rejected = tokenizer(str(row["text_rejected"]), truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)
        chosen_ids = [int(x) for x in chosen["input_ids"]]
        rejected_ids = [int(x) for x in rejected["input_ids"]]
        if not chosen_ids or not rejected_ids:
            continue
        records.append(
            {
                "chosen_input_ids": chosen_ids,
                "chosen_attention_mask": [int(x) for x in chosen.get("attention_mask", [1] * len(chosen_ids))],
                "rejected_input_ids": rejected_ids,
                "rejected_attention_mask": [int(x) for x in rejected.get("attention_mask", [1] * len(rejected_ids))],
                "pair_weight": float(row["pair_weight"]),
                "selection_bucket": str(row.get("selection_bucket", "") or ""),
                "selection_slot": int(row.get("selection_slot", 0) or 0),
                "chosen_label_source": str(row.get("chosen_label_source", "") or ""),
                "user_idx": int(row.get("user_idx", -1) or -1),
                "group_key": str(row.get("group_key", "") or ""),
                "chosen_item_idx": int(row.get("chosen_item_idx", -1) or -1),
                "rejected_item_idx": int(row.get("rejected_item_idx", -1) or -1),
                "chosen_pre_rank": int(row.get("chosen_pre_rank", -1) or -1),
                "rejected_pre_rank": int(row.get("rejected_pre_rank", -1) or -1),
                "chosen_learned_rank": int(row.get("chosen_learned_rank", -1) or -1),
                "rejected_learned_rank": int(row.get("rejected_learned_rank", -1) or -1),
                "chosen_learned_rank_band": str(row.get("chosen_learned_rank_band", "") or ""),
                "rejected_learned_rank_band": str(row.get("rejected_learned_rank_band", "") or ""),
                "chosen_primary_reason": str(row.get("chosen_primary_reason", "") or ""),
                "chosen_easy_but_useful": bool(row.get("chosen_easy_but_useful", False)),
                "chosen_hard_but_learnable": bool(row.get("chosen_hard_but_learnable", False)),
                "chosen_non_actionable": bool(row.get("chosen_non_actionable", False)),
            }
        )
        chosen_lens.append(len(chosen_ids))
        rejected_lens.append(len(rejected_ids))
        if len(chosen_ids) >= int(MAX_SEQ_LEN):
            chosen_at_max += 1
        if len(rejected_ids) >= int(MAX_SEQ_LEN):
            rejected_at_max += 1
    if not records:
        raise RuntimeError(f"no tokenized RM records built for dataset={dataset_name}")
    chosen_lens.sort()
    rejected_lens.sort()
    return records, {
        "dataset_name": dataset_name,
        "records": int(len(records)),
        "max_seq_len": int(MAX_SEQ_LEN),
        "chosen_token_len": {
            "min": int(chosen_lens[0]),
            "p50": int(chosen_lens[int(round((len(chosen_lens) - 1) * 0.5))]),
            "p95": int(chosen_lens[int(round((len(chosen_lens) - 1) * 0.95))]),
            "max": int(chosen_lens[-1]),
        },
        "chosen_at_max_seq_rate": float(chosen_at_max / max(1, len(records))),
        "rejected_token_len": {
            "min": int(rejected_lens[0]),
            "p50": int(rejected_lens[int(round((len(rejected_lens) - 1) * 0.5))]),
            "p95": int(rejected_lens[int(round((len(rejected_lens) - 1) * 0.95))]),
            "max": int(rejected_lens[-1]),
        },
        "rejected_at_max_seq_rate": float(rejected_at_max / max(1, len(records))),
    }


def build_tokenized_listwise_rm_records(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    dataset_name: str,
    min_negatives: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_negatives = max(1, int(min_negatives))
    records: list[dict[str, Any]] = []
    chosen_lens: list[int] = []
    neg_lens: list[int] = []
    neg_counts: list[int] = []
    chosen_at_max = 0
    negative_at_max = 0
    for row in rows:
        chosen = tokenizer(str(row["text_chosen"]), truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)
        chosen_ids = [int(x) for x in chosen["input_ids"]]
        if not chosen_ids:
            continue
        neg_input_ids: list[list[int]] = []
        neg_attention_masks: list[list[int]] = []
        neg_weights: list[float] = []
        text_rejected_list = row.get("text_rejected_list", [])
        raw_neg_weights = row.get("neg_weights", [])
        for neg_text, neg_weight in zip(text_rejected_list, raw_neg_weights):
            encoded = tokenizer(str(neg_text), truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)
            neg_ids = [int(x) for x in encoded["input_ids"]]
            if not neg_ids:
                continue
            neg_input_ids.append(neg_ids)
            neg_attention_masks.append([int(x) for x in encoded.get("attention_mask", [1] * len(neg_ids))])
            neg_weights.append(float(neg_weight))
            neg_lens.append(len(neg_ids))
            if len(neg_ids) >= int(MAX_SEQ_LEN):
                negative_at_max += 1
            if len(neg_input_ids) >= int(RM_LISTWISE_MAX_NEGATIVES):
                break
        if len(neg_input_ids) < int(min_negatives):
            continue
        records.append(
            {
                "chosen_input_ids": chosen_ids,
                "chosen_attention_mask": [int(x) for x in chosen.get("attention_mask", [1] * len(chosen_ids))],
                "neg_input_ids_list": neg_input_ids,
                "neg_attention_mask_list": neg_attention_masks,
                "neg_weights": neg_weights,
                "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
                "selection_buckets": [str(x) for x in row.get("selection_buckets", [])],
                "selection_slots": [int(x) for x in row.get("selection_slots", [])],
                "chosen_label_source": str(row.get("chosen_label_source", "") or ""),
                "group_key": str(row.get("group_key", "") or ""),
                "user_idx": int(row.get("user_idx", -1) or -1),
                "chosen_item_idx": int(row.get("chosen_item_idx", -1) or -1),
                "chosen_pre_rank": int(row.get("chosen_pre_rank", -1) or -1),
                "chosen_learned_rank": int(row.get("chosen_learned_rank", -1) or -1),
                "chosen_learned_rank_band": str(row.get("chosen_learned_rank_band", "") or ""),
                "chosen_primary_reason": str(row.get("chosen_primary_reason", "") or ""),
                "chosen_easy_but_useful": bool(row.get("chosen_easy_but_useful", False)),
                "chosen_hard_but_learnable": bool(row.get("chosen_hard_but_learnable", False)),
                "chosen_non_actionable": bool(row.get("chosen_non_actionable", False)),
                "neg_item_idxs": [int(x) for x in row.get("neg_item_idxs", [])],
                "neg_pre_ranks": [int(x) for x in row.get("neg_pre_ranks", [])],
                "neg_learned_ranks": [int(x) for x in row.get("neg_learned_ranks", [])],
            }
        )
        chosen_lens.append(len(chosen_ids))
        neg_counts.append(len(neg_input_ids))
        if len(chosen_ids) >= int(MAX_SEQ_LEN):
            chosen_at_max += 1
    if not records:
        raise RuntimeError(f"no tokenized listwise RM records built for dataset={dataset_name}")
    chosen_lens.sort()
    neg_lens.sort()
    neg_counts.sort()
    return records, {
        "dataset_name": dataset_name,
        "records": int(len(records)),
        "max_seq_len": int(MAX_SEQ_LEN),
        "chosen_token_len": {
            "min": int(chosen_lens[0]),
            "p50": int(chosen_lens[int(round((len(chosen_lens) - 1) * 0.5))]),
            "p95": int(chosen_lens[int(round((len(chosen_lens) - 1) * 0.95))]),
            "max": int(chosen_lens[-1]),
        },
        "chosen_at_max_seq_rate": float(chosen_at_max / max(1, len(records))),
        "negative_token_len": {
            "min": int(neg_lens[0]),
            "p50": int(neg_lens[int(round((len(neg_lens) - 1) * 0.5))]),
            "p95": int(neg_lens[int(round((len(neg_lens) - 1) * 0.95))]),
            "max": int(neg_lens[-1]),
        },
        "negative_at_max_seq_rate": float(negative_at_max / max(1, len(neg_lens))),
        "negatives_per_slate": {
            "min": int(neg_counts[0]),
            "p50": int(neg_counts[int(round((len(neg_counts) - 1) * 0.5))]),
            "p95": int(neg_counts[int(round((len(neg_counts) - 1) * 0.95))]),
            "max": int(neg_counts[-1]),
        },
        "listwise_config": {
            "max_negatives": int(RM_LISTWISE_MAX_NEGATIVES),
            "min_negatives": int(min_negatives),
        },
    }


def build_rm_collator(tokenizer: Any) -> Any:
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    def _pad(features: list[dict[str, Any]], ids_key: str, mask_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(f[ids_key]) for f in features)
        if PAD_TO_MULTIPLE_OF > 1:
            max_len = ((int(max_len) + int(PAD_TO_MULTIPLE_OF) - 1) // int(PAD_TO_MULTIPLE_OF)) * int(PAD_TO_MULTIPLE_OF)
        max_len = min(int(MAX_SEQ_LEN), int(max_len))
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for f in features:
            ids = [int(x) for x in f[ids_key]][:max_len]
            mask = [int(x) for x in f[mask_key]][:max_len]
            pad_len = int(max_len - len(ids))
            if pad_len > 0:
                ids = ids + ([pad_token_id] * pad_len)
                mask = mask + ([0] * pad_len)
            input_ids.append(ids)
            attention_mask.append(mask)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_ids, chosen_mask = _pad(features, "chosen_input_ids", "chosen_attention_mask")
        rejected_ids, rejected_mask = _pad(features, "rejected_input_ids", "rejected_attention_mask")
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "pair_weight": torch.tensor([float(f.get("pair_weight", 1.0)) for f in features], dtype=torch.float32),
            "labels": torch.ones((len(features),), dtype=torch.float32),
        }

    return _collate


def build_listwise_rm_collator(tokenizer: Any) -> Any:
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    def _pad_seq(seq: list[int], max_len: int, pad_value: int) -> list[int]:
        clipped = [int(x) for x in seq][:max_len]
        pad_len = int(max_len - len(clipped))
        if pad_len > 0:
            clipped = clipped + ([pad_value] * pad_len)
        return clipped

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = 1
        for feat in features:
            max_len = max(max_len, len(feat["chosen_input_ids"]))
            for neg_ids in feat["neg_input_ids_list"]:
                max_len = max(max_len, len(neg_ids))
        if PAD_TO_MULTIPLE_OF > 1:
            max_len = ((int(max_len) + int(PAD_TO_MULTIPLE_OF) - 1) // int(PAD_TO_MULTIPLE_OF)) * int(PAD_TO_MULTIPLE_OF)
        max_len = min(int(MAX_SEQ_LEN), int(max_len))
        max_neg = int(RM_LISTWISE_MAX_NEGATIVES)

        chosen_ids_batch: list[list[int]] = []
        chosen_mask_batch: list[list[int]] = []
        neg_ids_batch: list[list[list[int]]] = []
        neg_mask_batch: list[list[list[int]]] = []
        neg_weight_batch: list[list[float]] = []
        neg_presence_batch: list[list[float]] = []
        sample_weight_batch: list[float] = []

        for feat in features:
            chosen_ids_batch.append(_pad_seq(feat["chosen_input_ids"], max_len, pad_token_id))
            chosen_mask_batch.append(_pad_seq(feat["chosen_attention_mask"], max_len, 0))
            neg_ids_rows: list[list[int]] = []
            neg_mask_rows: list[list[int]] = []
            neg_weights: list[float] = []
            neg_presence: list[float] = []
            for neg_ids, neg_mask, neg_weight in zip(
                feat["neg_input_ids_list"],
                feat["neg_attention_mask_list"],
                feat["neg_weights"],
            ):
                if len(neg_ids_rows) >= max_neg:
                    break
                neg_ids_rows.append(_pad_seq(neg_ids, max_len, pad_token_id))
                neg_mask_rows.append(_pad_seq(neg_mask, max_len, 0))
                neg_weights.append(float(neg_weight))
                neg_presence.append(1.0)
            while len(neg_ids_rows) < max_neg:
                neg_ids_rows.append([pad_token_id] * max_len)
                neg_mask_rows.append([0] * max_len)
                neg_weights.append(0.0)
                neg_presence.append(0.0)
            neg_ids_batch.append(neg_ids_rows)
            neg_mask_batch.append(neg_mask_rows)
            neg_weight_batch.append(neg_weights)
            neg_presence_batch.append(neg_presence)
            sample_weight_batch.append(float(feat.get("sample_weight", 1.0)))

        return {
            "chosen_input_ids": torch.tensor(chosen_ids_batch, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_mask_batch, dtype=torch.long),
            "neg_input_ids": torch.tensor(neg_ids_batch, dtype=torch.long),
            "neg_attention_mask": torch.tensor(neg_mask_batch, dtype=torch.long),
            "neg_weight": torch.tensor(neg_weight_batch, dtype=torch.float32),
            "neg_presence_mask": torch.tensor(neg_presence_batch, dtype=torch.float32),
            "sample_weight": torch.tensor(sample_weight_batch, dtype=torch.float32),
            "labels": torch.zeros((len(features),), dtype=torch.float32),
        }

    return _collate


def reward_logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.reshape(-1)
    if logits.ndim != 2:
        raise RuntimeError(f"unexpected reward logits shape: {tuple(logits.shape)}")
    if logits.shape[-1] == 1:
        return logits[:, 0]
    # Generic sequence-classification heads may expose a 2-logit classifier even
    # when we conceptually want a scalar reward. Use the positive-vs-baseline
    # margin as a stable single score so train/eval stay aligned.
    return (logits[:, -1] - logits[:, 0]).reshape(-1)


def score_in_chunks(
    scorer: Any,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if int(chunk_size) <= 0 or int(input_ids.shape[0]) <= int(chunk_size):
        return scorer(model, input_ids, attention_mask)
    scores: list[torch.Tensor] = []
    total = int(input_ids.shape[0])
    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        scores.append(scorer(model, input_ids[start:end], attention_mask[start:end]))
    return torch.cat(scores, dim=0)


def compute_rm_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    arr = np.asarray(preds)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return {}
    margins = arr[:, 0].astype(np.float32) - arr[:, 1].astype(np.float32)
    return {
        "pair_accuracy": float(np.mean(margins > 0.0)),
        "pair_margin_mean": float(np.mean(margins)),
        "pair_margin_p50": float(np.median(margins)),
    }


def compute_listwise_rm_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    arr = np.asarray(preds)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return {}
    chosen = arr[:, 0].astype(np.float32)
    max_negative = arr[:, 1].astype(np.float32)
    top1_prob = arr[:, 2].astype(np.float32)
    negative_count = arr[:, 3].astype(np.float32)
    margins = chosen - max_negative
    return {
        "listwise_win_rate": float(np.mean(margins > 0.0)),
        "listwise_margin_mean": float(np.mean(margins)),
        "listwise_margin_p50": float(np.median(margins)),
        "top1_prob_mean": float(np.mean(top1_prob)),
        "negative_count_mean": float(np.mean(negative_count)),
    }


def learned_rank_band(rank: int) -> str:
    r = int(rank)
    if 0 < r <= 10:
        return "head_guard"
    if r <= 30:
        return "boundary_11_30"
    if r <= 60:
        return "rescue_31_60"
    if r <= 100:
        return "rescue_61_100"
    return "out_of_scope"


def learned_rank_probe_segment(rank: int) -> str | None:
    r = int(rank)
    if 31 <= r <= 40:
        return "rescue_31_40"
    if 41 <= r <= 60:
        return "rescue_41_60"
    return None


def _append_boundary_probe_metrics(out: dict[str, float], prefix: str, margins: list[float]) -> None:
    if not margins:
        return
    xs = np.asarray(margins, dtype=np.float32)
    out[f"{prefix}_count"] = float(xs.shape[0])
    out[f"{prefix}_win_rate"] = float(np.mean(xs > 0.0))
    out[f"{prefix}_margin_mean"] = float(np.mean(xs))
    out[f"{prefix}_margin_p50"] = float(np.median(xs))
    if xs.shape[0] >= int(BOUNDARY_TOO_EASY_MIN_COUNT):
        out[f"{prefix}_too_easy_flag"] = float(np.mean(xs > 0.0) >= float(BOUNDARY_TOO_EASY_WIN_RATE))


def compute_boundary_probe_metrics(dataset: Any, predictions: Any, objective: str) -> dict[str, float]:
    arr = np.asarray(predictions[0] if isinstance(predictions, tuple) else predictions)
    if arr.ndim != 2 or arr.shape[0] <= 0 or len(dataset) != int(arr.shape[0]):
        return {}
    band_margins: dict[str, list[float]] = defaultdict(list)
    band_source_margins: dict[tuple[str, str], list[float]] = defaultdict(list)
    segment_margins: dict[str, list[float]] = defaultdict(list)
    segment_source_margins: dict[tuple[str, str], list[float]] = defaultdict(list)
    reason_margins: dict[str, list[float]] = defaultdict(list)
    for idx in range(int(arr.shape[0])):
        row = dataset[idx]
        chosen_rank = int(row.get("chosen_learned_rank", row.get("chosen_pre_rank", -1)) or -1)
        band = learned_rank_band(chosen_rank)
        if band == "out_of_scope":
            continue
        source_name = str(row.get("chosen_label_source", "") or "unknown")
        primary_reason = str(row.get("chosen_primary_reason", "") or "").strip()
        if objective == "pairwise":
            if arr.shape[1] < 2:
                continue
            margin = float(arr[idx, 0] - arr[idx, 1])
        else:
            if arr.shape[1] < 2:
                continue
            margin = float(arr[idx, 0] - arr[idx, 1])
        band_margins[band].append(margin)
        band_source_margins[(band, source_name)].append(margin)
        segment_name = learned_rank_probe_segment(chosen_rank)
        if segment_name:
            segment_margins[segment_name].append(margin)
            segment_source_margins[(segment_name, source_name)].append(margin)
        if primary_reason:
            reason_margins[primary_reason].append(margin)

    metrics: dict[str, float] = {}
    for band_name, margins in band_margins.items():
        _append_boundary_probe_metrics(metrics, band_name, margins)
    for (band_name, source_name), margins in band_source_margins.items():
        _append_boundary_probe_metrics(metrics, f"{band_name}_{source_name}", margins)
    for segment_name, margins in segment_margins.items():
        _append_boundary_probe_metrics(metrics, segment_name, margins)
    for (segment_name, source_name), margins in segment_source_margins.items():
        _append_boundary_probe_metrics(metrics, f"{segment_name}_{source_name}", margins)
    for reason_name, margins in reason_margins.items():
        _append_boundary_probe_metrics(metrics, f"reason_{_metric_token(reason_name)}", margins)
    boundary_score = float(metrics.get("boundary_11_30_win_rate", 0.0))
    head_score = float(metrics.get("head_guard_win_rate", 0.0))
    mode = str(BOUNDARY_PROBE_SCORE_MODE or "boundary_only")
    if mode in {"boundary_head_weighted", "boundary_head_reason_weighted"}:
        head_w = min(max(float(BOUNDARY_PROBE_HEAD_WEIGHT), 0.0), 1.0)
        if mode == "boundary_head_reason_weighted":
            reason_w = min(max(float(BOUNDARY_PROBE_REASON_WEIGHT), 0.0), 1.0)
            target_scores = [
                float(metrics.get(f"reason_{_metric_token(reason_name)}_win_rate", 0.0))
                for reason_name in BOUNDARY_PROBE_REASON_TARGETS
            ]
            reason_score = float(sum(target_scores) / len(target_scores)) if target_scores else 0.0
            boundary_w = max(0.0, 1.0 - head_w - reason_w)
            total_w = boundary_w + head_w + reason_w
            if total_w <= 0.0:
                metrics["boundary_probe_score"] = boundary_score
            else:
                metrics["boundary_probe_score"] = float(
                    (boundary_w / total_w) * boundary_score
                    + (head_w / total_w) * head_score
                    + (reason_w / total_w) * reason_score
                )
            metrics["boundary_probe_reason_score"] = float(reason_score)
            metrics["boundary_probe_reason_weight"] = float(reason_w)
        else:
            metrics["boundary_probe_score"] = float((1.0 - head_w) * boundary_score + head_w * head_score)
    else:
        metrics["boundary_probe_score"] = boundary_score
    metrics["boundary_probe_head_weight"] = float(BOUNDARY_PROBE_HEAD_WEIGHT)
    return metrics


def _as_json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_metric_dict(metrics: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not metrics:
        return out
    for key, value in metrics.items():
        name = str(key).strip()
        if not name.startswith("eval_"):
            continue
        scalar = _as_json_scalar(value)
        if isinstance(scalar, (str, int, float, bool)) or scalar is None:
            out[name] = scalar
    return out


def _json_list(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]
    return json.dumps([_as_json_scalar(x) for x in raw], ensure_ascii=True)


def write_eval_score_dump(
    output_dir: Path,
    dataset: Any,
    predictions: Any,
    objective: str,
    global_step: int,
) -> Path | None:
    arr = np.asarray(predictions[0] if isinstance(predictions, tuple) else predictions)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return None
    if len(dataset) != int(arr.shape[0]):
        raise RuntimeError(f"eval score dump size mismatch: dataset={len(dataset)} predictions={arr.shape[0]}")
    dump_dir = output_dir / EVAL_SCORE_DUMPS_DIRNAME
    dump_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"checkpoint-{int(global_step)}"
    dump_path = dump_dir / f"{checkpoint_name}_eval_scores.jsonl"
    suffix = 1
    while dump_path.exists():
        dump_path = dump_dir / f"{checkpoint_name}_eval_scores_repeat{suffix}.jsonl"
        suffix += 1

    with dump_path.open("w", encoding="utf-8") as fh:
        for idx in range(int(arr.shape[0])):
            row = dataset[idx]
            payload: dict[str, Any] = {
                "checkpoint": checkpoint_name,
                "global_step": int(global_step),
                "objective": str(objective),
                "user_idx": int(row.get("user_idx", -1) or -1),
                "group_key": str(row.get("group_key", "") or ""),
                "chosen_label_source": str(row.get("chosen_label_source", "") or ""),
                "chosen_item_idx": int(row.get("chosen_item_idx", -1) or -1),
                "chosen_pre_rank": int(row.get("chosen_pre_rank", -1) or -1),
                "chosen_learned_rank": int(row.get("chosen_learned_rank", row.get("chosen_pre_rank", -1)) or -1),
                "chosen_learned_rank_band": str(row.get("chosen_learned_rank_band", "") or ""),
                "chosen_primary_reason": str(row.get("chosen_primary_reason", "") or ""),
                "chosen_easy_but_useful": bool(row.get("chosen_easy_but_useful", False)),
                "chosen_hard_but_learnable": bool(row.get("chosen_hard_but_learnable", False)),
                "chosen_non_actionable": bool(row.get("chosen_non_actionable", False)),
            }
            if objective == "pairwise":
                chosen_score = float(arr[idx, 0])
                rejected_score = float(arr[idx, 1]) if arr.shape[1] > 1 else float("nan")
                payload.update(
                    {
                        "selection_bucket": str(row.get("selection_bucket", "") or ""),
                        "selection_slot": int(row.get("selection_slot", 0) or 0),
                        "rejected_item_idx": int(row.get("rejected_item_idx", -1) or -1),
                        "rejected_pre_rank": int(row.get("rejected_pre_rank", -1) or -1),
                        "rejected_learned_rank": int(row.get("rejected_learned_rank", row.get("rejected_pre_rank", -1)) or -1),
                        "rejected_learned_rank_band": str(row.get("rejected_learned_rank_band", "") or ""),
                        "pair_weight": float(row.get("pair_weight", 1.0) or 1.0),
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "margin": float(chosen_score - rejected_score),
                    }
                )
            else:
                chosen_score = float(arr[idx, 0])
                max_negative_score = float(arr[idx, 1])
                top1_prob = float(arr[idx, 2])
                negative_count = int(round(float(arr[idx, 3])))
                neg_scores = arr[idx, 4 : 4 + max(0, negative_count)].astype(np.float32).tolist() if arr.shape[1] > 4 else []
                payload.update(
                    {
                        "selection_buckets": _json_list(row.get("selection_buckets", [])),
                        "selection_slots": _json_list(row.get("selection_slots", [])),
                        "neg_item_idxs": _json_list(row.get("neg_item_idxs", [])),
                        "neg_pre_ranks": _json_list(row.get("neg_pre_ranks", [])),
                        "neg_learned_ranks": _json_list(row.get("neg_learned_ranks", [])),
                        "neg_weights": _json_list(row.get("neg_weights", [])),
                        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
                        "chosen_score": chosen_score,
                        "max_negative_score": max_negative_score,
                        "margin": float(chosen_score - max_negative_score),
                        "top1_prob": top1_prob,
                        "negative_count": int(negative_count),
                        "neg_scores": _json_list(neg_scores),
                    }
                )
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return dump_path


class EvalScoreDumpMixin:
    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if (
            str(description or "").strip().lower().startswith("evaluation")
            and str(metric_key_prefix or "eval").strip().lower() == "eval"
            and getattr(output, "predictions", None) is not None
            and self.is_world_process_zero()
        ):
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None:
                probe_metrics = compute_boundary_probe_metrics(dataset, output.predictions, str(RM_OBJECTIVE))
                if probe_metrics:
                    output.metrics.update({f"{metric_key_prefix}_{k}": float(v) for k, v in probe_metrics.items()})
                if SAVE_EVAL_SCORE_DUMPS:
                    dump_path = write_eval_score_dump(
                        Path(self.args.output_dir),
                        dataset,
                        output.predictions,
                        str(RM_OBJECTIVE),
                        int(getattr(self.state, "global_step", 0) or 0),
                    )
                    if dump_path is not None:
                        print(f"[EVAL_DUMP] saved checkpoint scoring: {dump_path}")
        return output


class PairwiseRewardTrainer(EvalScoreDumpMixin, Trainer):
    def _score(self, model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return reward_logits_to_score(logits)

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        weights = inputs.pop("pair_weight").float()
        inputs.pop("labels", None)
        chosen_score = self._score(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
        rejected_score = self._score(model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"])
        per_example = -torch_f.logsigmoid(chosen_score - rejected_score)
        weights = weights.to(per_example.device)
        loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)
        if return_outputs:
            return loss, {"logits": torch.stack([chosen_score, rejected_score], dim=-1)}
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            weights = inputs["pair_weight"].float()
            chosen_score = self._score(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
            rejected_score = self._score(model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"])
            per_example = -torch_f.logsigmoid(chosen_score - rejected_score)
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-6)
            logits = torch.stack([chosen_score, rejected_score], dim=-1)
            labels = torch.ones((logits.shape[0],), dtype=torch.float32, device=logits.device)
        return (loss.detach(), None, None) if prediction_loss_only else (loss.detach(), logits.detach(), labels)


class ListwiseRewardTrainer(EvalScoreDumpMixin, Trainer):
    def _score(self, model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return reward_logits_to_score(logits)

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        sample_weight = inputs.pop("sample_weight").float()
        neg_weight = inputs.pop("neg_weight").float()
        neg_presence_mask = inputs.pop("neg_presence_mask").float()
        inputs.pop("labels", None)

        chosen_score = self._score(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
        batch_size, max_neg, seq_len = inputs["neg_input_ids"].shape
        flat_neg_ids = inputs["neg_input_ids"].reshape(batch_size * max_neg, seq_len)
        flat_neg_mask = inputs["neg_attention_mask"].reshape(batch_size * max_neg, seq_len)
        neg_score = score_in_chunks(
            self._score,
            model,
            flat_neg_ids,
            flat_neg_mask,
            RM_SCORE_CHUNK_SIZE,
        ).reshape(batch_size, max_neg)

        chosen_logit = chosen_score / max(float(RM_LISTWISE_TEMPERATURE), 1e-6)
        neg_logit = neg_score / max(float(RM_LISTWISE_TEMPERATURE), 1e-6)
        neg_log_weight = torch.log(neg_weight.clamp_min(1e-6))
        masked_neg = (neg_logit + neg_log_weight).masked_fill(neg_presence_mask <= 0.0, float("-inf"))
        all_logits = torch.cat([chosen_logit.unsqueeze(1), masked_neg], dim=1)
        log_prob_pos = torch.log_softmax(all_logits, dim=1)[:, 0]
        per_example = -log_prob_pos

        sample_weight = sample_weight.to(per_example.device)
        loss = (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        if return_outputs:
            max_neg_score = masked_neg.max(dim=1).values
            outputs = {
                "chosen_score": chosen_score,
                "max_negative_score": max_neg_score,
                "negative_count": neg_presence_mask.sum(dim=1),
            }
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            sample_weight = inputs["sample_weight"].float()
            neg_weight = inputs["neg_weight"].float()
            neg_presence_mask = inputs["neg_presence_mask"].float()
            chosen_score = self._score(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
            batch_size, max_neg, seq_len = inputs["neg_input_ids"].shape
            flat_neg_ids = inputs["neg_input_ids"].reshape(batch_size * max_neg, seq_len)
            flat_neg_mask = inputs["neg_attention_mask"].reshape(batch_size * max_neg, seq_len)
            neg_score = score_in_chunks(
                self._score,
                model,
                flat_neg_ids,
                flat_neg_mask,
                RM_SCORE_CHUNK_SIZE,
            ).reshape(batch_size, max_neg)
            chosen_logit = chosen_score / max(float(RM_LISTWISE_TEMPERATURE), 1e-6)
            neg_logit = neg_score / max(float(RM_LISTWISE_TEMPERATURE), 1e-6)
            neg_log_weight = torch.log(neg_weight.clamp_min(1e-6))
            masked_neg = (neg_logit + neg_log_weight).masked_fill(neg_presence_mask <= 0.0, float("-inf"))
            all_logits = torch.cat([chosen_logit.unsqueeze(1), masked_neg], dim=1)
            log_prob_pos = torch.log_softmax(all_logits, dim=1)[:, 0]
            per_example = -log_prob_pos
            loss = (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            max_negative_score = neg_score.masked_fill(neg_presence_mask <= 0.0, float("-inf")).max(dim=1).values
            top1_prob = torch.exp(log_prob_pos)
            negative_count = neg_presence_mask.sum(dim=1)
            neg_score_dump = neg_score.masked_fill(neg_presence_mask <= 0.0, float("nan"))
            logits = torch.cat(
                [
                    torch.stack([chosen_score, max_negative_score, top1_prob, negative_count], dim=-1),
                    neg_score_dump,
                ],
                dim=1,
            )
            labels = torch.ones((logits.shape[0],), dtype=torch.float32, device=logits.device)
        return (loss.detach(), None, None) if prediction_loss_only else (loss.detach(), logits.detach(), labels)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    source_11 = resolve_stage11_dataset_run()
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
    ds_meta_path = source_11 / "run_meta.json"
    if not ds_meta_path.exists():
        raise FileNotFoundError(f"dataset run_meta.json missing: {ds_meta_path}")
    ds_meta = json.loads(ds_meta_path.read_text(encoding="utf-8"))

    prompt_mode = str(ds_meta.get("config", {}).get("prompt_mode", ds_meta.get("prompt_mode", ""))).strip().lower()
    if RM_REQUIRE_SCORING_PROMPT and prompt_mode != RM_EXPECTED_PROMPT_MODE:
        raise RuntimeError(
            f"reward-model scoring prompt requires prompt_mode={RM_EXPECTED_PROMPT_MODE}, "
            f"got {prompt_mode or '<missing>'} in {ds_meta_path}"
        )
    if ENFORCE_STAGE09_GATE:
        gate = ds_meta.get("stage09_gate_result", {})
        if not isinstance(gate, dict) or not gate:
            raise RuntimeError("stage09 gate evidence missing in dataset run_meta")
    if ENFORCE_REQUIRED_BASE_MODEL and BASE_MODEL != REQUIRED_BASE_MODEL:
        raise RuntimeError(
            f"base model mismatch: QLORA_BASE_MODEL={BASE_MODEL}, expected={REQUIRED_BASE_MODEL}. "
            "Set QLORA_ENFORCE_REQUIRED_BASE_MODEL=false to override."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if not QLORA_RESUME_RUN_DIR else Path(QLORA_RESUME_RUN_DIR).name.split("_", 1)[0]
    out_dir = Path(QLORA_RESUME_RUN_DIR) if QLORA_RESUME_RUN_DIR else (OUTPUT_ROOT / f"{run_id}_{RUN_TAG}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows, data_files = collect_stage11_rows(source_11, buckets, PAIRWISE_SOURCE_MODE)
    train_band_filter_audit: dict[str, Any] = {
        "enabled": False,
        "target_true_bands": sorted(list(TARGET_TRUE_BANDS)),
        "input_rows": int(len(train_rows)),
        "output_rows": int(len(train_rows)),
    }
    eval_band_filter_audit: dict[str, Any] = {
        "enabled": False,
        "target_true_bands": sorted(list(TARGET_TRUE_BANDS)),
        "input_rows": int(len(eval_rows)),
        "output_rows": int(len(eval_rows)),
    }
    if TARGET_TRUE_BANDS:
        train_rows, train_band_filter_audit = filter_rows_for_target_true_bands(train_rows, TARGET_TRUE_BANDS)
        eval_rows, eval_band_filter_audit = filter_rows_for_target_true_bands(eval_rows, TARGET_TRUE_BANDS)
        if not train_rows:
            raise RuntimeError(
                f"no train rows remain after target_true_band filter={sorted(list(TARGET_TRUE_BANDS))}"
            )
        print(
            f"[DATA] target_true_bands={sorted(list(TARGET_TRUE_BANDS))} "
            f"filtered train_rows={len(train_rows)} eval_rows={len(eval_rows)}"
        )
    print(f"[DATA] raw_train rows: {len(train_rows)}")
    print(f"[CONFIG] base_model={BASE_MODEL} pair_policy={DPO_PAIR_POLICY} source_mode={PAIRWISE_SOURCE_MODE}")

    train_pairs_full, train_pair_audit = build_rich_sft_dpo_pairs(
        train_rows,
        DPO_MAX_PAIRS_PER_USER,
        SEED,
        true_max_pairs_per_user=DPO_TRUE_MAX_PAIRS_PER_USER,
        valid_max_pairs_per_user=DPO_VALID_MAX_PAIRS_PER_USER,
        hist_max_pairs_per_user=DPO_HIST_MAX_PAIRS_PER_USER,
        allow_mid_neg=DPO_ALLOW_MID_NEG,
    )
    train_reward_records = pair_records_for_reward_training(train_pairs_full)
    if RM_OBJECTIVE == "listwise":
        train_examples, train_weight_audit = build_listwise_rm_examples(
            train_reward_records,
            "train",
            RM_LISTWISE_TRAIN_MIN_NEGATIVES,
        )
    else:
        train_examples, train_weight_audit = build_rm_examples(train_reward_records, "train")
    if eval_rows:
        eval_pairs_full, eval_pair_audit = build_rich_sft_dpo_pairs(
            eval_rows,
            RM_EVAL_MAX_PAIRS_PER_USER,
            SEED + 1,
            true_max_pairs_per_user=RM_EVAL_TRUE_MAX_PAIRS_PER_USER,
            valid_max_pairs_per_user=RM_EVAL_VALID_MAX_PAIRS_PER_USER,
            hist_max_pairs_per_user=RM_EVAL_HIST_MAX_PAIRS_PER_USER,
            allow_mid_neg=DPO_ALLOW_MID_NEG,
        )
        eval_reward_records = pair_records_for_reward_training(eval_pairs_full)
        if RM_OBJECTIVE == "listwise":
            eval_examples, eval_weight_audit = build_listwise_rm_examples(
                eval_reward_records,
                "eval",
                RM_LISTWISE_EVAL_MIN_NEGATIVES,
            )
        else:
            eval_examples, eval_weight_audit = build_rm_examples(eval_reward_records, "eval")
    else:
        eval_pair_audit = {}
        eval_weight_audit = {}
        eval_examples = []

    has_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if (has_cuda and USE_BF16 and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = None
    if has_cuda and USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs: dict[str, Any] = {"trust_remote_code": TRUST_REMOTE_CODE}
    if has_cuda:
        model_kwargs["device_map"] = "auto"
    if ATTN_IMPLEMENTATION and ATTN_IMPLEMENTATION not in {"auto", "default"}:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype
    try:
        model_cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
        setattr(model_cfg, "num_labels", 1)
        if is_qwen35_model_type(str(getattr(model_cfg, "model_type", ""))):
            target_mamba_dtype = str(QWEN35_MAMBA_SSM_DTYPE or "auto").strip().lower()
            if target_mamba_dtype in {"", "auto"} and compute_dtype == torch.bfloat16:
                target_mamba_dtype = "bfloat16"
            if target_mamba_dtype not in {"", "auto"}:
                txt_cfg = getattr(model_cfg, "text_config", None)
                if isinstance(txt_cfg, dict):
                    txt_cfg["mamba_ssm_dtype"] = target_mamba_dtype
                elif txt_cfg is not None:
                    setattr(txt_cfg, "mamba_ssm_dtype", target_mamba_dtype)
        model_kwargs["config"] = model_cfg
    except Exception:
        pass

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, **model_kwargs)
    if bnb_config is not None:
        loaded_model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
        if is_qwen35_model_type(loaded_model_type) and QWEN35_SAFE_KBIT_PREP:
            model = prepare_model_for_kbit_training_qwen35_safe(model, enable_input_grads=bool(GRADIENT_CHECKPOINTING))
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[x.strip() for x in LORA_TARGET_MODULES.split(",") if x.strip()],
        modules_to_save=["score"] if hasattr(model, "score") else None,
    )
    model = get_peft_model(model, lora_cfg)
    if (
        is_qwen35_model_type(str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower())
        and has_cuda
        and compute_dtype == torch.bfloat16
        and QWEN35_FORCE_FLOAT_PARAMS_BF16
    ):
        recast_float_params(model, torch.bfloat16)
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.print_trainable_parameters()

    if RM_OBJECTIVE == "listwise":
        train_records, train_token_audit = build_tokenized_listwise_rm_records(
            train_examples,
            tokenizer,
            "train",
            RM_LISTWISE_TRAIN_MIN_NEGATIVES,
        )
        eval_records, eval_token_audit = (
            build_tokenized_listwise_rm_records(
                eval_examples,
                tokenizer,
                "eval",
                RM_LISTWISE_EVAL_MIN_NEGATIVES,
            )
            if eval_examples
            else ([], {})
        )
    else:
        train_records, train_token_audit = build_tokenized_rm_records(train_examples, tokenizer, "train")
        eval_records, eval_token_audit = (
            build_tokenized_rm_records(eval_examples, tokenizer, "eval") if eval_examples else ([], {})
        )
    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records) if eval_records else None

    best_metric_name = None
    if eval_ds is not None:
        best_metric_name = str(BEST_MODEL_METRIC_NAME or "").strip() or "eval_boundary_11_30_win_rate"

    args = TrainingArguments(
        output_dir=(out_dir / "trainer_output").as_posix(),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, EVAL_BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS if eval_ds is not None else None,
        save_steps=SAVE_STEPS,
        save_total_limit=max(1, int(SAVE_TOTAL_LIMIT)),
        bf16=bool(has_cuda and compute_dtype == torch.bfloat16),
        fp16=bool(has_cuda and compute_dtype == torch.float16),
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_strategy="steps" if eval_ds is not None else "no",
        load_best_model_at_end=bool(eval_ds is not None),
        metric_for_best_model=best_metric_name,
        greater_is_better=True if best_metric_name is not None else None,
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": build_listwise_rm_collator(tokenizer) if RM_OBJECTIVE == "listwise" else build_rm_collator(tokenizer),
        "compute_metrics": (
            compute_listwise_rm_metrics
            if (eval_ds is not None and RM_OBJECTIVE == "listwise")
            else (compute_rm_metrics if eval_ds is not None else None)
        ),
    }
    try:
        trainer_cls = ListwiseRewardTrainer if RM_OBJECTIVE == "listwise" else PairwiseRewardTrainer
        trainer = trainer_cls(**trainer_kwargs, tokenizer=tokenizer)
    except TypeError:
        trainer_cls = ListwiseRewardTrainer if RM_OBJECTIVE == "listwise" else PairwiseRewardTrainer
        trainer = trainer_cls(**trainer_kwargs, processing_class=tokenizer)

    resume_checkpoint = None
    trainer_out_dir = out_dir / "trainer_output"
    if QLORA_RESUME_RUN_DIR and trainer_out_dir.exists():
        cp_dirs = [d for d in trainer_out_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if cp_dirs:
            cp_dirs.sort(key=lambda d: int(d.name.split("-")[1]))
            resume_checkpoint = cp_dirs[-1].as_posix()

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_result = trainer.evaluate() if eval_ds is not None else {}

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir.as_posix())
    tokenizer.save_pretrained(adapter_dir.as_posix())

    eval_metric_payload = _json_metric_dict(eval_result)
    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "training_method": "RM_listwise" if RM_OBJECTIVE == "listwise" else "RM_pairwise",
        "source_stage11_dataset_run": str(source_11),
        "source_stage09_run": str(ds_meta.get("source_stage09_run", "")).strip(),
        "buckets": buckets,
        "base_model": BASE_MODEL,
        "use_4bit": bool(bnb_config is not None),
        "dtype": str(compute_dtype),
        "attn_implementation": str(ATTN_IMPLEMENTATION or ""),
        "per_device_train_batch_size": int(BATCH_SIZE),
        "per_device_eval_batch_size": int(EVAL_BATCH_SIZE),
        "gradient_accumulation_steps": int(GRAD_ACC),
        "train_pairs": len(train_ds),
        "eval_pairs": len(eval_ds) if eval_ds is not None else 0,
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
        "eval_loss": float(eval_result.get("eval_loss", 0.0)) if eval_result else None,
        "eval_pair_accuracy": float(eval_result.get("eval_pair_accuracy", 0.0)) if eval_result else None,
        "eval_pair_margin_mean": float(eval_result.get("eval_pair_margin_mean", 0.0)) if eval_result else None,
        "eval_listwise_win_rate": float(eval_result.get("eval_listwise_win_rate", 0.0)) if eval_result else None,
        "eval_listwise_margin_mean": float(eval_result.get("eval_listwise_margin_mean", 0.0)) if eval_result else None,
        "eval_top1_prob_mean": float(eval_result.get("eval_top1_prob_mean", 0.0)) if eval_result else None,
        "eval_negative_count_mean": float(eval_result.get("eval_negative_count_mean", 0.0)) if eval_result else None,
        "eval_score_dumps_enabled": bool(SAVE_EVAL_SCORE_DUMPS),
        "eval_score_dumps_dir": str((trainer_out_dir / EVAL_SCORE_DUMPS_DIRNAME)) if SAVE_EVAL_SCORE_DUMPS else None,
        "rm_config": {
            "objective": str(RM_OBJECTIVE),
            "pair_policy": str(DPO_PAIR_POLICY),
            "pair_prompt_style": str(PAIR_PROMPT_STYLE),
            "target_true_bands": sorted(list(TARGET_TRUE_BANDS)),
            "prompt_mode_expected": str(RM_EXPECTED_PROMPT_MODE),
            "true_pair_weight": float(RM_TRUE_PAIR_WEIGHT),
            "valid_pair_weight": float(RM_VALID_PAIR_WEIGHT),
            "hist_pair_weight": float(RM_HIST_PAIR_WEIGHT),
            "boundary_pair_weight": float(RM_BOUNDARY_PAIR_WEIGHT),
            "structured_pair_weight": float(RM_STRUCTURED_PAIR_WEIGHT),
            "slot_decay": float(RM_SLOT_DECAY),
            "sample_weight_mode": str(RM_SAMPLE_WEIGHT_MODE),
            "sample_weight_cap": float(RM_SAMPLE_WEIGHT_CAP),
            "listwise_max_negatives": int(RM_LISTWISE_MAX_NEGATIVES),
            "listwise_train_min_negatives": int(RM_LISTWISE_TRAIN_MIN_NEGATIVES),
            "listwise_eval_min_negatives": int(RM_LISTWISE_EVAL_MIN_NEGATIVES),
            "listwise_temperature": float(RM_LISTWISE_TEMPERATURE),
            "score_chunk_size": int(RM_SCORE_CHUNK_SIZE),
            "load_best_model_at_end": bool(eval_ds is not None),
            "metric_for_best_model": str(best_metric_name) if best_metric_name is not None else None,
            "boundary_probe_score_mode": str(BOUNDARY_PROBE_SCORE_MODE),
            "boundary_probe_head_weight": float(BOUNDARY_PROBE_HEAD_WEIGHT),
            "boundary_probe_reason_weight": float(BOUNDARY_PROBE_REASON_WEIGHT),
            "boundary_probe_reason_targets": list(BOUNDARY_PROBE_REASON_TARGETS),
            "boundary_too_easy_win_rate": float(BOUNDARY_TOO_EASY_WIN_RATE),
            "boundary_too_easy_min_count": int(BOUNDARY_TOO_EASY_MIN_COUNT),
            "eval_max_pairs_per_user": int(RM_EVAL_MAX_PAIRS_PER_USER),
            "eval_true_max_pairs_per_user": int(RM_EVAL_TRUE_MAX_PAIRS_PER_USER),
            "eval_valid_max_pairs_per_user": int(RM_EVAL_VALID_MAX_PAIRS_PER_USER),
            "eval_hist_max_pairs_per_user": int(RM_EVAL_HIST_MAX_PAIRS_PER_USER),
            "eval_score_dumps_dirname": str(EVAL_SCORE_DUMPS_DIRNAME),
        },
        "pair_audit": {"train": train_pair_audit, "eval": eval_pair_audit},
        "pair_weight_audit": {"train": train_weight_audit, "eval": eval_weight_audit},
        "target_true_band_filter": {"train": train_band_filter_audit, "eval": eval_band_filter_audit},
        "tokenization_audit": {"train": train_token_audit, "eval": eval_token_audit},
        "data_files": data_files,
        "adapter_dir": str(adapter_dir),
        "eval_model_type": "rm",
        "adapter_task_type": "SEQ_CLS",
        "eval_metrics": eval_metric_payload,
    }
    payload.update(eval_metric_payload)
    run_meta_path = out_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    pointer_path = write_latest_run_pointer(
        "stage11_2_rm_train",
        out_dir,
        extra={
            "run_tag": RUN_TAG,
            "base_model": BASE_MODEL,
            "training_method": "RM_listwise" if RM_OBJECTIVE == "listwise" else "RM_pairwise",
            "eval_model_type": "rm",
        },
    )
    print(f"[DONE] RM adapter saved to: {adapter_dir}")
    print(f"[DONE] run_meta: {run_meta_path}")
    print(f"[DONE] latest pointer: {pointer_path}")


if __name__ == "__main__":
    main()
