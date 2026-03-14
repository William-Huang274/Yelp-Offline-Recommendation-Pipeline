from __future__ import annotations

import gc
import json
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pipeline.project_paths import env_or_project_path, resolve_latest_run_pointer, write_latest_run_pointer

# Keep transformers on torch path in mixed TensorFlow/Keras env.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
import transformers.modeling_utils as _tf_modeling_utils


RUN_TAG = "stage11_2_qlora_train"
QLORA_RESUME_RUN_DIR = os.getenv("QLORA_RESUME_RUN_DIR", "").strip()
INPUT_11_RUN_DIR = os.getenv("INPUT_11_RUN_DIR", "").strip()
INPUT_11_ROOT = env_or_project_path("INPUT_11_ROOT_DIR", "data/output/11_qlora_data")
INPUT_11_SUFFIX = "_stage11_1_qlora_build_dataset"
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_MODELS_ROOT_DIR", "data/output/11_qlora_models")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
BASE_MODEL = os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen3-4B").strip()
REQUIRED_BASE_MODEL = os.getenv("QLORA_REQUIRED_BASE_MODEL", "Qwen/Qwen3-4B").strip()
ENFORCE_REQUIRED_BASE_MODEL = (
    os.getenv("QLORA_ENFORCE_REQUIRED_BASE_MODEL", "true").strip().lower() == "true"
)
TRUST_REMOTE_CODE = os.getenv("QLORA_TRUST_REMOTE_CODE", "true").strip().lower() == "true"

USE_4BIT = os.getenv("QLORA_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("QLORA_USE_BF16", "true").strip().lower() == "true"
DEVICE_MAP = os.getenv("QLORA_DEVICE_MAP", "auto").strip() or "auto"
MAX_MEMORY_CUDA = os.getenv("QLORA_MAX_MEMORY_CUDA", "").strip()
MAX_MEMORY_CPU = os.getenv("QLORA_MAX_MEMORY_CPU", "").strip()
OFFLOAD_FOLDER = os.getenv("QLORA_OFFLOAD_FOLDER", "").strip()
OFFLOAD_STATE_DICT = os.getenv("QLORA_OFFLOAD_STATE_DICT", "true").strip().lower() == "true"
LOW_CPU_MEM_USAGE = os.getenv("QLORA_LOW_CPU_MEM_USAGE", "true").strip().lower() == "true"
DISABLE_ALLOC_WARMUP = os.getenv("QLORA_DISABLE_ALLOC_WARMUP", "true").strip().lower() == "true"
DISABLE_PARALLEL_LOADING = os.getenv("QLORA_DISABLE_PARALLEL_LOADING", "true").strip().lower() == "true"
PARALLEL_LOADING_WORKERS = int(os.getenv("QLORA_PARALLEL_LOADING_WORKERS", "1").strip() or 1)
CLEAR_CUDA_CACHE_BEFORE_LORA = (
    os.getenv("QLORA_CLEAR_CUDA_CACHE_BEFORE_LORA", "true").strip().lower() == "true"
)
CAST_TRAINABLE_PARAMS_TO_COMPUTE = (
    os.getenv("QLORA_CAST_TRAINABLE_PARAMS_TO_COMPUTE", "false").strip().lower() == "true"
)
MANUAL_FP16_AUTOCAST = (
    os.getenv("QLORA_MANUAL_FP16_AUTOCAST", "false").strip().lower() == "true"
)
QWEN35_MAMBA_SSM_DTYPE = os.getenv("QLORA_QWEN35_MAMBA_SSM_DTYPE", "auto").strip().lower()
ATTN_IMPLEMENTATION = os.getenv("QLORA_ATTN_IMPLEMENTATION", "").strip().lower()
QWEN35_SAFE_KBIT_PREP = os.getenv("QLORA_QWEN35_SAFE_KBIT_PREP", "true").strip().lower() == "true"
QWEN35_FORCE_FLOAT_PARAMS_BF16 = (
    os.getenv("QLORA_QWEN35_FORCE_FLOAT_PARAMS_BF16", "true").strip().lower() == "true"
)
LORA_R = int(os.getenv("QLORA_LORA_R", "16").strip() or 16)
LORA_ALPHA = int(os.getenv("QLORA_LORA_ALPHA", "32").strip() or 32)
LORA_DROPOUT = float(os.getenv("QLORA_LORA_DROPOUT", "0.05").strip() or 0.05)
LORA_TARGET_MODULES = os.getenv(
    "QLORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
).strip()

MAX_SEQ_LEN = int(os.getenv("QLORA_MAX_SEQ_LEN", "768").strip() or 768)
MAX_TRAIN_ROWS = int(os.getenv("QLORA_MAX_TRAIN_ROWS", "0").strip() or 0)
MAX_EVAL_ROWS = int(os.getenv("QLORA_MAX_EVAL_ROWS", "0").strip() or 0)
MAX_NEG_POS_RATIO = float(os.getenv("QLORA_MAX_NEG_POS_RATIO", "4.0").strip() or 4.0)
TRAIN_SOURCE = os.getenv("QLORA_TRAIN_SOURCE", "default").strip().lower() or "default"
NEG_REBALANCE_MODE = os.getenv("QLORA_NEG_REBALANCE_MODE", "random").strip().lower() or "random"
MAX_TRAIN_USERS = int(os.getenv("QLORA_MAX_TRAIN_USERS", "0").strip() or 0)
MAX_EVAL_USERS = int(os.getenv("QLORA_MAX_EVAL_USERS", "0").strip() or 0)
USER_CAP_RANDOMIZED = os.getenv("QLORA_USER_CAP_RANDOMIZED", "true").strip().lower() == "true"
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "true").strip().lower() == "true"

EPOCHS = float(os.getenv("QLORA_EPOCHS", "1.0").strip() or 1.0)
LR = float(os.getenv("QLORA_LR", "2e-4").strip() or 2e-4)
WEIGHT_DECAY = float(os.getenv("QLORA_WEIGHT_DECAY", "0.01").strip() or 0.01)
WARMUP_RATIO = float(os.getenv("QLORA_WARMUP_RATIO", "0.03").strip() or 0.03)
BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "1").strip() or 1)
GRAD_ACC = int(os.getenv("QLORA_GRAD_ACC", "16").strip() or 16)
EVAL_BATCH_SIZE = int(os.getenv("QLORA_EVAL_BATCH_SIZE", str(BATCH_SIZE)).strip() or BATCH_SIZE)
EVAL_STEPS = int(os.getenv("QLORA_EVAL_STEPS", "100").strip() or 100)
SAVE_STEPS = int(os.getenv("QLORA_SAVE_STEPS", "100").strip() or 100)
SAVE_TOTAL_LIMIT = int(os.getenv("QLORA_SAVE_TOTAL_LIMIT", "2").strip() or 2)
LOGGING_STEPS = int(os.getenv("QLORA_LOGGING_STEPS", "20").strip() or 20)
DATASET_MAP_NUM_PROC = int(os.getenv("QLORA_DATASET_MAP_NUM_PROC", "1").strip() or 1)
FORMAT_MAP_NUM_PROC = int(os.getenv("QLORA_FORMAT_MAP_NUM_PROC", str(DATASET_MAP_NUM_PROC)).strip() or DATASET_MAP_NUM_PROC)
TOKENIZE_MAP_NUM_PROC = int(os.getenv("QLORA_TOKENIZE_MAP_NUM_PROC", str(DATASET_MAP_NUM_PROC)).strip() or DATASET_MAP_NUM_PROC)
DATALOADER_NUM_WORKERS = int(os.getenv("QLORA_DATALOADER_NUM_WORKERS", "0").strip() or 0)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_PAD_TO_MULTIPLE_OF", "0").strip() or 0)
GRADIENT_CHECKPOINTING = os.getenv("QLORA_GRADIENT_CHECKPOINTING", "true").strip().lower() == "true"
LOSS_MODE = os.getenv("QLORA_LOSS_MODE", "full_ce").strip().lower() or "full_ce"

TOKEN_AUDIT_ENABLED = os.getenv("QLORA_TOKEN_AUDIT_ENABLED", "true").strip().lower() == "true"
TOKEN_AUDIT_MAX_ROWS = int(os.getenv("QLORA_TOKEN_AUDIT_MAX_ROWS", "4000").strip() or 4000)
TOKEN_AUDIT_BATCH_SIZE = int(os.getenv("QLORA_TOKEN_AUDIT_BATCH_SIZE", "256").strip() or 256)
CHECKPOINT_AUDIT_ENABLED = os.getenv("QLORA_CHECKPOINT_AUDIT_ENABLED", "true").strip().lower() == "true"
CHECKPOINT_AUDIT_PROFILE = os.getenv("QLORA_CHECKPOINT_AUDIT_PROFILE", "smoke").strip().lower() or "smoke"
CHECKPOINT_AUDIT_PROMPT_MODE_RAW = os.getenv("QLORA_CHECKPOINT_AUDIT_PROMPT_MODE", "").strip().lower()
CHECKPOINT_AUDIT_INVERT_PROB = os.getenv("QLORA_CHECKPOINT_AUDIT_INVERT_PROB", "true").strip().lower() == "true"
CHECKPOINT_AUDIT_USE_EVAL_SPLIT = os.getenv("QLORA_CHECKPOINT_AUDIT_USE_EVAL_SPLIT", "true").strip().lower() == "true"
CHECKPOINT_AUDIT_MAX_USERS = int(os.getenv("QLORA_CHECKPOINT_AUDIT_MAX_USERS", "64").strip() or 64)
CHECKPOINT_AUDIT_MAX_ROWS = int(os.getenv("QLORA_CHECKPOINT_AUDIT_MAX_ROWS", "8192").strip() or 8192)

BINARY_CLASS_TOKEN_IDS: tuple[int, int] | None = None


def parse_bucket_override(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return sorted(list(set(out)))


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage11_dataset_run() -> Path:
    if INPUT_11_RUN_DIR:
        p = Path(INPUT_11_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_11_RUN_DIR not found: {p}")
        return p
    pinned = resolve_latest_run_pointer("stage11_1_qlora_build_dataset")
    if pinned is not None:
        return pinned
    return pick_latest_run(INPUT_11_ROOT, INPUT_11_SUFFIX)


def collect_json_files(
    source_11: Path,
    buckets: list[int],
    train_source: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    source_key = str(train_source or "default").strip().lower()
    if source_key == "default":
        train_dir_name = "train_json"
        eval_dir_name = "eval_json"
    elif source_key == "rich_sft":
        train_dir_name = "rich_sft_train_json"
        eval_dir_name = "rich_sft_eval_json"
    elif source_key == "pairwise_pool":
        train_dir_name = "pairwise_pool_train_json"
        eval_dir_name = "pairwise_pool_eval_json"
    else:
        raise ValueError(
            "unsupported QLORA_TRAIN_SOURCE="
            f"{train_source}. expected one of: default, rich_sft, pairwise_pool"
        )
    train_files: list[str] = []
    eval_files: list[str] = []
    summary: dict[str, Any] = {
        "train_source": source_key,
        "train_dir_name": train_dir_name,
        "eval_dir_name": eval_dir_name,
        "buckets": [],
    }
    for b in buckets:
        bdir = source_11 / f"bucket_{b}"
        train_dir = bdir / train_dir_name
        eval_dir = bdir / eval_dir_name
        if not bdir.exists():
            continue
        b_train = sorted([p.as_posix() for p in train_dir.glob("*.json")]) if train_dir.exists() else []
        b_eval = sorted([p.as_posix() for p in eval_dir.glob("*.json")]) if eval_dir.exists() else []
        train_files.extend(b_train)
        eval_files.extend(b_eval)
        summary["buckets"].append(
            {
                "bucket": int(b),
                "train_files": len(b_train),
                "eval_files": len(b_eval),
                "bucket_dir": str(bdir),
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
            }
        )
    if not train_files:
        raise RuntimeError(
            "no train json files found under "
            f"{source_11} for buckets={buckets} train_source={source_key}"
        )
    return train_files, eval_files, summary


def maybe_cap_rows(ds: Any, max_rows: int, seed: int) -> Any:
    if int(max_rows) <= 0:
        return ds
    if len(ds) <= int(max_rows):
        return ds
    idx = list(range(len(ds)))
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    idx = idx[: int(max_rows)]
    return ds.select(idx)


def maybe_cap_users(ds: Any, max_users: int, seed: int, randomized: bool) -> Any:
    if int(max_users) <= 0:
        return ds
    cols = list(getattr(ds, "column_names", []))
    if "user_idx" not in cols:
        return ds
    raw_users = [str(x) for x in ds["user_idx"]]
    if len(raw_users) <= 0:
        return ds
    seen: dict[str, None] = {}
    unique_users = [u for u in raw_users if not (u in seen or seen.setdefault(u, None))]
    if len(unique_users) <= int(max_users):
        return ds
    chosen = list(unique_users)
    if randomized:
        rng = random.Random(int(seed))
        rng.shuffle(chosen)
    chosen = chosen[: int(max_users)]
    chosen_set = set(chosen)
    keep = [i for i, user_key in enumerate(raw_users) if user_key in chosen_set]
    return ds.select(keep)


def build_checkpoint_audit_manifest(
    out_dir: Path,
    source_11: Path,
    ds_meta: dict[str, Any],
    buckets: list[int],
) -> dict[str, Any]:
    trainer_out_dir = out_dir / "trainer_output"
    cp_dirs = (
        sorted(
            [d for d in trainer_out_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if trainer_out_dir.exists()
        else []
    )
    audit_prompt_mode = (
        CHECKPOINT_AUDIT_PROMPT_MODE_RAW
        or str(ds_meta.get("prompt_mode", "")).strip().lower()
        or ("full_lite" if TRAIN_SOURCE == "rich_sft" else "full")
    )
    checkpoints = [
        {
            "name": d.name,
            "step": int(d.name.split("-")[1]),
            "path": str(d),
        }
        for d in cp_dirs
    ]
    return {
        "enabled": bool(CHECKPOINT_AUDIT_ENABLED),
        "selection_goal": "rank-side smoke audit for stage11_3; do not pick checkpoints by eval_loss alone",
        "source_stage11_dataset_run": str(source_11),
        "source_stage09_run": str(ds_meta.get("source_stage09_run", "")).strip(),
        "buckets": [int(b) for b in buckets],
        "train_source": str(TRAIN_SOURCE),
        "prompt_mode": str(audit_prompt_mode),
        "invert_prob": bool(CHECKPOINT_AUDIT_INVERT_PROB),
        "eval_profile": str(CHECKPOINT_AUDIT_PROFILE),
        "use_eval_split": bool(CHECKPOINT_AUDIT_USE_EVAL_SPLIT),
        "max_users_per_bucket": int(CHECKPOINT_AUDIT_MAX_USERS),
        "max_rows_per_bucket": int(CHECKPOINT_AUDIT_MAX_ROWS),
        "save_steps": int(SAVE_STEPS),
        "save_total_limit": int(SAVE_TOTAL_LIMIT),
        "available_checkpoints": checkpoints,
        "final_adapter_path": str(out_dir / "adapter"),
        "notes": [
            "Use the same prompt family for checkpoint smoke audit and final full-cohort eval.",
            "Increase QLORA_SAVE_TOTAL_LIMIT if you want to keep more checkpoints for rank-side selection.",
        ],
    }


def rebalance_negatives_random(ds: Any, max_neg_pos_ratio: float, seed: int) -> Any:
    if float(max_neg_pos_ratio) <= 0:
        return ds
    cols = list(getattr(ds, "column_names", []))
    if "label" not in cols:
        return ds
    labels = np.array(ds["label"], dtype=np.int32)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return ds
    target_neg = int(min(len(neg_idx), max(1, int(len(pos_idx) * float(max_neg_pos_ratio)))))
    if target_neg >= len(neg_idx):
        return ds
    rng = np.random.default_rng(int(seed))
    neg_pick = rng.choice(neg_idx, size=target_neg, replace=False)
    keep = np.sort(np.concatenate([pos_idx, neg_pick])).astype(np.int64).tolist()
    return ds.select(keep)


def rebalance_negatives_tier_preserve(ds: Any, max_neg_pos_ratio: float, seed: int) -> Any:
    if float(max_neg_pos_ratio) <= 0:
        return ds
    cols = list(getattr(ds, "column_names", []))
    if "label" not in cols or "neg_tier" not in cols:
        return ds
    labels = np.asarray(ds["label"], dtype=np.int32)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return ds
    target_neg = int(min(len(neg_idx), max(1, int(len(pos_idx) * float(max_neg_pos_ratio)))))
    if target_neg >= len(neg_idx):
        return ds

    raw_tiers = np.asarray([str(x or "") for x in ds["neg_tier"]], dtype=object)
    neg_tiers = raw_tiers[neg_idx]
    tier_keys = sorted({str(x or "") for x in neg_tiers.tolist()})
    if not tier_keys:
        return rebalance_negatives_random(ds, max_neg_pos_ratio, seed)

    neg_groups: dict[str, np.ndarray] = {
        tier: neg_idx[np.where(neg_tiers == tier)[0]]
        for tier in tier_keys
    }
    total_neg = float(len(neg_idx))
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for tier, tier_indices in neg_groups.items():
        raw_quota = float(target_neg) * (float(len(tier_indices)) / total_neg)
        quota = int(np.floor(raw_quota))
        quota = min(quota, int(len(tier_indices)))
        quotas[tier] = quota
        allocated += quota
        remainders.append((raw_quota - float(quota), tier))

    remaining = int(target_neg - allocated)
    if remaining > 0:
        for _, tier in sorted(remainders, key=lambda x: (-x[0], x[1])):
            spare = int(len(neg_groups[tier])) - int(quotas[tier])
            if spare <= 0:
                continue
            take = min(spare, remaining)
            quotas[tier] += int(take)
            remaining -= int(take)
            if remaining <= 0:
                break

    rng = np.random.default_rng(int(seed))
    picked_groups: list[np.ndarray] = []
    for tier, tier_indices in neg_groups.items():
        quota = int(quotas.get(tier, 0))
        if quota <= 0:
            continue
        if quota >= int(len(tier_indices)):
            picked_groups.append(np.asarray(tier_indices, dtype=np.int64))
            continue
        tier_pick = rng.choice(tier_indices, size=quota, replace=False)
        picked_groups.append(np.asarray(tier_pick, dtype=np.int64))

    if not picked_groups:
        return ds.select(pos_idx.astype(np.int64).tolist())

    neg_pick = np.sort(np.concatenate(picked_groups)).astype(np.int64)
    keep = np.sort(np.concatenate([pos_idx, neg_pick])).astype(np.int64).tolist()
    return ds.select(keep)


def rebalance_negatives(ds: Any, mode: str, max_neg_pos_ratio: float, seed: int) -> Any:
    key = str(mode or "random").strip().lower()
    if key in {"", "random"}:
        return rebalance_negatives_random(ds, max_neg_pos_ratio, seed)
    if key in {"off", "none", "disabled"}:
        return ds
    if key in {"tier_preserve", "tier-aware", "tier"}:
        return rebalance_negatives_tier_preserve(ds, max_neg_pos_ratio, seed)
    raise ValueError(
        "unsupported QLORA_NEG_REBALANCE_MODE="
        f"{mode}. expected one of: random, off, tier_preserve"
    )


def binary_label_stats(ds: Any) -> dict[str, int]:
    cols = list(getattr(ds, "column_names", []))
    if "label" not in cols:
        return {"rows": int(len(ds))}
    labels = np.array(ds["label"], dtype=np.int32)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    out = {"rows": int(len(ds)), "pos": pos, "neg": neg}
    if "user_idx" in cols:
        out["users"] = int(len({str(x) for x in ds["user_idx"]}))
    return out



def _length_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size <= 0:
        return {"rows": 0}
    q50, q90, q95, q99 = np.percentile(values, [50, 90, 95, 99]).tolist()
    return {
        "rows": int(values.size),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "p50": float(q50),
        "p90": float(q90),
        "p95": float(q95),
        "p99": float(q99),
        "max": int(values.max()),
    }


def token_length_audit_from_text_split(
    ds_split: Any,
    tokenizer: Any,
    max_seq_len: int,
    max_rows: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    total_rows = int(len(ds_split))
    probe = maybe_cap_rows(ds_split, int(max_rows), int(seed)) if int(max_rows) > 0 else ds_split
    if len(probe) <= 0:
        return {
            "rows_total": total_rows,
            "rows_measured": 0,
            "full_len": {"rows": 0},
            "prompt_len": {"rows": 0},
            "target_len": {"rows": 0},
            "clip_rate": 0.0,
            "prompt_clip_rate": 0.0,
        }

    def _lens(batch: dict[str, Any]) -> dict[str, Any]:
        texts = [str(x or "") for x in batch.get("text", [])]
        prompts = [t.rsplit(" ", 1)[0].strip() if t else "" for t in texts]
        full_tok = tokenizer(texts, truncation=False)
        prompt_tok = tokenizer(prompts, truncation=False)
        full_len = [int(len(x)) for x in full_tok["input_ids"]]
        prompt_len = [int(len(x)) for x in prompt_tok["input_ids"]]
        target_len = [max(0, int(f) - int(p)) for f, p in zip(full_len, prompt_len)]
        return {
            "full_len_raw": full_len,
            "prompt_len_raw": prompt_len,
            "target_len_raw": target_len,
        }

    measured = probe.map(
        _lens,
        batched=True,
        batch_size=max(1, int(batch_size)),
        remove_columns=list(probe.column_names),
    )
    full = np.asarray(measured["full_len_raw"], dtype=np.int32)
    prompt = np.asarray(measured["prompt_len_raw"], dtype=np.int32)
    target = np.asarray(measured["target_len_raw"], dtype=np.int32)

    return {
        "rows_total": total_rows,
        "rows_measured": int(full.size),
        "full_len": _length_stats(full),
        "prompt_len": _length_stats(prompt),
        "target_len": _length_stats(target),
        "clip_rate": float((full > int(max_seq_len)).mean()) if full.size > 0 else 0.0,
        "prompt_clip_rate": float((prompt > int(max_seq_len)).mean()) if prompt.size > 0 else 0.0,
    }


def label_supervision_audit_from_tok_split(
    ds_tok_split: Any,
    max_seq_len: int,
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    total_rows = int(len(ds_tok_split))
    probe = maybe_cap_rows(ds_tok_split, int(max_rows), int(seed)) if int(max_rows) > 0 else ds_tok_split
    if len(probe) <= 0:
        return {
            "rows_total": total_rows,
            "rows_measured": 0,
            "input_len": {"rows": 0},
            "supervised_tokens": {"rows": 0},
            "at_cap_rate": 0.0,
            "zero_supervision_rate": 0.0,
        }

    input_len = np.asarray([int(len(x)) for x in probe["input_ids"]], dtype=np.int32)
    supervised = np.asarray(
        [int(sum(1 for t in row if int(t) != -100)) for row in probe["labels"]],
        dtype=np.int32,
    )
    return {
        "rows_total": total_rows,
        "rows_measured": int(input_len.size),
        "input_len": _length_stats(input_len),
        "supervised_tokens": _length_stats(supervised),
        "at_cap_rate": float((input_len >= int(max_seq_len)).mean()) if input_len.size > 0 else 0.0,
        "zero_supervision_rate": float((supervised <= 0).mean()) if supervised.size > 0 else 0.0,
    }


def run_token_length_audit(
    ds_text: DatasetDict,
    ds_tok: DatasetDict,
    tokenizer: Any,
    max_seq_len: int,
    max_rows: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "enabled": True,
        "max_seq_len": int(max_seq_len),
        "max_rows_per_split": int(max_rows),
        "splits": {},
    }
    for split_name in ["train", "eval"]:
        if split_name not in ds_text or split_name not in ds_tok:
            continue
        raw_part = token_length_audit_from_text_split(
            ds_split=ds_text[split_name],
            tokenizer=tokenizer,
            max_seq_len=int(max_seq_len),
            max_rows=int(max_rows),
            batch_size=int(batch_size),
            seed=int(seed) + (0 if split_name == "train" else 1),
        )
        sup_part = label_supervision_audit_from_tok_split(
            ds_tok_split=ds_tok[split_name],
            max_seq_len=int(max_seq_len),
            max_rows=int(max_rows),
            seed=int(seed) + (11 if split_name == "train" else 12),
        )
        out["splits"][split_name] = {
            "raw_length": raw_part,
            "supervision": sup_part,
        }
    return out


def _build_text(prompt: str, target: str) -> str:
    p = str(prompt or "").strip()
    t = str(target or "").strip().upper()
    t = "YES" if t == "YES" else "NO"
    return f"{p} {t}"


def resolve_binary_class_token_ids(tokenizer: Any) -> tuple[int, int] | None:
    def _single_token_id(text: str) -> int | None:
        try:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception:
            return None
        if len(ids) != 1:
            return None
        return int(ids[0])

    yes_id = _single_token_id(" YES")
    no_id = _single_token_id(" NO")
    if yes_id is None or no_id is None:
        yes_id = _single_token_id("YES")
        no_id = _single_token_id("NO")
    if yes_id is None or no_id is None or yes_id == no_id:
        return None
    return (int(yes_id), int(no_id))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_model_for_kbit_training_qwen35_safe(model: Any, enable_input_grads: bool) -> Any:
    """
    Qwen3.5 linear-attention kernels require non-float32 activations.
    PEFT's generic helper upcasts non-4bit params to float32, which can
    propagate float32 activations into FLA kernels and fail at runtime.
    """
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


def recast_trainable_params(model: Any, target_dtype: Any) -> int:
    n_cast = 0
    for param in model.parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        if getattr(param, "dtype", None) != target_dtype:
            param.data = param.data.to(target_dtype)
            n_cast += 1
    return int(n_cast)


def build_weighted_causal_lm_collator(tokenizer: Any) -> Any:
    pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(int(len(f["input_ids"])) for f in features) if features else 0
        if PAD_TO_MULTIPLE_OF > 1 and max_len > 0:
            rounded_len = ((int(max_len) + int(PAD_TO_MULTIPLE_OF) - 1) // int(PAD_TO_MULTIPLE_OF)) * int(PAD_TO_MULTIPLE_OF)
            max_len = min(int(MAX_SEQ_LEN), int(rounded_len))
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        sample_weight: list[float] = []

        for f in features:
            ids = [int(x) for x in f["input_ids"]]
            att = [int(x) for x in f.get("attention_mask", [1] * len(ids))]
            lab = [int(x) for x in f["labels"]]
            pad_len = int(max_len - len(ids))
            if pad_len > 0:
                ids = ids + ([pad_token_id] * pad_len)
                att = att + ([0] * pad_len)
                lab = lab + ([-100] * pad_len)
            input_ids.append(ids)
            attention_mask.append(att)
            labels.append(lab)
            sample_weight.append(float(f.get("sample_weight", 1.0)))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
        }

    return _collate


def is_qwen35_model_type(model_type: str) -> bool:
    mt = str(model_type or "").strip().lower()
    return mt.startswith("qwen3_5")


class WeightedCausalLmTrainer(Trainer):
    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        weights = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels", None)
        use_bf16_autocast = bool(
            torch.cuda.is_available() and USE_BF16 and torch.cuda.is_bf16_supported()
        )
        if use_bf16_autocast:
            # torch>=2.4 prefers torch.amp.autocast("cuda", ...)
            try:
                amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            except Exception:
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif torch.cuda.is_available() and MANUAL_FP16_AUTOCAST:
            try:
                amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
            except Exception:
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None or logits is None:
            loss = outputs.get("loss")
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        valid_mask = shift_labels.ne(-100)
        if not bool(valid_mask.any()):
            # Preserve a valid autograd path for fully truncated samples.
            loss = logits[..., :1, :1].sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        # Only compute CE on supervised token positions. In this project the
        # target length is effectively 1, so full-sequence CE wastes memory
        # without changing semantics.
        # target_len is effectively 1 in this pipeline, so upcasting the
        # supervised logits to fp32 is cheap and avoids fp16 CE instability.
        valid_logits = shift_logits[valid_mask].float()
        valid_targets = shift_labels[valid_mask]
        use_binary_yesno = bool(LOSS_MODE == "binary_yesno" and BINARY_CLASS_TOKEN_IDS is not None)
        if use_binary_yesno:
            yes_id, no_id = BINARY_CLASS_TOKEN_IDS or (-1, -1)
            target_ok = valid_targets.eq(int(yes_id)) | valid_targets.eq(int(no_id))
            if bool(target_ok.all()):
                binary_logits = valid_logits[:, [int(no_id), int(yes_id)]]
                binary_targets = valid_targets.eq(int(yes_id)).long()
                token_loss = torch_f.cross_entropy(
                    binary_logits,
                    binary_targets,
                    reduction="none",
                )
            else:
                token_loss = torch_f.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="none",
                )
        else:
            token_loss = torch_f.cross_entropy(
                valid_logits,
                valid_targets,
                reduction="none",
            )
        sample_ids = valid_mask.nonzero(as_tuple=False)[:, 0]
        sample_assign = torch_f.one_hot(
            sample_ids,
            num_classes=int(shift_labels.size(0)),
        ).to(dtype=token_loss.dtype, device=token_loss.device)
        per_sample = sample_assign.transpose(0, 1).matmul(token_loss)
        token_counts = sample_assign.sum(dim=0).clamp_min(1.0)
        per_sample = per_sample / token_counts
        if per_sample.ndim != 1:
            per_sample = per_sample.view(-1)

        if per_sample.size(0) != int(shift_labels.size(0)):
            pad = torch.zeros(
                int(shift_labels.size(0)) - int(per_sample.size(0)),
                device=per_sample.device,
                dtype=per_sample.dtype,
            )
            per_sample = torch.cat([per_sample, pad], dim=0)

        if weights is not None:
            w = weights.to(token_loss.device).float().view(-1)
            loss = (per_sample * w).sum() / w.sum().clamp_min(1e-8)
        else:
            loss = per_sample.mean()
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    global BINARY_CLASS_TOKEN_IDS
    seed_everything(SEED)
    source_11 = resolve_stage11_dataset_run()
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
    ds_meta: dict[str, Any] = {}
    if ENFORCE_STAGE09_GATE:
        ds_meta_path = source_11 / "run_meta.json"
        if not ds_meta_path.exists():
            raise FileNotFoundError(
                f"dataset run_meta.json missing: {ds_meta_path}. "
                "Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
            )
        ds_meta = json.loads(ds_meta_path.read_text(encoding="utf-8"))
        gate = ds_meta.get("stage09_gate_result", {})
        if not isinstance(gate, dict) or not gate:
            raise RuntimeError(
                "stage09 gate evidence missing in dataset run_meta. "
                "Rebuild 11_1 with gate enabled or set QLORA_ENFORCE_STAGE09_GATE=false."
            )
        missing = [str(int(b)) for b in buckets if str(int(b)) not in gate]
        if missing:
            raise RuntimeError(
                f"stage09 gate evidence missing for buckets={missing} in {ds_meta_path}. "
                "Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
            )
    if int(MAX_SEQ_LEN) > 768:
        print(
            f"[WARN] QLORA_MAX_SEQ_LEN={int(MAX_SEQ_LEN)} is high for local training; "
            "prefer <=768 for stable iteration."
        )
    if int(MAX_TRAIN_ROWS) == 0:
        print("[WARN] QLORA_MAX_TRAIN_ROWS=0 uses uncapped train rows and may run much longer.")
    if int(MAX_EVAL_ROWS) == 0:
        print("[WARN] QLORA_MAX_EVAL_ROWS=0 uses uncapped eval rows and may slow periodic eval.")
    if TRAIN_SOURCE not in {"default", "rich_sft", "pairwise_pool"}:
        raise RuntimeError(
            "unsupported QLORA_TRAIN_SOURCE="
            f"{TRAIN_SOURCE}. expected one of: default, rich_sft, pairwise_pool"
        )
    if NEG_REBALANCE_MODE not in {"random", "off", "none", "disabled", "tier_preserve", "tier-aware", "tier"}:
        raise RuntimeError(
            "unsupported QLORA_NEG_REBALANCE_MODE="
            f"{NEG_REBALANCE_MODE}. expected one of: random, off, tier_preserve"
        )
    if LOSS_MODE not in {"full_ce", "binary_yesno"}:
        raise RuntimeError(
            "unsupported QLORA_LOSS_MODE="
            f"{LOSS_MODE}. expected one of: full_ce, binary_yesno"
        )
    if TRAIN_SOURCE == "pairwise_pool" and int(MAX_TRAIN_ROWS) > 0:
        raise RuntimeError(
            "pairwise_pool + QLORA_MAX_TRAIN_ROWS breaks user-level candidate structure. "
            "Use QLORA_MAX_TRAIN_USERS instead."
        )
    if TRAIN_SOURCE == "pairwise_pool" and int(MAX_EVAL_ROWS) > 0:
        raise RuntimeError(
            "pairwise_pool + QLORA_MAX_EVAL_ROWS breaks eval-side user structure. "
            "Use QLORA_MAX_EVAL_USERS instead."
        )
    if TRAIN_SOURCE == "pairwise_pool" and NEG_REBALANCE_MODE == "random":
        raise RuntimeError(
            "pairwise_pool + random negative rebalance destroys the upstream tier structure. "
            "Use QLORA_NEG_REBALANCE_MODE=off or tier_preserve."
        )
    if ENFORCE_REQUIRED_BASE_MODEL and BASE_MODEL != REQUIRED_BASE_MODEL:
        raise RuntimeError(
            "base model mismatch: "
            f"QLORA_BASE_MODEL={BASE_MODEL}, expected={REQUIRED_BASE_MODEL}. "
            "If you intentionally change the base model, set "
            "QLORA_ENFORCE_REQUIRED_BASE_MODEL=false."
        )
    print(f"[CONFIG] base_model={BASE_MODEL}")

    if QLORA_RESUME_RUN_DIR:
        out_dir = Path(QLORA_RESUME_RUN_DIR)
        if not out_dir.exists():
            raise FileNotFoundError(f"QLORA_RESUME_RUN_DIR not found: {out_dir}")
        run_id = out_dir.name.split("_", 1)[0]
        print(f"[CONFIG] Resuming run from: {out_dir}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
        out_dir.mkdir(parents=True, exist_ok=True)

    train_files, eval_files, summary = collect_json_files(source_11, buckets, TRAIN_SOURCE)
    data_files: dict[str, list[str]] = {"train": train_files}
    if eval_files:
        data_files["eval"] = eval_files

    ds: DatasetDict = load_dataset("json", data_files=data_files)  # type: ignore[assignment]
    before_stats = binary_label_stats(ds["train"])
    ds["train"] = maybe_cap_users(ds["train"], MAX_TRAIN_USERS, SEED, USER_CAP_RANDOMIZED)
    after_user_cap_stats = binary_label_stats(ds["train"])
    ds["train"] = rebalance_negatives(ds["train"], NEG_REBALANCE_MODE, MAX_NEG_POS_RATIO, SEED)
    after_rebalance_stats = binary_label_stats(ds["train"])
    ds["train"] = maybe_cap_rows(ds["train"], MAX_TRAIN_ROWS, SEED)
    after_cap_stats = binary_label_stats(ds["train"])
    if "eval" in ds:
        ds["eval"] = maybe_cap_users(ds["eval"], MAX_EVAL_USERS, SEED + 17, USER_CAP_RANDOMIZED)
        ds["eval"] = maybe_cap_rows(ds["eval"], MAX_EVAL_ROWS, SEED + 1)

    def _fmt(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row.get("prompt", "")
        target = row.get("target_text", "")
        try:
            w = float(row.get("sample_weight", 1.0))
        except Exception:
            w = 1.0
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        return {"text": _build_text(prompt, target), "sample_weight": float(w)}

    format_map_kwargs: dict[str, Any] = {}
    if FORMAT_MAP_NUM_PROC > 1:
        format_map_kwargs["num_proc"] = FORMAT_MAP_NUM_PROC
    ds = ds.map(
        _fmt,
        remove_columns=[c for c in ds["train"].column_names if c not in ("text", "sample_weight")],
        **format_map_kwargs,
    )
    print(
        "[DATA] train_stats "
        f"source={TRAIN_SOURCE} "
        f"before={before_stats} "
        f"after_user_cap={after_user_cap_stats} "
        f"after_rebalance={after_rebalance_stats} "
        f"after_cap={after_cap_stats} "
        f"neg_rebalance_mode={NEG_REBALANCE_MODE}"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"
    if LOSS_MODE == "binary_yesno":
        BINARY_CLASS_TOKEN_IDS = resolve_binary_class_token_ids(tokenizer)
        if BINARY_CLASS_TOKEN_IDS is None:
            print("[WARN] binary_yesno loss requested but YES/NO did not resolve to single-token ids; fallback to full_ce")
        else:
            yes_id, no_id = BINARY_CLASS_TOKEN_IDS
            print(f"[CONFIG] binary_yesno_token_ids yes={yes_id} no={no_id}")
    else:
        BINARY_CLASS_TOKEN_IDS = None

    def _tok(row: dict[str, Any]) -> dict[str, Any]:
        text = row["text"]
        prompt = text.rsplit(" ", 1)[0].strip()
        full = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)
        prompt_tok = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN)
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt_tok["input_ids"]), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        full["labels"] = labels
        full["sample_weight"] = float(row.get("sample_weight", 1.0))
        return full

    token_map_kwargs: dict[str, Any] = {}
    if TOKENIZE_MAP_NUM_PROC > 1:
        token_map_kwargs["num_proc"] = TOKENIZE_MAP_NUM_PROC
    ds_tok = ds.map(_tok, remove_columns=ds["train"].column_names, **token_map_kwargs)
    print(
        "[CONFIG] pipeline_workers "
        f"format_map={max(1, FORMAT_MAP_NUM_PROC)} "
        f"tokenize_map={max(1, TOKENIZE_MAP_NUM_PROC)} "
        f"dataloader={max(0, DATALOADER_NUM_WORKERS)} "
        f"pad_to_multiple_of={PAD_TO_MULTIPLE_OF}"
    )

    token_audit: dict[str, Any] = {"enabled": False}
    token_audit_path = out_dir / "token_length_audit.json"
    if TOKEN_AUDIT_ENABLED:
        token_audit = run_token_length_audit(
            ds_text=ds,
            ds_tok=ds_tok,
            tokenizer=tokenizer,
            max_seq_len=int(MAX_SEQ_LEN),
            max_rows=int(TOKEN_AUDIT_MAX_ROWS),
            batch_size=int(TOKEN_AUDIT_BATCH_SIZE),
            seed=int(SEED),
        )
        token_audit_path.write_text(json.dumps(token_audit, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[AUDIT] token_length: {token_audit_path}")
        for split_name, split_obj in token_audit.get("splits", {}).items():
            raw_obj = split_obj.get("raw_length", {})
            sup_obj = split_obj.get("supervision", {})
            print(
                f"[AUDIT] split={split_name} "
                f"clip_rate={float(raw_obj.get('clip_rate', 0.0)):.4f} "
                f"p95={float(raw_obj.get('full_len', {}).get('p95', 0.0)):.1f} "
                f"max={int(raw_obj.get('full_len', {}).get('max', 0))} "
                f"zero_supervision_rate={float(sup_obj.get('zero_supervision_rate', 0.0)):.4f}"
            )

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
    print(
        f"[CONFIG] has_cuda={has_cuda} use_4bit={USE_4BIT} "
        f"bnb_config={bool(bnb_config is not None)} compute_dtype={compute_dtype} "
        f"attn_implementation={(ATTN_IMPLEMENTATION or '<default>')}"
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
    }
    if ATTN_IMPLEMENTATION and ATTN_IMPLEMENTATION not in {"auto", "default"}:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    if has_cuda:
        model_kwargs["device_map"] = DEVICE_MAP
        max_memory: dict[Any, str] = {}
        if MAX_MEMORY_CUDA:
            max_memory[0] = str(MAX_MEMORY_CUDA)
        if MAX_MEMORY_CPU:
            max_memory["cpu"] = str(MAX_MEMORY_CPU)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if OFFLOAD_FOLDER:
            offload_dir = Path(OFFLOAD_FOLDER).expanduser()
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = offload_dir.as_posix()
        if OFFLOAD_STATE_DICT:
            model_kwargs["offload_state_dict"] = True
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype

    if DISABLE_ALLOC_WARMUP and hasattr(_tf_modeling_utils, "caching_allocator_warmup"):
        _tf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # type: ignore[assignment]
        print("[CONFIG] disable_transformers_alloc_warmup=true")
    if DISABLE_PARALLEL_LOADING:
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(max(1, int(PARALLEL_LOADING_WORKERS)))

    # Qwen3.5 linear-attention kernels expect non-float32 mamba_ssm_dtype.
    try:
        model_cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
        model_type = str(getattr(model_cfg, "model_type", "")).strip().lower()
        if is_qwen35_model_type(model_type):
            target_mamba_dtype = str(QWEN35_MAMBA_SSM_DTYPE or "auto").strip().lower()
            if target_mamba_dtype in {"", "auto"} and compute_dtype == torch.bfloat16:
                target_mamba_dtype = "bfloat16"
            if target_mamba_dtype not in {"", "auto"}:
                txt_cfg = getattr(model_cfg, "text_config", None)
                if isinstance(txt_cfg, dict):
                    txt_cfg["mamba_ssm_dtype"] = target_mamba_dtype
                elif txt_cfg is not None:
                    setattr(txt_cfg, "mamba_ssm_dtype", target_mamba_dtype)
                print(f"[CONFIG] qwen3.5 mamba_ssm_dtype={target_mamba_dtype}")
                model_kwargs["config"] = model_cfg
    except Exception as cfg_exc:
        short = str(cfg_exc).splitlines()[0][:180]
        print(f"[WARN] qwen3.5 dtype config override skipped: {short}")

    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if "1455" in msg or "paging file" in msg:
            raise RuntimeError(
                "model load failed with Windows os error 1455 (virtual memory/pagefile too small). "
                "Increase pagefile size (recommend >= 40GB), reboot to apply, then rerun stage11_2."
            ) from exc
        raise
    if bnb_config is not None:
        loaded_model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
        print(
            f"[CONFIG] loaded_model_type={loaded_model_type} "
            f"safe_kbit={QWEN35_SAFE_KBIT_PREP} force_float_params_bf16={QWEN35_FORCE_FLOAT_PARAMS_BF16}"
        )
        if is_qwen35_model_type(loaded_model_type) and QWEN35_SAFE_KBIT_PREP:
            model = prepare_model_for_kbit_training_qwen35_safe(
                model,
                enable_input_grads=bool(GRADIENT_CHECKPOINTING),
            )
            print("[CONFIG] qwen3.5 safe_kbit_prep=true (skip PEFT fp32 upcast)")
        else:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    loaded_model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
    if (
        is_qwen35_model_type(loaded_model_type)
        and has_cuda
        and compute_dtype == torch.bfloat16
        and QWEN35_FORCE_FLOAT_PARAMS_BF16
    ):
        n_recast = recast_float_params(model, torch.bfloat16)
        print(f"[CONFIG] qwen3.5 recast_float_params_to_bf16={n_recast}")
    if torch.cuda.is_available() and CLEAR_CUDA_CACHE_BEFORE_LORA:
        # Drop transient load-time allocator reservations before adapter injection.
        gc.collect()
        torch.cuda.empty_cache()
        print("[TRAIN] cleared_cuda_cache_before_lora=true")
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = [x.strip() for x in LORA_TARGET_MODULES.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    if CAST_TRAINABLE_PARAMS_TO_COMPUTE and has_cuda and compute_dtype in (torch.float16, torch.bfloat16):
        n_cast_trainable = recast_trainable_params(model, compute_dtype)
        print(f"[CONFIG] recast_trainable_params_to_compute={n_cast_trainable}")
        gc.collect()
        torch.cuda.empty_cache()
        print("[TRAIN] cleared_cuda_cache_after_lora_cast=true")
    model.print_trainable_parameters()

    has_eval = "eval" in ds_tok and len(ds_tok["eval"]) > 0
    if CHECKPOINT_AUDIT_ENABLED and int(SAVE_TOTAL_LIMIT) < 3:
        print(
            "[WARN] checkpoint audit enabled with SAVE_TOTAL_LIMIT<3; "
            "increase QLORA_SAVE_TOTAL_LIMIT if you want more rank-side checkpoint coverage."
        )
    args = TrainingArguments(
        output_dir=(out_dir / "trainer_output").as_posix(),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, EVAL_BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS if has_eval else None,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=bool(has_cuda and compute_dtype == torch.bfloat16),
        fp16=bool(has_cuda and compute_dtype == torch.float16 and not MANUAL_FP16_AUTOCAST),
        report_to=[],
        dataloader_num_workers=max(0, DATALOADER_NUM_WORKERS),
        remove_unused_columns=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_strategy="steps" if has_eval else "no",
        load_best_model_at_end=False,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": ds_tok["train"],
        "eval_dataset": (ds_tok["eval"] if has_eval else None),
        "data_collator": build_weighted_causal_lm_collator(tokenizer),
    }
    # transformers>=5 removed `tokenizer` from Trainer.__init__ in favor of
    # `processing_class`; keep compatibility with both APIs.
    try:
        trainer = WeightedCausalLmTrainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'tokenizer'" not in str(exc):
            raise
        trainer = WeightedCausalLmTrainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    if torch.cuda.is_available():
        # Drop transient allocations from dataset audit/model setup before step 0.
        gc.collect()
        torch.cuda.empty_cache()
        print("[TRAIN] cleared_cuda_cache_before_train=true")
    resume_checkpoint = None
    if QLORA_RESUME_RUN_DIR:
        trainer_out_dir = out_dir / "trainer_output"
        if trainer_out_dir.exists():
            cp_dirs = [d for d in trainer_out_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            if cp_dirs:
                cp_dirs.sort(key=lambda d: int(d.name.split("-")[1]))
                resume_checkpoint = cp_dirs[-1].as_posix()
                print(f"[TRAIN] Found checkpoint to resume from: {resume_checkpoint}")
            else:
                print(f"[WARN] QLORA_RESUME_RUN_DIR provided but no checkpoints found in {trainer_out_dir}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_result = trainer.evaluate() if has_eval else {}

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir.as_posix())
    tokenizer.save_pretrained(adapter_dir.as_posix())
    checkpoint_audit_manifest = build_checkpoint_audit_manifest(out_dir, source_11, ds_meta, buckets)
    checkpoint_audit_manifest_path = out_dir / "checkpoint_audit_manifest.json"
    checkpoint_audit_manifest_path.write_text(
        json.dumps(checkpoint_audit_manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage11_dataset_run": str(source_11),
        "source_stage09_run": str(ds_meta.get("source_stage09_run", "")).strip(),
        "resumed_from_dir": QLORA_RESUME_RUN_DIR if QLORA_RESUME_RUN_DIR else None,
        "resumed_checkpoint": resume_checkpoint if QLORA_RESUME_RUN_DIR else None,
        "buckets": buckets,
        "base_model": BASE_MODEL,
        "use_4bit": bool(bnb_config is not None),
        "dtype": str(compute_dtype),
        "enforce_stage09_gate": bool(ENFORCE_STAGE09_GATE),
        "train_rows": int(len(ds_tok["train"])),
        "eval_rows": int(len(ds_tok["eval"])) if has_eval else 0,
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
        "eval_loss": float(eval_result.get("eval_loss", 0.0)) if has_eval else None,
        "sample_weight_used": True,
        "token_audit_enabled": bool(TOKEN_AUDIT_ENABLED),
        "token_audit_file": (str(token_audit_path) if TOKEN_AUDIT_ENABLED else ""),
        "token_audit_summary": token_audit,
        "checkpoint_audit_manifest_file": str(checkpoint_audit_manifest_path),
        "checkpoint_audit_manifest": checkpoint_audit_manifest,
        "config": {
            "train_source": str(TRAIN_SOURCE),
            "neg_rebalance_mode": str(NEG_REBALANCE_MODE),
            "loss_mode": str(LOSS_MODE),
            "max_seq_len": int(MAX_SEQ_LEN),
            "max_train_users": int(MAX_TRAIN_USERS),
            "max_eval_users": int(MAX_EVAL_USERS),
            "user_cap_randomized": bool(USER_CAP_RANDOMIZED),
            "epochs": float(EPOCHS),
            "lr": float(LR),
            "batch_size": int(BATCH_SIZE),
            "eval_batch_size": int(EVAL_BATCH_SIZE),
            "grad_acc": int(GRAD_ACC),
            "eval_steps": int(EVAL_STEPS),
            "save_steps": int(SAVE_STEPS),
            "save_total_limit": int(SAVE_TOTAL_LIMIT),
            "lora_r": int(LORA_R),
            "lora_alpha": int(LORA_ALPHA),
            "lora_dropout": float(LORA_DROPOUT),
            "target_modules": target_modules,
            "attn_implementation": str(ATTN_IMPLEMENTATION or ""),
        },
        "data_files": summary,
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    pointer_path = write_latest_run_pointer(
        "stage11_2_qlora_train",
        out_dir,
        extra={
            "run_tag": RUN_TAG,
            "source_stage11_dataset_run": str(source_11),
            "source_stage09_run": str(ds_meta.get("source_stage09_run", "")).strip(),
            "base_model": BASE_MODEL,
            "adapter_dir": str(adapter_dir),
        },
    )
    print(f"[DONE] qlora adapter saved to: {adapter_dir}")
    print(f"[DONE] run_meta: {out_dir / 'run_meta.json'}")
    print(f"[DONE] checkpoint_audit_manifest: {checkpoint_audit_manifest_path}")
    print(f"[DONE] updated latest pointer: {pointer_path}")


if __name__ == "__main__":
    main()



