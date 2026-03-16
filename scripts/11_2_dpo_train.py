"""Stage 11-2 DPO pairwise trainer.

Drop-in alternative to ``11_2_qlora_train.py``:
it builds per-user preference pairs and trains with TRL ``DPOTrainer``.

The output LoRA adapter remains compatible with
``11_3_qlora_sidecar_eval.py`` (which scores P(YES) for prompts).
"""
from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from itertools import product as itertools_product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from pipeline.project_paths import env_or_project_path
from pipeline.stage11_pairwise import (
    build_dpo_pairs,
    build_rich_sft_dpo_pairs,
    pair_records_for_training,
)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from trl import DPOConfig, DPOTrainer
except ImportError:
    raise ImportError(
        "trl is required for DPO training.  Install it with:\n"
        "    pip install trl>=0.9\n"
        "Then re-run this script."
    )


# ------------------------------- config -------------------------------
RUN_TAG = os.getenv("RUN_TAG", "stage11_2_dpo_train").strip() or "stage11_2_dpo_train"
QLORA_RESUME_RUN_DIR = os.getenv("QLORA_RESUME_RUN_DIR", "").strip()
QLORA_SFT_ADAPTER_DIR = os.getenv("QLORA_SFT_ADAPTER_DIR", "").strip()
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
LORA_R = int(os.getenv("QLORA_LORA_R", "16").strip() or 16)
LORA_ALPHA = int(os.getenv("QLORA_LORA_ALPHA", "32").strip() or 32)
LORA_DROPOUT = float(os.getenv("QLORA_LORA_DROPOUT", "0.05").strip() or 0.05)
LORA_TARGET_MODULES = os.getenv(
    "QLORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
).strip()

MAX_SEQ_LEN = int(os.getenv("QLORA_MAX_SEQ_LEN", "768").strip() or 768)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_PAD_TO_MULTIPLE_OF", "0").strip() or 0)
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "false").strip().lower() == "true"
QWEN35_SAFE_KBIT_PREP = os.getenv("QLORA_QWEN35_SAFE_KBIT_PREP", "true").strip().lower() == "true"
QWEN35_FORCE_FLOAT_PARAMS_BF16 = os.getenv("QLORA_QWEN35_FORCE_FLOAT_PARAMS_BF16", "true").strip().lower() == "true"
QWEN35_MAMBA_SSM_DTYPE = os.getenv("QLORA_QWEN35_MAMBA_SSM_DTYPE", "auto").strip()

EPOCHS = float(os.getenv("QLORA_EPOCHS", "1.0").strip() or 1.0)
LR = float(os.getenv("QLORA_LR", "5e-5").strip() or 5e-5)
WEIGHT_DECAY = float(os.getenv("QLORA_WEIGHT_DECAY", "0.01").strip() or 0.01)
WARMUP_RATIO = float(os.getenv("QLORA_WARMUP_RATIO", "0.05").strip() or 0.05)
BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "1").strip() or 1)
GRAD_ACC = int(os.getenv("QLORA_GRAD_ACC", "8").strip() or 8)
EVAL_STEPS = int(os.getenv("QLORA_EVAL_STEPS", "1000").strip() or 1000)
SAVE_STEPS = int(os.getenv("QLORA_SAVE_STEPS", "1000").strip() or 1000)
SAVE_TOTAL_LIMIT = int(os.getenv("QLORA_SAVE_TOTAL_LIMIT", "2").strip() or 2)
LOGGING_STEPS = int(os.getenv("QLORA_LOGGING_STEPS", "10").strip() or 10)
GRADIENT_CHECKPOINTING = os.getenv("QLORA_GRADIENT_CHECKPOINTING", "true").strip().lower() == "true"

# DPO-specific
DPO_BETA = float(os.getenv("QLORA_DPO_BETA", "0.1").strip() or 0.1)
DPO_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_MAX_PAIRS", "8").strip() or 8)
DPO_TRUE_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_TRUE_MAX_PAIRS", "2").strip() or 2)
DPO_VALID_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_VALID_MAX_PAIRS", "1").strip() or 1)
DPO_HIST_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_HIST_MAX_PAIRS", "1").strip() or 1)
DPO_ALLOW_MID_NEG = os.getenv("QLORA_DPO_ALLOW_MID_NEG", "true").strip().lower() == "true"
DPO_LOSS_TYPE = os.getenv("QLORA_DPO_LOSS_TYPE", "sigmoid").strip() or "sigmoid"
DPO_MAX_PROMPT_LENGTH = int(os.getenv("QLORA_DPO_MAX_PROMPT_LENGTH", "512").strip() or 512)
DPO_MAX_TARGET_LENGTH = int(os.getenv("QLORA_DPO_MAX_TARGET_LENGTH", "16").strip() or 16)
DPO_PREFER_EASY_NEG = os.getenv("QLORA_DPO_PREFER_EASY_NEG", "true").strip().lower() == "true"
DPO_FILTER_INVERTED = os.getenv("QLORA_DPO_FILTER_INVERTED", "false").strip().lower() == "true"
DPO_STRIP_RANK_FEATURES = os.getenv("QLORA_DPO_STRIP_RANK_FEATURES", "false").strip().lower() == "true"
DPO_PRETOKENIZE = os.getenv("QLORA_DPO_PRETOKENIZE", "true").strip().lower() == "true"
PAIRWISE_SOURCE_MODE = os.getenv("QLORA_PAIRWISE_SOURCE_MODE", "auto").strip().lower() or "auto"

_RANK_FEATURE_RE = re.compile(
    r"; (?:candidate_sources|user_segment|als_rank|cluster_rank|profile_rank|popular_rank): [^;]*"
)


def is_qwen35_model_type(model_type: str) -> bool:
    mt = str(model_type or "").strip().lower()
    return mt.startswith("qwen3_5")


def prepare_model_for_kbit_training_qwen35_safe(model: Any, enable_input_grads: bool) -> Any:
    # Qwen3.5 linear-attention kernels are sensitive to float32 activations
    # that PEFT's generic helper may introduce via blanket upcasts.
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


def strip_rank_features(prompt: str) -> str:
    """Remove ranking-position features from a pre-built prompt string."""
    return _RANK_FEATURE_RE.sub("", prompt)


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def build_pretokenized_preference_records(
    pairs: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    *,
    dataset_name: str,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    records: list[dict[str, list[int]]] = []
    raw_prompt_lens: list[int] = []
    prompt_lens: list[int] = []
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    extra_shared_prefix_tokens: list[int] = []
    identical_pairs = 0

    for pair in pairs:
        prompt_text = str(pair.get("prompt", ""))
        chosen_text = str(pair.get("chosen", ""))
        rejected_text = str(pair.get("rejected", ""))

        raw_prompt_ids = tokenizer(text=prompt_text)["input_ids"]
        prompt_chosen_ids = tokenizer(text=prompt_text + chosen_text)["input_ids"]
        prompt_rejected_ids = tokenizer(text=prompt_text + rejected_text)["input_ids"]
        shared_prefix_len = _longest_common_prefix_len(prompt_chosen_ids, prompt_rejected_ids)

        prompt_ids = prompt_chosen_ids[:shared_prefix_len]
        chosen_ids = prompt_chosen_ids[shared_prefix_len:]
        rejected_ids = prompt_rejected_ids[shared_prefix_len:]

        if eos_token_id is not None:
            if not chosen_ids or chosen_ids[-1] != eos_token_id:
                chosen_ids = chosen_ids + [int(eos_token_id)]
            if not rejected_ids or rejected_ids[-1] != eos_token_id:
                rejected_ids = rejected_ids + [int(eos_token_id)]

        if prompt_chosen_ids == prompt_rejected_ids:
            identical_pairs += 1

        records.append(
            {
                "prompt_ids": [int(x) for x in prompt_ids],
                "chosen_ids": [int(x) for x in chosen_ids],
                "rejected_ids": [int(x) for x in rejected_ids],
            }
        )
        raw_prompt_lens.append(len(raw_prompt_ids))
        prompt_lens.append(len(prompt_ids))
        chosen_lens.append(len(chosen_ids))
        rejected_lens.append(len(rejected_ids))
        extra_shared_prefix_tokens.append(max(0, len(prompt_ids) - len(raw_prompt_ids)))

    def _summary(values: list[int]) -> dict[str, int]:
        if not values:
            return {"count": 0, "min": 0, "p50": 0, "p95": 0, "max": 0}
        xs = sorted(int(v) for v in values)
        n = len(xs)

        def _pick(q: float) -> int:
            idx = max(0, min(n - 1, int(round((n - 1) * q))))
            return int(xs[idx])

        return {
            "count": int(n),
            "min": int(xs[0]),
            "p50": _pick(0.50),
            "p95": _pick(0.95),
            "max": int(xs[-1]),
        }

    audit = {
        "dataset_name": dataset_name,
        "records": int(len(records)),
        "identical_full_pair_count": int(identical_pairs),
        "raw_prompt_token_len": _summary(raw_prompt_lens),
        "prompt_token_len": _summary(prompt_lens),
        "chosen_token_len": _summary(chosen_lens),
        "rejected_token_len": _summary(rejected_lens),
        "extra_shared_prefix_tokens": _summary(extra_shared_prefix_tokens),
    }
    return records, audit


class StablePrefixDPOTrainer(DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        first_example = next(iter(dataset))
        if isinstance(first_example, dict) and {
            "prompt_ids",
            "chosen_ids",
            "rejected_ids",
        }.issubset(first_example.keys()):
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)


# ------------------------------- helpers ------------------------------
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
    return pick_latest_run(INPUT_11_ROOT, INPUT_11_SUFFIX)


def collect_json_files(
    source_11: Path,
    buckets: list[int],
    *,
    train_dir_name: str = "train_json",
    eval_dir_name: str = "eval_json",
) -> tuple[list[str], list[str], dict[str, Any]]:
    train_files: list[str] = []
    eval_files: list[str] = []
    summary: dict[str, Any] = {
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
            {"bucket": int(b), "train_files": len(b_train), "eval_files": len(b_eval), "bucket_dir": str(bdir)}
        )
    if not train_files:
        raise RuntimeError(f"no train json files found under {source_11} for buckets={buckets}")
    return train_files, eval_files, summary


def resolve_pairwise_source_mode(source_11: Path, buckets: list[int]) -> str:
    valid_modes = {"auto", "pointwise", "rich_sft"}
    if PAIRWISE_SOURCE_MODE not in valid_modes:
        raise ValueError(
            f"unsupported QLORA_PAIRWISE_SOURCE_MODE={PAIRWISE_SOURCE_MODE}; "
            f"expected one of {sorted(valid_modes)}"
        )
    if PAIRWISE_SOURCE_MODE in {"pointwise", "rich_sft"}:
        return PAIRWISE_SOURCE_MODE
    for bucket in buckets:
        rich_dir = source_11 / f"bucket_{bucket}" / "rich_sft_train_json"
        if rich_dir.exists() and any(rich_dir.glob("*.json")):
            return "rich_sft"
    return "pointwise"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- main ---
def main() -> None:
    seed_everything(SEED)
    source_11 = resolve_stage11_dataset_run()
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]

    # Stage-09 gate enforcement (same as 11_2)
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
                f"stage09 gate evidence missing for buckets={missing}. "
                "Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
            )

    if ENFORCE_REQUIRED_BASE_MODEL and BASE_MODEL != REQUIRED_BASE_MODEL:
        raise RuntimeError(
            f"base model mismatch: QLORA_BASE_MODEL={BASE_MODEL}, "
            f"expected={REQUIRED_BASE_MODEL}. "
            "Set QLORA_ENFORCE_REQUIRED_BASE_MODEL=false to override."
        )
    print(f"[CONFIG] base_model={BASE_MODEL}")
    print(f"[CONFIG] DPO beta={DPO_BETA}, loss_type={DPO_LOSS_TYPE}, max_pairs_per_user={DPO_MAX_PAIRS_PER_USER}")
    print(f"[CONFIG] LR={LR}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, grad_acc={GRAD_ACC}")
    print(f"[CONFIG] max_seq_len={MAX_SEQ_LEN}, pad_to_multiple_of={PAD_TO_MULTIPLE_OF}")
    print(f"[CONFIG] dpo_pretokenize={DPO_PRETOKENIZE}")
    source_mode = resolve_pairwise_source_mode(source_11, buckets)
    print(f"[CONFIG] pairwise_source_mode={source_mode}")

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

    # --- Load raw JSON data ---
    if source_mode == "rich_sft":
        train_files, eval_files, file_summary = collect_json_files(
            source_11,
            buckets,
            train_dir_name="rich_sft_train_json",
            eval_dir_name="rich_sft_eval_json",
        )
    else:
        train_files, eval_files, file_summary = collect_json_files(source_11, buckets)
    raw_ds: DatasetDict = load_dataset("json", data_files={"train": train_files})  # type: ignore
    raw_train = raw_ds["train"]

    print(f"[DATA] raw_train rows: {len(raw_train)}")

    # Count labels
    labels = np.array(raw_train["label"], dtype=np.int32)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"[DATA] pos={n_pos}, neg={n_neg}, ratio=1:{n_neg / max(1, n_pos):.1f}")

    # --- Build DPO preference pairs ---
    train_rows = [raw_train[i] for i in range(len(raw_train))]

    # Optionally strip ranking features from pre-built prompts
    if DPO_STRIP_RANK_FEATURES:
        for row in train_rows:
            row["prompt"] = strip_rank_features(str(row.get("prompt", "")))
        print("[DATA] stripped ranking features from prompts (DPO_STRIP_RANK_FEATURES=true)")

    if source_mode == "rich_sft":
        train_pairs_full, train_pair_audit = build_rich_sft_dpo_pairs(
            train_rows,
            DPO_MAX_PAIRS_PER_USER,
            SEED,
            true_max_pairs_per_user=DPO_TRUE_MAX_PAIRS_PER_USER,
            valid_max_pairs_per_user=DPO_VALID_MAX_PAIRS_PER_USER,
            hist_max_pairs_per_user=DPO_HIST_MAX_PAIRS_PER_USER,
            allow_mid_neg=DPO_ALLOW_MID_NEG,
        )
    else:
        train_pairs_full, train_pair_audit = build_dpo_pairs(
            train_rows,
            DPO_MAX_PAIRS_PER_USER,
            SEED,
            prefer_easy_neg=DPO_PREFER_EASY_NEG,
            filter_inverted=DPO_FILTER_INVERTED,
        )
    train_pairs = pair_records_for_training(train_pairs_full)

    if not train_pairs:
        raise RuntimeError(
            "No DPO pairs could be built. Check that training data has "
            "both positive and negative samples for at least some users."
        )

    # Build eval pairs if available
    eval_pairs: list[dict[str, str]] = []
    if eval_files:
        raw_eval_ds: DatasetDict = load_dataset("json", data_files={"eval": eval_files})  # type: ignore
        eval_rows_list = [raw_eval_ds["eval"][i] for i in range(len(raw_eval_ds["eval"]))]
        if source_mode == "rich_sft":
            eval_pairs_full, eval_pair_audit = build_rich_sft_dpo_pairs(
                eval_rows_list,
                max(2, DPO_MAX_PAIRS_PER_USER // 2),
                SEED + 1,
                true_max_pairs_per_user=max(1, min(DPO_TRUE_MAX_PAIRS_PER_USER, 1)),
                valid_max_pairs_per_user=max(0, min(DPO_VALID_MAX_PAIRS_PER_USER, 1)),
                hist_max_pairs_per_user=max(0, min(DPO_HIST_MAX_PAIRS_PER_USER, 1)),
                allow_mid_neg=DPO_ALLOW_MID_NEG,
            )
        else:
            eval_pairs_full, eval_pair_audit = build_dpo_pairs(
                eval_rows_list,
                max(2, DPO_MAX_PAIRS_PER_USER // 2),
                SEED + 1,
                prefer_easy_neg=DPO_PREFER_EASY_NEG,
                filter_inverted=DPO_FILTER_INVERTED,
            )
        eval_pairs = pair_records_for_training(eval_pairs_full)
        print(f"[DATA] eval_pairs: {len(eval_pairs)}")
    else:
        eval_pair_audit = {}

    # --- Model ---
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
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype

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
                "model load failed with Windows os error 1455. "
                "Increase pagefile size (>= 40GB), reboot, then rerun."
            ) from exc
        raise

    # --- Optionally load and merge SFT adapter as initialization ---
    if QLORA_SFT_ADAPTER_DIR:
        sft_path = Path(QLORA_SFT_ADAPTER_DIR)
        if not sft_path.exists():
            raise FileNotFoundError(f"QLORA_SFT_ADAPTER_DIR not found: {sft_path}")
        print(f"[MODEL] Loading SFT adapter from: {sft_path}")
        model = PeftModel.from_pretrained(model, sft_path.as_posix())
        print("[MODEL] Merging SFT adapter into base model ...")
        model = model.merge_and_unload()
        print("[MODEL] SFT adapter merged successfully.")

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
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
            )
    loaded_model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
    if (
        is_qwen35_model_type(loaded_model_type)
        and has_cuda
        and compute_dtype == torch.bfloat16
        and QWEN35_FORCE_FLOAT_PARAMS_BF16
    ):
        n_recast = recast_float_params(model, torch.bfloat16)
        print(f"[CONFIG] qwen3.5 recast_float_params_to_bf16={n_recast}")
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
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"

    if DPO_PRETOKENIZE:
        train_records, train_token_audit = build_pretokenized_preference_records(
            train_pairs,
            tokenizer,
            dataset_name="train",
        )
        eval_records, eval_token_audit = (
            build_pretokenized_preference_records(eval_pairs, tokenizer, dataset_name="eval")
            if eval_pairs
            else ([], {})
        )
        dpo_train_ds = Dataset.from_list(train_records)
        dpo_eval_ds = Dataset.from_list(eval_records) if eval_records else None
        print(
            "[DATA] pretokenized DPO datasets: "
            f"train={len(dpo_train_ds)}, eval={len(dpo_eval_ds) if dpo_eval_ds else 0}"
        )
    else:
        train_token_audit = {}
        eval_token_audit = {}
        dpo_train_ds = Dataset.from_list(train_pairs)
        dpo_eval_ds = Dataset.from_list(eval_pairs) if eval_pairs else None

    print(f"[DATA] DPO train_pairs: {len(dpo_train_ds)}")

    # --- DPO Training ---
    has_eval = dpo_eval_ds is not None and len(dpo_eval_ds) > 0
    trainer_out_dir = out_dir / "trainer_output"

    dpo_config = DPOConfig(
        output_dir=trainer_out_dir.as_posix(),
        beta=DPO_BETA,
        loss_type=DPO_LOSS_TYPE,
        max_length=MAX_SEQ_LEN,
        pad_to_multiple_of=(PAD_TO_MULTIPLE_OF if PAD_TO_MULTIPLE_OF > 0 else None),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS if has_eval else None,
        save_steps=SAVE_STEPS,
        save_total_limit=max(1, int(SAVE_TOTAL_LIMIT)),
        bf16=bool(has_cuda and compute_dtype == torch.bfloat16),
        fp16=bool(has_cuda and compute_dtype == torch.float16),
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_strategy="steps" if has_eval else "no",
        load_best_model_at_end=False,
    )

    trainer = StablePrefixDPOTrainer(
        model=model,
        ref_model=None,   # with PEFT, TRL auto-uses frozen base as ref
        args=dpo_config,
        train_dataset=dpo_train_ds,
        eval_dataset=dpo_eval_ds,
        processing_class=tokenizer,
    )

    resume_checkpoint = None
    if QLORA_RESUME_RUN_DIR:
        if trainer_out_dir.exists():
            cp_dirs = [d for d in trainer_out_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            if cp_dirs:
                cp_dirs.sort(key=lambda d: int(d.name.split("-")[1]))
                resume_checkpoint = cp_dirs[-1].as_posix()
                print(f"[TRAIN] Found checkpoint to resume from: {resume_checkpoint}")
            else:
                print(f"[WARN] QLORA_RESUME_RUN_DIR provided but no checkpoints found in {trainer_out_dir}")

    print(f"[TRAIN] Starting DPO training ...")
    print(f"[TRAIN] train_pairs={len(dpo_train_ds)}, eval_pairs={len(dpo_eval_ds) if dpo_eval_ds else 0}")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_result = trainer.evaluate() if has_eval else {}

    # --- Save adapter ---
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir.as_posix())
    tokenizer.save_pretrained(adapter_dir.as_posix())

    # --- Run metadata ---
    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "training_method": "DPO_pairwise",
        "source_stage11_dataset_run": str(source_11),
        "sft_adapter_dir": QLORA_SFT_ADAPTER_DIR if QLORA_SFT_ADAPTER_DIR else None,
        "resumed_from_dir": QLORA_RESUME_RUN_DIR if QLORA_RESUME_RUN_DIR else None,
        "resumed_checkpoint": resume_checkpoint if QLORA_RESUME_RUN_DIR else None,
        "buckets": buckets,
        "base_model": BASE_MODEL,
        "use_4bit": bool(bnb_config is not None),
        "dtype": str(compute_dtype),
        "enforce_stage09_gate": bool(ENFORCE_STAGE09_GATE),
        "train_pairs": len(dpo_train_ds),
        "eval_pairs": len(dpo_eval_ds) if dpo_eval_ds else 0,
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
        "eval_loss": float(eval_result.get("eval_loss", 0.0)) if eval_result else None,
        "dpo_config": {
            "beta": DPO_BETA,
            "loss_type": DPO_LOSS_TYPE,
            "pretokenize": bool(DPO_PRETOKENIZE),
            "pairwise_source_mode": source_mode,
            "max_pairs_per_user": DPO_MAX_PAIRS_PER_USER,
            "true_max_pairs_per_user": DPO_TRUE_MAX_PAIRS_PER_USER,
            "valid_max_pairs_per_user": DPO_VALID_MAX_PAIRS_PER_USER,
            "hist_max_pairs_per_user": DPO_HIST_MAX_PAIRS_PER_USER,
            "allow_mid_neg": bool(DPO_ALLOW_MID_NEG),
            "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
            "max_target_length": DPO_MAX_TARGET_LENGTH,
        },
        "pair_audit": {
            "train": train_pair_audit,
            "eval": eval_pair_audit,
        },
        "tokenization_audit": {
            "train": train_token_audit,
            "eval": eval_token_audit,
        },
        "config": {
            "max_seq_len": int(MAX_SEQ_LEN),
            "pad_to_multiple_of": int(PAD_TO_MULTIPLE_OF),
            "epochs": float(EPOCHS),
            "lr": float(LR),
            "save_total_limit": int(SAVE_TOTAL_LIMIT),
            "batch_size": int(BATCH_SIZE),
            "grad_acc": int(GRAD_ACC),
            "lora_r": int(LORA_R),
            "lora_alpha": int(LORA_ALPHA),
            "lora_dropout": float(LORA_DROPOUT),
            "target_modules": target_modules,
        },
        "data_files": file_summary,
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"[DONE] DPO adapter saved to: {adapter_dir}")
    print(f"[DONE] run_meta: {out_dir / 'run_meta.json'}")
    print(f"[DONE] evaluate with: python scripts/11_3_qlora_sidecar_eval.py")
    print(f"[DONE]   set INPUT_11_2_RUN_DIR={out_dir}")


if __name__ == "__main__":
    main()

