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
from collections import defaultdict
from datetime import datetime
from itertools import product as itertools_product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from pipeline.project_paths import env_or_project_path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import (
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
RUN_TAG = "stage11_2_dpo_train"
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
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "false").strip().lower() == "true"

EPOCHS = float(os.getenv("QLORA_EPOCHS", "1.0").strip() or 1.0)
LR = float(os.getenv("QLORA_LR", "5e-5").strip() or 5e-5)
WEIGHT_DECAY = float(os.getenv("QLORA_WEIGHT_DECAY", "0.01").strip() or 0.01)
WARMUP_RATIO = float(os.getenv("QLORA_WARMUP_RATIO", "0.05").strip() or 0.05)
BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "1").strip() or 1)
GRAD_ACC = int(os.getenv("QLORA_GRAD_ACC", "8").strip() or 8)
EVAL_STEPS = int(os.getenv("QLORA_EVAL_STEPS", "1000").strip() or 1000)
SAVE_STEPS = int(os.getenv("QLORA_SAVE_STEPS", "1000").strip() or 1000)
LOGGING_STEPS = int(os.getenv("QLORA_LOGGING_STEPS", "10").strip() or 10)
GRADIENT_CHECKPOINTING = os.getenv("QLORA_GRADIENT_CHECKPOINTING", "true").strip().lower() == "true"

# DPO-specific
DPO_BETA = float(os.getenv("QLORA_DPO_BETA", "0.1").strip() or 0.1)
DPO_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_MAX_PAIRS", "8").strip() or 8)
DPO_LOSS_TYPE = os.getenv("QLORA_DPO_LOSS_TYPE", "sigmoid").strip() or "sigmoid"
DPO_MAX_PROMPT_LENGTH = int(os.getenv("QLORA_DPO_MAX_PROMPT_LENGTH", "512").strip() or 512)
DPO_MAX_TARGET_LENGTH = int(os.getenv("QLORA_DPO_MAX_TARGET_LENGTH", "16").strip() or 16)
DPO_PREFER_EASY_NEG = os.getenv("QLORA_DPO_PREFER_EASY_NEG", "true").strip().lower() == "true"
DPO_FILTER_INVERTED = os.getenv("QLORA_DPO_FILTER_INVERTED", "false").strip().lower() == "true"
DPO_STRIP_RANK_FEATURES = os.getenv("QLORA_DPO_STRIP_RANK_FEATURES", "false").strip().lower() == "true"

_RANK_FEATURE_RE = re.compile(
    r"; (?:candidate_sources|user_segment|als_rank|cluster_rank|profile_rank|popular_rank): [^;]*"
)


def strip_rank_features(prompt: str) -> str:
    """Remove ranking-position features from a pre-built prompt string."""
    return _RANK_FEATURE_RE.sub("", prompt)


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


def collect_json_files(source_11: Path, buckets: list[int]) -> tuple[list[str], list[str], dict[str, Any]]:
    train_files: list[str] = []
    eval_files: list[str] = []
    summary: dict[str, Any] = {"buckets": []}
    for b in buckets:
        bdir = source_11 / f"bucket_{b}"
        train_dir = bdir / "train_json"
        eval_dir = bdir / "eval_json"
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------- DPO pair construction -----------------------
def _extract_prompt_body(full_prompt: str) -> str:
    """Extract user+item portion from the full binary prompt.

    The original prompt looks like:
        "You are a recommendation assistant. ... Answer only YES or NO.\n
         User: <user_text>\nCandidate: <item_text>\nAnswer:"

    We keep the *entire* prompt (including system instruction) because DPO
    needs the model to see the same prefix it was pre-trained / will be
    evaluated on.  The chosen / rejected responses are just " YES" / " NO".
    """
    return full_prompt.strip()


def build_dpo_pairs(
    rows: list[dict[str, Any]],
    max_pairs_per_user: int,
    seed: int,
    prefer_easy_neg: bool = True,
    filter_inverted: bool = False,
) -> list[dict[str, str]]:
    """Group rows by user_idx, cross-pair positives with negatives.

    For each user we pair every positive-item prompt (chosen=" YES")
    with sampled negative-item prompts (rejected=" YES") 鈥?the key
    insight is: the *prompt itself contains the item info*, so the
    "chosen" prompt describes a good item and the "rejected" prompt
    describes a bad item.  Both get " YES" as the response text so
    DPO learns to *prefer generating YES for good items*.

    When ``prefer_easy_neg=True``, prioritise easy/near negatives
    (where the score gap is more intuitive) over hard negatives.

    When ``filter_inverted=True``, exclude pairs where the negative
    item has a higher pre_score than the positive item.
    """
    rng = random.Random(seed)

    # Group by user 鈥?store full row for score access
    user_pos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    user_neg: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        uid = int(row.get("user_idx", -1))
        label = int(row.get("label", 0))
        prompt = str(row.get("prompt", "")).strip()
        if uid < 0 or not prompt:
            continue
        if label == 1:
            user_pos[uid].append(row)
        else:
            user_neg[uid].append(row)

    pairs: list[dict[str, str]] = []
    users_with_pairs = 0
    users_skipped = 0
    n_filtered_inverted = 0

    for uid in sorted(user_pos.keys()):
        pos_rows = user_pos[uid]
        neg_rows = user_neg.get(uid, [])
        if not neg_rows:
            users_skipped += 1
            continue

        users_with_pairs += 1

        # Optionally prioritise easy/near negatives
        if prefer_easy_neg:
            easy_near = [r for r in neg_rows if r.get("neg_tier") in ("easy", "near", "fill")]
            hard = [r for r in neg_rows if r.get("neg_tier") == "hard"]
            ordered_neg = easy_near + hard  # easy first, hard as fallback
        else:
            ordered_neg = list(neg_rows)
            rng.shuffle(ordered_neg)

        # Generate cross-pairs with priority ordering
        all_combos: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for p_row in pos_rows:
            for n_row in ordered_neg:
                all_combos.append((p_row, n_row))

        # If preferring easy neg, keep order; otherwise shuffle
        if not prefer_easy_neg:
            rng.shuffle(all_combos)

        selected = all_combos[:max_pairs_per_user]

        for pos_row, neg_row in selected:
            # Optionally filter pairs where neg scores higher
            if filter_inverted:
                pos_score = float(pos_row.get("pre_score", 0))
                neg_score = float(neg_row.get("pre_score", 0))
                if neg_score > pos_score:
                    n_filtered_inverted += 1
                    continue

            pos_prompt = str(pos_row.get("prompt", "")).strip()
            neg_prompt = str(neg_row.get("prompt", "")).strip()

            # Extract the common prompt part (everything before "Candidate:")
            # Both pos and neg should have the same user context
            if "Candidate:" in pos_prompt:
                common_prompt = pos_prompt.split("Candidate:")[0] + "Candidate:"
            else:
                common_prompt = ""

            pairs.append({
                "prompt": common_prompt,
                "chosen": pos_prompt + " YES",
                "rejected": neg_prompt + " YES",
            })

    rng.shuffle(pairs)
    print(
        f"[DPO-PAIRS] users_with_pairs={users_with_pairs} "
        f"users_skipped={users_skipped} "
        f"total_pairs={len(pairs)} "
        f"max_pairs_per_user={max_pairs_per_user} "
        f"prefer_easy_neg={prefer_easy_neg} "
        f"filtered_inverted={n_filtered_inverted}"
    )
    return pairs


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€ main 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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
    print(f"[CONFIG] DPO 尾={DPO_BETA}, loss_type={DPO_LOSS_TYPE}, max_pairs_per_user={DPO_MAX_PAIRS_PER_USER}")
    print(f"[CONFIG] LR={LR}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, grad_acc={GRAD_ACC}")

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

    # 鈹€鈹€鈹€鈹€ Load raw JSON data 鈹€鈹€鈹€鈹€
    train_files, eval_files, file_summary = collect_json_files(source_11, buckets)
    raw_ds: DatasetDict = load_dataset("json", data_files={"train": train_files})  # type: ignore
    raw_train = raw_ds["train"]

    print(f"[DATA] raw_train rows: {len(raw_train)}")

    # Count labels
    labels = np.array(raw_train["label"], dtype=np.int32)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"[DATA] pos={n_pos}, neg={n_neg}, ratio=1:{n_neg / max(1, n_pos):.1f}")

    # 鈹€鈹€鈹€鈹€ Build DPO preference pairs 鈹€鈹€鈹€鈹€
    train_rows = [raw_train[i] for i in range(len(raw_train))]

    # Optionally strip ranking features from pre-built prompts
    if DPO_STRIP_RANK_FEATURES:
        for row in train_rows:
            row["prompt"] = strip_rank_features(str(row.get("prompt", "")))
        print("[DATA] stripped ranking features from prompts (DPO_STRIP_RANK_FEATURES=true)")

    train_pairs = build_dpo_pairs(
        train_rows, DPO_MAX_PAIRS_PER_USER, SEED,
        prefer_easy_neg=DPO_PREFER_EASY_NEG,
        filter_inverted=DPO_FILTER_INVERTED,
    )

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
        eval_pairs = build_dpo_pairs(
            eval_rows_list, max(2, DPO_MAX_PAIRS_PER_USER // 2), SEED + 1,
            prefer_easy_neg=DPO_PREFER_EASY_NEG,
            filter_inverted=DPO_FILTER_INVERTED,
        )
        print(f"[DATA] eval_pairs: {len(eval_pairs)}")

    dpo_train_ds = Dataset.from_list(train_pairs)
    dpo_eval_ds = Dataset.from_list(eval_pairs) if eval_pairs else None

    print(f"[DATA] DPO train_pairs: {len(dpo_train_ds)}")

    # 鈹€鈹€鈹€鈹€ Model 鈹€鈹€鈹€鈹€
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
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if "1455" in msg or "paging file" in msg:
            raise RuntimeError(
                "model load failed with Windows os error 1455. "
                "Increase pagefile size (>= 40GB), reboot, then rerun."
            ) from exc
        raise

    # 鈹€鈹€ Optionally load & merge SFT adapter as initialisation 鈹€鈹€
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
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING
        )
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

    # 鈹€鈹€鈹€鈹€ DPO Training 鈹€鈹€鈹€鈹€
    has_eval = dpo_eval_ds is not None and len(dpo_eval_ds) > 0
    trainer_out_dir = out_dir / "trainer_output"

    dpo_config = DPOConfig(
        output_dir=trainer_out_dir.as_posix(),
        beta=DPO_BETA,
        loss_type=DPO_LOSS_TYPE,
        max_length=MAX_SEQ_LEN,
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
        save_total_limit=2,
        bf16=bool(has_cuda and compute_dtype == torch.bfloat16),
        fp16=bool(has_cuda and compute_dtype == torch.float16),
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        eval_strategy="steps" if has_eval else "no",
        load_best_model_at_end=False,
    )

    trainer = DPOTrainer(
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

    # 鈹€鈹€鈹€鈹€ Save adapter 鈹€鈹€鈹€鈹€
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir.as_posix())
    tokenizer.save_pretrained(adapter_dir.as_posix())

    # 鈹€鈹€鈹€鈹€ Run metadata 鈹€鈹€鈹€鈹€
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
            "max_pairs_per_user": DPO_MAX_PAIRS_PER_USER,
            "max_prompt_length": DPO_MAX_PROMPT_LENGTH,
            "max_target_length": DPO_MAX_TARGET_LENGTH,
        },
        "config": {
            "max_seq_len": int(MAX_SEQ_LEN),
            "epochs": float(EPOCHS),
            "lr": float(LR),
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

