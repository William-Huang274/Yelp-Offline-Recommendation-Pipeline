from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Keep transformers on torch path in mixed TensorFlow/Keras env.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


RUN_TAG = "stage11_2_qlora_train"
QLORA_RESUME_RUN_DIR = os.getenv("QLORA_RESUME_RUN_DIR", "").strip()
INPUT_11_RUN_DIR = os.getenv("INPUT_11_RUN_DIR", "").strip()
INPUT_11_ROOT = Path(r"D:/5006_BDA_project/data/output/11_qlora_data")
INPUT_11_SUFFIX = "_stage11_1_qlora_build_dataset"
OUTPUT_ROOT = Path(r"D:/5006_BDA_project/data/output/11_qlora_models")

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
MAX_TRAIN_ROWS = int(os.getenv("QLORA_MAX_TRAIN_ROWS", "0").strip() or 0)
MAX_EVAL_ROWS = int(os.getenv("QLORA_MAX_EVAL_ROWS", "0").strip() or 0)
MAX_NEG_POS_RATIO = float(os.getenv("QLORA_MAX_NEG_POS_RATIO", "4.0").strip() or 4.0)
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "true").strip().lower() == "true"

EPOCHS = float(os.getenv("QLORA_EPOCHS", "1.0").strip() or 1.0)
LR = float(os.getenv("QLORA_LR", "2e-4").strip() or 2e-4)
WEIGHT_DECAY = float(os.getenv("QLORA_WEIGHT_DECAY", "0.01").strip() or 0.01)
WARMUP_RATIO = float(os.getenv("QLORA_WARMUP_RATIO", "0.03").strip() or 0.03)
BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "1").strip() or 1)
GRAD_ACC = int(os.getenv("QLORA_GRAD_ACC", "16").strip() or 16)
EVAL_STEPS = int(os.getenv("QLORA_EVAL_STEPS", "100").strip() or 100)
SAVE_STEPS = int(os.getenv("QLORA_SAVE_STEPS", "100").strip() or 100)
LOGGING_STEPS = int(os.getenv("QLORA_LOGGING_STEPS", "20").strip() or 20)
GRADIENT_CHECKPOINTING = os.getenv("QLORA_GRADIENT_CHECKPOINTING", "true").strip().lower() == "true"

TOKEN_AUDIT_ENABLED = os.getenv("QLORA_TOKEN_AUDIT_ENABLED", "true").strip().lower() == "true"
TOKEN_AUDIT_MAX_ROWS = int(os.getenv("QLORA_TOKEN_AUDIT_MAX_ROWS", "4000").strip() or 4000)
TOKEN_AUDIT_BATCH_SIZE = int(os.getenv("QLORA_TOKEN_AUDIT_BATCH_SIZE", "256").strip() or 256)


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
            {
                "bucket": int(b),
                "train_files": len(b_train),
                "eval_files": len(b_eval),
                "bucket_dir": str(bdir),
            }
        )
    if not train_files:
        raise RuntimeError(f"no train json files found under {source_11} for buckets={buckets}")
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


def rebalance_negatives(ds: Any, max_neg_pos_ratio: float, seed: int) -> Any:
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


def binary_label_stats(ds: Any) -> dict[str, int]:
    cols = list(getattr(ds, "column_names", []))
    if "label" not in cols:
        return {"rows": int(len(ds))}
    labels = np.array(ds["label"], dtype=np.int32)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return {"rows": int(len(ds)), "pos": pos, "neg": neg}



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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedCausalLmTrainer(Trainer):
    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        weights = inputs.pop("sample_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is None or logits is None:
            loss = outputs.get("loss")
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab = int(shift_logits.size(-1))
        token_loss = torch_f.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())
        valid_mask = (shift_labels != -100).float()
        per_sample = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        if weights is not None:
            w = weights.to(per_sample.device).float().view(-1)
            loss = (per_sample * w).sum() / w.sum().clamp_min(1e-8)
        else:
            loss = per_sample.mean()
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    seed_everything(SEED)
    source_11 = resolve_stage11_dataset_run()
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
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

    train_files, eval_files, summary = collect_json_files(source_11, buckets)
    data_files: dict[str, list[str]] = {"train": train_files}
    if eval_files:
        data_files["eval"] = eval_files

    ds: DatasetDict = load_dataset("json", data_files=data_files)  # type: ignore[assignment]
    before_stats = binary_label_stats(ds["train"])
    ds["train"] = rebalance_negatives(ds["train"], MAX_NEG_POS_RATIO, SEED)
    after_rebalance_stats = binary_label_stats(ds["train"])
    ds["train"] = maybe_cap_rows(ds["train"], MAX_TRAIN_ROWS, SEED)
    after_cap_stats = binary_label_stats(ds["train"])
    if "eval" in ds:
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

    ds = ds.map(_fmt, remove_columns=[c for c in ds["train"].column_names if c not in ("text", "sample_weight")])
    print(
        "[DATA] train_stats "
        f"before={before_stats} "
        f"after_rebalance={after_rebalance_stats} "
        f"after_cap={after_cap_stats}"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"

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

    ds_tok = ds.map(_tok, remove_columns=ds["train"].column_names)

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

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": TRUST_REMOTE_CODE,
    }
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
                "model load failed with Windows os error 1455 (virtual memory/pagefile too small). "
                "Increase pagefile size (recommend >= 40GB), reboot to apply, then rerun stage11_2."
            ) from exc
        raise
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
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

    has_eval = "eval" in ds_tok and len(ds_tok["eval"]) > 0
    args = TrainingArguments(
        output_dir=(out_dir / "trainer_output").as_posix(),
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

    trainer = WeightedCausalLmTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=(ds_tok["eval"] if has_eval else None),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

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

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage11_dataset_run": str(source_11),
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
        "data_files": summary,
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[DONE] qlora adapter saved to: {adapter_dir}")
    print(f"[DONE] run_meta: {out_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()

