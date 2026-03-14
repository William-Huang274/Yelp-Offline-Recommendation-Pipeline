from __future__ import annotations

import gc
import inspect
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.modeling_utils as _tf_modeling_utils

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    os.getenv("QLORA_EVAL_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128").strip()
    or "expandable_segments:True,max_split_size_mb:128",
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


RUN_TAG = "stage11_2_pointwise_confuser_scores"

INPUT_11_RUN_DIR = os.getenv("INPUT_11_RUN_DIR", "").strip()
INPUT_11_ROOT = env_or_project_path("INPUT_11_ROOT_DIR", "data/output/11_qlora_data")
INPUT_11_SUFFIX = "_stage11_1_qlora_build_dataset"

INPUT_11_2_RUN_DIR = os.getenv("INPUT_11_2_RUN_DIR", "").strip()
INPUT_11_2_ROOT = env_or_project_path("INPUT_11_2_ROOT_DIR", "data/output/11_qlora_models")
INPUT_11_2_SUFFIX = "_stage11_2_qlora_train"

OUTPUT_ROOT = env_or_project_path("OUTPUT_11_CONFUSER_SCORES_ROOT_DIR", "data/output/11_qlora_confuser_scores")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
PROMPT_SOURCE = os.getenv("QLORA_POINTWISE_SCORE_SOURCE", "rich_sft").strip().lower() or "rich_sft"
SCORE_SPLITS_RAW = os.getenv("QLORA_POINTWISE_SCORE_SPLITS", "train,eval").strip()
TRUE_LABEL_MODE = os.getenv("QLORA_POINTWISE_TRUE_LABEL_MODE", "label_source_true").strip().lower() or "label_source_true"
BLEND_ALPHA = float(os.getenv("QLORA_BLEND_ALPHA", "0.42").strip() or 0.42)
INVERT_PROB = os.getenv("QLORA_INVERT_PROB", "false").strip().lower() == "true"

USE_4BIT = os.getenv("QLORA_EVAL_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("QLORA_EVAL_USE_BF16", "true").strip().lower() == "true"
TRUST_REMOTE_CODE = os.getenv("QLORA_TRUST_REMOTE_CODE", "true").strip().lower() == "true"
ATTN_IMPLEMENTATION = os.getenv("QLORA_EVAL_ATTN_IMPLEMENTATION", "").strip().lower()
QWEN35_MAMBA_SSM_DTYPE = os.getenv("QLORA_QWEN35_MAMBA_SSM_DTYPE", "auto").strip().lower()
QWEN35_NO_THINK = os.getenv("QLORA_EVAL_QWEN35_NO_THINK", "false").strip().lower() == "true"
DISABLE_ALLOC_WARMUP = os.getenv("QLORA_DISABLE_ALLOC_WARMUP", "true").strip().lower() == "true"
DISABLE_PARALLEL_LOADING = os.getenv("QLORA_EVAL_DISABLE_PARALLEL_LOADING", "true").strip().lower() == "true"
PARALLEL_LOADING_WORKERS = int(os.getenv("QLORA_EVAL_PARALLEL_LOADING_WORKERS", "1").strip() or 1)
DEVICE_MAP = os.getenv("QLORA_EVAL_DEVICE_MAP", "auto").strip() or "auto"
MAX_MEMORY_CUDA = os.getenv("QLORA_EVAL_MAX_MEMORY_CUDA", "").strip()
MAX_MEMORY_CPU = os.getenv("QLORA_EVAL_MAX_MEMORY_CPU", "").strip()
OFFLOAD_FOLDER = os.getenv("QLORA_EVAL_OFFLOAD_FOLDER", "").strip()
OFFLOAD_STATE_DICT = os.getenv("QLORA_EVAL_OFFLOAD_STATE_DICT", "true").strip().lower() == "true"

MAX_SEQ_LEN = int(os.getenv("QLORA_EVAL_MAX_SEQ_LEN", "768").strip() or 768)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_EVAL_PAD_TO_MULTIPLE_OF", "0").strip() or 0)
BATCH_SIZE = int(os.getenv("QLORA_EVAL_BATCH_SIZE", "12").strip() or 12)
PROMPT_CHUNK_ROWS = int(os.getenv("QLORA_EVAL_PROMPT_CHUNK_ROWS", "8192").strip() or 8192)
PRETOKENIZE_PROMPT_CHUNK = os.getenv("QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK", "true").strip().lower() == "true"
PIN_MEMORY = os.getenv("QLORA_EVAL_PIN_MEMORY", "true").strip().lower() == "true"
NON_BLOCKING_H2D = os.getenv("QLORA_EVAL_NON_BLOCKING_H2D", "true").strip().lower() == "true"


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
    return sorted(set(out))


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage11_data_run() -> Path:
    if INPUT_11_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_11_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_11_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_11_ROOT, INPUT_11_SUFFIX)


def resolve_stage11_2_run() -> Path:
    if INPUT_11_2_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_11_2_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_11_2_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_11_2_ROOT, INPUT_11_2_SUFFIX)


def is_qwen35_model_type(model_type: str) -> bool:
    return str(model_type or "").strip().lower().startswith("qwen3_5")


def detect_final_token_logits_arg(model: Any) -> str:
    seen: set[int] = set()
    candidates: list[Any] = [model]
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        try:
            base_obj = get_base()
            if base_obj is not None:
                candidates.append(base_obj)
        except Exception:
            pass
    base_model_obj = getattr(model, "base_model", None)
    if base_model_obj is not None:
        candidates.append(base_model_obj)
        nested_model = getattr(base_model_obj, "model", None)
        if nested_model is not None:
            candidates.append(nested_model)

    for obj in candidates:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            sig = inspect.signature(obj.forward)
        except Exception:
            continue
        if "num_logits_to_keep" in sig.parameters:
            return "num_logits_to_keep"
        if "logits_to_keep" in sig.parameters:
            return "logits_to_keep"

    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
    if is_qwen35_model_type(model_type):
        return "logits_to_keep"
    return ""


def build_yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    def _pick(cands: list[str], fallback: str) -> int:
        for c in cands:
            ids = tokenizer(c, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return int(ids[0])
        ids = tokenizer(fallback, add_special_tokens=False).input_ids
        if not ids:
            raise RuntimeError(f"failed to build token id for fallback={fallback}")
        return int(ids[0])

    yes_id = _pick([" YES", "Yes", " yes", "YES"], "Y")
    no_id = _pick([" NO", "No", " no", "NO"], "N")
    return yes_id, no_id


def score_yes_probability(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_seq_len: int,
    yes_id: int,
    no_id: int,
) -> tuple[np.ndarray, int]:
    def _next_fallback_bs(cur_bs: int) -> int:
        b = int(cur_bs)
        if b <= 1:
            return 1
        if b == 3:
            return 2
        return max(1, b // 2)

    def _safe_cuda_empty_cache() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _pin_tensor_batch(batch_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not (torch.cuda.is_available() and PIN_MEMORY):
            return batch_tensors
        out_batch: dict[str, torch.Tensor] = {}
        for key, value in batch_tensors.items():
            if isinstance(value, torch.Tensor) and value.device.type == "cpu" and not value.is_pinned():
                out_batch[key] = value.pin_memory()
            else:
                out_batch[key] = value
        return out_batch

    def _move_tensor_batch_to_device(
        batch_tensors: dict[str, torch.Tensor],
        target_device: Any,
    ) -> dict[str, torch.Tensor]:
        non_blocking = bool(torch.cuda.is_available() and NON_BLOCKING_H2D)
        moved: dict[str, torch.Tensor] = {}
        for key, value in batch_tensors.items():
            use_non_blocking = bool(
                non_blocking
                and isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and value.is_pinned()
            )
            moved[key] = value.to(target_device, non_blocking=use_non_blocking)
        return moved

    def _wrap_prompt_for_chat(prompt_text: str) -> str:
        messages = [{"role": "user", "content": str(prompt_text)}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if template_enable_thinking_supported:
            kwargs["enable_thinking"] = False
        return str(tokenizer.apply_chat_template(messages, **kwargs))  # type: ignore[attr-defined]

    out: list[np.ndarray] = []
    template_enable_thinking_supported = False
    template_use_chat = bool(QWEN35_NO_THINK and hasattr(tokenizer, "apply_chat_template"))
    if template_use_chat:
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)  # type: ignore[attr-defined]
            template_enable_thinking_supported = "enable_thinking" in sig.parameters
        except Exception:
            template_enable_thinking_supported = False

    model.eval()
    forward_kwargs_base: dict[str, Any] = {"use_cache": False}
    final_token_logits_arg = detect_final_token_logits_arg(model)
    if final_token_logits_arg:
        forward_kwargs_base[final_token_logits_arg] = 1
        print(f"[CONFIG] final_token_logits_arg={final_token_logits_arg}")
    else:
        print("[CONFIG] final_token_logits_arg=<unsupported>")

    effective_bs = max(1, int(batch_size))
    prepared_prompts = prompts
    if template_use_chat:
        prepared_prompts = [_wrap_prompt_for_chat(p) for p in prompts]

    tokenized_prompts: dict[str, torch.Tensor] | None = None
    t0 = time.monotonic()
    if prepared_prompts and PRETOKENIZE_PROMPT_CHUNK:
        tokenized_prompts = tokenizer(
            prepared_prompts,
            padding=True,
            pad_to_multiple_of=(int(PAD_TO_MULTIPLE_OF) if int(PAD_TO_MULTIPLE_OF) > 0 else None),
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        tokenized_prompts = _pin_tensor_batch(tokenized_prompts)
        print(f"[TIMING] chunk tokenization done: {len(prepared_prompts)} rows in {time.monotonic() - t0:.1f}s")

    scored_total = 0
    last_log_time = t0
    with torch.inference_mode():
        i = 0
        n = len(prepared_prompts)
        while i < n:
            bs = min(effective_bs, n - i)
            try:
                if tokenized_prompts is not None:
                    enc = {k: v[i : i + bs] for k, v in tokenized_prompts.items()}
                else:
                    batch = prepared_prompts[i : i + bs]
                    enc = tokenizer(
                        batch,
                        padding=True,
                        pad_to_multiple_of=(int(PAD_TO_MULTIPLE_OF) if int(PAD_TO_MULTIPLE_OF) > 0 else None),
                        truncation=True,
                        max_length=max_seq_len,
                        return_tensors="pt",
                    )
                    enc = _pin_tensor_batch(enc)
                enc = _move_tensor_batch_to_device(enc, model.device)
                outputs = model(**enc, **forward_kwargs_base)
                logits = outputs.logits
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                pair = logits[:, [yes_id, no_id]]
                probs = torch.softmax(pair, dim=1)[:, 0]
                out.append(probs.detach().float().cpu().numpy())
                i += bs
                scored_total += bs
                now = time.monotonic()
                if now - last_log_time >= 30.0 or i >= n:
                    elapsed = now - t0
                    speed = scored_total / elapsed if elapsed > 0 else 0.0
                    remaining = (n - i) / speed if speed > 0 else 0.0
                    print(
                        f"[PROGRESS] scored={scored_total}/{n} speed={speed:.1f} rows/s "
                        f"elapsed={elapsed:.0f}s ETA={remaining:.0f}s batch_size={effective_bs}"
                    )
                    last_log_time = now
            except torch.OutOfMemoryError as oom:
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError("pointwise confuser scoring OOM at batch_size=1") from oom
                effective_bs = _next_fallback_bs(bs)
                print(f"[WARN] confuser scoring OOM: batch={bs}, fallback_batch={effective_bs}")
            except RuntimeError as rte:
                msg = str(rte).lower()
                if "cuda" not in msg:
                    raise
                is_retryable = ("cublas" in msg) or ("cudnn" in msg) or ("out of memory" in msg)
                if not is_retryable:
                    raise
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError("pointwise confuser scoring CUDA failure at batch_size=1") from rte
                effective_bs = _next_fallback_bs(bs)
                short = str(rte).splitlines()[0][:180]
                print(f"[WARN] confuser scoring CUDA failure: batch={bs}, fallback_batch={effective_bs}, err={short}")

    total_time = time.monotonic() - t0
    if prepared_prompts:
        print(f"[TIMING] inference done: {len(prepared_prompts)} rows in {total_time:.1f}s ({len(prepared_prompts) / total_time:.1f} rows/s)")
    merged = np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)
    return merged, int(effective_bs)


def normalize_pre_score(pdf: pd.DataFrame) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        a = s.min()
        b = s.max()
        if pd.isna(a) or pd.isna(b) or abs(float(b) - float(a)) < 1e-9:
            return pd.Series(np.full(len(s), 0.5, dtype=np.float32), index=s.index)
        return ((s - a) / (b - a + 1e-9)).astype(np.float32)

    return pdf.groupby("user_idx", sort=False)["pre_score"].transform(_norm).astype(np.float32)


def resolve_base_model(stage11_2_run: Path) -> str:
    explicit = str(os.getenv("QLORA_BASE_MODEL", "")).strip()
    if explicit:
        return explicit
    meta_path = stage11_2_run / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_model = str(meta.get("base_model", "")).strip()
            if base_model:
                return base_model
        except Exception:
            pass
    raise RuntimeError("failed to resolve base model; set QLORA_BASE_MODEL")


def iter_json_rows(dir_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(dir_path.glob("part-*.json")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if not txt:
                    continue
                rows.append(json.loads(txt))
    return rows


def load_pointwise_rows(stage11_data_run: Path, bucket: int, split: str) -> pd.DataFrame:
    if PROMPT_SOURCE != "rich_sft":
        raise ValueError(f"unsupported QLORA_POINTWISE_SCORE_SOURCE={PROMPT_SOURCE!r}")
    bdir = stage11_data_run / f"bucket_{int(bucket)}" / f"rich_sft_{split}_json"
    if not bdir.exists():
        raise FileNotFoundError(f"pointwise json dir not found: {bdir}")
    rows = iter_json_rows(bdir)
    if not rows:
        return pd.DataFrame(columns=["user_idx", "item_idx", "pre_rank", "pre_score", "prompt", "label", "label_source", "split"])
    pdf = pd.DataFrame(rows)
    keep_cols = [c for c in ["user_idx", "item_idx", "pre_rank", "pre_score", "prompt", "label", "label_source", "split"] if c in pdf.columns]
    pdf = pdf[keep_cols].copy()
    pdf["user_idx"] = pdf["user_idx"].astype("int64")
    pdf["item_idx"] = pdf["item_idx"].astype("int64")
    pdf["pre_rank"] = pd.to_numeric(pdf["pre_rank"], errors="coerce").fillna(999999).astype("int32")
    pdf["pre_score"] = pd.to_numeric(pdf["pre_score"], errors="coerce").fillna(0.0).astype("float32")
    pdf["prompt"] = pdf["prompt"].astype(str)
    pdf["label"] = pd.to_numeric(pdf.get("label", 0), errors="coerce").fillna(0).astype("int8")
    pdf["label_source"] = pdf.get("label_source", "").astype(str)
    pdf["split"] = split
    if TRUE_LABEL_MODE == "label_source_true":
        pdf["label_true"] = (pdf["label_source"].str.strip().str.lower() == "true").astype("int8")
    else:
        pdf["label_true"] = pdf["label"].astype("int8")
    return pdf


def main() -> None:
    stage11_data_run = resolve_stage11_data_run()
    stage11_2_run = resolve_stage11_2_run()
    adapter_dir = stage11_2_run / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")

    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
    score_splits = [s.strip().lower() for s in SCORE_SPLITS_RAW.split(",") if s.strip()]
    if not score_splits:
        score_splits = ["train", "eval"]

    base_model = resolve_base_model(stage11_2_run)
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

    model_kwargs: dict[str, Any] = {"trust_remote_code": TRUST_REMOTE_CODE, "low_cpu_mem_usage": True}
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

    try:
        model_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=TRUST_REMOTE_CODE)
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
                model_kwargs["config"] = model_cfg
                print(f"[CONFIG] qwen3.5 mamba_ssm_dtype={target_mamba_dtype}")
    except Exception as cfg_exc:
        short = str(cfg_exc).splitlines()[0][:180]
        print(f"[WARN] qwen3.5 dtype config override skipped: {short}")

    if DISABLE_ALLOC_WARMUP and hasattr(_tf_modeling_utils, "caching_allocator_warmup"):
        _tf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # type: ignore[assignment]
        print("[CONFIG] disable_transformers_alloc_warmup=true")
    if DISABLE_PARALLEL_LOADING:
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(max(1, int(PARALLEL_LOADING_WORKERS)))

    print(f"[CONFIG] base_model={base_model}")
    print(f"[CONFIG] stage11_data_run={stage11_data_run}")
    print(f"[CONFIG] stage11_2_run={stage11_2_run}")
    print(f"[CONFIG] prompt_source={PROMPT_SOURCE} score_splits={score_splits} blend_alpha={BLEND_ALPHA} invert_prob={INVERT_PROB}")
    print(f"[CONFIG] max_seq_len={MAX_SEQ_LEN} batch_size={BATCH_SIZE} pad_to_multiple_of={PAD_TO_MULTIPLE_OF}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "left"

    if has_cuda:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir.as_posix())
    try:
        if getattr(getattr(model, "config", None), "use_cache", None):
            model.config.use_cache = False
    except Exception:
        pass
    yes_id, no_id = build_yes_no_token_ids(tokenizer)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    audit: dict[str, Any] = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage11_dataset_run": str(stage11_data_run),
        "source_stage11_2_run": str(stage11_2_run),
        "source_prompt": str(PROMPT_SOURCE),
        "score_splits": score_splits,
        "buckets": buckets,
        "blend_alpha": float(BLEND_ALPHA),
        "invert_prob": bool(INVERT_PROB),
        "label_mode": str(TRUE_LABEL_MODE),
        "splits": {},
    }

    for bucket in buckets:
        for split in score_splits:
            pdf = load_pointwise_rows(stage11_data_run, int(bucket), split)
            if pdf.empty:
                print(f"[WARN] skip bucket={bucket} split={split}: empty pointwise rows")
                continue
            print(f"[DATA] bucket={bucket} split={split} rows={len(pdf)} users={pdf['user_idx'].nunique()}")
            probs_parts: list[np.ndarray] = []
            effective_bs = int(BATCH_SIZE)
            prompts = pdf["prompt"].tolist()
            for start in range(0, len(prompts), max(1, int(PROMPT_CHUNK_ROWS))):
                chunk_prompts = prompts[start : start + int(PROMPT_CHUNK_ROWS)]
                probs_chunk, effective_bs = score_yes_probability(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=chunk_prompts,
                    batch_size=effective_bs,
                    max_seq_len=int(MAX_SEQ_LEN),
                    yes_id=int(yes_id),
                    no_id=int(no_id),
                )
                probs_parts.append(probs_chunk.astype(np.float32))
            qprob = np.concatenate(probs_parts, axis=0) if probs_parts else np.zeros(0, dtype=np.float32)
            if INVERT_PROB:
                qprob = np.clip(1.0 - qprob, 0.0, 1.0).astype(np.float32)
            if len(qprob) != len(pdf):
                raise RuntimeError(f"score size mismatch bucket={bucket} split={split}: probs={len(qprob)} rows={len(pdf)}")
            pdf = pdf.copy()
            pdf["bucket"] = int(bucket)
            pdf["qlora_prob"] = qprob
            pdf["pre_norm"] = normalize_pre_score(pdf)
            pdf["blend_score"] = ((1.0 - float(BLEND_ALPHA)) * pdf["pre_norm"] + float(BLEND_ALPHA) * pdf["qlora_prob"]).astype(np.float32)
            out_csv = out_dir / f"bucket_{int(bucket)}_{split}_pointwise_scores.csv"
            pdf[
                [
                    "bucket",
                    "split",
                    "user_idx",
                    "item_idx",
                    "pre_rank",
                    "pre_score",
                    "label",
                    "label_true",
                    "label_source",
                    "qlora_prob",
                    "blend_score",
                ]
            ].to_csv(out_csv.as_posix(), index=False)
            audit["splits"][f"bucket_{int(bucket)}_{split}"] = {
                "rows": int(len(pdf)),
                "users": int(pdf["user_idx"].nunique()),
                "label_true_rows": int(pdf["label_true"].sum()),
                "qlora_prob_mean": float(pdf["qlora_prob"].mean()),
                "qlora_prob_std": float(pdf["qlora_prob"].std(ddof=0)),
                "blend_score_mean": float(pdf["blend_score"].mean()),
                "effective_batch_size": int(effective_bs),
                "output_file": str(out_csv),
            }
            all_rows.append(
                pdf[
                    [
                        "bucket",
                        "split",
                        "user_idx",
                        "item_idx",
                        "pre_rank",
                        "pre_score",
                        "label",
                        "label_true",
                        "label_source",
                        "qlora_prob",
                        "blend_score",
                    ]
                ].copy()
            )

    if not all_rows:
        raise RuntimeError("no pointwise score rows exported")

    combined = pd.concat(all_rows, ignore_index=True)
    combined_csv = out_dir / "pointwise_confuser_scores.csv"
    combined.to_csv(combined_csv.as_posix(), index=False)
    audit["combined"] = {
        "rows": int(len(combined)),
        "users": int(combined["user_idx"].nunique()),
        "label_true_rows": int(combined["label_true"].sum()),
        "splits": combined["split"].value_counts(dropna=False).to_dict(),
        "buckets": combined["bucket"].value_counts(dropna=False).to_dict(),
        "output_file": str(combined_csv),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[DONE] audit: {out_dir / 'audit.json'}")
    print(f"[DONE] combined_scores: {combined_csv}")


if __name__ == "__main__":
    main()
