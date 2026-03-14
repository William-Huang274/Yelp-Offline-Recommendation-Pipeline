from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling
except Exception:
    qwen35_modeling = None


BASE_MODEL = os.getenv("QLORA_BASE_MODEL", "").strip()
USE_4BIT = os.getenv("QLORA_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("QLORA_USE_BF16", "true").strip().lower() == "true"
TRUST_REMOTE_CODE = os.getenv("QLORA_TRUST_REMOTE_CODE", "true").strip().lower() == "true"
MAX_SEQ_LEN = int(os.getenv("QLORA_SMOKE_MAX_SEQ_LEN", "512").strip() or 512)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_PAD_TO_MULTIPLE_OF", "64").strip() or 64)
SFT_JSON_GLOB = os.getenv("QLORA_SMOKE_SFT_JSON_GLOB", "").strip()
DPO_JSONL = os.getenv("QLORA_SMOKE_DPO_JSONL", "").strip()
SFT_ROWS = int(os.getenv("QLORA_SMOKE_MAX_SFT_ROWS", "2").strip() or 2)
DPO_ROWS = int(os.getenv("QLORA_SMOKE_MAX_DPO_ROWS", "2").strip() or 2)
RESULT_JSON = os.getenv("QLORA_SMOKE_RESULT_JSON", "").strip()
WARMUP_ITERS = int(os.getenv("QLORA_SMOKE_WARMUP_ITERS", "2").strip() or 2)
BENCH_ITERS = int(os.getenv("QLORA_SMOKE_BENCH_ITERS", "5").strip() or 5)
BENCH_MODE = os.getenv("QLORA_QWEN35_BENCH_MODE", "fast").strip().lower() or "fast"


def is_qwen35_model_type(model_type: str) -> bool:
    return str(model_type or "").strip().lower().startswith("qwen3_5")


def prepare_model_for_kbit_training_qwen35_safe(model: Any) -> Any:
    for _, param in model.named_parameters():
        param.requires_grad = False
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


def rounded_length(length: int) -> int:
    if PAD_TO_MULTIPLE_OF > 1 and length > 0:
        return ((int(length) + int(PAD_TO_MULTIPLE_OF) - 1) // int(PAD_TO_MULTIPLE_OF)) * int(PAD_TO_MULTIPLE_OF)
    return int(length)


def load_json_rows(path_glob: str, max_rows: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(glob.glob(path_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
                if len(out) >= int(max_rows):
                    return out
    return out


def load_jsonl_rows(path: str, max_rows: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= int(max_rows):
                break
    return out


def tokenize_texts(tokenizer: AutoTokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(MAX_SEQ_LEN),
        pad_to_multiple_of=int(PAD_TO_MULTIPLE_OF) if PAD_TO_MULTIPLE_OF > 1 else None,
        return_tensors="pt",
    )


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def sequence_logprob(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(logits.dtype)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (token_log_probs * mask).sum(dim=1) / denom


def prepare_inputs(tokenizer: AutoTokenizer) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...], int, int]:
    sft_rows = load_json_rows(SFT_JSON_GLOB, SFT_ROWS)
    if not sft_rows:
        raise RuntimeError(f"no SFT rows found for glob={SFT_JSON_GLOB}")
    sft_texts = []
    for row in sft_rows:
        answer = " YES" if int(row.get("label", 0)) == 1 else " NO"
        sft_texts.append(str(row["prompt"]) + answer)
    sft_batch = tokenize_texts(tokenizer, sft_texts)

    dpo_rows = load_jsonl_rows(DPO_JSONL, DPO_ROWS)
    if not dpo_rows:
        raise RuntimeError(f"no DPO rows found in {DPO_JSONL}")

    chosen_input_ids: list[list[int]] = []
    chosen_attention_mask: list[list[int]] = []
    chosen_completion_mask: list[list[int]] = []
    rejected_input_ids: list[list[int]] = []
    rejected_attention_mask: list[list[int]] = []
    rejected_completion_mask: list[list[int]] = []
    eos_token_id = tokenizer.eos_token_id
    for row in dpo_rows:
        prompt = str(row["prompt"])
        chosen = str(row["chosen"])
        rejected = str(row["rejected"])
        chosen_ids = tokenizer(prompt + chosen, truncation=True, max_length=int(MAX_SEQ_LEN))["input_ids"]
        rejected_ids = tokenizer(prompt + rejected, truncation=True, max_length=int(MAX_SEQ_LEN))["input_ids"]
        prefix_len = longest_common_prefix_len(chosen_ids, rejected_ids)
        if eos_token_id is not None:
            if not chosen_ids or chosen_ids[-1] != eos_token_id:
                chosen_ids = chosen_ids + [int(eos_token_id)]
            if not rejected_ids or rejected_ids[-1] != eos_token_id:
                rejected_ids = rejected_ids + [int(eos_token_id)]
        chosen_input_ids.append(chosen_ids)
        chosen_attention_mask.append([1] * len(chosen_ids))
        chosen_completion_mask.append(([0] * min(prefix_len, len(chosen_ids))) + ([1] * max(0, len(chosen_ids) - prefix_len)))
        rejected_input_ids.append(rejected_ids)
        rejected_attention_mask.append([1] * len(rejected_ids))
        rejected_completion_mask.append(([0] * min(prefix_len, len(rejected_ids))) + ([1] * max(0, len(rejected_ids) - prefix_len)))

    def _pad(batch_ids: list[list[int]], batch_mask: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = min(int(MAX_SEQ_LEN), rounded_length(max(len(x) for x in batch_ids)))
        padded_ids = []
        padded_mask = []
        for ids, mask in zip(batch_ids, batch_mask):
            ids = ids[:max_len]
            mask = mask[:max_len]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + ([int(tokenizer.pad_token_id)] * pad_len)
                mask = mask + ([0] * pad_len)
            padded_ids.append(ids)
            padded_mask.append(mask)
        return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(padded_mask, dtype=torch.long)

    chosen_ids_t, chosen_att_t = _pad(chosen_input_ids, chosen_attention_mask)
    rejected_ids_t, rejected_att_t = _pad(rejected_input_ids, rejected_attention_mask)
    chosen_cmp_t, _ = _pad(chosen_completion_mask, chosen_attention_mask)
    rejected_cmp_t, _ = _pad(rejected_completion_mask, rejected_attention_mask)
    return sft_batch, (chosen_ids_t, chosen_att_t, chosen_cmp_t, rejected_ids_t, rejected_att_t, rejected_cmp_t), len(sft_rows), len(dpo_rows)


def load_model() -> tuple[Any, AutoTokenizer, torch.dtype]:
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

    cfg = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    model_type = str(getattr(cfg, "model_type", "")).strip().lower()
    text_cfg = getattr(cfg, "text_config", None)
    if is_qwen35_model_type(model_type):
        if text_cfg is not None and hasattr(text_cfg, "mamba_ssm_dtype") and compute_dtype == torch.bfloat16:
            setattr(text_cfg, "mamba_ssm_dtype", "bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "config": cfg,
    }
    if has_cuda:
        model_kwargs["device_map"] = "auto"
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    loaded_model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
    if is_qwen35_model_type(loaded_model_type):
        model = prepare_model_for_kbit_training_qwen35_safe(model)
        recast_float_params(model, compute_dtype)
    model.eval()
    return model, tokenizer, compute_dtype


def force_qwen35_fallback(model: Any) -> int:
    if qwen35_modeling is None:
        raise RuntimeError("transformers.models.qwen3_5.modeling_qwen3_5 unavailable")
    n_patched = 0
    fallback_chunk_rule = getattr(qwen35_modeling, "torch_chunk_gated_delta_rule", None)
    fallback_conv_update = getattr(qwen35_modeling, "torch_causal_conv1d_update", None)
    if fallback_chunk_rule is None or fallback_conv_update is None:
        raise RuntimeError("Qwen3.5 fallback functions unavailable in installed transformers")
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3_5GatedDeltaNet":
            continue
        if hasattr(module, "causal_conv1d_fn"):
            module.causal_conv1d_fn = None
        if hasattr(module, "causal_conv1d_update"):
            module.causal_conv1d_update = fallback_conv_update
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = fallback_chunk_rule
        n_patched += 1
    return int(n_patched)


def benchmark_forward_only(
    model: Any,
    sft_batch: dict[str, torch.Tensor],
    dpo_batch: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    device = next(model.parameters()).device
    sft_batch = {k: v.to(device) for k, v in sft_batch.items()}
    chosen_ids_t, chosen_att_t, chosen_cmp_t, rejected_ids_t, rejected_att_t, rejected_cmp_t = [
        x.to(device) for x in dpo_batch
    ]
    def _forward_step() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            sft_out = model(**sft_batch)
            chosen_lp = sequence_logprob(model, chosen_ids_t, chosen_att_t, chosen_cmp_t)
            rejected_lp = sequence_logprob(model, rejected_ids_t, rejected_att_t, rejected_cmp_t)
            sft_summary = sft_out.logits[..., -1, :].float().mean()
            dpo_summary = (chosen_lp - rejected_lp).float().mean()
        return sft_summary, dpo_summary

    for _ in range(max(0, WARMUP_ITERS)):
        _forward_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_sft = None
    last_dpo = None
    for _ in range(max(1, BENCH_ITERS)):
        last_sft, last_dpo = _forward_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return {
        "bench_iters": int(max(1, BENCH_ITERS)),
        "seconds_total": float(t1 - t0),
        "seconds_per_iter": float((t1 - t0) / max(1, BENCH_ITERS)),
        "tokens_per_iter": int(sum(v.numel() for v in sft_batch.values()) + chosen_ids_t.numel() + rejected_ids_t.numel()),
        "sft_forward_summary": float(last_sft.float().cpu().item() if last_sft is not None else 0.0),
        "dpo_forward_summary": float(last_dpo.float().cpu().item() if last_dpo is not None else 0.0),
    }


def main() -> None:
    if not BASE_MODEL:
        raise ValueError("QLORA_BASE_MODEL is required")
    if not SFT_JSON_GLOB:
        raise ValueError("QLORA_SMOKE_SFT_JSON_GLOB is required")
    if not DPO_JSONL:
        raise ValueError("QLORA_SMOKE_DPO_JSONL is required")

    if BENCH_MODE not in {"fast", "fallback"}:
        raise ValueError("QLORA_QWEN35_BENCH_MODE must be one of: fast, fallback")

    model_fast, tokenizer, compute_dtype = load_model()
    sft_batch, dpo_batch, sft_rows, dpo_rows = prepare_inputs(tokenizer)
    if BENCH_MODE == "fallback":
        patched_layers = force_qwen35_fallback(model_fast)
    else:
        patched_layers = 0

    linear_layer = next((m for m in model_fast.modules() if m.__class__.__name__ == "Qwen3_5GatedDeltaNet"), None)
    if linear_layer is None:
        raise RuntimeError("Qwen3_5GatedDeltaNet not found")

    bench_metrics = benchmark_forward_only(model_fast, sft_batch, dpo_batch)
    bench_meta = {
        "bench_mode": BENCH_MODE,
        "patched_layers": int(patched_layers),
        "linear_layer_causal_conv1d_fn_nonnull": bool(getattr(linear_layer, "causal_conv1d_fn", None) is not None),
        "linear_layer_causal_conv1d_update": getattr(getattr(linear_layer, "causal_conv1d_update", None), "__name__", str(getattr(linear_layer, "causal_conv1d_update", None))),
        "linear_layer_chunk_rule": getattr(getattr(linear_layer, "chunk_gated_delta_rule", None), "__name__", str(getattr(linear_layer, "chunk_gated_delta_rule", None))),
    }

    result = {
        "base_model": BASE_MODEL,
        "fla_available": bool(is_flash_linear_attention_available()),
        "causal_conv1d_available": bool(is_causal_conv1d_available()),
        "compute_dtype": str(compute_dtype),
        "sft_rows": int(sft_rows),
        "dpo_rows": int(dpo_rows),
        "bench": {**bench_meta, **bench_metrics},
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if RESULT_JSON:
        out_path = Path(RESULT_JSON)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
