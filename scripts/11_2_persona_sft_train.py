from __future__ import annotations

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script}")
    print("This stage script starts Stage11 persona SFT training and may load GPU/model dependencies.")
    print("Set the required INPUT_/OUTPUT_/PERSONA_SFT_* environment variables, then run without --help.")
    sys.exit(0)

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)

from pipeline.project_paths import (
    env_or_project_path,
    project_path,
    resolve_latest_run_pointer,
    write_latest_run_pointer,
)

RUN_TAG = os.getenv("RUN_TAG", "stage11_2_persona_sft_train").strip() or "stage11_2_persona_sft_train"
INPUT_11_ROOT = env_or_project_path("INPUT_11_PERSONA_SFT_DATA_ROOT_DIR", "data/output/11_persona_sft_data")
INPUT_11_RUN_DIR = os.getenv("INPUT_11_PERSONA_SFT_DATA_RUN_DIR", "").strip()
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_PERSONA_SFT_MODELS_ROOT_DIR", "data/output/11_persona_sft_models")
RESUME_RUN_DIR = os.getenv("PERSONA_SFT_RESUME_RUN_DIR", "").strip()

BASE_MODEL = os.getenv("PERSONA_SFT_BASE_MODEL", "/root/hf_models/Qwen3.5-9B").strip() or "/root/hf_models/Qwen3.5-9B"
USE_4BIT = os.getenv("PERSONA_SFT_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("PERSONA_SFT_USE_BF16", "true").strip().lower() == "true"
MAX_SEQ_LEN = int(os.getenv("PERSONA_SFT_MAX_SEQ_LEN", "1792").strip() or 1792)
MAX_TRAIN_ROWS = int(os.getenv("PERSONA_SFT_MAX_TRAIN_ROWS", "4096").strip() or 4096)
MAX_EVAL_ROWS = int(os.getenv("PERSONA_SFT_MAX_EVAL_ROWS", "512").strip() or 512)
EPOCHS = float(os.getenv("PERSONA_SFT_EPOCHS", "1.0").strip() or 1.0)
MAX_STEPS = int(os.getenv("PERSONA_SFT_MAX_STEPS", "0").strip() or 0)
LR = float(os.getenv("PERSONA_SFT_LR", "2e-4").strip() or 2e-4)
WEIGHT_DECAY = float(os.getenv("PERSONA_SFT_WEIGHT_DECAY", "0.01").strip() or 0.01)
WARMUP_RATIO = float(os.getenv("PERSONA_SFT_WARMUP_RATIO", "0.03").strip() or 0.03)
BATCH_SIZE = int(os.getenv("PERSONA_SFT_BATCH_SIZE", "1").strip() or 1)
EVAL_BATCH_SIZE = int(os.getenv("PERSONA_SFT_EVAL_BATCH_SIZE", "1").strip() or 1)
GRAD_ACC = int(os.getenv("PERSONA_SFT_GRAD_ACC", "8").strip() or 8)
SAVE_STEPS = int(os.getenv("PERSONA_SFT_SAVE_STEPS", "200").strip() or 200)
EVAL_STEPS = int(os.getenv("PERSONA_SFT_EVAL_STEPS", "200").strip() or 200)
LOGGING_STEPS = int(os.getenv("PERSONA_SFT_LOGGING_STEPS", "10").strip() or 10)
SAVE_TOTAL_LIMIT = int(os.getenv("PERSONA_SFT_SAVE_TOTAL_LIMIT", "2").strip() or 2)
SEED = int(os.getenv("PERSONA_SFT_RANDOM_SEED", "42").strip() or 42)
GRADIENT_CHECKPOINTING = os.getenv("PERSONA_SFT_GRADIENT_CHECKPOINTING", "true").strip().lower() == "true"
LORA_R = int(os.getenv("PERSONA_SFT_LORA_R", "16").strip() or 16)
LORA_ALPHA = int(os.getenv("PERSONA_SFT_LORA_ALPHA", "32").strip() or 32)
LORA_DROPOUT = float(os.getenv("PERSONA_SFT_LORA_DROPOUT", "0.05").strip() or 0.05)
LORA_TARGET_MODULES = [
    token.strip()
    for token in (
        os.getenv(
            "PERSONA_SFT_LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        or ""
    ).split(",")
    if token.strip()
]
GEN_EVAL_ROWS = int(os.getenv("PERSONA_SFT_GEN_EVAL_ROWS", "24").strip() or 24)
GEN_MAX_NEW_TOKENS = int(os.getenv("PERSONA_SFT_GEN_MAX_NEW_TOKENS", "384").strip() or 384)
GEN_BATCH_SIZE = int(os.getenv("PERSONA_SFT_GEN_BATCH_SIZE", "4").strip() or 4)
GEN_AUDIT_HOOK = os.getenv("PERSONA_SFT_GENERATION_AUDIT_HOOK", "eval").strip().lower() or "eval"
ATTN_IMPL = os.getenv("PERSONA_SFT_ATTN_IMPL", "flash_attention_2").strip() or "flash_attention_2"
DEFAULT_OPTIMIZER = "paged_adamw_8bit" if USE_4BIT else "adamw_torch"
OPTIMIZER = os.getenv("PERSONA_SFT_OPTIMIZER", DEFAULT_OPTIMIZER).strip() or DEFAULT_OPTIMIZER
SAVE_LOG_HISTORY = os.getenv("PERSONA_SFT_SAVE_LOG_HISTORY", "true").strip().lower() == "true"
SAVE_EVAL_GENERATIONS = os.getenv("PERSONA_SFT_SAVE_EVAL_GENERATIONS", "true").strip().lower() == "true"
TF32 = os.getenv("PERSONA_SFT_ALLOW_TF32", "true").strip().lower() == "true"
DISABLE_GROUPED_MM = os.getenv("PERSONA_SFT_DISABLE_GROUPED_MM", "false").strip().lower() == "true"
STRATIFIED_EVAL_CAP = os.getenv("PERSONA_SFT_STRATIFIED_EVAL_CAP", "true").strip().lower() == "true"
USE_WEIGHTED_SAMPLER = os.getenv("PERSONA_SFT_USE_WEIGHTED_SAMPLER", "false").strip().lower() == "true"
DENSITY_BALANCE_POWER = float(os.getenv("PERSONA_SFT_DENSITY_BALANCE_POWER", "0.5").strip() or 0.5)
CUISINE_BALANCE_POWER = float(os.getenv("PERSONA_SFT_CUISINE_BALANCE_POWER", "0.35").strip() or 0.35)
CUISINE_BALANCE_MIN_COUNT = int(os.getenv("PERSONA_SFT_CUISINE_BALANCE_MIN_COUNT", "16").strip() or 16)
SAMPLE_WEIGHT_MAX_MULTIPLIER = float(os.getenv("PERSONA_SFT_SAMPLE_WEIGHT_MAX_MULTIPLIER", "4.0").strip() or 4.0)
FORCE_SINGLE_GPU_DEVICE_MAP = os.getenv("PERSONA_SFT_FORCE_SINGLE_GPU_DEVICE_MAP", "false").strip().lower() == "true"
SKIP_TRAINER_DEVICE_MOVE = os.getenv("PERSONA_SFT_SKIP_TRAINER_DEVICE_MOVE", "false").strip().lower() == "true"


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"run dir not found: {path}")
        return path
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_persona_sft_input_run() -> Path:
    if INPUT_11_RUN_DIR:
        return resolve_run(INPUT_11_RUN_DIR, INPUT_11_ROOT, "_full_stage11_1_persona_sft_build_dataset")
    pointer_run = resolve_latest_run_pointer("11_persona_sft_data_latest")
    if pointer_run is not None and pointer_run.exists():
        return pointer_run
    return resolve_run(INPUT_11_RUN_DIR, INPUT_11_ROOT, "_full_stage11_1_persona_sft_build_dataset")


def resolve_optional_run_dir(raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = project_path(raw)
    if not path.exists():
        raise FileNotFoundError(f"resume run dir not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"resume run dir is not a directory: {path}")
    return path


def resolve_latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = []
    for path in run_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            step_id = int(path.name.split("-", 1)[1])
        except Exception:
            continue
        checkpoints.append((step_id, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_disable_transformers_grouped_mm() -> None:
    if not DISABLE_GROUPED_MM:
        return
    try:
        import transformers.integrations.moe as moe_integration
    except Exception as exc:
        print(f"[WARN] failed to import transformers.integrations.moe for grouped_mm patch: {exc}")
        return

    def _always_false(*args: Any, **kwargs: Any) -> bool:
        return False

    moe_integration._can_use_grouped_mm = _always_false
    print("[INFO] PERSONA_SFT_DISABLE_GROUPED_MM=true; forcing transformers MoE grouped_mm fallback path.")


def load_split(run_dir: Path, split_name: str, max_rows: int) -> pd.DataFrame:
    path = run_dir / f"{split_name}.parquet"
    df = pd.read_parquet(path)
    if max_rows > 0 and len(df) > max_rows:
        if split_name == "eval" and STRATIFIED_EVAL_CAP and "density_band" in df.columns:
            bands = ["b5_lower_visible_5_7", "b5_mid_visible_8_17", "b5_high_visible_18_plus"]
            per_band = max(1, max_rows // max(len(bands), 1))
            parts = []
            for band in bands:
                sub = df.loc[df["density_band"].eq(band)].head(per_band)
                if not sub.empty:
                    parts.append(sub)
            capped = pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()
            remaining = max_rows - len(capped)
            if remaining > 0:
                used_ids = set(capped["user_id"].tolist()) if not capped.empty else set()
                extra = df.loc[~df["user_id"].isin(used_ids)].head(remaining)
                capped = pd.concat([capped, extra], ignore_index=True)
            df = capped.head(max_rows).copy()
        else:
            df = df.head(max_rows).copy()
    return df.reset_index(drop=True)


def extract_primary_cuisine(target_text: str) -> str:
    try:
        target_obj = json.loads(target_text)
    except Exception:
        return "unknown"
    if isinstance(target_obj, dict) and "core_profile" in target_obj:
        target_obj = target_obj["core_profile"]
    stable = (target_obj.get("stable_preferences") or {}) if isinstance(target_obj, dict) else {}
    cuisines = stable.get("preferred_cuisines") or []
    if not isinstance(cuisines, list) or not cuisines:
        return "unknown"
    first = str(cuisines[0]).strip()
    return first or "unknown"


def build_train_sampling_weights(train_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    audit: dict[str, Any] = {
        "enabled": False,
        "density_balance_power": float(DENSITY_BALANCE_POWER),
        "cuisine_balance_power": float(CUISINE_BALANCE_POWER),
        "cuisine_balance_min_count": int(CUISINE_BALANCE_MIN_COUNT),
        "sample_weight_max_multiplier": float(SAMPLE_WEIGHT_MAX_MULTIPLIER),
    }
    if train_df.empty:
        audit["reason"] = "empty_train_df"
        return np.asarray([], dtype=np.float64), audit

    density_series = train_df.get("density_band", pd.Series(["unknown"] * len(train_df))).fillna("unknown").astype(str)
    cuisine_series = train_df["target_text"].map(extract_primary_cuisine).fillna("unknown").astype(str)
    density_counts = density_series.value_counts().to_dict()
    cuisine_counts_raw = cuisine_series.value_counts().to_dict()
    cuisine_counts = {
        key: max(int(value), int(CUISINE_BALANCE_MIN_COUNT))
        for key, value in cuisine_counts_raw.items()
    }
    max_density = max(density_counts.values()) if density_counts else 1
    max_cuisine = max(cuisine_counts.values()) if cuisine_counts else 1

    weights = []
    for density_band, cuisine_key in zip(density_series.tolist(), cuisine_series.tolist()):
        density_multiplier = (max_density / max(density_counts.get(density_band, 1), 1)) ** DENSITY_BALANCE_POWER
        cuisine_multiplier = (max_cuisine / max(cuisine_counts.get(cuisine_key, CUISINE_BALANCE_MIN_COUNT), 1)) ** CUISINE_BALANCE_POWER
        weight = float(density_multiplier * cuisine_multiplier)
        weight = min(weight, SAMPLE_WEIGHT_MAX_MULTIPLIER)
        weights.append(weight)

    weight_array = np.asarray(weights, dtype=np.float64)
    if len(weight_array) > 0:
        weight_array = weight_array / float(weight_array.mean())

    top_density = sorted(density_counts.items(), key=lambda item: item[1], reverse=True)[:8]
    top_cuisine = sorted(cuisine_counts_raw.items(), key=lambda item: item[1], reverse=True)[:12]
    audit.update(
        {
            "enabled": True,
            "rows": int(len(train_df)),
            "density_counts": {key: int(value) for key, value in density_counts.items()},
            "top_density_counts": [[key, int(value)] for key, value in top_density],
            "top_cuisine_counts": [[key, int(value)] for key, value in top_cuisine],
            "mean_weight": float(weight_array.mean()) if len(weight_array) else 0.0,
            "p50_weight": float(np.percentile(weight_array, 50)) if len(weight_array) else 0.0,
            "p90_weight": float(np.percentile(weight_array, 90)) if len(weight_array) else 0.0,
            "max_weight": float(weight_array.max()) if len(weight_array) else 0.0,
        }
    )
    return weight_array, audit


def build_chat_text(system_prompt: str, prompt_text: str, target_text: str) -> tuple[str, str]:
    prompt_only = (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{prompt_text}\n"
        "<|assistant|>\n"
    )
    full_text = prompt_only + target_text + (tokenizer_eos_token() or "")
    return prompt_only, full_text


def tokenizer_eos_token() -> str:
    return getattr(tokenizer_eos_token, "_value", "")


def set_tokenizer_eos_token(value: str) -> None:
    setattr(tokenizer_eos_token, "_value", value or "")


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def valid_json_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    ok = 0
    for text in texts:
        try:
            json.loads(text)
            ok += 1
        except Exception:
            continue
    return ok / len(texts)


def extract_key_match(pred_text: str, target_text: str, key_path: tuple[str, ...]) -> bool:
    try:
        pred_obj = json.loads(pred_text)
        target_obj = json.loads(target_text)
    except Exception:
        return False
    pred_value = pred_obj
    target_value = target_obj
    for key in key_path:
        if not isinstance(pred_value, dict) or key not in pred_value:
            return False
        if not isinstance(target_value, dict) or key not in target_value:
            return False
        pred_value = pred_value[key]
        target_value = target_value[key]
    return pred_value == target_value


def resolve_target_key_path(target_text: str, leaf_path: tuple[str, ...]) -> tuple[str, ...]:
    try:
        target_obj = json.loads(target_text)
    except Exception:
        return leaf_path
    if isinstance(target_obj, dict) and "core_profile" in target_obj:
        return ("core_profile",) + leaf_path
    return leaf_path


def sample_eval_generation_rows(eval_df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if eval_df.empty or limit <= 0:
        return []
    if "density_band" not in eval_df.columns:
        return eval_df.head(limit).to_dict(orient="records")
    groups = []
    bands = ["b5_lower_visible_5_7", "b5_mid_visible_8_17", "b5_high_visible_18_plus"]
    per_band = max(1, limit // max(len(bands), 1))
    for band in bands:
        sub = eval_df.loc[eval_df["density_band"].eq(band)].head(per_band)
        if not sub.empty:
            groups.append(sub)
    used = pd.concat(groups, ignore_index=True) if groups else eval_df.head(0).copy()
    remaining = limit - len(used)
    if remaining > 0:
        used_ids = set(used["user_id"].tolist()) if not used.empty else set()
        extra = eval_df.loc[~eval_df["user_id"].isin(used_ids)].head(remaining)
        used = pd.concat([used, extra], ignore_index=True)
    return used.head(limit).to_dict(orient="records")


def generate_eval_outputs(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_seq_len: int,
    max_new_tokens: int,
    batch_size: int,
    step_label: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generations = []
    model.eval()
    prompts = [build_chat_text(row["system_prompt"], row["prompt_text"], "")[0] for row in rows]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    def append_items(batch_rows: list[dict[str, Any]], batch_decoded: list[str]) -> None:
        for row, generated_text_raw in zip(batch_rows, batch_decoded):
            generated_text = extract_first_json_object(generated_text_raw) or generated_text_raw
            cuisines_path = resolve_target_key_path(row["target_text"], ("stable_preferences", "preferred_cuisines"))
            meals_path = resolve_target_key_path(row["target_text"], ("stable_preferences", "preferred_meals"))
            scenes_path = resolve_target_key_path(row["target_text"], ("stable_preferences", "preferred_scenes"))
            item = {
                "step_label": step_label,
                "user_id": row["user_id"],
                "density_band": row.get("density_band", "unknown"),
                "prediction_text_raw": generated_text_raw,
                "prediction_text": generated_text,
                "target_text": row["target_text"],
                "json_valid": False,
                "preferred_cuisines_exact": False,
                "preferred_meals_exact": False,
                "preferred_scenes_exact": False,
            }
            try:
                json.loads(generated_text)
                item["json_valid"] = True
            except Exception:
                item["json_valid"] = False
            item["preferred_cuisines_exact"] = extract_key_match(generated_text, row["target_text"], cuisines_path)
            item["preferred_meals_exact"] = extract_key_match(generated_text, row["target_text"], meals_path)
            item["preferred_scenes_exact"] = extract_key_match(generated_text, row["target_text"], scenes_path)
            generations.append(item)

    def generate_batch(batch_rows: list[dict[str, Any]], batch_prompts: list[str], current_batch_size: int) -> None:
        if not batch_rows:
            return
        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            padded_input_len = int(inputs["input_ids"].shape[1])
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = []
            for idx in range(len(batch_rows)):
                # For left-padded decoder-only batches, generation begins after the
                # padded batch input width, not after each row's non-pad token count.
                generated_ids = output_ids[idx][padded_input_len:]
                decoded.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
            append_items(batch_rows, decoded)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch_size <= 1 or len(batch_rows) <= 1:
                raise
            mid = max(1, len(batch_rows) // 2)
            next_batch_size = max(1, current_batch_size // 2)
            generate_batch(batch_rows[:mid], batch_prompts[:mid], next_batch_size)
            generate_batch(batch_rows[mid:], batch_prompts[mid:], next_batch_size)

    try:
        effective_batch_size = max(1, batch_size)
        for start in range(0, len(rows), effective_batch_size):
            end = min(start + effective_batch_size, len(rows))
            generate_batch(rows[start:end], prompts[start:end], effective_batch_size)
    finally:
        tokenizer.padding_side = original_padding_side

    audit = {
        "step_label": step_label,
        "rows": int(len(generations)),
        "generation_batch_size": int(max(1, batch_size)),
        "json_valid_rate": valid_json_rate([item["prediction_text"] for item in generations]),
        "json_start_rate": (
            float(np.mean([str(item["prediction_text"]).lstrip().startswith("{") for item in generations])) if generations else 0.0
        ),
        "json_extracted_rate": (
            float(np.mean([bool(item["prediction_text"]) and item["prediction_text"] != item["prediction_text_raw"] for item in generations]))
            if generations
            else 0.0
        ),
        "preferred_cuisines_exact_rate": float(np.mean([item["preferred_cuisines_exact"] for item in generations])) if generations else 0.0,
        "preferred_meals_exact_rate": float(np.mean([item["preferred_meals_exact"] for item in generations])) if generations else 0.0,
        "preferred_scenes_exact_rate": float(np.mean([item["preferred_scenes_exact"] for item in generations])) if generations else 0.0,
    }
    safe_json_write(output_dir / f"generation_audit_{step_label}.json", audit)
    safe_json_write(output_dir / f"generation_samples_{step_label}.json", generations[: min(24, len(generations))])
    return generations, audit


class PersonaTrainAuditCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer: Any,
        eval_rows: list[dict[str, Any]],
        run_dir: Path,
        max_seq_len: int,
        max_new_tokens: int,
        batch_size: int,
        audit_hook: str,
        save_log_history: bool,
        save_eval_generations: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.run_dir = run_dir
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.audit_hook = audit_hook
        self.save_log_history = save_log_history
        self.save_eval_generations = save_eval_generations
        self.eval_dir = run_dir / "eval_generations"
        self.log_path = run_dir / "log_history.jsonl"
        self.generated_steps: set[int] = set()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not self.save_log_history or not logs:
            return
        payload = {"global_step": int(state.global_step), "epoch": float(state.epoch or 0.0)}
        payload.update({key: (float(value) if isinstance(value, (np.floating, float)) else value) for key, value in logs.items()})
        append_jsonl(self.log_path, payload)

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> None:
        if self.audit_hook != "eval":
            return
        if not self.save_eval_generations or model is None or not self.eval_rows:
            return
        step_int = int(state.global_step)
        if step_int <= 0 or step_int in self.generated_steps:
            return
        self.generated_steps.add(step_int)
        step_label = f"step_{int(state.global_step):06d}"
        generate_eval_outputs(
            model=model,
            tokenizer=self.tokenizer,
            rows=self.eval_rows,
            max_seq_len=self.max_seq_len,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            step_label=step_label,
            output_dir=self.eval_dir,
        )

    def on_save(self, args: Any, state: Any, control: Any, model: Any | None = None, **kwargs: Any) -> None:
        if self.audit_hook != "save":
            return
        if not self.save_eval_generations or model is None or not self.eval_rows:
            return
        step_int = int(state.global_step)
        if step_int <= 0 or step_int in self.generated_steps:
            return
        self.generated_steps.add(step_int)
        step_label = f"step_{step_int:06d}"
        generate_eval_outputs(
            model=model,
            tokenizer=self.tokenizer,
            rows=self.eval_rows,
            max_seq_len=self.max_seq_len,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            step_label=step_label,
            output_dir=self.eval_dir,
        )


class PersonaTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        train_sampler_override: WeightedRandomSampler | None = None,
        skip_model_move: bool = False,
        **kwargs: Any,
    ) -> None:
        self.train_sampler_override = train_sampler_override
        self.skip_model_move = skip_model_move
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_sampler_override is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.train_sampler_override,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    def _move_model_to_device(self, model: Any, device: torch.device) -> None:
        if self.skip_model_move:
            print(f"[INFO] skipping Trainer._move_model_to_device; model already loaded on target device for {device}.")
            return
        return super()._move_model_to_device(model, device)


class PersonaDataCollator:
    def __init__(self, tokenizer: Any, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
        attention = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention = torch.nn.utils.rnn.pad_sequence(attention, batch_first=True, padding_value=0)
        if input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            labels = labels[:, : self.max_seq_len]
            attention = attention[:, : self.max_seq_len]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention,
        }


def build_tokenized_dataset(df: pd.DataFrame, tokenizer: Any, max_seq_len: int) -> Dataset:
    records = []
    for row in df.to_dict(orient="records"):
        prompt_only, full_text = build_chat_text(
            row["system_prompt"],
            row["prompt_text"],
            row["target_text"],
        )
        prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_len)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[min(len(prompt_ids), len(full_ids)) :]
        labels = labels[: len(full_ids)]
        attention = [1] * len(full_ids)
        records.append(
            {
                "user_id": row["user_id"],
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": attention,
                "target_text": row["target_text"],
            }
        )
    return Dataset.from_list(records)


def main() -> None:
    set_seed(SEED)
    if TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    maybe_disable_transformers_grouped_mm()
    input_run = resolve_persona_sft_input_run()
    train_df = load_split(input_run, "train", MAX_TRAIN_ROWS)
    eval_df = load_split(input_run, "eval", MAX_EVAL_ROWS)
    if train_df.empty or eval_df.empty:
        raise RuntimeError("persona SFT dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    set_tokenizer_eos_token(tokenizer.eos_token or "")

    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

    device_map = "auto" if USE_4BIT else None
    if not USE_4BIT and FORCE_SINGLE_GPU_DEVICE_MAP and torch.cuda.is_available():
        device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map=device_map,
        attn_implementation=ATTN_IMPL,
        low_cpu_mem_usage=True,
    )
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    sample_weights, sampler_audit = build_train_sampling_weights(train_df)
    train_sampler = None
    if USE_WEIGHTED_SAMPLER and len(sample_weights) == len(train_df) and len(sample_weights) > 0:
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
    train_ds = build_tokenized_dataset(train_df, tokenizer, MAX_SEQ_LEN)
    eval_ds = build_tokenized_dataset(eval_df, tokenizer, MAX_SEQ_LEN)

    resume_run_dir = resolve_optional_run_dir(RESUME_RUN_DIR)
    run_dir = resume_run_dir if resume_run_dir is not None else (OUTPUT_ROOT / now_run_id())
    resume_checkpoint = resolve_latest_checkpoint(run_dir) if resume_run_dir is not None else None
    if resume_run_dir is not None and resume_checkpoint is None:
        raise RuntimeError(
            f"PERSONA_SFT_RESUME_RUN_DIR was set but no checkpoint-* directory exists under {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_json_write(run_dir / "train_sampling_audit.json", sampler_audit)
    print(
        f"[RUN] run_dir={run_dir} "
        f"resume_run_dir={resume_run_dir if resume_run_dir is not None else '<none>'} "
        f"resume_checkpoint={resume_checkpoint if resume_checkpoint is not None else '<none>'}"
    )
    eval_rows = sample_eval_generation_rows(eval_df, GEN_EVAL_ROWS)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=USE_BF16,
        fp16=not USE_BF16,
        optim=OPTIMIZER,
        report_to=[],
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        seed=SEED,
        logging_dir=str(run_dir / "tb_logs"),
    )

    audit_callback = PersonaTrainAuditCallback(
        tokenizer=tokenizer,
        eval_rows=eval_rows,
        run_dir=run_dir,
        max_seq_len=MAX_SEQ_LEN,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        batch_size=GEN_BATCH_SIZE,
        audit_hook=GEN_AUDIT_HOOK,
        save_log_history=SAVE_LOG_HISTORY,
        save_eval_generations=SAVE_EVAL_GENERATIONS,
    )
    trainer = PersonaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PersonaDataCollator(tokenizer, MAX_SEQ_LEN),
        callbacks=[audit_callback],
        train_sampler_override=train_sampler,
        skip_model_move=SKIP_TRAINER_DEVICE_MOVE,
    )
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None)

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    generations, generation_audit = generate_eval_outputs(
        model=model,
        tokenizer=tokenizer,
        rows=eval_rows,
        max_seq_len=MAX_SEQ_LEN,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        batch_size=GEN_BATCH_SIZE,
        step_label="final",
        output_dir=run_dir,
    )

    if SAVE_LOG_HISTORY:
        safe_json_write(run_dir / "trainer_log_history.json", trainer.state.log_history)

    summary = {
        "train_rows": int(len(train_df)),
        "eval_rows": int(len(eval_df)),
        "base_model": BASE_MODEL,
        "use_4bit": bool(USE_4BIT),
        "use_bf16": bool(USE_BF16),
        "max_seq_len": int(MAX_SEQ_LEN),
        "attn_impl": ATTN_IMPL,
        "epochs": float(EPOCHS),
        "max_steps": int(MAX_STEPS),
        "learning_rate": float(LR),
        "gradient_accumulation_steps": int(GRAD_ACC),
        "lora_r": int(LORA_R),
        "lora_alpha": int(LORA_ALPHA),
        "lora_dropout": float(LORA_DROPOUT),
        "optimizer": OPTIMIZER,
        "use_weighted_sampler": bool(USE_WEIGHTED_SAMPLER),
        "sampling_audit_path": str(run_dir / "train_sampling_audit.json"),
        "gen_eval_rows": int(len(eval_rows)),
        "gen_batch_size": int(GEN_BATCH_SIZE),
        "gen_audit_hook": GEN_AUDIT_HOOK,
        "generation_audit": generation_audit,
    }
    run_meta = {
        "run_tag": RUN_TAG,
        "input_run_dir": str(input_run),
        "output_run_dir": str(run_dir),
        "resume_run_dir": str(resume_run_dir) if resume_run_dir is not None else "",
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
        "summary": summary,
    }
    safe_json_write(run_dir / "run_meta.json", run_meta)
    write_latest_run_pointer("11_persona_sft_models_latest", run_dir, {"run_tag": RUN_TAG, "summary": summary})
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
