from __future__ import annotations

import gc
import json
import math
import os
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

# Keep transformers on torch path in mixed TensorFlow/Keras env.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Reduce fragmentation risk on small VRAM GPUs.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    os.getenv("QLORA_EVAL_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128").strip()
    or "expandable_segments:True,max_split_size_mb:128",
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.modeling_utils as _tf_modeling_utils


from pipeline.project_paths import (
    env_or_project_path,
    normalize_legacy_project_path,
    project_path,
    read_latest_run_pointer,
    resolve_latest_run_pointer,
    write_latest_run_pointer,
)
from pipeline.qlora_prompting import (
    build_binary_prompt,
    build_binary_prompt_semantic,
    build_item_text,
    build_item_text_full_lite,
    build_item_text_semantic,
    build_item_text_sft_clean,
    build_pair_alignment_summary,
    build_user_text,
)
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context
from pipeline.stage11_text_features import (
    build_history_anchor_line,
    clean_text,
    extract_user_evidence_text,
    keyword_match_score,
)


RUN_TAG = "stage11_3_qlora_sidecar_eval"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"

INPUT_11_2_RUN_DIR = os.getenv("INPUT_11_2_RUN_DIR", "").strip()
INPUT_11_2_ROOT = env_or_project_path("INPUT_11_2_ROOT_DIR", "data/output/11_qlora_models")
INPUT_11_2_SUFFIX = "_stage11_2_qlora_train"
INPUT_11_DATA_RUN_DIR = os.getenv("INPUT_11_DATA_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path("OUTPUT_11_SIDECAR_ROOT_DIR", "data/output/11_qlora_sidecar_eval")
METRICS_PATH = env_or_project_path("METRICS_STAGE11_SIDECAR_PATH", "data/metrics/recsys_stage11_qlora_sidecar_results.csv")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
TOP_K = int(os.getenv("RANK_EVAL_TOP_K", "10").strip() or 10)
RERANK_TOPN = int(os.getenv("QLORA_RERANK_TOPN", "80").strip() or 80)
BLEND_ALPHA = float(os.getenv("QLORA_BLEND_ALPHA", "0.12").strip() or 0.12)
MAX_USERS_PER_BUCKET = int(os.getenv("QLORA_EVAL_MAX_USERS_PER_BUCKET", "1200").strip() or 1200)
MAX_ROWS_PER_BUCKET = int(os.getenv("QLORA_EVAL_MAX_ROWS_PER_BUCKET", "200000").strip() or 200000)
ROW_CAP_ORDERED = os.getenv("QLORA_EVAL_ROW_CAP_ORDERED", "false").strip().lower() == "true"
EVAL_PROFILE = os.getenv("QLORA_EVAL_PROFILE", "custom").strip().lower() or "custom"
EVAL_USER_COHORT_PATH_RAW = os.getenv("QLORA_EVAL_USER_COHORT_PATH", "").strip()
MAX_SEQ_LEN = int(os.getenv("QLORA_EVAL_MAX_SEQ_LEN", "768").strip() or 768)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_EVAL_PAD_TO_MULTIPLE_OF", "0").strip() or 0)
INFER_BATCH_SIZE = int(os.getenv("QLORA_EVAL_BATCH_SIZE", "12").strip() or 12)
PROMPT_CHUNK_ROWS = int(os.getenv("QLORA_EVAL_PROMPT_CHUNK_ROWS", "4096").strip() or 4096)
STREAM_LOG_ROWS = int(os.getenv("QLORA_EVAL_STREAM_LOG_ROWS", "4096").strip() or 4096)
ITER_COALESCE_PARTITIONS = int(os.getenv("QLORA_EVAL_ITER_COALESCE", "8").strip() or 8)
INTERMEDIATE_FLUSH_ROWS = int(os.getenv("QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS", "8192").strip() or 8192)
RANDOM_SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
USE_STAGE11_EVAL_SPLIT = os.getenv("QLORA_EVAL_USE_STAGE11_SPLIT", "true").strip().lower() == "true"
PROMPT_BUILD_MODE = os.getenv("QLORA_EVAL_PROMPT_BUILD_MODE", "driver").strip().lower() or "driver"
DRIVER_PROMPT_IMPL = os.getenv("QLORA_EVAL_DRIVER_PROMPT_IMPL", "apply").strip().lower() or "apply"
ARROW_TO_PANDAS = os.getenv("QLORA_EVAL_ARROW_TO_PANDAS", "false").strip().lower() == "true"
ARROW_FALLBACK = os.getenv("QLORA_EVAL_ARROW_FALLBACK", "false").strip().lower() == "true"
PRETOKENIZE_PROMPT_CHUNK = (
    os.getenv("QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK", "true").strip().lower() == "true"
)
PIN_MEMORY = os.getenv("QLORA_EVAL_PIN_MEMORY", "true").strip().lower() == "true"
NON_BLOCKING_H2D = os.getenv("QLORA_EVAL_NON_BLOCKING_H2D", "true").strip().lower() == "true"
RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER = int(os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER", "3").strip() or 3)
RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING = float(
    os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING", "4.5").strip() or 4.5
)
RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING = float(
    os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING", "4.0").strip() or 4.0
)
RICH_SFT_HISTORY_ANCHOR_MAX_CHARS = int(os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_MAX_CHARS", "180").strip() or 180)

ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "true").strip().lower() == "true"
STAGE09_GATE_METRICS_PATH = Path(
    os.getenv(
        "QLORA_STAGE09_GATE_METRICS_PATH",
        project_path("data/metrics/stage09_recall_audit_summary_latest.csv").as_posix(),
    ).strip()
)
STAGE09_GATE_MIN_TRUTH_IN_PRETRIM = float(os.getenv("QLORA_GATE_MIN_TRUTH_IN_PRETRIM", "0.82").strip() or 0.82)
STAGE09_GATE_MAX_PRETRIM_CUT_LOSS = float(os.getenv("QLORA_GATE_MAX_PRETRIM_CUT_LOSS", "0.08").strip() or 0.08)
STAGE09_GATE_MAX_HARD_MISS = float(os.getenv("QLORA_GATE_MAX_HARD_MISS", "0.10").strip() or 0.10)

ENABLE_SHORT_TEXT_SUMMARY = os.getenv("QLORA_ENABLE_SHORT_TEXT_SUMMARY", "true").strip().lower() == "true"
ENABLE_RAW_REVIEW_TEXT = os.getenv("QLORA_ENABLE_RAW_REVIEW_TEXT", "true").strip().lower() == "true"
REVIEW_TABLE_PATH = Path(
    os.getenv(
        "QLORA_REVIEW_TABLE_PATH",
        project_path("data/parquet/yelp_academic_dataset_review").as_posix(),
    ).strip()
)
USER_REVIEW_TOPN = int(os.getenv("QLORA_USER_REVIEW_TOPN", "2").strip() or 2)
ITEM_REVIEW_TOPN = int(os.getenv("QLORA_ITEM_REVIEW_TOPN", "2").strip() or 2)
REVIEW_SNIPPET_MAX_CHARS = int(os.getenv("QLORA_REVIEW_SNIPPET_MAX_CHARS", "220").strip() or 220)
USER_EVIDENCE_MAX_CHARS = int(os.getenv("QLORA_USER_EVIDENCE_MAX_CHARS", "260").strip() or 260)
ITEM_EVIDENCE_MAX_CHARS = int(os.getenv("QLORA_ITEM_EVIDENCE_MAX_CHARS", "260").strip() or 260)
ITEM_EVIDENCE_SCORE_UDF_MODE = os.getenv("QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE", "python").strip().lower() or "python"
REVIEW_BASE_CACHE_ENABLED = os.getenv("QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED", "false").strip().lower() == "true"
REVIEW_BASE_CACHE_ROOT = env_or_project_path(
    "QLORA_EVAL_REVIEW_BASE_CACHE_ROOT",
    "data/output/11_qlora_sidecar_eval_cache/review_base",
)

USE_4BIT = os.getenv("QLORA_EVAL_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("QLORA_EVAL_USE_BF16", "true").strip().lower() == "true"
QWEN35_MAMBA_SSM_DTYPE = os.getenv("QLORA_QWEN35_MAMBA_SSM_DTYPE", "auto").strip().lower()
ATTN_IMPLEMENTATION = os.getenv("QLORA_EVAL_ATTN_IMPLEMENTATION", "").strip().lower()
TRUST_REMOTE_CODE = os.getenv("QLORA_TRUST_REMOTE_CODE", "true").strip().lower() == "true"
DISABLE_ALLOC_WARMUP = os.getenv("QLORA_DISABLE_ALLOC_WARMUP", "true").strip().lower() == "true"
EVAL_DEVICE_MAP = os.getenv("QLORA_EVAL_DEVICE_MAP", "auto").strip() or "auto"
EVAL_MAX_MEMORY_CUDA = os.getenv("QLORA_EVAL_MAX_MEMORY_CUDA", "").strip()
EVAL_MAX_MEMORY_CPU = os.getenv("QLORA_EVAL_MAX_MEMORY_CPU", "").strip()
EVAL_OFFLOAD_FOLDER = os.getenv("QLORA_EVAL_OFFLOAD_FOLDER", "").strip()
EVAL_OFFLOAD_STATE_DICT = os.getenv("QLORA_EVAL_OFFLOAD_STATE_DICT", "true").strip().lower() == "true"
DISABLE_PARALLEL_LOADING = os.getenv("QLORA_EVAL_DISABLE_PARALLEL_LOADING", "true").strip().lower() == "true"
PARALLEL_LOADING_WORKERS = int(os.getenv("QLORA_EVAL_PARALLEL_LOADING_WORKERS", "1").strip() or 1)
LOAD_MODEL_BEFORE_SPARK = os.getenv("QLORA_EVAL_LOAD_MODEL_BEFORE_SPARK", "true").strip().lower() == "true"
INVERT_PROB = os.getenv("QLORA_INVERT_PROB", "false").strip().lower() == "true"
PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "full").strip().lower() or "full"  # "full" | "full_lite" | "semantic" | "sft_clean"
QWEN35_NO_THINK = os.getenv("QLORA_EVAL_QWEN35_NO_THINK", "false").strip().lower() == "true"

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
SPARK_DRIVER_BIND_ADDRESS = os.getenv("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1").strip() or "127.0.0.1"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
SPARK_RPC_MESSAGE_MAX_SIZE = os.getenv("SPARK_RPC_MESSAGE_MAX_SIZE", "1024").strip() or "1024"
SPARK_PYTHON_WORKER_REUSE = os.getenv("SPARK_PYTHON_WORKER_REUSE", "true").strip().lower() == "true"
SPARK_PYTHON_WORKER_MEMORY = os.getenv("SPARK_PYTHON_WORKER_MEMORY", "2g").strip() or "2g"

_SPARK_TMP_CTX: SparkTmpContext | None = None


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


def is_qwen35_model_type(model_type: str) -> bool:
    mt = str(model_type or "").strip().lower()
    return mt.startswith("qwen3_5")


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
        # PeftModelForCausalLM.forward hides this kwarg behind **kwargs,
        # but it is forwarded to the underlying Qwen3.5 base model.
        return "logits_to_keep"

    return ""


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage09_run(stage11_meta: dict[str, Any] | None = None, stage11_data_run: Path | None = None) -> Path:
    def _extract_stage09_path(payload: dict[str, Any] | None) -> Path | None:
        if not isinstance(payload, dict):
            return None
        for key in ("source_stage09_run", "source_run_09"):
            raw = str(payload.get(key, "")).strip()
            if not raw:
                continue
            p = normalize_legacy_project_path(raw)
            if p.exists():
                return p
        return None

    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    if stage11_data_run is not None:
        meta_path = stage11_data_run / "run_meta.json"
        if meta_path.exists():
            try:
                ds_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                p = _extract_stage09_path(ds_meta)
                if p is not None:
                    return p
            except Exception:
                pass
    if stage11_meta is not None:
        p = _extract_stage09_path(stage11_meta)
        if p is not None:
            return p
        ds_raw = str(stage11_meta.get("source_stage11_dataset_run", "")).strip()
        if ds_raw:
            ds_path = normalize_legacy_project_path(ds_raw)
            meta_path = ds_path / "run_meta.json"
            if meta_path.exists():
                try:
                    ds_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    p = _extract_stage09_path(ds_meta)
                    if p is not None:
                        return p
                except Exception:
                    pass
    latest_11_1 = read_latest_run_pointer("stage11_1_qlora_build_dataset")
    p = _extract_stage09_path(latest_11_1)
    if p is not None:
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def resolve_stage11_2_run() -> Path:
    def _is_complete_run(run_dir: Path) -> bool:
        return run_dir.is_dir() and (run_dir / "run_meta.json").exists() and (run_dir / "adapter").is_dir()

    if INPUT_11_2_RUN_DIR:
        p = Path(INPUT_11_2_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_11_2_RUN_DIR not found: {p}")
        if not _is_complete_run(p):
            raise FileNotFoundError(
                f"INPUT_11_2_RUN_DIR is incomplete (need run_meta.json + adapter dir): {p}"
            )
        return p
    pinned = resolve_latest_run_pointer("stage11_2_qlora_train")
    if pinned is not None and _is_complete_run(pinned):
        return pinned
    runs = [p for p in INPUT_11_2_ROOT.iterdir() if p.is_dir() and p.name.endswith(INPUT_11_2_SUFFIX)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if _is_complete_run(r):
            return r
    if not runs:
        raise FileNotFoundError(f"no run found in {INPUT_11_2_ROOT} with suffix={INPUT_11_2_SUFFIX}")
    checked = ", ".join([r.name for r in runs[:3]])
    raise FileNotFoundError(
        "no complete stage11_2 run found "
        f"(need run_meta.json + adapter dir). checked_latest={checked}"
    )


def resolve_stage11_data_run(stage11_meta: dict[str, Any]) -> Path | None:
    raw = INPUT_11_DATA_RUN_DIR.strip() or str(stage11_meta.get("source_stage11_dataset_run", "")).strip()
    if not raw:
        pinned = resolve_latest_run_pointer("stage11_1_qlora_build_dataset")
        if pinned is not None:
            raw = pinned.as_posix()
    if not raw:
        if USE_STAGE11_EVAL_SPLIT:
            raise FileNotFoundError(
                "stage11 dataset run missing; set INPUT_11_DATA_RUN_DIR or ensure "
                "source_stage11_dataset_run exists in stage11_2 run_meta.json."
            )
        return None
    p = Path(raw)
    if not p.exists():
        if USE_STAGE11_EVAL_SPLIT:
            raise FileNotFoundError(f"stage11 dataset run not found: {p}")
        return None
    return p


def enforce_stage09_gate(source_09: Path, buckets: list[int]) -> dict[str, dict[str, float]]:
    if not ENFORCE_STAGE09_GATE:
        return {}
    if not STAGE09_GATE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "stage09 gate metrics csv not found: "
            f"{STAGE09_GATE_METRICS_PATH}. Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
        )
    mdf = pd.read_csv(STAGE09_GATE_METRICS_PATH)
    if mdf.empty:
        raise RuntimeError(f"stage09 gate metrics csv is empty: {STAGE09_GATE_METRICS_PATH}")

    src_key = source_09.name
    if "source_run_09" in mdf.columns:
        s = mdf["source_run_09"].astype(str)
        scoped = mdf[(s == src_key) | (s == str(source_09)) | s.str.endswith(src_key)].copy()
    else:
        scoped = mdf.copy()
    if scoped.empty:
        raise RuntimeError(
            "no stage09 gate rows matched current source run: "
            f"source={source_09}. csv={STAGE09_GATE_METRICS_PATH}"
        )

    out: dict[str, dict[str, float]] = {}
    for b in sorted(list(set([int(x) for x in buckets]))):
        rows = scoped[scoped["bucket"].astype(int) == int(b)].copy()
        if rows.empty:
            raise RuntimeError(f"stage09 gate missing bucket={b} in {STAGE09_GATE_METRICS_PATH}")
        if "run_id" in rows.columns:
            rows = rows.sort_values("run_id")
        row = rows.iloc[-1]

        truth_in_pretrim = float(row.get("truth_in_pretrim", float("nan")))
        pretrim_cut_loss = float(row.get("pretrim_cut_loss", row.get("cut_loss", float("nan"))))
        hard_miss = float(row.get("hard_miss", float("nan")))
        if not (math.isfinite(truth_in_pretrim) and math.isfinite(pretrim_cut_loss) and math.isfinite(hard_miss)):
            raise RuntimeError(f"stage09 gate has non-finite metrics for bucket={b}: {dict(row)}")

        if truth_in_pretrim < float(STAGE09_GATE_MIN_TRUTH_IN_PRETRIM):
            raise RuntimeError(
                f"stage09 gate fail bucket={b}: truth_in_pretrim={truth_in_pretrim:.6f} "
                f"< {float(STAGE09_GATE_MIN_TRUTH_IN_PRETRIM):.6f}"
            )
        if pretrim_cut_loss > float(STAGE09_GATE_MAX_PRETRIM_CUT_LOSS):
            raise RuntimeError(
                f"stage09 gate fail bucket={b}: pretrim_cut_loss={pretrim_cut_loss:.6f} "
                f"> {float(STAGE09_GATE_MAX_PRETRIM_CUT_LOSS):.6f}"
            )
        if hard_miss > float(STAGE09_GATE_MAX_HARD_MISS):
            raise RuntimeError(
                f"stage09 gate fail bucket={b}: hard_miss={hard_miss:.6f} "
                f"> {float(STAGE09_GATE_MAX_HARD_MISS):.6f}"
            )
        out[str(b)] = {
            "truth_in_pretrim": truth_in_pretrim,
            "pretrim_cut_loss": pretrim_cut_loss,
            "hard_miss": hard_miss,
        }
    print(
        f"[GATE] stage09 pass source={source_09.name} buckets={sorted(list(out.keys()))} "
        f"metrics_csv={STAGE09_GATE_METRICS_PATH}"
    )
    return out


def load_eval_users_from_stage11_data(spark: SparkSession, stage11_data_run: Path, bucket: int) -> DataFrame:
    bdir = stage11_data_run / f"bucket_{int(bucket)}" / "all_parquet"
    if not bdir.exists():
        raise FileNotFoundError(f"stage11 all_parquet missing for bucket={bucket}: {bdir}")
    sdf = spark.read.parquet(bdir.as_posix())
    cols = set(sdf.columns)
    if "user_idx" not in cols or "split" not in cols:
        raise RuntimeError(f"stage11 all_parquet missing required columns user_idx/split: {bdir}")
    return sdf.filter(F.col("split") == F.lit("eval")).select("user_idx").dropDuplicates(["user_idx"])


def resolve_eval_user_cohort_path() -> Path | None:
    raw = str(EVAL_USER_COHORT_PATH_RAW or "").strip()
    if not raw:
        return None
    p = normalize_legacy_project_path(raw)
    if p.exists():
        return p
    raise FileNotFoundError(f"QLORA_EVAL_USER_COHORT_PATH not found: {p}")


def load_eval_users_from_cohort(spark: SparkSession, cohort_path: Path) -> tuple[DataFrame, int]:
    suffix = cohort_path.suffix.strip().lower()
    if cohort_path.is_dir() or suffix == ".parquet":
        pdf = pd.read_parquet(cohort_path.as_posix(), columns=["user_idx"])
    elif suffix == ".csv":
        pdf = pd.read_csv(cohort_path.as_posix(), usecols=["user_idx"])
    else:
        raise ValueError(
            f"Unsupported cohort file format: {cohort_path}. Expected .csv or .parquet."
        )
    if "user_idx" not in pdf.columns:
        raise RuntimeError(f"cohort file missing user_idx column: {cohort_path}")
    out = (
        pdf[["user_idx"]]
        .dropna()
        .astype({"user_idx": "int64"})
        .drop_duplicates(subset=["user_idx"])
        .sort_values("user_idx", kind="stable")
        .reset_index(drop=True)
    )
    return spark.createDataFrame(out), int(len(out))


def resolve_stage09_meta_path(raw_path: str) -> Path:
    p = normalize_legacy_project_path(str(raw_path or "").strip())
    if p.exists():
        return p
    return p


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=True,
    )
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} local={_SPARK_TMP_CTX.spark_local_dir} "
        f"py_temp={_SPARK_TMP_CTX.py_temp_dir} cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    worker_py = os.getenv("PYSPARK_PYTHON", sys.executable).strip() or sys.executable
    driver_py = os.getenv("PYSPARK_DRIVER_PYTHON", sys.executable).strip() or sys.executable
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    existing_pythonpath = [part for part in os.getenv("PYTHONPATH", "").split(os.pathsep) if part]
    pythonpath_parts: list[str] = []
    for candidate in [script_dir.as_posix(), repo_root.as_posix(), *existing_pythonpath]:
        if candidate and candidate not in pythonpath_parts:
            pythonpath_parts.append(candidate)
    pythonpath_value = os.pathsep.join(pythonpath_parts)
    os.environ["PYTHONPATH"] = pythonpath_value
    print(
        f"[SPARK] master={SPARK_MASTER} pyspark_python={worker_py} "
        f"pyspark_driver_python={driver_py} parallelism={SPARK_DEFAULT_PARALLELISM} "
        f"shuffle={SPARK_SQL_SHUFFLE_PARTITIONS} arrow_to_pandas={ARROW_TO_PANDAS}"
    )
    return (
        SparkSession.builder.appName(RUN_TAG)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.host", SPARK_DRIVER_HOST)
        .config("spark.driver.bindAddress", SPARK_DRIVER_BIND_ADDRESS)
        .config("spark.local.dir", str(_SPARK_TMP_CTX.spark_local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.python.worker.reuse", "true" if SPARK_PYTHON_WORKER_REUSE else "false")
        .config("spark.pyspark.python", worker_py)
        .config("spark.pyspark.driver.python", driver_py)
        .config("spark.executorEnv.PYTHONPATH", pythonpath_value)
        .config("spark.driverEnv.PYTHONPATH", pythonpath_value)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.rpc.message.maxSize", SPARK_RPC_MESSAGE_MAX_SIZE)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true" if ARROW_TO_PANDAS else "false")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true" if ARROW_FALLBACK else "false")
        .config("spark.python.worker.memory", SPARK_PYTHON_WORKER_MEMORY)
        .config("spark.task.maxFailures", "4")
        .config("spark.io.compression.codec", "lz4")
        .config("spark.file.transferTo", "false")
        .config("spark.shuffle.file.buffer", "32k")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def pick_candidate_file(bucket_dir: Path) -> Path:
    candidates = [
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates_pretrim.parquet",
        "candidates.parquet",
    ]
    for name in candidates:
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def _read_csv_subset(path: str, wanted_cols: list[str]) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=wanted_cols)
    cols = list(pd.read_csv(path, nrows=0).columns)
    usecols = [c for c in wanted_cols if c in cols]
    if not usecols:
        return pd.DataFrame(columns=wanted_cols)
    out = pd.read_csv(path, usecols=usecols)
    for c in wanted_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[wanted_cols]


def _str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and not np.isfinite(v):
        return ""
    s = str(v)
    return s if s.lower() != "nan" else ""


def _float_or_zero(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
        return float(f) if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def build_profile_lookup(profile_pdf: pd.DataFrame) -> dict[str, tuple[str, str, str, float]]:
    out: dict[str, tuple[str, str, str, float]] = {}
    if profile_pdf.empty:
        return out
    for r in profile_pdf.itertuples(index=False):
        uid = _str_or_empty(getattr(r, "user_id", ""))
        if not uid:
            continue
        out[uid] = (
            _str_or_empty(getattr(r, "profile_text", "")),
            _str_or_empty(getattr(r, "profile_top_pos_tags", "")),
            _str_or_empty(getattr(r, "profile_top_neg_tags", "")),
            _float_or_zero(getattr(r, "profile_confidence", 0.0)),
        )
    return out


def build_item_lookup(item_sem_pdf: pd.DataFrame) -> dict[str, tuple[str, str, float, float]]:
    out: dict[str, tuple[str, str, float, float]] = {}
    if item_sem_pdf.empty:
        return out
    for r in item_sem_pdf.itertuples(index=False):
        bid = _str_or_empty(getattr(r, "business_id", ""))
        if not bid:
            continue
        out[bid] = (
            _str_or_empty(getattr(r, "top_pos_tags", "")),
            _str_or_empty(getattr(r, "top_neg_tags", "")),
            _float_or_zero(getattr(r, "semantic_score", 0.0)),
            _float_or_zero(getattr(r, "semantic_confidence", 0.0)),
        )
    return out


def load_profile_item_tables(stage09_meta: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    item_sem_path = str(stage09_meta.get("item_semantic_features", "")).strip()

    pcols = ["user_id", "profile_text_short", "profile_text", "profile_top_pos_tags", "profile_top_neg_tags", "profile_confidence"]
    icols = ["business_id", "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence"]

    profile = _read_csv_subset(profile_path, pcols)
    profile["profile_text"] = profile["profile_text_short"].fillna(profile["profile_text"]).fillna("")
    profile = profile[["user_id", "profile_text", "profile_top_pos_tags", "profile_top_neg_tags", "profile_confidence"]].drop_duplicates(
        subset=["user_id"]
    )

    item_sem = _read_csv_subset(item_sem_path, icols)
    item_sem = item_sem.drop_duplicates(subset=["business_id"])
    return profile, item_sem


def build_user_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    schema = (
        "user_id string, profile_text string, profile_text_evidence string, "
        "profile_top_pos_tags string, profile_top_neg_tags string, profile_confidence double"
    )
    if not profile_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "user_id",
            F.coalesce(F.col("profile_text_short"), F.col("profile_text"), F.lit("")).alias("profile_text"),
            F.coalesce(F.col("profile_text_long"), F.col("profile_text"), F.col("profile_text_short"), F.lit("")).alias("profile_text_evidence"),
            F.coalesce(F.col("profile_top_pos_tags"), F.lit("")).alias("profile_top_pos_tags"),
            F.coalesce(F.col("profile_top_neg_tags"), F.lit("")).alias("profile_top_neg_tags"),
            F.col("profile_confidence").cast("double").alias("profile_confidence"),
        )
        .dropDuplicates(["user_id"])
    )


def build_item_sem_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    sem_path = str(stage09_meta.get("item_semantic_features", "")).strip()
    schema = "business_id string, top_pos_tags string, top_neg_tags string, semantic_score double, semantic_confidence double"
    if not sem_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(sem_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("top_pos_tags"), F.lit("")).alias("top_pos_tags"),
            F.coalesce(F.col("top_neg_tags"), F.lit("")).alias("top_neg_tags"),
            F.col("semantic_score").cast("double").alias("semantic_score"),
            F.col("semantic_confidence").cast("double").alias("semantic_confidence"),
        )
        .dropDuplicates(["business_id"])
    )


def build_business_meta_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("cluster_profile_csv", "")).strip()
    schema = "business_id string, name string, city string, categories string, primary_category string"
    if not profile_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    raw = spark.read.option("header", "true").csv(p.as_posix())
    primary_expr = F.coalesce(
        F.col("final_l2_label_top1"),
        F.col("final_l1_label"),
        F.lit(""),
    )
    return (
        raw.select(
            "business_id",
            F.coalesce(F.col("name"), F.lit("")).alias("name"),
            F.coalesce(F.col("city"), F.lit("")).alias("city"),
            F.coalesce(F.col("categories"), F.lit("")).alias("categories"),
            primary_expr.alias("primary_category"),
        )
        .dropDuplicates(["business_id"])
    )


def build_user_evidence_udf() -> Any:
    def _mk(profile_text: Any, profile_text_evidence: Any) -> str:
        return extract_user_evidence_text(profile_text, profile_text_evidence, max_chars=USER_EVIDENCE_MAX_CHARS)

    return F.udf(_mk, "string")


def build_pair_alignment_udf() -> Any:
    def _mk(user_pos: Any, user_neg: Any, item_pos: Any, item_neg: Any) -> str:
        return build_pair_alignment_summary(user_pos, user_neg, item_pos, item_neg)

    return F.udf(_mk, "string")


def build_history_anchor_text_udf() -> Any:
    def _mk(entries: Any, current_business_id: Any) -> str:
        current = str(current_business_id or "").strip()
        parts: list[str] = []
        seen: set[str] = set()
        for row in list(entries or []):
            try:
                biz = str(row["business_id"] or "").strip()
                anchor_text = str(row["anchor_text"] or "").strip()
            except Exception:
                try:
                    biz = str(getattr(row, "business_id", "") or "").strip()
                    anchor_text = str(getattr(row, "anchor_text", "") or "").strip()
                except Exception:
                    biz = ""
                    anchor_text = ""
            if not anchor_text or (current and biz == current):
                continue
            key = anchor_text.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(anchor_text)
            if len(parts) >= max(1, int(RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER)):
                break
        return clean_text(" || ".join(parts), max_chars=max(80, int(RICH_SFT_HISTORY_ANCHOR_MAX_CHARS)))

    return F.udf(_mk, "string")


def build_item_review_evidence(
    spark: SparkSession,
    bucket_dir: Path,
    cand_df: DataFrame,
    review_base_df: DataFrame,
) -> DataFrame:
    schema = "user_idx int, business_id string, item_evidence_text string"
    empty_df = spark.createDataFrame([], schema)
    hist_path = bucket_dir / "train_history.parquet"
    if not hist_path.exists():
        print(f"[WARN] train_history.parquet missing, skip item evidence features: {hist_path}")
        return empty_df

    user_cutoff = (
        spark.read.parquet(hist_path.as_posix())
        .select("user_idx", "test_ts")
        .filter(F.col("test_ts").isNotNull())
        .groupBy("user_idx")
        .agg(F.max("test_ts").alias("test_ts"))
    )
    item_base = (
        cand_df.select("user_idx", "business_id")
        .dropDuplicates(["user_idx", "business_id"])
        .join(user_cutoff, on="user_idx", how="inner")
    )
    item_rows = (
        item_base.join(review_base_df, on="business_id", how="inner")
        .filter(F.col("review_ts") < F.col("test_ts"))
        .dropDuplicates(["user_idx", "business_id", "snippet"])
    )
    w_item = Window.partitionBy("user_idx", "business_id").orderBy(F.col("snippet_score").desc(), F.col("review_ts").desc())
    item_top = item_rows.withColumn("rn", F.row_number().over(w_item)).filter(F.col("rn") <= F.lit(max(1, int(ITEM_REVIEW_TOPN))))
    return (
        item_top.groupBy("user_idx", "business_id")
        .agg(F.concat_ws(" || ", F.collect_list("snippet")).alias("item_evidence_text"))
        .select("user_idx", "business_id", "item_evidence_text")
    )


def build_review_base_df(
    spark: SparkSession,
    stage09_run: Path,
    bucket_dir: Path,
    cand_path: Path,
    item_sem_df: DataFrame,
) -> DataFrame:
    schema = "business_id string, review_ts timestamp, snippet string, snippet_score double"
    empty_df = spark.createDataFrame([], schema)
    if not ENABLE_RAW_REVIEW_TEXT:
        return empty_df
    if not REVIEW_TABLE_PATH.exists():
        print(f"[WARN] review table not found, skip item evidence features: {REVIEW_TABLE_PATH}")
        return empty_df

    cache_path = (
        REVIEW_BASE_CACHE_ROOT
        / stage09_run.name
        / f"bucket_{bucket_dir.name.split('_')[-1]}"
        / f"rerank_topn_{int(RERANK_TOPN)}_snippet_{int(REVIEW_SNIPPET_MAX_CHARS)}"
    )
    if bool(REVIEW_BASE_CACHE_ENABLED) and cache_path.exists():
        print(f"[CACHE] reuse review base cache: {cache_path}")
        return spark.read.parquet(cache_path.as_posix())

    max_chars = max(40, int(REVIEW_SNIPPET_MAX_CHARS))
    cand_business = (
        spark.read.parquet(cand_path.as_posix())
        .filter(F.col("pre_rank") <= F.lit(int(RERANK_TOPN)))
        .select("business_id")
        .dropDuplicates(["business_id"])
    )
    rvw = (
        spark.read.parquet(REVIEW_TABLE_PATH.as_posix())
        .select("business_id", "date", "text")
        .withColumn("review_ts", F.to_timestamp(F.col("date")))
        .withColumn("text_clean", F.regexp_replace(F.regexp_replace(F.coalesce(F.col("text"), F.lit("")), r"[\r\n]+", " "), r"\s+", " "))
        .filter(F.col("review_ts").isNotNull() & (F.length(F.col("text_clean")) > F.lit(0)))
        .join(F.broadcast(cand_business), on="business_id", how="semi")
        .join(item_sem_df.select("business_id", "top_pos_tags", "top_neg_tags"), on="business_id", how="left")
        .withColumn("snippet", F.substring(F.col("text_clean"), 1, int(max_chars)))
    )
    if ITEM_EVIDENCE_SCORE_UDF_MODE == "pandas":
        @F.pandas_udf("double")
        def score_udf(txt: pd.Series, pos: pd.Series, neg: pd.Series) -> pd.Series:
            return pd.Series(
                [float(keyword_match_score(t, p, n)) for t, p, n in zip(txt, pos, neg)],
                dtype="float64",
            )
    else:
        score_udf = F.udf(lambda txt, pos, neg: float(keyword_match_score(txt, pos, neg)), "double")

    review_base = (
        rvw.withColumn("tag_hit_score", score_udf(F.col("snippet"), F.col("top_pos_tags"), F.col("top_neg_tags")))
        .withColumn("snippet_len", F.length(F.col("snippet")).cast("double"))
        .withColumn(
            "snippet_score",
            F.col("tag_hit_score") * F.lit(10.0)
            + F.when(F.col("snippet_len") >= F.lit(80.0), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .select("business_id", "review_ts", "snippet", "snippet_score")
        .dropDuplicates(["business_id", "review_ts", "snippet"])
    )
    if bool(REVIEW_BASE_CACHE_ENABLED):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        review_base.write.mode("overwrite").parquet(cache_path.as_posix())
        print(f"[CACHE] write review base cache: {cache_path}")
        return spark.read.parquet(cache_path.as_posix())
    return review_base


def build_history_anchor_df(
    spark: SparkSession,
    bucket_dir: Path,
    users_df: DataFrame,
    cand_df: DataFrame,
    item_sem_df: DataFrame,
    business_meta_df: DataFrame,
) -> DataFrame:
    schema = (
        "user_idx int, "
        "history_anchor_entries array<struct<anchor_rank:int,business_id:string,anchor_text:string>>, "
        "history_anchor_available int"
    )
    hist_path = bucket_dir / "train_history.parquet"
    if PROMPT_MODE not in {"sft_clean", "full_lite"} or not hist_path.exists():
        return spark.createDataFrame([], schema)

    anchor_line_udf = F.udf(
        lambda n, c, pc, tags, rating: build_history_anchor_line(n, c, pc, tags, rating, max_chars=110),
        "string",
    )
    fallback_min = min(
        float(RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING),
        float(RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING),
    )
    anchor_keep = max(1, int(RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER) + 2)
    item_business_lookup = cand_df.select("item_idx", "business_id").dropDuplicates(["item_idx"])
    hist_anchor_base = (
        spark.read.parquet(hist_path.as_posix())
        .join(users_df, on="user_idx", how="inner")
        .filter(
            F.col("item_idx").isNotNull()
            & F.col("hist_rating").isNotNull()
            & (F.col("hist_rating").cast("double") >= F.lit(float(fallback_min)))
        )
        .select(
            "user_idx",
            "item_idx",
            F.col("hist_rating").cast("double").alias("hist_rating"),
            F.col("hist_ts").alias("hist_ts"),
        )
        .dropDuplicates(["user_idx", "item_idx"])
        .join(item_business_lookup, on="item_idx", how="left")
        .filter(F.col("business_id").isNotNull())
        .join(business_meta_df, on="business_id", how="left")
        .join(item_sem_df.select("business_id", "top_pos_tags"), on="business_id", how="left")
        .fillna(
            {
                "name": "",
                "city": "",
                "categories": "",
                "primary_category": "",
                "top_pos_tags": "",
            }
        )
        .withColumn(
            "anchor_pref",
            F.when(
                F.col("hist_rating") >= F.lit(float(RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING)),
                F.lit(2),
            ).otherwise(F.lit(1)),
        )
        .withColumn("anchor_city_norm", F.lower(F.trim(F.coalesce(F.col("city"), F.lit("")))))
        .withColumn("anchor_cat_norm", F.lower(F.trim(F.coalesce(F.col("primary_category"), F.lit("")))))
        .withColumn(
            "anchor_group",
            F.when(
                (F.length(F.col("anchor_city_norm")) > F.lit(0)) | (F.length(F.col("anchor_cat_norm")) > F.lit(0)),
                F.concat_ws("|", F.col("anchor_city_norm"), F.col("anchor_cat_norm")),
            ).otherwise(F.col("business_id")),
        )
        .withColumn(
            "anchor_text",
            anchor_line_udf(
                F.col("name"),
                F.col("city"),
                F.col("primary_category"),
                F.col("top_pos_tags"),
                F.col("hist_rating"),
            ),
        )
        .filter(F.length(F.col("anchor_text")) > F.lit(0))
    )
    w_anchor_group = Window.partitionBy("user_idx", "anchor_group").orderBy(
        F.col("anchor_pref").desc(),
        F.col("hist_rating").desc(),
        F.col("hist_ts").desc_nulls_last(),
    )
    hist_anchor_ranked = (
        hist_anchor_base.withColumn("_anchor_group_rn", F.row_number().over(w_anchor_group))
        .filter(F.col("_anchor_group_rn") <= F.lit(1))
    )
    w_anchor_user = Window.partitionBy("user_idx").orderBy(
        F.col("anchor_pref").desc(),
        F.col("hist_rating").desc(),
        F.col("hist_ts").desc_nulls_last(),
        F.col("business_id").asc(),
    )
    return (
        hist_anchor_ranked.withColumn("anchor_rank", F.row_number().over(w_anchor_user))
        .filter(F.col("anchor_rank") <= F.lit(int(anchor_keep)))
        .groupBy("user_idx")
        .agg(
            F.sort_array(
                F.collect_list(
                    F.struct(
                        F.col("anchor_rank").cast("int").alias("anchor_rank"),
                        F.col("business_id").alias("business_id"),
                        F.col("anchor_text").alias("anchor_text"),
                    )
                )
            ).alias("history_anchor_entries"),
            F.count("*").cast("int").alias("history_anchor_available"),
        )
    )


def build_prompt_udf() -> Any:
    def _mk(
        profile_text: Any,
        profile_text_evidence: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        pair_evidence_summary: Any,
        name: Any,
        city: Any,
        categories: Any,
        primary_category: Any,
        top_pos_tags: Any,
        top_neg_tags: Any,
        semantic_score: Any,
        semantic_confidence: Any,
        source_set_text: Any,
        user_segment: Any,
        als_rank: Any,
        cluster_rank: Any,
        profile_rank: Any,
        popular_rank: Any,
        semantic_support: Any,
        semantic_tag_richness: Any,
        tower_score: Any,
        seq_score: Any,
        cluster_for_recsys: Any,
        cluster_label_for_recsys: Any,
        item_evidence_text: Any,
    ) -> str:
        return build_binary_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text(
                name,
                city,
                categories,
                primary_category,
                top_pos_tags,
                top_neg_tags,
                semantic_score,
                semantic_confidence,
                source_set=source_set_text,
                user_segment=user_segment,
                als_rank=als_rank,
                cluster_rank=cluster_rank,
                profile_rank=profile_rank,
                popular_rank=popular_rank,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_semantic() -> Any:
    """Prompt UDF that drops ranking-position features for DPO eval."""
    def _mk(
        profile_text: Any,
        profile_text_evidence: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        pair_evidence_summary: Any,
        name: Any,
        city: Any,
        categories: Any,
        primary_category: Any,
        top_pos_tags: Any,
        top_neg_tags: Any,
        semantic_score: Any,
        semantic_confidence: Any,
        semantic_support: Any,
        semantic_tag_richness: Any,
        tower_score: Any,
        seq_score: Any,
        cluster_for_recsys: Any,
        cluster_label_for_recsys: Any,
        item_evidence_text: Any,
    ) -> str:
        return build_binary_prompt_semantic(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text_semantic(
                name,
                city,
                categories,
                primary_category,
                top_pos_tags,
                top_neg_tags,
                semantic_score,
                semantic_confidence,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_sft_clean() -> Any:
    def _mk(
        profile_text: Any,
        profile_text_evidence: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        history_anchor_text: Any,
        pair_evidence_summary: Any,
        name: Any,
        city: Any,
        categories: Any,
        primary_category: Any,
        top_pos_tags: Any,
        top_neg_tags: Any,
        semantic_score: Any,
        semantic_confidence: Any,
        cluster_label_for_recsys: Any,
        item_evidence_text: Any,
    ) -> str:
        return build_binary_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                history_anchors=history_anchor_text,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text_sft_clean(
                name,
                city,
                categories,
                primary_category,
                top_pos_tags,
                top_neg_tags,
                semantic_score,
                semantic_confidence,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_full_lite() -> Any:
    def _mk(
        profile_text: Any,
        profile_text_evidence: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        history_anchor_text: Any,
        pair_evidence_summary: Any,
        name: Any,
        city: Any,
        categories: Any,
        primary_category: Any,
        top_pos_tags: Any,
        top_neg_tags: Any,
        semantic_score: Any,
        semantic_confidence: Any,
        source_set_text: Any,
        user_segment: Any,
        semantic_support: Any,
        semantic_tag_richness: Any,
        tower_score: Any,
        seq_score: Any,
        cluster_label_for_recsys: Any,
        item_evidence_text: Any,
    ) -> str:
        return build_binary_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                history_anchors=history_anchor_text,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text_full_lite(
                name,
                city,
                categories,
                primary_category,
                top_pos_tags,
                top_neg_tags,
                semantic_score,
                semantic_confidence,
                source_set=source_set_text,
                user_segment=user_segment,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
            ),
        )

    return F.udf(_mk, "string")


def build_cluster_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("cluster_profile_csv", "")).strip()
    schema = "business_id string, cluster_for_recsys string, cluster_label_for_recsys string"
    if not profile_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("cluster_for_recsys"), F.lit("")).alias("cluster_for_recsys"),
            F.coalesce(F.col("cluster_label_for_recsys"), F.lit("")).alias("cluster_label_for_recsys"),
        )
        .dropDuplicates(["business_id"])
    )


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
        # User-requested path: if batch=3 fails, fallback to 2 first.
        if b == 3:
            return 2
        return max(1, b // 2)

    def _safe_cuda_empty_cache() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception as cache_exc:
            short = str(cache_exc).splitlines()[0][:180]
            print(f"[WARN] torch.cuda.empty_cache failed: {short}")

    def _wrap_prompt_for_chat(prompt_text: str) -> str:
        messages = [{"role": "user", "content": str(prompt_text)}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if template_enable_thinking_supported:
            kwargs["enable_thinking"] = False
        txt = tokenizer.apply_chat_template(messages, **kwargs)  # type: ignore[attr-defined]
        return str(txt)

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
        # Only score the final token position; avoids allocating full seq_len logits.
        forward_kwargs_base[final_token_logits_arg] = 1
        print(f"[CONFIG] final_token_logits_arg={final_token_logits_arg}")
    else:
        print("[CONFIG] final_token_logits_arg=<unsupported>")
    effective_bs = max(1, int(batch_size))
    if prompts:
        print(
            f"[CONFIG] pretokenize_prompt_chunk={PRETOKENIZE_PROMPT_CHUNK} "
            f"pin_memory={PIN_MEMORY and torch.cuda.is_available()} "
            f"non_blocking_h2d={NON_BLOCKING_H2D and torch.cuda.is_available()} "
            f"pad_to_multiple_of={int(PAD_TO_MULTIPLE_OF) if int(PAD_TO_MULTIPLE_OF) > 0 else 0}"
        )
    t0 = time.monotonic()
    scored_total = 0
    last_log_time = t0
    prepared_prompts = prompts
    if template_use_chat:
        prepared_prompts = [_wrap_prompt_for_chat(p) for p in prompts]

    tokenized_prompts: dict[str, torch.Tensor] | None = None
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
        print(
            f"[TIMING] chunk tokenization done: {len(prepared_prompts)} rows in "
            f"{time.monotonic() - t0:.1f}s"
        )

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
                    speed = scored_total / elapsed if elapsed > 0 else 0
                    remaining = (n - i) / speed if speed > 0 else 0
                    print(
                        f"[PROGRESS] scored={scored_total}/{n} "
                        f"speed={speed:.1f} rows/s "
                        f"elapsed={elapsed:.0f}s "
                        f"ETA={remaining:.0f}s "
                        f"batch_size={effective_bs}"
                    )
                    last_log_time = now
            except torch.OutOfMemoryError as oom:
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError(
                        "QLoRA sidecar inference OOM at batch_size=1. "
                        "Try lowering QLORA_EVAL_MAX_SEQ_LEN or disabling 4bit eval fallback."
                    ) from oom
                effective_bs = _next_fallback_bs(bs)
                print(f"[WARN] QLoRA eval OOM: batch={bs}, fallback_batch={effective_bs}")
            except RuntimeError as rte:
                msg = str(rte).lower()
                if "cuda" not in msg:
                    raise
                # These failures usually poison CUDA context; restarting process is safer.
                if "illegal memory access" in msg or "unspecified launch failure" in msg:
                    raise RuntimeError(
                        "QLoRA sidecar inference hit CUDA kernel failure. "
                        "Please restart Python process and retry with smaller "
                        "QLORA_EVAL_BATCH_SIZE (e.g. 12) and/or QLORA_EVAL_MAX_SEQ_LEN."
                    ) from rte
                is_retryable = ("cublas" in msg) or ("cudnn" in msg) or ("out of memory" in msg)
                if not is_retryable:
                    raise
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError(
                        "QLoRA sidecar inference CUDA runtime failure at batch_size=1. "
                        "Try lowering QLORA_EVAL_MAX_SEQ_LEN or setting QLORA_EVAL_USE_4BIT=false."
                    ) from rte
                effective_bs = _next_fallback_bs(bs)
                short = str(rte).splitlines()[0][:180]
                print(
                    f"[WARN] QLoRA eval CUDA failure: batch={bs}, "
                    f"fallback_batch={effective_bs}, err={short}"
                )
    total_time = time.monotonic() - t0
    if n > 0:
        print(f"[TIMING] inference done: {n} rows in {total_time:.1f}s ({n / total_time:.1f} rows/s)")
    merged = np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)
    return merged, int(effective_bs)


def _ndcg_from_rank(rank_1_based: int) -> float:
    return float(1.0 / math.log2(rank_1_based + 1.0))


def evaluate_topk(pdf: pd.DataFrame, score_col: str, top_k: int) -> tuple[float, float]:
    hits: list[float] = []
    ndcgs: list[float] = []
    for _, g in pdf.groupby("user_idx", sort=False):
        s = g.sort_values(score_col, ascending=False)
        s = s.head(int(top_k))
        pos = np.where(s["label_true"].to_numpy(dtype=np.int32) == 1)[0]
        if len(pos) == 0:
            hits.append(0.0)
            ndcgs.append(0.0)
            continue
        rank = int(pos[0]) + 1
        hits.append(1.0)
        ndcgs.append(_ndcg_from_rank(rank))
    if not hits:
        return 0.0, 0.0
    return float(np.mean(hits)), float(np.mean(ndcgs))


def attach_truth_labels(cand: DataFrame, truth: DataFrame) -> DataFrame:
    return cand.join(
        truth.select("user_idx", F.col("true_item_idx").alias("label_item_idx")),
        on="user_idx",
        how="inner",
    ).withColumn("label_true", F.when(F.col("item_idx") == F.col("label_item_idx"), F.lit(1)).otherwise(F.lit(0)))


def normalize_pre_score(pdf: pd.DataFrame) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        a = s.min()
        b = s.max()
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
        return (s - a) / (b - a + 1e-9)

    return pdf.groupby("user_idx", sort=False)["pre_score"].transform(_norm).astype(np.float32)


def validate_eval_profile() -> None:
    valid_profiles = {"custom", "report", "selector", "smoke"}
    if EVAL_PROFILE not in valid_profiles:
        raise ValueError(
            f"Unsupported QLORA_EVAL_PROFILE={EVAL_PROFILE!r}; "
            "expected one of: custom, report, selector, smoke."
        )

    sampled_user_cap = int(MAX_USERS_PER_BUCKET) > 0
    sampled_row_cap = int(MAX_ROWS_PER_BUCKET) > 0
    sampled = sampled_user_cap or sampled_row_cap
    has_explicit_user_cohort = bool(str(EVAL_USER_COHORT_PATH_RAW or "").strip())

    if EVAL_PROFILE == "report":
        if USE_STAGE11_EVAL_SPLIT:
            raise ValueError("QLORA_EVAL_PROFILE=report requires QLORA_EVAL_USE_STAGE11_SPLIT=false.")
        if sampled:
            raise ValueError(
                "QLORA_EVAL_PROFILE=report does not allow user/row caps. "
                "Set QLORA_EVAL_MAX_USERS_PER_BUCKET=0 and QLORA_EVAL_MAX_ROWS_PER_BUCKET=0."
            )
        return

    if EVAL_PROFILE == "selector":
        if not USE_STAGE11_EVAL_SPLIT:
            raise ValueError("QLORA_EVAL_PROFILE=selector requires QLORA_EVAL_USE_STAGE11_SPLIT=true.")
        if has_explicit_user_cohort:
            raise ValueError("QLORA_EVAL_PROFILE=selector does not allow QLORA_EVAL_USER_COHORT_PATH.")
        if sampled:
            raise ValueError(
                "QLORA_EVAL_PROFILE=selector does not allow user/row caps. "
                "Use QLORA_EVAL_PROFILE=smoke for sampled fast iterations."
            )
        return

    if EVAL_PROFILE == "smoke":
        if has_explicit_user_cohort:
            raise ValueError("QLORA_EVAL_PROFILE=smoke does not allow QLORA_EVAL_USER_COHORT_PATH.")
        if not sampled:
            print(
                "[WARN] eval_profile=smoke has no user/row cap. "
                "Prefer setting QLORA_EVAL_MAX_USERS_PER_BUCKET and/or QLORA_EVAL_MAX_ROWS_PER_BUCKET."
            )
        if not USE_STAGE11_EVAL_SPLIT:
            print(
                "[WARN] eval_profile=smoke with use_eval_split=false can still be expensive; "
                "prefer USE_STAGE11_EVAL_SPLIT=true for fast iteration."
            )
        return

    if has_explicit_user_cohort and USE_STAGE11_EVAL_SPLIT:
        raise ValueError(
            "QLORA_EVAL_USER_COHORT_PATH cannot be combined with QLORA_EVAL_USE_STAGE11_SPLIT=true. "
            "Use an explicit cohort or the stage11 eval split, not both."
        )
    if (not USE_STAGE11_EVAL_SPLIT) and sampled_user_cap:
        print(
            "[WARN] custom eval with use_eval_split=false and MAX_USERS_PER_BUCKET>0 "
            "produces a sampled all-user report; do not compare it against full-cohort report runs."
        )
    if (not USE_STAGE11_EVAL_SPLIT) and sampled_row_cap:
        print(
            "[WARN] custom eval with use_eval_split=false and MAX_ROWS_PER_BUCKET>0 "
            "truncates the report cohort; do not compare it against full-cohort report runs."
        )


def main() -> None:
    stage11_2_run = resolve_stage11_2_run()
    if PROMPT_BUILD_MODE not in {"driver", "spark"}:
        raise ValueError(
            f"Unsupported QLORA_EVAL_PROMPT_BUILD_MODE={PROMPT_BUILD_MODE!r}; expected 'driver' or 'spark'."
        )
    validate_eval_profile()
    if int(MAX_USERS_PER_BUCKET) > 800:
        print(
            f"[WARN] QLORA_EVAL_MAX_USERS_PER_BUCKET={int(MAX_USERS_PER_BUCKET)} is large for local runs; "
            "prefer <=600 for faster iteration."
        )
    if int(MAX_ROWS_PER_BUCKET) > 120000:
        print(
            f"[WARN] QLORA_EVAL_MAX_ROWS_PER_BUCKET={int(MAX_ROWS_PER_BUCKET)} may lead to long eval time; "
            "prefer <=80000 for debug cycles."
        )
    if int(MAX_ROWS_PER_BUCKET) > 0 and ROW_CAP_ORDERED:
        print("[WARN] QLORA_EVAL_ROW_CAP_ORDERED=true uses global orderBy before limit and can be slower.")
    if int(MAX_SEQ_LEN) > 512:
        print(
            f"[WARN] QLORA_EVAL_MAX_SEQ_LEN={int(MAX_SEQ_LEN)} increases latency and OOM risk; "
            "prefer 512 unless validated."
        )
    print(
        f"[CONFIG] eval_profile={EVAL_PROFILE} prompt_build_mode={PROMPT_BUILD_MODE} batch_size={int(INFER_BATCH_SIZE)} "
        f"max_seq_len={int(MAX_SEQ_LEN)} prompt_chunk_rows={int(PROMPT_CHUNK_ROWS)} "
        f"flush_rows={int(INTERMEDIATE_FLUSH_ROWS)} max_users_per_bucket={int(MAX_USERS_PER_BUCKET)} "
        f"max_rows_per_bucket={int(MAX_ROWS_PER_BUCKET)}"
    )
    eval_user_cohort_path = resolve_eval_user_cohort_path()
    adapter_dir = stage11_2_run / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")

    stage11_meta = json.loads((stage11_2_run / "run_meta.json").read_text(encoding="utf-8"))
    base_model = str(stage11_meta.get("base_model", "")).strip()
    if not base_model:
        raise RuntimeError(f"base_model missing in {stage11_2_run / 'run_meta.json'}")
    stage11_data_run = resolve_stage11_data_run(stage11_meta)
    stage09_run = resolve_stage09_run(stage11_meta=stage11_meta, stage11_data_run=stage11_data_run)
    print(
        f"[CONFIG] source_stage11_2={stage11_2_run} "
        f"source_stage11_data={stage11_data_run if stage11_data_run is not None else '<none>'} "
        f"use_eval_split={USE_STAGE11_EVAL_SPLIT} invert_prob={INVERT_PROB} "
        f"user_cohort_path={eval_user_cohort_path if eval_user_cohort_path is not None else '<none>'}"
    )
    if USE_STAGE11_EVAL_SPLIT:
        print("[WARN] use_eval_split=True produces selector-only metrics; use false for all-user report parity.")

    stage09_meta = json.loads((stage09_run / "run_meta.json").read_text(encoding="utf-8"))
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
    gate_result = enforce_stage09_gate(stage09_run, buckets)
    spark: SparkSession | None = None
    user_profile_df: DataFrame | None = None
    item_sem_df: DataFrame | None = None
    cluster_profile_df: DataFrame | None = None
    business_meta_df: DataFrame | None = None

    def ensure_spark_runtime() -> tuple[SparkSession, DataFrame, DataFrame, DataFrame, DataFrame]:
        nonlocal spark, user_profile_df, item_sem_df, cluster_profile_df, business_meta_df
        if spark is None:
            spark = build_spark()
            spark.sparkContext.setLogLevel("WARN")
            user_profile_df = build_user_profile_df(spark, stage09_meta)
            item_sem_df = build_item_sem_df(spark, stage09_meta)
            cluster_profile_df = build_cluster_profile_df(spark, stage09_meta)
            business_meta_df = build_business_meta_df(spark, stage09_meta)
            print(f"[CONFIG] PROMPT_MODE={PROMPT_MODE} prompt_build_mode={PROMPT_BUILD_MODE}")
        return spark, user_profile_df, item_sem_df, cluster_profile_df, business_meta_df


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
        "low_cpu_mem_usage": True,
    }
    if ATTN_IMPLEMENTATION and ATTN_IMPLEMENTATION not in {"auto", "default"}:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    if has_cuda:
        model_kwargs["device_map"] = EVAL_DEVICE_MAP
        max_memory: dict[Any, str] = {}
        if EVAL_MAX_MEMORY_CUDA:
            max_memory[0] = str(EVAL_MAX_MEMORY_CUDA)
        if EVAL_MAX_MEMORY_CPU:
            max_memory["cpu"] = str(EVAL_MAX_MEMORY_CPU)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if EVAL_OFFLOAD_FOLDER:
            offload_dir = Path(EVAL_OFFLOAD_FOLDER).expanduser()
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = offload_dir.as_posix()
            if EVAL_OFFLOAD_STATE_DICT:
                # Keep final placement unchanged while spilling transient load-time state dict tensors.
                model_kwargs["offload_state_dict"] = True
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype

    # Align qwen3.5 eval with train-time dtype path to avoid float32-heavy load/memory spikes.
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
    print(
        f"[CONFIG] model_load device_map={EVAL_DEVICE_MAP} "
        f"attn_implementation={(ATTN_IMPLEMENTATION or '<default>')} "
        f"max_memory_cuda={EVAL_MAX_MEMORY_CUDA or '<none>'} "
        f"max_memory_cpu={EVAL_MAX_MEMORY_CPU or '<none>'} "
        f"offload_folder={EVAL_OFFLOAD_FOLDER or '<none>'} "
        f"offload_state_dict={bool(model_kwargs.get('offload_state_dict', False))} "
        f"disable_parallel_loading={DISABLE_PARALLEL_LOADING}"
    )
    print(f"[CONFIG] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print(f"[CONFIG] qwen35_no_think={QWEN35_NO_THINK}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "left"

    if has_cuda and LOAD_MODEL_BEFORE_SPARK:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if "1455" in msg or "paging file" in msg:
            raise RuntimeError(
                "model load failed with Windows os error 1455 (virtual memory/pagefile too small). "
                "Increase pagefile size (recommend >= 40GB), reboot to apply, then rerun stage11_3."
            ) from exc
        raise
    model = PeftModel.from_pretrained(base, adapter_dir.as_posix())
    # This eval path only needs final-token logits, not generation KV cache.
    try:
        if getattr(getattr(model, "config", None), "use_cache", None):
            model.config.use_cache = False
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    yes_id, no_id = build_yes_no_token_ids(tokenizer)

    resume_target = os.getenv("QLORA_EVAL_RESUME_DIR", "").strip()
    if resume_target:
        out_dir = Path(resume_target)
        if not out_dir.exists():
            raise FileNotFoundError(f"QLORA_EVAL_RESUME_DIR not found: {out_dir}")
        run_id = out_dir.name.replace(f"_{RUN_TAG}", "")
        print(f"[RESUME] Resuming eval in {out_dir}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
        out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    for bucket in buckets:
        final_scores_path = out_dir / f"bucket_{bucket}_scores.csv"
        if final_scores_path.exists() and os.getenv("QLORA_EVAL_RESUME_DIR", "").strip():
            print(f"[RESUME] skip bucket={bucket} inference, final scores already exist. Recalculating metrics.")
            pdf = pd.read_csv(final_scores_path.as_posix())
            pre_recall, pre_ndcg = evaluate_topk(pdf, "pre_score", TOP_K)
            q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)
            n_users = int(pdf["user_idx"].nunique())
            n_items = int(pdf["item_idx"].nunique())
            n_rows = int(len(pdf))

            metrics_rows.append(
                {
                    "run_id_11": run_id,
                    "source_run_09": str(stage09_run),
                    "source_run_11_2": str(stage11_2_run),
                    "bucket_min_train_reviews": int(bucket),
                    "model": "PreScore@10",
                    "recall_at_k": float(pre_recall),
                    "ndcg_at_k": float(pre_ndcg),
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_candidates": n_rows,
                }
            )
            metrics_rows.append(
                {
                    "run_id_11": run_id,
                    "source_run_09": str(stage09_run),
                    "source_run_11_2": str(stage11_2_run),
                    "bucket_min_train_reviews": int(bucket),
                    "model": "QLoRASidecar@10",
                    "recall_at_k": float(q_recall),
                    "ndcg_at_k": float(q_ndcg),
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_candidates": n_rows,
                }
            )
            continue

        bdir = stage09_run / f"bucket_{bucket}"
        truth_path = bdir / "truth.parquet"
        if not truth_path.exists():
            print(f"[WARN] skip bucket={bucket}: missing truth.parquet")
            continue
        cand_path = pick_candidate_file(bdir)
        spark, user_profile_df, item_sem_df, cluster_profile_df, business_meta_df = ensure_spark_runtime()

        truth = (
            spark.read.parquet(truth_path.as_posix())
            .select("user_idx", "user_id", "true_item_idx")
            .dropDuplicates(["user_idx"])
        )
        users = truth.select("user_idx").dropDuplicates(["user_idx"])
        if eval_user_cohort_path is not None:
            cohort_users, cohort_user_count = load_eval_users_from_cohort(spark, eval_user_cohort_path)
            print(f"[COHORT] bucket={bucket} loaded explicit user cohort: {eval_user_cohort_path} users={cohort_user_count}")
            users = users.join(F.broadcast(cohort_users), on="user_idx", how="inner")
            truth = truth.join(users, on="user_idx", how="inner")
        elif USE_STAGE11_EVAL_SPLIT:
            if stage11_data_run is None:
                raise RuntimeError("USE_STAGE11_EVAL_SPLIT=true but stage11_data_run is not resolved")
            eval_users = load_eval_users_from_stage11_data(spark, stage11_data_run, int(bucket))
            users = users.join(eval_users, on="user_idx", how="inner")
            truth = truth.join(users, on="user_idx", how="inner")
        if MAX_USERS_PER_BUCKET > 0:
            users = users.orderBy(F.rand(int(RANDOM_SEED + bucket))).limit(int(MAX_USERS_PER_BUCKET))
            truth = truth.join(users, on="user_idx", how="inner")

        cand_raw = (
            spark.read.parquet(cand_path.as_posix())
            .join(users, on="user_idx", how="inner")
            .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
            .filter(F.col("pre_rank") <= F.lit(int(RERANK_TOPN)))
        )
        cand_cols = set(cand_raw.columns)

        def _num_col(name: str) -> Any:
            if name in cand_cols:
                return F.col(name).cast("double").alias(name)
            return F.lit(None).cast("double").alias(name)

        source_set_text = (
            F.when(F.col("source_set").isNull(), F.lit(""))
            .otherwise(F.concat_ws("|", F.col("source_set")))
            .alias("source_set_text")
            if "source_set" in cand_cols
            else F.lit("").alias("source_set_text")
        )
        user_segment_col = (
            F.coalesce(F.col("user_segment"), F.lit("")).alias("user_segment")
            if "user_segment" in cand_cols
            else F.lit("").alias("user_segment")
        )

        cand = (
            cand_raw.select(
                "user_idx",
                "user_id",
                "item_idx",
                "business_id",
                "pre_rank",
                "pre_score",
                "name",
                "city",
                "categories",
                "primary_category",
                source_set_text,
                user_segment_col,
                _num_col("als_rank"),
                _num_col("cluster_rank"),
                _num_col("profile_rank"),
                _num_col("popular_rank"),
                _num_col("semantic_support"),
                _num_col("semantic_tag_richness"),
                _num_col("tower_score"),
                _num_col("seq_score"),
            )
            .join(cluster_profile_df, on="business_id", how="left")
        )
        cand = attach_truth_labels(cand, truth)
        if MAX_ROWS_PER_BUCKET > 0:
            if ROW_CAP_ORDERED:
                cand = cand.orderBy(F.col("user_idx").asc(), F.col("pre_rank").asc()).limit(int(MAX_ROWS_PER_BUCKET))
            else:
                cand = cand.limit(int(MAX_ROWS_PER_BUCKET))

        review_base_df = build_review_base_df(
            spark=spark,
            stage09_run=stage09_run,
            bucket_dir=bdir,
            cand_path=cand_path,
            item_sem_df=item_sem_df,
        )
        history_anchor_df = build_history_anchor_df(
            spark=spark,
            bucket_dir=bdir,
            users_df=users,
            cand_df=cand.select("item_idx", "business_id"),
            item_sem_df=item_sem_df,
            business_meta_df=business_meta_df,
        )
        item_evidence_df = build_item_review_evidence(
            spark=spark,
            bucket_dir=bdir,
            cand_df=cand.select("user_idx", "business_id"),
            review_base_df=review_base_df,
        )
        clean_profile_udf = F.udf(lambda x: clean_text(x, max_chars=220), "string")
        user_evidence_udf = build_user_evidence_udf()
        pair_evidence_udf = build_pair_alignment_udf()
        history_anchor_text_udf = build_history_anchor_text_udf()
        cand = (
            cand.join(user_profile_df, on="user_id", how="left")
            .join(item_sem_df, on="business_id", how="left")
            .join(item_evidence_df, on=["user_idx", "business_id"], how="left")
            .join(history_anchor_df, on="user_idx", how="left")
            .fillna(
                {
                    "profile_text": "",
                    "profile_text_evidence": "",
                    "profile_top_pos_tags": "",
                    "profile_top_neg_tags": "",
                    "profile_confidence": 0.0,
                    "top_pos_tags": "",
                    "top_neg_tags": "",
                    "semantic_score": 0.0,
                    "semantic_confidence": 0.0,
                    "source_set_text": "",
                    "user_segment": "",
                    "als_rank": 0.0,
                    "cluster_rank": 0.0,
                    "profile_rank": 0.0,
                    "popular_rank": 0.0,
                    "semantic_support": 0.0,
                    "semantic_tag_richness": 0.0,
                    "tower_score": 0.0,
                    "seq_score": 0.0,
                    "cluster_for_recsys": "",
                    "cluster_label_for_recsys": "",
                    "item_evidence_text": "",
                }
            )
            .withColumn("profile_text", clean_profile_udf(F.col("profile_text")))
            .withColumn(
                "user_evidence_text",
                user_evidence_udf(F.col("profile_text"), F.col("profile_text_evidence")),
            )
            .withColumn(
                "pair_evidence_summary",
                pair_evidence_udf(
                    F.col("profile_top_pos_tags"),
                    F.col("profile_top_neg_tags"),
                    F.col("top_pos_tags"),
                    F.col("top_neg_tags"),
                ),
            )
            .withColumn(
                "history_anchor_text",
                history_anchor_text_udf(F.col("history_anchor_entries"), F.col("business_id")),
            )
        )

        partial_scores_path = out_dir / f"bucket_{bucket}_scores_partial.csv"
        flushed_rows = 0
        scored_pairs_df = None

        if partial_scores_path.exists() and os.getenv("QLORA_EVAL_RESUME_DIR", "").strip():
            print(f"[RESUME] Loading partial scores from {partial_scores_path}")
            try:
                partial_df = pd.read_csv(partial_scores_path.as_posix(), usecols=["user_idx", "item_idx"]).dropna()
                if not partial_df.empty:
                    flushed_rows = len(partial_df)
                    scored_pairs_df = spark.createDataFrame(
                        partial_df[["user_idx", "item_idx"]].astype(int)
                    )
            except Exception as e:
                print(f"[WARN] Failed to read partial scores: {e}. Will overwrite.")
                partial_scores_path.unlink()
        elif partial_scores_path.exists():
            partial_scores_path.unlink()

        pending_user_idx: list[int] = []
        pending_item_idx: list[int] = []
        pending_pre_rank: list[int] = []
        pending_label_true: list[int] = []
        pending_pre_score: list[float] = []
        pending_qlora_prob: list[float] = []

        prompt_buf: list[str] = []
        user_idx_buf: list[int] = []
        item_idx_buf: list[int] = []
        pre_rank_buf: list[int] = []
        label_true_buf: list[int] = []
        pre_score_buf: list[float] = []
        adaptive_batch_size = max(1, int(INFER_BATCH_SIZE))

        def _append_partial_scores() -> None:
            nonlocal flushed_rows
            if not pending_user_idx:
                return
            chunk = pd.DataFrame(
                {
                    "user_idx": np.asarray(pending_user_idx, dtype=np.int64),
                    "item_idx": np.asarray(pending_item_idx, dtype=np.int64),
                    "pre_rank": np.asarray(pending_pre_rank, dtype=np.int32),
                    "label_true": np.asarray(pending_label_true, dtype=np.int8),
                    "pre_score": np.asarray(pending_pre_score, dtype=np.float32),
                    "qlora_prob": np.asarray(pending_qlora_prob, dtype=np.float32),
                }
            )
            write_header = not partial_scores_path.exists()
            chunk.to_csv(partial_scores_path.as_posix(), mode="a", header=write_header, index=False)
            flushed_rows += int(len(chunk))
            pending_user_idx.clear()
            pending_item_idx.clear()
            pending_pre_rank.clear()
            pending_label_true.clear()
            pending_pre_score.clear()
            pending_qlora_prob.clear()

        def _flush_prompt_buf() -> None:
            nonlocal adaptive_batch_size
            if not prompt_buf:
                return
            prev_bs = int(adaptive_batch_size)
            q_prob_local, adaptive_batch_size = score_yes_probability(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_buf,
                batch_size=adaptive_batch_size,
                max_seq_len=MAX_SEQ_LEN,
                yes_id=yes_id,
                no_id=no_id,
            )
            q_prob_local = q_prob_local.astype(np.float32)
            if int(adaptive_batch_size) != int(prev_bs):
                print(
                    f"[CONFIG] adaptive_eval_batch_size update: "
                    f"{int(prev_bs)} -> {int(adaptive_batch_size)}"
                )
            if INVERT_PROB:
                q_prob_local = np.clip(1.0 - q_prob_local, 0.0, 1.0)
            if len(q_prob_local) != len(user_idx_buf):
                raise RuntimeError(
                    "qlora score size mismatch: "
                    f"probs={len(q_prob_local)} rows={len(user_idx_buf)}"
                )
            pending_user_idx.extend(user_idx_buf)
            pending_item_idx.extend(item_idx_buf)
            pending_pre_rank.extend(pre_rank_buf)
            pending_label_true.extend(label_true_buf)
            pending_pre_score.extend(pre_score_buf)
            pending_qlora_prob.extend([float(x) for x in q_prob_local.tolist()])
            if len(pending_user_idx) >= int(max(1, INTERMEDIATE_FLUSH_ROWS)):
                _append_partial_scores()
            prompt_buf.clear()
            user_idx_buf.clear()
            item_idx_buf.clear()
            pre_rank_buf.clear()
            label_true_buf.clear()
            pre_score_buf.clear()

        def _consume_prompt_row(row: Any) -> None:
            prompt_buf.append(_str_or_empty(row["prompt"]))
            user_idx_buf.append(int(row["user_idx"]))
            item_idx_buf.append(int(row["item_idx"]))
            pre_rank_buf.append(int(row["pre_rank"]))
            label_true_buf.append(int(row["label_true"]))
            pre_score_buf.append(_float_or_zero(row["pre_score"]))

        t_spark_start = time.monotonic()
        # ------------------------------------------------------------------
        # Candidate prep before driver-side scoring.
        # Cloud runs should prefer Spark-side prompt construction and stream
        # prompt-ready rows back to the driver, rather than full collect().
        # ------------------------------------------------------------------
        _collect_cols = [
            "user_idx", "item_idx", "pre_rank", "label_true", "pre_score",
            "profile_text", "profile_top_pos_tags", "profile_top_neg_tags",
            "profile_confidence", "user_evidence_text", "pair_evidence_summary",
            "history_anchor_text",
            "name", "city", "categories", "primary_category",
            "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence",
            "semantic_support", "semantic_tag_richness", "tower_score", "seq_score",
            "cluster_for_recsys", "cluster_label_for_recsys",
            "item_evidence_text",
        ]
        if PROMPT_MODE == "full":
            _collect_cols.extend(
                [
                    "source_set_text", "user_segment",
                    "als_rank", "cluster_rank", "profile_rank", "popular_rank",
                ]
            )
        elif PROMPT_MODE == "full_lite":
            _collect_cols.extend(["source_set_text", "user_segment"])
        cand_for_collect = cand.select(*_collect_cols)

        if scored_pairs_df is not None:
            cand_for_collect = cand_for_collect.join(
                F.broadcast(scored_pairs_df),
                on=["user_idx", "item_idx"],
                how="left_anti"
            )
            print(f"[RESUME] Skipping already evaluated rows. Pre-computed count = {flushed_rows}")

        if int(ITER_COALESCE_PARTITIONS) > 0:
            cand_for_collect = cand_for_collect.coalesce(int(ITER_COALESCE_PARTITIONS))

        if PROMPT_BUILD_MODE == "spark":
            print("[PROMPT] Building prompt on Spark workers ...")
            if PROMPT_MODE == "semantic":
                prompt_udf = build_prompt_udf_semantic()
                cand_for_collect = cand_for_collect.withColumn(
                    "prompt",
                    prompt_udf(
                        F.col("profile_text"),
                        F.col("user_evidence_text"),
                        F.col("profile_top_pos_tags"),
                        F.col("profile_top_neg_tags"),
                        F.col("profile_confidence"),
                        F.col("pair_evidence_summary"),
                        F.col("name"),
                        F.col("city"),
                        F.col("categories"),
                        F.col("primary_category"),
                        F.col("top_pos_tags"),
                        F.col("top_neg_tags"),
                        F.col("semantic_score"),
                        F.col("semantic_confidence"),
                        F.col("semantic_support"),
                        F.col("semantic_tag_richness"),
                        F.col("tower_score"),
                        F.col("seq_score"),
                        F.col("cluster_for_recsys"),
                        F.col("cluster_label_for_recsys"),
                        F.col("item_evidence_text"),
                    ),
                )
            elif PROMPT_MODE == "full_lite":
                prompt_udf = build_prompt_udf_full_lite()
                cand_for_collect = cand_for_collect.withColumn(
                    "prompt",
                    prompt_udf(
                        F.col("profile_text"),
                        F.col("user_evidence_text"),
                        F.col("profile_top_pos_tags"),
                        F.col("profile_top_neg_tags"),
                        F.col("profile_confidence"),
                        F.col("history_anchor_text"),
                        F.col("pair_evidence_summary"),
                        F.col("name"),
                        F.col("city"),
                        F.col("categories"),
                        F.col("primary_category"),
                        F.col("top_pos_tags"),
                        F.col("top_neg_tags"),
                        F.col("semantic_score"),
                        F.col("semantic_confidence"),
                        F.col("source_set_text"),
                        F.col("user_segment"),
                        F.col("semantic_support"),
                        F.col("semantic_tag_richness"),
                        F.col("tower_score"),
                        F.col("seq_score"),
                        F.col("cluster_label_for_recsys"),
                        F.col("item_evidence_text"),
                    ),
                )
            elif PROMPT_MODE == "sft_clean":
                prompt_udf = build_prompt_udf_sft_clean()
                cand_for_collect = cand_for_collect.withColumn(
                    "prompt",
                    prompt_udf(
                        F.col("profile_text"),
                        F.col("user_evidence_text"),
                        F.col("profile_top_pos_tags"),
                        F.col("profile_top_neg_tags"),
                        F.col("profile_confidence"),
                        F.col("history_anchor_text"),
                        F.col("pair_evidence_summary"),
                        F.col("name"),
                        F.col("city"),
                        F.col("categories"),
                        F.col("primary_category"),
                        F.col("top_pos_tags"),
                        F.col("top_neg_tags"),
                        F.col("semantic_score"),
                        F.col("semantic_confidence"),
                        F.col("cluster_label_for_recsys"),
                        F.col("item_evidence_text"),
                    ),
                )
            else:
                prompt_udf = build_prompt_udf()
                cand_for_collect = cand_for_collect.withColumn(
                    "prompt",
                    prompt_udf(
                        F.col("profile_text"),
                        F.col("user_evidence_text"),
                        F.col("profile_top_pos_tags"),
                        F.col("profile_top_neg_tags"),
                        F.col("profile_confidence"),
                        F.col("pair_evidence_summary"),
                        F.col("name"),
                        F.col("city"),
                        F.col("categories"),
                        F.col("primary_category"),
                        F.col("top_pos_tags"),
                        F.col("top_neg_tags"),
                        F.col("semantic_score"),
                        F.col("semantic_confidence"),
                        F.col("source_set_text"),
                        F.col("user_segment"),
                        F.col("als_rank"),
                        F.col("cluster_rank"),
                        F.col("profile_rank"),
                        F.col("popular_rank"),
                        F.col("semantic_support"),
                        F.col("semantic_tag_richness"),
                        F.col("tower_score"),
                        F.col("seq_score"),
                        F.col("cluster_for_recsys"),
                        F.col("cluster_label_for_recsys"),
                        F.col("item_evidence_text"),
                    ),
            )
            cand_for_collect = cand_for_collect.select(
                "user_idx", "item_idx", "pre_rank", "label_true", "pre_score", "prompt"
            )
            print("[COLLECT] Streaming prompt-ready candidate data to driver (toLocalIterator) ...")
            iter_row_count = 0
            for row in cand_for_collect.toLocalIterator():
                _consume_prompt_row(row)
                iter_row_count += 1
                if int(STREAM_LOG_ROWS) > 0 and (iter_row_count % int(STREAM_LOG_ROWS) == 0):
                    elapsed = time.monotonic() - t_spark_start
                    print(
                        f"[STREAM] streamed_rows={iter_row_count} "
                        f"buffered_prompts={len(prompt_buf)} elapsed={elapsed:.1f}s"
                    )
                if len(prompt_buf) >= int(PROMPT_CHUNK_ROWS):
                    elapsed = time.monotonic() - t_spark_start
                    print(
                        f"[TIMING] prompt chunk ready: streamed_rows={iter_row_count} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    _flush_prompt_buf()
            t_collect = time.monotonic() - t_spark_start
            print(f"[TIMING] spark prompt stream done: {iter_row_count} rows in {t_collect:.1f}s")
        else:
            print("[COLLECT] Collecting candidate data to driver (toPandas) ...")
            cand_pdf = cand_for_collect.toPandas()
            t_collect = time.monotonic() - t_spark_start
            print(f"[TIMING] toPandas done: {len(cand_pdf)} rows in {t_collect:.1f}s")

            # Build prompts on driver side (pure string ops, no Spark workers)
            def _build_prompt_for_row(r: pd.Series) -> str:
                user_text = build_user_text(
                    r.get("profile_text", ""),
                    r.get("profile_top_pos_tags", ""),
                    r.get("profile_top_neg_tags", ""),
                    r.get("profile_confidence", 0.0),
                    evidence_snippets=r.get("user_evidence_text", ""),
                    history_anchors=r.get("history_anchor_text", ""),
                    pair_evidence=r.get("pair_evidence_summary", ""),
                )
                if PROMPT_MODE == "semantic":
                    item_text = build_item_text_semantic(
                        r.get("name", ""),
                        r.get("city", ""),
                        r.get("categories", ""),
                        r.get("primary_category", ""),
                        r.get("top_pos_tags", ""),
                        r.get("top_neg_tags", ""),
                        r.get("semantic_score", 0.0),
                        r.get("semantic_confidence", 0.0),
                        semantic_support=r.get("semantic_support", 0.0),
                        semantic_tag_richness=r.get("semantic_tag_richness", 0.0),
                        tower_score=r.get("tower_score", 0.0),
                        seq_score=r.get("seq_score", 0.0),
                        cluster_for_recsys=r.get("cluster_for_recsys", ""),
                        cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                        item_review_snippet=r.get("item_evidence_text", ""),
                    )
                    return build_binary_prompt_semantic(user_text, item_text)
                elif PROMPT_MODE == "full_lite":
                    item_text = build_item_text_full_lite(
                        r.get("name", ""),
                        r.get("city", ""),
                        r.get("categories", ""),
                        r.get("primary_category", ""),
                        r.get("top_pos_tags", ""),
                        r.get("top_neg_tags", ""),
                        r.get("semantic_score", 0.0),
                        r.get("semantic_confidence", 0.0),
                        source_set=r.get("source_set_text", ""),
                        user_segment=r.get("user_segment", ""),
                        semantic_support=r.get("semantic_support", 0.0),
                        semantic_tag_richness=r.get("semantic_tag_richness", 0.0),
                        tower_score=r.get("tower_score", 0.0),
                        seq_score=r.get("seq_score", 0.0),
                        cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                        item_review_snippet=r.get("item_evidence_text", ""),
                    )
                    return build_binary_prompt(user_text, item_text)
                elif PROMPT_MODE == "sft_clean":
                    item_text = build_item_text_sft_clean(
                        r.get("name", ""),
                        r.get("city", ""),
                        r.get("categories", ""),
                        r.get("primary_category", ""),
                        r.get("top_pos_tags", ""),
                        r.get("top_neg_tags", ""),
                        r.get("semantic_score", 0.0),
                        r.get("semantic_confidence", 0.0),
                        cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                        item_review_snippet=r.get("item_evidence_text", ""),
                    )
                    return build_binary_prompt(user_text, item_text)
                else:
                    item_text = build_item_text(
                        r.get("name", ""),
                        r.get("city", ""),
                        r.get("categories", ""),
                        r.get("primary_category", ""),
                        r.get("top_pos_tags", ""),
                        r.get("top_neg_tags", ""),
                        r.get("semantic_score", 0.0),
                        r.get("semantic_confidence", 0.0),
                        source_set=r.get("source_set_text", ""),
                        user_segment=r.get("user_segment", ""),
                        als_rank=r.get("als_rank", 0.0),
                        cluster_rank=r.get("cluster_rank", 0.0),
                        profile_rank=r.get("profile_rank", 0.0),
                        popular_rank=r.get("popular_rank", 0.0),
                        semantic_support=r.get("semantic_support", 0.0),
                        semantic_tag_richness=r.get("semantic_tag_richness", 0.0),
                        tower_score=r.get("tower_score", 0.0),
                        seq_score=r.get("seq_score", 0.0),
                        cluster_for_recsys=r.get("cluster_for_recsys", ""),
                        cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                        item_review_snippet=r.get("item_evidence_text", ""),
                    )
                    return build_binary_prompt(user_text, item_text)

            print(f"[CONFIG] driver_prompt_impl={DRIVER_PROMPT_IMPL}")
            t_prompt_start = time.monotonic()
            if DRIVER_PROMPT_IMPL == "itertuples":
                def _build_prompt_for_tuple(r: Any) -> str:
                    user_text = build_user_text(
                        getattr(r, "profile_text", ""),
                        getattr(r, "profile_top_pos_tags", ""),
                        getattr(r, "profile_top_neg_tags", ""),
                        getattr(r, "profile_confidence", 0.0),
                        evidence_snippets=getattr(r, "user_evidence_text", ""),
                        history_anchors=getattr(r, "history_anchor_text", ""),
                        pair_evidence=getattr(r, "pair_evidence_summary", ""),
                    )
                    if PROMPT_MODE == "semantic":
                        item_text = build_item_text_semantic(
                            getattr(r, "name", ""),
                            getattr(r, "city", ""),
                            getattr(r, "categories", ""),
                            getattr(r, "primary_category", ""),
                            getattr(r, "top_pos_tags", ""),
                            getattr(r, "top_neg_tags", ""),
                            getattr(r, "semantic_score", 0.0),
                            getattr(r, "semantic_confidence", 0.0),
                            semantic_support=getattr(r, "semantic_support", 0.0),
                            semantic_tag_richness=getattr(r, "semantic_tag_richness", 0.0),
                            tower_score=getattr(r, "tower_score", 0.0),
                            seq_score=getattr(r, "seq_score", 0.0),
                            cluster_for_recsys=getattr(r, "cluster_for_recsys", ""),
                            cluster_label_for_recsys=getattr(r, "cluster_label_for_recsys", ""),
                            item_review_snippet=getattr(r, "item_evidence_text", ""),
                        )
                        return build_binary_prompt_semantic(user_text, item_text)
                    if PROMPT_MODE == "full_lite":
                        item_text = build_item_text_full_lite(
                            getattr(r, "name", ""),
                            getattr(r, "city", ""),
                            getattr(r, "categories", ""),
                            getattr(r, "primary_category", ""),
                            getattr(r, "top_pos_tags", ""),
                            getattr(r, "top_neg_tags", ""),
                            getattr(r, "semantic_score", 0.0),
                            getattr(r, "semantic_confidence", 0.0),
                            source_set=getattr(r, "source_set_text", ""),
                            user_segment=getattr(r, "user_segment", ""),
                            semantic_support=getattr(r, "semantic_support", 0.0),
                            semantic_tag_richness=getattr(r, "semantic_tag_richness", 0.0),
                            tower_score=getattr(r, "tower_score", 0.0),
                            seq_score=getattr(r, "seq_score", 0.0),
                            cluster_label_for_recsys=getattr(r, "cluster_label_for_recsys", ""),
                            item_review_snippet=getattr(r, "item_evidence_text", ""),
                        )
                        return build_binary_prompt(user_text, item_text)
                    if PROMPT_MODE == "sft_clean":
                        item_text = build_item_text_sft_clean(
                            getattr(r, "name", ""),
                            getattr(r, "city", ""),
                            getattr(r, "categories", ""),
                            getattr(r, "primary_category", ""),
                            getattr(r, "top_pos_tags", ""),
                            getattr(r, "top_neg_tags", ""),
                            getattr(r, "semantic_score", 0.0),
                            getattr(r, "semantic_confidence", 0.0),
                            cluster_label_for_recsys=getattr(r, "cluster_label_for_recsys", ""),
                            item_review_snippet=getattr(r, "item_evidence_text", ""),
                        )
                        return build_binary_prompt(user_text, item_text)
                    item_text = build_item_text(
                        getattr(r, "name", ""),
                        getattr(r, "city", ""),
                        getattr(r, "categories", ""),
                        getattr(r, "primary_category", ""),
                        getattr(r, "top_pos_tags", ""),
                        getattr(r, "top_neg_tags", ""),
                        getattr(r, "semantic_score", 0.0),
                        getattr(r, "semantic_confidence", 0.0),
                        source_set=getattr(r, "source_set_text", ""),
                        user_segment=getattr(r, "user_segment", ""),
                        als_rank=getattr(r, "als_rank", 0.0),
                        cluster_rank=getattr(r, "cluster_rank", 0.0),
                        profile_rank=getattr(r, "profile_rank", 0.0),
                        popular_rank=getattr(r, "popular_rank", 0.0),
                        semantic_support=getattr(r, "semantic_support", 0.0),
                        semantic_tag_richness=getattr(r, "semantic_tag_richness", 0.0),
                        tower_score=getattr(r, "tower_score", 0.0),
                        seq_score=getattr(r, "seq_score", 0.0),
                        cluster_for_recsys=getattr(r, "cluster_for_recsys", ""),
                        cluster_label_for_recsys=getattr(r, "cluster_label_for_recsys", ""),
                        item_review_snippet=getattr(r, "item_evidence_text", ""),
                    )
                    return build_binary_prompt(user_text, item_text)

                cand_pdf["prompt"] = [
                    _build_prompt_for_tuple(row)
                    for row in cand_pdf.itertuples(index=False, name="PromptRow")
                ]
            else:
                cand_pdf["prompt"] = cand_pdf.apply(_build_prompt_for_row, axis=1)
            t_prompt = time.monotonic() - t_prompt_start
            print(f"[TIMING] driver-side prompt generation done: {len(cand_pdf)} rows in {t_prompt:.1f}s")

            # Iterate over driver-side pandas DataFrame in chunks
            iter_row_count = 0
            for idx in range(len(cand_pdf)):
                row = cand_pdf.iloc[idx]
                _consume_prompt_row(row)
                iter_row_count += 1
                if len(prompt_buf) >= int(PROMPT_CHUNK_ROWS):
                    print(
                        f"[TIMING] prompt chunk ready: {iter_row_count}/{len(cand_pdf)} rows"
                    )
                    _flush_prompt_buf()

        t_spark_total = time.monotonic() - t_spark_start
        print(f"[TIMING] Spark iter done: {iter_row_count} total rows in {t_spark_total:.1f}s")
        _flush_prompt_buf()
        _append_partial_scores()
        if int(flushed_rows) <= 0 or not partial_scores_path.exists():
            print(f"[WARN] skip bucket={bucket}: empty candidate after filters")
            continue
        pdf = pd.read_csv(partial_scores_path.as_posix())
        if pdf.empty:
            print(f"[WARN] skip bucket={bucket}: empty partial scores")
            continue
        pdf["pre_norm"] = normalize_pre_score(pdf)
        pdf["blend_score"] = (1.0 - float(BLEND_ALPHA)) * pdf["pre_norm"] + float(BLEND_ALPHA) * pdf["qlora_prob"]

        pre_recall, pre_ndcg = evaluate_topk(pdf, "pre_score", TOP_K)
        q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)

        n_users = int(pdf["user_idx"].nunique())
        n_items = int(pdf["item_idx"].nunique())
        n_rows = int(len(pdf))

        metrics_rows.append(
            {
                "run_id_11": run_id,
                "source_run_09": str(stage09_run),
                "source_run_11_2": str(stage11_2_run),
                "bucket_min_train_reviews": int(bucket),
                "model": "PreScore@10",
                "recall_at_k": float(pre_recall),
                "ndcg_at_k": float(pre_ndcg),
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_rows,
            }
        )
        metrics_rows.append(
            {
                "run_id_11": run_id,
                "source_run_09": str(stage09_run),
                "source_run_11_2": str(stage11_2_run),
                "bucket_min_train_reviews": int(bucket),
                "model": "QLoRASidecar@10",
                "recall_at_k": float(q_recall),
                "ndcg_at_k": float(q_ndcg),
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_rows,
            }
        )
        print(
            f"[METRIC] bucket={bucket} pre_ndcg={pre_ndcg:.6f} qlora_ndcg={q_ndcg:.6f} "
            f"pre_recall={pre_recall:.6f} qlora_recall={q_recall:.6f}"
        )

        pd.DataFrame(
            {
                "user_idx": pdf["user_idx"],
                "item_idx": pdf["item_idx"],
                "pre_rank": pdf["pre_rank"],
                "label_true": pdf["label_true"],
                "pre_score": pdf["pre_score"],
                "qlora_prob": pdf["qlora_prob"],
                "blend_score": pdf["blend_score"],
            }
        ).to_csv((out_dir / f"bucket_{bucket}_scores.csv").as_posix(), index=False)
        print(f"[CHECKPOINT] bucket={bucket} flushed_rows={int(flushed_rows)} partial_file={partial_scores_path}")

    if not metrics_rows:
        raise RuntimeError("no bucket metrics produced")

    df_metrics = pd.DataFrame(metrics_rows)
    out_csv = out_dir / "qlora_sidecar_metrics.csv"
    df_metrics.to_csv(out_csv.as_posix(), index=False)

    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        merged = pd.concat([old, df_metrics], ignore_index=True)
        merged.to_csv(METRICS_PATH, index=False)
    else:
        df_metrics.to_csv(METRICS_PATH, index=False)

    meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_run_09": str(stage09_run),
        "source_run_11_2": str(stage11_2_run),
        "source_run_11_1_data": str(stage11_data_run) if stage11_data_run is not None else "",
        "enforce_stage09_gate": bool(ENFORCE_STAGE09_GATE),
        "stage09_gate_result": gate_result,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "buckets": buckets,
        "top_k": int(TOP_K),
        "rerank_topn": int(RERANK_TOPN),
        "blend_alpha": float(BLEND_ALPHA),
        "invert_prob": bool(INVERT_PROB),
        "eval_profile": str(EVAL_PROFILE),
        "eval_user_cohort_path": str(eval_user_cohort_path) if eval_user_cohort_path is not None else "",
        "use_stage11_eval_split": bool(USE_STAGE11_EVAL_SPLIT),
        "structured_feature_prompt_enabled": True,
        "short_text_summary_enabled": bool(ENABLE_SHORT_TEXT_SUMMARY),
        "raw_review_text_enabled": bool(ENABLE_RAW_REVIEW_TEXT),
        "review_snippet_max_chars": int(REVIEW_SNIPPET_MAX_CHARS),
        "user_evidence_max_chars": int(USER_EVIDENCE_MAX_CHARS),
        "item_evidence_max_chars": int(ITEM_EVIDENCE_MAX_CHARS),
        "max_users_per_bucket": int(MAX_USERS_PER_BUCKET),
        "max_rows_per_bucket": int(MAX_ROWS_PER_BUCKET),
        "row_cap_ordered": bool(ROW_CAP_ORDERED),
        "prompt_mode": str(PROMPT_MODE),
        "prompt_build_mode": str(PROMPT_BUILD_MODE),
        "attn_implementation": str(ATTN_IMPLEMENTATION or ""),
        "prompt_chunk_rows": int(PROMPT_CHUNK_ROWS),
        "pad_to_multiple_of": int(PAD_TO_MULTIPLE_OF),
        "stream_log_rows": int(STREAM_LOG_ROWS),
        "iter_coalesce_partitions": int(ITER_COALESCE_PARTITIONS),
        "intermediate_flush_rows": int(INTERMEDIATE_FLUSH_ROWS),
        "pretokenize_prompt_chunk": bool(PRETOKENIZE_PROMPT_CHUNK),
        "pin_memory": bool(PIN_MEMORY),
        "non_blocking_h2d": bool(NON_BLOCKING_H2D),
        "rich_sft_history_anchor_max_per_user": int(RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER),
        "rich_sft_history_anchor_primary_min_rating": float(RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING),
        "rich_sft_history_anchor_fallback_min_rating": float(RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING),
        "rich_sft_history_anchor_max_chars": int(RICH_SFT_HISTORY_ANCHOR_MAX_CHARS),
        "spark_python_worker_memory": str(SPARK_PYTHON_WORKER_MEMORY),
        "review_base_cache_enabled": bool(REVIEW_BASE_CACHE_ENABLED),
        "review_base_cache_root": str(REVIEW_BASE_CACHE_ROOT),
        "metrics_file": str(out_csv),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    pointer_path = write_latest_run_pointer(
        "stage11_3_qlora_sidecar_eval",
        out_dir,
        extra={
            "run_tag": RUN_TAG,
            "source_run_09": str(stage09_run),
            "source_run_11_2": str(stage11_2_run),
            "source_run_11_1_data": str(stage11_data_run) if stage11_data_run is not None else "",
            "metrics_file": str(out_csv),
            "eval_profile": str(EVAL_PROFILE),
            "eval_user_cohort_path": str(eval_user_cohort_path) if eval_user_cohort_path is not None else "",
        },
    )

    if spark is not None:
        spark.stop()
    print(f"[DONE] qlora sidecar metrics: {out_csv}")
    print(f"[DONE] merged metrics: {METRICS_PATH}")
    print(f"[DONE] updated latest pointer: {pointer_path}")


if __name__ == "__main__":
    main()


