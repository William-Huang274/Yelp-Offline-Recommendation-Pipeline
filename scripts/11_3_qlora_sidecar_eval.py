from __future__ import annotations

import gc
import json
import math
import os
import inspect
import sys
import time
from datetime import datetime
from functools import lru_cache
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

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
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
    build_scoring_prompt,
    build_item_text,
    build_item_text_full_lite,
    build_item_text_semantic_compact,
    build_item_text_semantic_compact_preserve,
    build_item_text_semantic_compact_targeted,
    build_item_text_semantic,
    build_item_text_sft_clean,
    build_pair_alignment_summary,
    build_user_text,
)
from pipeline.stage11_pairwise import (
    _build_boundary_item_text_from_row,
    _build_boundary_user_text_from_row,
    _prepare_local_listwise_prompt_context,
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
INPUT_09_MATCH_CHANNELS_ROOT = env_or_project_path(
    "INPUT_09_MATCH_CHANNELS_ROOT_DIR",
    "data/output/09_user_business_match_channels_v2",
)
INPUT_09_MATCH_CHANNELS_RUN_DIR = os.getenv("INPUT_09_MATCH_CHANNELS_RUN_DIR", "").strip()
INPUT_09_TEXT_MATCH_ROOT = env_or_project_path(
    "INPUT_09_TEXT_MATCH_ROOT_DIR",
    "data/output/09_candidate_wise_text_match_features_v1",
)
INPUT_09_TEXT_MATCH_RUN_DIR = os.getenv("INPUT_09_TEXT_MATCH_RUN_DIR", "").strip()
INPUT_09_GROUP_GAP_ROOT = env_or_project_path(
    "INPUT_09_GROUP_GAP_ROOT_DIR",
    "data/output/09_stage10_group_gap_features_v1",
)
INPUT_09_GROUP_GAP_RUN_DIR = os.getenv("INPUT_09_GROUP_GAP_RUN_DIR", "").strip()
INPUT_10_ROOT = env_or_project_path("INPUT_10_ROOT_DIR", "data/output/p0_stage10_context_eval")
INPUT_10_RUN_DIR = os.getenv("INPUT_10_RUN_DIR", "").strip()

INPUT_11_2_RUN_DIR = os.getenv("INPUT_11_2_RUN_DIR", "").strip()
INPUT_11_2_RUN_DIR_11_30 = os.getenv("INPUT_11_2_RUN_DIR_11_30", "").strip()
INPUT_11_2_RUN_DIR_31_60 = os.getenv("INPUT_11_2_RUN_DIR_31_60", "").strip()
INPUT_11_2_RUN_DIR_61_100 = os.getenv("INPUT_11_2_RUN_DIR_61_100", "").strip()
INPUT_11_2_ROOT = env_or_project_path("INPUT_11_2_ROOT_DIR", "data/output/11_qlora_models")
INPUT_11_2_SUFFIX = os.getenv("INPUT_11_2_SUFFIX", "_stage11_2_rm_train").strip() or "_stage11_2_rm_train"
INPUT_11_DATA_RUN_DIR = os.getenv("INPUT_11_DATA_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path("OUTPUT_11_SIDECAR_ROOT_DIR", "data/output/11_qlora_sidecar_eval")
METRICS_PATH = env_or_project_path("METRICS_STAGE11_SIDECAR_PATH", "data/metrics/recsys_stage11_qlora_sidecar_results.csv")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
TOP_K = int(os.getenv("RANK_EVAL_TOP_K", "10").strip() or 10)
# `RERANK_TOPN` is the candidate window rescored by Stage11.
# `TOP_K` remains the final metric cutoff reported to users.
RERANK_TOPN = int(os.getenv("QLORA_RERANK_TOPN", "80").strip() or 80)
BLEND_ALPHA = float(os.getenv("QLORA_BLEND_ALPHA", "0.12").strip() or 0.12)
BLEND_ALPHA_11_30 = float(os.getenv("QLORA_BLEND_ALPHA_11_30", str(BLEND_ALPHA)).strip() or BLEND_ALPHA)
BLEND_ALPHA_31_60 = float(os.getenv("QLORA_BLEND_ALPHA_31_60", str(BLEND_ALPHA)).strip() or BLEND_ALPHA)
BLEND_ALPHA_31_40 = float(os.getenv("QLORA_BLEND_ALPHA_31_40", str(BLEND_ALPHA_31_60)).strip() or BLEND_ALPHA_31_60)
BLEND_ALPHA_41_60 = float(os.getenv("QLORA_BLEND_ALPHA_41_60", str(BLEND_ALPHA_31_60)).strip() or BLEND_ALPHA_31_60)
BLEND_ALPHA_61_100 = float(os.getenv("QLORA_BLEND_ALPHA_61_100", str(BLEND_ALPHA)).strip() or BLEND_ALPHA)
EVAL_ROUTE_LOCAL_NORM = os.getenv("QLORA_EVAL_ROUTE_LOCAL_NORM", "false").strip().lower() == "true"
EVAL_FORCE_REFUSION = os.getenv("QLORA_EVAL_FORCE_REFUSION", "false").strip().lower() == "true"
EVAL_FUSION_MODE = os.getenv("QLORA_EVAL_FUSION_MODE", "rescue_bonus").strip().lower() or "rescue_bonus"
EVAL_BASELINE_SCORE_COL = os.getenv("QLORA_EVAL_BASELINE_SCORE_COL", "learned_blend_score").strip() or "learned_blend_score"
EVAL_BASELINE_RANK_COL = os.getenv("QLORA_EVAL_BASELINE_RANK_COL", "learned_rank").strip() or "learned_rank"
EVAL_BASELINE_SCORE_NORM = os.getenv("QLORA_EVAL_BASELINE_SCORE_NORM", "rank_pct").strip().lower() or "rank_pct"
RESCUE_BONUS_MIN_RANK = int(os.getenv("QLORA_EVAL_RESCUE_BONUS_MIN_RANK", "11").strip() or 11)
RESCUE_BONUS_MAX_RANK = int(os.getenv("QLORA_EVAL_RESCUE_BONUS_MAX_RANK", "100").strip() or 100)
RESCUE_BONUS_BOUNDARY_WEIGHT = float(os.getenv("QLORA_EVAL_RESCUE_BONUS_BOUNDARY_WEIGHT", "1.0").strip() or 1.0)
RESCUE_BONUS_MID_WEIGHT = float(os.getenv("QLORA_EVAL_RESCUE_BONUS_MID_WEIGHT", "0.7").strip() or 0.7)
RESCUE_BONUS_MID_WEIGHT_31_40 = float(os.getenv("QLORA_EVAL_RESCUE_BONUS_MID_WEIGHT_31_40", str(RESCUE_BONUS_MID_WEIGHT)).strip() or RESCUE_BONUS_MID_WEIGHT)
RESCUE_BONUS_MID_WEIGHT_41_60 = float(os.getenv("QLORA_EVAL_RESCUE_BONUS_MID_WEIGHT_41_60", str(RESCUE_BONUS_MID_WEIGHT)).strip() or RESCUE_BONUS_MID_WEIGHT)
RESCUE_BONUS_DEEP_WEIGHT = float(os.getenv("QLORA_EVAL_RESCUE_BONUS_DEEP_WEIGHT", "0.35").strip() or 0.35)
MAX_USERS_PER_BUCKET = int(os.getenv("QLORA_EVAL_MAX_USERS_PER_BUCKET", "1200").strip() or 1200)
MAX_ROWS_PER_BUCKET = int(os.getenv("QLORA_EVAL_MAX_ROWS_PER_BUCKET", "200000").strip() or 200000)
ROW_CAP_ORDERED = os.getenv("QLORA_EVAL_ROW_CAP_ORDERED", "false").strip().lower() == "true"
EVAL_PROFILE = os.getenv("QLORA_EVAL_PROFILE", "custom").strip().lower() or "custom"
EVAL_USER_COHORT_PATH_RAW = os.getenv("QLORA_EVAL_USER_COHORT_PATH", "").strip()
EVAL_TARGET_TRUE_BANDS_RAW = os.getenv("QLORA_EVAL_TARGET_TRUE_BANDS", "").strip()
MAX_SEQ_LEN = int(os.getenv("QLORA_EVAL_MAX_SEQ_LEN", "768").strip() or 768)
PAD_TO_MULTIPLE_OF = int(os.getenv("QLORA_EVAL_PAD_TO_MULTIPLE_OF", "0").strip() or 0)
INFER_BATCH_SIZE = int(os.getenv("QLORA_EVAL_BATCH_SIZE", "12").strip() or 12)
PROMPT_CHUNK_ROWS = int(os.getenv("QLORA_EVAL_PROMPT_CHUNK_ROWS", "4096").strip() or 4096)
STREAM_LOG_ROWS = int(os.getenv("QLORA_EVAL_STREAM_LOG_ROWS", "4096").strip() or 4096)
ITER_COALESCE_PARTITIONS = int(os.getenv("QLORA_EVAL_ITER_COALESCE", "8").strip() or 8)
INTERMEDIATE_FLUSH_ROWS = int(os.getenv("QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS", "24576").strip() or 24576)
RANDOM_SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
USE_STAGE11_EVAL_SPLIT = os.getenv("QLORA_EVAL_USE_STAGE11_SPLIT", "true").strip().lower() == "true"
PROMPT_BUILD_MODE = os.getenv("QLORA_EVAL_PROMPT_BUILD_MODE", "driver").strip().lower() or "driver"
DRIVER_PROMPT_IMPL = os.getenv("QLORA_EVAL_DRIVER_PROMPT_IMPL", "apply").strip().lower() or "apply"
ARROW_TO_PANDAS = os.getenv("QLORA_EVAL_ARROW_TO_PANDAS", "false").strip().lower() == "true"
ARROW_FALLBACK = os.getenv("QLORA_EVAL_ARROW_FALLBACK", "false").strip().lower() == "true"
PRETOKENIZE_PROMPT_CHUNK = (
    os.getenv("QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK", "true").strip().lower() == "true"
)
GPU_PRELOAD_PROMPT_CHUNK = (
    os.getenv("QLORA_EVAL_GPU_PRELOAD_PROMPT_CHUNK", "false").strip().lower() == "true"
)
BUCKET_SORT_PROMPT_CHUNK = (
    os.getenv("QLORA_EVAL_BUCKET_SORT_PROMPT_CHUNK", "true").strip().lower() == "true"
)
LOCAL_TRIM_PROMPT_BATCH = (
    os.getenv("QLORA_EVAL_LOCAL_TRIM_PROMPT_BATCH", "true").strip().lower() == "true"
)
PIN_MEMORY = os.getenv("QLORA_EVAL_PIN_MEMORY", "true").strip().lower() == "true"
NON_BLOCKING_H2D = os.getenv("QLORA_EVAL_NON_BLOCKING_H2D", "true").strip().lower() == "true"
DUAL_BAND_ROUTE_ENABLED = (
    os.getenv("QLORA_EVAL_DUAL_BAND_ROUTE", "false").strip().lower() == "true"
    or bool(INPUT_11_2_RUN_DIR_11_30.strip())
    or bool(INPUT_11_2_RUN_DIR_31_60.strip())
    or bool(INPUT_11_2_RUN_DIR_61_100.strip())
)
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
PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "local_listwise_compare_rm").strip().lower() or "local_listwise_compare_rm"  # "full" | "full_lite" | "semantic" | "semantic_rm" | "semantic_compact_rm" | "semantic_compact_preserve_rm" | "semantic_compact_targeted_rm" | "local_listwise_compare_rm" | "sft_clean"
EVAL_MODEL_TYPE = os.getenv("QLORA_EVAL_MODEL_TYPE", "rm").strip().lower() or "rm"
RM_SCORE_NORM = os.getenv("QLORA_EVAL_RM_SCORE_NORM", "rank_pct").strip().lower() or "rank_pct"
EVAL_BASE_MODEL = os.getenv("QLORA_EVAL_BASE_MODEL", "").strip()
ATTACH_STAGE10_LEARNED = os.getenv("QLORA_ATTACH_STAGE10_LEARNED", "true").strip().lower() == "true"
ATTACH_STAGE10_TEXT_MATCH = os.getenv("QLORA_ATTACH_STAGE10_TEXT_MATCH", "false").strip().lower() == "true"
ATTACH_STAGE10_MATCH_CHANNELS = os.getenv("QLORA_ATTACH_STAGE10_MATCH_CHANNELS", "false").strip().lower() == "true"
ATTACH_STAGE10_GROUP_GAP = os.getenv("QLORA_ATTACH_STAGE10_GROUP_GAP", "false").strip().lower() == "true"
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
LOCAL_LISTWISE_COMPARE_PROMPT_MODE = "local_listwise_compare_rm"
LOCAL_LISTWISE_EVAL_MAX_RIVALS = max(
    2,
    int(os.getenv("QLORA_EVAL_LOCAL_LISTWISE_MAX_RIVALS", "4").strip() or 4),
)
EVAL_SECOND_STAGE_ENABLE = os.getenv("QLORA_EVAL_SECOND_STAGE_ENABLE", "false").strip().lower() == "true"
SECOND_STAGE_SHORTLIST_SIZE = max(
    0,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_SHORTLIST_SIZE", "12").strip() or 12),
)
SECOND_STAGE_GLOBAL_TOPK = max(
    0,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GLOBAL_TOPK", "8").strip() or 8),
)
SECOND_STAGE_ROUTE_31_40_TOPK = max(
    0,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_ROUTE_31_40_TOPK", "4").strip() or 4),
)
SECOND_STAGE_ROUTE_41_60_TOPK = max(
    0,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_ROUTE_41_60_TOPK", "3").strip() or 3),
)
SECOND_STAGE_ROUTE_61_100_TOPK = max(
    0,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_ROUTE_61_100_TOPK", "2").strip() or 2),
)
SECOND_STAGE_BLEND_ALPHA = float(
    os.getenv("QLORA_EVAL_SECOND_STAGE_BLEND_ALPHA", "0.12").strip() or 0.12
)
SECOND_STAGE_ROUTE_LOCAL_NORM = (
    os.getenv("QLORA_EVAL_SECOND_STAGE_ROUTE_LOCAL_NORM", "false").strip().lower() == "true"
)
SECOND_STAGE_GATE_ENABLE = (
    os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_ENABLE", "false").strip().lower() == "true"
)
SECOND_STAGE_GATE_TARGET_RANK_31_40 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_TARGET_RANK_31_40", "10").strip() or 10),
)
SECOND_STAGE_GATE_TARGET_RANK_41_60 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_TARGET_RANK_41_60", "20").strip() or 20),
)
SECOND_STAGE_GATE_TARGET_RANK_61_100 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_TARGET_RANK_61_100", "30").strip() or 30),
)
SECOND_STAGE_GATE_MAX_BLEND_RANK_31_40 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_MAX_BLEND_RANK_31_40", "15").strip() or 15),
)
SECOND_STAGE_GATE_MAX_BLEND_RANK_41_60 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_MAX_BLEND_RANK_41_60", "25").strip() or 25),
)
SECOND_STAGE_GATE_MAX_BLEND_RANK_61_100 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_MAX_BLEND_RANK_61_100", "40").strip() or 40),
)
SECOND_STAGE_GATE_MIN_MARGIN_31_40 = float(
    os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_MIN_MARGIN_31_40", "0.03").strip() or 0.03
)
SECOND_STAGE_GATE_MIN_MARGIN_41_60 = float(
    os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_MIN_MARGIN_41_60", "0.06").strip() or 0.06
)
SECOND_STAGE_GATE_MIN_MARGIN_61_100 = float(
    os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_MIN_MARGIN_61_100", "0.10").strip() or 0.10
)
SECOND_STAGE_GATE_CAP_RANK_31_40 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_CAP_RANK_31_40", "8").strip() or 8),
)
SECOND_STAGE_GATE_CAP_RANK_41_60 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_CAP_RANK_41_60", "10").strip() or 10),
)
SECOND_STAGE_GATE_CAP_RANK_61_100 = max(
    1,
    int(os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_CAP_RANK_61_100", "20").strip() or 20),
)
SECOND_STAGE_GATE_CAP_EPSILON = float(
    os.getenv("QLORA_EVAL_SECOND_STAGE_GATE_CAP_EPSILON", "0.0001").strip() or 0.0001
)
SECOND_STAGE_LOCAL_LISTWISE_MAX_RIVALS = max(
    2,
    int(
        os.getenv(
            "QLORA_EVAL_SECOND_STAGE_LOCAL_LISTWISE_MAX_RIVALS",
            str(max(2, SECOND_STAGE_SHORTLIST_SIZE - 1)),
        ).strip()
        or max(2, SECOND_STAGE_SHORTLIST_SIZE - 1)
    ),
)


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


def resolve_match_channel_run() -> Path:
    if INPUT_09_MATCH_CHANNELS_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_MATCH_CHANNELS_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_MATCH_CHANNELS_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_MATCH_CHANNELS_ROOT, "_full_stage09_user_business_match_channels_v2_build")


def resolve_text_match_run() -> Path:
    if INPUT_09_TEXT_MATCH_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_TEXT_MATCH_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_TEXT_MATCH_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_TEXT_MATCH_ROOT, "_full_stage09_candidate_wise_text_match_features_v1_build")


def resolve_group_gap_run() -> Path:
    if INPUT_09_GROUP_GAP_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_GROUP_GAP_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_GROUP_GAP_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_GROUP_GAP_ROOT, "_full_stage09_stage10_group_gap_features_v1_build")


def resolve_stage10_run() -> Path:
    if INPUT_10_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_10_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_10_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_10_ROOT, "_stage10_2_rank_infer_eval")


def _is_stage11_2_complete_run(run_dir: Path) -> bool:
    return run_dir.is_dir() and (run_dir / "run_meta.json").exists() and (run_dir / "adapter").is_dir()


def _is_stage11_2_checkpoint_dir(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    if not run_dir.name.startswith("checkpoint-"):
        return False
    has_adapter_weights = (
        (run_dir / "adapter_model.safetensors").exists()
        or (run_dir / "adapter_model.bin").exists()
    )
    return has_adapter_weights and (run_dir / "adapter_config.json").exists()


def _checkpoint_sort_key(run_dir: Path) -> tuple[int, float]:
    try:
        step = int(str(run_dir.name).split("checkpoint-", 1)[1])
    except Exception:
        step = -1
    try:
        mtime = float(run_dir.stat().st_mtime)
    except Exception:
        mtime = -1.0
    return step, mtime


def _pick_stage11_2_checkpoint(run_root: Path) -> Path | None:
    trainer_output = run_root / "trainer_output"
    if not trainer_output.is_dir():
        return None
    checkpoints = [p for p in trainer_output.iterdir() if _is_stage11_2_checkpoint_dir(p)]
    if not checkpoints:
        return None
    checkpoints.sort(key=_checkpoint_sort_key, reverse=True)
    return checkpoints[0]


def _resolve_stage11_2_meta_root(stage11_2_run: Path) -> Path | None:
    candidates: list[Path] = []
    if _is_stage11_2_checkpoint_dir(stage11_2_run):
        candidates.extend([stage11_2_run.parent.parent, stage11_2_run.parent, stage11_2_run])
    else:
        candidates.extend([stage11_2_run, stage11_2_run / "trainer_output"])
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "run_meta.json").exists():
            return candidate
    return None


def _resolve_stage11_2_run_from_inputs(
    explicit_run_dir: str,
    run_root: Path,
    run_suffix: str,
) -> Path:
    def _resolve_usable_path(run_dir: Path) -> Path | None:
        if _is_stage11_2_complete_run(run_dir) or _is_stage11_2_checkpoint_dir(run_dir):
            return run_dir
        ckpt = _pick_stage11_2_checkpoint(run_dir)
        if ckpt is not None:
            return ckpt
        return None

    preferred_pointer_names = (
        ["stage11_2_rm_train", "stage11_2_qlora_train"]
        if EVAL_MODEL_TYPE == "rm"
        else ["stage11_2_qlora_train", "stage11_2_rm_train"]
    )
    suffix_candidates = [str(run_suffix)]
    if EVAL_MODEL_TYPE == "rm":
        suffix_candidates.append("_stage11_2_rm_train")
    else:
        suffix_candidates.append("_stage11_2_qlora_train")
    suffixes: list[str] = []
    for suffix in suffix_candidates:
        clean = str(suffix or "").strip()
        if clean and clean not in suffixes:
            suffixes.append(clean)

    if explicit_run_dir:
        p = Path(explicit_run_dir)
        if not p.exists():
            raise FileNotFoundError(f"stage11_2 run not found: {p}")
        resolved = _resolve_usable_path(p)
        if resolved is None:
            raise FileNotFoundError(
                "stage11_2 run is incomplete "
                f"(need run_meta.json + adapter dir, or a checkpoint dir with adapter weights): {p}"
            )
        return resolved
    for pointer_name in preferred_pointer_names:
        pinned = resolve_latest_run_pointer(pointer_name)
        if pinned is not None:
            resolved = _resolve_usable_path(pinned)
            if resolved is not None:
                return resolved

    runs: list[Path] = []
    seen_run_dirs: set[str] = set()
    for suffix in suffixes:
        matched = [p for p in run_root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
        matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in matched:
            key = run_dir.as_posix()
            if key in seen_run_dirs:
                continue
            seen_run_dirs.add(key)
            runs.append(run_dir)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        resolved = _resolve_usable_path(r)
        if resolved is not None:
            return resolved
    if not runs:
        raise FileNotFoundError(
            f"no run found in {run_root} with suffixes={suffixes}"
        )
    checked = ", ".join([r.name for r in runs[:3]])
    raise FileNotFoundError(
        "no complete stage11_2 run found "
        "(need run_meta.json + adapter dir, or a checkpoint dir with adapter weights). "
        f"checked_latest={checked}"
    )


def resolve_stage11_2_run() -> Path:
    explicit_run_dir = INPUT_11_2_RUN_DIR_11_30.strip() or INPUT_11_2_RUN_DIR.strip()
    return _resolve_stage11_2_run_from_inputs(
        explicit_run_dir=explicit_run_dir,
        run_root=INPUT_11_2_ROOT,
        run_suffix=INPUT_11_2_SUFFIX,
    )


def resolve_stage11_2_run_secondary() -> Path | None:
    explicit_run_dir = INPUT_11_2_RUN_DIR_31_60.strip()
    if not explicit_run_dir:
        return None
    return _resolve_stage11_2_run_from_inputs(
        explicit_run_dir=explicit_run_dir,
        run_root=INPUT_11_2_ROOT,
        run_suffix=INPUT_11_2_SUFFIX,
    )


def resolve_stage11_2_run_tertiary() -> Path | None:
    explicit_run_dir = INPUT_11_2_RUN_DIR_61_100.strip()
    if not explicit_run_dir:
        return None
    return _resolve_stage11_2_run_from_inputs(
        explicit_run_dir=explicit_run_dir,
        run_root=INPUT_11_2_ROOT,
        run_suffix=INPUT_11_2_SUFFIX,
    )


def _iter_sidecar_band_run_paths() -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = [("boundary_11_30", "band_11_30", resolve_stage11_2_run())]
    secondary = resolve_stage11_2_run_secondary()
    if secondary is not None:
        out.append(("rescue_31_60", "band_31_60", secondary))
    tertiary = resolve_stage11_2_run_tertiary()
    if tertiary is not None:
        out.append(("rescue_61_100", "band_61_100", tertiary))
    return out


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
    bdir = resolve_stage11_bucket_parquet_dir(stage11_data_run, bucket)
    sdf = spark.read.parquet(bdir.as_posix())
    cols = set(sdf.columns)
    if "user_idx" not in cols or "split" not in cols:
        raise RuntimeError(f"stage11 all_parquet missing required columns user_idx/split: {bdir}")
    return sdf.filter(F.col("split") == F.lit("eval")).select("user_idx").dropDuplicates(["user_idx"])


def parse_target_true_bands(raw: str) -> set[str]:
    out: set[str] = set()
    for part in str(raw or "").split(","):
        token = str(part or "").strip().lower()
        if token in {"all", "any", "*", "none"}:
            return set()
        if token:
            out.add(token)
    return out


EVAL_TARGET_TRUE_BANDS = parse_target_true_bands(EVAL_TARGET_TRUE_BANDS_RAW)


@lru_cache(maxsize=16)
def _effective_pad_to_multiple(max_seq_len: int) -> int | None:
    requested = int(PAD_TO_MULTIPLE_OF) if int(PAD_TO_MULTIPLE_OF) > 0 else 0
    if requested <= 0:
        return None
    seq_len = max(0, int(max_seq_len or 0))
    if seq_len <= 0:
        return requested
    if seq_len % requested == 0:
        return requested
    for candidate in (32, 16, 8):
        if requested >= candidate and seq_len % candidate == 0:
            print(
                f"[WARN] adjust pad_to_multiple_of from {requested} to {candidate} "
                f"for max_seq_len={seq_len}"
            )
            return candidate
    print(
        f"[WARN] disable pad_to_multiple_of={requested} because it is incompatible "
        f"with max_seq_len={seq_len}"
    )
    return None


def _spark_boundary_rank_band(col: Any) -> Any:
    col_d = col.cast("double")
    return (
        F.when(col_d <= F.lit(10.0), F.lit("head_guard"))
        .when(col_d <= F.lit(30.0), F.lit("boundary_11_30"))
        .when(col_d <= F.lit(60.0), F.lit("rescue_31_60"))
        .when(col_d <= F.lit(100.0), F.lit("rescue_61_100"))
        .otherwise(F.lit("outside_100"))
    )


def resolve_stage11_bucket_parquet_dir(stage11_data_run: Path, bucket: int) -> Path:
    bucket_dir = stage11_data_run / f"bucket_{int(bucket)}"
    candidates = [
        bucket_dir / "all_parquet",
        bucket_dir / "pairwise_pool_all_parquet",
        bucket_dir / "pointwise_all_parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "stage11 parquet missing for bucket="
        f"{bucket}: tried {[p.as_posix() for p in candidates]}"
    )


def load_eval_users_from_stage11_target_bands(
    spark: SparkSession,
    stage11_data_run: Path,
    bucket: int,
    target_true_bands: set[str],
) -> DataFrame:
    if not target_true_bands:
        raise ValueError("target_true_bands is empty")
    bdir = resolve_stage11_bucket_parquet_dir(stage11_data_run, bucket)
    sdf = spark.read.parquet(bdir.as_posix())
    cols = set(sdf.columns)
    if "user_idx" not in cols or "label" not in cols:
        raise RuntimeError(f"stage11 all_parquet missing required columns user_idx/label: {bdir}")
    rank_col = "learned_rank" if "learned_rank" in cols else ("pre_rank" if "pre_rank" in cols else "")
    if not rank_col:
        raise RuntimeError(f"stage11 all_parquet missing learned_rank/pre_rank: {bdir}")
    return (
        sdf.filter(F.col("label") == F.lit(1))
        .withColumn("target_true_band", _spark_boundary_rank_band(F.col(rank_col)))
        .filter(F.col("target_true_band").isin(sorted(list(target_true_bands))))
        .select("user_idx")
        .dropDuplicates(["user_idx"])
    )


def resolve_eval_user_cohort_path() -> Path | None:
    raw = str(EVAL_USER_COHORT_PATH_RAW or "").strip()
    if not raw:
        return None
    p = normalize_legacy_project_path(raw)
    if p.exists():
        return p
    raise FileNotFoundError(f"QLORA_EVAL_USER_COHORT_PATH not found: {p}")


def load_eval_users_from_cohort(spark: SparkSession, cohort_path: Path) -> tuple[DataFrame, int, list[str]]:
    suffix = cohort_path.suffix.strip().lower()
    if cohort_path.is_dir() or suffix == ".parquet":
        pdf = pd.read_parquet(cohort_path.as_posix())
    elif suffix == ".csv":
        pdf = pd.read_csv(cohort_path.as_posix())
    else:
        raise ValueError(
            f"Unsupported cohort file format: {cohort_path}. Expected .csv or .parquet."
        )
    cols = [c for c in ("user_id", "user_idx") if c in pdf.columns]
    if not cols:
        raise RuntimeError(f"cohort file missing user_id/user_idx columns: {cohort_path}")
    out = pd.DataFrame()
    if "user_id" in cols:
        out["user_id"] = pdf["user_id"].astype(str).str.strip()
        out["user_id"] = out["user_id"].replace({"": pd.NA})
    if "user_idx" in cols:
        out["user_idx"] = pd.to_numeric(pdf["user_idx"], errors="coerce").astype("Int64")
    subset = [c for c in ("user_id", "user_idx") if c in out.columns]
    out = out.dropna(subset=subset, how="all")
    if "user_id" in out.columns:
        out = out.dropna(subset=["user_id"])
    if "user_idx" in out.columns:
        out = out.dropna(subset=["user_idx"])
        out["user_idx"] = out["user_idx"].astype("int64")
        out = out.sort_values("user_idx", kind="stable")
    elif "user_id" in out.columns:
        out = out.sort_values("user_id", kind="stable")
    out = out.drop_duplicates(subset=subset).reset_index(drop=True)
    return spark.createDataFrame(out), int(len(out)), subset


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
        "candidates_pretrim.parquet",
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates.parquet",
    ]
    for name in candidates:
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def canonical_bucket_scores_path(out_dir: Path, bucket: int) -> Path:
    return out_dir / f"stage11_bucket_{bucket}_scores.csv"


def legacy_bucket_scores_path(out_dir: Path, bucket: int) -> Path:
    return out_dir / f"bucket_{bucket}_scores.csv"


def canonical_bucket_partial_scores_path(out_dir: Path, bucket: int) -> Path:
    return out_dir / f"stage11_bucket_{bucket}_scores_partial.csv"


def legacy_bucket_partial_scores_path(out_dir: Path, bucket: int) -> Path:
    return out_dir / f"bucket_{bucket}_scores_partial.csv"


def choose_existing_score_path(canonical_path: Path, legacy_path: Path) -> Path:
    if canonical_path.exists():
        return canonical_path
    return legacy_path


def sync_csv_aliases(canonical_path: Path, legacy_path: Path) -> None:
    if canonical_path == legacy_path or not canonical_path.exists():
        return
    shutil.copyfile(canonical_path, legacy_path)


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


def attach_stage10_match_channel_features(spark: SparkSession, cand_raw: DataFrame) -> DataFrame:
    if not ATTACH_STAGE10_MATCH_CHANNELS:
        return cand_raw
    cols = set(cand_raw.columns)
    if "user_id" not in cols or "business_id" not in cols:
        return cand_raw
    feature_cols = [
        "channel_preference_core_v1",
        "channel_recent_intent_v1",
        "channel_context_time_v1",
        "channel_conflict_v1",
        "channel_evidence_support_v1",
    ]
    missing_cols = [c for c in feature_cols if c not in cols]
    if not missing_cols:
        print("[STAGE11-EVAL] match_channels attach skipped; columns already present")
        return cand_raw
    match_run = resolve_match_channel_run()
    channel_path = match_run / "user_business_match_channels_v2_user_item.parquet"
    if not channel_path.exists():
        raise FileNotFoundError(f"user-business channel parquet missing: {channel_path}")
    channel_df = (
        spark.read.parquet(channel_path.as_posix())
        .select(
            F.col("user_id").cast("string").alias("user_id"),
            F.col("business_id").cast("string").alias("business_id"),
            *[
                F.col(f"mean_{c}").cast("double").alias(c)
                for c in missing_cols
            ],
        )
        .dropDuplicates(["user_id", "business_id"])
    )
    print(f"[STAGE11-EVAL] attach match_channels run={match_run} cols={missing_cols}")
    return cand_raw.join(F.broadcast(channel_df), on=["user_id", "business_id"], how="left")


def attach_stage10_text_match_features(spark: SparkSession, cand_raw: DataFrame) -> DataFrame:
    if not ATTACH_STAGE10_TEXT_MATCH:
        return cand_raw
    cols = set(cand_raw.columns)
    if "user_idx" not in cols or "item_idx" not in cols:
        return cand_raw
    feature_cols = [
        "sim_negative_avoid_neg",
        "sim_negative_avoid_core",
        "sim_conflict_gap",
    ]
    missing_cols = [c for c in feature_cols if c not in cols]
    if not missing_cols:
        print("[STAGE11-EVAL] text_match attach skipped; columns already present")
        return cand_raw
    text_run = resolve_text_match_run()
    text_path = text_run / "candidate_text_match_features_v1.parquet"
    if not text_path.exists():
        raise FileNotFoundError(f"candidate text match parquet missing: {text_path}")
    text_df = (
        spark.read.parquet(text_path.as_posix())
        .select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            *[F.col(c).cast("double").alias(c) for c in missing_cols],
        )
        .dropDuplicates(["user_idx", "item_idx"])
    )
    print(f"[STAGE11-EVAL] attach text_match run={text_run} cols={missing_cols}")
    return cand_raw.join(F.broadcast(text_df), on=["user_idx", "item_idx"], how="left")


def attach_stage10_group_gap_features(spark: SparkSession, cand_raw: DataFrame) -> DataFrame:
    if not ATTACH_STAGE10_GROUP_GAP:
        return cand_raw
    cols = set(cand_raw.columns)
    if "user_idx" not in cols or "item_idx" not in cols:
        return cand_raw
    feature_cols = [
        "schema_weighted_overlap_user_ratio_v2_rank_pct_v3",
        "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3",
        "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3",
        "schema_weighted_net_score_v2_rank_pct_v3",
    ]
    missing_cols = [c for c in feature_cols if c not in cols]
    if not missing_cols:
        print("[STAGE11-EVAL] group_gap attach skipped; columns already present")
        return cand_raw
    gap_run = resolve_group_gap_run()
    gap_path = gap_run / "candidate_group_gap_features_v1.parquet"
    if not gap_path.exists():
        raise FileNotFoundError(f"candidate group-gap parquet missing: {gap_path}")
    gap_df = (
        spark.read.parquet(gap_path.as_posix())
        .select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            *[F.col(c).cast("double").alias(c) for c in missing_cols],
        )
        .dropDuplicates(["user_idx", "item_idx"])
    )
    print(f"[STAGE11-EVAL] attach group_gap run={gap_run} cols={missing_cols}")
    return cand_raw.join(F.broadcast(gap_df), on=["user_idx", "item_idx"], how="left")


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
    bucket = int(bucket_dir.name.split("_")[-1])
    cand_business = (
        spark.read.parquet(cand_path.as_posix())
        .transform(lambda df: attach_stage10_learned_features(spark, df, bucket))
        .filter(
            F.col(str(EVAL_BASELINE_RANK_COL)).isNotNull()
            & (F.col(str(EVAL_BASELINE_RANK_COL)).cast("double") <= F.lit(float(RERANK_TOPN)))
        )
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


def build_prompt_udf_semantic_rm() -> Any:
    """Single-candidate scoring prompt for reward-model eval."""

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
        return build_scoring_prompt(
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


def build_prompt_udf_semantic_compact_rm() -> Any:
    """Single-candidate scoring prompt with compact structured hints for RM eval."""

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
        group_gap_rank_pct: Any,
        group_gap_to_top3: Any,
        group_gap_to_top10: Any,
        net_score_rank_pct: Any,
        avoid_neg: Any,
        avoid_core: Any,
        conflict_gap: Any,
        channel_preference_core_v1: Any,
        channel_recent_intent_v1: Any,
        channel_context_time_v1: Any,
        channel_conflict_v1: Any,
        channel_evidence_support_v1: Any,
    ) -> str:
        return build_scoring_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text_semantic_compact(
                name=name,
                city=city,
                categories=categories,
                primary_category=primary_category,
                top_pos_tags=top_pos_tags,
                top_neg_tags=top_neg_tags,
                semantic_score=semantic_score,
                semantic_confidence=semantic_confidence,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
                group_gap_rank_pct=group_gap_rank_pct,
                group_gap_to_top3=group_gap_to_top3,
                group_gap_to_top10=group_gap_to_top10,
                net_score_rank_pct=net_score_rank_pct,
                avoid_neg=avoid_neg,
                avoid_core=avoid_core,
                conflict_gap=conflict_gap,
                channel_preference_core_v1=channel_preference_core_v1,
                channel_recent_intent_v1=channel_recent_intent_v1,
                channel_context_time_v1=channel_context_time_v1,
                channel_conflict_v1=channel_conflict_v1,
                channel_evidence_support_v1=channel_evidence_support_v1,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_semantic_compact_preserve_rm() -> Any:
    """Single-candidate scoring prompt with preserve-oriented structured hints for RM eval."""

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
        group_gap_rank_pct: Any,
        group_gap_to_top3: Any,
        group_gap_to_top10: Any,
        net_score_rank_pct: Any,
        avoid_neg: Any,
        avoid_core: Any,
        conflict_gap: Any,
        channel_preference_core_v1: Any,
        channel_recent_intent_v1: Any,
        channel_context_time_v1: Any,
        channel_conflict_v1: Any,
        channel_evidence_support_v1: Any,
        source_set_text: Any,
        source_count: Any,
        nonpopular_source_count: Any,
        profile_cluster_source_count: Any,
        context_rank: Any,
    ) -> str:
        return build_scoring_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text_semantic_compact_preserve(
                name=name,
                city=city,
                categories=categories,
                primary_category=primary_category,
                top_pos_tags=top_pos_tags,
                top_neg_tags=top_neg_tags,
                semantic_score=semantic_score,
                semantic_confidence=semantic_confidence,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
                group_gap_rank_pct=group_gap_rank_pct,
                group_gap_to_top3=group_gap_to_top3,
                group_gap_to_top10=group_gap_to_top10,
                net_score_rank_pct=net_score_rank_pct,
                avoid_neg=avoid_neg,
                avoid_core=avoid_core,
                conflict_gap=conflict_gap,
                channel_preference_core_v1=channel_preference_core_v1,
                channel_recent_intent_v1=channel_recent_intent_v1,
                channel_context_time_v1=channel_context_time_v1,
                channel_conflict_v1=channel_conflict_v1,
                channel_evidence_support_v1=channel_evidence_support_v1,
                source_set=source_set_text,
                source_count=source_count,
                nonpopular_source_count=nonpopular_source_count,
                profile_cluster_source_count=profile_cluster_source_count,
                context_rank=context_rank,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_semantic_compact_targeted_rm() -> Any:
    """Single-candidate scoring prompt with rank-aware targeted structured hints for RM eval."""

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
        pre_rank: Any,
        group_gap_rank_pct: Any,
        group_gap_to_top3: Any,
        group_gap_to_top10: Any,
        net_score_rank_pct: Any,
        avoid_neg: Any,
        avoid_core: Any,
        conflict_gap: Any,
        channel_preference_core_v1: Any,
        channel_recent_intent_v1: Any,
        channel_context_time_v1: Any,
        channel_conflict_v1: Any,
        channel_evidence_support_v1: Any,
        source_set_text: Any,
        source_count: Any,
        nonpopular_source_count: Any,
        profile_cluster_source_count: Any,
        context_rank: Any,
    ) -> str:
        return build_scoring_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                evidence_snippets=profile_text_evidence,
                pair_evidence=pair_evidence_summary,
            ),
            build_item_text_semantic_compact_targeted(
                name=name,
                city=city,
                categories=categories,
                primary_category=primary_category,
                top_pos_tags=top_pos_tags,
                top_neg_tags=top_neg_tags,
                semantic_score=semantic_score,
                semantic_confidence=semantic_confidence,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
                pre_rank=pre_rank,
                group_gap_rank_pct=group_gap_rank_pct,
                group_gap_to_top3=group_gap_to_top3,
                group_gap_to_top10=group_gap_to_top10,
                net_score_rank_pct=net_score_rank_pct,
                avoid_neg=avoid_neg,
                avoid_core=avoid_core,
                conflict_gap=conflict_gap,
                channel_preference_core_v1=channel_preference_core_v1,
                channel_recent_intent_v1=channel_recent_intent_v1,
                channel_context_time_v1=channel_context_time_v1,
                channel_conflict_v1=channel_conflict_v1,
                channel_evidence_support_v1=channel_evidence_support_v1,
                source_set=source_set_text,
                source_count=source_count,
                nonpopular_source_count=nonpopular_source_count,
                profile_cluster_source_count=profile_cluster_source_count,
                context_rank=context_rank,
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
        pre_rank: Any,
        pre_score: Any,
        group_gap_rank_pct: Any,
        group_gap_to_top3: Any,
        group_gap_to_top10: Any,
        net_score_rank_pct: Any,
        avoid_neg: Any,
        avoid_core: Any,
        conflict_gap: Any,
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
                pre_rank=pre_rank,
                pre_score=pre_score,
                group_gap_rank_pct=group_gap_rank_pct,
                group_gap_to_top3=group_gap_to_top3,
                group_gap_to_top10=group_gap_to_top10,
                net_score_rank_pct=net_score_rank_pct,
                avoid_neg=avoid_neg,
                avoid_core=avoid_core,
                conflict_gap=conflict_gap,
                source_set=source_set_text,
                user_segment=user_segment,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_snippet=item_evidence_text,
                pair_evidence_summary=pair_evidence_summary,
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
            f"pad_to_multiple_of={int(_effective_pad_to_multiple(max_seq_len) or 0)}"
        )
    t0 = time.monotonic()
    scored_total = 0
    last_log_time = t0
    prepared_prompts = prompts
    if template_use_chat:
        prepared_prompts = [_wrap_prompt_for_chat(p) for p in prompts]

    tokenized_prompts: dict[str, torch.Tensor] | None = None
    if prepared_prompts and PRETOKENIZE_PROMPT_CHUNK:
        pad_multiple = _effective_pad_to_multiple(max_seq_len)
        tokenized_prompts = tokenizer(
            prepared_prompts,
            padding=True,
            pad_to_multiple_of=pad_multiple,
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
                    pad_multiple = _effective_pad_to_multiple(max_seq_len)
                    enc = tokenizer(
                        batch,
                        padding=True,
                        pad_to_multiple_of=pad_multiple,
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


def set_active_adapter(model: Any, adapter_name: str) -> None:
    if not adapter_name:
        return
    setter = getattr(model, "set_adapter", None)
    if setter is None:
        return
    setter(adapter_name)


def score_reward_model(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_seq_len: int,
) -> tuple[np.ndarray, int]:
    def _rm_forward_kwargs() -> dict[str, Any]:
        # RM eval only needs final sequence scores; disable extra structured outputs
        # and KV cache to trim Python/object overhead and avoid useless cache work.
        kwargs: dict[str, Any] = {"return_dict": False}
        try:
            model_cfg = getattr(model, "config", None)
            if hasattr(model_cfg, "use_cache"):
                kwargs["use_cache"] = False
        except Exception:
            pass
        return kwargs

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
        except Exception as cache_exc:
            short = str(cache_exc).splitlines()[0][:180]
            print(f"[WARN] torch.cuda.empty_cache failed: {short}")

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

    def _trim_tokenized_batch_to_local_max(
        batch_tensors: dict[str, torch.Tensor],
        local_max: int,
    ) -> dict[str, torch.Tensor]:
        # Batch-local trim is only helpful when a whole prompt chunk was padded to the
        # chunk-global max width. Use CPU-precomputed valid lengths to avoid GPU sync.
        if local_max <= 0:
            return batch_tensors
        attn = batch_tensors.get("attention_mask")
        if not isinstance(attn, torch.Tensor) or attn.ndim != 2:
            return batch_tensors
        seq_width = int(attn.shape[1])
        if seq_width <= 1 or local_max >= seq_width:
            return batch_tensors
        trimmed: dict[str, torch.Tensor] = {}
        for key, value in batch_tensors.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[1]) == seq_width:
                trimmed[key] = value[:, :local_max]
            else:
                trimmed[key] = value
        return trimmed

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

    def _bucket_sort_tokenized_chunk(
        batch_tensors: dict[str, torch.Tensor],
        prompt_lengths: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray | None]:
        if not BUCKET_SORT_PROMPT_CHUNK:
            return batch_tensors, prompt_lengths, None
        if prompt_lengths.ndim != 1 or prompt_lengths.size <= max(1, int(batch_size)):
            return batch_tensors, prompt_lengths, None
        bucket_width = max(16, int(_effective_pad_to_multiple(max_seq_len) or 16))
        bucket_ids = ((prompt_lengths.astype(np.int32) + bucket_width - 1) // bucket_width).astype(np.int32)
        order = np.argsort(bucket_ids, kind="stable")
        if np.array_equal(order, np.arange(order.size, dtype=order.dtype)):
            return batch_tensors, prompt_lengths, None
        order_t = torch.as_tensor(order, dtype=torch.long)
        sorted_batch: dict[str, torch.Tensor] = {}
        for key, value in batch_tensors.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == int(order.size):
                sorted_batch[key] = value.index_select(0, order_t)
            else:
                sorted_batch[key] = value
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(order.size, dtype=order.dtype)
        return sorted_batch, prompt_lengths[order], inverse_order

    out: list[np.ndarray] = []
    model.eval()
    effective_bs = max(1, int(batch_size))
    t0 = time.monotonic()
    scored_total = 0
    last_log_time = t0
    rm_forward_kwargs = _rm_forward_kwargs()
    if prompts:
        print(
            f"[CONFIG] pretokenize_prompt_chunk={PRETOKENIZE_PROMPT_CHUNK} "
            f"gpu_preload_prompt_chunk={GPU_PRELOAD_PROMPT_CHUNK and torch.cuda.is_available()} "
            f"bucket_sort_prompt_chunk={BUCKET_SORT_PROMPT_CHUNK} "
            f"local_trim_prompt_batch={LOCAL_TRIM_PROMPT_BATCH} "
            f"pin_memory={PIN_MEMORY and torch.cuda.is_available()} "
            f"non_blocking_h2d={NON_BLOCKING_H2D and torch.cuda.is_available()} "
            f"pad_to_multiple_of={int(_effective_pad_to_multiple(max_seq_len) or 0)}"
        )

    tokenized_prompts: dict[str, torch.Tensor] | None = None
    gpu_tokenized_prompts: dict[str, torch.Tensor] | None = None
    tokenized_prompt_lengths: np.ndarray | None = None
    inverse_prompt_order: np.ndarray | None = None
    if prompts and PRETOKENIZE_PROMPT_CHUNK:
        pad_multiple = _effective_pad_to_multiple(max_seq_len)
        tokenized_prompts = tokenizer(
            prompts,
            padding=True,
            pad_to_multiple_of=pad_multiple,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        if LOCAL_TRIM_PROMPT_BATCH:
            try:
                attn_cpu = tokenized_prompts.get("attention_mask")
                if isinstance(attn_cpu, torch.Tensor) and attn_cpu.ndim == 2:
                    tokenized_prompt_lengths = (
                        attn_cpu.sum(dim=1).to(dtype=torch.int32).cpu().numpy()
                    )
            except Exception:
                tokenized_prompt_lengths = None
        if tokenized_prompt_lengths is not None:
            tokenized_prompts, tokenized_prompt_lengths, inverse_prompt_order = _bucket_sort_tokenized_chunk(
                tokenized_prompts,
                tokenized_prompt_lengths,
            )
        tokenized_prompts = _pin_tensor_batch(tokenized_prompts)
        print(
            f"[TIMING] chunk tokenization done: {len(prompts)} rows in "
            f"{time.monotonic() - t0:.1f}s"
        )
        if GPU_PRELOAD_PROMPT_CHUNK and torch.cuda.is_available():
            try:
                preload_t0 = time.monotonic()
                gpu_tokenized_prompts = _move_tensor_batch_to_device(tokenized_prompts, model.device)
                tokenized_prompts = None
                print(
                    f"[TIMING] chunk gpu preload done: {len(prompts)} rows in "
                    f"{time.monotonic() - preload_t0:.1f}s"
                )
            except torch.OutOfMemoryError:
                _safe_cuda_empty_cache()
                gpu_tokenized_prompts = None
                print("[WARN] RM eval gpu preload OOM; falling back to per-batch H2D")
            except RuntimeError as rte:
                msg = str(rte).lower()
                is_retryable = ("cuda" in msg) and (
                    ("cublas" in msg) or ("cudnn" in msg) or ("out of memory" in msg)
                )
                if not is_retryable:
                    raise
                _safe_cuda_empty_cache()
                gpu_tokenized_prompts = None
                short = str(rte).splitlines()[0][:180]
                print(
                    f"[WARN] RM eval gpu preload fallback; err={short}"
                )

    with torch.inference_mode():
        i = 0
        n = len(prompts)
        while i < n:
            bs = min(effective_bs, n - i)
            try:
                local_max = 0
                if tokenized_prompt_lengths is not None:
                    try:
                        local_max = int(tokenized_prompt_lengths[i : i + bs].max())
                    except Exception:
                        local_max = 0
                if gpu_tokenized_prompts is not None:
                    enc = {k: v[i : i + bs] for k, v in gpu_tokenized_prompts.items()}
                    if local_max > 0:
                        enc = _trim_tokenized_batch_to_local_max(enc, local_max)
                elif tokenized_prompts is not None:
                    enc = {k: v[i : i + bs] for k, v in tokenized_prompts.items()}
                    if local_max > 0:
                        enc = _trim_tokenized_batch_to_local_max(enc, local_max)
                    enc = _move_tensor_batch_to_device(enc, model.device)
                else:
                    batch = prompts[i : i + bs]
                    pad_multiple = _effective_pad_to_multiple(max_seq_len)
                    enc = tokenizer(
                        batch,
                        padding=True,
                        pad_to_multiple_of=pad_multiple,
                        truncation=True,
                        max_length=max_seq_len,
                        return_tensors="pt",
                    )
                    enc = _pin_tensor_batch(enc)
                    enc = _move_tensor_batch_to_device(enc, model.device)
                outputs = model(**enc, **rm_forward_kwargs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                score = reward_logits_to_score_tensor(logits)
                out.append(score.detach().float().cpu().numpy())
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
                        "reward-model sidecar inference OOM at batch_size=1. "
                        "Try lowering QLORA_EVAL_MAX_SEQ_LEN."
                    ) from oom
                effective_bs = _next_fallback_bs(bs)
                print(f"[WARN] RM eval OOM: batch={bs}, fallback_batch={effective_bs}")
            except RuntimeError as rte:
                msg = str(rte).lower()
                if "cuda" not in msg:
                    raise
                if "illegal memory access" in msg or "unspecified launch failure" in msg:
                    raise RuntimeError(
                        "reward-model sidecar inference hit CUDA kernel failure. "
                        "Restart Python process and retry with smaller batch size/seq len."
                    ) from rte
                is_retryable = ("cublas" in msg) or ("cudnn" in msg) or ("out of memory" in msg)
                if not is_retryable:
                    raise
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError(
                        "reward-model sidecar inference CUDA runtime failure at batch_size=1. "
                        "Try lowering QLORA_EVAL_MAX_SEQ_LEN."
                    ) from rte
                effective_bs = _next_fallback_bs(bs)
                short = str(rte).splitlines()[0][:180]
                print(
                    f"[WARN] RM eval CUDA failure: batch={bs}, "
                    f"fallback_batch={effective_bs}, err={short}"
                )
    total_time = time.monotonic() - t0
    if prompts:
        print(f"[TIMING] reward inference done: {len(prompts)} rows in {total_time:.1f}s ({len(prompts) / total_time:.1f} rows/s)")
    merged = np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)
    if inverse_prompt_order is not None and int(merged.shape[0]) == int(inverse_prompt_order.shape[0]):
        merged = merged[inverse_prompt_order]
    return merged.astype(np.float32), int(effective_bs)


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


def attach_stage10_learned_features(
    spark: SparkSession,
    cand_raw: DataFrame,
    bucket: int,
) -> DataFrame:
    if not ATTACH_STAGE10_LEARNED:
        return cand_raw
    cols = set(cand_raw.columns)
    if "user_idx" not in cols or "item_idx" not in cols:
        return cand_raw
    feature_cols = [
        "learned_blend_score",
        "learned_rank",
    ]
    missing_cols = [c for c in feature_cols if c not in cols]
    if not missing_cols:
        print("[STAGE11] stage10 learned attach skipped; columns already present")
        return cand_raw
    learned_run = resolve_stage10_run()
    learned_path = learned_run / f"bucket_{int(bucket)}" / "learned_scored.parquet"
    if not learned_path.exists():
        raise FileNotFoundError(f"stage10 learned scored parquet missing: {learned_path}")
    learned_df = (
        spark.read.parquet(learned_path.as_posix())
        .select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            *[F.col(c).cast("double").alias(c) for c in missing_cols],
        )
        .dropDuplicates(["user_idx", "item_idx"])
    )
    print(
        f"[STAGE11] attach stage10_learned run={learned_run} bucket={int(bucket)} cols={missing_cols}"
    )
    return cand_raw.join(F.broadcast(learned_df), on=["user_idx", "item_idx"], how="left")


def normalize_group_score(
    pdf: pd.DataFrame,
    score_col: str,
    method: str,
    group_cols: str | list[str] | tuple[str, ...] = "user_idx",
) -> pd.Series:
    mode = str(method or "rank_pct").strip().lower() or "rank_pct"
    if isinstance(group_cols, str):
        group_keys = [group_cols]
    else:
        group_keys = list(group_cols)

    def _minmax(s: pd.Series) -> pd.Series:
        a = s.min()
        b = s.max()
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
        return ((s - a) / (b - a + 1e-9)).astype(np.float32)

    def _rank_pct(s: pd.Series) -> pd.Series:
        n = int(len(s))
        if n <= 1:
            return pd.Series(np.zeros(n, dtype=np.float32), index=s.index)
        ranks = s.rank(method="average", ascending=True)
        return (((ranks - 1.0) / max(1.0, float(n - 1)))).astype(np.float32)

    def _zscore_sigmoid(s: pd.Series) -> pd.Series:
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-9:
            return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
        z = (s - mean) / std
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    if mode == "minmax":
        fn = _minmax
    elif mode == "zscore_sigmoid":
        fn = _zscore_sigmoid
    else:
        fn = _rank_pct
    return pdf.groupby(group_keys, sort=False)[score_col].transform(fn).astype(np.float32)


def resolve_baseline_columns(pdf: pd.DataFrame) -> tuple[str, str]:
    score_candidates = [str(EVAL_BASELINE_SCORE_COL or "").strip(), "learned_blend_score", "pre_score"]
    rank_candidates = [str(EVAL_BASELINE_RANK_COL or "").strip(), "learned_rank", "pre_rank"]
    score_col = "pre_score"
    rank_col = "pre_rank"
    for cand in score_candidates:
        if cand and cand in pdf.columns and pd.to_numeric(pdf[cand], errors="coerce").notna().any():
            score_col = cand
            break
    for cand in rank_candidates:
        if cand and cand in pdf.columns and pd.to_numeric(pdf[cand], errors="coerce").notna().any():
            rank_col = cand
            break
    return score_col, rank_col


def baseline_model_label(score_col: str) -> str:
    return "XGBBlend@10" if str(score_col or "").strip() == "learned_blend_score" else "PreScore@10"


def _rescue_band_weights(rank_series: pd.Series) -> pd.Series:
    ranks = pd.to_numeric(rank_series, errors="coerce").fillna(999999).astype(np.int32)
    weights = np.zeros(len(ranks), dtype=np.float32)
    route_segments = ranks.map(_route_segment_from_rank)
    weights[route_segments.to_numpy() == "boundary_11_30"] = float(RESCUE_BONUS_BOUNDARY_WEIGHT)
    weights[route_segments.to_numpy() == "rescue_31_40"] = float(RESCUE_BONUS_MID_WEIGHT_31_40)
    weights[route_segments.to_numpy() == "rescue_41_60"] = float(RESCUE_BONUS_MID_WEIGHT_41_60)
    weights[route_segments.to_numpy() == "rescue_61_100"] = float(RESCUE_BONUS_DEEP_WEIGHT)
    valid_mask = (ranks >= int(RESCUE_BONUS_MIN_RANK)) & (ranks <= int(RESCUE_BONUS_MAX_RANK))
    weights[~valid_mask.to_numpy()] = np.float32(0.0)
    return pd.Series(weights, index=rank_series.index, dtype=np.float32)


def _band_alpha_series(pdf: pd.DataFrame) -> pd.Series:
    if "band_alpha" in pdf.columns:
        return pd.to_numeric(pdf["band_alpha"], errors="coerce").fillna(0.0).astype(np.float32)
    if not DUAL_BAND_ROUTE_ENABLED:
        return pd.Series(np.full(len(pdf), float(BLEND_ALPHA), dtype=np.float32), index=pdf.index)
    route_bands = pdf["baseline_rank"].map(_route_segment_from_rank)
    alpha = np.zeros(len(pdf), dtype=np.float32)
    alpha[route_bands.to_numpy() == "boundary_11_30"] = float(BLEND_ALPHA_11_30)
    alpha[route_bands.to_numpy() == "rescue_31_40"] = float(BLEND_ALPHA_31_40)
    alpha[route_bands.to_numpy() == "rescue_41_60"] = float(BLEND_ALPHA_41_60)
    alpha[route_bands.to_numpy() == "rescue_61_100"] = float(BLEND_ALPHA_61_100)
    return pd.Series(alpha, index=pdf.index, dtype=np.float32)


def _sidecar_norm_group_cols(pdf: pd.DataFrame) -> list[str]:
    if not (DUAL_BAND_ROUTE_ENABLED and EVAL_ROUTE_LOCAL_NORM):
        return ["user_idx"]
    if "route_band" in pdf.columns:
        route_band = pdf["route_band"].fillna("unrouted")
    else:
        route_band = pdf["baseline_rank"].map(_route_segment_from_rank).fillna("unrouted")
    pdf["_sidecar_route_band"] = route_band.astype(str)
    return ["user_idx", "_sidecar_route_band"]


def apply_sidecar_fusion(pdf: pd.DataFrame, sidecar_raw_col: str) -> tuple[pd.DataFrame, str, str]:
    out = pdf.copy()
    base_score_col, base_rank_col = resolve_baseline_columns(out)
    out["baseline_score"] = pd.to_numeric(out[base_score_col], errors="coerce").fillna(0.0).astype(np.float32)
    out["baseline_rank"] = pd.to_numeric(out[base_rank_col], errors="coerce").fillna(999999).astype(np.int32)
    out["base_norm"] = normalize_group_score(
        out.assign(_baseline_score=out["baseline_score"]),
        "_baseline_score",
        EVAL_BASELINE_SCORE_NORM,
    )
    out["sidecar_score_present"] = pd.to_numeric(out[sidecar_raw_col], errors="coerce").notna().astype(np.int8)
    if EVAL_MODEL_TYPE == "rm":
        out["_sidecar_for_norm"] = pd.to_numeric(out[sidecar_raw_col], errors="coerce").astype(np.float32)
        sidecar_norm_group_cols = _sidecar_norm_group_cols(out)
        out["sidecar_norm"] = (
            normalize_group_score(out, "_sidecar_for_norm", RM_SCORE_NORM, group_cols=sidecar_norm_group_cols)
            .fillna(0.0)
            .astype(np.float32)
        )
        out.drop(columns=["_sidecar_for_norm"], inplace=True)
        if "_sidecar_route_band" in out.columns:
            out.drop(columns=["_sidecar_route_band"], inplace=True)
    else:
        out["sidecar_norm"] = pd.to_numeric(out[sidecar_raw_col], errors="coerce").fillna(0.0).astype(np.float32)
    out["band_alpha"] = _band_alpha_series(out)

    if EVAL_FUSION_MODE == "rescue_bonus":
        out["rescue_band_weight"] = _rescue_band_weights(out["baseline_rank"])
        out["rescue_bonus"] = (
            out["band_alpha"].astype(np.float32)
            * out["sidecar_norm"].astype(np.float32)
            * out["rescue_band_weight"].astype(np.float32)
        ).astype(np.float32)
        out["blend_score"] = (out["base_norm"].astype(np.float32) + out["rescue_bonus"]).astype(np.float32)
    else:
        out["rescue_band_weight"] = np.float32(0.0)
        out["rescue_bonus"] = np.float32(0.0)
        out["blend_score"] = (
            (1.0 - out["band_alpha"].astype(np.float32)) * out["base_norm"].astype(np.float32)
            + out["band_alpha"].astype(np.float32) * out["sidecar_norm"].astype(np.float32)
        ).astype(np.float32)
    return out, base_score_col, base_rank_col


def build_second_stage_shortlist(pdf: pd.DataFrame) -> pd.DataFrame:
    if (
        not EVAL_SECOND_STAGE_ENABLE
        or pdf.empty
        or int(SECOND_STAGE_SHORTLIST_SIZE) <= 1
    ):
        return pd.DataFrame(columns=["user_idx", "item_idx"])

    work = pdf.sort_values(["user_idx", "blend_score", "item_idx"], ascending=[True, False, True], kind="stable")
    selected_rows: list[dict[str, int]] = []
    for uid, g in work.groupby("user_idx", sort=False):
        picked: list[tuple[int, int]] = []
        seen_items: set[int] = set()
        route_band = g.get("route_band", pd.Series([""] * len(g), index=g.index)).astype(str)

        def _add(sub: pd.DataFrame, limit: int) -> None:
            if int(limit) <= 0 or len(picked) >= int(SECOND_STAGE_SHORTLIST_SIZE) or sub.empty:
                return
            for _, row in sub.head(int(limit)).iterrows():
                item_idx = int(_safe_rank_int(row.get("item_idx", -1), -1))
                if item_idx < 0 or item_idx in seen_items:
                    continue
                seen_items.add(item_idx)
                picked.append((int(uid), item_idx))
                if len(picked) >= int(SECOND_STAGE_SHORTLIST_SIZE):
                    break

        _add(g, int(SECOND_STAGE_GLOBAL_TOPK))
        _add(g.loc[route_band == "rescue_31_40"], int(SECOND_STAGE_ROUTE_31_40_TOPK))
        _add(g.loc[route_band == "rescue_41_60"], int(SECOND_STAGE_ROUTE_41_60_TOPK))
        _add(g.loc[route_band == "rescue_61_100"], int(SECOND_STAGE_ROUTE_61_100_TOPK))
        if len(picked) < int(SECOND_STAGE_SHORTLIST_SIZE):
            _add(g, int(SECOND_STAGE_SHORTLIST_SIZE) - len(picked))
        selected_rows.extend({"user_idx": u, "item_idx": i} for u, i in picked)

    if not selected_rows:
        return pd.DataFrame(columns=["user_idx", "item_idx"])
    return pd.DataFrame(selected_rows).drop_duplicates(["user_idx", "item_idx"])


def _attach_frontier_score(
    pdf: pd.DataFrame,
    score_col: str,
    rank_col: str,
    target_rank: int,
    out_col: str,
) -> pd.DataFrame:
    if pdf.empty:
        out = pdf.copy()
        out[out_col] = np.float32(np.nan)
        return out
    work = pdf.copy()
    work[rank_col] = pd.to_numeric(work.get(rank_col), errors="coerce").fillna(999999).astype(np.int32)
    frontier = (
        work.loc[work[rank_col] == int(target_rank), ["user_idx", score_col]]
        .rename(columns={score_col: out_col})
        .drop_duplicates(["user_idx"])
    )
    out = work.merge(frontier, on=["user_idx"], how="left")
    out[out_col] = pd.to_numeric(out.get(out_col), errors="coerce").astype(np.float32)
    return out


def _safe_rank_int(value: Any, default: int = 999999) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _eval_rank_from_row(row: dict[str, Any], default: int = 999999) -> int:
    learned_rank = _safe_rank_int(row.get("learned_rank", default), default)
    if int(learned_rank) < int(default):
        return int(max(1, learned_rank))
    return int(max(1, _safe_rank_int(row.get("pre_rank", default), default)))


def _eval_score_from_row(row: dict[str, Any]) -> float:
    try:
        val = row.get("learned_blend_score", None)
        if val is not None and np.isfinite(float(val)):
            return float(val)
    except Exception:
        pass
    try:
        return float(row.get("pre_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _eval_route_rank_from_row(row: Any, default: int = 999999) -> int:
    rank_col = str(EVAL_BASELINE_RANK_COL or "").strip()
    if rank_col == "pre_rank":
        return int(max(1, _safe_rank_int(getattr(row, "pre_rank", row.get("pre_rank", default) if isinstance(row, dict) else default), default)))
    learned_rank = _safe_rank_int(
        getattr(row, "learned_rank", row.get("learned_rank", default) if isinstance(row, dict) else default),
        default,
    )
    if int(learned_rank) < int(default):
        return int(max(1, learned_rank))
    return int(max(1, _safe_rank_int(getattr(row, "pre_rank", row.get("pre_rank", default) if isinstance(row, dict) else default), default)))


def _route_band_from_rank(rank_value: int) -> str:
    rank_int = int(_safe_rank_int(rank_value, 999999))
    if 11 <= rank_int <= 30:
        return "boundary_11_30"
    if 31 <= rank_int <= 60:
        return "rescue_31_60"
    if 61 <= rank_int <= 100:
        return "rescue_61_100"
    if 1 <= rank_int <= 10:
        return "top10_anchor"
    return "unrouted"


def _route_segment_from_rank(rank_value: int) -> str:
    rank_int = int(_safe_rank_int(rank_value, 999999))
    if 11 <= rank_int <= 30:
        return "boundary_11_30"
    if 31 <= rank_int <= 40:
        return "rescue_31_40"
    if 41 <= rank_int <= 60:
        return "rescue_41_60"
    if 61 <= rank_int <= 100:
        return "rescue_61_100"
    if 1 <= rank_int <= 10:
        return "top10_anchor"
    return "unrouted"


def _sidecar_route_from_rank(rank_value: int) -> tuple[str, str, float]:
    route_band = _route_segment_from_rank(rank_value)
    if not DUAL_BAND_ROUTE_ENABLED:
        return route_band, "global", float(BLEND_ALPHA)
    if route_band == "boundary_11_30":
        return route_band, "band_11_30", float(BLEND_ALPHA_11_30)
    if route_band == "rescue_31_40":
        return route_band, "band_31_60", float(BLEND_ALPHA_31_40)
    if route_band == "rescue_41_60":
        return route_band, "band_31_60", float(BLEND_ALPHA_41_60)
    if route_band == "rescue_61_100" and INPUT_11_2_RUN_DIR_61_100.strip():
        return route_band, "band_61_100", float(BLEND_ALPHA_61_100)
    return route_band, "", 0.0


def _select_eval_local_rivals(
    group_rows: list[dict[str, Any]],
    focus_row: dict[str, Any],
    max_rivals: int,
) -> list[dict[str, Any]]:
    focus_item_idx = _safe_rank_int(focus_row.get("item_idx", -1), -1)
    focus_rank = _eval_rank_from_row(focus_row, 999999)
    others = [
        row for row in group_rows
        if _safe_rank_int(row.get("item_idx", -1), -1) != focus_item_idx
    ]
    if not others:
        return []

    def _sort_rows(rows: list[dict[str, Any]], *, closer_first: bool = False) -> list[dict[str, Any]]:
        if closer_first:
            return sorted(
                rows,
                key=lambda row: (
                    abs(_eval_rank_from_row(row, 999999) - focus_rank),
                    _eval_rank_from_row(row, 999999),
                    -_eval_score_from_row(row),
                ),
            )
        return sorted(
            rows,
            key=lambda row: (
                _eval_rank_from_row(row, 999999),
                -_eval_score_from_row(row),
            ),
        )

    head_rows = _sort_rows([row for row in others if _eval_rank_from_row(row, 999999) <= 10])
    boundary_rows = _sort_rows([
        row for row in others
        if 11 <= _eval_rank_from_row(row, 999999) <= 30
    ], closer_first=True)
    mid_rows = _sort_rows([
        row for row in others
        if 31 <= _eval_rank_from_row(row, 999999) <= 60
    ], closer_first=True)
    deep_rows = _sort_rows([
        row for row in others
        if 61 <= _eval_rank_from_row(row, 999999) <= 100
    ], closer_first=True)
    other_rows = _sort_rows([
        row for row in others
        if _eval_rank_from_row(row, 999999) > 100
    ], closer_first=True)

    if focus_rank <= 10:
        quota_plan = [(boundary_rows, 2), (mid_rows, 2), (deep_rows, 1)]
    elif focus_rank <= 30:
        quota_plan = [(head_rows, 2), (boundary_rows, 2), (mid_rows, 1)]
    elif focus_rank <= 60:
        quota_plan = [(head_rows, 1), (boundary_rows, 2), (mid_rows, 1)]
    elif focus_rank <= 100:
        quota_plan = [(head_rows, 1), (boundary_rows, 1), (mid_rows, 2)]
    else:
        quota_plan = [(head_rows, 1), (boundary_rows, 2), (mid_rows, 1)]

    selected: list[dict[str, Any]] = []
    seen_item_idx: set[int] = set()

    def _take(rows: list[dict[str, Any]], limit: int) -> None:
        if limit <= 0:
            return
        built = 0
        for row in rows:
            item_idx = _safe_rank_int(row.get("item_idx", -1), -1)
            if item_idx < 0 or item_idx in seen_item_idx:
                continue
            seen_item_idx.add(item_idx)
            selected.append(row)
            built += 1
            if built >= limit or len(selected) >= int(max_rivals):
                break

    for rows, limit in quota_plan:
        if len(selected) >= int(max_rivals):
            break
        _take(rows, limit)

    if len(selected) < int(max_rivals):
        fill_rows = boundary_rows + head_rows + mid_rows + deep_rows + other_rows
        _take(fill_rows, int(max_rivals) - len(selected))

    return selected[: int(max_rivals)]


def _build_single_boundary_compare_prompt(row: dict[str, Any]) -> str:
    return build_scoring_prompt(
        _build_boundary_user_text_from_row(row),
        _build_boundary_item_text_from_row(row),
    )


def build_driver_local_listwise_prompts(
    cand_pdf: pd.DataFrame,
    max_rivals: int | None = None,
) -> pd.Series:
    if cand_pdf.empty:
        return pd.Series(dtype="string")
    rival_cap = int(max_rivals or LOCAL_LISTWISE_EVAL_MAX_RIVALS)
    prompts: list[str] = [""] * int(len(cand_pdf))
    ordered_pdf = cand_pdf.sort_values(["user_idx", "pre_rank", "item_idx"], kind="stable")
    for _, g in ordered_pdf.groupby("user_idx", sort=False):
        group_rows = g.to_dict("records")
        group_index = list(g.index)
        for idx, focus_row in zip(group_index, group_rows):
            rival_rows = _select_eval_local_rivals(
                group_rows,
                focus_row,
                rival_cap,
            )
            if rival_rows:
                focus_item_idx = _safe_rank_int(focus_row.get("item_idx", -1), -1)
                prompt_builder, _ = _prepare_local_listwise_prompt_context(focus_row, rival_rows)
                prompt = str(prompt_builder(focus_item_idx) if prompt_builder else "" or "").strip()
                if prompt:
                    prompts[int(idx)] = prompt
                    continue
            prompts[int(idx)] = _build_single_boundary_compare_prompt(focus_row)
    return pd.Series(prompts, index=cand_pdf.index, dtype="string")


def reward_logits_to_score_tensor(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.reshape(-1)
    if logits.ndim != 2:
        raise RuntimeError(f"unexpected reward logits shape: {tuple(logits.shape)}")
    if logits.shape[-1] == 1:
        return logits[:, 0]
    return (logits[:, -1] - logits[:, 0]).reshape(-1)


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
    if EVAL_MODEL_TYPE not in {"causal_yesno", "rm"}:
        raise ValueError(
            f"Unsupported QLORA_EVAL_MODEL_TYPE={EVAL_MODEL_TYPE!r}; expected 'causal_yesno' or 'rm'."
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
    stage11_2_meta_root = _resolve_stage11_2_meta_root(stage11_2_run)
    stage11_2_run_secondary = resolve_stage11_2_run_secondary()
    stage11_2_meta_root_secondary = (
        _resolve_stage11_2_meta_root(stage11_2_run_secondary) if stage11_2_run_secondary is not None else None
    )
    stage11_2_run_tertiary = resolve_stage11_2_run_tertiary()
    stage11_2_meta_root_tertiary = (
        _resolve_stage11_2_meta_root(stage11_2_run_tertiary) if stage11_2_run_tertiary is not None else None
    )

    if _is_stage11_2_checkpoint_dir(stage11_2_run):
        adapter_dir = stage11_2_run
    else:
        adapter_dir = stage11_2_run / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")

    adapter_dir_secondary: Path | None = None
    if stage11_2_run_secondary is not None:
        if _is_stage11_2_checkpoint_dir(stage11_2_run_secondary):
            adapter_dir_secondary = stage11_2_run_secondary
        else:
            adapter_dir_secondary = stage11_2_run_secondary / "adapter"
        if not adapter_dir_secondary.exists():
            raise FileNotFoundError(f"secondary adapter dir not found: {adapter_dir_secondary}")

    adapter_dir_tertiary: Path | None = None
    if stage11_2_run_tertiary is not None:
        if _is_stage11_2_checkpoint_dir(stage11_2_run_tertiary):
            adapter_dir_tertiary = stage11_2_run_tertiary
        else:
            adapter_dir_tertiary = stage11_2_run_tertiary / "adapter"
        if not adapter_dir_tertiary.exists():
            raise FileNotFoundError(f"tertiary adapter dir not found: {adapter_dir_tertiary}")

    stage11_meta_path = (stage11_2_meta_root / "run_meta.json") if stage11_2_meta_root is not None else None
    stage11_meta: dict[str, Any] = {}
    if stage11_meta_path is not None and stage11_meta_path.exists():
        stage11_meta = json.loads(stage11_meta_path.read_text(encoding="utf-8"))

    stage11_meta_secondary: dict[str, Any] = {}
    stage11_meta_path_secondary = (
        (stage11_2_meta_root_secondary / "run_meta.json") if stage11_2_meta_root_secondary is not None else None
    )
    if stage11_meta_path_secondary is not None and stage11_meta_path_secondary.exists():
        stage11_meta_secondary = json.loads(stage11_meta_path_secondary.read_text(encoding="utf-8"))

    stage11_meta_tertiary: dict[str, Any] = {}
    stage11_meta_path_tertiary = (
        (stage11_2_meta_root_tertiary / "run_meta.json") if stage11_2_meta_root_tertiary is not None else None
    )
    if stage11_meta_path_tertiary is not None and stage11_meta_path_tertiary.exists():
        stage11_meta_tertiary = json.loads(stage11_meta_path_tertiary.read_text(encoding="utf-8"))

    base_model = str(stage11_meta.get("base_model", "")).strip() or str(EVAL_BASE_MODEL or "").strip()
    if not base_model:
        raise RuntimeError(
            "base_model missing for stage11_2 eval; "
            f"set QLORA_EVAL_BASE_MODEL or ensure run_meta.json exists under {stage11_2_run}"
        )
    if stage11_meta_secondary:
        base_model_secondary = str(stage11_meta_secondary.get("base_model", "")).strip() or base_model
        if base_model_secondary != base_model:
            raise RuntimeError(
                "dual-band eval requires the same base_model for both adapters; "
                f"11_30={base_model} 31_60={base_model_secondary}"
            )
        for meta_key in ("source_stage11_dataset_run", "source_run_09"):
            left = str(stage11_meta.get(meta_key, "")).strip()
            right = str(stage11_meta_secondary.get(meta_key, "")).strip()
            if left and right and left != right:
                raise RuntimeError(
                    "dual-band eval requires aligned upstream lineage; "
                    f"mismatch on {meta_key}: {left} vs {right}"
                )
    if stage11_meta_tertiary:
        base_model_tertiary = str(stage11_meta_tertiary.get("base_model", "")).strip() or base_model
        if base_model_tertiary != base_model:
            raise RuntimeError(
                "band-routed eval requires the same base_model for all adapters; "
                f"11_30={base_model} 61_100={base_model_tertiary}"
            )
        for meta_key in ("source_stage11_dataset_run", "source_run_09"):
            left = str(stage11_meta.get(meta_key, "")).strip()
            right = str(stage11_meta_tertiary.get(meta_key, "")).strip()
            if left and right and left != right:
                raise RuntimeError(
                    "band-routed eval requires aligned upstream lineage; "
                    f"mismatch on {meta_key}: {left} vs {right}"
                )

    stage11_data_run = resolve_stage11_data_run(stage11_meta)
    stage09_run = resolve_stage09_run(stage11_meta=stage11_meta, stage11_data_run=stage11_data_run)
    print(
        f"[CONFIG] source_stage11_2={stage11_2_run} "
        f"stage11_2_meta_root={stage11_2_meta_root if stage11_2_meta_root is not None else '<none>'} "
        f"adapter_dir={adapter_dir} "
        f"dual_band_route={DUAL_BAND_ROUTE_ENABLED} "
        f"source_stage11_2_31_60={stage11_2_run_secondary if stage11_2_run_secondary is not None else '<none>'} "
        f"adapter_dir_31_60={adapter_dir_secondary if adapter_dir_secondary is not None else '<none>'} "
        f"source_stage11_2_61_100={stage11_2_run_tertiary if stage11_2_run_tertiary is not None else '<none>'} "
        f"adapter_dir_61_100={adapter_dir_tertiary if adapter_dir_tertiary is not None else '<none>'} "
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
        f"eval_model_type={EVAL_MODEL_TYPE} "
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
    tokenizer.padding_side = "right" if EVAL_MODEL_TYPE == "rm" else "left"

    if has_cuda and LOAD_MODEL_BEFORE_SPARK:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    base_model_cls: Any = AutoModelForSequenceClassification if EVAL_MODEL_TYPE == "rm" else AutoModelForCausalLM
    if EVAL_MODEL_TYPE == "rm":
        try:
            rm_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=TRUST_REMOTE_CODE)
            setattr(rm_cfg, "num_labels", 1)
            model_kwargs["config"] = rm_cfg
        except Exception:
            pass
    try:
        base = base_model_cls.from_pretrained(base_model, **model_kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if "1455" in msg or "paging file" in msg:
            raise RuntimeError(
                "model load failed with Windows os error 1455 (virtual memory/pagefile too small). "
                "Increase pagefile size (recommend >= 40GB), reboot to apply, then rerun stage11_3."
            ) from exc
        raise
    if EVAL_MODEL_TYPE == "rm":
        try:
            if getattr(getattr(base, "config", None), "pad_token_id", None) is None:
                base.config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
    primary_adapter_name = "band_11_30" if DUAL_BAND_ROUTE_ENABLED else "default"
    model = PeftModel.from_pretrained(base, adapter_dir.as_posix(), adapter_name=primary_adapter_name)
    if DUAL_BAND_ROUTE_ENABLED and adapter_dir_secondary is not None:
        model.load_adapter(adapter_dir_secondary.as_posix(), adapter_name="band_31_60")
    if DUAL_BAND_ROUTE_ENABLED and adapter_dir_tertiary is not None:
        model.load_adapter(adapter_dir_tertiary.as_posix(), adapter_name="band_61_100")
    if DUAL_BAND_ROUTE_ENABLED:
        set_active_adapter(model, primary_adapter_name)
    if EVAL_MODEL_TYPE == "rm":
        try:
            if getattr(getattr(model, "config", None), "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
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
    yes_id, no_id = (-1, -1)
    if EVAL_MODEL_TYPE != "rm":
        yes_id, no_id = build_yes_no_token_ids(tokenizer)
    sidecar_raw_col = "reward_score" if EVAL_MODEL_TYPE == "rm" else "qlora_prob"

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
    bucket_run_summaries: list[dict[str, Any]] = []
    for bucket in buckets:
        eval_user_count_requested = 0
        eval_user_count_surviving = 0
        eval_user_count_dropped = 0
        eval_user_join_cols: list[str] = []
        final_scores_path = canonical_bucket_scores_path(out_dir, int(bucket))
        final_scores_legacy_path = legacy_bucket_scores_path(out_dir, int(bucket))
        final_scores_resume_path = choose_existing_score_path(final_scores_path, final_scores_legacy_path)
        if final_scores_resume_path.exists() and os.getenv("QLORA_EVAL_RESUME_DIR", "").strip():
            print(f"[RESUME] skip bucket={bucket} inference, final scores already exist. Recalculating metrics.")
            pdf = pd.read_csv(final_scores_resume_path.as_posix())
            refusion_applied = False
            if EVAL_FORCE_REFUSION or "blend_score" not in pdf.columns or "baseline_score" not in pdf.columns:
                pdf, base_score_col, base_rank_col = apply_sidecar_fusion(pdf, sidecar_raw_col)
                refusion_applied = True
            else:
                base_score_col, base_rank_col = resolve_baseline_columns(pdf)
            if refusion_applied:
                # Keep resumed score artifacts aligned with the metrics that were just recomputed.
                pdf.to_csv(final_scores_path.as_posix(), index=False)
                sync_csv_aliases(final_scores_path, final_scores_legacy_path)
            pre_recall, pre_ndcg = evaluate_topk(pdf, "baseline_score", TOP_K)
            q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)
            n_users = int(pdf["user_idx"].nunique())
            n_items = int(pdf["item_idx"].nunique())
            n_rows = int(len(pdf))
            if eval_user_cohort_path is not None and int(eval_user_count_surviving) <= 0:
                eval_user_count_surviving = int(n_users)
                eval_user_count_dropped = int(max(0, eval_user_count_requested - eval_user_count_surviving))

            metrics_rows.append(
                {
                    "run_id_11": run_id,
                    "source_run_09": str(stage09_run),
                    "source_run_11_2": str(stage11_2_run),
                    "bucket_min_train_reviews": int(bucket),
                    "model": baseline_model_label(base_score_col),
                    "recall_at_k": float(pre_recall),
                    "ndcg_at_k": float(pre_ndcg),
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_candidates": n_rows,
                    "eval_user_count_requested": int(eval_user_count_requested),
                    "eval_user_count_surviving": int(eval_user_count_surviving),
                    "eval_user_count_dropped": int(eval_user_count_dropped),
                    "baseline_score_col": str(base_score_col),
                    "baseline_rank_col": str(base_rank_col),
                    "fusion_mode": str(EVAL_FUSION_MODE),
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
                    "eval_user_count_requested": int(eval_user_count_requested),
                    "eval_user_count_surviving": int(eval_user_count_surviving),
                    "eval_user_count_dropped": int(eval_user_count_dropped),
                    "baseline_score_col": str(base_score_col),
                    "baseline_rank_col": str(base_rank_col),
                    "fusion_mode": str(EVAL_FUSION_MODE),
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
            cohort_users, cohort_user_count, cohort_cols = load_eval_users_from_cohort(spark, eval_user_cohort_path)
            eval_user_count_requested = int(cohort_user_count)
            eval_user_join_cols = ["user_id"] if "user_id" in cohort_cols else ["user_idx"]
            print(
                f"[COHORT] bucket={bucket} loaded explicit user cohort: {eval_user_cohort_path} "
                f"users={cohort_user_count} join_cols={eval_user_join_cols}"
            )
            truth = truth.join(
                F.broadcast(cohort_users.select(*eval_user_join_cols).dropDuplicates(eval_user_join_cols)),
                on=eval_user_join_cols,
                how="inner",
            )
            users = truth.select("user_idx").dropDuplicates(["user_idx"])
        elif USE_STAGE11_EVAL_SPLIT:
            if stage11_data_run is None:
                raise RuntimeError("USE_STAGE11_EVAL_SPLIT=true but stage11_data_run is not resolved")
            eval_users = load_eval_users_from_stage11_data(spark, stage11_data_run, int(bucket))
            users = users.join(eval_users, on="user_idx", how="inner")
            truth = truth.join(users, on="user_idx", how="inner")
        if EVAL_TARGET_TRUE_BANDS:
            if stage11_data_run is None:
                raise RuntimeError(
                    "QLORA_EVAL_TARGET_TRUE_BANDS requires stage11_data_run; set INPUT_11_DATA_RUN_DIR or "
                    "ensure source_stage11_dataset_run exists in stage11_2 run_meta.json."
                )
            band_users = load_eval_users_from_stage11_target_bands(
                spark,
                stage11_data_run,
                int(bucket),
                EVAL_TARGET_TRUE_BANDS,
            )
            users = users.join(band_users, on="user_idx", how="inner")
            truth = truth.join(users, on="user_idx", how="inner")
            print(
                f"[COHORT] bucket={bucket} target_true_bands={sorted(list(EVAL_TARGET_TRUE_BANDS))} "
                "applied on eval users"
            )
        if MAX_USERS_PER_BUCKET > 0:
            users = users.orderBy(F.rand(int(RANDOM_SEED + bucket))).limit(int(MAX_USERS_PER_BUCKET))
            truth = truth.join(users, on="user_idx", how="inner")
        if eval_user_cohort_path is not None:
            print(
                f"[COHORT] bucket={bucket} requested={eval_user_count_requested} "
                "surviving=<defer-to-final-pdf> dropped=<defer-to-final-pdf>"
            )

        cand_raw = (
            spark.read.parquet(cand_path.as_posix())
            .join(users, on="user_idx", how="inner")
            .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
        )
        cand_raw = attach_stage10_learned_features(spark, cand_raw, bucket)
        cand_raw = cand_raw.filter(
            F.col(str(EVAL_BASELINE_RANK_COL)).isNotNull()
            & (F.col(str(EVAL_BASELINE_RANK_COL)).cast("double") <= F.lit(float(RERANK_TOPN)))
        )
        cand_raw = attach_stage10_text_match_features(spark, cand_raw)
        cand_raw = attach_stage10_match_channel_features(spark, cand_raw)
        cand_raw = attach_stage10_group_gap_features(spark, cand_raw)
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
                _num_col("learned_blend_score"),
                _num_col("learned_rank"),
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
                _num_col("schema_weighted_overlap_user_ratio_v2_rank_pct_v3"),
                _num_col("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3"),
                _num_col("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3"),
                _num_col("schema_weighted_net_score_v2_rank_pct_v3"),
                _num_col("sim_negative_avoid_neg"),
                _num_col("sim_negative_avoid_core"),
                _num_col("sim_conflict_gap"),
                _num_col("channel_preference_core_v1"),
                _num_col("channel_recent_intent_v1"),
                _num_col("channel_context_time_v1"),
                _num_col("channel_conflict_v1"),
                _num_col("channel_evidence_support_v1"),
                _num_col("source_count"),
                _num_col("nonpopular_source_count"),
                _num_col("profile_cluster_source_count"),
                _num_col("context_rank"),
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
                    "pre_rank": 0.0,
                    "pre_score": 0.0,
                    "learned_blend_score": 0.0,
                    "learned_rank": 0.0,
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
                    "schema_weighted_overlap_user_ratio_v2_rank_pct_v3": 0.0,
                    "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3": 0.0,
                    "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3": 0.0,
                    "schema_weighted_net_score_v2_rank_pct_v3": 0.0,
                    "sim_negative_avoid_neg": 0.0,
                    "sim_negative_avoid_core": 0.0,
                    "sim_conflict_gap": 0.0,
                    "channel_preference_core_v1": 0.0,
                    "channel_recent_intent_v1": 0.0,
                    "channel_context_time_v1": 0.0,
                    "channel_conflict_v1": 0.0,
                    "channel_evidence_support_v1": 0.0,
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

        partial_scores_path = canonical_bucket_partial_scores_path(out_dir, int(bucket))
        partial_scores_legacy_path = legacy_bucket_partial_scores_path(out_dir, int(bucket))
        partial_scores_resume_path = choose_existing_score_path(partial_scores_path, partial_scores_legacy_path)
        flushed_rows = 0
        scored_pairs_df = None

        if partial_scores_resume_path.exists() and os.getenv("QLORA_EVAL_RESUME_DIR", "").strip():
            print(f"[RESUME] Loading partial scores from {partial_scores_resume_path}")
            try:
                partial_df = pd.read_csv(partial_scores_resume_path.as_posix(), usecols=["user_idx", "item_idx"]).dropna()
                if not partial_df.empty:
                    flushed_rows = len(partial_df)
                    scored_pairs_df = spark.createDataFrame(
                        partial_df[["user_idx", "item_idx"]].astype(int)
                    )
            except Exception as e:
                print(f"[WARN] Failed to read partial scores: {e}. Will overwrite.")
                if partial_scores_resume_path.exists():
                    partial_scores_resume_path.unlink()
        elif partial_scores_resume_path.exists():
            partial_scores_resume_path.unlink()

        pending_user_idx: list[int] = []
        pending_item_idx: list[int] = []
        pending_pre_rank: list[int] = []
        pending_label_true: list[int] = []
        pending_pre_score: list[float] = []
        pending_learned_rank: list[int] = []
        pending_learned_blend_score: list[float] = []
        pending_sidecar_score: list[float] = []
        pending_route_band: list[str] = []
        pending_sidecar_model_id: list[str] = []
        pending_band_alpha: list[float] = []

        prompt_buf: list[str] = []
        user_idx_buf: list[int] = []
        item_idx_buf: list[int] = []
        pre_rank_buf: list[int] = []
        label_true_buf: list[int] = []
        pre_score_buf: list[float] = []
        learned_rank_buf: list[int] = []
        learned_blend_score_buf: list[float] = []
        route_band_buf: list[str] = []
        sidecar_model_id_buf: list[str] = []
        band_alpha_buf: list[float] = []
        default_adaptive_batch_size = max(1, int(INFER_BATCH_SIZE))
        adaptive_batch_size_by_model: dict[str, int] = {"global": default_adaptive_batch_size}
        if DUAL_BAND_ROUTE_ENABLED:
            adaptive_batch_size_by_model["band_11_30"] = default_adaptive_batch_size
            adaptive_batch_size_by_model["band_31_60"] = default_adaptive_batch_size
            if adapter_dir_tertiary is not None:
                adaptive_batch_size_by_model["band_61_100"] = default_adaptive_batch_size

        def _append_pending_row(
            row: Any,
            sidecar_score: float,
            route_band: str,
            sidecar_model_id: str,
            band_alpha: float,
        ) -> None:
            pending_user_idx.append(int(row["user_idx"]))
            pending_item_idx.append(int(row["item_idx"]))
            pending_pre_rank.append(int(row["pre_rank"]))
            pending_label_true.append(int(row["label_true"]))
            pending_pre_score.append(_float_or_zero(row["pre_score"]))
            pending_learned_rank.append(int(_safe_rank_int(row.get("learned_rank", 999999), 999999)))
            pending_learned_blend_score.append(_float_or_zero(row.get("learned_blend_score", 0.0)))
            pending_sidecar_score.append(float(sidecar_score))
            pending_route_band.append(str(route_band or ""))
            pending_sidecar_model_id.append(str(sidecar_model_id or ""))
            pending_band_alpha.append(float(band_alpha))
            if len(pending_user_idx) >= int(max(1, INTERMEDIATE_FLUSH_ROWS)):
                _append_partial_scores()

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
                    "learned_rank": np.asarray(pending_learned_rank, dtype=np.int32),
                    "learned_blend_score": np.asarray(pending_learned_blend_score, dtype=np.float32),
                    sidecar_raw_col: np.asarray(pending_sidecar_score, dtype=np.float32),
                    "route_band": np.asarray(pending_route_band, dtype=object),
                    "sidecar_model_id": np.asarray(pending_sidecar_model_id, dtype=object),
                    "band_alpha": np.asarray(pending_band_alpha, dtype=np.float32),
                }
            )
            write_header = not partial_scores_path.exists()
            chunk.to_csv(partial_scores_path.as_posix(), mode="a", header=write_header, index=False)
            sync_csv_aliases(partial_scores_path, partial_scores_legacy_path)
            flushed_rows += int(len(chunk))
            pending_user_idx.clear()
            pending_item_idx.clear()
            pending_pre_rank.clear()
            pending_label_true.clear()
            pending_pre_score.clear()
            pending_learned_rank.clear()
            pending_learned_blend_score.clear()
            pending_sidecar_score.clear()
            pending_route_band.clear()
            pending_sidecar_model_id.clear()
            pending_band_alpha.clear()

        def _score_prompt_subset(
            active_model_id: str,
            prompts_subset: list[str],
        ) -> tuple[np.ndarray, int, int]:
            prev_bs = int(adaptive_batch_size_by_model.get(active_model_id, default_adaptive_batch_size))
            if DUAL_BAND_ROUTE_ENABLED and active_model_id != "global":
                set_active_adapter(model, active_model_id)
            if EVAL_MODEL_TYPE == "rm":
                q_prob_local, next_bs = score_reward_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts_subset,
                    batch_size=prev_bs,
                    max_seq_len=MAX_SEQ_LEN,
                )
            else:
                q_prob_local, next_bs = score_yes_probability(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts_subset,
                    batch_size=prev_bs,
                    max_seq_len=MAX_SEQ_LEN,
                    yes_id=yes_id,
                    no_id=no_id,
                )
            adaptive_batch_size_by_model[active_model_id] = int(next_bs)
            return q_prob_local.astype(np.float32), int(prev_bs), int(next_bs)

        def _flush_prompt_buf() -> None:
            if not prompt_buf:
                return
            q_prob_local = np.full(len(prompt_buf), np.nan, dtype=np.float32)
            if not DUAL_BAND_ROUTE_ENABLED:
                q_prob_local, prev_bs, next_bs = _score_prompt_subset("global", prompt_buf)
                if int(next_bs) != int(prev_bs):
                    print(
                        f"[CONFIG] adaptive_eval_batch_size update: "
                        f"{int(prev_bs)} -> {int(next_bs)}"
                    )
            else:
                active_model_ids = ["band_11_30", "band_31_60"]
                if adapter_dir_tertiary is not None:
                    active_model_ids.append("band_61_100")
                for active_model_id in active_model_ids:
                    idxs = [idx for idx, mid in enumerate(sidecar_model_id_buf) if mid == active_model_id]
                    if not idxs:
                        continue
                    prompt_subset = [prompt_buf[idx] for idx in idxs]
                    subset_scores, prev_bs, next_bs = _score_prompt_subset(active_model_id, prompt_subset)
                    if int(next_bs) != int(prev_bs):
                        print(
                            f"[CONFIG] adaptive_eval_batch_size[{active_model_id}] update: "
                            f"{int(prev_bs)} -> {int(next_bs)}"
                        )
                    q_prob_local[np.asarray(idxs, dtype=np.int32)] = subset_scores
            if INVERT_PROB and EVAL_MODEL_TYPE != "rm":
                q_prob_local = np.clip(1.0 - q_prob_local, 0.0, 1.0)
            if len(q_prob_local) != len(user_idx_buf):
                raise RuntimeError(
                    "qlora score size mismatch: "
                    f"probs={len(q_prob_local)} rows={len(user_idx_buf)}"
                )
            for idx, score_value in enumerate(q_prob_local.tolist()):
                _append_pending_row(
                    {
                        "user_idx": user_idx_buf[idx],
                        "item_idx": item_idx_buf[idx],
                        "pre_rank": pre_rank_buf[idx],
                        "label_true": label_true_buf[idx],
                        "pre_score": pre_score_buf[idx],
                        "learned_rank": learned_rank_buf[idx],
                        "learned_blend_score": learned_blend_score_buf[idx],
                    },
                    float(score_value),
                    route_band_buf[idx],
                    sidecar_model_id_buf[idx],
                    band_alpha_buf[idx],
                )
            prompt_buf.clear()
            user_idx_buf.clear()
            item_idx_buf.clear()
            pre_rank_buf.clear()
            label_true_buf.clear()
            pre_score_buf.clear()
            learned_rank_buf.clear()
            learned_blend_score_buf.clear()
            route_band_buf.clear()
            sidecar_model_id_buf.clear()
            band_alpha_buf.clear()

        def _score_prompt_pdf_direct(prompt_pdf: pd.DataFrame) -> np.ndarray:
            if prompt_pdf.empty:
                return np.zeros(0, dtype=np.float32)
            prompts_local = prompt_pdf["prompt"].fillna("").astype(str).tolist()
            model_ids = (
                prompt_pdf.get("sidecar_model_id", pd.Series(["global"] * len(prompt_pdf)))
                .fillna("")
                .astype(str)
                .tolist()
            )
            q_prob_local = np.full(len(prompt_pdf), np.nan, dtype=np.float32)
            if not DUAL_BAND_ROUTE_ENABLED:
                subset_scores, prev_bs, next_bs = _score_prompt_subset("global", prompts_local)
                if int(next_bs) != int(prev_bs):
                    print(
                        f"[CONFIG] adaptive_eval_batch_size update: "
                        f"{int(prev_bs)} -> {int(next_bs)}"
                    )
                return subset_scores.astype(np.float32)

            active_model_ids = ["band_11_30", "band_31_60"]
            if adapter_dir_tertiary is not None:
                active_model_ids.append("band_61_100")
            for active_model_id in active_model_ids:
                idxs = [idx for idx, mid in enumerate(model_ids) if mid == active_model_id]
                if not idxs:
                    continue
                prompt_subset = [prompts_local[idx] for idx in idxs]
                subset_scores, prev_bs, next_bs = _score_prompt_subset(active_model_id, prompt_subset)
                if int(next_bs) != int(prev_bs):
                    print(
                        f"[CONFIG] adaptive_eval_batch_size[{active_model_id}] update: "
                        f"{int(prev_bs)} -> {int(next_bs)}"
                    )
                q_prob_local[np.asarray(idxs, dtype=np.int32)] = subset_scores.astype(np.float32)
            return q_prob_local

        def _consume_prompt_row(row: Any) -> None:
            route_rank = _eval_route_rank_from_row(row, 999999)
            route_band, sidecar_model_id, band_alpha = _sidecar_route_from_rank(route_rank)
            if DUAL_BAND_ROUTE_ENABLED and not sidecar_model_id:
                _append_pending_row(
                    row,
                    float("nan"),
                    route_band,
                    sidecar_model_id,
                    band_alpha,
                )
                return
            prompt_buf.append(_str_or_empty(row["prompt"]))
            user_idx_buf.append(int(row["user_idx"]))
            item_idx_buf.append(int(row["item_idx"]))
            pre_rank_buf.append(int(row["pre_rank"]))
            label_true_buf.append(int(row["label_true"]))
            pre_score_buf.append(_float_or_zero(row["pre_score"]))
            learned_rank_buf.append(int(_safe_rank_int(row.get("learned_rank", 999999), 999999)))
            learned_blend_score_buf.append(_float_or_zero(row.get("learned_blend_score", 0.0)))
            route_band_buf.append(route_band)
            sidecar_model_id_buf.append(sidecar_model_id)
            band_alpha_buf.append(float(band_alpha))

        t_spark_start = time.monotonic()
        # ------------------------------------------------------------------
        # Candidate prep before driver-side scoring.
        # Cloud runs should prefer Spark-side prompt construction and stream
        # prompt-ready rows back to the driver, rather than full collect().
        # ------------------------------------------------------------------
        _collect_cols = [
            "user_idx", "item_idx", "pre_rank", "label_true", "pre_score", "learned_rank", "learned_blend_score",
            "profile_text", "profile_top_pos_tags", "profile_top_neg_tags",
            "profile_confidence", "user_evidence_text", "pair_evidence_summary",
            "history_anchor_text",
            "name", "city", "categories", "primary_category",
            "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence",
            "semantic_support", "semantic_tag_richness", "tower_score", "seq_score",
            "schema_weighted_overlap_user_ratio_v2_rank_pct_v3",
            "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3",
            "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3",
            "schema_weighted_net_score_v2_rank_pct_v3",
            "sim_negative_avoid_neg",
            "sim_negative_avoid_core",
            "sim_conflict_gap",
            "channel_preference_core_v1",
            "channel_recent_intent_v1",
            "channel_context_time_v1",
            "channel_conflict_v1",
            "channel_evidence_support_v1",
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
        elif PROMPT_MODE in {"semantic_compact_preserve_rm", "semantic_compact_targeted_rm"}:
            _collect_cols.extend(
                [
                    "source_set_text",
                    "source_count",
                    "nonpopular_source_count",
                    "profile_cluster_source_count",
                    "context_rank",
                ]
            )
        elif PROMPT_MODE == LOCAL_LISTWISE_COMPARE_PROMPT_MODE:
            _collect_cols.extend(
                [
                    "source_set_text",
                    "source_count",
                    "nonpopular_source_count",
                    "profile_cluster_source_count",
                    "context_rank",
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

        effective_prompt_build_mode = PROMPT_BUILD_MODE
        if PROMPT_MODE == LOCAL_LISTWISE_COMPARE_PROMPT_MODE and effective_prompt_build_mode != "driver":
            print(
                "[PROMPT] local_listwise_compare_rm requires driver prompt build; "
                f"override {PROMPT_BUILD_MODE} -> driver"
            )
            effective_prompt_build_mode = "driver"

        if effective_prompt_build_mode == "spark":
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
            elif PROMPT_MODE == "semantic_rm":
                prompt_udf = build_prompt_udf_semantic_rm()
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
            elif PROMPT_MODE == "semantic_compact_rm":
                prompt_udf = build_prompt_udf_semantic_compact_rm()
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
                        F.col("schema_weighted_overlap_user_ratio_v2_rank_pct_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3"),
                        F.col("schema_weighted_net_score_v2_rank_pct_v3"),
                        F.col("sim_negative_avoid_neg"),
                        F.col("sim_negative_avoid_core"),
                        F.col("sim_conflict_gap"),
                        F.col("channel_preference_core_v1"),
                        F.col("channel_recent_intent_v1"),
                        F.col("channel_context_time_v1"),
                        F.col("channel_conflict_v1"),
                        F.col("channel_evidence_support_v1"),
                    ),
                )
            elif PROMPT_MODE == "semantic_compact_preserve_rm":
                prompt_udf = build_prompt_udf_semantic_compact_preserve_rm()
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
                        F.col("schema_weighted_overlap_user_ratio_v2_rank_pct_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3"),
                        F.col("schema_weighted_net_score_v2_rank_pct_v3"),
                        F.col("sim_negative_avoid_neg"),
                        F.col("sim_negative_avoid_core"),
                        F.col("sim_conflict_gap"),
                        F.col("channel_preference_core_v1"),
                        F.col("channel_recent_intent_v1"),
                        F.col("channel_context_time_v1"),
                        F.col("channel_conflict_v1"),
                        F.col("channel_evidence_support_v1"),
                        F.col("source_set_text"),
                        F.col("source_count"),
                        F.col("nonpopular_source_count"),
                        F.col("profile_cluster_source_count"),
                        F.col("context_rank"),
                    ),
                )
            elif PROMPT_MODE == "semantic_compact_targeted_rm":
                prompt_udf = build_prompt_udf_semantic_compact_targeted_rm()
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
                        F.col("pre_rank"),
                        F.col("schema_weighted_overlap_user_ratio_v2_rank_pct_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3"),
                        F.col("schema_weighted_net_score_v2_rank_pct_v3"),
                        F.col("sim_negative_avoid_neg"),
                        F.col("sim_negative_avoid_core"),
                        F.col("sim_conflict_gap"),
                        F.col("channel_preference_core_v1"),
                        F.col("channel_recent_intent_v1"),
                        F.col("channel_context_time_v1"),
                        F.col("channel_conflict_v1"),
                        F.col("channel_evidence_support_v1"),
                        F.col("source_set_text"),
                        F.col("source_count"),
                        F.col("nonpopular_source_count"),
                        F.col("profile_cluster_source_count"),
                        F.col("context_rank"),
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
                        F.col("pre_rank"),
                        F.col("pre_score"),
                        F.col("schema_weighted_overlap_user_ratio_v2_rank_pct_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3"),
                        F.col("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3"),
                        F.col("schema_weighted_net_score_v2_rank_pct_v3"),
                        F.col("sim_negative_avoid_neg"),
                        F.col("sim_negative_avoid_core"),
                        F.col("sim_conflict_gap"),
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

            if PROMPT_MODE == LOCAL_LISTWISE_COMPARE_PROMPT_MODE:
                print("[PROMPT] driver-side local listwise prompt generation ...")
                t_prompt_start = time.monotonic()
                routed_prompt_pdf = cand_pdf
                bypass_pdf = pd.DataFrame(columns=cand_pdf.columns)
                if DUAL_BAND_ROUTE_ENABLED and not cand_pdf.empty:
                    route_rank_col = "learned_rank" if str(EVAL_BASELINE_RANK_COL or "").strip() != "pre_rank" else "pre_rank"
                    route_rank_series = pd.to_numeric(
                        cand_pdf.get(route_rank_col, cand_pdf.get("pre_rank")),
                        errors="coerce",
                    ).fillna(pd.to_numeric(cand_pdf.get("pre_rank"), errors="coerce").fillna(999999))
                    # Keep pandas-side bypass routing aligned with the adapter routing
                    # logic so tri-band eval does not drop the 61-100 scorer path.
                    sidecar_model_series = route_rank_series.astype(np.int32).map(
                        lambda rank: _sidecar_route_from_rank(int(rank))[1]
                    )
                    score_mask = sidecar_model_series != ""
                    bypass_pdf = cand_pdf.loc[~score_mask].copy()
                    routed_prompt_pdf = cand_pdf.loc[score_mask].copy().reset_index(drop=True)
                    for idx in range(len(bypass_pdf)):
                        _consume_prompt_row(bypass_pdf.iloc[idx])
                    print(
                        f"[PROMPT] dual-band route skip: score_rows={len(routed_prompt_pdf)} "
                        f"baseline_only_rows={len(bypass_pdf)}"
                    )
                if not routed_prompt_pdf.empty:
                    routed_prompt_pdf["prompt"] = build_driver_local_listwise_prompts(routed_prompt_pdf)
                t_prompt = time.monotonic() - t_prompt_start
                print(
                    f"[TIMING] driver-side local listwise prompt generation done: "
                    f"{len(routed_prompt_pdf)} scored rows in {t_prompt:.1f}s"
                )
                iter_row_count = 0
                for idx in range(len(routed_prompt_pdf)):
                    row = routed_prompt_pdf.iloc[idx]
                    _consume_prompt_row(row)
                    iter_row_count += 1
                    if len(prompt_buf) >= int(PROMPT_CHUNK_ROWS):
                        print(
                            f"[TIMING] prompt chunk ready: {iter_row_count}/{len(routed_prompt_pdf)} scored rows"
                        )
                        _flush_prompt_buf()
                t_spark_total = time.monotonic() - t_spark_start
                total_rows_processed = int(iter_row_count + len(bypass_pdf))
                print(f"[TIMING] Spark iter done: {total_rows_processed} total rows in {t_spark_total:.1f}s")
                _flush_prompt_buf()
                _append_partial_scores()
                if int(flushed_rows) <= 0 or not partial_scores_path.exists():
                    print(f"[WARN] skip bucket={bucket}: empty candidate after filters")
                    continue
                pdf = pd.read_csv(partial_scores_path.as_posix())
                if pdf.empty:
                    print(f"[WARN] skip bucket={bucket}: empty partial scores")
                    continue
                pdf, base_score_col, base_rank_col = apply_sidecar_fusion(pdf, sidecar_raw_col)
                pdf["blend_rank"] = (
                    pdf.groupby("user_idx", sort=False)["blend_score"]
                    .rank(method="first", ascending=False)
                    .astype(np.int32)
                )
                pre_recall, pre_ndcg = evaluate_topk(pdf, "baseline_score", TOP_K)
                q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)
                final_score_col = "blend_score"
                joint_recall = float("nan")
                joint_ndcg = float("nan")
                # Stage11 second-stage rerank is intentionally bounded:
                # first shortlist routed candidates, then apply joint scoring,
                # then gate/cap replacements so deeper bands do not directly
                # take over the full top-k frontier.
                if EVAL_SECOND_STAGE_ENABLE and PROMPT_MODE == LOCAL_LISTWISE_COMPARE_PROMPT_MODE and not cand_pdf.empty:
                    t_joint_start = time.monotonic()
                    shortlist_keys = build_second_stage_shortlist(pdf)
                    if not shortlist_keys.empty:
                        shortlist_ctx = cand_pdf.merge(shortlist_keys, on=["user_idx", "item_idx"], how="inner")
                        if not shortlist_ctx.empty:
                            shortlist_ctx = shortlist_ctx.merge(
                                pdf[
                                    [
                                        "user_idx",
                                        "item_idx",
                                        "blend_score",
                                        "blend_rank",
                                        "route_band",
                                        "sidecar_model_id",
                                        "band_alpha",
                                    ]
                                ],
                                on=["user_idx", "item_idx"],
                                how="left",
                            )
                            shortlist_ctx["pre_rank"] = pd.to_numeric(
                                shortlist_ctx.get("blend_rank"), errors="coerce"
                            ).fillna(pd.to_numeric(shortlist_ctx.get("pre_rank"), errors="coerce")).astype(np.int32)
                            shortlist_ctx["learned_rank"] = shortlist_ctx["pre_rank"].astype(np.int32)
                            shortlist_ctx["prompt"] = build_driver_local_listwise_prompts(
                                shortlist_ctx,
                                max_rivals=min(
                                    int(SECOND_STAGE_LOCAL_LISTWISE_MAX_RIVALS),
                                    max(2, int(SECOND_STAGE_SHORTLIST_SIZE) - 1),
                                ),
                            )
                            joint_score = _score_prompt_pdf_direct(shortlist_ctx)
                            shortlist_score_pdf = shortlist_ctx[
                                ["user_idx", "item_idx", "route_band", "sidecar_model_id"]
                            ].copy()
                            shortlist_score_pdf["joint_reward_score"] = joint_score.astype(np.float32)
                            routed_joint_mask = shortlist_score_pdf["sidecar_model_id"].fillna("").astype(str) != ""
                            shortlist_score_pdf["joint_norm"] = np.float32(0.0)
                            if routed_joint_mask.any():
                                joint_group_cols: str | list[str] = "user_idx"
                                if SECOND_STAGE_ROUTE_LOCAL_NORM:
                                    joint_group_cols = ["user_idx", "route_band"]
                                shortlist_score_pdf.loc[routed_joint_mask, "joint_norm"] = normalize_group_score(
                                    shortlist_score_pdf.loc[routed_joint_mask].copy(),
                                    "joint_reward_score",
                                    "rank_pct",
                                    group_cols=joint_group_cols,
                                ).astype(np.float32)
                            shortlist_score_pdf["joint_bonus"] = np.float32(0.0)
                            shortlist_score_pdf.loc[routed_joint_mask, "joint_bonus"] = (
                                float(SECOND_STAGE_BLEND_ALPHA)
                                * (shortlist_score_pdf.loc[routed_joint_mask, "joint_norm"].astype(np.float32) - np.float32(0.5))
                            ).astype(np.float32)
                            pdf = pdf.merge(
                                shortlist_score_pdf[
                                    ["user_idx", "item_idx", "joint_reward_score", "joint_norm", "joint_bonus"]
                                ],
                                on=["user_idx", "item_idx"],
                                how="left",
                            )
                            pdf["joint_reward_score"] = pd.to_numeric(
                                pdf.get("joint_reward_score"), errors="coerce"
                            ).astype(np.float32)
                            pdf["joint_norm"] = pd.to_numeric(
                                pdf.get("joint_norm"), errors="coerce"
                            ).fillna(0.0).astype(np.float32)
                            pdf["joint_bonus"] = pd.to_numeric(
                                pdf.get("joint_bonus"), errors="coerce"
                            ).fillna(0.0).astype(np.float32)
                            pdf["joint_gate_pass"] = True
                            pdf["joint_bonus_effective"] = pdf["joint_bonus"].astype(np.float32)
                            if SECOND_STAGE_GATE_ENABLE:
                                # The gate compares each routed candidate only
                                # against the frontier rank it is allowed to
                                # challenge (top10/top20/top30), then caps how
                                # far it may jump to keep the final rerank
                                # conservative and production-safe.
                                pdf = _attach_frontier_score(
                                    pdf,
                                    "blend_score",
                                    "blend_rank",
                                    int(SECOND_STAGE_GATE_TARGET_RANK_31_40),
                                    "gate_frontier_score_31_40",
                                )
                                pdf = _attach_frontier_score(
                                    pdf,
                                    "blend_score",
                                    "blend_rank",
                                    int(SECOND_STAGE_GATE_TARGET_RANK_41_60),
                                    "gate_frontier_score_41_60",
                                )
                                pdf = _attach_frontier_score(
                                    pdf,
                                    "blend_score",
                                    "blend_rank",
                                    int(SECOND_STAGE_GATE_TARGET_RANK_61_100),
                                    "gate_frontier_score_61_100",
                                )
                                pdf = _attach_frontier_score(
                                    pdf,
                                    "blend_score",
                                    "blend_rank",
                                    int(SECOND_STAGE_GATE_CAP_RANK_31_40),
                                    "gate_cap_score_31_40",
                                )
                                pdf = _attach_frontier_score(
                                    pdf,
                                    "blend_score",
                                    "blend_rank",
                                    int(SECOND_STAGE_GATE_CAP_RANK_41_60),
                                    "gate_cap_score_41_60",
                                )
                                pdf = _attach_frontier_score(
                                    pdf,
                                    "blend_score",
                                    "blend_rank",
                                    int(SECOND_STAGE_GATE_CAP_RANK_61_100),
                                    "gate_cap_score_61_100",
                                )
                                gate_31_40 = pdf["route_band"].fillna("").astype(str) == "rescue_31_40"
                                gate_41_60 = pdf["route_band"].fillna("").astype(str) == "rescue_41_60"
                                gate_61_100 = pdf["route_band"].fillna("").astype(str) == "rescue_61_100"
                                pass_31_40 = (
                                    gate_31_40
                                    & (pdf["blend_rank"].astype(np.int32) <= int(SECOND_STAGE_GATE_MAX_BLEND_RANK_31_40))
                                    & (
                                        (pdf["blend_score"].astype(np.float32) + pdf["joint_bonus"].astype(np.float32))
                                        >= (
                                            pdf["gate_frontier_score_31_40"].astype(np.float32)
                                            + np.float32(SECOND_STAGE_GATE_MIN_MARGIN_31_40)
                                        )
                                    )
                                )
                                pass_41_60 = (
                                    gate_41_60
                                    & (pdf["blend_rank"].astype(np.int32) <= int(SECOND_STAGE_GATE_MAX_BLEND_RANK_41_60))
                                    & (
                                        (pdf["blend_score"].astype(np.float32) + pdf["joint_bonus"].astype(np.float32))
                                        >= (
                                            pdf["gate_frontier_score_41_60"].astype(np.float32)
                                            + np.float32(SECOND_STAGE_GATE_MIN_MARGIN_41_60)
                                        )
                                    )
                                )
                                pass_61_100 = (
                                    gate_61_100
                                    & (pdf["blend_rank"].astype(np.int32) <= int(SECOND_STAGE_GATE_MAX_BLEND_RANK_61_100))
                                    & (
                                        (pdf["blend_score"].astype(np.float32) + pdf["joint_bonus"].astype(np.float32))
                                        >= (
                                            pdf["gate_frontier_score_61_100"].astype(np.float32)
                                            + np.float32(SECOND_STAGE_GATE_MIN_MARGIN_61_100)
                                        )
                                    )
                                )
                                rescue_mask = gate_31_40 | gate_41_60 | gate_61_100
                                rescue_pass = pass_31_40 | pass_41_60 | pass_61_100
                                pdf.loc[rescue_mask, "joint_gate_pass"] = rescue_pass.loc[rescue_mask].astype(bool)
                                pdf.loc[rescue_mask & (~rescue_pass), "joint_bonus_effective"] = np.float32(0.0)
                                if rescue_pass.any():
                                    eff_bonus = pdf["joint_bonus_effective"].astype(np.float32).copy()
                                    cap_eps = np.float32(SECOND_STAGE_GATE_CAP_EPSILON)
                                    cap_31_40 = (
                                        pdf["gate_cap_score_31_40"].astype(np.float32)
                                        - pdf["blend_score"].astype(np.float32)
                                        - cap_eps
                                    )
                                    cap_41_60 = (
                                        pdf["gate_cap_score_41_60"].astype(np.float32)
                                        - pdf["blend_score"].astype(np.float32)
                                        - cap_eps
                                    )
                                    cap_61_100 = (
                                        pdf["gate_cap_score_61_100"].astype(np.float32)
                                        - pdf["blend_score"].astype(np.float32)
                                        - cap_eps
                                    )
                                    cap_31_40 = np.where(np.isfinite(cap_31_40), cap_31_40, eff_bonus).astype(np.float32)
                                    cap_41_60 = np.where(np.isfinite(cap_41_60), cap_41_60, eff_bonus).astype(np.float32)
                                    cap_61_100 = np.where(np.isfinite(cap_61_100), cap_61_100, eff_bonus).astype(np.float32)
                                    cap_31_40 = np.maximum(cap_31_40, np.float32(0.0)).astype(np.float32)
                                    cap_41_60 = np.maximum(cap_41_60, np.float32(0.0)).astype(np.float32)
                                    cap_61_100 = np.maximum(cap_61_100, np.float32(0.0)).astype(np.float32)
                                    eff_bonus.loc[gate_31_40] = np.minimum(
                                        eff_bonus.loc[gate_31_40].astype(np.float32),
                                        cap_31_40[gate_31_40.to_numpy()],
                                    ).astype(np.float32)
                                    eff_bonus.loc[gate_41_60] = np.minimum(
                                        eff_bonus.loc[gate_41_60].astype(np.float32),
                                        cap_41_60[gate_41_60.to_numpy()],
                                    ).astype(np.float32)
                                    eff_bonus.loc[gate_61_100] = np.minimum(
                                        eff_bonus.loc[gate_61_100].astype(np.float32),
                                        cap_61_100[gate_61_100.to_numpy()],
                                    ).astype(np.float32)
                                    pdf.loc[rescue_pass, "joint_bonus_effective"] = eff_bonus.loc[rescue_pass].astype(np.float32)
                            pdf["final_score"] = (
                                pdf["blend_score"].astype(np.float32) + pdf["joint_bonus_effective"].astype(np.float32)
                            ).astype(np.float32)
                            final_score_col = "final_score"
                            joint_recall, joint_ndcg = evaluate_topk(pdf, final_score_col, TOP_K)
                            t_joint = time.monotonic() - t_joint_start
                            print(
                                f"[JOINT] shortlist_rows={len(shortlist_ctx)} "
                                f"shortlist_users={int(shortlist_ctx['user_idx'].nunique())} "
                                f"alpha={float(SECOND_STAGE_BLEND_ALPHA):.3f} "
                                f"recall={joint_recall:.6f} ndcg={joint_ndcg:.6f} elapsed={t_joint:.1f}s"
                            )
                    if "final_score" not in pdf.columns:
                        pdf["joint_reward_score"] = np.float32(np.nan)
                        pdf["joint_norm"] = np.float32(0.0)
                        pdf["joint_bonus"] = np.float32(0.0)
                        pdf["joint_bonus_effective"] = np.float32(0.0)
                        pdf["joint_gate_pass"] = True
                        pdf["gate_frontier_score_31_40"] = np.float32(np.nan)
                        pdf["gate_frontier_score_41_60"] = np.float32(np.nan)
                        pdf["gate_frontier_score_61_100"] = np.float32(np.nan)
                        pdf["gate_cap_score_31_40"] = np.float32(np.nan)
                        pdf["gate_cap_score_41_60"] = np.float32(np.nan)
                        pdf["gate_cap_score_61_100"] = np.float32(np.nan)
                        pdf["final_score"] = pdf["blend_score"].astype(np.float32)

                n_users = int(pdf["user_idx"].nunique())
                n_items = int(pdf["item_idx"].nunique())
                n_rows = int(len(pdf))
                if eval_user_cohort_path is not None and int(eval_user_count_surviving) <= 0:
                    eval_user_count_surviving = int(n_users)
                    eval_user_count_dropped = int(max(0, eval_user_count_requested - eval_user_count_surviving))

                metrics_rows.append(
                    {
                        "run_id_11": run_id,
                        "source_run_09": str(stage09_run),
                        "source_run_11_2": str(stage11_2_run),
                        "bucket_min_train_reviews": int(bucket),
                        "model": baseline_model_label(base_score_col),
                        "recall_at_k": float(pre_recall),
                        "ndcg_at_k": float(pre_ndcg),
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_candidates": n_rows,
                        "eval_user_count_requested": int(eval_user_count_requested),
                        "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                        "eval_user_count_dropped": int(eval_user_count_dropped),
                        "baseline_score_col": str(base_score_col),
                        "baseline_rank_col": str(base_rank_col),
                        "fusion_mode": str(EVAL_FUSION_MODE),
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
                        "eval_user_count_requested": int(eval_user_count_requested),
                        "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                        "eval_user_count_dropped": int(eval_user_count_dropped),
                        "baseline_score_col": str(base_score_col),
                        "baseline_rank_col": str(base_rank_col),
                        "fusion_mode": str(EVAL_FUSION_MODE),
                    }
                )
                if EVAL_SECOND_STAGE_ENABLE and np.isfinite(joint_recall) and np.isfinite(joint_ndcg):
                    metrics_rows.append(
                        {
                            "run_id_11": run_id,
                            "source_run_09": str(stage09_run),
                            "source_run_11_2": str(stage11_2_run),
                            "bucket_min_train_reviews": int(bucket),
                            "model": "QLoRASidecarJoint@10",
                            "recall_at_k": float(joint_recall),
                            "ndcg_at_k": float(joint_ndcg),
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_candidates": n_rows,
                            "eval_user_count_requested": int(eval_user_count_requested),
                            "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                            "eval_user_count_dropped": int(eval_user_count_dropped),
                            "baseline_score_col": str(base_score_col),
                            "baseline_rank_col": str(base_rank_col),
                            "fusion_mode": str(EVAL_FUSION_MODE),
                        }
                    )
                bucket_run_summaries.append(
                    {
                        "bucket": int(bucket),
                        "eval_user_count_requested": int(eval_user_count_requested),
                        "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                        "eval_user_count_dropped": int(eval_user_count_dropped),
                        "eval_user_join_cols": list(eval_user_join_cols),
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_candidates": n_rows,
                        "pre_recall_at_k": float(pre_recall),
                        "pre_ndcg_at_k": float(pre_ndcg),
                        "qlora_recall_at_k": float(q_recall),
                        "qlora_ndcg_at_k": float(q_ndcg),
                        "joint_recall_at_k": float(joint_recall) if np.isfinite(joint_recall) else None,
                        "joint_ndcg_at_k": float(joint_ndcg) if np.isfinite(joint_ndcg) else None,
                        "baseline_score_col": str(base_score_col),
                        "baseline_rank_col": str(base_rank_col),
                        "fusion_mode": str(EVAL_FUSION_MODE),
                    }
                )
                print(
                    f"[METRIC] bucket={bucket} baseline={baseline_model_label(base_score_col)} "
                    f"base_ndcg={pre_ndcg:.6f} qlora_ndcg={q_ndcg:.6f} "
                    f"base_recall={pre_recall:.6f} qlora_recall={q_recall:.6f}"
                )
                pd.DataFrame(
                    {
                        "user_idx": pdf["user_idx"],
                        "item_idx": pdf["item_idx"],
                        "pre_rank": pdf["pre_rank"],
                        "label_true": pdf["label_true"],
                        "pre_score": pdf["pre_score"],
                        "baseline_score": pdf["baseline_score"],
                        "baseline_rank": pdf["baseline_rank"],
                        "base_norm": pdf["base_norm"],
                        "blend_rank": pdf.get("blend_rank", pd.Series(np.zeros(len(pdf), dtype=np.int32))),
                        "learned_rank": pdf.get("learned_rank", pd.Series(np.zeros(len(pdf), dtype=np.int32))),
                        "learned_blend_score": pdf.get("learned_blend_score", pd.Series(np.zeros(len(pdf), dtype=np.float32))),
                        sidecar_raw_col: pdf[sidecar_raw_col],
                        "sidecar_norm": pdf["sidecar_norm"],
                        "sidecar_score_present": pdf.get("sidecar_score_present", pd.Series(np.zeros(len(pdf), dtype=np.int8))),
                        "route_band": pdf.get("route_band", pd.Series([""] * len(pdf))),
                        "sidecar_model_id": pdf.get("sidecar_model_id", pd.Series([""] * len(pdf))),
                        "band_alpha": pdf.get("band_alpha", pd.Series(np.full(len(pdf), float(BLEND_ALPHA), dtype=np.float32))),
                        "rescue_band_weight": pdf["rescue_band_weight"],
                        "rescue_bonus": pdf["rescue_bonus"],
                        "blend_score": pdf["blend_score"],
                            "joint_reward_score": pdf.get("joint_reward_score", pd.Series(np.full(len(pdf), np.nan, dtype=np.float32))),
                            "joint_norm": pdf.get("joint_norm", pd.Series(np.zeros(len(pdf), dtype=np.float32))),
                            "joint_bonus": pdf.get("joint_bonus", pd.Series(np.zeros(len(pdf), dtype=np.float32))),
                            "joint_bonus_effective": pdf.get(
                                "joint_bonus_effective",
                                pd.Series(np.zeros(len(pdf), dtype=np.float32)),
                            ),
                            "joint_gate_pass": pdf.get(
                                "joint_gate_pass",
                                pd.Series(np.ones(len(pdf), dtype=bool)),
                            ),
                            "gate_frontier_score_31_40": pdf.get(
                                "gate_frontier_score_31_40",
                                pd.Series(np.full(len(pdf), np.nan, dtype=np.float32)),
                            ),
                            "gate_frontier_score_41_60": pdf.get(
                                "gate_frontier_score_41_60",
                                pd.Series(np.full(len(pdf), np.nan, dtype=np.float32)),
                            ),
                            "gate_frontier_score_61_100": pdf.get(
                                "gate_frontier_score_61_100",
                                pd.Series(np.full(len(pdf), np.nan, dtype=np.float32)),
                            ),
                            "gate_cap_score_31_40": pdf.get(
                                "gate_cap_score_31_40",
                                pd.Series(np.full(len(pdf), np.nan, dtype=np.float32)),
                            ),
                            "gate_cap_score_41_60": pdf.get(
                                "gate_cap_score_41_60",
                                pd.Series(np.full(len(pdf), np.nan, dtype=np.float32)),
                            ),
                            "gate_cap_score_61_100": pdf.get(
                                "gate_cap_score_61_100",
                                pd.Series(np.full(len(pdf), np.nan, dtype=np.float32)),
                            ),
                            "final_score": pdf.get("final_score", pdf["blend_score"]),
                        }
                    ).to_csv(final_scores_path.as_posix(), index=False)
                print(f"[CHECKPOINT] bucket={bucket} flushed_rows={int(flushed_rows)} partial_file={partial_scores_path}")
                continue

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
                elif PROMPT_MODE == "semantic_rm":
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
                    return build_scoring_prompt(user_text, item_text)
                elif PROMPT_MODE == "semantic_compact_rm":
                    item_text = build_item_text_semantic_compact(
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
                        group_gap_rank_pct=r.get("schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                        group_gap_to_top3=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                        group_gap_to_top10=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                        net_score_rank_pct=r.get("schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                        avoid_neg=r.get("sim_negative_avoid_neg", 0.0),
                        avoid_core=r.get("sim_negative_avoid_core", 0.0),
                        conflict_gap=r.get("sim_conflict_gap", 0.0),
                        channel_preference_core_v1=r.get("channel_preference_core_v1", 0.0),
                        channel_recent_intent_v1=r.get("channel_recent_intent_v1", 0.0),
                        channel_context_time_v1=r.get("channel_context_time_v1", 0.0),
                        channel_conflict_v1=r.get("channel_conflict_v1", 0.0),
                        channel_evidence_support_v1=r.get("channel_evidence_support_v1", 0.0),
                    )
                    return build_scoring_prompt(user_text, item_text)
                elif PROMPT_MODE == "semantic_compact_preserve_rm":
                    item_text = build_item_text_semantic_compact_preserve(
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
                        group_gap_rank_pct=r.get("schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                        group_gap_to_top3=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                        group_gap_to_top10=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                        net_score_rank_pct=r.get("schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                        avoid_neg=r.get("sim_negative_avoid_neg", 0.0),
                        avoid_core=r.get("sim_negative_avoid_core", 0.0),
                        conflict_gap=r.get("sim_conflict_gap", 0.0),
                        channel_preference_core_v1=r.get("channel_preference_core_v1", 0.0),
                        channel_recent_intent_v1=r.get("channel_recent_intent_v1", 0.0),
                        channel_context_time_v1=r.get("channel_context_time_v1", 0.0),
                        channel_conflict_v1=r.get("channel_conflict_v1", 0.0),
                        channel_evidence_support_v1=r.get("channel_evidence_support_v1", 0.0),
                        source_set=r.get("source_set_text", ""),
                        source_count=r.get("source_count", 0.0),
                        nonpopular_source_count=r.get("nonpopular_source_count", 0.0),
                        profile_cluster_source_count=r.get("profile_cluster_source_count", 0.0),
                        context_rank=r.get("context_rank", 0.0),
                    )
                    return build_scoring_prompt(user_text, item_text)
                elif PROMPT_MODE == "semantic_compact_targeted_rm":
                    item_text = build_item_text_semantic_compact_targeted(
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
                        pre_rank=r.get("pre_rank", 0.0),
                        group_gap_rank_pct=r.get("schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                        group_gap_to_top3=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                        group_gap_to_top10=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                        net_score_rank_pct=r.get("schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                        avoid_neg=r.get("sim_negative_avoid_neg", 0.0),
                        avoid_core=r.get("sim_negative_avoid_core", 0.0),
                        conflict_gap=r.get("sim_conflict_gap", 0.0),
                        channel_preference_core_v1=r.get("channel_preference_core_v1", 0.0),
                        channel_recent_intent_v1=r.get("channel_recent_intent_v1", 0.0),
                        channel_context_time_v1=r.get("channel_context_time_v1", 0.0),
                        channel_conflict_v1=r.get("channel_conflict_v1", 0.0),
                        channel_evidence_support_v1=r.get("channel_evidence_support_v1", 0.0),
                        source_set=r.get("source_set_text", ""),
                        source_count=r.get("source_count", 0.0),
                        nonpopular_source_count=r.get("nonpopular_source_count", 0.0),
                        profile_cluster_source_count=r.get("profile_cluster_source_count", 0.0),
                        context_rank=r.get("context_rank", 0.0),
                    )
                    return build_scoring_prompt(user_text, item_text)
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
                        pre_rank=r.get("pre_rank", 0.0),
                        pre_score=r.get("pre_score", 0.0),
                        group_gap_rank_pct=r.get("schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                        group_gap_to_top3=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                        group_gap_to_top10=r.get("schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                        net_score_rank_pct=r.get("schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                        avoid_neg=r.get("sim_negative_avoid_neg", 0.0),
                        avoid_core=r.get("sim_negative_avoid_core", 0.0),
                        conflict_gap=r.get("sim_conflict_gap", 0.0),
                        source_set=r.get("source_set_text", ""),
                        user_segment=r.get("user_segment", ""),
                        semantic_support=r.get("semantic_support", 0.0),
                        semantic_tag_richness=r.get("semantic_tag_richness", 0.0),
                        tower_score=r.get("tower_score", 0.0),
                        seq_score=r.get("seq_score", 0.0),
                        cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                        item_review_snippet=r.get("item_evidence_text", ""),
                        pair_evidence_summary=r.get("pair_evidence_summary", ""),
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
                    if PROMPT_MODE == "semantic_rm":
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
                        return build_scoring_prompt(user_text, item_text)
                    if PROMPT_MODE == "semantic_compact_rm":
                        item_text = build_item_text_semantic_compact(
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
                            group_gap_rank_pct=getattr(r, "schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                            group_gap_to_top3=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                            group_gap_to_top10=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                            net_score_rank_pct=getattr(r, "schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                            avoid_neg=getattr(r, "sim_negative_avoid_neg", 0.0),
                            avoid_core=getattr(r, "sim_negative_avoid_core", 0.0),
                            conflict_gap=getattr(r, "sim_conflict_gap", 0.0),
                            channel_preference_core_v1=getattr(r, "channel_preference_core_v1", 0.0),
                            channel_recent_intent_v1=getattr(r, "channel_recent_intent_v1", 0.0),
                            channel_context_time_v1=getattr(r, "channel_context_time_v1", 0.0),
                            channel_conflict_v1=getattr(r, "channel_conflict_v1", 0.0),
                            channel_evidence_support_v1=getattr(r, "channel_evidence_support_v1", 0.0),
                        )
                        return build_scoring_prompt(user_text, item_text)
                    if PROMPT_MODE == "semantic_compact_preserve_rm":
                        item_text = build_item_text_semantic_compact_preserve(
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
                            group_gap_rank_pct=getattr(r, "schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                            group_gap_to_top3=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                            group_gap_to_top10=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                            net_score_rank_pct=getattr(r, "schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                            avoid_neg=getattr(r, "sim_negative_avoid_neg", 0.0),
                            avoid_core=getattr(r, "sim_negative_avoid_core", 0.0),
                            conflict_gap=getattr(r, "sim_conflict_gap", 0.0),
                            channel_preference_core_v1=getattr(r, "channel_preference_core_v1", 0.0),
                            channel_recent_intent_v1=getattr(r, "channel_recent_intent_v1", 0.0),
                            channel_context_time_v1=getattr(r, "channel_context_time_v1", 0.0),
                            channel_conflict_v1=getattr(r, "channel_conflict_v1", 0.0),
                            channel_evidence_support_v1=getattr(r, "channel_evidence_support_v1", 0.0),
                            source_set=getattr(r, "source_set_text", ""),
                            source_count=getattr(r, "source_count", 0.0),
                            nonpopular_source_count=getattr(r, "nonpopular_source_count", 0.0),
                            profile_cluster_source_count=getattr(r, "profile_cluster_source_count", 0.0),
                            context_rank=getattr(r, "context_rank", 0.0),
                        )
                        return build_scoring_prompt(user_text, item_text)
                    if PROMPT_MODE == "semantic_compact_targeted_rm":
                        item_text = build_item_text_semantic_compact_targeted(
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
                            pre_rank=getattr(r, "pre_rank", 0.0),
                            group_gap_rank_pct=getattr(r, "schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                            group_gap_to_top3=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                            group_gap_to_top10=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                            net_score_rank_pct=getattr(r, "schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                            avoid_neg=getattr(r, "sim_negative_avoid_neg", 0.0),
                            avoid_core=getattr(r, "sim_negative_avoid_core", 0.0),
                            conflict_gap=getattr(r, "sim_conflict_gap", 0.0),
                            channel_preference_core_v1=getattr(r, "channel_preference_core_v1", 0.0),
                            channel_recent_intent_v1=getattr(r, "channel_recent_intent_v1", 0.0),
                            channel_context_time_v1=getattr(r, "channel_context_time_v1", 0.0),
                            channel_conflict_v1=getattr(r, "channel_conflict_v1", 0.0),
                            channel_evidence_support_v1=getattr(r, "channel_evidence_support_v1", 0.0),
                            source_set=getattr(r, "source_set_text", ""),
                            source_count=getattr(r, "source_count", 0.0),
                            nonpopular_source_count=getattr(r, "nonpopular_source_count", 0.0),
                            profile_cluster_source_count=getattr(r, "profile_cluster_source_count", 0.0),
                            context_rank=getattr(r, "context_rank", 0.0),
                        )
                        return build_scoring_prompt(user_text, item_text)
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
                            pre_rank=getattr(r, "pre_rank", 0.0),
                            pre_score=getattr(r, "pre_score", 0.0),
                            group_gap_rank_pct=getattr(r, "schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0),
                            group_gap_to_top3=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3", 0.0),
                            group_gap_to_top10=getattr(r, "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3", 0.0),
                            net_score_rank_pct=getattr(r, "schema_weighted_net_score_v2_rank_pct_v3", 0.0),
                            avoid_neg=getattr(r, "sim_negative_avoid_neg", 0.0),
                            avoid_core=getattr(r, "sim_negative_avoid_core", 0.0),
                            conflict_gap=getattr(r, "sim_conflict_gap", 0.0),
                            source_set=getattr(r, "source_set_text", ""),
                            user_segment=getattr(r, "user_segment", ""),
                            semantic_support=getattr(r, "semantic_support", 0.0),
                            semantic_tag_richness=getattr(r, "semantic_tag_richness", 0.0),
                            tower_score=getattr(r, "tower_score", 0.0),
                            seq_score=getattr(r, "seq_score", 0.0),
                            cluster_label_for_recsys=getattr(r, "cluster_label_for_recsys", ""),
                            item_review_snippet=getattr(r, "item_evidence_text", ""),
                            pair_evidence_summary=getattr(r, "pair_evidence_summary", ""),
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
        pdf, base_score_col, base_rank_col = apply_sidecar_fusion(pdf, sidecar_raw_col)
        pre_recall, pre_ndcg = evaluate_topk(pdf, "baseline_score", TOP_K)
        q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)

        n_users = int(pdf["user_idx"].nunique())
        n_items = int(pdf["item_idx"].nunique())
        n_rows = int(len(pdf))
        if eval_user_cohort_path is not None and int(eval_user_count_surviving) <= 0:
            eval_user_count_surviving = int(n_users)
            eval_user_count_dropped = int(max(0, eval_user_count_requested - eval_user_count_surviving))

        metrics_rows.append(
            {
                "run_id_11": run_id,
                "source_run_09": str(stage09_run),
                "source_run_11_2": str(stage11_2_run),
                "bucket_min_train_reviews": int(bucket),
                "model": baseline_model_label(base_score_col),
                "recall_at_k": float(pre_recall),
                "ndcg_at_k": float(pre_ndcg),
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_rows,
                "eval_user_count_requested": int(eval_user_count_requested),
                "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                "eval_user_count_dropped": int(eval_user_count_dropped),
                "baseline_score_col": str(base_score_col),
                "baseline_rank_col": str(base_rank_col),
                "fusion_mode": str(EVAL_FUSION_MODE),
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
                "eval_user_count_requested": int(eval_user_count_requested),
                "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                "eval_user_count_dropped": int(eval_user_count_dropped),
                "baseline_score_col": str(base_score_col),
                "baseline_rank_col": str(base_rank_col),
                "fusion_mode": str(EVAL_FUSION_MODE),
            }
        )
        bucket_run_summaries.append(
            {
                "bucket": int(bucket),
                "eval_user_count_requested": int(eval_user_count_requested),
                "eval_user_count_surviving": int(eval_user_count_surviving or n_users),
                "eval_user_count_dropped": int(eval_user_count_dropped),
                "eval_user_join_cols": list(eval_user_join_cols),
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_rows,
                "pre_recall_at_k": float(pre_recall),
                "pre_ndcg_at_k": float(pre_ndcg),
                "qlora_recall_at_k": float(q_recall),
                "qlora_ndcg_at_k": float(q_ndcg),
                "baseline_score_col": str(base_score_col),
                "baseline_rank_col": str(base_rank_col),
                "fusion_mode": str(EVAL_FUSION_MODE),
            }
        )
        print(
            f"[METRIC] bucket={bucket} baseline={baseline_model_label(base_score_col)} "
            f"base_ndcg={pre_ndcg:.6f} qlora_ndcg={q_ndcg:.6f} "
            f"base_recall={pre_recall:.6f} qlora_recall={q_recall:.6f}"
        )

        pd.DataFrame(
            {
                "user_idx": pdf["user_idx"],
                "item_idx": pdf["item_idx"],
                "pre_rank": pdf["pre_rank"],
                "label_true": pdf["label_true"],
                "pre_score": pdf["pre_score"],
                "baseline_score": pdf["baseline_score"],
                "baseline_rank": pdf["baseline_rank"],
                "base_norm": pdf["base_norm"],
                "learned_rank": pdf.get("learned_rank", pd.Series(np.zeros(len(pdf), dtype=np.int32))),
                "learned_blend_score": pdf.get("learned_blend_score", pd.Series(np.zeros(len(pdf), dtype=np.float32))),
                sidecar_raw_col: pdf[sidecar_raw_col],
                "sidecar_norm": pdf["sidecar_norm"],
                "sidecar_score_present": pdf.get("sidecar_score_present", pd.Series(np.zeros(len(pdf), dtype=np.int8))),
                "route_band": pdf.get("route_band", pd.Series([""] * len(pdf))),
                "sidecar_model_id": pdf.get("sidecar_model_id", pd.Series([""] * len(pdf))),
                "band_alpha": pdf.get("band_alpha", pd.Series(np.full(len(pdf), float(BLEND_ALPHA), dtype=np.float32))),
                "rescue_band_weight": pdf["rescue_band_weight"],
                "rescue_bonus": pdf["rescue_bonus"],
                "blend_score": pdf["blend_score"],
            }
        ).to_csv(final_scores_path.as_posix(), index=False)
        sync_csv_aliases(final_scores_path, final_scores_legacy_path)
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
        "source_run_11_2_11_30": str(stage11_2_run),
        "source_run_11_2_31_60": str(stage11_2_run_secondary) if stage11_2_run_secondary is not None else "",
        "source_run_11_2_61_100": str(stage11_2_run_tertiary) if stage11_2_run_tertiary is not None else "",
        "source_run_11_1_data": str(stage11_data_run) if stage11_data_run is not None else "",
        "enforce_stage09_gate": bool(ENFORCE_STAGE09_GATE),
        "stage09_gate_result": gate_result,
        "adapter_dir": str(adapter_dir),
        "adapter_dir_11_30": str(adapter_dir),
        "adapter_dir_31_60": str(adapter_dir_secondary) if adapter_dir_secondary is not None else "",
        "adapter_dir_61_100": str(adapter_dir_tertiary) if adapter_dir_tertiary is not None else "",
        "dual_band_route_enabled": bool(DUAL_BAND_ROUTE_ENABLED),
        "route_local_norm_enabled": bool(EVAL_ROUTE_LOCAL_NORM),
        "force_refusion_enabled": bool(EVAL_FORCE_REFUSION),
        "blend_alpha_11_30": float(BLEND_ALPHA_11_30),
        "blend_alpha_31_60": float(BLEND_ALPHA_31_60),
        "blend_alpha_31_40": float(BLEND_ALPHA_31_40),
        "blend_alpha_41_60": float(BLEND_ALPHA_41_60),
        "blend_alpha_61_100": float(BLEND_ALPHA_61_100),
        "second_stage_enable": bool(EVAL_SECOND_STAGE_ENABLE),
        "second_stage_shortlist_size": int(SECOND_STAGE_SHORTLIST_SIZE),
        "second_stage_global_topk": int(SECOND_STAGE_GLOBAL_TOPK),
        "second_stage_route_31_40_topk": int(SECOND_STAGE_ROUTE_31_40_TOPK),
        "second_stage_route_41_60_topk": int(SECOND_STAGE_ROUTE_41_60_TOPK),
        "second_stage_route_61_100_topk": int(SECOND_STAGE_ROUTE_61_100_TOPK),
        "second_stage_blend_alpha": float(SECOND_STAGE_BLEND_ALPHA),
        "second_stage_route_local_norm_enabled": bool(SECOND_STAGE_ROUTE_LOCAL_NORM),
        "second_stage_gate_enable": bool(SECOND_STAGE_GATE_ENABLE),
        "second_stage_gate_target_rank_31_40": int(SECOND_STAGE_GATE_TARGET_RANK_31_40),
        "second_stage_gate_target_rank_41_60": int(SECOND_STAGE_GATE_TARGET_RANK_41_60),
        "second_stage_gate_target_rank_61_100": int(SECOND_STAGE_GATE_TARGET_RANK_61_100),
        "second_stage_gate_max_blend_rank_31_40": int(SECOND_STAGE_GATE_MAX_BLEND_RANK_31_40),
        "second_stage_gate_max_blend_rank_41_60": int(SECOND_STAGE_GATE_MAX_BLEND_RANK_41_60),
        "second_stage_gate_max_blend_rank_61_100": int(SECOND_STAGE_GATE_MAX_BLEND_RANK_61_100),
        "second_stage_gate_min_margin_31_40": float(SECOND_STAGE_GATE_MIN_MARGIN_31_40),
        "second_stage_gate_min_margin_41_60": float(SECOND_STAGE_GATE_MIN_MARGIN_41_60),
        "second_stage_gate_min_margin_61_100": float(SECOND_STAGE_GATE_MIN_MARGIN_61_100),
        "second_stage_gate_cap_rank_31_40": int(SECOND_STAGE_GATE_CAP_RANK_31_40),
        "second_stage_gate_cap_rank_41_60": int(SECOND_STAGE_GATE_CAP_RANK_41_60),
        "second_stage_gate_cap_rank_61_100": int(SECOND_STAGE_GATE_CAP_RANK_61_100),
        "second_stage_gate_cap_epsilon": float(SECOND_STAGE_GATE_CAP_EPSILON),
        "second_stage_local_listwise_max_rivals": int(SECOND_STAGE_LOCAL_LISTWISE_MAX_RIVALS),
        "base_model": base_model,
        "buckets": buckets,
        "top_k": int(TOP_K),
        "rerank_topn": int(RERANK_TOPN),
        "candidate_rerank_topn": int(RERANK_TOPN),
        "bucket_score_file_pattern_canonical": "stage11_bucket_{bucket}_scores.csv",
        "bucket_score_file_pattern_legacy": "bucket_{bucket}_scores.csv",
        "bucket_partial_score_file_pattern_canonical": "stage11_bucket_{bucket}_scores_partial.csv",
        "bucket_partial_score_file_pattern_legacy": "bucket_{bucket}_scores_partial.csv",
        "blend_alpha": float(BLEND_ALPHA),
        "fusion_mode": str(EVAL_FUSION_MODE),
        "baseline_score_col": str(EVAL_BASELINE_SCORE_COL),
        "baseline_rank_col": str(EVAL_BASELINE_RANK_COL),
        "baseline_score_norm": str(EVAL_BASELINE_SCORE_NORM),
        "rescue_bonus_min_rank": int(RESCUE_BONUS_MIN_RANK),
        "rescue_bonus_max_rank": int(RESCUE_BONUS_MAX_RANK),
        "rescue_bonus_boundary_weight": float(RESCUE_BONUS_BOUNDARY_WEIGHT),
        "rescue_bonus_mid_weight": float(RESCUE_BONUS_MID_WEIGHT),
        "rescue_bonus_mid_weight_31_40": float(RESCUE_BONUS_MID_WEIGHT_31_40),
        "rescue_bonus_mid_weight_41_60": float(RESCUE_BONUS_MID_WEIGHT_41_60),
        "rescue_bonus_deep_weight": float(RESCUE_BONUS_DEEP_WEIGHT),
        "invert_prob": bool(INVERT_PROB),
        "eval_profile": str(EVAL_PROFILE),
        "eval_user_cohort_path": str(eval_user_cohort_path) if eval_user_cohort_path is not None else "",
        "eval_target_true_bands": sorted(list(EVAL_TARGET_TRUE_BANDS)),
        "use_stage11_eval_split": bool(USE_STAGE11_EVAL_SPLIT),
        "bucket_run_summaries": bucket_run_summaries,
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
        "eval_model_type": str(EVAL_MODEL_TYPE),
        "sidecar_score_col": str(sidecar_raw_col),
        "rm_score_norm": str(RM_SCORE_NORM) if EVAL_MODEL_TYPE == "rm" else "",
        "prompt_build_mode": str(PROMPT_BUILD_MODE),
        "attn_implementation": str(ATTN_IMPLEMENTATION or ""),
        "prompt_chunk_rows": int(PROMPT_CHUNK_ROWS),
        "pad_to_multiple_of": int(PAD_TO_MULTIPLE_OF),
        "stream_log_rows": int(STREAM_LOG_ROWS),
        "iter_coalesce_partitions": int(ITER_COALESCE_PARTITIONS),
        "intermediate_flush_rows": int(INTERMEDIATE_FLUSH_ROWS),
        "pretokenize_prompt_chunk": bool(PRETOKENIZE_PROMPT_CHUNK),
        "gpu_preload_prompt_chunk": bool(GPU_PRELOAD_PROMPT_CHUNK),
        "bucket_sort_prompt_chunk": bool(BUCKET_SORT_PROMPT_CHUNK),
        "local_trim_prompt_batch": bool(LOCAL_TRIM_PROMPT_BATCH),
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
            "source_run_11_2_31_60": str(stage11_2_run_secondary) if stage11_2_run_secondary is not None else "",
            "source_run_11_2_61_100": str(stage11_2_run_tertiary) if stage11_2_run_tertiary is not None else "",
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


