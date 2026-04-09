from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

from pipeline.project_paths import (
    env_or_project_path,
    normalize_legacy_project_path,
    project_path,
    write_latest_run_pointer,
)
from pipeline.qlora_prompting import (
    build_binary_prompt,
    build_binary_prompt_semantic,
    build_scoring_prompt,
    build_item_text,
    build_item_text_semantic_compact_boundary,
    build_item_text_full_lite,
    build_item_text_semantic_compact,
    build_item_text_semantic_compact_preserve,
    build_item_text_semantic_compact_targeted,
    build_item_text_semantic,
    build_item_text_sft_clean,
    build_pair_alignment_summary,
    build_user_text,
)
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context
from pipeline.stage11_text_features import (
    build_history_anchor_line,
    build_history_anchor_summary,
    clean_text,
    extract_user_evidence_text,
    keyword_match_score,
)


RUN_TAG = "stage11_1_qlora_build_dataset"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
INPUT_09_TEXT_MATCH_ROOT = env_or_project_path(
    "INPUT_09_TEXT_MATCH_ROOT_DIR",
    "data/output/09_candidate_wise_text_match_features_v1",
)
INPUT_09_TEXT_MATCH_RUN_DIR = os.getenv("INPUT_09_TEXT_MATCH_RUN_DIR", "").strip()
INPUT_09_USER_PROFILE_TEXT_ROOT = env_or_project_path(
    "INPUT_09_USER_PROFILE_TEXT_ROOT_DIR",
    "data/output/09_user_profiles",
)
INPUT_09_USER_PROFILE_TEXT_RUN_DIR = os.getenv("INPUT_09_USER_PROFILE_TEXT_RUN_DIR", "").strip()
INPUT_09_USER_TEXT_VIEWS_ROOT = env_or_project_path(
    "INPUT_09_USER_TEXT_VIEWS_ROOT_DIR",
    "data/output/09_candidate_wise_text_views_v1",
)
INPUT_09_USER_TEXT_VIEWS_RUN_DIR = os.getenv("INPUT_09_USER_TEXT_VIEWS_RUN_DIR", "").strip()
INPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_ROOT = env_or_project_path(
    "INPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_ROOT_DIR",
    "data/output/09_stage11_semantic_text_assets_v1",
)
INPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_RUN_DIR = os.getenv(
    "INPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_RUN_DIR",
    "",
).strip()
INPUT_09_MATCH_CHANNELS_ROOT = env_or_project_path(
    "INPUT_09_MATCH_CHANNELS_ROOT_DIR",
    "data/output/09_user_business_match_channels_v2",
)
INPUT_09_MATCH_CHANNELS_RUN_DIR = os.getenv("INPUT_09_MATCH_CHANNELS_RUN_DIR", "").strip()
INPUT_09_GROUP_GAP_ROOT = env_or_project_path(
    "INPUT_09_GROUP_GAP_ROOT_DIR",
    "data/output/09_stage10_group_gap_features_v1",
)
INPUT_09_GROUP_GAP_RUN_DIR = os.getenv("INPUT_09_GROUP_GAP_RUN_DIR", "").strip()
INPUT_10_ROOT = env_or_project_path("INPUT_10_ROOT_DIR", "data/output/p0_stage10_context_eval")
INPUT_10_RUN_DIR = os.getenv("INPUT_10_RUN_DIR", "").strip()
INPUT_10_13_RESCUE_AUDIT_ROOT = env_or_project_path(
    "INPUT_10_13_RESCUE_AUDIT_ROOT_DIR",
    "data/output/stage11_rescue_reason_audit",
)
INPUT_10_13_RESCUE_AUDIT_RUN_DIR = os.getenv("INPUT_10_13_RESCUE_AUDIT_RUN_DIR", "").strip()
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_DATA_ROOT_DIR", "data/output/11_qlora_data")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
# Prefer the generic Stage09 handoff name. Older runs may still only expose
# the legacy `candidates_pretrim150.parquet` alias, so keep it in fallback order.
CANDIDATE_FILE = os.getenv("TRAIN_CANDIDATE_FILE", "candidates_pretrim.parquet").strip() or "candidates_pretrim.parquet"
# Candidate window pulled from Stage09/10 for Stage11 data construction.
# This is not the final evaluation top-k metric.
TOPN_PER_USER = int(os.getenv("QLORA_TOPN_PER_USER", "120").strip() or 120)
ENABLE_PAIRWISE_POOL_EXPORT = os.getenv("QLORA_ENABLE_PAIRWISE_POOL_EXPORT", "false").strip().lower() == "true"
# Learned-rank cutoff used only for pairwise-pool export / rescue data mining.
PAIRWISE_POOL_TOPN = int(os.getenv("QLORA_PAIRWISE_POOL_TOPN", "100").strip() or 100)
PAIRWISE_POOL_MINIMAL_EXPORT = os.getenv("QLORA_PAIRWISE_POOL_MINIMAL_EXPORT", "true").strip().lower() == "true"
NEG_PER_USER = int(os.getenv("QLORA_NEG_PER_USER", "8").strip() or 8)
NEG_LAYERED_ENABLED = os.getenv("QLORA_NEG_LAYERED_ENABLED", "true").strip().lower() == "true"
NEG_SAMPLER_MODE = os.getenv("QLORA_NEG_SAMPLER_MODE", "layered").strip().lower() or "layered"
NEG_HARD_RATIO = float(os.getenv("QLORA_NEG_HARD_RATIO", "0.50").strip() or 0.50)
NEG_NEAR_RATIO = float(os.getenv("QLORA_NEG_NEAR_RATIO", "0.30").strip() or 0.30)
NEG_HARD_RANK_MAX = int(os.getenv("QLORA_NEG_HARD_RANK_MAX", "120").strip() or 120)
NEG_HARD_WEIGHT = float(os.getenv("QLORA_NEG_HARD_WEIGHT", "1.20").strip() or 1.20)
NEG_NEAR_WEIGHT = float(os.getenv("QLORA_NEG_NEAR_WEIGHT", "1.10").strip() or 1.10)
NEG_EASY_WEIGHT = float(os.getenv("QLORA_NEG_EASY_WEIGHT", "0.90").strip() or 0.90)
NEG_FILL_WEIGHT = float(os.getenv("QLORA_NEG_FILL_WEIGHT", "1.00").strip() or 1.00)
NEG_BAND_TOP10 = int(os.getenv("QLORA_NEG_BAND_TOP10", "1").strip() or 1)
NEG_BAND_11_30 = int(os.getenv("QLORA_NEG_BAND_11_30", "2").strip() or 2)
NEG_BAND_31_80 = int(os.getenv("QLORA_NEG_BAND_31_80", "2").strip() or 2)
NEG_BAND_81_150 = int(os.getenv("QLORA_NEG_BAND_81_150", "2").strip() or 2)
NEG_BAND_NEAR = int(os.getenv("QLORA_NEG_BAND_NEAR", "1").strip() or 1)
MAX_USERS_PER_BUCKET = int(os.getenv("QLORA_MAX_USERS_PER_BUCKET", "0").strip() or 0)
USER_CAP_RANDOMIZED = os.getenv("QLORA_USER_CAP_RANDOMIZED", "false").strip().lower() == "true"
INCLUDE_VALID_POS = os.getenv("QLORA_INCLUDE_VALID_POS", "false").strip().lower() == "true"
VALID_POS_WEIGHT = float(os.getenv("QLORA_VALID_POS_WEIGHT", "0.35").strip() or 0.35)
VALID_POS_ONLY_IF_NO_TRUE = os.getenv("QLORA_VALID_POS_ONLY_IF_NO_TRUE", "false").strip().lower() == "true"
EVAL_USER_FRAC = float(os.getenv("QLORA_EVAL_USER_FRAC", "0.1").strip() or 0.1)
DATASET_EVAL_USER_COHORT_PATH_RAW = os.getenv("QLORA_DATASET_EVAL_USER_COHORT_PATH", "").strip()
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
MAX_ROWS_PER_BUCKET = int(os.getenv("QLORA_MAX_ROWS_PER_BUCKET", "300000").strip() or 300000)
ROW_CAP_ORDERED = os.getenv("QLORA_ROW_CAP_ORDERED", "false").strip().lower() == "true"
AUDIT_ONLY = os.getenv("QLORA_AUDIT_ONLY", "false").strip().lower() == "true"
SKIP_POINTWISE_EXPORT = os.getenv("QLORA_SKIP_POINTWISE_EXPORT", "false").strip().lower() == "true"
SKIP_RICH_SFT_JSON_EXPORT = os.getenv("QLORA_SKIP_RICH_SFT_JSON_EXPORT", "false").strip().lower() == "true"
SKIP_PAIRWISE_POOL_JSON_EXPORT = os.getenv("QLORA_SKIP_PAIRWISE_POOL_JSON_EXPORT", "false").strip().lower() == "true"

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

PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "full").strip().lower() or "full"  # "full" | "full_lite" | "semantic" | "semantic_rm" | "semantic_compact_rm" | "semantic_compact_preserve_rm" | "semantic_compact_targeted_rm" | "semantic_compact_boundary_rm" | "sft_clean"
ATTACH_STAGE10_TEXT_MATCH = os.getenv("QLORA_ATTACH_STAGE10_TEXT_MATCH", "false").strip().lower() == "true"
ATTACH_STAGE10_MATCH_CHANNELS = os.getenv("QLORA_ATTACH_STAGE10_MATCH_CHANNELS", "false").strip().lower() == "true"
ATTACH_STAGE10_GROUP_GAP = os.getenv("QLORA_ATTACH_STAGE10_GROUP_GAP", "false").strip().lower() == "true"
ATTACH_STAGE10_LEARNED = os.getenv("QLORA_ATTACH_STAGE10_LEARNED", "false").strip().lower() == "true"
STAGE10_LEARNED_FALLBACK_TO_PRE = os.getenv("QLORA_STAGE10_LEARNED_FALLBACK_TO_PRE", "false").strip().lower() == "true"
ATTACH_STAGE11_RESCUE_REASON = os.getenv("QLORA_ATTACH_STAGE11_RESCUE_REASON", "false").strip().lower() == "true"
ENABLE_STAGE11_WEAK_REASON_BACKFILL = os.getenv("QLORA_ENABLE_STAGE11_WEAK_REASON_BACKFILL", "true").strip().lower() == "true"
INCLUDE_HIST_POS = os.getenv("QLORA_INCLUDE_HIST_POS", "false").strip().lower() == "true"
HIST_POS_MIN_RATING = float(os.getenv("QLORA_HIST_POS_MIN_RATING", "4.0").strip() or 4.0)
HIST_POS_MAX_PER_USER = int(os.getenv("QLORA_HIST_POS_MAX_PER_USER", "3").strip() or 3)
HIST_POS_WEIGHT = float(os.getenv("QLORA_HIST_POS_WEIGHT", "0.25").strip() or 0.25)

ENABLE_RICH_SFT_EXPORT = os.getenv("QLORA_ENABLE_RICH_SFT_EXPORT", "false").strip().lower() == "true"
RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER = int(os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER", "3").strip() or 3)
RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING = float(
    os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING", "4.5").strip() or 4.5
)
RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING = float(
    os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING", "4.0").strip() or 4.0
)
RICH_SFT_HISTORY_ANCHOR_MAX_CHARS = int(os.getenv("QLORA_RICH_SFT_HISTORY_ANCHOR_MAX_CHARS", "180").strip() or 180)
RICH_SFT_NEG_EXPLICIT = int(os.getenv("QLORA_RICH_SFT_NEG_EXPLICIT", "1").strip() or 1)
RICH_SFT_NEG_HARD = int(os.getenv("QLORA_RICH_SFT_NEG_HARD", "2").strip() or 2)
RICH_SFT_NEG_NEAR = int(os.getenv("QLORA_RICH_SFT_NEG_NEAR", "1").strip() or 1)
RICH_SFT_NEG_MID = int(os.getenv("QLORA_RICH_SFT_NEG_MID", "0").strip() or 0)
RICH_SFT_NEG_TAIL = int(os.getenv("QLORA_RICH_SFT_NEG_TAIL", "0").strip() or 0)
RICH_SFT_NEG_HARD_RANK_MAX = int(os.getenv("QLORA_RICH_SFT_NEG_HARD_RANK_MAX", "20").strip() or 20)
RICH_SFT_NEG_MID_RANK_MAX = int(os.getenv("QLORA_RICH_SFT_NEG_MID_RANK_MAX", "60").strip() or 60)
RICH_SFT_NEG_MAX_RATING = float(os.getenv("QLORA_RICH_SFT_NEG_MAX_RATING", "2.5").strip() or 2.5)
RICH_SFT_ALLOW_NEG_FILL = os.getenv("QLORA_RICH_SFT_ALLOW_NEG_FILL", "false").strip().lower() == "true"

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_SQL_CONSTRAINT_PROPAGATION_ENABLED = (
    os.getenv("SPARK_SQL_CONSTRAINT_PROPAGATION_ENABLED", "false").strip().lower() == "true"
)
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
SPARK_PYTHON_AUTH_SOCKET_TIMEOUT = os.getenv("SPARK_PYTHON_AUTH_SOCKET_TIMEOUT", "120s").strip() or "120s"
SPARK_PYTHON_UNIX_DOMAIN_SOCKET_ENABLED = (
    os.getenv("SPARK_PYTHON_UNIX_DOMAIN_SOCKET_ENABLED", "false").strip().lower() == "true"
)
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
SPARK_DRIVER_BIND_ADDRESS = os.getenv("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1").strip() or "127.0.0.1"
ENABLE_HEAVY_STAGE_CHECKPOINTS = os.getenv("QLORA_ENABLE_HEAVY_STAGE_CHECKPOINTS", "true").strip().lower() == "true"
HEAVY_STAGE_CHECKPOINT_EAGER = os.getenv("QLORA_HEAVY_STAGE_CHECKPOINT_EAGER", "true").strip().lower() == "true"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
WRITE_JSON_PARTITIONS = int(os.getenv("QLORA_WRITE_JSON_PARTITIONS", "1").strip() or 1)
WRITE_PARQUET_PARTITIONS = int(os.getenv("QLORA_WRITE_PARQUET_PARTITIONS", "1").strip() or 1)
WRITE_AUX_PARTITIONS = int(os.getenv("QLORA_WRITE_AUX_PARTITIONS", "1").strip() or 1)
PAIRWISE_POOL_WRITE_JSON_PARTITIONS = int(os.getenv("QLORA_PAIRWISE_POOL_WRITE_JSON_PARTITIONS", "1").strip() or 1)
PAIRWISE_POOL_WRITE_PARQUET_PARTITIONS = int(os.getenv("QLORA_PAIRWISE_POOL_WRITE_PARQUET_PARTITIONS", "1").strip() or 1)

REASON_AUDIT_EXPORT_COLUMNS = [
    "primary_reason",
    "semantic_edge_underweighted",
    "multi_route_underweighted",
    "channel_context_underweighted",
    "head_prior_blocked",
    "easy_but_useful",
    "hard_but_learnable",
    "non_actionable",
]

DPO_SUPPORT_EXPORT_COLUMNS = [
    "name",
    "city",
    "categories",
    "primary_category",
    "top_pos_tags",
    "top_neg_tags",
    "semantic_score",
    "semantic_confidence",
    "source_set_text",
    "source_count",
    "nonpopular_source_count",
    "profile_cluster_source_count",
    "user_segment",
    "als_rank",
    "cluster_rank",
    "profile_rank",
    "popular_rank",
    "context_rank",
    "has_context",
    "context_detail_count",
    "semantic_support",
    "semantic_tag_richness",
    "tower_score",
    "seq_score",
    "cluster_for_recsys",
    "cluster_label_for_recsys",
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
    "learned_blend_score",
    "learned_rank",
    "learned_rank_band",
    *REASON_AUDIT_EXPORT_COLUMNS,
]

STAGE11_SEMANTIC_TEXT_EXPORT_COLUMNS = [
    "stable_preferences_text",
    "recent_intent_text_v2",
    "avoidance_text_v2",
    "history_anchor_hint_text",
    "user_semantic_profile_text_v2",
    "user_profile_richness_v2",
    "user_profile_richness_tier_v2",
    "core_offering_text",
    "scene_fit_text",
    "strengths_text",
    "risk_points_text",
    "merchant_semantic_profile_text_v2",
    "merchant_profile_richness_v2",
    "fit_reasons_text_v1",
    "friction_reasons_text_v1",
    "evidence_basis_text_v1",
    "pair_alignment_richness_v1",
    "semantic_prompt_readiness_tier_v1",
]


_SPARK_TMP_CTX: SparkTmpContext | None = None


def _resolve_python_exec(raw: str, fallback: str) -> str:
    raw_text = str(raw or "").strip()
    if raw_text:
        p = Path(raw_text)
        if p.exists() and p.is_file():
            return str(p)
    fb = Path(str(fallback or "").strip() or sys.executable)
    return str(fb)


def _configure_pyspark_python() -> tuple[str, str]:
    # Force Spark workers to use the same interpreter as driver by default.
    worker_py = _resolve_python_exec(os.getenv("PYSPARK_PYTHON", "").strip(), sys.executable)
    driver_py = _resolve_python_exec(os.getenv("PYSPARK_DRIVER_PYTHON", "").strip(), worker_py)
    os.environ["PYSPARK_PYTHON"] = worker_py
    os.environ["PYSPARK_DRIVER_PYTHON"] = driver_py
    return worker_py, driver_py


def parse_bucket_override(raw: str) -> set[int]:
    out: set[int] = set()
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def resolve_optional_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = normalize_legacy_project_path(raw)
        if not p.exists():
            raise FileNotFoundError(f"run dir not found: {p}")
        return p
    return pick_latest_run(root, suffix)


def resolve_text_match_run() -> Path:
    return resolve_optional_run(
        INPUT_09_TEXT_MATCH_RUN_DIR,
        INPUT_09_TEXT_MATCH_ROOT,
        "_full_stage09_candidate_wise_text_match_features_v1_build",
    )


def resolve_user_profile_text_run() -> Path:
    return resolve_optional_run(
        INPUT_09_USER_PROFILE_TEXT_RUN_DIR,
        INPUT_09_USER_PROFILE_TEXT_ROOT,
        "_full_stage09_user_profile_build",
    )


def resolve_user_text_views_run() -> Path:
    return resolve_optional_run(
        INPUT_09_USER_TEXT_VIEWS_RUN_DIR,
        INPUT_09_USER_TEXT_VIEWS_ROOT,
        "_full_stage09_candidate_wise_text_views_v1_build",
    )


def resolve_stage11_semantic_text_assets_run() -> Path:
    return resolve_optional_run(
        INPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_RUN_DIR,
        INPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_ROOT,
        "_full_stage09_stage11_semantic_text_assets_v1_build",
    )


def resolve_match_channel_run() -> Path:
    return resolve_optional_run(
        INPUT_09_MATCH_CHANNELS_RUN_DIR,
        INPUT_09_MATCH_CHANNELS_ROOT,
        "_full_stage09_user_business_match_channels_v2_build",
    )


def resolve_group_gap_run() -> Path:
    return resolve_optional_run(
        INPUT_09_GROUP_GAP_RUN_DIR,
        INPUT_09_GROUP_GAP_ROOT,
        "_full_stage09_stage10_group_gap_features_v1_build",
    )


def resolve_stage10_run() -> Path:
    if INPUT_10_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_10_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_10_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_10_ROOT, "_stage10_2_rank_infer_eval")


def resolve_stage11_rescue_audit_run() -> Path:
    if INPUT_10_13_RESCUE_AUDIT_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_10_13_RESCUE_AUDIT_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_10_13_RESCUE_AUDIT_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_10_13_RESCUE_AUDIT_ROOT, "_stage11_rescue_reason_audit")


def resolve_stage09_meta_path(raw_path: str) -> Path:
    p = normalize_legacy_project_path(str(raw_path or "").strip())
    if p.exists():
        return p
    return p


def resolve_dataset_eval_user_cohort_path() -> Path | None:
    raw = str(DATASET_EVAL_USER_COHORT_PATH_RAW or "").strip()
    if not raw:
        return None
    p = normalize_legacy_project_path(raw)
    if p.exists():
        return p
    raise FileNotFoundError(f"QLORA_DATASET_EVAL_USER_COHORT_PATH not found: {p}")


def load_eval_users_from_cohort(
    spark: SparkSession,
    cohort_path: Path,
) -> tuple[DataFrame, int, list[str]]:
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
    if "user_idx" in cols:
        out["user_idx"] = pd.to_numeric(pdf["user_idx"], errors="coerce").astype("Int64")

    if "user_id" in out.columns:
        out["user_id"] = out["user_id"].replace({"": pd.NA})
    subset = [c for c in ("user_id", "user_idx") if c in out.columns]
    out = out.dropna(subset=subset, how="all")
    if "user_idx" in out.columns:
        out = out.dropna(subset=["user_idx"])
        out["user_idx"] = out["user_idx"].astype("int64")
    if "user_id" in out.columns:
        out = out.dropna(subset=["user_id"])
    out = out.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return spark.createDataFrame(out), int(len(out)), subset


def with_split_assignment(
    df: DataFrame,
    *,
    eval_user_frac: float,
    fixed_eval_users: DataFrame | None,
    fixed_eval_join_cols: list[str],
) -> DataFrame:
    if fixed_eval_users is None or not fixed_eval_join_cols:
        return df.withColumn(
            "split",
            F.when(
                (F.col("user_idx").cast("int") % F.lit(100)) < F.lit(int(100 * eval_user_frac)),
                F.lit("eval"),
            ).otherwise(F.lit("train")),
        )

    marked_eval = F.broadcast(
        fixed_eval_users.select(*fixed_eval_join_cols)
        .dropDuplicates(fixed_eval_join_cols)
        .withColumn("_fixed_eval", F.lit(1))
    )
    return (
        df.join(marked_eval, on=fixed_eval_join_cols, how="left")
        .withColumn(
            "split",
            F.when(F.col("_fixed_eval") == F.lit(1), F.lit("eval")).otherwise(F.lit("train")),
        )
        .drop("_fixed_eval")
    )


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    worker_py, driver_py = _configure_pyspark_python()
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    existing_pythonpath = [part for part in os.getenv("PYTHONPATH", "").split(os.pathsep) if part]
    pythonpath_parts: list[str] = []
    for candidate in [script_dir.as_posix(), repo_root.as_posix(), *existing_pythonpath]:
        if candidate and candidate not in pythonpath_parts:
            pythonpath_parts.append(candidate)
    pythonpath_value = os.pathsep.join(pythonpath_parts)
    os.environ["PYTHONPATH"] = pythonpath_value
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
    print(
        f"[SPARK] master={SPARK_MASTER} pyspark_python={worker_py} "
        f"pyspark_driver_python={driver_py} pythonpath={pythonpath_value}"
    )
    spark = (
        SparkSession.builder.appName(RUN_TAG)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.host", SPARK_DRIVER_HOST)
        .config("spark.driver.bindAddress", SPARK_DRIVER_BIND_ADDRESS)
        .config("spark.local.dir", str(_SPARK_TMP_CTX.spark_local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config(
            "spark.sql.constraintPropagation.enabled",
            "true" if SPARK_SQL_CONSTRAINT_PROPAGATION_ENABLED else "false",
        )
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.unix.domain.socket.enabled", "true" if SPARK_PYTHON_UNIX_DOMAIN_SOCKET_ENABLED else "false")
        .config("spark.python.authenticate.socketTimeout", SPARK_PYTHON_AUTH_SOCKET_TIMEOUT)
        .config("spark.pyspark.python", worker_py)
        .config("spark.pyspark.driver.python", driver_py)
        .config("spark.executorEnv.PYTHONPATH", pythonpath_value)
        .config("spark.driverEnv.PYTHONPATH", pythonpath_value)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    checkpoint_dir = Path(_SPARK_TMP_CTX.spark_local_dir) / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir.as_posix())
    print(f"[SPARK] checkpoint_dir={checkpoint_dir.as_posix()} enabled={ENABLE_HEAVY_STAGE_CHECKPOINTS}")
    return spark


def maybe_checkpoint_df(df: DataFrame, label: str) -> DataFrame:
    if not ENABLE_HEAVY_STAGE_CHECKPOINTS:
        return df
    print(f"[STEP] {label} checkpoint start")
    out = df.checkpoint(eager=HEAVY_STAGE_CHECKPOINT_EAGER)
    print(f"[STEP] {label} checkpoint ready")
    return out


def _optional_text_expr(df: DataFrame, column: str, alias: str | None = None) -> F.Column:
    target = alias or column
    if column in df.columns:
        return F.coalesce(F.col(column), F.lit("")).alias(target)
    return F.lit("").alias(target)


def _optional_int_expr(df: DataFrame, column: str, alias: str | None = None) -> F.Column:
    target = alias or column
    if column in df.columns:
        return F.coalesce(F.col(column).cast("int"), F.lit(0)).alias(target)
    return F.lit(0).alias(target)


def choose_candidate_file(bucket_dir: Path) -> Path:
    candidates = [
        CANDIDATE_FILE,
        "candidates_pretrim.parquet",
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
    ]
    for name in candidates:
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate file not found in {bucket_dir}")


def coalesce_for_write(df: DataFrame, partitions: int) -> DataFrame:
    n_parts = int(partitions)
    if n_parts <= 0:
        return df
    return df.coalesce(int(n_parts))


def collect_split_label_user_summary(
    df: DataFrame,
    *,
    include_history_anchor_nonempty: bool = False,
    include_split_user_counts: bool = False,
) -> dict[str, int]:
    agg_exprs: list[Any] = [
        F.count("*").alias("rows_total"),
        F.sum(F.when((F.col("split") == F.lit("train")) & (F.col("label") == F.lit(0)), F.lit(1)).otherwise(F.lit(0))).alias("train_label_0"),
        F.sum(F.when((F.col("split") == F.lit("train")) & (F.col("label") == F.lit(1)), F.lit(1)).otherwise(F.lit(0))).alias("train_label_1"),
        F.sum(F.when((F.col("split") == F.lit("eval")) & (F.col("label") == F.lit(0)), F.lit(1)).otherwise(F.lit(0))).alias("eval_label_0"),
        F.sum(F.when((F.col("split") == F.lit("eval")) & (F.col("label") == F.lit(1)), F.lit(1)).otherwise(F.lit(0))).alias("eval_label_1"),
        F.countDistinct("user_idx").alias("users_total"),
        F.countDistinct(F.when(F.col("label") == F.lit(1), F.col("user_idx"))).alias("users_with_pos"),
    ]
    if include_split_user_counts:
        agg_exprs.extend(
            [
                F.countDistinct(F.when(F.col("split") == F.lit("train"), F.col("user_idx"))).alias("train_users"),
                F.countDistinct(F.when(F.col("split") == F.lit("eval"), F.col("user_idx"))).alias("eval_users"),
            ]
        )
    if include_history_anchor_nonempty and "history_anchor_text" in df.columns:
        agg_exprs.append(
            F.sum(
                F.when(F.length(F.coalesce(F.col("history_anchor_text"), F.lit(""))) > F.lit(0), F.lit(1)).otherwise(F.lit(0))
            ).alias("history_anchor_nonempty")
        )
    row = df.agg(*agg_exprs).collect()[0]
    return {str(k): int(row[k] or 0) for k in row.asDict().keys()}


def collect_split_label_user_summary_safe(
    df: DataFrame,
    *,
    include_history_anchor_nonempty: bool = False,
    include_split_user_counts: bool = False,
) -> tuple[dict[str, int], str]:
    try:
        return (
            collect_split_label_user_summary(
                df,
                include_history_anchor_nonempty=include_history_anchor_nonempty,
                include_split_user_counts=include_split_user_counts,
            ),
            "",
        )
    except Exception as exc:
        return ({}, f"{type(exc).__name__}: {exc}")


def pre_rank_band_expr(col_name: str = "pre_rank") -> Any:
    col = F.col(col_name).cast("int")
    return (
        F.when(col <= F.lit(10), F.lit("001_010"))
        .when(col <= F.lit(30), F.lit("011_030"))
        .when(col <= F.lit(80), F.lit("031_080"))
        .when(col <= F.lit(150), F.lit("081_150"))
        .otherwise(F.lit("151_plus"))
    )


def learned_rank_band_expr(col_name: str = "learned_rank") -> Any:
    col = F.col(col_name).cast("int")
    return (
        F.when(col <= F.lit(10), F.lit("001_010"))
        .when(col <= F.lit(30), F.lit("011_030"))
        .when(col <= F.lit(60), F.lit("031_060"))
        .when(col <= F.lit(100), F.lit("061_100"))
        .otherwise(F.lit("101_plus"))
    )


def project_candidate_rows(cand_raw: DataFrame) -> DataFrame:
    cand_cols = set(cand_raw.columns)

    def _num_col(name: str) -> Any:
        if name in cand_cols:
            return F.col(name).cast("double").alias(name)
        return F.lit(None).cast("double").alias(name)

    def _str_col(name: str) -> Any:
        if name in cand_cols:
            return F.coalesce(F.col(name).cast("string"), F.lit("")).alias(name)
        return F.lit("").alias(name)

    def _bool_col(name: str) -> Any:
        if name in cand_cols:
            return F.coalesce(F.col(name).cast("boolean"), F.lit(False)).alias(name)
        return F.lit(False).alias(name)

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

    return cand_raw.select(
        "user_idx",
        "item_idx",
        "business_id",
        "pre_rank",
        "pre_score",
        pre_rank_band_expr("pre_rank").alias("pre_rank_band"),
        _num_col("learned_blend_score"),
        _num_col("learned_rank"),
        learned_rank_band_expr("learned_rank").alias("learned_rank_band"),
        _str_col("primary_reason"),
        _bool_col("semantic_edge_underweighted"),
        _bool_col("multi_route_underweighted"),
        _bool_col("channel_context_underweighted"),
        _bool_col("head_prior_blocked"),
        _bool_col("easy_but_useful"),
        _bool_col("hard_but_learnable"),
        _bool_col("non_actionable"),
        "name",
        "city",
        "categories",
        "primary_category",
        source_set_text,
        _num_col("source_count"),
        _num_col("nonpopular_source_count"),
        _num_col("profile_cluster_source_count"),
        user_segment_col,
        _num_col("als_rank"),
        _num_col("cluster_rank"),
        _num_col("profile_rank"),
        _num_col("popular_rank"),
        _num_col("context_rank"),
        _num_col("has_context"),
        _num_col("context_detail_count"),
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
    )


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
        print("[STAGE11] text_match attach skipped; columns already present")
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
    print(f"[STAGE11] attach text_match run={text_run} cols={missing_cols}")
    return cand_raw.join(F.broadcast(text_df), on=["user_idx", "item_idx"], how="left")


def attach_stage10_match_channel_features(
    spark: SparkSession,
    cand_raw: DataFrame,
    truth_df: DataFrame,
) -> DataFrame:
    if not ATTACH_STAGE10_MATCH_CHANNELS:
        return cand_raw
    cols = set(cand_raw.columns)
    if "user_idx" not in cols or "business_id" not in cols:
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
        print("[STAGE11] match_channels attach skipped; columns already present")
        return cand_raw
    match_run = resolve_match_channel_run()
    channel_path = match_run / "user_business_match_channels_v2_user_item.parquet"
    if not channel_path.exists():
        raise FileNotFoundError(f"user-business channel parquet missing: {channel_path}")
    user_map = truth_df.select("user_idx", "user_id").dropDuplicates(["user_idx"])
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
    print(f"[STAGE11] attach match_channels run={match_run} cols={missing_cols}")
    return (
        cand_raw.join(F.broadcast(user_map), on="user_idx", how="left")
        .join(F.broadcast(channel_df), on=["user_id", "business_id"], how="left")
        .drop("user_id")
    )


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
        print("[STAGE11] group_gap attach skipped; columns already present")
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
    print(f"[STAGE11] attach group_gap run={gap_run} cols={missing_cols}")
    return cand_raw.join(F.broadcast(gap_df), on=["user_idx", "item_idx"], how="left")


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
        if STAGE10_LEARNED_FALLBACK_TO_PRE and "pre_score" in cols and "pre_rank" in cols:
            print(
                f"[STAGE11] stage10 learned parquet missing for bucket={int(bucket)}; "
                "fallback_to_pre=true so learned_blend_score/pre_rank are derived from pre_score/pre_rank"
            )
            out = cand_raw
            if "learned_blend_score" in missing_cols:
                out = out.withColumn("learned_blend_score", F.col("pre_score").cast("double"))
            if "learned_rank" in missing_cols:
                out = out.withColumn("learned_rank", F.col("pre_rank").cast("double"))
            return out
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


def attach_stage11_rescue_reason_features(
    spark: SparkSession,
    cand_raw: DataFrame,
) -> DataFrame:
    if not ATTACH_STAGE11_RESCUE_REASON:
        return cand_raw
    cols = set(cand_raw.columns)
    if "user_idx" not in cols or "item_idx" not in cols:
        return cand_raw
    missing_cols = [c for c in REASON_AUDIT_EXPORT_COLUMNS if c not in cols]
    if not missing_cols:
        print("[STAGE11] rescue_reason attach skipped; columns already present")
        return cand_raw
    reason_run = resolve_stage11_rescue_audit_run()
    reason_path = reason_run / "rescue_reason_user_table.parquet"
    if not reason_path.exists():
        raise FileNotFoundError(f"stage11 rescue reason parquet missing: {reason_path}")
    reason_df = (
        spark.read.parquet(reason_path.as_posix())
        .select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("true_item_idx").cast("int").alias("item_idx"),
            F.coalesce(F.col("primary_reason").cast("string"), F.lit("")).alias("primary_reason"),
            F.coalesce(F.col("semantic_edge_underweighted").cast("boolean"), F.lit(False)).alias("semantic_edge_underweighted"),
            F.coalesce(F.col("multi_route_underweighted").cast("boolean"), F.lit(False)).alias("multi_route_underweighted"),
            F.coalesce(F.col("channel_context_underweighted").cast("boolean"), F.lit(False)).alias("channel_context_underweighted"),
            F.coalesce(F.col("head_prior_blocked").cast("boolean"), F.lit(False)).alias("head_prior_blocked"),
            F.coalesce(F.col("easy_but_useful").cast("boolean"), F.lit(False)).alias("easy_but_useful"),
            F.coalesce(F.col("hard_but_learnable").cast("boolean"), F.lit(False)).alias("hard_but_learnable"),
            F.coalesce(F.col("non_actionable").cast("boolean"), F.lit(False)).alias("non_actionable"),
        )
        .dropDuplicates(["user_idx", "item_idx"])
    )
    print(f"[STAGE11] attach rescue_reason run={reason_run} cols={missing_cols}")
    cand_raw = cand_raw.join(F.broadcast(reason_df), on=["user_idx", "item_idx"], how="left")
    if not ENABLE_STAGE11_WEAK_REASON_BACKFILL:
        return cand_raw

    cand_cols = set(cand_raw.columns)

    def _num_col(name: str, default: float = 0.0) -> Any:
        if name in cand_cols:
            return F.coalesce(F.col(name).cast("double"), F.lit(default))
        return F.lit(default)

    def _bool_col(name: str, default: bool = False) -> Any:
        if name in cand_cols:
            return F.coalesce(F.col(name).cast("boolean"), F.lit(default))
        return F.lit(default)

    tmp_prefix = "_s11r_"
    existing_easy = _bool_col("easy_but_useful", False)
    existing_hard = _bool_col("hard_but_learnable", False)
    existing_non_actionable = _bool_col("non_actionable", False)

    enriched = (
        cand_raw
        .withColumn(f"{tmp_prefix}primary_reason_norm", F.trim(F.coalesce(F.col("primary_reason").cast("string"), F.lit(""))))
        .withColumn(f"{tmp_prefix}learned_rank_num", _num_col("learned_rank", 999999.0))
        .withColumn(f"{tmp_prefix}learned_rank_band", learned_rank_band_expr("learned_rank"))
        .withColumn(
            f"{tmp_prefix}in_rescue_scope",
            F.col(f"{tmp_prefix}learned_rank_num").between(11.0, 100.0),
        )
        .withColumn(
            f"{tmp_prefix}empty_reason",
            F.col(f"{tmp_prefix}primary_reason_norm") == F.lit(""),
        )
        .withColumn(
            f"{tmp_prefix}eligible",
            F.col(f"{tmp_prefix}in_rescue_scope") & F.col(f"{tmp_prefix}empty_reason") & (~existing_non_actionable),
        )
        .withColumn(
            f"{tmp_prefix}unclear_non_actionable",
            F.col(f"{tmp_prefix}in_rescue_scope") & F.col(f"{tmp_prefix}empty_reason") & existing_non_actionable,
        )
        .withColumn(
            f"{tmp_prefix}multi_route_signal",
            (_num_col("source_count", 0.0) >= F.lit(3.0))
            | (
                (_num_col("source_count", 0.0) >= F.lit(2.0))
                & (
                    (_num_col("nonpopular_source_count", 0.0) >= F.lit(1.0))
                    | (_num_col("profile_cluster_source_count", 0.0) >= F.lit(1.0))
                )
            ),
        )
        .withColumn(
            f"{tmp_prefix}channel_context_signal",
            (_num_col("has_context", 0.0) >= F.lit(1.0))
            | (_num_col("context_detail_count", 0.0) >= F.lit(1.0))
            | (_num_col("channel_preference_core_v1", 0.0) > F.lit(0.0))
            | (_num_col("channel_recent_intent_v1", 0.0) > F.lit(0.0))
            | (_num_col("channel_context_time_v1", 0.0) > F.lit(0.0))
            | (_num_col("channel_evidence_support_v1", 0.0) > F.lit(0.0)),
        )
        .withColumn(
            f"{tmp_prefix}semantic_signal",
            (_num_col("semantic_support", 0.0) > F.lit(0.0))
            | (_num_col("semantic_tag_richness", 0.0) >= F.lit(1.0))
            | (_num_col("schema_weighted_overlap_user_ratio_v2_rank_pct_v3", 0.0) >= F.lit(0.5))
            | (_num_col("schema_weighted_net_score_v2_rank_pct_v3", 0.0) >= F.lit(0.5)),
        )
        .withColumn(
            f"{tmp_prefix}head_prior_signal",
            F.col(f"{tmp_prefix}learned_rank_num").between(11.0, 30.0),
        )
        .withColumn(
            f"{tmp_prefix}signal_count",
            F.when(F.col(f"{tmp_prefix}multi_route_signal"), F.lit(1)).otherwise(F.lit(0))
            + F.when(F.col(f"{tmp_prefix}channel_context_signal"), F.lit(1)).otherwise(F.lit(0))
            + F.when(F.col(f"{tmp_prefix}semantic_signal"), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            f"{tmp_prefix}derived_reason",
            F.when(
                F.col(f"{tmp_prefix}unclear_non_actionable"),
                F.lit("unclear_or_non_actionable"),
            ).when(
                F.col(f"{tmp_prefix}eligible") & F.col(f"{tmp_prefix}multi_route_signal"),
                F.lit("multi_route_underweighted_weak"),
            )
            .when(
                F.col(f"{tmp_prefix}eligible") & F.col(f"{tmp_prefix}channel_context_signal"),
                F.lit("channel_context_underweighted_weak"),
            )
            .when(
                F.col(f"{tmp_prefix}eligible") & F.col(f"{tmp_prefix}semantic_signal"),
                F.lit("semantic_edge_xgb_ambiguous_weak"),
            )
            .when(
                F.col(f"{tmp_prefix}eligible") & F.col(f"{tmp_prefix}head_prior_signal"),
                F.lit("head_prior_blocked_weak"),
            )
            .when(F.col(f"{tmp_prefix}eligible"), F.lit("semantic_edge_xgb_ambiguous_weak"))
            .otherwise(F.col(f"{tmp_prefix}primary_reason_norm")),
        )
        .withColumn(
            f"{tmp_prefix}derived_easy",
            F.col(f"{tmp_prefix}eligible")
            & (
                (F.col(f"{tmp_prefix}signal_count") >= F.lit(2))
                | (F.col(f"{tmp_prefix}learned_rank_band") == F.lit("061_100"))
            ),
        )
        .withColumn(
            f"{tmp_prefix}derived_hard",
            F.col(f"{tmp_prefix}eligible") & (~F.col(f"{tmp_prefix}derived_easy")),
        )
        .withColumn("primary_reason", F.col(f"{tmp_prefix}derived_reason"))
        .withColumn("easy_but_useful", existing_easy | F.col(f"{tmp_prefix}derived_easy"))
        .withColumn("hard_but_learnable", existing_hard | F.col(f"{tmp_prefix}derived_hard"))
        .withColumn("non_actionable", existing_non_actionable | F.col(f"{tmp_prefix}unclear_non_actionable"))
    )

    return enriched.drop(
        f"{tmp_prefix}primary_reason_norm",
        f"{tmp_prefix}learned_rank_num",
        f"{tmp_prefix}learned_rank_band",
        f"{tmp_prefix}in_rescue_scope",
        f"{tmp_prefix}empty_reason",
        f"{tmp_prefix}eligible",
        f"{tmp_prefix}multi_route_signal",
        f"{tmp_prefix}channel_context_signal",
        f"{tmp_prefix}semantic_signal",
        f"{tmp_prefix}head_prior_signal",
        f"{tmp_prefix}signal_count",
        f"{tmp_prefix}derived_reason",
        f"{tmp_prefix}derived_easy",
        f"{tmp_prefix}derived_hard",
        f"{tmp_prefix}unclear_non_actionable",
    )


def build_user_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    schema = (
        "user_id string, profile_text string, profile_text_evidence string, profile_text_short string, profile_text_long string, "
        "profile_top_pos_tags string, profile_top_neg_tags string, profile_top_pos_tags_by_type string, profile_top_neg_tags_by_type string, "
        "profile_confidence double, profile_pos_text string, profile_neg_text string, user_long_pref_text string, "
        "user_recent_intent_text string, user_negative_avoid_text string, user_context_text string"
    )
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    if not profile_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    pdf = (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "user_id",
            F.coalesce(F.col("profile_text_short"), F.col("profile_text"), F.lit("")).alias("profile_text"),
            F.coalesce(F.col("profile_text_long"), F.col("profile_text"), F.col("profile_text_short"), F.lit("")).alias("profile_text_evidence"),
            F.coalesce(F.col("profile_text_short"), F.col("profile_text"), F.lit("")).alias("profile_text_short"),
            F.coalesce(F.col("profile_text_long"), F.col("profile_text"), F.lit("")).alias("profile_text_long"),
            F.coalesce(F.col("profile_top_pos_tags"), F.lit("")).alias("profile_top_pos_tags"),
            F.coalesce(F.col("profile_top_neg_tags"), F.lit("")).alias("profile_top_neg_tags"),
            F.coalesce(F.col("profile_top_pos_tags_by_type"), F.lit("")).alias("profile_top_pos_tags_by_type"),
            F.coalesce(F.col("profile_top_neg_tags_by_type"), F.lit("")).alias("profile_top_neg_tags_by_type"),
            F.col("profile_confidence").cast("double").alias("profile_confidence"),
        )
        .dropDuplicates(["user_id"])
    )
    def _coalesce_pd(frame: pd.DataFrame, *cols: str) -> pd.Series:
        out = pd.Series([""] * len(frame), index=frame.index, dtype="object")
        for col in cols:
            if col not in frame.columns:
                continue
            series = frame[col].fillna("").astype(str)
            out = out.where(out.astype(str).str.len() > 0, series)
        return out.fillna("")

    try:
        raw_pdf = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
        if "user_id" in raw_pdf.columns:
            cooked_pdf = pd.DataFrame(
                {
                    "user_id": raw_pdf["user_id"].fillna("").astype(str),
                    "profile_text": _coalesce_pd(raw_pdf, "profile_text_short", "profile_text"),
                    "profile_text_evidence": _coalesce_pd(raw_pdf, "profile_text_long", "profile_text", "profile_text_short"),
                    "profile_text_short": _coalesce_pd(raw_pdf, "profile_text_short", "profile_text"),
                    "profile_text_long": _coalesce_pd(raw_pdf, "profile_text_long", "profile_text", "profile_text_short"),
                    "profile_top_pos_tags": _coalesce_pd(raw_pdf, "profile_top_pos_tags"),
                    "profile_top_neg_tags": _coalesce_pd(raw_pdf, "profile_top_neg_tags"),
                    "profile_top_pos_tags_by_type": _coalesce_pd(raw_pdf, "profile_top_pos_tags_by_type"),
                    "profile_top_neg_tags_by_type": _coalesce_pd(raw_pdf, "profile_top_neg_tags_by_type"),
                    "profile_confidence": pd.to_numeric(raw_pdf.get("profile_confidence", ""), errors="coerce"),
                }
            ).drop_duplicates(subset=["user_id"], keep="first")
            pdf = spark.createDataFrame(cooked_pdf, schema=schema).dropDuplicates(["user_id"])
    except Exception:
        pass
    try:
        clean_run = resolve_user_profile_text_run()
        clean_path = clean_run / "user_profile_multi_vector_texts.csv"
        if clean_path.exists():
            clean_df = (
                spark.read.option("header", "true").csv(clean_path.as_posix())
                .select(
                    "user_id",
                    F.coalesce(F.col("profile_short_text"), F.lit("")).alias("profile_pos_text_short_unused"),
                    F.coalesce(F.col("profile_short_text"), F.lit("")).alias("profile_clean_short_text"),
                    F.coalesce(F.col("profile_long_text"), F.lit("")).alias("profile_clean_long_text"),
                    F.coalesce(F.col("profile_pos_text"), F.lit("")).alias("profile_pos_text"),
                    F.coalesce(F.col("profile_neg_text"), F.lit("")).alias("profile_neg_text"),
                )
                .drop("profile_pos_text_short_unused")
                .dropDuplicates(["user_id"])
            )
            pdf = pdf.join(clean_df, on="user_id", how="left")
    except Exception:
        pass
    try:
        text_run = resolve_user_text_views_run()
        text_path = text_run / "user_text_views_v1.parquet"
        if text_path.exists():
            text_df = (
                spark.read.parquet(text_path.as_posix())
                .select(
                    "user_id",
                    F.coalesce(F.col("user_long_pref_text"), F.lit("")).alias("user_long_pref_text"),
                    F.coalesce(F.col("user_recent_intent_text"), F.lit("")).alias("user_recent_intent_text"),
                    F.coalesce(F.col("user_negative_avoid_text"), F.lit("")).alias("user_negative_avoid_text"),
                    F.coalesce(F.col("user_context_text"), F.lit("")).alias("user_context_text"),
                )
                .dropDuplicates(["user_id"])
            )
            pdf = pdf.join(text_df, on="user_id", how="left")
    except Exception:
        pass
    pdf = (
        pdf.fillna(
            {
                "profile_clean_short_text": "",
                "profile_clean_long_text": "",
                "profile_pos_text": "",
                "profile_neg_text": "",
                "user_long_pref_text": "",
                "user_recent_intent_text": "",
                "user_negative_avoid_text": "",
                "user_context_text": "",
            }
        )
        .withColumn("profile_text_short", F.coalesce(F.col("profile_clean_short_text"), F.col("profile_text_short"), F.col("profile_text"), F.lit("")))
        .withColumn("profile_text_long", F.coalesce(F.col("profile_clean_long_text"), F.col("profile_text_long"), F.col("profile_text_evidence"), F.lit("")))
        .drop("profile_clean_short_text", "profile_clean_long_text")
    )
    return pdf


def build_stage11_user_semantic_profile_df(spark: SparkSession) -> DataFrame:
    schema = (
        "bucket int, user_idx int, user_id string, stable_preferences_text string, recent_intent_text_v2 string, "
        "avoidance_text_v2 string, history_anchor_hint_text string, user_evidence_text_v2 string, user_semantic_profile_text_v2 string, "
        "user_profile_richness_v2 int, user_profile_richness_tier_v2 string, user_semantic_facts_v1 string, "
        "user_visible_fact_count_v1 int, user_focus_fact_count_v1 int, user_recent_fact_count_v1 int, "
        "user_avoid_fact_count_v1 int, user_history_fact_count_v1 int, user_evidence_fact_count_v1 int"
    )
    try:
        run_dir = resolve_stage11_semantic_text_assets_run()
    except Exception:
        return spark.createDataFrame([], schema)
    path = run_dir / "user_semantic_profile_text_v2.parquet"
    if not path.exists():
        return spark.createDataFrame([], schema)
    sdf = spark.read.parquet(path.as_posix())
    return (
        sdf.select(
            F.col("bucket").cast("int").alias("bucket"),
            F.col("user_idx").cast("int").alias("user_idx"),
            _optional_text_expr(sdf, "user_id"),
            _optional_text_expr(sdf, "stable_preferences_text"),
            _optional_text_expr(sdf, "recent_intent_text_v2"),
            _optional_text_expr(sdf, "avoidance_text_v2"),
            _optional_text_expr(sdf, "history_anchor_hint_text"),
            _optional_text_expr(sdf, "user_evidence_text_v2"),
            _optional_text_expr(sdf, "user_semantic_profile_text_v2"),
            _optional_int_expr(sdf, "user_profile_richness_v2"),
            _optional_text_expr(sdf, "user_profile_richness_tier_v2"),
            _optional_text_expr(sdf, "user_semantic_facts_v1"),
            _optional_int_expr(sdf, "user_visible_fact_count_v1"),
            _optional_int_expr(sdf, "user_focus_fact_count_v1"),
            _optional_int_expr(sdf, "user_recent_fact_count_v1"),
            _optional_int_expr(sdf, "user_avoid_fact_count_v1"),
            _optional_int_expr(sdf, "user_history_fact_count_v1"),
            _optional_int_expr(sdf, "user_evidence_fact_count_v1"),
        )
        .withColumn(
            "user_semantic_facts_v1",
            F.when(F.col("user_semantic_facts_v1") == "", F.lit("[]")).otherwise(F.col("user_semantic_facts_v1")),
        )
        .dropDuplicates(["bucket", "user_idx"])
    )


def build_stage11_merchant_semantic_profile_df(spark: SparkSession) -> DataFrame:
    schema = (
        "bucket int, business_id string, core_offering_text string, scene_fit_text string, strengths_text string, "
        "risk_points_text string, merchant_semantic_profile_text_v2 string, merchant_profile_richness_v2 int, "
        "merchant_semantic_facts_v1 string, merchant_visible_fact_count_v1 int, merchant_core_fact_count_v1 int, "
        "merchant_scene_fact_count_v1 int, merchant_strength_fact_count_v1 int, merchant_risk_fact_count_v1 int"
    )
    try:
        run_dir = resolve_stage11_semantic_text_assets_run()
    except Exception:
        return spark.createDataFrame([], schema)
    path = run_dir / "merchant_semantic_profile_text_v2.parquet"
    if not path.exists():
        return spark.createDataFrame([], schema)
    sdf = spark.read.parquet(path.as_posix())
    return (
        sdf.select(
            F.col("bucket").cast("int").alias("bucket"),
            _optional_text_expr(sdf, "business_id"),
            _optional_text_expr(sdf, "core_offering_text"),
            _optional_text_expr(sdf, "scene_fit_text"),
            _optional_text_expr(sdf, "strengths_text"),
            _optional_text_expr(sdf, "risk_points_text"),
            _optional_text_expr(sdf, "merchant_semantic_profile_text_v2"),
            _optional_int_expr(sdf, "merchant_profile_richness_v2"),
            _optional_text_expr(sdf, "merchant_semantic_facts_v1"),
            _optional_int_expr(sdf, "merchant_visible_fact_count_v1"),
            _optional_int_expr(sdf, "merchant_core_fact_count_v1"),
            _optional_int_expr(sdf, "merchant_scene_fact_count_v1"),
            _optional_int_expr(sdf, "merchant_strength_fact_count_v1"),
            _optional_int_expr(sdf, "merchant_risk_fact_count_v1"),
        )
        .withColumn(
            "merchant_semantic_facts_v1",
            F.when(F.col("merchant_semantic_facts_v1") == "", F.lit("[]")).otherwise(F.col("merchant_semantic_facts_v1")),
        )
        .dropDuplicates(["bucket", "business_id"])
    )


def build_stage11_user_business_alignment_df(spark: SparkSession) -> DataFrame:
    schema = (
        "bucket int, user_idx int, item_idx int, user_id string, business_id string, fit_reasons_text_v1 string, "
        "friction_reasons_text_v1 string, evidence_basis_text_v1 string, pair_alignment_richness_v1 int, "
        "semantic_prompt_readiness_tier_v1 string, pair_alignment_facts_v1 string, pair_fit_fact_count_v1 int, "
        "pair_friction_fact_count_v1 int, pair_evidence_fact_count_v1 int, pair_stable_fit_fact_count_v1 int, "
        "pair_recent_fit_fact_count_v1 int, pair_history_fit_fact_count_v1 int, pair_practical_fit_fact_count_v1 int, "
        "pair_fit_scope_count_v1 int, pair_has_visible_user_fit_v1 int, pair_has_visible_conflict_v1 int, "
        "pair_has_recent_context_visible_bridge_v1 int, pair_has_detail_support_v1 int, pair_has_multisource_fit_v1 int, "
        "pair_has_candidate_fit_support_v1 int, pair_has_candidate_strong_fit_support_v1 int, pair_has_contrastive_support_v1 int, "
        "pair_fact_signal_count_v1 int, boundary_constructability_class_v1 string, boundary_constructability_reason_codes_v1 string, "
        "boundary_prompt_ready_v1 int, boundary_rival_total_v1 int, boundary_rival_head_or_boundary_v1 int"
    )
    try:
        run_dir = resolve_stage11_semantic_text_assets_run()
    except Exception:
        return spark.createDataFrame([], schema)
    filtered_path = run_dir / "stage11_target_pair_assets_filtered_v1.parquet"
    path = filtered_path if filtered_path.exists() else (run_dir / "user_business_alignment_text_v1.parquet")
    if not path.exists():
        return spark.createDataFrame([], schema)
    sdf = spark.read.parquet(path.as_posix())
    return (
        sdf.select(
            F.col("bucket").cast("int").alias("bucket"),
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            _optional_text_expr(sdf, "user_id"),
            _optional_text_expr(sdf, "business_id"),
            _optional_text_expr(sdf, "fit_reasons_text_v1"),
            _optional_text_expr(sdf, "friction_reasons_text_v1"),
            _optional_text_expr(sdf, "evidence_basis_text_v1"),
            _optional_int_expr(sdf, "pair_alignment_richness_v1"),
            _optional_text_expr(sdf, "semantic_prompt_readiness_tier_v1"),
            _optional_text_expr(sdf, "pair_alignment_facts_v1"),
            _optional_int_expr(sdf, "pair_fit_fact_count_v1"),
            _optional_int_expr(sdf, "pair_friction_fact_count_v1"),
            _optional_int_expr(sdf, "pair_evidence_fact_count_v1"),
            _optional_int_expr(sdf, "pair_stable_fit_fact_count_v1"),
            _optional_int_expr(sdf, "pair_recent_fit_fact_count_v1"),
            _optional_int_expr(sdf, "pair_history_fit_fact_count_v1"),
            _optional_int_expr(sdf, "pair_practical_fit_fact_count_v1"),
            _optional_int_expr(sdf, "pair_fit_scope_count_v1"),
            _optional_int_expr(sdf, "pair_has_visible_user_fit_v1"),
            _optional_int_expr(sdf, "pair_has_visible_conflict_v1"),
            _optional_int_expr(sdf, "pair_has_recent_context_visible_bridge_v1"),
            _optional_int_expr(sdf, "pair_has_detail_support_v1"),
            _optional_int_expr(sdf, "pair_has_multisource_fit_v1"),
            _optional_int_expr(sdf, "pair_has_candidate_fit_support_v1"),
            _optional_int_expr(sdf, "pair_has_candidate_strong_fit_support_v1"),
            _optional_int_expr(sdf, "pair_has_contrastive_support_v1"),
            _optional_int_expr(sdf, "pair_fact_signal_count_v1"),
            _optional_text_expr(sdf, "boundary_constructability_class_v1"),
            _optional_text_expr(sdf, "boundary_constructability_reason_codes_v1"),
            _optional_int_expr(sdf, "boundary_prompt_ready_v1"),
            _optional_int_expr(sdf, "boundary_rival_total_v1"),
            _optional_int_expr(sdf, "boundary_rival_head_or_boundary_v1"),
        )
        .withColumn(
            "pair_alignment_facts_v1",
            F.when(F.col("pair_alignment_facts_v1") == "", F.lit("[]")).otherwise(F.col("pair_alignment_facts_v1")),
        )
        .withColumn(
            "boundary_constructability_reason_codes_v1",
            F.when(F.col("boundary_constructability_reason_codes_v1") == "", F.lit("[]")).otherwise(F.col("boundary_constructability_reason_codes_v1")),
        )
        .dropDuplicates(["bucket", "user_idx", "item_idx"])
    )


def build_stage11_target_users_df(spark: SparkSession) -> DataFrame:
    schema = "bucket int, user_idx int, user_id string, truth_learned_rank int"
    try:
        run_dir = resolve_stage11_semantic_text_assets_run()
    except Exception:
        return spark.createDataFrame([], schema)
    filtered_path = run_dir / "stage11_target_users_filtered_v1.parquet"
    path = filtered_path if filtered_path.exists() else (run_dir / "stage11_target_users_v1.parquet")
    if not path.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.parquet(path.as_posix())
        .select(
            F.col("bucket").cast("int").alias("bucket"),
            F.col("user_idx").cast("int").alias("user_idx"),
            F.coalesce(F.col("user_id"), F.lit("")).alias("user_id"),
            F.coalesce(F.col("truth_learned_rank").cast("int"), F.lit(0)).alias("truth_learned_rank"),
        )
        .dropDuplicates(["bucket", "user_idx"])
    )


def build_item_sem_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    sem_path = str(stage09_meta.get("item_semantic_features", "")).strip()
    if not sem_path:
        return spark.createDataFrame([], "business_id string, top_pos_tags string, top_neg_tags string, semantic_score double, semantic_confidence double")
    p = resolve_stage09_meta_path(sem_path)
    if not p.exists():
        return spark.createDataFrame([], "business_id string, top_pos_tags string, top_neg_tags string, semantic_score double, semantic_confidence double")
    sdf = (
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
    return sdf


def build_cluster_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("cluster_profile_csv", "")).strip()
    schema = (
        "business_id string, "
        "cluster_for_recsys string, "
        "cluster_label_for_recsys string, "
        "final_l1_label string, "
        "final_l2_label_top1 string"
    )
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
            F.coalesce(F.col("final_l1_label"), F.lit("")).alias("final_l1_label"),
            F.coalesce(F.col("final_l2_label_top1"), F.lit("")).alias("final_l2_label_top1"),
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


def maybe_cap_users(df_truth: DataFrame, max_users: int) -> DataFrame:
    if int(max_users) <= 0:
        return df_truth
    users = df_truth.select("user_idx").dropDuplicates(["user_idx"])
    if USER_CAP_RANDOMIZED:
        # Slow path kept for parity fallback in case strict random cap is required.
        sample_users = users.orderBy(F.rand(SEED)).limit(int(max_users))
    else:
        # Fast path: avoid global random sort on large user sets.
        sample_users = users.limit(int(max_users))
    return df_truth.join(sample_users, on="user_idx", how="inner")


def resolve_neg_quotas(total_neg: int) -> dict[str, int]:
    total = max(0, int(total_neg))
    if total <= 0:
        return {"hard": 0, "near": 0, "easy": 0}

    hard = int(round(float(total) * max(0.0, float(NEG_HARD_RATIO))))
    near = int(round(float(total) * max(0.0, float(NEG_NEAR_RATIO))))
    hard = max(1, min(total, hard))
    near = max(0, min(total - hard, near))
    easy = max(0, total - hard - near)
    return {"hard": int(hard), "near": int(near), "easy": int(easy)}


def resolve_banded_neg_quotas(total_neg: int) -> dict[str, int]:
    total = max(0, int(total_neg))
    if total <= 0:
        return {"top10": 0, "11_30": 0, "31_80": 0, "81_150": 0, "near": 0}
    requested = {
        "top10": max(0, int(NEG_BAND_TOP10)),
        "11_30": max(0, int(NEG_BAND_11_30)),
        "31_80": max(0, int(NEG_BAND_31_80)),
        "81_150": max(0, int(NEG_BAND_81_150)),
        "near": max(0, int(NEG_BAND_NEAR)),
    }
    if sum(requested.values()) <= total:
        return requested
    out = {k: 0 for k in requested}
    remain = total
    for key in ("top10", "11_30", "31_80", "81_150", "near"):
        if remain <= 0:
            break
        take = min(int(requested[key]), int(remain))
        out[key] = int(take)
        remain -= int(take)
    return out


def enforce_stage09_gate(source_09: Path, buckets: list[int]) -> dict[str, dict[str, float]]:
    if not ENFORCE_STAGE09_GATE:
        return {}
    try:
        semantic_run_dir = resolve_stage11_semantic_text_assets_run()
    except Exception:
        semantic_run_dir = None

    if semantic_run_dir is not None:
        semantic_meta_path = semantic_run_dir / "run_meta.json"
        filtered_users_path = semantic_run_dir / "stage11_target_users_filtered_v1.parquet"
        readiness_path = semantic_run_dir / "stage11_target_user_readiness_v1.parquet"
        if semantic_meta_path.exists() and filtered_users_path.exists():
            semantic_meta = json.loads(semantic_meta_path.read_text(encoding="utf-8"))
            semantic_source_raw = str(semantic_meta.get("source_stage09_run", "")).strip()
            if semantic_source_raw:
                semantic_source = normalize_legacy_project_path(semantic_source_raw)
                src_key = source_09.name
                semantic_key = semantic_source.name
                same_source = (
                    semantic_source == source_09
                    or str(semantic_source).endswith(str(source_09))
                    or str(source_09).endswith(str(semantic_source))
                    or semantic_key == src_key
                )
                if not same_source:
                    raise RuntimeError(
                        "stage09 semantic gate source mismatch: "
                        f"source={source_09} semantic_source={semantic_source} run={semantic_run_dir}"
                    )

            filtered_pdf = pd.read_parquet(filtered_users_path, columns=["bucket", "user_idx"])
            if filtered_pdf.empty:
                raise RuntimeError(f"stage09 semantic gate filtered users empty: {filtered_users_path}")
            filtered_pdf["bucket"] = pd.to_numeric(filtered_pdf["bucket"], errors="coerce").astype("Int64")
            filtered_pdf["user_idx"] = pd.to_numeric(filtered_pdf["user_idx"], errors="coerce").astype("Int64")
            filtered_pdf = filtered_pdf.dropna(subset=["bucket", "user_idx"]).drop_duplicates()

            readiness_pdf = None
            if readiness_path.exists():
                try:
                    readiness_pdf = pd.read_parquet(
                        readiness_path,
                        columns=["bucket", "user_idx", "truth_learned_rank", "boundary_prompt_ready_v1"],
                    )
                    if not readiness_pdf.empty:
                        readiness_pdf["bucket"] = pd.to_numeric(readiness_pdf["bucket"], errors="coerce").astype("Int64")
                        readiness_pdf["user_idx"] = pd.to_numeric(readiness_pdf["user_idx"], errors="coerce").astype("Int64")
                        readiness_pdf["truth_learned_rank"] = pd.to_numeric(
                            readiness_pdf.get("truth_learned_rank"), errors="coerce"
                        ).astype("Int64")
                        readiness_pdf["boundary_prompt_ready_v1"] = pd.to_numeric(
                            readiness_pdf.get("boundary_prompt_ready_v1"), errors="coerce"
                        ).fillna(0).astype(int)
                except Exception:
                    readiness_pdf = None

            out: dict[str, dict[str, float]] = {}
            for b in sorted(list(set([int(x) for x in buckets]))):
                bucket_pdf = filtered_pdf.loc[filtered_pdf["bucket"] == int(b)]
                bucket_user_count = int(bucket_pdf["user_idx"].nunique())
                if bucket_user_count <= 0:
                    raise RuntimeError(
                        f"stage09 semantic gate filtered users missing bucket={b} in {filtered_users_path}"
                    )
                metrics: dict[str, float] = {"filtered_users": float(bucket_user_count)}
                if readiness_pdf is not None and not readiness_pdf.empty:
                    ready_bucket = readiness_pdf.loc[readiness_pdf["bucket"] == int(b)]
                    if not ready_bucket.empty:
                        metrics["boundary_prompt_ready_rate"] = float(
                            ready_bucket["boundary_prompt_ready_v1"].astype(int).gt(0).mean()
                        )
                        boundary_rows = ready_bucket.loc[
                            ready_bucket["truth_learned_rank"].between(11, 30, inclusive="both")
                        ]
                        if not boundary_rows.empty:
                            metrics["boundary_11_30_prompt_ready_rate"] = float(
                                boundary_rows["boundary_prompt_ready_v1"].astype(int).gt(0).mean()
                            )
                out[str(b)] = metrics
            print(
                f"[GATE] stage09 semantic assets pass run={semantic_run_dir.name} "
                f"source={source_09.name} buckets={sorted(list(out.keys()))}"
            )
            return out

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


def build_user_evidence_udf() -> Any:
    def _mk(
        profile_text: Any,
        profile_text_evidence: Any,
        profile_pos_text: Any,
        profile_neg_text: Any,
        user_long_pref_text: Any,
        user_recent_intent_text: Any,
        user_negative_avoid_text: Any,
        user_context_text: Any,
    ) -> str:
        return extract_user_evidence_text(
            profile_text,
            profile_text_evidence,
            profile_pos_text=profile_pos_text,
            profile_neg_text=profile_neg_text,
            user_long_pref_text=user_long_pref_text,
            user_recent_intent_text=user_recent_intent_text,
            user_negative_avoid_text=user_negative_avoid_text,
            user_context_text=user_context_text,
            max_chars=USER_EVIDENCE_MAX_CHARS,
        )

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
        return build_history_anchor_summary(
            " || ".join(parts),
            max_items=max(1, int(RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER)),
            max_chars=max(80, int(RICH_SFT_HISTORY_ANCHOR_MAX_CHARS)),
        )

    return F.udf(_mk, "string")


def build_item_review_evidence(
    spark: SparkSession,
    bucket_dir: Path,
    row_df: DataFrame,
    item_sem_df: DataFrame,
) -> DataFrame:
    schema = "user_idx int, business_id string, item_evidence_text string"
    empty_df = spark.createDataFrame([], schema)
    if not ENABLE_RAW_REVIEW_TEXT:
        return empty_df
    if not REVIEW_TABLE_PATH.exists():
        print(f"[WARN] review table not found, skip item evidence features: {REVIEW_TABLE_PATH}")
        return empty_df
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
    rvw = (
        spark.read.parquet(REVIEW_TABLE_PATH.as_posix())
        .select("business_id", "date", "text")
        .withColumn("review_ts", F.to_timestamp(F.col("date")))
        .withColumn("text_clean", F.regexp_replace(F.regexp_replace(F.coalesce(F.col("text"), F.lit("")), r"[\r\n]+", " "), r"\s+", " "))
        .filter(F.col("review_ts").isNotNull() & (F.length(F.col("text_clean")) > F.lit(0)))
    )
    max_chars = max(40, int(REVIEW_SNIPPET_MAX_CHARS))
    if ITEM_EVIDENCE_SCORE_UDF_MODE == "pandas":
        @F.pandas_udf("double")
        def score_udf(txt: pd.Series, pos: pd.Series, neg: pd.Series) -> pd.Series:
            return pd.Series(
                [float(keyword_match_score(t, p, n)) for t, p, n in zip(txt, pos, neg)],
                dtype="float64",
            )
    else:
        score_udf = F.udf(lambda txt, pos, neg: float(keyword_match_score(txt, pos, neg)), "double")
    cand_business = row_df.select("business_id").dropDuplicates(["business_id"])
    rvw = rvw.join(F.broadcast(cand_business), on="business_id", how="semi")
    item_base = (
        row_df.select("user_idx", "business_id")
        .dropDuplicates(["user_idx", "business_id"])
        .join(user_cutoff, on="user_idx", how="inner")
        .join(item_sem_df.select("business_id", "top_pos_tags", "top_neg_tags"), on="business_id", how="left")
    )
    item_rows = (
        item_base.join(rvw, on="business_id", how="inner")
        .filter(F.col("review_ts") < F.col("test_ts"))
        .withColumn("snippet", F.substring(F.col("text_clean"), 1, int(max_chars)))
        .withColumn("tag_hit_score", score_udf(F.col("snippet"), F.col("top_pos_tags"), F.col("top_neg_tags")))
        .withColumn("snippet_len", F.length(F.col("snippet")).cast("double"))
        .withColumn(
            "snippet_score",
            F.col("tag_hit_score") * F.lit(10.0)
            + F.when(F.col("snippet_len") >= F.lit(80.0), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .dropDuplicates(["user_idx", "business_id", "snippet"])
    )
    w_item = Window.partitionBy("user_idx", "business_id").orderBy(F.col("snippet_score").desc(), F.col("review_ts").desc())
    item_top = item_rows.withColumn("rn", F.row_number().over(w_item)).filter(F.col("rn") <= F.lit(max(1, int(ITEM_REVIEW_TOPN))))
    return (
        item_top.groupBy("user_idx", "business_id")
        .agg(F.concat_ws(" || ", F.collect_list("snippet")).alias("item_evidence_text"))
        .select("user_idx", "business_id", "item_evidence_text")
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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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
    """Prompt UDF that drops ranking-position features for DPO training."""
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
    """Single-candidate scoring prompt for reward-model training/eval."""

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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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
    """Single-candidate scoring prompt with compact structured hints."""

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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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
    """Single-candidate scoring prompt with compact structured hints + preserve hints."""

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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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
    """Single-candidate scoring prompt with compact structured hints + rank-aware targeted hints."""

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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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


def build_prompt_udf_semantic_compact_boundary_rm() -> Any:
    """Single-candidate scoring prompt focused on xgbblend boundary rescue bands."""

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
        learned_rank: Any,
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
            build_item_text_semantic_compact_boundary(
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
                learned_rank=learned_rank,
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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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
                profile_text=profile_text,
                top_pos_tags=profile_top_pos_tags,
                top_neg_tags=profile_top_neg_tags,
                confidence=profile_confidence,
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


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    est_max_rows = int(MAX_ROWS_PER_BUCKET) if int(MAX_ROWS_PER_BUCKET) > 0 else -1
    if est_max_rows > 150000:
        print(
            f"[WARN] QLORA_MAX_ROWS_PER_BUCKET={est_max_rows} may cause slow/unstable prompt generation. "
            "For faster local iteration, prefer <=120000."
        )
    if int(MAX_ROWS_PER_BUCKET) > 0 and ROW_CAP_ORDERED:
        print("[WARN] QLORA_ROW_CAP_ORDERED=true uses global orderBy before limit and can be slower.")
    if NEG_LAYERED_ENABLED:
        if NEG_SAMPLER_MODE == "banded":
            print(
                "[CONFIG] neg_sampler=banded "
                f"top10={NEG_BAND_TOP10} band11_30={NEG_BAND_11_30} band31_80={NEG_BAND_31_80} "
                f"band81_150={NEG_BAND_81_150} near={NEG_BAND_NEAR}"
            )
        else:
            print(
                "[CONFIG] neg_sampler=layered "
                f"hard_ratio={NEG_HARD_RATIO:.2f} near_ratio={NEG_NEAR_RATIO:.2f} hard_rank_max={NEG_HARD_RANK_MAX}"
            )

    source_09 = resolve_stage09_run()
    run_meta_path = source_09 / "run_meta.json"
    if not run_meta_path.exists():
        raise FileNotFoundError(f"missing run_meta.json: {run_meta_path}")
    meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    dataset_eval_user_cohort_path = resolve_dataset_eval_user_cohort_path()
    fixed_eval_users_df: DataFrame | None = None
    fixed_eval_user_count = 0
    fixed_eval_join_cols: list[str] = []
    if dataset_eval_user_cohort_path is not None:
        fixed_eval_users_df, fixed_eval_user_count, fixed_eval_available_cols = load_eval_users_from_cohort(
            spark,
            dataset_eval_user_cohort_path,
        )
        fixed_eval_join_cols = ["user_id"] if "user_id" in fixed_eval_available_cols else ["user_idx"]
        print(
            "[COHORT] dataset explicit eval cohort "
            f"path={dataset_eval_user_cohort_path} users={fixed_eval_user_count} join_cols={fixed_eval_join_cols}"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = parse_bucket_override(BUCKETS_OVERRIDE)
    bucket_dirs = sorted([p for p in source_09.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    if wanted:
        bucket_dirs = [p for p in bucket_dirs if int(p.name.split("_")[-1]) in wanted]
    if not bucket_dirs:
        raise RuntimeError("no bucket dirs to process")
    processed_buckets = [int(p.name.split("_")[-1]) for p in bucket_dirs]
    gate_result = enforce_stage09_gate(source_09, processed_buckets)

    user_profile_df = build_user_profile_df(spark, meta)
    stage11_user_semantic_profile_df = build_stage11_user_semantic_profile_df(spark)
    stage11_merchant_semantic_profile_df = build_stage11_merchant_semantic_profile_df(spark)
    stage11_user_business_alignment_df = build_stage11_user_business_alignment_df(spark)
    stage11_target_users_df = build_stage11_target_users_df(spark)
    try:
        _stage11_semantic_run_dir = resolve_stage11_semantic_text_assets_run()
        stage11_target_users_filtered_asset_enabled = (
            _stage11_semantic_run_dir / "stage11_target_users_filtered_v1.parquet"
        ).exists()
        stage11_target_users_asset_enabled = stage11_target_users_filtered_asset_enabled or (
            _stage11_semantic_run_dir / "stage11_target_users_v1.parquet"
        ).exists()
        if stage11_target_users_filtered_asset_enabled:
            print("[CONFIG] stage11 target users scope=filtered 11-100 surviving users")
        elif stage11_target_users_asset_enabled:
            print("[CONFIG] stage11 target users scope=raw 11-100 target users")
    except Exception:
        stage11_target_users_asset_enabled = False
    item_sem_df = build_item_sem_df(spark, meta)
    cluster_profile_df = build_cluster_profile_df(spark, meta)
    business_meta_df = build_business_meta_df(spark, meta)
    history_anchor_text_udf = build_history_anchor_text_udf()
    rich_prompt_udf = None
    prompt_udf = None
    if AUDIT_ONLY:
        print("[CONFIG] QLORA_AUDIT_ONLY=true (skip prompt generation and dataset writes; collect summary stats only)")
        print(f"[CONFIG] PROMPT_MODE={PROMPT_MODE} (declared, audit-only path does not materialize prompts)")
    else:
        if ENABLE_RICH_SFT_EXPORT:
            if PROMPT_MODE == "semantic":
                rich_prompt_udf = build_prompt_udf_semantic()
            elif PROMPT_MODE == "semantic_rm":
                rich_prompt_udf = build_prompt_udf_semantic_rm()
            elif PROMPT_MODE == "semantic_compact_rm":
                rich_prompt_udf = build_prompt_udf_semantic_compact_rm()
            elif PROMPT_MODE == "semantic_compact_preserve_rm":
                rich_prompt_udf = build_prompt_udf_semantic_compact_preserve_rm()
            elif PROMPT_MODE == "semantic_compact_targeted_rm":
                rich_prompt_udf = build_prompt_udf_semantic_compact_targeted_rm()
            elif PROMPT_MODE == "semantic_compact_boundary_rm":
                rich_prompt_udf = build_prompt_udf_semantic_compact_boundary_rm()
            elif PROMPT_MODE == "sft_clean":
                rich_prompt_udf = build_prompt_udf_sft_clean()
            elif PROMPT_MODE == "full_lite":
                rich_prompt_udf = build_prompt_udf_full_lite()
            else:
                rich_prompt_udf = build_prompt_udf()
        if PROMPT_MODE == "semantic":
            prompt_udf = build_prompt_udf_semantic()
            print(f"[CONFIG] PROMPT_MODE=semantic (ranking features removed from prompt)")
        elif PROMPT_MODE == "semantic_rm":
            prompt_udf = build_prompt_udf_semantic_rm()
            print("[CONFIG] PROMPT_MODE=semantic_rm (single-candidate reward scoring prompt)")
        elif PROMPT_MODE == "semantic_compact_rm":
            prompt_udf = build_prompt_udf_semantic_compact_rm()
            print("[CONFIG] PROMPT_MODE=semantic_compact_rm (semantic prompt + compact stage09/10 hints)")
        elif PROMPT_MODE == "semantic_compact_preserve_rm":
            prompt_udf = build_prompt_udf_semantic_compact_preserve_rm()
            print(
                "[CONFIG] PROMPT_MODE=semantic_compact_preserve_rm "
                "(semantic_compact_rm + preserve hints from route/support/stable-fit signals)"
            )
        elif PROMPT_MODE == "semantic_compact_targeted_rm":
            prompt_udf = build_prompt_udf_semantic_compact_targeted_rm()
            print(
                "[CONFIG] PROMPT_MODE=semantic_compact_targeted_rm "
                "(semantic_compact_rm + rank-aware route/support/stability profiles)"
            )
        elif PROMPT_MODE == "semantic_compact_boundary_rm":
            prompt_udf = build_prompt_udf_semantic_compact_boundary_rm()
            print(
                "[CONFIG] PROMPT_MODE=semantic_compact_boundary_rm "
                "(semantic_compact_rm + xgbblend boundary rescue band hints)"
            )
        elif PROMPT_MODE == "full_lite":
            prompt_udf = build_prompt_udf_full_lite()
            print("[CONFIG] PROMPT_MODE=full_lite (history anchors + selected model signals, no exact rank leakage)")
        elif PROMPT_MODE == "sft_clean":
            prompt_udf = build_prompt_udf_sft_clean()
            print("[CONFIG] PROMPT_MODE=sft_clean (history anchors + semantic text, no rank/source leakage)")
        else:
            prompt_udf = build_prompt_udf()
            print(f"[CONFIG] PROMPT_MODE=full (all features in prompt)")

    summary: list[dict[str, Any]] = []

    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        hist_path = bdir / "train_history.parquet"
        cand_file = choose_candidate_file(bdir)
        truth = (
            spark.read.parquet((bdir / "truth.parquet").as_posix())
            .select("user_idx", "user_id", "true_item_idx", "valid_item_idx")
            .dropDuplicates(["user_idx"])
        )
        truth = maybe_cap_users(truth, MAX_USERS_PER_BUCKET).cache()

        users = truth.select("user_idx").dropDuplicates(["user_idx"])
        cand_raw = (
            spark.read.parquet(cand_file.as_posix())
            .join(users, on="user_idx", how="inner")
            .filter(F.col("pre_rank") <= F.lit(int(TOPN_PER_USER)))
        )
        cand_raw = attach_stage10_text_match_features(spark, cand_raw)
        cand_raw = attach_stage10_match_channel_features(spark, cand_raw, truth)
        cand_raw = attach_stage10_group_gap_features(spark, cand_raw)
        cand_raw = attach_stage10_learned_features(spark, cand_raw, bucket)
        cand_raw = attach_stage11_rescue_reason_features(spark, cand_raw)
        cand = project_candidate_rows(cand_raw)
        cand = maybe_checkpoint_df(cand, f"bucket={bucket} candidate_rows")

        pos_true = (
            truth.select(
                "user_idx",
                "user_id",
                F.col("true_item_idx").alias("item_idx"),
                F.lit("true").alias("label_source"),
                F.lit(1).alias("label"),
                F.lit(1.0).alias("sample_weight"),
            )
            .filter(F.col("item_idx").isNotNull())
            .dropDuplicates(["user_idx", "item_idx"])
        )
        pos_true_in_topn_users = (
            pos_true.join(cand.select("user_idx", "item_idx"), on=["user_idx", "item_idx"], how="inner")
            .select("user_idx")
            .dropDuplicates(["user_idx"])
        )
        pos_all = pos_true
        if INCLUDE_VALID_POS:
            pos_valid = (
                truth.select(
                    "user_idx",
                    "user_id",
                    F.col("valid_item_idx").alias("item_idx"),
                    F.lit("valid").alias("label_source"),
                    F.lit(1).alias("label"),
                    F.lit(float(VALID_POS_WEIGHT)).alias("sample_weight"),
                )
                .filter(F.col("item_idx").isNotNull())
                .dropDuplicates(["user_idx", "item_idx"])
            )
            if VALID_POS_ONLY_IF_NO_TRUE:
                pos_valid = pos_valid.join(pos_true_in_topn_users, on="user_idx", how="left_anti")
            pos_all = pos_all.unionByName(pos_valid)

        # ── hist_pos: add high-rating train_history items as extra positives ──
        if INCLUDE_HIST_POS:
            if hist_path.exists():
                w_hist_pos = Window.partitionBy("user_idx").orderBy(
                    F.col("hist_rating").desc(), F.col("hist_ts").desc()
                )
                hist_pos = (
                    spark.read.parquet(hist_path.as_posix())
                    .filter(
                        F.col("hist_rating").isNotNull()
                        & (F.col("hist_rating").cast("double") >= F.lit(float(HIST_POS_MIN_RATING)))
                    )
                    .withColumn("_hp_rn", F.row_number().over(w_hist_pos))
                    .filter(F.col("_hp_rn") <= F.lit(int(HIST_POS_MAX_PER_USER)))
                    .drop("_hp_rn")
                    .select(
                        "user_idx",
                        F.col("item_idx"),
                    )
                    .join(
                        truth.select("user_idx", "user_id"),
                        on="user_idx",
                        how="inner",
                    )
                    .join(
                        pos_all.select("user_idx", "item_idx"),
                        on=["user_idx", "item_idx"],
                        how="left_anti",
                    )
                    .select(
                        "user_idx",
                        "user_id",
                        "item_idx",
                        F.lit("hist_pos").alias("label_source"),
                        F.lit(1).alias("label"),
                        F.lit(float(HIST_POS_WEIGHT)).alias("sample_weight"),
                    )
                    .dropDuplicates(["user_idx", "item_idx"])
                )
                pos_all = pos_all.unionByName(hist_pos)
                print(f"[DATA] bucket={bucket} hist_pos added (min_rating={HIST_POS_MIN_RATING}, max_per_user={HIST_POS_MAX_PER_USER})")
            else:
                print(f"[WARN] bucket={bucket} train_history.parquet missing, skip hist_pos")

        pos_in_topn = pos_all.join(cand.select("user_idx", "item_idx"), on=["user_idx", "item_idx"], how="inner")

        pos_rows = (
            pos_in_topn.join(cand, on=["user_idx", "item_idx"], how="inner")
            .withColumn("neg_tier", F.lit("pos"))
            .withColumn("neg_pick_rank", F.lit(0).cast("int"))
            .withColumn("neg_is_near", F.lit(False))
            .withColumn("neg_is_hard", F.lit(False))
        )

        neg_base = cand.join(pos_in_topn.select("user_idx", "item_idx"), on=["user_idx", "item_idx"], how="left_anti")
        if NEG_LAYERED_ENABLED:
            pos_ctx = (
                pos_rows.select(
                    "user_idx",
                    F.lower(F.trim(F.coalesce(F.col("city"), F.lit("")))).alias("pos_city_norm"),
                    F.lower(F.trim(F.coalesce(F.col("primary_category"), F.lit("")))).alias("pos_cat_norm"),
                    F.col("pre_score").cast("double").alias("pos_pre_score"),
                )
                .groupBy("user_idx")
                .agg(
                    F.collect_set("pos_city_norm").alias("pos_city_set"),
                    F.collect_set("pos_cat_norm").alias("pos_cat_set"),
                    F.max("pos_pre_score").alias("pos_pre_score_max"),
                )
            )

            neg_scored = (
                neg_base.join(pos_ctx, on="user_idx", how="left")
                .withColumn("city_norm", F.lower(F.trim(F.coalesce(F.col("city"), F.lit("")))))
                .withColumn("cat_norm", F.lower(F.trim(F.coalesce(F.col("primary_category"), F.lit("")))))
                .withColumn(
                    "neg_is_near_city",
                    F.coalesce(F.array_contains(F.col("pos_city_set"), F.col("city_norm")), F.lit(False)),
                )
                .withColumn(
                    "neg_is_near_cat",
                    F.coalesce(F.array_contains(F.col("pos_cat_set"), F.col("cat_norm")), F.lit(False)),
                )
                .withColumn("neg_is_near", F.col("neg_is_near_city") | F.col("neg_is_near_cat"))
                .withColumn("pos_pre_score_max", F.coalesce(F.col("pos_pre_score_max"), F.lit(0.0)))
                .withColumn(
                    "score_gap_to_pos",
                    F.abs(F.col("pre_score").cast("double") - F.col("pos_pre_score_max")),
                )
            )
            neg_template = (
                neg_scored.withColumn("label_source", F.lit(""))
                .withColumn("label", F.lit(0))
                .withColumn("sample_weight", F.lit(1.0))
                .withColumn("neg_tier", F.lit(""))
                .withColumn("neg_pick_rank", F.lit(0).cast("int"))
                .withColumn("neg_is_hard", F.lit(False))
                .limit(0)
            )

            if NEG_SAMPLER_MODE == "banded":
                band_quota = resolve_banded_neg_quotas(NEG_PER_USER)
                picked_ui = neg_template.select("user_idx", "item_idx")

                def _pick_band(
                    pool_df: DataFrame,
                    quota: int,
                    label_source: str,
                    sample_weight: float,
                    neg_tier: str,
                    *,
                    is_hard: bool = False,
                    order_desc: bool = False,
                ) -> DataFrame:
                    if int(quota) <= 0:
                        return neg_template
                    ranked = (
                        pool_df.withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(
                                    F.col("pre_rank").desc() if order_desc else F.col("pre_rank").asc(),
                                    F.col("pre_score").asc() if order_desc else F.col("pre_score").desc(),
                                )
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(quota)))
                        .withColumn("label_source", F.lit(label_source))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(sample_weight)))
                        .withColumn("neg_tier", F.lit(neg_tier))
                        .withColumn("neg_is_hard", F.lit(bool(is_hard)))
                    )
                    return ranked

                neg_top10 = _pick_band(
                    neg_scored.filter(F.col("pre_rank") <= F.lit(10)),
                    int(band_quota["top10"]),
                    "neg_band_top10",
                    float(NEG_HARD_WEIGHT),
                    "band_top10",
                    is_hard=True,
                )
                picked_ui = neg_top10.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"])

                neg_11_30 = _pick_band(
                    neg_scored.filter((F.col("pre_rank") >= F.lit(11)) & (F.col("pre_rank") <= F.lit(30))).join(
                        picked_ui, on=["user_idx", "item_idx"], how="left_anti"
                    ),
                    int(band_quota["11_30"]),
                    "neg_band_11_30",
                    float(NEG_FILL_WEIGHT),
                    "band_11_30",
                )
                picked_ui = picked_ui.unionByName(
                    neg_11_30.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])

                neg_31_80 = _pick_band(
                    neg_scored.filter((F.col("pre_rank") >= F.lit(31)) & (F.col("pre_rank") <= F.lit(80))).join(
                        picked_ui, on=["user_idx", "item_idx"], how="left_anti"
                    ),
                    int(band_quota["31_80"]),
                    "neg_band_31_80",
                    float(NEG_FILL_WEIGHT),
                    "band_31_80",
                )
                picked_ui = picked_ui.unionByName(
                    neg_31_80.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])

                neg_81_150 = _pick_band(
                    neg_scored.filter(F.col("pre_rank") >= F.lit(81)).join(
                        picked_ui, on=["user_idx", "item_idx"], how="left_anti"
                    ),
                    int(band_quota["81_150"]),
                    "neg_band_81_150",
                    float(NEG_EASY_WEIGHT),
                    "band_81_150",
                )
                picked_ui = picked_ui.unionByName(
                    neg_81_150.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])

                neg_near = neg_template
                if int(band_quota["near"]) > 0:
                    near_pool = neg_scored.filter(F.col("neg_is_near")).join(
                        picked_ui,
                        on=["user_idx", "item_idx"],
                        how="left_anti",
                    )
                    neg_near = (
                        near_pool.withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(
                                    F.col("score_gap_to_pos").asc(),
                                    F.col("pre_rank").asc(),
                                )
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(band_quota["near"])))
                        .withColumn("label_source", F.lit("neg_near_ctx"))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(NEG_NEAR_WEIGHT)))
                        .withColumn("neg_tier", F.lit("near"))
                        .withColumn("neg_is_hard", F.lit(False))
                    )
                    picked_ui = picked_ui.unionByName(
                        neg_near.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                        allowMissingColumns=False,
                    ).dropDuplicates(["user_idx", "item_idx"])

                neg_selected = (
                    neg_top10.unionByName(neg_11_30, allowMissingColumns=True)
                    .unionByName(neg_31_80, allowMissingColumns=True)
                    .unionByName(neg_81_150, allowMissingColumns=True)
                    .unionByName(neg_near, allowMissingColumns=True)
                    .dropDuplicates(["user_idx", "item_idx"])
                )
            else:
                neg_quota = resolve_neg_quotas(NEG_PER_USER)
                hard_quota = int(neg_quota["hard"])
                near_quota = int(neg_quota["near"])
                easy_quota = int(neg_quota["easy"])

                neg_hard = neg_template
                if hard_quota > 0:
                    neg_hard = (
                        neg_scored.filter(F.col("pre_rank") <= F.lit(int(max(1, NEG_HARD_RANK_MAX))))
                        .withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(F.col("pre_rank").asc(), F.col("pre_score").desc())
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(hard_quota)))
                        .withColumn("label_source", F.lit("neg_hard_top"))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(NEG_HARD_WEIGHT)))
                        .withColumn("neg_tier", F.lit("hard"))
                        .withColumn("neg_is_hard", F.lit(True))
                    )

                picked_ui = neg_hard.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"])

                neg_near = neg_template
                if near_quota > 0:
                    near_pool = neg_scored.filter(F.col("neg_is_near")).join(
                        picked_ui,
                        on=["user_idx", "item_idx"],
                        how="left_anti",
                    )
                    neg_near = (
                        near_pool.withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(F.col("score_gap_to_pos").asc(), F.col("pre_rank").asc())
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(near_quota)))
                        .withColumn("label_source", F.lit("neg_near_ctx"))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(NEG_NEAR_WEIGHT)))
                        .withColumn("neg_tier", F.lit("near"))
                        .withColumn("neg_is_hard", F.lit(False))
                    )
                    picked_ui = picked_ui.unionByName(
                        neg_near.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                        allowMissingColumns=False,
                    ).dropDuplicates(["user_idx", "item_idx"])

                neg_easy = neg_template
                if easy_quota > 0:
                    easy_pool = neg_scored.join(picked_ui, on=["user_idx", "item_idx"], how="left_anti")
                    neg_easy = (
                        easy_pool.withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(F.col("pre_rank").desc(), F.col("pre_score").asc())
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(easy_quota)))
                        .withColumn("label_source", F.lit("neg_easy_tail"))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(NEG_EASY_WEIGHT)))
                        .withColumn("neg_tier", F.lit("easy"))
                        .withColumn("neg_is_hard", F.lit(False))
                    )
                    picked_ui = picked_ui.unionByName(
                        neg_easy.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                        allowMissingColumns=False,
                    ).dropDuplicates(["user_idx", "item_idx"])

                neg_selected = (
                    neg_hard.unionByName(neg_near, allowMissingColumns=True)
                    .unionByName(neg_easy, allowMissingColumns=True)
                    .dropDuplicates(["user_idx", "item_idx"])
                )
            neg_counts = neg_selected.groupBy("user_idx").agg(F.count("*").alias("picked_neg"))
            need_neg_users = (
                users.join(neg_counts, on="user_idx", how="left")
                .fillna({"picked_neg": 0})
                .withColumn("need_neg", F.greatest(F.lit(int(NEG_PER_USER)) - F.col("picked_neg"), F.lit(0)))
                .filter(F.col("need_neg") > F.lit(0))
                .select("user_idx", "need_neg")
            )
            neg_fill_pool = (
                neg_scored.join(picked_ui, on=["user_idx", "item_idx"], how="left_anti")
                .join(need_neg_users, on="user_idx", how="inner")
            )
            neg_fill = (
                neg_fill_pool.withColumn(
                    "neg_pick_rank",
                    F.row_number().over(Window.partitionBy("user_idx").orderBy(F.col("pre_rank").asc(), F.col("pre_score").desc())),
                )
                .filter(F.col("neg_pick_rank") <= F.col("need_neg"))
                .withColumn("label_source", F.lit("neg_fill_top"))
                .withColumn("label", F.lit(0))
                .withColumn("sample_weight", F.lit(float(NEG_FILL_WEIGHT)))
                .withColumn("neg_tier", F.lit("fill"))
                .withColumn("neg_is_hard", F.lit(False))
            )
            neg = (
                neg_selected.unionByName(neg_fill, allowMissingColumns=True)
                .dropDuplicates(["user_idx", "item_idx"])
                .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
            )
        else:
            neg = (
                neg_base.withColumn("neg_pick_rank", F.row_number().over(Window.partitionBy("user_idx").orderBy(F.col("pre_rank").asc())))
                .filter(F.col("neg_pick_rank") <= F.lit(int(NEG_PER_USER)))
                .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
                .withColumn("label_source", F.lit("neg_topn"))
                .withColumn("label", F.lit(0))
                .withColumn("sample_weight", F.lit(1.0))
                .withColumn("neg_tier", F.lit("topn"))
                .withColumn("neg_is_near", F.lit(False))
                .withColumn("neg_is_hard", F.lit(False))
            )

        rich_train_rows: DataFrame | None = None
        rich_sft_stat: dict[str, Any] = {
            "enabled": bool(ENABLE_RICH_SFT_EXPORT),
            "prompt_mode": str(PROMPT_MODE),
            "target_neg_per_user": 0,
            "rows_total": 0,
            "rows_train": 0,
            "rows_eval": 0,
            "train_pos": 0,
            "train_neg": 0,
            "eval_pos": 0,
            "eval_neg": 0,
            "history_anchor_nonempty": 0,
            "output_train_dir": "",
            "output_eval_dir": "",
            "output_parquet_dir": "",
        }
        if ENABLE_RICH_SFT_EXPORT:
            rich_pos_rows = pos_rows
            rich_total_neg = max(
                0,
                int(RICH_SFT_NEG_EXPLICIT)
                + int(RICH_SFT_NEG_HARD)
                + int(RICH_SFT_NEG_NEAR)
                + int(RICH_SFT_NEG_MID)
                + int(RICH_SFT_NEG_TAIL),
            )
            rich_sft_stat["target_neg_per_user"] = int(rich_total_neg)
            if NEG_LAYERED_ENABLED:
                rich_neg_template = (
                    neg_scored.withColumn("label_source", F.lit(""))
                    .withColumn("label", F.lit(0))
                    .withColumn("sample_weight", F.lit(1.0))
                    .withColumn("neg_tier", F.lit(""))
                    .withColumn("neg_pick_rank", F.lit(0).cast("int"))
                    .withColumn("neg_is_hard", F.lit(False))
                    .limit(0)
                )
                picked_ui = rich_neg_template.select("user_idx", "item_idx")

                explicit_neg = rich_neg_template
                if hist_path.exists() and int(RICH_SFT_NEG_EXPLICIT) > 0:
                    hist_low = (
                        spark.read.parquet(hist_path.as_posix())
                        .join(users, on="user_idx", how="inner")
                        .filter(
                            F.col("hist_rating").isNotNull()
                            & (F.col("hist_rating").cast("double") <= F.lit(float(RICH_SFT_NEG_MAX_RATING)))
                        )
                        .select(
                            "user_idx",
                            "item_idx",
                            F.col("hist_rating").cast("double").alias("hist_rating"),
                            F.col("hist_ts").alias("hist_ts"),
                        )
                        .dropDuplicates(["user_idx", "item_idx"])
                    )
                    explicit_neg = (
                        neg_scored.join(hist_low, on=["user_idx", "item_idx"], how="inner")
                        .withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(
                                    F.col("hist_rating").asc(),
                                    F.col("hist_ts").desc_nulls_last(),
                                    F.col("pre_rank").asc(),
                                )
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(RICH_SFT_NEG_EXPLICIT)))
                        .withColumn("label_source", F.lit("neg_hist_lowrate"))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(NEG_HARD_WEIGHT)))
                        .withColumn("neg_tier", F.lit("observed_dislike"))
                        .withColumn("neg_is_hard", F.lit(True))
                    )
                    picked_ui = explicit_neg.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"])

                def _pick_rich(
                    pool_df: DataFrame,
                    quota: int,
                    label_source: str,
                    sample_weight: float,
                    neg_tier: str,
                    *order_exprs: Any,
                    is_hard: bool = False,
                ) -> DataFrame:
                    if int(quota) <= 0:
                        return rich_neg_template
                    ranked = (
                        pool_df.join(picked_ui, on=["user_idx", "item_idx"], how="left_anti")
                        .withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(
                                    *(list(order_exprs) if order_exprs else [F.col("pre_rank").asc()])
                                )
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.lit(int(quota)))
                        .withColumn("label_source", F.lit(label_source))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(sample_weight)))
                        .withColumn("neg_tier", F.lit(neg_tier))
                        .withColumn("neg_is_hard", F.lit(bool(is_hard)))
                    )
                    return ranked

                rich_hard = _pick_rich(
                    neg_scored.filter(F.col("pre_rank") <= F.lit(int(max(1, RICH_SFT_NEG_HARD_RANK_MAX)))),
                    int(RICH_SFT_NEG_HARD),
                    "neg_rich_hard",
                    float(NEG_HARD_WEIGHT),
                    "hard",
                    F.col("pre_rank").asc(),
                    F.col("pre_score").desc(),
                    is_hard=True,
                )
                picked_ui = picked_ui.unionByName(
                    rich_hard.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])

                rich_near = _pick_rich(
                    neg_scored.filter(F.col("neg_is_near")),
                    int(RICH_SFT_NEG_NEAR),
                    "neg_rich_near",
                    float(NEG_NEAR_WEIGHT),
                    "near",
                    F.col("score_gap_to_pos").asc(),
                    F.col("pre_rank").asc(),
                )
                picked_ui = picked_ui.unionByName(
                    rich_near.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])

                rich_mid = _pick_rich(
                    neg_scored.filter(
                        (F.col("pre_rank") > F.lit(int(max(1, RICH_SFT_NEG_HARD_RANK_MAX))))
                        & (F.col("pre_rank") <= F.lit(int(max(1, RICH_SFT_NEG_MID_RANK_MAX))))
                    ),
                    int(RICH_SFT_NEG_MID),
                    "neg_rich_mid",
                    float(NEG_FILL_WEIGHT),
                    "mid",
                    F.col("pre_rank").asc(),
                    F.col("pre_score").desc(),
                )
                picked_ui = picked_ui.unionByName(
                    rich_mid.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])

                rich_tail = _pick_rich(
                    neg_scored.filter(F.col("pre_rank") > F.lit(int(max(1, RICH_SFT_NEG_MID_RANK_MAX)))),
                    int(RICH_SFT_NEG_TAIL),
                    "neg_rich_tail",
                    float(NEG_EASY_WEIGHT),
                    "tail",
                    F.col("pre_rank").desc(),
                    F.col("pre_score").asc(),
                )
                picked_ui = picked_ui.unionByName(
                    rich_tail.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]),
                    allowMissingColumns=False,
                ).dropDuplicates(["user_idx", "item_idx"])
                rich_selected = (
                    explicit_neg.unionByName(rich_hard, allowMissingColumns=True)
                    .unionByName(rich_near, allowMissingColumns=True)
                    .unionByName(rich_mid, allowMissingColumns=True)
                    .unionByName(rich_tail, allowMissingColumns=True)
                    .dropDuplicates(["user_idx", "item_idx"])
                )

                if RICH_SFT_ALLOW_NEG_FILL and rich_total_neg > 0:
                    rich_counts = rich_selected.groupBy("user_idx").agg(F.count("*").alias("picked_neg"))
                    need_rich = (
                        users.join(rich_counts, on="user_idx", how="left")
                        .fillna({"picked_neg": 0})
                        .withColumn("need_neg", F.greatest(F.lit(int(rich_total_neg)) - F.col("picked_neg"), F.lit(0)))
                        .filter(F.col("need_neg") > F.lit(0))
                        .select("user_idx", "need_neg")
                    )
                    rich_fill = (
                        neg_scored.join(picked_ui, on=["user_idx", "item_idx"], how="left_anti")
                        .join(need_rich, on="user_idx", how="inner")
                        .withColumn(
                            "neg_pick_rank",
                            F.row_number().over(
                                Window.partitionBy("user_idx").orderBy(F.col("pre_rank").asc(), F.col("pre_score").desc())
                            ),
                        )
                        .filter(F.col("neg_pick_rank") <= F.col("need_neg"))
                        .withColumn("label_source", F.lit("neg_rich_fill"))
                        .withColumn("label", F.lit(0))
                        .withColumn("sample_weight", F.lit(float(NEG_FILL_WEIGHT)))
                        .withColumn("neg_tier", F.lit("fill"))
                        .withColumn("neg_is_hard", F.lit(False))
                    )
                    rich_neg = (
                        rich_selected.unionByName(rich_fill, allowMissingColumns=True)
                        .dropDuplicates(["user_idx", "item_idx"])
                        .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
                    )
                else:
                    rich_neg = rich_selected.join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
            else:
                rich_neg = (
                    neg_base.withColumn(
                        "neg_pick_rank",
                        F.row_number().over(Window.partitionBy("user_idx").orderBy(F.col("pre_rank").asc())),
                    )
                    .filter(F.col("neg_pick_rank") <= F.lit(int(rich_total_neg)))
                    .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
                    .withColumn("label_source", F.lit("neg_rich_topn"))
                    .withColumn("label", F.lit(0))
                    .withColumn("sample_weight", F.lit(1.0))
                    .withColumn("neg_tier", F.lit("topn"))
                    .withColumn("neg_is_near", F.lit(False))
                    .withColumn("neg_is_hard", F.lit(False))
                )

            rich_train_rows = rich_pos_rows.select(
                "user_idx",
                "user_id",
                "item_idx",
                "business_id",
                "pre_rank",
                "pre_score",
                "learned_blend_score",
                "learned_rank",
                "learned_rank_band",
                *REASON_AUDIT_EXPORT_COLUMNS,
                "name",
                "city",
                "categories",
                "primary_category",
                "source_set_text",
                "user_segment",
                "source_count",
                "nonpopular_source_count",
                "profile_cluster_source_count",
                "als_rank",
                "cluster_rank",
                "profile_rank",
                "popular_rank",
                "context_rank",
                "has_context",
                "context_detail_count",
                "semantic_support",
                "semantic_tag_richness",
                "tower_score",
                "seq_score",
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
                "label_source",
                "label",
                "sample_weight",
                "neg_tier",
                "neg_pick_rank",
                "neg_is_near",
                "neg_is_hard",
            ).unionByName(
                rich_neg.select(
                    "user_idx",
                    "user_id",
                    "item_idx",
                    "business_id",
                    "pre_rank",
                    "pre_score",
                    "learned_blend_score",
                    "learned_rank",
                    "learned_rank_band",
                    *REASON_AUDIT_EXPORT_COLUMNS,
                    "name",
                    "city",
                    "categories",
                    "primary_category",
                    "source_set_text",
                    "user_segment",
                    "source_count",
                    "nonpopular_source_count",
                    "profile_cluster_source_count",
                    "als_rank",
                    "cluster_rank",
                    "profile_rank",
                    "popular_rank",
                    "context_rank",
                    "has_context",
                    "context_detail_count",
                    "semantic_support",
                    "semantic_tag_richness",
                    "tower_score",
                    "seq_score",
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
                    "label_source",
                    "label",
                    "sample_weight",
                    "neg_tier",
                    "neg_pick_rank",
                    "neg_is_near",
                    "neg_is_hard",
                ),
                allowMissingColumns=False,
            ).dropDuplicates(["user_idx", "item_idx", "label_source"])
            rich_train_rows = maybe_checkpoint_df(rich_train_rows, f"bucket={bucket} rich_train_rows")

        train_rows = pos_rows.select(
            "user_idx",
            "user_id",
            "item_idx",
            "business_id",
            "pre_rank",
            "pre_score",
            "learned_blend_score",
            "learned_rank",
            "learned_rank_band",
            *REASON_AUDIT_EXPORT_COLUMNS,
            "name",
            "city",
            "categories",
            "primary_category",
            "source_set_text",
            "user_segment",
            "source_count",
            "nonpopular_source_count",
            "profile_cluster_source_count",
            "als_rank",
            "cluster_rank",
            "profile_rank",
            "popular_rank",
            "context_rank",
            "has_context",
            "context_detail_count",
            "semantic_support",
            "semantic_tag_richness",
            "tower_score",
            "seq_score",
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
            "label_source",
            "label",
            "sample_weight",
            "neg_tier",
            "neg_pick_rank",
            "neg_is_near",
            "neg_is_hard",
        ).unionByName(
            neg.select(
                "user_idx",
                "user_id",
                "item_idx",
                "business_id",
                "pre_rank",
                "pre_score",
                "learned_blend_score",
                "learned_rank",
                "learned_rank_band",
                *REASON_AUDIT_EXPORT_COLUMNS,
                "name",
                "city",
                "categories",
                "primary_category",
                "source_set_text",
                "user_segment",
                "source_count",
                "nonpopular_source_count",
                "profile_cluster_source_count",
                "als_rank",
                "cluster_rank",
                "profile_rank",
                "popular_rank",
                "context_rank",
                "has_context",
                "context_detail_count",
                "semantic_support",
                "semantic_tag_richness",
                "tower_score",
                "seq_score",
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
                "label_source",
                "label",
                "sample_weight",
                "neg_tier",
                "neg_pick_rank",
                "neg_is_near",
                "neg_is_hard",
            ),
            allowMissingColumns=False,
        ).dropDuplicates(["user_idx", "item_idx", "label_source"])
        train_rows = maybe_checkpoint_df(train_rows, f"bucket={bucket} train_rows")

        if MAX_ROWS_PER_BUCKET > 0:
            if ROW_CAP_ORDERED:
                train_rows = train_rows.orderBy(F.col("user_idx").asc(), F.col("pre_rank").asc()).limit(int(MAX_ROWS_PER_BUCKET))
            else:
                train_rows = train_rows.limit(int(MAX_ROWS_PER_BUCKET))

        if AUDIT_ONLY:
            b_out = out_dir / f"bucket_{bucket}"
            b_out.mkdir(parents=True, exist_ok=True)

            out_df_audit = with_split_assignment(
                train_rows.select("user_idx", "user_id", "item_idx", "label", "neg_tier"),
                eval_user_frac=float(EVAL_USER_FRAC),
                fixed_eval_users=fixed_eval_users_df,
                fixed_eval_join_cols=fixed_eval_join_cols,
            )
            out_summary = collect_split_label_user_summary(out_df_audit, include_split_user_counts=True)
            split_stat = {
                "train_label_0": int(out_summary.get("train_label_0", 0)),
                "train_label_1": int(out_summary.get("train_label_1", 0)),
                "eval_label_0": int(out_summary.get("eval_label_0", 0)),
                "eval_label_1": int(out_summary.get("eval_label_1", 0)),
            }
            users_total = int(out_summary.get("users_total", 0))
            users_with_pos = int(out_summary.get("users_with_pos", 0))
            users_no_pos = int(max(0, users_total - users_with_pos))
            users_no_pos_ratio = float(users_no_pos / users_total) if users_total > 0 else 0.0
            neg_tier_counts: dict[str, int] = {}
            for r in out_df_audit.filter(F.col("label") == F.lit(0)).groupBy("neg_tier").count().collect():
                neg_tier_counts[str(r["neg_tier"])] = int(r["count"])

            if ENABLE_RICH_SFT_EXPORT and rich_train_rows is not None:
                rich_sft_audit_df = with_split_assignment(
                    rich_train_rows.select("user_idx", "user_id", "item_idx", "label"),
                    eval_user_frac=float(EVAL_USER_FRAC),
                    fixed_eval_users=fixed_eval_users_df,
                    fixed_eval_join_cols=fixed_eval_join_cols,
                )
                rich_summary = collect_split_label_user_summary(rich_sft_audit_df)
                rich_sft_stat.update(
                    {
                        "rows_total": int(rich_summary.get("rows_total", 0)),
                        "train_pos": int(rich_summary.get("train_label_1", 0)),
                        "train_neg": int(rich_summary.get("train_label_0", 0)),
                        "eval_pos": int(rich_summary.get("eval_label_1", 0)),
                        "eval_neg": int(rich_summary.get("eval_label_0", 0)),
                    }
                )
                rich_sft_stat["rows_train"] = int(rich_sft_stat["train_pos"]) + int(rich_sft_stat["train_neg"])
                rich_sft_stat["rows_eval"] = int(rich_sft_stat["eval_pos"]) + int(rich_sft_stat["eval_neg"])

            summary.append(
                {
                    "bucket": int(bucket),
                    "candidate_file": cand_file.name,
                    "rows_total": int(sum(split_stat.values())),
                    "rows_train": int(split_stat.get("train_label_0", 0) + split_stat.get("train_label_1", 0)),
                    "rows_eval": int(split_stat.get("eval_label_0", 0) + split_stat.get("eval_label_1", 0)),
                    "train_pos": int(split_stat.get("train_label_1", 0)),
                    "train_neg": int(split_stat.get("train_label_0", 0)),
                    "eval_pos": int(split_stat.get("eval_label_1", 0)),
                    "eval_neg": int(split_stat.get("eval_label_0", 0)),
                    "users_total": users_total,
                    "users_with_positive": users_with_pos,
                    "users_no_positive": users_no_pos,
                    "users_no_positive_ratio": users_no_pos_ratio,
                    "dataset_eval_split_mode": "explicit_cohort" if fixed_eval_users_df is not None else "hash_frac",
                    "dataset_eval_user_count_requested": int(fixed_eval_user_count),
                    "dataset_eval_user_join_cols": fixed_eval_join_cols,
                    "dataset_eval_user_count_surviving": int(out_summary.get("eval_users", 0)),
                    "dataset_eval_user_count_dropped": int(max(0, int(fixed_eval_user_count) - int(out_summary.get("eval_users", 0)))),
                    "neg_tier_counts": neg_tier_counts,
                    "user_evidence_dir": "",
                    "item_evidence_dir": "",
                    "pair_evidence_audit_dir": "",
                    "rich_sft": rich_sft_stat,
                    "pairwise_pool": {
                        "enabled": bool(ENABLE_PAIRWISE_POOL_EXPORT),
                        "topn_per_user": int(PAIRWISE_POOL_TOPN),
                        "rows_total": 0,
                        "rows_train": 0,
                        "rows_eval": 0,
                        "train_pos": 0,
                        "train_neg": 0,
                        "eval_pos": 0,
                        "eval_neg": 0,
                        "users_total": 0,
                        "train_users": 0,
                        "eval_users": 0,
                        "neg_tier_counts": {},
                        "output_train_dir": "",
                        "output_eval_dir": "",
                        "output_parquet_dir": "",
                    },
                    "output_bucket_dir": str(b_out),
                }
            )
            print(f"[DONE] bucket={bucket} audit_only stats={split_stat}")
            truth.unpersist()
            continue

        b_out = out_dir / f"bucket_{bucket}"
        train_dir = b_out / "train_json"
        eval_dir = b_out / "eval_json"
        parquet_dir = b_out / "all_parquet"
        rich_sft_train_dir = b_out / "rich_sft_train_json"
        rich_sft_eval_dir = b_out / "rich_sft_eval_json"
        rich_sft_parquet_dir = b_out / "rich_sft_all_parquet"
        user_evidence_dir = b_out / "user_evidence_table"
        item_evidence_dir = b_out / "item_evidence_table"
        pair_audit_dir = b_out / "pair_evidence_audit"
        pairwise_pool_train_dir = b_out / "pairwise_pool_train_json"
        pairwise_pool_eval_dir = b_out / "pairwise_pool_eval_json"
        pairwise_pool_parquet_dir = b_out / "pairwise_pool_all_parquet"

        history_anchor_df = spark.createDataFrame(
            [],
            "user_idx int, history_anchor_entries array<struct<anchor_rank:int,business_id:string,anchor_text:string>>, history_anchor_available int",
        )
        hist_path = bdir / "train_history.parquet"
        if (ENABLE_RICH_SFT_EXPORT or ENABLE_PAIRWISE_POOL_EXPORT or PROMPT_MODE in {"sft_clean", "full_lite"}) and hist_path.exists():
            anchor_line_udf = F.udf(
                lambda n, c, pc, tags, rating: build_history_anchor_line(n, c, pc, tags, rating, max_chars=110),
                "string",
            )
            fallback_min = min(
                float(RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING),
                float(RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING),
            )
            anchor_keep = max(1, int(RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER) + 2)
            item_business_lookup = cand.select("item_idx", "business_id").dropDuplicates(["item_idx"])
            hist_anchor_base = (
                spark.read.parquet(hist_path.as_posix())
                .join(users, on="user_idx", how="inner")
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
            history_anchor_df = (
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
        clean_profile_udf = F.udf(lambda x: clean_text(x, max_chars=220), "string")
        clean_profile_evidence_udf = F.udf(lambda x: clean_text(x, max_chars=420), "string")
        user_evidence_udf = build_user_evidence_udf()
        pair_evidence_udf = build_pair_alignment_udf()

        enrich: DataFrame | None = None
        out_df: DataFrame | None = None
        split_stat = {"train_label_0": 0, "train_label_1": 0, "eval_label_0": 0, "eval_label_1": 0}
        users_total = 0
        users_with_pos = 0
        users_no_pos = 0
        users_no_pos_ratio = 0.0
        out_summary: dict[str, int] = {}
        neg_tier_counts: dict[str, int] = {}
        out_summary_error = ""
        neg_tier_counts_error = ""

        if not SKIP_POINTWISE_EXPORT:
            print(f"[STEP] bucket={bucket} build pointwise item_review_evidence")
            item_evidence_df = build_item_review_evidence(
                spark=spark,
                bucket_dir=bdir,
                row_df=train_rows.select("user_idx", "business_id"),
                item_sem_df=item_sem_df,
            )
            print(f"[STEP] bucket={bucket} pointwise item_review_evidence ready")

            enrich = (
                train_rows.join(user_profile_df, on="user_id", how="left")
                .join(item_sem_df, on="business_id", how="left")
                .join(cluster_profile_df, on="business_id", how="left")
                .join(item_evidence_df, on=["user_idx", "business_id"], how="left")
                .join(history_anchor_df, on="user_idx", how="left")
                .fillna(
                    {
                        "profile_text": "",
                        "profile_text_evidence": "",
                        "profile_text_short": "",
                        "profile_text_long": "",
                        "profile_pos_text": "",
                        "profile_neg_text": "",
                        "profile_top_pos_tags": "",
                        "profile_top_neg_tags": "",
                        "profile_top_pos_tags_by_type": "",
                        "profile_top_neg_tags_by_type": "",
                        "profile_confidence": 0.0,
                        "user_long_pref_text": "",
                        "user_recent_intent_text": "",
                        "user_negative_avoid_text": "",
                        "user_context_text": "",
                        "top_pos_tags": "",
                        "top_neg_tags": "",
                        "semantic_score": 0.0,
                        "semantic_confidence": 0.0,
                        "source_set_text": "",
                        "user_segment": "",
                        "source_count": 0.0,
                        "nonpopular_source_count": 0.0,
                        "profile_cluster_source_count": 0.0,
                        "als_rank": 0.0,
                        "cluster_rank": 0.0,
                        "profile_rank": 0.0,
                        "popular_rank": 0.0,
                        "context_rank": 0.0,
                        "has_context": 0.0,
                        "context_detail_count": 0.0,
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
                .withColumn("profile_text_short", clean_profile_udf(F.col("profile_text_short")))
                .withColumn("profile_text_long", clean_profile_evidence_udf(F.col("profile_text_long")))
                .withColumn("profile_text", clean_profile_udf(F.col("profile_text")))
                .withColumn("profile_text_evidence", clean_profile_evidence_udf(F.col("profile_text_evidence")))
                .withColumn("profile_pos_text", clean_profile_evidence_udf(F.col("profile_pos_text")))
                .withColumn("profile_neg_text", clean_profile_evidence_udf(F.col("profile_neg_text")))
                .withColumn("user_long_pref_text", clean_profile_evidence_udf(F.col("user_long_pref_text")))
                .withColumn("user_recent_intent_text", clean_profile_evidence_udf(F.col("user_recent_intent_text")))
                .withColumn("user_negative_avoid_text", clean_profile_evidence_udf(F.col("user_negative_avoid_text")))
                .withColumn("user_context_text", clean_profile_evidence_udf(F.col("user_context_text")))
                .withColumn(
                    "user_evidence_text",
                    user_evidence_udf(
                        F.coalesce(F.col("profile_text_short"), F.col("profile_text")),
                        F.coalesce(F.col("profile_text_long"), F.col("profile_text_evidence")),
                        F.col("profile_pos_text"),
                        F.col("profile_neg_text"),
                        F.col("user_long_pref_text"),
                        F.col("user_recent_intent_text"),
                        F.col("user_negative_avoid_text"),
                        F.col("user_context_text"),
                    ),
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
                .withColumn(
                    "history_anchor_count",
                    F.when(F.length(F.col("history_anchor_text")) > F.lit(0), F.lit(1)).otherwise(F.lit(0)),
                )
            ).persist(StorageLevel.DISK_ONLY)

        if PROMPT_MODE in {"semantic", "semantic_rm"}:
            prompt_col = prompt_udf(
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
            )
        elif PROMPT_MODE == "semantic_compact_rm":
            prompt_col = prompt_udf(
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
            )
        elif PROMPT_MODE == "semantic_compact_preserve_rm":
            prompt_col = prompt_udf(
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
            )
        elif PROMPT_MODE == "semantic_compact_targeted_rm":
            prompt_col = prompt_udf(
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
            )
        elif PROMPT_MODE == "semantic_compact_boundary_rm":
            prompt_col = prompt_udf(
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
                F.col("learned_rank"),
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
            )
        elif PROMPT_MODE == "full_lite":
            prompt_col = prompt_udf(
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
            )
        elif PROMPT_MODE == "sft_clean":
            prompt_col = prompt_udf(
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
            )
        else:
            prompt_col = prompt_udf(
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
            )

        if SKIP_POINTWISE_EXPORT or enrich is None:
            out_df = spark.createDataFrame(
                [],
                (
                    "bucket int, user_idx int, item_idx int, business_id string, "
                    "label int, sample_weight double, label_source string, neg_tier string, "
                    "neg_pick_rank int, neg_is_near boolean, neg_is_hard boolean, "
                    "pre_rank double, pre_score double, prompt string, target_text string, "
                    "user_evidence_text string, history_anchor_text string, item_evidence_text string, "
                    "pair_evidence_summary string, split string"
                ),
            )
        else:
            out_df = (
                with_split_assignment(
                    enrich.withColumn("prompt", prompt_col),
                    eval_user_frac=float(EVAL_USER_FRAC),
                    fixed_eval_users=fixed_eval_users_df,
                    fixed_eval_join_cols=fixed_eval_join_cols,
                )
                .withColumn("target_text", F.when(F.col("label").cast("int") == F.lit(1), F.lit("YES")).otherwise(F.lit("NO")))
                .withColumn("bucket", F.lit(int(bucket)))
                .dropDuplicates(["user_idx", "item_idx", "label_source"])
                .select(
                    "bucket",
                    "user_idx",
                    "item_idx",
                    "business_id",
                    "label",
                    "sample_weight",
                    "label_source",
                    "neg_tier",
                    "neg_pick_rank",
                    "neg_is_near",
                    "neg_is_hard",
                    "pre_rank",
                    "pre_score",
                    "prompt",
                    "target_text",
                    "user_evidence_text",
                    "history_anchor_text",
                    "item_evidence_text",
                    "pair_evidence_summary",
                    "split",
                )
            ).persist(StorageLevel.DISK_ONLY)

        out_dirs = []
        if not SKIP_POINTWISE_EXPORT:
            out_dirs.extend([train_dir, eval_dir, parquet_dir, user_evidence_dir, item_evidence_dir, pair_audit_dir])
        if ENABLE_RICH_SFT_EXPORT:
            out_dirs.extend([rich_sft_train_dir, rich_sft_eval_dir, rich_sft_parquet_dir])
        if ENABLE_PAIRWISE_POOL_EXPORT:
            out_dirs.extend([pairwise_pool_train_dir, pairwise_pool_eval_dir, pairwise_pool_parquet_dir])
        for d in out_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        if not SKIP_POINTWISE_EXPORT and out_df is not None and enrich is not None:
            coalesce_for_write(
                out_df.filter(F.col("split") == F.lit("train")),
                WRITE_JSON_PARTITIONS,
            ).write.mode("overwrite").json(train_dir.as_posix())
            coalesce_for_write(
                out_df.filter(F.col("split") == F.lit("eval")),
                WRITE_JSON_PARTITIONS,
            ).write.mode("overwrite").json(eval_dir.as_posix())
            coalesce_for_write(out_df, WRITE_PARQUET_PARTITIONS).write.mode("overwrite").parquet(parquet_dir.as_posix())
            (
                coalesce_for_write(
                    enrich.select(
                        "user_id",
                        "profile_text",
                        "user_evidence_text",
                        "profile_top_pos_tags",
                        "profile_top_neg_tags",
                        "profile_confidence",
                    ).dropDuplicates(["user_id"]),
                    WRITE_AUX_PARTITIONS,
                )
                .write.mode("overwrite")
                .parquet(user_evidence_dir.as_posix())
            )
            (
                coalesce_for_write(
                    enrich.select("user_idx", "business_id", "item_evidence_text").dropDuplicates(["user_idx", "business_id"]),
                    WRITE_AUX_PARTITIONS,
                )
                .write.mode("overwrite")
                .parquet(item_evidence_dir.as_posix())
            )
            (
                coalesce_for_write(
                    enrich.select(
                        F.when(F.length(F.col("pair_evidence_summary")) > F.lit(0), F.lit(1)).otherwise(F.lit(0)).alias("has_pair_evidence"),
                        F.col("label").cast("int").alias("label"),
                    )
                    .groupBy("has_pair_evidence", "label")
                    .count(),
                    WRITE_AUX_PARTITIONS,
                )
                .write.mode("overwrite")
                .option("header", "true")
                .csv(pair_audit_dir.as_posix())
            )
            out_summary, out_summary_error = collect_split_label_user_summary_safe(out_df, include_split_user_counts=True)
            split_stat = {
                "train_label_0": int(out_summary.get("train_label_0", 0)),
                "train_label_1": int(out_summary.get("train_label_1", 0)),
                "eval_label_0": int(out_summary.get("eval_label_0", 0)),
                "eval_label_1": int(out_summary.get("eval_label_1", 0)),
            }
            users_total = int(out_summary.get("users_total", 0))
            users_with_pos = int(out_summary.get("users_with_pos", 0))
            users_no_pos = int(max(0, users_total - users_with_pos))
            users_no_pos_ratio = float(users_no_pos / users_total) if users_total > 0 else 0.0
            try:
                for r in out_df.filter(F.col("label") == F.lit(0)).groupBy("neg_tier").count().collect():
                    neg_tier_counts[str(r["neg_tier"])] = int(r["count"])
            except Exception as exc:
                neg_tier_counts_error = f"{type(exc).__name__}: {exc}"
        else:
            print(f"[SKIP] bucket={bucket} pointwise export disabled; skip pointwise prompt/evidence materialization")
            out_df_audit = with_split_assignment(
                train_rows.select("user_idx", "user_id", "item_idx", "label", "neg_tier"),
                eval_user_frac=float(EVAL_USER_FRAC),
                fixed_eval_users=fixed_eval_users_df,
                fixed_eval_join_cols=fixed_eval_join_cols,
            )
            out_summary, out_summary_error = collect_split_label_user_summary_safe(
                out_df_audit,
                include_split_user_counts=True,
            )
            split_stat = {
                "train_label_0": int(out_summary.get("train_label_0", 0)),
                "train_label_1": int(out_summary.get("train_label_1", 0)),
                "eval_label_0": int(out_summary.get("eval_label_0", 0)),
                "eval_label_1": int(out_summary.get("eval_label_1", 0)),
            }
            users_total = int(out_summary.get("users_total", 0))
            users_with_pos = int(out_summary.get("users_with_pos", 0))
            users_no_pos = int(max(0, users_total - users_with_pos))
            users_no_pos_ratio = float(users_no_pos / users_total) if users_total > 0 else 0.0
            try:
                for r in out_df_audit.filter(F.col("label") == F.lit(0)).groupBy("neg_tier").count().collect():
                    neg_tier_counts[str(r["neg_tier"])] = int(r["count"])
            except Exception as exc:
                neg_tier_counts_error = f"{type(exc).__name__}: {exc}"

        if ENABLE_RICH_SFT_EXPORT and rich_train_rows is not None and rich_prompt_udf is not None:
            print(f"[STEP] bucket={bucket} build rich_sft item_review_evidence")
            rich_item_evidence_df = build_item_review_evidence(
                spark=spark,
                bucket_dir=bdir,
                row_df=rich_train_rows.select("user_idx", "business_id"),
                item_sem_df=item_sem_df,
            )
            print(f"[STEP] bucket={bucket} rich_sft item_review_evidence ready")
            print(f"[STEP] bucket={bucket} build rich_sft join_base")
            rich_join_base = (
                rich_train_rows.join(user_profile_df, on="user_id", how="left")
                .join(item_sem_df, on="business_id", how="left")
                .join(cluster_profile_df, on="business_id", how="left")
                .join(rich_item_evidence_df, on=["user_idx", "business_id"], how="left")
                .join(history_anchor_df, on="user_idx", how="left")
                .fillna(
                    {
                        "profile_text": "",
                        "profile_text_evidence": "",
                        "profile_text_short": "",
                        "profile_text_long": "",
                        "profile_pos_text": "",
                        "profile_neg_text": "",
                        "profile_top_pos_tags": "",
                        "profile_top_neg_tags": "",
                        "profile_top_pos_tags_by_type": "",
                        "profile_top_neg_tags_by_type": "",
                        "profile_confidence": 0.0,
                        "user_long_pref_text": "",
                        "user_recent_intent_text": "",
                        "user_negative_avoid_text": "",
                        "user_context_text": "",
                        "top_pos_tags": "",
                        "top_neg_tags": "",
                        "semantic_score": 0.0,
                        "semantic_confidence": 0.0,
                        "pre_rank": 0.0,
                        "pre_score": 0.0,
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
                        "source_set_text": "",
                        "user_segment": "",
                        "source_count": 0.0,
                        "nonpopular_source_count": 0.0,
                        "profile_cluster_source_count": 0.0,
                        "context_rank": 0.0,
                        "has_context": 0.0,
                        "context_detail_count": 0.0,
                        "cluster_label_for_recsys": "",
                        "item_evidence_text": "",
                    }
                )
            )
            rich_join_base = maybe_checkpoint_df(rich_join_base, f"bucket={bucket} rich_sft_join_base")
            print(f"[STEP] bucket={bucket} rich_sft join_base ready")
            print(f"[STEP] bucket={bucket} build rich_sft late_text_features")
            rich_enrich = (
                rich_join_base.withColumn("profile_text_short", clean_profile_udf(F.col("profile_text_short")))
                .withColumn("profile_text_long", clean_profile_evidence_udf(F.col("profile_text_long")))
                .withColumn("profile_text", clean_profile_udf(F.col("profile_text")))
                .withColumn("profile_text_evidence", clean_profile_evidence_udf(F.col("profile_text_evidence")))
                .withColumn("profile_pos_text", clean_profile_evidence_udf(F.col("profile_pos_text")))
                .withColumn("profile_neg_text", clean_profile_evidence_udf(F.col("profile_neg_text")))
                .withColumn("user_long_pref_text", clean_profile_evidence_udf(F.col("user_long_pref_text")))
                .withColumn("user_recent_intent_text", clean_profile_evidence_udf(F.col("user_recent_intent_text")))
                .withColumn("user_negative_avoid_text", clean_profile_evidence_udf(F.col("user_negative_avoid_text")))
                .withColumn("user_context_text", clean_profile_evidence_udf(F.col("user_context_text")))
                .withColumn(
                    "user_evidence_text",
                    user_evidence_udf(
                        F.coalesce(F.col("profile_text_short"), F.col("profile_text")),
                        F.coalesce(F.col("profile_text_long"), F.col("profile_text_evidence")),
                        F.col("profile_pos_text"),
                        F.col("profile_neg_text"),
                        F.col("user_long_pref_text"),
                        F.col("user_recent_intent_text"),
                        F.col("user_negative_avoid_text"),
                        F.col("user_context_text"),
                    ),
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
            print(f"[STEP] bucket={bucket} rich_sft late_text_features ready")
            rich_enrich = rich_enrich.persist(StorageLevel.DISK_ONLY)

            if PROMPT_MODE in {"semantic", "semantic_rm"}:
                rich_prompt_col = rich_prompt_udf(
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
                )
            elif PROMPT_MODE == "semantic_compact_rm":
                rich_prompt_col = rich_prompt_udf(
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
                )
            elif PROMPT_MODE == "semantic_compact_preserve_rm":
                rich_prompt_col = rich_prompt_udf(
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
                )
            elif PROMPT_MODE == "semantic_compact_targeted_rm":
                rich_prompt_col = rich_prompt_udf(
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
                )
            elif PROMPT_MODE == "semantic_compact_boundary_rm":
                rich_prompt_col = rich_prompt_udf(
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
                    F.col("learned_rank"),
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
                )
            elif PROMPT_MODE == "full_lite":
                rich_prompt_col = rich_prompt_udf(
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
                )
            elif PROMPT_MODE == "sft_clean":
                rich_prompt_col = rich_prompt_udf(
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
                )
            else:
                rich_prompt_col = rich_prompt_udf(
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
                )
            rich_sft_df = (
                with_split_assignment(
                    rich_enrich.withColumn("prompt", rich_prompt_col),
                    eval_user_frac=float(EVAL_USER_FRAC),
                    fixed_eval_users=fixed_eval_users_df,
                    fixed_eval_join_cols=fixed_eval_join_cols,
                )
                .withColumn("target_text", F.when(F.col("label").cast("int") == F.lit(1), F.lit("YES")).otherwise(F.lit("NO")))
                .withColumn("bucket", F.lit(int(bucket)))
                .dropDuplicates(["user_idx", "item_idx", "label_source"])
                .select(
                    "bucket",
                    "user_idx",
                    "item_idx",
                    "business_id",
                    "label",
                    "sample_weight",
                    "label_source",
                    "neg_tier",
                    "neg_pick_rank",
                    "neg_is_near",
                    "neg_is_hard",
                    "pre_rank",
                    "pre_score",
                    "prompt",
                    "target_text",
                    "user_evidence_text",
                    "history_anchor_text",
                    "item_evidence_text",
                    "pair_evidence_summary",
                    *DPO_SUPPORT_EXPORT_COLUMNS,
                    "split",
                )
            ).persist(StorageLevel.DISK_ONLY)

            if not SKIP_RICH_SFT_JSON_EXPORT:
                coalesce_for_write(
                    rich_sft_df.filter(F.col("split") == F.lit("train")),
                    WRITE_JSON_PARTITIONS,
                ).write.mode("overwrite").json(rich_sft_train_dir.as_posix())
                coalesce_for_write(
                    rich_sft_df.filter(F.col("split") == F.lit("eval")),
                    WRITE_JSON_PARTITIONS,
                ).write.mode("overwrite").json(rich_sft_eval_dir.as_posix())
            coalesce_for_write(
                rich_sft_df,
                WRITE_PARQUET_PARTITIONS,
            ).write.mode("overwrite").parquet(rich_sft_parquet_dir.as_posix())

            rich_summary = collect_split_label_user_summary(
                rich_sft_df,
                include_history_anchor_nonempty=True,
            )
            rich_sft_stat.update(
                {
                    "rows_total": int(rich_summary.get("rows_total", 0)),
                    "history_anchor_nonempty": int(rich_summary.get("history_anchor_nonempty", 0)),
                    "output_train_dir": str(rich_sft_train_dir) if not SKIP_RICH_SFT_JSON_EXPORT else "",
                    "output_eval_dir": str(rich_sft_eval_dir) if not SKIP_RICH_SFT_JSON_EXPORT else "",
                    "output_parquet_dir": str(rich_sft_parquet_dir),
                    "train_pos": int(rich_summary.get("train_label_1", 0)),
                    "train_neg": int(rich_summary.get("train_label_0", 0)),
                    "eval_pos": int(rich_summary.get("eval_label_1", 0)),
                    "eval_neg": int(rich_summary.get("eval_label_0", 0)),
                }
            )
            rich_sft_stat["rows_train"] = int(rich_sft_stat["train_pos"]) + int(rich_sft_stat["train_neg"])
            rich_sft_stat["rows_eval"] = int(rich_sft_stat["eval_pos"]) + int(rich_sft_stat["eval_neg"])
            rich_join_base.unpersist(blocking=False)
            rich_enrich.unpersist(blocking=False)
            rich_sft_df.unpersist(blocking=False)

        pairwise_pool_stat: dict[str, Any] = {
            "enabled": bool(ENABLE_PAIRWISE_POOL_EXPORT),
            "topn_per_user": int(PAIRWISE_POOL_TOPN),
            "rows_total": 0,
            "rows_train": 0,
            "rows_eval": 0,
            "train_pos": 0,
            "train_neg": 0,
            "eval_pos": 0,
            "eval_neg": 0,
            "users_total": 0,
            "train_users": 0,
            "eval_users": 0,
            "neg_tier_counts": {},
            "output_train_dir": "",
            "output_eval_dir": "",
            "output_parquet_dir": "",
        }
        if ENABLE_PAIRWISE_POOL_EXPORT:
            print(f"[STEP] bucket={bucket} build pairwise_pool item_review_evidence")
            cand_pool_raw = (
                spark.read.parquet(cand_file.as_posix())
                .join(users, on="user_idx", how="inner")
            )
            if stage11_target_users_asset_enabled:
                print(f"[STEP] bucket={bucket} restrict pairwise_pool users to stage11 target 11-100 universe")
                pool_target_users_df = (
                    stage11_target_users_df.filter(F.col("bucket") == F.lit(int(bucket)))
                    .select("user_idx", "user_id", "truth_learned_rank")
                    .dropDuplicates(["user_idx"])
                )
                cand_pool_raw = cand_pool_raw.join(
                    pool_target_users_df.select("user_idx").dropDuplicates(["user_idx"]),
                    on=["user_idx"],
                    how="inner",
                )
            cand_pool_raw = attach_stage10_learned_features(spark, cand_pool_raw, bucket)
            cand_pool_raw = attach_stage11_rescue_reason_features(spark, cand_pool_raw)
            print(
                f"[STEP] bucket={bucket} filter pairwise_pool candidates by learned_rank<= {int(PAIRWISE_POOL_TOPN)} "
                f"(fallback pre_rank when learned_rank is null)"
            )
            cand_pool_raw = cand_pool_raw.withColumn(
                "_pairwise_pool_rank",
                F.coalesce(F.col("learned_rank").cast("double"), F.col("pre_rank").cast("double")),
            ).filter(F.col("_pairwise_pool_rank") <= F.lit(float(PAIRWISE_POOL_TOPN)))
            cand_pool = project_candidate_rows(cand_pool_raw)
            truth_pool_users = (
                cand_pool.select("user_idx", "item_idx")
                .join(
                    truth.select("user_idx", F.col("true_item_idx").alias("item_idx")),
                    on=["user_idx", "item_idx"],
                    how="inner",
                )
                .select("user_idx")
                .dropDuplicates(["user_idx"])
            )
            pool_truth = (
                truth.join(truth_pool_users, on="user_idx", how="inner")
                .select("user_idx", "user_id", "true_item_idx")
                .dropDuplicates(["user_idx"])
            )
            pool_rows_base = (
                cand_pool.join(pool_truth, on="user_idx", how="inner")
                .withColumn("label", F.when(F.col("item_idx") == F.col("true_item_idx"), F.lit(1)).otherwise(F.lit(0)))
                .withColumn(
                    "label_source",
                    F.when(F.col("item_idx") == F.col("true_item_idx"), F.lit("true")).otherwise(F.lit("pair_pool")),
                )
                .withColumn("sample_weight", F.lit(1.0))
                .withColumn("neg_pick_rank", F.lit(0).cast("int"))
            )
            pool_pos_ctx = (
                pool_rows_base.filter(F.col("label") == F.lit(1))
                .select(
                    "user_idx",
                    F.lower(F.trim(F.coalesce(F.col("city"), F.lit("")))).alias("pos_city_norm"),
                    F.lower(F.trim(F.coalesce(F.col("primary_category"), F.lit("")))).alias("pos_cat_norm"),
                    F.col("pre_score").cast("double").alias("pos_pre_score"),
                )
                .groupBy("user_idx")
                .agg(
                    F.collect_set("pos_city_norm").alias("pos_city_set"),
                    F.collect_set("pos_cat_norm").alias("pos_cat_set"),
                    F.max("pos_pre_score").alias("pos_pre_score_max"),
                )
            )
            pool_rows = (
                pool_rows_base.join(pool_pos_ctx, on="user_idx", how="left")
                .withColumn("pool_rank_num", F.coalesce(F.col("learned_rank").cast("double"), F.col("pre_rank").cast("double")))
                .withColumn("city_norm", F.lower(F.trim(F.coalesce(F.col("city"), F.lit("")))))
                .withColumn("cat_norm", F.lower(F.trim(F.coalesce(F.col("primary_category"), F.lit("")))))
                .withColumn(
                    "neg_is_near_city",
                    F.coalesce(F.array_contains(F.col("pos_city_set"), F.col("city_norm")), F.lit(False)),
                )
                .withColumn(
                    "neg_is_near_cat",
                    F.coalesce(F.array_contains(F.col("pos_cat_set"), F.col("cat_norm")), F.lit(False)),
                )
                .withColumn(
                    "neg_is_near",
                    F.when(F.col("label") == F.lit(1), F.lit(False)).otherwise(F.col("neg_is_near_city") | F.col("neg_is_near_cat")),
                )
                .withColumn(
                    "neg_is_hard",
                    F.when(F.col("label") == F.lit(1), F.lit(False)).otherwise(F.col("pool_rank_num") <= F.lit(10.0)),
                )
                .withColumn(
                    "neg_tier",
                    F.when(F.col("label") == F.lit(1), F.lit("pos"))
                    .when(F.col("neg_is_near"), F.lit("near"))
                    .when(F.col("pool_rank_num") <= F.lit(10.0), F.lit("hard"))
                    .when(F.col("pool_rank_num") <= F.lit(80.0), F.lit("mid"))
                    .otherwise(F.lit("easy")),
                )
                .drop(
                    "true_item_idx",
                    "pos_city_set",
                    "pos_cat_set",
                    "pos_pre_score_max",
                    "city_norm",
                    "cat_norm",
                    "neg_is_near_city",
                    "neg_is_near_cat",
                    "pool_rank_num",
                )
            )
            pool_item_evidence_df = build_item_review_evidence(
                spark=spark,
                bucket_dir=bdir,
                row_df=pool_rows.select("user_idx", "business_id"),
                item_sem_df=item_sem_df,
            )
            print(f"[STEP] bucket={bucket} pairwise_pool item_review_evidence ready")
            pool_rows = maybe_checkpoint_df(pool_rows, f"bucket={bucket} pairwise_pool_rows")
            pool_user_profile_df = user_profile_df.select(
                "user_id",
                "profile_text",
                "profile_text_evidence",
                "profile_text_short",
                "profile_text_long",
                "profile_pos_text",
                "profile_neg_text",
                "profile_top_pos_tags",
                "profile_top_neg_tags",
                "profile_top_pos_tags_by_type",
                "profile_top_neg_tags_by_type",
                "profile_confidence",
                "user_long_pref_text",
                "user_recent_intent_text",
                "user_negative_avoid_text",
                "user_context_text",
            )
            pool_item_sem_df = item_sem_df.select(
                "business_id",
                "top_pos_tags",
                "top_neg_tags",
                "semantic_score",
                "semantic_confidence",
            )
            pool_cluster_df = cluster_profile_df.select(
                "business_id",
                "cluster_for_recsys",
                "cluster_label_for_recsys",
            )
            pool_history_anchor_df = history_anchor_df.select(
                "user_idx",
                "history_anchor_entries",
                "history_anchor_available",
            )
            pool_stage11_user_semantic_df = stage11_user_semantic_profile_df.select(
                "bucket",
                "user_idx",
                "user_id",
                "stable_preferences_text",
                "recent_intent_text_v2",
                "avoidance_text_v2",
                "history_anchor_hint_text",
                "user_semantic_profile_text_v2",
                "user_evidence_text_v2",
                "user_profile_richness_v2",
                "user_profile_richness_tier_v2",
                "user_semantic_facts_v1",
                "user_visible_fact_count_v1",
                "user_focus_fact_count_v1",
                "user_recent_fact_count_v1",
                "user_avoid_fact_count_v1",
                "user_history_fact_count_v1",
                "user_evidence_fact_count_v1",
            )
            pool_stage11_merchant_semantic_df = stage11_merchant_semantic_profile_df.select(
                "bucket",
                "business_id",
                "core_offering_text",
                "scene_fit_text",
                "strengths_text",
                "risk_points_text",
                "merchant_semantic_profile_text_v2",
                "merchant_profile_richness_v2",
                "merchant_semantic_facts_v1",
                "merchant_visible_fact_count_v1",
                "merchant_core_fact_count_v1",
                "merchant_scene_fact_count_v1",
                "merchant_strength_fact_count_v1",
                "merchant_risk_fact_count_v1",
            )
            pool_stage11_pair_alignment_df = stage11_user_business_alignment_df.select(
                "bucket",
                "user_idx",
                "item_idx",
                "user_id",
                "business_id",
                "fit_reasons_text_v1",
                "friction_reasons_text_v1",
                "evidence_basis_text_v1",
                "pair_alignment_richness_v1",
                "semantic_prompt_readiness_tier_v1",
                "pair_alignment_facts_v1",
                "pair_fit_fact_count_v1",
                "pair_friction_fact_count_v1",
                "pair_evidence_fact_count_v1",
                "pair_stable_fit_fact_count_v1",
                "pair_recent_fit_fact_count_v1",
                "pair_history_fit_fact_count_v1",
                "pair_practical_fit_fact_count_v1",
                "pair_fit_scope_count_v1",
                "pair_has_visible_user_fit_v1",
                "pair_has_visible_conflict_v1",
                "pair_has_recent_context_visible_bridge_v1",
                "pair_has_detail_support_v1",
                "pair_has_multisource_fit_v1",
                "pair_has_candidate_fit_support_v1",
                "pair_has_candidate_strong_fit_support_v1",
                "pair_has_contrastive_support_v1",
                "pair_fact_signal_count_v1",
                "boundary_constructability_class_v1",
                "boundary_constructability_reason_codes_v1",
                "boundary_prompt_ready_v1",
                "boundary_rival_total_v1",
                "boundary_rival_head_or_boundary_v1",
            )
            pool_item_evidence_slim_df = pool_item_evidence_df.select(
                "user_idx",
                "business_id",
                "item_evidence_text",
            )
            print(f"[STEP] bucket={bucket} build pairwise_pool join_base core")
            pool_join_base_core = (
                pool_rows.join(pool_cluster_df, on="business_id", how="left")
                .join(pool_item_sem_df, on="business_id", how="left")
                .fillna(
                    {
                        "top_pos_tags": "",
                        "top_neg_tags": "",
                        "semantic_score": 0.0,
                        "semantic_confidence": 0.0,
                        "pre_rank": 0.0,
                        "pre_score": 0.0,
                        "source_set_text": "",
                        "user_segment": "",
                        "source_count": 0.0,
                        "nonpopular_source_count": 0.0,
                        "profile_cluster_source_count": 0.0,
                        "als_rank": 0.0,
                        "cluster_rank": 0.0,
                        "profile_rank": 0.0,
                        "popular_rank": 0.0,
                        "context_rank": 0.0,
                        "has_context": 0.0,
                        "context_detail_count": 0.0,
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
                    }
                )
            )
            pool_core_export_cols = [
                "user_idx",
                "user_id",
                "item_idx",
                "business_id",
                "label",
                "sample_weight",
                "label_source",
                "neg_tier",
                "neg_pick_rank",
                "neg_is_near",
                "neg_is_hard",
                "pre_rank",
                "pre_rank_band",
                "pre_score",
                *DPO_SUPPORT_EXPORT_COLUMNS,
            ]
            pool_join_base_core = pool_join_base_core.select(*pool_core_export_cols)
            print(f"[STEP] bucket={bucket} pairwise_pool join_base core ready")
            pool_join_base_core = maybe_checkpoint_df(pool_join_base_core, f"bucket={bucket} pairwise_pool_join_base_core")
            print(f"[STEP] bucket={bucket} pairwise_pool join_base core checkpoint ready")
            print(f"[STEP] bucket={bucket} dedupe pairwise_pool join_base core")
            pool_join_base_core = pool_join_base_core.dropDuplicates(["user_idx", "item_idx"])
            pool_join_base_core = maybe_checkpoint_df(pool_join_base_core, f"bucket={bucket} pairwise_pool_join_base_unique")
            print(f"[STEP] bucket={bucket} pairwise_pool join_base unique ready")

            print(f"[STEP] bucket={bucket} build pairwise_pool text view")
            pool_text_base = (
                pool_join_base_core.select("user_idx", "user_id", "item_idx", "business_id")
                .withColumn("bucket", F.lit(int(bucket)))
                .join(pool_user_profile_df, on="user_id", how="left")
                .join(pool_history_anchor_df, on="user_idx", how="left")
                .join(pool_stage11_user_semantic_df, on=["bucket", "user_idx", "user_id"], how="left")
                .join(pool_stage11_merchant_semantic_df, on=["bucket", "business_id"], how="left")
                .join(pool_stage11_pair_alignment_df, on=["bucket", "user_idx", "item_idx", "user_id", "business_id"], how="left")
                .join(pool_item_evidence_slim_df, on=["user_idx", "business_id"], how="left")
                .fillna(
                    {
                        "profile_text": "",
                        "profile_text_evidence": "",
                        "profile_text_short": "",
                        "profile_text_long": "",
                        "profile_pos_text": "",
                        "profile_neg_text": "",
                        "profile_top_pos_tags": "",
                        "profile_top_neg_tags": "",
                        "profile_top_pos_tags_by_type": "",
                        "profile_top_neg_tags_by_type": "",
                        "profile_confidence": 0.0,
                        "user_long_pref_text": "",
                        "user_recent_intent_text": "",
                        "user_negative_avoid_text": "",
                        "user_context_text": "",
                        "stable_preferences_text": "",
                        "recent_intent_text_v2": "",
                        "avoidance_text_v2": "",
                        "history_anchor_hint_text": "",
                        "user_evidence_text_v2": "",
                        "user_semantic_profile_text_v2": "",
                        "user_profile_richness_v2": 0,
                        "user_profile_richness_tier_v2": "",
                        "user_semantic_facts_v1": "[]",
                        "user_visible_fact_count_v1": 0,
                        "user_focus_fact_count_v1": 0,
                        "user_recent_fact_count_v1": 0,
                        "user_avoid_fact_count_v1": 0,
                        "user_history_fact_count_v1": 0,
                        "user_evidence_fact_count_v1": 0,
                        "core_offering_text": "",
                        "scene_fit_text": "",
                        "strengths_text": "",
                        "risk_points_text": "",
                        "merchant_semantic_profile_text_v2": "",
                        "merchant_profile_richness_v2": 0,
                        "merchant_semantic_facts_v1": "[]",
                        "merchant_visible_fact_count_v1": 0,
                        "merchant_core_fact_count_v1": 0,
                        "merchant_scene_fact_count_v1": 0,
                        "merchant_strength_fact_count_v1": 0,
                        "merchant_risk_fact_count_v1": 0,
                        "fit_reasons_text_v1": "",
                        "friction_reasons_text_v1": "",
                        "evidence_basis_text_v1": "",
                        "pair_alignment_richness_v1": 0,
                        "semantic_prompt_readiness_tier_v1": "",
                        "pair_alignment_facts_v1": "[]",
                        "pair_fit_fact_count_v1": 0,
                        "pair_friction_fact_count_v1": 0,
                        "pair_evidence_fact_count_v1": 0,
                        "pair_stable_fit_fact_count_v1": 0,
                        "pair_recent_fit_fact_count_v1": 0,
                        "pair_history_fit_fact_count_v1": 0,
                        "pair_practical_fit_fact_count_v1": 0,
                        "pair_fit_scope_count_v1": 0,
                        "pair_has_visible_user_fit_v1": 0,
                        "pair_has_visible_conflict_v1": 0,
                        "pair_has_recent_context_visible_bridge_v1": 0,
                        "pair_has_detail_support_v1": 0,
                        "pair_has_multisource_fit_v1": 0,
                        "pair_has_candidate_fit_support_v1": 0,
                        "pair_has_candidate_strong_fit_support_v1": 0,
                        "pair_has_contrastive_support_v1": 0,
                        "pair_fact_signal_count_v1": 0,
                        "boundary_constructability_class_v1": "",
                        "boundary_constructability_reason_codes_v1": "[]",
                        "boundary_prompt_ready_v1": 0,
                        "boundary_rival_total_v1": 0,
                        "boundary_rival_head_or_boundary_v1": 0,
                        "history_anchor_available": 0,
                        "item_evidence_text": "",
                    }
                )
            )
            pool_text_cols = [
                "user_idx",
                "user_id",
                "item_idx",
                "business_id",
                "profile_text",
                "profile_text_evidence",
                "profile_text_short",
                "profile_text_long",
                "profile_pos_text",
                "profile_neg_text",
                "profile_top_pos_tags",
                "profile_top_neg_tags",
                "profile_top_pos_tags_by_type",
                "profile_top_neg_tags_by_type",
                "profile_confidence",
                "user_long_pref_text",
                "user_recent_intent_text",
                "user_negative_avoid_text",
                "user_context_text",
                "stable_preferences_text",
                "recent_intent_text_v2",
                "avoidance_text_v2",
                "history_anchor_hint_text",
                "user_evidence_text_v2",
                "user_semantic_profile_text_v2",
                "user_profile_richness_v2",
                "user_profile_richness_tier_v2",
                "user_semantic_facts_v1",
                "user_visible_fact_count_v1",
                "user_focus_fact_count_v1",
                "user_recent_fact_count_v1",
                "user_avoid_fact_count_v1",
                "user_history_fact_count_v1",
                "user_evidence_fact_count_v1",
                "core_offering_text",
                "scene_fit_text",
                "strengths_text",
                "risk_points_text",
                "merchant_semantic_profile_text_v2",
                "merchant_profile_richness_v2",
                "merchant_semantic_facts_v1",
                "merchant_visible_fact_count_v1",
                "merchant_core_fact_count_v1",
                "merchant_scene_fact_count_v1",
                "merchant_strength_fact_count_v1",
                "merchant_risk_fact_count_v1",
                "fit_reasons_text_v1",
                "friction_reasons_text_v1",
                "evidence_basis_text_v1",
                "pair_alignment_richness_v1",
                "semantic_prompt_readiness_tier_v1",
                "pair_alignment_facts_v1",
                "pair_fit_fact_count_v1",
                "pair_friction_fact_count_v1",
                "pair_evidence_fact_count_v1",
                "pair_stable_fit_fact_count_v1",
                "pair_recent_fit_fact_count_v1",
                "pair_history_fit_fact_count_v1",
                "pair_practical_fit_fact_count_v1",
                "pair_fit_scope_count_v1",
                "pair_has_visible_user_fit_v1",
                "pair_has_visible_conflict_v1",
                "pair_has_recent_context_visible_bridge_v1",
                "pair_has_detail_support_v1",
                "pair_has_multisource_fit_v1",
                "pair_has_candidate_fit_support_v1",
                "pair_has_candidate_strong_fit_support_v1",
                "pair_has_contrastive_support_v1",
                "pair_fact_signal_count_v1",
                "boundary_constructability_class_v1",
                "boundary_constructability_reason_codes_v1",
                "boundary_prompt_ready_v1",
                "boundary_rival_total_v1",
                "boundary_rival_head_or_boundary_v1",
                "history_anchor_entries",
                "item_evidence_text",
            ]
            pool_text_base = pool_text_base.select(*pool_text_cols)
            print(f"[STEP] bucket={bucket} pairwise_pool text view ready")

            if PAIRWISE_POOL_MINIMAL_EXPORT:
                print(
                    f"[STEP] bucket={bucket} pairwise_pool minimal export enabled; "
                    "delay prompt, user-focus, and item-match text structuring to pair/listwise pack; keep online-available semantic text"
                )
                pool_key_cols = ["user_idx", "user_id", "item_idx", "business_id"]
                pool_user_key_cols = ["user_idx", "user_id"]
                pool_merchant_key_cols = ["item_idx", "business_id"]
                pool_user_semantic_payload_cols = pool_user_key_cols + [
                    "stable_preferences_text",
                    "recent_intent_text_v2",
                    "avoidance_text_v2",
                    "history_anchor_hint_text",
                    "user_evidence_text_v2",
                    "user_semantic_profile_text_v2",
                    "user_profile_richness_v2",
                    "user_profile_richness_tier_v2",
                    "user_semantic_facts_v1",
                    "user_visible_fact_count_v1",
                    "user_focus_fact_count_v1",
                    "user_recent_fact_count_v1",
                    "user_avoid_fact_count_v1",
                    "user_history_fact_count_v1",
                    "user_evidence_fact_count_v1",
                ]
                pool_merchant_text_payload_cols = pool_merchant_key_cols + [
                    "core_offering_text",
                    "scene_fit_text",
                    "strengths_text",
                    "risk_points_text",
                    "merchant_semantic_profile_text_v2",
                ]
                pool_merchant_fact_payload_cols = pool_merchant_key_cols + [
                    "merchant_profile_richness_v2",
                    "merchant_semantic_facts_v1",
                    "merchant_visible_fact_count_v1",
                    "merchant_core_fact_count_v1",
                    "merchant_scene_fact_count_v1",
                    "merchant_strength_fact_count_v1",
                    "merchant_risk_fact_count_v1",
                ]
                pool_pair_text_payload_cols = pool_key_cols + [
                    "fit_reasons_text_v1",
                    "friction_reasons_text_v1",
                    "evidence_basis_text_v1",
                    "item_evidence_text",
                ]
                pool_pair_fact_payload_cols = pool_key_cols + [
                    "pair_alignment_richness_v1",
                    "semantic_prompt_readiness_tier_v1",
                    "pair_alignment_facts_v1",
                    "pair_fit_fact_count_v1",
                    "pair_friction_fact_count_v1",
                    "pair_evidence_fact_count_v1",
                    "pair_stable_fit_fact_count_v1",
                    "pair_recent_fit_fact_count_v1",
                    "pair_history_fit_fact_count_v1",
                    "pair_practical_fit_fact_count_v1",
                    "pair_fit_scope_count_v1",
                    "pair_has_visible_user_fit_v1",
                    "pair_has_visible_conflict_v1",
                    "pair_has_recent_context_visible_bridge_v1",
                    "pair_has_detail_support_v1",
                    "pair_has_multisource_fit_v1",
                    "pair_has_candidate_fit_support_v1",
                    "pair_has_candidate_strong_fit_support_v1",
                    "pair_has_contrastive_support_v1",
                    "pair_fact_signal_count_v1",
                    "boundary_constructability_class_v1",
                    "boundary_constructability_reason_codes_v1",
                    "boundary_prompt_ready_v1",
                    "boundary_rival_total_v1",
                    "boundary_rival_head_or_boundary_v1",
                ]
                pool_profile_text_payload_cols = pool_user_key_cols + [
                    "profile_text",
                    "profile_text_evidence",
                    "profile_text_short",
                    "profile_text_long",
                    "profile_pos_text",
                    "profile_neg_text",
                    "profile_top_pos_tags",
                    "profile_top_neg_tags",
                    "profile_top_pos_tags_by_type",
                    "profile_top_neg_tags_by_type",
                    "profile_confidence",
                    "user_long_pref_text",
                    "user_recent_intent_text",
                    "user_negative_avoid_text",
                    "user_context_text",
                    "history_anchor_entries",
                ]
                print(f"[STEP] bucket={bucket} build pairwise_pool user semantic payload")
                pool_user_semantic_payload = (
                    pool_text_base.select(*pool_user_semantic_payload_cols)
                    .dropDuplicates(pool_user_key_cols)
                )
                pool_user_semantic_payload = maybe_checkpoint_df(
                    pool_user_semantic_payload,
                    f"bucket={bucket} pairwise_pool_user_semantic_payload",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool user semantic payload ready")

                print(f"[STEP] bucket={bucket} build pairwise_pool merchant text payload")
                pool_merchant_text_payload = (
                    pool_text_base.select(*pool_merchant_text_payload_cols)
                    .dropDuplicates(pool_merchant_key_cols)
                )
                pool_merchant_text_payload = maybe_checkpoint_df(
                    pool_merchant_text_payload,
                    f"bucket={bucket} pairwise_pool_merchant_text_payload",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool merchant text payload ready")

                print(f"[STEP] bucket={bucket} build pairwise_pool merchant fact payload")
                pool_merchant_fact_payload = (
                    pool_text_base.select(*pool_merchant_fact_payload_cols)
                    .dropDuplicates(pool_merchant_key_cols)
                )
                pool_merchant_fact_payload = maybe_checkpoint_df(
                    pool_merchant_fact_payload,
                    f"bucket={bucket} pairwise_pool_merchant_fact_payload",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool merchant fact payload ready")

                print(f"[STEP] bucket={bucket} build pairwise_pool pair text payload")
                pool_pair_text_payload = pool_text_base.select(*pool_pair_text_payload_cols)
                pool_pair_text_payload = maybe_checkpoint_df(
                    pool_pair_text_payload,
                    f"bucket={bucket} pairwise_pool_pair_text_payload",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool pair text payload ready")

                print(f"[STEP] bucket={bucket} build pairwise_pool pair fact payload")
                pool_pair_fact_payload = pool_text_base.select(*pool_pair_fact_payload_cols)
                pool_pair_fact_payload = maybe_checkpoint_df(
                    pool_pair_fact_payload,
                    f"bucket={bucket} pairwise_pool_pair_fact_payload",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool pair fact payload ready")

                print(f"[STEP] bucket={bucket} build pairwise_pool user profile text clean")
                pool_profile_text_payload = (
                    pool_text_base.select(*pool_profile_text_payload_cols)
                    .dropDuplicates(pool_user_key_cols)
                    .withColumn("profile_text_short", clean_profile_udf(F.col("profile_text_short")))
                    .withColumn("profile_text_long", clean_profile_evidence_udf(F.col("profile_text_long")))
                    .withColumn("profile_text", clean_profile_udf(F.col("profile_text")))
                    .withColumn("profile_text_evidence", clean_profile_evidence_udf(F.col("profile_text_evidence")))
                    .withColumn("profile_pos_text", clean_profile_evidence_udf(F.col("profile_pos_text")))
                    .withColumn("profile_neg_text", clean_profile_evidence_udf(F.col("profile_neg_text")))
                    .withColumn("user_long_pref_text", clean_profile_evidence_udf(F.col("user_long_pref_text")))
                    .withColumn("user_recent_intent_text", clean_profile_evidence_udf(F.col("user_recent_intent_text")))
                    .withColumn("user_negative_avoid_text", clean_profile_evidence_udf(F.col("user_negative_avoid_text")))
                    .withColumn("user_context_text", clean_profile_evidence_udf(F.col("user_context_text")))
                )
                pool_profile_text_payload = maybe_checkpoint_df(
                    pool_profile_text_payload,
                    f"bucket={bucket} pairwise_pool_user_profile_text_clean",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool user profile text clean ready")

                print(f"[STEP] bucket={bucket} build pairwise_pool evidence text")
                pool_evidence_payload = (
                    pool_join_base_core.select(*pool_key_cols)
                    .join(pool_user_semantic_payload.select(*pool_user_key_cols, "user_evidence_text_v2"), on=pool_user_key_cols, how="left")
                    .join(pool_profile_text_payload, on=pool_user_key_cols, how="left")
                    .withColumn(
                        "user_evidence_text",
                        F.coalesce(
                            F.col("user_evidence_text_v2"),
                            user_evidence_udf(
                                F.coalesce(F.col("profile_text_short"), F.col("profile_text")),
                                F.coalesce(F.col("profile_text_long"), F.col("profile_text_evidence")),
                                F.col("profile_pos_text"),
                                F.col("profile_neg_text"),
                                F.col("user_long_pref_text"),
                                F.col("user_recent_intent_text"),
                                F.col("user_negative_avoid_text"),
                                F.col("user_context_text"),
                            ),
                        ),
                    )
                    .withColumn(
                        "history_anchor_text",
                        history_anchor_text_udf(F.col("history_anchor_entries"), F.col("business_id")),
                    )
                    .withColumn("pair_evidence_summary", F.lit(""))
                    .drop("user_evidence_text_v2")
                    .drop("history_anchor_entries")
                )
                print(f"[STEP] bucket={bucket} pairwise_pool evidence text view ready")
                pool_evidence_payload = maybe_checkpoint_df(
                    pool_evidence_payload,
                    f"bucket={bucket} pairwise_pool_evidence_text",
                )
                print(f"[STEP] bucket={bucket} pairwise_pool evidence text checkpoint ready")
                print(f"[STEP] bucket={bucket} attach pairwise_pool late semantic text")
                pool_materialized = (
                    pool_join_base_core.join(
                        pool_user_semantic_payload,
                        on=pool_user_key_cols,
                        how="left",
                    )
                    .join(
                        pool_merchant_text_payload,
                        on=pool_merchant_key_cols,
                        how="left",
                    )
                    .join(
                        pool_merchant_fact_payload,
                        on=pool_merchant_key_cols,
                        how="left",
                    )
                    .join(
                        pool_pair_text_payload,
                        on=pool_key_cols,
                        how="left",
                    )
                    .join(
                        pool_pair_fact_payload,
                        on=pool_key_cols,
                        how="left",
                    )
                    .join(
                        pool_evidence_payload,
                        on=["user_idx", "user_id", "item_idx", "business_id"],
                        how="left",
                    )
                    .fillna(
                        {
                        "profile_text": "",
                        "profile_text_evidence": "",
                        "profile_text_short": "",
                        "profile_text_long": "",
                        "profile_pos_text": "",
                        "profile_neg_text": "",
                        "profile_top_pos_tags": "",
                        "profile_top_neg_tags": "",
                        "profile_top_pos_tags_by_type": "",
                        "profile_top_neg_tags_by_type": "",
                        "profile_confidence": 0.0,
                        "user_long_pref_text": "",
                        "user_recent_intent_text": "",
                        "user_negative_avoid_text": "",
                        "user_context_text": "",
                        "stable_preferences_text": "",
                        "recent_intent_text_v2": "",
                        "avoidance_text_v2": "",
                        "history_anchor_hint_text": "",
                        "user_semantic_profile_text_v2": "",
                        "user_evidence_text_v2": "",
                        "user_profile_richness_v2": 0,
                        "user_profile_richness_tier_v2": "",
                        "user_semantic_facts_v1": "[]",
                        "user_visible_fact_count_v1": 0,
                        "user_focus_fact_count_v1": 0,
                        "user_recent_fact_count_v1": 0,
                        "user_avoid_fact_count_v1": 0,
                        "user_history_fact_count_v1": 0,
                        "user_evidence_fact_count_v1": 0,
                        "core_offering_text": "",
                        "scene_fit_text": "",
                        "strengths_text": "",
                        "risk_points_text": "",
                        "merchant_semantic_profile_text_v2": "",
                        "merchant_profile_richness_v2": 0,
                        "merchant_semantic_facts_v1": "[]",
                        "merchant_visible_fact_count_v1": 0,
                        "merchant_core_fact_count_v1": 0,
                        "merchant_scene_fact_count_v1": 0,
                        "merchant_strength_fact_count_v1": 0,
                        "merchant_risk_fact_count_v1": 0,
                        "fit_reasons_text_v1": "",
                        "friction_reasons_text_v1": "",
                        "evidence_basis_text_v1": "",
                        "pair_alignment_richness_v1": 0,
                        "semantic_prompt_readiness_tier_v1": "",
                        "pair_alignment_facts_v1": "[]",
                        "pair_fit_fact_count_v1": 0,
                        "pair_friction_fact_count_v1": 0,
                        "pair_evidence_fact_count_v1": 0,
                        "pair_stable_fit_fact_count_v1": 0,
                        "pair_recent_fit_fact_count_v1": 0,
                        "pair_history_fit_fact_count_v1": 0,
                        "pair_practical_fit_fact_count_v1": 0,
                        "pair_fit_scope_count_v1": 0,
                        "pair_has_visible_user_fit_v1": 0,
                        "pair_has_visible_conflict_v1": 0,
                        "pair_has_recent_context_visible_bridge_v1": 0,
                        "pair_has_detail_support_v1": 0,
                        "pair_has_multisource_fit_v1": 0,
                        "pair_has_candidate_fit_support_v1": 0,
                        "pair_has_candidate_strong_fit_support_v1": 0,
                        "pair_has_contrastive_support_v1": 0,
                        "pair_fact_signal_count_v1": 0,
                        "boundary_constructability_class_v1": "",
                        "boundary_constructability_reason_codes_v1": "[]",
                        "boundary_prompt_ready_v1": 0,
                        "boundary_rival_total_v1": 0,
                        "boundary_rival_head_or_boundary_v1": 0,
                        "user_evidence_text": "",
                        "history_anchor_text": "",
                        "item_evidence_text": "",
                            "pair_evidence_summary": "",
                        }
                    )
                    .withColumn(
                        "pair_evidence_summary",
                        F.when(
                            F.length(F.trim(F.coalesce(F.col("pair_evidence_summary"), F.lit("")))) > F.lit(0),
                            F.col("pair_evidence_summary"),
                        ).otherwise(F.coalesce(F.col("evidence_basis_text_v1"), F.lit(""))),
                    )
                    .withColumn("prompt", F.lit(""))
                )
                print(f"[STEP] bucket={bucket} pairwise_pool late semantic text ready")
            else:
                raise RuntimeError("pairwise_pool non-minimal export path disabled; set QLORA_PAIRWISE_POOL_MINIMAL_EXPORT=true")

            print(f"[STEP] bucket={bucket} assign pairwise_pool split")
            pairwise_pool_df = (
                with_split_assignment(
                    pool_materialized,
                    eval_user_frac=float(EVAL_USER_FRAC),
                    fixed_eval_users=fixed_eval_users_df,
                    fixed_eval_join_cols=fixed_eval_join_cols,
                )
                .withColumn("target_text", F.when(F.col("label").cast("int") == F.lit(1), F.lit("YES")).otherwise(F.lit("NO")))
                .withColumn("bucket", F.lit(int(bucket)))
                .select(
                    "bucket",
                    "user_idx",
                    "item_idx",
                    "business_id",
                    "label",
                    "sample_weight",
                    "label_source",
                    "neg_tier",
                    "neg_pick_rank",
                    "neg_is_near",
                    "neg_is_hard",
                    "pre_rank",
                    "pre_rank_band",
                    "pre_score",
                    "prompt",
                    "target_text",
                    "profile_text",
                    "profile_text_evidence",
                    "profile_text_short",
                    "profile_text_long",
                    "profile_pos_text",
                    "profile_neg_text",
                    "profile_top_pos_tags",
                    "profile_top_neg_tags",
                    "profile_top_pos_tags_by_type",
                    "profile_top_neg_tags_by_type",
                    "profile_confidence",
                    "user_long_pref_text",
                    "user_recent_intent_text",
                    "user_negative_avoid_text",
                    "user_context_text",
                    "stable_preferences_text",
                    "recent_intent_text_v2",
                    "avoidance_text_v2",
                    "history_anchor_hint_text",
                    "user_semantic_profile_text_v2",
                    "user_profile_richness_v2",
                    "user_profile_richness_tier_v2",
                    "user_semantic_facts_v1",
                    "user_visible_fact_count_v1",
                    "user_focus_fact_count_v1",
                    "user_recent_fact_count_v1",
                    "user_avoid_fact_count_v1",
                    "user_history_fact_count_v1",
                    "user_evidence_fact_count_v1",
                    "core_offering_text",
                    "scene_fit_text",
                    "strengths_text",
                    "risk_points_text",
                    "merchant_semantic_profile_text_v2",
                    "merchant_profile_richness_v2",
                    "merchant_semantic_facts_v1",
                    "merchant_visible_fact_count_v1",
                    "merchant_core_fact_count_v1",
                    "merchant_scene_fact_count_v1",
                    "merchant_strength_fact_count_v1",
                    "merchant_risk_fact_count_v1",
                    "fit_reasons_text_v1",
                    "friction_reasons_text_v1",
                    "evidence_basis_text_v1",
                    "pair_alignment_richness_v1",
                    "semantic_prompt_readiness_tier_v1",
                    "pair_alignment_facts_v1",
                    "pair_fit_fact_count_v1",
                    "pair_friction_fact_count_v1",
                    "pair_evidence_fact_count_v1",
                    "pair_stable_fit_fact_count_v1",
                    "pair_recent_fit_fact_count_v1",
                    "pair_history_fit_fact_count_v1",
                    "pair_practical_fit_fact_count_v1",
                    "pair_fit_scope_count_v1",
                    "pair_has_visible_user_fit_v1",
                    "pair_has_visible_conflict_v1",
                    "pair_has_recent_context_visible_bridge_v1",
                    "pair_has_detail_support_v1",
                    "pair_has_multisource_fit_v1",
                    "pair_has_candidate_fit_support_v1",
                    "pair_has_candidate_strong_fit_support_v1",
                    "pair_has_contrastive_support_v1",
                    "pair_fact_signal_count_v1",
                    "boundary_constructability_class_v1",
                    "boundary_constructability_reason_codes_v1",
                    "boundary_prompt_ready_v1",
                    "boundary_rival_total_v1",
                    "boundary_rival_head_or_boundary_v1",
                    "user_evidence_text",
                    "history_anchor_text",
                    "item_evidence_text",
                    "pair_evidence_summary",
                    *DPO_SUPPORT_EXPORT_COLUMNS,
                    "split",
                )
            ).persist(StorageLevel.DISK_ONLY)
            print(f"[STEP] bucket={bucket} pairwise_pool split ready")
            print(f"[STEP] bucket={bucket} pairwise_pool_df ready")

            if not SKIP_PAIRWISE_POOL_JSON_EXPORT:
                print(f"[STEP] bucket={bucket} write pairwise_pool json")
                coalesce_for_write(
                    pairwise_pool_df.filter(F.col("split") == F.lit("train")),
                    PAIRWISE_POOL_WRITE_JSON_PARTITIONS,
                ).write.mode("overwrite").json(pairwise_pool_train_dir.as_posix())
                coalesce_for_write(
                    pairwise_pool_df.filter(F.col("split") == F.lit("eval")),
                    PAIRWISE_POOL_WRITE_JSON_PARTITIONS,
                ).write.mode("overwrite").json(pairwise_pool_eval_dir.as_posix())
                print(f"[STEP] bucket={bucket} pairwise_pool json ready")
            print(f"[STEP] bucket={bucket} write pairwise_pool parquet")
            coalesce_for_write(
                pairwise_pool_df,
                PAIRWISE_POOL_WRITE_PARQUET_PARTITIONS,
            ).write.mode("overwrite").parquet(pairwise_pool_parquet_dir.as_posix())
            print(f"[STEP] bucket={bucket} pairwise_pool parquet ready")

            print(f"[STEP] bucket={bucket} collect pairwise_pool summary")
            pair_pool_summary, pair_pool_summary_error = collect_split_label_user_summary_safe(
                pairwise_pool_df,
                include_split_user_counts=True,
            )
            pool_neg_tier_counts: dict[str, int] = {}
            pool_neg_tier_counts_error = ""
            try:
                for r in pairwise_pool_df.filter(F.col("label") == F.lit(0)).groupBy("neg_tier").count().collect():
                    pool_neg_tier_counts[str(r["neg_tier"])] = int(r["count"])
            except Exception as exc:
                pool_neg_tier_counts_error = f"{type(exc).__name__}: {exc}"
            print(f"[STEP] bucket={bucket} pairwise_pool summary ready")
            pairwise_pool_stat = {
                "enabled": True,
                "topn_per_user": int(PAIRWISE_POOL_TOPN),
                "rows_total": int(pair_pool_summary.get("rows_total", 0)),
                "rows_train": int(pair_pool_summary.get("train_label_0", 0) + pair_pool_summary.get("train_label_1", 0)),
                "rows_eval": int(pair_pool_summary.get("eval_label_0", 0) + pair_pool_summary.get("eval_label_1", 0)),
                "train_pos": int(pair_pool_summary.get("train_label_1", 0)),
                "train_neg": int(pair_pool_summary.get("train_label_0", 0)),
                "eval_pos": int(pair_pool_summary.get("eval_label_1", 0)),
                "eval_neg": int(pair_pool_summary.get("eval_label_0", 0)),
                "users_total": int(pair_pool_summary.get("users_total", 0)),
                "train_users": int(pair_pool_summary.get("train_users", 0)),
                "eval_users": int(pair_pool_summary.get("eval_users", 0)),
                "neg_tier_counts": pool_neg_tier_counts,
                "summary_collection_error": pair_pool_summary_error,
                "neg_tier_collection_error": pool_neg_tier_counts_error,
                "output_train_dir": str(pairwise_pool_train_dir) if not SKIP_PAIRWISE_POOL_JSON_EXPORT else "",
                "output_eval_dir": str(pairwise_pool_eval_dir) if not SKIP_PAIRWISE_POOL_JSON_EXPORT else "",
                "output_parquet_dir": str(pairwise_pool_parquet_dir),
            }
            pairwise_pool_df.unpersist(blocking=False)
            pool_join_base_core.unpersist(blocking=False)
            print(
                f"[DONE] bucket={bucket} pairwise_pool "
                f"rows_train={pairwise_pool_stat['rows_train']} rows_eval={pairwise_pool_stat['rows_eval']}"
            )

        summary.append(
            {
                "bucket": int(bucket),
                "candidate_file": cand_file.name,
                "rows_total": int(sum(split_stat.values())),
                "rows_train": int(split_stat.get("train_label_0", 0) + split_stat.get("train_label_1", 0)),
                "rows_eval": int(split_stat.get("eval_label_0", 0) + split_stat.get("eval_label_1", 0)),
                "train_pos": int(split_stat.get("train_label_1", 0)),
                "train_neg": int(split_stat.get("train_label_0", 0)),
                "eval_pos": int(split_stat.get("eval_label_1", 0)),
                "eval_neg": int(split_stat.get("eval_label_0", 0)),
                "users_total": users_total,
                "users_with_positive": users_with_pos,
                "users_no_positive": users_no_pos,
                "users_no_positive_ratio": users_no_pos_ratio,
                "dataset_eval_split_mode": "explicit_cohort" if fixed_eval_users_df is not None else "hash_frac",
                "dataset_eval_user_count_requested": int(fixed_eval_user_count),
                "dataset_eval_user_join_cols": fixed_eval_join_cols,
                "dataset_eval_user_count_surviving": int(out_summary.get("eval_users", 0)),
                "dataset_eval_user_count_dropped": int(max(0, int(fixed_eval_user_count) - int(out_summary.get("eval_users", 0)))),
                "neg_tier_counts": neg_tier_counts,
                "summary_collection_error": out_summary_error,
                "neg_tier_collection_error": neg_tier_counts_error,
                "user_evidence_dir": str(user_evidence_dir) if not SKIP_POINTWISE_EXPORT else "",
                "item_evidence_dir": str(item_evidence_dir) if not SKIP_POINTWISE_EXPORT else "",
                "pair_evidence_audit_dir": str(pair_audit_dir) if not SKIP_POINTWISE_EXPORT else "",
                "rich_sft": rich_sft_stat,
                "pairwise_pool": pairwise_pool_stat,
                "output_bucket_dir": str(b_out),
            }
        )
        print(f"[DONE] bucket={bucket} stats={split_stat}")
        if out_df is not None:
            out_df.unpersist(blocking=False)
        if enrich is not None:
            enrich.unpersist(blocking=False)

        truth.unpersist()

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage09_run": str(source_09),
        "buckets_override": sorted(list(wanted)) if wanted else [],
        "buckets_processed": processed_buckets,
        "topn_per_user": int(TOPN_PER_USER),
        "candidate_rerank_topn_per_user": int(TOPN_PER_USER),
        "prompt_mode": str(PROMPT_MODE),
        "audit_only": bool(AUDIT_ONLY),
        "skip_pointwise_export": bool(SKIP_POINTWISE_EXPORT),
        "skip_rich_sft_json_export": bool(SKIP_RICH_SFT_JSON_EXPORT),
        "skip_pairwise_pool_json_export": bool(SKIP_PAIRWISE_POOL_JSON_EXPORT),
        "enable_rich_sft_export": bool(ENABLE_RICH_SFT_EXPORT),
        "rich_sft_history_anchor_max_per_user": int(RICH_SFT_HISTORY_ANCHOR_MAX_PER_USER),
        "rich_sft_history_anchor_primary_min_rating": float(RICH_SFT_HISTORY_ANCHOR_PRIMARY_MIN_RATING),
        "rich_sft_history_anchor_fallback_min_rating": float(RICH_SFT_HISTORY_ANCHOR_FALLBACK_MIN_RATING),
        "rich_sft_history_anchor_max_chars": int(RICH_SFT_HISTORY_ANCHOR_MAX_CHARS),
        "rich_sft_neg_explicit": int(RICH_SFT_NEG_EXPLICIT),
        "rich_sft_neg_hard": int(RICH_SFT_NEG_HARD),
        "rich_sft_neg_near": int(RICH_SFT_NEG_NEAR),
        "rich_sft_neg_mid": int(RICH_SFT_NEG_MID),
        "rich_sft_neg_tail": int(RICH_SFT_NEG_TAIL),
        "rich_sft_neg_hard_rank_max": int(RICH_SFT_NEG_HARD_RANK_MAX),
        "rich_sft_neg_mid_rank_max": int(RICH_SFT_NEG_MID_RANK_MAX),
        "rich_sft_neg_max_rating": float(RICH_SFT_NEG_MAX_RATING),
        "rich_sft_allow_neg_fill": bool(RICH_SFT_ALLOW_NEG_FILL),
        "enable_pairwise_pool_export": bool(ENABLE_PAIRWISE_POOL_EXPORT),
        "pairwise_pool_topn": int(PAIRWISE_POOL_TOPN),
        "pairwise_pool_rank_limit": int(PAIRWISE_POOL_TOPN),
        "pairwise_pool_minimal_export": bool(PAIRWISE_POOL_MINIMAL_EXPORT),
        "neg_per_user": int(NEG_PER_USER),
        "neg_layered_enabled": bool(NEG_LAYERED_ENABLED),
        "neg_sampler_mode": str(NEG_SAMPLER_MODE),
        "neg_hard_ratio": float(NEG_HARD_RATIO),
        "neg_near_ratio": float(NEG_NEAR_RATIO),
        "neg_hard_rank_max": int(NEG_HARD_RANK_MAX),
        "neg_hard_weight": float(NEG_HARD_WEIGHT),
        "neg_near_weight": float(NEG_NEAR_WEIGHT),
        "neg_easy_weight": float(NEG_EASY_WEIGHT),
        "neg_fill_weight": float(NEG_FILL_WEIGHT),
        "neg_band_top10": int(NEG_BAND_TOP10),
        "neg_band_11_30": int(NEG_BAND_11_30),
        "neg_band_31_80": int(NEG_BAND_31_80),
        "neg_band_81_150": int(NEG_BAND_81_150),
        "neg_band_near": int(NEG_BAND_NEAR),
        "user_cap_randomized": bool(USER_CAP_RANDOMIZED),
        "include_valid_pos": bool(INCLUDE_VALID_POS),
        "valid_pos_weight": float(VALID_POS_WEIGHT),
        "valid_pos_only_if_no_true": bool(VALID_POS_ONLY_IF_NO_TRUE),
        "eval_user_frac": float(EVAL_USER_FRAC),
        "dataset_eval_user_cohort_path": str(dataset_eval_user_cohort_path) if dataset_eval_user_cohort_path is not None else "",
        "dataset_eval_user_count_requested": int(fixed_eval_user_count),
        "dataset_eval_user_join_cols": fixed_eval_join_cols,
        "dataset_eval_split_mode": "explicit_cohort" if fixed_eval_users_df is not None else "hash_frac",
        "structured_feature_prompt_enabled": True,
        "short_text_summary_enabled": bool(ENABLE_SHORT_TEXT_SUMMARY),
        "raw_review_text_enabled": bool(ENABLE_RAW_REVIEW_TEXT),
        "review_snippet_max_chars": int(REVIEW_SNIPPET_MAX_CHARS),
        "user_evidence_max_chars": int(USER_EVIDENCE_MAX_CHARS),
        "item_evidence_max_chars": int(ITEM_EVIDENCE_MAX_CHARS),
        "user_review_topn": int(USER_REVIEW_TOPN),
        "item_review_topn": int(ITEM_REVIEW_TOPN),
        "row_cap_ordered": bool(ROW_CAP_ORDERED),
        "enforce_stage09_gate": bool(ENFORCE_STAGE09_GATE),
        "stage09_gate_result": gate_result,
        "cluster_profile_csv": str(meta.get("cluster_profile_csv", "")),
        "summary": summary,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out_dir / 'run_meta.json'}")
    pointer_path = write_latest_run_pointer(
        "stage11_1_qlora_build_dataset",
        out_dir,
        extra={
            "run_tag": RUN_TAG,
            "source_run_09": str(source_09),
            "source_stage09_run": str(source_09),
            "buckets": processed_buckets,
        },
    )
    print(f"[INFO] updated latest pointer: {pointer_path}")
    spark.stop()


if __name__ == "__main__":
    main()

