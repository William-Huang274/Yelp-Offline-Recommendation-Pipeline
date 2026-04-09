from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from datetime import datetime
import heapq
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window
from pipeline.project_paths import env_or_project_path, project_path
from pipeline.spark_tmp_manager import (
    SparkTmpContext,
    alloc_scratch_file,
    build_spark_tmp_context,
)
from pipeline.local_parquet_writer import write_spark_df_to_parquet_dir


# Run mode
RUN_PROFILE = os.getenv("RUN_PROFILE_OVERRIDE", "full").strip().lower() or "full"  # "sample" | "full"
RUN_TAG = "stage09_candidate_fusion"
RECALL_PROFILE = os.getenv("RECALL_PROFILE_OVERRIDE", "balanced").strip().lower() or "balanced"

# Data paths
PARQUET_BASE = env_or_project_path("PARQUET_BASE_DIR", "data/parquet")
OUTPUT_ROOT = env_or_project_path("OUTPUT_ROOT_DIR", "data/output/09_candidate_fusion")
CLUSTER_PROFILE_ROOT = env_or_project_path("CLUSTER_PROFILE_ROOT_DIR", "data/output/08_cluster_labels/full")
CLUSTER_PROFILE_DIR_SUFFIX = "_full_profile_merged"
CLUSTER_PROFILE_FILENAME = "biz_profile_recsys.csv"
CLUSTER_PROFILE_CLUSTER_COL = "cluster_for_recsys"
CLUSTER_PROFILE_CSV = ""  # optional explicit path
USER_PROFILE_ROOT = env_or_project_path("USER_PROFILE_ROOT_DIR", "data/output/09_user_profiles")
USER_PROFILE_RUN_DIR = os.getenv("USER_PROFILE_RUN_DIR", "").strip()  # optional explicit path
USER_PROFILE_RUN_SUFFIX = "_stage09_user_profile_build"
USER_PROFILE_VECTOR_FILE = "user_profile_vectors.npz"
USER_PROFILE_MULTI_VECTOR_FILE_TEMPLATE = "user_profile_vectors_{scope}.npz"
USER_PROFILE_TABLE_FILE = "user_profiles.csv"
USER_PROFILE_TAG_LONG_FILE = "user_profile_tag_profile_long.csv"
ENABLE_PROFILE_RECALL = os.getenv("ENABLE_PROFILE_RECALL", "true").strip().lower() == "true"
ENABLE_PROFILE_VECTOR_ROUTE = os.getenv("ENABLE_PROFILE_VECTOR_ROUTE", "true").strip().lower() == "true"
ENABLE_TAG_SHARED_ROUTE = os.getenv("ENABLE_TAG_SHARED_ROUTE", "false").strip().lower() == "true"
ENABLE_BRIDGE_USER_ROUTE = os.getenv("ENABLE_BRIDGE_USER_ROUTE", "false").strip().lower() == "true"
ENABLE_BRIDGE_TYPE_ROUTE = os.getenv("ENABLE_BRIDGE_TYPE_ROUTE", "false").strip().lower() == "true"
PROFILE_ROUTE_POLICY_JSON = os.getenv("PROFILE_ROUTE_POLICY_JSON", "").strip()
PROFILE_ROUTE_THRESHOLDS_JSON = os.getenv("PROFILE_ROUTE_THRESHOLDS_JSON", "").strip()
PROFILE_CALIBRATION_JSON_OVERRIDE = os.getenv("PROFILE_CALIBRATION_JSON_OVERRIDE", "").strip()
RECALL_LIMITS_BY_BUCKET_JSON = os.getenv("RECALL_LIMITS_BY_BUCKET_JSON", "").strip()
LAYERED_PRETRIM_BY_BUCKET_JSON = os.getenv("LAYERED_PRETRIM_BY_BUCKET_JSON", "").strip()
PRETRIM_SEGMENT_TOPK_BY_BUCKET_JSON = os.getenv("PRETRIM_SEGMENT_TOPK_BY_BUCKET_JSON", "").strip()
SOURCE_WEIGHTS_BY_BUCKET_JSON = os.getenv("SOURCE_WEIGHTS_BY_BUCKET_JSON", "").strip()
PRETRIM_HEAD_POLICY_JSON = os.getenv("PRETRIM_HEAD_POLICY_JSON", "").strip()
ITEM_SEMANTIC_ROOT = env_or_project_path("ITEM_SEMANTIC_ROOT_DIR", "data/output/09_item_semantics")
ITEM_SEMANTIC_RUN_DIR = os.getenv("ITEM_SEMANTIC_RUN_DIR", "").strip()  # optional explicit path
ITEM_SEMANTIC_RUN_SUFFIX = "_stage09_item_semantic_build"
ITEM_SEMANTIC_FEATURE_FILE = "item_semantic_features.csv"
ITEM_SEMANTIC_TAG_LONG_FILE = "item_tag_profile_long.csv"
ITEM_SEMANTIC_DENSE_VECTOR_FILE_TEMPLATE = "merchant_dense_vectors_{scope}.npz"
BUSINESS_CONTEXT_ROOT = env_or_project_path("BUSINESS_CONTEXT_ROOT_DIR", "data/output/09_business_context")
BUSINESS_CONTEXT_RUN_DIR = os.getenv("BUSINESS_CONTEXT_RUN_DIR", "").strip()
BUSINESS_CONTEXT_RUN_SUFFIX = "_stage09_business_context_build"
BUSINESS_CONTEXT_FEATURE_FILE = "business_context_features.parquet"
REVIEW_EVIDENCE_ROOT = env_or_project_path("REVIEW_EVIDENCE_ROOT_DIR", "data/output/09_review_evidence")
REVIEW_EVIDENCE_RUN_DIR = os.getenv("REVIEW_EVIDENCE_RUN_DIR", "").strip()
REVIEW_EVIDENCE_RUN_SUFFIX = "_stage09_review_evidence_weight_build"
REVIEW_EVIDENCE_BUSINESS_FILE = "business_review_evidence_agg.parquet"
TIP_SIGNAL_ROOT = env_or_project_path("TIP_SIGNAL_ROOT_DIR", "data/output/09_tip_signals")
TIP_SIGNAL_RUN_DIR = os.getenv("TIP_SIGNAL_RUN_DIR", "").strip()
TIP_SIGNAL_RUN_SUFFIX = "_stage09_tip_signal_build"
TIP_SIGNAL_BUSINESS_FILE = "business_tip_signal_agg.parquet"
CHECKIN_CONTEXT_ROOT = env_or_project_path("CHECKIN_CONTEXT_ROOT_DIR", "data/output/09_checkin_context")
CHECKIN_CONTEXT_RUN_DIR = os.getenv("CHECKIN_CONTEXT_RUN_DIR", "").strip()
CHECKIN_CONTEXT_RUN_SUFFIX = "_stage09_checkin_context_build"
CHECKIN_CONTEXT_BUSINESS_FILE = "business_checkin_context_agg.parquet"
ENABLE_ITEM_SEMANTIC = os.getenv("ENABLE_ITEM_SEMANTIC", "true").strip().lower() == "true"
ENABLE_CONTEXT_SURFACE = os.getenv("ENABLE_CONTEXT_SURFACE", "false").strip().lower() == "true"
CONTEXT_SURFACE_BUCKETS_RAW = os.getenv("CONTEXT_SURFACE_BUCKETS", "5").strip()
CONTEXT_SURFACE_TOP_K = int(os.getenv("CONTEXT_SURFACE_TOP_K", "72").strip() or 72)
CONTEXT_SURFACE_SCORE_MIN = float(os.getenv("CONTEXT_SURFACE_SCORE_MIN", "0.18").strip() or 0.18)
CONTEXT_SURFACE_BATCH_USERS = int(os.getenv("CONTEXT_SURFACE_BATCH_USERS", "256").strip() or 256)
CONTEXT_GEO_SCALE_KM = float(os.getenv("CONTEXT_GEO_SCALE_KM", "18.0").strip() or 18.0)
CONTEXT_ROUTE_DETAIL_RAW = os.getenv("CONTEXT_ROUTE_DETAILS", "attr,time,geo").strip()
ENABLE_PROFILE_MULTIVECTOR_ROUTE_AUDIT = (
    os.getenv("ENABLE_PROFILE_MULTIVECTOR_ROUTE_AUDIT", "true").strip().lower() == "true"
)
PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS_RAW = os.getenv("PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS", "5,10").strip()
PROFILE_MULTIVECTOR_ROUTE_TOP_K = int(os.getenv("PROFILE_MULTIVECTOR_ROUTE_TOP_K", "120").strip() or 120)
PROFILE_MULTIVECTOR_ROUTE_SCORE_MIN = float(
    os.getenv("PROFILE_MULTIVECTOR_ROUTE_SCORE_MIN", "0.0").strip() or 0.0
)
PROFILE_MULTIVECTOR_ROUTE_BATCH_USERS = int(
    os.getenv("PROFILE_MULTIVECTOR_ROUTE_BATCH_USERS", "128").strip() or 128
)
ENABLE_PROFILE_MULTIVECTOR_ROUTE_INTEGRATION = (
    os.getenv("ENABLE_PROFILE_MULTIVECTOR_ROUTE_INTEGRATION", "false").strip().lower() == "true"
)
PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS_RAW = os.getenv(
    "PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS", "5"
).strip()
PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_TOP_K = int(
    os.getenv("PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_TOP_K", "80").strip() or 80
)
PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_SCORE_MIN = float(
    os.getenv("PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_SCORE_MIN", "0.25").strip() or 0.25
)
PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BATCH_USERS = int(
    os.getenv("PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BATCH_USERS", "128").strip() or 128
)
ENABLE_PROFILE_EVIDENCE_PROTECTED_LANE = (
    os.getenv("ENABLE_PROFILE_EVIDENCE_PROTECTED_LANE", "false").strip().lower() == "true"
)
PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS_RAW = os.getenv(
    "PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS", "5"
).strip()
PROFILE_EVIDENCE_PROTECTED_LANE_QUOTA = int(
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_QUOTA", "72").strip() or 72
)
PROFILE_EVIDENCE_PROTECTED_LANE_MAX_PRE_RANK = int(
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_MAX_PRE_RANK", "320").strip() or 320
)
PROFILE_EVIDENCE_PROTECTED_LANE_MAX_POP_COUNT = int(
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_MAX_POP_COUNT", "140").strip() or 140
)
PROFILE_EVIDENCE_PROTECTED_LANE_MIN_MV_ROUTES = int(
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_MIN_MV_ROUTES", "1").strip() or 1
)
PROFILE_EVIDENCE_PROTECTED_LANE_MIN_SEMANTIC_SCORE = float(
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_MIN_SEMANTIC_SCORE", "0.82").strip() or 0.82
)
PROFILE_EVIDENCE_PROTECTED_LANE_MIN_PROFILE_CLUSTER = int(
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_MIN_PROFILE_CLUSTER", "1").strip() or 1
)
PROFILE_EVIDENCE_PROTECTED_LANE_EXCLUDE_POPULAR = (
    os.getenv("PROFILE_EVIDENCE_PROTECTED_LANE_EXCLUDE_POPULAR", "false").strip().lower() == "true"
)
ENABLE_BUCKET_DEBIAS_REWRITE = (
    os.getenv("ENABLE_BUCKET_DEBIAS_REWRITE", "false").strip().lower() == "true"
)
DEBIAS_BUCKETS_RAW = os.getenv("DEBIAS_BUCKETS", "5").strip()
DEBIAS_QUALITY_REVIEW_POWER = float(
    os.getenv("DEBIAS_QUALITY_REVIEW_POWER", "0.35").strip() or 0.35
)
DEBIAS_ITEM_POP_CAP = float(os.getenv("DEBIAS_ITEM_POP_CAP", "400").strip() or 400.0)
DEBIAS_ALS_BACKBONE_MIN_MULT = float(
    os.getenv("DEBIAS_ALS_BACKBONE_MIN_MULT", "0.55").strip() or 0.55
)
DEBIAS_ALS_BACKBONE_POP_PENALTY = float(
    os.getenv("DEBIAS_ALS_BACKBONE_POP_PENALTY", "0.45").strip() or 0.45
)
DEBIAS_NICHE_SEMANTIC_BONUS_WEIGHT = float(
    os.getenv("DEBIAS_NICHE_SEMANTIC_BONUS_WEIGHT", "0.025").strip() or 0.025
)
DEBIAS_QUALITY_BASE_FLOOR = float(
    os.getenv("DEBIAS_QUALITY_BASE_FLOOR", "0.72").strip() or 0.72
)
DEBIAS_SEMANTIC_SUPPORT_POP_ALPHA = float(
    os.getenv("DEBIAS_SEMANTIC_SUPPORT_POP_ALPHA", "0.65").strip() or 0.65
)
DEBIAS_SEMANTIC_SUPPORT_ADJ_CAP = float(
    os.getenv("DEBIAS_SEMANTIC_SUPPORT_ADJ_CAP", "140").strip() or 140.0
)
DEBIAS_SEMANTIC_SUPPORT_GATE_FLOOR = float(
    os.getenv("DEBIAS_SEMANTIC_SUPPORT_GATE_FLOOR", "0.18").strip() or 0.18
)
DEBIAS_ALS_BACKBONE_HEAVY_POP_PENALTY = float(
    os.getenv("DEBIAS_ALS_BACKBONE_HEAVY_POP_PENALTY", "0.18").strip() or 0.18
)
DEBIAS_ALS_BACKBONE_UNCORROBORATED_PENALTY = float(
    os.getenv("DEBIAS_ALS_BACKBONE_UNCORROBORATED_PENALTY", "0.16").strip() or 0.16
)
DEBIAS_NICHE_PROFILE_CLUSTER_BONUS = float(
    os.getenv("DEBIAS_NICHE_PROFILE_CLUSTER_BONUS", "0.012").strip() or 0.012
)
ENABLE_PERSONALIZED_CHALLENGER_SURFACE = (
    os.getenv("ENABLE_PERSONALIZED_CHALLENGER_SURFACE", "false").strip().lower() == "true"
)
PERSONALIZED_CHALLENGER_BUCKETS_RAW = os.getenv(
    "PERSONALIZED_CHALLENGER_BUCKETS", "5"
).strip()
PERSONALIZED_CHALLENGER_QUOTA = int(
    os.getenv("PERSONALIZED_CHALLENGER_QUOTA", "96").strip() or 96
)
PERSONALIZED_CHALLENGER_MAX_PRE_RANK = int(
    os.getenv("PERSONALIZED_CHALLENGER_MAX_PRE_RANK", "320").strip() or 320
)
PERSONALIZED_CHALLENGER_HEAD_WINDOW = int(
    os.getenv("PERSONALIZED_CHALLENGER_HEAD_WINDOW", "250").strip() or 250
)
PERSONALIZED_CHALLENGER_HEAD_BONUS = float(
    os.getenv("PERSONALIZED_CHALLENGER_HEAD_BONUS", "0.0").strip() or 0.0
)
PERSONALIZED_CHALLENGER_MAX_POP_COUNT = int(
    os.getenv("PERSONALIZED_CHALLENGER_MAX_POP_COUNT", "180").strip() or 180
)
PERSONALIZED_CHALLENGER_MIN_MV_ROUTES = int(
    os.getenv("PERSONALIZED_CHALLENGER_MIN_MV_ROUTES", "1").strip() or 1
)
PERSONALIZED_CHALLENGER_MIN_PROFILE_CLUSTER = int(
    os.getenv("PERSONALIZED_CHALLENGER_MIN_PROFILE_CLUSTER", "1").strip() or 1
)
PERSONALIZED_CHALLENGER_MIN_NONPOPULAR_SOURCES = int(
    os.getenv("PERSONALIZED_CHALLENGER_MIN_NONPOPULAR_SOURCES", "1").strip() or 1
)
PERSONALIZED_CHALLENGER_MIN_SEMANTIC_SCORE = float(
    os.getenv("PERSONALIZED_CHALLENGER_MIN_SEMANTIC_SCORE", "0.80").strip() or 0.80
)
PERSONALIZED_CHALLENGER_EXCLUDE_POPULAR = (
    os.getenv("PERSONALIZED_CHALLENGER_EXCLUDE_POPULAR", "false").strip().lower() == "true"
)
PROFILE_MULTIVECTOR_ROUTE_SCOPES = tuple(
    x.strip().lower()
    for x in os.getenv("PROFILE_MULTIVECTOR_ROUTE_SCOPES", "short,long,pos,neg").split(",")
    if x.strip()
)
ENABLE_TOWER_SEQ_FEATURES = os.getenv("ENABLE_TOWER_SEQ_FEATURES", "true").strip().lower() == "true"
TOWER_SEQ_MAX_CAND_ROWS = int(os.getenv("TOWER_SEQ_MAX_CAND_ROWS", "2500000").strip() or 2500000)
TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET_JSON = os.getenv("TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET_JSON", "").strip()
TOWER_SEQ_HEAD_WINDOW_BY_BUCKET_JSON = os.getenv("TOWER_SEQ_HEAD_WINDOW_BY_BUCKET_JSON", "").strip()
TOWER_SEQ_DIM = int(os.getenv("TOWER_SEQ_DIM", "64").strip() or 64)
TOWER_SEQ_EPOCHS = int(os.getenv("TOWER_SEQ_EPOCHS", "6").strip() or 6)
TOWER_SEQ_LR = float(os.getenv("TOWER_SEQ_LR", "0.04").strip() or 0.04)
TOWER_SEQ_REG = float(os.getenv("TOWER_SEQ_REG", "0.0001").strip() or 0.0001)
TOWER_SEQ_NEG_TRIES = int(os.getenv("TOWER_SEQ_NEG_TRIES", "6").strip() or 6)
TOWER_SEQ_STEPS_PER_EPOCH = int(os.getenv("TOWER_SEQ_STEPS_PER_EPOCH", "0").strip() or 0)
TOWER_SEQ_OUTPUT_MODE = (os.getenv("TOWER_SEQ_OUTPUT_MODE", "csv").strip().lower() or "csv")
TOWER_SEQ_APPLY_HEAD_WINDOW_FIRST = (
    os.getenv("TOWER_SEQ_APPLY_HEAD_WINDOW_FIRST", "false").strip().lower() == "true"
)
PROFILE_OUTPUT_MODE = (os.getenv("PROFILE_OUTPUT_MODE", "csv").strip().lower() or "csv")
PROFILE_OUTPUT_SPARK_DF_MAX_ROWS = int(os.getenv("PROFILE_OUTPUT_SPARK_DF_MAX_ROWS", "800000").strip() or 800000)
PANDAS_TO_SPARK_OUTPUT_PARTITIONS = int(os.getenv("PANDAS_TO_SPARK_OUTPUT_PARTITIONS", "4").strip() or 4)
ALLOW_PROFILE_CSV_FALLBACK = os.getenv("ALLOW_PROFILE_CSV_FALLBACK", "true").strip().lower() == "true"
ALLOW_TOWER_SEQ_CSV_FALLBACK = os.getenv("ALLOW_TOWER_SEQ_CSV_FALLBACK", "true").strip().lower() == "true"
PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_USERS = (
    os.getenv("PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_USERS", "true").strip().lower() == "true"
)
PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_ITEMS = (
    os.getenv("PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_ITEMS", "false").strip().lower() == "true"
)
TOWER_SEQ_SCOPE_USERS_ONLY = os.getenv("TOWER_SEQ_SCOPE_USERS_ONLY", "true").strip().lower() == "true"
TOWER_SEQ_SCOPE_ITEMS_ONLY = os.getenv("TOWER_SEQ_SCOPE_ITEMS_ONLY", "false").strip().lower() == "true"
TOWER_SEQ_TRAIN_MAX_PER_USER = int(os.getenv("TOWER_SEQ_TRAIN_MAX_PER_USER", "120").strip() or 120)
SEQ_RECENT_LEN = int(os.getenv("SEQ_RECENT_LEN", "10").strip() or 10)
SEQ_RECENT_MIN_LEN = int(os.getenv("SEQ_RECENT_MIN_LEN", "2").strip() or 2)
SEQ_DECAY = float(os.getenv("SEQ_DECAY", "0.85").strip() or 0.85)

# Dataset scope
TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
MIN_TRAIN_REVIEWS_BUCKETS = [2, 5, 10]
MIN_USER_REVIEWS_OFFSET = 2
RANDOM_SEED = 42

# Hard filters (keep in one place for maintainability)
FILTER_POLICY: dict[str, Any] = {
    "require_is_open": True,
    "min_business_stars": 3.0,
    "min_business_review_count": 20,
    "stale_cutoff_date": "2020-01-01",  # use last review time
}

# ALS + candidate generation
IMPLICIT_PARAMS_BY_BUCKET = {
    2: {"rank": 20, "reg": 0.1, "alpha": 20.0},
    5: {"rank": 20, "reg": 0.1, "alpha": 20.0},
    10: {"rank": 20, "reg": 0.1, "alpha": 20.0},
}
ALS_TOP_K = 200
CLUSTER_TOP_K = 80
POPULAR_TOP_K = 50
PROFILE_TOP_K = 80
PRETRIM_TOP_K = 150
TOP_K_EVAL = 10
CLUSTER_USER_TOPN = 3
RECALL_LIMITS_BY_BUCKET: dict[int, dict[str, int]] = {}
PRETRIM_SEGMENT_TOPK_BY_BUCKET: dict[int, dict[str, int]] = {}
TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET: dict[int, int] = {}
TOWER_SEQ_HEAD_WINDOW_BY_BUCKET: dict[int, int] = {}
PROFILE_ROUTE_POLICY_DEFAULT_BY_BUCKET: dict[int, dict[str, Any]] = {}
PROFILE_ROUTE_THRESHOLDS_DEFAULT_BY_BUCKET: dict[int, dict[str, Any]] = {}
PRETRIM_HEAD_POLICY_DEFAULT_BY_BUCKET: dict[int, dict[str, Any]] = {}
if RECALL_PROFILE == "coverage_first":
    ALS_TOP_K = 300
    CLUSTER_TOP_K = 150
    POPULAR_TOP_K = 80
    PROFILE_TOP_K = 150
    PRETRIM_TOP_K = 250
    CLUSTER_USER_TOPN = 5
    # Lower-data buckets get larger recall pools to improve truth-in-candidate rate.
    RECALL_LIMITS_BY_BUCKET = {
        2: {"als_top_k": 450, "cluster_top_k": 260, "popular_top_k": 120, "profile_top_k": 260, "pretrim_top_k": 360, "cluster_user_topn": 6},
        5: {"als_top_k": 460, "cluster_top_k": 260, "popular_top_k": 120, "profile_top_k": 260, "pretrim_top_k": 380, "cluster_user_topn": 6},
        # Bucket10 is recall-bottlenecked; raise candidate budget for higher
        # true-item-in-candidate rate before rerank.
        10: {"als_top_k": 700, "cluster_top_k": 380, "popular_top_k": 150, "profile_top_k": 380, "pretrim_top_k": 620, "cluster_user_topn": 8},
    }
elif RECALL_PROFILE == "coverage_stage2":
    # Balanced high-recall profile from 2026-02-25 full probes:
    # keeps hard-miss low while reducing pretrim cut-loss vs naive expansion.
    ALS_TOP_K = 300
    CLUSTER_TOP_K = 180
    POPULAR_TOP_K = 100
    PROFILE_TOP_K = 220
    PRETRIM_TOP_K = 280
    CLUSTER_USER_TOPN = 6
    RECALL_LIMITS_BY_BUCKET = {
        2: {"als_top_k": 340, "cluster_top_k": 220, "popular_top_k": 110, "profile_top_k": 240, "pretrim_top_k": 360, "cluster_user_topn": 6},
        5: {"als_top_k": 420, "cluster_top_k": 260, "popular_top_k": 130, "profile_top_k": 280, "pretrim_top_k": 420, "cluster_user_topn": 7},
        10: {"als_top_k": 620, "cluster_top_k": 360, "popular_top_k": 150, "profile_top_k": 400, "pretrim_top_k": 520, "cluster_user_topn": 8},
    }
    # Per-segment pretrim budgets (heavy users get deeper pretrim to reduce
    # deep-truth truncation; light stays bounded to control total compute).
    PRETRIM_SEGMENT_TOPK_BY_BUCKET = {
        2: {"light": 360, "mid": 390, "heavy": 500, "unknown": 360},
        5: {"light": 390, "mid": 440, "heavy": 560, "unknown": 390},
        10: {"light": 500, "mid": 560, "heavy": 700, "unknown": 500},
    }
    PROFILE_ROUTE_POLICY_DEFAULT_BY_BUCKET = {
        2: {
            "enable_profile_vector_route": True,
            "enable_tag_shared_route": True,
            "enable_bridge_user_route": True,
            "enable_bridge_type_route": False,
        },
        5: {
            "enable_profile_vector_route": True,
            "enable_tag_shared_route": True,
            "enable_bridge_user_route": True,
            "enable_bridge_type_route": False,
        },
        10: {
            "enable_profile_vector_route": True,
            "enable_tag_shared_route": True,
            "enable_bridge_user_route": True,
            "enable_bridge_type_route": False,
        },
    }
    PROFILE_ROUTE_THRESHOLDS_DEFAULT_BY_BUCKET = {
        2: {
            "profile_tag_shared_top_k": 110,
            "profile_bridge_user_top_k": 110,
            "profile_bridge_user_min_sim": 0.06,
            "profile_shared_score_min": 0.02,
            "profile_bridge_score_min": 0.02,
        },
        5: {
            "profile_tag_shared_top_k": 100,
            "profile_bridge_user_top_k": 100,
            "profile_bridge_user_min_sim": 0.08,
            "profile_shared_score_min": 0.03,
            "profile_bridge_score_min": 0.03,
        },
        10: {
            "profile_tag_shared_top_k": 100,
            "profile_bridge_user_top_k": 100,
            "profile_bridge_user_min_sim": 0.08,
            "profile_shared_score_min": 0.03,
            "profile_bridge_score_min": 0.03,
        },
    }
    # Bucket10 already has strong source coverage; the cheaper gain is improving
    # pre-rank/head protection for corroborated profile/cluster candidates.
    PRETRIM_HEAD_POLICY_DEFAULT_BY_BUCKET = {
        10: {
            "source_count_weight": 0.02,
            "source_count_cap": 2,
            "multi_source_min": 2,
            "profile_rank_weight": 0.055,
            "profile_rank_cap": 160,
            "cluster_rank_weight": 0.035,
            "cluster_rank_cap": 160,
            "popular_only_penalty": 0.02,
        },
    }

# Layered pretrim policy: keep high-precision candidates protected in head,
# then fill from broader layers. This is enabled only for selected buckets.
LAYERED_PRETRIM_BY_BUCKET: dict[int, dict[str, int]] = {
    5: {
        "front_guard_topk": 120,
        "l1_quota": 180,
        "l2_quota": 130,
        "l3_quota": 70,
        "l1_als_rank_max": 110,
        "l1_cluster_rank_max": 65,
        "l1_profile_rank_max": 65,
        "l2_als_rank_max": 260,
        "l2_cluster_rank_max": 180,
        "l2_profile_rank_max": 180,
    },
    10: {
        "front_guard_topk": 140,
        "l1_quota": 220,
        "l2_quota": 180,
        "l3_quota": 100,
        "l1_als_rank_max": 120,
        "l1_cluster_rank_max": 70,
        "l1_profile_rank_max": 70,
        "l2_als_rank_max": 300,
        "l2_cluster_rank_max": 200,
        "l2_profile_rank_max": 220,
    }
}
if RECALL_PROFILE == "coverage_stage2":
    LAYERED_PRETRIM_BY_BUCKET = {
        2: {
            "front_guard_topk": 120,
            "l1_quota": 220,
            "l2_quota": 160,
            "l3_quota": 90,
            "l1_als_rank_max": 120,
            "l1_cluster_rank_max": 80,
            "l1_profile_rank_max": 80,
            "l2_als_rank_max": 320,
            "l2_cluster_rank_max": 220,
            "l2_profile_rank_max": 220,
        },
        5: {
            "front_guard_topk": 140,
            "l1_quota": 260,
            "l2_quota": 190,
            "l3_quota": 120,
            "l1_als_rank_max": 130,
            "l1_cluster_rank_max": 85,
            "l1_profile_rank_max": 85,
            "l2_als_rank_max": 420,
            "l2_cluster_rank_max": 260,
            "l2_profile_rank_max": 260,
        },
        10: {
            "front_guard_topk": 140,
            "l1_quota": 320,
            "l2_quota": 260,
            "l3_quota": 160,
            "l1_als_rank_max": 140,
            "l1_cluster_rank_max": 90,
            "l1_profile_rank_max": 90,
            "l2_als_rank_max": 560,
            "l2_cluster_rank_max": 360,
            "l2_profile_rank_max": 360,
        },
    }

# Segment policy
LIGHT_MAX_TRAIN = 7
MID_MAX_TRAIN = 19

# Pre-score policy (single config block)
QUALITY_WEIGHT = 0.04
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.03").strip() or 0.03)
ENABLE_SEMANTIC_GATES = os.getenv("ENABLE_SEMANTIC_GATES", "false").strip().lower() == "true"
SEMANTIC_SUPPORT_CAP = float(os.getenv("SEMANTIC_SUPPORT_CAP", "120").strip() or 120.0)
SEMANTIC_RICHNESS_CAP = float(os.getenv("SEMANTIC_RICHNESS_CAP", "3").strip() or 3.0)
SEMANTIC_RICHNESS_BLEND = float(os.getenv("SEMANTIC_RICHNESS_BLEND", "0.3").strip() or 0.3)
SOURCE_WEIGHTS = {
    "light": {"als": 0.45, "cluster": 0.25, "popular": 0.15, "profile": 0.15, "context": 0.14},
    "mid": {"als": 0.70, "cluster": 0.15, "popular": 0.08, "profile": 0.07, "context": 0.08},
    "heavy": {"als": 0.85, "cluster": 0.08, "popular": 0.03, "profile": 0.04, "context": 0.05},
}
SOURCE_WEIGHTS_BY_BUCKET: dict[int, dict[str, dict[str, float]]] = {}
PROFILE_RECALL_BATCH_USERS = 256
CLUSTER_CONF_FLOOR = 0.35
PROFILE_CONFIDENCE_MIN = 0.30
PROFILE_CONFIDENCE_FLOOR = 0.25
PROFILE_CONF_V2_BLEND = float(os.getenv("PROFILE_CONF_V2_BLEND", "0.35").strip() or 0.35)
PROFILE_MIN_SENTENCES = 4
PROFILE_ITEM_WEIGHT_CONF_POWER = 1.0
PROFILE_ITEM_WEIGHT_RATING_POWER = 0.5
PROFILE_ITEM_WEIGHT_ACTIVITY_POWER = 0.5
PROFILE_ITEM_WEIGHT_MIN = 0.05
PROFILE_ITEM_AGG_CHUNK_ROWS = int(os.getenv("PROFILE_ITEM_AGG_CHUNK_ROWS", "8192").strip() or 8192)
PROFILE_TAG_SHARED_TOP_K = int(os.getenv("PROFILE_TAG_SHARED_TOP_K", "120").strip() or 120)
PROFILE_BRIDGE_USER_TOP_K = int(os.getenv("PROFILE_BRIDGE_USER_TOP_K", "100").strip() or 100)
PROFILE_BRIDGE_TYPE_TOP_K = int(os.getenv("PROFILE_BRIDGE_TYPE_TOP_K", "100").strip() or 100)
PROFILE_BRIDGE_USER_NEIGHBORS = int(os.getenv("PROFILE_BRIDGE_USER_NEIGHBORS", "24").strip() or 24)
PROFILE_BRIDGE_USER_MIN_SIM = float(os.getenv("PROFILE_BRIDGE_USER_MIN_SIM", "0.02").strip() or 0.02)
PROFILE_SHARED_SCORE_MIN = float(os.getenv("PROFILE_SHARED_SCORE_MIN", "0.0").strip() or 0.0)
PROFILE_BRIDGE_SCORE_MIN = float(os.getenv("PROFILE_BRIDGE_SCORE_MIN", "0.0").strip() or 0.0)
PROFILE_ROUTE_POLICY_BY_BUCKET: dict[int, dict[str, Any]] = {}
PROFILE_ROUTE_THRESHOLDS_BY_BUCKET: dict[int, dict[str, Any]] = {}
ENABLE_PRETRIM_HEAD_POLICY = os.getenv("ENABLE_PRETRIM_HEAD_POLICY", "true").strip().lower() == "true"
ENABLE_PRETRIM_CONSENSUS_RESCUE = os.getenv("ENABLE_PRETRIM_CONSENSUS_RESCUE", "true").strip().lower() == "true"
ENABLE_LAYERED_PRETRIM_FINAL_RANK = (
    os.getenv("ENABLE_LAYERED_PRETRIM_FINAL_RANK", "true").strip().lower() == "true"
)
_write_generic_pretrim_alias_legacy = os.getenv("WRITE_GENERIC_PRETRIM_ALIAS", "true").strip().lower() == "true"
WRITE_LEGACY_PRETRIM150_ALIAS = (
    os.getenv(
        "WRITE_LEGACY_PRETRIM150_ALIAS",
        "true" if _write_generic_pretrim_alias_legacy else "false",
    ).strip().lower()
    == "true"
)
WRITE_TRUTH_USER_ROSTER = os.getenv("WRITE_TRUTH_USER_ROSTER", "true").strip().lower() == "true"
WRITE_ENRICHED_AUDIT_EXPORT = os.getenv("WRITE_ENRICHED_AUDIT_EXPORT", "false").strip().lower() == "true"
ENRICHED_AUDIT_BUCKETS_RAW = os.getenv("ENRICHED_AUDIT_BUCKETS", "10").strip()
ALS_BACKBONE_TOPN = {"light": 3, "mid": 5, "heavy": 8}
ALS_BACKBONE_WEIGHT = 0.35
ALS_SAFETY_KEEP_TOPK = 20
PROFILE_ROUTE_POLICY_BY_BUCKET = dict(PROFILE_ROUTE_POLICY_DEFAULT_BY_BUCKET)
PROFILE_ROUTE_THRESHOLDS_BY_BUCKET = dict(PROFILE_ROUTE_THRESHOLDS_DEFAULT_BY_BUCKET)
PRETRIM_HEAD_POLICY_BY_BUCKET = dict(PRETRIM_HEAD_POLICY_DEFAULT_BY_BUCKET)
STAGE09_CLOUD_FAST_MODE = os.getenv("STAGE09_CLOUD_FAST_MODE", "false").strip().lower() == "true"
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[4]").strip() or "local[4]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_SQL_ADAPTIVE_ENABLED = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "true").strip().lower() == "true"
_spark_sql_parquet_vectorized_raw = os.getenv("SPARK_SQL_PARQUET_ENABLE_VECTORIZED", "").strip()
if _spark_sql_parquet_vectorized_raw:
    SPARK_SQL_PARQUET_ENABLE_VECTORIZED = _spark_sql_parquet_vectorized_raw.lower() == "true"
else:
    SPARK_SQL_PARQUET_ENABLE_VECTORIZED = STAGE09_CLOUD_FAST_MODE
SPARK_SQL_FILES_MAX_PARTITION_BYTES = (
    os.getenv(
        "SPARK_SQL_FILES_MAX_PARTITION_BYTES",
        "128m" if STAGE09_CLOUD_FAST_MODE else "64m",
    ).strip()
    or ("128m" if STAGE09_CLOUD_FAST_MODE else "64m")
)
SPARK_SQL_MAX_PLAN_STRING_LENGTH = (
    os.getenv("SPARK_SQL_MAX_PLAN_STRING_LENGTH", "8192").strip() or "8192"
)
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = (
    os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
)
SPARK_DRIVER_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_DRIVER_EXTRA_JAVA_OPTIONS",
    (
        "-XX:+UseG1GC -XX:+ParallelRefProcEnabled -XX:InitiatingHeapOccupancyPercent=35 "
        "-XX:ReservedCodeCacheSize=256m -XX:MaxMetaspaceSize=512m -Xss1m"
        if STAGE09_CLOUD_FAST_MODE
        else "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 "
        "-XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m -Xss512k"
    ),
).strip()
SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS",
    (
        "-XX:+UseG1GC -XX:+ParallelRefProcEnabled -XX:InitiatingHeapOccupancyPercent=35 "
        "-XX:ReservedCodeCacheSize=256m -XX:MaxMetaspaceSize=512m -Xss1m"
        if STAGE09_CLOUD_FAST_MODE
        else "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 "
        "-XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m -Xss512k"
    ),
).strip()
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
SPARK_DRIVER_BIND_ADDRESS = os.getenv("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1").strip() or "127.0.0.1"
SPARK_PYTHON_WORKER_REUSE = os.getenv("SPARK_PYTHON_WORKER_REUSE", "true").strip().lower() == "true"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
PY_TEMP_DIR = os.getenv("PY_TEMP_DIR", "").strip()
SKIP_INTERNAL_METRICS = os.getenv("SKIP_INTERNAL_METRICS", "false").strip().lower() == "true"
ENABLE_BUCKET_WRITE_CHECKPOINT = os.getenv("ENABLE_BUCKET_WRITE_CHECKPOINT", "false").strip().lower() == "true"
BUCKET_WRITE_CHECKPOINT_DIR = (
    os.getenv("BUCKET_WRITE_CHECKPOINT_DIR", project_path("data/spark-tmp/stage09-checkpoint").as_posix()).strip()
    or project_path("data/spark-tmp/stage09-checkpoint").as_posix()
)
SKIP_CHECKPOINT_ON_PYTHON_PLAN = os.getenv("SKIP_CHECKPOINT_ON_PYTHON_PLAN", "true").strip().lower() == "true"
OUTPUT_COALESCE_PARTITIONS = int(os.getenv("OUTPUT_COALESCE_PARTITIONS", "8").strip() or 8)
LOCAL_PARQUET_WRITE_MODE = (os.getenv("LOCAL_PARQUET_WRITE_MODE", "spark").strip().lower() or "spark")
LOCAL_PARQUET_WRITE_CHUNK_ROWS = int(os.getenv("LOCAL_PARQUET_WRITE_CHUNK_ROWS", "50000").strip() or 50000)
TOWER_SEQ_INPUT_MATERIALIZE_MODE = (
    os.getenv("TOWER_SEQ_INPUT_MATERIALIZE_MODE", "pandas").strip().lower() or "pandas"
)
TOWER_SEQ_INPUT_PARQUET_CHUNK_ROWS = int(
    os.getenv("TOWER_SEQ_INPUT_PARQUET_CHUNK_ROWS", "50000").strip() or 50000
)
_BUCKET_ENV = os.getenv("MIN_TRAIN_BUCKETS_OVERRIDE", "").strip()
if _BUCKET_ENV:
    MIN_TRAIN_REVIEWS_BUCKETS = [int(x.strip()) for x in _BUCKET_ENV.split(",") if x.strip()]


_SPARK_TMP_CTX: SparkTmpContext | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_bucket_json(raw: str) -> dict[int, dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in payload.items():
        try:
            bk = int(str(k).strip())
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        out[bk] = dict(v)
    return out


def _merge_bucket_int_policy(
    base: dict[int, dict[str, int]], override_raw: str
) -> dict[int, dict[str, int]]:
    parsed = _parse_bucket_json(override_raw)
    if not parsed:
        return base
    merged: dict[int, dict[str, int]] = {}
    for k, vals in base.items():
        row: dict[str, int] = {}
        for kk, vv in vals.items():
            try:
                row[str(kk)] = int(vv)
            except Exception:
                continue
        merged[int(k)] = row
    for bk, cfg in parsed.items():
        cur = dict(merged.get(int(bk), {}))
        for k, v in cfg.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                cur[key] = int(v)
            except Exception:
                continue
        merged[int(bk)] = cur
    return merged


def _merge_bucket_any_policy(
    base: dict[int, dict[str, Any]], override_raw: str
) -> dict[int, dict[str, Any]]:
    parsed = _parse_bucket_json(override_raw)
    if not parsed:
        return base
    merged: dict[int, dict[str, Any]] = {}
    for k, vals in base.items():
        merged[int(k)] = dict(vals)
    for k, vals in parsed.items():
        row = dict(merged.get(int(k), {}))
        row.update(dict(vals))
        merged[int(k)] = row
    return merged


def _parse_bucket_int_map(raw: str, key: str | None = None) -> dict[int, int]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[int, int] = {}
    for k, v in payload.items():
        try:
            bk = int(str(k).strip())
        except Exception:
            continue
        vv: Any = v
        if isinstance(v, dict):
            if key is None:
                continue
            vv = v.get(str(key), None)
        try:
            iv = int(vv)
        except Exception:
            continue
        if iv > 0:
            out[int(bk)] = int(iv)
    return out


def _parse_int_set(raw: str) -> set[int]:
    text = str(raw or "").strip()
    if not text:
        return set()
    out: set[int] = set()
    for part in text.split(","):
        p = str(part).strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


ENRICHED_AUDIT_BUCKETS = _parse_int_set(ENRICHED_AUDIT_BUCKETS_RAW)
PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS = _parse_int_set(PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS_RAW)
PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS = _parse_int_set(PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS_RAW)
PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS = _parse_int_set(PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS_RAW)
DEBIAS_BUCKETS = _parse_int_set(DEBIAS_BUCKETS_RAW)
PERSONALIZED_CHALLENGER_BUCKETS = _parse_int_set(PERSONALIZED_CHALLENGER_BUCKETS_RAW)
CONTEXT_SURFACE_BUCKETS = _parse_int_set(CONTEXT_SURFACE_BUCKETS_RAW)
CONTEXT_ROUTE_DETAILS = tuple(
    x.strip().lower() for x in CONTEXT_ROUTE_DETAIL_RAW.split(",") if x.strip().lower() in {"attr", "time", "geo"}
)


def should_write_enriched_audit(bucket: int) -> bool:
    if not WRITE_ENRICHED_AUDIT_EXPORT:
        return False
    if not ENRICHED_AUDIT_BUCKETS:
        return True
    return int(bucket) in ENRICHED_AUDIT_BUCKETS


def should_write_profile_multivector_route_audit(bucket: int) -> bool:
    if not ENABLE_PROFILE_MULTIVECTOR_ROUTE_AUDIT:
        return False
    if not PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS:
        return True
    return int(bucket) in PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS


def should_integrate_profile_multivector_routes(bucket: int) -> bool:
    if not ENABLE_PROFILE_MULTIVECTOR_ROUTE_INTEGRATION:
        return False
    if not PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS:
        return True
    return int(bucket) in PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS


def should_enable_profile_evidence_protected_lane(bucket: int) -> bool:
    if not ENABLE_PROFILE_EVIDENCE_PROTECTED_LANE:
        return False
    if not PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS:
        return True
    return int(bucket) in PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS


def should_enable_bucket_debias_rewrite(bucket: int) -> bool:
    if not ENABLE_BUCKET_DEBIAS_REWRITE:
        return False
    if not DEBIAS_BUCKETS:
        return True
    return int(bucket) in DEBIAS_BUCKETS


def should_enable_personalized_challenger_surface(bucket: int) -> bool:
    if not ENABLE_PERSONALIZED_CHALLENGER_SURFACE:
        return False
    if not PERSONALIZED_CHALLENGER_BUCKETS:
        return True
    return int(bucket) in PERSONALIZED_CHALLENGER_BUCKETS


def should_enable_context_surface(bucket: int) -> bool:
    if not ENABLE_CONTEXT_SURFACE:
        return False
    if not CONTEXT_SURFACE_BUCKETS:
        return True
    return int(bucket) in CONTEXT_SURFACE_BUCKETS


def _apply_env_overrides() -> None:
    global PROFILE_ITEM_WEIGHT_CONF_POWER
    global PROFILE_ITEM_WEIGHT_RATING_POWER
    global PROFILE_ITEM_WEIGHT_ACTIVITY_POWER
    global PROFILE_ITEM_WEIGHT_MIN
    global PROFILE_CONFIDENCE_MIN
    global PROFILE_CONFIDENCE_FLOOR
    global PROFILE_CONF_V2_BLEND
    global ALS_TOP_K
    global CLUSTER_TOP_K
    global POPULAR_TOP_K
    global PROFILE_TOP_K
    global PRETRIM_TOP_K
    global CLUSTER_USER_TOPN
    global RECALL_LIMITS_BY_BUCKET
    global LAYERED_PRETRIM_BY_BUCKET
    global PRETRIM_SEGMENT_TOPK_BY_BUCKET
    global TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET
    global TOWER_SEQ_HEAD_WINDOW_BY_BUCKET
    global PROFILE_ROUTE_POLICY_BY_BUCKET
    global PROFILE_ROUTE_THRESHOLDS_BY_BUCKET
    global PRETRIM_HEAD_POLICY_BY_BUCKET
    global SOURCE_WEIGHTS_BY_BUCKET

    PROFILE_ITEM_WEIGHT_CONF_POWER = _env_float(
        "PROFILE_ITEM_WEIGHT_CONF_POWER_OVERRIDE", PROFILE_ITEM_WEIGHT_CONF_POWER
    )
    PROFILE_ITEM_WEIGHT_RATING_POWER = _env_float(
        "PROFILE_ITEM_WEIGHT_RATING_POWER_OVERRIDE", PROFILE_ITEM_WEIGHT_RATING_POWER
    )
    PROFILE_ITEM_WEIGHT_ACTIVITY_POWER = _env_float(
        "PROFILE_ITEM_WEIGHT_ACTIVITY_POWER_OVERRIDE", PROFILE_ITEM_WEIGHT_ACTIVITY_POWER
    )
    PROFILE_ITEM_WEIGHT_MIN = _env_float(
        "PROFILE_ITEM_WEIGHT_MIN_OVERRIDE", PROFILE_ITEM_WEIGHT_MIN
    )
    PROFILE_CONFIDENCE_MIN = _env_float(
        "PROFILE_CONFIDENCE_MIN_OVERRIDE", PROFILE_CONFIDENCE_MIN
    )
    PROFILE_CONFIDENCE_FLOOR = _env_float(
        "PROFILE_CONFIDENCE_FLOOR_OVERRIDE", PROFILE_CONFIDENCE_FLOOR
    )
    PROFILE_CONF_V2_BLEND = _env_float(
        "PROFILE_CONF_V2_BLEND_OVERRIDE", PROFILE_CONF_V2_BLEND
    )
    ALS_TOP_K = _env_int("ALS_TOP_K_OVERRIDE", ALS_TOP_K)
    CLUSTER_TOP_K = _env_int("CLUSTER_TOP_K_OVERRIDE", CLUSTER_TOP_K)
    POPULAR_TOP_K = _env_int("POPULAR_TOP_K_OVERRIDE", POPULAR_TOP_K)
    PROFILE_TOP_K = _env_int("PROFILE_TOP_K_OVERRIDE", PROFILE_TOP_K)
    PRETRIM_TOP_K = _env_int("PRETRIM_TOP_K_OVERRIDE", PRETRIM_TOP_K)
    CLUSTER_USER_TOPN = _env_int("CLUSTER_USER_TOPN_OVERRIDE", CLUSTER_USER_TOPN)
    RECALL_LIMITS_BY_BUCKET = _merge_bucket_int_policy(RECALL_LIMITS_BY_BUCKET, RECALL_LIMITS_BY_BUCKET_JSON)
    LAYERED_PRETRIM_BY_BUCKET = _merge_bucket_int_policy(LAYERED_PRETRIM_BY_BUCKET, LAYERED_PRETRIM_BY_BUCKET_JSON)
    PRETRIM_SEGMENT_TOPK_BY_BUCKET = _merge_bucket_int_policy(
        PRETRIM_SEGMENT_TOPK_BY_BUCKET, PRETRIM_SEGMENT_TOPK_BY_BUCKET_JSON
    )
    TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET = _parse_bucket_int_map(
        TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET_JSON,
        key="tower_seq_max_cand_rows",
    )
    TOWER_SEQ_HEAD_WINDOW_BY_BUCKET = _parse_bucket_int_map(
        TOWER_SEQ_HEAD_WINDOW_BY_BUCKET_JSON,
        key="tower_seq_head_window",
    )
    PROFILE_ROUTE_POLICY_BY_BUCKET = _merge_bucket_any_policy(
        PROFILE_ROUTE_POLICY_BY_BUCKET, PROFILE_ROUTE_POLICY_JSON
    )
    PROFILE_ROUTE_THRESHOLDS_BY_BUCKET = _merge_bucket_any_policy(
        PROFILE_ROUTE_THRESHOLDS_BY_BUCKET, PROFILE_ROUTE_THRESHOLDS_JSON
    )
    PRETRIM_HEAD_POLICY_BY_BUCKET = _merge_bucket_any_policy(
        PRETRIM_HEAD_POLICY_BY_BUCKET, PRETRIM_HEAD_POLICY_JSON
    )
    SOURCE_WEIGHTS_BY_BUCKET = _merge_bucket_any_policy(
        SOURCE_WEIGHTS_BY_BUCKET, SOURCE_WEIGHTS_BY_BUCKET_JSON
    )


def get_bucket_recall_limits(min_train: int) -> dict[str, int]:
    limits = {
        "als_top_k": int(ALS_TOP_K),
        "cluster_top_k": int(CLUSTER_TOP_K),
        "popular_top_k": int(POPULAR_TOP_K),
        "profile_top_k": int(PROFILE_TOP_K),
        "pretrim_top_k": int(PRETRIM_TOP_K),
        "cluster_user_topn": int(CLUSTER_USER_TOPN),
    }
    override = RECALL_LIMITS_BY_BUCKET.get(int(min_train), {})
    for k, v in override.items():
        limits[k] = int(v)
    return limits


def get_layered_pretrim_policy(min_train: int) -> dict[str, int] | None:
    raw = LAYERED_PRETRIM_BY_BUCKET.get(int(min_train))
    if raw is None:
        return None
    return {k: int(v) for k, v in raw.items()}


def get_tower_seq_max_cand_rows(min_train: int) -> int:
    per_bucket = TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET.get(int(min_train))
    if per_bucket is not None:
        return int(max(1, int(per_bucket)))
    return int(max(1, TOWER_SEQ_MAX_CAND_ROWS))


def get_tower_seq_head_window(min_train: int) -> int:
    per_bucket = TOWER_SEQ_HEAD_WINDOW_BY_BUCKET.get(int(min_train))
    if per_bucket is None:
        return 0
    return int(max(0, int(per_bucket)))


def get_pretrim_segment_policy(min_train: int, default_topk: int) -> dict[str, int]:
    policy = {
        "light": int(default_topk),
        "mid": int(default_topk),
        "heavy": int(default_topk),
        "unknown": int(default_topk),
    }
    raw = PRETRIM_SEGMENT_TOPK_BY_BUCKET.get(int(min_train), {})
    for k, v in raw.items():
        key = str(k).strip().lower()
        if key not in policy:
            continue
        try:
            policy[key] = max(1, int(v))
        except Exception:
            continue
    return policy


def get_pretrim_head_policy(min_train: int) -> dict[str, float | int | bool]:
    cfg: dict[str, float | int | bool] = {
        "enable": bool(ENABLE_PRETRIM_HEAD_POLICY),
        "source_count_weight": 0.0,
        "source_count_cap": 0,
        "multi_source_min": 2,
        "profile_rank_weight": 0.0,
        "profile_rank_cap": 0,
        "cluster_rank_weight": 0.0,
        "cluster_rank_cap": 0,
        "popular_only_penalty": 0.0,
    }
    raw = PRETRIM_HEAD_POLICY_BY_BUCKET.get(int(min_train), {})
    alias = {
        "enabled": "enable",
        "multi_source_weight": "source_count_weight",
        "multi_source_cap": "source_count_cap",
    }
    for k, v in raw.items():
        key = alias.get(str(k).strip(), str(k).strip())
        if key not in cfg:
            continue
        base = cfg[key]
        try:
            if isinstance(base, bool):
                cfg[key] = _as_bool(v, bool(base))
            elif isinstance(base, int):
                cfg[key] = int(v)
            else:
                cfg[key] = float(v)
        except Exception:
            continue
    return cfg


def get_profile_route_policy(min_train: int) -> dict[str, bool]:
    cfg: dict[str, bool] = {
        "enable_profile_vector_route": bool(ENABLE_PROFILE_VECTOR_ROUTE),
        "enable_tag_shared_route": bool(ENABLE_TAG_SHARED_ROUTE),
        "enable_bridge_user_route": bool(ENABLE_BRIDGE_USER_ROUTE),
        "enable_bridge_type_route": bool(ENABLE_BRIDGE_TYPE_ROUTE),
    }
    raw = PROFILE_ROUTE_POLICY_BY_BUCKET.get(int(min_train), {})
    alias = {
        "vector": "enable_profile_vector_route",
        "shared": "enable_tag_shared_route",
        "bridge_user": "enable_bridge_user_route",
        "bridge_type": "enable_bridge_type_route",
    }
    for k, v in raw.items():
        key = str(k).strip()
        key = alias.get(key, key)
        if key in cfg:
            cfg[key] = _as_bool(v, cfg[key])
    return cfg


def get_profile_threshold_policy(min_train: int) -> dict[str, float | int]:
    cfg: dict[str, float | int] = {
        "profile_tag_shared_top_k": int(PROFILE_TAG_SHARED_TOP_K),
        "profile_bridge_user_top_k": int(PROFILE_BRIDGE_USER_TOP_K),
        "profile_bridge_type_top_k": int(PROFILE_BRIDGE_TYPE_TOP_K),
        "profile_bridge_user_neighbors": int(PROFILE_BRIDGE_USER_NEIGHBORS),
        "profile_bridge_user_min_sim": float(PROFILE_BRIDGE_USER_MIN_SIM),
        "profile_shared_score_min": float(PROFILE_SHARED_SCORE_MIN),
        "profile_bridge_score_min": float(PROFILE_BRIDGE_SCORE_MIN),
    }
    raw = PROFILE_ROUTE_THRESHOLDS_BY_BUCKET.get(int(min_train), {})
    alias = {
        "tag_shared_top_k": "profile_tag_shared_top_k",
        "bridge_user_top_k": "profile_bridge_user_top_k",
        "bridge_type_top_k": "profile_bridge_type_top_k",
        "bridge_user_neighbors": "profile_bridge_user_neighbors",
        "bridge_user_min_sim": "profile_bridge_user_min_sim",
        "shared_score_min": "profile_shared_score_min",
        "bridge_score_min": "profile_bridge_score_min",
    }
    for k, v in raw.items():
        key = str(k).strip()
        key = alias.get(key, key)
        if key not in cfg:
            continue
        base = cfg[key]
        try:
            if isinstance(base, int):
                cfg[key] = int(v)
            else:
                cfg[key] = float(v)
        except Exception:
            continue
    return cfg


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        py_temp_root_override=PY_TEMP_DIR,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=True,
    )
    local_dir = _SPARK_TMP_CTX.spark_local_dir
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} spark_local_dir={local_dir} py_temp={_SPARK_TMP_CTX.py_temp_dir} "
        f"auto_clean={SPARK_TMP_AUTOCLEAN_ENABLED} retention_h={SPARK_TMP_RETENTION_HOURS} "
        f"cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    return (
        SparkSession.builder.appName("stage09-candidate-fusion")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.extraJavaOptions", SPARK_DRIVER_EXTRA_JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS)
        .config("spark.driver.host", SPARK_DRIVER_HOST)
        .config("spark.driver.bindAddress", SPARK_DRIVER_BIND_ADDRESS)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.sql.adaptive.enabled", "true" if SPARK_SQL_ADAPTIVE_ENABLED else "false")
        .config("spark.sql.parquet.enableVectorizedReader", "true" if SPARK_SQL_PARQUET_ENABLE_VECTORIZED else "false")
        .config("spark.sql.files.maxPartitionBytes", SPARK_SQL_FILES_MAX_PARTITION_BYTES)
        .config("spark.sql.maxPlanStringLength", SPARK_SQL_MAX_PLAN_STRING_LENGTH)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.python.worker.reuse", "true" if SPARK_PYTHON_WORKER_REUSE else "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _alloc_stage_tmp_csv(subdir: str, prefix: str) -> Path:
    if _SPARK_TMP_CTX is not None:
        return alloc_scratch_file(ctx=_SPARK_TMP_CTX, subdir=subdir, prefix=prefix, suffix=".csv")
    fallback_root = Path(SPARK_LOCAL_DIR) / "_scratch" / subdir
    fallback_root.mkdir(parents=True, exist_ok=True)
    return fallback_root / f"{prefix}_{uuid.uuid4().hex}.csv"


def _alloc_stage_tmp_path(subdir: str, prefix: str, suffix: str = "") -> Path:
    if _SPARK_TMP_CTX is not None:
        return alloc_scratch_file(ctx=_SPARK_TMP_CTX, subdir=subdir, prefix=prefix, suffix=suffix)
    fallback_root = Path(SPARK_LOCAL_DIR) / "_scratch" / subdir
    fallback_root.mkdir(parents=True, exist_ok=True)
    return fallback_root / f"{prefix}_{uuid.uuid4().hex}{suffix}"


def _checkpoint_for_write(spark: SparkSession, df: DataFrame) -> DataFrame:
    if not ENABLE_BUCKET_WRITE_CHECKPOINT:
        return df
    if SKIP_CHECKPOINT_ON_PYTHON_PLAN:
        try:
            plan_txt = df._jdf.queryExecution().toString()
            if "Python" in str(plan_txt):
                return df
        except Exception:
            pass
    ckpt_dir = Path(BUCKET_WRITE_CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(ckpt_dir.as_posix())
    return df.checkpoint(eager=True)


def _coalesce_for_write(df: DataFrame) -> DataFrame:
    target = int(max(1, OUTPUT_COALESCE_PARTITIONS))
    try:
        current = int(df.rdd.getNumPartitions())
    except Exception:
        return df
    if current <= target:
        return df
    return df.coalesce(target)


def _write_output_df(df: DataFrame, output_dir: Path) -> dict[str, Any]:
    mode = str(LOCAL_PARQUET_WRITE_MODE).strip().lower() or "spark"
    if mode == "driver_parquet":
        meta = write_spark_df_to_parquet_dir(
            df,
            output_dir,
            chunk_rows=LOCAL_PARQUET_WRITE_CHUNK_ROWS,
            compression="snappy",
        )
        print(
            f"[WRITE] mode=driver_parquet path={output_dir} rows={meta['row_count']} "
            f"files={meta['file_count']} chunk_rows={meta['chunk_rows']}"
        )
        return meta
    if mode != "spark":
        raise ValueError(f"unsupported LOCAL_PARQUET_WRITE_MODE={LOCAL_PARQUET_WRITE_MODE}")
    df.write.mode("overwrite").parquet(str(output_dir))
    print(f"[WRITE] mode=spark path={output_dir}")
    return {"output_dir": str(output_dir), "mode": "spark"}


def _materialize_small_df_to_pandas(
    df: DataFrame,
    *,
    subdir: str,
    prefix: str,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mat_mode = str(mode or "pandas").strip().lower() or "pandas"
    meta: dict[str, Any] = {"mode": mat_mode, "path": ""}
    if mat_mode == "pandas":
        return df.toPandas(), meta

    tmp_dir = _alloc_stage_tmp_path(subdir, prefix, suffix="_parquet")
    meta["path"] = tmp_dir.as_posix()
    write_df = _coalesce_for_write(df)
    if mat_mode == "spark_parquet":
        write_df.write.mode("overwrite").parquet(tmp_dir.as_posix())
    elif mat_mode == "driver_parquet":
        write_spark_df_to_parquet_dir(
            write_df,
            tmp_dir,
            chunk_rows=TOWER_SEQ_INPUT_PARQUET_CHUNK_ROWS,
            compression="snappy",
        )
    else:
        raise ValueError(f"unsupported TOWER_SEQ_INPUT_MATERIALIZE_MODE={mat_mode}")
    return pd.read_parquet(tmp_dir.as_posix()), meta


def build_source_evidence_df(candidates_scored: DataFrame) -> DataFrame:
    return candidates_scored.select(
        "user_idx",
        "item_idx",
        "business_id",
        "user_segment",
        "user_train_count",
        "source",
        "source_detail",
        "source_rank",
        "source_score",
        "source_confidence",
        "source_weight",
        "source_norm",
        "signal_score",
    )


def build_enriched_audit_df(
    candidates_scored: DataFrame,
    fused: DataFrame,
    fused_pretrim: DataFrame,
    als_top_k: int,
    cluster_top_k: int,
    popular_top_k: int,
    profile_top_k: int,
) -> DataFrame:
    def _src_max(src: str, col_name: str, alias: str) -> Any:
        return F.max(F.when(F.col("source") == F.lit(src), F.col(col_name))).alias(alias)

    def _src_sum(src: str, col_name: str, alias: str) -> Any:
        return F.sum(F.when(F.col("source") == F.lit(src), F.col(col_name)).otherwise(F.lit(0.0))).alias(alias)

    route_summary = candidates_scored.groupBy("user_idx", "item_idx").agg(
        _src_max("als", "source_rank", "als_source_rank"),
        _src_max("als", "source_score", "als_source_score"),
        _src_max("als", "source_confidence", "als_source_confidence"),
        _src_max("als", "source_weight", "als_source_weight"),
        _src_max("als", "source_norm", "als_source_norm"),
        _src_sum("als", "signal_score", "als_signal_score"),
        _src_max("cluster", "source_rank", "cluster_source_rank"),
        _src_max("cluster", "source_score", "cluster_source_score"),
        _src_max("cluster", "source_confidence", "cluster_source_confidence"),
        _src_max("cluster", "source_weight", "cluster_source_weight"),
        _src_max("cluster", "source_norm", "cluster_source_norm"),
        _src_sum("cluster", "signal_score", "cluster_signal_score"),
        _src_max("profile", "source_rank", "profile_source_rank"),
        _src_max("profile", "source_score", "profile_source_score"),
        _src_max("profile", "source_confidence", "profile_source_confidence"),
        _src_max("profile", "source_weight", "profile_source_weight"),
        _src_max("profile", "source_norm", "profile_source_norm"),
        _src_sum("profile", "signal_score", "profile_signal_score"),
        _src_max("popular", "source_rank", "popular_source_rank"),
        _src_max("popular", "source_score", "popular_source_score"),
        _src_max("popular", "source_confidence", "popular_source_confidence"),
        _src_max("popular", "source_weight", "popular_source_weight"),
        _src_max("popular", "source_norm", "popular_source_norm"),
        _src_sum("popular", "signal_score", "popular_signal_score"),
        _src_max("context", "source_rank", "context_source_rank"),
        _src_max("context", "source_score", "context_source_score"),
        _src_max("context", "source_confidence", "context_source_confidence"),
        _src_max("context", "source_weight", "context_source_weight"),
        _src_max("context", "source_norm", "context_source_norm"),
        _src_sum("context", "signal_score", "context_signal_score"),
    )

    final_rank = fused_pretrim.select(
        "user_idx",
        "item_idx",
        F.col("pre_rank").cast("int").alias("final_pre_rank"),
        F.col("tower_score").cast("double").alias("tower_score"),
        F.col("seq_score").cast("double").alias("seq_score"),
        F.col("tower_inv").cast("double").alias("tower_inv"),
        F.col("seq_inv").cast("double").alias("seq_inv"),
    )

    enriched = (
        fused.select(
            "user_idx",
            "item_idx",
            "business_id",
            "name",
            "city",
            "categories",
            "primary_category",
            "user_train_count",
            "user_segment",
            "item_train_pop_count",
            "source_set",
            "source_count",
            "nonpopular_source_count",
            "profile_cluster_source_count",
            "als_rank",
            "cluster_rank",
            "profile_rank",
            "popular_rank",
            "context_rank",
            "signal_score",
            "quality_score",
            "semantic_score",
            "semantic_confidence",
            "semantic_support",
            "semantic_tag_richness",
            "semantic_effective_score",
            "als_backbone_score",
            "pre_score",
            "head_score_seed",
            "head_multisource_boost",
            "head_profile_boost",
            "head_cluster_boost",
            "head_popular_penalty",
            "head_challenger_bonus",
            "head_score",
            "pre_head_seed_rank",
            "challenger_candidate_ok",
            "has_context",
            "context_detail_count",
        )
        .join(route_summary, on=["user_idx", "item_idx"], how="left")
        .join(final_rank, on=["user_idx", "item_idx"], how="left")
        .withColumn("is_in_pretrim", F.col("final_pre_rank").isNotNull().cast("int"))
        .withColumn(
            "is_in_top150",
            F.when(F.col("final_pre_rank").isNotNull() & (F.col("final_pre_rank") <= F.lit(150)), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "als_rank_pct",
            F.when(F.col("als_source_rank").isNotNull(), F.col("als_source_rank").cast("double") / F.lit(float(max(1, als_top_k)))).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "cluster_rank_pct",
            F.when(F.col("cluster_source_rank").isNotNull(), F.col("cluster_source_rank").cast("double") / F.lit(float(max(1, cluster_top_k)))).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "profile_rank_pct",
            F.when(F.col("profile_source_rank").isNotNull(), F.col("profile_source_rank").cast("double") / F.lit(float(max(1, profile_top_k)))).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "popular_rank_pct",
            F.when(F.col("popular_source_rank").isNotNull(), F.col("popular_source_rank").cast("double") / F.lit(float(max(1, popular_top_k)))).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "context_rank_pct",
            F.when(
                F.col("context_source_rank").isNotNull(),
                F.col("context_source_rank").cast("double") / F.lit(float(max(1, CONTEXT_SURFACE_TOP_K))),
            ).otherwise(F.lit(None).cast("double")),
        )
    )
    return enriched


def _create_spark_df_from_pandas_safe(
    spark: SparkSession,
    pdf: pd.DataFrame,
) -> DataFrame:
    out = spark.createDataFrame(pdf)
    target = int(max(1, PANDAS_TO_SPARK_OUTPUT_PARTITIONS))
    try:
        current = int(out.rdd.getNumPartitions())
        if current > target:
            out = out.coalesce(target)
    except Exception:
        pass
    return out


def compute_recall_ndcg(rank_df: DataFrame) -> tuple[float, float]:
    metrics = (
        rank_df.withColumn(
            "recall",
            F.when(F.col("rank") > 0, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ndcg",
            F.when(
                F.col("rank") > 0,
                F.lit(1.0) / (F.log(F.col("rank") + F.lit(1.0)) / F.log(F.lit(2.0))),
            ).otherwise(F.lit(0.0)),
        )
    )
    row = metrics.agg(F.avg("recall").alias("recall"), F.avg("ndcg").alias("ndcg")).first()
    return float(row["recall"] or 0.0), float(row["ndcg"] or 0.0)


def build_global_popular_topk_df(spark: SparkSession, pop_stats: DataFrame, top_k: int) -> DataFrame:
    # Avoid PythonRDD in stage09: keep this path fully in Spark SQL.
    # Avoid global unpartitioned window to remove WindowExec warnings.
    kk = max(1, int(top_k))
    top_df = (
        pop_stats.select(
            F.col("item_idx").cast("int").alias("item_idx"),
            F.col("item_train_pop_count").cast("double").alias("source_score"),
        )
        .orderBy(F.desc("source_score"), F.asc("item_idx"))
        .limit(kk)
    )
    ranked = (
        top_df.select(
            F.struct(
                (-F.col("source_score")).cast("double").alias("sort_neg_score"),
                F.col("item_idx").cast("int").alias("sort_item_idx"),
                F.col("item_idx").cast("int").alias("item_idx"),
                F.col("source_score").cast("double").alias("source_score"),
            ).alias("s")
        )
        .agg(F.sort_array(F.collect_list("s"), asc=True).alias("arr"))
        .select(F.posexplode("arr").alias("pos", "s"))
        .select(
            F.col("s.item_idx").cast("int").alias("item_idx"),
            (F.col("pos") + F.lit(1)).cast("int").alias("source_rank"),
            F.col("s.source_score").cast("double").alias("source_score"),
        )
    )
    return ranked


def pick_latest_run(root: Path, suffix: str, filename: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        if (run / filename).exists():
            return run
    raise FileNotFoundError(f"no run found in {root} suffix={suffix} file={filename}")


def resolve_cluster_profile_csv() -> Path:
    if CLUSTER_PROFILE_CSV.strip():
        p = Path(CLUSTER_PROFILE_CSV.strip())
        if not p.exists():
            raise FileNotFoundError(f"CLUSTER_PROFILE_CSV not found: {p}")
        return p
    run = pick_latest_run(
        CLUSTER_PROFILE_ROOT,
        CLUSTER_PROFILE_DIR_SUFFIX,
        CLUSTER_PROFILE_FILENAME,
    )
    return run / CLUSTER_PROFILE_FILENAME


def resolve_user_profile_vectors() -> Path:
    if USER_PROFILE_RUN_DIR.strip():
        run = Path(USER_PROFILE_RUN_DIR.strip())
        p = run / USER_PROFILE_VECTOR_FILE
        if not p.exists():
            raise FileNotFoundError(f"USER_PROFILE_VECTOR_FILE not found: {p}")
        return p
    suffix = f"_{RUN_PROFILE}{USER_PROFILE_RUN_SUFFIX}"
    run = pick_latest_run(USER_PROFILE_ROOT, suffix, USER_PROFILE_VECTOR_FILE)
    return run / USER_PROFILE_VECTOR_FILE


def resolve_user_profile_table(vector_path: Path) -> Path:
    run_dir = vector_path.parent
    table_path = run_dir / USER_PROFILE_TABLE_FILE
    if not table_path.exists():
        raise FileNotFoundError(f"user profile table not found: {table_path}")
    return table_path


def resolve_user_profile_tag_long(table_path: Path) -> Path | None:
    p = table_path.parent / USER_PROFILE_TAG_LONG_FILE
    if p.exists():
        return p
    return None


def resolve_user_profile_multivector_file(vector_path: Path, scope: str) -> Path | None:
    p = vector_path.parent / USER_PROFILE_MULTI_VECTOR_FILE_TEMPLATE.format(scope=str(scope).strip().lower())
    if p.exists():
        return p
    return None


def resolve_item_semantic_features() -> Path:
    if ITEM_SEMANTIC_RUN_DIR.strip():
        run = Path(ITEM_SEMANTIC_RUN_DIR.strip())
        p = run / ITEM_SEMANTIC_FEATURE_FILE
        if not p.exists():
            raise FileNotFoundError(f"ITEM_SEMANTIC_FEATURE_FILE not found: {p}")
        return p
    suffix = f"_{RUN_PROFILE}{ITEM_SEMANTIC_RUN_SUFFIX}"
    run = pick_latest_run(ITEM_SEMANTIC_ROOT, suffix, ITEM_SEMANTIC_FEATURE_FILE)
    return run / ITEM_SEMANTIC_FEATURE_FILE


def resolve_item_semantic_tag_long(features_path: Path) -> Path | None:
    p = features_path.parent / ITEM_SEMANTIC_TAG_LONG_FILE
    if p.exists():
        return p
    return None


def resolve_item_semantic_dense_vector_file(features_path: Path, scope: str) -> Path | None:
    p = features_path.parent / ITEM_SEMANTIC_DENSE_VECTOR_FILE_TEMPLATE.format(scope=str(scope).strip().lower())
    if p.exists():
        return p
    return None


def resolve_business_context_features() -> Path:
    if BUSINESS_CONTEXT_RUN_DIR.strip():
        run = Path(BUSINESS_CONTEXT_RUN_DIR.strip())
        p = run / BUSINESS_CONTEXT_FEATURE_FILE
        if not p.exists():
            raise FileNotFoundError(f"BUSINESS_CONTEXT_FEATURE_FILE not found: {p}")
        return p
    suffix = f"_{RUN_PROFILE}{BUSINESS_CONTEXT_RUN_SUFFIX}"
    run = pick_latest_run(BUSINESS_CONTEXT_ROOT, suffix, BUSINESS_CONTEXT_FEATURE_FILE)
    return run / BUSINESS_CONTEXT_FEATURE_FILE


def resolve_review_evidence_business_agg() -> Path:
    if REVIEW_EVIDENCE_RUN_DIR.strip():
        run = Path(REVIEW_EVIDENCE_RUN_DIR.strip())
        p = run / REVIEW_EVIDENCE_BUSINESS_FILE
        if not p.exists():
            raise FileNotFoundError(f"REVIEW_EVIDENCE_BUSINESS_FILE not found: {p}")
        return p
    suffix = f"_{RUN_PROFILE}{REVIEW_EVIDENCE_RUN_SUFFIX}"
    run = pick_latest_run(REVIEW_EVIDENCE_ROOT, suffix, REVIEW_EVIDENCE_BUSINESS_FILE)
    return run / REVIEW_EVIDENCE_BUSINESS_FILE


def resolve_tip_signal_business_agg() -> Path:
    if TIP_SIGNAL_RUN_DIR.strip():
        run = Path(TIP_SIGNAL_RUN_DIR.strip())
        p = run / TIP_SIGNAL_BUSINESS_FILE
        if not p.exists():
            raise FileNotFoundError(f"TIP_SIGNAL_BUSINESS_FILE not found: {p}")
        return p
    suffix = f"_{RUN_PROFILE}{TIP_SIGNAL_RUN_SUFFIX}"
    run = pick_latest_run(TIP_SIGNAL_ROOT, suffix, TIP_SIGNAL_BUSINESS_FILE)
    return run / TIP_SIGNAL_BUSINESS_FILE


def resolve_checkin_context_business_agg() -> Path:
    if CHECKIN_CONTEXT_RUN_DIR.strip():
        run = Path(CHECKIN_CONTEXT_RUN_DIR.strip())
        p = run / CHECKIN_CONTEXT_BUSINESS_FILE
        if not p.exists():
            raise FileNotFoundError(f"CHECKIN_CONTEXT_BUSINESS_FILE not found: {p}")
        return p
    suffix = f"_{RUN_PROFILE}{CHECKIN_CONTEXT_RUN_SUFFIX}"
    run = pick_latest_run(CHECKIN_CONTEXT_ROOT, suffix, CHECKIN_CONTEXT_BUSINESS_FILE)
    return run / CHECKIN_CONTEXT_BUSINESS_FILE


def load_business_context_features(spark: SparkSession, parquet_path: Path) -> DataFrame:
    return (
        spark.read.parquet(parquet_path.as_posix())
        .select(
            "business_id",
            F.col("latitude").cast("double").alias("ctx_latitude"),
            F.col("longitude").cast("double").alias("ctx_longitude"),
            F.coalesce(F.col("has_hours").cast("double"), F.lit(0.0)).alias("ctx_has_hours"),
            F.coalesce(F.col("open_weekend").cast("double"), F.lit(0.0)).alias("ctx_open_weekend"),
            F.coalesce(F.col("open_late_any").cast("double"), F.lit(0.0)).alias("ctx_open_late_any"),
            F.coalesce(F.col("meal_any").cast("double"), F.lit(0.0)).alias("ctx_meal_any"),
            F.coalesce(F.col("meal_breakfast").cast("double"), F.lit(0.0)).alias("ctx_meal_breakfast"),
            F.coalesce(F.col("meal_brunch").cast("double"), F.lit(0.0)).alias("ctx_meal_brunch"),
            F.coalesce(F.col("meal_lunch").cast("double"), F.lit(0.0)).alias("ctx_meal_lunch"),
            F.coalesce(F.col("meal_dinner").cast("double"), F.lit(0.0)).alias("ctx_meal_dinner"),
            F.coalesce(F.col("meal_latenight").cast("double"), F.lit(0.0)).alias("ctx_meal_latenight"),
            F.coalesce(F.col("meal_dessert").cast("double"), F.lit(0.0)).alias("ctx_meal_dessert"),
            F.coalesce(F.col("attr_delivery").cast("double"), F.lit(0.0)).alias("ctx_attr_delivery"),
            F.coalesce(F.col("attr_takeout").cast("double"), F.lit(0.0)).alias("ctx_attr_takeout"),
            F.coalesce(F.col("attr_reservations").cast("double"), F.lit(0.0)).alias("ctx_attr_reservations"),
            F.coalesce(F.col("attr_groups").cast("double"), F.lit(0.0)).alias("ctx_attr_groups"),
            F.coalesce(F.col("attr_good_for_kids").cast("double"), F.lit(0.0)).alias("ctx_attr_good_for_kids"),
            F.coalesce(F.col("attr_outdoor").cast("double"), F.lit(0.0)).alias("ctx_attr_outdoor"),
            F.coalesce(F.col("attr_table_service").cast("double"), F.lit(0.0)).alias("ctx_attr_table_service"),
            F.coalesce(F.col("attr_alcohol_full_bar").cast("double"), F.lit(0.0)).alias("ctx_attr_alcohol_full_bar"),
            F.coalesce(F.col("attr_alcohol_beer_wine").cast("double"), F.lit(0.0)).alias("ctx_attr_alcohol_beer_wine"),
            F.coalesce(F.col("attr_price_range").cast("double"), F.lit(0.0)).alias("ctx_attr_price_range"),
        )
        .withColumn(
            "ctx_attr_price_range_norm",
            F.when(F.col("ctx_attr_price_range") > F.lit(0.0), F.least(F.col("ctx_attr_price_range") / F.lit(4.0), F.lit(1.0)))
            .otherwise(F.lit(0.0)),
        )
        .drop("ctx_attr_price_range")
    )


def load_review_evidence_business_agg(spark: SparkSession, parquet_path: Path) -> DataFrame:
    return (
        spark.read.parquet(parquet_path.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("review_rows").cast("double"), F.lit(0.0)).alias("ctx_review_rows"),
            F.coalesce(F.col("avg_evidence_weight_v1").cast("double"), F.lit(0.0)).alias("ctx_review_avg_evidence"),
            F.coalesce(F.col("high_vote_review_rows").cast("double"), F.lit(0.0)).alias("ctx_review_high_vote_rows"),
            F.coalesce(F.col("recent_2y_review_rows").cast("double"), F.lit(0.0)).alias("ctx_review_recent_rows"),
        )
        .withColumn(
            "ctx_review_high_vote_share",
            F.when(F.col("ctx_review_rows") > F.lit(0.0), F.col("ctx_review_high_vote_rows") / F.col("ctx_review_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_review_recent_share",
            F.when(F.col("ctx_review_rows") > F.lit(0.0), F.col("ctx_review_recent_rows") / F.col("ctx_review_rows"))
            .otherwise(F.lit(0.0)),
        )
    )


def load_tip_signal_business_agg(spark: SparkSession, parquet_path: Path) -> DataFrame:
    return (
        spark.read.parquet(parquet_path.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("tip_rows").cast("double"), F.lit(0.0)).alias("ctx_tip_rows"),
            F.coalesce(F.col("avg_tip_weight_v1").cast("double"), F.lit(0.0)).alias("ctx_tip_avg_weight"),
            F.coalesce(F.col("recommend_tip_rows").cast("double"), F.lit(0.0)).alias("ctx_tip_recommend_rows"),
            F.coalesce(F.col("time_tip_rows").cast("double"), F.lit(0.0)).alias("ctx_tip_time_rows"),
            F.coalesce(F.col("dish_tip_rows").cast("double"), F.lit(0.0)).alias("ctx_tip_dish_rows"),
            F.coalesce(F.col("recent_1y_tip_rows").cast("double"), F.lit(0.0)).alias("ctx_tip_recent_rows"),
        )
        .withColumn(
            "ctx_tip_recommend_share",
            F.when(F.col("ctx_tip_rows") > F.lit(0.0), F.col("ctx_tip_recommend_rows") / F.col("ctx_tip_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_tip_time_share",
            F.when(F.col("ctx_tip_rows") > F.lit(0.0), F.col("ctx_tip_time_rows") / F.col("ctx_tip_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_tip_dish_share",
            F.when(F.col("ctx_tip_rows") > F.lit(0.0), F.col("ctx_tip_dish_rows") / F.col("ctx_tip_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_tip_recent_share",
            F.when(F.col("ctx_tip_rows") > F.lit(0.0), F.col("ctx_tip_recent_rows") / F.col("ctx_tip_rows"))
            .otherwise(F.lit(0.0)),
        )
    )


def load_checkin_context_business_agg(spark: SparkSession, parquet_path: Path) -> DataFrame:
    return (
        spark.read.parquet(parquet_path.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("checkin_rows").cast("double"), F.lit(0.0)).alias("ctx_checkin_rows"),
            F.coalesce(F.col("weekend_share").cast("double"), F.lit(0.0)).alias("ctx_checkin_weekend_share"),
            F.coalesce(F.col("breakfast_rows").cast("double"), F.lit(0.0)).alias("ctx_checkin_breakfast_rows"),
            F.coalesce(F.col("lunch_rows").cast("double"), F.lit(0.0)).alias("ctx_checkin_lunch_rows"),
            F.coalesce(F.col("dinner_rows").cast("double"), F.lit(0.0)).alias("ctx_checkin_dinner_rows"),
            F.coalesce(F.col("late_rows").cast("double"), F.lit(0.0)).alias("ctx_checkin_late_rows"),
            F.coalesce(F.col("recent_2y_checkin_rows").cast("double"), F.lit(0.0)).alias("ctx_checkin_recent_rows"),
            F.coalesce(F.col("repeat_density_proxy").cast("double"), F.lit(0.0)).alias("ctx_checkin_repeat_density"),
        )
        .withColumn(
            "ctx_checkin_lunch_share",
            F.when(F.col("ctx_checkin_rows") > F.lit(0.0), F.col("ctx_checkin_lunch_rows") / F.col("ctx_checkin_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_checkin_dinner_share",
            F.when(F.col("ctx_checkin_rows") > F.lit(0.0), F.col("ctx_checkin_dinner_rows") / F.col("ctx_checkin_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_checkin_late_share",
            F.when(F.col("ctx_checkin_rows") > F.lit(0.0), F.col("ctx_checkin_late_rows") / F.col("ctx_checkin_rows"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_checkin_recent_share",
            F.when(F.col("ctx_checkin_rows") > F.lit(0.0), F.col("ctx_checkin_recent_rows") / F.col("ctx_checkin_rows"))
            .otherwise(F.lit(0.0)),
        )
    )


def load_item_semantic_features(spark: SparkSession, csv_path: Path) -> DataFrame:
    sem = (
        spark.read.option("header", True)
        .csv(csv_path.as_posix())
        .select(
            F.col("business_id").cast("string").alias("business_id"),
            F.col("semantic_score").cast("double").alias("semantic_score"),
            F.col("semantic_confidence").cast("double").alias("semantic_confidence"),
            F.col("semantic_support").cast("double").alias("semantic_support"),
            F.col("semantic_tag_richness").cast("double").alias("semantic_tag_richness"),
        )
        .filter(F.col("business_id").isNotNull())
        .dropDuplicates(["business_id"])
    )
    return sem


def _norm_tag_type(x: Any) -> str:
    t = str(x or "").strip().lower()
    if not t:
        return "other"
    alias = {
        "ambience": "scene",
        "audience": "scene",
        "beverage": "beverage",
        "meal": "meal",
        "taste": "taste",
        "service": "service",
        "value": "value",
        "cuisine": "cuisine",
    }
    return alias.get(t, t)


def load_user_profile_tag_long(path: Path | None) -> pd.DataFrame:
    if path is None or (not path.exists()):
        return pd.DataFrame(columns=["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"])
    usecols = ["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support", "abs_net_w"]
    pdf = pd.read_csv(path.as_posix(), usecols=lambda c: c in set(usecols), low_memory=False)
    if pdf.empty:
        return pd.DataFrame(columns=["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"])
    for c in ["user_id", "tag"]:
        if c not in pdf.columns:
            pdf[c] = ""
    if "tag_type" not in pdf.columns:
        pdf["tag_type"] = "other"
    if "net_w" not in pdf.columns:
        if "abs_net_w" in pdf.columns:
            pdf["net_w"] = pd.to_numeric(pdf["abs_net_w"], errors="coerce").fillna(0.0)
        else:
            pdf["net_w"] = 0.0
    if "tag_confidence" not in pdf.columns:
        pdf["tag_confidence"] = 1.0
    if "support" not in pdf.columns:
        pdf["support"] = 1.0
    pdf["user_id"] = pdf["user_id"].astype(str)
    pdf["tag"] = pdf["tag"].astype(str).str.strip().str.lower()
    pdf["tag_type"] = pdf["tag_type"].map(_norm_tag_type)
    pdf["net_w"] = pd.to_numeric(pdf["net_w"], errors="coerce").fillna(0.0)
    pdf["tag_confidence"] = pd.to_numeric(pdf["tag_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    pdf["support"] = pd.to_numeric(pdf["support"], errors="coerce").fillna(1.0).clip(lower=0.0)
    pdf = pdf[(pdf["user_id"] != "") & (pdf["tag"] != "")]
    return pdf[["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"]]


def load_item_semantic_tag_long(path: Path | None) -> pd.DataFrame:
    if path is None or (not path.exists()):
        return pd.DataFrame(
            columns=["business_id", "tag", "tag_type", "net_weight_sum", "tag_confidence", "support_count"]
        )
    usecols = [
        "business_id",
        "tag",
        "facet",
        "tag_type",
        "net_weight_sum",
        "tag_confidence",
        "support_count",
    ]
    pdf = pd.read_csv(path.as_posix(), usecols=lambda c: c in set(usecols), low_memory=False)
    if pdf.empty:
        return pd.DataFrame(
            columns=["business_id", "tag", "tag_type", "net_weight_sum", "tag_confidence", "support_count"]
        )
    for c in ["business_id", "tag"]:
        if c not in pdf.columns:
            pdf[c] = ""
    if "tag_type" not in pdf.columns:
        pdf["tag_type"] = pdf.get("facet", "other")
    if "net_weight_sum" not in pdf.columns:
        pdf["net_weight_sum"] = 0.0
    if "tag_confidence" not in pdf.columns:
        pdf["tag_confidence"] = 1.0
    if "support_count" not in pdf.columns:
        pdf["support_count"] = 1.0
    pdf["business_id"] = pdf["business_id"].astype(str)
    pdf["tag"] = pdf["tag"].astype(str).str.strip().str.lower()
    pdf["tag_type"] = pdf["tag_type"].map(_norm_tag_type)
    pdf["net_weight_sum"] = pd.to_numeric(pdf["net_weight_sum"], errors="coerce").fillna(0.0)
    pdf["tag_confidence"] = pd.to_numeric(pdf["tag_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    pdf["support_count"] = pd.to_numeric(pdf["support_count"], errors="coerce").fillna(1.0).clip(lower=0.0)
    pdf = pdf[(pdf["business_id"] != "") & (pdf["tag"] != "")]
    return pdf[["business_id", "tag", "tag_type", "net_weight_sum", "tag_confidence", "support_count"]]


def load_user_profile_confidence(table_path: Path) -> dict[str, float]:
    use_cols = [
        "user_id",
        "profile_confidence_v1",
        "profile_confidence",
        "profile_confidence_v2",
        "profile_conf_consistency",
        "profile_tag_support",
        "n_sentences_selected",
    ]
    prof = pd.read_csv(table_path.as_posix(), usecols=lambda c: c in set(use_cols))
    if "user_id" not in prof.columns:
        return {}
    prof["user_id"] = prof["user_id"].astype(str)
    has_v1 = "profile_confidence_v1" in prof.columns or "profile_confidence" in prof.columns
    has_v2 = "profile_confidence_v2" in prof.columns
    if "profile_confidence_v1" not in prof.columns:
        prof["profile_confidence_v1"] = prof.get("profile_confidence", 0.0)
    if "profile_confidence" not in prof.columns:
        prof["profile_confidence"] = prof.get("profile_confidence_v1", 0.0)
    if not has_v2:
        prof["profile_confidence_v2"] = prof["profile_confidence"]
    if "n_sentences_selected" not in prof.columns:
        prof["n_sentences_selected"] = 0
    if "profile_conf_consistency" not in prof.columns:
        prof["profile_conf_consistency"] = 0.5
    if "profile_tag_support" not in prof.columns:
        prof["profile_tag_support"] = 0.0
    conf_v1 = pd.to_numeric(prof["profile_confidence_v1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    conf_v2 = pd.to_numeric(prof["profile_confidence_v2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    blend = float(np.clip(float(PROFILE_CONF_V2_BLEND), 0.0, 1.0))
    if has_v1 and has_v2:
        conf_raw = (1.0 - blend) * conf_v1 + blend * conf_v2
    elif has_v2:
        conf_raw = conf_v2
    else:
        conf_raw = conf_v1
    prof["profile_confidence"] = conf_raw
    prof["n_sentences_selected"] = pd.to_numeric(prof["n_sentences_selected"], errors="coerce").fillna(0).astype(int)
    prof["profile_conf_consistency"] = pd.to_numeric(
        prof["profile_conf_consistency"], errors="coerce"
    ).fillna(0.5)
    prof["profile_tag_support"] = pd.to_numeric(prof["profile_tag_support"], errors="coerce").fillna(0.0)
    prof["profile_confidence"] = np.clip(prof["profile_confidence"], 0.0, 1.0)
    # Light sentence-quality gating to avoid unstable profile vectors dominating scores.
    sentence_factor = np.minimum(1.0, prof["n_sentences_selected"].to_numpy(dtype=np.float32) / float(max(1, PROFILE_MIN_SENTENCES)))
    consistency = np.clip(prof["profile_conf_consistency"].to_numpy(dtype=np.float32), 0.0, 1.0)
    support_factor = np.minimum(1.0, np.log1p(prof["profile_tag_support"].to_numpy(dtype=np.float32)) / np.log(5.0))
    prof["profile_confidence"] = (
        prof["profile_confidence"].to_numpy(dtype=np.float32)
        * sentence_factor
        * (0.8 + 0.2 * consistency)
        * (0.85 + 0.15 * support_factor)
    )
    return dict(zip(prof["user_id"].tolist(), prof["profile_confidence"].tolist()))


def resolve_profile_calibration_json() -> Path | None:
    path_text = PROFILE_CALIBRATION_JSON_OVERRIDE.strip()
    if not path_text:
        return None
    p = Path(path_text)
    if not p.exists():
        raise FileNotFoundError(f"PROFILE_CALIBRATION_JSON_OVERRIDE not found: {p}")
    return p


def load_profile_calibration_models(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    models_raw = payload.get("models_by_bucket", {})
    out: dict[int, dict[str, Any]] = {}
    for k, v in models_raw.items():
        try:
            out[int(k)] = dict(v)
        except Exception:
            continue
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_profile_scores(
    raw_scores: np.ndarray,
    ranks: np.ndarray,
    source_conf: np.ndarray,
    model_cfg: dict[str, Any],
) -> np.ndarray:
    mean = np.asarray(model_cfg.get("mean", []), dtype=np.float32)
    std = np.asarray(model_cfg.get("std", []), dtype=np.float32)
    coef = np.asarray(model_cfg.get("coef", []), dtype=np.float32)
    intercept = float(model_cfg.get("intercept", 0.0))
    if mean.shape[0] != 5 or std.shape[0] != 5 or coef.shape[0] != 5:
        return raw_scores
    std[std == 0.0] = 1.0
    rank_norm = 1.0 / np.log2(ranks.astype(np.float32) + 1.0)
    feat = np.stack(
        [
            raw_scores.astype(np.float32),
            ranks.astype(np.float32),
            source_conf.astype(np.float32),
            raw_scores.astype(np.float32) * source_conf.astype(np.float32),
            raw_scores.astype(np.float32) * rank_norm,
        ],
        axis=1,
    )
    z = np.matmul((feat - mean) / std, coef) + intercept
    return _sigmoid_np(z.astype(np.float32)).astype(np.float32)


def _load_named_vectors(npz_path: Path, id_key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path.as_posix(), allow_pickle=False)
    if id_key not in data:
        raise RuntimeError(f"missing '{id_key}' in vector file: {npz_path}")
    user_ids = data[id_key].astype(str)
    vectors = data["vectors"].astype(np.float32)
    if vectors.ndim != 2:
        raise RuntimeError(f"invalid vector shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = (vectors / norms).astype(np.float32)
    return user_ids, vectors


def load_profile_vectors(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return _load_named_vectors(npz_path, "user_ids")


def load_merchant_dense_vectors(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return _load_named_vectors(npz_path, "business_ids")


def _empty_source_df(spark: SparkSession, source: str) -> DataFrame:
    return (
        spark.range(0)
        .select(
            F.lit(None).cast("int").alias("user_idx"),
            F.lit(None).cast("int").alias("item_idx"),
            F.lit(None).cast("int").alias("source_rank"),
            F.lit(None).cast("double").alias("source_score"),
            F.lit(None).cast("double").alias("source_confidence"),
            F.lit(source).cast("string").alias("source"),
            F.lit(source).cast("string").alias("source_detail"),
        )
        .limit(0)
    )


def _empty_multivector_route_audit_df(spark: SparkSession) -> DataFrame:
    return (
        spark.range(0)
        .select(
            F.lit(None).cast("int").alias("user_idx"),
            F.lit(None).cast("int").alias("item_idx"),
            F.lit(None).cast("int").alias("route_rank"),
            F.lit(None).cast("double").alias("route_score"),
            F.lit(None).cast("double").alias("route_confidence"),
            F.lit(None).cast("string").alias("route_name"),
            F.lit(None).cast("string").alias("route_role"),
            F.lit(None).cast("string").alias("user_vector_scope"),
            F.lit(None).cast("string").alias("merchant_vector_scope"),
        )
        .limit(0)
    )


def _build_multivector_scope_pair_list() -> list[tuple[str, str, str]]:
    target_by_scope = {
        "short": "semantic",
        "long": "core",
        "pos": "pos",
        "neg": "neg",
    }
    role_by_scope = {
        "short": "candidate",
        "long": "candidate",
        "pos": "candidate",
        "neg": "penalty",
    }
    out: list[tuple[str, str, str]] = []
    for scope in PROFILE_MULTIVECTOR_ROUTE_SCOPES:
        merchant_scope = target_by_scope.get(str(scope).strip().lower())
        if merchant_scope is None:
            continue
        out.append((str(scope).strip().lower(), merchant_scope, role_by_scope.get(str(scope).strip().lower(), "candidate")))
    return out


def build_profile_multivector_route_audit_df(
    spark: SparkSession,
    test_users: DataFrame,
    user_map: DataFrame,
    item_map: DataFrame,
    profile_confidence_by_user_id: dict[str, float],
    user_vectors_by_scope: dict[str, tuple[np.ndarray, np.ndarray]],
    merchant_vectors_by_scope: dict[str, tuple[np.ndarray, np.ndarray]],
    route_top_k: int,
    route_score_min: float,
    batch_users: int,
) -> tuple[DataFrame, dict[str, Any]]:
    scope_pairs = _build_multivector_scope_pair_list()
    if not scope_pairs:
        return _empty_multivector_route_audit_df(spark), {"status": "no_scope_pairs"}
    if not user_vectors_by_scope:
        return _empty_multivector_route_audit_df(spark), {"status": "no_user_multivectors"}
    if not merchant_vectors_by_scope:
        return _empty_multivector_route_audit_df(spark), {"status": "no_merchant_dense_vectors"}

    test_pdf = (
        test_users.join(user_map, on="user_idx", how="left")
        .select("user_idx", "user_id")
        .toPandas()
    )
    if test_pdf.empty:
        return _empty_multivector_route_audit_df(spark), {"status": "empty_test_users"}
    test_pdf["user_idx"] = test_pdf["user_idx"].astype(np.int32)
    test_pdf["user_id"] = test_pdf["user_id"].fillna("").astype(str)
    test_pdf["profile_confidence"] = (
        test_pdf["user_id"].map(profile_confidence_by_user_id).fillna(0.0).astype(np.float32)
    )
    test_pdf = test_pdf[test_pdf["profile_confidence"] >= float(PROFILE_CONFIDENCE_MIN)].copy()
    if test_pdf.empty:
        return _empty_multivector_route_audit_df(spark), {"status": "no_confident_test_users"}

    item_pdf = item_map.select("item_idx", "business_id").dropDuplicates(["item_idx"]).toPandas()
    if item_pdf.empty:
        return _empty_multivector_route_audit_df(spark), {"status": "empty_item_map"}
    item_pdf["item_idx"] = item_pdf["item_idx"].astype(np.int32)
    item_pdf["business_id"] = item_pdf["business_id"].astype(str)

    merchant_index_by_scope: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for merchant_scope, (business_ids, merchant_vectors) in merchant_vectors_by_scope.items():
        bid_to_vecidx = {str(bid): int(i) for i, bid in enumerate(business_ids.tolist())}
        scope_item_pdf = item_pdf.copy()
        scope_item_pdf["vec_idx"] = scope_item_pdf["business_id"].map(bid_to_vecidx)
        scope_item_pdf = scope_item_pdf.dropna(subset=["vec_idx"]).copy()
        if scope_item_pdf.empty:
            continue
        scope_item_pdf["vec_idx"] = scope_item_pdf["vec_idx"].astype(np.int32)
        item_indices = scope_item_pdf["item_idx"].to_numpy(np.int32)
        vec_indices = scope_item_pdf["vec_idx"].to_numpy(np.int32)
        merchant_index_by_scope[str(merchant_scope)] = (
            item_indices,
            merchant_vectors[vec_indices],
        )

    if not merchant_index_by_scope:
        return _empty_multivector_route_audit_df(spark), {"status": "no_items_with_dense_vectors"}

    rows: list[tuple[int, int, int, float, float, str, str, str, str]] = []
    route_rows: dict[str, int] = {}
    route_users: dict[str, int] = {}
    scope_status: dict[str, Any] = {}

    for user_scope, merchant_scope, route_role in scope_pairs:
        route_name = f"profile_mv_{user_scope}_{merchant_scope}"
        user_entry = user_vectors_by_scope.get(str(user_scope))
        merchant_entry = merchant_index_by_scope.get(str(merchant_scope))
        if user_entry is None:
            scope_status[route_name] = {"status": "missing_user_scope"}
            continue
        if merchant_entry is None:
            scope_status[route_name] = {"status": "missing_merchant_scope"}
            continue

        scope_user_ids, scope_user_vectors = user_entry
        uid_to_vecidx = {str(uid): int(i) for i, uid in enumerate(scope_user_ids.tolist())}
        route_user_pdf = test_pdf.copy()
        route_user_pdf["vec_idx"] = route_user_pdf["user_id"].map(uid_to_vecidx)
        route_user_pdf = route_user_pdf.dropna(subset=["vec_idx"]).copy()
        if route_user_pdf.empty:
            scope_status[route_name] = {"status": "no_users_with_scope_vector"}
            continue
        route_user_pdf["vec_idx"] = route_user_pdf["vec_idx"].astype(np.int32)
        item_indices, merchant_vectors = merchant_entry
        if merchant_vectors.shape[0] == 0:
            scope_status[route_name] = {"status": "empty_merchant_matrix"}
            continue
        kk = min(int(max(1, route_top_k)), int(merchant_vectors.shape[0]))
        min_score = float(route_score_min)
        written_before = len(rows)
        route_user_count = 0

        for start in range(0, int(route_user_pdf.shape[0]), int(max(1, batch_users))):
            batch_pdf = route_user_pdf.iloc[start : start + int(max(1, batch_users))].copy()
            if batch_pdf.empty:
                continue
            batch_vecs = scope_user_vectors[batch_pdf["vec_idx"].to_numpy(np.int32)]
            score = np.matmul(batch_vecs, merchant_vectors.T).astype(np.float32)
            if score.size == 0:
                continue
            top_idx = np.argpartition(-score, kth=kk - 1, axis=1)[:, :kk]
            batch_conf = batch_pdf["profile_confidence"].to_numpy(np.float32)
            batch_uidx = batch_pdf["user_idx"].to_numpy(np.int32)
            for row_pos in range(top_idx.shape[0]):
                idxs = top_idx[row_pos]
                order = idxs[np.argsort(-score[row_pos, idxs], kind="mergesort")]
                src_conf = float(
                    PROFILE_CONFIDENCE_FLOOR
                    + (1.0 - float(PROFILE_CONFIDENCE_FLOOR)) * float(np.clip(batch_conf[row_pos], 0.0, 1.0))
                )
                rank_j = 0
                for item_pos in order.tolist():
                    sc = float(score[row_pos, int(item_pos)])
                    if sc < min_score:
                        continue
                    rank_j += 1
                    rows.append(
                        (
                            int(batch_uidx[row_pos]),
                            int(item_indices[int(item_pos)]),
                            int(rank_j),
                            float(sc),
                            float(src_conf),
                            route_name,
                            route_role,
                            str(user_scope),
                            str(merchant_scope),
                        )
                    )
                if rank_j > 0:
                    route_user_count += 1

        route_rows[route_name] = int(len(rows) - written_before)
        route_users[route_name] = int(route_user_count)
        scope_status[route_name] = {
            "status": "ok",
            "rows": int(route_rows[route_name]),
            "users": int(route_user_count),
            "top_k": int(kk),
            "route_role": str(route_role),
        }

    if not rows:
        return _empty_multivector_route_audit_df(spark), {
            "status": "no_multivector_rows",
            "route_status": scope_status,
        }

    route_pdf = pd.DataFrame(
        rows,
        columns=[
            "user_idx",
            "item_idx",
            "route_rank",
            "route_score",
            "route_confidence",
            "route_name",
            "route_role",
            "user_vector_scope",
            "merchant_vector_scope",
        ],
    )
    route_pdf["user_idx"] = pd.to_numeric(route_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    route_pdf["item_idx"] = pd.to_numeric(route_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    route_pdf["route_rank"] = pd.to_numeric(route_pdf["route_rank"], errors="coerce").fillna(0).astype(np.int32)
    route_pdf["route_score"] = pd.to_numeric(route_pdf["route_score"], errors="coerce").fillna(0.0).astype(np.float64)
    route_pdf["route_confidence"] = (
        pd.to_numeric(route_pdf["route_confidence"], errors="coerce").fillna(0.0).astype(np.float64)
    )
    route_pdf = route_pdf[(route_pdf["user_idx"] >= 0) & (route_pdf["item_idx"] >= 0) & (route_pdf["route_rank"] > 0)].copy()
    route_pdf = route_pdf.sort_values(
        ["route_name", "user_idx", "route_rank", "item_idx"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    if route_pdf.empty:
        return _empty_multivector_route_audit_df(spark), {
            "status": "multivector_rows_invalid_after_cast",
            "route_status": scope_status,
        }

    route_df = _create_spark_df_from_pandas_safe(spark=spark, pdf=route_pdf).select(
        F.col("user_idx").cast("int"),
        F.col("item_idx").cast("int"),
        F.col("route_rank").cast("int"),
        F.col("route_score").cast("double"),
        F.col("route_confidence").cast("double"),
        F.col("route_name").cast("string"),
        F.col("route_role").cast("string"),
        F.col("user_vector_scope").cast("string"),
        F.col("merchant_vector_scope").cast("string"),
    )
    meta = {
        "status": "ok",
        "rows": int(route_pdf.shape[0]),
        "route_rows": {str(k): int(v) for k, v in route_rows.items()},
        "route_users": {str(k): int(v) for k, v in route_users.items()},
        "route_status": scope_status,
        "batch_users": int(batch_users),
        "top_k": int(route_top_k),
        "score_min": float(route_score_min),
    }
    return route_df, meta


def build_profile_multivector_candidate_pdf(
    test_pdf_conf: pd.DataFrame,
    item_map: DataFrame,
    user_vectors_by_scope: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    merchant_vectors_by_scope: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    route_top_k: int,
    route_score_min: float,
    batch_users: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_pdf = pd.DataFrame(
        columns=[
            "user_idx",
            "item_idx",
            "source_rank",
            "source_score",
            "source_confidence",
            "source",
            "source_detail",
        ]
    )
    if test_pdf_conf.empty:
        return empty_pdf, {"status": "empty_test_users"}
    if not user_vectors_by_scope:
        return empty_pdf, {"status": "no_user_multivectors"}
    if not merchant_vectors_by_scope:
        return empty_pdf, {"status": "no_merchant_dense_vectors"}

    scope_pairs = [
        (user_scope, merchant_scope, route_role)
        for user_scope, merchant_scope, route_role in _build_multivector_scope_pair_list()
        if str(route_role) == "candidate"
    ]
    if not scope_pairs:
        return empty_pdf, {"status": "no_candidate_scope_pairs"}

    item_pdf = item_map.select("item_idx", "business_id").dropDuplicates(["item_idx"]).toPandas()
    if item_pdf.empty:
        return empty_pdf, {"status": "empty_item_map"}
    item_pdf["item_idx"] = item_pdf["item_idx"].astype(np.int32)
    item_pdf["business_id"] = item_pdf["business_id"].astype(str)

    merchant_index_by_scope: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for merchant_scope, (business_ids, merchant_vectors) in merchant_vectors_by_scope.items():
        bid_to_vecidx = {str(bid): int(i) for i, bid in enumerate(business_ids.tolist())}
        scope_item_pdf = item_pdf.copy()
        scope_item_pdf["vec_idx"] = scope_item_pdf["business_id"].map(bid_to_vecidx)
        scope_item_pdf = scope_item_pdf.dropna(subset=["vec_idx"]).copy()
        if scope_item_pdf.empty:
            continue
        scope_item_pdf["vec_idx"] = scope_item_pdf["vec_idx"].astype(np.int32)
        item_indices = scope_item_pdf["item_idx"].to_numpy(np.int32)
        vec_indices = scope_item_pdf["vec_idx"].to_numpy(np.int32)
        merchant_index_by_scope[str(merchant_scope)] = (
            item_indices,
            merchant_vectors[vec_indices],
        )
    if not merchant_index_by_scope:
        return empty_pdf, {"status": "no_items_with_dense_vectors"}

    rows: list[tuple[int, int, int, float, float, str, str]] = []
    route_rows: dict[str, int] = {}
    route_users: dict[str, int] = {}
    route_status: dict[str, Any] = {}
    batch_n = int(max(1, batch_users))
    min_score = float(route_score_min)

    for user_scope, merchant_scope, _route_role in scope_pairs:
        route_name = f"profile_mv_{user_scope}_{merchant_scope}"
        user_entry = user_vectors_by_scope.get(str(user_scope))
        merchant_entry = merchant_index_by_scope.get(str(merchant_scope))
        if user_entry is None:
            route_status[route_name] = {"status": "missing_user_scope"}
            continue
        if merchant_entry is None:
            route_status[route_name] = {"status": "missing_merchant_scope"}
            continue

        scope_user_ids, scope_user_vectors = user_entry
        uid_to_vecidx = {str(uid): int(i) for i, uid in enumerate(scope_user_ids.tolist())}
        route_user_pdf = test_pdf_conf[["user_idx", "user_id", "profile_conf"]].copy()
        route_user_pdf["vec_idx"] = route_user_pdf["user_id"].map(uid_to_vecidx)
        route_user_pdf = route_user_pdf.dropna(subset=["vec_idx"]).copy()
        if route_user_pdf.empty:
            route_status[route_name] = {"status": "no_users_with_scope_vector"}
            continue
        route_user_pdf["vec_idx"] = route_user_pdf["vec_idx"].astype(np.int32)
        item_indices, merchant_vectors = merchant_entry
        if merchant_vectors.shape[0] == 0:
            route_status[route_name] = {"status": "empty_merchant_matrix"}
            continue
        kk = min(int(max(1, route_top_k)), int(merchant_vectors.shape[0]))
        written_before = len(rows)
        route_user_count = 0

        for start in range(0, int(route_user_pdf.shape[0]), batch_n):
            batch_pdf = route_user_pdf.iloc[start : start + batch_n].copy()
            if batch_pdf.empty:
                continue
            batch_vecs = scope_user_vectors[batch_pdf["vec_idx"].to_numpy(np.int32)]
            score = np.matmul(batch_vecs, merchant_vectors.T).astype(np.float32)
            if score.size == 0:
                continue
            top_idx = np.argpartition(-score, kth=kk - 1, axis=1)[:, :kk]
            batch_conf = batch_pdf["profile_conf"].to_numpy(np.float32)
            batch_uidx = batch_pdf["user_idx"].to_numpy(np.int32)
            for row_pos in range(top_idx.shape[0]):
                idxs = top_idx[row_pos]
                order = idxs[np.argsort(-score[row_pos, idxs], kind="mergesort")]
                src_conf = float(
                    PROFILE_CONFIDENCE_FLOOR
                    + (1.0 - float(PROFILE_CONFIDENCE_FLOOR)) * float(np.clip(batch_conf[row_pos], 0.0, 1.0))
                )
                rank_j = 0
                for item_pos in order.tolist():
                    sc = float(score[row_pos, int(item_pos)])
                    if sc < min_score:
                        continue
                    rank_j += 1
                    rows.append(
                        (
                            int(batch_uidx[row_pos]),
                            int(item_indices[int(item_pos)]),
                            int(rank_j),
                            float(sc),
                            float(src_conf),
                            "profile",
                            route_name,
                        )
                    )
                if rank_j > 0:
                    route_user_count += 1

        route_rows[route_name] = int(len(rows) - written_before)
        route_users[route_name] = int(route_user_count)
        route_status[route_name] = {
            "status": "ok",
            "rows": int(route_rows[route_name]),
            "users": int(route_user_count),
            "top_k": int(kk),
        }

    if not rows:
        return empty_pdf, {"status": "no_multivector_candidate_rows", "route_status": route_status}

    route_pdf = pd.DataFrame(
        rows,
        columns=[
            "user_idx",
            "item_idx",
            "source_rank",
            "source_score",
            "source_confidence",
            "source",
            "source_detail",
        ],
    )
    route_pdf["user_idx"] = pd.to_numeric(route_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    route_pdf["item_idx"] = pd.to_numeric(route_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    route_pdf["source_rank"] = pd.to_numeric(route_pdf["source_rank"], errors="coerce").fillna(0).astype(np.int32)
    route_pdf["source_score"] = pd.to_numeric(route_pdf["source_score"], errors="coerce").fillna(0.0).astype(np.float64)
    route_pdf["source_confidence"] = (
        pd.to_numeric(route_pdf["source_confidence"], errors="coerce").fillna(0.0).astype(np.float64)
    )
    route_pdf = route_pdf[(route_pdf["user_idx"] >= 0) & (route_pdf["item_idx"] >= 0) & (route_pdf["source_rank"] > 0)].copy()
    route_pdf = route_pdf.sort_values(
        ["source_detail", "user_idx", "source_rank", "item_idx"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    if route_pdf.empty:
        return empty_pdf, {
            "status": "multivector_candidate_rows_invalid_after_cast",
            "route_status": route_status,
        }

    meta = {
        "status": "ok",
        "rows": int(route_pdf.shape[0]),
        "route_rows": {str(k): int(v) for k, v in route_rows.items()},
        "route_users": {str(k): int(v) for k, v in route_users.items()},
        "route_status": route_status,
        "batch_users": int(batch_n),
        "top_k": int(route_top_k),
        "score_min": float(route_score_min),
    }
    return route_pdf, meta


def build_profile_candidates(
    spark: SparkSession,
    train_idx: DataFrame,
    test_users: DataFrame,
    user_map: DataFrame,
    item_map: DataFrame,
    profile_user_ids: np.ndarray,
    profile_vectors: np.ndarray,
    profile_confidence_by_user_id: dict[str, float],
    profile_top_k: int,
    profile_calibration_model: dict[str, Any] | None,
    user_tag_long_pdf: pd.DataFrame | None,
    item_tag_long_pdf: pd.DataFrame | None,
    profile_route_policy: dict[str, Any] | None = None,
    profile_threshold_policy: dict[str, Any] | None = None,
    bucket_min_train: int | None = None,
    profile_multivector_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    merchant_dense_vector_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[DataFrame, dict[str, Any]]:
    route_policy = dict(profile_route_policy or {})
    threshold_policy = dict(profile_threshold_policy or {})
    enable_profile_vector_route = _as_bool(
        route_policy.get("enable_profile_vector_route", ENABLE_PROFILE_VECTOR_ROUTE),
        bool(ENABLE_PROFILE_VECTOR_ROUTE),
    )
    enable_tag_shared_route = _as_bool(
        route_policy.get("enable_tag_shared_route", ENABLE_TAG_SHARED_ROUTE),
        bool(ENABLE_TAG_SHARED_ROUTE),
    )
    enable_bridge_user_route = _as_bool(
        route_policy.get("enable_bridge_user_route", ENABLE_BRIDGE_USER_ROUTE),
        bool(ENABLE_BRIDGE_USER_ROUTE),
    )
    enable_bridge_type_route = _as_bool(
        route_policy.get("enable_bridge_type_route", ENABLE_BRIDGE_TYPE_ROUTE),
        bool(ENABLE_BRIDGE_TYPE_ROUTE),
    )
    profile_tag_shared_top_k = int(
        threshold_policy.get("profile_tag_shared_top_k", PROFILE_TAG_SHARED_TOP_K)
    )
    profile_bridge_user_top_k = int(
        threshold_policy.get("profile_bridge_user_top_k", PROFILE_BRIDGE_USER_TOP_K)
    )
    profile_bridge_type_top_k = int(
        threshold_policy.get("profile_bridge_type_top_k", PROFILE_BRIDGE_TYPE_TOP_K)
    )
    profile_bridge_user_neighbors = int(
        threshold_policy.get("profile_bridge_user_neighbors", PROFILE_BRIDGE_USER_NEIGHBORS)
    )
    profile_bridge_user_min_sim = float(
        threshold_policy.get("profile_bridge_user_min_sim", PROFILE_BRIDGE_USER_MIN_SIM)
    )
    profile_shared_score_min = float(
        threshold_policy.get("profile_shared_score_min", PROFILE_SHARED_SCORE_MIN)
    )
    profile_bridge_score_min = float(
        threshold_policy.get("profile_bridge_score_min", PROFILE_BRIDGE_SCORE_MIN)
    )
    enable_multivector_route_integration = (
        bucket_min_train is not None
        and should_integrate_profile_multivector_routes(int(bucket_min_train))
        and bool(profile_multivector_data)
        and bool(merchant_dense_vector_data)
    )
    multivector_route_top_k = max(1, min(int(profile_top_k), int(PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_TOP_K)))
    multivector_route_score_min = float(PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_SCORE_MIN)
    multivector_route_batch_users = int(max(1, PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BATCH_USERS))
    uid_to_vec = {str(uid): int(i) for i, uid in enumerate(profile_user_ids.tolist())}

    user_scope = user_map.select("user_idx")
    if PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_USERS:
        user_scope = (
            train_idx.select("user_idx")
            .unionByName(test_users.select("user_idx"))
            .dropDuplicates(["user_idx"])
        )
    user_pdf = user_map.join(user_scope, on="user_idx", how="inner").select("user_idx", "user_id").toPandas()
    if user_pdf.empty:
        return _empty_source_df(spark, "profile"), {"status": "empty_user_map"}
    user_pdf["user_idx"] = user_pdf["user_idx"].astype(np.int32)
    user_pdf["user_id"] = user_pdf["user_id"].astype(str)
    user_pdf["profile_conf"] = (
        user_pdf["user_id"].map(profile_confidence_by_user_id).fillna(0.0).astype(np.float32)
    )
    user_pdf["vec_idx"] = user_pdf["user_id"].map(uid_to_vec)
    useridx_to_profile_conf = dict(zip(user_pdf["user_idx"].tolist(), user_pdf["profile_conf"].tolist()))
    vec_map_pdf = user_pdf.dropna(subset=["vec_idx"])[["user_idx", "vec_idx"]].copy()
    if not vec_map_pdf.empty:
        vec_map_pdf["vec_idx"] = vec_map_pdf["vec_idx"].astype(np.int32)
        useridx_to_vecidx = dict(
            zip(
                vec_map_pdf["user_idx"].astype(np.int32).tolist(),
                vec_map_pdf["vec_idx"].astype(np.int32).tolist(),
            )
        )
    else:
        useridx_to_vecidx = {}

    test_pdf = test_users.select("user_idx").toPandas()
    if test_pdf.empty:
        return _empty_source_df(spark, "profile"), {"status": "empty_test_users"}
    test_pdf["user_idx"] = test_pdf["user_idx"].astype(np.int32)
    test_pdf = test_pdf.merge(user_pdf[["user_idx", "user_id"]], on="user_idx", how="left")
    test_pdf["user_id"] = test_pdf["user_id"].fillna("").astype(str)
    test_pdf["profile_conf"] = (
        test_pdf["user_idx"].map(useridx_to_profile_conf).fillna(0.0).astype(np.float32)
    )
    test_users_with_vector = int(test_pdf["user_idx"].map(useridx_to_vecidx).notna().sum())
    test_pdf_conf = test_pdf[test_pdf["profile_conf"] >= float(PROFILE_CONFIDENCE_MIN)].copy()
    if test_pdf_conf.empty:
        return _empty_source_df(spark, "profile"), {"status": "no_test_with_confident_profile"}
    test_users_with_confident_profile = int(test_pdf_conf.shape[0])
    test_conf_by_uidx = dict(
        zip(
            test_pdf_conf["user_idx"].astype(np.int32).tolist(),
            test_pdf_conf["profile_conf"].astype(np.float32).tolist(),
        )
    )

    train_pdf_all = train_idx.select("user_idx", "item_idx", "rating").toPandas()
    if train_pdf_all.empty:
        return _empty_source_df(spark, "profile"), {"status": "empty_train_idx"}
    train_pdf_all["user_idx"] = train_pdf_all["user_idx"].astype(np.int32)
    train_pdf_all["item_idx"] = train_pdf_all["item_idx"].astype(np.int32)
    train_pdf_all["rating"] = pd.to_numeric(train_pdf_all["rating"], errors="coerce").fillna(3.0).astype(np.float32)
    train_pdf_all["rating_norm"] = np.clip((train_pdf_all["rating"] - 1.0) / 4.0, 0.0, 1.0)

    rows: list[tuple[int, int, int, float, float, str]] = []
    profile_parts: list[pd.DataFrame] = []
    route_rows: dict[str, int] = {"vector": 0, "shared": 0, "bridge_user": 0, "bridge_type": 0}
    multivector_route_rows: dict[str, int] = {}
    calibration_applied_rows = 0
    enabled_routes: list[str] = []
    item_ids_final = np.array([], dtype=np.int32)
    multivector_meta: dict[str, Any] = {"status": "disabled"}

    # Route 0: original profile-vector route (kept for compatibility / ablation).
    if bool(enable_profile_vector_route) and profile_vectors.shape[0] > 0:
        enabled_routes.append("vector")
        train_pdf = train_pdf_all.copy()
        train_pdf["vec_idx"] = train_pdf["user_idx"].map(useridx_to_vecidx)
        train_pdf["profile_conf"] = train_pdf["user_idx"].map(useridx_to_profile_conf)
        train_pdf = train_pdf.dropna(subset=["vec_idx"]).copy()
        if not train_pdf.empty:
            train_pdf["vec_idx"] = train_pdf["vec_idx"].astype(np.int32)
            train_pdf["profile_conf"] = train_pdf["profile_conf"].fillna(0.0).astype(np.float32)
            user_activity = train_pdf.groupby("user_idx").size().rename("user_activity")
            train_pdf = train_pdf.merge(user_activity, on="user_idx", how="left")
            train_pdf["user_activity"] = train_pdf["user_activity"].fillna(1).astype(np.float32)

            item_ids = train_pdf["item_idx"].to_numpy(np.int32)
            vec_indices = train_pdf["vec_idx"].to_numpy(np.int32)
            rating_norm = np.clip((train_pdf["rating"].to_numpy(np.float32) - 1.0) / 4.0, 0.0, 1.0)
            conf_gate = (
                float(PROFILE_CONFIDENCE_FLOOR)
                + (1.0 - float(PROFILE_CONFIDENCE_FLOOR)) * np.clip(train_pdf["profile_conf"].to_numpy(np.float32), 0.0, 1.0)
            )
            rating_gate = 0.5 + 0.5 * rating_norm
            activity_gate = 1.0 / np.sqrt(np.maximum(train_pdf["user_activity"].to_numpy(np.float32), 1.0))
            sample_weight = (
                np.power(conf_gate, float(PROFILE_ITEM_WEIGHT_CONF_POWER))
                * np.power(rating_gate, float(PROFILE_ITEM_WEIGHT_RATING_POWER))
                * np.power(activity_gate, float(PROFILE_ITEM_WEIGHT_ACTIVITY_POWER))
            ).astype(np.float32)
            sample_weight = np.maximum(sample_weight, float(PROFILE_ITEM_WEIGHT_MIN)).astype(np.float32)

            unique_item_ids, inv = np.unique(item_ids, return_inverse=True)
            dim = int(profile_vectors.shape[1])
            item_sum = np.zeros((unique_item_ids.shape[0], dim), dtype=np.float32)
            item_weight_sum = np.zeros((unique_item_ids.shape[0],), dtype=np.float32)
            # Memory-safe aggregation: chunk rows to avoid allocating a full
            # (n_train_rows x dim) temporary matrix.
            chunk_rows = max(1024, int(PROFILE_ITEM_AGG_CHUNK_ROWS))
            n_rows = int(inv.shape[0])
            for s in range(0, n_rows, chunk_rows):
                e = min(n_rows, s + chunk_rows)
                inv_chunk = inv[s:e]
                vec_chunk = profile_vectors[vec_indices[s:e]]
                w_chunk = sample_weight[s:e]
                np.add.at(item_sum, inv_chunk, vec_chunk * w_chunk[:, None])
                np.add.at(item_weight_sum, inv_chunk, w_chunk)
            valid_items = item_weight_sum > 0.0
            item_ids_final = unique_item_ids[valid_items]
            item_mat = item_sum[valid_items] / np.maximum(item_weight_sum[valid_items][:, None], 1e-6)
            item_norm = np.linalg.norm(item_mat, axis=1, keepdims=True)
            item_norm[item_norm == 0.0] = 1.0
            item_mat = (item_mat / item_norm).astype(np.float32)

            test_vec_pdf = test_pdf_conf.copy()
            test_vec_pdf["vec_idx"] = test_vec_pdf["user_idx"].map(useridx_to_vecidx)
            test_vec_pdf = test_vec_pdf.dropna(subset=["vec_idx"]).copy()
            if not test_vec_pdf.empty and item_mat.shape[0] > 0:
                test_vec_pdf["vec_idx"] = test_vec_pdf["vec_idx"].astype(np.int32)
                top_k = min(int(profile_top_k), int(item_mat.shape[0]))
                uids = test_vec_pdf["user_idx"].to_numpy(np.int32)
                uvec = test_vec_pdf["vec_idx"].to_numpy(np.int32)
                uconf = test_vec_pdf["profile_conf"].to_numpy(np.float32)
                rank_vec = np.arange(1, top_k + 1, dtype=np.int32)
                for s in range(0, uids.shape[0], int(PROFILE_RECALL_BATCH_USERS)):
                    e = min(uids.shape[0], s + int(PROFILE_RECALL_BATCH_USERS))
                    batch_uids = uids[s:e]
                    batch_vecs = profile_vectors[uvec[s:e]]
                    batch_conf = uconf[s:e]
                    score = np.matmul(batch_vecs, item_mat.T)
                    if score.size == 0:
                        continue
                    top_idx = np.argpartition(-score, kth=top_k - 1, axis=1)[:, :top_k]
                    top_score = np.take_along_axis(score, top_idx, axis=1)
                    order = np.argsort(-top_score, axis=1)
                    top_idx = np.take_along_axis(top_idx, order, axis=1)
                    top_score = np.take_along_axis(top_score, order, axis=1)
                    source_conf_batch = (
                        float(PROFILE_CONFIDENCE_FLOOR)
                        + (1.0 - float(PROFILE_CONFIDENCE_FLOOR)) * np.clip(batch_conf.astype(np.float32), 0.0, 1.0)
                    ).astype(np.float32)
                    if profile_calibration_model:
                        for i, uid in enumerate(batch_uids.tolist()):
                            source_conf_user = float(source_conf_batch[i])
                            ranks = rank_vec.astype(np.float32)
                            conf_vec = np.full((top_k,), source_conf_user, dtype=np.float32)
                            cal_scores = calibrate_profile_scores(
                                raw_scores=top_score[i].astype(np.float32),
                                ranks=ranks,
                                source_conf=conf_vec,
                                model_cfg=profile_calibration_model,
                            )
                            cal_order = np.argsort(-cal_scores)
                            top_idx[i] = top_idx[i, cal_order]
                            top_score[i] = cal_scores[cal_order]
                            calibration_applied_rows += int(top_k)
                        uid_col = np.repeat(batch_uids.astype(np.int32), int(top_k))
                        item_col = item_ids_final[top_idx.reshape(-1)].astype(np.int32)
                        rank_col = np.tile(rank_vec, int(batch_uids.shape[0])).astype(np.int32)
                        score_col = top_score.reshape(-1).astype(np.float64)
                        conf_col = np.repeat(source_conf_batch.astype(np.float64), int(top_k))
                        profile_parts.append(
                            pd.DataFrame(
                                {
                                    "user_idx": uid_col,
                                    "item_idx": item_col,
                                    "source_rank": rank_col,
                                    "source_score": score_col,
                                    "source_confidence": conf_col,
                                    "source": "profile",
                                    "source_detail": "profile_vector",
                                }
                            )
                        )
                        route_rows["vector"] += int(uid_col.shape[0])
                    else:
                        uid_col = np.repeat(batch_uids.astype(np.int32), int(top_k))
                        item_col = item_ids_final[top_idx.reshape(-1)].astype(np.int32)
                        rank_col = np.tile(rank_vec, int(batch_uids.shape[0])).astype(np.int32)
                        score_col = top_score.reshape(-1).astype(np.float64)
                        conf_col = np.repeat(source_conf_batch.astype(np.float64), int(top_k))
                        profile_parts.append(
                            pd.DataFrame(
                                {
                                    "user_idx": uid_col,
                                    "item_idx": item_col,
                                    "source_rank": rank_col,
                                    "source_score": score_col,
                                    "source_confidence": conf_col,
                                    "source": "profile",
                                    "source_detail": "profile_vector",
                                }
                            )
                        )
                        route_rows["vector"] += int(uid_col.shape[0])

    if bool(enable_multivector_route_integration):
        multivector_pdf, multivector_meta = build_profile_multivector_candidate_pdf(
            test_pdf_conf=test_pdf_conf,
            item_map=item_map,
            user_vectors_by_scope=profile_multivector_data,
            merchant_vectors_by_scope=merchant_dense_vector_data,
            route_top_k=multivector_route_top_k,
            route_score_min=multivector_route_score_min,
            batch_users=multivector_route_batch_users,
        )
        if not multivector_pdf.empty:
            profile_parts.append(multivector_pdf)
            for route_name, row_count in multivector_meta.get("route_rows", {}).items():
                multivector_route_rows[str(route_name)] = int(row_count)
            enabled_routes.extend(
                [
                    str(route_name)
                    for route_name, row_count in multivector_meta.get("route_rows", {}).items()
                    if int(row_count) > 0
                ]
            )

    # Prepare typed-tag tables for shared/bridge routes.
    user_map_pdf = user_pdf[["user_idx", "user_id"]].copy()
    item_map_pdf = pd.DataFrame(columns=["item_idx", "business_id"])
    if bool(enable_tag_shared_route) or bool(enable_bridge_type_route):
        item_scope = item_map.select("item_idx")
        if PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_ITEMS:
            item_scope = train_idx.select("item_idx").dropDuplicates(["item_idx"])
        item_map_pdf = item_map.join(item_scope, on="item_idx", how="inner").select("item_idx", "business_id").toPandas()
        item_map_pdf["item_idx"] = item_map_pdf["item_idx"].astype(np.int32)
        item_map_pdf["business_id"] = item_map_pdf["business_id"].astype(str)

    user_tags = pd.DataFrame(columns=["user_idx", "tag", "tag_type", "user_tag_weight"])
    item_tags = pd.DataFrame(columns=["item_idx", "tag", "tag_type", "item_tag_weight"])
    if user_tag_long_pdf is not None and item_tag_long_pdf is not None:
        if (not user_tag_long_pdf.empty) and (not item_tag_long_pdf.empty):
            user_tags = user_tag_long_pdf.merge(user_map_pdf, on="user_id", how="inner")
            item_tags = item_tag_long_pdf.merge(item_map_pdf, on="business_id", how="inner")
            if not user_tags.empty:
                user_tags["profile_conf"] = user_tags["user_idx"].map(useridx_to_profile_conf).fillna(0.0)
                conf_gate = float(PROFILE_CONFIDENCE_FLOOR) + (
                    1.0 - float(PROFILE_CONFIDENCE_FLOOR)
                ) * np.clip(user_tags["profile_conf"].to_numpy(np.float32), 0.0, 1.0)
                user_tags["user_tag_weight"] = (
                    user_tags["net_w"].to_numpy(np.float64)
                    * user_tags["tag_confidence"].to_numpy(np.float64)
                    * np.log1p(np.maximum(user_tags["support"].to_numpy(np.float64), 0.0))
                    * conf_gate.astype(np.float64)
                )
                user_tags = user_tags[
                    (user_tags["tag"] != "")
                    & (user_tags["tag_type"] != "")
                    & np.isfinite(user_tags["user_tag_weight"])
                ][["user_idx", "tag", "tag_type", "user_tag_weight"]]
            if not item_tags.empty:
                item_tags["item_tag_weight"] = (
                    item_tags["net_weight_sum"].to_numpy(np.float64)
                    * item_tags["tag_confidence"].to_numpy(np.float64)
                    * np.log1p(np.maximum(item_tags["support_count"].to_numpy(np.float64), 0.0))
                )
                item_tags = item_tags[
                    (item_tags["tag"] != "")
                    & (item_tags["tag_type"] != "")
                    & np.isfinite(item_tags["item_tag_weight"])
                ][["item_idx", "tag", "tag_type", "item_tag_weight"]]

    # Shared route: exact typed tag overlap user<->item.
    if bool(enable_tag_shared_route) and (not user_tags.empty) and (not item_tags.empty):
        enabled_routes.append("shared")
        shared_top_k = max(1, min(int(profile_top_k), int(profile_tag_shared_top_k)))
        test_uids = test_pdf_conf["user_idx"].tolist()
        user_tags_test = user_tags[user_tags["user_idx"].isin(test_uids)].copy()
        if not user_tags_test.empty:
            key_to_items: dict[tuple[str, str], list[tuple[int, float]]] = {}
            for r in item_tags.itertuples(index=False):
                key = (str(r.tag_type), str(r.tag))
                key_to_items.setdefault(key, []).append((int(r.item_idx), float(r.item_tag_weight)))
            user_tag_map: dict[int, list[tuple[str, str, float]]] = {}
            for r in user_tags_test.itertuples(index=False):
                user_tag_map.setdefault(int(r.user_idx), []).append((str(r.tag_type), str(r.tag), float(r.user_tag_weight)))

            for uid in test_uids:
                tags_u = user_tag_map.get(int(uid), [])
                if not tags_u:
                    continue
                accum: dict[int, float] = {}
                user_norm = 0.0
                for t_type, tag, u_w in tags_u:
                    user_norm += abs(float(u_w))
                    for item_idx, i_w in key_to_items.get((t_type, tag), []):
                        accum[item_idx] = accum.get(item_idx, 0.0) + float(u_w) * float(i_w)
                if not accum:
                    continue
                denom = max(user_norm, 1e-6)
                scored = [
                    (iid, float(s / denom))
                    for iid, s in accum.items()
                    if float(s / denom) > float(profile_shared_score_min)
                ]
                if not scored:
                    continue
                top = heapq.nlargest(shared_top_k, scored, key=lambda z: z[1])
                u_conf = float(test_conf_by_uidx.get(int(uid), 0.0))
                src_conf = float(
                    PROFILE_CONFIDENCE_FLOOR + (1.0 - PROFILE_CONFIDENCE_FLOOR) * np.clip(u_conf, 0.0, 1.0)
                )
                for rnk, (iid, sc) in enumerate(top, start=1):
                    rows.append((int(uid), int(iid), int(rnk), float(sc), float(src_conf), "profile", "profile_shared"))
                    route_rows["shared"] += 1

    # Bridge A: similar users by typed tags -> transfer neighbor items.
    if bool(enable_bridge_user_route) and (not user_tags.empty):
        enabled_routes.append("bridge_user")
        bridge_top_k = max(1, min(int(profile_top_k), int(profile_bridge_user_top_k)))
        neigh_k = max(1, int(profile_bridge_user_neighbors))
        min_sim = float(profile_bridge_user_min_sim)
        test_uids = test_pdf_conf["user_idx"].tolist()
        train_uids = set(train_pdf_all["user_idx"].astype(int).tolist())
        user_tags_test = user_tags[user_tags["user_idx"].isin(test_uids)].copy()
        user_tags_train = user_tags[user_tags["user_idx"].isin(train_uids)].copy()
        if (not user_tags_test.empty) and (not user_tags_train.empty):
            key_to_train_users: dict[tuple[str, str], list[tuple[int, float]]] = {}
            for r in user_tags_train.itertuples(index=False):
                key = (str(r.tag_type), str(r.tag))
                key_to_train_users.setdefault(key, []).append((int(r.user_idx), float(r.user_tag_weight)))
            # limit very broad keys to keep compute stable
            for key, vals in list(key_to_train_users.items()):
                if len(vals) > 3000:
                    key_to_train_users[key] = heapq.nlargest(3000, vals, key=lambda z: abs(float(z[1])))
            user_tag_map_test: dict[int, list[tuple[str, str, float]]] = {}
            for r in user_tags_test.itertuples(index=False):
                user_tag_map_test.setdefault(int(r.user_idx), []).append((str(r.tag_type), str(r.tag), float(r.user_tag_weight)))
            train_user_items: dict[int, list[tuple[int, float]]] = {}
            for r in train_pdf_all[["user_idx", "item_idx", "rating_norm"]].itertuples(index=False):
                train_user_items.setdefault(int(r.user_idx), []).append((int(r.item_idx), float(0.5 + 0.5 * float(r.rating_norm))))

            for uid in test_uids:
                tags_u = user_tag_map_test.get(int(uid), [])
                if not tags_u:
                    continue
                neigh_score: dict[int, float] = {}
                for t_type, tag, u_w in tags_u:
                    for nb_uid, nb_w in key_to_train_users.get((t_type, tag), []):
                        if int(nb_uid) == int(uid):
                            continue
                        neigh_score[int(nb_uid)] = neigh_score.get(int(nb_uid), 0.0) + float(u_w) * float(nb_w)
                cand_neigh = [(nid, s) for nid, s in neigh_score.items() if float(s) > min_sim]
                if not cand_neigh:
                    continue
                neigh_top = heapq.nlargest(neigh_k, cand_neigh, key=lambda z: z[1])
                sim_sum = sum(float(s) for _, s in neigh_top)
                if sim_sum <= 0:
                    continue
                item_acc: dict[int, float] = {}
                for nid, sim in neigh_top:
                    for iid, iw in train_user_items.get(int(nid), []):
                        item_acc[int(iid)] = item_acc.get(int(iid), 0.0) + float(sim) * float(iw)
                scored = [
                    (iid, float(s / sim_sum))
                    for iid, s in item_acc.items()
                    if float(s / sim_sum) > float(profile_bridge_score_min)
                ]
                if not scored:
                    continue
                top = heapq.nlargest(bridge_top_k, scored, key=lambda z: z[1])
                u_conf = float(test_conf_by_uidx.get(int(uid), 0.0))
                neigh_factor = min(1.0, float(len(neigh_top)) / float(neigh_k))
                src_conf = float(
                    (PROFILE_CONFIDENCE_FLOOR + (1.0 - PROFILE_CONFIDENCE_FLOOR) * np.clip(u_conf, 0.0, 1.0))
                    * (0.8 + 0.2 * neigh_factor)
                )
                for rnk, (iid, sc) in enumerate(top, start=1):
                    rows.append(
                        (int(uid), int(iid), int(rnk), float(sc), float(src_conf), "profile", "profile_bridge_user")
                    )
                    route_rows["bridge_user"] += 1

    # Bridge B: user type preference -> item type profile.
    if bool(enable_bridge_type_route) and (not user_tags.empty) and (not item_tags.empty):
        enabled_routes.append("bridge_type")
        bridge_type_top_k = max(1, min(int(profile_top_k), int(profile_bridge_type_top_k)))
        test_uids = test_pdf_conf["user_idx"].tolist()
        user_type_df = (
            user_tags[user_tags["user_idx"].isin(test_uids)]
            .groupby(["user_idx", "tag_type"], as_index=False)["user_tag_weight"]
            .sum()
        )
        item_type_df = item_tags.groupby(["item_idx", "tag_type"], as_index=False)["item_tag_weight"].sum()
        if (not user_type_df.empty) and (not item_type_df.empty):
            type_list = sorted(set(user_type_df["tag_type"].tolist()) | set(item_type_df["tag_type"].tolist()))
            type_to_pos = {t: i for i, t in enumerate(type_list)}
            item_pivot = item_type_df.pivot_table(
                index="item_idx", columns="tag_type", values="item_tag_weight", aggfunc="sum", fill_value=0.0
            )
            item_ids = item_pivot.index.to_numpy(np.int32)
            item_mat = item_pivot.reindex(columns=type_list, fill_value=0.0).to_numpy(np.float32)
            user_type_map: dict[int, np.ndarray] = {}
            for r in user_type_df.itertuples(index=False):
                uid = int(r.user_idx)
                pos = type_to_pos.get(str(r.tag_type))
                if pos is None:
                    continue
                if uid not in user_type_map:
                    user_type_map[uid] = np.zeros((len(type_list),), dtype=np.float32)
                user_type_map[uid][pos] += float(r.user_tag_weight)

            for uid in test_uids:
                uvec = user_type_map.get(int(uid))
                if uvec is None:
                    continue
                u_denom = float(np.sum(np.abs(uvec)))
                if u_denom <= 1e-6:
                    continue
                score = (item_mat @ uvec) / u_denom
                if score.size == 0:
                    continue
                k = min(int(bridge_type_top_k), int(score.shape[0]))
                if k <= 0:
                    continue
                top_idx = np.argpartition(-score, kth=k - 1)[:k]
                top_idx = top_idx[np.argsort(-score[top_idx])]
                u_conf = float(test_conf_by_uidx.get(int(uid), 0.0))
                src_conf = float(
                    PROFILE_CONFIDENCE_FLOOR + (1.0 - PROFILE_CONFIDENCE_FLOOR) * np.clip(u_conf, 0.0, 1.0)
                )
                rank_j = 0
                for pos in top_idx.tolist():
                    sc = float(score[int(pos)])
                    if sc <= float(profile_bridge_score_min):
                        continue
                    rank_j += 1
                    rows.append(
                        (int(uid), int(item_ids[int(pos)]), int(rank_j), float(sc), float(src_conf), "profile", "profile_bridge_type")
                    )
                    route_rows["bridge_type"] += 1

    if rows:
        profile_parts.append(
            pd.DataFrame(
                rows,
                columns=[
                    "user_idx",
                    "item_idx",
                    "source_rank",
                    "source_score",
                    "source_confidence",
                    "source",
                    "source_detail",
                ],
            )
        )
    if not profile_parts:
        return _empty_source_df(spark, "profile"), {"status": "no_profile_rows"}
    rows_before_dedup = int(sum(int(p.shape[0]) for p in profile_parts))
    profile_pdf = pd.concat(profile_parts, ignore_index=True)
    # Collapse duplicated user-item rows from multiple profile sub-routes before
    # moving data back to Spark. Keep strongest row, then rebuild profile rank.
    profile_pdf["user_idx"] = pd.to_numeric(profile_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    profile_pdf["item_idx"] = pd.to_numeric(profile_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    profile_pdf["source_score"] = pd.to_numeric(profile_pdf["source_score"], errors="coerce").fillna(0.0).astype(np.float64)
    profile_pdf["source_confidence"] = (
        pd.to_numeric(profile_pdf["source_confidence"], errors="coerce").fillna(0.0).astype(np.float64)
    )
    profile_pdf = profile_pdf[(profile_pdf["user_idx"] >= 0) & (profile_pdf["item_idx"] >= 0)].copy()
    if profile_pdf.empty:
        return _empty_source_df(spark, "profile"), {"status": "profile_rows_invalid_after_cast"}
    if bool(enable_multivector_route_integration):
        profile_pdf = profile_pdf.sort_values(
            ["user_idx", "item_idx", "source_detail", "source_score", "source_confidence", "source_rank"],
            ascending=[True, True, True, False, False, True],
            kind="mergesort",
        ).drop_duplicates(subset=["user_idx", "item_idx", "source_detail"], keep="first")
        profile_pdf["is_profile_mv_route"] = profile_pdf["source_detail"].astype(str).str.startswith("profile_mv_").astype(np.int8)
        profile_pdf["profile_mv_short_rank_src"] = np.where(
            profile_pdf["source_detail"] == "profile_mv_short_semantic",
            profile_pdf["source_rank"].astype(np.float64),
            np.nan,
        )
        profile_pdf["profile_mv_long_rank_src"] = np.where(
            profile_pdf["source_detail"] == "profile_mv_long_core",
            profile_pdf["source_rank"].astype(np.float64),
            np.nan,
        )
        profile_pdf["profile_mv_pos_rank_src"] = np.where(
            profile_pdf["source_detail"] == "profile_mv_pos_pos",
            profile_pdf["source_rank"].astype(np.float64),
            np.nan,
        )
        profile_detail_pdf = (
            profile_pdf.groupby(["user_idx", "item_idx"], sort=False)
            .agg(
                profile_detail_count=("source_detail", "size"),
                profile_mv_route_count=("is_profile_mv_route", "sum"),
                profile_mv_short_rank=("profile_mv_short_rank_src", "min"),
                profile_mv_long_rank=("profile_mv_long_rank_src", "min"),
                profile_mv_pos_rank=("profile_mv_pos_rank_src", "min"),
            )
            .reset_index()
        )
        profile_best_pdf = profile_pdf.sort_values(
            ["user_idx", "item_idx", "source_score", "source_confidence", "source_rank"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        ).drop_duplicates(subset=["user_idx", "item_idx"], keep="first")
        profile_pdf = profile_best_pdf.merge(
            profile_detail_pdf,
            on=["user_idx", "item_idx"],
            how="left",
        )
    else:
        profile_pdf = profile_pdf.sort_values(
            ["user_idx", "item_idx", "source_score", "source_confidence", "source_rank"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        ).drop_duplicates(subset=["user_idx", "item_idx"], keep="first")
    for col_name, default_value in [
        ("profile_detail_count", 1.0),
        ("profile_mv_route_count", 0.0),
        ("profile_mv_short_rank", np.nan),
        ("profile_mv_long_rank", np.nan),
        ("profile_mv_pos_rank", np.nan),
    ]:
        if col_name not in profile_pdf.columns:
            profile_pdf[col_name] = default_value
    profile_pdf["profile_detail_count"] = pd.to_numeric(
        profile_pdf["profile_detail_count"], errors="coerce"
    ).fillna(1.0).astype(np.float64)
    profile_pdf["profile_mv_route_count"] = pd.to_numeric(
        profile_pdf["profile_mv_route_count"], errors="coerce"
    ).fillna(0.0).astype(np.float64)
    profile_pdf["profile_mv_short_rank"] = pd.to_numeric(
        profile_pdf["profile_mv_short_rank"], errors="coerce"
    ).astype(np.float64)
    profile_pdf["profile_mv_long_rank"] = pd.to_numeric(
        profile_pdf["profile_mv_long_rank"], errors="coerce"
    ).astype(np.float64)
    profile_pdf["profile_mv_pos_rank"] = pd.to_numeric(
        profile_pdf["profile_mv_pos_rank"], errors="coerce"
    ).astype(np.float64)
    profile_pdf = profile_pdf.sort_values(
        ["user_idx", "source_score", "source_confidence", "profile_mv_route_count", "item_idx"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    profile_pdf["source_rank"] = profile_pdf.groupby("user_idx", sort=False).cumcount() + 1
    profile_pdf = profile_pdf[profile_pdf["source_rank"] <= int(profile_top_k)].copy()
    if profile_pdf.empty:
        return _empty_source_df(spark, "profile"), {"status": "profile_rows_empty_after_dedup"}
    profile_pdf["source"] = "profile"
    profile_df = None
    tmp_csv: Path | None = None
    output_mode_effective = str(PROFILE_OUTPUT_MODE)
    if output_mode_effective == "auto":
        output_mode_effective = (
            "spark_df" if int(profile_pdf.shape[0]) <= int(PROFILE_OUTPUT_SPARK_DF_MAX_ROWS) else "csv"
        )
    if output_mode_effective == "csv" and (not ALLOW_PROFILE_CSV_FALLBACK):
        output_mode_effective = "spark_df"
    if output_mode_effective != "csv":
        try:
            profile_df = _create_spark_df_from_pandas_safe(
                spark=spark,
                pdf=profile_pdf,
            ).select(
                F.col("user_idx").cast("int"),
                F.col("item_idx").cast("int"),
                F.col("source_rank").cast("int"),
                F.col("source_score").cast("double"),
                F.col("source_confidence").cast("double"),
                F.col("source").cast("string"),
                F.col("source_detail").cast("string"),
                F.col("profile_detail_count").cast("double"),
                F.col("profile_mv_route_count").cast("double"),
                F.col("profile_mv_short_rank").cast("double"),
                F.col("profile_mv_long_rank").cast("double"),
                F.col("profile_mv_pos_rank").cast("double"),
            )
        except Exception:
            profile_df = None
    if profile_df is None and ALLOW_PROFILE_CSV_FALLBACK:
        tmp_csv = _alloc_stage_tmp_csv("profile_recall", "profile_cand")
        profile_pdf.to_csv(tmp_csv.as_posix(), index=False, encoding="utf-8")
        profile_df = spark.read.csv(tmp_csv.as_posix(), header=True, inferSchema=True).select(
            F.col("user_idx").cast("int"),
            F.col("item_idx").cast("int"),
            F.col("source_rank").cast("int"),
            F.col("source_score").cast("double"),
            F.col("source_confidence").cast("double"),
            F.col("source").cast("string"),
            F.col("source_detail").cast("string"),
            F.col("profile_detail_count").cast("double"),
            F.col("profile_mv_route_count").cast("double"),
            F.col("profile_mv_short_rank").cast("double"),
            F.col("profile_mv_long_rank").cast("double"),
            F.col("profile_mv_pos_rank").cast("double"),
        )
    if profile_df is None:
        raise RuntimeError("profile dataframe conversion failed; enable ALLOW_PROFILE_CSV_FALLBACK=true for debug fallback")
    meta = {
        "status": "ok",
        "users_with_vector": int(len(useridx_to_vecidx)),
        "test_users_with_vector": int(test_users_with_vector),
        "test_users_with_confident_profile": int(test_users_with_confident_profile),
        "items_with_profile_vector": int(item_ids_final.shape[0]),
        "train_pairs_with_profile_vector": int(train_pdf_all.shape[0]),
        "rows_before_dedup": int(rows_before_dedup),
        "rows": int(profile_pdf.shape[0]),
        "top_k": int(profile_top_k),
        "profile_calibration_applied": bool(profile_calibration_model is not None),
        "profile_calibration_rows": int(calibration_applied_rows),
        "enable_profile_vector_route": bool(enable_profile_vector_route),
        "enable_tag_shared_route": bool(enable_tag_shared_route),
        "enable_bridge_user_route": bool(enable_bridge_user_route),
        "enable_bridge_type_route": bool(enable_bridge_type_route),
        "profile_multivector_integration_enabled": bool(enable_multivector_route_integration),
        "profile_multivector_integration_bucket": int(bucket_min_train) if bucket_min_train is not None else None,
        "profile_multivector_integration_status": str(multivector_meta.get("status", "disabled")),
        "profile_multivector_integration_top_k": int(multivector_route_top_k),
        "profile_multivector_integration_score_min": float(multivector_route_score_min),
        "profile_multivector_integration_batch_users": int(multivector_route_batch_users),
        "profile_multivector_integration_route_rows": {
            str(k): int(v) for k, v in multivector_meta.get("route_rows", {}).items()
        },
        "profile_multivector_integration_route_users": {
            str(k): int(v) for k, v in multivector_meta.get("route_users", {}).items()
        },
        "profile_tag_shared_top_k": int(profile_tag_shared_top_k),
        "profile_bridge_user_top_k": int(profile_bridge_user_top_k),
        "profile_bridge_type_top_k": int(profile_bridge_type_top_k),
        "profile_bridge_user_neighbors": int(profile_bridge_user_neighbors),
        "profile_bridge_user_min_sim": float(profile_bridge_user_min_sim),
        "profile_shared_score_min": float(profile_shared_score_min),
        "profile_bridge_score_min": float(profile_bridge_score_min),
        "profile_item_agg_chunk_rows": int(PROFILE_ITEM_AGG_CHUNK_ROWS),
        "enabled_routes": enabled_routes,
        "rows_vector": int(route_rows["vector"]),
        "rows_shared": int(route_rows["shared"]),
        "rows_bridge_user": int(route_rows["bridge_user"]),
        "rows_bridge_type": int(route_rows["bridge_type"]),
        "rows_multivector": int(sum(int(v) for v in multivector_route_rows.values())),
        "profile_scope_users_only_relevant": bool(PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_USERS),
        "profile_scope_items_only_relevant": bool(PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_ITEMS),
        "user_map_rows_scoped": int(user_pdf.shape[0]),
        "tmp_csv": str(tmp_csv) if tmp_csv is not None else "",
        "profile_output_mode": str(PROFILE_OUTPUT_MODE),
        "profile_output_mode_effective": str(output_mode_effective),
    }
    return profile_df, meta


def build_context_candidates(
    spark: SparkSession,
    train_idx: DataFrame,
    test_users: DataFrame,
    item_map: DataFrame,
    business_context_df: DataFrame | None,
    review_evidence_df: DataFrame | None,
    tip_signal_df: DataFrame | None,
    checkin_context_df: DataFrame | None,
    context_top_k: int,
    context_score_min: float,
    batch_users: int,
) -> tuple[DataFrame, dict[str, Any]]:
    if business_context_df is None:
        return _empty_source_df(spark, "context"), {"status": "missing_business_context", "route_rows": {}, "route_users": {}}
    if review_evidence_df is None:
        review_evidence_df = spark.range(0).select(F.lit(None).cast("string").alias("business_id")).limit(0)
    if tip_signal_df is None:
        tip_signal_df = spark.range(0).select(F.lit(None).cast("string").alias("business_id")).limit(0)
    if checkin_context_df is None:
        checkin_context_df = spark.range(0).select(F.lit(None).cast("string").alias("business_id")).limit(0)
    route_details = [r for r in CONTEXT_ROUTE_DETAILS if r in {"attr", "time", "geo"}]
    if not route_details:
        return _empty_source_df(spark, "context"), {"status": "no_context_route_details", "route_rows": {}, "route_users": {}}

    item_ctx = (
        item_map.join(business_context_df, on="business_id", how="left")
        .join(review_evidence_df, on="business_id", how="left")
        .join(tip_signal_df, on="business_id", how="left")
        .join(checkin_context_df, on="business_id", how="left")
        .withColumn("ctx_has_hours", F.coalesce(F.col("ctx_has_hours"), F.lit(0.0)))
        .withColumn("ctx_open_weekend", F.coalesce(F.col("ctx_open_weekend"), F.lit(0.0)))
        .withColumn("ctx_open_late_any", F.coalesce(F.col("ctx_open_late_any"), F.lit(0.0)))
        .withColumn("ctx_meal_any", F.coalesce(F.col("ctx_meal_any"), F.lit(0.0)))
        .withColumn("ctx_meal_breakfast", F.coalesce(F.col("ctx_meal_breakfast"), F.lit(0.0)))
        .withColumn("ctx_meal_brunch", F.coalesce(F.col("ctx_meal_brunch"), F.lit(0.0)))
        .withColumn("ctx_meal_lunch", F.coalesce(F.col("ctx_meal_lunch"), F.lit(0.0)))
        .withColumn("ctx_meal_dinner", F.coalesce(F.col("ctx_meal_dinner"), F.lit(0.0)))
        .withColumn("ctx_meal_latenight", F.coalesce(F.col("ctx_meal_latenight"), F.lit(0.0)))
        .withColumn("ctx_meal_dessert", F.coalesce(F.col("ctx_meal_dessert"), F.lit(0.0)))
        .withColumn("ctx_attr_delivery", F.coalesce(F.col("ctx_attr_delivery"), F.lit(0.0)))
        .withColumn("ctx_attr_takeout", F.coalesce(F.col("ctx_attr_takeout"), F.lit(0.0)))
        .withColumn("ctx_attr_reservations", F.coalesce(F.col("ctx_attr_reservations"), F.lit(0.0)))
        .withColumn("ctx_attr_groups", F.coalesce(F.col("ctx_attr_groups"), F.lit(0.0)))
        .withColumn("ctx_attr_good_for_kids", F.coalesce(F.col("ctx_attr_good_for_kids"), F.lit(0.0)))
        .withColumn("ctx_attr_outdoor", F.coalesce(F.col("ctx_attr_outdoor"), F.lit(0.0)))
        .withColumn("ctx_attr_table_service", F.coalesce(F.col("ctx_attr_table_service"), F.lit(0.0)))
        .withColumn("ctx_attr_alcohol_full_bar", F.coalesce(F.col("ctx_attr_alcohol_full_bar"), F.lit(0.0)))
        .withColumn("ctx_attr_alcohol_beer_wine", F.coalesce(F.col("ctx_attr_alcohol_beer_wine"), F.lit(0.0)))
        .withColumn("ctx_attr_price_range_norm", F.coalesce(F.col("ctx_attr_price_range_norm"), F.lit(0.0)))
        .withColumn("ctx_review_rows", F.coalesce(F.col("ctx_review_rows"), F.lit(0.0)))
        .withColumn("ctx_review_avg_evidence", F.coalesce(F.col("ctx_review_avg_evidence"), F.lit(0.0)))
        .withColumn("ctx_review_high_vote_share", F.coalesce(F.col("ctx_review_high_vote_share"), F.lit(0.0)))
        .withColumn("ctx_review_recent_share", F.coalesce(F.col("ctx_review_recent_share"), F.lit(0.0)))
        .withColumn("ctx_tip_rows", F.coalesce(F.col("ctx_tip_rows"), F.lit(0.0)))
        .withColumn("ctx_tip_avg_weight", F.coalesce(F.col("ctx_tip_avg_weight"), F.lit(0.0)))
        .withColumn("ctx_tip_recommend_share", F.coalesce(F.col("ctx_tip_recommend_share"), F.lit(0.0)))
        .withColumn("ctx_tip_time_share", F.coalesce(F.col("ctx_tip_time_share"), F.lit(0.0)))
        .withColumn("ctx_tip_dish_share", F.coalesce(F.col("ctx_tip_dish_share"), F.lit(0.0)))
        .withColumn("ctx_tip_recent_share", F.coalesce(F.col("ctx_tip_recent_share"), F.lit(0.0)))
        .withColumn("ctx_checkin_rows", F.coalesce(F.col("ctx_checkin_rows"), F.lit(0.0)))
        .withColumn("ctx_checkin_weekend_share", F.coalesce(F.col("ctx_checkin_weekend_share"), F.lit(0.0)))
        .withColumn("ctx_checkin_lunch_share", F.coalesce(F.col("ctx_checkin_lunch_share"), F.lit(0.0)))
        .withColumn("ctx_checkin_dinner_share", F.coalesce(F.col("ctx_checkin_dinner_share"), F.lit(0.0)))
        .withColumn("ctx_checkin_late_share", F.coalesce(F.col("ctx_checkin_late_share"), F.lit(0.0)))
        .withColumn("ctx_checkin_recent_share", F.coalesce(F.col("ctx_checkin_recent_share"), F.lit(0.0)))
        .withColumn("ctx_checkin_repeat_density", F.coalesce(F.col("ctx_checkin_repeat_density"), F.lit(0.0)))
        .withColumn(
            "ctx_business_gate",
            F.lit(0.5) * F.col("ctx_has_hours") + F.lit(0.5) * F.col("ctx_meal_any"),
        )
        .withColumn(
            "ctx_review_gate",
            F.least(F.col("ctx_review_avg_evidence") / F.lit(1.0), F.lit(1.0))
            * (F.lit(0.5) + F.lit(0.5) * F.col("ctx_review_recent_share")),
        )
        .withColumn(
            "ctx_tip_gate",
            F.least(F.log(F.col("ctx_tip_rows") + F.lit(1.0)) / F.log(F.lit(11.0)), F.lit(1.0))
            * (
                F.lit(0.5)
                + F.lit(0.5)
                * F.greatest("ctx_tip_recommend_share", "ctx_tip_time_share", "ctx_tip_dish_share")
            ),
        )
        .withColumn(
            "ctx_checkin_gate",
            F.least(F.log(F.col("ctx_checkin_rows") + F.lit(1.0)) / F.log(F.lit(51.0)), F.lit(1.0))
            * (
                F.lit(0.5)
                + F.lit(0.5)
                * F.greatest("ctx_checkin_weekend_share", "ctx_checkin_dinner_share", "ctx_checkin_late_share")
            ),
        )
        .withColumn(
            "ctx_support_gate",
            F.lit(0.25) * F.col("ctx_business_gate")
            + F.lit(0.25) * F.col("ctx_review_gate")
            + F.lit(0.20) * F.col("ctx_tip_gate")
            + F.lit(0.30) * F.col("ctx_checkin_gate"),
        )
        .persist(StorageLevel.DISK_ONLY)
    )

    attr_cols = [
        "ctx_attr_delivery",
        "ctx_attr_takeout",
        "ctx_attr_reservations",
        "ctx_attr_groups",
        "ctx_attr_good_for_kids",
        "ctx_attr_outdoor",
        "ctx_attr_table_service",
        "ctx_attr_alcohol_full_bar",
        "ctx_attr_alcohol_beer_wine",
        "ctx_attr_price_range_norm",
    ]
    time_cols = [
        "ctx_open_weekend",
        "ctx_open_late_any",
        "ctx_meal_breakfast",
        "ctx_meal_brunch",
        "ctx_meal_lunch",
        "ctx_meal_dinner",
        "ctx_meal_latenight",
        "ctx_meal_dessert",
        "ctx_tip_time_share",
        "ctx_tip_recent_share",
        "ctx_checkin_weekend_share",
        "ctx_checkin_lunch_share",
        "ctx_checkin_dinner_share",
        "ctx_checkin_late_share",
        "ctx_checkin_recent_share",
    ]

    test_scope = test_users.select("user_idx").dropDuplicates(["user_idx"]).persist(StorageLevel.DISK_ONLY)
    train_anchor = (
        train_idx.join(test_scope, on="user_idx", how="inner")
        .groupBy("user_idx")
        .agg(F.max("ts").alias("anchor_ts"))
    )
    weighted_train = (
        train_idx.join(test_scope, on="user_idx", how="inner")
        .join(train_anchor, on="user_idx", how="left")
        .join(item_ctx, on="item_idx", how="left")
        .withColumn(
            "rating_norm",
            F.least(
                F.greatest((F.col("rating").cast("double") - F.lit(1.0)) / F.lit(4.0), F.lit(0.0)),
                F.lit(1.0),
            ),
        )
        .withColumn(
            "ctx_days_since_anchor",
            F.when(
                F.col("anchor_ts").isNotNull() & F.col("ts").isNotNull(),
                F.datediff(F.col("anchor_ts"), F.col("ts")).cast("double"),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "ctx_event_weight",
            (F.lit(0.35) + F.lit(0.65) * F.col("rating_norm"))
            * F.exp(-F.greatest(F.col("ctx_days_since_anchor"), F.lit(0.0)) / F.lit(180.0)),
        )
        .withColumn(
            "ctx_geo_weight",
            F.when(
                F.col("ctx_latitude").isNotNull() & F.col("ctx_longitude").isNotNull(),
                F.col("ctx_event_weight"),
            ).otherwise(F.lit(0.0)),
        )
        .persist(StorageLevel.DISK_ONLY)
    )

    agg_exprs: list[Any] = [
        F.sum("ctx_event_weight").alias("ctx_total_weight"),
        F.count("*").alias("ctx_event_rows"),
        F.sum("ctx_geo_weight").alias("ctx_geo_weight_sum"),
    ]
    for col_name in attr_cols + time_cols:
        agg_exprs.append(F.sum(F.col("ctx_event_weight") * F.coalesce(F.col(col_name), F.lit(0.0))).alias(f"{col_name}_sum"))
    agg_exprs.extend(
        [
            F.sum(F.col("ctx_geo_weight") * F.coalesce(F.col("ctx_latitude"), F.lit(0.0))).alias("ctx_latitude_sum"),
            F.sum(F.col("ctx_geo_weight") * F.coalesce(F.col("ctx_longitude"), F.lit(0.0))).alias("ctx_longitude_sum"),
        ]
    )
    user_ctx = weighted_train.groupBy("user_idx").agg(*agg_exprs)
    for col_name in attr_cols + time_cols:
        user_ctx = user_ctx.withColumn(
            f"user_{col_name}",
            F.when(F.col("ctx_total_weight") > F.lit(0.0), F.col(f"{col_name}_sum") / F.col("ctx_total_weight")).otherwise(F.lit(0.0)),
        )
    user_ctx = (
        user_ctx.withColumn(
            "user_ctx_latitude",
            F.when(F.col("ctx_geo_weight_sum") > F.lit(0.0), F.col("ctx_latitude_sum") / F.col("ctx_geo_weight_sum")).otherwise(F.lit(None).cast("double")),
        )
        .withColumn(
            "user_ctx_longitude",
            F.when(F.col("ctx_geo_weight_sum") > F.lit(0.0), F.col("ctx_longitude_sum") / F.col("ctx_geo_weight_sum")).otherwise(F.lit(None).cast("double")),
        )
        .select(
            "user_idx",
            "ctx_total_weight",
            "ctx_event_rows",
            "ctx_geo_weight_sum",
            *[f"user_{c}" for c in attr_cols + time_cols],
            "user_ctx_latitude",
            "user_ctx_longitude",
        )
    )

    item_cols = [
        "item_idx",
        "business_id",
        "ctx_latitude",
        "ctx_longitude",
        "ctx_support_gate",
        *attr_cols,
        *time_cols,
    ]
    item_pdf = item_ctx.select(*item_cols).dropDuplicates(["item_idx"]).toPandas()
    user_pdf = user_ctx.join(test_scope, on="user_idx", how="inner").toPandas()
    test_scope.unpersist()
    weighted_train.unpersist()
    item_ctx.unpersist()

    if item_pdf.empty:
        return _empty_source_df(spark, "context"), {"status": "empty_item_context", "route_rows": {}, "route_users": {}}
    if user_pdf.empty:
        return _empty_source_df(spark, "context"), {"status": "empty_user_context", "route_rows": {}, "route_users": {}}

    item_pdf["item_idx"] = pd.to_numeric(item_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    item_pdf = item_pdf[item_pdf["item_idx"] >= 0].copy()
    if item_pdf.empty:
        return _empty_source_df(spark, "context"), {"status": "invalid_item_context", "route_rows": {}, "route_users": {}}
    for col_name in ["ctx_latitude", "ctx_longitude", "ctx_support_gate", *attr_cols, *time_cols]:
        item_pdf[col_name] = pd.to_numeric(item_pdf[col_name], errors="coerce")
    item_pdf["ctx_support_gate"] = (
        item_pdf["ctx_support_gate"].fillna(0.0).clip(lower=0.0, upper=1.0).astype(np.float32)
    )
    item_indices = item_pdf["item_idx"].to_numpy(np.int32)
    item_support = item_pdf["ctx_support_gate"].to_numpy(np.float32)
    item_lat = pd.to_numeric(item_pdf["ctx_latitude"], errors="coerce").to_numpy(np.float32)
    item_lon = pd.to_numeric(item_pdf["ctx_longitude"], errors="coerce").to_numpy(np.float32)
    item_geo_valid = np.isfinite(item_lat) & np.isfinite(item_lon)

    user_pdf["user_idx"] = pd.to_numeric(user_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    user_pdf = user_pdf[user_pdf["user_idx"] >= 0].copy()
    if user_pdf.empty:
        return _empty_source_df(spark, "context"), {"status": "invalid_user_context", "route_rows": {}, "route_users": {}}
    for col_name in ["ctx_total_weight", "ctx_event_rows", "ctx_geo_weight_sum", "user_ctx_latitude", "user_ctx_longitude"]:
        user_pdf[col_name] = pd.to_numeric(user_pdf[col_name], errors="coerce")
    for col_name in [f"user_{c}" for c in attr_cols + time_cols]:
        user_pdf[col_name] = pd.to_numeric(user_pdf[col_name], errors="coerce").fillna(0.0).astype(np.float32)

    def _norm_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (mat / norms).astype(np.float32), norms.reshape(-1).astype(np.float32)

    item_attr_mat = item_pdf[attr_cols].fillna(0.0).to_numpy(np.float32)
    item_time_mat = item_pdf[time_cols].fillna(0.0).to_numpy(np.float32)
    item_attr_normed, item_attr_norm = _norm_rows(item_attr_mat)
    item_time_normed, item_time_norm = _norm_rows(item_time_mat)
    user_weight_gate = np.clip(
        np.log1p(
            np.maximum(
                pd.to_numeric(user_pdf["ctx_total_weight"], errors="coerce").fillna(0.0).to_numpy(np.float32),
                0.0,
            )
        )
        / np.log(8.0),
        0.0,
        1.0,
    ).astype(np.float32)

    rows: list[tuple[int, int, int, float, float, str, str]] = []
    route_rows: dict[str, int] = {}
    route_users: dict[str, int] = {}
    route_status: dict[str, Any] = {}
    top_k = max(1, min(int(context_top_k), int(item_pdf.shape[0])))
    batch_n = int(max(1, batch_users))
    min_score = float(context_score_min)

    def _append_route_rows(route_name: str, user_batch: np.ndarray, score: np.ndarray, conf: np.ndarray) -> None:
        written_before = len(rows)
        active_users = 0
        if score.size == 0:
            route_rows.setdefault(route_name, 0)
            route_users.setdefault(route_name, 0)
            route_status[route_name] = {"status": "empty_score"}
            return
        top_idx = np.argpartition(-score, kth=top_k - 1, axis=1)[:, :top_k]
        for row_pos in range(top_idx.shape[0]):
            idxs = top_idx[row_pos]
            order = idxs[np.argsort(-score[row_pos, idxs], kind="mergesort")]
            rank_j = 0
            for item_pos in order.tolist():
                sc = float(score[row_pos, int(item_pos)])
                if (not np.isfinite(sc)) or sc < min_score:
                    continue
                rank_j += 1
                rows.append(
                    (
                        int(user_batch[row_pos]),
                        int(item_indices[int(item_pos)]),
                        int(rank_j),
                        float(sc),
                        float(np.clip(conf[row_pos, int(item_pos)], 0.0, 1.0)),
                        "context",
                        str(route_name),
                    )
                )
            if rank_j > 0:
                active_users += 1
        route_rows[route_name] = int(route_rows.get(route_name, 0) + int(len(rows) - written_before))
        route_users[route_name] = int(route_users.get(route_name, 0) + int(active_users))
        route_status[route_name] = {
            "status": "ok",
            "rows": int(route_rows[route_name]),
            "users": int(route_users[route_name]),
            "top_k": int(top_k),
            "score_min": float(min_score),
        }

    for start in range(0, int(user_pdf.shape[0]), batch_n):
        batch_pdf = user_pdf.iloc[start : start + batch_n].copy()
        if batch_pdf.empty:
            continue
        user_batch = batch_pdf["user_idx"].to_numpy(np.int32)
        batch_user_gate = user_weight_gate[start : start + batch_pdf.shape[0]]
        if "attr" in route_details:
            user_attr_mat = batch_pdf[[f"user_{c}" for c in attr_cols]].fillna(0.0).to_numpy(np.float32)
            user_attr_normed, user_attr_norm = _norm_rows(user_attr_mat)
            attr_score = np.matmul(user_attr_normed, item_attr_normed.T).astype(np.float32)
            attr_score = np.where(user_attr_norm[:, None] > 1e-6, attr_score, np.float32(0.0))
            attr_score = np.clip(0.82 * attr_score + 0.18 * item_support[None, :], 0.0, 1.5)
            attr_conf = np.clip(
                batch_user_gate[:, None] * (0.55 + 0.45 * item_support[None, :]),
                0.0,
                1.0,
            ).astype(np.float32)
            _append_route_rows("context_attr", user_batch, attr_score, attr_conf)
        if "time" in route_details:
            user_time_mat = batch_pdf[[f"user_{c}" for c in time_cols]].fillna(0.0).to_numpy(np.float32)
            user_time_normed, user_time_norm = _norm_rows(user_time_mat)
            time_score = np.matmul(user_time_normed, item_time_normed.T).astype(np.float32)
            time_score = np.where(user_time_norm[:, None] > 1e-6, time_score, np.float32(0.0))
            time_score = np.clip(0.78 * time_score + 0.22 * item_support[None, :], 0.0, 1.5)
            time_conf = np.clip(
                batch_user_gate[:, None] * (0.50 + 0.50 * item_support[None, :]),
                0.0,
                1.0,
            ).astype(np.float32)
            _append_route_rows("context_time", user_batch, time_score, time_conf)
        if "geo" in route_details:
            batch_lat = pd.to_numeric(batch_pdf["user_ctx_latitude"], errors="coerce").to_numpy(np.float32)
            batch_lon = pd.to_numeric(batch_pdf["user_ctx_longitude"], errors="coerce").to_numpy(np.float32)
            geo_score = np.zeros((batch_pdf.shape[0], item_indices.shape[0]), dtype=np.float32)
            valid_user = np.isfinite(batch_lat) & np.isfinite(batch_lon)
            if valid_user.any() and item_geo_valid.any():
                lat1 = np.deg2rad(batch_lat[valid_user])[:, None]
                lon1 = np.deg2rad(batch_lon[valid_user])[:, None]
                lat2 = np.deg2rad(item_lat[item_geo_valid])[None, :]
                lon2 = np.deg2rad(item_lon[item_geo_valid])[None, :]
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
                dist_km = 6371.0 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(np.maximum(a, 0.0))))
                geo_valid_score = np.exp(-dist_km / float(max(CONTEXT_GEO_SCALE_KM, 1.0))).astype(np.float32)
                geo_score[np.where(valid_user)[0][:, None], np.where(item_geo_valid)[0][None, :]] = geo_valid_score
            geo_score = np.clip(0.88 * geo_score + 0.12 * item_support[None, :], 0.0, 1.5)
            geo_conf = np.clip(
                batch_user_gate[:, None]
                * (0.50 + 0.50 * item_support[None, :])
                * valid_user[:, None].astype(np.float32),
                0.0,
                1.0,
            ).astype(np.float32)
            _append_route_rows("context_geo", user_batch, geo_score, geo_conf)

    if not rows:
        return _empty_source_df(spark, "context"), {
            "status": "no_context_rows",
            "route_rows": route_rows,
            "route_users": route_users,
            "top_k": int(top_k),
            "score_min": float(min_score),
        }

    context_pdf = pd.DataFrame(
        rows,
        columns=["user_idx", "item_idx", "source_rank", "source_score", "source_confidence", "source", "source_detail"],
    )
    context_pdf["user_idx"] = pd.to_numeric(context_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    context_pdf["item_idx"] = pd.to_numeric(context_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    context_pdf["source_rank"] = pd.to_numeric(context_pdf["source_rank"], errors="coerce").fillna(0).astype(np.int32)
    context_pdf["source_score"] = pd.to_numeric(context_pdf["source_score"], errors="coerce").fillna(0.0).astype(np.float64)
    context_pdf["source_confidence"] = (
        pd.to_numeric(context_pdf["source_confidence"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).astype(np.float64)
    )
    context_pdf = context_pdf[
        (context_pdf["user_idx"] >= 0)
        & (context_pdf["item_idx"] >= 0)
        & (context_pdf["source_rank"] > 0)
        & np.isfinite(context_pdf["source_score"].to_numpy(np.float64))
    ].copy()
    context_pdf = context_pdf.sort_values(
        ["source_detail", "user_idx", "source_rank", "item_idx"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    if context_pdf.empty:
        return _empty_source_df(spark, "context"), {
            "status": "context_rows_invalid_after_cast",
            "route_rows": route_rows,
            "route_users": route_users,
            "top_k": int(top_k),
            "score_min": float(min_score),
        }
    context_df = _create_spark_df_from_pandas_safe(spark=spark, pdf=context_pdf).select(
        F.col("user_idx").cast("int"),
        F.col("item_idx").cast("int"),
        F.col("source_rank").cast("int"),
        F.col("source_score").cast("double"),
        F.col("source_confidence").cast("double"),
        F.col("source").cast("string"),
        F.col("source_detail").cast("string"),
    )
    return context_df, {
        "status": "ok",
        "rows": int(context_pdf.shape[0]),
        "top_k": int(top_k),
        "score_min": float(min_score),
        "route_rows": {str(k): int(v) for k, v in route_rows.items()},
        "route_users": {str(k): int(v) for k, v in route_users.items()},
        "route_status": route_status,
        "n_item_rows": int(item_pdf.shape[0]),
        "n_user_rows": int(user_pdf.shape[0]),
    }


def load_business_pool(spark: SparkSession) -> tuple[DataFrame, dict[str, Any]]:
    business = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select(
            "business_id",
            "name",
            "state",
            "city",
            "categories",
            "is_open",
            "stars",
            "review_count",
        )
        .withColumn("business_id", F.col("business_id").cast("string"))
    )
    cat = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    biz = business.filter(F.col("state") == TARGET_STATE)
    cond = None
    if REQUIRE_RESTAURANTS:
        cond = cat.contains("restaurants")
    if REQUIRE_FOOD:
        cond = (cond | cat.contains("food")) if cond is not None else cat.contains("food")
    if cond is not None:
        biz = biz.filter(cond)

    stats: dict[str, Any] = {"n_scope_before_hard_filter": int(biz.count())}

    if FILTER_POLICY["require_is_open"]:
        biz = biz.filter(F.col("is_open") == 1)
    biz = biz.filter(F.col("stars") >= F.lit(float(FILTER_POLICY["min_business_stars"])))
    biz = biz.filter(F.col("review_count") >= F.lit(int(FILTER_POLICY["min_business_review_count"])))

    # Time filter via last review timestamp.
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("business_id", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
    )
    last_review = rvw.groupBy("business_id").agg(F.max("ts").alias("last_review_ts"))
    biz = biz.join(last_review, on="business_id", how="left")
    stale_cutoff = str(FILTER_POLICY["stale_cutoff_date"]).strip()
    biz = biz.filter(F.col("last_review_ts") >= F.to_timestamp(F.lit(stale_cutoff)))

    biz = biz.withColumn(
        "primary_category",
        F.lower(F.trim(F.element_at(F.split(F.coalesce(F.col("categories"), F.lit("")), r"\s*,\s*"), 1))),
    ).persist(StorageLevel.DISK_ONLY)

    stats["n_scope_after_hard_filter"] = int(biz.count())
    stats["stale_cutoff_date"] = stale_cutoff
    return biz, stats


def load_interactions(spark: SparkSession, biz: DataFrame) -> DataFrame:
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "stars", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(biz.select("business_id"), on="business_id", how="inner")
    )
    if RUN_PROFILE == "sample":
        rvw = rvw.sample(False, 0.15, RANDOM_SEED)
    return rvw.persist(StorageLevel.DISK_ONLY)


def leave_two_out(
    rvw: DataFrame, min_user_reviews: int
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    user_counts = rvw.groupBy("user_id").agg(F.count("*").alias("n_reviews"))
    users = user_counts.filter(F.col("n_reviews") >= min_user_reviews).select("user_id")
    rvw2 = rvw.join(users, on="user_id", how="inner")
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = rvw2.withColumn("rn", F.row_number().over(w))
    test = ranked.filter(F.col("rn") == 1)
    valid = ranked.filter(F.col("rn") == 2)
    train = ranked.filter(F.col("rn") > 2)
    return rvw2, train, valid, test


def index_ids(
    full_df: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip").fit(full_df)
    item_indexer = StringIndexer(inputCol="business_id", outputCol="item_idx", handleInvalid="skip").fit(full_df)

    def _transform(df: DataFrame) -> DataFrame:
        out = user_indexer.transform(df)
        out = item_indexer.transform(out)
        return (
            out.withColumn("user_idx", F.col("user_idx").cast("int"))
            .withColumn("item_idx", F.col("item_idx").cast("int"))
            .withColumn("rating", F.col("stars").cast("float"))
        )

    return _transform(train), _transform(valid), _transform(test)


def build_source_weight_expr(min_train: int) -> Any:
    expr = F.lit(0.0)
    source_order = ["als", "cluster", "popular", "profile", "context"]
    bucket_cfg = SOURCE_WEIGHTS_BY_BUCKET.get(int(min_train), {})
    for seg in ("light", "mid", "heavy"):
        for src in source_order:
            seg_cfg = bucket_cfg.get(seg, {})
            value = float(seg_cfg.get(src, SOURCE_WEIGHTS[seg][src]))
            expr = F.when(
                (F.col("user_segment") == F.lit(seg)) & (F.col("source") == F.lit(src)),
                F.lit(value),
            ).otherwise(expr)
    return expr


def build_segment_value_expr(value_by_segment: dict[str, float], default_value: float = 0.0) -> Any:
    expr = F.lit(float(default_value))
    for seg in ("light", "mid", "heavy"):
        expr = F.when(
            F.col("user_segment") == F.lit(seg),
            F.lit(float(value_by_segment.get(seg, default_value))),
        ).otherwise(expr)
    return expr


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _inv_from_scores(scores: np.ndarray) -> np.ndarray:
    n = int(scores.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    order = np.argsort(scores)[::-1]
    ranks = np.empty((n,), dtype=np.int32)
    ranks[order] = np.arange(1, n + 1, dtype=np.int32)
    return (1.0 / np.log2(ranks.astype(np.float64) + 1.0)).astype(np.float32)


def _train_tower_bpr(
    n_users: int,
    n_items: int,
    pairs: np.ndarray,
    user_pos: dict[int, set[int]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(int(seed))
    user_vec = (rng.normal(0, 0.1, size=(n_users, TOWER_SEQ_DIM)) / math.sqrt(max(1, TOWER_SEQ_DIM))).astype(np.float32)
    item_vec = (rng.normal(0, 0.1, size=(n_items, TOWER_SEQ_DIM)) / math.sqrt(max(1, TOWER_SEQ_DIM))).astype(np.float32)
    if pairs.shape[0] == 0:
        return user_vec, item_vec, 0
    steps = int(TOWER_SEQ_STEPS_PER_EPOCH) if TOWER_SEQ_STEPS_PER_EPOCH > 0 else int(np.clip(pairs.shape[0] * 6, 4000, 50000))
    updates = 0
    for ep in range(int(TOWER_SEQ_EPOCHS)):
        lr = float(TOWER_SEQ_LR) * (0.95**ep)
        for _ in range(steps):
            ridx = int(rng.integers(0, pairs.shape[0]))
            uid = int(pairs[ridx, 0])
            iid = int(pairs[ridx, 1])
            pos_set = user_pos.get(uid)
            if not pos_set or len(pos_set) >= n_items:
                continue
            j = int(rng.integers(0, n_items))
            t = 0
            while j in pos_set and t < int(TOWER_SEQ_NEG_TRIES):
                j = int(rng.integers(0, n_items))
                t += 1
            if j in pos_set:
                continue
            u0 = user_vec[uid].copy()
            i0 = item_vec[iid].copy()
            j0 = item_vec[j].copy()
            x = float(np.clip(np.dot(u0, i0) - np.dot(u0, j0), -20.0, 20.0))
            g = 1.0 - (1.0 / (1.0 + math.exp(-x)))
            user_vec[uid] = u0 + lr * (g * (i0 - j0) - float(TOWER_SEQ_REG) * u0)
            item_vec[iid] = i0 + lr * (g * u0 - float(TOWER_SEQ_REG) * i0)
            item_vec[j] = j0 + lr * (-g * u0 - float(TOWER_SEQ_REG) * j0)
            updates += 1
    return user_vec, item_vec, int(updates)


def _build_seq_vectors(train_pdf: pd.DataFrame, item_vec: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    seq = np.zeros((n_users, item_vec.shape[1]), dtype=np.float32)
    ready = np.zeros((n_users,), dtype=np.int32)
    if train_pdf.empty:
        return seq, ready
    grouped = train_pdf.sort_values(["user_idx", "ts"], ascending=[True, False]).groupby("user_idx", sort=False)
    for uid, g in grouped:
        ids = g["item_idx"].astype(np.int32).tolist()[: int(SEQ_RECENT_LEN)]
        if len(ids) < int(SEQ_RECENT_MIN_LEN):
            continue
        arr = np.array(ids, dtype=np.int32)
        arr = arr[(arr >= 0) & (arr < item_vec.shape[0])]
        if arr.size < int(SEQ_RECENT_MIN_LEN):
            continue
        w = np.array([float(SEQ_DECAY) ** i for i in range(arr.size)], dtype=np.float32)
        w = w / max(1e-6, float(w.sum()))
        seq[int(uid)] = (item_vec[arr] * w[:, None]).sum(axis=0)
        ready[int(uid)] = 1
    return seq, ready


def build_tower_seq_features_for_candidates(
    spark: SparkSession,
    fused_pretrim: DataFrame,
    train_idx: DataFrame,
    bucket: int,
) -> tuple[DataFrame | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": bool(ENABLE_TOWER_SEQ_FEATURES),
        "status": "disabled",
        "rows": 0,
        "tower_seq_max_cand_rows_effective": 0,
        "tower_seq_candidate_window_mode": "full",
        "tower_seq_candidate_window_topk": 0,
        "tower_seq_rows_total_before_window_probe": 0,
        "n_users": 0,
        "n_items": 0,
        "updates": 0,
        "seq_ready_rate": 0.0,
        "tower_seq_scope_users_only": bool(TOWER_SEQ_SCOPE_USERS_ONLY),
        "tower_seq_scope_items_only": bool(TOWER_SEQ_SCOPE_ITEMS_ONLY),
        "tower_seq_train_max_per_user": int(TOWER_SEQ_TRAIN_MAX_PER_USER),
        "tower_seq_apply_head_window_first": bool(TOWER_SEQ_APPLY_HEAD_WINDOW_FIRST),
        "tower_seq_input_materialize_mode": str(TOWER_SEQ_INPUT_MATERIALIZE_MODE),
        "tower_seq_input_candidate_path": "",
        "tower_seq_input_train_path": "",
    }
    if not ENABLE_TOWER_SEQ_FEATURES:
        return None, meta

    # Probe only a bounded candidate surface before materializing Pandas data.
    # For buckets with a configured head window, optionally apply that window
    # before the probe so large pretrim surfaces do not force a heavy full-DAG
    # count just to decide whether tower/seq is feasible.
    cap = int(get_tower_seq_max_cand_rows(bucket))
    meta["tower_seq_max_cand_rows_effective"] = int(cap)
    cand_source = fused_pretrim
    head_window = int(get_tower_seq_head_window(bucket))
    if bool(TOWER_SEQ_APPLY_HEAD_WINDOW_FIRST) and head_window > 0:
        cand_source = fused_pretrim.filter(F.col("pre_rank") <= F.lit(int(head_window)))
        meta["tower_seq_candidate_window_mode"] = "pre_rank_head"
        meta["tower_seq_candidate_window_topk"] = int(head_window)
    probed_rows = int(cand_source.limit(cap + 1).count())
    meta["tower_seq_rows_total_before_window_probe"] = int(probed_rows)
    cand_rows = int(probed_rows)
    if probed_rows <= 0:
        meta["rows"] = int(probed_rows)
        meta["status"] = "empty_candidates"
        return None, meta
    if probed_rows > cap:
        if meta["tower_seq_candidate_window_mode"] == "pre_rank_head" or head_window <= 0:
            meta["rows"] = int(probed_rows)
            meta["status"] = "skip_row_cap"
            return None, meta
        cand_source = fused_pretrim.filter(F.col("pre_rank") <= F.lit(int(head_window)))
        cand_rows = int(cand_source.limit(cap + 1).count())
        meta["tower_seq_candidate_window_mode"] = "pre_rank_head"
        meta["tower_seq_candidate_window_topk"] = int(head_window)
        meta["rows"] = int(cand_rows)
        if cand_rows <= 0:
            meta["status"] = "empty_windowed_candidates"
            return None, meta
        if cand_rows > cap:
            meta["status"] = "skip_row_cap"
            return None, meta
    else:
        meta["rows"] = int(cand_rows)

    cand_base = cand_source.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"])
    cand_pdf, cand_pdf_meta = _materialize_small_df_to_pandas(
        cand_base,
        subdir="tower_seq_inputs",
        prefix="tower_seq_cand",
        mode=TOWER_SEQ_INPUT_MATERIALIZE_MODE,
    )
    meta["tower_seq_input_candidate_path"] = str(cand_pdf_meta.get("path", ""))
    train_src = train_idx.select("user_idx", "item_idx", "ts")
    if TOWER_SEQ_SCOPE_USERS_ONLY:
        train_src = train_src.join(cand_base.select("user_idx").dropDuplicates(["user_idx"]), on="user_idx", how="inner")
    if TOWER_SEQ_SCOPE_ITEMS_ONLY:
        train_src = train_src.join(cand_base.select("item_idx").dropDuplicates(["item_idx"]), on="item_idx", how="inner")
    if int(TOWER_SEQ_TRAIN_MAX_PER_USER) > 0:
        w_train_recent = Window.partitionBy("user_idx").orderBy(F.col("ts").desc_nulls_last(), F.col("item_idx").desc())
        train_src = (
            train_src.withColumn("_rn", F.row_number().over(w_train_recent))
            .filter(F.col("_rn") <= F.lit(int(TOWER_SEQ_TRAIN_MAX_PER_USER)))
            .drop("_rn")
        )
    train_pdf, train_pdf_meta = _materialize_small_df_to_pandas(
        train_src,
        subdir="tower_seq_inputs",
        prefix="tower_seq_train",
        mode=TOWER_SEQ_INPUT_MATERIALIZE_MODE,
    )
    meta["tower_seq_input_train_path"] = str(train_pdf_meta.get("path", ""))
    if cand_pdf.empty or train_pdf.empty:
        meta["status"] = "empty_mapping"
        return None, meta

    cand_pdf["user_idx"] = pd.to_numeric(cand_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    cand_pdf["item_idx"] = pd.to_numeric(cand_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    train_pdf["user_idx"] = pd.to_numeric(train_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    train_pdf["item_idx"] = pd.to_numeric(train_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    cand_pdf = cand_pdf[(cand_pdf["user_idx"] >= 0) & (cand_pdf["item_idx"] >= 0)].copy()
    train_pdf = train_pdf[(train_pdf["user_idx"] >= 0) & (train_pdf["item_idx"] >= 0)].copy()
    if cand_pdf.empty or train_pdf.empty:
        meta["status"] = "empty_filtered"
        return None, meta

    n_users = int(max(int(cand_pdf["user_idx"].max()), int(train_pdf["user_idx"].max())) + 1)
    n_items = int(max(int(cand_pdf["item_idx"].max()), int(train_pdf["item_idx"].max())) + 1)
    meta["n_users"] = int(n_users)
    meta["n_items"] = int(n_items)
    if n_users <= 0 or n_items <= 0:
        meta["status"] = "invalid_dims"
        return None, meta

    pairs_pdf = train_pdf[["user_idx", "item_idx"]].drop_duplicates()
    pairs = pairs_pdf.to_numpy(dtype=np.int32, copy=True)
    if pairs.shape[0] == 0:
        meta["status"] = "empty_pairs"
        return None, meta

    user_pos: dict[int, set[int]] = {}
    for uid, iid in pairs:
        user_pos.setdefault(int(uid), set()).add(int(iid))

    uvec, ivec, updates = _train_tower_bpr(
        n_users=n_users,
        n_items=n_items,
        pairs=pairs,
        user_pos=user_pos,
        seed=int(RANDOM_SEED + bucket * 97),
    )
    svec, sready = _build_seq_vectors(train_pdf[["user_idx", "item_idx", "ts"]].copy(), ivec, n_users)
    meta["updates"] = int(updates)
    meta["seq_ready_rate"] = float(sready.sum()) / float(max(1, n_users))

    cand_pdf["tower_score"] = np.float32(0.0)
    cand_pdf["seq_score"] = np.float32(0.0)
    cand_pdf["tower_inv"] = np.float32(0.0)
    cand_pdf["seq_inv"] = np.float32(0.0)
    idx_tower_score = cand_pdf.columns.get_loc("tower_score")
    idx_seq_score = cand_pdf.columns.get_loc("seq_score")
    idx_tower_inv = cand_pdf.columns.get_loc("tower_inv")
    idx_seq_inv = cand_pdf.columns.get_loc("seq_inv")
    for uid, idx in cand_pdf.groupby("user_idx", sort=False).groups.items():
        if uid < 0 or uid >= n_users:
            continue
        row_idx = np.asarray(list(idx), dtype=np.int32)
        items = cand_pdf.iloc[row_idx]["item_idx"].to_numpy(dtype=np.int32, copy=True)
        valid = (items >= 0) & (items < n_items)
        if not valid.any():
            continue
        vv = np.where(valid)[0]
        ii = items[vv]
        tv = ivec[ii] @ uvec[int(uid)]
        sv = ivec[ii] @ svec[int(uid)]
        cand_pdf.iloc[row_idx[vv], idx_tower_score] = tv.astype(np.float32)
        cand_pdf.iloc[row_idx[vv], idx_seq_score] = sv.astype(np.float32)
        cand_pdf.iloc[row_idx[vv], idx_tower_inv] = _inv_from_scores(tv)
        cand_pdf.iloc[row_idx[vv], idx_seq_inv] = _inv_from_scores(sv)

    out_pdf = cand_pdf[["user_idx", "item_idx", "tower_score", "seq_score", "tower_inv", "seq_inv"]].copy()
    out_pdf["user_idx"] = pd.to_numeric(out_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
    out_pdf["item_idx"] = pd.to_numeric(out_pdf["item_idx"], errors="coerce").fillna(-1).astype(np.int32)
    for c in ("tower_score", "seq_score", "tower_inv", "seq_inv"):
        out_pdf[c] = pd.to_numeric(out_pdf[c], errors="coerce").fillna(0.0).astype(np.float64)
    out_pdf = out_pdf[(out_pdf["user_idx"] >= 0) & (out_pdf["item_idx"] >= 0)].copy()
    if out_pdf.empty:
        meta["status"] = "empty_output"
        return None, meta

    out_df: DataFrame | None = None
    output_mode_effective = str(TOWER_SEQ_OUTPUT_MODE)
    if output_mode_effective == "auto":
        output_mode_effective = (
            "spark_df" if int(out_pdf.shape[0]) <= int(PROFILE_OUTPUT_SPARK_DF_MAX_ROWS) else "csv"
        )
    if output_mode_effective == "csv" and (not ALLOW_TOWER_SEQ_CSV_FALLBACK):
        output_mode_effective = "spark_df"
    if output_mode_effective != "csv":
        try:
            out_df = _create_spark_df_from_pandas_safe(
                spark=spark,
                pdf=out_pdf,
            ).select(
                F.col("user_idx").cast("int").alias("user_idx"),
                F.col("item_idx").cast("int").alias("item_idx"),
                F.col("tower_score").cast("double").alias("tower_score"),
                F.col("seq_score").cast("double").alias("seq_score"),
                F.col("tower_inv").cast("double").alias("tower_inv"),
                F.col("seq_inv").cast("double").alias("seq_inv"),
            )
        except Exception:
            out_df = None
    if out_df is None and ALLOW_TOWER_SEQ_CSV_FALLBACK:
        tmp_csv = _alloc_stage_tmp_csv("tower_seq_features", "tower_seq")
        out_pdf.to_csv(tmp_csv.as_posix(), index=False, encoding="utf-8")
        out_df = spark.read.csv(tmp_csv.as_posix(), header=True, inferSchema=True).select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("item_idx").cast("int").alias("item_idx"),
            F.col("tower_score").cast("double").alias("tower_score"),
            F.col("seq_score").cast("double").alias("seq_score"),
            F.col("tower_inv").cast("double").alias("tower_inv"),
            F.col("seq_inv").cast("double").alias("seq_inv"),
        )
        meta["tmp_csv"] = tmp_csv.as_posix()
    else:
        meta["tmp_csv"] = ""
    if out_df is None:
        raise RuntimeError(
            "tower/seq feature dataframe conversion failed; enable ALLOW_TOWER_SEQ_CSV_FALLBACK=true for debug fallback"
        )
    meta["status"] = "ok"
    return out_df, meta


def main() -> None:
    _apply_env_overrides()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_PROFILE}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    deferred_tmp_csv_cleanup: list[str] = []

    print("[STEP] load business pool with hard filters")
    biz, biz_stats = load_business_pool(spark)
    print(f"[COUNT] business_before={biz_stats['n_scope_before_hard_filter']} after={biz_stats['n_scope_after_hard_filter']}")
    print(f"[CONFIG] stale_cutoff={biz_stats['stale_cutoff_date']}")

    print("[STEP] load interactions")
    rvw = load_interactions(spark, biz)
    scoped_interaction_rows = int(rvw.count())
    scoped_interaction_users = int(rvw.select("user_id").distinct().count())
    print(f"[COUNT] interactions={scoped_interaction_rows} users={scoped_interaction_users}")

    print("[STEP] load cluster profile map")
    profile_csv = resolve_cluster_profile_csv()
    cluster_map_raw = (
        spark.read.csv(profile_csv.as_posix(), header=True)
        .select(
            F.col("business_id").cast("string").alias("business_id"),
            F.col(CLUSTER_PROFILE_CLUSTER_COL).cast("string").alias("cluster_id"),
        )
        .filter(F.col("business_id").isNotNull() & F.col("cluster_id").isNotNull() & (F.col("cluster_id") != ""))
        .dropDuplicates(["business_id"])
        .persist(StorageLevel.DISK_ONLY)
    )
    print(f"[INFO] cluster_profile_csv={profile_csv}")
    print(f"[COUNT] cluster_businesses={cluster_map_raw.count()}")

    profile_vec_path = None
    profile_table_path = None
    profile_tag_long_path = None
    profile_calibration_path = None
    profile_user_ids = np.array([], dtype=str)
    profile_vectors = np.zeros((0, 0), dtype=np.float32)
    profile_confidence_by_user_id: dict[str, float] = {}
    profile_calibration_models: dict[int, dict[str, Any]] = {}
    profile_multivector_paths: dict[str, str] = {}
    profile_multivector_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    user_profile_tag_long_pdf = pd.DataFrame(
        columns=["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"]
    )
    if ENABLE_PROFILE_RECALL:
        print("[STEP] load user profile vectors")
        profile_vec_path = resolve_user_profile_vectors()
        profile_table_path = resolve_user_profile_table(profile_vec_path)
        profile_tag_long_path = resolve_user_profile_tag_long(profile_table_path)
        profile_user_ids, profile_vectors = load_profile_vectors(profile_vec_path)
        profile_confidence_by_user_id = load_user_profile_confidence(profile_table_path)
        profile_calibration_path = resolve_profile_calibration_json()
        profile_calibration_models = load_profile_calibration_models(profile_calibration_path)
        user_profile_tag_long_pdf = load_user_profile_tag_long(profile_tag_long_path)
        print(
            f"[INFO] user_profile_vectors={profile_vec_path} "
            f"users={int(profile_user_ids.shape[0])} dim={int(profile_vectors.shape[1])}"
        )
        print(
            f"[INFO] user_profile_table={profile_table_path} "
            f"confidence_users={int(len(profile_confidence_by_user_id))}"
        )
        if profile_calibration_path is not None:
            print(
                f"[INFO] profile_calibration_json={profile_calibration_path} "
                f"buckets={sorted(profile_calibration_models.keys())}"
            )
        print(
            f"[INFO] user_profile_tag_long={profile_tag_long_path if profile_tag_long_path is not None else '<missing>'} "
            f"rows={int(user_profile_tag_long_pdf.shape[0])}"
        )
        if ENABLE_PROFILE_MULTIVECTOR_ROUTE_AUDIT or ENABLE_PROFILE_MULTIVECTOR_ROUTE_INTEGRATION:
            for scope in PROFILE_MULTIVECTOR_ROUTE_SCOPES:
                path = resolve_user_profile_multivector_file(profile_vec_path, scope)
                if path is None:
                    continue
                try:
                    profile_multivector_data[str(scope)] = load_profile_vectors(path)
                    profile_multivector_paths[str(scope)] = str(path)
                except Exception as e:
                    print(f"[WARN] user multivector scope={scope} load failed: {e}")
            print(
                f"[INFO] user_profile_multivector_scopes={sorted(profile_multivector_data.keys())} "
                f"count={int(len(profile_multivector_data))}"
            )
    else:
        print("[STEP] profile recall disabled by ENABLE_PROFILE_RECALL=false")

    item_semantic_path = None
    item_semantic_tag_long_path = None
    item_semantic_df = None
    item_semantic_rows = 0
    merchant_dense_vector_paths: dict[str, str] = {}
    merchant_dense_vector_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    item_semantic_tag_long_pdf = pd.DataFrame(
        columns=["business_id", "tag", "tag_type", "net_weight_sum", "tag_confidence", "support_count"]
    )
    if ENABLE_ITEM_SEMANTIC:
        try:
            item_semantic_path = resolve_item_semantic_features()
            item_semantic_tag_long_path = resolve_item_semantic_tag_long(item_semantic_path)
            item_semantic_df = load_item_semantic_features(spark, item_semantic_path).persist(StorageLevel.DISK_ONLY)
            item_semantic_rows = int(item_semantic_df.count())
            item_semantic_tag_long_pdf = load_item_semantic_tag_long(item_semantic_tag_long_path)
            print(f"[INFO] item_semantic_features={item_semantic_path} rows={item_semantic_rows}")
            print(
                f"[INFO] item_semantic_tag_long={item_semantic_tag_long_path if item_semantic_tag_long_path is not None else '<missing>'} "
                f"rows={int(item_semantic_tag_long_pdf.shape[0])}"
            )
            if ENABLE_PROFILE_MULTIVECTOR_ROUTE_AUDIT or ENABLE_PROFILE_MULTIVECTOR_ROUTE_INTEGRATION:
                for merchant_scope in sorted({pair[1] for pair in _build_multivector_scope_pair_list()}):
                    path = resolve_item_semantic_dense_vector_file(item_semantic_path, merchant_scope)
                    if path is None:
                        continue
                    try:
                        merchant_dense_vector_data[str(merchant_scope)] = load_merchant_dense_vectors(path)
                        merchant_dense_vector_paths[str(merchant_scope)] = str(path)
                    except Exception as e:
                        print(f"[WARN] merchant dense vector scope={merchant_scope} load failed: {e}")
                print(
                    f"[INFO] merchant_dense_vector_scopes={sorted(merchant_dense_vector_data.keys())} "
                    f"count={int(len(merchant_dense_vector_data))}"
                )
        except Exception as e:
            print(f"[WARN] item semantic disabled due to load failure: {e}")
            item_semantic_df = None
            item_semantic_tag_long_pdf = pd.DataFrame(
                columns=["business_id", "tag", "tag_type", "net_weight_sum", "tag_confidence", "support_count"]
            )
    else:
        print("[STEP] item semantic disabled by ENABLE_ITEM_SEMANTIC=false")

    business_context_df: DataFrame | None = None
    review_evidence_business_df: DataFrame | None = None
    tip_signal_business_df: DataFrame | None = None
    checkin_context_business_df: DataFrame | None = None
    if ENABLE_CONTEXT_SURFACE:
        print("[STEP] loading P0 context assets for stage09 context surface")
        try:
            business_context_features_path = resolve_business_context_features()
            review_evidence_business_path = resolve_review_evidence_business_agg()
            tip_signal_business_path = resolve_tip_signal_business_agg()
            checkin_context_business_path = resolve_checkin_context_business_agg()
            business_context_df = load_business_context_features(spark, business_context_features_path).persist(StorageLevel.DISK_ONLY)
            review_evidence_business_df = load_review_evidence_business_agg(
                spark, review_evidence_business_path
            ).persist(StorageLevel.DISK_ONLY)
            tip_signal_business_df = load_tip_signal_business_agg(spark, tip_signal_business_path).persist(
                StorageLevel.DISK_ONLY
            )
            checkin_context_business_df = load_checkin_context_business_agg(
                spark, checkin_context_business_path
            ).persist(StorageLevel.DISK_ONLY)
            print(
                "[INFO] context_assets "
                f"business={business_context_features_path.parent.name} "
                f"review={review_evidence_business_path.parent.name} "
                f"tip={tip_signal_business_path.parent.name} "
                f"checkin={checkin_context_business_path.parent.name}"
            )
        except Exception as exc:
            print(f"[WARN] context surface assets disabled due to load failure: {exc}")
            business_context_df = None
            review_evidence_business_df = None
            tip_signal_business_df = None
            checkin_context_business_df = None

    bucket_rows: list[dict[str, Any]] = []
    for min_train in MIN_TRAIN_REVIEWS_BUCKETS:
        min_user = min_train + MIN_USER_REVIEWS_OFFSET
        print(f"\n[BUCKET] min_train={min_train} min_user={min_user}")
        bucket_dir = out_dir / f"bucket_{min_train}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        recall_limits = get_bucket_recall_limits(min_train)
        profile_route_policy_bucket = get_profile_route_policy(min_train)
        profile_threshold_policy_bucket = get_profile_threshold_policy(min_train)
        als_top_k = int(recall_limits["als_top_k"])
        cluster_top_k = int(recall_limits["cluster_top_k"])
        popular_top_k = int(recall_limits["popular_top_k"])
        profile_top_k = int(recall_limits["profile_top_k"])
        pretrim_top_k = int(recall_limits["pretrim_top_k"])
        cluster_user_topn = int(recall_limits["cluster_user_topn"])
        pretrim_segment_policy = get_pretrim_segment_policy(min_train, pretrim_top_k)
        pretrim_head_policy = get_pretrim_head_policy(min_train)
        pretrim_head_enabled = bool(pretrim_head_policy.get("enable", True))
        pretrim_output_max_top_k = max(int(v) for v in pretrim_segment_policy.values())
        pretrim_segment_variable = len({int(v) for v in pretrim_segment_policy.values()}) > 1
        print(
            "[RECALL] "
            f"als_top_k={als_top_k} cluster_top_k={cluster_top_k} "
            f"popular_top_k={popular_top_k} profile_top_k={profile_top_k} "
            f"pretrim_top_k={pretrim_top_k} cluster_user_topn={cluster_user_topn}"
        )
        print(f"[PRETRIM_SEG] bucket={min_train} policy={pretrim_segment_policy}")
        print(f"[HEAD_POLICY] bucket={min_train} enabled={pretrim_head_enabled} policy={pretrim_head_policy}")
        print(
            "[PROFILE_ROUTE] "
            f"bucket={min_train} route={profile_route_policy_bucket} "
            f"threshold={profile_threshold_policy_bucket}"
        )

        full_df, train, valid, test = leave_two_out(rvw, min_user)
        full_df = full_df.persist(StorageLevel.DISK_ONLY)
        train = train.persist(StorageLevel.DISK_ONLY)
        valid = valid.persist(StorageLevel.DISK_ONLY)
        test = test.persist(StorageLevel.DISK_ONLY)

        n_users = int(full_df.select("user_id").distinct().count())
        n_items = int(full_df.select("business_id").distinct().count())
        n_train = int(train.count())
        n_valid = int(valid.count())
        n_test = int(test.count())
        print(f"[COUNT] users={n_users} items={n_items} train={n_train} valid={n_valid} test={n_test}")

        train_idx, valid_idx, test_idx = index_ids(full_df, train, valid, test)
        train_idx = train_idx.persist(StorageLevel.DISK_ONLY)
        valid_idx = valid_idx.persist(StorageLevel.DISK_ONLY)
        test_idx = test_idx.persist(StorageLevel.DISK_ONLY)

        # Map indexed ids back to original ids.
        item_map = (
            train_idx.select("item_idx", "business_id")
            .unionByName(valid_idx.select("item_idx", "business_id"), allowMissingColumns=True)
            .unionByName(test_idx.select("item_idx", "business_id"), allowMissingColumns=True)
            .dropDuplicates(["item_idx"])
            .persist(StorageLevel.DISK_ONLY)
        )
        user_map = (
            train_idx.select("user_idx", "user_id")
            .unionByName(valid_idx.select("user_idx", "user_id"), allowMissingColumns=True)
            .unionByName(test_idx.select("user_idx", "user_id"), allowMissingColumns=True)
            .dropDuplicates(["user_idx"])
            .persist(StorageLevel.DISK_ONLY)
        )

        # Segment and pop stats.
        user_train_counts = (
            train_idx.groupBy("user_idx")
            .agg(F.count("*").alias("user_train_count"))
            .withColumn(
                "user_segment",
                F.when(F.col("user_train_count") <= F.lit(LIGHT_MAX_TRAIN), F.lit("light"))
                .when(F.col("user_train_count") <= F.lit(MID_MAX_TRAIN), F.lit("mid"))
                .otherwise(F.lit("heavy")),
            )
            .persist(StorageLevel.DISK_ONLY)
        )
        pop_stats = (
            train_idx.groupBy("item_idx")
            .agg(F.count("*").alias("item_train_pop_count"))
            .persist(StorageLevel.DISK_ONLY)
        )
        total_train_events = int(n_train)

        biz_by_item = (
            item_map.join(biz.select("business_id", "name", "city", "categories", "stars", "review_count", "primary_category"), on="business_id", how="left")
            .persist(StorageLevel.DISK_ONLY)
        )
        test_users = test_idx.select("user_idx").distinct().persist(StorageLevel.DISK_ONLY)
        truth_test = (
            test_idx.select("user_idx", F.col("item_idx").alias("true_item_idx"))
            .dropDuplicates(["user_idx"])
        )
        truth_valid = (
            valid_idx.select("user_idx", F.col("item_idx").alias("valid_item_idx"))
            .dropDuplicates(["user_idx"])
        )
        truth = (
            truth_test.join(truth_valid, on="user_idx", how="left")
            .persist(StorageLevel.DISK_ONLY)
        )
        test_anchor = (
            test_idx.select("user_idx", F.col("ts").alias("test_ts"))
            .dropDuplicates(["user_idx"])
            .persist(StorageLevel.DISK_ONLY)
        )
        profile_multivector_route_audit_df = _empty_multivector_route_audit_df(spark)
        profile_multivector_route_meta: dict[str, Any] = {"status": "disabled"}
        if should_write_profile_multivector_route_audit(int(min_train)):
            profile_multivector_route_audit_df, profile_multivector_route_meta = build_profile_multivector_route_audit_df(
                spark=spark,
                test_users=test_users,
                user_map=user_map,
                item_map=item_map,
                profile_confidence_by_user_id=profile_confidence_by_user_id,
                user_vectors_by_scope=profile_multivector_data,
                merchant_vectors_by_scope=merchant_dense_vector_data,
                route_top_k=int(PROFILE_MULTIVECTOR_ROUTE_TOP_K),
                route_score_min=float(PROFILE_MULTIVECTOR_ROUTE_SCORE_MIN),
                batch_users=int(PROFILE_MULTIVECTOR_ROUTE_BATCH_USERS),
            )
            print(f"[INFO] bucket={min_train} profile_multivector_route_audit={profile_multivector_route_meta}")
        w_hist_latest = Window.partitionBy("user_idx", "item_idx").orderBy(F.col("hist_ts").desc(), F.col("review_id").desc())
        train_history = (
            train_idx.select(
                "user_idx",
                "item_idx",
                F.col("ts").alias("hist_ts"),
                F.col("rating").cast("double").alias("hist_rating"),
                "review_id",
            )
            .join(test_anchor, on="user_idx", how="left")
            .withColumn("days_to_test", F.datediff(F.col("test_ts"), F.col("hist_ts")))
            .filter(F.col("days_to_test").isNotNull() & (F.col("days_to_test") >= F.lit(1)))
            .withColumn("_rn", F.row_number().over(w_hist_latest))
            .filter(F.col("_rn") == F.lit(1))
            .drop("_rn", "review_id")
            .withColumn("hist_recency_w", F.exp(-F.col("days_to_test").cast("double") / F.lit(180.0)))
            .persist(StorageLevel.DISK_ONLY)
        )

        params = IMPLICIT_PARAMS_BY_BUCKET[min_train]
        train_imp = train_idx.withColumn("rating", F.lit(1.0))
        model = ALS(
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="rating",
            implicitPrefs=True,
            alpha=float(params["alpha"]),
            coldStartStrategy="drop",
            rank=int(params["rank"]),
            maxIter=5,
            regParam=float(params["reg"]),
            seed=RANDOM_SEED,
        ).fit(train_imp)

        # ALS candidates.
        als_cand = (
            model.recommendForUserSubset(test_users, als_top_k)
            .select("user_idx", F.posexplode("recommendations").alias("pos", "rec"))
            .select(
                "user_idx",
                F.col("rec.item_idx").cast("int").alias("item_idx"),
                (F.col("pos") + F.lit(1)).cast("int").alias("source_rank"),
                F.col("rec.rating").cast("double").alias("source_score"),
            )
            .withColumn("source_confidence", F.lit(1.0))
            .withColumn("source", F.lit("als"))
            .withColumn("source_detail", F.lit("als"))
        )

        # Cluster popular candidates.
        cluster_item_map = (
            item_map.join(cluster_map_raw, on="business_id", how="inner")
            .select("item_idx", "cluster_id")
            .dropDuplicates(["item_idx"])
            .persist(StorageLevel.DISK_ONLY)
        )
        user_cluster = (
            train_idx.join(cluster_item_map, on="item_idx", how="inner")
            .groupBy("user_idx", "cluster_id")
            .agg(F.count("*").alias("cnt"))
        )
        w_user_cluster = Window.partitionBy("user_idx").orderBy(F.desc("cnt"), F.asc("cluster_id"))
        user_cluster_total = (
            user_cluster.groupBy("user_idx")
            .agg(F.sum("cnt").alias("user_cluster_total"))
        )
        user_top_cluster = (
            user_cluster.withColumn("cluster_user_rank", F.row_number().over(w_user_cluster))
            .filter(F.col("cluster_user_rank") <= F.lit(int(cluster_user_topn)))
            .join(user_cluster_total, on="user_idx", how="left")
            .withColumn(
                "cluster_share",
                F.when(F.col("user_cluster_total") > 0, F.col("cnt").cast("double") / F.col("user_cluster_total").cast("double"))
                .otherwise(F.lit(0.0)),
            )
            .withColumn(
                "cluster_confidence",
                F.lit(float(CLUSTER_CONF_FLOOR))
                + (F.lit(1.0) - F.lit(float(CLUSTER_CONF_FLOOR))) * F.col("cluster_share"),
            )
            .select("user_idx", "cluster_id", "cluster_user_rank", "cluster_confidence")
            .persist(StorageLevel.DISK_ONLY)
        )
        cluster_pop = (
            train_idx.join(cluster_item_map, on="item_idx", how="inner")
            .groupBy("cluster_id", "item_idx")
            .agg(F.count("*").alias("cnt"))
        )
        w_cluster = Window.partitionBy("cluster_id").orderBy(F.desc("cnt"), F.asc("item_idx"))
        cluster_topk = (
            cluster_pop.withColumn("source_rank", F.row_number().over(w_cluster))
            .filter(F.col("source_rank") <= F.lit(cluster_top_k))
            .select("cluster_id", "item_idx", "source_rank", F.col("cnt").cast("double").alias("source_score"))
        )
        cluster_raw = (
            test_users.join(user_top_cluster, on="user_idx", how="left")
            .join(cluster_topk, on="cluster_id", how="inner")
            .select(
                "user_idx",
                "item_idx",
                F.col("source_rank").cast("int").alias("cluster_item_rank"),
                "source_score",
                F.col("cluster_confidence").cast("double").alias("source_confidence"),
            )
        )
        cluster_log2_const = float(math.log(2.0))
        w_cluster_user = Window.partitionBy("user_idx").orderBy(
            F.desc(
                F.col("source_confidence")
                * (
                    F.lit(1.0)
                    / (F.log(F.col("cluster_item_rank").cast("double") + F.lit(1.0)) / F.lit(cluster_log2_const))
                )
            ),
            F.asc("item_idx"),
        )
        cluster_cand = (
            cluster_raw.withColumn("cluster_signal_score", F.col("source_confidence") * F.col("source_score"))
            .withColumn("source_rank", F.row_number().over(w_cluster_user))
            .filter(F.col("source_rank") <= F.lit(cluster_top_k))
            .select(
                "user_idx",
                "item_idx",
                "source_rank",
                F.col("cluster_signal_score").cast("double").alias("source_score"),
                "source_confidence",
            )
            .withColumn("source", F.lit("cluster"))
            .withColumn("source_detail", F.lit("cluster"))
        )

        # Global popular candidates.
        popular_topk = build_global_popular_topk_df(
            spark=spark,
            pop_stats=pop_stats,
            top_k=popular_top_k,
        )
        pop_cand = (
            test_users.crossJoin(popular_topk)
            .select("user_idx", "item_idx", "source_rank", "source_score")
            .withColumn("source_confidence", F.lit(1.0))
            .withColumn("source", F.lit("popular"))
            .withColumn("source_detail", F.lit("popular"))
        )

        if ENABLE_PROFILE_RECALL:
            bucket_profile_calibration = profile_calibration_models.get(int(min_train))
            profile_cand, profile_meta = build_profile_candidates(
                spark=spark,
                train_idx=train_idx,
                test_users=test_users,
                user_map=user_map,
                item_map=item_map,
                profile_user_ids=profile_user_ids,
                profile_vectors=profile_vectors,
                profile_confidence_by_user_id=profile_confidence_by_user_id,
                profile_top_k=profile_top_k,
                profile_calibration_model=bucket_profile_calibration,
                user_tag_long_pdf=user_profile_tag_long_pdf,
                item_tag_long_pdf=item_semantic_tag_long_pdf,
                profile_route_policy=profile_route_policy_bucket,
                profile_threshold_policy=profile_threshold_policy_bucket,
                bucket_min_train=int(min_train),
                profile_multivector_data=profile_multivector_data,
                merchant_dense_vector_data=merchant_dense_vector_data,
            )
        else:
            profile_cand = _empty_source_df(spark, "profile")
            profile_meta = {"status": "disabled"}
        print(f"[INFO] bucket={min_train} profile_recall={profile_meta}")

        context_enabled = should_enable_context_surface(int(min_train))
        context_cand = _empty_source_df(spark, "context")
        context_meta: dict[str, Any] = {"status": "disabled", "route_rows": {}, "route_users": {}}
        if context_enabled:
            context_cand, context_meta = build_context_candidates(
                spark=spark,
                train_idx=train_idx,
                test_users=test_users,
                item_map=item_map,
                business_context_df=business_context_df,
                review_evidence_df=review_evidence_business_df,
                tip_signal_df=tip_signal_business_df,
                checkin_context_df=checkin_context_business_df,
                context_top_k=int(CONTEXT_SURFACE_TOP_K),
                context_score_min=float(CONTEXT_SURFACE_SCORE_MIN),
                batch_users=int(CONTEXT_SURFACE_BATCH_USERS),
            )
        print(f"[INFO] bucket={min_train} context_surface={context_meta}")

        candidates_all = (
            als_cand.unionByName(cluster_cand, allowMissingColumns=True)
            .unionByName(pop_cand, allowMissingColumns=True)
            .unionByName(profile_cand, allowMissingColumns=True)
            .unionByName(context_cand, allowMissingColumns=True)
            .persist(StorageLevel.DISK_ONLY)
        )
        source_weight_expr = build_source_weight_expr(int(min_train))
        log2_const = float(math.log(2.0))
        debias_rewrite_enabled = should_enable_bucket_debias_rewrite(int(min_train))
        review_volume_gate_expr = F.least(
            F.log(F.coalesce(F.col("review_count"), F.lit(0)) + F.lit(1.0)) / F.log(F.lit(501.0)),
            F.lit(1.0),
        )
        if debias_rewrite_enabled:
            quality_review_gate_expr = F.pow(
                review_volume_gate_expr,
                F.lit(float(DEBIAS_QUALITY_REVIEW_POWER)),
            )
            quality_score_expr = (F.coalesce(F.col("stars"), F.lit(0.0)) / F.lit(5.0)) * (
                F.lit(float(DEBIAS_QUALITY_BASE_FLOOR))
                + F.lit(1.0 - float(DEBIAS_QUALITY_BASE_FLOOR)) * quality_review_gate_expr
            )
        else:
            quality_score_expr = (F.coalesce(F.col("stars"), F.lit(0.0)) / F.lit(5.0)) * review_volume_gate_expr
        candidates_scored = (
            candidates_all.join(user_train_counts, on="user_idx", how="left")
            .join(biz_by_item, on="item_idx", how="left")
            .join(pop_stats, on="item_idx", how="left")
            .withColumn("source_confidence", F.coalesce(F.col("source_confidence"), F.lit(1.0)))
            .withColumn(
                "source_norm",
                F.lit(1.0) / (F.log(F.col("source_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
            )
            .withColumn("source_weight", source_weight_expr)
            .withColumn("signal_score", F.col("source_weight") * F.col("source_norm") * F.col("source_confidence"))
            .withColumn("quality_score", quality_score_expr)
        )

        fused_base = (
            candidates_scored.groupBy("user_idx", "item_idx")
            .agg(
                F.sum("signal_score").alias("signal_score"),
                F.max("quality_score").alias("quality_score"),
                F.max("item_train_pop_count").alias("item_train_pop_count"),
                F.first("business_id", ignorenulls=True).alias("business_id"),
                F.first("name", ignorenulls=True).alias("name"),
                F.first("city", ignorenulls=True).alias("city"),
                F.first("categories", ignorenulls=True).alias("categories"),
                F.first("primary_category", ignorenulls=True).alias("primary_category"),
                F.first("user_train_count", ignorenulls=True).alias("user_train_count"),
                F.first("user_segment", ignorenulls=True).alias("user_segment"),
                F.collect_set("source").alias("source_set"),
                F.collect_set("source_detail").alias("source_detail_set"),
                F.max(F.coalesce(F.col("profile_detail_count"), F.lit(0.0))).alias("profile_detail_count"),
                F.max(F.coalesce(F.col("profile_mv_route_count"), F.lit(0.0))).alias("profile_mv_route_count"),
                F.min(F.col("profile_mv_short_rank")).alias("profile_mv_short_rank"),
                F.min(F.col("profile_mv_long_rank")).alias("profile_mv_long_rank"),
                F.min(F.col("profile_mv_pos_rank")).alias("profile_mv_pos_rank"),
                F.min(F.when(F.col("source") == F.lit("als"), F.col("source_rank"))).alias("als_rank"),
                F.min(F.when(F.col("source") == F.lit("cluster"), F.col("source_rank"))).alias("cluster_rank"),
                F.min(F.when(F.col("source") == F.lit("profile"), F.col("source_rank"))).alias("profile_rank"),
                F.min(F.when(F.col("source") == F.lit("popular"), F.col("source_rank"))).alias("popular_rank"),
                F.min(F.when(F.col("source") == F.lit("context"), F.col("source_rank"))).alias("context_rank"),
            )
        )
        if item_semantic_df is not None:
            fused_base = (
                fused_base.join(item_semantic_df, on="business_id", how="left")
                .withColumn("semantic_score", F.coalesce(F.col("semantic_score"), F.lit(0.0)))
                .withColumn("semantic_confidence", F.coalesce(F.col("semantic_confidence"), F.lit(0.0)))
                .withColumn("semantic_support", F.coalesce(F.col("semantic_support"), F.lit(0.0)))
                .withColumn("semantic_tag_richness", F.coalesce(F.col("semantic_tag_richness"), F.lit(0.0)))
            )
        else:
            fused_base = (
                fused_base.withColumn("semantic_score", F.lit(0.0))
                .withColumn("semantic_confidence", F.lit(0.0))
                .withColumn("semantic_support", F.lit(0.0))
                .withColumn("semantic_tag_richness", F.lit(0.0))
            )
        fused_base = (
            fused_base.withColumn("source_count", F.coalesce(F.size(F.col("source_set")).cast("double"), F.lit(0.0)))
            .withColumn(
                "has_als",
                F.when(F.array_contains(F.col("source_set"), F.lit("als")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_cluster",
                F.when(F.array_contains(F.col("source_set"), F.lit("cluster")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_profile",
                F.when(F.array_contains(F.col("source_set"), F.lit("profile")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_popular",
                F.when(F.array_contains(F.col("source_set"), F.lit("popular")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_context",
                F.when(F.array_contains(F.col("source_set"), F.lit("context")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "nonpopular_source_count",
                F.col("has_als") + F.col("has_cluster") + F.col("has_profile") + F.col("has_context"),
            )
            .withColumn(
                "profile_cluster_source_count",
                F.col("has_cluster") + F.col("has_profile"),
            )
            .withColumn("profile_detail_count", F.coalesce(F.col("profile_detail_count"), F.lit(0.0)))
            .withColumn("profile_mv_route_count", F.coalesce(F.col("profile_mv_route_count"), F.lit(0.0)))
            .withColumn(
                "has_profile_mv_short",
                F.when(
                    F.array_contains(F.col("source_detail_set"), F.lit("profile_mv_short_semantic")),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_profile_mv_long",
                F.when(
                    F.array_contains(F.col("source_detail_set"), F.lit("profile_mv_long_core")),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_profile_mv_pos",
                F.when(
                    F.array_contains(F.col("source_detail_set"), F.lit("profile_mv_pos_pos")),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_context_attr",
                F.when(F.array_contains(F.col("source_detail_set"), F.lit("context_attr")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_context_time",
                F.when(F.array_contains(F.col("source_detail_set"), F.lit("context_time")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "has_context_geo",
                F.when(F.array_contains(F.col("source_detail_set"), F.lit("context_geo")), F.lit(1.0)).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "context_detail_count",
                F.col("has_context_attr") + F.col("has_context_time") + F.col("has_context_geo"),
            )
        )
        protected_lane_enabled = should_enable_profile_evidence_protected_lane(int(min_train))
        protected_lane_quota = max(0, int(PROFILE_EVIDENCE_PROTECTED_LANE_QUOTA)) if protected_lane_enabled else 0
        protected_lane_max_pre_rank = max(0, int(PROFILE_EVIDENCE_PROTECTED_LANE_MAX_PRE_RANK))
        protected_lane_max_pop_count = max(0, int(PROFILE_EVIDENCE_PROTECTED_LANE_MAX_POP_COUNT))
        protected_lane_min_mv_routes = max(0, int(PROFILE_EVIDENCE_PROTECTED_LANE_MIN_MV_ROUTES))
        protected_lane_min_semantic_score = float(PROFILE_EVIDENCE_PROTECTED_LANE_MIN_SEMANTIC_SCORE)
        protected_lane_min_profile_cluster = max(0, int(PROFILE_EVIDENCE_PROTECTED_LANE_MIN_PROFILE_CLUSTER))
        protected_lane_exclude_popular = bool(PROFILE_EVIDENCE_PROTECTED_LANE_EXCLUDE_POPULAR)
        als_backbone_topn_expr = build_segment_value_expr(ALS_BACKBONE_TOPN, default_value=0.0)
        item_pop_gate_expr = F.least(
            F.log(F.coalesce(F.col("item_train_pop_count"), F.lit(0.0)) + F.lit(1.0))
            / F.log(F.lit(float(DEBIAS_ITEM_POP_CAP) + 1.0)),
            F.lit(1.0),
        )
        item_pop_inv_expr = F.lit(1.0) / (
            F.log(F.coalesce(F.col("item_train_pop_count"), F.lit(0.0)) + F.lit(2.0)) / F.lit(log2_const)
        )
        als_rank_inv_expr = F.when(
            F.col("als_rank").isNotNull(),
            F.lit(1.0) / (F.log(F.col("als_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
        ).otherwise(F.lit(0.0))
        profile_rank_inv_expr = F.when(
            F.col("profile_rank").isNotNull(),
            F.lit(1.0) / (F.log(F.col("profile_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
        ).otherwise(F.lit(0.0))
        cluster_rank_inv_expr = F.when(
            F.col("cluster_rank").isNotNull(),
            F.lit(1.0) / (F.log(F.col("cluster_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
        ).otherwise(F.lit(0.0))
        profile_mv_short_rank_inv_expr = F.when(
            F.col("profile_mv_short_rank").isNotNull(),
            F.lit(1.0) / (F.log(F.col("profile_mv_short_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
        ).otherwise(F.lit(0.0))
        profile_mv_long_rank_inv_expr = F.when(
            F.col("profile_mv_long_rank").isNotNull(),
            F.lit(1.0) / (F.log(F.col("profile_mv_long_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
        ).otherwise(F.lit(0.0))
        profile_mv_pos_rank_inv_expr = F.when(
            F.col("profile_mv_pos_rank").isNotNull(),
            F.lit(1.0) / (F.log(F.col("profile_mv_pos_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const)),
        ).otherwise(F.lit(0.0))
        fused_base = fused_base.withColumn("als_backbone_topn", als_backbone_topn_expr)
        item_pop_log2_expr = (
            F.log(F.coalesce(F.col("item_train_pop_count"), F.lit(0.0)) + F.lit(2.0)) / F.lit(log2_const)
        )
        semantic_support_adj_expr = F.col("semantic_support").cast("double")
        if debias_rewrite_enabled:
            semantic_support_adj_expr = F.col("semantic_support").cast("double") / F.pow(
                F.greatest(item_pop_log2_expr, F.lit(1.0)),
                F.lit(float(DEBIAS_SEMANTIC_SUPPORT_POP_ALPHA)),
            )
        fused_base = fused_base.withColumn("semantic_support_adj", semantic_support_adj_expr)
        if ENABLE_SEMANTIC_GATES:
            semantic_support_cap_value = (
                float(DEBIAS_SEMANTIC_SUPPORT_ADJ_CAP) if debias_rewrite_enabled else float(SEMANTIC_SUPPORT_CAP)
            )
            semantic_support_gate_core_expr = F.least(
                F.log(F.col("semantic_support_adj").cast("double") + F.lit(1.0))
                / F.log(F.lit(float(semantic_support_cap_value) + 1.0)),
                F.lit(1.0),
            )
            if debias_rewrite_enabled:
                semantic_support_gate_expr = (
                    F.lit(float(DEBIAS_SEMANTIC_SUPPORT_GATE_FLOOR))
                    + F.lit(1.0 - float(DEBIAS_SEMANTIC_SUPPORT_GATE_FLOOR)) * semantic_support_gate_core_expr
                )
            else:
                semantic_support_gate_expr = semantic_support_gate_core_expr
            fused_base = (
                fused_base.withColumn("semantic_support_gate", semantic_support_gate_expr)
                .withColumn(
                    "semantic_richness_gate",
                    F.least(
                        F.col("semantic_tag_richness").cast("double") / F.lit(float(SEMANTIC_RICHNESS_CAP)),
                        F.lit(1.0),
                    ),
                )
                .withColumn(
                    "semantic_effective_score",
                    F.col("semantic_score")
                    * F.col("semantic_confidence")
                    * F.col("semantic_support_gate")
                    * (
                        F.lit(1.0 - float(SEMANTIC_RICHNESS_BLEND))
                        + F.lit(float(SEMANTIC_RICHNESS_BLEND)) * F.col("semantic_richness_gate")
                    ),
                )
            )
        else:
            fused_base = (
                fused_base.withColumn("semantic_support_gate", F.lit(1.0))
                .withColumn(
                    "semantic_effective_score",
                    F.col("semantic_score") * F.col("semantic_confidence"),
                )
            )
        head_source_count_weight = float(pretrim_head_policy.get("source_count_weight", 0.0))
        head_source_count_cap = max(0, int(pretrim_head_policy.get("source_count_cap", 0)))
        head_multi_source_min = max(1, int(pretrim_head_policy.get("multi_source_min", 2)))
        head_profile_rank_weight = float(pretrim_head_policy.get("profile_rank_weight", 0.0))
        head_profile_rank_cap = max(0, int(pretrim_head_policy.get("profile_rank_cap", 0)))
        head_cluster_rank_weight = float(pretrim_head_policy.get("cluster_rank_weight", 0.0))
        head_cluster_rank_cap = max(0, int(pretrim_head_policy.get("cluster_rank_cap", 0)))
        head_popular_only_penalty = float(pretrim_head_policy.get("popular_only_penalty", 0.0))
        multi_source_boost_expr = F.lit(0.0)
        if pretrim_head_enabled and abs(head_source_count_weight) > 1e-12 and head_source_count_cap > 0:
            multi_source_boost_expr = F.lit(float(head_source_count_weight)) * F.least(
                F.greatest(
                    F.col("source_count") - F.lit(float(head_multi_source_min) - 1.0),
                    F.lit(0.0),
                ),
                F.lit(float(head_source_count_cap)),
            )
        profile_rank_boost_expr = F.lit(0.0)
        if pretrim_head_enabled and abs(head_profile_rank_weight) > 1e-12 and head_profile_rank_cap > 0:
            profile_rank_boost_expr = F.when(
                F.col("profile_rank").isNotNull()
                & (F.col("profile_rank").cast("double") <= F.lit(float(head_profile_rank_cap))),
                F.lit(float(head_profile_rank_weight)) * profile_rank_inv_expr,
            ).otherwise(F.lit(0.0))
        cluster_rank_boost_expr = F.lit(0.0)
        if pretrim_head_enabled and abs(head_cluster_rank_weight) > 1e-12 and head_cluster_rank_cap > 0:
            cluster_rank_boost_expr = F.when(
                F.col("cluster_rank").isNotNull()
                & (F.col("cluster_rank").cast("double") <= F.lit(float(head_cluster_rank_cap))),
                F.lit(float(head_cluster_rank_weight)) * cluster_rank_inv_expr,
            ).otherwise(F.lit(0.0))
        popular_only_penalty_expr = F.lit(0.0)
        if pretrim_head_enabled and abs(head_popular_only_penalty) > 1e-12:
            popular_only_penalty_expr = F.when(
                (F.col("has_popular") > F.lit(0.5)) & (F.col("source_count") <= F.lit(1.0)),
                F.lit(float(head_popular_only_penalty)),
            ).otherwise(F.lit(0.0))
        profile_mv_evidence_expr = F.lit(0.0)
        if should_integrate_profile_multivector_routes(int(min_train)):
            profile_mv_evidence_expr = (
                F.lit(0.010) * F.least(F.col("profile_mv_route_count"), F.lit(3.0))
                + F.lit(0.010) * profile_mv_short_rank_inv_expr
                + F.lit(0.008) * profile_mv_pos_rank_inv_expr
                + F.lit(0.006) * profile_mv_long_rank_inv_expr
            )
        als_uncorroborated_expr = F.when(
            F.col("profile_cluster_source_count") <= F.lit(0.5),
            F.lit(1.0),
        ).otherwise(F.lit(0.0))
        user_heavy_expr = F.when(F.col("user_segment") == F.lit("heavy"), F.lit(1.0)).otherwise(F.lit(0.0))
        profile_cluster_support_expr = F.greatest(F.col("has_profile"), F.col("has_cluster"))
        als_backbone_mult_expr = F.lit(1.0)
        niche_semantic_bonus_expr = F.lit(0.0)
        niche_profile_cluster_bonus_expr = F.lit(0.0)
        if debias_rewrite_enabled:
            als_backbone_mult_expr = F.greatest(
                F.lit(float(DEBIAS_ALS_BACKBONE_MIN_MULT)),
                F.lit(1.0)
                - F.lit(float(DEBIAS_ALS_BACKBONE_POP_PENALTY)) * item_pop_gate_expr
                - F.lit(float(DEBIAS_ALS_BACKBONE_HEAVY_POP_PENALTY)) * item_pop_gate_expr * user_heavy_expr
                - F.lit(float(DEBIAS_ALS_BACKBONE_UNCORROBORATED_PENALTY))
                * item_pop_gate_expr
                * als_uncorroborated_expr,
            )
            niche_semantic_bonus_expr = (
                F.lit(float(DEBIAS_NICHE_SEMANTIC_BONUS_WEIGHT))
                * F.col("semantic_effective_score")
                * (F.lit(1.0) - item_pop_gate_expr)
                * (F.lit(0.5) + F.lit(0.5) * F.col("semantic_support_gate"))
                * profile_cluster_support_expr
            )
            niche_profile_cluster_bonus_expr = (
                F.lit(float(DEBIAS_NICHE_PROFILE_CLUSTER_BONUS))
                * (F.lit(1.0) - item_pop_gate_expr)
                * profile_cluster_support_expr
                * item_pop_inv_expr
            )
        profile_lane_score_expr = (
            F.col("profile_mv_evidence_score")
            + F.lit(0.45) * F.col("profile_rank_inv")
            + F.lit(0.18) * F.col("cluster_rank_inv")
            + F.lit(0.14) * F.col("semantic_effective_score")
            + F.lit(0.03) * F.col("nonpopular_source_count")
            + F.lit(0.08) * item_pop_inv_expr
        )
        fused = (
            fused_base.withColumn(
                "als_backbone_score",
                F.when(
                    F.col("als_rank").isNotNull() & (F.col("als_rank").cast("double") <= F.col("als_backbone_topn")),
                    F.lit(float(ALS_BACKBONE_WEIGHT)) * als_rank_inv_expr * als_backbone_mult_expr,
                ).otherwise(F.lit(0.0)),
            )
            .withColumn("als_rank_inv", als_rank_inv_expr)
            .withColumn("cluster_rank_inv", cluster_rank_inv_expr)
            .withColumn("profile_rank_inv", profile_rank_inv_expr)
            .withColumn("profile_mv_short_rank_inv", profile_mv_short_rank_inv_expr)
            .withColumn("profile_mv_long_rank_inv", profile_mv_long_rank_inv_expr)
            .withColumn("profile_mv_pos_rank_inv", profile_mv_pos_rank_inv_expr)
            .withColumn("profile_mv_evidence_score", profile_mv_evidence_expr)
            .withColumn("niche_semantic_bonus", niche_semantic_bonus_expr)
            .withColumn("niche_profile_cluster_bonus", niche_profile_cluster_bonus_expr)
            .withColumn(
                "pre_score",
                F.col("signal_score")
                + F.lit(float(QUALITY_WEIGHT)) * F.col("quality_score")
                + F.lit(float(SEMANTIC_WEIGHT)) * F.col("semantic_effective_score")
                + F.col("als_backbone_score")
                + F.col("profile_mv_evidence_score")
                + F.col("niche_semantic_bonus")
                + F.col("niche_profile_cluster_bonus"),
            )
            .withColumn("head_multisource_boost", multi_source_boost_expr)
            .withColumn("head_profile_boost", profile_rank_boost_expr)
            .withColumn("head_cluster_boost", cluster_rank_boost_expr)
            .withColumn("head_popular_penalty", popular_only_penalty_expr)
            .withColumn(
                "head_score_seed",
                F.col("pre_score")
                + F.col("head_multisource_boost")
                + F.col("head_profile_boost")
                + F.col("head_cluster_boost")
                - F.col("head_popular_penalty"),
            )
            .withColumn(
                "consensus_rescue_score",
                F.col("profile_rank_inv")
                + F.col("cluster_rank_inv")
                + F.lit(0.75) * F.col("als_rank_inv")
                + F.lit(0.03) * F.col("nonpopular_source_count")
                + F.lit(0.01) * F.col("source_count")
                + F.lit(0.02) * F.col("profile_mv_route_count")
                + F.lit(0.01) * F.col("has_profile_mv_short")
                + F.lit(0.01) * F.col("has_profile_mv_pos"),
            )
            .persist(StorageLevel.DISK_ONLY)
        )

        w_pre_base = Window.partitionBy("user_idx").orderBy(F.desc("pre_score"), F.asc("item_idx"))
        w_pre_seed = Window.partitionBy("user_idx").orderBy(
            F.desc("head_score_seed"),
            F.desc("pre_score"),
            F.asc("item_idx"),
        )
        fused_ranked = (
            fused.withColumn(
                "user_pretrim_top_k",
                F.when(F.col("user_segment") == F.lit("light"), F.lit(int(pretrim_segment_policy["light"])))
                .when(F.col("user_segment") == F.lit("mid"), F.lit(int(pretrim_segment_policy["mid"])))
                .when(F.col("user_segment") == F.lit("heavy"), F.lit(int(pretrim_segment_policy["heavy"])))
                .otherwise(F.lit(int(pretrim_segment_policy["unknown"]))),
            )
            .withColumn("pre_base_rank", F.row_number().over(w_pre_base))
            .withColumn("pre_head_seed_rank", F.row_number().over(w_pre_seed))
        )
        challenger_enabled = should_enable_personalized_challenger_surface(int(min_train))
        challenger_tower_seq_meta: dict[str, Any] = {"status": "disabled"}
        if challenger_enabled:
            challenger_window_topk = max(1, int(PERSONALIZED_CHALLENGER_HEAD_WINDOW))
            challenger_window_df = fused_ranked.filter(
                F.col("pre_head_seed_rank") <= F.lit(int(challenger_window_topk))
            ).select("user_idx", "item_idx", F.col("pre_head_seed_rank").alias("pre_rank"))
            challenger_tower_seq_df, challenger_tower_seq_meta = build_tower_seq_features_for_candidates(
                spark=spark,
                fused_pretrim=challenger_window_df,
                train_idx=train_idx,
                bucket=int(min_train),
            )
            if challenger_tower_seq_df is not None:
                fused_ranked = (
                    fused_ranked.join(
                        challenger_tower_seq_df.select(
                            "user_idx",
                            "item_idx",
                            F.col("tower_score").cast("double").alias("challenger_tower_score"),
                            F.col("seq_score").cast("double").alias("challenger_seq_score"),
                            F.col("tower_inv").cast("double").alias("challenger_tower_inv"),
                            F.col("seq_inv").cast("double").alias("challenger_seq_inv"),
                        ),
                        on=["user_idx", "item_idx"],
                        how="left",
                    )
                    .withColumn("challenger_tower_score", F.coalesce(F.col("challenger_tower_score"), F.lit(0.0)))
                    .withColumn("challenger_seq_score", F.coalesce(F.col("challenger_seq_score"), F.lit(0.0)))
                    .withColumn("challenger_tower_inv", F.coalesce(F.col("challenger_tower_inv"), F.lit(0.0)))
                    .withColumn("challenger_seq_inv", F.coalesce(F.col("challenger_seq_inv"), F.lit(0.0)))
                )
            else:
                fused_ranked = (
                    fused_ranked.withColumn("challenger_tower_score", F.lit(0.0))
                    .withColumn("challenger_seq_score", F.lit(0.0))
                    .withColumn("challenger_tower_inv", F.lit(0.0))
                    .withColumn("challenger_seq_inv", F.lit(0.0))
                )
            challenger_score_expr = (
                F.col("profile_mv_evidence_score")
                + F.lit(0.32) * F.col("profile_rank_inv")
                + F.lit(0.16) * F.col("cluster_rank_inv")
                + F.lit(0.16) * F.col("semantic_effective_score")
                + F.lit(0.10) * F.col("semantic_support_gate")
                + F.lit(0.16) * F.col("challenger_tower_inv")
                + F.lit(0.12) * F.col("challenger_seq_inv")
                + F.lit(0.04) * F.col("nonpopular_source_count")
                + F.lit(0.03) * F.col("profile_mv_route_count")
                + F.lit(0.05) * F.col("profile_cluster_source_count")
                + F.lit(0.10) * item_pop_inv_expr
                - F.lit(0.08) * item_pop_gate_expr
                - F.lit(0.04) * F.col("has_popular")
            )
            challenger_candidate_ok_expr = (
                (F.col("pre_head_seed_rank") <= F.lit(int(PERSONALIZED_CHALLENGER_MAX_PRE_RANK)))
                & (F.col("item_train_pop_count") <= F.lit(float(PERSONALIZED_CHALLENGER_MAX_POP_COUNT)))
                & (F.col("semantic_score") >= F.lit(float(PERSONALIZED_CHALLENGER_MIN_SEMANTIC_SCORE)))
                & (F.col("nonpopular_source_count") >= F.lit(float(PERSONALIZED_CHALLENGER_MIN_NONPOPULAR_SOURCES)))
                & (F.col("profile_cluster_source_count") >= F.lit(float(PERSONALIZED_CHALLENGER_MIN_PROFILE_CLUSTER)))
                & (
                    (F.col("profile_mv_route_count") >= F.lit(float(PERSONALIZED_CHALLENGER_MIN_MV_ROUTES)))
                    | F.col("profile_rank").isNotNull()
                )
            )
            if PERSONALIZED_CHALLENGER_EXCLUDE_POPULAR:
                challenger_candidate_ok_expr = challenger_candidate_ok_expr & (F.col("has_popular") < F.lit(0.5))
            fused_ranked = (
                fused_ranked.withColumn("challenger_candidate_ok", challenger_candidate_ok_expr)
                .withColumn(
                    "challenger_score",
                    F.when(F.col("challenger_candidate_ok"), challenger_score_expr).otherwise(F.lit(0.0)),
                )
            )
        else:
            fused_ranked = (
                fused_ranked.withColumn("challenger_tower_score", F.lit(0.0))
                .withColumn("challenger_seq_score", F.lit(0.0))
                .withColumn("challenger_tower_inv", F.lit(0.0))
                .withColumn("challenger_seq_inv", F.lit(0.0))
                .withColumn("challenger_candidate_ok", F.lit(False))
                .withColumn("challenger_score", F.lit(0.0))
            )
        w_pre = Window.partitionBy("user_idx").orderBy(
            F.desc("head_score"),
            F.desc("head_score_seed"),
            F.desc("pre_score"),
            F.asc("item_idx"),
        )
        fused_ranked = (
            fused_ranked.withColumn(
                "head_challenger_bonus",
                F.when(
                    F.col("challenger_candidate_ok"),
                    F.lit(float(PERSONALIZED_CHALLENGER_HEAD_BONUS)),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn("head_score", F.col("head_score_seed") + F.col("head_challenger_bonus"))
            .withColumn("pre_rank", F.row_number().over(w_pre))
            .withColumn("pre_rank_before_layered", F.col("pre_rank"))
        )
        layered_policy = get_layered_pretrim_policy(min_train)
        layered_pretrim_enabled = layered_policy is not None
        if layered_pretrim_enabled:
            front_guard_topk = int(layered_policy["front_guard_topk"])
            challenger_quota = max(0, int(PERSONALIZED_CHALLENGER_QUOTA)) if challenger_enabled else 0
            rescue_quota = (
                max(0, int(layered_policy.get("rescue_quota", 0))) if ENABLE_PRETRIM_CONSENSUS_RESCUE else 0
            )
            rescue_max_pre_rank = max(0, int(layered_policy.get("rescue_max_pre_rank", 0)))
            rescue_min_nonpopular_sources = max(0, int(layered_policy.get("rescue_min_nonpopular_sources", 0)))
            rescue_profile_cluster_min = max(0, int(layered_policy.get("rescue_profile_cluster_min", 0)))
            rescue_als_rank_max = max(0, int(layered_policy.get("rescue_als_rank_max", 0)))
            rescue_cluster_rank_max = max(0, int(layered_policy.get("rescue_cluster_rank_max", 0)))
            rescue_profile_rank_max = max(0, int(layered_policy.get("rescue_profile_rank_max", 0)))
            l1_quota = int(layered_policy["l1_quota"])
            l2_quota = int(layered_policy["l2_quota"])
            l3_quota = int(layered_policy["l3_quota"])

            rescue_enabled = (
                rescue_quota > 0
                and rescue_max_pre_rank > front_guard_topk
                and rescue_min_nonpopular_sources > 0
                and rescue_profile_cluster_min > 0
            )
            rescue_rank_ok = F.lit(True)
            if rescue_enabled:
                if rescue_als_rank_max > 0:
                    rescue_rank_ok = rescue_rank_ok & (
                        F.col("als_rank").isNull() | (F.col("als_rank") <= F.lit(int(rescue_als_rank_max)))
                    )
                if rescue_cluster_rank_max > 0:
                    rescue_rank_ok = rescue_rank_ok & (
                        F.col("cluster_rank").isNull()
                        | (F.col("cluster_rank") <= F.lit(int(rescue_cluster_rank_max)))
                    )
                if rescue_profile_rank_max > 0:
                    rescue_rank_ok = rescue_rank_ok & (
                        F.col("profile_rank").isNull()
                        | (F.col("profile_rank") <= F.lit(int(rescue_profile_rank_max)))
                    )
                is_rescue = (
                    (F.col("pre_rank") > F.lit(front_guard_topk))
                    & (F.col("pre_rank") <= F.lit(rescue_max_pre_rank))
                    & (F.col("nonpopular_source_count") >= F.lit(float(rescue_min_nonpopular_sources)))
                    & (F.col("profile_cluster_source_count") >= F.lit(float(rescue_profile_cluster_min)))
                    & rescue_rank_ok
                )
            else:
                is_rescue = F.lit(False)
            is_challenger = F.lit(False)
            if challenger_enabled and challenger_quota > 0 and PERSONALIZED_CHALLENGER_MAX_PRE_RANK > front_guard_topk:
                is_challenger = (
                    (F.col("pre_rank") > F.lit(front_guard_topk))
                    & F.col("challenger_candidate_ok")
                )
            is_l1 = (
                (F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(int(layered_policy["l1_als_rank_max"]))))
                | (F.col("cluster_rank").isNotNull() & (F.col("cluster_rank") <= F.lit(int(layered_policy["l1_cluster_rank_max"]))))
                | (F.col("profile_rank").isNotNull() & (F.col("profile_rank") <= F.lit(int(layered_policy["l1_profile_rank_max"]))))
            )
            is_l2 = (
                (F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(int(layered_policy["l2_als_rank_max"]))))
                | (F.col("cluster_rank").isNotNull() & (F.col("cluster_rank") <= F.lit(int(layered_policy["l2_cluster_rank_max"]))))
                | (F.col("profile_rank").isNotNull() & (F.col("profile_rank") <= F.lit(int(layered_policy["l2_profile_rank_max"]))))
            )
            is_profile_lane = F.lit(False)
            if protected_lane_enabled and protected_lane_quota > 0 and protected_lane_max_pre_rank > front_guard_topk:
                is_profile_lane = (
                    (F.col("pre_rank") > F.lit(front_guard_topk))
                    & (F.col("pre_rank") <= F.lit(int(protected_lane_max_pre_rank)))
                    & (F.col("profile_mv_route_count") >= F.lit(float(protected_lane_min_mv_routes)))
                    & (F.col("semantic_score") >= F.lit(float(protected_lane_min_semantic_score)))
                    & (F.col("item_train_pop_count") <= F.lit(float(protected_lane_max_pop_count)))
                    & (F.col("profile_cluster_source_count") >= F.lit(float(protected_lane_min_profile_cluster)))
                )
                if protected_lane_exclude_popular:
                    is_profile_lane = is_profile_lane & (F.col("has_popular") < F.lit(0.5))
            layer_tiered = (
                fused_ranked.withColumn(
                    "layer_tier",
                    F.when(is_challenger, F.lit(0))
                    .when(is_profile_lane, F.lit(1))
                    .when(is_rescue, F.lit(2))
                    .when(is_l1, F.lit(3))
                    .when(is_l2, F.lit(4))
                    .otherwise(F.lit(5)),
                )
                .withColumn(
                    "layer_sort_score",
                    F.when(F.col("layer_tier") == F.lit(0), F.col("challenger_score"))
                    .when(F.col("layer_tier") == F.lit(1), profile_lane_score_expr)
                    .when(F.col("layer_tier") == F.lit(2), F.col("consensus_rescue_score"))
                    .otherwise(F.lit(0.0)),
                )
                .persist(StorageLevel.DISK_ONLY)
            )
            w_tier = Window.partitionBy("user_idx", "layer_tier").orderBy(
                F.desc("layer_sort_score"),
                F.asc("pre_rank"),
                F.asc("item_idx"),
            )
            layer_ranked = (
                layer_tiered.withColumn("layer_rank", F.row_number().over(w_tier))
                .withColumn(
                    "pretrim_priority",
                    F.when(F.col("pre_rank") <= F.lit(front_guard_topk), F.lit(0))
                    .when(F.col("layer_tier") == F.lit(0), F.lit(1))
                    .when(F.col("layer_tier") == F.lit(1), F.lit(2))
                    .when(F.col("layer_tier") == F.lit(2), F.lit(3))
                    .when(F.col("layer_tier") == F.lit(3), F.lit(4))
                    .when(F.col("layer_tier") == F.lit(4), F.lit(5))
                    .otherwise(F.lit(6)),
                )
                .withColumn(
                    "pretrim_priority_rank",
                    F.when(F.col("pre_rank") <= F.lit(front_guard_topk), F.col("pre_rank")).otherwise(F.col("layer_rank")),
                )
            )
            layered_seed = (
                layer_ranked.filter(
                    (F.col("pre_rank") <= F.lit(front_guard_topk))
                    | ((F.col("layer_tier") == F.lit(0)) & (F.col("layer_rank") <= F.lit(challenger_quota)))
                    | ((F.col("layer_tier") == F.lit(1)) & (F.col("layer_rank") <= F.lit(protected_lane_quota)))
                    | ((F.col("layer_tier") == F.lit(2)) & (F.col("layer_rank") <= F.lit(rescue_quota)))
                    | ((F.col("layer_tier") == F.lit(3)) & (F.col("layer_rank") <= F.lit(l1_quota)))
                    | ((F.col("layer_tier") == F.lit(4)) & (F.col("layer_rank") <= F.lit(l2_quota)))
                    | ((F.col("layer_tier") == F.lit(5)) & (F.col("layer_rank") <= F.lit(l3_quota)))
                )
                .dropDuplicates(["user_idx", "item_idx"])
                .persist(StorageLevel.DISK_ONLY)
            )

            seed_counts = layered_seed.groupBy("user_idx").agg(F.count("*").alias("seed_cnt"))
            remainder = (
                layer_tiered.join(
                    layered_seed.select("user_idx", "item_idx"),
                    on=["user_idx", "item_idx"],
                    how="left_anti",
                )
                .persist(StorageLevel.DISK_ONLY)
            )
            w_remainder = Window.partitionBy("user_idx").orderBy(F.asc("pre_rank"), F.asc("item_idx"))
            remainder_ranked = (
                remainder.withColumn("remainder_rank", F.row_number().over(w_remainder))
                .join(seed_counts, on="user_idx", how="left")
                .fillna({"seed_cnt": 0})
                .withColumn("pretrim_priority", F.lit(5))
                .withColumn("pretrim_priority_rank", F.col("remainder_rank"))
                .withColumn(
                    "need_cnt",
                    F.greatest(F.col("user_pretrim_top_k").cast("int") - F.col("seed_cnt"), F.lit(0)),
                )
            )
            layered_fill = remainder_ranked.filter(F.col("remainder_rank") <= F.col("need_cnt")).drop(
                "remainder_rank", "seed_cnt", "need_cnt"
            )
            layered_pool = (
                layered_seed.unionByName(layered_fill, allowMissingColumns=True)
                .dropDuplicates(["user_idx", "item_idx"])
                .persist(StorageLevel.DISK_ONLY)
            )
            if ENABLE_LAYERED_PRETRIM_FINAL_RANK:
                w_layered_pre = Window.partitionBy("user_idx").orderBy(
                    F.asc("pretrim_priority"),
                    F.asc("pretrim_priority_rank"),
                    F.asc("pre_rank"),
                    F.asc("item_idx"),
                )
                fused_pretrim = (
                    layered_pool.withColumn("layered_pre_rank", F.row_number().over(w_layered_pre))
                    .filter(
                        (F.col("layered_pre_rank") <= F.col("user_pretrim_top_k").cast("int"))
                        | (F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(int(ALS_SAFETY_KEEP_TOPK))))
                    )
                    .withColumn("pre_rank", F.col("layered_pre_rank").cast("int"))
                    .drop(
                        "layer_tier",
                        "layer_sort_score",
                        "layer_rank",
                        "pretrim_priority",
                        "pretrim_priority_rank",
                        "layered_pre_rank",
                    )
                    .persist(StorageLevel.DISK_ONLY)
                )
            else:
                w_layered_pre = Window.partitionBy("user_idx").orderBy(F.asc("pre_rank"), F.asc("item_idx"))
                fused_pretrim = (
                    layered_pool.withColumn("layered_pre_rank", F.row_number().over(w_layered_pre))
                    .filter(
                        (F.col("layered_pre_rank") <= F.col("user_pretrim_top_k").cast("int"))
                        | (F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(int(ALS_SAFETY_KEEP_TOPK))))
                    )
                    .drop(
                        "layer_tier",
                        "layer_sort_score",
                        "layer_rank",
                        "pretrim_priority",
                        "pretrim_priority_rank",
                        "layered_pre_rank",
                    )
                    .persist(StorageLevel.DISK_ONLY)
                )
            layered_seed.unpersist()
            remainder.unpersist()
            layered_pool.unpersist()
            layer_tiered.unpersist()
        else:
            fused_pretrim = (
                fused_ranked.filter(
                    (F.col("pre_rank") <= F.col("user_pretrim_top_k").cast("int"))
                    | (F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(int(ALS_SAFETY_KEEP_TOPK))))
                )
                .persist(StorageLevel.DISK_ONLY)
            )

        tower_seq_df, tower_seq_meta = build_tower_seq_features_for_candidates(
            spark=spark,
            fused_pretrim=fused_pretrim,
            train_idx=train_idx,
            bucket=int(min_train),
        )
        if tower_seq_df is not None:
            fused_pretrim_aug = (
                fused_pretrim.join(tower_seq_df, on=["user_idx", "item_idx"], how="left")
                .withColumn("tower_score", F.coalesce(F.col("tower_score").cast("double"), F.lit(0.0)))
                .withColumn("seq_score", F.coalesce(F.col("seq_score").cast("double"), F.lit(0.0)))
                .withColumn("tower_inv", F.coalesce(F.col("tower_inv").cast("double"), F.lit(0.0)))
                .withColumn("seq_inv", F.coalesce(F.col("seq_inv").cast("double"), F.lit(0.0)))
                .persist(StorageLevel.DISK_ONLY)
            )
            fused_pretrim.unpersist()
            fused_pretrim = fused_pretrim_aug
        else:
            fused_pretrim_aug = (
                fused_pretrim.withColumn("tower_score", F.lit(0.0))
                .withColumn("seq_score", F.lit(0.0))
                .withColumn("tower_inv", F.lit(0.0))
                .withColumn("seq_inv", F.lit(0.0))
                .persist(StorageLevel.DISK_ONLY)
            )
            fused_pretrim.unpersist()
            fused_pretrim = fused_pretrim_aug
        print(f"[INFO] bucket={min_train} tower_seq_features={tower_seq_meta}")

        # Write bucket outputs early so recall audit can proceed even if optional
        # internal metrics fail on unstable Python workers.
        # Cut lineage before wide writes to reduce local worker timeout risk.
        candidates_all_out = _coalesce_for_write(_checkpoint_for_write(spark, candidates_all))
        fused_pretrim_out = _coalesce_for_write(_checkpoint_for_write(spark, fused_pretrim))
        truth_out = _coalesce_for_write(_checkpoint_for_write(spark, truth.join(user_map, on="user_idx", how="left")))
        train_history_out = _coalesce_for_write(_checkpoint_for_write(spark, train_history))
        _write_output_df(candidates_all_out, bucket_dir / "candidates_all.parquet")
        # `candidates_pretrim.parquet` is the canonical Stage09 handoff.
        # Keep `candidates_pretrim150.parquet` as a legacy compatibility alias
        # because older Stage10/11 runs and external notebooks still reference it.
        canonical_pretrim_path = bucket_dir / "candidates_pretrim.parquet"
        legacy_pretrim_alias_path = bucket_dir / "candidates_pretrim150.parquet"
        _write_output_df(fused_pretrim_out, canonical_pretrim_path)
        if WRITE_LEGACY_PRETRIM150_ALIAS:
            if LOCAL_PARQUET_WRITE_MODE == "driver_parquet":
                shutil.rmtree(legacy_pretrim_alias_path, ignore_errors=True)
                shutil.copytree(canonical_pretrim_path, legacy_pretrim_alias_path)
                print(f"[WRITE] mode=driver_parquet-copy path={legacy_pretrim_alias_path}")
            else:
                _write_output_df(fused_pretrim_out, legacy_pretrim_alias_path)
        _write_output_df(truth_out, bucket_dir / "truth.parquet")
        if WRITE_TRUTH_USER_ROSTER:
            truth_roster_cols = [c for c in ("user_id", "user_idx") if c in truth_out.columns]
            truth_roster_dedupe_cols = ["user_id"] if "user_id" in truth_roster_cols else truth_roster_cols
            (
                truth_out.select(*truth_roster_cols)
                .dropDuplicates(truth_roster_dedupe_cols)
                .coalesce(1)
                .write.mode("overwrite")
                .option("header", True)
                .csv((bucket_dir / "truth_user_roster.csv").as_posix())
            )
        _write_output_df(train_history_out, bucket_dir / "train_history.parquet")
        if should_write_enriched_audit(int(min_train)):
            source_evidence_out = _coalesce_for_write(_checkpoint_for_write(spark, build_source_evidence_df(candidates_scored)))
            enriched_audit_out = _coalesce_for_write(
                _checkpoint_for_write(
                    spark,
                    build_enriched_audit_df(
                        candidates_scored=candidates_scored,
                        fused=fused_ranked,
                        fused_pretrim=fused_pretrim,
                        als_top_k=als_top_k,
                        cluster_top_k=cluster_top_k,
                        popular_top_k=popular_top_k,
                        profile_top_k=profile_top_k,
                    ),
                )
            )
            _write_output_df(source_evidence_out, bucket_dir / "candidate_source_evidence.parquet")
            _write_output_df(enriched_audit_out, bucket_dir / "candidates_enriched_audit.parquet")
            for _tmp_df in (source_evidence_out, enriched_audit_out):
                try:
                    _tmp_df.unpersist(blocking=False)
                except Exception:
                    pass
        if should_write_profile_multivector_route_audit(int(min_train)):
            multivector_route_out = _coalesce_for_write(
                _checkpoint_for_write(
                    spark,
                    profile_multivector_route_audit_df.join(
                        biz_by_item.select("item_idx", "business_id", "name", "city", "categories"),
                        on="item_idx",
                        how="left",
                    ),
                )
            )
            _write_output_df(multivector_route_out, bucket_dir / "profile_multivector_route_audit.parquet")
            try:
                multivector_route_out.unpersist(blocking=False)
            except Exception:
                pass
        for _tmp_df in (candidates_all_out, fused_pretrim_out, truth_out, train_history_out):
            try:
                _tmp_df.unpersist(blocking=False)
            except Exception:
                pass
        tower_seq_tmp_csv = str(tower_seq_meta.get("tmp_csv", "")).strip()
        if tower_seq_tmp_csv:
            deferred_tmp_csv_cleanup.append(tower_seq_tmp_csv)
        profile_tmp_csv = str(profile_meta.get("tmp_csv", "")).strip()
        if profile_tmp_csv:
            deferred_tmp_csv_cleanup.append(profile_tmp_csv)

        # Optional quick metrics (for local sanity only). Primary recall quality
        # should be evaluated by scripts/09_1_recall_audit.py.
        als_recall = float("nan")
        als_ndcg = float("nan")
        profile_recall = float("nan")
        profile_ndcg = float("nan")
        fusion_recall = float("nan")
        fusion_ndcg = float("nan")
        if SKIP_INTERNAL_METRICS:
            print(f"[INFO] bucket={min_train} skip_internal_metrics=true (use 09_1_recall_audit)")
        else:
            try:
                als_eval = (
                    truth.join(
                        als_cand.filter(F.col("source_rank") <= F.lit(TOP_K_EVAL)).select(
                            "user_idx", F.col("item_idx").alias("pred_item"), F.col("source_rank").alias("rank")
                        ),
                        on="user_idx",
                        how="left",
                    )
                    .withColumn(
                        "rank",
                        F.when(F.col("pred_item") == F.col("true_item_idx"), F.col("rank")).otherwise(F.lit(0)),
                    )
                    .groupBy("user_idx")
                    .agg(F.max("rank").alias("rank"))
                )
                als_recall, als_ndcg = compute_recall_ndcg(als_eval)

                if ENABLE_PROFILE_RECALL:
                    profile_eval = (
                        truth.join(
                            profile_cand.filter(F.col("source_rank") <= F.lit(TOP_K_EVAL)).select(
                                "user_idx", F.col("item_idx").alias("pred_item"), F.col("source_rank").alias("rank")
                            ),
                            on="user_idx",
                            how="left",
                        )
                        .withColumn(
                            "rank",
                            F.when(F.col("pred_item") == F.col("true_item_idx"), F.col("rank")).otherwise(F.lit(0)),
                        )
                        .groupBy("user_idx")
                        .agg(F.max("rank").alias("rank"))
                    )
                    profile_recall, profile_ndcg = compute_recall_ndcg(profile_eval)

                fusion_eval = (
                    truth.join(
                        fused_pretrim.filter(F.col("pre_rank") <= F.lit(TOP_K_EVAL)).select(
                            "user_idx", F.col("item_idx").alias("pred_item"), F.col("pre_rank").alias("rank")
                        ),
                        on="user_idx",
                        how="left",
                    )
                    .withColumn(
                        "rank",
                        F.when(F.col("pred_item") == F.col("true_item_idx"), F.col("rank")).otherwise(F.lit(0)),
                    )
                    .groupBy("user_idx")
                    .agg(F.max("rank").alias("rank"))
                )
                fusion_recall, fusion_ndcg = compute_recall_ndcg(fusion_eval)
            except Exception as exc:
                print(f"[WARN] bucket={min_train} internal metrics failed: {exc}")

        def _metric_or_none(v: float) -> float | None:
            try:
                if math.isfinite(float(v)):
                    return round(float(v), 6)
            except Exception:
                return None
            return None

        seg_counts = user_train_counts.groupBy("user_segment").count().collect()
        seg_map = {r["user_segment"]: int(r["count"]) for r in seg_counts}
        bucket_row = {
            "bucket_min_train_reviews": min_train,
            "min_user_reviews": min_user,
            "n_users": n_users,
            "n_items": n_items,
            "n_train": n_train,
            "n_valid": n_valid,
            "n_test": n_test,
            "als_recall_at_10": _metric_or_none(als_recall),
            "als_ndcg_at_10": _metric_or_none(als_ndcg),
            "profile_recall_at_10": _metric_or_none(profile_recall),
            "profile_ndcg_at_10": _metric_or_none(profile_ndcg),
            "fusion_recall_at_10": _metric_or_none(fusion_recall),
            "fusion_ndcg_at_10": _metric_or_none(fusion_ndcg),
            "segment_light_users": int(seg_map.get("light", 0)),
            "segment_mid_users": int(seg_map.get("mid", 0)),
            "segment_heavy_users": int(seg_map.get("heavy", 0)),
            "scope_business_before_hard_filter": int(biz_stats["n_scope_before_hard_filter"]),
            "scope_business_after_hard_filter": int(biz_stats["n_scope_after_hard_filter"]),
            "scope_interaction_rows": int(scoped_interaction_rows),
            "scope_interaction_users": int(scoped_interaction_users),
            "total_train_events": total_train_events,
            "implicit_rank": int(params["rank"]),
            "implicit_reg": float(params["reg"]),
            "implicit_alpha": float(params["alpha"]),
            "als_top_k_used": int(als_top_k),
            "cluster_top_k_used": int(cluster_top_k),
            "popular_top_k_used": int(popular_top_k),
            "profile_top_k_used": int(profile_top_k),
            "pretrim_top_k_used": int(pretrim_top_k),
            "pretrim_segment_topk_policy_used": pretrim_segment_policy,
            "pretrim_output_max_top_k": int(pretrim_output_max_top_k),
            "pretrim_segment_variable": bool(pretrim_segment_variable),
            "pretrim_head_policy_used": pretrim_head_policy,
                "write_generic_pretrim_alias": True,
                "write_legacy_pretrim150_alias": bool(WRITE_LEGACY_PRETRIM150_ALIAS),
            "write_enriched_audit_export": bool(should_write_enriched_audit(int(min_train))),
            "cluster_user_topn_used": int(cluster_user_topn),
            "layered_pretrim_enabled": bool(layered_pretrim_enabled),
            "layered_pretrim_policy": layered_policy if layered_policy is not None else {},
            "pretrim_consensus_rescue_enabled": bool(ENABLE_PRETRIM_CONSENSUS_RESCUE),
            "layered_pretrim_final_rank_enabled": bool(ENABLE_LAYERED_PRETRIM_FINAL_RANK),
            "profile_recall_status": str(profile_meta.get("status", "")),
            "profile_route_policy_used": profile_route_policy_bucket,
            "profile_threshold_policy_used": profile_threshold_policy_bucket,
            "profile_recall_enabled_routes": profile_meta.get("enabled_routes", []),
            "profile_recall_rows_total": int(profile_meta.get("rows", 0)),
            "profile_recall_rows_vector": int(profile_meta.get("rows_vector", 0)),
            "profile_recall_rows_shared": int(profile_meta.get("rows_shared", 0)),
            "profile_recall_rows_bridge_user": int(profile_meta.get("rows_bridge_user", 0)),
            "profile_recall_rows_bridge_type": int(profile_meta.get("rows_bridge_type", 0)),
            "profile_recall_rows_multivector": int(profile_meta.get("rows_multivector", 0)),
            "profile_multivector_integration_enabled": bool(
                profile_meta.get("profile_multivector_integration_enabled", False)
            ),
            "profile_multivector_integration_status": str(
                profile_meta.get("profile_multivector_integration_status", "")
            ),
            "profile_multivector_integration_route_rows": profile_meta.get(
                "profile_multivector_integration_route_rows", {}
            ),
            "profile_multivector_integration_route_users": profile_meta.get(
                "profile_multivector_integration_route_users", {}
            ),
            "profile_evidence_protected_lane_enabled": bool(protected_lane_enabled),
            "personalized_challenger_enabled": bool(challenger_enabled),
            "personalized_challenger_quota": int(challenger_quota) if layered_pretrim_enabled else 0,
            "personalized_challenger_max_pre_rank": int(PERSONALIZED_CHALLENGER_MAX_PRE_RANK),
            "personalized_challenger_head_window": int(PERSONALIZED_CHALLENGER_HEAD_WINDOW),
            "personalized_challenger_head_bonus": float(PERSONALIZED_CHALLENGER_HEAD_BONUS),
            "personalized_challenger_tower_seq_status": str(challenger_tower_seq_meta.get("status", "")),
            "personalized_challenger_tower_seq_rows": int(challenger_tower_seq_meta.get("rows", 0)),
            "personalized_challenger_tower_seq_window_mode": str(
                challenger_tower_seq_meta.get("tower_seq_candidate_window_mode", "")
            ),
            "personalized_challenger_tower_seq_window_topk": int(
                challenger_tower_seq_meta.get("tower_seq_candidate_window_topk", 0)
            ),
            "bucket_debias_rewrite_enabled": bool(debias_rewrite_enabled),
            "profile_recall_users_with_vector": int(profile_meta.get("users_with_vector", 0)),
            "profile_recall_test_users_with_vector": int(profile_meta.get("test_users_with_vector", 0)),
            "profile_recall_test_users_with_confident_profile": int(
                profile_meta.get("test_users_with_confident_profile", 0)
            ),
            "profile_recall_items": int(profile_meta.get("items_with_profile_vector", 0)),
            "profile_multivector_route_audit_enabled": bool(should_write_profile_multivector_route_audit(int(min_train))),
            "profile_multivector_route_status": str(profile_multivector_route_meta.get("status", "")),
            "profile_multivector_route_rows": int(profile_multivector_route_meta.get("rows", 0)),
            "profile_multivector_route_route_rows": profile_multivector_route_meta.get("route_rows", {}),
            "profile_multivector_route_route_users": profile_multivector_route_meta.get("route_users", {}),
            "context_surface_enabled": bool(context_enabled),
            "context_surface_status": str(context_meta.get("status", "")),
            "context_surface_rows": int(context_meta.get("rows", 0)),
            "context_surface_top_k": int(context_meta.get("top_k", 0)),
            "context_surface_score_min": float(context_meta.get("score_min", 0.0)),
            "context_surface_route_rows": context_meta.get("route_rows", {}),
            "context_surface_route_users": context_meta.get("route_users", {}),
            "profile_calibration_applied": bool(profile_meta.get("profile_calibration_applied", False)),
            "profile_calibration_rows": int(profile_meta.get("profile_calibration_rows", 0)),
            "train_history_rows": int(train_history.count()),
            "item_semantic_enabled": bool(item_semantic_df is not None),
            "item_semantic_weight": float(SEMANTIC_WEIGHT),
            "source_weights_used": {
                seg: {
                    src: float(
                        SOURCE_WEIGHTS_BY_BUCKET.get(int(min_train), {}).get(seg, {}).get(src, SOURCE_WEIGHTS[seg][src])
                    )
                    for src in ("als", "cluster", "popular", "profile", "context")
                }
                for seg in ("light", "mid", "heavy")
            },
            "tower_seq_enabled": bool(tower_seq_meta.get("enabled", False)),
            "tower_seq_status": str(tower_seq_meta.get("status", "")),
            "tower_seq_rows": int(tower_seq_meta.get("rows", 0)),
            "tower_seq_n_users": int(tower_seq_meta.get("n_users", 0)),
            "tower_seq_n_items": int(tower_seq_meta.get("n_items", 0)),
            "tower_seq_updates": int(tower_seq_meta.get("updates", 0)),
            "tower_seq_seq_ready_rate": float(tower_seq_meta.get("seq_ready_rate", 0.0)),
            "tower_seq_max_cand_rows_effective": int(tower_seq_meta.get("tower_seq_max_cand_rows_effective", 0)),
            "tower_seq_candidate_window_mode": str(tower_seq_meta.get("tower_seq_candidate_window_mode", "")),
            "tower_seq_candidate_window_topk": int(tower_seq_meta.get("tower_seq_candidate_window_topk", 0)),
            "tower_seq_rows_total_before_window_probe": int(tower_seq_meta.get("tower_seq_rows_total_before_window_probe", 0)),
        }
        bucket_rows.append(bucket_row)
        write_json(bucket_dir / "bucket_meta.json", bucket_row)
        if bucket_row["als_ndcg_at_10"] is None or bucket_row["fusion_ndcg_at_10"] is None:
            print(f"[METRIC] bucket={min_train} internal metrics unavailable")
        else:
            print(
                f"[METRIC] bucket={min_train} ALS NDCG@10={als_ndcg:.4f} "
                f"Fusion NDCG@10={fusion_ndcg:.4f}"
            )

        # Cleanup bucket caches.
        for df in [
            full_df,
            train,
            valid,
            test,
            train_idx,
            valid_idx,
            test_idx,
            test_anchor,
            train_history,
            item_map,
            user_map,
            user_train_counts,
            pop_stats,
            biz_by_item,
            test_users,
            truth,
            cluster_item_map,
            user_top_cluster,
            profile_cand,
            profile_multivector_route_audit_df,
            candidates_all,
            fused,
            fused_pretrim,
        ]:
            df.unpersist()

    run_meta = {
        "run_id": run_id,
        "run_profile": RUN_PROFILE,
        "recall_profile": RECALL_PROFILE,
        "run_tag": RUN_TAG,
        "output_dir": str(out_dir),
        "cluster_profile_csv": str(profile_csv),
        "cluster_user_topn": int(CLUSTER_USER_TOPN),
        "cluster_conf_floor": float(CLUSTER_CONF_FLOOR),
        "user_profile_vectors": str(profile_vec_path) if profile_vec_path is not None else "",
        "user_profile_multivector_paths": profile_multivector_paths,
        "user_profile_table": str(profile_table_path) if profile_table_path is not None else "",
        "user_profile_tag_long": str(profile_tag_long_path) if profile_tag_long_path is not None else "",
        "profile_calibration_json": str(profile_calibration_path) if profile_calibration_path is not None else "",
        "item_semantic_features": str(item_semantic_path) if item_semantic_path is not None else "",
        "item_semantic_tag_long": str(item_semantic_tag_long_path) if item_semantic_tag_long_path is not None else "",
        "merchant_dense_vector_paths": merchant_dense_vector_paths,
        "item_semantic_rows": int(item_semantic_rows),
        "enable_item_semantic": bool(item_semantic_df is not None),
        "enable_profile_multivector_route_audit": bool(ENABLE_PROFILE_MULTIVECTOR_ROUTE_AUDIT),
        "profile_multivector_route_audit_buckets": sorted(int(x) for x in PROFILE_MULTIVECTOR_ROUTE_AUDIT_BUCKETS),
        "profile_multivector_route_scopes": list(PROFILE_MULTIVECTOR_ROUTE_SCOPES),
        "profile_multivector_route_top_k": int(PROFILE_MULTIVECTOR_ROUTE_TOP_K),
        "profile_multivector_route_score_min": float(PROFILE_MULTIVECTOR_ROUTE_SCORE_MIN),
        "profile_multivector_route_batch_users": int(PROFILE_MULTIVECTOR_ROUTE_BATCH_USERS),
        "enable_profile_multivector_route_integration": bool(ENABLE_PROFILE_MULTIVECTOR_ROUTE_INTEGRATION),
        "profile_multivector_route_integration_buckets": sorted(
            int(x) for x in PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BUCKETS
        ),
        "profile_multivector_route_integration_top_k": int(PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_TOP_K),
        "profile_multivector_route_integration_score_min": float(
            PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_SCORE_MIN
        ),
        "profile_multivector_route_integration_batch_users": int(
            PROFILE_MULTIVECTOR_ROUTE_INTEGRATION_BATCH_USERS
        ),
        "enable_profile_evidence_protected_lane": bool(ENABLE_PROFILE_EVIDENCE_PROTECTED_LANE),
        "profile_evidence_protected_lane_buckets": sorted(
            int(x) for x in PROFILE_EVIDENCE_PROTECTED_LANE_BUCKETS
        ),
        "profile_evidence_protected_lane_quota": int(PROFILE_EVIDENCE_PROTECTED_LANE_QUOTA),
        "profile_evidence_protected_lane_max_pre_rank": int(PROFILE_EVIDENCE_PROTECTED_LANE_MAX_PRE_RANK),
        "profile_evidence_protected_lane_max_pop_count": int(PROFILE_EVIDENCE_PROTECTED_LANE_MAX_POP_COUNT),
        "profile_evidence_protected_lane_min_mv_routes": int(PROFILE_EVIDENCE_PROTECTED_LANE_MIN_MV_ROUTES),
        "profile_evidence_protected_lane_min_semantic_score": float(PROFILE_EVIDENCE_PROTECTED_LANE_MIN_SEMANTIC_SCORE),
        "profile_evidence_protected_lane_min_profile_cluster": int(PROFILE_EVIDENCE_PROTECTED_LANE_MIN_PROFILE_CLUSTER),
        "profile_evidence_protected_lane_exclude_popular": bool(PROFILE_EVIDENCE_PROTECTED_LANE_EXCLUDE_POPULAR),
        "enable_personalized_challenger_surface": bool(ENABLE_PERSONALIZED_CHALLENGER_SURFACE),
        "personalized_challenger_buckets": sorted(int(x) for x in PERSONALIZED_CHALLENGER_BUCKETS),
        "personalized_challenger_quota": int(PERSONALIZED_CHALLENGER_QUOTA),
        "personalized_challenger_max_pre_rank": int(PERSONALIZED_CHALLENGER_MAX_PRE_RANK),
        "personalized_challenger_head_window": int(PERSONALIZED_CHALLENGER_HEAD_WINDOW),
        "personalized_challenger_head_bonus": float(PERSONALIZED_CHALLENGER_HEAD_BONUS),
        "personalized_challenger_max_pop_count": int(PERSONALIZED_CHALLENGER_MAX_POP_COUNT),
        "personalized_challenger_min_mv_routes": int(PERSONALIZED_CHALLENGER_MIN_MV_ROUTES),
        "personalized_challenger_min_profile_cluster": int(PERSONALIZED_CHALLENGER_MIN_PROFILE_CLUSTER),
        "personalized_challenger_min_nonpopular_sources": int(PERSONALIZED_CHALLENGER_MIN_NONPOPULAR_SOURCES),
        "personalized_challenger_min_semantic_score": float(PERSONALIZED_CHALLENGER_MIN_SEMANTIC_SCORE),
        "personalized_challenger_exclude_popular": bool(PERSONALIZED_CHALLENGER_EXCLUDE_POPULAR),
        "enable_bucket_debias_rewrite": bool(ENABLE_BUCKET_DEBIAS_REWRITE),
        "debias_buckets": sorted(int(x) for x in DEBIAS_BUCKETS),
        "debias_quality_review_power": float(DEBIAS_QUALITY_REVIEW_POWER),
        "debias_quality_base_floor": float(DEBIAS_QUALITY_BASE_FLOOR),
        "debias_item_pop_cap": float(DEBIAS_ITEM_POP_CAP),
        "debias_semantic_support_pop_alpha": float(DEBIAS_SEMANTIC_SUPPORT_POP_ALPHA),
        "debias_semantic_support_adj_cap": float(DEBIAS_SEMANTIC_SUPPORT_ADJ_CAP),
        "debias_semantic_support_gate_floor": float(DEBIAS_SEMANTIC_SUPPORT_GATE_FLOOR),
        "debias_als_backbone_min_mult": float(DEBIAS_ALS_BACKBONE_MIN_MULT),
        "debias_als_backbone_pop_penalty": float(DEBIAS_ALS_BACKBONE_POP_PENALTY),
        "debias_als_backbone_heavy_pop_penalty": float(DEBIAS_ALS_BACKBONE_HEAVY_POP_PENALTY),
        "debias_als_backbone_uncorroborated_penalty": float(DEBIAS_ALS_BACKBONE_UNCORROBORATED_PENALTY),
        "debias_niche_semantic_bonus_weight": float(DEBIAS_NICHE_SEMANTIC_BONUS_WEIGHT),
        "debias_niche_profile_cluster_bonus": float(DEBIAS_NICHE_PROFILE_CLUSTER_BONUS),
        "enable_profile_vector_route": bool(ENABLE_PROFILE_VECTOR_ROUTE),
        "enable_tag_shared_route": bool(ENABLE_TAG_SHARED_ROUTE),
        "enable_bridge_user_route": bool(ENABLE_BRIDGE_USER_ROUTE),
        "enable_bridge_type_route": bool(ENABLE_BRIDGE_TYPE_ROUTE),
        "profile_route_policy_by_bucket": PROFILE_ROUTE_POLICY_BY_BUCKET,
        "profile_route_thresholds_by_bucket": PROFILE_ROUTE_THRESHOLDS_BY_BUCKET,
        "profile_tag_shared_top_k": int(PROFILE_TAG_SHARED_TOP_K),
        "profile_bridge_user_top_k": int(PROFILE_BRIDGE_USER_TOP_K),
        "profile_bridge_type_top_k": int(PROFILE_BRIDGE_TYPE_TOP_K),
        "profile_bridge_user_neighbors": int(PROFILE_BRIDGE_USER_NEIGHBORS),
        "profile_bridge_user_min_sim": float(PROFILE_BRIDGE_USER_MIN_SIM),
        "profile_shared_score_min": float(PROFILE_SHARED_SCORE_MIN),
        "profile_bridge_score_min": float(PROFILE_BRIDGE_SCORE_MIN),
        "profile_confidence_min": float(PROFILE_CONFIDENCE_MIN),
        "profile_confidence_floor": float(PROFILE_CONFIDENCE_FLOOR),
        "profile_conf_v2_blend": float(PROFILE_CONF_V2_BLEND),
        "profile_min_sentences": int(PROFILE_MIN_SENTENCES),
        "profile_item_weight_conf_power": float(PROFILE_ITEM_WEIGHT_CONF_POWER),
        "profile_item_weight_rating_power": float(PROFILE_ITEM_WEIGHT_RATING_POWER),
        "profile_item_weight_activity_power": float(PROFILE_ITEM_WEIGHT_ACTIVITY_POWER),
        "profile_item_weight_min": float(PROFILE_ITEM_WEIGHT_MIN),
        "profile_item_agg_chunk_rows": int(PROFILE_ITEM_AGG_CHUNK_ROWS),
        "enable_profile_recall": bool(ENABLE_PROFILE_RECALL),
        "filter_policy": FILTER_POLICY,
        "als_top_k": ALS_TOP_K,
        "cluster_top_k": CLUSTER_TOP_K,
        "popular_top_k": POPULAR_TOP_K,
        "profile_top_k": PROFILE_TOP_K,
        "pretrim_top_k": PRETRIM_TOP_K,
        "recall_limits_by_bucket": RECALL_LIMITS_BY_BUCKET,
        "layered_pretrim_by_bucket": LAYERED_PRETRIM_BY_BUCKET,
        "pretrim_segment_topk_by_bucket": PRETRIM_SEGMENT_TOPK_BY_BUCKET,
        "source_weights_by_bucket": SOURCE_WEIGHTS_BY_BUCKET,
        "enable_pretrim_head_policy": bool(ENABLE_PRETRIM_HEAD_POLICY),
        "enable_pretrim_consensus_rescue": bool(ENABLE_PRETRIM_CONSENSUS_RESCUE),
        "enable_layered_pretrim_final_rank": bool(ENABLE_LAYERED_PRETRIM_FINAL_RANK),
        "pretrim_head_policy_by_bucket": PRETRIM_HEAD_POLICY_BY_BUCKET,
                "write_generic_pretrim_alias": True,
                "write_legacy_pretrim150_alias": bool(WRITE_LEGACY_PRETRIM150_ALIAS),
        "als_backbone_topn": ALS_BACKBONE_TOPN,
        "als_backbone_weight": ALS_BACKBONE_WEIGHT,
        "als_safety_keep_topk": ALS_SAFETY_KEEP_TOPK,
        "semantic_weight": float(SEMANTIC_WEIGHT),
        "enable_semantic_gates": bool(ENABLE_SEMANTIC_GATES),
        "semantic_support_cap": float(SEMANTIC_SUPPORT_CAP),
        "semantic_richness_cap": float(SEMANTIC_RICHNESS_CAP),
        "semantic_richness_blend": float(SEMANTIC_RICHNESS_BLEND),
        "enable_tower_seq_features": bool(ENABLE_TOWER_SEQ_FEATURES),
        "tower_seq_max_cand_rows": int(TOWER_SEQ_MAX_CAND_ROWS),
        "tower_seq_max_cand_rows_by_bucket": {str(k): int(v) for k, v in TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET.items()},
        "tower_seq_head_window_by_bucket": {str(k): int(v) for k, v in TOWER_SEQ_HEAD_WINDOW_BY_BUCKET.items()},
        "tower_seq_output_mode": str(TOWER_SEQ_OUTPUT_MODE),
        "profile_output_mode": str(PROFILE_OUTPUT_MODE),
        "profile_output_spark_df_max_rows": int(PROFILE_OUTPUT_SPARK_DF_MAX_ROWS),
        "tower_seq_dim": int(TOWER_SEQ_DIM),
        "tower_seq_epochs": int(TOWER_SEQ_EPOCHS),
        "tower_seq_lr": float(TOWER_SEQ_LR),
        "tower_seq_reg": float(TOWER_SEQ_REG),
        "tower_seq_neg_tries": int(TOWER_SEQ_NEG_TRIES),
        "tower_seq_steps_per_epoch": int(TOWER_SEQ_STEPS_PER_EPOCH),
        "tower_seq_scope_users_only": bool(TOWER_SEQ_SCOPE_USERS_ONLY),
        "tower_seq_scope_items_only": bool(TOWER_SEQ_SCOPE_ITEMS_ONLY),
        "tower_seq_train_max_per_user": int(TOWER_SEQ_TRAIN_MAX_PER_USER),
        "profile_pandas_scope_only_relevant_users": bool(PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_USERS),
        "profile_pandas_scope_only_relevant_items": bool(PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_ITEMS),
        "seq_recent_len": int(SEQ_RECENT_LEN),
        "seq_recent_min_len": int(SEQ_RECENT_MIN_LEN),
        "seq_decay": float(SEQ_DECAY),
        "spark_master": str(SPARK_MASTER),
        "spark_driver_memory": str(SPARK_DRIVER_MEMORY),
        "spark_executor_memory": str(SPARK_EXECUTOR_MEMORY),
        "spark_sql_shuffle_partitions": str(SPARK_SQL_SHUFFLE_PARTITIONS),
        "spark_default_parallelism": str(SPARK_DEFAULT_PARALLELISM),
        "spark_sql_adaptive_enabled": bool(SPARK_SQL_ADAPTIVE_ENABLED),
        "spark_sql_parquet_enable_vectorized": bool(SPARK_SQL_PARQUET_ENABLE_VECTORIZED),
        "spark_sql_files_max_partition_bytes": str(SPARK_SQL_FILES_MAX_PARTITION_BYTES),
        "spark_sql_max_plan_string_length": str(SPARK_SQL_MAX_PLAN_STRING_LENGTH),
        "spark_network_timeout": str(SPARK_NETWORK_TIMEOUT),
        "spark_executor_heartbeat_interval": str(SPARK_EXECUTOR_HEARTBEAT_INTERVAL),
        "spark_python_worker_reuse": bool(SPARK_PYTHON_WORKER_REUSE),
        "spark_tmp_session_isolation": bool(SPARK_TMP_SESSION_ISOLATION),
        "spark_tmp_autoclean_enabled": bool(SPARK_TMP_AUTOCLEAN_ENABLED),
        "spark_tmp_clean_on_exit": bool(SPARK_TMP_CLEAN_ON_EXIT),
        "spark_tmp_retention_hours": int(SPARK_TMP_RETENTION_HOURS),
        "spark_tmp_clean_max_entries": int(SPARK_TMP_CLEAN_MAX_ENTRIES),
        "spark_tmp_context": (
            {
                "base_dir": str(_SPARK_TMP_CTX.base_dir),
                "spark_local_dir": str(_SPARK_TMP_CTX.spark_local_dir),
                "py_temp_dir": str(_SPARK_TMP_CTX.py_temp_dir),
                "scratch_root": str(_SPARK_TMP_CTX.scratch_root),
                "cleanup_summary": dict(_SPARK_TMP_CTX.cleanup_summary),
            }
            if _SPARK_TMP_CTX is not None
            else {}
        ),
        "enable_bucket_write_checkpoint": bool(ENABLE_BUCKET_WRITE_CHECKPOINT),
        "bucket_write_checkpoint_dir": str(BUCKET_WRITE_CHECKPOINT_DIR),
        "skip_checkpoint_on_python_plan": bool(SKIP_CHECKPOINT_ON_PYTHON_PLAN),
        "output_coalesce_partitions": int(OUTPUT_COALESCE_PARTITIONS),
        "skip_internal_metrics": bool(SKIP_INTERNAL_METRICS),
        "buckets": bucket_rows,
    }
    write_json(out_dir / "run_meta.json", run_meta)
    print(f"[INFO] wrote {out_dir}")

    if item_semantic_df is not None:
        item_semantic_df.unpersist()
    rvw.unpersist()
    biz.unpersist()
    cluster_map_raw.unpersist()
    spark.stop()
    for tmp_csv in sorted({str(p).strip() for p in deferred_tmp_csv_cleanup if str(p).strip()}):
        try:
            Path(tmp_csv).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()

