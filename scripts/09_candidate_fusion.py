from __future__ import annotations

import json
import math
import os
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
from pipeline.spark_tmp_manager import (
    SparkTmpContext,
    alloc_scratch_file,
    build_spark_tmp_context,
)


# Run mode
RUN_PROFILE = os.getenv("RUN_PROFILE_OVERRIDE", "full").strip().lower() or "full"  # "sample" | "full"
RUN_TAG = "stage09_candidate_fusion"
RECALL_PROFILE = os.getenv("RECALL_PROFILE_OVERRIDE", "balanced").strip().lower() or "balanced"

# Data paths
PARQUET_BASE = Path(r"D:/5006 BDA project/data/parquet")
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/09_candidate_fusion")
CLUSTER_PROFILE_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels/full")
CLUSTER_PROFILE_DIR_SUFFIX = "_full_profile_merged"
CLUSTER_PROFILE_FILENAME = "biz_profile_recsys.csv"
CLUSTER_PROFILE_CLUSTER_COL = "cluster_for_recsys"
CLUSTER_PROFILE_CSV = ""  # optional explicit path
USER_PROFILE_ROOT = Path(r"D:/5006 BDA project/data/output/09_user_profiles")
USER_PROFILE_RUN_DIR = os.getenv("USER_PROFILE_RUN_DIR", "").strip()  # optional explicit path
USER_PROFILE_RUN_SUFFIX = "_stage09_user_profile_build"
USER_PROFILE_VECTOR_FILE = "user_profile_vectors.npz"
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
ITEM_SEMANTIC_ROOT = Path(r"D:/5006 BDA project/data/output/09_item_semantics")
ITEM_SEMANTIC_RUN_DIR = os.getenv("ITEM_SEMANTIC_RUN_DIR", "").strip()  # optional explicit path
ITEM_SEMANTIC_RUN_SUFFIX = "_stage09_item_semantic_build"
ITEM_SEMANTIC_FEATURE_FILE = "item_semantic_features.csv"
ITEM_SEMANTIC_TAG_LONG_FILE = "item_tag_profile_long.csv"
ENABLE_ITEM_SEMANTIC = os.getenv("ENABLE_ITEM_SEMANTIC", "true").strip().lower() == "true"
ENABLE_TOWER_SEQ_FEATURES = os.getenv("ENABLE_TOWER_SEQ_FEATURES", "true").strip().lower() == "true"
TOWER_SEQ_MAX_CAND_ROWS = int(os.getenv("TOWER_SEQ_MAX_CAND_ROWS", "2500000").strip() or 2500000)
TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET_JSON = os.getenv("TOWER_SEQ_MAX_CAND_ROWS_BY_BUCKET_JSON", "").strip()
TOWER_SEQ_DIM = int(os.getenv("TOWER_SEQ_DIM", "64").strip() or 64)
TOWER_SEQ_EPOCHS = int(os.getenv("TOWER_SEQ_EPOCHS", "6").strip() or 6)
TOWER_SEQ_LR = float(os.getenv("TOWER_SEQ_LR", "0.04").strip() or 0.04)
TOWER_SEQ_REG = float(os.getenv("TOWER_SEQ_REG", "0.0001").strip() or 0.0001)
TOWER_SEQ_NEG_TRIES = int(os.getenv("TOWER_SEQ_NEG_TRIES", "6").strip() or 6)
TOWER_SEQ_STEPS_PER_EPOCH = int(os.getenv("TOWER_SEQ_STEPS_PER_EPOCH", "0").strip() or 0)
TOWER_SEQ_OUTPUT_MODE = (os.getenv("TOWER_SEQ_OUTPUT_MODE", "csv").strip().lower() or "csv")
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
PROFILE_ROUTE_POLICY_DEFAULT_BY_BUCKET: dict[int, dict[str, Any]] = {}
PROFILE_ROUTE_THRESHOLDS_DEFAULT_BY_BUCKET: dict[int, dict[str, Any]] = {}
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
            "front_guard_topk": 160,
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
    "light": {"als": 0.45, "cluster": 0.25, "popular": 0.15, "profile": 0.15},
    "mid": {"als": 0.70, "cluster": 0.15, "popular": 0.08, "profile": 0.07},
    "heavy": {"als": 0.85, "cluster": 0.08, "popular": 0.03, "profile": 0.04},
}
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
ALS_BACKBONE_TOPN = {"light": 3, "mid": 5, "heavy": 8}
ALS_BACKBONE_WEIGHT = 0.35
ALS_SAFETY_KEEP_TOPK = 20
PROFILE_ROUTE_POLICY_BY_BUCKET = dict(PROFILE_ROUTE_POLICY_DEFAULT_BY_BUCKET)
PROFILE_ROUTE_THRESHOLDS_BY_BUCKET = dict(PROFILE_ROUTE_THRESHOLDS_DEFAULT_BY_BUCKET)
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[4]").strip() or "local[4]"
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", "D:/5006 BDA project/data/spark-tmp").strip() or "D:/5006 BDA project/data/spark-tmp"
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_SQL_ADAPTIVE_ENABLED = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "true").strip().lower() == "true"
SPARK_SQL_MAX_PLAN_STRING_LENGTH = (
    os.getenv("SPARK_SQL_MAX_PLAN_STRING_LENGTH", "8192").strip() or "8192"
)
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = (
    os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
)
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
    os.getenv("BUCKET_WRITE_CHECKPOINT_DIR", "D:/5006 BDA project/data/spark-tmp/stage09-checkpoint").strip()
    or "D:/5006 BDA project/data/spark-tmp/stage09-checkpoint"
)
SKIP_CHECKPOINT_ON_PYTHON_PLAN = os.getenv("SKIP_CHECKPOINT_ON_PYTHON_PLAN", "true").strip().lower() == "true"
OUTPUT_COALESCE_PARTITIONS = int(os.getenv("OUTPUT_COALESCE_PARTITIONS", "8").strip() or 8)
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
    global PROFILE_ROUTE_POLICY_BY_BUCKET
    global PROFILE_ROUTE_THRESHOLDS_BY_BUCKET

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
    PROFILE_ROUTE_POLICY_BY_BUCKET = _merge_bucket_any_policy(
        PROFILE_ROUTE_POLICY_BY_BUCKET, PROFILE_ROUTE_POLICY_JSON
    )
    PROFILE_ROUTE_THRESHOLDS_BY_BUCKET = _merge_bucket_any_policy(
        PROFILE_ROUTE_THRESHOLDS_BY_BUCKET, PROFILE_ROUTE_THRESHOLDS_JSON
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
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.sql.adaptive.enabled", "true" if SPARK_SQL_ADAPTIVE_ENABLED else "false")
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


def load_profile_vectors(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path.as_posix(), allow_pickle=False)
    user_ids = data["user_ids"].astype(str)
    vectors = data["vectors"].astype(np.float32)
    if vectors.ndim != 2:
        raise RuntimeError(f"invalid profile vector shape: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = (vectors / norms).astype(np.float32)
    return user_ids, vectors


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
        )
        .limit(0)
    )


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
    calibration_applied_rows = 0
    enabled_routes: list[str] = []
    item_ids_final = np.array([], dtype=np.int32)

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
                                }
                            )
                        )
                        route_rows["vector"] += int(uid_col.shape[0])

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
                    rows.append((int(uid), int(iid), int(rnk), float(sc), float(src_conf), "profile"))
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
                    rows.append((int(uid), int(iid), int(rnk), float(sc), float(src_conf), "profile"))
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
                    rows.append((int(uid), int(item_ids[int(pos)]), int(rank_j), float(sc), float(src_conf), "profile"))
                    route_rows["bridge_type"] += 1

    if rows:
        profile_parts.append(
            pd.DataFrame(
                rows,
                columns=["user_idx", "item_idx", "source_rank", "source_score", "source_confidence", "source"],
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
    profile_pdf = profile_pdf.sort_values(
        ["user_idx", "item_idx", "source_score", "source_confidence", "source_rank"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).drop_duplicates(subset=["user_idx", "item_idx"], keep="first")
    profile_pdf = profile_pdf.sort_values(
        ["user_idx", "source_score", "source_confidence", "item_idx"],
        ascending=[True, False, False, True],
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
        "profile_scope_users_only_relevant": bool(PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_USERS),
        "profile_scope_items_only_relevant": bool(PROFILE_PANDAS_SCOPE_ONLY_RELEVANT_ITEMS),
        "user_map_rows_scoped": int(user_pdf.shape[0]),
        "tmp_csv": str(tmp_csv) if tmp_csv is not None else "",
        "profile_output_mode": str(PROFILE_OUTPUT_MODE),
        "profile_output_mode_effective": str(output_mode_effective),
    }
    return profile_df, meta


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


def build_source_weight_expr() -> Any:
    expr = F.lit(0.0)
    source_order = ["als", "cluster", "popular", "profile"]
    for seg in ("light", "mid", "heavy"):
        for src in source_order:
            expr = F.when(
                (F.col("user_segment") == F.lit(seg)) & (F.col("source") == F.lit(src)),
                F.lit(float(SOURCE_WEIGHTS[seg][src])),
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
        "n_users": 0,
        "n_items": 0,
        "updates": 0,
        "seq_ready_rate": 0.0,
        "tower_seq_scope_users_only": bool(TOWER_SEQ_SCOPE_USERS_ONLY),
        "tower_seq_scope_items_only": bool(TOWER_SEQ_SCOPE_ITEMS_ONLY),
        "tower_seq_train_max_per_user": int(TOWER_SEQ_TRAIN_MAX_PER_USER),
    }
    if not ENABLE_TOWER_SEQ_FEATURES:
        return None, meta

    # Avoid full count action: only probe up to cap+1.
    cap = int(get_tower_seq_max_cand_rows(bucket))
    meta["tower_seq_max_cand_rows_effective"] = int(cap)
    cand_rows = int(fused_pretrim.limit(cap + 1).count())
    meta["rows"] = int(cand_rows)
    if cand_rows <= 0:
        meta["status"] = "empty_candidates"
        return None, meta
    if cand_rows > cap:
        meta["status"] = "skip_row_cap"
        return None, meta

    cand_base = fused_pretrim.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"])
    cand_pdf = cand_base.toPandas()
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
    train_pdf = train_src.toPandas()
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

    print("[STEP] load business pool with hard filters")
    biz, biz_stats = load_business_pool(spark)
    print(f"[COUNT] business_before={biz_stats['n_scope_before_hard_filter']} after={biz_stats['n_scope_after_hard_filter']}")
    print(f"[CONFIG] stale_cutoff={biz_stats['stale_cutoff_date']}")

    print("[STEP] load interactions")
    rvw = load_interactions(spark, biz)
    print(f"[COUNT] interactions={rvw.count()} users={rvw.select('user_id').distinct().count()}")

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
    else:
        print("[STEP] profile recall disabled by ENABLE_PROFILE_RECALL=false")

    item_semantic_path = None
    item_semantic_tag_long_path = None
    item_semantic_df = None
    item_semantic_rows = 0
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
        except Exception as e:
            print(f"[WARN] item semantic disabled due to load failure: {e}")
            item_semantic_df = None
            item_semantic_tag_long_pdf = pd.DataFrame(
                columns=["business_id", "tag", "tag_type", "net_weight_sum", "tag_confidence", "support_count"]
            )
    else:
        print("[STEP] item semantic disabled by ENABLE_ITEM_SEMANTIC=false")

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
        print(
            "[RECALL] "
            f"als_top_k={als_top_k} cluster_top_k={cluster_top_k} "
            f"popular_top_k={popular_top_k} profile_top_k={profile_top_k} "
            f"pretrim_top_k={pretrim_top_k} cluster_user_topn={cluster_user_topn}"
        )
        print(f"[PRETRIM_SEG] bucket={min_train} policy={pretrim_segment_policy}")
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
            )
        else:
            profile_cand = _empty_source_df(spark, "profile")
            profile_meta = {"status": "disabled"}
        print(f"[INFO] bucket={min_train} profile_recall={profile_meta}")

        candidates_all = (
            als_cand.unionByName(cluster_cand)
            .unionByName(pop_cand)
            .unionByName(profile_cand)
            .persist(StorageLevel.DISK_ONLY)
        )
        source_weight_expr = build_source_weight_expr()
        log2_const = float(math.log(2.0))
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
            .withColumn(
                "quality_score",
                (F.coalesce(F.col("stars"), F.lit(0.0)) / F.lit(5.0))
                * F.least(
                    F.log(F.coalesce(F.col("review_count"), F.lit(0)) + F.lit(1.0))
                    / F.log(F.lit(501.0)),
                    F.lit(1.0),
                ),
            )
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
                F.min(F.when(F.col("source") == F.lit("als"), F.col("source_rank"))).alias("als_rank"),
                F.min(F.when(F.col("source") == F.lit("cluster"), F.col("source_rank"))).alias("cluster_rank"),
                F.min(F.when(F.col("source") == F.lit("profile"), F.col("source_rank"))).alias("profile_rank"),
                F.min(F.when(F.col("source") == F.lit("popular"), F.col("source_rank"))).alias("popular_rank"),
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
        als_backbone_topn_expr = build_segment_value_expr(ALS_BACKBONE_TOPN, default_value=0.0)
        fused_base = fused_base.withColumn("als_backbone_topn", als_backbone_topn_expr)
        if ENABLE_SEMANTIC_GATES:
            fused_base = (
                fused_base.withColumn(
                    "semantic_support_gate",
                    F.least(
                        F.log(F.col("semantic_support").cast("double") + F.lit(1.0))
                        / F.log(F.lit(float(SEMANTIC_SUPPORT_CAP) + 1.0)),
                        F.lit(1.0),
                    ),
                )
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
            fused_base = fused_base.withColumn(
                "semantic_effective_score",
                F.col("semantic_score") * F.col("semantic_confidence"),
            )
        fused = (
            fused_base.withColumn(
                "als_backbone_score",
                F.when(
                    F.col("als_rank").isNotNull() & (F.col("als_rank").cast("double") <= F.col("als_backbone_topn")),
                    F.lit(float(ALS_BACKBONE_WEIGHT))
                    * (F.lit(1.0) / (F.log(F.col("als_rank").cast("double") + F.lit(1.0)) / F.lit(log2_const))),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "pre_score",
                F.col("signal_score")
                + F.lit(float(QUALITY_WEIGHT)) * F.col("quality_score")
                + F.lit(float(SEMANTIC_WEIGHT)) * F.col("semantic_effective_score")
                + F.col("als_backbone_score"),
            )
            .persist(StorageLevel.DISK_ONLY)
        )

        w_pre = Window.partitionBy("user_idx").orderBy(F.desc("pre_score"), F.asc("item_idx"))
        fused_ranked = (
            fused.withColumn(
                "user_pretrim_top_k",
                F.when(F.col("user_segment") == F.lit("light"), F.lit(int(pretrim_segment_policy["light"])))
                .when(F.col("user_segment") == F.lit("mid"), F.lit(int(pretrim_segment_policy["mid"])))
                .when(F.col("user_segment") == F.lit("heavy"), F.lit(int(pretrim_segment_policy["heavy"])))
                .otherwise(F.lit(int(pretrim_segment_policy["unknown"]))),
            )
            .withColumn("pre_rank", F.row_number().over(w_pre))
        )
        layered_policy = get_layered_pretrim_policy(min_train)
        layered_pretrim_enabled = layered_policy is not None
        if layered_pretrim_enabled:
            front_guard_topk = int(layered_policy["front_guard_topk"])
            l1_quota = int(layered_policy["l1_quota"])
            l2_quota = int(layered_policy["l2_quota"])
            l3_quota = int(layered_policy["l3_quota"])

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
            layer_tiered = (
                fused_ranked.withColumn(
                    "layer_tier",
                    F.when(is_l1, F.lit(1)).when(is_l2, F.lit(2)).otherwise(F.lit(3)),
                )
                .persist(StorageLevel.DISK_ONLY)
            )
            w_tier = Window.partitionBy("user_idx", "layer_tier").orderBy(F.asc("pre_rank"), F.asc("item_idx"))
            layer_ranked = layer_tiered.withColumn("layer_rank", F.row_number().over(w_tier))
            layered_seed = (
                layer_ranked.filter(
                    (F.col("pre_rank") <= F.lit(front_guard_topk))
                    | ((F.col("layer_tier") == F.lit(1)) & (F.col("layer_rank") <= F.lit(l1_quota)))
                    | ((F.col("layer_tier") == F.lit(2)) & (F.col("layer_rank") <= F.lit(l2_quota)))
                    | ((F.col("layer_tier") == F.lit(3)) & (F.col("layer_rank") <= F.lit(l3_quota)))
                )
                .drop("layer_rank")
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
            w_layered_pre = Window.partitionBy("user_idx").orderBy(F.asc("pre_rank"), F.asc("item_idx"))
            fused_pretrim = (
                layered_pool.withColumn("layered_pre_rank", F.row_number().over(w_layered_pre))
                .filter(
                    (F.col("layered_pre_rank") <= F.col("user_pretrim_top_k").cast("int"))
                    | (F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(int(ALS_SAFETY_KEEP_TOPK))))
                )
                .drop("layer_tier", "layered_pre_rank")
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
        candidates_all_out.write.mode("overwrite").parquet((bucket_dir / "candidates_all.parquet").as_posix())
        fused_pretrim_out.write.mode("overwrite").parquet((bucket_dir / "candidates_pretrim150.parquet").as_posix())
        truth_out.write.mode("overwrite").parquet((bucket_dir / "truth.parquet").as_posix())
        train_history_out.write.mode("overwrite").parquet((bucket_dir / "train_history.parquet").as_posix())
        for _tmp_df in (candidates_all_out, fused_pretrim_out, truth_out, train_history_out):
            try:
                _tmp_df.unpersist(blocking=False)
            except Exception:
                pass
        tower_seq_tmp_csv = str(tower_seq_meta.get("tmp_csv", "")).strip()
        if tower_seq_tmp_csv:
            try:
                Path(tower_seq_tmp_csv).unlink(missing_ok=True)
            except Exception:
                pass
        profile_tmp_csv = str(profile_meta.get("tmp_csv", "")).strip()
        if profile_tmp_csv:
            try:
                Path(profile_tmp_csv).unlink(missing_ok=True)
            except Exception:
                pass

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
                        fused_ranked.filter(F.col("pre_rank") <= F.lit(TOP_K_EVAL)).select(
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
            "cluster_user_topn_used": int(cluster_user_topn),
            "layered_pretrim_enabled": bool(layered_pretrim_enabled),
            "layered_pretrim_policy": layered_policy if layered_policy is not None else {},
            "profile_recall_status": str(profile_meta.get("status", "")),
            "profile_route_policy_used": profile_route_policy_bucket,
            "profile_threshold_policy_used": profile_threshold_policy_bucket,
            "profile_recall_enabled_routes": profile_meta.get("enabled_routes", []),
            "profile_recall_rows_total": int(profile_meta.get("rows", 0)),
            "profile_recall_rows_vector": int(profile_meta.get("rows_vector", 0)),
            "profile_recall_rows_shared": int(profile_meta.get("rows_shared", 0)),
            "profile_recall_rows_bridge_user": int(profile_meta.get("rows_bridge_user", 0)),
            "profile_recall_rows_bridge_type": int(profile_meta.get("rows_bridge_type", 0)),
            "profile_recall_users_with_vector": int(profile_meta.get("users_with_vector", 0)),
            "profile_recall_test_users_with_vector": int(profile_meta.get("test_users_with_vector", 0)),
            "profile_recall_test_users_with_confident_profile": int(
                profile_meta.get("test_users_with_confident_profile", 0)
            ),
            "profile_recall_items": int(profile_meta.get("items_with_profile_vector", 0)),
            "profile_calibration_applied": bool(profile_meta.get("profile_calibration_applied", False)),
            "profile_calibration_rows": int(profile_meta.get("profile_calibration_rows", 0)),
            "train_history_rows": int(train_history.count()),
            "item_semantic_enabled": bool(item_semantic_df is not None),
            "item_semantic_weight": float(SEMANTIC_WEIGHT),
            "tower_seq_enabled": bool(tower_seq_meta.get("enabled", False)),
            "tower_seq_status": str(tower_seq_meta.get("status", "")),
            "tower_seq_rows": int(tower_seq_meta.get("rows", 0)),
            "tower_seq_n_users": int(tower_seq_meta.get("n_users", 0)),
            "tower_seq_n_items": int(tower_seq_meta.get("n_items", 0)),
            "tower_seq_updates": int(tower_seq_meta.get("updates", 0)),
            "tower_seq_seq_ready_rate": float(tower_seq_meta.get("seq_ready_rate", 0.0)),
            "tower_seq_max_cand_rows_effective": int(tower_seq_meta.get("tower_seq_max_cand_rows_effective", 0)),
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
        "user_profile_table": str(profile_table_path) if profile_table_path is not None else "",
        "user_profile_tag_long": str(profile_tag_long_path) if profile_tag_long_path is not None else "",
        "profile_calibration_json": str(profile_calibration_path) if profile_calibration_path is not None else "",
        "item_semantic_features": str(item_semantic_path) if item_semantic_path is not None else "",
        "item_semantic_tag_long": str(item_semantic_tag_long_path) if item_semantic_tag_long_path is not None else "",
        "item_semantic_rows": int(item_semantic_rows),
        "enable_item_semantic": bool(item_semantic_df is not None),
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


if __name__ == "__main__":
    main()
