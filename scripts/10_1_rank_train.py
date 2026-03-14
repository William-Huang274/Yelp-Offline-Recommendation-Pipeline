from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window
from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, project_path
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
try:
    from xgboost import XGBClassifier
    from xgboost import XGBRanker
except Exception:
    XGBClassifier = None
    XGBRanker = None


RUN_TAG = "stage10_1_rank_train"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_10_1_ROOT_DIR", "data/output/10_rank_models")

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "16").strip() or 16)
SPARK_DEFAULT_PARALLELISM = int(os.getenv("SPARK_DEFAULT_PARALLELISM", "16").strip() or 16)
SPARK_SQL_ADAPTIVE_ENABLED = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "false").strip().lower() == "true"
SPARK_SQL_PARQUET_ENABLE_VECTORIZED = (
    os.getenv("SPARK_SQL_PARQUET_ENABLE_VECTORIZED", "false").strip().lower() == "true"
)
SPARK_SQL_FILES_MAX_PARTITION_BYTES = (
    os.getenv("SPARK_SQL_FILES_MAX_PARTITION_BYTES", "64m").strip() or "64m"
)
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = (
    os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
)
SPARK_DRIVER_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_DRIVER_EXTRA_JAVA_OPTIONS",
    "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m -Xss512k",
).strip()
SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS",
    "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m -Xss512k",
).strip()
PY_TEMP_SESSION_ISOLATION = os.getenv("PY_TEMP_SESSION_ISOLATION", "true").strip().lower() == "true"
PY_TEMP_SESSION_PREFIX = os.getenv("PY_TEMP_SESSION_PREFIX", "stage10_1").strip() or "stage10_1"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
TOP_K = int(os.getenv("TRAIN_EVAL_TOP_K", "10").strip() or 10)
HOLDOUT_USER_FRAC = float(os.getenv("TRAIN_HOLDOUT_USER_FRAC", "0.2").strip() or 0.2)
CALIB_USER_FRAC = float(os.getenv("TRAIN_CALIB_USER_FRAC", "0.1").strip() or 0.1)
TEST_USER_FRAC = float(os.getenv("TRAIN_TEST_USER_FRAC", str(HOLDOUT_USER_FRAC)).strip() or HOLDOUT_USER_FRAC)
RANDOM_SEED = int(os.getenv("TRAIN_RANDOM_SEED", "42").strip() or 42)
HARD_NEG_TOPK = int(os.getenv("TRAIN_HARD_NEG_TOPK", "30").strip() or 30)
MAX_NEG_PER_USER = int(os.getenv("TRAIN_MAX_NEG_PER_USER", "20").strip() or 20)
HARD_NEG_PER_USER = int(os.getenv("TRAIN_HARD_NEG_PER_USER", "10").strip() or 10)
TOTAL_NEG_PER_USER = int(os.getenv("TRAIN_TOTAL_NEG_PER_USER", str(MAX_NEG_PER_USER)).strip() or MAX_NEG_PER_USER)
EASY_NEG_MIN_RANK = int(os.getenv("TRAIN_EASY_NEG_MIN_RANK", "31").strip() or 31)
EASY_NEG_MAX_RANK = int(os.getenv("TRAIN_EASY_NEG_MAX_RANK", "250").strip() or 250)
MAX_EASY_NEG_PER_USER = int(os.getenv("TRAIN_MAX_EASY_NEG_PER_USER", "12").strip() or 12)
MAX_RANDOM_NEG_PER_USER = int(os.getenv("TRAIN_MAX_RANDOM_NEG_PER_USER", "8").strip() or 8)
MAX_SAME_CATEGORY_NEG_PER_USER = int(os.getenv("TRAIN_MAX_SAME_CATEGORY_NEG_PER_USER", "8").strip() or 8)
MAX_NEAR_POS_NEG_PER_USER = int(os.getenv("TRAIN_MAX_NEAR_POS_NEG_PER_USER", "10").strip() or 10)
NEAR_POS_RANK_WINDOW = int(os.getenv("TRAIN_NEAR_POS_RANK_WINDOW", "40").strip() or 40)
MAX_ROUTE_MIX_NEG_PER_USER = int(os.getenv("TRAIN_MAX_ROUTE_MIX_NEG_PER_USER", "6").strip() or 6)
ROUTE_NEG_FLOOR_PER_USER = int(os.getenv("TRAIN_ROUTE_NEG_FLOOR_PER_USER", "4").strip() or 4)
SAME_CATEGORY_POP_LOG_DIFF_MAX = float(os.getenv("TRAIN_SAME_CATEGORY_POP_LOG_DIFF_MAX", "0.8").strip() or 0.8)
MAX_TRAIN_ROWS = int(os.getenv("TRAIN_MAX_ROWS", "1200000").strip() or 1200000)
MAX_VALID_USERS = int(os.getenv("TRAIN_MAX_VALID_USERS", "5000").strip() or 5000)
VALID_MAX_CAND_PER_USER = int(os.getenv("TRAIN_VALID_MAX_CAND_PER_USER", "200").strip() or 200)
MIN_POS_TRAIN = int(os.getenv("TRAIN_MIN_POS", "80").strip() or 80)
BLEND_ALPHA = float(os.getenv("TRAIN_BLEND_ALPHA", "0.08").strip() or 0.08)
BLEND_ALPHA_GRID = [
    float(x.strip())
    for x in os.getenv("TRAIN_BLEND_ALPHA_GRID", "0.0,0.02,0.05,0.08,0.1,0.12,0.15,0.2,0.3").split(",")
    if x.strip()
]
if not BLEND_ALPHA_GRID:
    BLEND_ALPHA_GRID = [float(BLEND_ALPHA)]
TRAIN_TOPK_FOCUS_K = int(os.getenv("TRAIN_TOPK_FOCUS_K", "50").strip() or 50)
TRAIN_TOPK_MID_K = int(os.getenv("TRAIN_TOPK_MID_K", "150").strip() or 150)
TRAIN_POS_W_TOP = float(os.getenv("TRAIN_POS_W_TOP", "3.0").strip() or 3.0)
TRAIN_POS_W_MID = float(os.getenv("TRAIN_POS_W_MID", "1.8").strip() or 1.8)
TRAIN_POS_W_TAIL = float(os.getenv("TRAIN_POS_W_TAIL", "1.2").strip() or 1.2)
TRAIN_NEG_W_TOP = float(os.getenv("TRAIN_NEG_W_TOP", "1.2").strip() or 1.2)
TRAIN_NEG_W_MID = float(os.getenv("TRAIN_NEG_W_MID", "1.0").strip() or 1.0)
TRAIN_NEG_W_TAIL = float(os.getenv("TRAIN_NEG_W_TAIL", "0.9").strip() or 0.9)
TRAIN_RANK_PRIOR_POWER = float(os.getenv("TRAIN_RANK_PRIOR_POWER", "0.4").strip() or 0.4)
ROUTE_CALIB_TOPK = int(os.getenv("TRAIN_ROUTE_CALIB_TOPK", "200").strip() or 200)
ROUTE_CALIB_PRIOR_STRENGTH = float(os.getenv("TRAIN_ROUTE_CALIB_PRIOR_STRENGTH", "200.0").strip() or 200.0)
ROUTE_CALIB_LIFT_MIN = float(os.getenv("TRAIN_ROUTE_CALIB_LIFT_MIN", "0.85").strip() or 0.85)
ROUTE_CALIB_LIFT_MAX = float(os.getenv("TRAIN_ROUTE_CALIB_LIFT_MAX", "1.15").strip() or 1.15)
ROUTE_BLEND_GAMMA = float(os.getenv("TRAIN_ROUTE_BLEND_GAMMA", "0.25").strip() or 0.25)
CALIBRATION_METHOD = os.getenv("TRAIN_CALIBRATION_METHOD", "platt").strip().lower() or "platt"
CALIBRATION_MIN_POS = int(os.getenv("TRAIN_CALIBRATION_MIN_POS", "30").strip() or 30)
CALIBRATION_MIN_COEF = float(os.getenv("TRAIN_CALIBRATION_MIN_COEF", "0.0").strip() or 0.0)
ENABLE_GLOBAL_CALIBRATION = os.getenv("TRAIN_ENABLE_GLOBAL_CALIBRATION", "false").strip().lower() == "true"
RANKER_REQUIRE_POSITIVE_GROUP = os.getenv("TRAIN_RANKER_REQUIRE_POSITIVE_GROUP", "true").strip().lower() == "true"
RANKER_REQUIRE_POSITIVE_GROUP_VALID = os.getenv("TRAIN_RANKER_REQUIRE_POSITIVE_GROUP_VALID", "true").strip().lower() == "true"
TRAIN_USE_VALID_AS_POSITIVE = os.getenv("TRAIN_USE_VALID_AS_POSITIVE", "false").strip().lower() == "true"
TRAIN_TRUE_POS_WEIGHT = float(os.getenv("TRAIN_TRUE_POS_WEIGHT", "1.0").strip() or 1.0)
TRAIN_VALID_POS_WEIGHT = float(os.getenv("TRAIN_VALID_POS_WEIGHT", "0.6").strip() or 0.6)
TRAIN_ENABLE_TIMEWINDOW_POS = os.getenv("TRAIN_ENABLE_TIMEWINDOW_POS", "true").strip().lower() == "true"
TRAIN_WINDOW_SHORT_DAYS = int(os.getenv("TRAIN_WINDOW_SHORT_DAYS", "90").strip() or 90)
TRAIN_WINDOW_MID_DAYS = int(os.getenv("TRAIN_WINDOW_MID_DAYS", "365").strip() or 365)
TRAIN_MAX_SHORT_POS_PER_USER = int(os.getenv("TRAIN_MAX_SHORT_POS_PER_USER", "3").strip() or 3)
TRAIN_MAX_MID_POS_PER_USER = int(os.getenv("TRAIN_MAX_MID_POS_PER_USER", "4").strip() or 4)
TRAIN_MAX_OLD_POS_PER_USER = int(os.getenv("TRAIN_MAX_OLD_POS_PER_USER", "3").strip() or 3)
TRAIN_SHORT_POS_WEIGHT = float(os.getenv("TRAIN_SHORT_POS_WEIGHT", "0.35").strip() or 0.35)
TRAIN_MID_POS_WEIGHT = float(os.getenv("TRAIN_MID_POS_WEIGHT", "0.22").strip() or 0.22)
TRAIN_OLD_POS_WEIGHT = float(os.getenv("TRAIN_OLD_POS_WEIGHT", "0.12").strip() or 0.12)
TOWER_SCORE_SCALE = float(os.getenv("TRAIN_TOWER_SCORE_SCALE", "1.0").strip() or 1.0)
SEQ_SCORE_SCALE = float(os.getenv("TRAIN_SEQ_SCORE_SCALE", "1.0").strip() or 1.0)
TOWER_INV_SCALE = float(os.getenv("TRAIN_TOWER_INV_SCALE", "1.0").strip() or 1.0)
SEQ_INV_SCALE = float(os.getenv("TRAIN_SEQ_INV_SCALE", "1.0").strip() or 1.0)
MODEL_BACKEND = os.getenv("TRAIN_MODEL_BACKEND", "sklearn_lr").strip().lower() or "sklearn_lr"
XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", "300").strip() or 300)
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "6").strip() or 6)
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", "0.05").strip() or 0.05)
XGB_SUBSAMPLE = float(os.getenv("XGB_SUBSAMPLE", "0.8").strip() or 0.8)
XGB_COLSAMPLE = float(os.getenv("XGB_COLSAMPLE", "0.8").strip() or 0.8)
XGB_REG_LAMBDA = float(os.getenv("XGB_REG_LAMBDA", "2.0").strip() or 2.0)
XGB_RANK_OBJECTIVE = os.getenv("XGB_RANK_OBJECTIVE", "rank:pairwise").strip() or "rank:pairwise"
CANDIDATE_FILE = os.getenv("TRAIN_CANDIDATE_FILE", "candidates_pretrim150.parquet").strip() or "candidates_pretrim150.parquet"
CANDIDATE_FILE_FALLBACKS = ["candidates_pretrim.parquet", "candidates_pretrim150.parquet"]
ROUTE_KEYS = ("als", "cluster", "profile", "popular")
BUCKETS_OVERRIDE = os.getenv("TRAIN_BUCKETS_OVERRIDE", "").strip()
TRAIN_BUCKET_POLICY_JSON = os.getenv("TRAIN_BUCKET_POLICY_JSON", "").strip()
CACHE_MODE = os.getenv("TRAIN_CACHE_MODE", "auto").strip().lower() or "auto"
if CACHE_MODE not in {"auto", "disk", "none"}:
    CACHE_MODE = "auto"


_SPARK_TMP_CTX: SparkTmpContext | None = None

FEATURE_COLUMNS = [
    "pre_score",
    "signal_score",
    "quality_score",
    "semantic_score",
    "semantic_confidence",
    "semantic_support_log",
    "semantic_tag_richness",
    "item_pop_log",
    "user_train_log",
    "inv_pre_rank",
    "inv_als_rank",
    "inv_cluster_rank",
    "inv_popular_rank",
    "has_als",
    "has_cluster",
    "has_profile",
    "has_popular",
    "hist_feat_short",
    "hist_feat_mid",
    "hist_feat_old",
    "hist_feat_any",
    "hist_feat_recency",
    "tower_score",
    "seq_score",
    "tower_inv",
    "seq_inv",
    "source_count",
    "is_light",
    "is_mid",
    "is_heavy",
    "pre_score_x_profile",
    "pre_score_x_cluster",
    "pre_score_x_als",
    "signal_x_profile",
    "quality_x_profile",
    "semantic_x_profile",
    "semantic_x_heavy",
    "profile_x_heavy",
    "cluster_x_heavy",
    "inv_pre_minus_best_route",
    "pre_minus_signal",
    "pre_minus_semantic",
]


def normalize_model_backend(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in {"xgboost_ranker", "xgb_ranker"}:
        return "xgboost_ranker"
    if v in {"xgboost_cls", "xgb_cls", "xgboost_classifier"}:
        return "xgboost_cls"
    if v in {"xgboost", "xgb"}:
        # Backward-compatible alias; default to ranking objective for better
        # consistency with recall-stage candidate ordering.
        return "xgboost_ranker"
    return "sklearn_lr"


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX

    def _to_short_path(path: Path) -> str:
        # Convert Windows paths to short form when possible; this avoids
        # occasional command parsing issues on paths containing spaces.
        try:
            import ctypes  # local import to keep non-Windows environments safe

            p = str(path)
            size = 260
            while True:
                buf = ctypes.create_unicode_buffer(size)
                out_len = ctypes.windll.kernel32.GetShortPathNameW(p, buf, size)
                if out_len == 0:
                    break
                if out_len < size:
                    return buf.value
                size = out_len + 1
        except Exception:
            pass
        return str(path)

    py_temp_override = os.getenv("PY_TEMP_DIR", "").strip()
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=PY_TEMP_SESSION_PREFIX,
        spark_local_dir=SPARK_LOCAL_DIR,
        py_temp_root_override=py_temp_override,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=False,
    )

    temp_dir = _SPARK_TMP_CTX.py_temp_dir if PY_TEMP_SESSION_ISOLATION else _SPARK_TMP_CTX.py_temp_root
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_for_env = _to_short_path(temp_dir)
    os.environ["TEMP"] = temp_dir_for_env
    os.environ["TMP"] = temp_dir_for_env
    os.environ["TMPDIR"] = temp_dir_for_env
    tempfile.tempdir = temp_dir_for_env

    # Spark spill/cache dir (also normalized to short path for JVM args).
    local_dir = _SPARK_TMP_CTX.spark_local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    local_dir_for_conf = _to_short_path(local_dir)
    print(
        f"[SPARK] master={SPARK_MASTER} driver_mem={SPARK_DRIVER_MEMORY} executor_mem={SPARK_EXECUTOR_MEMORY} "
        f"temp_dir={temp_dir_for_env} local_dir={local_dir_for_conf} shuffle={SPARK_SQL_SHUFFLE_PARTITIONS} "
        f"default_parallelism={SPARK_DEFAULT_PARALLELISM} adaptive={SPARK_SQL_ADAPTIVE_ENABLED} "
        f"parquet_vectorized={SPARK_SQL_PARQUET_ENABLE_VECTORIZED} max_part_bytes={SPARK_SQL_FILES_MAX_PARTITION_BYTES} "
        f"tmp_cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    builder = (
        SparkSession.builder.appName("stage10-1-rank-train")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.extraJavaOptions", SPARK_DRIVER_EXTRA_JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS)
        .config("spark.local.dir", str(local_dir_for_conf))
        .config("spark.sql.shuffle.partitions", str(SPARK_SQL_SHUFFLE_PARTITIONS))
        .config("spark.default.parallelism", str(SPARK_DEFAULT_PARALLELISM))
        .config("spark.sql.adaptive.enabled", str(SPARK_SQL_ADAPTIVE_ENABLED).lower())
        .config("spark.sql.parquet.enableVectorizedReader", str(SPARK_SQL_PARQUET_ENABLE_VECTORIZED).lower())
        .config("spark.sql.files.maxPartitionBytes", SPARK_SQL_FILES_MAX_PARTITION_BYTES)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    return builder.getOrCreate()


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def parse_bucket_override(raw: str) -> set[int]:
    out: set[int] = set()
    text = (raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except ValueError:
            continue
    return out


def parse_bucket_policy_json(raw: str) -> dict[int, dict[str, Any]]:
    text = (raw or "").strip()
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


def bucket_int_param(bucket_policy: dict[str, Any], key: str, default: int, min_value: int | None = None) -> int:
    raw = bucket_policy.get(key, default)
    try:
        v = int(raw)
    except Exception:
        v = int(default)
    if min_value is not None:
        v = max(int(min_value), v)
    return v


def bucket_float_param(bucket_policy: dict[str, Any], key: str, default: float, min_value: float | None = None) -> float:
    raw = bucket_policy.get(key, default)
    try:
        v = float(raw)
    except Exception:
        v = float(default)
    if min_value is not None:
        v = max(float(min_value), v)
    return v


def normalize_split_fracs(test_frac: float, calib_frac: float) -> tuple[float, float]:
    t = float(np.clip(test_frac, 0.01, 0.9))
    c = float(np.clip(calib_frac, 0.01, 0.9))
    s = t + c
    if s >= 0.95:
        scale = 0.94 / max(1e-9, s)
        t *= scale
        c *= scale
    return t, c


def resolve_bucket_candidate_path(bucket_dir: Path) -> Path:
    primary = bucket_dir / CANDIDATE_FILE
    if primary.exists():
        return primary
    for name in CANDIDATE_FILE_FALLBACKS:
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no candidate file in {bucket_dir}, checked {[CANDIDATE_FILE, *CANDIDATE_FILE_FALLBACKS]}")


def add_feature_columns(
    cand: DataFrame,
    tower_score_scale: float = 1.0,
    seq_score_scale: float = 1.0,
    tower_inv_scale: float = 1.0,
    seq_inv_scale: float = 1.0,
) -> DataFrame:
    cols = set(cand.columns)

    def _col(name: str) -> Any:
        if name in cols:
            return F.col(name)
        return F.lit(None)

    def _inv_rank(col_name: str) -> Any:
        c = _col(col_name).cast("double")
        return F.when(c.isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / (F.log(c + F.lit(1.0)) / F.log(F.lit(2.0))))

    base = (
        cand.withColumn("pre_score", F.coalesce(_col("pre_score").cast("double"), F.lit(0.0)))
        .withColumn("signal_score", F.coalesce(_col("signal_score").cast("double"), F.lit(0.0)))
        .withColumn("quality_score", F.coalesce(_col("quality_score").cast("double"), F.lit(0.0)))
        .withColumn("semantic_score", F.coalesce(_col("semantic_score").cast("double"), F.lit(0.0)))
        .withColumn("semantic_confidence", F.coalesce(_col("semantic_confidence").cast("double"), F.lit(0.0)))
        .withColumn("semantic_support", F.coalesce(_col("semantic_support").cast("double"), F.lit(0.0)))
        .withColumn("semantic_tag_richness", F.coalesce(_col("semantic_tag_richness").cast("double"), F.lit(0.0)))
        .withColumn(
            "tower_score",
            F.coalesce(_col("tower_score").cast("double"), F.lit(0.0)) * F.lit(float(tower_score_scale)),
        )
        .withColumn(
            "seq_score",
            F.coalesce(_col("seq_score").cast("double"), F.lit(0.0)) * F.lit(float(seq_score_scale)),
        )
        .withColumn(
            "tower_inv",
            F.coalesce(_col("tower_inv").cast("double"), F.lit(0.0)) * F.lit(float(tower_inv_scale)),
        )
        .withColumn(
            "seq_inv",
            F.coalesce(_col("seq_inv").cast("double"), F.lit(0.0)) * F.lit(float(seq_inv_scale)),
        )
        .withColumn("semantic_support_log", F.log1p(F.col("semantic_support")))
        .withColumn("item_pop_log", F.log1p(F.coalesce(_col("item_train_pop_count").cast("double"), F.lit(0.0))))
        .withColumn("user_train_log", F.log1p(F.coalesce(_col("user_train_count").cast("double"), F.lit(0.0))))
        .withColumn("inv_pre_rank", _inv_rank("pre_rank"))
        .withColumn("inv_als_rank", _inv_rank("als_rank"))
        .withColumn("inv_cluster_rank", _inv_rank("cluster_rank"))
        .withColumn("inv_popular_rank", _inv_rank("popular_rank"))
        .withColumn("has_als", F.when(F.array_contains(_col("source_set"), F.lit("als")), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn(
            "has_cluster",
            F.when(F.array_contains(_col("source_set"), F.lit("cluster")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "has_profile",
            F.when(F.array_contains(_col("source_set"), F.lit("profile")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "has_popular",
            F.when(F.array_contains(_col("source_set"), F.lit("popular")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn("is_light", F.when(_col("user_segment") == F.lit("light"), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("is_mid", F.when(_col("user_segment") == F.lit("mid"), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("is_heavy", F.when(_col("user_segment") == F.lit("heavy"), F.lit(1.0)).otherwise(F.lit(0.0)))
    )
    return (
        base.withColumn("source_count", F.coalesce(F.size(F.col("source_set")).cast("double"), F.lit(0.0)))
        .withColumn("pre_score_x_profile", F.col("pre_score") * F.col("has_profile"))
        .withColumn("pre_score_x_cluster", F.col("pre_score") * F.col("has_cluster"))
        .withColumn("pre_score_x_als", F.col("pre_score") * F.col("has_als"))
        .withColumn("signal_x_profile", F.col("signal_score") * F.col("has_profile"))
        .withColumn("quality_x_profile", F.col("quality_score") * F.col("has_profile"))
        .withColumn("semantic_x_profile", F.col("semantic_score") * F.col("has_profile"))
        .withColumn("semantic_x_heavy", F.col("semantic_score") * F.col("is_heavy"))
        .withColumn("profile_x_heavy", F.col("has_profile") * F.col("is_heavy"))
        .withColumn("cluster_x_heavy", F.col("has_cluster") * F.col("is_heavy"))
        .withColumn(
            "inv_pre_minus_best_route",
            F.col("inv_pre_rank") - F.greatest(F.col("inv_als_rank"), F.col("inv_cluster_rank"), F.col("inv_popular_rank")),
        )
        .withColumn("pre_minus_signal", F.col("pre_score") - F.col("signal_score"))
        .withColumn("pre_minus_semantic", F.col("pre_score") - F.col("semantic_score"))
    )


def rank_metrics_from_pdf(pdf: pd.DataFrame, rank_col: str, top_k: int) -> tuple[float, float]:
    if pdf.empty:
        return 0.0, 0.0
    g = pdf.groupby("user_idx", dropna=False)
    ranks: list[int] = []
    for _, d in g:
        p = d[d["label"] == 1]
        if p.empty:
            ranks.append(0)
        else:
            r = int(p[rank_col].iloc[0])
            ranks.append(r if r <= int(top_k) else 0)
    rank_arr = np.asarray(ranks, dtype=np.int32)
    recall = float((rank_arr > 0).mean())
    if (rank_arr > 0).any():
        ndcg = float((1.0 / np.log2(rank_arr[rank_arr > 0] + 1.0)).sum() / max(1, len(rank_arr)))
    else:
        ndcg = 0.0
    return recall, ndcg


def score_and_rank(pdf: pd.DataFrame, score_col: str, out_rank_col: str) -> pd.DataFrame:
    if pdf.empty:
        return pdf
    # Keep ranking frame minimal to reduce peak pandas memory usage.
    cols = ["user_idx", "label", "pre_score", score_col]
    out = pdf.loc[:, cols].sort_values(
        ["user_idx", score_col, "pre_score"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    out[out_rank_col] = out.groupby("user_idx", sort=False).cumcount() + 1
    return out.loc[:, ["user_idx", "label", out_rank_col]]


def inv_rank_numpy(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.log2(x + 1.0)


def standardize(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (train_x - mean) / std, (valid_x - mean) / std, mean, std


def fit_platt_calibrator(scores: np.ndarray, labels: np.ndarray) -> dict[str, float] | None:
    if CALIBRATION_METHOD != "platt":
        return None
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0:
        return None
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos < CALIBRATION_MIN_POS or neg < CALIBRATION_MIN_POS:
        return None
    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=400, random_state=RANDOM_SEED)
        clf.fit(s.reshape(-1, 1), y)
        a = float(clf.coef_[0][0])
        b = float(clf.intercept_[0])
        if a <= CALIBRATION_MIN_COEF:
            return None
        return {"method": "platt", "a": a, "b": b}
    except Exception:
        return None


def merge_calibration_batches(score_batches: list[np.ndarray], label_batches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not score_batches or not label_batches:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int32)
    s = np.concatenate([np.asarray(x, dtype=np.float64) for x in score_batches if x is not None and len(x) > 0], axis=0)
    y = np.concatenate([np.asarray(x, dtype=np.int32) for x in label_batches if x is not None and len(x) > 0], axis=0)
    if s.size != y.size:
        n = min(int(s.size), int(y.size))
        s = s[:n]
        y = y[:n]
    return s, y


def apply_platt(scores: np.ndarray, cal: dict[str, float] | None) -> np.ndarray:
    if not cal:
        return scores.astype(np.float64)
    a = float(cal.get("a", 1.0))
    b = float(cal.get("b", 0.0))
    z = np.clip(a * scores.astype(np.float64) + b, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _safe_pre_rank(rank: np.ndarray) -> np.ndarray:
    r = np.asarray(rank, dtype=np.float64)
    return np.where(np.isfinite(r) & (r > 0.0), r, 10000.0)


def build_sample_weights(pre_rank: np.ndarray, labels: np.ndarray) -> np.ndarray:
    r = _safe_pre_rank(pre_rank)
    y = np.asarray(labels, dtype=np.int32)
    inv = np.clip(1.0 / np.log2(r + 1.0), 0.05, 1.0)
    pos_w = np.where(
        r <= float(TRAIN_TOPK_FOCUS_K),
        float(TRAIN_POS_W_TOP),
        np.where(r <= float(TRAIN_TOPK_MID_K), float(TRAIN_POS_W_MID), float(TRAIN_POS_W_TAIL)),
    )
    neg_w = np.where(
        r <= float(TRAIN_TOPK_FOCUS_K),
        float(TRAIN_NEG_W_TOP),
        np.where(r <= float(TRAIN_TOPK_MID_K), float(TRAIN_NEG_W_MID), float(TRAIN_NEG_W_TAIL)),
    )
    base = np.where(y == 1, pos_w, neg_w)
    return (base * np.power(inv, float(TRAIN_RANK_PRIOR_POWER))).astype(np.float32)


def compute_route_quality_calibration(valid_pdf: pd.DataFrame) -> dict[str, Any]:
    subset = valid_pdf.copy()
    subset = subset[subset["pre_rank"].fillna(1e9) <= float(ROUTE_CALIB_TOPK)]
    if subset.empty:
        weights = {k: 1.0 for k in ROUTE_KEYS}
        return {
            "topk_cap": int(ROUTE_CALIB_TOPK),
            "prior_strength": float(ROUTE_CALIB_PRIOR_STRENGTH),
            "lift_clip": [float(ROUTE_CALIB_LIFT_MIN), float(ROUTE_CALIB_LIFT_MAX)],
            "gamma": float(ROUTE_BLEND_GAMMA),
            "global_hit_rate": 0.0,
            "weights": weights,
            "stats": {},
        }
    global_hit = float(subset["label"].mean())
    prior_pos = float(ROUTE_CALIB_PRIOR_STRENGTH) * global_hit
    stats: dict[str, Any] = {}
    weights: dict[str, float] = {}
    for route in ROUTE_KEYS:
        col = f"has_{route}"
        if col not in subset.columns:
            weights[route] = 1.0
            stats[route] = {"count": 0, "pos": 0, "smoothed_hit_rate": global_hit, "lift": 1.0}
            continue
        mask = subset[col].to_numpy(dtype=np.float32) > 0.5
        cnt = int(mask.sum())
        pos = int(subset.loc[mask, "label"].sum()) if cnt > 0 else 0
        if cnt <= 0:
            smooth = global_hit
        else:
            smooth = float((pos + prior_pos) / (cnt + float(ROUTE_CALIB_PRIOR_STRENGTH)))
        lift = float(smooth / max(global_hit, 1e-9)) if global_hit > 0.0 else 1.0
        w = float(np.clip(lift, float(ROUTE_CALIB_LIFT_MIN), float(ROUTE_CALIB_LIFT_MAX)))
        weights[route] = w
        stats[route] = {
            "count": int(cnt),
            "pos": int(pos),
            "smoothed_hit_rate": float(smooth),
            "lift": float(lift),
            "weight": float(w),
        }
    return {
        "topk_cap": int(ROUTE_CALIB_TOPK),
        "prior_strength": float(ROUTE_CALIB_PRIOR_STRENGTH),
        "lift_clip": [float(ROUTE_CALIB_LIFT_MIN), float(ROUTE_CALIB_LIFT_MAX)],
        "gamma": float(ROUTE_BLEND_GAMMA),
        "global_hit_rate": float(global_hit),
        "weights": weights,
        "stats": stats,
    }


def compute_route_factor(valid_pdf: pd.DataFrame, route_weights: dict[str, float]) -> np.ndarray:
    n = len(valid_pdf)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    denom = np.maximum(valid_pdf["source_count"].to_numpy(dtype=np.float64), 1.0)
    total = np.zeros(n, dtype=np.float64)
    for route in ROUTE_KEYS:
        col = f"has_{route}"
        w = float(route_weights.get(route, 1.0))
        if col in valid_pdf.columns:
            total += valid_pdf[col].to_numpy(dtype=np.float64) * w
    fac = total / denom
    return np.clip(fac, 0.5, 1.5).astype(np.float64)


def tune_blend_alpha(calib_pdf: pd.DataFrame, alpha_grid: list[float]) -> tuple[float, list[dict[str, float]]]:
    if calib_pdf.empty:
        return float(BLEND_ALPHA), []
    grid = sorted(set([float(np.clip(a, 0.0, 1.0)) for a in alpha_grid])) or [float(BLEND_ALPHA)]
    base = calib_pdf[["user_idx", "label", "pre_score", "pre_prior_route", "learned_score_calibrated"]].copy()
    best_alpha = float(grid[0])
    best_ndcg = -1.0
    best_recall = -1.0
    rows: list[dict[str, float]] = []
    for alpha in grid:
        tmp = base.copy()
        tmp["learned_blend_score"] = (1.0 - float(alpha)) * tmp["pre_prior_route"] + float(alpha) * tmp["learned_score_calibrated"]
        ranked = score_and_rank(tmp, "learned_blend_score", "learned_blend_rank")
        recall, ndcg = rank_metrics_from_pdf(ranked, "learned_blend_rank", TOP_K)
        rows.append({"alpha": float(alpha), "recall_at_k": float(recall), "ndcg_at_k": float(ndcg)})
        if ndcg > best_ndcg or (abs(ndcg - best_ndcg) <= 1e-12 and recall > best_recall):
            best_alpha = float(alpha)
            best_ndcg = float(ndcg)
            best_recall = float(recall)
    return best_alpha, rows


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    source_run = resolve_stage09_run()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_dirs = sorted([p for p in source_run.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    wanted = parse_bucket_override(BUCKETS_OVERRIDE)
    if wanted:
        bucket_dirs = [p for p in bucket_dirs if int(p.name.split("_")[-1]) in wanted]
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs in {source_run}")

    model_backend = normalize_model_backend(MODEL_BACKEND)
    split_test_frac, split_calib_frac = normalize_split_fracs(TEST_USER_FRAC, CALIB_USER_FRAC)
    blend_alpha_grid = sorted(set([float(np.clip(a, 0.0, 1.0)) for a in BLEND_ALPHA_GRID])) or [float(BLEND_ALPHA)]
    bucket_policy_by_bucket = parse_bucket_policy_json(TRAIN_BUCKET_POLICY_JSON)

    payload: dict[str, Any] = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage09_run": str(source_run),
        "feature_columns": FEATURE_COLUMNS,
        "params": {
            "holdout_user_frac": HOLDOUT_USER_FRAC,
            "split_test_user_frac": split_test_frac,
            "split_calib_user_frac": split_calib_frac,
            "hard_neg_topk": HARD_NEG_TOPK,
            "max_neg_per_user": MAX_NEG_PER_USER,
            "hard_neg_per_user": HARD_NEG_PER_USER,
            "total_neg_per_user": TOTAL_NEG_PER_USER,
            "easy_neg_min_rank": EASY_NEG_MIN_RANK,
            "easy_neg_max_rank": EASY_NEG_MAX_RANK,
            "max_easy_neg_per_user": MAX_EASY_NEG_PER_USER,
            "max_random_neg_per_user": MAX_RANDOM_NEG_PER_USER,
            "max_same_category_neg_per_user": MAX_SAME_CATEGORY_NEG_PER_USER,
            "max_near_pos_neg_per_user": MAX_NEAR_POS_NEG_PER_USER,
            "near_pos_rank_window": NEAR_POS_RANK_WINDOW,
            "max_route_mix_neg_per_user": MAX_ROUTE_MIX_NEG_PER_USER,
            "route_neg_floor_per_user": ROUTE_NEG_FLOOR_PER_USER,
            "same_category_pop_log_diff_max": SAME_CATEGORY_POP_LOG_DIFF_MAX,
            "max_train_rows": MAX_TRAIN_ROWS,
            "max_valid_users": MAX_VALID_USERS,
            "valid_max_cand_per_user": VALID_MAX_CAND_PER_USER,
            "min_pos_train": MIN_POS_TRAIN,
            "random_seed": RANDOM_SEED,
            "blend_alpha": BLEND_ALPHA,
            "blend_alpha_grid": blend_alpha_grid,
            "train_topk_focus_k": TRAIN_TOPK_FOCUS_K,
            "train_topk_mid_k": TRAIN_TOPK_MID_K,
            "train_pos_w_top": TRAIN_POS_W_TOP,
            "train_pos_w_mid": TRAIN_POS_W_MID,
            "train_pos_w_tail": TRAIN_POS_W_TAIL,
            "train_neg_w_top": TRAIN_NEG_W_TOP,
            "train_neg_w_mid": TRAIN_NEG_W_MID,
            "train_neg_w_tail": TRAIN_NEG_W_TAIL,
            "train_rank_prior_power": TRAIN_RANK_PRIOR_POWER,
            "route_calib_topk": ROUTE_CALIB_TOPK,
            "route_calib_prior_strength": ROUTE_CALIB_PRIOR_STRENGTH,
            "route_calib_lift_min": ROUTE_CALIB_LIFT_MIN,
            "route_calib_lift_max": ROUTE_CALIB_LIFT_MAX,
            "route_blend_gamma": ROUTE_BLEND_GAMMA,
            "candidate_file": CANDIDATE_FILE,
            "buckets_override": sorted(wanted) if wanted else [],
            "bucket_policy_by_bucket": bucket_policy_by_bucket,
            "ranker_require_positive_group": RANKER_REQUIRE_POSITIVE_GROUP,
            "ranker_require_positive_group_valid": RANKER_REQUIRE_POSITIVE_GROUP_VALID,
            "train_use_valid_as_positive": TRAIN_USE_VALID_AS_POSITIVE,
            "train_true_pos_weight": TRAIN_TRUE_POS_WEIGHT,
            "train_valid_pos_weight": TRAIN_VALID_POS_WEIGHT,
            "train_enable_timewindow_pos": TRAIN_ENABLE_TIMEWINDOW_POS,
            "train_window_short_days": TRAIN_WINDOW_SHORT_DAYS,
            "train_window_mid_days": TRAIN_WINDOW_MID_DAYS,
            "train_max_short_pos_per_user": TRAIN_MAX_SHORT_POS_PER_USER,
            "train_max_mid_pos_per_user": TRAIN_MAX_MID_POS_PER_USER,
            "train_max_old_pos_per_user": TRAIN_MAX_OLD_POS_PER_USER,
            "train_short_pos_weight": TRAIN_SHORT_POS_WEIGHT,
            "train_mid_pos_weight": TRAIN_MID_POS_WEIGHT,
            "train_old_pos_weight": TRAIN_OLD_POS_WEIGHT,
            "model_backend": model_backend,
            "xgb_n_estimators": XGB_N_ESTIMATORS,
            "xgb_max_depth": XGB_MAX_DEPTH,
            "xgb_learning_rate": XGB_LEARNING_RATE,
            "xgb_subsample": XGB_SUBSAMPLE,
            "xgb_colsample": XGB_COLSAMPLE,
            "xgb_reg_lambda": XGB_REG_LAMBDA,
            "xgb_rank_objective": XGB_RANK_OBJECTIVE,
            "calibration_method": CALIBRATION_METHOD,
            "calibration_min_pos": CALIBRATION_MIN_POS,
            "calibration_min_coef": CALIBRATION_MIN_COEF,
            "enable_global_calibration": ENABLE_GLOBAL_CALIBRATION,
            "cache_mode": CACHE_MODE,
            "spark_master": SPARK_MASTER,
            "spark_local_dir": SPARK_LOCAL_DIR,
            "spark_sql_shuffle_partitions": SPARK_SQL_SHUFFLE_PARTITIONS,
            "spark_default_parallelism": SPARK_DEFAULT_PARALLELISM,
            "spark_sql_adaptive_enabled": SPARK_SQL_ADAPTIVE_ENABLED,
            "spark_tmp_session_isolation": SPARK_TMP_SESSION_ISOLATION,
            "spark_tmp_autoclean_enabled": SPARK_TMP_AUTOCLEAN_ENABLED,
            "spark_tmp_clean_on_exit": SPARK_TMP_CLEAN_ON_EXIT,
            "spark_tmp_retention_hours": SPARK_TMP_RETENTION_HOURS,
            "spark_tmp_clean_max_entries": SPARK_TMP_CLEAN_MAX_ENTRIES,
            "spark_tmp_context": (
                {
                    "base_dir": str(_SPARK_TMP_CTX.base_dir),
                    "spark_local_dir": str(_SPARK_TMP_CTX.spark_local_dir),
                    "py_temp_dir": str(_SPARK_TMP_CTX.py_temp_dir),
                    "cleanup_summary": dict(_SPARK_TMP_CTX.cleanup_summary),
                }
                if _SPARK_TMP_CTX is not None
                else {}
            ),
        },
        "models_by_bucket": {},
        "summaries": [],
    }
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    global_calib_scores: list[np.ndarray] = []
    global_calib_labels: list[np.ndarray] = []

    for bdir in bucket_dirs:
        bucket_start_ts = time.time()
        bucket = int(bdir.name.split("_")[-1])
        bucket_policy = bucket_policy_by_bucket.get(int(bucket), {})
        hard_neg_topk = bucket_int_param(bucket_policy, "hard_neg_topk", HARD_NEG_TOPK, min_value=1)
        hard_neg_per_user = bucket_int_param(bucket_policy, "hard_neg_per_user", HARD_NEG_PER_USER, min_value=1)
        total_neg_per_user = bucket_int_param(bucket_policy, "total_neg_per_user", TOTAL_NEG_PER_USER, min_value=1)
        easy_neg_min_rank = bucket_int_param(bucket_policy, "easy_neg_min_rank", EASY_NEG_MIN_RANK, min_value=1)
        easy_neg_max_rank = bucket_int_param(bucket_policy, "easy_neg_max_rank", EASY_NEG_MAX_RANK, min_value=easy_neg_min_rank)
        max_easy_neg_per_user = bucket_int_param(bucket_policy, "max_easy_neg_per_user", MAX_EASY_NEG_PER_USER, min_value=0)
        max_random_neg_per_user = bucket_int_param(bucket_policy, "max_random_neg_per_user", MAX_RANDOM_NEG_PER_USER, min_value=0)
        max_same_category_neg_per_user = bucket_int_param(
            bucket_policy, "max_same_category_neg_per_user", MAX_SAME_CATEGORY_NEG_PER_USER, min_value=0
        )
        max_near_pos_neg_per_user = bucket_int_param(bucket_policy, "max_near_pos_neg_per_user", MAX_NEAR_POS_NEG_PER_USER, min_value=0)
        near_pos_rank_window = bucket_int_param(bucket_policy, "near_pos_rank_window", NEAR_POS_RANK_WINDOW, min_value=1)
        max_route_mix_neg_per_user = bucket_int_param(bucket_policy, "max_route_mix_neg_per_user", MAX_ROUTE_MIX_NEG_PER_USER, min_value=0)
        route_neg_floor_per_user = bucket_int_param(bucket_policy, "route_neg_floor_per_user", ROUTE_NEG_FLOOR_PER_USER, min_value=0)
        same_category_pop_log_diff_max = bucket_float_param(
            bucket_policy, "same_category_pop_log_diff_max", SAME_CATEGORY_POP_LOG_DIFF_MAX, min_value=0.0
        )
        max_train_rows = bucket_int_param(bucket_policy, "max_train_rows", MAX_TRAIN_ROWS, min_value=1000)
        max_valid_users = bucket_int_param(bucket_policy, "max_valid_users", MAX_VALID_USERS, min_value=100)
        min_pos_train = bucket_int_param(bucket_policy, "min_pos_train", MIN_POS_TRAIN, min_value=1)
        tower_score_scale = bucket_float_param(bucket_policy, "tower_score_scale", TOWER_SCORE_SCALE, min_value=0.0)
        seq_score_scale = bucket_float_param(bucket_policy, "seq_score_scale", SEQ_SCORE_SCALE, min_value=0.0)
        tower_inv_scale = bucket_float_param(bucket_policy, "tower_inv_scale", TOWER_INV_SCALE, min_value=0.0)
        seq_inv_scale = bucket_float_param(bucket_policy, "seq_inv_scale", SEQ_INV_SCALE, min_value=0.0)
        print(
            f"[POLICY] bucket={bucket} hard_topk={hard_neg_topk} hard_per={hard_neg_per_user} "
            f"total_neg={total_neg_per_user} route_mix={max_route_mix_neg_per_user} route_floor={route_neg_floor_per_user} "
            f"near_per={max_near_pos_neg_per_user} easy_range=[{easy_neg_min_rank},{easy_neg_max_rank}] "
            f"tower_scale={tower_score_scale:.3f}/{seq_score_scale:.3f} inv_scale={tower_inv_scale:.3f}/{seq_inv_scale:.3f}"
        )
        try:
            cand_path = resolve_bucket_candidate_path(bdir)
        except FileNotFoundError as e:
            print(f"[WARN] bucket={bucket} {e}")
            continue
        truth_path = bdir / "truth.parquet"
        if not cand_path.exists() or not truth_path.exists():
            continue

        cache_this_bucket = CACHE_MODE == "disk" or (CACHE_MODE == "auto" and bucket != 2)
        if CACHE_MODE == "none":
            cache_this_bucket = False

        def _timed_count(df: DataFrame, label: str) -> int:
            t0 = time.time()
            c = int(df.count())
            print(f"[STEP] bucket={bucket} {label}={c} took={time.time() - t0:.1f}s")
            return c

        print(f"[BUCKET] {bucket} cache={cache_this_bucket}")
        cand = spark.read.parquet(cand_path.as_posix())
        truth_raw = spark.read.parquet(truth_path.as_posix())
        truth_cols = set(truth_raw.columns)
        if "valid_item_idx" in truth_cols:
            valid_item_col = F.col("valid_item_idx").cast("int")
        else:
            valid_item_col = F.lit(None).cast("int")
        truth = (
            truth_raw.select(
                F.col("user_idx").cast("int").alias("user_idx"),
                F.col("true_item_idx").cast("int").alias("true_item_idx"),
                valid_item_col.alias("valid_item_idx"),
            )
            .dropDuplicates(["user_idx"])
        )
        history_path = bdir / "train_history.parquet"
        window_short_days = max(1, int(TRAIN_WINDOW_SHORT_DAYS))
        window_mid_days = max(window_short_days + 1, int(TRAIN_WINDOW_MID_DAYS))
        max_short_pos_per_user = max(0, int(TRAIN_MAX_SHORT_POS_PER_USER))
        max_mid_pos_per_user = max(0, int(TRAIN_MAX_MID_POS_PER_USER))
        max_old_pos_per_user = max(0, int(TRAIN_MAX_OLD_POS_PER_USER))
        use_timewindow_pos_this_bucket = bool(TRAIN_ENABLE_TIMEWINDOW_POS and history_path.exists())
        if TRAIN_ENABLE_TIMEWINDOW_POS and not history_path.exists():
            print(f"[WARN] bucket={bucket} train_history.parquet missing, disable timewindow positives")
        history_flags_df: DataFrame | None = None
        history_rows_raw = 0
        if use_timewindow_pos_this_bucket:
            hist_raw = (
                spark.read.parquet(history_path.as_posix())
                .select(
                    F.col("user_idx").cast("int").alias("user_idx"),
                    F.col("item_idx").cast("int").alias("item_idx"),
                    F.col("days_to_test").cast("int").alias("days_to_test"),
                    F.col("hist_ts"),
                    F.col("hist_rating").cast("double").alias("hist_rating"),
                )
                .filter(F.col("days_to_test").isNotNull() & (F.col("days_to_test") >= F.lit(1)))
                .dropDuplicates(["user_idx", "item_idx"])
            )
            if cache_this_bucket:
                hist_raw = hist_raw.persist(StorageLevel.DISK_ONLY)
            history_rows_raw = int(hist_raw.count())
            w_hist_order = Window.partitionBy("user_idx").orderBy(
                F.asc(F.col("days_to_test")),
                F.desc(F.col("hist_ts")),
                F.desc(F.col("hist_rating")),
                F.asc(F.col("item_idx")),
            )

            def _cap_hist(df: DataFrame, max_n: int) -> DataFrame:
                if max_n <= 0:
                    return df.filter(F.lit(False))
                return (
                    df.withColumn("_hist_rank", F.row_number().over(w_hist_order))
                    .filter(F.col("_hist_rank") <= F.lit(int(max_n)))
                    .drop("_hist_rank")
                )

            hist_short = _cap_hist(hist_raw.filter(F.col("days_to_test") <= F.lit(window_short_days)), max_short_pos_per_user)
            hist_mid = _cap_hist(
                hist_raw.filter(
                    (F.col("days_to_test") > F.lit(window_short_days)) & (F.col("days_to_test") <= F.lit(window_mid_days))
                ),
                max_mid_pos_per_user,
            )
            hist_old = _cap_hist(hist_raw.filter(F.col("days_to_test") > F.lit(window_mid_days)), max_old_pos_per_user)
            history_flags_df = (
                hist_short.select("user_idx", "item_idx")
                .withColumn("label_hist_short", F.lit(1))
                .unionByName(
                    hist_mid.select("user_idx", "item_idx").withColumn("label_hist_mid", F.lit(1)),
                    allowMissingColumns=True,
                )
                .unionByName(
                    hist_old.select("user_idx", "item_idx").withColumn("label_hist_old", F.lit(1)),
                    allowMissingColumns=True,
                )
                .groupBy("user_idx", "item_idx")
                .agg(
                    F.max(F.coalesce(F.col("label_hist_short"), F.lit(0))).cast("int").alias("label_hist_short"),
                    F.max(F.coalesce(F.col("label_hist_mid"), F.lit(0))).cast("int").alias("label_hist_mid"),
                    F.max(F.coalesce(F.col("label_hist_old"), F.lit(0))).cast("int").alias("label_hist_old"),
                )
            )
            if cache_this_bucket:
                history_flags_df = history_flags_df.persist(StorageLevel.DISK_ONLY)
                hist_raw.unpersist()
        feats = add_feature_columns(
            cand,
            tower_score_scale=tower_score_scale,
            seq_score_scale=seq_score_scale,
            tower_inv_scale=tower_inv_scale,
            seq_inv_scale=seq_inv_scale,
        )
        if cache_this_bucket:
            feats = feats.persist(StorageLevel.DISK_ONLY)
        ds = feats.join(truth, on="user_idx", how="left")
        if history_flags_df is not None:
            ds = ds.join(history_flags_df, on=["user_idx", "item_idx"], how="left")
        else:
            ds = (
                ds.withColumn("label_hist_short", F.lit(0).cast("int"))
                .withColumn("label_hist_mid", F.lit(0).cast("int"))
                .withColumn("label_hist_old", F.lit(0).cast("int"))
            )
        ds = ds.withColumn("label_hist_short", F.coalesce(F.col("label_hist_short"), F.lit(0)).cast("int"))
        ds = ds.withColumn("label_hist_mid", F.coalesce(F.col("label_hist_mid"), F.lit(0)).cast("int"))
        ds = ds.withColumn("label_hist_old", F.coalesce(F.col("label_hist_old"), F.lit(0)).cast("int"))
        ds = ds.withColumn(
            "label_hist_any",
            F.when(
                (F.col("label_hist_short") + F.col("label_hist_mid") + F.col("label_hist_old")) > F.lit(0),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        ds = (
            ds.withColumn("hist_feat_short", F.col("label_hist_short").cast("double"))
            .withColumn("hist_feat_mid", F.col("label_hist_mid").cast("double"))
            .withColumn("hist_feat_old", F.col("label_hist_old").cast("double"))
            .withColumn("hist_feat_any", F.col("label_hist_any").cast("double"))
            .withColumn(
                "hist_feat_recency",
                F.when(F.col("label_hist_short") == F.lit(1), F.lit(1.0))
                .when(F.col("label_hist_mid") == F.lit(1), F.lit(0.5))
                .when(F.col("label_hist_old") == F.lit(1), F.lit(0.2))
                .otherwise(F.lit(0.0)),
            )
        )
        ds = ds.withColumn(
            "label_true",
            F.when(F.col("item_idx") == F.col("true_item_idx"), F.lit(1)).otherwise(F.lit(0)),
        ).withColumn(
            "label_valid",
            F.when(
                F.col("valid_item_idx").isNotNull() & (F.col("item_idx") == F.col("valid_item_idx")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        label_expr = F.col("label_true")
        if TRAIN_USE_VALID_AS_POSITIVE:
            label_expr = F.when(
                (F.col("label_true") == F.lit(1)) | (F.col("label_valid") == F.lit(1)),
                F.lit(1),
            ).otherwise(F.lit(0))
        ds = ds.withColumn("label", label_expr).drop("true_item_idx", "valid_item_idx")
        if cache_this_bucket:
            ds = ds.persist(StorageLevel.DISK_ONLY)

        users = (
            ds.select("user_idx")
            .distinct()
            .withColumn("_u", F.rand(seed=RANDOM_SEED + bucket))
            .withColumn(
                "split_tag",
                F.when(F.col("_u") < F.lit(split_test_frac), F.lit("test"))
                .when(F.col("_u") < F.lit(split_test_frac + split_calib_frac), F.lit("calib"))
                .otherwise(F.lit("train")),
            )
            .drop("_u")
        )
        ds = ds.join(users, on="user_idx", how="left")
        if cache_this_bucket:
            ds = ds.persist(StorageLevel.DISK_ONLY)
        train_ds = ds.filter(F.col("split_tag") == F.lit("train"))
        calib_ds = ds.filter(F.col("split_tag") == F.lit("calib"))
        test_user_df = ds.filter(F.col("split_tag") == F.lit("test")).select("user_idx").distinct()
        test_users_pdf = test_user_df.toPandas()
        test_users_pdf["user_idx"] = pd.to_numeric(test_users_pdf["user_idx"], errors="coerce").fillna(-1).astype(np.int32)
        test_users_pdf = test_users_pdf[test_users_pdf["user_idx"] >= 0].drop_duplicates("user_idx")
        test_users_rel = f"splits/bucket_{bucket}_test_users.csv"
        test_users_path = out_dir / test_users_rel
        test_users_pdf.to_csv(test_users_path, index=False, encoding="utf-8-sig")
        test_user_count = int(test_users_pdf.shape[0])
        print(f"[SPLIT] bucket={bucket} test_users={test_user_count} file={test_users_rel}")

        pos_train = _timed_count(train_ds.filter(F.col("label") == 1), "train_pos")
        if pos_train < min_pos_train:
            msg = {"bucket": bucket, "status": "skip_low_positive", "train_pos": int(pos_train)}
            payload["summaries"].append(msg)
            print(f"[WARN] {msg}")
            for df in (feats, ds, train_ds, calib_ds):
                df.unpersist()
            if history_flags_df is not None and cache_this_bucket:
                history_flags_df.unpersist()
            continue

        pos_users = train_ds.filter(F.col("label") == 1).select("user_idx").distinct()
        neg_base = train_ds.filter(F.col("label") == 0).join(pos_users, on="user_idx", how="inner")
        if cache_this_bucket:
            neg_base = neg_base.persist(StorageLevel.DISK_ONLY)
        rank_ord = F.coalesce(F.col("pre_rank").cast("double"), F.lit(1e9))
        score_ord = F.coalesce(F.col("pre_score").cast("double"), F.lit(0.0))
        source_ord = F.coalesce(F.col("source_count").cast("double"), F.lit(0.0))

        w_hard = Window.partitionBy("user_idx").orderBy(rank_ord.asc(), score_ord.desc(), F.asc("item_idx"))
        hard_neg = (
            neg_base.filter(F.col("pre_rank") <= F.lit(hard_neg_topk))
            .withColumn("neg_rank", F.row_number().over(w_hard))
            .filter(F.col("neg_rank") <= F.lit(hard_neg_per_user))
            .drop("neg_rank")
            .withColumn("_neg_type", F.lit("hard_pre"))
            .withColumn("_neg_priority", F.lit(1))
        )
        w_pos_profile = Window.partitionBy("user_idx").orderBy(
            F.coalesce(F.col("pre_rank").cast("double"), F.lit(1e9)).asc(),
            F.asc("item_idx"),
        )
        pos_profile = (
            train_ds.filter(F.col("label") == 1)
            .withColumn("_pos_profile_rank", F.row_number().over(w_pos_profile))
            .filter(F.col("_pos_profile_rank") == F.lit(1))
            .select(
                "user_idx",
                F.col("pre_rank").cast("double").alias("pos_pre_rank"),
                F.col("primary_category").alias("pos_primary_category"),
                F.log1p(F.coalesce(F.col("item_train_pop_count").cast("double"), F.lit(0.0))).alias("pos_item_pop_log"),
            )
            .dropDuplicates(["user_idx"])
        )
        w_near_pos = Window.partitionBy("user_idx").orderBy(
            F.asc(F.abs(F.col("pre_rank").cast("double") - F.col("pos_pre_rank"))),
            rank_ord.asc(),
            source_ord.desc(),
            score_ord.desc(),
            F.asc("item_idx"),
        )
        near_pos_neg = (
            neg_base.join(pos_profile.select("user_idx", "pos_pre_rank"), on="user_idx", how="inner")
            .filter(F.col("pos_pre_rank").isNotNull() & F.col("pre_rank").isNotNull())
            .filter(F.col("pre_rank") > F.lit(hard_neg_topk))
            .filter(F.abs(F.col("pre_rank").cast("double") - F.col("pos_pre_rank")) <= F.lit(float(near_pos_rank_window)))
            .withColumn("neg_rank", F.row_number().over(w_near_pos))
            .filter(F.col("neg_rank") <= F.lit(max_near_pos_neg_per_user))
            .drop("neg_rank", "pos_pre_rank")
            .withColumn("_neg_type", F.lit("near_pos"))
            .withColumn("_neg_priority", F.lit(2))
        )

        w_route_mix = Window.partitionBy("user_idx").orderBy(
            rank_ord.asc(),
            source_ord.desc(),
            F.desc(F.col("has_profile")),
            F.desc(F.col("has_cluster")),
            score_ord.desc(),
            F.asc("item_idx"),
        )
        route_mix_neg = (
            neg_base.filter(
                (F.col("pre_rank").isNotNull())
                & (F.col("pre_rank") > F.lit(hard_neg_topk))
                & (F.col("pre_rank") <= F.lit(easy_neg_max_rank))
                & (
                (F.col("source_count").cast("double") >= F.lit(2.0))
                | (F.col("has_profile").cast("double") > F.lit(0.5))
                | (F.col("has_cluster").cast("double") > F.lit(0.5))
                )
            )
            .withColumn("neg_rank", F.row_number().over(w_route_mix))
            .filter(F.col("neg_rank") <= F.lit(max_route_mix_neg_per_user))
            .drop("neg_rank")
            .withColumn("_neg_type", F.lit("route_mix"))
            .withColumn("_neg_priority", F.lit(3))
        )
        easy_low = max(int(hard_neg_topk) + 1, int(easy_neg_min_rank))
        w_easy = Window.partitionBy("user_idx").orderBy(rank_ord.asc(), source_ord.desc(), score_ord.desc(), F.asc("item_idx"))
        easy_neg = (
            neg_base.filter((F.col("pre_rank") >= F.lit(easy_low)) & (F.col("pre_rank") <= F.lit(easy_neg_max_rank)))
            .withColumn("neg_rank", F.row_number().over(w_easy))
            .filter(F.col("neg_rank") <= F.lit(max_easy_neg_per_user))
            .drop("neg_rank")
            .withColumn("_neg_type", F.lit("easy"))
            .withColumn("_neg_priority", F.lit(5))
        )
        w_rand = Window.partitionBy("user_idx").orderBy(F.rand(seed=RANDOM_SEED + bucket + 101), F.asc("item_idx"))
        random_neg = (
            neg_base.filter(F.col("pre_rank").isNull() | (F.col("pre_rank") > F.lit(easy_neg_max_rank)))
            .withColumn("neg_rank", F.row_number().over(w_rand))
            .filter(F.col("neg_rank") <= F.lit(max_random_neg_per_user))
            .drop("neg_rank")
            .withColumn("_neg_type", F.lit("random"))
            .withColumn("_neg_priority", F.lit(6))
        )
        w_same_cat = Window.partitionBy("user_idx").orderBy(
            F.asc(F.abs(F.col("item_pop_log") - F.col("pos_item_pop_log"))),
            rank_ord.asc(),
            source_ord.desc(),
            score_ord.desc(),
            F.asc("item_idx"),
        )
        same_category_neg = (
            neg_base.join(pos_profile, on="user_idx", how="inner")
            .filter(F.col("primary_category").isNotNull() & (F.col("primary_category") == F.col("pos_primary_category")))
            .filter(F.col("pre_rank").isNotNull() & (F.col("pre_rank") > F.lit(hard_neg_topk)))
            .filter(F.abs(F.col("item_pop_log") - F.col("pos_item_pop_log")) <= F.lit(same_category_pop_log_diff_max))
            .withColumn("neg_rank", F.row_number().over(w_same_cat))
            .filter(F.col("neg_rank") <= F.lit(max_same_category_neg_per_user))
            .drop("neg_rank", "pos_pre_rank", "pos_primary_category", "pos_item_pop_log")
            .withColumn("_neg_type", F.lit("same_category"))
            .withColumn("_neg_priority", F.lit(4))
        )

        neg_union = (
            hard_neg.unionByName(near_pos_neg, allowMissingColumns=True)
            .unionByName(route_mix_neg, allowMissingColumns=True)
            .unionByName(same_category_neg, allowMissingColumns=True)
            .unionByName(easy_neg, allowMissingColumns=True)
            .unionByName(random_neg, allowMissingColumns=True)
        )
        w_neg_dedup = Window.partitionBy("user_idx", "item_idx").orderBy(
            F.asc("_neg_priority"),
            rank_ord.asc(),
            source_ord.desc(),
            score_ord.desc(),
            F.asc("item_idx"),
        )
        neg_dedup = neg_union.withColumn("_dup_rank", F.row_number().over(w_neg_dedup)).filter(F.col("_dup_rank") == F.lit(1))
        neg_budget = max(1, int(total_neg_per_user))
        route_floor = max(0, min(int(route_neg_floor_per_user), int(max_route_mix_neg_per_user), int(neg_budget)))
        w_route_floor = Window.partitionBy("user_idx").orderBy(
            rank_ord.asc(),
            source_ord.desc(),
            score_ord.desc(),
            F.asc("item_idx"),
        )
        route_floor_keep = (
            neg_dedup.filter(F.col("_neg_type") == F.lit("route_mix"))
            .withColumn("_route_floor_rank", F.row_number().over(w_route_floor))
            .filter(F.col("_route_floor_rank") <= F.lit(route_floor))
            .drop("_route_floor_rank")
        )
        route_floor_counts = route_floor_keep.groupBy("user_idx").agg(F.count("*").alias("_route_floor_kept"))
        neg_fill_pool = (
            neg_dedup.join(route_floor_keep.select("user_idx", "item_idx"), on=["user_idx", "item_idx"], how="left_anti")
            .join(route_floor_counts, on="user_idx", how="left")
            .withColumn(
                "_remaining_budget",
                F.greatest(F.lit(0), F.lit(neg_budget) - F.coalesce(F.col("_route_floor_kept"), F.lit(0))),
            )
        )
        w_neg_budget = Window.partitionBy("user_idx").orderBy(
            F.asc("_neg_priority"),
            rank_ord.asc(),
            source_ord.desc(),
            score_ord.desc(),
            F.asc("item_idx"),
        )
        neg_fill = (
            neg_fill_pool.withColumn("_neg_budget_rank", F.row_number().over(w_neg_budget))
            .filter(F.col("_neg_budget_rank") <= F.col("_remaining_budget"))
            .drop("_neg_budget_rank", "_route_floor_kept", "_remaining_budget")
        )
        neg_keep = (
            route_floor_keep.unionByName(neg_fill, allowMissingColumns=True)
            .dropDuplicates(["user_idx", "item_idx"])
            .drop("_dup_rank")
        )

        train_pos = train_ds.filter(F.col("label") == 1)
        train_keep = (
            train_pos.unionByName(neg_keep.drop("_neg_type", "_neg_priority"), allowMissingColumns=True)
            .dropDuplicates(["user_idx", "item_idx"])
        )
        hard_count = _timed_count(hard_neg, "hard_neg")
        near_pos_count = _timed_count(near_pos_neg, "near_pos_neg")
        route_mix_count = _timed_count(route_mix_neg, "route_mix_neg")
        route_floor_keep_count = _timed_count(route_floor_keep, "route_mix_floor_kept")
        same_category_count = _timed_count(same_category_neg, "same_cat_neg")
        easy_count = _timed_count(easy_neg, "easy_neg")
        random_count = _timed_count(random_neg, "random_neg")
        neg_keep_count = _timed_count(neg_keep, "neg_after_budget")
        neg_type_rows = neg_keep.groupBy("_neg_type").count().collect()
        neg_type_summary = ", ".join(f"{r['_neg_type']}={int(r['count'])}" for r in neg_type_rows)
        neg_base.unpersist()

        n_train_rows = _timed_count(train_keep, "train_rows_before_cap")
        if n_train_rows > max_train_rows:
            frac = float(max_train_rows) / float(max(1, n_train_rows))
            train_keep = train_keep.sample(withReplacement=False, fraction=frac, seed=RANDOM_SEED + bucket + 7)
            n_train_rows = _timed_count(train_keep, "train_rows_after_cap")
        print(
            f"[SAMPLE] bucket={bucket} neg_mix hard={hard_count} near_pos={near_pos_count} route_mix={route_mix_count} "
            f"route_floor={route_floor} route_floor_kept={route_floor_keep_count} "
            f"same_cat={same_category_count} easy={easy_count} random={random_count} "
            f"kept={neg_keep_count} by_type=[{neg_type_summary}] train_rows={n_train_rows}"
        )

        calib_users = calib_ds.select("user_idx").distinct()
        calib_user_count = _timed_count(calib_users, "calib_users_raw")
        if max_valid_users > 0 and calib_user_count > max_valid_users:
            frac = float(max_valid_users) / float(calib_user_count)
            calib_users = calib_users.sample(withReplacement=False, fraction=frac, seed=RANDOM_SEED + bucket + 11)
        calib_eval = calib_ds.join(calib_users, on="user_idx", how="inner")
        if VALID_MAX_CAND_PER_USER > 0:
            # Memory guard for local machine:
            # keep only top-N candidates per calibration user before toPandas.
            # This does not alter label definition/metric formula, only limits
            # validation candidate table size used for model/alpha selection.
            w_valid = Window.partitionBy("user_idx").orderBy(
                F.col("pre_rank").asc_nulls_last(),
                F.col("pre_score").desc_nulls_last(),
                F.col("item_idx").asc(),
            )
            calib_eval = (
                calib_eval.withColumn("_valid_rank", F.row_number().over(w_valid))
                .filter(F.col("_valid_rank") <= F.lit(int(VALID_MAX_CAND_PER_USER)))
                .drop("_valid_rank")
            )

        train_pdf = train_keep.select(
            "user_idx",
            "item_idx",
            "pre_rank",
            "label",
            "label_true",
            "label_valid",
            "label_hist_short",
            "label_hist_mid",
            "label_hist_old",
            "label_hist_any",
            *FEATURE_COLUMNS,
        ).toPandas()
        valid_pdf = calib_eval.select(
            "user_idx",
            "item_idx",
            "pre_rank",
            "label",
            "label_true",
            "label_valid",
            "label_hist_short",
            "label_hist_mid",
            "label_hist_old",
            "label_hist_any",
            *FEATURE_COLUMNS,
        ).toPandas()
        if train_pdf.empty or valid_pdf.empty:
            msg = {"bucket": bucket, "status": "skip_empty_after_sampling"}
            payload["summaries"].append(msg)
            print(f"[WARN] {msg}")
            for df in (feats, ds, train_ds, calib_ds):
                df.unpersist()
            if history_flags_df is not None and cache_this_bucket:
                history_flags_df.unpersist()
            continue

        train_pdf = train_pdf.copy()
        valid_pdf = valid_pdf.copy()
        train_pdf["label_train"] = pd.to_numeric(train_pdf["label"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_true"] = pd.to_numeric(train_pdf["label_true"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_valid"] = pd.to_numeric(train_pdf["label_valid"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_hist_short"] = pd.to_numeric(train_pdf["label_hist_short"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_hist_mid"] = pd.to_numeric(train_pdf["label_hist_mid"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_hist_old"] = pd.to_numeric(train_pdf["label_hist_old"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_hist_any"] = pd.to_numeric(train_pdf["label_hist_any"], errors="coerce").fillna(0).astype(np.int32)
        train_pdf["label_valid_only_positive"] = np.where(
            (train_pdf["label_train"] == 1) & (train_pdf["label_true"] == 0) & (train_pdf["label_valid"] == 1),
            1,
            0,
        ).astype(np.int32)
        train_pdf["label_hist_only_positive"] = np.where(
            (train_pdf["label_train"] == 1)
            & (train_pdf["label_true"] == 0)
            & (train_pdf["label_valid"] == 0)
            & (train_pdf["label_hist_any"] == 1),
            1,
            0,
        ).astype(np.int32)
        valid_pdf["label_train"] = pd.to_numeric(valid_pdf["label"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_true"] = pd.to_numeric(valid_pdf["label_true"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_valid"] = pd.to_numeric(valid_pdf["label_valid"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_hist_short"] = pd.to_numeric(valid_pdf["label_hist_short"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_hist_mid"] = pd.to_numeric(valid_pdf["label_hist_mid"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_hist_old"] = pd.to_numeric(valid_pdf["label_hist_old"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_hist_any"] = pd.to_numeric(valid_pdf["label_hist_any"], errors="coerce").fillna(0).astype(np.int32)
        valid_pdf["label_eval"] = valid_pdf["label_true"].astype(np.int32)
        valid_pdf["label"] = valid_pdf["label_eval"].astype(np.int32)
        train_pdf["sample_weight"] = build_sample_weights(
            train_pdf["pre_rank"].to_numpy(dtype=np.float64),
            train_pdf["label_train"].to_numpy(dtype=np.int32),
        )
        pos_mult = np.ones(len(train_pdf), dtype=np.float32)
        is_pos = train_pdf["label_train"].to_numpy(dtype=np.int32) == 1
        is_true = train_pdf["label_true"].to_numpy(dtype=np.int32) == 1
        is_valid_only = (train_pdf["label_valid_only_positive"].to_numpy(dtype=np.int32) == 1) & (~is_true)
        is_hist_short_only = (
            (train_pdf["label_hist_short"].to_numpy(dtype=np.int32) == 1)
            & (~is_true)
            & (~is_valid_only)
        )
        is_hist_mid_only = (
            (train_pdf["label_hist_mid"].to_numpy(dtype=np.int32) == 1)
            & (~is_true)
            & (~is_valid_only)
            & (~is_hist_short_only)
        )
        is_hist_old_only = (
            (train_pdf["label_hist_old"].to_numpy(dtype=np.int32) == 1)
            & (~is_true)
            & (~is_valid_only)
            & (~is_hist_short_only)
            & (~is_hist_mid_only)
        )
        pos_mult = np.where(is_pos & is_true, float(TRAIN_TRUE_POS_WEIGHT), pos_mult).astype(np.float32)
        pos_mult = np.where(is_pos & is_valid_only, float(TRAIN_VALID_POS_WEIGHT), pos_mult).astype(np.float32)
        pos_mult = np.where(is_pos & is_hist_short_only, float(TRAIN_SHORT_POS_WEIGHT), pos_mult).astype(np.float32)
        pos_mult = np.where(is_pos & is_hist_mid_only, float(TRAIN_MID_POS_WEIGHT), pos_mult).astype(np.float32)
        pos_mult = np.where(is_pos & is_hist_old_only, float(TRAIN_OLD_POS_WEIGHT), pos_mult).astype(np.float32)
        train_pdf["sample_weight"] = train_pdf["sample_weight"].to_numpy(dtype=np.float32) * pos_mult
        valid_pdf["sample_weight"] = build_sample_weights(
            valid_pdf["pre_rank"].to_numpy(dtype=np.float64),
            valid_pdf["label_eval"].to_numpy(dtype=np.int32),
        )

        x_train = train_pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_train = train_pdf["label_train"].to_numpy(dtype=np.int32)
        x_valid = valid_pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_valid = valid_pdf["label_eval"].to_numpy(dtype=np.int32)
        metric_train_rows = int(len(train_pdf))
        metric_valid_rows = int(len(valid_pdf))
        metric_train_pos = int((y_train == 1).sum())
        metric_valid_pos = int((y_valid == 1).sum())
        metric_train_pos_true = int((train_pdf["label_true"].to_numpy(dtype=np.int32) == 1).sum())
        metric_train_pos_valid_only = int((train_pdf["label_valid_only_positive"].to_numpy(dtype=np.int32) == 1).sum())
        metric_train_pos_hist_only = int((train_pdf["label_hist_only_positive"].to_numpy(dtype=np.int32) == 1).sum())
        metric_train_pos_hist_short_only = int(
            (
                (train_pdf["label_train"].to_numpy(dtype=np.int32) == 1)
                & (train_pdf["label_true"].to_numpy(dtype=np.int32) == 0)
                & (train_pdf["label_valid"].to_numpy(dtype=np.int32) == 0)
                & (train_pdf["label_hist_short"].to_numpy(dtype=np.int32) == 1)
            ).sum()
        )
        metric_train_pos_hist_mid_only = int(
            (
                (train_pdf["label_train"].to_numpy(dtype=np.int32) == 1)
                & (train_pdf["label_true"].to_numpy(dtype=np.int32) == 0)
                & (train_pdf["label_valid"].to_numpy(dtype=np.int32) == 0)
                & (train_pdf["label_hist_mid"].to_numpy(dtype=np.int32) == 1)
            ).sum()
        )
        metric_train_pos_hist_old_only = int(
            (
                (train_pdf["label_train"].to_numpy(dtype=np.int32) == 1)
                & (train_pdf["label_true"].to_numpy(dtype=np.int32) == 0)
                & (train_pdf["label_valid"].to_numpy(dtype=np.int32) == 0)
                & (train_pdf["label_hist_old"].to_numpy(dtype=np.int32) == 1)
            ).sum()
        )
        metric_valid_pos_true = int((valid_pdf["label_eval"].to_numpy(dtype=np.int32) == 1).sum())
        metric_valid_pos_valid = int((valid_pdf["label_valid"].to_numpy(dtype=np.int32) == 1).sum())
        metric_valid_pos_hist_any = int((valid_pdf["label_hist_any"].to_numpy(dtype=np.int32) == 1).sum())
        train_users_kept = int(train_pdf["user_idx"].nunique())
        valid_users_kept = int(valid_pdf["user_idx"].nunique())

        model_payload: dict[str, Any]
        if model_backend == "xgboost_ranker":
            if XGBRanker is None:
                raise RuntimeError("TRAIN_MODEL_BACKEND=xgboost_ranker but xgboost is not installed")

            train_rank_pdf = train_pdf.sort_values(["user_idx", "pre_rank", "item_idx"], ascending=[True, True, True]).copy()
            valid_rank_pdf = valid_pdf.sort_values(["user_idx", "pre_rank", "item_idx"], ascending=[True, True, True]).copy()
            dropped_train_groups = 0
            dropped_valid_groups = 0
            if RANKER_REQUIRE_POSITIVE_GROUP:
                train_pos_users = set(train_rank_pdf.loc[train_rank_pdf["label_train"] == 1, "user_idx"].tolist())
                before_train_groups = int(train_rank_pdf["user_idx"].nunique())
                train_rank_pdf = train_rank_pdf[train_rank_pdf["user_idx"].isin(train_pos_users)].copy()
                dropped_train_groups = max(0, before_train_groups - int(train_rank_pdf["user_idx"].nunique()))
            if RANKER_REQUIRE_POSITIVE_GROUP_VALID:
                valid_pos_users = set(valid_rank_pdf.loc[valid_rank_pdf["label_eval"] == 1, "user_idx"].tolist())
                before_valid_groups = int(valid_rank_pdf["user_idx"].nunique())
                valid_rank_pdf = valid_rank_pdf[valid_rank_pdf["user_idx"].isin(valid_pos_users)].copy()
                dropped_valid_groups = max(0, before_valid_groups - int(valid_rank_pdf["user_idx"].nunique()))
            if train_rank_pdf.empty or valid_rank_pdf.empty:
                msg = {
                    "bucket": bucket,
                    "status": "skip_empty_rank_groups",
                    "train_groups_dropped": int(dropped_train_groups),
                    "valid_groups_dropped": int(dropped_valid_groups),
                }
                payload["summaries"].append(msg)
                print(f"[WARN] {msg}")
                for df in (feats, ds, train_ds, calib_ds):
                    df.unpersist()
                if history_flags_df is not None and cache_this_bucket:
                    history_flags_df.unpersist()
                continue
            x_train_rank = train_rank_pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            y_train_rank = train_rank_pdf["label_train"].to_numpy(dtype=np.int32)
            x_valid_rank = valid_rank_pdf[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            y_valid_rank = valid_rank_pdf["label_eval"].to_numpy(dtype=np.int32)
            qid_train = train_rank_pdf.groupby("user_idx", sort=False).size().to_numpy(dtype=np.int32)
            qid_valid = valid_rank_pdf.groupby("user_idx", sort=False).size().to_numpy(dtype=np.int32)

            ranker = XGBRanker(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE,
                reg_lambda=XGB_REG_LAMBDA,
                random_state=RANDOM_SEED + bucket,
                objective=XGB_RANK_OBJECTIVE,
                eval_metric="ndcg@10",
                tree_method="hist",
                n_jobs=4,
            )
            fit_kwargs = {
                "group": qid_train,
                "eval_set": [(x_valid_rank, y_valid_rank)],
                "eval_group": [qid_valid],
                "verbose": False,
            }
            # NOTE: For current local xgboost version, ranking mode expects
            # sample_weight length == n_groups (group weights), not per-row
            # weights. Keep fit unweighted here for compatibility.
            ranker.fit(
                x_train_rank,
                y_train_rank,
                **fit_kwargs,
            )
            valid_pdf = valid_rank_pdf
            y_valid = y_valid_rank
            metric_train_rows = int(len(train_rank_pdf))
            metric_valid_rows = int(len(valid_rank_pdf))
            metric_train_pos = int((y_train_rank == 1).sum())
            metric_valid_pos = int((y_valid_rank == 1).sum())
            metric_train_pos_true = int((train_rank_pdf["label_true"].to_numpy(dtype=np.int32) == 1).sum())
            metric_train_pos_valid_only = int((train_rank_pdf["label_valid_only_positive"].to_numpy(dtype=np.int32) == 1).sum())
            metric_train_pos_hist_only = int((train_rank_pdf["label_hist_only_positive"].to_numpy(dtype=np.int32) == 1).sum())
            metric_train_pos_hist_short_only = int(
                (
                    (train_rank_pdf["label_train"].to_numpy(dtype=np.int32) == 1)
                    & (train_rank_pdf["label_true"].to_numpy(dtype=np.int32) == 0)
                    & (train_rank_pdf["label_valid"].to_numpy(dtype=np.int32) == 0)
                    & (train_rank_pdf["label_hist_short"].to_numpy(dtype=np.int32) == 1)
                ).sum()
            )
            metric_train_pos_hist_mid_only = int(
                (
                    (train_rank_pdf["label_train"].to_numpy(dtype=np.int32) == 1)
                    & (train_rank_pdf["label_true"].to_numpy(dtype=np.int32) == 0)
                    & (train_rank_pdf["label_valid"].to_numpy(dtype=np.int32) == 0)
                    & (train_rank_pdf["label_hist_mid"].to_numpy(dtype=np.int32) == 1)
                ).sum()
            )
            metric_train_pos_hist_old_only = int(
                (
                    (train_rank_pdf["label_train"].to_numpy(dtype=np.int32) == 1)
                    & (train_rank_pdf["label_true"].to_numpy(dtype=np.int32) == 0)
                    & (train_rank_pdf["label_valid"].to_numpy(dtype=np.int32) == 0)
                    & (train_rank_pdf["label_hist_old"].to_numpy(dtype=np.int32) == 1)
                ).sum()
            )
            metric_valid_pos_true = int((valid_rank_pdf["label_eval"].to_numpy(dtype=np.int32) == 1).sum())
            metric_valid_pos_valid = int((valid_rank_pdf["label_valid"].to_numpy(dtype=np.int32) == 1).sum())
            metric_valid_pos_hist_any = int((valid_rank_pdf["label_hist_any"].to_numpy(dtype=np.int32) == 1).sum())
            train_users_kept = int(train_rank_pdf["user_idx"].nunique())
            valid_users_kept = int(valid_rank_pdf["user_idx"].nunique())
            valid_pdf["learned_score"] = ranker.predict(x_valid_rank).astype(np.float64)
            model_rel = f"models/bucket_{bucket}_xgb_ranker.json"
            model_path = out_dir / model_rel
            model_path.parent.mkdir(parents=True, exist_ok=True)
            ranker.save_model(model_path.as_posix())
            model_payload = {
                "model_backend": "xgboost_ranker",
                "feature_columns": FEATURE_COLUMNS,
                "model_file": model_rel,
                "train_groups_kept": int(train_users_kept),
                "valid_groups_kept": int(valid_users_kept),
                "train_groups_dropped": int(dropped_train_groups),
                "valid_groups_dropped": int(dropped_valid_groups),
            }
        elif model_backend == "xgboost_cls":
            if XGBClassifier is None:
                raise RuntimeError("TRAIN_MODEL_BACKEND=xgboost_cls but xgboost is not installed")
            pos_count = int((y_train == 1).sum())
            neg_count = int((y_train == 0).sum())
            scale_pos_weight = float(neg_count / max(1, pos_count))
            clf_xgb = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE,
                reg_lambda=XGB_REG_LAMBDA,
                random_state=RANDOM_SEED + bucket,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=4,
                scale_pos_weight=scale_pos_weight,
            )
            clf_xgb.fit(
                x_train,
                y_train,
                sample_weight=train_pdf["sample_weight"].to_numpy(dtype=np.float32),
                verbose=False,
            )
            valid_pdf["learned_score"] = clf_xgb.predict_proba(x_valid)[:, 1].astype(np.float64)
            model_rel = f"models/bucket_{bucket}_xgb_cls.json"
            model_path = out_dir / model_rel
            model_path.parent.mkdir(parents=True, exist_ok=True)
            clf_xgb.save_model(model_path.as_posix())
            model_payload = {
                "model_backend": "xgboost_cls",
                "feature_columns": FEATURE_COLUMNS,
                "model_file": model_rel,
            }
        else:
            x_train_z, x_valid_z, mean, std = standardize(x_train, x_valid)
            clf = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_SEED + bucket,
                max_iter=300,
            )
            clf.fit(x_train_z, y_train, sample_weight=train_pdf["sample_weight"].to_numpy(dtype=np.float64))
            valid_pdf["learned_score"] = clf.predict_proba(x_valid_z)[:, 1].astype(np.float64)
            model_payload = {
                "model_backend": "sklearn_lr",
                "feature_columns": FEATURE_COLUMNS,
                "mean": mean.astype(float).tolist(),
                "std": std.astype(float).tolist(),
                "coef": clf.coef_[0].astype(float).tolist(),
                "intercept": float(clf.intercept_[0]),
            }
        valid_pdf["pre_prior"] = inv_rank_numpy(valid_pdf["pre_rank"].to_numpy(dtype=np.float64))
        route_cal = compute_route_quality_calibration(valid_pdf)
        route_factor = compute_route_factor(valid_pdf, route_cal.get("weights", {}))
        route_gamma = float(route_cal.get("gamma", ROUTE_BLEND_GAMMA))
        valid_pdf["route_factor"] = route_factor
        valid_pdf["pre_prior_route"] = valid_pdf["pre_prior"] * np.power(route_factor, route_gamma)
        calibrator = fit_platt_calibrator(valid_pdf["learned_score"].to_numpy(dtype=np.float64), y_valid)
        if ENABLE_GLOBAL_CALIBRATION:
            global_calib_scores.append(valid_pdf["learned_score"].to_numpy(dtype=np.float64))
            global_calib_labels.append(np.asarray(y_valid, dtype=np.int32))
        valid_pdf["learned_score_calibrated"] = apply_platt(
            valid_pdf["learned_score"].to_numpy(dtype=np.float64),
            calibrator,
        )
        best_alpha, alpha_grid_rows = tune_blend_alpha(valid_pdf, blend_alpha_grid)
        valid_pdf["learned_blend_score"] = (1.0 - float(best_alpha)) * valid_pdf["pre_prior_route"] + float(best_alpha) * valid_pdf[
            "learned_score_calibrated"
        ]
        valid_ranked = score_and_rank(valid_pdf, "learned_score", "learned_rank")
        valid_ranked_cal = score_and_rank(valid_pdf, "learned_score_calibrated", "learned_rank_calibrated")
        valid_blended = score_and_rank(valid_pdf, "learned_blend_score", "learned_blend_rank")
        base_ranked = valid_pdf.loc[:, ["user_idx", "label", "pre_rank"]].copy()
        base_ranked["base_rank"] = base_ranked["pre_rank"].astype(np.int32)
        base_ranked = base_ranked.loc[:, ["user_idx", "label", "base_rank"]]
        route_prior_ranked = score_and_rank(valid_pdf, "pre_prior_route", "route_prior_rank")

        base_recall, base_ndcg = rank_metrics_from_pdf(base_ranked, "base_rank", TOP_K)
        route_prior_recall, route_prior_ndcg = rank_metrics_from_pdf(route_prior_ranked, "route_prior_rank", TOP_K)
        learned_recall, learned_ndcg = rank_metrics_from_pdf(valid_ranked, "learned_rank", TOP_K)
        learned_cal_recall, learned_cal_ndcg = rank_metrics_from_pdf(valid_ranked_cal, "learned_rank_calibrated", TOP_K)
        blend_recall, blend_ndcg = rank_metrics_from_pdf(valid_blended, "learned_blend_rank", TOP_K)
        auc = float("nan")
        auc_cal = float("nan")
        ll = float("nan")
        ll_cal = float("nan")
        try:
            if len(np.unique(y_valid)) > 1:
                auc = float(roc_auc_score(y_valid, valid_pdf["learned_score"].to_numpy(dtype=np.float64)))
                auc_cal = float(roc_auc_score(y_valid, valid_pdf["learned_score_calibrated"].to_numpy(dtype=np.float64)))
            ll = float(log_loss(y_valid, valid_pdf["learned_score"].to_numpy(dtype=np.float64), labels=[0, 1]))
            ll_cal = float(log_loss(y_valid, valid_pdf["learned_score_calibrated"].to_numpy(dtype=np.float64), labels=[0, 1]))
        except Exception:
            pass

        model_payload["calibration"] = calibrator if calibrator else {"method": "none"}
        model_payload["calibration_meta"] = {
                "n_rows": int(metric_valid_rows),
                "n_pos": int(metric_valid_pos),
                "n_neg": int(max(0, metric_valid_rows - metric_valid_pos)),
            }
        model_payload["label_strategy"] = {
            "train_label": ("true_or_valid" if TRAIN_USE_VALID_AS_POSITIVE else "true_only"),
            "eval_label": "true_only",
            "truth_has_valid_item": bool("valid_item_idx" in truth_cols),
            "timewindow_enabled_for_bucket": bool(use_timewindow_pos_this_bucket),
            "history_used_as_feature": bool(use_timewindow_pos_this_bucket),
            "timewindow_history_file_exists": bool(history_path.exists()),
            "timewindow_history_rows_raw": int(history_rows_raw),
            "timewindow_short_days": int(window_short_days),
            "timewindow_mid_days": int(window_mid_days),
            "timewindow_max_short_pos_per_user": int(max_short_pos_per_user),
            "timewindow_max_mid_pos_per_user": int(max_mid_pos_per_user),
            "timewindow_max_old_pos_per_user": int(max_old_pos_per_user),
            "train_true_pos_weight": float(TRAIN_TRUE_POS_WEIGHT),
            "train_valid_pos_weight": float(TRAIN_VALID_POS_WEIGHT),
            "train_short_pos_weight": float(TRAIN_SHORT_POS_WEIGHT),
            "train_mid_pos_weight": float(TRAIN_MID_POS_WEIGHT),
            "train_old_pos_weight": float(TRAIN_OLD_POS_WEIGHT),
        }
        model_payload["label_stats"] = {
            "train_pos_total": int(metric_train_pos),
            "train_pos_true": int(metric_train_pos_true),
            "train_pos_valid_only": int(metric_train_pos_valid_only),
            "train_pos_hist_only": int(metric_train_pos_hist_only),
            "train_pos_hist_short_only": int(metric_train_pos_hist_short_only),
            "train_pos_hist_mid_only": int(metric_train_pos_hist_mid_only),
            "train_pos_hist_old_only": int(metric_train_pos_hist_old_only),
            "valid_pos_eval_true": int(metric_valid_pos_true),
            "valid_pos_valid_item_in_candidates": int(metric_valid_pos_valid),
            "valid_pos_hist_any_in_candidates": int(metric_valid_pos_hist_any),
        }
        model_payload["split_users"] = {
            "test_users_file": test_users_rel,
            "test_user_count": int(test_user_count),
        }
        model_payload["bucket_policy_applied"] = {
            "hard_neg_topk": int(hard_neg_topk),
            "hard_neg_per_user": int(hard_neg_per_user),
            "total_neg_per_user": int(total_neg_per_user),
            "easy_neg_min_rank": int(easy_neg_min_rank),
            "easy_neg_max_rank": int(easy_neg_max_rank),
            "max_easy_neg_per_user": int(max_easy_neg_per_user),
            "max_random_neg_per_user": int(max_random_neg_per_user),
            "max_same_category_neg_per_user": int(max_same_category_neg_per_user),
            "max_near_pos_neg_per_user": int(max_near_pos_neg_per_user),
            "near_pos_rank_window": int(near_pos_rank_window),
            "max_route_mix_neg_per_user": int(max_route_mix_neg_per_user),
            "route_neg_floor_per_user": int(route_neg_floor_per_user),
            "same_category_pop_log_diff_max": float(same_category_pop_log_diff_max),
            "max_train_rows": int(max_train_rows),
            "max_valid_users": int(max_valid_users),
            "min_pos_train": int(min_pos_train),
            "train_enable_timewindow_pos": bool(use_timewindow_pos_this_bucket),
            "train_window_short_days": int(window_short_days),
            "train_window_mid_days": int(window_mid_days),
            "train_max_short_pos_per_user": int(max_short_pos_per_user),
            "train_max_mid_pos_per_user": int(max_mid_pos_per_user),
            "train_max_old_pos_per_user": int(max_old_pos_per_user),
            "tower_score_scale": float(tower_score_scale),
            "seq_score_scale": float(seq_score_scale),
            "tower_inv_scale": float(tower_inv_scale),
            "seq_inv_scale": float(seq_inv_scale),
        }
        model_payload["blend_alpha_tuning"] = {
            "grid": [float(a) for a in blend_alpha_grid],
            "selected_alpha": float(best_alpha),
            "rows": alpha_grid_rows,
        }
        model_payload["route_quality_calibration"] = route_cal
        model_payload["metrics"] = {
            "baseline_recall_at_k": float(base_recall),
            "baseline_ndcg_at_k": float(base_ndcg),
            "route_prior_recall_at_k": float(route_prior_recall),
            "route_prior_ndcg_at_k": float(route_prior_ndcg),
            "route_prior_delta_ndcg_at_k": float(route_prior_ndcg - base_ndcg),
            "learned_recall_at_k": float(learned_recall),
            "learned_ndcg_at_k": float(learned_ndcg),
            "learned_calibrated_recall_at_k": float(learned_cal_recall),
            "learned_calibrated_ndcg_at_k": float(learned_cal_ndcg),
            "delta_ndcg_at_k": float(learned_ndcg - base_ndcg),
            "blend_alpha": float(best_alpha),
            "blend_alpha_default": float(BLEND_ALPHA),
            "blend_recall_at_k": float(blend_recall),
            "blend_ndcg_at_k": float(blend_ndcg),
            "blend_delta_ndcg_at_k": float(blend_ndcg - base_ndcg),
            "valid_auc": None if (isinstance(auc, float) and math.isnan(auc)) else float(auc),
            "valid_auc_calibrated": None if (isinstance(auc_cal, float) and math.isnan(auc_cal)) else float(auc_cal),
            "valid_logloss": None if (isinstance(ll, float) and math.isnan(ll)) else float(ll),
            "valid_logloss_calibrated": None if (isinstance(ll_cal, float) and math.isnan(ll_cal)) else float(ll_cal),
            "train_rows": int(n_train_rows),
            "train_rows_kept_for_fit": int(metric_train_rows),
            "valid_rows_kept_for_eval": int(metric_valid_rows),
            "train_pos": int(metric_train_pos),
            "valid_pos": int(metric_valid_pos),
            "valid_users": int(valid_users_kept),
            "calib_users": int(valid_users_kept),
            "test_users": int(test_user_count),
            "train_users": int(train_users_kept),
            "hard_neg_rows": int(hard_count),
            "near_pos_neg_rows": int(near_pos_count),
            "route_mix_neg_rows": int(route_mix_count),
            "route_mix_floor_per_user": int(route_floor),
            "route_mix_floor_kept_rows": int(route_floor_keep_count),
            "same_category_neg_rows": int(same_category_count),
            "easy_neg_rows": int(easy_count),
            "random_neg_rows": int(random_count),
            "neg_rows_after_budget": int(neg_keep_count),
        }
        payload["models_by_bucket"][str(bucket)] = model_payload
        payload["summaries"].append({"bucket": bucket, "status": "ok", **model_payload["metrics"]})
        print(
            f"[METRIC] bucket={bucket} base_ndcg={base_ndcg:.4f} "
            f"backend={model_payload.get('model_backend')} "
            f"route_ndcg={route_prior_ndcg:.4f} "
            f"learned_ndcg={learned_ndcg:.4f} delta={learned_ndcg - base_ndcg:+.4f} "
            f"blend_ndcg={blend_ndcg:.4f} delta={blend_ndcg - base_ndcg:+.4f} alpha={best_alpha:.3f}"
        )

        for df in (feats, ds, train_ds, calib_ds):
            df.unpersist()
        if history_flags_df is not None and cache_this_bucket:
            history_flags_df.unpersist()
        print(f"[DONE] bucket={bucket} elapsed={time.time() - bucket_start_ts:.1f}s")

    global_scores, global_labels = merge_calibration_batches(global_calib_scores, global_calib_labels)
    if ENABLE_GLOBAL_CALIBRATION and global_scores.size > 0:
        global_cal = fit_platt_calibrator(global_scores, global_labels)
        payload["global_calibration"] = global_cal if global_cal else {"method": "none"}
        payload["global_calibration_meta"] = {
            "n_rows": int(global_scores.size),
            "n_pos": int((global_labels == 1).sum()),
            "n_neg": int((global_labels == 0).sum()),
        }
        print(
            f"[CALIB] global enabled rows={global_scores.size} "
            f"pos={(global_labels == 1).sum()} neg={(global_labels == 0).sum()} "
            f"method={payload['global_calibration'].get('method', 'none')}"
        )
    else:
        payload["global_calibration"] = {"method": "none"}

    out_json = out_dir / "rank_model.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] wrote model: {out_json}")

    spark.stop()


if __name__ == "__main__":
    main()
