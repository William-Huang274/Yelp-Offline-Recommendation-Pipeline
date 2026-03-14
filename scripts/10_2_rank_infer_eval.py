from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import uuid
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
try:
    from xgboost import XGBClassifier
    from xgboost import XGBRanker
except Exception:
    XGBClassifier = None
    XGBRanker = None
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None


RUN_TAG = "stage10_2_rank_infer_eval"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"

RANK_MODEL_JSON = os.getenv("RANK_MODEL_JSON", "").strip()
RANK_MODEL_ROOT = env_or_project_path("RANK_MODEL_ROOT_DIR", "data/output/10_rank_models")
RANK_MODEL_SUFFIX = "_stage10_1_rank_train"
RANK_MODEL_FILE = "rank_model.json"

OUTPUT_ROOT = env_or_project_path("OUTPUT_10_2_ROOT_DIR", "data/output/10_2_rank_infer_eval")
METRICS_PATH = env_or_project_path("STAGE10_RESULTS_METRICS_PATH", "data/metrics/recsys_stage10_results.csv")

TOP_K = int(os.getenv("RANK_EVAL_TOP_K", "10").strip() or 10)
TAIL_QUANTILE = 0.8
BLEND_ALPHA_OVERRIDE = os.getenv("RANK_BLEND_ALPHA", "").strip()
BLEND_ALPHA_BY_BUCKET = os.getenv("RANK_BLEND_ALPHA_BY_BUCKET", "").strip()
RANK_BLEND_MODE = (os.getenv("RANK_BLEND_MODE", "prob").strip().lower() or "prob")
RANK_BLEND_MODE_BY_BUCKET = os.getenv("RANK_BLEND_MODE_BY_BUCKET", "").strip()
RANK_BLEND_GUARD_ENABLE = os.getenv("RANK_BLEND_GUARD_ENABLE", "true").strip().lower() == "true"
RANK_BLEND_GUARD_MIN_STD = float(os.getenv("RANK_BLEND_GUARD_MIN_STD", "0.015").strip() or 0.015)
RANK_BLEND_GUARD_MIN_P95_P05 = float(os.getenv("RANK_BLEND_GUARD_MIN_P95_P05", "0.06").strip() or 0.06)
RANK_BLEND_GUARD_MIN_UNIQUE = int(os.getenv("RANK_BLEND_GUARD_MIN_UNIQUE", "20").strip() or 20)
RERANK_TOPN_OVERRIDE = int(os.getenv("RANK_RERANK_TOPN", "0").strip() or 0)
EVAL_USER_COHORT_PATH = os.getenv("RANK_EVAL_USER_COHORT_PATH", "").strip()
EVAL_CANDIDATE_TOPN = int(os.getenv("RANK_EVAL_CANDIDATE_TOPN", "0").strip() or 0)
XGB_MAX_SCORE_ROWS = int(os.getenv("XGB_MAX_SCORE_ROWS", "1500000").strip() or 1500000)
XGB_USER_BATCH_SIZE = int(os.getenv("XGB_USER_BATCH_SIZE", "2500").strip() or 2500)
XGB_BATCH_MODE = (os.getenv("XGB_BATCH_MODE", "hash_partition_disk").strip().lower() or "hash_partition_disk")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_SQL_ADAPTIVE_ENABLED = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "false").strip().lower() == "true"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = (
    os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
)
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
SPARK_DRIVER_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_DRIVER_EXTRA_JAVA_OPTIONS",
    "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m -Xss512k",
).strip()
BUCKETS_OVERRIDE = os.getenv("RANK_BUCKETS_OVERRIDE", "").strip()
SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS",
    "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=128m -XX:MaxMetaspaceSize=256m -Xss512k",
).strip()
DEFAULT_ROUTE_BLEND_GAMMA = float(os.getenv("RANK_ROUTE_BLEND_GAMMA", "0.25").strip() or 0.25)
CALIBRATION_SCOPE = (os.getenv("RANK_CALIBRATION_SCOPE", "bucket").strip().lower() or "bucket")
AUDIT_ENABLE = os.getenv("RANK_AUDIT_ENABLE", "false").strip().lower() == "true"
AUDIT_SAMPLE_ROWS = int(os.getenv("RANK_AUDIT_SAMPLE_ROWS", "80000").strip() or 80000)
AUDIT_BINS = int(os.getenv("RANK_AUDIT_BINS", "20").strip() or 20)
DIAGNOSTICS_ENABLE = os.getenv("RANK_DIAGNOSTICS_ENABLE", "true").strip().lower() == "true"
CALIB_RESIDUAL_SHRINK_TAU = float(os.getenv("RANK_CALIB_RESIDUAL_SHRINK_TAU", "120").strip() or 120.0)
CALIB_RESIDUAL_MIN_POS = int(os.getenv("RANK_CALIB_RESIDUAL_MIN_POS", "30").strip() or 30)
CALIB_RESIDUAL_MIN_LAMBDA = float(os.getenv("RANK_CALIB_RESIDUAL_MIN_LAMBDA", "0.0").strip() or 0.0)
CALIB_RESIDUAL_MAX_LAMBDA = float(os.getenv("RANK_CALIB_RESIDUAL_MAX_LAMBDA", "1.0").strip() or 1.0)
ROUTE_KEYS = ("als", "cluster", "profile", "popular")
VALID_BLEND_MODES = {"prob", "rank_prior", "auto"}


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

RESULT_FIELDS = [
    "run_id_10",
    "source_run_09",
    "bucket_min_train_reviews",
    "model",
    "recall_at_k",
    "ndcg_at_k",
    "user_coverage_at_k",
    "item_coverage_at_k",
    "tail_coverage_at_k",
    "novelty_at_k",
    "n_users",
    "n_items",
    "n_candidates",
]


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX

    def _to_short_path(path: Path) -> str:
        try:
            import ctypes

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
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        py_temp_root_override=py_temp_override,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=False,
    )
    temp_root = _to_short_path(_SPARK_TMP_CTX.py_temp_dir)
    os.environ["TEMP"] = temp_root
    os.environ["TMP"] = temp_root
    os.environ["TMPDIR"] = temp_root
    tempfile.tempdir = temp_root
    local_dir = _SPARK_TMP_CTX.spark_local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    local_dir_short = _to_short_path(local_dir)
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} spark_local_dir={local_dir} py_temp={_SPARK_TMP_CTX.py_temp_dir} "
        f"auto_clean={SPARK_TMP_AUTOCLEAN_ENABLED} retention_h={SPARK_TMP_RETENTION_HOURS} "
        f"cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    return (
        SparkSession.builder.appName("stage10-2-rank-infer-eval")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.extraJavaOptions", SPARK_DRIVER_EXTRA_JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS)
        .config("spark.local.dir", str(local_dir_short))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.sql.adaptive.enabled", "true" if SPARK_SQL_ADAPTIVE_ENABLED else "false")
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def pick_latest_run(root: Path, suffix: str, required_file: str | None = None) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if required_file is None:
        if not runs:
            raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
        return runs[0]
    for r in runs:
        if (r / required_file).exists():
            return r
    raise FileNotFoundError(f"no run in {root} with suffix={suffix} file={required_file}")


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def resolve_rank_model_json() -> Path:
    if RANK_MODEL_JSON:
        p = normalize_legacy_project_path(RANK_MODEL_JSON)
        if not p.exists():
            raise FileNotFoundError(f"RANK_MODEL_JSON not found: {p}")
        return p
    run = pick_latest_run(RANK_MODEL_ROOT, RANK_MODEL_SUFFIX, RANK_MODEL_FILE)
    return run / RANK_MODEL_FILE


def resolve_eval_user_cohort_path() -> Path | None:
    raw = str(EVAL_USER_COHORT_PATH or "").strip()
    if not raw:
        return None
    p = normalize_legacy_project_path(raw)
    if not p.exists():
        raise FileNotFoundError(f"RANK_EVAL_USER_COHORT_PATH not found: {p}")
    return p


def load_eval_users_df(spark: SparkSession, cohort_path: Path) -> DataFrame:
    suffixes = {s.lower() for s in cohort_path.suffixes}
    if ".parquet" in suffixes:
        base = spark.read.parquet(cohort_path.as_posix())
    else:
        base = spark.read.csv(cohort_path.as_posix(), header=True, inferSchema=True)
    return (
        base.select(F.col("user_idx").cast("int").alias("user_idx"))
        .filter(F.col("user_idx").isNotNull() & (F.col("user_idx") >= F.lit(0)))
        .dropDuplicates(["user_idx"])
    )


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


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def parse_bucket_float_map(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    text = (raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            out[int(k.strip())] = float(v.strip())
        except ValueError:
            continue
    return out


def parse_bucket_str_map(raw: str) -> dict[int, str]:
    out: dict[int, str] = {}
    text = (raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p or ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            bk = int(k.strip())
        except ValueError:
            continue
        mode = str(v or "").strip().lower()
        if not mode:
            continue
        out[int(bk)] = mode
    return out


def normalize_blend_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode in VALID_BLEND_MODES:
        return mode
    return "prob"


def _rank_prior_from_score(pdf: pd.DataFrame, score_col: str) -> np.ndarray:
    rank = pdf.groupby("user_idx")[score_col].rank(method="first", ascending=False).to_numpy(dtype=np.float64)
    return (1.0 / np.log2(rank + 1.0)).astype(np.float64)


def _blend_guard_stats(scores: np.ndarray) -> dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size <= 0:
        return {
            "score_std": 0.0,
            "score_p95_p05": 0.0,
            "score_unique_6dp": 0.0,
        }
    std_v = float(np.std(arr))
    q05 = float(np.quantile(arr, 0.05))
    q95 = float(np.quantile(arr, 0.95))
    unique_6dp = float(np.unique(np.round(arr, 6)).size)
    return {
        "score_std": float(std_v),
        "score_p95_p05": float(q95 - q05),
        "score_unique_6dp": float(unique_6dp),
    }


def summarize_blend_infos(blend_infos: list[dict[str, Any]], fallback_requested: str) -> dict[str, Any]:
    if not blend_infos:
        mode = normalize_blend_mode(fallback_requested)
        return {
            "requested_blend_mode": mode,
            "effective_blend_mode": mode,
            "effective_mode_counts": {mode: 0},
            "blend_guard_triggered_rate": 0.0,
            "score_std_mean": 0.0,
            "score_p95_p05_mean": 0.0,
            "score_unique_6dp_mean": 0.0,
        }
    counts: dict[str, int] = {}
    guard = 0
    std_vals: list[float] = []
    spread_vals: list[float] = []
    uniq_vals: list[float] = []
    requested_mode = normalize_blend_mode(str(blend_infos[0].get("requested_blend_mode", fallback_requested)))
    for info in blend_infos:
        em = normalize_blend_mode(str(info.get("effective_blend_mode", requested_mode)))
        counts[em] = int(counts.get(em, 0) + 1)
        guard += 1 if bool(info.get("blend_guard_triggered", False)) else 0
        std_vals.append(float(info.get("score_std", 0.0)))
        spread_vals.append(float(info.get("score_p95_p05", 0.0)))
        uniq_vals.append(float(info.get("score_unique_6dp", 0.0)))
    effective_mode = max(counts.items(), key=lambda kv: kv[1])[0]
    n = float(max(1, len(blend_infos)))
    return {
        "requested_blend_mode": requested_mode,
        "effective_blend_mode": str(effective_mode),
        "effective_mode_counts": {str(k): int(v) for k, v in counts.items()},
        "blend_guard_triggered_rate": float(guard / n),
        "score_std_mean": float(sum(std_vals) / n),
        "score_p95_p05_mean": float(sum(spread_vals) / n),
        "score_unique_6dp_mean": float(sum(uniq_vals) / n),
    }


def _build_history_flags(
    spark: SparkSession,
    bucket_dir: Path,
    window_short_days: int,
    window_mid_days: int,
    max_short_pos_per_user: int,
    max_mid_pos_per_user: int,
    max_old_pos_per_user: int,
) -> DataFrame | None:
    history_path = bucket_dir / "train_history.parquet"
    if not history_path.exists():
        return None
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

    hist_short = _cap_hist(hist_raw.filter(F.col("days_to_test") <= F.lit(int(window_short_days))), int(max_short_pos_per_user))
    hist_mid = _cap_hist(
        hist_raw.filter(
            (F.col("days_to_test") > F.lit(int(window_short_days))) & (F.col("days_to_test") <= F.lit(int(window_mid_days)))
        ),
        int(max_mid_pos_per_user),
    )
    hist_old = _cap_hist(hist_raw.filter(F.col("days_to_test") > F.lit(int(window_mid_days))), int(max_old_pos_per_user))
    return (
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


def attach_history_features(cand_raw: DataFrame, bucket_dir: Path, model_bucket: dict[str, Any] | None) -> DataFrame:
    out = cand_raw
    label_strategy = model_bucket.get("label_strategy", {}) if isinstance(model_bucket, dict) else {}
    bucket_policy = model_bucket.get("bucket_policy_applied", {}) if isinstance(model_bucket, dict) else {}
    use_history = bool(
        (label_strategy.get("history_used_as_feature") is True)
        or (label_strategy.get("timewindow_enabled_for_bucket") is True)
    )
    if use_history:
        short_days = int(bucket_policy.get("train_window_short_days", 90) or 90)
        mid_days = int(bucket_policy.get("train_window_mid_days", 365) or 365)
        max_short = int(bucket_policy.get("train_max_short_pos_per_user", 3) or 3)
        max_mid = int(bucket_policy.get("train_max_mid_pos_per_user", 4) or 4)
        max_old = int(bucket_policy.get("train_max_old_pos_per_user", 3) or 3)
        hist_flags = _build_history_flags(
            spark=cand_raw.sparkSession,
            bucket_dir=bucket_dir,
            window_short_days=max(1, short_days),
            window_mid_days=max(max(1, short_days) + 1, mid_days),
            max_short_pos_per_user=max(0, max_short),
            max_mid_pos_per_user=max(0, max_mid),
            max_old_pos_per_user=max(0, max_old),
        )
        if hist_flags is not None:
            out = out.join(hist_flags, on=["user_idx", "item_idx"], how="left")
    cols = set(out.columns)
    if "label_hist_short" not in cols:
        out = out.withColumn("label_hist_short", F.lit(0).cast("int"))
    if "label_hist_mid" not in cols:
        out = out.withColumn("label_hist_mid", F.lit(0).cast("int"))
    if "label_hist_old" not in cols:
        out = out.withColumn("label_hist_old", F.lit(0).cast("int"))
    out = out.withColumn("label_hist_short", F.coalesce(F.col("label_hist_short"), F.lit(0)).cast("int"))
    out = out.withColumn("label_hist_mid", F.coalesce(F.col("label_hist_mid"), F.lit(0)).cast("int"))
    out = out.withColumn("label_hist_old", F.coalesce(F.col("label_hist_old"), F.lit(0)).cast("int"))
    out = out.withColumn(
        "label_hist_any",
        F.when(
            (F.col("label_hist_short") + F.col("label_hist_mid") + F.col("label_hist_old")) > F.lit(0),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    return (
        out.withColumn("hist_feat_short", F.col("label_hist_short").cast("double"))
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


def normalize_calibration_scope(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in {"global", "global_only"}:
        return "global"
    if v in {"global_then_bucket", "hybrid", "residual", "global_bucket_residual"}:
        return "global_then_bucket"
    return "bucket"


def is_platt_calibration(cal: dict[str, Any] | None) -> bool:
    if not isinstance(cal, dict):
        return False
    return str(cal.get("method", "none")).strip().lower() == "platt"


def extract_bucket_calibration_stats(model_bucket: dict[str, Any]) -> tuple[int, int]:
    meta = model_bucket.get("calibration_meta", {})
    if isinstance(meta, dict):
        n_pos = int(meta.get("n_pos", 0) or 0)
        n_rows = int(meta.get("n_rows", 0) or 0)
        if n_rows > 0:
            return max(0, n_pos), max(0, n_rows)
    metrics = model_bucket.get("metrics", {})
    if isinstance(metrics, dict):
        n_pos = int(metrics.get("valid_pos", 0) or 0)
        n_rows = int(metrics.get("valid_rows_kept_for_eval", 0) or 0)
        return max(0, n_pos), max(0, n_rows)
    return 0, 0


def resolve_bucket_residual_lambda(model_bucket: dict[str, Any]) -> tuple[float, dict[str, float | int]]:
    n_pos, n_rows = extract_bucket_calibration_stats(model_bucket)
    tau = max(1.0, float(CALIB_RESIDUAL_SHRINK_TAU))
    min_pos = max(1, int(CALIB_RESIDUAL_MIN_POS))
    lam_raw = float(n_pos) / float(n_pos + tau) if n_pos > 0 else 0.0
    if n_pos < min_pos:
        lam_raw *= float(n_pos) / float(min_pos)
    lam = float(np.clip(lam_raw, float(CALIB_RESIDUAL_MIN_LAMBDA), float(CALIB_RESIDUAL_MAX_LAMBDA)))
    meta: dict[str, float | int] = {
        "bucket_valid_pos": int(n_pos),
        "bucket_valid_rows": int(n_rows),
        "tau": float(tau),
        "min_pos": int(min_pos),
        "lambda_raw": float(lam_raw),
        "lambda": float(lam),
    }
    return lam, meta


def resolve_calibration_payload(model_bucket: dict[str, Any], model_data: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    scope = normalize_calibration_scope(CALIBRATION_SCOPE)
    bucket_cal = model_bucket.get("calibration")
    global_cal = model_data.get("global_calibration")
    if scope == "global":
        return (global_cal if is_platt_calibration(global_cal) else {"method": "none"}, "global")
    if scope == "global_then_bucket":
        has_global = is_platt_calibration(global_cal)
        has_bucket = is_platt_calibration(bucket_cal)
        if has_global and has_bucket:
            lam, meta = resolve_bucket_residual_lambda(model_bucket)
            payload = {
                "method": "platt_residual",
                "global": global_cal,
                "bucket": bucket_cal,
                "lambda": float(lam),
                "shrinkage_meta": meta,
            }
            if lam <= 1e-8:
                return payload, "global_plus_bucket_residual_shrink0"
            return payload, "global_plus_bucket_residual"
        if has_bucket:
            return bucket_cal, "bucket_only_fallback"
        if has_global:
            return global_cal, "global_only_fallback"
        return {"method": "none"}, "none"
    return (bucket_cal if is_platt_calibration(bucket_cal) else {"method": "none"}, "bucket")


def summarize_array(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "p01": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(np.quantile(arr, 0.01)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def compute_calibration_audit(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> dict[str, float | None]:
    y = np.asarray(labels, dtype=np.int32)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if y.size == 0:
        return {"brier": None, "ece": None, "slope": None, "intercept": None}
    brier = float(np.mean((p - y) ** 2))
    bins = max(5, int(n_bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges, right=True) - 1
    idx = np.clip(idx, 0, bins - 1)
    ece = 0.0
    n = float(y.size)
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += abs(conf - acc) * (float(mask.sum()) / n)

    slope = None
    intercept = None
    if LogisticRegression is not None and len(np.unique(y)) > 1:
        try:
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs", max_iter=300)
            lr.fit(logits, y)
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
        except Exception:
            slope = None
            intercept = None
    return {"brier": brier, "ece": float(ece), "slope": slope, "intercept": intercept}


def build_score_audit(
    scored_pdf: pd.DataFrame,
    truth_pdf: pd.DataFrame,
    sample_rows: int,
    n_bins: int,
    calibration_source: str,
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    if scored_pdf is None or scored_pdf.empty:
        return {
            "n_rows_raw": 0,
            "n_rows_audited": 0,
            "calibration_source": calibration_source,
            "calibration": calibration or {"method": "none"},
            "score_stats": {},
            "calibration_metrics": {},
        }
    pdf = scored_pdf.copy()
    n_raw = int(len(pdf))
    if sample_rows > 0 and len(pdf) > sample_rows:
        pdf = pdf.sample(n=sample_rows, random_state=42)

    truth = truth_pdf[["user_idx", "true_item_idx"]].copy()
    merged = pdf.merge(
        truth,
        how="left",
        left_on=["user_idx", "item_idx"],
        right_on=["user_idx", "true_item_idx"],
    )
    merged["label"] = np.where(merged["true_item_idx"].notna(), 1, 0).astype(np.int32)
    label = merged["label"].to_numpy(dtype=np.int32)

    raw = merged["learned_score"].to_numpy(dtype=np.float64)
    p_model = merged["learned_score_for_blend"].to_numpy(dtype=np.float64)
    p_pre = merged["pre_prior_route"].to_numpy(dtype=np.float64)
    p_final = merged["learned_blend_score"].to_numpy(dtype=np.float64)
    p_model_clip = np.clip(p_model, 1e-6, 1.0 - 1e-6)
    p_pre_clip = np.clip(p_pre, 1e-6, 1.0 - 1e-6)
    p_final_clip = np.clip(p_final, 1e-6, 1.0 - 1e-6)

    return {
        "n_rows_raw": n_raw,
        "n_rows_audited": int(len(merged)),
        "n_positive_audited": int(label.sum()),
        "positive_rate_audited": float(label.mean()) if len(label) else 0.0,
        "calibration_source": calibration_source,
        "calibration": calibration or {"method": "none"},
        "score_stats": {
            "raw": summarize_array(raw),
            "p_model": summarize_array(p_model),
            "p_pre": summarize_array(p_pre),
            "p_final": summarize_array(p_final),
        },
        "calibration_metrics": {
            "model": compute_calibration_audit(label, p_model_clip, n_bins),
            "pre": compute_calibration_audit(label, p_pre_clip, n_bins),
            "final": compute_calibration_audit(label, p_final_clip, n_bins),
        },
    }


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


def resolve_tower_seq_scales(model_bucket: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not isinstance(model_bucket, dict):
        return 1.0, 1.0, 1.0, 1.0
    policy = model_bucket.get("bucket_policy_applied", {})
    if not isinstance(policy, dict):
        return 1.0, 1.0, 1.0, 1.0

    def _safe(name: str, default: float = 1.0) -> float:
        try:
            return max(0.0, float(policy.get(name, default)))
        except Exception:
            return float(default)

    return (
        _safe("tower_score_scale", 1.0),
        _safe("seq_score_scale", 1.0),
        _safe("tower_inv_scale", 1.0),
        _safe("seq_inv_scale", 1.0),
    )


def apply_platt_numpy(scores: np.ndarray, calibration: dict[str, Any] | None) -> np.ndarray:
    arr = scores.astype(np.float64)
    if not calibration:
        return arr
    method = str(calibration.get("method", "none")).strip().lower()
    if method == "platt":
        a = float(calibration.get("a", 1.0))
        b = float(calibration.get("b", 0.0))
        z = np.clip(a * arr + b, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))
    if method == "platt_residual":
        global_cal = calibration.get("global")
        bucket_cal = calibration.get("bucket")
        if not is_platt_calibration(global_cal) or not is_platt_calibration(bucket_cal):
            return arr
        lam = float(np.clip(float(calibration.get("lambda", 1.0)), 0.0, 1.0))
        a_g = float(global_cal.get("a", 1.0))
        b_g = float(global_cal.get("b", 0.0))
        a_b = float(bucket_cal.get("a", 1.0))
        b_b = float(bucket_cal.get("b", 0.0))
        z_global = np.clip(a_g * arr + b_g, -50.0, 50.0)
        z_bucket = np.clip(a_b * arr + b_b, -50.0, 50.0)
        z = np.clip(z_global + lam * (z_bucket - z_global), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))
    return arr


def build_platt_expr(score_col: str, calibration: dict[str, Any] | None) -> Any:
    if not calibration:
        return F.col(score_col).cast("double")
    method = str(calibration.get("method", "none")).strip().lower()
    if method == "platt":
        a = float(calibration.get("a", 1.0))
        b = float(calibration.get("b", 0.0))
        z = F.greatest(F.least(F.lit(a) * F.col(score_col).cast("double") + F.lit(b), F.lit(50.0)), F.lit(-50.0))
        return F.lit(1.0) / (F.lit(1.0) + F.exp(-z))
    if method == "platt_residual":
        global_cal = calibration.get("global")
        bucket_cal = calibration.get("bucket")
        if not is_platt_calibration(global_cal) or not is_platt_calibration(bucket_cal):
            return F.col(score_col).cast("double")
        lam = float(np.clip(float(calibration.get("lambda", 1.0)), 0.0, 1.0))
        z_global = F.lit(float(global_cal.get("a", 1.0))) * F.col(score_col).cast("double") + F.lit(float(global_cal.get("b", 0.0)))
        z_bucket = F.lit(float(bucket_cal.get("a", 1.0))) * F.col(score_col).cast("double") + F.lit(float(bucket_cal.get("b", 0.0)))
        z_global = F.greatest(F.least(z_global, F.lit(50.0)), F.lit(-50.0))
        z_bucket = F.greatest(F.least(z_bucket, F.lit(50.0)), F.lit(-50.0))
        z = F.greatest(F.least(z_global + F.lit(lam) * (z_bucket - z_global), F.lit(50.0)), F.lit(-50.0))
        return F.lit(1.0) / (F.lit(1.0) + F.exp(-z))
    return F.col(score_col).cast("double")


def resolve_route_blend_params(model_bucket: dict[str, Any]) -> tuple[dict[str, float], float]:
    raw = model_bucket.get("route_quality_calibration", {})
    weights_raw = raw.get("weights", {}) if isinstance(raw, dict) else {}
    weights = {k: float(weights_raw.get(k, 1.0)) for k in ROUTE_KEYS}
    gamma = float(raw.get("gamma", DEFAULT_ROUTE_BLEND_GAMMA)) if isinstance(raw, dict) else float(DEFAULT_ROUTE_BLEND_GAMMA)
    return weights, gamma


def build_route_factor_expr(route_weights: dict[str, float]) -> Any:
    denom = F.greatest(F.col("source_count").cast("double"), F.lit(1.0))
    total = F.lit(0.0)
    for route in ROUTE_KEYS:
        col = F.col(f"has_{route}").cast("double")
        total = total + col * F.lit(float(route_weights.get(route, 1.0)))
    fac = total / denom
    return F.greatest(F.least(fac, F.lit(1.5)), F.lit(0.5))


def compute_recall_ndcg(rank_df: DataFrame) -> tuple[float, float]:
    metrics = (
        rank_df.withColumn("recall", F.when(F.col("rank") > 0, F.lit(1.0)).otherwise(F.lit(0.0)))
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


def eval_from_pred(
    pred_topk: DataFrame,
    truth: DataFrame,
    candidate_pool: DataFrame,
    total_train_events: int,
) -> tuple[float, float, float, float, float, float]:
    eval_df = (
        truth.join(
            pred_topk.select("user_idx", F.col("item_idx").alias("pred_item"), "rank"),
            on="user_idx",
            how="left",
        )
        .withColumn("rank", F.when(F.col("pred_item") == F.col("true_item_idx"), F.col("rank")).otherwise(F.lit(0)))
        .groupBy("user_idx")
        .agg(F.max("rank").alias("rank"))
    )
    recall, ndcg = compute_recall_ndcg(eval_df)

    n_users = int(truth.select("user_idx").distinct().count())
    pred_users = int(pred_topk.select("user_idx").distinct().count())
    user_cov = float(pred_users / max(1, n_users))

    n_items_pool = int(candidate_pool.select("item_idx").distinct().count())
    n_items_pred = int(pred_topk.select("item_idx").distinct().count())
    item_cov = float(n_items_pred / max(1, n_items_pool))

    item_pop = (
        candidate_pool.select("item_idx", "item_train_pop_count")
        .dropDuplicates(["item_idx"])
        .filter(F.col("item_train_pop_count").isNotNull())
    )
    quant = item_pop.approxQuantile("item_train_pop_count", [TAIL_QUANTILE], 0.01)
    tail_cutoff = float(quant[0]) if quant else 0.0

    pred_diag = (
        pred_topk.join(item_pop, on="item_idx", how="left")
        .fillna({"item_train_pop_count": 1.0})
        .withColumn(
            "pop_prob",
            F.greatest(F.col("item_train_pop_count").cast("double") / F.lit(float(max(1, total_train_events))), F.lit(1e-12)),
        )
        .withColumn("is_tail", F.when(F.col("item_train_pop_count") <= F.lit(tail_cutoff), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("novelty", -(F.log(F.col("pop_prob")) / F.log(F.lit(2.0))))
    )
    row = pred_diag.agg(F.avg("is_tail").alias("tail_cov"), F.avg("novelty").alias("novelty")).first()
    tail_cov = float(row["tail_cov"] or 0.0)
    novelty = float(row["novelty"] or 0.0)
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty


def build_learned_score_expr(feature_cols: list[str], mean: list[float], std: list[float], coef: list[float], intercept: float) -> Any:
    linear = F.lit(float(intercept))
    for i, name in enumerate(feature_cols):
        s = float(std[i]) if abs(float(std[i])) > 1e-12 else 1.0
        m = float(mean[i])
        w = float(coef[i])
        z = (F.col(name).cast("double") - F.lit(m)) / F.lit(s)
        linear = linear + F.lit(w) * z
    # Sigmoid probability for ranking score.
    return F.lit(1.0) / (F.lit(1.0) + F.exp(-linear))


def infer_topk_pdf_with_xgboost(
    cand_pdf: pd.DataFrame,
    feature_cols: list[str],
    model_path: Path,
    model_backend: str,
    blend_alpha: float,
    blend_mode: str,
    rerank_topn: int,
    calibration: dict[str, Any] | None,
    route_weights: dict[str, float],
    route_gamma: float,
    top_k: int,
    return_scored: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    backend = str(model_backend or "").strip().lower()
    if backend == "xgboost_ranker":
        if XGBRanker is None:
            raise RuntimeError("xgboost is not installed for xgboost_ranker inference backend")
        model = XGBRanker()
        model.load_model(model_path.as_posix())
        is_ranker = True
    else:
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed for xgboost_cls inference backend")
        model = XGBClassifier()
        model.load_model(model_path.as_posix())
        is_ranker = False
    if cand_pdf.empty:
        empty_topk = pd.DataFrame(columns=["user_idx", "item_idx", "rank"])
        return empty_topk, (pd.DataFrame() if return_scored else None), {
            "requested_blend_mode": normalize_blend_mode(blend_mode),
            "effective_blend_mode": normalize_blend_mode(blend_mode),
            "blend_guard_triggered": False,
            "score_std": 0.0,
            "score_p95_p05": 0.0,
            "score_unique_6dp": 0.0,
        }

    pdf = cand_pdf.copy()
    x = pdf[feature_cols].to_numpy(dtype=np.float32)
    if is_ranker:
        pdf["learned_score"] = model.predict(x).astype(np.float64)
    else:
        pdf["learned_score"] = model.predict_proba(x)[:, 1].astype(np.float64)
    pdf["learned_score_for_blend"] = apply_platt_numpy(pdf["learned_score"].to_numpy(dtype=np.float64), calibration)
    pdf["pre_prior"] = 1.0 / np.log2(pdf["pre_rank"].to_numpy(dtype=np.float64) + 1.0)
    denom = np.maximum(pdf["source_count"].to_numpy(dtype=np.float64), 1.0)
    route_total = np.zeros(len(pdf), dtype=np.float64)
    for route in ROUTE_KEYS:
        c = f"has_{route}"
        if c in pdf.columns:
            route_total += pdf[c].to_numpy(dtype=np.float64) * float(route_weights.get(route, 1.0))
    route_factor = np.clip(route_total / denom, 0.5, 1.5)
    pdf["pre_prior_route"] = pdf["pre_prior"] * np.power(route_factor, float(route_gamma))
    requested_mode = normalize_blend_mode(blend_mode)
    guard_stats = _blend_guard_stats(pdf["learned_score_for_blend"].to_numpy(dtype=np.float64))
    guard_triggered = False
    effective_mode = requested_mode
    if requested_mode == "auto":
        guard_triggered = bool(
            RANK_BLEND_GUARD_ENABLE
            and (
                guard_stats["score_std"] < float(RANK_BLEND_GUARD_MIN_STD)
                or guard_stats["score_p95_p05"] < float(RANK_BLEND_GUARD_MIN_P95_P05)
                or guard_stats["score_unique_6dp"] < float(RANK_BLEND_GUARD_MIN_UNIQUE)
            )
        )
        effective_mode = "rank_prior" if guard_triggered else "prob"
    if effective_mode == "rank_prior":
        pdf["learned_model_signal"] = _rank_prior_from_score(pdf, "learned_score_for_blend")
    else:
        pdf["learned_model_signal"] = pdf["learned_score_for_blend"].to_numpy(dtype=np.float64)
    blend = (1.0 - float(blend_alpha)) * pdf["pre_prior_route"] + float(blend_alpha) * pdf["learned_model_signal"]
    if int(rerank_topn) > 0:
        # Safe rerank: keep pre-rank order outside topN and only rerank head candidates.
        use_model = (pdf["pre_rank"].to_numpy(dtype=np.int32) <= int(rerank_topn))
        pdf["learned_blend_score"] = np.where(use_model, blend, pdf["pre_prior"])
    else:
        pdf["learned_blend_score"] = blend
    pdf = pdf.sort_values(["user_idx", "learned_blend_score", "pre_score", "item_idx"], ascending=[True, False, False, True])
    pdf["rank"] = pdf.groupby("user_idx").cumcount() + 1
    topk = pdf[pdf["rank"] <= int(top_k)][["user_idx", "item_idx", "rank"]].copy()
    scored = None
    if return_scored:
        keep_cols = [
            "user_idx",
            "item_idx",
            "pre_score",
            "pre_rank",
            "learned_score",
            "learned_score_for_blend",
            "pre_prior",
            "pre_prior_route",
            "learned_model_signal",
            "learned_blend_score",
            "rank",
        ]
        keep_cols = [c for c in keep_cols if c in pdf.columns]
        scored = pdf[keep_cols].copy()
    blend_meta = {
        "requested_blend_mode": str(requested_mode),
        "effective_blend_mode": str(effective_mode),
        "blend_guard_triggered": bool(guard_triggered),
        "score_std": float(guard_stats["score_std"]),
        "score_p95_p05": float(guard_stats["score_p95_p05"]),
        "score_unique_6dp": float(guard_stats["score_unique_6dp"]),
    }
    return topk, scored, blend_meta


def eval_from_pred_pdf(
    pred_topk_pdf: pd.DataFrame,
    truth_pdf: pd.DataFrame,
    cand_pdf: pd.DataFrame,
    total_train_events: int,
) -> tuple[float, float, float, float, float, float]:
    if truth_pdf.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    t = truth_pdf[["user_idx", "true_item_idx"]].drop_duplicates("user_idx").copy()
    p = pred_topk_pdf[["user_idx", "item_idx", "rank"]].copy()
    m = p.merge(t, on="user_idx", how="right")
    m["hit_rank"] = np.where(m["item_idx"] == m["true_item_idx"], m["rank"], np.nan)
    rank_df = m.groupby("user_idx", as_index=False)["hit_rank"].min()
    rank = rank_df["hit_rank"].fillna(0).to_numpy(dtype=np.int32)
    recall = float((rank > 0).mean())
    if (rank > 0).any():
        ndcg = float((1.0 / np.log2(rank[rank > 0] + 1.0)).sum() / max(1, len(rank)))
    else:
        ndcg = 0.0

    n_users = int(t["user_idx"].nunique())
    pred_users = int(p["user_idx"].nunique())
    user_cov = float(pred_users / max(1, n_users))

    item_pool = cand_pdf[["item_idx"]].drop_duplicates()
    n_items_pool = int(item_pool["item_idx"].nunique())
    n_items_pred = int(p["item_idx"].nunique())
    item_cov = float(n_items_pred / max(1, n_items_pool))

    item_pop = cand_pdf[["item_idx", "item_train_pop_count"]].drop_duplicates("item_idx")
    item_pop["item_train_pop_count"] = item_pop["item_train_pop_count"].fillna(1.0).astype(float)
    tail_cutoff = float(item_pop["item_train_pop_count"].quantile(TAIL_QUANTILE)) if not item_pop.empty else 0.0
    pred_diag = p.merge(item_pop, on="item_idx", how="left")
    pred_diag["item_train_pop_count"] = pred_diag["item_train_pop_count"].fillna(1.0).astype(float)
    pred_diag["pop_prob"] = np.maximum(pred_diag["item_train_pop_count"].to_numpy(dtype=np.float64) / max(1.0, float(total_train_events)), 1e-12)
    pred_diag["is_tail"] = (pred_diag["item_train_pop_count"] <= tail_cutoff).astype(float)
    pred_diag["novelty"] = -np.log2(pred_diag["pop_prob"])
    tail_cov = float(pred_diag["is_tail"].mean()) if not pred_diag.empty else 0.0
    novelty = float(pred_diag["novelty"].mean()) if not pred_diag.empty else 0.0
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty


def eval_from_pred_pdf_compact(
    pred_topk_pdf: pd.DataFrame,
    truth_pdf: pd.DataFrame,
    n_items_pool: int,
    item_pop_pdf: pd.DataFrame,
    total_train_events: int,
) -> tuple[float, float, float, float, float, float]:
    if truth_pdf.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    t = truth_pdf[["user_idx", "true_item_idx"]].drop_duplicates("user_idx").copy()
    p = pred_topk_pdf[["user_idx", "item_idx", "rank"]].copy()
    m = p.merge(t, on="user_idx", how="right")
    m["hit_rank"] = np.where(m["item_idx"] == m["true_item_idx"], m["rank"], np.nan)
    rank_df = m.groupby("user_idx", as_index=False)["hit_rank"].min()
    rank = rank_df["hit_rank"].fillna(0).to_numpy(dtype=np.int32)
    recall = float((rank > 0).mean())
    if (rank > 0).any():
        ndcg = float((1.0 / np.log2(rank[rank > 0] + 1.0)).sum() / max(1, len(rank)))
    else:
        ndcg = 0.0

    n_users = int(t["user_idx"].nunique())
    pred_users = int(p["user_idx"].nunique())
    user_cov = float(pred_users / max(1, n_users))

    n_items_pred = int(p["item_idx"].nunique())
    item_cov = float(n_items_pred / max(1, int(n_items_pool)))

    ip = item_pop_pdf[["item_idx", "item_train_pop_count"]].drop_duplicates("item_idx").copy()
    ip["item_train_pop_count"] = ip["item_train_pop_count"].fillna(1.0).astype(float)
    tail_cutoff = float(ip["item_train_pop_count"].quantile(TAIL_QUANTILE)) if not ip.empty else 0.0
    pred_diag = p.merge(ip, on="item_idx", how="left")
    pred_diag["item_train_pop_count"] = pred_diag["item_train_pop_count"].fillna(1.0).astype(float)
    pred_diag["pop_prob"] = np.maximum(pred_diag["item_train_pop_count"].to_numpy(dtype=np.float64) / max(1.0, float(total_train_events)), 1e-12)
    pred_diag["is_tail"] = (pred_diag["item_train_pop_count"] <= tail_cutoff).astype(float)
    pred_diag["novelty"] = -np.log2(pred_diag["pop_prob"])
    tail_cov = float(pred_diag["is_tail"].mean()) if not pred_diag.empty else 0.0
    novelty = float(pred_diag["novelty"].mean()) if not pred_diag.empty else 0.0
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty


def compute_bucket_diagnostics(
    bucket: int,
    top_k: int,
    truth_pdf: pd.DataFrame,
    truth_diag_pdf: pd.DataFrame,
    pre_topk_pdf: pd.DataFrame,
    learned_topk_pdf: pd.DataFrame | None,
) -> dict[str, Any]:
    truth_base = truth_pdf[["user_idx", "true_item_idx"]].drop_duplicates("user_idx").copy()
    diag = truth_diag_pdf.copy()
    if "truth_in_pool" not in diag.columns:
        diag["truth_in_pool"] = diag["truth_pre_rank"].notna().astype(int)
    for c in ("has_als", "has_cluster", "has_profile", "has_popular"):
        if c not in diag.columns:
            diag[c] = 0.0
        diag[c] = diag[c].fillna(0.0).astype(float)
    base = truth_base.merge(diag, on=["user_idx", "true_item_idx"], how="left")

    def _truth_rank(pred_pdf: pd.DataFrame | None, col_name: str) -> pd.DataFrame:
        out = truth_base[["user_idx"]].copy()
        out[col_name] = 0
        if pred_pdf is None or pred_pdf.empty:
            return out
        p = pred_pdf[["user_idx", "item_idx", "rank"]].copy()
        m = truth_base.merge(p, left_on=["user_idx", "true_item_idx"], right_on=["user_idx", "item_idx"], how="left")
        out[col_name] = m["rank"].fillna(0).astype(int)
        return out

    pre_rank_df = _truth_rank(pre_topk_pdf, "pre_hit_rank")
    learned_rank_df = _truth_rank(learned_topk_pdf, "learned_hit_rank")
    merged = base.merge(pre_rank_df, on="user_idx", how="left").merge(learned_rank_df, on="user_idx", how="left")
    merged["pre_hit_rank"] = merged["pre_hit_rank"].fillna(0).astype(int)
    merged["learned_hit_rank"] = merged["learned_hit_rank"].fillna(0).astype(int)

    n_users = int(len(merged))
    truth_in_pool = merged["truth_in_pool"].fillna(0).astype(int)
    oracle_recall = float(truth_in_pool.mean()) if n_users > 0 else 0.0
    oracle_ndcg = oracle_recall

    pre_hit = merged["pre_hit_rank"] > 0
    learned_hit = merged["learned_hit_rank"] > 0
    both_hit = pre_hit & learned_hit
    improved = (~pre_hit) & learned_hit
    degraded = pre_hit & (~learned_hit)
    both_rank_improved = both_hit & (merged["learned_hit_rank"] < merged["pre_hit_rank"])
    both_rank_degraded = both_hit & (merged["learned_hit_rank"] > merged["pre_hit_rank"])

    pre_map: dict[int, set[int]] = {}
    if pre_topk_pdf is not None and not pre_topk_pdf.empty:
        for u, g in pre_topk_pdf.groupby("user_idx")["item_idx"]:
            pre_map[int(u)] = set(int(x) for x in g.tolist())
    learned_map: dict[int, set[int]] = {}
    if learned_topk_pdf is not None and not learned_topk_pdf.empty:
        for u, g in learned_topk_pdf.groupby("user_idx")["item_idx"]:
            learned_map[int(u)] = set(int(x) for x in g.tolist())
    changed = 0
    jaccard_sum = 0.0
    for u in truth_base["user_idx"].tolist():
        uu = int(u)
        a = pre_map.get(uu, set())
        b = learned_map.get(uu, set())
        if a != b:
            changed += 1
        union = a | b
        if union:
            jaccard_sum += float(len(a & b) / len(union))
        else:
            jaccard_sum += 1.0
    swap_rate = float(changed / max(1, n_users))
    jaccard_mean = float(jaccard_sum / max(1, n_users))

    route_stats: dict[str, Any] = {}
    for route in ROUTE_KEYS:
        c = f"has_{route}"
        flag = merged[c] > 0.5
        pre_via = pre_hit & flag
        learned_via = learned_hit & flag
        route_stats[route] = {
            "truth_pool_rate": float((truth_in_pool.astype(bool) & flag).mean()) if n_users > 0 else 0.0,
            "pre_hit_user_rate": float(pre_via.mean()) if n_users > 0 else 0.0,
            "learned_hit_user_rate": float(learned_via.mean()) if n_users > 0 else 0.0,
            "pre_hit_share": float(pre_via.sum() / max(1, int(pre_hit.sum()))),
            "learned_hit_share": float(learned_via.sum() / max(1, int(learned_hit.sum()))),
        }

    truth_pre_rank_nonnull = merged["truth_pre_rank"].dropna()
    return {
        "bucket": int(bucket),
        "top_k": int(top_k),
        "n_users": int(n_users),
        "oracle": {
            "truth_in_pool_users": int(truth_in_pool.sum()),
            "oracle_recall_at_k": float(oracle_recall),
            "oracle_ndcg_at_k": float(oracle_ndcg),
            "truth_pre_rank_p50": None if truth_pre_rank_nonnull.empty else float(truth_pre_rank_nonnull.quantile(0.5)),
            "truth_pre_rank_p90": None if truth_pre_rank_nonnull.empty else float(truth_pre_rank_nonnull.quantile(0.9)),
            "truth_pre_rank_le_10_rate": float((merged["truth_pre_rank"].fillna(1e9) <= 10).mean()) if n_users > 0 else 0.0,
            "truth_pre_rank_le_50_rate": float((merged["truth_pre_rank"].fillna(1e9) <= 50).mean()) if n_users > 0 else 0.0,
            "truth_pre_rank_le_150_rate": float((merged["truth_pre_rank"].fillna(1e9) <= 150).mean()) if n_users > 0 else 0.0,
        },
        "pre_vs_learned": {
            "pre_hit_users": int(pre_hit.sum()),
            "learned_hit_users": int(learned_hit.sum()),
            "improved_users": int(improved.sum()),
            "degraded_users": int(degraded.sum()),
            "both_hit_users": int(both_hit.sum()),
            "both_rank_improved_users": int(both_rank_improved.sum()),
            "both_rank_degraded_users": int(both_rank_degraded.sum()),
            "topk_set_swap_rate": float(swap_rate),
            "topk_set_jaccard_mean": float(jaccard_mean),
        },
        "route_attribution": route_stats,
    }


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    ensure_results_file(METRICS_PATH)

    source_09 = resolve_stage09_run()
    model_json = resolve_rank_model_json()
    eval_user_cohort_path = resolve_eval_user_cohort_path()
    model_data = json.loads(model_json.read_text(encoding="utf-8"))
    models_by_bucket: dict[str, Any] = model_data.get("models_by_bucket", {})
    bucket_alpha_map = parse_bucket_float_map(BLEND_ALPHA_BY_BUCKET)
    bucket_blend_mode_map = parse_bucket_str_map(RANK_BLEND_MODE_BY_BUCKET)

    run_id_10 = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id_10}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_dirs = sorted([p for p in source_09.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    wanted = parse_bucket_override(BUCKETS_OVERRIDE)
    if wanted:
        bucket_dirs = [p for p in bucket_dirs if int(p.name.split("_")[-1]) in wanted]
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs under {source_09}")

    meta_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    blend_mode_rows: list[dict[str, Any]] = []
    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        print(f"[BUCKET] {bucket}")
        bucket_out = out_dir / bdir.name
        bucket_out.mkdir(parents=True, exist_ok=True)
        cand_path = bdir / "candidates_pretrim150.parquet"
        truth_path = bdir / "truth.parquet"
        meta_path = bdir / "bucket_meta.json"
        if not cand_path.exists() or not truth_path.exists() or not meta_path.exists():
            print(f"[WARN] skip bucket={bucket}: missing input files")
            continue

        model_bucket = models_by_bucket.get(str(bucket))
        bucket_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        total_train_events = int(bucket_meta.get("total_train_events", 1))

        cand_raw_base = spark.read.parquet(cand_path.as_posix())
        cand_raw = attach_history_features(cand_raw_base, bdir, model_bucket).persist(StorageLevel.DISK_ONLY)
        tower_score_scale, seq_score_scale, tower_inv_scale, seq_inv_scale = resolve_tower_seq_scales(model_bucket)
        cand = add_feature_columns(
            cand_raw,
            tower_score_scale=tower_score_scale,
            seq_score_scale=seq_score_scale,
            tower_inv_scale=tower_inv_scale,
            seq_inv_scale=seq_inv_scale,
        ).persist(StorageLevel.DISK_ONLY)
        truth = spark.read.parquet(truth_path.as_posix()).select("user_idx", "true_item_idx").dropDuplicates(["user_idx"]).persist(
            StorageLevel.DISK_ONLY
        )

        eval_user_count = None
        if eval_user_cohort_path is not None:
            cohort_users = load_eval_users_df(spark, eval_user_cohort_path)
            eval_user_count = int(cohort_users.count())
            if eval_user_count > 0:
                truth.unpersist()
                cand.unpersist()
                truth = truth.join(F.broadcast(cohort_users), on="user_idx", how="inner").persist(StorageLevel.DISK_ONLY)
                cand = cand.join(F.broadcast(cohort_users), on="user_idx", how="inner").persist(StorageLevel.DISK_ONLY)
                print(f"[COHORT] bucket={bucket} using explicit eval cohort={eval_user_cohort_path} users={eval_user_count}")
        elif model_bucket is not None:
            split_users = model_bucket.get("split_users", {}) if isinstance(model_bucket, dict) else {}
            test_users_rel = str(split_users.get("test_users_file", "")).strip() if isinstance(split_users, dict) else ""
            if test_users_rel:
                test_users_path = model_json.parent / test_users_rel
                if test_users_path.exists():
                    try:
                        test_users_sp = (
                            spark.read.csv(test_users_path.as_posix(), header=True, inferSchema=True)
                            .select(F.col("user_idx").cast("int").alias("user_idx"))
                            .filter(F.col("user_idx").isNotNull() & (F.col("user_idx") >= F.lit(0)))
                            .dropDuplicates(["user_idx"])
                        )
                        eval_user_count = int(test_users_sp.count())
                        if eval_user_count > 0:
                            truth.unpersist()
                            cand.unpersist()
                            truth = truth.join(test_users_sp, on="user_idx", how="inner").persist(StorageLevel.DISK_ONLY)
                            cand = cand.join(test_users_sp, on="user_idx", how="inner").persist(StorageLevel.DISK_ONLY)
                            print(f"[SPLIT] bucket={bucket} using test_users_file={test_users_rel} users={eval_user_count}")
                    except Exception as e:
                        print(f"[WARN] bucket={bucket} failed to load split users from {test_users_path}: {e}")

        if EVAL_CANDIDATE_TOPN > 0:
            cand.unpersist()
            cand = cand.filter(F.col("pre_rank") <= F.lit(int(EVAL_CANDIDATE_TOPN))).persist(StorageLevel.DISK_ONLY)
            print(f"[COHORT] bucket={bucket} using candidate_topn={int(EVAL_CANDIDATE_TOPN)}")

        n_users = int(truth.select("user_idx").distinct().count())
        n_items = int(cand.select("item_idx").distinct().count())
        n_candidates = int(cand.count())

        als_topk = (
            cand.filter(F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(TOP_K)))
            .select("user_idx", "item_idx", F.col("als_rank").cast("int").alias("rank"))
        )
        pre_topk = (
            cand.filter(F.col("pre_rank") <= F.lit(TOP_K))
            .select("user_idx", "item_idx", F.col("pre_rank").cast("int").alias("rank"))
        )
        pre_topk_pdf = pre_topk.toPandas() if DIAGNOSTICS_ENABLE else None
        truth_pdf = truth.select("user_idx", "true_item_idx").toPandas()
        truth_diag_pdf = None
        if DIAGNOSTICS_ENABLE:
            truth_diag = (
                truth.join(
                    cand.select(
                        "user_idx",
                        "item_idx",
                        F.col("pre_rank").cast("double").alias("truth_pre_rank"),
                        "has_als",
                        "has_cluster",
                        "has_profile",
                        "has_popular",
                    ),
                    (truth["user_idx"] == cand["user_idx"]) & (truth["true_item_idx"] == cand["item_idx"]),
                    how="left",
                )
                .select(
                    truth["user_idx"].alias("user_idx"),
                    truth["true_item_idx"].alias("true_item_idx"),
                    F.col("truth_pre_rank"),
                    F.when(F.col("truth_pre_rank").isNotNull(), F.lit(1)).otherwise(F.lit(0)).alias("truth_in_pool"),
                    F.coalesce(F.col("has_als").cast("double"), F.lit(0.0)).alias("has_als"),
                    F.coalesce(F.col("has_cluster").cast("double"), F.lit(0.0)).alias("has_cluster"),
                    F.coalesce(F.col("has_profile").cast("double"), F.lit(0.0)).alias("has_profile"),
                    F.coalesce(F.col("has_popular").cast("double"), F.lit(0.0)).alias("has_popular"),
                )
            )
            truth_diag_pdf = truth_diag.toPandas()

        als_metrics = eval_from_pred(als_topk, truth, cand, total_train_events)
        pre_metrics = eval_from_pred(pre_topk, truth, cand, total_train_events)

        metric_rows: list[tuple[str, tuple[float, float, float, float, float, float]]] = [
            ("ALS@10_from_candidates", als_metrics),
            ("PreScore@10", pre_metrics),
        ]
        learned_topk_pdf: pd.DataFrame | None = None
        audit_scored_pdf: pd.DataFrame | None = None
        calibration_source = "none"
        selected_calibration: dict[str, Any] | None = {"method": "none"}

        if model_bucket is not None:
            feature_cols = model_bucket.get("feature_columns", FEATURE_COLUMNS)
            backend = str(model_bucket.get("model_backend", "sklearn_lr")).strip().lower() or "sklearn_lr"
            if backend == "xgboost":
                # backward compatibility for old model payloads
                backend = "xgboost_cls"
            blend_alpha = float(model_bucket.get("metrics", {}).get("blend_alpha", 0.2))
            if BLEND_ALPHA_OVERRIDE:
                blend_alpha = float(BLEND_ALPHA_OVERRIDE)
            if bucket in bucket_alpha_map:
                blend_alpha = float(bucket_alpha_map[bucket])
            blend_mode_requested = normalize_blend_mode(bucket_blend_mode_map.get(bucket, RANK_BLEND_MODE))
            selected_calibration, calibration_source = resolve_calibration_payload(model_bucket, model_data)
            route_weights, route_gamma = resolve_route_blend_params(model_bucket)
            label_name = "LearnedBlendLR@10"
            learned_topk = None
            blend_mode_info = summarize_blend_infos([], blend_mode_requested)
            if backend in {"xgboost_cls", "xgboost_ranker"}:
                label_name = "LearnedBlendXGBCls@10" if backend == "xgboost_cls" else "LearnedBlendXGBRanker@10"
                if XGB_USER_BATCH_SIZE <= 0 and n_candidates > XGB_MAX_SCORE_ROWS:
                    print(
                        f"[WARN] bucket={bucket} {backend} inference skipped: "
                        f"rows={n_candidates} > XGB_MAX_SCORE_ROWS={XGB_MAX_SCORE_ROWS} and XGB_USER_BATCH_SIZE<=0"
                    )
                    learned_topk = None
                else:
                    model_rel = str(model_bucket.get("model_file", "")).strip()
                    if not model_rel:
                        print(f"[WARN] bucket={bucket} {backend} model_file missing, skip {label_name}")
                        learned_topk = None
                    else:
                        model_path = model_json.parent / model_rel
                        if not model_path.exists():
                            print(f"[WARN] bucket={bucket} {backend} model missing: {model_path}")
                            learned_topk = None
                        else:
                            use_cols: list[str] = []
                            for c in ["user_idx", "item_idx", "pre_score", "pre_rank", "item_train_pop_count", *feature_cols]:
                                if c not in use_cols:
                                    use_cols.append(c)
                            item_pop_pdf = cand.select("item_idx", "item_train_pop_count").dropDuplicates(["item_idx"]).toPandas()
                            if XGB_USER_BATCH_SIZE > 0:
                                pred_batches: list[pd.DataFrame] = []
                                audit_batches: list[pd.DataFrame] = []
                                blend_infos: list[dict[str, Any]] = []
                                n_batch_total = max(1, int(math.ceil(float(n_users) / max(1, XGB_USER_BATCH_SIZE))))
                                audit_per_batch = max(1, int(math.ceil(max(1, AUDIT_SAMPLE_ROWS) / n_batch_total)))
                                loop_meta: list[tuple[int, int | None, DataFrame]] = []
                                cand_batched = None
                                batch_root: Path | None = None
                                mode_effective = str(XGB_BATCH_MODE)
                                if mode_effective == "legacy_user_collect":
                                    print(
                                        f"[WARN] bucket={bucket} XGB_BATCH_MODE=legacy_user_collect is deprecated; "
                                        "fallback to hash_partition_disk"
                                    )
                                    mode_effective = "hash_partition_disk"
                                if mode_effective == "hash_partition_disk":
                                    batch_col = "_batch_id"
                                    batch_root_base = (
                                        _SPARK_TMP_CTX.scratch_root
                                        if _SPARK_TMP_CTX is not None
                                        else (Path(SPARK_LOCAL_DIR) / "_scratch")
                                    )
                                    batch_root_base.mkdir(parents=True, exist_ok=True)
                                    batch_root = batch_root_base / f"stage10_2_xgb_batches_b{bucket}_{uuid.uuid4().hex}"
                                    if batch_root.exists():
                                        shutil.rmtree(batch_root, ignore_errors=True)
                                    (
                                        cand.select(*use_cols)
                                        .withColumn(
                                            batch_col,
                                            F.pmod(F.xxhash64(F.col("user_idx")), F.lit(int(n_batch_total))).cast("int"),
                                        )
                                        .write.mode("overwrite")
                                        .partitionBy(batch_col)
                                        .parquet(batch_root.as_posix())
                                    )
                                    for bid in range(n_batch_total):
                                        part_dir = batch_root / f"{batch_col}={int(bid)}"
                                        if not part_dir.exists():
                                            continue
                                        loop_meta.append((bid, None, spark.read.parquet(part_dir.as_posix()).select(*use_cols)))
                                elif mode_effective == "hash_partition_memory":
                                    batch_col = "_batch_id"
                                    cand_batched = (
                                        cand.select(*use_cols)
                                        .withColumn(
                                            batch_col,
                                            F.pmod(F.xxhash64(F.col("user_idx")), F.lit(int(n_batch_total))).cast("int"),
                                        )
                                        .repartition(int(n_batch_total), F.col(batch_col))
                                        .persist(StorageLevel.DISK_ONLY)
                                    )
                                    for bid in range(n_batch_total):
                                        loop_meta.append(
                                            (bid, None, cand_batched.filter(F.col(batch_col) == F.lit(int(bid))).drop(batch_col))
                                        )
                                else:
                                    raise ValueError(
                                        f"unsupported XGB_BATCH_MODE={XGB_BATCH_MODE}; expected hash_partition_disk/hash_partition_memory"
                                    )
                                processed_users = 0
                                try:
                                    for batch_id, batch_user_cnt, batch_df in loop_meta:
                                        batch_pdf = batch_df.toPandas()
                                        if batch_user_cnt is None:
                                            batch_user_cnt = int(batch_pdf["user_idx"].nunique()) if not batch_pdf.empty else 0
                                        processed_users += int(batch_user_cnt)
                                        if batch_pdf.empty:
                                            continue
                                        topk_pdf, scored_pdf, blend_info = infer_topk_pdf_with_xgboost(
                                            cand_pdf=batch_pdf,
                                            feature_cols=feature_cols,
                                            model_path=model_path,
                                            model_backend=backend,
                                            blend_alpha=blend_alpha,
                                            blend_mode=blend_mode_requested,
                                            rerank_topn=RERANK_TOPN_OVERRIDE,
                                            calibration=selected_calibration,
                                            route_weights=route_weights,
                                            route_gamma=route_gamma,
                                            top_k=TOP_K,
                                            return_scored=AUDIT_ENABLE,
                                        )
                                        blend_infos.append(blend_info)
                                        if not topk_pdf.empty:
                                            pred_batches.append(topk_pdf)
                                        if AUDIT_ENABLE and scored_pdf is not None and not scored_pdf.empty:
                                            if len(scored_pdf) > audit_per_batch:
                                                scored_pdf = scored_pdf.sample(n=audit_per_batch, random_state=42)
                                            audit_batches.append(scored_pdf)
                                        if (batch_id + 1) % 5 == 0 or (batch_id + 1) >= n_batch_total:
                                            print(
                                                f"[XGB-BATCH] bucket={bucket} backend={backend} "
                                                f"users={min(processed_users, n_users)}/{n_users}"
                                            )
                                finally:
                                    if mode_effective == "hash_partition_disk" and batch_root is not None:
                                        shutil.rmtree(batch_root, ignore_errors=True)
                                    if mode_effective == "hash_partition_memory" and cand_batched is not None:
                                        cand_batched.unpersist()
                                blend_mode_info = summarize_blend_infos(blend_infos, blend_mode_requested)
                                if pred_batches:
                                    pred_pdf = pd.concat(pred_batches, ignore_index=True)
                                    learned_topk_pdf = pred_pdf[["user_idx", "item_idx", "rank"]].copy()
                                    learned_metrics = eval_from_pred_pdf_compact(
                                        pred_topk_pdf=pred_pdf,
                                        truth_pdf=truth_pdf,
                                        n_items_pool=n_items,
                                        item_pop_pdf=item_pop_pdf,
                                        total_train_events=total_train_events,
                                    )
                                    metric_rows.append((label_name, learned_metrics))
                                    if AUDIT_ENABLE and audit_batches:
                                        audit_scored_pdf = pd.concat(audit_batches, ignore_index=True)
                                else:
                                    learned_topk = None
                            else:
                                cand_pdf = cand.select(*use_cols).toPandas()
                                topk_pdf, scored_pdf, blend_info = infer_topk_pdf_with_xgboost(
                                    cand_pdf=cand_pdf,
                                    feature_cols=feature_cols,
                                    model_path=model_path,
                                    model_backend=backend,
                                    blend_alpha=blend_alpha,
                                    blend_mode=blend_mode_requested,
                                    rerank_topn=RERANK_TOPN_OVERRIDE,
                                    calibration=selected_calibration,
                                    route_weights=route_weights,
                                    route_gamma=route_gamma,
                                    top_k=TOP_K,
                                    return_scored=AUDIT_ENABLE,
                                )
                                blend_mode_info = summarize_blend_infos([blend_info], blend_mode_requested)
                                learned_topk_pdf = topk_pdf[["user_idx", "item_idx", "rank"]].copy()
                                learned_metrics = eval_from_pred_pdf_compact(
                                    pred_topk_pdf=topk_pdf,
                                    truth_pdf=truth_pdf,
                                    n_items_pool=n_items,
                                    item_pop_pdf=item_pop_pdf,
                                    total_train_events=total_train_events,
                                )
                                metric_rows.append((label_name, learned_metrics))
                                if AUDIT_ENABLE and scored_pdf is not None and not scored_pdf.empty:
                                    if len(scored_pdf) > AUDIT_SAMPLE_ROWS:
                                        scored_pdf = scored_pdf.sample(n=AUDIT_SAMPLE_ROWS, random_state=42)
                                    audit_scored_pdf = scored_pdf
            else:
                mean = model_bucket["mean"]
                std = model_bucket["std"]
                coef = model_bucket["coef"]
                intercept = float(model_bucket["intercept"])
                score_expr = build_learned_score_expr(feature_cols, mean, std, coef, intercept)
                score_for_blend_expr = build_platt_expr("learned_score", selected_calibration)
                route_factor_expr = build_route_factor_expr(route_weights)
                w_user = Window.partitionBy("user_idx").orderBy(F.desc("learned_blend_score"), F.desc("pre_score"), F.asc("item_idx"))
                blend_mode_effective = (
                    "rank_prior"
                    if blend_mode_requested == "rank_prior"
                    else ("prob" if blend_mode_requested in {"prob", "auto"} else "prob")
                )
                if blend_mode_requested == "auto":
                    print(f"[WARN] bucket={bucket} backend={backend} blend_mode=auto uses prob path (guard only on xgboost backends)")
                scored_base = (
                    cand.withColumn("learned_score", score_expr)
                    .withColumn("learned_score_for_blend", score_for_blend_expr)
                    .withColumn("pre_prior", F.lit(1.0) / (F.log(F.col("pre_rank").cast("double") + F.lit(1.0)) / F.log(F.lit(2.0))))
                    .withColumn("route_factor", route_factor_expr)
                    .withColumn("pre_prior_route", F.col("pre_prior") * F.pow(F.col("route_factor"), F.lit(float(route_gamma))))
                )
                if blend_mode_effective == "rank_prior":
                    w_model = Window.partitionBy("user_idx").orderBy(F.desc("learned_score_for_blend"), F.desc("pre_score"), F.asc("item_idx"))
                    scored_base = (
                        scored_base
                        .withColumn("learned_model_rank", F.row_number().over(w_model))
                        .withColumn(
                            "learned_model_signal",
                            F.lit(1.0)
                            / (F.log(F.col("learned_model_rank").cast("double") + F.lit(1.0)) / F.log(F.lit(2.0))),
                        )
                    )
                else:
                    scored_base = scored_base.withColumn("learned_model_signal", F.col("learned_score_for_blend"))
                scored = (
                    scored_base
                    .withColumn(
                        "learned_blend_raw",
                        F.lit(1.0 - float(blend_alpha)) * F.col("pre_prior_route")
                        + F.lit(float(blend_alpha)) * F.col("learned_model_signal"),
                    )
                    .withColumn(
                        "learned_blend_score",
                        F.when(
                            F.lit(int(RERANK_TOPN_OVERRIDE) > 0) & (F.col("pre_rank") > F.lit(int(RERANK_TOPN_OVERRIDE))),
                            F.col("pre_prior"),
                        ).otherwise(F.col("learned_blend_raw")),
                    )
                    .withColumn("learned_rank", F.row_number().over(w_user))
                )
                blend_mode_info = summarize_blend_infos(
                    [
                        {
                            "requested_blend_mode": blend_mode_requested,
                            "effective_blend_mode": blend_mode_effective,
                            "blend_guard_triggered": False,
                            "score_std": 0.0,
                            "score_p95_p05": 0.0,
                            "score_unique_6dp": 0.0,
                        }
                    ],
                    blend_mode_requested,
                )
                learned_topk = scored.filter(F.col("learned_rank") <= F.lit(TOP_K)).select(
                    "user_idx", "item_idx", F.col("learned_rank").cast("int").alias("rank")
                )
                learned_topk_pdf = learned_topk.toPandas()
                if AUDIT_ENABLE:
                    audit_cols = [
                        "user_idx",
                        "item_idx",
                        "pre_score",
                        "pre_rank",
                        "learned_score",
                        "learned_score_for_blend",
                        "pre_prior",
                        "pre_prior_route",
                        "learned_model_signal",
                        "learned_blend_score",
                    ]
                    audit_df = (
                        scored.select(*audit_cols)
                        .orderBy(F.rand(seed=42))
                        .limit(int(max(1, AUDIT_SAMPLE_ROWS)))
                        .toPandas()
                    )
                    audit_scored_pdf = audit_df
            if backend not in {"xgboost_cls", "xgboost_ranker"} and learned_topk is not None:
                learned_metrics = eval_from_pred(learned_topk, truth, cand, total_train_events)
                metric_rows.append((label_name, learned_metrics))
                print(
                    f"[METRIC] bucket={bucket} backend={backend} ALS_ndcg={als_metrics[1]:.4f} "
                    f"pre_ndcg={pre_metrics[1]:.4f} learned_ndcg={learned_metrics[1]:.4f} "
                    f"alpha={blend_alpha:.4f} rerank_topn={int(RERANK_TOPN_OVERRIDE)} "
                    f"blend_req={blend_mode_info.get('requested_blend_mode')} "
                    f"blend_eff={blend_mode_info.get('effective_blend_mode')} "
                    f"guard_rate={float(blend_mode_info.get('blend_guard_triggered_rate', 0.0)):.3f} "
                    f"cal={calibration_source}"
                )
            elif backend in {"xgboost_cls", "xgboost_ranker"} and any(x[0] == label_name for x in metric_rows):
                learned_metrics = [x[1] for x in metric_rows if x[0] == label_name][0]
                print(
                    f"[METRIC] bucket={bucket} backend={backend} ALS_ndcg={als_metrics[1]:.4f} "
                    f"pre_ndcg={pre_metrics[1]:.4f} learned_ndcg={learned_metrics[1]:.4f} "
                    f"alpha={blend_alpha:.4f} rerank_topn={int(RERANK_TOPN_OVERRIDE)} "
                    f"blend_req={blend_mode_info.get('requested_blend_mode')} "
                    f"blend_eff={blend_mode_info.get('effective_blend_mode')} "
                    f"guard_rate={float(blend_mode_info.get('blend_guard_triggered_rate', 0.0)):.3f} "
                    f"cal={calibration_source}"
                )
            else:
                learned_metrics = None
                print(f"[WARN] bucket={bucket} skip learned model metrics for backend={backend}")
            if backend not in {"xgboost_cls", "xgboost_ranker"} and learned_metrics is not None:
                bucket_out = out_dir / bdir.name
                bucket_out.mkdir(parents=True, exist_ok=True)
                scored.select("user_idx", "item_idx", "pre_score", "pre_rank", "learned_score", "learned_blend_score", "learned_rank").write.mode(
                    "overwrite"
                ).parquet((bucket_out / "learned_scored.parquet").as_posix())
                learned_topk.write.mode("overwrite").parquet((bucket_out / "learned_top10.parquet").as_posix())
            if AUDIT_ENABLE and audit_scored_pdf is not None and not audit_scored_pdf.empty:
                audit_payload = build_score_audit(
                    scored_pdf=audit_scored_pdf,
                    truth_pdf=truth_pdf,
                    sample_rows=AUDIT_SAMPLE_ROWS,
                    n_bins=AUDIT_BINS,
                    calibration_source=calibration_source,
                    calibration=selected_calibration,
                )
                (bucket_out / "score_audit.json").write_text(json.dumps(audit_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                print(
                    f"[AUDIT] bucket={bucket} rows={audit_payload['n_rows_audited']} "
                    f"pos_rate={audit_payload['positive_rate_audited']:.6f} "
                    f"ece_final={audit_payload['calibration_metrics'].get('final', {}).get('ece')}"
                )
        else:
            print(f"[WARN] bucket={bucket} model missing in {model_json.name}, skip learned model")

        if DIAGNOSTICS_ENABLE and pre_topk_pdf is not None and truth_diag_pdf is not None:
            diag = compute_bucket_diagnostics(
                bucket=bucket,
                top_k=TOP_K,
                truth_pdf=truth_pdf,
                truth_diag_pdf=truth_diag_pdf,
                pre_topk_pdf=pre_topk_pdf,
                learned_topk_pdf=learned_topk_pdf,
            )
            (bucket_out / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=True, indent=2), encoding="utf-8")
            diag_rows.append(diag)
            pv = diag["pre_vs_learned"]
            print(
                f"[DIAG] bucket={bucket} oracle_recall={diag['oracle']['oracle_recall_at_k']:.4f} "
                f"swap_rate={pv['topk_set_swap_rate']:.4f} improved={pv['improved_users']} degraded={pv['degraded_users']}"
            )

        if model_bucket is not None:
            blend_mode_rows.append(
                {
                    "bucket": int(bucket),
                    "backend": str(backend),
                    "blend_alpha": float(blend_alpha),
                    "requested_blend_mode": str(blend_mode_info.get("requested_blend_mode", blend_mode_requested)),
                    "effective_blend_mode": str(blend_mode_info.get("effective_blend_mode", blend_mode_requested)),
                    "effective_mode_counts": dict(blend_mode_info.get("effective_mode_counts", {})),
                    "blend_guard_triggered_rate": float(blend_mode_info.get("blend_guard_triggered_rate", 0.0)),
                    "score_std_mean": float(blend_mode_info.get("score_std_mean", 0.0)),
                    "score_p95_p05_mean": float(blend_mode_info.get("score_p95_p05_mean", 0.0)),
                    "score_unique_6dp_mean": float(blend_mode_info.get("score_unique_6dp_mean", 0.0)),
                }
            )

        for model_name, m in metric_rows:
            row = {
                "run_id_10": run_id_10,
                "source_run_09": source_09.name,
                "bucket_min_train_reviews": bucket,
                "model": model_name,
                "recall_at_k": f"{m[0]:.6f}",
                "ndcg_at_k": f"{m[1]:.6f}",
                "user_coverage_at_k": f"{m[2]:.6f}",
                "item_coverage_at_k": f"{m[3]:.6f}",
                "tail_coverage_at_k": f"{m[4]:.6f}",
                "novelty_at_k": f"{m[5]:.6f}",
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_candidates,
            }
            append_result(METRICS_PATH, row)
            meta_rows.append(row)

        cand_raw.unpersist()
        cand.unpersist()
        truth.unpersist()

    diagnostics_summary_path = out_dir / "diagnostics_summary.json"
    if DIAGNOSTICS_ENABLE:
        diagnostics_summary_path.write_text(json.dumps(diag_rows, ensure_ascii=True, indent=2), encoding="utf-8")

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id_10": run_id_10,
                "run_tag": RUN_TAG,
                "source_stage09_run": str(source_09),
                "rank_model_json": str(model_json),
                "eval_user_cohort_path": str(eval_user_cohort_path) if eval_user_cohort_path is not None else "",
                "eval_candidate_topn": int(EVAL_CANDIDATE_TOPN),
                "top_k": TOP_K,
                "blend_alpha_override": BLEND_ALPHA_OVERRIDE,
                "blend_alpha_by_bucket": bucket_alpha_map,
                "rank_blend_mode": normalize_blend_mode(RANK_BLEND_MODE),
                "rank_blend_mode_by_bucket": {str(k): normalize_blend_mode(v) for k, v in bucket_blend_mode_map.items()},
                "rank_blend_guard_enable": bool(RANK_BLEND_GUARD_ENABLE),
                "rank_blend_guard_min_std": float(RANK_BLEND_GUARD_MIN_STD),
                "rank_blend_guard_min_p95_p05": float(RANK_BLEND_GUARD_MIN_P95_P05),
                "rank_blend_guard_min_unique": int(RANK_BLEND_GUARD_MIN_UNIQUE),
                "rerank_topn_override": int(RERANK_TOPN_OVERRIDE),
                "xgb_batch_mode": str(XGB_BATCH_MODE),
                "xgb_user_batch_size": int(XGB_USER_BATCH_SIZE),
                "default_route_blend_gamma": float(DEFAULT_ROUTE_BLEND_GAMMA),
                "calibration_scope": normalize_calibration_scope(CALIBRATION_SCOPE),
                "calib_residual_shrink_tau": float(CALIB_RESIDUAL_SHRINK_TAU),
                "calib_residual_min_pos": int(CALIB_RESIDUAL_MIN_POS),
                "calib_residual_lambda_min": float(CALIB_RESIDUAL_MIN_LAMBDA),
                "calib_residual_lambda_max": float(CALIB_RESIDUAL_MAX_LAMBDA),
                "audit_enable": bool(AUDIT_ENABLE),
                "diagnostics_enable": bool(DIAGNOSTICS_ENABLE),
                "audit_sample_rows": int(AUDIT_SAMPLE_ROWS),
                "audit_bins": int(AUDIT_BINS),
                "spark_master": str(SPARK_MASTER),
                "spark_driver_memory": str(SPARK_DRIVER_MEMORY),
                "spark_executor_memory": str(SPARK_EXECUTOR_MEMORY),
                "spark_local_dir": str(SPARK_LOCAL_DIR),
                "spark_sql_shuffle_partitions": str(SPARK_SQL_SHUFFLE_PARTITIONS),
                "spark_default_parallelism": str(SPARK_DEFAULT_PARALLELISM),
                "spark_sql_adaptive_enabled": bool(SPARK_SQL_ADAPTIVE_ENABLED),
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
                        "cleanup_summary": dict(_SPARK_TMP_CTX.cleanup_summary),
                    }
                    if _SPARK_TMP_CTX is not None
                    else {}
                ),
                "buckets_override": sorted(wanted) if wanted else [],
                "diagnostics_summary_json": str(diagnostics_summary_path) if DIAGNOSTICS_ENABLE else "",
                "blend_mode_summary": blend_mode_rows,
                "rows": meta_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {out_dir}")
    print(f"[INFO] appended metrics to {METRICS_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
