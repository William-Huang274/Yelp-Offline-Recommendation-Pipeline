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
    build_history_anchor_summary,
    clean_text,
    extract_user_evidence_text,
    keyword_match_score,
)


RUN_TAG = "stage11_1_qlora_build_dataset"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_DATA_ROOT_DIR", "data/output/11_qlora_data")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
CANDIDATE_FILE = os.getenv("TRAIN_CANDIDATE_FILE", "candidates_pretrim150.parquet").strip() or "candidates_pretrim150.parquet"
TOPN_PER_USER = int(os.getenv("QLORA_TOPN_PER_USER", "120").strip() or 120)
ENABLE_PAIRWISE_POOL_EXPORT = os.getenv("QLORA_ENABLE_PAIRWISE_POOL_EXPORT", "false").strip().lower() == "true"
PAIRWISE_POOL_TOPN = int(os.getenv("QLORA_PAIRWISE_POOL_TOPN", "150").strip() or 150)
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
INCLUDE_VALID_POS = os.getenv("QLORA_INCLUDE_VALID_POS", "true").strip().lower() == "true"
VALID_POS_WEIGHT = float(os.getenv("QLORA_VALID_POS_WEIGHT", "0.35").strip() or 0.35)
EVAL_USER_FRAC = float(os.getenv("QLORA_EVAL_USER_FRAC", "0.1").strip() or 0.1)
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
MAX_ROWS_PER_BUCKET = int(os.getenv("QLORA_MAX_ROWS_PER_BUCKET", "300000").strip() or 300000)
ROW_CAP_ORDERED = os.getenv("QLORA_ROW_CAP_ORDERED", "false").strip().lower() == "true"

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

PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "full").strip().lower() or "full"  # "full" | "full_lite" | "semantic" | "sft_clean"
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
RICH_SFT_NEG_NEAR = int(os.getenv("QLORA_RICH_SFT_NEG_NEAR", "2").strip() or 2)
RICH_SFT_NEG_MID = int(os.getenv("QLORA_RICH_SFT_NEG_MID", "1").strip() or 1)
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
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
SPARK_PYTHON_AUTH_SOCKET_TIMEOUT = os.getenv("SPARK_PYTHON_AUTH_SOCKET_TIMEOUT", "120s").strip() or "120s"
SPARK_PYTHON_UNIX_DOMAIN_SOCKET_ENABLED = (
    os.getenv("SPARK_PYTHON_UNIX_DOMAIN_SOCKET_ENABLED", "false").strip().lower() == "true"
)
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
SPARK_DRIVER_BIND_ADDRESS = os.getenv("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1").strip() or "127.0.0.1"
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


def resolve_stage09_meta_path(raw_path: str) -> Path:
    p = normalize_legacy_project_path(str(raw_path or "").strip())
    if p.exists():
        return p
    return p


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    worker_py, driver_py = _configure_pyspark_python()
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
        f"pyspark_driver_python={driver_py}"
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
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.unix.domain.socket.enabled", "true" if SPARK_PYTHON_UNIX_DOMAIN_SOCKET_ENABLED else "false")
        .config("spark.python.authenticate.socketTimeout", SPARK_PYTHON_AUTH_SOCKET_TIMEOUT)
        .config("spark.pyspark.python", worker_py)
        .config("spark.pyspark.driver.python", driver_py)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def choose_candidate_file(bucket_dir: Path) -> Path:
    candidates = [
        CANDIDATE_FILE,
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates_pretrim.parquet",
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


def has_rows(df: DataFrame) -> bool:
    return bool(df.limit(1).collect())


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


def pre_rank_band_expr(col_name: str = "pre_rank") -> Any:
    col = F.col(col_name).cast("int")
    return (
        F.when(col <= F.lit(10), F.lit("001_010"))
        .when(col <= F.lit(30), F.lit("011_030"))
        .when(col <= F.lit(80), F.lit("031_080"))
        .when(col <= F.lit(150), F.lit("081_150"))
        .otherwise(F.lit("151_plus"))
    )


def project_candidate_rows(cand_raw: DataFrame) -> DataFrame:
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

    return cand_raw.select(
        "user_idx",
        "item_idx",
        "business_id",
        "pre_rank",
        "pre_score",
        pre_rank_band_expr("pre_rank").alias("pre_rank_band"),
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


def build_user_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    if not profile_path:
        return spark.createDataFrame(
            [],
            "user_id string, profile_text string, profile_text_evidence string, profile_top_pos_tags string, profile_top_neg_tags string, profile_confidence double",
        )
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame(
            [],
            "user_id string, profile_text string, profile_text_evidence string, profile_top_pos_tags string, profile_top_neg_tags string, profile_confidence double",
        )
    pdf = (
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
    return pdf


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
    item_sem_df = build_item_sem_df(spark, meta)
    cluster_profile_df = build_cluster_profile_df(spark, meta)
    business_meta_df = build_business_meta_df(spark, meta)
    history_anchor_text_udf = build_history_anchor_text_udf()
    rich_prompt_udf = None
    if ENABLE_RICH_SFT_EXPORT:
        if PROMPT_MODE == "semantic":
            rich_prompt_udf = build_prompt_udf_semantic()
        elif PROMPT_MODE == "sft_clean":
            rich_prompt_udf = build_prompt_udf_sft_clean()
        elif PROMPT_MODE == "full_lite":
            rich_prompt_udf = build_prompt_udf_full_lite()
        else:
            rich_prompt_udf = build_prompt_udf()
    if PROMPT_MODE == "semantic":
        prompt_udf = build_prompt_udf_semantic()
        print(f"[CONFIG] PROMPT_MODE=semantic (ranking features removed from prompt)")
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
        cand = project_candidate_rows(cand_raw)

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
                "name",
                "city",
                "categories",
                "primary_category",
                "source_set_text",
                "user_segment",
                "als_rank",
                "cluster_rank",
                "profile_rank",
                "popular_rank",
                "semantic_support",
                "semantic_tag_richness",
                "tower_score",
                "seq_score",
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
                    "name",
                    "city",
                    "categories",
                    "primary_category",
                    "source_set_text",
                    "user_segment",
                    "als_rank",
                    "cluster_rank",
                    "profile_rank",
                    "popular_rank",
                    "semantic_support",
                    "semantic_tag_richness",
                    "tower_score",
                    "seq_score",
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

        train_rows = pos_rows.select(
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
            "source_set_text",
            "user_segment",
            "als_rank",
            "cluster_rank",
            "profile_rank",
            "popular_rank",
            "semantic_support",
            "semantic_tag_richness",
            "tower_score",
            "seq_score",
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
                "name",
                "city",
                "categories",
                "primary_category",
                "source_set_text",
                "user_segment",
                "als_rank",
                "cluster_rank",
                "profile_rank",
                "popular_rank",
                "semantic_support",
                "semantic_tag_richness",
                "tower_score",
                "seq_score",
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

        if MAX_ROWS_PER_BUCKET > 0:
            if ROW_CAP_ORDERED:
                train_rows = train_rows.orderBy(F.col("user_idx").asc(), F.col("pre_rank").asc()).limit(int(MAX_ROWS_PER_BUCKET))
            else:
                train_rows = train_rows.limit(int(MAX_ROWS_PER_BUCKET))

        item_evidence_df = build_item_review_evidence(
            spark=spark,
            bucket_dir=bdir,
            row_df=train_rows.select("user_idx", "business_id"),
            item_sem_df=item_sem_df,
        )
        history_anchor_df = spark.createDataFrame(
            [],
            "user_idx int, history_anchor_entries array<struct<anchor_rank:int,business_id:string,anchor_text:string>>, history_anchor_available int",
        )
        hist_path = bdir / "train_history.parquet"
        if (ENABLE_RICH_SFT_EXPORT or PROMPT_MODE in {"sft_clean", "full_lite"}) and hist_path.exists():
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
        user_evidence_udf = build_user_evidence_udf()
        pair_evidence_udf = build_pair_alignment_udf()

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
            .withColumn(
                "history_anchor_count",
                F.when(F.length(F.col("history_anchor_text")) > F.lit(0), F.lit(1)).otherwise(F.lit(0)),
            )
        ).persist(StorageLevel.DISK_ONLY)

        if PROMPT_MODE == "semantic":
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

        out_df = (
            enrich.withColumn("prompt", prompt_col)
            .withColumn("target_text", F.when(F.col("label").cast("int") == F.lit(1), F.lit("YES")).otherwise(F.lit("NO")))
            .withColumn(
                "split",
                F.when(
                    (F.col("user_idx").cast("int") % F.lit(100)) < F.lit(int(100 * EVAL_USER_FRAC)),
                    F.lit("eval"),
                ).otherwise(F.lit("train")),
            )
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

        if not has_rows(out_df):
            print(f"[WARN] bucket={bucket} empty after enrich")
            out_df.unpersist(blocking=False)
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
        out_dirs = [train_dir, eval_dir, parquet_dir, user_evidence_dir, item_evidence_dir, pair_audit_dir]
        if ENABLE_RICH_SFT_EXPORT:
            out_dirs.extend([rich_sft_train_dir, rich_sft_eval_dir, rich_sft_parquet_dir])
        if ENABLE_PAIRWISE_POOL_EXPORT:
            out_dirs.extend([pairwise_pool_train_dir, pairwise_pool_eval_dir, pairwise_pool_parquet_dir])
        for d in out_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

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

        if ENABLE_RICH_SFT_EXPORT and rich_train_rows is not None and rich_prompt_udf is not None:
            rich_item_evidence_df = build_item_review_evidence(
                spark=spark,
                bucket_dir=bdir,
                row_df=rich_train_rows.select("user_idx", "business_id"),
                item_sem_df=item_sem_df,
            )
            rich_enrich = (
                rich_train_rows.join(user_profile_df, on="user_id", how="left")
                .join(item_sem_df, on="business_id", how="left")
                .join(cluster_profile_df, on="business_id", how="left")
                .join(rich_item_evidence_df, on=["user_idx", "business_id"], how="left")
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
            ).persist(StorageLevel.DISK_ONLY)

            if PROMPT_MODE == "semantic":
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
                rich_enrich.withColumn("prompt", rich_prompt_col)
                .withColumn("target_text", F.when(F.col("label").cast("int") == F.lit(1), F.lit("YES")).otherwise(F.lit("NO")))
                .withColumn(
                    "split",
                    F.when(
                        (F.col("user_idx").cast("int") % F.lit(100)) < F.lit(int(100 * EVAL_USER_FRAC)),
                        F.lit("eval"),
                    ).otherwise(F.lit("train")),
                )
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
                    "output_train_dir": str(rich_sft_train_dir),
                    "output_eval_dir": str(rich_sft_eval_dir),
                    "output_parquet_dir": str(rich_sft_parquet_dir),
                    "train_pos": int(rich_summary.get("train_label_1", 0)),
                    "train_neg": int(rich_summary.get("train_label_0", 0)),
                    "eval_pos": int(rich_summary.get("eval_label_1", 0)),
                    "eval_neg": int(rich_summary.get("eval_label_0", 0)),
                }
            )
            rich_sft_stat["rows_train"] = int(rich_sft_stat["train_pos"]) + int(rich_sft_stat["train_neg"])
            rich_sft_stat["rows_eval"] = int(rich_sft_stat["eval_pos"]) + int(rich_sft_stat["eval_neg"])
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
            cand_pool_raw = (
                spark.read.parquet(cand_file.as_posix())
                .join(users, on="user_idx", how="inner")
                .filter(F.col("pre_rank") <= F.lit(int(PAIRWISE_POOL_TOPN)))
            )
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
                    F.when(F.col("label") == F.lit(1), F.lit(False)).otherwise(F.col("pre_rank") <= F.lit(10)),
                )
                .withColumn(
                    "neg_tier",
                    F.when(F.col("label") == F.lit(1), F.lit("pos"))
                    .when(F.col("neg_is_near"), F.lit("near"))
                    .when(F.col("pre_rank") <= F.lit(10), F.lit("hard"))
                    .when(F.col("pre_rank") <= F.lit(80), F.lit("mid"))
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
                )
            )
            pool_item_evidence_df = build_item_review_evidence(
                spark=spark,
                bucket_dir=bdir,
                row_df=pool_rows.select("user_idx", "business_id"),
                item_sem_df=item_sem_df,
            )
            pool_enrich = (
                pool_rows.join(user_profile_df, on="user_id", how="left")
                .join(item_sem_df, on="business_id", how="left")
                .join(cluster_profile_df, on="business_id", how="left")
                .join(pool_item_evidence_df, on=["user_idx", "business_id"], how="left")
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
            ).persist(StorageLevel.DISK_ONLY)

            if PROMPT_MODE == "semantic":
                pool_prompt_col = prompt_udf(
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
            elif PROMPT_MODE == "full_lite":
                pool_prompt_col = prompt_udf(
                    F.col("profile_text"),
                    F.col("user_evidence_text"),
                    F.col("profile_top_pos_tags"),
                    F.col("profile_top_neg_tags"),
                    F.col("profile_confidence"),
                    F.lit(""),
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
                )
            elif PROMPT_MODE == "sft_clean":
                pool_prompt_col = prompt_udf(
                    F.col("profile_text"),
                    F.col("user_evidence_text"),
                    F.col("profile_top_pos_tags"),
                    F.col("profile_top_neg_tags"),
                    F.col("profile_confidence"),
                    F.lit(""),
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
                pool_prompt_col = prompt_udf(
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

            pairwise_pool_df = (
                pool_enrich.withColumn("prompt", pool_prompt_col)
                .withColumn("target_text", F.when(F.col("label").cast("int") == F.lit(1), F.lit("YES")).otherwise(F.lit("NO")))
                .withColumn(
                    "split",
                    F.when(
                        (F.col("user_idx").cast("int") % F.lit(100)) < F.lit(int(100 * EVAL_USER_FRAC)),
                        F.lit("eval"),
                    ).otherwise(F.lit("train")),
                )
                .withColumn("bucket", F.lit(int(bucket)))
                .dropDuplicates(["user_idx", "item_idx"])
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
                    "user_evidence_text",
                    "item_evidence_text",
                    "pair_evidence_summary",
                    "split",
                )
            ).persist(StorageLevel.DISK_ONLY)

            coalesce_for_write(
                pairwise_pool_df.filter(F.col("split") == F.lit("train")),
                PAIRWISE_POOL_WRITE_JSON_PARTITIONS,
            ).write.mode("overwrite").json(pairwise_pool_train_dir.as_posix())
            coalesce_for_write(
                pairwise_pool_df.filter(F.col("split") == F.lit("eval")),
                PAIRWISE_POOL_WRITE_JSON_PARTITIONS,
            ).write.mode("overwrite").json(pairwise_pool_eval_dir.as_posix())
            coalesce_for_write(
                pairwise_pool_df,
                PAIRWISE_POOL_WRITE_PARQUET_PARTITIONS,
            ).write.mode("overwrite").parquet(pairwise_pool_parquet_dir.as_posix())

            pair_pool_summary = collect_split_label_user_summary(
                pairwise_pool_df,
                include_split_user_counts=True,
            )
            pool_neg_tier_counts: dict[str, int] = {}
            for r in pairwise_pool_df.filter(F.col("label") == F.lit(0)).groupBy("neg_tier").count().collect():
                pool_neg_tier_counts[str(r["neg_tier"])] = int(r["count"])
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
                "output_train_dir": str(pairwise_pool_train_dir),
                "output_eval_dir": str(pairwise_pool_eval_dir),
                "output_parquet_dir": str(pairwise_pool_parquet_dir),
            }
            pairwise_pool_df.unpersist(blocking=False)
            pool_enrich.unpersist(blocking=False)
            print(
                f"[DONE] bucket={bucket} pairwise_pool "
                f"rows_train={pairwise_pool_stat['rows_train']} rows_eval={pairwise_pool_stat['rows_eval']}"
            )

        out_summary = collect_split_label_user_summary(out_df)
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
        for r in out_df.filter(F.col("label") == F.lit(0)).groupBy("neg_tier").count().collect():
            neg_tier_counts[str(r["neg_tier"])] = int(r["count"])
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
                "neg_tier_counts": neg_tier_counts,
                "user_evidence_dir": str(user_evidence_dir),
                "item_evidence_dir": str(item_evidence_dir),
                "pair_evidence_audit_dir": str(pair_audit_dir),
                "rich_sft": rich_sft_stat,
                "pairwise_pool": pairwise_pool_stat,
                "output_bucket_dir": str(b_out),
            }
        )
        print(f"[DONE] bucket={bucket} stats={split_stat}")
        out_df.unpersist(blocking=False)
        enrich.unpersist(blocking=False)

        truth.unpersist()

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage09_run": str(source_09),
        "buckets_override": sorted(list(wanted)) if wanted else [],
        "buckets_processed": processed_buckets,
        "topn_per_user": int(TOPN_PER_USER),
        "prompt_mode": str(PROMPT_MODE),
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
        "eval_user_frac": float(EVAL_USER_FRAC),
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

