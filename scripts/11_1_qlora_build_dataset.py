from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

from pipeline.qlora_prompting import (
    build_binary_prompt,
    build_binary_prompt_semantic,
    build_item_text,
    build_item_text_semantic,
    build_user_text,
)
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context


RUN_TAG = "stage11_1_qlora_build_dataset"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = Path(r"D:/5006_BDA_project/data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = Path(r"D:/5006_BDA_project/data/output/11_qlora_data")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
CANDIDATE_FILE = os.getenv("TRAIN_CANDIDATE_FILE", "candidates_pretrim150.parquet").strip() or "candidates_pretrim150.parquet"
TOPN_PER_USER = int(os.getenv("QLORA_TOPN_PER_USER", "120").strip() or 120)
NEG_PER_USER = int(os.getenv("QLORA_NEG_PER_USER", "8").strip() or 8)
NEG_LAYERED_ENABLED = os.getenv("QLORA_NEG_LAYERED_ENABLED", "true").strip().lower() == "true"
NEG_HARD_RATIO = float(os.getenv("QLORA_NEG_HARD_RATIO", "0.50").strip() or 0.50)
NEG_NEAR_RATIO = float(os.getenv("QLORA_NEG_NEAR_RATIO", "0.30").strip() or 0.30)
NEG_HARD_RANK_MAX = int(os.getenv("QLORA_NEG_HARD_RANK_MAX", "120").strip() or 120)
NEG_HARD_WEIGHT = float(os.getenv("QLORA_NEG_HARD_WEIGHT", "1.20").strip() or 1.20)
NEG_NEAR_WEIGHT = float(os.getenv("QLORA_NEG_NEAR_WEIGHT", "1.10").strip() or 1.10)
NEG_EASY_WEIGHT = float(os.getenv("QLORA_NEG_EASY_WEIGHT", "0.90").strip() or 0.90)
NEG_FILL_WEIGHT = float(os.getenv("QLORA_NEG_FILL_WEIGHT", "1.00").strip() or 1.00)
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
    os.getenv("QLORA_STAGE09_GATE_METRICS_PATH", "D:/5006_BDA_project/data/metrics/stage09_recall_audit_summary_latest.csv").strip()
)
STAGE09_GATE_MIN_TRUTH_IN_PRETRIM = float(os.getenv("QLORA_GATE_MIN_TRUTH_IN_PRETRIM", "0.82").strip() or 0.82)
STAGE09_GATE_MAX_PRETRIM_CUT_LOSS = float(os.getenv("QLORA_GATE_MAX_PRETRIM_CUT_LOSS", "0.08").strip() or 0.08)
STAGE09_GATE_MAX_HARD_MISS = float(os.getenv("QLORA_GATE_MAX_HARD_MISS", "0.10").strip() or 0.10)

ENABLE_SHORT_TEXT_SUMMARY = os.getenv("QLORA_ENABLE_SHORT_TEXT_SUMMARY", "true").strip().lower() == "true"
ENABLE_RAW_REVIEW_TEXT = os.getenv("QLORA_ENABLE_RAW_REVIEW_TEXT", "true").strip().lower() == "true"
REVIEW_TABLE_PATH = Path(
    os.getenv("QLORA_REVIEW_TABLE_PATH", "D:/5006_BDA_project/data/parquet/yelp_academic_dataset_review").strip()
)
USER_REVIEW_TOPN = int(os.getenv("QLORA_USER_REVIEW_TOPN", "2").strip() or 2)
ITEM_REVIEW_TOPN = int(os.getenv("QLORA_ITEM_REVIEW_TOPN", "2").strip() or 2)
REVIEW_SNIPPET_MAX_CHARS = int(os.getenv("QLORA_REVIEW_SNIPPET_MAX_CHARS", "220").strip() or 220)

PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "full").strip().lower() or "full"  # "full" | "semantic"
INCLUDE_HIST_POS = os.getenv("QLORA_INCLUDE_HIST_POS", "false").strip().lower() == "true"
HIST_POS_MIN_RATING = float(os.getenv("QLORA_HIST_POS_MIN_RATING", "4.0").strip() or 4.0)
HIST_POS_MAX_PER_USER = int(os.getenv("QLORA_HIST_POS_MAX_PER_USER", "3").strip() or 3)
HIST_POS_WEIGHT = float(os.getenv("QLORA_HIST_POS_WEIGHT", "0.25").strip() or 0.25)

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", "D:/5006_BDA_project/data/spark-tmp").strip() or "D:/5006_BDA_project/data/spark-tmp"
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


_SPARK_TMP_CTX: SparkTmpContext | None = None


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
    p = Path(str(raw_path or "").strip())
    if p.exists():
        return p
    legacy = str(raw_path or "").replace("\\", "/")
    if "D:/5006 BDA project" in legacy:
        alt = Path(legacy.replace("D:/5006 BDA project", "D:/5006_BDA_project"))
        if alt.exists():
            return alt
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


def build_user_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    if not profile_path:
        return spark.createDataFrame([], "user_id string, profile_text string, profile_top_pos_tags string, profile_top_neg_tags string, profile_confidence double")
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], "user_id string, profile_text string, profile_top_pos_tags string, profile_top_neg_tags string, profile_confidence double")
    pdf = (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "user_id",
            F.coalesce(F.col("profile_text_short"), F.col("profile_text"), F.lit("")).alias("profile_text"),
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


def build_temporal_review_features(
    spark: SparkSession,
    bucket_dir: Path,
    truth_df: DataFrame,
    row_df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    user_schema = "user_idx int, user_review_summary string, user_review_raw string"
    item_schema = "user_idx int, business_id string, item_review_summary string, item_review_raw string"
    empty_user = spark.createDataFrame([], user_schema)
    empty_item = spark.createDataFrame([], item_schema)
    if not (ENABLE_SHORT_TEXT_SUMMARY or ENABLE_RAW_REVIEW_TEXT):
        return empty_user, empty_item
    if not REVIEW_TABLE_PATH.exists():
        print(f"[WARN] review table not found, skip review text features: {REVIEW_TABLE_PATH}")
        return empty_user, empty_item
    hist_path = bucket_dir / "train_history.parquet"
    if not hist_path.exists():
        print(f"[WARN] train_history.parquet missing, skip review text features: {hist_path}")
        return empty_user, empty_item

    user_cutoff = (
        spark.read.parquet(hist_path.as_posix())
        .select("user_idx", "test_ts")
        .filter(F.col("test_ts").isNotNull())
        .groupBy("user_idx")
        .agg(F.max("test_ts").alias("test_ts"))
    )

    rvw = (
        spark.read.parquet(REVIEW_TABLE_PATH.as_posix())
        .select("user_id", "business_id", "date", "text", "stars")
        .withColumn("review_ts", F.to_timestamp(F.col("date")))
        .withColumn("text_clean", F.regexp_replace(F.regexp_replace(F.coalesce(F.col("text"), F.lit("")), r"[\r\n]+", " "), r"\s+", " "))
        .filter(F.col("review_ts").isNotNull() & (F.length(F.col("text_clean")) > F.lit(0)))
    )

    u_topn = max(1, int(USER_REVIEW_TOPN))
    i_topn = max(1, int(ITEM_REVIEW_TOPN))
    max_chars = max(40, int(REVIEW_SNIPPET_MAX_CHARS))

    user_base = truth_df.select("user_idx", "user_id").dropDuplicates(["user_idx"]).join(user_cutoff, on="user_idx", how="inner")
    user_rows = (
        user_base.join(rvw.select("user_id", "review_ts", "text_clean", "stars"), on="user_id", how="inner")
        .filter(F.col("review_ts") < F.col("test_ts"))
        .withColumn("snippet", F.substring(F.col("text_clean"), 1, int(max_chars)))
    )
    w_user = Window.partitionBy("user_idx").orderBy(F.col("review_ts").desc())
    user_top = user_rows.withColumn("rn", F.row_number().over(w_user)).filter(F.col("rn") <= F.lit(int(u_topn)))
    user_feat = (
        user_top.groupBy("user_idx")
        .agg(
            F.count("*").alias("n_reviews"),
            F.avg(F.col("stars").cast("double")).alias("avg_stars"),
            F.concat_ws(" || ", F.collect_list("snippet")).alias("user_review_raw"),
        )
        .withColumn(
            "user_review_summary",
            F.concat(
                F.lit("recent_user_reviews="),
                F.col("n_reviews").cast("string"),
                F.lit(", avg_stars="),
                F.format_number(F.coalesce(F.col("avg_stars"), F.lit(0.0)), 2),
            ),
        )
        .select("user_idx", "user_review_summary", "user_review_raw")
    )

    item_base = (
        row_df.select("user_idx", "business_id")
        .dropDuplicates(["user_idx", "business_id"])
        .join(user_cutoff, on="user_idx", how="inner")
    )
    item_rows = (
        item_base.join(rvw.select("business_id", "review_ts", "text_clean", "stars"), on="business_id", how="inner")
        .filter(F.col("review_ts") < F.col("test_ts"))
        .withColumn("snippet", F.substring(F.col("text_clean"), 1, int(max_chars)))
    )
    w_item = Window.partitionBy("user_idx", "business_id").orderBy(F.col("review_ts").desc())
    item_top = item_rows.withColumn("rn", F.row_number().over(w_item)).filter(F.col("rn") <= F.lit(int(i_topn)))
    item_feat = (
        item_top.groupBy("user_idx", "business_id")
        .agg(
            F.count("*").alias("n_reviews"),
            F.avg(F.col("stars").cast("double")).alias("avg_stars"),
            F.concat_ws(" || ", F.collect_list("snippet")).alias("item_review_raw"),
        )
        .withColumn(
            "item_review_summary",
            F.concat(
                F.lit("recent_item_reviews="),
                F.col("n_reviews").cast("string"),
                F.lit(", avg_stars="),
                F.format_number(F.coalesce(F.col("avg_stars"), F.lit(0.0)), 2),
            ),
        )
        .select("user_idx", "business_id", "item_review_summary", "item_review_raw")
    )

    if not ENABLE_SHORT_TEXT_SUMMARY:
        user_feat = user_feat.withColumn("user_review_summary", F.lit(""))
        item_feat = item_feat.withColumn("item_review_summary", F.lit(""))
    if not ENABLE_RAW_REVIEW_TEXT:
        user_feat = user_feat.withColumn("user_review_raw", F.lit(""))
        item_feat = item_feat.withColumn("item_review_raw", F.lit(""))
    return user_feat, item_feat


def build_prompt_udf() -> Any:
    def _mk(
        profile_text: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        user_review_summary: Any,
        user_review_raw: Any,
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
        item_review_summary: Any,
        item_review_raw: Any,
    ) -> str:
        return build_binary_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                review_summary=user_review_summary,
                review_raw_snippet=user_review_raw,
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
                item_review_summary=item_review_summary,
                item_review_snippet=item_review_raw,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_semantic() -> Any:
    """Prompt UDF that drops ranking-position features for DPO training."""
    def _mk(
        profile_text: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        user_review_summary: Any,
        user_review_raw: Any,
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
        item_review_summary: Any,
        item_review_raw: Any,
    ) -> str:
        return build_binary_prompt_semantic(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                review_summary=user_review_summary,
                review_raw_snippet=user_review_raw,
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
                item_review_summary=item_review_summary,
                item_review_snippet=item_review_raw,
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
    gate_result = enforce_stage09_gate(source_09, [int(p.name.split("_")[-1]) for p in bucket_dirs])

    user_profile_df = build_user_profile_df(spark, meta)
    item_sem_df = build_item_sem_df(spark, meta)
    cluster_profile_df = build_cluster_profile_df(spark, meta)
    if PROMPT_MODE == "semantic":
        prompt_udf = build_prompt_udf_semantic()
        print(f"[CONFIG] PROMPT_MODE=semantic (ranking features removed from prompt)")
    else:
        prompt_udf = build_prompt_udf()
        print(f"[CONFIG] PROMPT_MODE=full (all features in prompt)")

    summary: list[dict[str, Any]] = []

    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
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

        cand = cand_raw.select(
            "user_idx",
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
            hist_path = bdir / "train_history.parquet"
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
            neg_quota = resolve_neg_quotas(NEG_PER_USER)
            hard_quota = int(neg_quota["hard"])
            near_quota = int(neg_quota["near"])
            easy_quota = int(neg_quota["easy"])

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

        user_review_df, item_review_df = build_temporal_review_features(
            spark=spark,
            bucket_dir=bdir,
            truth_df=truth.select("user_idx", "user_id"),
            row_df=train_rows.select("user_idx", "business_id"),
        )

        enrich = (
            train_rows.join(user_profile_df, on="user_id", how="left")
            .join(item_sem_df, on="business_id", how="left")
            .join(cluster_profile_df, on="business_id", how="left")
            .join(user_review_df, on="user_idx", how="left")
            .join(item_review_df, on=["user_idx", "business_id"], how="left")
            .fillna(
                {
                    "profile_text": "",
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
                    "user_review_summary": "",
                    "user_review_raw": "",
                    "item_review_summary": "",
                    "item_review_raw": "",
                }
            )
        )

        if PROMPT_MODE == "semantic":
            prompt_col = prompt_udf(
                F.col("profile_text"),
                F.col("profile_top_pos_tags"),
                F.col("profile_top_neg_tags"),
                F.col("profile_confidence"),
                F.col("user_review_summary"),
                F.col("user_review_raw"),
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
                F.col("item_review_summary"),
                F.col("item_review_raw"),
            )
        else:
            prompt_col = prompt_udf(
                F.col("profile_text"),
                F.col("profile_top_pos_tags"),
                F.col("profile_top_neg_tags"),
                F.col("profile_confidence"),
                F.col("user_review_summary"),
                F.col("user_review_raw"),
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
                F.col("item_review_summary"),
                F.col("item_review_raw"),
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
                "split",
            )
        ).persist(StorageLevel.DISK_ONLY)

        split_pairs = out_df.groupBy("split", "label").count().collect()
        if not split_pairs:
            print(f"[WARN] bucket={bucket} empty after enrich")
            out_df.unpersist(blocking=False)
            truth.unpersist()
            continue

        b_out = out_dir / f"bucket_{bucket}"
        train_dir = b_out / "train_json"
        eval_dir = b_out / "eval_json"
        parquet_dir = b_out / "all_parquet"
        for d in (train_dir, eval_dir, parquet_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        out_df.filter(F.col("split") == F.lit("train")).coalesce(1).write.mode("overwrite").json(train_dir.as_posix())
        out_df.filter(F.col("split") == F.lit("eval")).coalesce(1).write.mode("overwrite").json(eval_dir.as_posix())
        out_df.coalesce(1).write.mode("overwrite").parquet(parquet_dir.as_posix())

        split_stat_map: dict[tuple[str, int], int] = {}
        for r in split_pairs:
            split_stat_map[(str(r["split"]), int(r["label"]))] = int(r["count"])
        split_stat = {
            "train_label_0": int(split_stat_map.get(("train", 0), 0)),
            "train_label_1": int(split_stat_map.get(("train", 1), 0)),
            "eval_label_0": int(split_stat_map.get(("eval", 0), 0)),
            "eval_label_1": int(split_stat_map.get(("eval", 1), 0)),
        }
        user_stats_row = (
            out_df.agg(
                F.countDistinct("user_idx").alias("users_total"),
                F.countDistinct(F.when(F.col("label") == F.lit(1), F.col("user_idx"))).alias("users_with_pos"),
            ).collect()[0]
        )
        users_total = int(user_stats_row["users_total"] or 0)
        users_with_pos = int(user_stats_row["users_with_pos"] or 0)
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
                "output_bucket_dir": str(b_out),
            }
        )
        print(f"[DONE] bucket={bucket} stats={split_stat}")
        out_df.unpersist(blocking=False)

        truth.unpersist()

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage09_run": str(source_09),
        "buckets_override": sorted(list(wanted)) if wanted else [],
        "topn_per_user": int(TOPN_PER_USER),
        "neg_per_user": int(NEG_PER_USER),
        "neg_layered_enabled": bool(NEG_LAYERED_ENABLED),
        "neg_hard_ratio": float(NEG_HARD_RATIO),
        "neg_near_ratio": float(NEG_NEAR_RATIO),
        "neg_hard_rank_max": int(NEG_HARD_RANK_MAX),
        "neg_hard_weight": float(NEG_HARD_WEIGHT),
        "neg_near_weight": float(NEG_NEAR_WEIGHT),
        "neg_easy_weight": float(NEG_EASY_WEIGHT),
        "neg_fill_weight": float(NEG_FILL_WEIGHT),
        "user_cap_randomized": bool(USER_CAP_RANDOMIZED),
        "include_valid_pos": bool(INCLUDE_VALID_POS),
        "valid_pos_weight": float(VALID_POS_WEIGHT),
        "eval_user_frac": float(EVAL_USER_FRAC),
        "structured_feature_prompt_enabled": True,
        "short_text_summary_enabled": bool(ENABLE_SHORT_TEXT_SUMMARY),
        "raw_review_text_enabled": bool(ENABLE_RAW_REVIEW_TEXT),
        "review_snippet_max_chars": int(REVIEW_SNIPPET_MAX_CHARS),
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
    spark.stop()


if __name__ == "__main__":
    main()

