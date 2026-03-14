from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window
from pipeline.project_paths import env_or_project_path


# Runtime
RUN_PROFILE = os.getenv("RUN_PROFILE_OVERRIDE", "full").strip().lower() or "full"  # "sample" | "full"
RUN_TAG = "stage09_user_profile_stats"
RANDOM_SEED = 42

# Paths
PARQUET_BASE = env_or_project_path("PARQUET_BASE_DIR", "data/parquet")
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_USER_PROFILE_STATS_ROOT_DIR", "data/output/09_user_profile_stats")

# Scope
TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
SAMPLE_MAX_BUSINESSES = 600

# Split (strict no-leak for profile side)
HOLDOUT_PER_USER = 2  # leave last 2 interactions for valid/test

# Time windows
PRIMARY_WINDOW_MONTHS = 12
FALLBACK_WINDOW_MONTHS = 24
PRIMARY_MIN_SENTENCES = 8

# Tier definition (agreed in discussion)
TIER_COLD_MAX = 1
TIER_LIGHT_MAX = 5

# High-recall text filter (remove obvious noise only)
TEXT_MIN_CHARS = 20
TEXT_MAX_CHARS = 1200
TEXT_MIN_WORDS = 6
TEXT_MAX_WORDS = 220
TEXT_MIN_ALPHA_RATIO = 0.45
MIN_SENTENCE_CHARS = 20

# Readiness thresholds
READINESS_SENTENCE_THRESHOLDS = [5, 10, 15, 20]

# Optional heavy output
WRITE_USER_LEVEL_PARQUET = False


def build_spark() -> SparkSession:
    local_dir = env_or_project_path("SPARK_LOCAL_DIR", "data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage09-user-profile-stats")
        .master("local[4]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.default.parallelism", "32")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_business_scope(spark: SparkSession) -> tuple[DataFrame, dict[str, Any]]:
    business = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
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
    biz = biz.select("business_id").distinct()

    stats: dict[str, Any] = {"n_scope_full": int(biz.count())}
    if RUN_PROFILE == "sample":
        biz = biz.orderBy(F.rand(RANDOM_SEED)).limit(int(SAMPLE_MAX_BUSINESSES))
    stats["n_scope_after_profile"] = int(biz.count())
    return biz.persist(StorageLevel.DISK_ONLY), stats


def load_reviews(spark: SparkSession, biz: DataFrame) -> DataFrame:
    return (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "date", "text")
        .withColumn("review_id", F.col("review_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(biz, on="business_id", how="inner")
        .persist(StorageLevel.DISK_ONLY)
    )


def split_train_only(reviews: DataFrame) -> tuple[DataFrame, DataFrame]:
    # Strictly use train side for profile feature generation.
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = reviews.withColumn("rn", F.row_number().over(w))
    train = ranked.filter(F.col("rn") > F.lit(int(HOLDOUT_PER_USER))).drop("rn")
    n_train = train.groupBy("user_id").agg(F.count("*").alias("n_train"))
    return train.persist(StorageLevel.DISK_ONLY), n_train.persist(StorageLevel.DISK_ONLY)


def build_train_text_pool(train: DataFrame) -> DataFrame:
    text_clean = F.trim(F.regexp_replace(F.coalesce(F.col("text"), F.lit("")), r"\s+", " "))
    char_len = F.length(text_clean)
    alpha_len = F.length(F.regexp_replace(text_clean, r"[^A-Za-z]", ""))
    alpha_ratio = F.when(char_len > 0, alpha_len / char_len).otherwise(F.lit(0.0))
    text_norm = F.lower(text_clean)
    word_count = F.size(F.split(text_norm, r"\s+"))

    # Keep only a compact key for dedup to reduce shuffle payload.
    pool = (
        train.select("user_id", "business_id", "review_id", "ts", "text")
        .withColumn("text_clean", text_clean)
        .withColumn("text_norm", text_norm)
        .withColumn("char_len", char_len)
        .withColumn("word_count", word_count)
        .withColumn("alpha_ratio", alpha_ratio)
        .filter(F.col("text_clean") != "")
        .filter((F.col("char_len") >= F.lit(TEXT_MIN_CHARS)) & (F.col("char_len") <= F.lit(TEXT_MAX_CHARS)))
        .filter((F.col("word_count") >= F.lit(TEXT_MIN_WORDS)) & (F.col("word_count") <= F.lit(TEXT_MAX_WORDS)))
        .filter(F.col("alpha_ratio") >= F.lit(float(TEXT_MIN_ALPHA_RATIO)))
        .withColumn("text_hash", F.xxhash64("text_norm"))
        .dropDuplicates(["user_id", "business_id", "text_hash"])
        .withColumn(
            "n_sent",
            F.size(F.expr(f"filter(split(text_clean, '[.!?]+'), x -> length(trim(x)) >= {int(MIN_SENTENCE_CHARS)})")),
        )
        .persist(StorageLevel.DISK_ONLY)
    )
    return pool


def with_user_tier(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "tier",
        F.when(F.col("n_train") <= F.lit(int(TIER_COLD_MAX)), F.lit("cold_lt2"))
        .when(F.col("n_train") <= F.lit(int(TIER_LIGHT_MAX)), F.lit("light_2_5"))
        .otherwise(F.lit("warm_6p")),
    )


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_PROFILE}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP] load business scope")
    biz, scope_stats = load_business_scope(spark)
    print(
        f"[COUNT] businesses_full={scope_stats['n_scope_full']} "
        f"after_profile={scope_stats['n_scope_after_profile']}"
    )

    print("[STEP] load reviews")
    reviews = load_reviews(spark, biz)
    n_reviews_raw = int(reviews.count())
    print(f"[COUNT] reviews_raw={n_reviews_raw}")

    print("[STEP] split train-only")
    train, n_train = split_train_only(reviews)
    n_users_train = int(n_train.count())
    n_train_events = int(train.count())
    print(f"[COUNT] users_train={n_users_train} train_events={n_train_events}")

    print("[STEP] build train text pool (high-recall filter + hash dedup)")
    pool = build_train_text_pool(train)
    n_pool_reviews = int(pool.count())
    print(f"[COUNT] pool_reviews={n_pool_reviews}")

    print("[STEP] aggregate by 12m priority + 24m fallback")
    last_train_ts = train.groupBy("user_id").agg(F.max("ts").alias("last_train_ts"))
    pool2 = (
        pool.join(last_train_ts, on="user_id", how="inner")
        .withColumn("m_diff", F.months_between(F.col("last_train_ts"), F.col("ts")))
        .filter((F.col("m_diff") >= F.lit(0.0)) & (F.col("m_diff") <= F.lit(float(FALLBACK_WINDOW_MONTHS))))
    )

    agg12 = (
        pool2.filter(F.col("m_diff") <= F.lit(float(PRIMARY_WINDOW_MONTHS)))
        .groupBy("user_id")
        .agg(
            F.count("*").alias("n_r12"),
            F.sum("n_sent").alias("n_s12"),
        )
    )
    agg24 = pool2.groupBy("user_id").agg(F.count("*").alias("n_r24"), F.sum("n_sent").alias("n_s24"))

    users = (
        n_train.join(last_train_ts, on="user_id", how="left")
        .join(agg12, on="user_id", how="left")
        .join(agg24, on="user_id", how="left")
        .fillna(0, subset=["n_r12", "n_s12", "n_r24", "n_s24"])
    )
    users = with_user_tier(users)
    users = (
        users.withColumn("use_24m_fallback", F.col("n_s12") < F.lit(int(PRIMARY_MIN_SENTENCES)))
        .withColumn("n_r_final", F.when(F.col("use_24m_fallback"), F.col("n_r24")).otherwise(F.col("n_r12")))
        .withColumn("n_s_final", F.when(F.col("use_24m_fallback"), F.col("n_s24")).otherwise(F.col("n_s12")))
        .persist(StorageLevel.DISK_ONLY)
    )

    summary_row = users.agg(
        F.count("*").alias("total_users"),
        F.sum(F.when(F.col("n_r12") > 0, 1).otherwise(0)).alias("users_with_r12"),
        F.sum(F.when(F.col("n_r24") > 0, 1).otherwise(0)).alias("users_with_r24"),
        F.sum(F.when(F.col("use_24m_fallback"), 1).otherwise(0)).alias("users_use_24m_fallback"),
        F.sum("n_r_final").alias("total_final_reviews"),
        F.sum("n_s_final").alias("total_final_sentences"),
    ).first()

    summary = {
        "run_id": run_id,
        "run_profile": RUN_PROFILE,
        "target_state": TARGET_STATE,
        "holdout_per_user": int(HOLDOUT_PER_USER),
        "window_primary_months": int(PRIMARY_WINDOW_MONTHS),
        "window_fallback_months": int(FALLBACK_WINDOW_MONTHS),
        "primary_min_sentences": int(PRIMARY_MIN_SENTENCES),
        "businesses_full": int(scope_stats["n_scope_full"]),
        "businesses_after_profile": int(scope_stats["n_scope_after_profile"]),
        "reviews_raw": int(n_reviews_raw),
        "train_events": int(n_train_events),
        "pool_reviews": int(n_pool_reviews),
        "total_users": int(summary_row["total_users"] or 0),
        "users_with_r12": int(summary_row["users_with_r12"] or 0),
        "users_with_r24": int(summary_row["users_with_r24"] or 0),
        "users_use_24m_fallback": int(summary_row["users_use_24m_fallback"] or 0),
        "total_final_reviews": int(summary_row["total_final_reviews"] or 0),
        "total_final_sentences": int(summary_row["total_final_sentences"] or 0),
    }

    thresholds_rows: list[dict[str, Any]] = []
    total_users = max(1, int(summary["total_users"]))
    for th in READINESS_SENTENCE_THRESHOLDS:
        c = int(users.filter(F.col("n_s_final") >= F.lit(int(th))).count())
        thresholds_rows.append(
            {
                "sentence_threshold": int(th),
                "user_count": c,
                "user_ratio": round(float(c) / float(total_users), 6),
            }
        )

    tier_rows: list[dict[str, Any]] = []
    tier_stat_df = (
        users.groupBy("tier")
        .agg(
            F.count("*").alias("n_users"),
            F.sum("n_r_final").alias("sum_reviews"),
            F.sum("n_s_final").alias("sum_sentences"),
            F.expr("percentile_approx(n_s_final, 0.5)").alias("p50_sentences"),
            F.expr("percentile_approx(n_s_final, 0.9)").alias("p90_sentences"),
            F.sum(F.when(F.col("n_s_final") >= F.lit(10), 1).otherwise(0)).alias("users_ge10"),
            F.sum(F.when(F.col("n_s_final") >= F.lit(15), 1).otherwise(0)).alias("users_ge15"),
        )
        .orderBy("tier")
    )
    for row in tier_stat_df.collect():
        n_users = int(row["n_users"] or 0)
        tier_rows.append(
            {
                "tier": row["tier"],
                "n_users": n_users,
                "user_ratio": round(float(n_users) / float(total_users), 6),
                "sum_reviews": int(row["sum_reviews"] or 0),
                "sum_sentences": int(row["sum_sentences"] or 0),
                "p50_sentences": int(row["p50_sentences"] or 0),
                "p90_sentences": int(row["p90_sentences"] or 0),
                "users_ge10": int(row["users_ge10"] or 0),
                "users_ge15": int(row["users_ge15"] or 0),
            }
        )

    write_csv_rows(
        out_dir / "user_profile_stats_summary.csv",
        list(summary.keys()),
        [summary],
    )
    write_csv_rows(
        out_dir / "user_profile_stats_thresholds.csv",
        ["sentence_threshold", "user_count", "user_ratio"],
        thresholds_rows,
    )
    write_csv_rows(
        out_dir / "user_profile_stats_tier.csv",
        [
            "tier",
            "n_users",
            "user_ratio",
            "sum_reviews",
            "sum_sentences",
            "p50_sentences",
            "p90_sentences",
            "users_ge10",
            "users_ge15",
        ],
        tier_rows,
    )

    if WRITE_USER_LEVEL_PARQUET:
        users.select(
            "user_id",
            "n_train",
            "tier",
            "last_train_ts",
            "n_r12",
            "n_s12",
            "n_r24",
            "n_s24",
            "use_24m_fallback",
            "n_r_final",
            "n_s_final",
        ).write.mode("overwrite").parquet((out_dir / "user_profile_stats_user_level.parquet").as_posix())

    write_json(
        out_dir / "run_meta.json",
        {
            "run_id": run_id,
            "run_profile": RUN_PROFILE,
            "run_tag": RUN_TAG,
            "scope": {
                "target_state": TARGET_STATE,
                "require_restaurants": bool(REQUIRE_RESTAURANTS),
                "require_food": bool(REQUIRE_FOOD),
                "sample_max_businesses": int(SAMPLE_MAX_BUSINESSES),
            },
            "split": {"holdout_per_user": int(HOLDOUT_PER_USER)},
            "windows": {
                "primary_months": int(PRIMARY_WINDOW_MONTHS),
                "fallback_months": int(FALLBACK_WINDOW_MONTHS),
                "primary_min_sentences": int(PRIMARY_MIN_SENTENCES),
            },
            "text_filter": {
                "min_chars": int(TEXT_MIN_CHARS),
                "max_chars": int(TEXT_MAX_CHARS),
                "min_words": int(TEXT_MIN_WORDS),
                "max_words": int(TEXT_MAX_WORDS),
                "min_alpha_ratio": float(TEXT_MIN_ALPHA_RATIO),
                "min_sentence_chars": int(MIN_SENTENCE_CHARS),
            },
            "summary": summary,
        },
    )

    print(f"[OK] wrote: {out_dir}")

    users.unpersist()
    pool.unpersist()
    n_train.unpersist()
    train.unpersist()
    reviews.unpersist()
    biz.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
