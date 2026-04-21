import os
import sys
from pathlib import Path

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("Usage: python scripts/stage01_to_stage08/06_insight_text.py")
    print("Builds stage06 insight text and category-level term summaries from review data.")
    print("Set parquet paths and Spark env vars, then run without --help.")
    sys.exit(0)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pyspark import StorageLevel
from pyspark.ml.feature import NGram, RegexTokenizer, StopWordsRemover
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window


TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
GROUP_BY = "category"  # "category" or "city_category"

QUIET_LOGS = True
SAMPLE_FRACTION = 0.0
RANDOM_SEED = 42

TOP_TERMS_PER_GROUP = 15
TOP_GROUPS_FOR_PLOTS = 10
HIGH_STARS = 4.0
LOW_STARS = 2.0
MAX_TERMS_TOPIC = 2000  # 0 means keep all

FAST_MODE = True
FAST_SAMPLE_FRACTION = 0.1
FAST_MAX_TERMS_TOPIC = 500
FAST_TOP_TERMS_PER_GROUP = 10
FAST_TOP_GROUPS_FOR_PLOTS = 8
FAST_TOP_CATEGORIES = 50

USE_DOMAIN_STOPWORDS = True
CUSTOM_STOPWORDS = [
    "food",
    "good",
    "great",
    "place",
    "service",
    "time",
    "one",
    "get",
    "back",
    "like",
    "really",
    "also",
    "just",
    "would",
    "could",
    "im",
    "ive",
    "dont",
    "didnt",
    "wasnt",
    "isnt",
    "wont",
    "cant",
]

ENABLE_BIGRAMS = True

BASE_DIR = Path(r"D:/5006 BDA project/data/parquet")
OUTPUT_DIR = Path(r"D:/5006 BDA project/data/insights")
OUTPUT_KEYWORDS = OUTPUT_DIR / "insight_keywords_by_group.csv"
OUTPUT_TOPIC_STARS = OUTPUT_DIR / "insight_topic_vs_stars.csv"
FIG_TOP_TERMS = OUTPUT_DIR / "fig_top_terms_overall.png"
FIG_HIGH_LOW = OUTPUT_DIR / "fig_high_low_terms.png"
FIG_GROUP_STARS = OUTPUT_DIR / "fig_avg_stars_by_group.png"


def build_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder
        .appName("yelp-la-insight-text")
        .master(local_master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.local.dir", str(local_dir))
        .config("spark.ui.showConsoleProgress", "false" if QUIET_LOGS else "true")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )
    return builder.getOrCreate()


def set_log_level(spark: SparkSession) -> None:
    level = "ERROR" if QUIET_LOGS else "WARN"
    spark.sparkContext.setLogLevel(level)
    if not QUIET_LOGS:
        return
    try:
        log4j = spark._jvm.org.apache.log4j
        log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
    except Exception:
        pass
    try:
        log4j2 = spark._jvm.org.apache.logging.log4j
        log4j2.LogManager.getRootLogger().setLevel(log4j2.Level.ERROR)
    except Exception:
        pass


def load_business_filtered(spark: SparkSession) -> DataFrame:
    business = (
        spark.read.parquet((BASE_DIR / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories", "city")
    )
    biz = business.filter(F.col("state") == TARGET_STATE)
    cat_col = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    cond = None
    if REQUIRE_RESTAURANTS:
        cond = cat_col.contains("restaurants")
    if REQUIRE_FOOD:
        cond = (cond | cat_col.contains("food")) if cond is not None else cat_col.contains("food")
    if cond is not None:
        biz = biz.filter(cond)
    return biz.select("business_id", "categories", "city").distinct()


def load_reviews(spark: SparkSession, biz: DataFrame) -> DataFrame:
    review = (
        spark.read.parquet((BASE_DIR / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "business_id", "stars", "text")
    )
    rvw = review.join(biz.select("business_id", "city"), on="business_id", how="inner")
    rvw = rvw.filter(F.col("text").isNotNull() & (F.length(F.col("text")) > 0))
    rvw = rvw.filter(F.col("stars").isNotNull())
    return rvw


def tokenize_reviews(df: DataFrame) -> DataFrame:
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens_raw",
        pattern="[^a-zA-Z]+",
        toLowercase=True,
    )
    df = tokenizer.transform(df)
    stopwords = StopWordsRemover.loadDefaultStopWords("english")
    if USE_DOMAIN_STOPWORDS:
        stopwords = list(set(stopwords + CUSTOM_STOPWORDS))
    remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens", stopWords=stopwords)
    df = remover.transform(df)
    df = df.withColumn("tokens", F.expr("filter(tokens, x -> length(x) >= 3)"))
    if ENABLE_BIGRAMS:
        ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
        df = ngram.transform(df)
        df = df.withColumn("tokens", F.expr("concat(tokens, bigrams)")).drop("bigrams")
    df = df.filter(F.size("tokens") > 0)
    return df.drop("tokens_raw")


def explode_categories(biz: DataFrame) -> DataFrame:
    return (
        biz.filter(F.col("categories").isNotNull() & (F.trim(F.col("categories")) != ""))
        .withColumn("category", F.explode(F.split(F.col("categories"), r"\s*,\s*")))
        .withColumn("category", F.lower(F.trim(F.col("category"))))
        .filter(F.col("category") != "")
        .select("business_id", "city", "category")
        .distinct()
    )


def compute_idf(docs: DataFrame) -> tuple[DataFrame, int]:
    docs = docs.select("review_id", "tokens").filter(F.size("tokens") > 0)
    doc_terms = docs.select(
        "review_id",
        F.explode(F.array_distinct("tokens")).alias("term"),
    )
    n_docs = docs.select("review_id").distinct().count()
    term_df = doc_terms.groupBy("term").agg(F.countDistinct("review_id").alias("df"))
    term_idf = term_df.withColumn(
        "idf",
        F.log((F.lit(n_docs) + F.lit(1.0)) / (F.col("df") + F.lit(1.0))) + F.lit(1.0),
    )
    return term_idf, n_docs


def compute_group_terms(
    reviews_grouped: DataFrame,
    term_idf: DataFrame,
    group_cols: list[str],
    top_terms_per_group: int,
) -> tuple[DataFrame, DataFrame]:
    group_stats = reviews_grouped.groupBy(*group_cols).agg(
        F.countDistinct("review_id").alias("n_reviews"),
        F.avg("stars").alias("avg_stars"),
    )
    term_tf = reviews_grouped.select(
        *group_cols, "review_id", F.explode("tokens").alias("term")
    ).groupBy(*group_cols, "term").agg(
        F.count("*").alias("tf"),
        F.countDistinct("review_id").alias("doc_count"),
    )
    term_tfidf = term_tf.join(term_idf, on="term", how="left").withColumn(
        "tfidf", F.col("tf") * F.col("idf")
    )
    w = Window.partitionBy(*group_cols).orderBy(F.desc("tfidf"), F.desc("tf"))
    top_terms = term_tfidf.withColumn("rn", F.row_number().over(w)).filter(
        F.col("rn") <= F.lit(top_terms_per_group)
    )
    top_terms = top_terms.join(group_stats, on=group_cols, how="left")
    return top_terms, group_stats


def compute_topic_vs_stars(docs: DataFrame, term_idf: DataFrame, max_terms_topic: int) -> DataFrame:
    doc_terms = docs.select(
        "review_id",
        "stars",
        F.explode(F.array_distinct("tokens")).alias("term"),
    )
    term_stats = doc_terms.groupBy("term").agg(
        F.countDistinct("review_id").alias("n_reviews"),
        F.avg("stars").alias("avg_stars"),
        F.sum(F.when(F.col("stars") >= HIGH_STARS, 1).otherwise(0)).alias("high_reviews"),
        F.sum(F.when(F.col("stars") <= LOW_STARS, 1).otherwise(0)).alias("low_reviews"),
    )
    term_stats = term_stats.join(term_idf, on="term", how="left").withColumn(
        "strength", F.col("n_reviews") * F.col("idf")
    )
    if max_terms_topic and max_terms_topic > 0:
        term_stats = term_stats.orderBy(F.desc("strength")).limit(max_terms_topic)
    return term_stats


def format_group_label(row: pd.Series) -> str:
    if GROUP_BY == "city_category":
        return f"{row['city']} | {row['category']}"
    return str(row["category"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_fraction = SAMPLE_FRACTION
    max_terms_topic = MAX_TERMS_TOPIC
    top_terms_per_group = TOP_TERMS_PER_GROUP
    top_groups_for_plots = TOP_GROUPS_FOR_PLOTS

    if FAST_MODE:
        sample_fraction = FAST_SAMPLE_FRACTION
        max_terms_topic = FAST_MAX_TERMS_TOPIC
        top_terms_per_group = FAST_TOP_TERMS_PER_GROUP
        top_groups_for_plots = FAST_TOP_GROUPS_FOR_PLOTS
        print(
            "[INFO] FAST_MODE enabled: "
            f"sample={sample_fraction}, max_terms={max_terms_topic}, "
            f"top_terms_per_group={top_terms_per_group}, "
            f"top_groups_for_plots={top_groups_for_plots}"
        )

    spark = build_spark()
    set_log_level(spark)

    print("[STEP] load + filter businesses")
    biz = load_business_filtered(spark).persist(StorageLevel.DISK_ONLY)
    if biz.limit(1).count() == 0:
        print("[ERROR] no businesses after filtering")
        spark.stop()
        return

    print("[STEP] load + filter reviews")
    reviews = load_reviews(spark, biz)
    if sample_fraction and sample_fraction > 0:
        reviews = reviews.sample(False, sample_fraction, RANDOM_SEED)
    reviews = reviews.persist(StorageLevel.DISK_ONLY)
    if reviews.limit(1).count() == 0:
        print("[ERROR] no reviews after filtering")
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] build categories")
    biz_cats = explode_categories(biz).persist(StorageLevel.DISK_ONLY)

    if FAST_MODE and FAST_TOP_CATEGORIES > 0:
        print(f"[STEP] fast mode: select top {FAST_TOP_CATEGORIES} categories")
        cat_counts = (
            reviews.join(biz_cats, on="business_id", how="inner")
            .groupBy("category")
            .agg(F.countDistinct("review_id").alias("n_reviews"))
            .orderBy(F.desc("n_reviews"))
            .limit(FAST_TOP_CATEGORIES)
        )
        top_categories = [row["category"] for row in cat_counts.collect()]
        biz_cats = biz_cats.filter(F.col("category").isin(top_categories)).persist(
            StorageLevel.DISK_ONLY
        )
        top_businesses = biz_cats.select("business_id").distinct()
        reviews.unpersist()
        reviews = reviews.join(top_businesses, on="business_id", how="inner").persist(
            StorageLevel.DISK_ONLY
        )

    print("[STEP] tokenize reviews")
    reviews_tok = tokenize_reviews(reviews).persist(StorageLevel.DISK_ONLY)

    print("[STEP] compute IDF")
    term_idf, n_docs = compute_idf(reviews_tok)
    print(f"[COUNT] docs={n_docs}")

    print("[STEP] build groups")
    reviews_grouped = reviews_tok.join(biz_cats, on="business_id", how="inner")
    reviews_grouped = reviews_grouped.persist(StorageLevel.DISK_ONLY)

    group_cols = ["category"] if GROUP_BY == "category" else ["city", "category"]
    top_terms, group_stats = compute_group_terms(
        reviews_grouped, term_idf, group_cols, top_terms_per_group
    )

    print("[STEP] compute topic vs stars")
    term_topic = compute_topic_vs_stars(reviews_tok, term_idf, max_terms_topic)

    top_terms_pd = top_terms.toPandas()
    group_stats_pd = group_stats.toPandas()
    term_topic_pd = term_topic.toPandas()

    top_terms_pd.to_csv(OUTPUT_KEYWORDS, index=False)
    term_topic_pd.to_csv(OUTPUT_TOPIC_STARS, index=False)
    print(f"[INFO] wrote {OUTPUT_KEYWORDS}")
    print(f"[INFO] wrote {OUTPUT_TOPIC_STARS}")

    if not group_stats_pd.empty:
        group_stats_pd = group_stats_pd.sort_values("n_reviews", ascending=False).head(
            top_groups_for_plots
        )
        group_stats_pd["group"] = group_stats_pd.apply(format_group_label, axis=1)
        plt.figure(figsize=(8, 4))
        plt.barh(group_stats_pd["group"], group_stats_pd["avg_stars"])
        plt.gca().invert_yaxis()
        plt.xlabel("avg_stars")
        plt.title("Average Stars by Group (Top by Reviews)")
        plt.tight_layout()
        plt.savefig(FIG_GROUP_STARS, dpi=150)
        plt.close()

    if not term_topic_pd.empty:
        term_topic_pd = term_topic_pd.sort_values("strength", ascending=False)
        top_terms_pd2 = term_topic_pd.head(20)
        plt.figure(figsize=(8, 4))
        plt.barh(top_terms_pd2["term"], top_terms_pd2["strength"])
        plt.gca().invert_yaxis()
        plt.xlabel("strength (tf * idf)")
        plt.title("Top Terms by Topic Strength")
        plt.tight_layout()
        plt.savefig(FIG_TOP_TERMS, dpi=150)
        plt.close()

        term_topic_pd["log_ratio"] = np.log(
            (term_topic_pd["high_reviews"] + 1.0) / (term_topic_pd["low_reviews"] + 1.0)
        )
        top_pos = term_topic_pd.sort_values("log_ratio", ascending=False).head(15)
        top_neg = term_topic_pd.sort_values("log_ratio", ascending=True).head(15)
        combined = pd.concat([top_pos, top_neg], axis=0)
        plt.figure(figsize=(8, 5))
        plt.barh(combined["term"], combined["log_ratio"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("log((high+1)/(low+1))")
        plt.title("Terms Skewed to High vs Low Stars")
        plt.tight_layout()
        plt.savefig(FIG_HIGH_LOW, dpi=150)
        plt.close()

    reviews_grouped.unpersist()
    biz_cats.unpersist()
    reviews_tok.unpersist()
    reviews.unpersist()
    biz.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
