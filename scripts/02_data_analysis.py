import os
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


SHOW_TABLE_FIELDS = False
SHOW_SAMPLE_ROWS = 5

RUN_CITY_STATS = False
RUN_BUSINESS_STATS = False
RUN_CATEGORY_STATS = False
RUN_STATE_STATS = False
RUN_LA_SLICE_STATS = False
RUN_LA_USER_ACTIVITY_STATS = True
RUN_LA_LEAVE_LAST_OUT = True

RUN_PROFILE_USER = False
RUN_PROFILE_TIP = False
RUN_PROFILE_CHECKIN = False

RUN_TIME_DIST_REVIEW = False
RUN_TIME_DIST_TIP = False
RUN_TIME_DIST_CHECKIN = False

CITY_MIN_REVIEWS = 1000
CITY_TOP_N = 20
BUSINESS_MIN_REVIEWS = 200
BUSINESS_TOP_N = 20
CATEGORY_MIN_REVIEWS = 5000
CATEGORY_TOP_N = 30
STATE_TOP_N = 60

TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
USE_EXPLICIT_RATING = True
SLICE_SHOW_SAMPLE = 0
USER_COUNT_THRESHOLDS = [2, 3, 5, 10]
MIN_USER_REVIEWS_FOR_SPLIT = 2
LEAVE_LAST_SHOW_SAMPLE = 0


def build_spark() -> SparkSession:
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")

    return (
        SparkSession.builder
        .appName("yelp-analysis")
        .master(local_master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", executor_mem)
        .config("spark.default.parallelism", "4")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def show_table_fields(spark: SparkSession, base: Path, n: int = 5) -> None:
    tables = {
        "business": "yelp_academic_dataset_business",
        "checkin": "yelp_academic_dataset_checkin",
        "tip": "yelp_academic_dataset_tip",
        "user": "yelp_academic_dataset_user",
        "review": "yelp_academic_dataset_review",
    }
    for name, folder in tables.items():
        print(f"\n[{name.upper()}]")
        df = spark.read.parquet((base / folder).as_posix())
        print("columns:", df.columns)
        df.printSchema()
        df.show(n, truncate=80)
        spark.catalog.clearCache()


def city_stats(biz_joined, min_reviews: int, top_n: int) -> None:
    print("\n[CITY STATS]")
    stats = (
        biz_joined.groupBy("state", "city")
        .agg(
            F.sum("n_reviews").alias("n_reviews"),
            F.round(F.sum("sum_stars") / F.sum("n_reviews"), 3).alias("avg_stars"),
        )
        .filter(F.col("n_reviews") >= min_reviews)
        .orderBy(F.desc("avg_stars"), F.desc("n_reviews"))
    )
    stats.show(top_n, truncate=False)


def business_stats(biz_joined, min_reviews: int, top_n: int) -> None:
    print("\n[BUSINESS STATS]")
    stats = (
        biz_joined.withColumn("avg_stars", F.round(F.col("sum_stars") / F.col("n_reviews"), 3))
        .filter(F.col("n_reviews") >= min_reviews)
        .select("business_id", "name", "city", "state", "n_reviews", "avg_stars")
        .orderBy(F.desc("avg_stars"), F.desc("n_reviews"))
    )
    stats.show(top_n, truncate=False)


def category_stats(biz_joined, min_reviews: int, top_n: int) -> None:
    print("\n[CATEGORY STATS]")
    stats = (
        biz_joined.select("categories", "n_reviews", "sum_stars")
        .filter(F.col("categories").isNotNull() & (F.trim(F.col("categories")) != ""))
        .withColumn("category", F.explode(F.split(F.col("categories"), r"\s*,\s*")))
        .filter(F.col("category") != "")
        .groupBy("category")
        .agg(
            F.sum("n_reviews").alias("n_reviews"),
            F.round(F.sum("sum_stars") / F.sum("n_reviews"), 3).alias("avg_stars"),
        )
        .filter(F.col("n_reviews") >= min_reviews)
        .orderBy(F.desc("avg_stars"), F.desc("n_reviews"))
    )
    stats.show(top_n, truncate=False)


def la_slice_stats(
    spark: SparkSession,
    base: Path,
    state: str,
    require_restaurants: bool,
    require_food: bool,
    use_explicit_rating: bool,
    sample_n: int = 0,
) -> None:
    print("\n[LA SLICE STATS]")
    print(
        f"[CONFIG] state={state}, require_restaurants={require_restaurants}, "
        f"require_food={require_food}, explicit_rating={use_explicit_rating}"
    )

    business = (
        spark.read.parquet((base / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
    )
    review = (
        spark.read.parquet((base / "yelp_academic_dataset_review").as_posix())
        .select("user_id", "business_id", "stars", "date")
    )

    biz = business.filter(F.col("state") == state)
    cat_col = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    cond = None
    if require_restaurants:
        cond = cat_col.contains("restaurants")
    if require_food:
        cond = (cond | cat_col.contains("food")) if cond is not None else cat_col.contains("food")
    if cond is not None:
        biz = biz.filter(cond)

    biz = biz.select("business_id").distinct().persist(StorageLevel.DISK_ONLY)
    n_businesses = biz.count()

    rvw = review
    if use_explicit_rating:
        rvw = rvw.filter(F.col("stars").isNotNull())

    rvw = rvw.join(biz, on="business_id", how="inner").persist(StorageLevel.DISK_ONLY)
    n_reviews = rvw.count()
    n_users = rvw.select("user_id").distinct().count()

    print(f"[COUNT] businesses={n_businesses}, reviews={n_reviews}, users={n_users}")
    (
        rvw.select(F.to_timestamp("date").alias("ts"))
        .filter(F.col("ts").isNotNull())
        .agg(F.min("ts").alias("min_date"), F.max("ts").alias("max_date"))
        .show(truncate=False)
    )
    if use_explicit_rating:
        rvw.agg(F.round(F.avg("stars"), 3).alias("avg_stars")).show(truncate=False)

    if sample_n > 0:
        rvw.show(sample_n, truncate=80)

    rvw.unpersist()
    biz.unpersist()


def la_user_activity_stats(
    spark: SparkSession,
    base: Path,
    state: str,
    require_restaurants: bool,
    require_food: bool,
    use_explicit_rating: bool,
    thresholds: list[int],
) -> None:
    print("\n[LA USER ACTIVITY]")
    print(
        f"[CONFIG] state={state}, require_restaurants={require_restaurants}, "
        f"require_food={require_food}, explicit_rating={use_explicit_rating}"
    )

    business = (
        spark.read.parquet((base / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
    )
    review = (
        spark.read.parquet((base / "yelp_academic_dataset_review").as_posix())
        .select("user_id", "business_id", "stars", "date")
    )

    biz = business.filter(F.col("state") == state)
    cat_col = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    cond = None
    if require_restaurants:
        cond = cat_col.contains("restaurants")
    if require_food:
        cond = (cond | cat_col.contains("food")) if cond is not None else cat_col.contains("food")
    if cond is not None:
        biz = biz.filter(cond)

    biz = biz.select("business_id").distinct().persist(StorageLevel.DISK_ONLY)

    rvw = review
    if use_explicit_rating:
        rvw = rvw.filter(F.col("stars").isNotNull())
    rvw = rvw.join(biz, on="business_id", how="inner").persist(StorageLevel.DISK_ONLY)

    user_counts = rvw.groupBy("user_id").agg(F.count("*").alias("n_reviews"))

    (
        user_counts.agg(
            F.count("*").alias("n_users"),
            F.min("n_reviews").alias("min_reviews"),
            F.round(F.avg("n_reviews"), 2).alias("avg_reviews"),
            F.max("n_reviews").alias("max_reviews"),
        )
        .show(truncate=False)
    )

    quantiles = user_counts.approxQuantile("n_reviews", [0.5, 0.9, 0.95, 0.99], 0.01)
    print("[QUANTILES] p50/p90/p95/p99 =", [int(q) for q in quantiles])

    if thresholds:
        exprs = [
            F.sum(F.when(F.col("n_reviews") >= t, 1).otherwise(0)).alias(f"users_ge_{t}")
            for t in thresholds
        ]
        row = user_counts.agg(*exprs).first().asDict()
        total_users = user_counts.count()
        print(f"[TOTAL_USERS] {total_users}")
        for t in thresholds:
            key = f"users_ge_{t}"
            val = int(row.get(key, 0))
            pct = (val / total_users * 100.0) if total_users else 0.0
            print(f"[USERS >= {t}] {val} ({pct:.2f}%)")

    rvw.unpersist()
    biz.unpersist()


def la_leave_last_out_stats(
    spark: SparkSession,
    base: Path,
    state: str,
    require_restaurants: bool,
    require_food: bool,
    use_explicit_rating: bool,
    min_user_reviews: int,
    sample_n: int = 0,
) -> None:
    print("\n[LA LEAVE-LAST-OUT]")
    print(
        f"[CONFIG] state={state}, require_restaurants={require_restaurants}, "
        f"require_food={require_food}, explicit_rating={use_explicit_rating}, "
        f"min_user_reviews={min_user_reviews}"
    )

    business = (
        spark.read.parquet((base / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
    )
    review = (
        spark.read.parquet((base / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "stars", "date")
    )

    biz = business.filter(F.col("state") == state)
    cat_col = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    cond = None
    if require_restaurants:
        cond = cat_col.contains("restaurants")
    if require_food:
        cond = (cond | cat_col.contains("food")) if cond is not None else cat_col.contains("food")
    if cond is not None:
        biz = biz.filter(cond)

    biz = biz.select("business_id").distinct().persist(StorageLevel.DISK_ONLY)

    rvw = review
    if use_explicit_rating:
        rvw = rvw.filter(F.col("stars").isNotNull())
    rvw = rvw.join(biz, on="business_id", how="inner").persist(StorageLevel.DISK_ONLY)

    user_counts = rvw.groupBy("user_id").agg(F.count("*").alias("n_reviews"))
    eligible_users = user_counts.filter(F.col("n_reviews") >= min_user_reviews).select("user_id")

    rvw = rvw.join(eligible_users, on="user_id", how="inner")
    rvw = rvw.withColumn("ts", F.to_timestamp("date")).filter(F.col("ts").isNotNull())

    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = rvw.withColumn("rn", F.row_number().over(w)).persist(StorageLevel.DISK_ONLY)
    test = ranked.filter(F.col("rn") == 1)
    train = ranked.filter(F.col("rn") > 1)

    n_users = ranked.select("user_id").distinct().count()
    n_items = ranked.select("business_id").distinct().count()
    n_train = train.count()
    n_test = test.count()

    print(f"[COUNT] users={n_users}, items={n_items}, train={n_train}, test={n_test}")

    train_items = train.select("business_id").distinct()
    cold_test_items = test.select("business_id").distinct().join(
        train_items, on="business_id", how="left_anti"
    )
    n_cold_items = cold_test_items.count()
    if n_cold_items > 0:
        print(f"[WARN] test items not in train: {n_cold_items}")
    else:
        print("[OK] all test items appear in train")

    if sample_n > 0:
        test.select("user_id", "business_id", "stars", "date").show(sample_n, truncate=80)

    ranked.unpersist()
    rvw.unpersist()
    biz.unpersist()


def state_stats(business, rvw_by_biz, top_n: int) -> None:
    print("\n[STATE STATS]")
    biz_counts = (
        business.select("state")
        .groupBy("state")
        .agg(F.count("*").alias("n_businesses"))
    )
    rvw_counts = (
        rvw_by_biz.join(business.select("business_id", "state"), on="business_id", how="inner")
        .groupBy("state")
        .agg(F.sum("n_reviews").alias("n_reviews"))
    )
    stats = (
        biz_counts.join(rvw_counts, on="state", how="left")
        .fillna(0, subset=["n_reviews"])
        .orderBy(F.desc("n_reviews"), F.desc("n_businesses"))
    )
    stats.show(top_n, truncate=False)


def profile_user(spark: SparkSession, base: Path) -> None:
    print("\n[USER]")
    df = spark.read.parquet((base / "yelp_academic_dataset_user").as_posix()).select(
        "user_id",
        "name",
        "review_count",
        "average_stars",
    )
    df.show(5, truncate=False)
    (
        df.agg(
            F.count("*").alias("n_users"),
            F.round(F.avg("review_count"), 2).alias("avg_review_count"),
            F.round(F.avg("average_stars"), 3).alias("avg_user_stars"),
        )
        .show(truncate=False)
    )


def profile_tip(spark: SparkSession, base: Path) -> None:
    print("\n[TIP]")
    df = spark.read.parquet((base / "yelp_academic_dataset_tip").as_posix()).select(
        "user_id",
        "business_id",
        "text",
    )
    df.show(5, truncate=False)
    (
        df.agg(
            F.count("*").alias("n_tips"),
            F.approx_count_distinct("user_id").alias("n_users"),
            F.approx_count_distinct("business_id").alias("n_businesses"),
            F.round(F.avg(F.length("text")), 1).alias("avg_text_len"),
        )
        .show(truncate=False)
    )


def profile_checkin(spark: SparkSession, base: Path) -> None:
    print("\n[CHECKIN]")
    df = spark.read.parquet((base / "yelp_academic_dataset_checkin").as_posix()).select(
        "business_id",
        "date",
    )
    df.show(5, truncate=False)
    checkin_counts = df.select(
        F.size(F.split(F.coalesce(F.col("date"), F.lit("")), ",")).alias("n_checkins")
    )
    (
        checkin_counts.agg(
            F.count("*").alias("n_businesses"),
            F.round(F.avg("n_checkins"), 2).alias("avg_checkins"),
            F.max("n_checkins").alias("max_checkins"),
        )
        .show(truncate=False)
    )


def time_dist_review(spark: SparkSession, base: Path) -> None:
    print("\n[TIME DISTRIBUTION - REVIEW]")
    df = spark.read.parquet((base / "yelp_academic_dataset_review").as_posix()).select("date")
    ts = F.to_timestamp("date")
    clean = df.select(ts.alias("ts")).filter(F.col("ts").isNotNull())
    (
        clean.agg(F.min("ts").alias("min_date"), F.max("ts").alias("max_date"))
        .show(truncate=False)
    )
    (
        clean.withColumn("ym", F.date_format("ts", "yyyy-MM"))
        .groupBy("ym")
        .count()
        .orderBy("ym")
        .show(150, truncate=False)
    )


def time_dist_tip(spark: SparkSession, base: Path) -> None:
    print("\n[TIME DISTRIBUTION - TIP]")
    df = spark.read.parquet((base / "yelp_academic_dataset_tip").as_posix()).select("date")
    ts = F.to_timestamp("date")
    clean = df.select(ts.alias("ts")).filter(F.col("ts").isNotNull())
    (
        clean.agg(F.min("ts").alias("min_date"), F.max("ts").alias("max_date"))
        .show(truncate=False)
    )
    (
        clean.withColumn("ym", F.date_format("ts", "yyyy-MM"))
        .groupBy("ym")
        .count()
        .orderBy("ym")
        .show(150, truncate=False)
    )


def time_dist_checkin(spark: SparkSession, base: Path) -> None:
    print("\n[TIME DISTRIBUTION - CHECKIN]")
    df = spark.read.parquet((base / "yelp_academic_dataset_checkin").as_posix()).select("date")
    exploded = (
        df.select(
            F.explode(
                F.split(F.coalesce(F.col("date"), F.lit("")), r",\s*")
            ).alias("ts_str")
        )
        .filter(F.col("ts_str") != "")
    )
    ts = F.to_timestamp("ts_str")
    clean = exploded.select(ts.alias("ts")).filter(F.col("ts").isNotNull())
    (
        clean.agg(F.min("ts").alias("min_date"), F.max("ts").alias("max_date"))
        .show(truncate=False)
    )
    (
        clean.withColumn("ym", F.date_format("ts", "yyyy-MM"))
        .groupBy("ym")
        .count()
        .orderBy("ym")
        .show(150, truncate=False)
    )


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    base = Path(r"D:/5006 BDA project/data/parquet")
    any_output = any(
        [
            SHOW_TABLE_FIELDS,
            RUN_CITY_STATS,
            RUN_BUSINESS_STATS,
            RUN_CATEGORY_STATS,
            RUN_STATE_STATS,
            RUN_LA_SLICE_STATS,
            RUN_LA_USER_ACTIVITY_STATS,
            RUN_LA_LEAVE_LAST_OUT,
            RUN_PROFILE_USER,
            RUN_PROFILE_TIP,
            RUN_PROFILE_CHECKIN,
            RUN_TIME_DIST_REVIEW,
            RUN_TIME_DIST_TIP,
            RUN_TIME_DIST_CHECKIN,
        ]
    )
    if not any_output:
        print("[INFO] No modules enabled. Toggle flags at the top of the file.")

    if SHOW_TABLE_FIELDS:
        show_table_fields(spark, base, SHOW_SAMPLE_ROWS)

    if RUN_LA_SLICE_STATS:
        la_slice_stats(
            spark,
            base,
            TARGET_STATE,
            REQUIRE_RESTAURANTS,
            REQUIRE_FOOD,
            USE_EXPLICIT_RATING,
            SLICE_SHOW_SAMPLE,
        )

    if RUN_LA_USER_ACTIVITY_STATS:
        la_user_activity_stats(
            spark,
            base,
            TARGET_STATE,
            REQUIRE_RESTAURANTS,
            REQUIRE_FOOD,
            USE_EXPLICIT_RATING,
            USER_COUNT_THRESHOLDS,
        )

    if RUN_LA_LEAVE_LAST_OUT:
        la_leave_last_out_stats(
            spark,
            base,
            TARGET_STATE,
            REQUIRE_RESTAURANTS,
            REQUIRE_FOOD,
            USE_EXPLICIT_RATING,
            MIN_USER_REVIEWS_FOR_SPLIT,
            LEAVE_LAST_SHOW_SAMPLE,
        )

    need_business = RUN_CITY_STATS or RUN_BUSINESS_STATS or RUN_CATEGORY_STATS or RUN_STATE_STATS
    need_review = RUN_CITY_STATS or RUN_BUSINESS_STATS or RUN_CATEGORY_STATS or RUN_STATE_STATS

    business = None
    review = None
    rvw_by_biz = None
    biz_joined = None

    if need_business:
        business = spark.read.parquet((base / "yelp_academic_dataset_business").as_posix())
    if need_review:
        review = spark.read.parquet((base / "yelp_academic_dataset_review").as_posix())
        rvw_by_biz = (
            review.select("business_id", "stars")
            .groupBy("business_id")
            .agg(
                F.count("*").alias("n_reviews"),
                F.sum("stars").alias("sum_stars"),
            )
        )

    need_biz_joined = RUN_CITY_STATS or RUN_BUSINESS_STATS or RUN_CATEGORY_STATS
    if need_biz_joined and business is not None and rvw_by_biz is not None:
        biz = business.select("business_id", "name", "city", "state", "categories")
        biz_joined = rvw_by_biz.join(biz, on="business_id", how="inner").persist(StorageLevel.DISK_ONLY)

    if RUN_CITY_STATS and biz_joined is not None:
        city_stats(biz_joined, CITY_MIN_REVIEWS, CITY_TOP_N)
    if RUN_BUSINESS_STATS and biz_joined is not None:
        business_stats(biz_joined, BUSINESS_MIN_REVIEWS, BUSINESS_TOP_N)
    if RUN_CATEGORY_STATS and biz_joined is not None:
        category_stats(biz_joined, CATEGORY_MIN_REVIEWS, CATEGORY_TOP_N)

    if biz_joined is not None:
        biz_joined.unpersist()

    if RUN_STATE_STATS and business is not None and rvw_by_biz is not None:
        state_stats(business, rvw_by_biz, STATE_TOP_N)

    if RUN_PROFILE_USER:
        profile_user(spark, base)
        spark.catalog.clearCache()
    if RUN_PROFILE_TIP:
        profile_tip(spark, base)
        spark.catalog.clearCache()
    if RUN_PROFILE_CHECKIN:
        profile_checkin(spark, base)
        spark.catalog.clearCache()

    if RUN_TIME_DIST_REVIEW:
        time_dist_review(spark, base)
        spark.catalog.clearCache()
    if RUN_TIME_DIST_TIP:
        time_dist_tip(spark, base)
        spark.catalog.clearCache()
    if RUN_TIME_DIST_CHECKIN:
        time_dist_checkin(spark, base)
        spark.catalog.clearCache()

    spark.stop()


if __name__ == "__main__":
    main()
