import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window


TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
USE_EXPLICIT_RATING = True

QUIET_LOGS = True
RUN_BASELINES = True
RUN_ALS = True
RUN_IMPLICIT_ALS = True
RUN_GRID = True
SELECT_BEST = True
SUMMARY_BEST = True
WRITE_RESULTS = True

MIN_TRAIN_REVIEWS_BUCKETS = [2, 5, 10]
MIN_USER_REVIEWS_OFFSET = 2  # for second-last-out: min_user = min_train + 2
SAMPLE_FRACTION = 0.0  # 0.0 disables sampling
RANDOM_SEED = 42

ALS_RANK = 20
ALS_MAX_ITER = 5
ALS_REG_PARAM = 0.1
ALS_IMPLICIT_ALPHA = 40.0
IMPLICIT_GRID_DEFAULT = [
    {"rank": 20, "reg": 0.05, "alpha": 10.0},
    {"rank": 30, "reg": 0.1, "alpha": 20.0},
    {"rank": 50, "reg": 0.2, "alpha": 40.0},
    {"rank": 50, "reg": 0.2, "alpha": 60.0},
]
# Optional per-bucket grids (key = min_train_reviews). Falls back to IMPLICIT_GRID_DEFAULT.
IMPLICIT_GRID_BY_BUCKET = {
    2: [
        {"rank": 50, "reg": 0.1, "alpha": 40.0},
        {"rank": 30, "reg": 0.1, "alpha": 20.0},
    ],
    5: [
        {"rank": 20, "reg": 0.1, "alpha": 20.0},
        {"rank": 30, "reg": 0.1, "alpha": 20.0},
        {"rank": 50, "reg": 0.1, "alpha": 40.0},
    ],
    10: [
        {"rank": 20, "reg": 0.1, "alpha": 20.0},
        {"rank": 30, "reg": 0.1, "alpha": 20.0},
        {"rank": 50, "reg": 0.2, "alpha": 40.0},
    ],
}
TOP_K = 10
TUNE_METRIC = "ndcg"  # "ndcg" or "recall"

RESULTS_PATH = Path(r"D:/5006 BDA project/data/metrics/la_recsys_valid_test_results.csv")
RESULT_FIELDS = [
    "run_id",
    "bucket_min_train_reviews",
    "min_user_reviews",
    "model",
    "split",
    "rank",
    "reg",
    "alpha",
    "rmse",
    "recall_at_k",
    "ndcg_at_k",
    "top_k",
    "n_users",
    "n_items",
    "n_train",
    "n_valid",
    "n_test",
]


def build_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)

    log4j_path = None
    if QUIET_LOGS:
        log4j_path = local_dir / "log4j2.properties"
        if not log4j_path.exists():
            log4j_path.write_text(
                "\n".join(
                    [
                        "status = error",
                        "name = SparkLog4j2",
                        "appender.console.type = Console",
                        "appender.console.name = console",
                        "appender.console.target = SYSTEM_ERR",
                        "appender.console.layout.type = PatternLayout",
                        "appender.console.layout.pattern = %d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n",
                        "rootLogger.level = error",
                        "rootLogger.appenderRefs = console",
                        "rootLogger.appenderRef.console.ref = console",
                    ]
                ),
                encoding="utf-8",
            )

    builder = (
        SparkSession.builder
        .appName("yelp-la-recsys")
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
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.worker.connectTimeout", "120")
        .config("spark.network.timeout", "600s")
    )

    if log4j_path is not None:
        log4j_uri = log4j_path.as_uri()
        java_opts = f"-Dlog4j.configurationFile={log4j_uri}"
        builder = builder.config("spark.driver.extraJavaOptions", java_opts)
        builder = builder.config("spark.executor.extraJavaOptions", java_opts)

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


def load_business_filtered(spark: SparkSession, base: Path) -> DataFrame:
    business = (
        spark.read.parquet((base / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
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
    return biz.select("business_id", "categories").distinct()


def load_interactions(spark: SparkSession, base: Path, biz: DataFrame) -> DataFrame:
    review = (
        spark.read.parquet((base / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "stars", "date")
    )
    rvw = review.join(biz.select("business_id"), on="business_id", how="inner")

    if USE_EXPLICIT_RATING:
        rvw = rvw.filter(F.col("stars").isNotNull())

    rvw = rvw.withColumn("ts", F.to_timestamp("date")).filter(F.col("ts").isNotNull())

    if SAMPLE_FRACTION and SAMPLE_FRACTION > 0:
        rvw = rvw.sample(False, SAMPLE_FRACTION, RANDOM_SEED)

    return rvw


def leave_two_out(
    rvw: DataFrame, min_user_reviews: int
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    user_counts = rvw.groupBy("user_id").agg(F.count("*").alias("n_reviews"))
    eligible_users = user_counts.filter(F.col("n_reviews") >= min_user_reviews).select("user_id")
    rvw = rvw.join(eligible_users, on="user_id", how="inner")

    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = rvw.withColumn("rn", F.row_number().over(w))
    test = ranked.filter(F.col("rn") == 1)
    valid = ranked.filter(F.col("rn") == 2)
    train = ranked.filter(F.col("rn") > 2)
    return rvw, train, valid, test


def index_ids(
    full_df: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    user_indexer = StringIndexer(
        inputCol="user_id",
        outputCol="user_idx",
        handleInvalid="skip",
    ).fit(full_df)
    item_indexer = StringIndexer(
        inputCol="business_id",
        outputCol="item_idx",
        handleInvalid="skip",
    ).fit(full_df)

    def transform(df: DataFrame) -> DataFrame:
        out = user_indexer.transform(df)
        out = item_indexer.transform(out)
        out = (
            out.withColumn("user_idx", F.col("user_idx").cast("int"))
            .withColumn("item_idx", F.col("item_idx").cast("int"))
            .withColumn("rating", F.col("stars").cast("float"))
        )
        return out

    return transform(train), transform(valid), transform(test)


def build_item_categories(biz: DataFrame) -> DataFrame:
    return (
        biz.filter(F.col("categories").isNotNull() & (F.trim(F.col("categories")) != ""))
        .withColumn("category", F.explode(F.split(F.col("categories"), r"\s*,\s*")))
        .withColumn("category", F.lower(F.trim(F.col("category"))))
        .filter(F.col("category") != "")
        .select("business_id", "category")
        .distinct()
    )


def compute_recall_ndcg(eval_df: DataFrame) -> tuple[float, float]:
    metrics = (
        eval_df.withColumn(
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
    return float(row["recall"]), float(row["ndcg"])


def evaluate_topk(model: ALS, test_idx: DataFrame, top_k: int) -> tuple[float, float]:
    users = test_idx.select("user_idx").distinct()
    recs = model.recommendForUserSubset(users, top_k)
    pred = recs.select(
        "user_idx",
        F.expr("transform(recommendations, x -> int(x.item_idx))").alias("pred_items"),
    )
    truth = test_idx.groupBy("user_idx").agg(F.collect_set("item_idx").alias("true_items"))
    eval_df = (
        pred.join(truth, on="user_idx", how="inner")
        .withColumn("true_item", F.element_at("true_items", 1))
        .withColumn("rank", F.expr("array_position(pred_items, true_item)"))
        .select("user_idx", "rank")
    )
    return compute_recall_ndcg(eval_df)


def evaluate_popular_baseline(train: DataFrame, test: DataFrame, top_k: int) -> tuple[float, float]:
    pop = train.groupBy("business_id").agg(F.count("*").alias("cnt"))
    w = Window.orderBy(F.desc("cnt"), F.asc("business_id"))
    pop_ranked = pop.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= top_k)

    eval_df = (
        test.select("user_id", "business_id")
        .distinct()
        .join(pop_ranked.select("business_id", "rank"), on="business_id", how="left")
    )
    return compute_recall_ndcg(eval_df)


def evaluate_category_popular_baseline(
    train: DataFrame, test: DataFrame, biz: DataFrame, top_k: int
) -> tuple[float, float]:
    cats = build_item_categories(biz).persist(StorageLevel.DISK_ONLY)
    item_pop = train.groupBy("business_id").agg(F.count("*").alias("cnt"))
    cat_items = item_pop.join(cats, on="business_id", how="inner")
    w_cat = Window.partitionBy("category").orderBy(F.desc("cnt"), F.asc("business_id"))
    cat_topk = (
        cat_items.withColumn("rank", F.row_number().over(w_cat))
        .filter(F.col("rank") <= top_k)
        .select("category", "business_id", "rank")
    )

    user_cat = (
        train.join(cats, on="business_id", how="inner")
        .groupBy("user_id", "category")
        .agg(F.count("*").alias("cnt"))
    )
    w_user = Window.partitionBy("user_id").orderBy(F.desc("cnt"), F.asc("category"))
    user_top = user_cat.withColumn("rn", F.row_number().over(w_user)).filter(F.col("rn") == 1)
    user_top = user_top.select("user_id", "category")

    eval_df = (
        test.select("user_id", "business_id")
        .distinct()
        .join(user_top, on="user_id", how="left")
        .join(cat_topk, on=["category", "business_id"], how="left")
        .select("user_id", "business_id", "rank")
    )

    cats.unpersist()
    return compute_recall_ndcg(eval_df)


def implicit_grid_search(
    train_idx: DataFrame, valid_idx: DataFrame, grid: list[dict]
) -> tuple[dict, list[tuple[dict, float, float]]]:
    train_imp = train_idx.withColumn("rating", F.lit(1.0))
    results: list[tuple[dict, float, float]] = []
    best_params: dict | None = None
    best_score = -1.0

    for params in grid:
        als_imp = ALS(
            userCol="user_idx",
            itemCol="item_idx",
            ratingCol="rating",
            implicitPrefs=True,
            alpha=params["alpha"],
            coldStartStrategy="drop",
            rank=params["rank"],
            maxIter=ALS_MAX_ITER,
            regParam=params["reg"],
        )
        model_imp = als_imp.fit(train_imp)
        recall, ndcg = evaluate_topk(model_imp, valid_idx, TOP_K)
        results.append((params, recall, ndcg))

        score = ndcg if TUNE_METRIC == "ndcg" else recall
        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None and grid:
        best_params = grid[0]
    return best_params, results


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()


def append_result(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writerow(row)


def main() -> None:
    spark = build_spark()
    set_log_level(spark)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if WRITE_RESULTS:
        ensure_results_file(RESULTS_PATH)

    base = Path(r"D:/5006 BDA project/data/parquet")

    print("[STEP] load + filter businesses")
    biz = load_business_filtered(spark, base).persist(StorageLevel.DISK_ONLY)

    print("[STEP] load + filter interactions")
    rvw = load_interactions(spark, base, biz).persist(StorageLevel.DISK_ONLY)
    if rvw.rdd.isEmpty():
        print("[ERROR] no interactions after filtering")
        biz.unpersist()
        spark.stop()
        return

    n_users = rvw.select("user_id").distinct().count()
    n_items = rvw.select("business_id").distinct().count()
    n_events = rvw.count()
    print(f"[COUNT] users={n_users}, items={n_items}, events={n_events}")

    best_summary_rows: list[dict] = []
    for min_train in MIN_TRAIN_REVIEWS_BUCKETS:
        min_user_reviews = min_train + MIN_USER_REVIEWS_OFFSET
        print(f"\n[BUCKET] min_train_reviews={min_train} (min_user_reviews={min_user_reviews})")
        rvw_filtered, train, valid, test = leave_two_out(rvw, min_user_reviews)
        rvw_filtered = rvw_filtered.persist(StorageLevel.DISK_ONLY)
        train = train.persist(StorageLevel.DISK_ONLY)
        valid = valid.persist(StorageLevel.DISK_ONLY)
        test = test.persist(StorageLevel.DISK_ONLY)

        n_users = rvw_filtered.select("user_id").distinct().count()
        n_items = rvw_filtered.select("business_id").distinct().count()
        n_train = train.count()
        n_valid = valid.count()
        n_test = test.count()
        print(
            f"[COUNT] users={n_users}, items={n_items}, "
            f"train={n_train}, valid={n_valid}, test={n_test}"
        )

        if RUN_BASELINES:
            print(f"[STEP] baselines @ {TOP_K}")
            for split_name, split_df in [("valid", valid), ("test", test)]:
                pop_recall, pop_ndcg = evaluate_popular_baseline(train, split_df, TOP_K)
                print(
                    f"[Popular:{split_name}] Recall@{TOP_K}={pop_recall:.4f} "
                    f"NDCG@{TOP_K}={pop_ndcg:.4f}"
                )
                if WRITE_RESULTS:
                    append_result(
                        RESULTS_PATH,
                        {
                            "run_id": run_id,
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "model": "Popular",
                            "split": split_name,
                            "rank": "",
                            "reg": "",
                            "alpha": "",
                            "rmse": "",
                            "recall_at_k": f"{pop_recall:.6f}",
                            "ndcg_at_k": f"{pop_ndcg:.6f}",
                            "top_k": TOP_K,
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_train": n_train,
                            "n_valid": n_valid,
                            "n_test": n_test,
                        },
                    )

                cat_recall, cat_ndcg = evaluate_category_popular_baseline(train, split_df, biz, TOP_K)
                print(
                    f"[Category Popular:{split_name}] Recall@{TOP_K}={cat_recall:.4f} "
                    f"NDCG@{TOP_K}={cat_ndcg:.4f}"
                )
                if WRITE_RESULTS:
                    append_result(
                        RESULTS_PATH,
                        {
                            "run_id": run_id,
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "model": "Category Popular",
                            "split": split_name,
                            "rank": "",
                            "reg": "",
                            "alpha": "",
                            "rmse": "",
                            "recall_at_k": f"{cat_recall:.6f}",
                            "ndcg_at_k": f"{cat_ndcg:.6f}",
                            "top_k": TOP_K,
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_train": n_train,
                            "n_valid": n_valid,
                            "n_test": n_test,
                        },
                    )

        train_idx = None
        valid_idx = None
        test_idx = None
        if RUN_ALS or RUN_IMPLICIT_ALS:
            print("[STEP] index user/item ids")
            train_idx, valid_idx, test_idx = index_ids(rvw_filtered, train, valid, test)
            train_idx = train_idx.persist(StorageLevel.DISK_ONLY)
            valid_idx = valid_idx.persist(StorageLevel.DISK_ONLY)
            test_idx = test_idx.persist(StorageLevel.DISK_ONLY)

        if RUN_ALS:
            print("[STEP] fit ALS (explicit)")
            als = ALS(
                userCol="user_idx",
                itemCol="item_idx",
                ratingCol="rating",
                implicitPrefs=False,
                coldStartStrategy="drop",
                rank=ALS_RANK,
                maxIter=ALS_MAX_ITER,
                regParam=ALS_REG_PARAM,
            )
            model = als.fit(train_idx)

            for split_name, split_df in [("valid", valid_idx), ("test", test_idx)]:
                preds = model.transform(split_df)
                rmse = RegressionEvaluator(
                    metricName="rmse",
                    labelCol="rating",
                    predictionCol="prediction",
                ).evaluate(preds)
                recall, ndcg = evaluate_topk(model, split_df, TOP_K)
                print(
                    f"[Explicit ALS:{split_name}] RMSE={rmse:.4f} "
                    f"Recall@{TOP_K}={recall:.4f} NDCG@{TOP_K}={ndcg:.4f}"
                )
                if WRITE_RESULTS:
                    append_result(
                        RESULTS_PATH,
                        {
                            "run_id": run_id,
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "model": "Explicit ALS",
                            "split": split_name,
                            "rank": ALS_RANK,
                            "reg": ALS_REG_PARAM,
                            "alpha": "",
                            "rmse": f"{rmse:.6f}",
                            "recall_at_k": f"{recall:.6f}",
                            "ndcg_at_k": f"{ndcg:.6f}",
                            "top_k": TOP_K,
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_train": n_train,
                            "n_valid": n_valid,
                            "n_test": n_test,
                        },
                    )

        if RUN_IMPLICIT_ALS and RUN_GRID:
            grid = IMPLICIT_GRID_BY_BUCKET.get(min_train, IMPLICIT_GRID_DEFAULT)
            if not grid:
                print("[WARN] implicit grid is empty for this bucket")
            else:
                print("[STEP] implicit ALS grid (validate)")
                best_params, grid_results = implicit_grid_search(train_idx, valid_idx, grid)
                for params, recall, ndcg in grid_results:
                    print(
                        "[Implicit ALS:valid] "
                        f"rank={params['rank']} reg={params['reg']} alpha={params['alpha']} "
                        f"Recall@{TOP_K}={recall:.4f} NDCG@{TOP_K}={ndcg:.4f}"
                    )
                    if WRITE_RESULTS:
                        append_result(
                            RESULTS_PATH,
                            {
                                "run_id": run_id,
                                "bucket_min_train_reviews": min_train,
                                "min_user_reviews": min_user_reviews,
                                "model": "Implicit ALS",
                                "split": "valid",
                                "rank": params["rank"],
                                "reg": params["reg"],
                                "alpha": params["alpha"],
                                "rmse": "",
                                "recall_at_k": f"{recall:.6f}",
                                "ndcg_at_k": f"{ndcg:.6f}",
                                "top_k": TOP_K,
                                "n_users": n_users,
                                "n_items": n_items,
                                "n_train": n_train,
                                "n_valid": n_valid,
                                "n_test": n_test,
                            },
                        )

                print(
                    "[Implicit ALS:best] "
                    f"rank={best_params['rank']} reg={best_params['reg']} alpha={best_params['alpha']} "
                    f"(selected by {TUNE_METRIC})"
                )
                if SUMMARY_BEST:
                    best_summary_rows.append(
                        {
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "rank": best_params["rank"],
                            "reg": best_params["reg"],
                            "alpha": best_params["alpha"],
                        }
                    )

                if SELECT_BEST:
                    train_full_idx = train_idx.unionByName(valid_idx)
                    train_full_imp = train_full_idx.withColumn("rating", F.lit(1.0))
                    als_imp = ALS(
                        userCol="user_idx",
                        itemCol="item_idx",
                        ratingCol="rating",
                        implicitPrefs=True,
                        alpha=best_params["alpha"],
                        coldStartStrategy="drop",
                        rank=best_params["rank"],
                        maxIter=ALS_MAX_ITER,
                        regParam=best_params["reg"],
                    )
                    model_imp = als_imp.fit(train_full_imp)
                    recall, ndcg = evaluate_topk(model_imp, test_idx, TOP_K)
                    print(f"[Implicit ALS:test] Recall@{TOP_K}={recall:.4f} NDCG@{TOP_K}={ndcg:.4f}")
                    if WRITE_RESULTS:
                        append_result(
                            RESULTS_PATH,
                            {
                                "run_id": run_id,
                                "bucket_min_train_reviews": min_train,
                                "min_user_reviews": min_user_reviews,
                                "model": "Implicit ALS (best)",
                                "split": "test",
                                "rank": best_params["rank"],
                                "reg": best_params["reg"],
                                "alpha": best_params["alpha"],
                                "rmse": "",
                                "recall_at_k": f"{recall:.6f}",
                                "ndcg_at_k": f"{ndcg:.6f}",
                                "top_k": TOP_K,
                                "n_users": n_users,
                                "n_items": n_items,
                                "n_train": n_train,
                                "n_valid": n_valid,
                                "n_test": n_test,
                            },
                        )
        elif RUN_IMPLICIT_ALS:
            print("[STEP] fit ALS (implicit)")
            train_full_idx = train_idx.unionByName(valid_idx)
            train_full_imp = train_full_idx.withColumn("rating", F.lit(1.0))
            als_imp = ALS(
                userCol="user_idx",
                itemCol="item_idx",
                ratingCol="rating",
                implicitPrefs=True,
                alpha=ALS_IMPLICIT_ALPHA,
                coldStartStrategy="drop",
                rank=ALS_RANK,
                maxIter=ALS_MAX_ITER,
                regParam=ALS_REG_PARAM,
            )
            model_imp = als_imp.fit(train_full_imp)
            recall, ndcg = evaluate_topk(model_imp, test_idx, TOP_K)
            print(f"[Implicit ALS:test] Recall@{TOP_K} {recall:.4f}")
            print(f"[Implicit ALS:test] NDCG@{TOP_K} {ndcg:.4f}")
            if WRITE_RESULTS:
                append_result(
                    RESULTS_PATH,
                    {
                        "run_id": run_id,
                        "bucket_min_train_reviews": min_train,
                        "min_user_reviews": min_user_reviews,
                        "model": "Implicit ALS",
                        "split": "test",
                        "rank": ALS_RANK,
                        "reg": ALS_REG_PARAM,
                        "alpha": ALS_IMPLICIT_ALPHA,
                        "rmse": "",
                        "recall_at_k": f"{recall:.6f}",
                        "ndcg_at_k": f"{ndcg:.6f}",
                        "top_k": TOP_K,
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_train": n_train,
                        "n_valid": n_valid,
                        "n_test": n_test,
                    },
                )

        if train_idx is not None:
            train_idx.unpersist()
        if valid_idx is not None:
            valid_idx.unpersist()
        if test_idx is not None:
            test_idx.unpersist()
        rvw_filtered.unpersist()
        train.unpersist()
        valid.unpersist()
        test.unpersist()

    rvw.unpersist()
    biz.unpersist()
    if SUMMARY_BEST and best_summary_rows:
        print("\n[BEST PARAMS SUMMARY]")
        for row in best_summary_rows:
            print(
                f"min_train={row['bucket_min_train_reviews']} "
                f"(min_user={row['min_user_reviews']}) "
                f"rank={row['rank']} reg={row['reg']} alpha={row['alpha']}"
            )
    if WRITE_RESULTS:
        print(f"[INFO] results saved to {RESULTS_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
