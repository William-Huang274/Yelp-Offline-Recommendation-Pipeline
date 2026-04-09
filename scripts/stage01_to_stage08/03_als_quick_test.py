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


# === Dataset config (edit these for each dataset) ===
INTERACTIONS_PATH = Path(r"D:/5006 BDA project/data/parquet/yelp_academic_dataset_review")
USER_COL = "user_id"
ITEM_COL = "business_id"
RATING_COL = "stars"
TIME_COL = "date"
TIE_BREAK_COL = "review_id"  # optional, set "" to ignore
FILTER_EXPR = ""  # e.g. "state = 'LA'" if column exists

# === Experiment config ===
USE_EXPLICIT_RATING = True
MIN_USER_REVIEWS = 2
SAMPLE_FRACTION = 0.0  # 0.0 disables sampling
RANDOM_SEED = 42
TOP_K = 10

RUN_BASELINES = True
RUN_EXPLICIT_ALS = True
RUN_IMPLICIT_ALS = True

ALS_RANK = 20
ALS_MAX_ITER = 5
ALS_REG_PARAM = 0.1
ALS_IMPLICIT_ALPHA = 40.0

QUIET_LOGS = True
WRITE_RESULTS = True
RESULTS_PATH = Path(r"D:/5006 BDA project/data/metrics/als_quick_test_results.csv")
RESULT_FIELDS = [
    "run_id",
    "dataset_path",
    "model",
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
        .appName("als-quick-test")
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


def load_interactions(spark: SparkSession) -> DataFrame:
    df = spark.read.parquet(INTERACTIONS_PATH.as_posix())
    required = [USER_COL, ITEM_COL, TIME_COL]
    if USE_EXPLICIT_RATING:
        required.append(RATING_COL)
    if TIE_BREAK_COL:
        required.append(TIE_BREAK_COL)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in interactions: {missing}")

    df = df.select(*required)
    if FILTER_EXPR:
        df = df.filter(F.expr(FILTER_EXPR))
    if USE_EXPLICIT_RATING:
        df = df.filter(F.col(RATING_COL).isNotNull())

    df = df.withColumn("ts", F.to_timestamp(F.col(TIME_COL))).filter(F.col("ts").isNotNull())

    if SAMPLE_FRACTION and SAMPLE_FRACTION > 0:
        df = df.sample(False, SAMPLE_FRACTION, RANDOM_SEED)
    return df


def leave_last_out(rvw: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    user_counts = rvw.groupBy(USER_COL).agg(F.count("*").alias("n_reviews"))
    eligible = user_counts.filter(F.col("n_reviews") >= MIN_USER_REVIEWS).select(USER_COL)
    rvw = rvw.join(eligible, on=USER_COL, how="inner")

    order_cols = [F.col("ts").desc()]
    if TIE_BREAK_COL:
        order_cols.append(F.col(TIE_BREAK_COL).desc())
    w = Window.partitionBy(USER_COL).orderBy(*order_cols)
    ranked = rvw.withColumn("rn", F.row_number().over(w))
    test = ranked.filter(F.col("rn") == 1)
    train = ranked.filter(F.col("rn") > 1)
    return rvw, train, test


def index_ids(full_df: DataFrame, train: DataFrame, test: DataFrame) -> tuple[DataFrame, DataFrame]:
    user_indexer = StringIndexer(
        inputCol=USER_COL,
        outputCol="user_idx",
        handleInvalid="skip",
    ).fit(full_df)
    item_indexer = StringIndexer(
        inputCol=ITEM_COL,
        outputCol="item_idx",
        handleInvalid="skip",
    ).fit(full_df)

    def transform(df: DataFrame) -> DataFrame:
        out = user_indexer.transform(df)
        out = item_indexer.transform(out)
        out = (
            out.withColumn("user_idx", F.col("user_idx").cast("int"))
            .withColumn("item_idx", F.col("item_idx").cast("int"))
            .withColumn("rating", F.col(RATING_COL).cast("float"))
        )
        return out

    return transform(train), transform(test)


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
    pop = train.groupBy(ITEM_COL).agg(F.count("*").alias("cnt"))
    w = Window.orderBy(F.desc("cnt"), F.asc(ITEM_COL))
    pop_ranked = pop.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= top_k)

    eval_df = (
        test.select(USER_COL, ITEM_COL)
        .distinct()
        .join(pop_ranked.select(ITEM_COL, "rank"), on=ITEM_COL, how="left")
    )
    return compute_recall_ndcg(eval_df)


def main() -> None:
    spark = build_spark()
    set_log_level(spark)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if WRITE_RESULTS:
        ensure_results_file(RESULTS_PATH)

    print("[STEP] load interactions")
    rvw = load_interactions(spark).persist(StorageLevel.DISK_ONLY)
    if rvw.rdd.isEmpty():
        print("[ERROR] no interactions after filtering")
        spark.stop()
        return

    n_users = rvw.select(USER_COL).distinct().count()
    n_items = rvw.select(ITEM_COL).distinct().count()
    n_events = rvw.count()
    print(f"[COUNT] users={n_users}, items={n_items}, events={n_events}")

    print("[STEP] leave-last-out split")
    rvw_filtered, train, test = leave_last_out(rvw)
    rvw_filtered = rvw_filtered.persist(StorageLevel.DISK_ONLY)
    train = train.persist(StorageLevel.DISK_ONLY)
    test = test.persist(StorageLevel.DISK_ONLY)

    n_users = rvw_filtered.select(USER_COL).distinct().count()
    n_items = rvw_filtered.select(ITEM_COL).distinct().count()
    n_train = train.count()
    n_test = test.count()
    print(f"[COUNT] users={n_users}, items={n_items}, train={n_train}, test={n_test}")

    if RUN_BASELINES:
        print(f"[STEP] baselines @ {TOP_K}")
        pop_recall, pop_ndcg = evaluate_popular_baseline(train, test, TOP_K)
        print(f"[Popular] Recall@{TOP_K}={pop_recall:.4f} NDCG@{TOP_K}={pop_ndcg:.4f}")
        if WRITE_RESULTS:
            append_result(
                RESULTS_PATH,
                {
                    "run_id": run_id,
                    "dataset_path": str(INTERACTIONS_PATH),
                    "model": "Popular",
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
                    "n_test": n_test,
                },
            )

    train_idx = None
    test_idx = None
    if RUN_EXPLICIT_ALS or RUN_IMPLICIT_ALS:
        print("[STEP] index user/item ids")
        train_idx, test_idx = index_ids(rvw_filtered, train, test)
        train_idx = train_idx.persist(StorageLevel.DISK_ONLY)
        test_idx = test_idx.persist(StorageLevel.DISK_ONLY)

    if RUN_EXPLICIT_ALS:
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
        preds = model.transform(test_idx)
        rmse = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        ).evaluate(preds)
        recall, ndcg = evaluate_topk(model, test_idx, TOP_K)
        print(f"[Explicit ALS] RMSE={rmse:.4f} Recall@{TOP_K}={recall:.4f} NDCG@{TOP_K}={ndcg:.4f}")
        if WRITE_RESULTS:
            append_result(
                RESULTS_PATH,
                {
                    "run_id": run_id,
                    "dataset_path": str(INTERACTIONS_PATH),
                    "model": "Explicit ALS",
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
                    "n_test": n_test,
                },
            )

    if RUN_IMPLICIT_ALS:
        print("[STEP] fit ALS (implicit)")
        train_imp = train_idx.withColumn("rating", F.lit(1.0))
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
        model_imp = als_imp.fit(train_imp)
        recall, ndcg = evaluate_topk(model_imp, test_idx, TOP_K)
        print(f"[Implicit ALS] Recall@{TOP_K}={recall:.4f} NDCG@{TOP_K}={ndcg:.4f}")
        if WRITE_RESULTS:
            append_result(
                RESULTS_PATH,
                {
                    "run_id": run_id,
                    "dataset_path": str(INTERACTIONS_PATH),
                    "model": "Implicit ALS",
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
                    "n_test": n_test,
                },
            )

    if train_idx is not None:
        train_idx.unpersist()
    if test_idx is not None:
        test_idx.unpersist()
    rvw.unpersist()
    rvw_filtered.unpersist()
    train.unpersist()
    test.unpersist()

    if WRITE_RESULTS:
        print(f"[INFO] results saved to {RESULTS_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
