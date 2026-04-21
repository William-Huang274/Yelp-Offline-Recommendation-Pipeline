import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("Usage: python scripts/stage01_to_stage08/04_la_recsys_valid.py")
    print("Runs stage04 validation / hybrid recommendation experiments on the LA slice.")
    print("Set parquet, metrics, and Spark env vars, then run without --help.")
    sys.exit(0)

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
RUN_ALS = False
RUN_IMPLICIT_ALS = True
RUN_GRID = False
SELECT_BEST = True
SUMMARY_BEST = True
WRITE_RESULTS = True
RUN_HYBRID = True
RUN_HYBRID_CLUSTER = True
RUN_HYBRID_CLUSTER_RERANK = True

MIN_TRAIN_REVIEWS_BUCKETS = [2, 5, 10]
MIN_USER_REVIEWS_OFFSET = 2  # for second-last-out: min_user = min_train + 2
SAMPLE_FRACTION = 0.0  # 0.0 disables sampling
RANDOM_SEED = 42

ALS_RANK = 20
ALS_MAX_ITER = 5
ALS_REG_PARAM = 0.1
ALS_IMPLICIT_ALPHA = 40.0
USE_PRESET_IMPLICIT_BEST = True
IMPLICIT_PRESET_BEST_BY_BUCKET = {
    2: {"rank": 20, "reg": 0.1, "alpha": 20.0},
    5: {"rank": 20, "reg": 0.1, "alpha": 20.0},
    10: {"rank": 20, "reg": 0.1, "alpha": 20.0},
}
IMPLICIT_GRID_DEFAULT = [
    {"rank": 20, "reg": 0.05, "alpha": 10.0},
    {"rank": 30, "reg": 0.1, "alpha": 20.0},
    {"rank": 50, "reg": 0.2, "alpha": 40.0},
    {"rank": 50, "reg": 0.2, "alpha": 60.0},
]
# Optional per-bucket grids (key = min_train_reviews). Falls back to IMPLICIT_GRID_DEFAULT.
IMPLICIT_GRID_BY_BUCKET = {
    2: [
    {"rank": 20, "reg": 0.1, "alpha": 20.0},
    {"rank": 30, "reg": 0.1, "alpha": 20.0},
    {"rank": 50, "reg": 0.1, "alpha": 20.0},
    ],
    5: [
    {"rank": 20, "reg": 0.1, "alpha": 20.0},
    {"rank": 30, "reg": 0.1, "alpha": 20.0},
    {"rank": 50, "reg": 0.1, "alpha": 20.0},
    ],
    10: [
    {"rank": 20, "reg": 0.1, "alpha": 20.0},
    {"rank": 30, "reg": 0.1, "alpha": 20.0},
    {"rank": 50, "reg": 0.1, "alpha": 20.0},
    ],
}
TOP_K = 10
TUNE_METRIC = "ndcg"  # "ndcg" or "recall"
HYBRID_MIN_TRAIN_DEFAULT = 5
HYBRID_MIN_TRAIN_BY_BUCKET = {2: 5, 5: 8, 10: 12}
TAIL_QUANTILE = 0.8
CLUSTER_PROFILE_CSV = ""  # optional explicit: .../biz_profile_recsys.csv
CLUSTER_PROFILE_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels/full")
CLUSTER_PROFILE_DIR_SUFFIX = "_full_profile_merged"
CLUSTER_PROFILE_FILENAME = "biz_profile_recsys.csv"
CLUSTER_PROFILE_CLUSTER_COL = "cluster_for_recsys"
RERANK_CANDIDATE_MULTIPLIER = 3
RERANK_CLUSTER_BONUS = 0.15

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
    "user_coverage_at_k",
    "item_coverage_at_k",
    "tail_coverage_at_k",
    "novelty_at_k",
    "hybrid_min_train_threshold",
    "hybrid_heavy_users",
    "hybrid_light_users",
    "hybrid_light_cluster_user_coverage",
    "rerank_user_cluster_coverage",
]


def build_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")
    local_dir = Path(os.environ.get("SPARK_LOCAL_DIR", r"D:/5006 BDA project/data/spark-tmp"))
    py_tmp_dir = Path(os.environ.get("SPARK_PY_TMP_DIR", str(local_dir / "py-tmp")))
    local_dir.mkdir(parents=True, exist_ok=True)
    py_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Force Python/JVM temp spill away from C:.
    os.environ["TEMP"] = str(py_tmp_dir)
    os.environ["TMP"] = str(py_tmp_dir)
    os.environ["TMPDIR"] = str(py_tmp_dir)
    os.environ["SPARK_LOCAL_DIRS"] = str(local_dir)

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

    java_opts_parts = [f"-Djava.io.tmpdir={py_tmp_dir.as_posix()}"]
    if log4j_path is not None:
        java_opts_parts.append(f"-Dlog4j.configurationFile={log4j_path.as_uri()}")
    java_opts = " ".join(java_opts_parts)

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
        .config("spark.executorEnv.TEMP", str(py_tmp_dir))
        .config("spark.executorEnv.TMP", str(py_tmp_dir))
        .config("spark.executorEnv.TMPDIR", str(py_tmp_dir))
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
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


def resolve_cluster_profile_csv() -> Optional[Path]:
    if CLUSTER_PROFILE_CSV.strip():
        p = Path(CLUSTER_PROFILE_CSV.strip())
        if p.exists():
            return p
        print(f"[WARN] CLUSTER_PROFILE_CSV not found: {p}")
        return None
    if not CLUSTER_PROFILE_ROOT.exists():
        print(f"[WARN] CLUSTER_PROFILE_ROOT not found: {CLUSTER_PROFILE_ROOT}")
        return None
    runs = [
        p for p in CLUSTER_PROFILE_ROOT.iterdir()
        if p.is_dir() and p.name.endswith(CLUSTER_PROFILE_DIR_SUFFIX)
    ]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        cand = run / CLUSTER_PROFILE_FILENAME
        if cand.exists():
            return cand
    print(
        f"[WARN] no cluster profile csv found under {CLUSTER_PROFILE_ROOT} "
        f"(suffix={CLUSTER_PROFILE_DIR_SUFFIX})"
    )
    return None


def load_cluster_profile_map(spark: SparkSession, profile_csv: Path) -> DataFrame:
    sdf = spark.read.csv(profile_csv.as_posix(), header=True)
    if "business_id" not in sdf.columns:
        raise RuntimeError(f"cluster profile missing business_id: {profile_csv}")
    if CLUSTER_PROFILE_CLUSTER_COL not in sdf.columns:
        raise RuntimeError(
            f"cluster profile missing {CLUSTER_PROFILE_CLUSTER_COL}: {profile_csv}"
        )
    return (
        sdf.select(
            F.col("business_id").cast("string").alias("business_id"),
            F.col(CLUSTER_PROFILE_CLUSTER_COL).cast("string").alias("cluster_id"),
        )
        .filter(F.col("business_id").isNotNull())
        .filter(F.col("cluster_id").isNotNull())
        .filter(F.col("cluster_id") != "")
        .dropDuplicates(["business_id"])
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
    pred = recs_to_pred_items(recs)
    truth = test_idx.groupBy("user_idx").agg(F.collect_set("item_idx").alias("true_items"))
    eval_df = (
        pred.join(truth, on="user_idx", how="inner")
        .withColumn("true_item", F.element_at("true_items", 1))
        .withColumn("rank", F.expr("array_position(pred_items, true_item)"))
        .select("user_idx", "rank")
    )
    return compute_recall_ndcg(eval_df)


def recs_to_pred_items(recs: DataFrame) -> DataFrame:
    return recs.select(
        "user_idx",
        F.expr("transform(recommendations, x -> int(x.item_idx))").alias("pred_items"),
    )


def build_item_pop_stats(
    train_full_idx: DataFrame, tail_quantile: float = TAIL_QUANTILE
) -> tuple[DataFrame, float]:
    item_pop = train_full_idx.groupBy("item_idx").agg(F.count("*").alias("cnt"))
    total_events = train_full_idx.count()
    default_pop_prob = float(1.0 / max(1, total_events))
    quantile = item_pop.approxQuantile("cnt", [tail_quantile], 0.01)
    tail_cutoff = float(quantile[0]) if quantile else 0.0
    stats = (
        item_pop.withColumn(
            "pop_prob",
            F.col("cnt").cast("double") / F.lit(float(max(1, total_events))),
        )
        .withColumn(
            "is_tail",
            F.when(F.col("cnt").cast("double") <= F.lit(tail_cutoff), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .select("item_idx", "pop_prob", "is_tail")
    )
    return stats, default_pop_prob


def evaluate_pred_items_with_diagnostics(
    pred_items: DataFrame,
    test_idx: DataFrame,
    item_pop_stats: DataFrame,
    default_pop_prob: float,
    n_items: int,
) -> tuple[float, float, float, float, float, float]:
    eval_users = test_idx.select("user_idx").distinct()
    n_eval_users = eval_users.count()

    pred_user = pred_items.select("user_idx", "pred_items").dropDuplicates(["user_idx"])
    eval_pred = (
        eval_users.join(pred_user, on="user_idx", how="left")
        .withColumn("pred_items", F.coalesce(F.col("pred_items"), F.expr("cast(array() as array<int>)")))
    )
    truth = test_idx.groupBy("user_idx").agg(F.first("item_idx").cast("int").alias("true_item"))
    eval_rank = (
        eval_pred.join(truth, on="user_idx", how="inner")
        .withColumn("rank", F.expr("array_position(pred_items, true_item)"))
        .select("user_idx", "rank")
    )
    recall, ndcg = compute_recall_ndcg(eval_rank)

    user_cov_row = eval_pred.agg(
        F.avg(F.when(F.size("pred_items") > 0, F.lit(1.0)).otherwise(F.lit(0.0))).alias("user_cov")
    ).first()
    user_cov = float(user_cov_row["user_cov"] or 0.0)

    exposures = (
        eval_pred.select("user_idx", F.posexplode_outer("pred_items").alias("pos", "item_idx"))
        .filter(F.col("item_idx").isNotNull())
        .withColumn("item_idx", F.col("item_idx").cast("int"))
        .persist(StorageLevel.DISK_ONLY)
    )
    n_exposures = exposures.count()
    if n_exposures == 0 or n_eval_users == 0:
        exposures.unpersist()
        return recall, ndcg, user_cov, 0.0, 0.0, 0.0

    unique_items = exposures.select("item_idx").distinct().count()
    item_cov = float(unique_items / max(1, n_items))

    expo_with_pop = (
        exposures.join(item_pop_stats, on="item_idx", how="left")
        .fillna({"pop_prob": default_pop_prob, "is_tail": 1.0})
        .withColumn(
            "novelty",
            -(
                F.log(F.greatest(F.col("pop_prob"), F.lit(default_pop_prob)))
                / F.log(F.lit(2.0))
            ),
        )
    )
    diag = expo_with_pop.agg(
        F.avg("is_tail").alias("tail_cov"),
        F.avg("novelty").alias("novelty"),
    ).first()
    tail_cov = float(diag["tail_cov"] or 0.0)
    novelty = float(diag["novelty"] or 0.0)
    exposures.unpersist()
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty


def rank_from_recs(recs: DataFrame, test_idx: DataFrame) -> DataFrame:
    pred = recs_to_pred_items(recs)
    truth = test_idx.groupBy("user_idx").agg(F.collect_set("item_idx").alias("true_items"))
    return (
        pred.join(truth, on="user_idx", how="inner")
        .withColumn("true_item", F.element_at("true_items", 1))
        .withColumn("rank", F.expr("array_position(pred_items, true_item)"))
        .select("user_idx", "rank")
    )


def evaluate_hybrid_popular(
    model_imp: ALS,
    train_full_idx: DataFrame,
    test_idx: DataFrame,
    min_train_count: int,
    top_k: int,
    item_pop_stats: DataFrame,
    default_pop_prob: float,
    n_items: int,
) -> tuple[float, float, int, int, float, float, float, float]:
    user_counts = train_full_idx.groupBy("user_idx").agg(F.count("*").alias("n_train"))
    heavy_users = user_counts.filter(F.col("n_train") >= min_train_count).select("user_idx")
    light_users = user_counts.filter(F.col("n_train") < min_train_count).select("user_idx")

    heavy_test = test_idx.join(heavy_users, on="user_idx", how="inner")
    light_test = test_idx.join(light_users, on="user_idx", how="inner")

    heavy_n = heavy_test.select("user_idx").distinct().count()
    light_n = light_test.select("user_idx").distinct().count()

    pop_items = (
        train_full_idx.groupBy("item_idx")
        .agg(F.count("*").alias("cnt"))
        .orderBy(F.desc("cnt"), F.asc("item_idx"))
        .limit(top_k)
        .agg(F.collect_list(F.col("item_idx").cast("int")).alias("pred_items"))
    )
    light_pred = light_users.crossJoin(pop_items)

    pred_parts = [light_pred]
    if heavy_n > 0:
        recs = model_imp.recommendForUserSubset(heavy_users, top_k)
        heavy_pred = recs_to_pred_items(recs)
        pred_parts.append(heavy_pred)

    pred_df = pred_parts[0]
    for part in pred_parts[1:]:
        pred_df = pred_df.unionByName(part, allowMissingColumns=True)

    recall, ndcg, user_cov, item_cov, tail_cov, novelty = evaluate_pred_items_with_diagnostics(
        pred_df,
        test_idx,
        item_pop_stats=item_pop_stats,
        default_pop_prob=default_pop_prob,
        n_items=n_items,
    )
    return recall, ndcg, heavy_n, light_n, user_cov, item_cov, tail_cov, novelty


def evaluate_hybrid_cluster_popular(
    model_imp: ALS,
    train_full_idx: DataFrame,
    test_idx: DataFrame,
    cluster_map: DataFrame,
    min_train_count: int,
    top_k: int,
    item_pop_stats: DataFrame,
    default_pop_prob: float,
    n_items: int,
) -> tuple[float, float, int, int, float, float, float, float, float]:
    user_counts = train_full_idx.groupBy("user_idx").agg(F.count("*").alias("n_train"))
    heavy_users = user_counts.filter(F.col("n_train") >= min_train_count).select("user_idx")
    light_users = user_counts.filter(F.col("n_train") < min_train_count).select("user_idx")

    heavy_test = test_idx.join(heavy_users, on="user_idx", how="inner")
    light_test = test_idx.join(light_users, on="user_idx", how="inner")

    heavy_n = heavy_test.select("user_idx").distinct().count()
    light_n = light_test.select("user_idx").distinct().count()

    # Global popular fallback (item-level rank fallback).
    pop = train_full_idx.groupBy("item_idx").agg(F.count("*").alias("cnt"))
    w_pop = Window.orderBy(F.desc("cnt"), F.asc("item_idx"))
    pop_ranked = (
        pop.withColumn("pop_rank", F.row_number().over(w_pop))
        .filter(F.col("pop_rank") <= top_k)
        .select(F.col("item_idx").cast("int").alias("item_idx"), "pop_rank")
    )
    global_pop_items = (
        pop_ranked.orderBy(F.asc("pop_rank"))
        .agg(F.collect_list(F.col("item_idx")).alias("global_pred_items"))
    )

    train_with_cluster = train_full_idx.join(cluster_map, on="business_id", how="left")
    train_with_cluster = train_with_cluster.filter(F.col("cluster_id").isNotNull())

    cluster_pop = train_with_cluster.groupBy("cluster_id", "item_idx").agg(F.count("*").alias("cnt"))
    w_cluster = Window.partitionBy("cluster_id").orderBy(F.desc("cnt"), F.asc("item_idx"))
    cluster_topk = (
        cluster_pop.withColumn("cluster_rank", F.row_number().over(w_cluster))
        .filter(F.col("cluster_rank") <= top_k)
        .select("cluster_id", "item_idx", "cluster_rank")
    )
    cluster_items = (
        cluster_topk.groupBy("cluster_id")
        .agg(
            F.sort_array(
                F.collect_list(
                    F.struct(
                        F.col("cluster_rank").alias("cluster_rank"),
                        F.col("item_idx").cast("int").alias("item_idx"),
                    )
                )
            ).alias("cluster_pairs")
        )
        .withColumn("pred_items", F.expr("transform(cluster_pairs, x -> int(x.item_idx))"))
        .select("cluster_id", "pred_items")
    )

    user_cluster = train_with_cluster.groupBy("user_idx", "cluster_id").agg(F.count("*").alias("cnt"))
    w_user = Window.partitionBy("user_idx").orderBy(F.desc("cnt"), F.asc("cluster_id"))
    user_top_cluster = (
        user_cluster.withColumn("rn", F.row_number().over(w_user))
        .filter(F.col("rn") == 1)
        .select("user_idx", "cluster_id")
    )

    light_eval_base = (
        light_test.select("user_idx", "item_idx")
        .distinct()
        .join(user_top_cluster, on="user_idx", how="left")
        .join(
            cluster_topk.select("cluster_id", F.col("item_idx").cast("int").alias("item_idx"), "cluster_rank"),
            on=["cluster_id", "item_idx"],
            how="left",
        )
        .join(pop_ranked, on="item_idx", how="left")
        .withColumn("rank", F.coalesce(F.col("cluster_rank"), F.col("pop_rank")))
        .select("user_idx", "cluster_rank", "rank")
    )
    light_cluster_hit_users = (
        light_eval_base.filter(F.col("cluster_rank").isNotNull())
        .select("user_idx")
        .distinct()
        .count()
    )
    light_cluster_user_coverage = float(light_cluster_hit_users / max(1, light_n))
    light_eval = light_eval_base.select("user_idx", "rank")
    light_pred = (
        light_users
        .join(user_top_cluster, on="user_idx", how="left")
        .join(cluster_items, on="cluster_id", how="left")
        .crossJoin(global_pop_items)
        .withColumn("cluster_items", F.col("pred_items"))
        .withColumn(
            "pred_items",
            F.when(
                F.col("cluster_items").isNull(),
                F.col("global_pred_items"),
            ).otherwise(
                F.expr(
                    "slice("
                    "concat(cluster_items, filter(global_pred_items, x -> NOT array_contains(cluster_items, x))),"
                    "1,"
                    f"{top_k}"
                    ")"
                )
            ),
        )
        .select("user_idx", "pred_items")
    )

    pred_parts = [light_pred]
    eval_parts = [light_eval]
    if heavy_n > 0:
        recs = model_imp.recommendForUserSubset(heavy_users, top_k)
        heavy_pred = recs_to_pred_items(recs)
        heavy_eval = rank_from_recs(recs, heavy_test)
        pred_parts.append(heavy_pred)
        eval_parts.append(heavy_eval)

    pred_df = pred_parts[0]
    for part in pred_parts[1:]:
        pred_df = pred_df.unionByName(part, allowMissingColumns=True)

    eval_df = eval_parts[0]
    for part in eval_parts[1:]:
        eval_df = eval_df.unionByName(part)
    recall, ndcg = compute_recall_ndcg(eval_df)

    _, _, user_cov, item_cov, tail_cov, novelty = evaluate_pred_items_with_diagnostics(
        pred_df,
        test_idx,
        item_pop_stats=item_pop_stats,
        default_pop_prob=default_pop_prob,
        n_items=n_items,
    )
    return (
        recall,
        ndcg,
        heavy_n,
        light_n,
        light_cluster_user_coverage,
        user_cov,
        item_cov,
        tail_cov,
        novelty,
    )


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
            seed=RANDOM_SEED,
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


def get_implicit_params_for_bucket(min_train: int) -> dict:
    if USE_PRESET_IMPLICIT_BEST:
        params = IMPLICIT_PRESET_BEST_BY_BUCKET.get(min_train)
        if params is not None:
            return {
                "rank": int(params["rank"]),
                "reg": float(params["reg"]),
                "alpha": float(params["alpha"]),
            }
    return {"rank": ALS_RANK, "reg": ALS_REG_PARAM, "alpha": ALS_IMPLICIT_ALPHA}


def evaluate_hybrid_cluster_rerank(
    model_imp: ALS,
    train_full_idx: DataFrame,
    test_idx: DataFrame,
    cluster_map: DataFrame,
    top_k: int,
    item_pop_stats: DataFrame,
    default_pop_prob: float,
    n_items: int,
    candidate_multiplier: int = RERANK_CANDIDATE_MULTIPLIER,
    cluster_bonus: float = RERANK_CLUSTER_BONUS,
) -> tuple[float, float, float, float, float, float, float]:
    cand_k = max(top_k, top_k * max(1, int(candidate_multiplier)))
    test_users = test_idx.select("user_idx").distinct()

    train_with_cluster = train_full_idx.join(cluster_map, on="business_id", how="left")
    train_with_cluster = train_with_cluster.filter(F.col("cluster_id").isNotNull())

    user_cluster = train_with_cluster.groupBy("user_idx", "cluster_id").agg(F.count("*").alias("cnt"))
    w_user = Window.partitionBy("user_idx").orderBy(F.desc("cnt"), F.asc("cluster_id"))
    user_top_cluster = (
        user_cluster.withColumn("rn", F.row_number().over(w_user))
        .filter(F.col("rn") == 1)
        .select(F.col("user_idx"), F.col("cluster_id").alias("user_cluster_id"))
    )
    item_cluster = (
        train_with_cluster.select(
            F.col("item_idx").cast("int").alias("item_idx"),
            F.col("cluster_id").alias("item_cluster_id"),
        )
        .dropDuplicates(["item_idx"])
    )

    recs = model_imp.recommendForUserSubset(test_users, cand_k)
    recs_exploded = (
        recs.select("user_idx", F.posexplode("recommendations").alias("orig_pos", "rec"))
        .select(
            F.col("user_idx"),
            F.col("orig_pos"),
            F.col("rec.item_idx").cast("int").alias("item_idx"),
            F.col("rec.rating").cast("double").alias("als_score"),
        )
        .join(item_cluster, on="item_idx", how="left")
        .join(user_top_cluster, on="user_idx", how="left")
        .withColumn(
            "cluster_bonus",
            F.when(
                F.col("user_cluster_id").isNotNull()
                & (F.col("item_cluster_id") == F.col("user_cluster_id")),
                F.lit(float(cluster_bonus)),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("rerank_score", F.col("als_score") + F.col("cluster_bonus"))
    )

    w_rank = Window.partitionBy("user_idx").orderBy(
        F.desc("rerank_score"),
        F.desc("als_score"),
        F.asc("orig_pos"),
    )
    reranked_topk = (
        recs_exploded.withColumn("rn", F.row_number().over(w_rank))
        .filter(F.col("rn") <= top_k)
        .groupBy("user_idx")
        .agg(
            F.sort_array(
                F.collect_list(F.struct(F.col("rn").alias("rn"), F.col("item_idx").alias("item_idx")))
            ).alias("pairs")
        )
        .withColumn("pred_items", F.expr("transform(pairs, x -> int(x.item_idx))"))
        .select("user_idx", "pred_items")
    )

    covered_cluster_users = (
        test_users.join(user_top_cluster.select("user_idx").distinct(), on="user_idx", how="inner")
        .count()
    )
    total_users = test_users.count()
    user_cluster_coverage = float(covered_cluster_users / max(1, total_users))

    recall, ndcg, user_cov, item_cov, tail_cov, novelty = evaluate_pred_items_with_diagnostics(
        reranked_topk,
        test_idx,
        item_pop_stats=item_pop_stats,
        default_pop_prob=default_pop_prob,
        n_items=n_items,
    )
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty, user_cluster_coverage


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        old_header = next(reader, [])
    if old_header == RESULT_FIELDS:
        return

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: row.get(k, "") for k in RESULT_FIELDS})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] upgraded results header fields for {path}")


def append_result(path: Path, row: dict) -> None:
    norm_row = {k: row.get(k, "") for k in RESULT_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writerow(norm_row)


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

    cluster_map = None
    if RUN_HYBRID_CLUSTER:
        profile_csv = resolve_cluster_profile_csv()
        if profile_csv is not None:
            try:
                cluster_map = load_cluster_profile_map(spark, profile_csv).persist(StorageLevel.DISK_ONLY)
                print(f"[CONFIG] cluster_profile_csv={profile_csv}")
                print(f"[COUNT] cluster_profile_businesses={cluster_map.count()}")
            except Exception as e:
                print(f"[WARN] failed to load cluster profile map: {e}")
                cluster_map = None
        else:
            print("[WARN] cluster profile disabled (no csv found)")

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
                seed=RANDOM_SEED,
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
                    item_pop_stats, default_pop_prob = build_item_pop_stats(train_full_idx)
                    item_pop_stats = item_pop_stats.persist(StorageLevel.DISK_ONLY)
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
                        seed=RANDOM_SEED,
                    )
                    model_imp = als_imp.fit(train_full_imp)
                    imp_pred = recs_to_pred_items(
                        model_imp.recommendForUserSubset(test_idx.select("user_idx").distinct(), TOP_K)
                    )
                    recall, ndcg, user_cov, item_cov, tail_cov, novelty = evaluate_pred_items_with_diagnostics(
                        imp_pred,
                        test_idx,
                        item_pop_stats=item_pop_stats,
                        default_pop_prob=default_pop_prob,
                        n_items=n_items,
                    )
                    print(
                        f"[Implicit ALS:test] Recall@{TOP_K}={recall:.4f} NDCG@{TOP_K}={ndcg:.4f} "
                        f"user_cov={user_cov:.2%} item_cov={item_cov:.2%} novelty={novelty:.3f}"
                    )
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
                                "user_coverage_at_k": f"{user_cov:.6f}",
                                "item_coverage_at_k": f"{item_cov:.6f}",
                                "tail_coverage_at_k": f"{tail_cov:.6f}",
                                "novelty_at_k": f"{novelty:.6f}",
                                "hybrid_min_train_threshold": "",
                                "hybrid_heavy_users": "",
                                "hybrid_light_users": "",
                                "hybrid_light_cluster_user_coverage": "",
                            },
                        )
                    if RUN_HYBRID:
                        hybrid_min = HYBRID_MIN_TRAIN_BY_BUCKET.get(min_train, HYBRID_MIN_TRAIN_DEFAULT)
                        (
                            h_recall,
                            h_ndcg,
                            h_heavy,
                            h_light,
                            h_user_cov,
                            h_item_cov,
                            h_tail_cov,
                            h_novelty,
                        ) = evaluate_hybrid_popular(
                            model_imp=model_imp,
                            train_full_idx=train_full_idx,
                            test_idx=test_idx,
                            min_train_count=hybrid_min,
                            top_k=TOP_K,
                            item_pop_stats=item_pop_stats,
                            default_pop_prob=default_pop_prob,
                            n_items=n_items,
                        )
                        print(
                            f"[Hybrid:test] min_train>={hybrid_min} "
                            f"Recall@{TOP_K}={h_recall:.4f} NDCG@{TOP_K}={h_ndcg:.4f} "
                            f"(heavy={h_heavy}, light={h_light}, user_cov={h_user_cov:.2%}, "
                            f"item_cov={h_item_cov:.2%}, novelty={h_novelty:.3f})"
                        )
                        if WRITE_RESULTS:
                            append_result(
                                RESULTS_PATH,
                                {
                                    "run_id": run_id,
                                    "bucket_min_train_reviews": min_train,
                                    "min_user_reviews": min_user_reviews,
                                    "model": f"Hybrid ALS+Popular (min_train>={hybrid_min})",
                                    "split": "test",
                                    "rank": best_params["rank"],
                                    "reg": best_params["reg"],
                                    "alpha": best_params["alpha"],
                                    "rmse": "",
                                    "recall_at_k": f"{h_recall:.6f}",
                                    "ndcg_at_k": f"{h_ndcg:.6f}",
                                    "top_k": TOP_K,
                                    "n_users": n_users,
                                    "n_items": n_items,
                                    "n_train": n_train,
                                    "n_valid": n_valid,
                                    "n_test": n_test,
                                    "user_coverage_at_k": f"{h_user_cov:.6f}",
                                    "item_coverage_at_k": f"{h_item_cov:.6f}",
                                    "tail_coverage_at_k": f"{h_tail_cov:.6f}",
                                    "novelty_at_k": f"{h_novelty:.6f}",
                                    "hybrid_min_train_threshold": hybrid_min,
                                    "hybrid_heavy_users": h_heavy,
                                    "hybrid_light_users": h_light,
                                    "hybrid_light_cluster_user_coverage": "",
                                },
                            )
                    if RUN_HYBRID_CLUSTER and cluster_map is not None:
                        hybrid_min = HYBRID_MIN_TRAIN_BY_BUCKET.get(min_train, HYBRID_MIN_TRAIN_DEFAULT)
                        (
                            hc_recall,
                            hc_ndcg,
                            hc_heavy,
                            hc_light,
                            hc_cov,
                            hc_user_cov,
                            hc_item_cov,
                            hc_tail_cov,
                            hc_novelty,
                        ) = evaluate_hybrid_cluster_popular(
                            model_imp=model_imp,
                            train_full_idx=train_full_idx,
                            test_idx=test_idx,
                            cluster_map=cluster_map,
                            min_train_count=hybrid_min,
                            top_k=TOP_K,
                            item_pop_stats=item_pop_stats,
                            default_pop_prob=default_pop_prob,
                            n_items=n_items,
                        )
                        print(
                            f"[Hybrid+Cluster:test] min_train>={hybrid_min} "
                            f"Recall@{TOP_K}={hc_recall:.4f} NDCG@{TOP_K}={hc_ndcg:.4f} "
                            f"(heavy={hc_heavy}, light={hc_light}, light_cluster_cov={hc_cov:.2%}, "
                            f"user_cov={hc_user_cov:.2%}, item_cov={hc_item_cov:.2%}, "
                            f"novelty={hc_novelty:.3f})"
                        )
                        if WRITE_RESULTS:
                            append_result(
                                RESULTS_PATH,
                                {
                                    "run_id": run_id,
                                    "bucket_min_train_reviews": min_train,
                                    "min_user_reviews": min_user_reviews,
                                    "model": f"Hybrid ALS+ClusterPopular (min_train>={hybrid_min})",
                                    "split": "test",
                                    "rank": best_params["rank"],
                                    "reg": best_params["reg"],
                                    "alpha": best_params["alpha"],
                                    "rmse": "",
                                    "recall_at_k": f"{hc_recall:.6f}",
                                    "ndcg_at_k": f"{hc_ndcg:.6f}",
                                    "top_k": TOP_K,
                                    "n_users": n_users,
                                    "n_items": n_items,
                                    "n_train": n_train,
                                    "n_valid": n_valid,
                                    "n_test": n_test,
                                    "user_coverage_at_k": f"{hc_user_cov:.6f}",
                                    "item_coverage_at_k": f"{hc_item_cov:.6f}",
                                    "tail_coverage_at_k": f"{hc_tail_cov:.6f}",
                                    "novelty_at_k": f"{hc_novelty:.6f}",
                                    "hybrid_min_train_threshold": hybrid_min,
                                    "hybrid_heavy_users": hc_heavy,
                                    "hybrid_light_users": hc_light,
                                    "hybrid_light_cluster_user_coverage": f"{hc_cov:.6f}",
                                },
                            )
                    if RUN_HYBRID_CLUSTER_RERANK and cluster_map is not None:
                        (
                            hr_recall,
                            hr_ndcg,
                            hr_user_cov,
                            hr_item_cov,
                            hr_tail_cov,
                            hr_novelty,
                            hr_cluster_user_cov,
                        ) = evaluate_hybrid_cluster_rerank(
                            model_imp=model_imp,
                            train_full_idx=train_full_idx,
                            test_idx=test_idx,
                            cluster_map=cluster_map,
                            top_k=TOP_K,
                            item_pop_stats=item_pop_stats,
                            default_pop_prob=default_pop_prob,
                            n_items=n_items,
                            candidate_multiplier=RERANK_CANDIDATE_MULTIPLIER,
                            cluster_bonus=RERANK_CLUSTER_BONUS,
                        )
                        print(
                            f"[Hybrid+ClusterRerank:test] Recall@{TOP_K}={hr_recall:.4f} "
                            f"NDCG@{TOP_K}={hr_ndcg:.4f} user_cov={hr_user_cov:.2%} "
                            f"item_cov={hr_item_cov:.2%} novelty={hr_novelty:.3f} "
                            f"user_cluster_cov={hr_cluster_user_cov:.2%}"
                        )
                        if WRITE_RESULTS:
                            append_result(
                                RESULTS_PATH,
                                {
                                    "run_id": run_id,
                                    "bucket_min_train_reviews": min_train,
                                    "min_user_reviews": min_user_reviews,
                                    "model": (
                                        f"Hybrid ALS+ClusterRerank"
                                        f" (candx{RERANK_CANDIDATE_MULTIPLIER},bonus={RERANK_CLUSTER_BONUS})"
                                    ),
                                    "split": "test",
                                    "rank": best_params["rank"],
                                    "reg": best_params["reg"],
                                    "alpha": best_params["alpha"],
                                    "rmse": "",
                                    "recall_at_k": f"{hr_recall:.6f}",
                                    "ndcg_at_k": f"{hr_ndcg:.6f}",
                                    "top_k": TOP_K,
                                    "n_users": n_users,
                                    "n_items": n_items,
                                    "n_train": n_train,
                                    "n_valid": n_valid,
                                    "n_test": n_test,
                                    "user_coverage_at_k": f"{hr_user_cov:.6f}",
                                    "item_coverage_at_k": f"{hr_item_cov:.6f}",
                                    "tail_coverage_at_k": f"{hr_tail_cov:.6f}",
                                    "novelty_at_k": f"{hr_novelty:.6f}",
                                    "hybrid_min_train_threshold": "",
                                    "hybrid_heavy_users": "",
                                    "hybrid_light_users": "",
                                    "hybrid_light_cluster_user_coverage": "",
                                    "rerank_user_cluster_coverage": f"{hr_cluster_user_cov:.6f}",
                                },
                            )
                    item_pop_stats.unpersist()
        elif RUN_IMPLICIT_ALS:
            best_params = get_implicit_params_for_bucket(min_train)
            print(
                "[STEP] fit ALS (implicit, best-only) "
                f"rank={best_params['rank']} reg={best_params['reg']} alpha={best_params['alpha']}"
            )
            train_full_idx = train_idx.unionByName(valid_idx)
            train_full_imp = train_full_idx.withColumn("rating", F.lit(1.0))
            item_pop_stats, default_pop_prob = build_item_pop_stats(train_full_idx)
            item_pop_stats = item_pop_stats.persist(StorageLevel.DISK_ONLY)
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
                seed=RANDOM_SEED,
            )
            model_imp = als_imp.fit(train_full_imp)
            imp_pred = recs_to_pred_items(
                model_imp.recommendForUserSubset(test_idx.select("user_idx").distinct(), TOP_K)
            )
            recall, ndcg, user_cov, item_cov, tail_cov, novelty = evaluate_pred_items_with_diagnostics(
                imp_pred,
                test_idx,
                item_pop_stats=item_pop_stats,
                default_pop_prob=default_pop_prob,
                n_items=n_items,
            )
            print(
                f"[Implicit ALS:test] Recall@{TOP_K} {recall:.4f} "
                f"NDCG@{TOP_K} {ndcg:.4f} user_cov={user_cov:.2%} "
                f"item_cov={item_cov:.2%} novelty={novelty:.3f}"
            )
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
                        "user_coverage_at_k": f"{user_cov:.6f}",
                        "item_coverage_at_k": f"{item_cov:.6f}",
                        "tail_coverage_at_k": f"{tail_cov:.6f}",
                        "novelty_at_k": f"{novelty:.6f}",
                        "hybrid_min_train_threshold": "",
                        "hybrid_heavy_users": "",
                        "hybrid_light_users": "",
                        "hybrid_light_cluster_user_coverage": "",
                    },
                )
            if RUN_HYBRID:
                hybrid_min = HYBRID_MIN_TRAIN_BY_BUCKET.get(min_train, HYBRID_MIN_TRAIN_DEFAULT)
                (
                    h_recall,
                    h_ndcg,
                    h_heavy,
                    h_light,
                    h_user_cov,
                    h_item_cov,
                    h_tail_cov,
                    h_novelty,
                ) = evaluate_hybrid_popular(
                    model_imp=model_imp,
                    train_full_idx=train_full_idx,
                    test_idx=test_idx,
                    min_train_count=hybrid_min,
                    top_k=TOP_K,
                    item_pop_stats=item_pop_stats,
                    default_pop_prob=default_pop_prob,
                    n_items=n_items,
                )
                print(
                    f"[Hybrid:test] min_train>={hybrid_min} "
                    f"Recall@{TOP_K}={h_recall:.4f} NDCG@{TOP_K}={h_ndcg:.4f} "
                    f"(heavy={h_heavy}, light={h_light}, user_cov={h_user_cov:.2%}, "
                    f"item_cov={h_item_cov:.2%}, novelty={h_novelty:.3f})"
                )
                if WRITE_RESULTS:
                    append_result(
                        RESULTS_PATH,
                        {
                            "run_id": run_id,
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "model": f"Hybrid ALS+Popular (min_train>={hybrid_min})",
                            "split": "test",
                            "rank": best_params["rank"],
                            "reg": best_params["reg"],
                            "alpha": best_params["alpha"],
                            "rmse": "",
                            "recall_at_k": f"{h_recall:.6f}",
                            "ndcg_at_k": f"{h_ndcg:.6f}",
                            "top_k": TOP_K,
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_train": n_train,
                            "n_valid": n_valid,
                            "n_test": n_test,
                            "user_coverage_at_k": f"{h_user_cov:.6f}",
                            "item_coverage_at_k": f"{h_item_cov:.6f}",
                            "tail_coverage_at_k": f"{h_tail_cov:.6f}",
                            "novelty_at_k": f"{h_novelty:.6f}",
                            "hybrid_min_train_threshold": hybrid_min,
                            "hybrid_heavy_users": h_heavy,
                            "hybrid_light_users": h_light,
                            "hybrid_light_cluster_user_coverage": "",
                        },
                    )
            if RUN_HYBRID_CLUSTER and cluster_map is not None:
                hybrid_min = HYBRID_MIN_TRAIN_BY_BUCKET.get(min_train, HYBRID_MIN_TRAIN_DEFAULT)
                (
                    hc_recall,
                    hc_ndcg,
                    hc_heavy,
                    hc_light,
                    hc_cov,
                    hc_user_cov,
                    hc_item_cov,
                    hc_tail_cov,
                    hc_novelty,
                ) = evaluate_hybrid_cluster_popular(
                    model_imp=model_imp,
                    train_full_idx=train_full_idx,
                    test_idx=test_idx,
                    cluster_map=cluster_map,
                    min_train_count=hybrid_min,
                    top_k=TOP_K,
                    item_pop_stats=item_pop_stats,
                    default_pop_prob=default_pop_prob,
                    n_items=n_items,
                )
                print(
                    f"[Hybrid+Cluster:test] min_train>={hybrid_min} "
                    f"Recall@{TOP_K}={hc_recall:.4f} NDCG@{TOP_K}={hc_ndcg:.4f} "
                    f"(heavy={hc_heavy}, light={hc_light}, light_cluster_cov={hc_cov:.2%}, "
                    f"user_cov={hc_user_cov:.2%}, item_cov={hc_item_cov:.2%}, novelty={hc_novelty:.3f})"
                )
                if WRITE_RESULTS:
                    append_result(
                        RESULTS_PATH,
                        {
                            "run_id": run_id,
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "model": f"Hybrid ALS+ClusterPopular (min_train>={hybrid_min})",
                            "split": "test",
                            "rank": best_params["rank"],
                            "reg": best_params["reg"],
                            "alpha": best_params["alpha"],
                            "rmse": "",
                            "recall_at_k": f"{hc_recall:.6f}",
                            "ndcg_at_k": f"{hc_ndcg:.6f}",
                            "top_k": TOP_K,
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_train": n_train,
                            "n_valid": n_valid,
                            "n_test": n_test,
                            "user_coverage_at_k": f"{hc_user_cov:.6f}",
                            "item_coverage_at_k": f"{hc_item_cov:.6f}",
                            "tail_coverage_at_k": f"{hc_tail_cov:.6f}",
                            "novelty_at_k": f"{hc_novelty:.6f}",
                            "hybrid_min_train_threshold": hybrid_min,
                            "hybrid_heavy_users": hc_heavy,
                            "hybrid_light_users": hc_light,
                            "hybrid_light_cluster_user_coverage": f"{hc_cov:.6f}",
                        },
                    )
            if RUN_HYBRID_CLUSTER_RERANK and cluster_map is not None:
                (
                    hr_recall,
                    hr_ndcg,
                    hr_user_cov,
                    hr_item_cov,
                    hr_tail_cov,
                    hr_novelty,
                    hr_cluster_user_cov,
                ) = evaluate_hybrid_cluster_rerank(
                    model_imp=model_imp,
                    train_full_idx=train_full_idx,
                    test_idx=test_idx,
                    cluster_map=cluster_map,
                    top_k=TOP_K,
                    item_pop_stats=item_pop_stats,
                    default_pop_prob=default_pop_prob,
                    n_items=n_items,
                    candidate_multiplier=RERANK_CANDIDATE_MULTIPLIER,
                    cluster_bonus=RERANK_CLUSTER_BONUS,
                )
                print(
                    f"[Hybrid+ClusterRerank:test] Recall@{TOP_K}={hr_recall:.4f} "
                    f"NDCG@{TOP_K}={hr_ndcg:.4f} user_cov={hr_user_cov:.2%} "
                    f"item_cov={hr_item_cov:.2%} novelty={hr_novelty:.3f} "
                    f"user_cluster_cov={hr_cluster_user_cov:.2%}"
                )
                if WRITE_RESULTS:
                    append_result(
                        RESULTS_PATH,
                        {
                            "run_id": run_id,
                            "bucket_min_train_reviews": min_train,
                            "min_user_reviews": min_user_reviews,
                            "model": (
                                f"Hybrid ALS+ClusterRerank"
                                f" (candx{RERANK_CANDIDATE_MULTIPLIER},bonus={RERANK_CLUSTER_BONUS})"
                            ),
                            "split": "test",
                            "rank": best_params["rank"],
                            "reg": best_params["reg"],
                            "alpha": best_params["alpha"],
                            "rmse": "",
                            "recall_at_k": f"{hr_recall:.6f}",
                            "ndcg_at_k": f"{hr_ndcg:.6f}",
                            "top_k": TOP_K,
                            "n_users": n_users,
                            "n_items": n_items,
                            "n_train": n_train,
                            "n_valid": n_valid,
                            "n_test": n_test,
                            "user_coverage_at_k": f"{hr_user_cov:.6f}",
                            "item_coverage_at_k": f"{hr_item_cov:.6f}",
                            "tail_coverage_at_k": f"{hr_tail_cov:.6f}",
                            "novelty_at_k": f"{hr_novelty:.6f}",
                            "hybrid_min_train_threshold": "",
                            "hybrid_heavy_users": "",
                            "hybrid_light_users": "",
                            "hybrid_light_cluster_user_coverage": "",
                            "rerank_user_cluster_coverage": f"{hr_cluster_user_cov:.6f}",
                        },
                    )
            item_pop_stats.unpersist()

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
    if cluster_map is not None:
        cluster_map.unpersist()
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
