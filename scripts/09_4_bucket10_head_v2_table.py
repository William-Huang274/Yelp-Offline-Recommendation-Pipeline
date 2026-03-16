from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

from pipeline.project_paths import env_or_project_path, project_path
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context, cleanup_context_tmp


RUN_TAG = "stage09_bucket10_head_v2_table"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_HEAD_V2_ROOT_DIR", "data/output/09_head_v2_table")

HEAD_V2_BUCKET = int(os.getenv("HEAD_V2_BUCKET", "10").strip() or 10)
HEAD_V2_TOP150 = int(os.getenv("HEAD_V2_TOP150", "150").strip() or 150)
HEAD_V2_TOP250 = int(os.getenv("HEAD_V2_TOP250", "250").strip() or 250)
HEAD_V2_COHORT_PATH = os.getenv("HEAD_V2_COHORT_PATH", "").strip()
OUTPUT_COALESCE_PARTITIONS = int(os.getenv("OUTPUT_COALESCE_PARTITIONS", "16").strip() or 16)

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "8g").strip() or "8g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "8g").strip() or "8g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[4]").strip() or "local[4]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "16").strip() or "16"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "16").strip() or "16"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "900s").strip() or "900s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = (
    os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
)
SPARK_SQL_ADAPTIVE_ENABLED = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
PY_TEMP_DIR = os.getenv("PY_TEMP_DIR", "").strip()

_SPARK_TMP_CTX: SparkTmpContext | None = None

PROFILE_DETAILS = (
    "profile_vector",
    "profile_shared",
    "profile_bridge_user",
    "profile_bridge_type",
)


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        py_temp_root_override=PY_TEMP_DIR,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=True,
    )
    local_dir = _SPARK_TMP_CTX.spark_local_dir
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} spark_local_dir={local_dir} py_temp={_SPARK_TMP_CTX.py_temp_dir} "
        f"auto_clean={SPARK_TMP_AUTOCLEAN_ENABLED} retention_h={SPARK_TMP_RETENTION_HOURS} "
        f"cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    return (
        SparkSession.builder.appName(RUN_TAG)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.sql.adaptive.enabled", str(SPARK_SQL_ADAPTIVE_ENABLED).lower())
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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


def resolve_cohort_df(spark: SparkSession, truth_df: DataFrame) -> DataFrame:
    path = str(HEAD_V2_COHORT_PATH or "").strip()
    if not path:
        return truth_df
    cohort = spark.read.option("header", True).csv(path)
    if "user_idx" in cohort.columns:
        return truth_df.join(
            cohort.select(F.col("user_idx").cast("int").alias("user_idx")).dropDuplicates(["user_idx"]),
            on="user_idx",
            how="inner",
        )
    if "user_id" in cohort.columns:
        return truth_df.join(
            cohort.select(F.col("user_id").cast("string").alias("user_id")).dropDuplicates(["user_id"]),
            on="user_id",
            how="inner",
        )
    raise RuntimeError(f"cohort file missing user_idx/user_id column: {path}")


def resolve_bucket_paths(run_dir: Path, bucket: int) -> dict[str, Any]:
    bucket_dir = run_dir / f"bucket_{int(bucket)}"
    if not bucket_dir.exists():
        raise FileNotFoundError(f"bucket dir not found: {bucket_dir}")
    enriched = bucket_dir / "candidates_enriched_audit.parquet"
    pretrim = bucket_dir / "candidates_pretrim150.parquet"
    if not pretrim.exists():
        pretrim = bucket_dir / "candidates_pretrim.parquet"
    source_evidence = bucket_dir / "candidate_source_evidence.parquet"
    truth = bucket_dir / "truth.parquet"
    if not truth.exists():
        raise FileNotFoundError(f"truth parquet not found: {truth}")
    if enriched.exists():
        candidate_path = enriched
        candidate_mode = "enriched"
    elif pretrim.exists():
        candidate_path = pretrim
        candidate_mode = "pretrim"
    else:
        raise FileNotFoundError(f"no candidate parquet found in {bucket_dir}")
    return {
        "bucket_dir": bucket_dir,
        "candidate_path": candidate_path,
        "candidate_mode": candidate_mode,
        "source_evidence_path": source_evidence if source_evidence.exists() else None,
        "truth_path": truth,
    }


def _coalesce_for_write(df: DataFrame) -> DataFrame:
    target = int(max(1, OUTPUT_COALESCE_PARTITIONS))
    try:
        current = int(df.rdd.getNumPartitions())
    except Exception:
        return df
    if current <= target:
        return df
    return df.coalesce(target)


def load_candidate_df(spark: SparkSession, candidate_path: Path, candidate_mode: str, truth_users: DataFrame) -> DataFrame:
    cand = spark.read.parquet(candidate_path.as_posix()).join(truth_users, on="user_idx", how="inner")
    if "source_count" not in cand.columns:
        cand = cand.withColumn("source_count", F.coalesce(F.size(F.col("source_set")).cast("double"), F.lit(0.0)))
    if "has_als" not in cand.columns:
        cand = cand.withColumn(
            "has_als",
            F.when(F.array_contains(F.col("source_set"), F.lit("als")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    if "has_cluster" not in cand.columns:
        cand = cand.withColumn(
            "has_cluster",
            F.when(F.array_contains(F.col("source_set"), F.lit("cluster")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    if "has_profile" not in cand.columns:
        cand = cand.withColumn(
            "has_profile",
            F.when(F.array_contains(F.col("source_set"), F.lit("profile")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    if "has_popular" not in cand.columns:
        cand = cand.withColumn(
            "has_popular",
            F.when(F.array_contains(F.col("source_set"), F.lit("popular")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    if "nonpopular_source_count" not in cand.columns:
        cand = cand.withColumn("nonpopular_source_count", F.col("has_als") + F.col("has_cluster") + F.col("has_profile"))
    if "profile_cluster_source_count" not in cand.columns:
        cand = cand.withColumn("profile_cluster_source_count", F.col("has_cluster") + F.col("has_profile"))
    if candidate_mode == "enriched":
        baseline_rank = F.col("final_pre_rank")
        in_pretrim = F.coalesce(F.col("is_in_pretrim").cast("int"), F.lit(0))
    else:
        baseline_rank = F.col("pre_rank")
        in_pretrim = F.lit(1)
    return (
        cand.withColumn("baseline_pre_rank", baseline_rank.cast("int"))
        .withColumn("is_in_pretrim", in_pretrim.cast("int"))
        .withColumn(
            "is_in_top150_baseline",
            F.when(F.col("baseline_pre_rank").isNotNull() & (F.col("baseline_pre_rank") <= F.lit(HEAD_V2_TOP150)), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "is_in_top250_baseline",
            F.when(F.col("baseline_pre_rank").isNotNull() & (F.col("baseline_pre_rank") <= F.lit(HEAD_V2_TOP250)), F.lit(1)).otherwise(F.lit(0)),
        )
    )


def build_source_feature_df(spark: SparkSession, source_evidence_path: Path | None, truth_users: DataFrame) -> DataFrame | None:
    if source_evidence_path is None or not source_evidence_path.exists():
        return None
    src = spark.read.parquet(source_evidence_path.as_posix()).join(truth_users, on="user_idx", how="inner")

    def _src_max(src_name: str, col_name: str, alias: str) -> Any:
        return F.max(F.when(F.col("source") == F.lit(src_name), F.col(col_name))).alias(alias)

    def _detail_max(detail_name: str, col_name: str, alias: str) -> Any:
        return F.max(F.when(F.col("source_detail") == F.lit(detail_name), F.col(col_name))).alias(alias)

    aggs: list[Any] = [
        _src_max("als", "source_rank", "als_evd_rank"),
        _src_max("als", "source_score", "als_evd_score"),
        _src_max("als", "source_confidence", "als_evd_confidence"),
        _src_max("als", "source_weight", "als_evd_weight"),
        _src_max("als", "source_norm", "als_evd_norm"),
        _src_max("als", "signal_score", "als_evd_signal"),
        _src_max("cluster", "source_rank", "cluster_evd_rank"),
        _src_max("cluster", "source_score", "cluster_evd_score"),
        _src_max("cluster", "source_confidence", "cluster_evd_confidence"),
        _src_max("cluster", "source_weight", "cluster_evd_weight"),
        _src_max("cluster", "source_norm", "cluster_evd_norm"),
        _src_max("cluster", "signal_score", "cluster_evd_signal"),
        _src_max("profile", "source_rank", "profile_evd_rank"),
        _src_max("profile", "source_score", "profile_evd_score"),
        _src_max("profile", "source_confidence", "profile_evd_confidence"),
        _src_max("profile", "source_weight", "profile_evd_weight"),
        _src_max("profile", "source_norm", "profile_evd_norm"),
        _src_max("profile", "signal_score", "profile_evd_signal"),
        _src_max("popular", "source_rank", "popular_evd_rank"),
        _src_max("popular", "source_score", "popular_evd_score"),
        _src_max("popular", "source_confidence", "popular_evd_confidence"),
        _src_max("popular", "source_weight", "popular_evd_weight"),
        _src_max("popular", "source_norm", "popular_evd_norm"),
        _src_max("popular", "signal_score", "popular_evd_signal"),
        F.sum(F.when(F.col("source") == F.lit("profile"), F.lit(1)).otherwise(F.lit(0))).alias("profile_route_rows"),
    ]
    for detail in PROFILE_DETAILS:
        prefix = detail.replace("profile_", "")
        aggs.extend(
            [
                _detail_max(detail, "source_rank", f"profile_{prefix}_rank"),
                _detail_max(detail, "source_score", f"profile_{prefix}_score"),
                _detail_max(detail, "source_confidence", f"profile_{prefix}_confidence"),
                _detail_max(detail, "source_weight", f"profile_{prefix}_weight"),
                _detail_max(detail, "source_norm", f"profile_{prefix}_norm"),
                _detail_max(detail, "signal_score", f"profile_{prefix}_signal"),
            ]
        )
    return (
        src.groupBy("user_idx", "item_idx")
        .agg(*aggs)
        .withColumn(
            "profile_active_detail_count",
            F.coalesce(F.col("profile_vector_score").isNotNull().cast("int"), F.lit(0))
            + F.coalesce(F.col("profile_shared_score").isNotNull().cast("int"), F.lit(0))
            + F.coalesce(F.col("profile_bridge_user_score").isNotNull().cast("int"), F.lit(0))
            + F.coalesce(F.col("profile_bridge_type_score").isNotNull().cast("int"), F.lit(0)),
        )
    )


def build_training_df(candidates: DataFrame, truth: DataFrame, source_features: DataFrame | None) -> DataFrame:
    base = (
        candidates.join(truth, on="user_idx", how="inner")
        .withColumn("item_idx_int", F.col("item_idx").cast("int"))
        .withColumn("true_item_idx_int", F.col("true_item_idx").cast("int"))
        .withColumn("label", F.when(F.col("item_idx_int") == F.col("true_item_idx_int"), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("source_combo", F.concat_ws("+", F.sort_array(F.col("source_set"))))
        .withColumn(
            "has_profile_or_cluster",
            F.when((F.col("has_profile") > F.lit(0.5)) | (F.col("has_cluster") > F.lit(0.5)), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "is_als_only",
            F.when((F.col("has_als") > F.lit(0.5)) & (F.col("source_count") <= F.lit(1.0)), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "is_als_popular_only",
            F.when(
                (F.col("has_als") > F.lit(0.5))
                & (F.col("has_popular") > F.lit(0.5))
                & (F.col("has_cluster") <= F.lit(0.5))
                & (F.col("has_profile") <= F.lit(0.5))
                & (F.col("source_count") <= F.lit(2.0)),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("popular_log1p", F.log1p(F.coalesce(F.col("item_train_pop_count").cast("double"), F.lit(0.0))))
        .withColumn(
            "baseline_rank_bucket",
            F.when(F.col("baseline_pre_rank").isNull(), F.lit("absent"))
            .when(F.col("baseline_pre_rank") <= F.lit(80), F.lit("top80"))
            .when(F.col("baseline_pre_rank") <= F.lit(HEAD_V2_TOP150), F.lit("81-150"))
            .when(F.col("baseline_pre_rank") <= F.lit(HEAD_V2_TOP250), F.lit("151-250"))
            .otherwise(F.lit("251+")),
        )
        .withColumn(
            "head_shortfall_to_top150",
            F.when(F.col("baseline_pre_rank").isNull(), F.lit(None).cast("int"))
            .otherwise(F.greatest(F.col("baseline_pre_rank") - F.lit(HEAD_V2_TOP150), F.lit(0))),
        )
    )
    if source_features is not None:
        base = base.join(source_features, on=["user_idx", "item_idx"], how="left")
    return base.drop("item_idx_int", "true_item_idx_int")


def build_truth_audit_df(train_df: DataFrame, truth: DataFrame) -> DataFrame:
    truth_hit = train_df.filter(F.col("label") > F.lit(0)).select(
        "user_idx",
        "item_idx",
        "business_id",
        "user_segment",
        "source_combo",
        "baseline_pre_rank",
        "baseline_rank_bucket",
        "is_in_pretrim",
        "is_in_top150_baseline",
        "is_in_top250_baseline",
        "head_shortfall_to_top150",
        "source_count",
        "nonpopular_source_count",
        "profile_cluster_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
    )
    return (
        truth.join(truth_hit, on="user_idx", how="left")
        .withColumn("item_idx", F.coalesce(F.col("item_idx").cast("int"), F.col("true_item_idx").cast("int")))
        .withColumn("baseline_rank_bucket", F.coalesce(F.col("baseline_rank_bucket"), F.lit("absent")))
        .withColumn("is_in_pretrim", F.coalesce(F.col("is_in_pretrim").cast("int"), F.lit(0)))
        .withColumn("is_in_top150_baseline", F.coalesce(F.col("is_in_top150_baseline").cast("int"), F.lit(0)))
        .withColumn("is_in_top250_baseline", F.coalesce(F.col("is_in_top250_baseline").cast("int"), F.lit(0)))
    )


def main() -> None:
    run_dir = resolve_stage09_run()
    paths = resolve_bucket_paths(run_dir, HEAD_V2_BUCKET)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{timestamp}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    train_df: DataFrame | None = None
    truth_audit_df: DataFrame | None = None
    try:
        truth = (
            spark.read.parquet(paths["truth_path"].as_posix())
            .select("user_idx", F.col("true_item_idx").cast("int").alias("true_item_idx"), "user_id")
            .dropDuplicates(["user_idx"])
        )
        truth = resolve_cohort_df(spark, truth)
        truth_users = truth.select("user_idx").dropDuplicates(["user_idx"])

        candidates = load_candidate_df(spark, paths["candidate_path"], paths["candidate_mode"], truth_users)
        source_features = build_source_feature_df(spark, paths["source_evidence_path"], truth_users)
        train_df = build_training_df(candidates, truth, source_features).persist(StorageLevel.DISK_ONLY)
        truth_audit_df = build_truth_audit_df(train_df, truth).persist(StorageLevel.DISK_ONLY)

        feature_out = _coalesce_for_write(train_df)
        positive_out = _coalesce_for_write(truth_audit_df)
        segment_out = _coalesce_for_write(
            truth_audit_df.groupBy("user_segment", "baseline_rank_bucket")
            .agg(F.count("*").alias("users"))
            .orderBy(F.asc("user_segment"), F.asc("baseline_rank_bucket"))
        )

        feature_out.write.mode("overwrite").parquet((out_dir / "head_v2_training_table.parquet").as_posix())
        positive_out.write.mode("overwrite").parquet((out_dir / "head_v2_positive_audit.parquet").as_posix())
        segment_out.write.mode("overwrite").parquet((out_dir / "head_v2_segment_summary.parquet").as_posix())

        metrics_row = (
            truth_audit_df.agg(
                F.count("*").alias("n_users"),
                F.sum(F.when(F.col("baseline_rank_bucket") != F.lit("absent"), F.lit(1)).otherwise(F.lit(0))).alias("truth_in_table"),
                F.sum(F.when(F.col("is_in_pretrim") > F.lit(0), F.lit(1)).otherwise(F.lit(0))).alias("truth_in_pretrim"),
                F.sum(F.when(F.col("is_in_top150_baseline") > F.lit(0), F.lit(1)).otherwise(F.lit(0))).alias("truth_in_top150"),
                F.sum(F.when(F.col("is_in_top250_baseline") > F.lit(0), F.lit(1)).otherwise(F.lit(0))).alias("truth_in_top250"),
            )
            .first()
        )
        train_rows = int(train_df.count())
        positive_rows = int(truth_audit_df.count())
        feature_columns = [
            c
            for c in train_df.columns
            if c
            not in {
                "user_idx",
                "item_idx",
                "user_id",
                "true_item_idx",
                "business_id",
                "name",
                "city",
            "categories",
            "source_set",
            "source_combo",
            "baseline_rank_bucket",
            "label",
            }
        ]
        run_meta = {
            "run_tag": RUN_TAG,
            "input_09_run_dir": run_dir.as_posix(),
            "bucket": int(HEAD_V2_BUCKET),
            "head_v2_top150": int(HEAD_V2_TOP150),
            "head_v2_top250": int(HEAD_V2_TOP250),
            "candidate_mode": str(paths["candidate_mode"]),
            "candidate_path": paths["candidate_path"].as_posix(),
            "source_evidence_path": paths["source_evidence_path"].as_posix() if paths["source_evidence_path"] is not None else "",
            "truth_path": paths["truth_path"].as_posix(),
            "cohort_path": str(HEAD_V2_COHORT_PATH or ""),
            "train_rows": int(train_rows),
            "truth_audit_rows": int(positive_rows),
            "n_users": int(metrics_row["n_users"] or 0),
            "truth_in_table": int(metrics_row["truth_in_table"] or 0),
            "truth_in_pretrim": int(metrics_row["truth_in_pretrim"] or 0),
            "truth_in_top150": int(metrics_row["truth_in_top150"] or 0),
            "truth_in_top250": int(metrics_row["truth_in_top250"] or 0),
            "has_source_evidence_features": bool(paths["source_evidence_path"] is not None),
            "feature_columns": feature_columns,
            "spark_master": SPARK_MASTER,
            "spark_driver_memory": SPARK_DRIVER_MEMORY,
            "spark_executor_memory": SPARK_EXECUTOR_MEMORY,
        }
        write_json(out_dir / "run_meta.json", run_meta)
        print(f"[INFO] wrote {out_dir.as_posix()}")
    finally:
        try:
            spark.stop()
        finally:
            global _SPARK_TMP_CTX
            try:
                if train_df is not None:
                    train_df.unpersist(blocking=False)
            except Exception:
                pass
            try:
                if truth_audit_df is not None:
                    truth_audit_df.unpersist(blocking=False)
            except Exception:
                pass
            if _SPARK_TMP_CTX is not None:
                cleanup_context_tmp(ctx=_SPARK_TMP_CTX)
                _SPARK_TMP_CTX = None


if __name__ == "__main__":
    main()
