from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F

from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context


RUN_TAG = "stage11_0_qlora_data_gate_audit"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = Path(r"D:/5006 BDA project/data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
TOPN_GRID_RAW = os.getenv("QLORA_AUDIT_TOPN_GRID", "80,120,150,200,300").strip()
AUDIT_CANDIDATE_FILES_RAW = os.getenv("QLORA_AUDIT_CANDIDATE_FILES", "").strip()
INCLUDE_VALID_POS = os.getenv("QLORA_AUDIT_INCLUDE_VALID_POS", "true").strip().lower() == "true"

GATE_A_MIN_TRUE_IN_TOPN = float(os.getenv("QLORA_GATE_A_MIN_TRUE_IN_TOPN", "0.82").strip() or 0.82)
GATE_B_MAX_NO_POS_RATIO = float(os.getenv("QLORA_GATE_B_MAX_NO_POS_RATIO", "0.30").strip() or 0.30)

OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/11_qlora_data_gate_audit")
METRICS_LATEST = Path(r"D:/5006 BDA project/data/metrics/stage11_qlora_data_gate_audit_latest.csv")

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", "D:/5006 BDA project/data/spark-tmp").strip() or "D:/5006 BDA project/data/spark-tmp"
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)

_SPARK_TMP_CTX: SparkTmpContext | None = None


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return sorted(list(set(out)))


def parse_str_list(raw: str) -> list[str]:
    out: list[str] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p)
    seen: set[str] = set()
    dedup: list[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def build_literal_int_df(spark: SparkSession, *, col_name: str, values: list[int]) -> DataFrame:
    nums = [int(x) for x in values]
    if not nums:
        raise ValueError(f"empty int literals for {col_name}")
    arr_sql = ",".join(str(x) for x in nums)
    return spark.sql(f"SELECT explode(array({arr_sql})) AS {col_name}")


def build_literal_str_df(spark: SparkSession, *, col_name: str, values: list[str]) -> DataFrame:
    vals = [str(x) for x in values if str(x)]
    if not vals:
        raise ValueError(f"empty string literals for {col_name}")
    # Escape single quote for Spark SQL string literal.
    arr_sql = ",".join("'" + x.replace("'", "''") + "'" for x in vals)
    return spark.sql(f"SELECT explode(array({arr_sql})) AS {col_name}")


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


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
        .config("spark.local.dir", str(_SPARK_TMP_CTX.spark_local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.python.worker.reuse", "true")
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def list_candidate_files(bucket_dir: Path) -> list[str]:
    if AUDIT_CANDIDATE_FILES_RAW:
        wanted = parse_str_list(AUDIT_CANDIDATE_FILES_RAW)
        keep = [x for x in wanted if (bucket_dir / x).exists()]
        return keep

    ordered = [
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates_pretrim.parquet",
        "candidates_all.parquet",
        "candidates.parquet",
    ]
    keep: list[str] = []
    for name in ordered:
        if (bucket_dir / name).exists():
            keep.append(name)

    # Add any additional pretrim parquet not covered by fixed list.
    for p in sorted(bucket_dir.glob("candidates_pretrim*.parquet")):
        if p.name not in keep:
            keep.append(p.name)
    return keep


def read_candidates_union(
    spark: SparkSession,
    *,
    bucket_dir: Path,
    user_idx_df: DataFrame,
    candidate_files: list[str],
) -> DataFrame:
    frames: list[DataFrame] = []
    for name in candidate_files:
        path = bucket_dir / name
        if not path.exists():
            continue
        raw = spark.read.parquet(path.as_posix())
        cols = set(raw.columns)
        rank_src_col = ""
        if "pre_rank" in cols:
            rank_src_col = "pre_rank"
        elif "source_rank" in cols:
            rank_src_col = "source_rank"
        else:
            print(f"[WARN] skip candidate_file={name}: missing pre_rank/source_rank column")
            continue
        c = (
            raw
            .select(
                F.col("user_idx").cast("int").alias("user_idx"),
                F.col("item_idx").cast("int").alias("item_idx"),
                F.col(rank_src_col).cast("int").alias("pre_rank"),
            )
            .filter(F.col("user_idx").isNotNull() & F.col("item_idx").isNotNull() & F.col("pre_rank").isNotNull())
            .join(user_idx_df, on="user_idx", how="inner")
            .withColumn("candidate_file", F.lit(name))
            .select("candidate_file", "user_idx", "item_idx", "pre_rank")
        )
        frames.append(c)

    if not frames:
        raise RuntimeError(f"no candidate parquet loaded under {bucket_dir}")
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f, allowMissingColumns=False)
    return out


def build_bucket_metrics_df(
    spark: SparkSession,
    *,
    bucket: int,
    bucket_dir: Path,
    topn_grid: list[int],
    include_valid_pos: bool,
) -> DataFrame | None:
    truth_path = bucket_dir / "truth.parquet"
    if not truth_path.exists():
        print(f"[WARN] skip bucket={bucket}: missing truth.parquet")
        return None

    candidate_files = list_candidate_files(bucket_dir)
    if not candidate_files:
        print(f"[WARN] skip bucket={bucket}: no candidate files")
        return None

    truth = (
        spark.read.parquet(truth_path.as_posix())
        .select(
            F.col("user_idx").cast("int").alias("user_idx"),
            F.col("true_item_idx").cast("int").alias("true_item_idx"),
            F.col("valid_item_idx").cast("int").alias("valid_item_idx"),
        )
        .filter(F.col("user_idx").isNotNull())
        .dropDuplicates(["user_idx"])
    )
    user_idx_df = truth.select("user_idx").dropDuplicates(["user_idx"])

    cand_union = read_candidates_union(
        spark,
        bucket_dir=bucket_dir,
        user_idx_df=user_idx_df,
        candidate_files=candidate_files,
    )

    topn_df = build_literal_int_df(spark, col_name="topn", values=topn_grid)
    cand_file_df = build_literal_str_df(spark, col_name="candidate_file", values=candidate_files)
    combos = cand_file_df.crossJoin(topn_df)

    cand_top = (
        cand_union.crossJoin(F.broadcast(topn_df))
        .filter(F.col("pre_rank") <= F.col("topn"))
        .select("candidate_file", "topn", "user_idx", "item_idx")
    )

    cand_user_cnt = cand_top.groupBy("candidate_file", "topn", "user_idx").agg(F.count("*").alias("cand_cnt"))
    pool_stats = cand_user_cnt.groupBy("candidate_file", "topn").agg(
        F.sum("cand_cnt").alias("rows_in_topn"),
        F.avg("cand_cnt").alias("cand_per_user_avg"),
        F.expr("percentile_approx(cand_cnt, 0.5)").alias("cand_per_user_median"),
    )

    truth_true = truth.filter(F.col("true_item_idx").isNotNull()).select(
        "user_idx", F.col("true_item_idx").alias("item_idx")
    )
    true_hits_user = (
        cand_top.join(truth_true, on=["user_idx", "item_idx"], how="inner")
        .select("candidate_file", "topn", "user_idx")
        .dropDuplicates(["candidate_file", "topn", "user_idx"])
    )
    true_hit_counts = true_hits_user.groupBy("candidate_file", "topn").agg(F.count("*").alias("true_hit_users"))

    truth_valid = truth.filter(F.col("valid_item_idx").isNotNull()).select(
        "user_idx", F.col("valid_item_idx").alias("item_idx")
    )
    valid_hits_user = (
        cand_top.join(truth_valid, on=["user_idx", "item_idx"], how="inner")
        .select("candidate_file", "topn", "user_idx")
        .dropDuplicates(["candidate_file", "topn", "user_idx"])
    )
    valid_hit_counts = valid_hits_user.groupBy("candidate_file", "topn").agg(F.count("*").alias("valid_hit_users"))

    if include_valid_pos:
        pos_hits_user = true_hits_user.unionByName(valid_hits_user, allowMissingColumns=False).dropDuplicates(
            ["candidate_file", "topn", "user_idx"]
        )
    else:
        pos_hits_user = true_hits_user
    pos_hit_counts = pos_hits_user.groupBy("candidate_file", "topn").agg(F.count("*").alias("pos_hit_users"))

    n_users_df = user_idx_df.agg(F.count("*").alias("n_users"))
    out = (
        combos.crossJoin(n_users_df)
        .join(pool_stats, on=["candidate_file", "topn"], how="left")
        .join(true_hit_counts, on=["candidate_file", "topn"], how="left")
        .join(valid_hit_counts, on=["candidate_file", "topn"], how="left")
        .join(pos_hit_counts, on=["candidate_file", "topn"], how="left")
        .fillna(
            {
                "rows_in_topn": 0,
                "cand_per_user_avg": 0.0,
                "cand_per_user_median": 0,
                "true_hit_users": 0,
                "valid_hit_users": 0,
                "pos_hit_users": 0,
            }
        )
        .withColumn("bucket", F.lit(int(bucket)))
        .withColumn("users_no_positive", F.greatest(F.col("n_users") - F.col("pos_hit_users"), F.lit(0)))
        .withColumn(
            "truth_in_topn",
            F.when(F.col("n_users") > F.lit(0), F.col("true_hit_users") / F.col("n_users")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "valid_in_topn",
            F.when(F.col("n_users") > F.lit(0), F.col("valid_hit_users") / F.col("n_users")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "pos_user_coverage",
            F.when(F.col("n_users") > F.lit(0), F.col("pos_hit_users") / F.col("n_users")).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "users_no_positive_ratio",
            F.when(F.col("n_users") > F.lit(0), F.col("users_no_positive") / F.col("n_users")).otherwise(F.lit(0.0)),
        )
        .withColumn("gate_a_pass", F.col("truth_in_topn") >= F.lit(float(GATE_A_MIN_TRUE_IN_TOPN)))
        .withColumn("gate_b_pass", F.col("users_no_positive_ratio") <= F.lit(float(GATE_B_MAX_NO_POS_RATIO)))
        .withColumn("gate_ab_pass", F.col("gate_a_pass") & F.col("gate_b_pass"))
        .select(
            "bucket",
            "candidate_file",
            "topn",
            "n_users",
            "rows_in_topn",
            "cand_per_user_avg",
            "cand_per_user_median",
            "true_hit_users",
            "valid_hit_users",
            "pos_hit_users",
            "users_no_positive",
            "truth_in_topn",
            "valid_in_topn",
            "pos_user_coverage",
            "users_no_positive_ratio",
            "gate_a_pass",
            "gate_b_pass",
            "gate_ab_pass",
        )
    )
    return out


def main() -> None:
    stage09_run = resolve_stage09_run()
    buckets = parse_int_list(BUCKETS_OVERRIDE) or [10]
    topn_grid = parse_int_list(TOPN_GRID_RAW) or [80, 120, 150, 200, 300]

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    rows: list[DataFrame] = []
    for bucket in buckets:
        bdir = stage09_run / f"bucket_{bucket}"
        if not bdir.exists():
            print(f"[WARN] skip bucket={bucket}: bucket dir missing")
            continue
        r = build_bucket_metrics_df(
            spark,
            bucket=int(bucket),
            bucket_dir=bdir,
            topn_grid=topn_grid,
            include_valid_pos=INCLUDE_VALID_POS,
        )
        if r is not None:
            rows.append(r)

    if not rows:
        raise RuntimeError("no bucket metrics produced in data gate audit")

    out_df = rows[0]
    for r in rows[1:]:
        out_df = out_df.unionByName(r, allowMissingColumns=False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_df.orderBy(
        F.col("bucket").asc(),
        F.col("gate_ab_pass").desc(),
        F.col("pos_user_coverage").desc(),
        F.col("truth_in_topn").desc(),
        F.col("topn").asc(),
    ).toPandas()

    out_csv = out_dir / "stage11_qlora_data_gate_audit.csv"
    out_pdf.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_pdf.to_csv(METRICS_LATEST, index=False, encoding="utf-8-sig")

    meta: dict[str, Any] = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_run_09": str(stage09_run),
        "buckets": [int(x) for x in buckets],
        "topn_grid": [int(x) for x in topn_grid],
        "include_valid_pos": bool(INCLUDE_VALID_POS),
        "gate_a_min_true_in_topn": float(GATE_A_MIN_TRUE_IN_TOPN),
        "gate_b_max_no_pos_ratio": float(GATE_B_MAX_NO_POS_RATIO),
        "metrics_file": str(out_csv),
        "metrics_latest_file": str(METRICS_LATEST),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[DONE] data gate audit: {out_csv}")
    print(f"[DONE] latest summary: {METRICS_LATEST}")

    top_show = out_pdf.head(12)
    if len(top_show) > 0:
        print("[TOP]")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(
                top_show[
                    [
                        "bucket",
                        "candidate_file",
                        "topn",
                        "truth_in_topn",
                        "pos_user_coverage",
                        "users_no_positive_ratio",
                        "gate_ab_pass",
                    ]
                ].to_string(index=False)
            )

    spark.stop()


if __name__ == "__main__":
    main()
