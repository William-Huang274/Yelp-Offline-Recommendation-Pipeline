from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pipeline.project_paths import env_or_project_path, project_path
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context


RUN_TAG = "stage09_recall_audit"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_AUDIT_ROOT_DIR", "data/output/09_recall_audit")
METRICS_DIR = env_or_project_path("METRICS_DIR", "data/metrics")

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "8").strip() or "8"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "8").strip() or "8"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
PY_TEMP_DIR = os.getenv("PY_TEMP_DIR", "").strip()
BUCKETS_OVERRIDE = os.getenv("MIN_TRAIN_BUCKETS_OVERRIDE", "").strip()
AUDIT_USER_COHORT_CSV = os.getenv("AUDIT_USER_COHORT_CSV", "").strip()
AUDIT_WRITE_USER_DETAIL_CSV = os.getenv("AUDIT_WRITE_USER_DETAIL_CSV", "true").strip().lower() == "true"
PRE_RANK_THRESHOLDS_RAW = os.getenv("AUDIT_PRE_RANK_THRESHOLDS", "10,20,50,100,150").strip()

LIGHT_MAX_TRAIN = int(os.getenv("AUDIT_LIGHT_MAX_TRAIN", "7").strip() or 7)
MID_MAX_TRAIN = int(os.getenv("AUDIT_MID_MAX_TRAIN", "19").strip() or 19)

PRETRIM_CANDIDATE_FILES = [
    "candidates_pretrim.parquet",
    "candidates_pretrim150.parquet",
    "candidates_pretrim250.parquet",
    "candidates_pretrim300.parquet",
    "candidates_pretrim360.parquet",
    "candidates_pretrim500.parquet",
]
ROUTES = ["als", "cluster", "profile", "popular"]


_SPARK_TMP_CTX: SparkTmpContext | None = None


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
        SparkSession.builder.appName("stage09-recall-audit")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def parse_bucket_override(raw: str) -> set[int]:
    out: set[int] = set()
    text = str(raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            continue
    return out


def parse_rank_thresholds(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    out = sorted(set(out))
    return out or [10, 20, 50, 100, 150]


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


def pick_pretrim_file(bucket_dir: Path) -> Path:
    for name in PRETRIM_CANDIDATE_FILES:
        p = bucket_dir / name
        if p.exists():
            return p
    for p in sorted(bucket_dir.glob("candidates_pretrim*.parquet")):
        if p.exists():
            return p
    raise FileNotFoundError(f"no pretrim parquet found under {bucket_dir}")


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def to_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def collect_single(df: DataFrame) -> dict[str, Any]:
    row = df.first()
    if row is None:
        return {}
    return row.asDict(recursive=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_cohort_users(path_raw: str) -> pd.DataFrame:
    path = Path(path_raw)
    if not path.exists():
        raise FileNotFoundError(f"AUDIT_USER_COHORT_CSV not found: {path}")
    pdf = pd.read_csv(path)
    cols: list[str] = []
    if "user_id" in pdf.columns:
        pdf["user_id"] = pdf["user_id"].astype(str).str.strip()
        pdf = pdf[pdf["user_id"] != ""].copy()
        cols.append("user_id")
    if "user_idx" in pdf.columns:
        pdf["user_idx"] = pd.to_numeric(pdf["user_idx"], errors="coerce")
        pdf = pdf[pdf["user_idx"].notna()].copy()
        pdf["user_idx"] = pdf["user_idx"].astype("int64")
        cols.append("user_idx")
    if not cols:
        raise RuntimeError(f"AUDIT_USER_COHORT_CSV missing user_id/user_idx column: {path}")
    return pdf[cols].drop_duplicates()


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    run_dir = resolve_stage09_run()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    bucket_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    wanted = parse_bucket_override(BUCKETS_OVERRIDE)
    if wanted:
        bucket_dirs = [p for p in bucket_dirs if int(p.name.split("_")[-1]) in wanted]
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs under {run_dir}")

    summary_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    rank_thresholds = parse_rank_thresholds(PRE_RANK_THRESHOLDS_RAW)
    cohort_sp = None
    cohort_users_count = 0
    cohort_join_key = "user_idx"
    if AUDIT_USER_COHORT_CSV:
        cohort_pdf = load_cohort_users(AUDIT_USER_COHORT_CSV)
        cohort_users_count = int(cohort_pdf.shape[0])
        cohort_sp = spark.createDataFrame(cohort_pdf)
        cohort_join_key = "user_id" if "user_id" in cohort_pdf.columns else "user_idx"
        print(f"[COHORT] using AUDIT_USER_COHORT_CSV={AUDIT_USER_COHORT_CSV} users={cohort_users_count}")

    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        meta_path = bdir / "bucket_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        pretrim_path = pick_pretrim_file(bdir)

        truth_base = spark.read.parquet((bdir / "truth.parquet").as_posix())
        truth_cols = [c for c in ("user_idx", "user_id", "true_item_idx") if c in truth_base.columns]
        truth = truth_base.select(*truth_cols).dropDuplicates(["user_idx"])
        if cohort_sp is not None:
            if cohort_join_key == "user_id" and "user_id" in truth.columns:
                truth = truth.join(
                    F.broadcast(cohort_sp.select("user_id").dropDuplicates(["user_id"])),
                    on="user_id",
                    how="inner",
                )
            else:
                truth = truth.join(
                    F.broadcast(cohort_sp.select("user_idx").dropDuplicates(["user_idx"])),
                    on="user_idx",
                    how="inner",
                )
        truth = truth.persist(StorageLevel.DISK_ONLY)
        truth_user_filter = truth.select("user_idx").dropDuplicates(["user_idx"]).persist(StorageLevel.DISK_ONLY)
        cand_all = (
            spark.read.parquet((bdir / "candidates_all.parquet").as_posix())
            .select("user_idx", "item_idx", "source")
        )
        if cohort_sp is not None:
            cand_all = cand_all.join(F.broadcast(truth_user_filter), on="user_idx", how="inner")
        cand_all = cand_all.persist(StorageLevel.DISK_ONLY)
        cand_all_u = cand_all.select("user_idx", "item_idx").dropDuplicates(["user_idx", "item_idx"]).persist(StorageLevel.DISK_ONLY)
        cand_pre_u = (
            spark.read.parquet(pretrim_path.as_posix())
            .select("user_idx", "item_idx", "pre_rank", "pre_score")
            .dropDuplicates(["user_idx", "item_idx"])
        )
        if cohort_sp is not None:
            cand_pre_u = cand_pre_u.join(F.broadcast(truth_user_filter), on="user_idx", how="inner")
        cand_pre_u = cand_pre_u.persist(StorageLevel.DISK_ONLY)

        n_users = int(truth.count())
        truth_hit = (
            truth.alias("t")
            .join(
                cand_all_u.alias("a"),
                (F.col("t.user_idx") == F.col("a.user_idx")) & (F.col("t.true_item_idx") == F.col("a.item_idx")),
                how="left",
            )
            .join(
                cand_pre_u.alias("p"),
                (F.col("t.user_idx") == F.col("p.user_idx")) & (F.col("t.true_item_idx") == F.col("p.item_idx")),
                how="left",
            )
            .select(
                F.col("t.user_idx").alias("user_idx"),
                F.col("t.true_item_idx").alias("true_item_idx"),
                F.when(F.col("a.item_idx").isNotNull(), F.lit(1)).otherwise(F.lit(0)).alias("hit_all"),
                F.when(F.col("p.item_idx").isNotNull(), F.lit(1)).otherwise(F.lit(0)).alias("hit_pretrim"),
                F.col("p.pre_rank").cast("int").alias("true_pre_rank"),
                F.col("p.pre_score").cast("double").alias("true_pre_score"),
            )
            .persist(StorageLevel.DISK_ONLY)
        )
        hit_counts = collect_single(
            truth_hit.agg(
                F.sum("hit_all").alias("hit_all_users"),
                F.sum("hit_pretrim").alias("hit_pretrim_users"),
            )
        )
        hit_all_users = to_int(hit_counts.get("hit_all_users"))
        hit_pretrim_users = to_int(hit_counts.get("hit_pretrim_users"))

        truth_in_all = float(hit_all_users / max(1, n_users))
        truth_in_pretrim = float(hit_pretrim_users / max(1, n_users))
        pretrim_cut_loss = float(max(0.0, truth_in_all - truth_in_pretrim))
        hard_miss = float(max(0.0, 1.0 - truth_in_all))

        top1_scores = (
            cand_pre_u.filter(F.col("pre_rank") == F.lit(1))
            .select("user_idx", F.col("pre_score").cast("double").alias("top1_pre_score"))
        )
        top10_scores = (
            cand_pre_u.filter(F.col("pre_rank") <= F.lit(10))
            .groupBy("user_idx")
            .agg(F.min(F.col("pre_score").cast("double")).alias("top10_floor_pre_score"))
        )
        truth_pre_metrics = (
            truth_hit.filter(F.col("hit_pretrim") == F.lit(1))
            .join(top1_scores, on="user_idx", how="left")
            .join(top10_scores, on="user_idx", how="left")
            .withColumn(
                "top1_minus_truth_pre_score",
                F.when(
                    F.col("true_pre_score").isNotNull() & F.col("top1_pre_score").isNotNull(),
                    F.col("top1_pre_score") - F.col("true_pre_score"),
                ),
            )
            .withColumn(
                "top10_floor_minus_truth_pre_score",
                F.when(
                    F.col("true_pre_score").isNotNull() & F.col("top10_floor_pre_score").isNotNull(),
                    F.col("top10_floor_pre_score") - F.col("true_pre_score"),
                ),
            )
        ).persist(StorageLevel.DISK_ONLY)
        rank_metric_exprs = []
        for k in rank_thresholds:
            rank_metric_exprs.append(
                F.avg(F.when(F.col("true_pre_rank") <= F.lit(int(k)), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                    f"truth_pre_rank_le_{int(k)}"
                )
            )
        truth_pre_summary = collect_single(
            truth_pre_metrics.agg(
                F.expr("percentile_approx(true_pre_rank, 0.5)").alias("true_pre_rank_hit_median"),
                F.avg("top1_minus_truth_pre_score").alias("top1_minus_truth_pre_score_mean"),
                F.avg("top10_floor_minus_truth_pre_score").alias("top10_floor_minus_truth_pre_score_mean"),
                *rank_metric_exprs,
            )
        )

        user_pool_stats = (
            truth.select("user_idx")
            .join(cand_all_u.groupBy("user_idx").agg(F.count("*").alias("cand_all_cnt")), on="user_idx", how="left")
            .join(cand_pre_u.groupBy("user_idx").agg(F.count("*").alias("cand_pretrim_cnt")), on="user_idx", how="left")
            .fillna({"cand_all_cnt": 0, "cand_pretrim_cnt": 0})
            .agg(
                F.avg("cand_all_cnt").alias("cand_all_avg"),
                F.expr("percentile_approx(cand_all_cnt, 0.5)").alias("cand_all_median"),
                F.avg("cand_pretrim_cnt").alias("cand_pretrim_avg"),
                F.expr("percentile_approx(cand_pretrim_cnt, 0.5)").alias("cand_pretrim_median"),
            )
        )
        pool_stats = collect_single(user_pool_stats)

        route_hit_users = (
            truth.alias("t")
            .join(
                cand_all.alias("c"),
                (F.col("t.user_idx") == F.col("c.user_idx")) & (F.col("t.true_item_idx") == F.col("c.item_idx")),
                how="inner",
            )
            .select(F.col("t.user_idx").alias("user_idx"), F.col("c.source").alias("source"))
            .dropDuplicates(["user_idx", "source"])
        )
        route_sets = (
            truth.select("user_idx")
            .join(
                route_hit_users.groupBy("user_idx").agg(
                    F.collect_set("source").alias("hit_routes"),
                    F.countDistinct("source").alias("hit_route_cnt"),
                ),
                on="user_idx",
                how="left",
            )
            .fillna({"hit_route_cnt": 0})
            .persist(StorageLevel.DISK_ONLY)
        )
        route_agg_exprs = [
            F.avg(F.when(F.col("hit_route_cnt") >= F.lit(2), F.lit(1.0)).otherwise(F.lit(0.0))).alias("route_overlap_rate_on_all"),
            F.avg(F.when(F.col("hit_route_cnt") == F.lit(0), F.lit(1.0)).otherwise(F.lit(0.0))).alias("route_zero_hit_rate_on_all"),
        ]
        for r in ROUTES:
            route_agg_exprs.append(
                F.avg(F.when(F.array_contains(F.col("hit_routes"), F.lit(r)), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                    f"route_any_{r}"
                )
            )
            route_agg_exprs.append(
                F.avg(
                    F.when(
                        (F.col("hit_route_cnt") == F.lit(1)) & F.array_contains(F.col("hit_routes"), F.lit(r)),
                        F.lit(1.0),
                    ).otherwise(F.lit(0.0))
                ).alias(f"route_unique_{r}")
            )
        route_stats = collect_single(route_sets.agg(*route_agg_exprs))

        overlap_on_hit = collect_single(
            route_sets.filter(F.col("hit_route_cnt") > F.lit(0)).agg(
                F.avg(F.when(F.col("hit_route_cnt") >= F.lit(2), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                    "route_overlap_rate_on_hit"
                )
            )
        )

        train_history = spark.read.parquet((bdir / "train_history.parquet").as_posix()).select("user_idx", "item_idx").persist(
            StorageLevel.DISK_ONLY
        )
        user_seg = (
            train_history.groupBy("user_idx")
            .agg(F.count("*").alias("user_train_count"))
            .withColumn(
                "user_segment",
                F.when(F.col("user_train_count") <= F.lit(LIGHT_MAX_TRAIN), F.lit("light"))
                .when(F.col("user_train_count") <= F.lit(MID_MAX_TRAIN), F.lit("mid"))
                .otherwise(F.lit("heavy")),
            )
            .persist(StorageLevel.DISK_ONLY)
        )
        seg_hit = (
            truth_hit.join(user_seg, on="user_idx", how="left")
            .groupBy("user_segment")
            .agg(
                F.count("*").alias("n_users"),
                F.avg(F.col("hit_all").cast("double")).alias("truth_in_all"),
                F.avg(F.col("hit_pretrim").cast("double")).alias("truth_in_pretrim"),
            )
        )
        seg_pdf = seg_hit.toPandas()
        seg_map: dict[str, dict[str, Any]] = {}
        for _, rr in seg_pdf.iterrows():
            seg = str(rr["user_segment"] or "unknown")
            seg_map[seg] = {
                "n_users": int(rr["n_users"]),
                "truth_in_all": to_float(rr["truth_in_all"]),
                "truth_in_pretrim": to_float(rr["truth_in_pretrim"]),
            }

        item_pop = train_history.groupBy("item_idx").agg(F.count("*").alias("item_pop"))
        pop_stats = collect_single(
            truth_hit.join(item_pop, truth_hit.true_item_idx == item_pop.item_idx, how="left")
            .fillna({"item_pop": 0})
            .agg(
                F.expr("percentile_approx(item_pop, 0.5)").alias("true_item_pop_median_all"),
                F.expr("percentile_approx(IF(hit_all = 1, item_pop, NULL), 0.5)").alias("true_item_pop_median_hit"),
                F.expr("percentile_approx(IF(hit_all = 0, item_pop, NULL), 0.5)").alias("true_item_pop_median_miss"),
            )
        )

        row = {
            "run_id": run_id,
            "source_run_09": run_dir.name,
            "bucket": int(bucket),
            "n_users": int(n_users),
            "cohort_users": int(cohort_users_count) if cohort_sp is not None else None,
            "cohort_csv": AUDIT_USER_COHORT_CSV or None,
            "truth_in_all": round(truth_in_all, 6),
            "truth_in_pretrim": round(truth_in_pretrim, 6),
            "pretrim_cut_loss": round(pretrim_cut_loss, 6),
            "hard_miss": round(hard_miss, 6),
            "hit_all_users": int(hit_all_users),
            "hit_pretrim_users": int(hit_pretrim_users),
            "truth_pre_rank_hit_median": int(to_int(truth_pre_summary.get("true_pre_rank_hit_median"))),
            "top1_minus_truth_pre_score_mean": round(
                to_float(truth_pre_summary.get("top1_minus_truth_pre_score_mean")), 6
            ),
            "top10_floor_minus_truth_pre_score_mean": round(
                to_float(truth_pre_summary.get("top10_floor_minus_truth_pre_score_mean")), 6
            ),
            "pretrim_file": pretrim_path.name,
            "pretrim_top_k_used": int(meta.get("pretrim_top_k_used", 0)),
            "cand_all_avg": round(to_float(pool_stats.get("cand_all_avg")), 3),
            "cand_all_median": int(to_int(pool_stats.get("cand_all_median"))),
            "cand_pretrim_avg": round(to_float(pool_stats.get("cand_pretrim_avg")), 3),
            "cand_pretrim_median": int(to_int(pool_stats.get("cand_pretrim_median"))),
            "route_overlap_rate_on_all": round(to_float(route_stats.get("route_overlap_rate_on_all")), 6),
            "route_overlap_rate_on_hit": round(to_float(overlap_on_hit.get("route_overlap_rate_on_hit")), 6),
            "route_zero_hit_rate_on_all": round(to_float(route_stats.get("route_zero_hit_rate_on_all")), 6),
            "true_item_pop_median_all": int(to_int(pop_stats.get("true_item_pop_median_all"))),
            "true_item_pop_median_hit": int(to_int(pop_stats.get("true_item_pop_median_hit"))),
            "true_item_pop_median_miss": int(to_int(pop_stats.get("true_item_pop_median_miss"))),
            "segment_stats": seg_map,
            "profile_routes_enabled": ",".join(meta.get("profile_recall_enabled_routes", []) or []),
        }
        for r in ROUTES:
            row[f"route_any_{r}"] = round(to_float(route_stats.get(f"route_any_{r}")), 6)
            row[f"route_unique_{r}"] = round(to_float(route_stats.get(f"route_unique_{r}")), 6)
            route_rows.append(
                {
                    "run_id": run_id,
                    "source_run_09": run_dir.name,
                    "bucket": int(bucket),
                    "route": r,
                    "route_any_hit_rate": row[f"route_any_{r}"],
                    "route_unique_hit_rate": row[f"route_unique_{r}"],
                }
            )
        for k in rank_thresholds:
            row[f"truth_pre_rank_le_{int(k)}"] = round(
                to_float(truth_pre_summary.get(f"truth_pre_rank_le_{int(k)}")), 6
            )
        summary_rows.append(row)

        print(
            f"[METRIC] bucket={bucket} truth@all={truth_in_all:.4f} "
            f"truth@pretrim={truth_in_pretrim:.4f} cut_loss={pretrim_cut_loss:.4f} "
            f"hard_miss={hard_miss:.4f} pre_rank<=10={to_float(truth_pre_summary.get('truth_pre_rank_le_10')):.4f}"
        )

        if AUDIT_WRITE_USER_DETAIL_CSV:
            detail_dir = out_dir / f"bucket_{bucket}_truth_user_detail_csv"
            (
                truth_hit.join(top1_scores, on="user_idx", how="left")
                .join(top10_scores, on="user_idx", how="left")
                .withColumn(
                    "top1_minus_truth_pre_score",
                    F.when(
                        F.col("true_pre_score").isNotNull() & F.col("top1_pre_score").isNotNull(),
                        F.col("top1_pre_score") - F.col("true_pre_score"),
                    ),
                )
                .withColumn(
                    "top10_floor_minus_truth_pre_score",
                    F.when(
                        F.col("true_pre_score").isNotNull() & F.col("top10_floor_pre_score").isNotNull(),
                        F.col("top10_floor_pre_score") - F.col("true_pre_score"),
                    ),
                )
                .coalesce(1)
                .write.mode("overwrite")
                .option("header", True)
                .csv(detail_dir.as_posix())
            )

        route_sets.unpersist()
        user_seg.unpersist()
        train_history.unpersist()
        truth_pre_metrics.unpersist()
        truth_hit.unpersist()
        cand_pre_u.unpersist()
        cand_all_u.unpersist()
        cand_all.unpersist()
        truth_user_filter.unpersist()
        truth.unpersist()

    summary_pdf = pd.DataFrame(summary_rows) if summary_rows else None
    routes_pdf = pd.DataFrame(route_rows) if route_rows else None

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_run_09": str(run_dir),
        "rows": summary_rows,
        "routes": route_rows,
    }
    write_json(out_dir / "stage09_recall_audit.json", payload)

    if summary_pdf is not None:
        summary_csv = out_dir / "stage09_recall_audit_summary.csv"
        summary_pdf.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        summary_pdf.to_csv(METRICS_DIR / "stage09_recall_audit_summary_latest.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {summary_csv}")

    if routes_pdf is not None:
        routes_csv = out_dir / "stage09_recall_audit_routes.csv"
        routes_pdf.to_csv(routes_csv, index=False, encoding="utf-8-sig")
        routes_pdf.to_csv(METRICS_DIR / "stage09_recall_audit_routes_latest.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {routes_csv}")

    print(f"[INFO] wrote {out_dir / 'stage09_recall_audit.json'}")
    spark.stop()


if __name__ == "__main__":
    main()
