from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from sklearn.metrics import roc_auc_score


RUN_TAG = "stage10_3_xgb_diagnose"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = Path(r"D:/5006 BDA project/data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"

RANK_MODEL_JSON = os.getenv("RANK_MODEL_JSON", "").strip()
RANK_MODEL_ROOT = Path(r"D:/5006 BDA project/data/output/10_rank_models")
RANK_MODEL_SUFFIX = "_stage10_1_rank_train"
RANK_MODEL_FILE = "rank_model.json"

OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/10_3_xgb_diagnose")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
DIAG_MAX_ROWS = int(os.getenv("DIAG_MAX_ROWS", "350000").strip() or 350000)


def build_spark() -> SparkSession:
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage10-3-xgb-diagnose")
        .master("local[2]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def pick_latest_run(root: Path, suffix: str, required_file: str | None = None) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if required_file is None:
        if not runs:
            raise FileNotFoundError(f"no run in {root} suffix={suffix}")
        return runs[0]
    for run in runs:
        if (run / required_file).exists():
            return run
    raise FileNotFoundError(f"no run in {root} suffix={suffix} file={required_file}")


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def resolve_rank_model_json() -> Path:
    if RANK_MODEL_JSON:
        p = Path(RANK_MODEL_JSON)
        if not p.exists():
            raise FileNotFoundError(f"RANK_MODEL_JSON not found: {p}")
        return p
    run = pick_latest_run(RANK_MODEL_ROOT, RANK_MODEL_SUFFIX, RANK_MODEL_FILE)
    return run / RANK_MODEL_FILE


def collect_single(row_df: DataFrame, col_name: str) -> float:
    r = row_df.first()
    if r is None:
        return 0.0
    v = r[col_name]
    if v is None:
        return 0.0
    return float(v)


def add_feature_columns(cand: DataFrame) -> DataFrame:
    def _inv_rank(col_name: str) -> Any:
        c = F.col(col_name).cast("double")
        return F.when(c.isNull(), F.lit(0.0)).otherwise(F.lit(1.0) / (F.log(c + F.lit(1.0)) / F.log(F.lit(2.0))))

    base = (
        cand.withColumn("pre_score", F.coalesce(F.col("pre_score").cast("double"), F.lit(0.0)))
        .withColumn("signal_score", F.coalesce(F.col("signal_score").cast("double"), F.lit(0.0)))
        .withColumn("quality_score", F.coalesce(F.col("quality_score").cast("double"), F.lit(0.0)))
        .withColumn("item_pop_log", F.log1p(F.coalesce(F.col("item_train_pop_count").cast("double"), F.lit(0.0))))
        .withColumn("user_train_log", F.log1p(F.coalesce(F.col("user_train_count").cast("double"), F.lit(0.0))))
        .withColumn("inv_pre_rank", _inv_rank("pre_rank"))
        .withColumn("inv_als_rank", _inv_rank("als_rank"))
        .withColumn("inv_cluster_rank", _inv_rank("cluster_rank"))
        .withColumn("inv_popular_rank", _inv_rank("popular_rank"))
        .withColumn("has_als", F.when(F.array_contains(F.col("source_set"), F.lit("als")), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn(
            "has_cluster",
            F.when(F.array_contains(F.col("source_set"), F.lit("cluster")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "has_profile",
            F.when(F.array_contains(F.col("source_set"), F.lit("profile")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "has_popular",
            F.when(F.array_contains(F.col("source_set"), F.lit("popular")), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn("source_count", F.coalesce(F.size(F.col("source_set")).cast("double"), F.lit(0.0)))
        .withColumn("is_light", F.when(F.col("user_segment") == F.lit("light"), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("is_mid", F.when(F.col("user_segment") == F.lit("mid"), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("is_heavy", F.when(F.col("user_segment") == F.lit("heavy"), F.lit(1.0)).otherwise(F.lit(0.0)))
    )
    return (
        base.withColumn("pre_score_x_profile", F.col("pre_score") * F.col("has_profile"))
        .withColumn("pre_score_x_cluster", F.col("pre_score") * F.col("has_cluster"))
        .withColumn("pre_score_x_als", F.col("pre_score") * F.col("has_als"))
        .withColumn("signal_x_profile", F.col("signal_score") * F.col("has_profile"))
        .withColumn("quality_x_profile", F.col("quality_score") * F.col("has_profile"))
        .withColumn("profile_x_heavy", F.col("has_profile") * F.col("is_heavy"))
        .withColumn("cluster_x_heavy", F.col("has_cluster") * F.col("is_heavy"))
        .withColumn(
            "inv_pre_minus_best_route",
            F.col("inv_pre_rank") - F.greatest(F.col("inv_als_rank"), F.col("inv_cluster_rank"), F.col("inv_popular_rank")),
        )
        .withColumn("pre_minus_signal", F.col("pre_score") - F.col("signal_score"))
    )


def to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        x = float(v)
    except Exception:
        return default
    if np.isnan(x) or np.isinf(x):
        return default
    return x


def build_feature_stats(pdf: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if pdf.empty or "label" not in pdf.columns:
        return rows
    y = pdf["label"].astype(int).to_numpy(dtype=np.int32)
    for c in feature_cols:
        if c not in pdf.columns:
            continue
        x = pd.to_numeric(pdf[c], errors="coerce")
        m = x.notna()
        if int(m.sum()) < 100:
            continue
        xv = x[m].to_numpy(dtype=np.float64)
        yv = y[m.to_numpy()]
        if np.unique(yv).size < 2:
            auc = None
        else:
            try:
                auc = float(roc_auc_score(yv, xv))
            except Exception:
                auc = None
        pos_mean = float(np.nanmean(xv[yv == 1])) if np.any(yv == 1) else 0.0
        neg_mean = float(np.nanmean(xv[yv == 0])) if np.any(yv == 0) else 0.0

        deciles: list[dict[str, Any]] = []
        try:
            bins = pd.qcut(x[m], 10, labels=False, duplicates="drop")
            d = pd.DataFrame({"bin": bins.astype("float"), "label": yv}).dropna()
            if not d.empty:
                grp = d.groupby("bin", dropna=False)["label"].agg(["count", "mean"]).reset_index()
                grp = grp.sort_values("bin")
                for _, rr in grp.iterrows():
                    deciles.append(
                        {
                            "bin": int(rr["bin"]),
                            "count": int(rr["count"]),
                            "hit_rate": to_float(rr["mean"]),
                        }
                    )
        except Exception:
            pass
        rows.append(
            {
                "feature": c,
                "auc": auc,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "mean_gap": pos_mean - neg_mean,
                "deciles": deciles,
            }
        )
    rows.sort(
        key=lambda r: (
            -abs(to_float(r.get("auc"), 0.5) - 0.5),
            -abs(to_float(r.get("mean_gap"), 0.0)),
        )
    )
    return rows


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    stage09_run = resolve_stage09_run()
    model_json = resolve_rank_model_json()
    model_data = json.loads(model_json.read_text(encoding="utf-8"))
    params = model_data.get("params", {})
    holdout_user_frac = float(params.get("holdout_user_frac", 0.2))
    random_seed = int(params.get("random_seed", 42))
    model_backend = str(params.get("model_backend", "unknown"))
    feature_cols = list(model_data.get("feature_columns", []))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_dirs = sorted([p for p in stage09_run.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs under {stage09_run}")

    summary_rows: list[dict[str, Any]] = []
    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        cand_path = bdir / "candidates_pretrim150.parquet"
        truth_path = bdir / "truth.parquet"
        if not cand_path.exists() or not truth_path.exists():
            continue

        print(f"[BUCKET] {bucket}")
        cand_raw = spark.read.parquet(cand_path.as_posix()).persist(StorageLevel.DISK_ONLY)
        cand = add_feature_columns(cand_raw).persist(StorageLevel.DISK_ONLY)
        truth = spark.read.parquet(truth_path.as_posix()).select("user_idx", "true_item_idx").dropDuplicates(["user_idx"]).persist(
            StorageLevel.DISK_ONLY
        )

        holdout_users = (
            truth.select("user_idx")
            .distinct()
            .withColumn("is_valid", F.rand(seed=random_seed + bucket) < F.lit(holdout_user_frac))
            .filter(F.col("is_valid"))
            .select("user_idx")
        ).persist(StorageLevel.DISK_ONLY)

        n_valid_users = int(holdout_users.count())

        pos = (
            holdout_users.join(truth, on="user_idx", how="inner")
            .join(cand, on="user_idx", how="left")
            .filter(F.col("item_idx") == F.col("true_item_idx"))
            .select(
                "user_idx",
                "pre_rank",
                "source_set",
                F.when(F.array_contains(F.col("source_set"), F.lit("als")), F.lit(1.0)).otherwise(F.lit(0.0)).alias("hit_has_als"),
                F.when(F.array_contains(F.col("source_set"), F.lit("cluster")), F.lit(1.0)).otherwise(F.lit(0.0)).alias("hit_has_cluster"),
                F.when(F.array_contains(F.col("source_set"), F.lit("profile")), F.lit(1.0)).otherwise(F.lit(0.0)).alias("hit_has_profile"),
                F.when(F.array_contains(F.col("source_set"), F.lit("popular")), F.lit(1.0)).otherwise(F.lit(0.0)).alias("hit_has_popular"),
            )
            .dropDuplicates(["user_idx"])
            .persist(StorageLevel.DISK_ONLY)
        )

        n_hit_users = int(pos.select("user_idx").distinct().count())
        n_miss_users = int(max(0, n_valid_users - n_hit_users))

        hit_rate = float(n_hit_users / max(1, n_valid_users))
        miss_rate = float(n_miss_users / max(1, n_valid_users))

        top10_rate = collect_single(
            pos.select(F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(10)), F.lit(1.0)).otherwise(F.lit(0.0))).alias("v")),
            "v",
        )
        top50_rate = collect_single(
            pos.select(F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(50)), F.lit(1.0)).otherwise(F.lit(0.0))).alias("v")),
            "v",
        )
        top150_rate = collect_single(
            pos.select(F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(150)), F.lit(1.0)).otherwise(F.lit(0.0))).alias("v")),
            "v",
        )

        src_stats = pos.agg(
            F.avg("hit_has_als").alias("src_hit_als"),
            F.avg("hit_has_cluster").alias("src_hit_cluster"),
            F.avg("hit_has_profile").alias("src_hit_profile"),
            F.avg("hit_has_popular").alias("src_hit_popular"),
        ).first()

        row = {
            "bucket": bucket,
            "valid_users": n_valid_users,
            "valid_hit_users_in_candidates": n_hit_users,
            "valid_miss_users_not_in_candidates": n_miss_users,
            "candidate_hit_rate": hit_rate,
            "candidate_miss_rate": miss_rate,
            "hit_top10_rate": top10_rate,
            "hit_top50_rate": top50_rate,
            "hit_top150_rate": top150_rate,
            "source_hit_als_rate": float(src_stats["src_hit_als"] or 0.0),
            "source_hit_cluster_rate": float(src_stats["src_hit_cluster"] or 0.0),
            "source_hit_profile_rate": float(src_stats["src_hit_profile"] or 0.0),
            "source_hit_popular_rate": float(src_stats["src_hit_popular"] or 0.0),
        }

        pos_ranks = (
            holdout_users.join(truth, on="user_idx", how="inner")
            .join(cand.select("user_idx", "item_idx", "pre_rank"), on="user_idx", how="left")
            .filter(F.col("item_idx") == F.col("true_item_idx"))
            .select("user_idx", "pre_rank")
            .dropDuplicates(["user_idx"])
            .persist(StorageLevel.DISK_ONLY)
        )
        pos_rank_stats = pos_ranks.agg(
            F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(10)), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "top10"
            ),
            F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(50)), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "top50"
            ),
            F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(150)), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "top150"
            ),
            F.avg(F.when(F.col("pre_rank").isNotNull() & (F.col("pre_rank") <= F.lit(300)), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "top300"
            ),
            F.avg(F.when(F.col("pre_rank").isNull(), F.lit(1.0)).otherwise(F.lit(0.0))).alias("missing"),
        ).first()
        row["true_item_rank_distribution"] = {
            "top10": to_float(pos_rank_stats["top10"]),
            "top50": to_float(pos_rank_stats["top50"]),
            "top150": to_float(pos_rank_stats["top150"]),
            "top300": to_float(pos_rank_stats["top300"]),
            "missing": to_float(pos_rank_stats["missing"]),
        }

        label_ds = (
            holdout_users.join(cand, on="user_idx", how="inner")
            .join(truth, on="user_idx", how="left")
            .withColumn("label", F.when(F.col("item_idx") == F.col("true_item_idx"), F.lit(1)).otherwise(F.lit(0)))
            .select("label", "pre_rank", *[c for c in feature_cols if c in cand.columns])
        ).persist(StorageLevel.DISK_ONLY)
        n_label_rows = int(label_ds.count())
        frac = 1.0
        if DIAG_MAX_ROWS > 0 and n_label_rows > DIAG_MAX_ROWS:
            frac = float(DIAG_MAX_ROWS) / float(n_label_rows)
        label_sample = label_ds.sample(withReplacement=False, fraction=min(1.0, frac), seed=random_seed + bucket + 77)
        label_pdf = label_sample.toPandas()
        row["feature_diagnostics"] = build_feature_stats(label_pdf, feature_cols)
        row["feature_diag_sample_rows"] = int(len(label_pdf))
        row["feature_diag_total_rows"] = int(n_label_rows)
        label_ds.unpersist()
        pos_ranks.unpersist()
        summary_rows.append(row)
        print(f"[DIAG] {row}")

        pos.unpersist()
        holdout_users.unpersist()
        truth.unpersist()
        cand.unpersist()
        cand_raw.unpersist()

    out_payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "stage09_run": str(stage09_run),
        "rank_model_json": str(model_json),
        "model_backend": model_backend,
        "holdout_user_frac": holdout_user_frac,
        "random_seed": random_seed,
        "rows": summary_rows,
    }
    out_json = out_dir / "xgb_diagnose_summary.json"
    out_json.write_text(json.dumps(out_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] wrote {out_json}")
    spark.stop()


if __name__ == "__main__":
    main()
