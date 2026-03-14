from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window


RUN_TAG = "stage10_rerank_eval"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()  # optional explicit path
INPUT_09_ROOT = Path(r"D:/5006 BDA project/data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/10_rerank_eval")
METRICS_PATH = Path(r"D:/5006 BDA project/data/metrics/recsys_stage10_results.csv")

TOP_K = 10
MAX_PER_PRIMARY_CATEGORY = 4
TAIL_QUANTILE = 0.8

# Keep rerank config centralized for maintainability.
RERANK_POLICY: dict[str, Any] = {
    "source_bonus": {"als": 0.03, "cluster": 0.04, "popular": 0.00, "profile": 0.02},
    "segment_cluster_extra": {"light": 0.03, "mid": 0.015, "heavy": 0.0},
    "als_guard_topn": {"light": 3, "mid": 5, "heavy": 8},
    "quality_bonus_weight": 0.04,
    "popular_only_penalty": 0.01,
    "max_per_primary_category": MAX_PER_PRIMARY_CATEGORY,
}

RESULT_FIELDS = [
    "run_id_10",
    "source_run_09",
    "bucket_min_train_reviews",
    "model",
    "recall_at_k",
    "ndcg_at_k",
    "user_coverage_at_k",
    "item_coverage_at_k",
    "tail_coverage_at_k",
    "novelty_at_k",
    "n_users",
    "n_items",
    "n_candidates",
]


def build_spark() -> SparkSession:
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage10-rerank-eval")
        .master("local[2]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR.strip():
        p = Path(INPUT_09_RUN_DIR.strip())
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def compute_recall_ndcg(rank_df: DataFrame) -> tuple[float, float]:
    metrics = (
        rank_df.withColumn(
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
    return float(row["recall"] or 0.0), float(row["ndcg"] or 0.0)


def build_segment_value_expr(value_by_segment: dict[str, float], default_value: float = 0.0) -> Any:
    expr = F.lit(float(default_value))
    for seg in ("light", "mid", "heavy"):
        expr = F.when(
            F.col("user_segment") == F.lit(seg),
            F.lit(float(value_by_segment.get(seg, default_value))),
        ).otherwise(expr)
    return expr


def eval_from_pred(
    pred_topk: DataFrame,
    truth: DataFrame,
    candidate_pool: DataFrame,
    total_train_events: int,
    top_k: int,
) -> tuple[float, float, float, float, float, float]:
    # pred_topk: user_idx, item_idx, rank
    eval_df = (
        truth.join(
            pred_topk.select("user_idx", F.col("item_idx").alias("pred_item"), "rank"),
            on="user_idx",
            how="left",
        )
        .withColumn("rank", F.when(F.col("pred_item") == F.col("true_item_idx"), F.col("rank")).otherwise(F.lit(0)))
        .groupBy("user_idx")
        .agg(F.max("rank").alias("rank"))
    )
    recall, ndcg = compute_recall_ndcg(eval_df)

    n_users = int(truth.select("user_idx").distinct().count())
    pred_users = int(pred_topk.select("user_idx").distinct().count())
    user_cov = float(pred_users / max(1, n_users))

    n_items_pool = int(candidate_pool.select("item_idx").distinct().count())
    n_items_pred = int(pred_topk.select("item_idx").distinct().count())
    item_cov = float(n_items_pred / max(1, n_items_pool))

    item_pop = (
        candidate_pool.select("item_idx", "item_train_pop_count")
        .dropDuplicates(["item_idx"])
        .filter(F.col("item_train_pop_count").isNotNull())
    )
    quant = item_pop.approxQuantile("item_train_pop_count", [TAIL_QUANTILE], 0.01)
    tail_cutoff = float(quant[0]) if quant else 0.0

    pred_diag = (
        pred_topk.join(item_pop, on="item_idx", how="left")
        .fillna({"item_train_pop_count": 1.0})
        .withColumn(
            "pop_prob",
            F.greatest(F.col("item_train_pop_count").cast("double") / F.lit(float(max(1, total_train_events))), F.lit(1e-12)),
        )
        .withColumn("is_tail", F.when(F.col("item_train_pop_count") <= F.lit(tail_cutoff), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("novelty", -(F.log(F.col("pop_prob")) / F.log(F.lit(2.0))))
    )
    row = pred_diag.agg(F.avg("is_tail").alias("tail_cov"), F.avg("novelty").alias("novelty")).first()
    tail_cov = float(row["tail_cov"] or 0.0)
    novelty = float(row["novelty"] or 0.0)
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    ensure_results_file(METRICS_PATH)

    source_09 = resolve_stage09_run()
    run_id_10 = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id_10}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_dirs = sorted([p for p in source_09.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs under {source_09}")

    meta_rows: list[dict[str, Any]] = []
    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        print(f"\n[BUCKET] {bucket}")
        cand_path = bdir / "candidates_pretrim150.parquet"
        truth_path = bdir / "truth.parquet"
        meta_path = bdir / "bucket_meta.json"
        if not cand_path.exists() or not truth_path.exists() or not meta_path.exists():
            print(f"[WARN] skip bucket={bucket} due to missing files")
            continue

        bucket_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        total_train_events = int(bucket_meta.get("total_train_events", 1))

        cand = spark.read.parquet(cand_path.as_posix()).persist(StorageLevel.DISK_ONLY)
        truth = spark.read.parquet(truth_path.as_posix()).select("user_idx", "true_item_idx").dropDuplicates(["user_idx"]).persist(StorageLevel.DISK_ONLY)
        n_users = int(truth.select("user_idx").distinct().count())
        n_items = int(cand.select("item_idx").distinct().count())
        n_candidates = int(cand.count())
        print(f"[COUNT] users={n_users} items={n_items} candidates={n_candidates}")

        # Baseline 1: ALS@10 (from candidate aux rank).
        als_topk = (
            cand.filter(F.col("als_rank").isNotNull() & (F.col("als_rank") <= F.lit(TOP_K)))
            .select("user_idx", "item_idx", F.col("als_rank").cast("int").alias("rank"))
        )
        als_metrics = eval_from_pred(als_topk, truth, cand, total_train_events, TOP_K)

        # Baseline 2: pre_score top@10.
        pre_topk = (
            cand.filter(F.col("pre_rank") <= F.lit(TOP_K))
            .select("user_idx", "item_idx", F.col("pre_rank").cast("int").alias("rank"))
        )
        pre_metrics = eval_from_pred(pre_topk, truth, cand, total_train_events, TOP_K)

        # Rerank
        source_bonus = RERANK_POLICY["source_bonus"]
        seg_extra = RERANK_POLICY["segment_cluster_extra"]
        als_guard_topn = RERANK_POLICY["als_guard_topn"]
        max_per_cat = int(RERANK_POLICY["max_per_primary_category"])
        als_guard_topn_expr = build_segment_value_expr(als_guard_topn, default_value=0.0)
        cand2 = (
            cand.withColumn("has_als", F.array_contains(F.col("source_set"), F.lit("als")))
            .withColumn("has_cluster", F.array_contains(F.col("source_set"), F.lit("cluster")))
            .withColumn("has_popular", F.array_contains(F.col("source_set"), F.lit("popular")))
            .withColumn("has_profile", F.array_contains(F.col("source_set"), F.lit("profile")))
            .withColumn("als_guard_topn", als_guard_topn_expr)
            .withColumn(
                "is_als_guard",
                F.col("als_rank").isNotNull() & (F.col("als_rank").cast("double") <= F.col("als_guard_topn")),
            )
            .withColumn(
                "blend_bonus",
                F.when(F.col("has_als"), F.lit(float(source_bonus["als"]))).otherwise(F.lit(0.0))
                + F.when(F.col("has_cluster"), F.lit(float(source_bonus["cluster"]))).otherwise(F.lit(0.0))
                + F.when(F.col("has_popular"), F.lit(float(source_bonus["popular"]))).otherwise(F.lit(0.0))
                + F.when(F.col("has_profile"), F.lit(float(source_bonus["profile"]))).otherwise(F.lit(0.0))
            )
            .withColumn(
                "segment_cluster_bonus",
                F.when((F.col("user_segment") == F.lit("light")) & F.col("has_cluster"), F.lit(float(seg_extra["light"])))
                .when((F.col("user_segment") == F.lit("mid")) & F.col("has_cluster"), F.lit(float(seg_extra["mid"])))
                .otherwise(F.lit(float(seg_extra["heavy"]))),
            )
            .withColumn(
                "popular_only_penalty",
                F.when(
                    F.col("has_popular") & (~F.col("has_als")) & (~F.col("has_cluster")),
                    F.lit(float(RERANK_POLICY["popular_only_penalty"])),
                ).otherwise(F.lit(0.0)),
            )
            .withColumn(
                "final_score",
                F.col("pre_score")
                + F.col("blend_bonus")
                + F.lit(float(RERANK_POLICY["quality_bonus_weight"])) * F.coalesce(F.col("quality_score"), F.lit(0.0))
                + F.col("segment_cluster_bonus")
                - F.col("popular_only_penalty"),
            )
        )

        # Keep strong ALS positions stable; apply category cap only to non-guard candidates.
        guard_pool = cand2.filter(F.col("is_als_guard"))
        non_guard_pool = cand2.filter(~F.col("is_als_guard"))
        w_cat = Window.partitionBy("user_idx", "primary_category").orderBy(F.desc("final_score"), F.asc("item_idx"))
        non_guard_capped = (
            non_guard_pool.withColumn("cat_rank", F.row_number().over(w_cat))
            .filter(F.col("cat_rank") <= F.lit(max_per_cat))
        )
        rerank_pool = (
            guard_pool.withColumn("cat_rank", F.lit(0))
            .unionByName(non_guard_capped, allowMissingColumns=True)
            .withColumn("is_als_guard_int", F.when(F.col("is_als_guard"), F.lit(1)).otherwise(F.lit(0)))
        )
        w_user = Window.partitionBy("user_idx").orderBy(
            F.desc("is_als_guard_int"), F.desc("final_score"), F.desc("pre_score"), F.asc("item_idx")
        )
        reranked = (
            rerank_pool.withColumn("final_rank", F.row_number().over(w_user))
            .persist(StorageLevel.DISK_ONLY)
        )
        rerank_topk = reranked.filter(F.col("final_rank") <= F.lit(TOP_K)).select(
            "user_idx", "item_idx", F.col("final_rank").cast("int").alias("rank")
        )
        rerank_metrics = eval_from_pred(rerank_topk, truth, cand, total_train_events, TOP_K)

        # Persist outputs
        bucket_out = out_dir / bdir.name
        bucket_out.mkdir(parents=True, exist_ok=True)
        reranked.write.mode("overwrite").parquet((bucket_out / "reranked_candidates.parquet").as_posix())
        rerank_topk.write.mode("overwrite").parquet((bucket_out / "final_top10.parquet").as_posix())

        metric_models = [
            ("ALS@10_from_candidates", als_metrics),
            ("PreScore@10", pre_metrics),
            ("RerankV1@10", rerank_metrics),
        ]
        for model_name, m in metric_models:
            row = {
                "run_id_10": run_id_10,
                "source_run_09": source_09.name,
                "bucket_min_train_reviews": bucket,
                "model": model_name,
                "recall_at_k": f"{m[0]:.6f}",
                "ndcg_at_k": f"{m[1]:.6f}",
                "user_coverage_at_k": f"{m[2]:.6f}",
                "item_coverage_at_k": f"{m[3]:.6f}",
                "tail_coverage_at_k": f"{m[4]:.6f}",
                "novelty_at_k": f"{m[5]:.6f}",
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_candidates,
            }
            append_result(METRICS_PATH, row)
            meta_rows.append(row)

        print(
            f"[METRIC] bucket={bucket} "
            f"ALS NDCG@10={als_metrics[1]:.4f} "
            f"PreScore NDCG@10={pre_metrics[1]:.4f} "
            f"Rerank NDCG@10={rerank_metrics[1]:.4f}"
        )

        for df in [cand, truth, reranked]:
            df.unpersist()

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id_10": run_id_10,
                "source_run_09": str(source_09),
                "output_dir": str(out_dir),
                "policy": RERANK_POLICY,
                "top_k": TOP_K,
                "rows": meta_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] wrote {out_dir}")
    print(f"[INFO] appended metrics to {METRICS_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
