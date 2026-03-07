from __future__ import annotations

import importlib.util
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


BASE_SCRIPT = Path(__file__).resolve().with_name("04_la_recsys_valid.py")
PROFILE_MERGED_CSV = ""  # optional explicit path to biz_profile_recsys.csv

PROFILE_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels/full")
RESULTS_PATH = Path(r"D:/5006 BDA project/data/metrics/la_recsys_cluster_probe_results.csv")


def load_base_module():
    spec = importlib.util.spec_from_file_location("recsys04", str(BASE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {BASE_SCRIPT}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pick_profile_csv() -> Path:
    if PROFILE_MERGED_CSV.strip():
        p = Path(PROFILE_MERGED_CSV.strip())
        if not p.exists():
            raise FileNotFoundError(f"PROFILE_MERGED_CSV not found: {p}")
        return p
    runs = [p for p in PROFILE_ROOT.iterdir() if p.is_dir() and p.name.endswith("_full_profile_merged")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        csv_path = run / "biz_profile_recsys.csv"
        if csv_path.exists():
            return csv_path
    raise FileNotFoundError(f"No profile merged run found under {PROFILE_ROOT}")


def load_cluster_map(spark, profile_csv: Path) -> DataFrame:
    sdf = spark.read.csv(str(profile_csv), header=True)
    if "business_id" not in sdf.columns:
        raise RuntimeError("Profile csv missing business_id")
    cluster_col = "cluster_for_recsys" if "cluster_for_recsys" in sdf.columns else None
    if cluster_col is None:
        raise RuntimeError("Profile csv missing cluster_for_recsys")
    return (
        sdf.select(
            F.col("business_id").cast("string").alias("business_id"),
            F.col(cluster_col).cast("string").alias("cluster_id"),
        )
        .filter(F.col("business_id").isNotNull() & F.col("cluster_id").isNotNull() & (F.col("cluster_id") != ""))
        .dropDuplicates(["business_id"])
    )


def evaluate_cluster_popular_baseline(
    train: DataFrame,
    test: DataFrame,
    cluster_map: DataFrame,
    top_k: int,
    compute_recall_ndcg_fn,
    popular_fallback_fn,
) -> tuple[float, float]:
    train_c = train.join(cluster_map, on="business_id", how="inner")
    if train_c.rdd.isEmpty():
        return popular_fallback_fn(train, test, top_k)

    # Global fallback list when user/cluster evidence is missing.
    pop = train.groupBy("business_id").agg(F.count("*").alias("cnt"))
    w_pop = Window.orderBy(F.desc("cnt"), F.asc("business_id"))
    pop_ranked = (
        pop.withColumn("global_rank", F.row_number().over(w_pop))
        .filter(F.col("global_rank") <= top_k)
        .select("business_id", "global_rank")
    )

    # Cluster-specific popularity.
    c_pop = train_c.groupBy("cluster_id", "business_id").agg(F.count("*").alias("cnt"))
    w_c = Window.partitionBy("cluster_id").orderBy(F.desc("cnt"), F.asc("business_id"))
    c_top = (
        c_pop.withColumn("cluster_rank", F.row_number().over(w_c))
        .filter(F.col("cluster_rank") <= top_k)
        .select("cluster_id", "business_id", "cluster_rank")
    )

    # User's dominant cluster from history.
    u_c = train_c.groupBy("user_id", "cluster_id").agg(F.count("*").alias("cnt"))
    w_u = Window.partitionBy("user_id").orderBy(F.desc("cnt"), F.asc("cluster_id"))
    u_top = (
        u_c.withColumn("rn", F.row_number().over(w_u))
        .filter(F.col("rn") == 1)
        .select("user_id", "cluster_id")
    )

    eval_df = (
        test.select("user_id", "business_id")
        .distinct()
        .join(u_top, on="user_id", how="left")
        .join(c_top, on=["cluster_id", "business_id"], how="left")
        .join(pop_ranked, on="business_id", how="left")
        .withColumn("rank", F.coalesce(F.col("cluster_rank"), F.col("global_rank")))
        .select("user_id", "business_id", "rank")
    )
    return compute_recall_ndcg_fn(eval_df)


def main() -> None:
    tmp_dir = Path(r"D:/5006 BDA project/data/spark-tmp/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    m04 = load_base_module()
    profile_csv = pick_profile_csv()
    print(f"[CONFIG] profile_csv={profile_csv}")

    spark = m04.build_spark()
    m04.set_log_level(spark)

    base = Path(r"D:/5006 BDA project/data/parquet")
    print("[STEP] load + filter businesses")
    biz = m04.load_business_filtered(spark, base).persist(m04.StorageLevel.DISK_ONLY)
    print("[STEP] load + filter interactions")
    rvw = m04.load_interactions(spark, base, biz).persist(m04.StorageLevel.DISK_ONLY)
    if rvw.rdd.isEmpty():
        raise RuntimeError("No interactions after filtering.")

    cluster_map = load_cluster_map(spark, profile_csv).persist(m04.StorageLevel.DISK_ONLY)
    print(f"[COUNT] cluster_map_businesses={cluster_map.count()}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    for min_train in m04.MIN_TRAIN_REVIEWS_BUCKETS:
        min_user_reviews = min_train + m04.MIN_USER_REVIEWS_OFFSET
        print(f"[BUCKET] min_train={min_train} min_user={min_user_reviews}")
        rvw_filtered, train, valid, test = m04.leave_two_out(rvw, min_user_reviews)
        rvw_filtered = rvw_filtered.persist(m04.StorageLevel.DISK_ONLY)
        train = train.persist(m04.StorageLevel.DISK_ONLY)
        valid = valid.persist(m04.StorageLevel.DISK_ONLY)
        test = test.persist(m04.StorageLevel.DISK_ONLY)

        n_users = rvw_filtered.select("user_id").distinct().count()
        n_items = rvw_filtered.select("business_id").distinct().count()
        n_train = train.count()
        n_valid = valid.count()
        n_test = test.count()

        for split_name, split_df in [("valid", valid), ("test", test)]:
            pop_recall, pop_ndcg = m04.evaluate_popular_baseline(train, split_df, m04.TOP_K)
            cat_recall, cat_ndcg = m04.evaluate_category_popular_baseline(train, split_df, biz, m04.TOP_K)
            clu_recall, clu_ndcg = evaluate_cluster_popular_baseline(
                train=train,
                test=split_df,
                cluster_map=cluster_map,
                top_k=m04.TOP_K,
                compute_recall_ndcg_fn=m04.compute_recall_ndcg,
                popular_fallback_fn=m04.evaluate_popular_baseline,
            )
            print(
                f"[{split_name}] Popular={pop_ndcg:.4f} Category={cat_ndcg:.4f} Cluster={clu_ndcg:.4f}"
            )
            rows.extend(
                [
                    {
                        "run_id": run_id,
                        "bucket_min_train_reviews": min_train,
                        "min_user_reviews": min_user_reviews,
                        "model": "Popular",
                        "split": split_name,
                        "recall_at_k": pop_recall,
                        "ndcg_at_k": pop_ndcg,
                        "top_k": m04.TOP_K,
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_train": n_train,
                        "n_valid": n_valid,
                        "n_test": n_test,
                        "profile_csv": str(profile_csv),
                    },
                    {
                        "run_id": run_id,
                        "bucket_min_train_reviews": min_train,
                        "min_user_reviews": min_user_reviews,
                        "model": "Category Popular",
                        "split": split_name,
                        "recall_at_k": cat_recall,
                        "ndcg_at_k": cat_ndcg,
                        "top_k": m04.TOP_K,
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_train": n_train,
                        "n_valid": n_valid,
                        "n_test": n_test,
                        "profile_csv": str(profile_csv),
                    },
                    {
                        "run_id": run_id,
                        "bucket_min_train_reviews": min_train,
                        "min_user_reviews": min_user_reviews,
                        "model": "Cluster Popular",
                        "split": split_name,
                        "recall_at_k": clu_recall,
                        "ndcg_at_k": clu_ndcg,
                        "top_k": m04.TOP_K,
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_train": n_train,
                        "n_valid": n_valid,
                        "n_test": n_test,
                        "profile_csv": str(profile_csv),
                    },
                ]
            )

        rvw_filtered.unpersist()
        train.unpersist()
        valid.unpersist()
        test.unpersist()

    rvw.unpersist()
    biz.unpersist()
    cluster_map.unpersist()
    spark.stop()

    out_df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
