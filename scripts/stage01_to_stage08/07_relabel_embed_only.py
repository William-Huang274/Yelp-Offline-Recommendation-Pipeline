from datetime import datetime

from pyspark import StorageLevel
from pyspark.sql import functions as F

from stage07_core import configure_stage07, load_stage07_module


RUN_PROFILE_OVERRIDE = "sample"  # optional: "sample" | "full"
RUN_TAG_SUFFIX = "relabel_embed_only"
USE_BGE_M3_OVERRIDE = "true"  # optional: "true" | "false"


def main() -> None:
    m07 = load_stage07_module()
    configure_stage07(
        m07,
        run_profile_override=RUN_PROFILE_OVERRIDE,
        use_bge_m3_override=USE_BGE_M3_OVERRIDE,
    )

    orig_tag = m07.RUN_TAG
    m07.RUN_TAG = f"{orig_tag}_{RUN_TAG_SUFFIX}" if orig_tag else RUN_TAG_SUFFIX

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = m07.resolve_profile()
    paths = m07.build_output_paths(run_id, cfg["profile"])
    print(f"[CONFIG] profile={cfg['profile']}")
    print(f"[CONFIG] use_bge_m3={m07.USE_BGE_M3}")
    print(f"[CONFIG] relabel_embed_use_cache={getattr(m07, 'RELABEL_EMBED_RECALL_USE_CACHE', False)}")
    print(f"[CONFIG] output_dir={paths['run_dir']}")

    spark = m07.build_spark()
    m07.set_log_level(spark)

    print("[STEP] load + filter businesses")
    biz = m07.load_business_filtered(spark).persist(StorageLevel.DISK_ONLY)
    if biz.limit(1).count() == 0:
        print("[ERROR] no businesses after filtering")
        spark.stop()
        return

    print("[STEP] load + filter reviews")
    reviews = m07.load_reviews(spark, biz, cfg["sample_fraction"]).persist(StorageLevel.DISK_ONLY)
    if reviews.limit(1).count() == 0:
        print("[ERROR] no reviews after filtering")
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] build business texts")
    biz_text = m07.build_business_texts(
        reviews,
        cfg["min_reviews_per_business"],
        cfg["max_reviews_per_business"],
    ).persist(StorageLevel.DISK_ONLY)
    if biz_text.limit(1).count() == 0:
        print("[ERROR] no businesses after review aggregation")
        reviews.unpersist()
        biz.unpersist()
        spark.stop()
        return

    biz_joined = biz_text.join(biz, on="business_id", how="left")
    if cfg["max_businesses"] and cfg["max_businesses"] > 0:
        biz_joined = (
            biz_joined.orderBy(F.desc("n_reviews"), F.desc("review_count"))
            .limit(cfg["max_businesses"])
        )

    pdf = biz_joined.select(
        "business_id",
        "name",
        "city",
        "categories",
        "stars",
        "review_count",
        "avg_review_stars",
        "n_reviews",
        "text",
    ).toPandas()
    print(f"[COUNT] businesses_for_embed_cache={len(pdf)}")
    if len(pdf) == 0:
        print("[ERROR] no businesses collected")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] precompute relabel embedding cache")
    prep_stats = m07.precompute_relabel_embedding_cache(
        pdf,
        profile=cfg["profile"],
        cache_dir=paths["cache_dir"],
    )

    run_meta = {
        "run_id": run_id,
        "profile": cfg["profile"],
        "stage_mode": "relabel_embed_only",
        "n_businesses": int(len(pdf)),
        "use_bge_m3": bool(m07.USE_BGE_M3),
        "embed_cache_use": bool(getattr(m07, "RELABEL_EMBED_RECALL_USE_CACHE", False)),
        "embed_cache_file": str(prep_stats.get("cache_file", "")),
        "embed_rows": int(prep_stats.get("n_rows", 0)),
        "embed_unique_keys": int(prep_stats.get("n_unique_keys", 0)),
        "embed_missing_computed": int(prep_stats.get("n_missing_computed", 0)),
        "embed_cache_size": int(prep_stats.get("cache_size", 0)),
        "embed_model_requested": str(prep_stats.get("model_requested", "")),
        "embed_model_used": str(prep_stats.get("model_used", "")),
        "embed_batch_size_used": int(prep_stats.get("batch_size_used", 0)),
    }
    m07.pd.DataFrame([run_meta]).to_csv(paths["run_meta_csv"], index=False)
    print(f"[INFO] wrote {paths['run_meta_csv']}")

    reviews.unpersist()
    biz_text.unpersist()
    biz.unpersist()
    spark.stop()
    print(f"[INFO] run_id={run_id}")


if __name__ == "__main__":
    main()

