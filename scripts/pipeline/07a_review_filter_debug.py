import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.window import Window

try:
    from pipeline.review_text_filter import (
        build_text_views_from_reviews,
        select_reviews_for_sentence_stage,
    )
except ImportError:
    from review_text_filter import (  # type: ignore
        build_text_views_from_reviews,
        select_reviews_for_sentence_stage,
    )


RUN_PROFILE = "full"  # "sample" | "full"
MAX_BUSINESSES = 0
MAX_REVIEWS_PER_BIZ = 10
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/07_review_filter_debug")
RUN_TAG = "review_filter_debug"


def load_stage07_module():
    try:
        from stage07_core import load_stage07_module as _loader
    except ImportError:
        scripts_dir = str(Path(__file__).resolve().parents[1])
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from stage07_core import load_stage07_module as _loader
    return _loader()


def extract_reviews(review_items: Any) -> list[str]:
    out: list[tuple[int, str]] = []
    if not isinstance(review_items, list):
        return []
    for it in review_items:
        rn = 0
        text = ""
        if hasattr(it, "asDict"):
            d = it.asDict(recursive=True)
            rn = int(d.get("rn", 0) or 0)
            text = str(d.get("text", "") or "")
        elif isinstance(it, dict):
            rn = int(it.get("rn", 0) or 0)
            text = str(it.get("text", "") or "")
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            rn = int(it[0] or 0)
            text = str(it[1] or "")
        if text.strip():
            out.append((rn, text))
    out.sort(key=lambda x: x[0])
    return [t for _, t in out]


def main() -> None:
    m07 = load_stage07_module()
    m07.RUN_PROFILE = RUN_PROFILE
    cfg = m07.resolve_profile()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{cfg['profile']}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "review_filter_debug.csv"
    meta_csv = out_dir / "run_meta.csv"

    spark = m07.build_spark()
    m07.set_log_level(spark)

    print(f"[CONFIG] profile={cfg['profile']}")
    print(f"[CONFIG] max_businesses={MAX_BUSINESSES}, max_reviews_per_biz={MAX_REVIEWS_PER_BIZ}")

    biz = m07.load_business_filtered(spark).persist(StorageLevel.DISK_ONLY)
    reviews = m07.load_reviews(spark, biz, cfg["sample_fraction"]).persist(StorageLevel.DISK_ONLY)

    w = Window.partitionBy("business_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = reviews.withColumn("rn", F.row_number().over(w))
    limited = ranked.filter(F.col("rn") <= int(MAX_REVIEWS_PER_BIZ))

    agg = (
        limited.groupBy("business_id")
        .agg(
            F.count("*").alias("n_reviews"),
            F.collect_list(F.struct("rn", "text")).alias("review_items"),
            F.avg("stars").alias("avg_review_stars"),
        )
        .filter(F.col("n_reviews") >= int(cfg["min_reviews_per_business"]))
    )

    joined = agg.join(
        biz.select("business_id", "name", "city", "categories", "stars", "review_count"),
        on="business_id",
        how="left",
    )
    if int(MAX_BUSINESSES) > 0:
        joined = joined.orderBy(F.desc("n_reviews"), F.desc("review_count")).limit(int(MAX_BUSINESSES))

    pdf = joined.toPandas()
    print(f"[COUNT] businesses_for_debug={len(pdf)}")
    if pdf.empty:
        print("[WARN] no rows for debug")
        reviews.unpersist()
        biz.unpersist()
        spark.stop()
        return

    rows: list[dict[str, Any]] = []
    for _, row in pdf.iterrows():
        reviews_list = extract_reviews(row.get("review_items"))
        pick = select_reviews_for_sentence_stage(
            reviews=reviews_list,
            categories=str(row.get("categories", "") or ""),
            name=str(row.get("name", "") or ""),
            target_reviews=6,
        )
        text_view = build_text_views_from_reviews(
            reviews=reviews_list,
            categories=str(row.get("categories", "") or ""),
            name=str(row.get("name", "") or ""),
            target_reviews=6,
            relabel_sentences=8,
            embed_sentences=12,
            relabel_max_chars=1500,
            embed_max_chars=2500,
        )
        selected_preview = " || ".join(pick["selected_reviews"][:3])[:1500]
        rows.append(
            {
                "business_id": row.get("business_id", ""),
                "name": row.get("name", ""),
                "city": row.get("city", ""),
                "categories": row.get("categories", ""),
                "stars": row.get("stars", ""),
                "review_count": row.get("review_count", ""),
                "n_reviews_considered": len(reviews_list),
                "review_filter_total_reviews": text_view.get("review_filter_total_reviews", 0),
                "review_filter_kept_reviews": text_view.get("review_filter_kept_reviews", 0),
                "review_filter_dropped_reviews": text_view.get("review_filter_dropped_reviews", 0),
                "review_filter_mean_info_score": text_view.get("review_filter_mean_info_score", 0.0),
                "review_filter_mean_spam_score": text_view.get("review_filter_mean_spam_score", 0.0),
                "review_filter_selected_keywords": text_view.get("review_filter_selected_keywords", ""),
                "text_filter_total_sentences": text_view.get("text_filter_total_sentences", 0),
                "text_filter_kept_sentences": text_view.get("text_filter_kept_sentences", 0),
                "text_filter_dropped_sentences": text_view.get("text_filter_dropped_sentences", 0),
                "text_filter_mean_info_score": text_view.get("text_filter_mean_info_score", 0.0),
                "text_filter_mean_spam_score": text_view.get("text_filter_mean_spam_score", 0.0),
                "text_filter_selected_keywords": text_view.get("text_filter_selected_keywords", ""),
                "selected_reviews_preview": selected_preview,
                "text_for_relabel": text_view.get("text_for_relabel", ""),
                "text_for_embed": text_view.get("text_for_embed", ""),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "run_id": run_id,
        "profile": cfg["profile"],
        "max_businesses": int(MAX_BUSINESSES),
        "max_reviews_per_biz": int(MAX_REVIEWS_PER_BIZ),
        "rows": int(len(out_df)),
        "output_csv": str(out_csv),
    }
    pd.DataFrame([meta]).to_csv(meta_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote {out_csv}")
    print(f"[INFO] wrote {meta_csv}")

    reviews.unpersist()
    biz.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
