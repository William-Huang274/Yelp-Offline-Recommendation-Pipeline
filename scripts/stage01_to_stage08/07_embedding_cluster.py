import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window


TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True

QUIET_LOGS = True
RANDOM_SEED = 42

# Runtime profile: "sample" (quick sanity run) or "full"
RUN_PROFILE = "sample"

# Embedding model switch
USE_BGE_M3 = True
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_LOCAL_MODEL_PATH = ""  # Example: r"D:/models/bge-m3"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOW_MODEL_FALLBACK = True

# Auto batch policy by model/device
AUTO_BATCH = True
MANUAL_BATCH_SIZE = 64
BATCH_SIZE_BGE_GPU = 8
BATCH_SIZE_BGE_CPU = 4
BATCH_SIZE_MINILM_GPU = 64
BATCH_SIZE_MINILM_CPU = 32
NORMALIZE_EMBEDDINGS = True

REUSE_EMBEDDINGS = True

BASE_DIR = Path(r"D:/5006 BDA project/data/parquet")
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/07_embedding_cluster")
RUN_TAG = ""  # optional suffix in output dir name

FULL_CONFIG = {
    "sample_fraction": 0.0,
    "min_reviews_per_business": 3,
    "max_reviews_per_business": 20,
    "max_businesses": 0,  # 0 means no limit
    "cluster_k": 20,
    "top_terms_per_cluster": 15,
    "tfidf_max_features": 20000,
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.8,
    "tfidf_ngram_range": (1, 2),
}

SAMPLE_CONFIG = {
    "sample_fraction": 0.08,
    "min_reviews_per_business": 3,
    "max_reviews_per_business": 10,
    "max_businesses": 600,
    "cluster_k": 8,
    "top_terms_per_cluster": 10,
    "tfidf_max_features": 8000,
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.9,
    "tfidf_ngram_range": (1, 2),
}


def build_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder
        .appName("yelp-la-embed-cluster")
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


def load_business_filtered(spark: SparkSession) -> DataFrame:
    business = (
        spark.read.parquet((BASE_DIR / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "name", "state", "city", "categories", "stars", "review_count")
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
    return biz.select(
        "business_id",
        "name",
        "city",
        "categories",
        "stars",
        "review_count",
    ).distinct()


def load_reviews(spark: SparkSession, biz: DataFrame, sample_fraction: float) -> DataFrame:
    review = (
        spark.read.parquet((BASE_DIR / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "business_id", "text", "date", "stars")
    )
    rvw = review.join(biz.select("business_id"), on="business_id", how="inner")
    rvw = rvw.filter(F.col("text").isNotNull() & (F.length(F.col("text")) > 0))
    rvw = rvw.withColumn("ts", F.to_timestamp("date")).filter(F.col("ts").isNotNull())
    if sample_fraction and sample_fraction > 0:
        rvw = rvw.sample(False, sample_fraction, RANDOM_SEED)
    return rvw


def build_business_texts(
    reviews: DataFrame, min_reviews_per_business: int, max_reviews_per_business: int
) -> DataFrame:
    w = Window.partitionBy("business_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = reviews.withColumn("rn", F.row_number().over(w))
    limited = ranked.filter(F.col("rn") <= max_reviews_per_business)
    agg = (
        limited.groupBy("business_id")
        .agg(
            F.count("*").alias("n_reviews"),
            F.concat_ws(" ", F.collect_list("text")).alias("text"),
            F.avg("stars").alias("avg_review_stars"),
        )
        .filter(F.col("n_reviews") >= min_reviews_per_business)
    )
    return agg


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def resolve_profile() -> dict:
    if RUN_PROFILE.lower() == "sample":
        cfg = SAMPLE_CONFIG.copy()
    else:
        cfg = FULL_CONFIG.copy()
    cfg["profile"] = RUN_PROFILE.lower()
    return cfg


def choose_model_name() -> str:
    if USE_BGE_M3:
        local = BGE_LOCAL_MODEL_PATH.strip()
        return local if local else BGE_MODEL_NAME
    return MINILM_MODEL_NAME


def pick_batch_size(model_name: str, device: str) -> int:
    if not AUTO_BATCH:
        return MANUAL_BATCH_SIZE
    name = model_name.lower()
    if "bge-m3" in name:
        return BATCH_SIZE_BGE_GPU if device == "cuda" else BATCH_SIZE_BGE_CPU
    return BATCH_SIZE_MINILM_GPU if device == "cuda" else BATCH_SIZE_MINILM_CPU


def load_model(primary_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(primary_name, device=device)
        return model, primary_name
    except Exception as exc:
        if not (ALLOW_MODEL_FALLBACK and USE_BGE_M3):
            raise
        print(f"[WARN] failed to load {primary_name}: {exc}")
        print(f"[WARN] fallback to {MINILM_MODEL_NAME}")
        model = SentenceTransformer(MINILM_MODEL_NAME, device=device)
        return model, MINILM_MODEL_NAME


def compute_embeddings(texts: list[str], model_name: str) -> tuple[np.ndarray, str, int]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, actual_model_name = load_model(model_name, device)
    batch_size = max(1, pick_batch_size(actual_model_name, device))

    print(f"[INFO] embedding device={device}")
    print(f"[INFO] embedding model={actual_model_name}")
    print(f"[INFO] embedding batch_size={batch_size}")

    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            return np.asarray(embeddings, dtype=np.float32), actual_model_name, batch_size
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"[WARN] OOM detected, retry with batch_size={batch_size}")
            if device == "cuda":
                torch.cuda.empty_cache()


def build_output_paths(run_id: str, profile: str) -> dict:
    suffix = RUN_TAG.strip()
    run_name = f"{run_id}_{profile}" if not suffix else f"{run_id}_{profile}_{suffix}"
    run_dir = OUTPUT_ROOT / run_name
    cache_dir = OUTPUT_ROOT / "_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "assignments_csv": run_dir / "biz_cluster_assignments.csv",
        "cluster_keywords_csv": run_dir / "biz_cluster_keywords.csv",
        "cluster_summary_csv": run_dir / "biz_cluster_summary.csv",
        "cluster_examples_csv": run_dir / "biz_cluster_examples.csv",
        "run_meta_csv": run_dir / "run_meta.csv",
    }


def make_cache_file(cache_dir: Path, model_name: str, cfg: dict) -> Path:
    model_key = sanitize_name(model_name.lower())
    data_key = (
        f"{TARGET_STATE}_r{int(REQUIRE_RESTAURANTS)}_f{int(REQUIRE_FOOD)}_"
        f"s{cfg['sample_fraction']}_mn{cfg['min_reviews_per_business']}_"
        f"mx{cfg['max_reviews_per_business']}_mb{cfg['max_businesses']}"
    )
    return cache_dir / f"emb_{model_key}_{sanitize_name(data_key)}.npz"


def try_load_cached_embeddings(cache_file: Path, business_ids: list[str]) -> np.ndarray | None:
    if not cache_file.exists():
        return None
    data = np.load(cache_file, allow_pickle=False)
    cached_ids = data["business_id"].astype(str).tolist()
    if cached_ids != business_ids:
        print("[WARN] embedding cache business_id mismatch, recomputing")
        return None
    print(f"[INFO] reuse embeddings from {cache_file}")
    return data["embeddings"]


def save_cache_embeddings(cache_file: Path, business_ids: list[str], embeddings: np.ndarray) -> None:
    np.savez(
        cache_file,
        business_id=np.asarray(business_ids),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = resolve_profile()
    paths = build_output_paths(run_id, cfg["profile"])
    model_name = choose_model_name()

    print(f"[CONFIG] profile={cfg['profile']}")
    print(f"[CONFIG] use_bge_m3={USE_BGE_M3}, model={model_name}")
    print(f"[CONFIG] output_dir={paths['run_dir']}")

    spark = build_spark()
    set_log_level(spark)

    print("[STEP] load + filter businesses")
    biz = load_business_filtered(spark).persist(StorageLevel.DISK_ONLY)
    if biz.limit(1).count() == 0:
        print("[ERROR] no businesses after filtering")
        spark.stop()
        return

    print("[STEP] load + filter reviews")
    reviews = load_reviews(spark, biz, cfg["sample_fraction"]).persist(StorageLevel.DISK_ONLY)
    if reviews.limit(1).count() == 0:
        print("[ERROR] no reviews after filtering")
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] build business texts")
    biz_text = build_business_texts(
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

    print(f"[COUNT] businesses={len(pdf)}")
    if len(pdf) == 0:
        print("[ERROR] no businesses collected")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return

    cache_file = make_cache_file(paths["cache_dir"], model_name, cfg)
    business_ids = pdf["business_id"].astype(str).tolist()
    embeddings = None
    actual_model_name = model_name
    actual_batch_size = MANUAL_BATCH_SIZE
    if REUSE_EMBEDDINGS:
        embeddings = try_load_cached_embeddings(cache_file, business_ids)

    if embeddings is None:
        print("[STEP] compute embeddings")
        embeddings, actual_model_name, actual_batch_size = compute_embeddings(
            pdf["text"].tolist(), model_name
        )
        save_cache_embeddings(cache_file, business_ids, embeddings)
        print(f"[INFO] saved embeddings cache to {cache_file}")

    print("[STEP] kmeans clustering")
    from sklearn.cluster import KMeans

    cluster_k = min(int(cfg["cluster_k"]), len(pdf))
    if cluster_k < 2:
        print("[ERROR] not enough businesses for clustering")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return

    kmeans = KMeans(n_clusters=cluster_k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    pdf["cluster"] = labels

    print("[STEP] tf-idf keywords per cluster")
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=int(cfg["tfidf_max_features"]),
        min_df=int(cfg["tfidf_min_df"]),
        max_df=float(cfg["tfidf_max_df"]),
        ngram_range=tuple(cfg["tfidf_ngram_range"]),
        stop_words="english",
    )
    X = vectorizer.fit_transform(pdf["text"].tolist())
    terms = vectorizer.get_feature_names_out()

    keyword_rows = []
    summary_rows = []
    example_rows = []
    top_terms = int(cfg["top_terms_per_cluster"])
    for c in range(cluster_k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        cluster_df = pdf.iloc[idx]
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_terms]

        for rank, term_idx in enumerate(top_idx, start=1):
            score = float(mean_tfidf[term_idx])
            if score <= 0:
                continue
            keyword_rows.append(
                {
                    "cluster": c,
                    "rank": rank,
                    "term": terms[term_idx],
                    "score": score,
                    "n_businesses": int(idx.size),
                }
            )

        summary_rows.append(
            {
                "cluster": c,
                "n_businesses": int(idx.size),
                "avg_review_stars": float(cluster_df["avg_review_stars"].mean()),
                "avg_business_stars": float(cluster_df["stars"].mean()),
                "avg_n_reviews": float(cluster_df["n_reviews"].mean()),
            }
        )

        examples = cluster_df.sort_values("n_reviews", ascending=False).head(5)
        for _, row in examples.iterrows():
            example_rows.append(
                {
                    "cluster": c,
                    "business_id": row["business_id"],
                    "name": row["name"],
                    "city": row["city"],
                    "categories": row["categories"],
                    "n_reviews": int(row["n_reviews"]),
                    "avg_review_stars": float(row["avg_review_stars"]),
                }
            )

    pdf.drop(columns=["text"]).to_csv(paths["assignments_csv"], index=False)
    pd.DataFrame(keyword_rows).to_csv(paths["cluster_keywords_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["cluster_summary_csv"], index=False)
    pd.DataFrame(example_rows).to_csv(paths["cluster_examples_csv"], index=False)
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "profile": cfg["profile"],
                "model_requested": model_name,
                "model_used": actual_model_name,
                "batch_size_used": actual_batch_size,
                "cluster_k_used": cluster_k,
                "n_businesses": len(pdf),
                "cache_file": str(cache_file),
            }
        ]
    ).to_csv(paths["run_meta_csv"], index=False)

    print(f"[INFO] wrote {paths['assignments_csv']}")
    print(f"[INFO] wrote {paths['cluster_keywords_csv']}")
    print(f"[INFO] wrote {paths['cluster_summary_csv']}")
    print(f"[INFO] wrote {paths['cluster_examples_csv']}")
    print(f"[INFO] wrote {paths['run_meta_csv']}")

    reviews.unpersist()
    biz_text.unpersist()
    biz.unpersist()
    spark.stop()
    print(f"[INFO] run_id={run_id}")


if __name__ == "__main__":
    main()
