import json
import os
import hashlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window
from pipeline.project_paths import env_or_project_path

# Keep sentence-transformers on PyTorch path in mixed TensorFlow/Keras setups.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True

QUIET_LOGS = True
RANDOM_SEED = 42

# Runtime profile: "sample" (quick sanity run) or "full"
RUN_PROFILE = "full"

# Embedding model switch
USE_BGE_M3 = False
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_LOCAL_MODEL_PATH = r"D:/hf_cache/hub/models--BAAI--bge-m3"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOW_MODEL_FALLBACK = True
REQUIRE_LOCAL_BGE_IF_SET = True

# Auto batch policy by model/device
AUTO_BATCH = True
MANUAL_BATCH_SIZE = 64
BATCH_SIZE_BGE_GPU = 8
BATCH_SIZE_BGE_CPU = 4
BATCH_SIZE_MINILM_GPU = 64
BATCH_SIZE_MINILM_CPU = 32
NORMALIZE_EMBEDDINGS = True
REUSE_EMBEDDINGS = True

# Labeling config
LABEL_CONFIG_DIR = env_or_project_path("LABEL_CONFIG_DIR", "config/labeling/food_service/v1")
RELABEL_USE_LLM = True
RELABEL_MAX_LLM_CALLS_SAMPLE = 230
RELABEL_MAX_LLM_CALLS_FULL = 900
RELABEL_LLM_WORKERS_SAMPLE = 2
RELABEL_LLM_WORKERS_FULL = 2
RELABEL_LLM_RETRY_SERIAL_ON_FAILURE = True
RELABEL_PROGRESS_EVERY_N = 20
RELABEL_FLUSH_EVERY_N = 50
RELABEL_WRITE_IN_PROGRESS = True
RELABEL_REVIEW_CHARS = 700
KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER = True
CLUSTER_INPUT_MODE = "strict"  # "strict" | "food_service"
CLUSTER_STRICT_MIN_CONFIDENCE = 3.0
CLUSTER_STRICT_EXCLUDE_REVIEW_QUEUE = True
CLUSTER_STRICT_EXCLUDE_GENERAL = True

# Embedding-assisted relabel recall (rule -> embedding recall -> llm)
RELABEL_USE_EMBED_RECALL = True
RELABEL_EXPERIMENT_STRATEGY = "selective"  # "selective" | "dense"
RELABEL_EMBED_RECALL_CONFIDENCE_MAX = 3
RELABEL_EMBED_RECALL_MAX_CANDIDATES_SAMPLE = 250
RELABEL_EMBED_RECALL_MAX_CANDIDATES_FULL = 1600
RELABEL_EMBED_RECALL_MAX_CANDIDATES_DENSE_SAMPLE = 1200
RELABEL_EMBED_RECALL_MAX_CANDIDATES_DENSE_FULL = 6000
RELABEL_EMBED_RECALL_DENSE_INCLUDE_NON_FOOD = False
RELABEL_EMBED_RECALL_REVIEW_CHARS = 600
RELABEL_EMBED_RECALL_USE_CACHE = True
RELABEL_EMBED_RECALL_PRECOMPUTE_ALL = False
RELABEL_EMBED_RECALL_MIN_SIM = 0.50
RELABEL_EMBED_RECALL_MIN_MARGIN = 0.03
RELABEL_EMBED_RECALL_STRONG_SIM = 0.58
RELABEL_EMBED_RECALL_MIN_SIM_UNCERTAIN = 0.54
RELABEL_EMBED_RECALL_SKIP_WEAK_SCENE = True
RELABEL_EMBED_RECALL_STORE_TOPK = 6

# L2 scoring behavior
RULE_USE_REVIEW_SIGNAL = True
RULE_REVIEW_SIGNAL_CHARS = 600
L2_WEIGHT_CATEGORIES = 1.4
L2_WEIGHT_NAME = 1.6
L2_WEIGHT_REVIEW = 0.75
L2_AMBIGUOUS_GAP_MAX = 0.45
L2_LOW_SIGNAL_MIN_SCORE = 1.6
L2_FORCE_GENERAL_MAX_SCORE = 1.0
L2_GENERAL_IS_FALLBACK_ONLY = True

# LLM trigger expansion for relabel
LLM_TRIGGER_GENERAL_FALLBACK = True
LLM_TRIGGER_L2_AMBIGUOUS = True
LLM_TRIGGER_L2_LOW_SIGNAL = True
LLM_TRIGGER_L2_AMBIGUOUS_MAX_GAP = 0.35
LLM_TRIGGER_L2_AMBIGUOUS_MAX_TOP1_SCORE = 2.5
LLM_REVIEW_CONFIDENCE_MAX = 2
LLM_REVIEW_TRIGGER_CONFIDENCE_MAX = 3
LLM_CANDIDATE_TOPK_HIGH_CONF = 2
LLM_CANDIDATE_TOPK_MID_CONF = 4
LLM_CANDIDATE_TOPK_LOW_CONF = 6
LLM_CANDIDATE_MIN_SIM = 0.45
LLM_CANDIDATE_HARD_LIMIT = 8
LLM_CANDIDATE_BACKFILL_ALL = False

# Conservative rescue for L1 uncertain cases with clearly stronger service evidence.
UNCERTAIN_RESCUE_ENABLE = True
UNCERTAIN_RESCUE_MIN_SERVICE_HITS = 2
UNCERTAIN_RESCUE_MIN_SERVICE_OVER_RETAIL = 1
UNCERTAIN_RESCUE_REQUIRE_RESTAURANT_TOKEN = True

# LLM review action mode
LLM_ACTION_MODE = True
LLM_ACTIONS = {"KEEP", "MODIFY", "ADD", "REJECT"}

# Decision constraints for broad scene labels
BAR_SCENE_CHILD_LABELS = ["cocktail_bar", "wine_bar", "pub_gastropub"]
BAR_SCENE_STRONG_TERMS = {
    "cocktail bar",
    "cocktail bars",
    "wine bar",
    "wine bars",
    "sports bars",
    "pub",
    "pubs",
    "gastropub",
    "gastropubs",
    "dive bars",
    "lounges",
    "night clubs",
    "nightclub",
    "beer bar",
}
BAR_SCENE_MIN_TERMS = 2
BAR_SCENE_REQUIRE_STRONG_TERM = True
BAR_SCENE_CHILD_PROMOTE_MIN_SCORE = 2.0
BAR_SCENE_CHILD_PROMOTE_GAP = 2.0

SCENE_LABELS = {"bar_nightlife", "cocktail_bar", "wine_bar", "pub_gastropub"}
CUISINE_LABELS = {
    "bbq",
    "breakfast_brunch",
    "burgers",
    "cajun_creole",
    "chinese",
    "italian",
    "japanese_sushi",
    "mexican",
    "pizza",
    "seafood",
    "southern",
    "steakhouse",
    "thai",
    "vietnamese",
    "french",
    "mediterranean",
    "middle_eastern",
    "american_new",
    "american_traditional",
    "asian_fusion",
    "indian",
    "pakistani_halal",
    "vegetarian_vegan",
    "salad_healthy",
}
SCENE_VS_CUISINE_MIN_SCORE = 2.0
SCENE_VS_CUISINE_OVERRIDE_GAP = 0.75

# Pairwise conflict resolver for high-frequency ambiguous L2 pairs.
PAIRWISE_CONFLICT_MAX_GAP = 0.9
PAIRWISE_CONFLICT_BOOST = 0.6
PAIRWISE_CONFLICT_RULES: list[dict[str, Any]] = [
    {
        "left": "breakfast_brunch",
        "right": "coffee_tea",
        "left_terms": {"breakfast", "brunch"},
        "right_terms": {"coffee", "tea", "espresso"},
        "left_text_terms": {"breakfast", "brunch"},
        "right_text_terms": {"coffee", "tea", "espresso", "cafe", "cafes", "coffeehouse"},
    },
    {
        "left": "breakfast_brunch",
        "right": "cafe",
        "left_terms": {"breakfast", "brunch"},
        "right_terms": {"cafe", "cafes", "coffeehouse"},
        "left_text_terms": {"breakfast", "brunch"},
        "right_text_terms": {"cafe", "cafes", "coffeehouse", "coffee", "tea", "espresso"},
    },
    {
        "left": "cajun_creole",
        "right": "seafood",
        "left_terms": {"cajun", "creole", "gumbo", "jambalaya", "etouffee", "po boy", "po' boy"},
        "right_terms": {"seafood", "oyster", "crawfish", "shrimp"},
        "left_text_terms": {"cajun", "creole", "gumbo", "jambalaya", "etouffee", "po boy", "po' boy"},
        "right_text_terms": {"seafood", "oyster", "crawfish", "shrimp"},
    },
    {
        "left": "american_new",
        "right": "american_traditional",
        "left_terms": {"american (new)", "new american", "american new"},
        "right_terms": {"american (traditional)", "american traditional", "diners", "diner"},
        "left_text_terms": {"new american", "american new"},
        "right_text_terms": {"american traditional", "diner", "diners"},
    },
    {
        "left": "japanese_sushi",
        "right": "asian_fusion",
        "left_terms": {"japanese", "sushi", "ramen", "izakaya"},
        "right_terms": {"asian fusion", "pan asian", "fusion"},
        "left_text_terms": {"japanese", "sushi", "ramen", "izakaya"},
        "right_text_terms": {"asian fusion", "pan asian", "fusion"},
    },
]

# Cluster features: semantic embedding + L2 profile
USE_HYBRID_CLUSTER_FEATURES = True
HYBRID_L2_FEATURE_WEIGHT = 0.45
HYBRID_USE_L2_TOP2 = True
HYBRID_L2_TOP2_WEIGHT = 0.45
HYBRID_IGNORE_GENERAL_L2 = True

# Optional: run cluster-level labeling script after this pipeline.
RUN_08_AFTER_CLUSTER = False

# Output tables
WRITE_LAYERED_LABEL_OUTPUTS = True

# Local Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_TIMEOUT_SEC = 180
OLLAMA_NUM_CTX = 4096
OLLAMA_NUM_PREDICT = 320
OLLAMA_THINKING = False
OLLAMA_JSON_RETRY_ON_PARSE_FAIL = True
OLLAMA_JSON_RETRY_NUM_PREDICT = 192
LLM_PROMPT_REVIEW_CHARS = 520
LLM_REASON_SHORT_MAX_CHARS = 120

BASE_DIR = env_or_project_path("PARQUET_BASE_DIR", "data/parquet")
OUTPUT_ROOT = env_or_project_path("OUTPUT_07_ROOT_DIR", "data/output/07_embedding_cluster")
RUN_TAG = "relabel_minilm"
SPARK_LOCAL_DIR = env_or_project_path("SPARK_LOCAL_DIR", "data/spark-tmp")

FULL_CONFIG = {
    "sample_fraction": 0.0,
    "min_reviews_per_business": 3,
    "max_reviews_per_business": 20,
    "max_businesses": 0,
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

DEFAULT_ALLOWED_L1 = {"food_service", "food_retail", "non_food", "uncertain"}

DEFAULT_ALLOWED_L2_SERVICE = {
    "restaurants_general",
    "coffee_tea",
    "cafe",
    "bakery",
    "dessert",
    "ice_cream_frozen_yogurt",
    "juice_smoothies",
    "food_trucks",
    "fast_food",
    "pizza",
    "sandwiches",
    "burgers",
    "seafood",
    "cajun_creole",
    "mexican",
    "chinese",
    "japanese_sushi",
    "vietnamese",
    "italian",
    "bbq",
    "breakfast_brunch",
    "bar_nightlife",
    "cocktail_bar",
    "wine_bar",
    "pub_gastropub",
    "american_new",
    "american_traditional",
    "southern",
    "french",
    "steakhouse",
    "asian_fusion",
    "indian",
    "pakistani_halal",
    "thai",
    "mediterranean",
    "middle_eastern",
    "salad_healthy",
    "vegetarian_vegan",
    "other_service",
}

BASE_SERVICE_TERMS = [
    "restaurants",
    "restaurant",
    "cafe",
    "coffee",
    "tea",
    "bakery",
    "dessert",
    "pizza",
    "sushi",
    "barbecue",
    "bbq",
    "food truck",
    "breakfast",
    "brunch",
    "bars",
    "nightlife",
    "cocktail bars",
    "wine bars",
    "pubs",
    "gastropubs",
    "steakhouse",
    "southern",
    "french",
    "mediterranean",
    "middle eastern",
    "thai",
    "asian fusion",
    "indian",
    "pakistani",
    "halal",
    "vegetarian",
    "vegan",
    "fast food",
    "sandwich",
    "sandwiches",
    "deli",
    "delis",
    "burger",
    "food court",
    "food courts",
    "food stand",
    "food stands",
    "juice bar",
    "juice bars",
    "smoothie",
    "smoothies",
    "cajun",
    "creole",
    "seafood",
]

BASE_RETAIL_TERMS = [
    "drugstore",
    "drugstores",
    "pharmacy",
    "convenience store",
    "convenience stores",
    "grocery",
    "grocery store",
    "grocery stores",
    "supermarket",
    "farmers market",
    "public market",
    "food mart",
    "specialty food",
    "liquor store",
    "liquor stores",
    "walmart",
    "cvs",
    "walgreens",
    "dollar general",
    "target",
    "costco",
]

RETAIL_DOMINANT_TERMS = [
    "farmers market",
    "public market",
    "food mart",
    "convenience store",
    "convenience stores",
    "grocery",
    "grocery store",
    "grocery stores",
    "supermarket",
    "specialty food",
]

BASE_NON_FOOD_TERMS = [
    "auto repair",
    "car wash",
    "insurance",
    "lawyer",
    "dentist",
    "financial services",
]

L2_KEYWORDS = {
    "coffee_tea": ["coffee", "tea", "espresso"],
    "cafe": ["cafe", "cafes", "coffeehouse"],
    "bakery": ["bakery", "bread", "pastry", "croissant"],
    "dessert": ["dessert", "cake", "sweet"],
    "ice_cream_frozen_yogurt": ["ice cream", "frozen yogurt", "gelato"],
    "juice_smoothies": ["smoothie", "juice bar", "juice"],
    "food_trucks": ["food truck", "food trucks"],
    "fast_food": ["fast food", "drive thru", "drive-thru"],
    "pizza": ["pizza", "pizzeria"],
    "sandwiches": ["sandwich", "deli"],
    "burgers": ["burger", "burgers"],
    "seafood": ["seafood", "oyster", "crawfish", "shrimp"],
    "cajun_creole": [
        "cajun",
        "creole",
        "gumbo",
        "jambalaya",
        "etouffee",
        "po boy",
        "po' boy",
    ],
    "mexican": ["mexican", "taco", "burrito", "quesadilla"],
    "chinese": ["chinese", "dim sum", "szechuan"],
    "japanese_sushi": ["japanese", "sushi", "ramen", "izakaya"],
    "vietnamese": ["vietnamese", "pho", "banh mi", "bun", "vermicelli"],
    "italian": ["italian", "pasta", "risotto"],
    "bbq": ["barbecue", "bbq", "smoked meat"],
    "breakfast_brunch": ["breakfast", "brunch"],
    "bar_nightlife": [
        "bars",
        "nightlife",
        "dive bars",
        "sports bars",
        "lounges",
        "night clubs",
        "nightclub",
    ],
    "cocktail_bar": ["cocktail bar", "cocktail bars", "mixology"],
    "wine_bar": ["wine bar", "wine bars"],
    "pub_gastropub": ["pub", "pubs", "gastropub", "gastropubs"],
    "american_new": ["american (new)", "new american", "american new"],
    "american_traditional": ["american (traditional)", "american traditional", "diners", "diner"],
    "southern": ["southern", "soul food"],
    "french": ["french", "brasserie", "bistro", "bistros"],
    "steakhouse": ["steakhouse", "steakhouses", "steak house"],
    "asian_fusion": ["asian fusion", "pan asian", "fusion"],
    "indian": ["indian", "curry", "masala", "tandoori", "naan"],
    "pakistani_halal": ["pakistani", "halal", "biryani", "nihari", "karahi"],
    "thai": ["thai", "pad thai", "tom yum"],
    "mediterranean": ["mediterranean", "greek"],
    "middle_eastern": ["middle eastern", "lebanese", "turkish", "persian"],
    "salad_healthy": ["salad", "salads", "healthy", "health food"],
    "vegetarian_vegan": ["vegetarian", "vegan", "plant based", "plant-based"],
    "restaurants_general": ["restaurant", "restaurants", "dining"],
}


def build_spark() -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    executor_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    local_master = os.environ.get("SPARK_LOCAL_MASTER", "local[2]")
    local_dir = SPARK_LOCAL_DIR
    local_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder
        .appName("yelp-la-relabel-cluster")
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


def normalize_text(*parts: Any) -> str:
    joined = " ".join([str(p) if p is not None else "" for p in parts]).lower()
    return re.sub(r"\s+", " ", joined).strip()


def resolve_profile() -> dict:
    if RUN_PROFILE.lower() == "sample":
        cfg = SAMPLE_CONFIG.copy()
    else:
        cfg = FULL_CONFIG.copy()
    cfg["profile"] = RUN_PROFILE.lower()
    return cfg


def is_valid_local_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists()
    )


def resolve_bge_local_path(path_text: str) -> str | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        print(f"[WARN] bge local path not found: {path}")
        return None
    if is_valid_local_model_dir(path):
        return str(path)

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates: list[Path] = []
    refs_main = path / "refs" / "main"
    if refs_main.exists():
        try:
            main_snapshot = refs_main.read_text(encoding="utf-8").strip()
            if main_snapshot:
                candidates.append(snapshots_dir / main_snapshot)
        except Exception:
            pass

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates.extend(snapshots)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if is_valid_local_model_dir(candidate):
            return key
    return None


def choose_model_name() -> tuple[str, str]:
    if USE_BGE_M3:
        local = resolve_bge_local_path(BGE_LOCAL_MODEL_PATH.strip())
        if local:
            return local, "local_cache"
        if BGE_LOCAL_MODEL_PATH.strip() and REQUIRE_LOCAL_BGE_IF_SET:
            raise RuntimeError(
                "BGE local path is set but no valid local snapshot was found: "
                f"{BGE_LOCAL_MODEL_PATH}"
            )
        return BGE_MODEL_NAME, "huggingface_hub"
    return MINILM_MODEL_NAME, "huggingface_hub"


def pick_batch_size(model_name: str, device: str) -> int:
    if not AUTO_BATCH:
        return MANUAL_BATCH_SIZE
    if "bge-m3" in model_name.lower():
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
        "cluster_input_csv": run_dir / "biz_cluster_input.csv",
        "assignments_csv": run_dir / "biz_cluster_assignments.csv",
        "cluster_keywords_csv": run_dir / "biz_cluster_keywords.csv",
        "cluster_summary_csv": run_dir / "biz_cluster_summary.csv",
        "cluster_examples_csv": run_dir / "biz_cluster_examples.csv",
        "relabels_csv": run_dir / "biz_relabels.csv",
        "relabels_inprogress_csv": run_dir / "biz_relabels_inprogress.csv",
        "labels_final_csv": run_dir / "biz_labels_final.csv",
        "labels_review_csv": run_dir / "biz_labels_review_queue.csv",
        "labels_audit_csv": run_dir / "biz_labels_audit_full.csv",
        "label_stats_csv": run_dir / "biz_relabel_stats.csv",
        "run_meta_csv": run_dir / "run_meta.csv",
    }


def cache_model_key(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return sanitize_name(model_path.name.lower())
    return sanitize_name(model_name.lower())


def make_cache_file(cache_dir: Path, model_name: str, cfg: dict) -> Path:
    model_key = cache_model_key(model_name)
    data_key = (
        f"{TARGET_STATE}_r{int(REQUIRE_RESTAURANTS)}_f{int(REQUIRE_FOOD)}_"
        f"s{cfg['sample_fraction']}_mn{cfg['min_reviews_per_business']}_"
        f"mx{cfg['max_reviews_per_business']}_mb{cfg['max_businesses']}_relabel1"
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


def make_relabel_embed_cache_file(cache_dir: Path, model_name: str, profile: str) -> Path:
    model_key = cache_model_key(model_name)
    data_key = (
        f"{profile}_rc{RELABEL_EMBED_RECALL_REVIEW_CHARS}_"
        f"r{int(REQUIRE_RESTAURANTS)}_f{int(REQUIRE_FOOD)}"
    )
    return cache_dir / f"emb_relabel_{model_key}_{sanitize_name(data_key)}.npz"


def make_relabel_proto_cache_file(cache_dir: Path, model_name: str, profile: str) -> Path:
    model_key = cache_model_key(model_name)
    return cache_dir / f"emb_relabel_proto_{model_key}_{sanitize_name(profile)}.npz"


def build_relabel_embed_text(row: pd.Series | dict[str, Any]) -> str:
    if isinstance(row, pd.Series):
        getter = row.get
    else:
        getter = row.get
    return normalize_text(
        getter("name", ""),
        getter("categories", ""),
        str(getter("text", "") or "")[:RELABEL_EMBED_RECALL_REVIEW_CHARS],
    )


def relabel_embed_cache_key(business_id: Any, text: str) -> str:
    bid = sanitize_name(str(business_id or "")) or "_"
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{bid}|{digest}"


def load_relabel_embed_cache(cache_file: Path) -> dict[str, np.ndarray]:
    if not cache_file.exists():
        return {}
    try:
        data = np.load(cache_file, allow_pickle=False)
        keys = data["cache_key"].astype(str).tolist()
        emb = np.asarray(data["embeddings"], dtype=np.float32)
    except Exception as exc:
        print(f"[WARN] failed to load relabel embed cache: {cache_file} ({exc})")
        return {}
    if len(keys) != int(emb.shape[0]):
        print(f"[WARN] invalid relabel embed cache shape: {cache_file}")
        return {}
    return {k: emb[i] for i, k in enumerate(keys)}


def save_relabel_embed_cache(cache_file: Path, cache_map: dict[str, np.ndarray]) -> None:
    if not cache_map:
        return
    keys = sorted(cache_map.keys())
    emb = np.vstack([np.asarray(cache_map[k], dtype=np.float32) for k in keys]).astype(np.float32)
    np.savez(cache_file, cache_key=np.asarray(keys), embeddings=emb)


def ensure_relabel_embed_cache(
    records: list[tuple[str, str]],
    model_name: str,
    cache_file: Path,
    use_cache: bool,
) -> tuple[dict[str, np.ndarray], str, int, int]:
    cache_map = load_relabel_embed_cache(cache_file) if use_cache else {}
    missing_map: dict[str, str] = {}
    for key, text in records:
        if key not in cache_map and key not in missing_map:
            missing_map[key] = text

    actual_model_name = model_name
    actual_batch_size = MANUAL_BATCH_SIZE
    n_missing = int(len(missing_map))
    if n_missing > 0:
        miss_keys = list(missing_map.keys())
        miss_texts = [missing_map[k] for k in miss_keys]
        miss_emb, actual_model_name, actual_batch_size = compute_embeddings(miss_texts, model_name)
        for k, v in zip(miss_keys, miss_emb):
            cache_map[k] = np.asarray(v, dtype=np.float32)
        if use_cache:
            save_relabel_embed_cache(cache_file, cache_map)

    return cache_map, actual_model_name, actual_batch_size, n_missing


def get_relabel_proto_embeddings(
    proto_labels: list[str],
    proto_texts: list[str],
    model_name: str,
    profile: str,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[np.ndarray, str, int]:
    if not proto_labels or not proto_texts:
        return np.zeros((0, 0), dtype=np.float32), model_name, MANUAL_BATCH_SIZE

    cache_file = make_relabel_proto_cache_file(cache_dir, model_name, profile)
    signature_src = "\n".join([f"{l}:{t}" for l, t in zip(proto_labels, proto_texts)])
    signature = hashlib.sha1(signature_src.encode("utf-8", errors="ignore")).hexdigest()
    if use_cache and cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=False)
            cached_sig = str(data["signature"].item())
            cached_emb = np.asarray(data["embeddings"], dtype=np.float32)
            if cached_sig == signature and int(cached_emb.shape[0]) == len(proto_labels):
                return cached_emb, model_name, MANUAL_BATCH_SIZE
        except Exception:
            pass

    emb, actual_model_name, actual_batch_size = compute_embeddings(proto_texts, model_name)
    emb_arr = np.asarray(emb, dtype=np.float32)
    if use_cache:
        np.savez(cache_file, signature=np.asarray(signature), embeddings=emb_arr)
    return emb_arr, actual_model_name, actual_batch_size


def precompute_relabel_embedding_cache(
    pdf: pd.DataFrame,
    profile: str,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    if pdf.empty:
        return {
            "cache_file": "",
            "n_rows": 0,
            "n_unique_keys": 0,
            "n_missing_computed": 0,
            "model_requested": "",
            "model_used": "",
            "batch_size_used": MANUAL_BATCH_SIZE,
        }

    cache_base = cache_dir or (OUTPUT_ROOT / "_cache")
    cache_base.mkdir(parents=True, exist_ok=True)
    model_name, model_source = choose_model_name()
    cache_file = make_relabel_embed_cache_file(cache_base, model_name, profile)

    records: list[tuple[str, str]] = []
    for _, row in pdf.iterrows():
        text = build_relabel_embed_text(row)
        key = relabel_embed_cache_key(row.get("business_id", ""), text)
        records.append((key, text))

    unique_keys = len({k for k, _ in records})
    cache_map, actual_model_name, actual_batch_size, n_missing = ensure_relabel_embed_cache(
        records=records,
        model_name=model_name,
        cache_file=cache_file,
        use_cache=RELABEL_EMBED_RECALL_USE_CACHE,
    )
    print(
        f"[INFO] relabel embedding cache prepared rows={len(records)} unique_keys={unique_keys} "
        f"missing_computed={n_missing} cache={cache_file} model={actual_model_name} source={model_source}"
    )
    return {
        "cache_file": str(cache_file),
        "n_rows": int(len(records)),
        "n_unique_keys": int(unique_keys),
        "n_missing_computed": int(n_missing),
        "model_requested": model_name,
        "model_used": actual_model_name,
        "batch_size_used": int(actual_batch_size),
        "cache_size": int(len(cache_map)),
    }


def build_l2_feature_matrix(cluster_pdf: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    n = int(len(cluster_pdf))
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    if "l2_label_top1" not in cluster_pdf.columns:
        return np.zeros((n, 0), dtype=np.float32), []

    labels: set[str] = set()
    for col in ["l2_label_top1", "l2_label_top2"]:
        if col not in cluster_pdf.columns:
            continue
        for value in cluster_pdf[col].fillna("").astype(str).tolist():
            token = value.strip().lower()
            if not token:
                continue
            if token == "other_service":
                continue
            if HYBRID_IGNORE_GENERAL_L2 and token == "restaurants_general":
                continue
            labels.add(token)

    ordered = sorted(list(labels))
    if not ordered:
        return np.zeros((n, 0), dtype=np.float32), []

    idx = {label: i for i, label in enumerate(ordered)}
    mat = np.zeros((n, len(ordered)), dtype=np.float32)

    for i, row in cluster_pdf.reset_index(drop=True).iterrows():
        l2_1 = str(row.get("l2_label_top1", "")).strip().lower()
        l2_2 = str(row.get("l2_label_top2", "")).strip().lower()
        if l2_1 in idx:
            mat[i, idx[l2_1]] += 1.0
        if HYBRID_USE_L2_TOP2 and l2_2 in idx and l2_2 != l2_1:
            mat[i, idx[l2_2]] += float(HYBRID_L2_TOP2_WEIGHT)

    for i in range(mat.shape[0]):
        norm = float(np.linalg.norm(mat[i]))
        if norm > 0:
            mat[i] = mat[i] / norm

    mat = mat * float(HYBRID_L2_FEATURE_WEIGHT)
    return mat.astype(np.float32), ordered


def clamp_confidence(x: Any, default_value: int = 2) -> int:
    try:
        raw = float(x)
        # Accept either 1-5 scale or fractional confidence from some models.
        if 0.0 < raw < 1.0:
            raw = 1.0 + (4.0 * raw)
        v = int(round(raw))
    except Exception:
        v = default_value
    return max(1, min(5, v))


def load_label_whitelist(config_dir: Path) -> tuple[set[str], set[str]]:
    csv_path = config_dir / "label_whitelist_v1.csv"
    if not csv_path.exists():
        return DEFAULT_ALLOWED_L1.copy(), DEFAULT_ALLOWED_L2_SERVICE.copy()

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    level_col = cols.get("label_level")
    code_col = cols.get("label_code")
    active_col = cols.get("active")
    parent_col = cols.get("parent_label")
    if not level_col or not code_col:
        return DEFAULT_ALLOWED_L1.copy(), DEFAULT_ALLOWED_L2_SERVICE.copy()

    allowed_l1 = set()
    allowed_l2 = set()
    for _, row in df.iterrows():
        active = str(row.get(active_col, "1")).strip().lower() not in {"0", "false", "no"}
        if not active:
            continue
        level = str(row[level_col]).strip().upper()
        code = str(row[code_col]).strip()
        if not code:
            continue
        if level == "L1":
            allowed_l1.add(code)
        elif level == "L2":
            parent = str(row.get(parent_col, "")).strip().lower()
            if parent == "food_service":
                allowed_l2.add(code)

    if not allowed_l1:
        allowed_l1 = DEFAULT_ALLOWED_L1.copy()
    if not allowed_l2:
        allowed_l2 = DEFAULT_ALLOWED_L2_SERVICE.copy()
    return allowed_l1, allowed_l2


def load_brand_terms(config_dir: Path) -> dict[str, list[str]]:
    out = {"food_service": [], "food_retail": [], "non_food": []}
    csv_path = config_dir / "brand_lexicon_v1.csv"
    if not csv_path.exists():
        return out

    df = pd.read_csv(csv_path)
    if not {"token", "category_hint"}.issubset(set(df.columns)):
        return out
    for _, row in df.iterrows():
        token = str(row.get("token", "")).strip().lower()
        hint = str(row.get("category_hint", "")).strip().lower()
        if not token or hint not in out:
            continue
        out[hint].append(token)
    return out


def any_hits(text: str, terms: list[str]) -> list[str]:
    hits = []
    for term in terms:
        token = normalize_text(term)
        if not token:
            continue
        pattern = _ANY_HIT_PATTERN_CACHE.get(token)
        if pattern is None:
            escaped = re.escape(token).replace(r"\ ", r"\s+")
            pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
            _ANY_HIT_PATTERN_CACHE[token] = pattern
        if pattern.search(text):
            hits.append(term)
    return hits


def should_rescue_uncertain_to_food_service(
    text: str,
    service_hits: list[str],
    retail_hits: list[str],
    non_food_hits: list[str],
) -> bool:
    if not UNCERTAIN_RESCUE_ENABLE:
        return False
    if non_food_hits:
        return False

    service_n = int(len(service_hits))
    retail_n = int(len(retail_hits))
    if service_n < int(UNCERTAIN_RESCUE_MIN_SERVICE_HITS):
        return False
    if service_n < (retail_n + int(UNCERTAIN_RESCUE_MIN_SERVICE_OVER_RETAIL)):
        return False
    if UNCERTAIN_RESCUE_REQUIRE_RESTAURANT_TOKEN and ("restaurant" not in text):
        return False
    return True


_TERM_PATTERN_CACHE: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
_ANY_HIT_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def compile_term_patterns(label: str) -> list[tuple[str, re.Pattern[str]]]:
    cached = _TERM_PATTERN_CACHE.get(label)
    if cached is not None:
        return cached

    out: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for raw in L2_KEYWORDS.get(label, []):
        term = normalize_text(raw)
        if not term or term in seen:
            continue
        seen.add(term)
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        out.append((term, re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")))
    _TERM_PATTERN_CACHE[label] = out
    return out


def score_l2_labels(
    name_text: str,
    categories_text: str,
    review_excerpt: str,
    allowed_l2: set[str],
) -> tuple[dict[str, float], dict[str, list[str]], float]:
    fields = [
        (categories_text, float(L2_WEIGHT_CATEGORIES)),
        (name_text, float(L2_WEIGHT_NAME)),
        (review_excerpt, float(L2_WEIGHT_REVIEW)),
    ]

    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for label, _ in L2_KEYWORDS.items():
        if label not in allowed_l2:
            continue
        if label == "restaurants_general" and L2_GENERAL_IS_FALLBACK_ONLY:
            continue
        if label == "other_service":
            continue

        label_score = 0.0
        label_hits: list[str] = []
        for term, pattern in compile_term_patterns(label):
            term_score = 0.0
            for text, weight in fields:
                if text and pattern.search(text):
                    term_score += weight
            if term_score > 0:
                label_score += term_score
                label_hits.append(term)
        if label_score > 0:
            scores[label] = float(label_score)
            evidence[label] = label_hits

    general_score = 0.0
    if "restaurants_general" in allowed_l2:
        for _, pattern in compile_term_patterns("restaurants_general"):
            term_score = 0.0
            for text, weight in fields:
                if text and pattern.search(text):
                    term_score += weight
            if term_score > 0:
                general_score += term_score

    return scores, evidence, float(general_score)


def apply_scene_decision_constraints(
    scores: dict[str, float],
    evidence: dict[str, list[str]],
) -> tuple[dict[str, float], dict[str, list[str]], list[str]]:
    adjusted_scores = dict(scores)
    adjusted_evidence = {k: list(v) for k, v in evidence.items()}
    notes: list[str] = []

    # Broad scene label guardrail: bar_nightlife needs stronger evidence than generic bars/nightlife noise.
    if "bar_nightlife" in adjusted_scores:
        bar_hits = {
            normalize_text(t)
            for t in adjusted_evidence.get("bar_nightlife", [])
            if normalize_text(t)
        }
        has_strong_scene_term = any(t in BAR_SCENE_STRONG_TERMS for t in bar_hits)
        if (len(bar_hits) < int(BAR_SCENE_MIN_TERMS)) or (
            BAR_SCENE_REQUIRE_STRONG_TERM and not has_strong_scene_term
        ):
            adjusted_scores.pop("bar_nightlife", None)
            adjusted_evidence.pop("bar_nightlife", None)
            notes.append("drop_bar_nightlife_weak_scene_signal")

    return adjusted_scores, adjusted_evidence, notes


def apply_pairwise_conflict_rules(
    scores: dict[str, float],
    evidence: dict[str, list[str]],
    *,
    name_text: str,
    categories_text: str,
    review_excerpt: str,
) -> tuple[dict[str, float], list[str]]:
    adjusted_scores = dict(scores)
    notes: list[str] = []
    combined_text = normalize_text(name_text, categories_text, review_excerpt)

    for rule in PAIRWISE_CONFLICT_RULES:
        left = str(rule.get("left", "")).strip()
        right = str(rule.get("right", "")).strip()
        if not left or not right:
            continue
        if left not in adjusted_scores or right not in adjusted_scores:
            continue

        gap = abs(float(adjusted_scores.get(left, 0.0)) - float(adjusted_scores.get(right, 0.0)))
        if gap > float(PAIRWISE_CONFLICT_MAX_GAP):
            continue

        left_hits = {normalize_text(x) for x in evidence.get(left, []) if normalize_text(x)}
        right_hits = {normalize_text(x) for x in evidence.get(right, []) if normalize_text(x)}
        left_terms = {normalize_text(x) for x in rule.get("left_terms", set()) if normalize_text(x)}
        right_terms = {normalize_text(x) for x in rule.get("right_terms", set()) if normalize_text(x)}
        left_text_terms = [normalize_text(x) for x in rule.get("left_text_terms", set()) if normalize_text(x)]
        right_text_terms = [normalize_text(x) for x in rule.get("right_text_terms", set()) if normalize_text(x)]

        left_signal = int(len(left_hits & left_terms)) + int(len(any_hits(combined_text, left_text_terms)))
        right_signal = int(len(right_hits & right_terms)) + int(len(any_hits(combined_text, right_text_terms)))
        if left_signal == right_signal:
            continue

        winner = left if left_signal > right_signal else right
        loser = right if winner == left else left
        adjusted_scores[winner] = float(adjusted_scores.get(winner, 0.0)) + float(PAIRWISE_CONFLICT_BOOST)
        notes.append(
            f"pairwise_{left}_vs_{right}:prefer_{winner}(signal={max(left_signal, right_signal)}>{min(left_signal, right_signal)})"
        )
        # no hard penalty on loser; keep conservative behavior and only give winner a nudge.
        _ = loser

    return adjusted_scores, notes


def reorder_ranked_labels(
    ranked_labels: list[str],
    scores: dict[str, float],
) -> tuple[list[str], list[str]]:
    ordered = list(ranked_labels)
    notes: list[str] = []

    if "bar_nightlife" in ordered:
        bar_score = float(scores.get("bar_nightlife", 0.0))
        child_candidates = [
            (label, float(scores.get(label, 0.0)))
            for label in BAR_SCENE_CHILD_LABELS
            if label in ordered
        ]
        if child_candidates:
            child_label, child_score = sorted(
                child_candidates,
                key=lambda kv: (-kv[1], kv[0]),
            )[0]
            if (
                child_score >= float(BAR_SCENE_CHILD_PROMOTE_MIN_SCORE)
                and child_score >= (bar_score - float(BAR_SCENE_CHILD_PROMOTE_GAP))
            ):
                ordered = [x for x in ordered if x != child_label]
                bar_idx = ordered.index("bar_nightlife")
                ordered.insert(bar_idx, child_label)
                notes.append(f"promote_{child_label}_over_bar_nightlife")

    if ordered and ordered[0] in SCENE_LABELS:
        scene_label = ordered[0]
        scene_score = float(scores.get(scene_label, 0.0))
        cuisine_candidates = [
            (label, float(scores.get(label, 0.0)))
            for label in ordered
            if label in CUISINE_LABELS
        ]
        if cuisine_candidates:
            cuisine_label, cuisine_score = sorted(
                cuisine_candidates,
                key=lambda kv: (-kv[1], kv[0]),
            )[0]
            if (
                cuisine_score >= float(SCENE_VS_CUISINE_MIN_SCORE)
                and cuisine_score >= (scene_score - float(SCENE_VS_CUISINE_OVERRIDE_GAP))
                and cuisine_label != scene_label
            ):
                ordered = [cuisine_label] + [x for x in ordered if x != cuisine_label]
                notes.append(f"promote_{cuisine_label}_over_scene_{scene_label}")

    return ordered, notes


def choose_l2_labels(
    name_text: str,
    categories_text: str,
    review_excerpt: str,
    allowed_l2: set[str],
) -> tuple[str, str, dict[str, Any]]:
    scores, evidence, general_score = score_l2_labels(
        name_text=name_text,
        categories_text=categories_text,
        review_excerpt=review_excerpt,
        allowed_l2=allowed_l2,
    )

    fallback = "restaurants_general" if "restaurants_general" in allowed_l2 else "other_service"
    if fallback not in allowed_l2 and allowed_l2:
        fallback = sorted(list(allowed_l2))[0]

    scores, evidence, constraint_notes = apply_scene_decision_constraints(scores, evidence)
    scores, pairwise_notes = apply_pairwise_conflict_rules(
        scores=scores,
        evidence=evidence,
        name_text=name_text,
        categories_text=categories_text,
        review_excerpt=review_excerpt,
    )
    constraint_notes = constraint_notes + pairwise_notes

    if not scores:
        return (
            fallback,
            "",
            {
                "source": "fallback_general" if fallback == "restaurants_general" else "fallback_other",
                "top1_score": float(general_score),
                "top2_score": 0.0,
                "score_gap": 0.0,
                "is_ambiguous": False,
                "top1_terms": [],
                "top2_terms": [],
                "general_score": float(general_score),
                "decision_notes": ";".join(constraint_notes),
            },
        )

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked_labels = [label for label, _ in ranked]
    ranked_labels, reorder_notes = reorder_ranked_labels(ranked_labels, scores)
    decision_notes = constraint_notes + reorder_notes

    top1 = ranked_labels[0]
    top1_score = float(scores.get(top1, 0.0))
    top2 = ""
    top2_score = 0.0
    if len(ranked_labels) > 1:
        top2 = ranked_labels[1]
        top2_score = float(scores.get(top2, 0.0))
    score_gap = float(top1_score - top2_score)
    is_ambiguous = bool(top2) and score_gap <= float(L2_AMBIGUOUS_GAP_MAX)

    # Guardrail: with extremely weak evidence, force fallback to restaurants_general.
    if (
        fallback == "restaurants_general"
        and float(top1_score) <= float(L2_FORCE_GENERAL_MAX_SCORE)
    ):
        return (
            fallback,
            "",
            {
                "source": "fallback_low_score_general",
                "top1_score": float(top1_score),
                "top2_score": float(top2_score),
                "score_gap": float(score_gap),
                "is_ambiguous": bool(is_ambiguous),
                "top1_terms": evidence.get(top1, []),
                "top2_terms": evidence.get(top2, []) if top2 else [],
                "general_score": float(general_score),
                "fallback_from_label": top1,
                "decision_notes": ";".join(decision_notes),
            },
        )

    if is_ambiguous:
        chosen_top2 = top2
    else:
        chosen_top2 = ""

    return (
        top1,
        chosen_top2,
        {
            "source": "specific",
            "top1_score": float(top1_score),
            "top2_score": float(top2_score),
            "score_gap": float(score_gap),
            "is_ambiguous": bool(is_ambiguous),
            "top1_terms": evidence.get(top1, []),
            "top2_terms": evidence.get(top2, []) if top2 else [],
            "general_score": float(general_score),
            "decision_notes": ";".join(decision_notes),
        },
    )


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a_arr, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b_arr, axis=1, keepdims=True)
    a_safe = a_arr / np.clip(a_norm, 1e-8, None)
    b_safe = b_arr / np.clip(b_norm, 1e-8, None)
    return np.matmul(a_safe, b_safe.T)


def build_l2_prototype_texts(allowed_l2_service: set[str]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    texts: list[str] = []
    for label in sorted(list(allowed_l2_service)):
        if label in {"restaurants_general", "other_service"}:
            continue
        keywords = [normalize_text(x) for x in L2_KEYWORDS.get(label, []) if normalize_text(x)]
        label_text = label.replace("_", " ")
        merged = [label_text] + keywords
        seen: set[str] = set()
        dedup: list[str] = []
        for term in merged:
            if term and term not in seen:
                seen.add(term)
                dedup.append(term)
        proto = f"label={label_text}; key_terms={', '.join(dedup)}"
        labels.append(label)
        texts.append(proto)
    return labels, texts


def get_relabel_embed_strategy() -> str:
    strategy = str(RELABEL_EXPERIMENT_STRATEGY).strip().lower()
    if strategy not in {"selective", "dense"}:
        return "selective"
    return strategy


def apply_embedding_recall_to_relabels(
    out: pd.DataFrame,
    allowed_l2_service: set[str],
    profile: str,
    cache_dir: Path | None = None,
) -> dict[str, int]:
    strategy = get_relabel_embed_strategy()
    stats = {
        "n_embed_candidate_pool": 0,
        "n_embed_called": 0,
        "n_embed_applied": 0,
        "n_embed_llm_skipped": 0,
        "n_embed_forced_llm": 0,
        "n_embed_cache_miss": 0,
    }
    if not RELABEL_USE_EMBED_RECALL:
        return stats
    if out.empty:
        return stats

    for col, default in [
        ("embed_top1_label", ""),
        ("embed_top1_sim", 0.0),
        ("embed_top2_label", ""),
        ("embed_top2_sim", 0.0),
        ("embed_topk_labels", ""),
        ("embed_topk_sims", ""),
        ("embed_score_gap", 0.0),
        ("embed_applied", False),
    ]:
        if col not in out.columns:
            out[col] = default

    conf_series = pd.to_numeric(out.get("label_confidence", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    l1_series = out.get("l1_label", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    l2_series = out.get("l2_label_top1", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    needs_llm_series = out.get("needs_llm", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    rule_hit_series = out.get("rule_hit", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    trigger_series = out.get("llm_trigger_reason", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    l2_source_series = (
        out.get("rule_l2_source", out.get("l2_source", pd.Series("", index=out.index)))
        .fillna("")
        .astype(str)
        .str.lower()
    )

    base_service_mask = l1_series.isin(["uncertain", "food_service"])
    low_conf_mask = conf_series <= int(RELABEL_EMBED_RECALL_CONFIDENCE_MAX)
    conflict_mask = rule_hit_series.str.contains("conflict", na=False) | trigger_series.str.contains("l2_ambiguous", na=False)
    general_fallback_mask = (
        ((l1_series == "food_service") & l2_series.isin(["", "restaurants_general"]))
        | l2_source_series.isin(["fallback_general", "fallback_low_score_general"])
        | trigger_series.str.contains("general_fallback", na=False)
    )

    if strategy == "dense":
        if bool(RELABEL_EMBED_RECALL_DENSE_INCLUDE_NON_FOOD):
            candidate_mask = pd.Series(True, index=out.index, dtype=bool)
        else:
            candidate_mask = base_service_mask
        max_candidates = (
            RELABEL_EMBED_RECALL_MAX_CANDIDATES_DENSE_SAMPLE
            if profile == "sample"
            else RELABEL_EMBED_RECALL_MAX_CANDIDATES_DENSE_FULL
        )
    else:
        candidate_mask = base_service_mask & (
            needs_llm_series
            | low_conf_mask
            | conflict_mask
            | general_fallback_mask
        )
        max_candidates = (
            RELABEL_EMBED_RECALL_MAX_CANDIDATES_SAMPLE
            if profile == "sample"
            else RELABEL_EMBED_RECALL_MAX_CANDIDATES_FULL
        )

    candidate_pool = out[candidate_mask].copy()
    stats["n_embed_candidate_pool"] = int(len(candidate_pool))
    cand = candidate_pool
    if cand.empty:
        return stats

    cand = cand.sort_values(
        by=["needs_llm", "llm_priority", "label_confidence", "n_reviews"],
        ascending=[False, False, True, False],
    ).head(int(max_candidates))
    if cand.empty:
        return stats

    proto_labels, proto_texts = build_l2_prototype_texts(allowed_l2_service)
    if not proto_labels:
        return stats

    model_name, model_source = choose_model_name()
    cache_base = cache_dir or (OUTPUT_ROOT / "_cache")
    cache_base.mkdir(parents=True, exist_ok=True)
    relabel_cache_file = make_relabel_embed_cache_file(cache_base, model_name, profile)

    def build_records(df: pd.DataFrame) -> list[tuple[int, str, str]]:
        rows: list[tuple[int, str, str]] = []
        for idx in df.index.tolist():
            row = out.loc[idx]
            text = build_relabel_embed_text(row)
            key = relabel_embed_cache_key(row.get("business_id", ""), text)
            rows.append((int(idx), key, text))
        return rows

    cand_records = build_records(cand)
    warm_records = cand_records
    if RELABEL_EMBED_RECALL_PRECOMPUTE_ALL:
        warm_records = build_records(out)

    print(
        f"[STEP] embedding recall strategy={strategy} candidates={len(cand)}/{len(candidate_pool)} "
        f"labels={len(proto_labels)} model={model_name} source={model_source}"
    )

    proto_emb, actual_model_name, actual_batch_size = get_relabel_proto_embeddings(
        proto_labels=proto_labels,
        proto_texts=proto_texts,
        model_name=model_name,
        profile=profile,
        cache_dir=cache_base,
        use_cache=RELABEL_EMBED_RECALL_USE_CACHE,
    )
    cache_map, actual_model_name, actual_batch_size, cache_miss = ensure_relabel_embed_cache(
        records=[(k, t) for _, k, t in warm_records],
        model_name=actual_model_name,
        cache_file=relabel_cache_file,
        use_cache=RELABEL_EMBED_RECALL_USE_CACHE,
    )
    stats["n_embed_cache_miss"] = int(cache_miss)
    cand_emb = np.vstack([cache_map[k] for _, k, _ in cand_records]).astype(np.float32)
    sim_mat = cosine_similarity_matrix(cand_emb, proto_emb)

    stats["n_embed_called"] = int(len(cand))
    min_sim = float(RELABEL_EMBED_RECALL_MIN_SIM)
    min_margin = float(RELABEL_EMBED_RECALL_MIN_MARGIN)
    strong_sim = float(RELABEL_EMBED_RECALL_STRONG_SIM)
    min_sim_uncertain = float(RELABEL_EMBED_RECALL_MIN_SIM_UNCERTAIN)
    scene_terms = sorted(list(BAR_SCENE_STRONG_TERMS))
    scene_guard_enabled = bool(RELABEL_EMBED_RECALL_SKIP_WEAK_SCENE)

    def append_trigger(idx: int, token: str) -> None:
        trigger = str(out.at[idx, "llm_trigger_reason"]).strip()
        parts = [t for t in trigger.split(",") if t]
        if token not in parts:
            parts.append(token)
        out.at[idx, "llm_trigger_reason"] = ",".join(parts)

    text_map = {idx: text for idx, _, text in cand_records}
    cand_indices = [idx for idx, _, _ in cand_records]
    for row_pos, idx in enumerate(cand_indices):
        sims = np.asarray(sim_mat[row_pos], dtype=np.float32)
        if sims.size <= 0:
            continue
        order = np.argsort(sims)[::-1]
        top1_i = int(order[0])
        top2_i = int(order[1]) if order.size > 1 else int(order[0])
        top1_label = str(proto_labels[top1_i])
        top2_label = str(proto_labels[top2_i]) if top2_i != top1_i else ""
        top1_sim = float(sims[top1_i])
        top2_sim = float(sims[top2_i]) if top2_label else 0.0
        score_gap = float(top1_sim - top2_sim)
        topk_n = max(1, min(int(RELABEL_EMBED_RECALL_STORE_TOPK), int(order.size)))
        topk_labels = [str(proto_labels[int(i)]) for i in order[:topk_n]]
        topk_sims = [float(sims[int(i)]) for i in order[:topk_n]]

        out.at[idx, "embed_top1_label"] = top1_label
        out.at[idx, "embed_top1_sim"] = top1_sim
        out.at[idx, "embed_top2_label"] = top2_label
        out.at[idx, "embed_top2_sim"] = top2_sim
        out.at[idx, "embed_topk_labels"] = "|".join(topk_labels)
        out.at[idx, "embed_topk_sims"] = "|".join(f"{x:.4f}" for x in topk_sims)
        out.at[idx, "embed_score_gap"] = score_gap

        if top1_sim < min_sim or score_gap < min_margin:
            if strategy == "dense":
                out.at[idx, "needs_llm"] = True
                out.at[idx, "llm_priority"] = max(int(out.at[idx, "llm_priority"]), 2)
                append_trigger(idx, "embed_dense_weak")
                stats["n_embed_forced_llm"] += 1
            continue

        row_l1 = str(out.at[idx, "l1_label"]).strip().lower()
        row_l2 = str(out.at[idx, "l2_label_top1"]).strip().lower()
        row_conf = clamp_confidence(out.at[idx, "label_confidence"], default_value=2)

        if (
            top1_label == "bar_nightlife"
            and scene_guard_enabled
            and not any_hits(text_map.get(idx, ""), scene_terms)
        ):
            if strategy == "dense":
                out.at[idx, "needs_llm"] = True
                out.at[idx, "llm_priority"] = max(int(out.at[idx, "llm_priority"]), 2)
                append_trigger(idx, "embed_scene_guard")
                stats["n_embed_forced_llm"] += 1
            continue

        if strategy == "dense":
            should_apply = True
        else:
            should_apply = False
            if row_l1 == "uncertain" and top1_sim >= min_sim_uncertain:
                should_apply = True
            elif row_l1 == "food_service" and (row_l2 in {"", "restaurants_general"} or row_conf <= 2):
                should_apply = True
        if not should_apply:
            if strategy == "dense":
                out.at[idx, "needs_llm"] = True
                out.at[idx, "llm_priority"] = max(int(out.at[idx, "llm_priority"]), 2)
                append_trigger(idx, "embed_dense_hold_rule")
                stats["n_embed_forced_llm"] += 1
            continue

        prev_label = f"{out.at[idx, 'l1_label']}|{out.at[idx, 'l2_label_top1']}"
        out.at[idx, "l1_label"] = "food_service"
        out.at[idx, "l2_label_top1"] = top1_label

        if top2_label and top2_label != top1_label and top2_sim >= min_sim and score_gap <= (min_margin + 0.04):
            out.at[idx, "l2_label_top2"] = top2_label
        else:
            out.at[idx, "l2_label_top2"] = ""

        new_conf = max(row_conf, 4 if top1_sim >= strong_sim else 3)
        out.at[idx, "label_confidence"] = clamp_confidence(new_conf, default_value=3)

        reason = str(out.at[idx, "label_reason"])
        embed_note = (
            f"embed_top1={top1_label}:{top1_sim:.3f}; "
            f"embed_top2={top2_label}:{top2_sim:.3f}; gap={score_gap:.3f}"
        )
        out.at[idx, "label_reason"] = f"{reason} | {embed_note}"[:500]
        out.at[idx, "l2_source"] = "embedding_recall_dense" if strategy == "dense" else "embedding_recall"
        out.at[idx, "l2_score_top1"] = top1_sim
        out.at[idx, "l2_score_top2"] = top2_sim
        out.at[idx, "l2_score_gap"] = score_gap
        out.at[idx, "l2_top1_terms"] = ", ".join(L2_KEYWORDS.get(top1_label, [])[:6])
        out.at[idx, "l2_top2_terms"] = ", ".join(L2_KEYWORDS.get(top2_label, [])[:6]) if top2_label else ""

        if top1_sim >= strong_sim:
            out.at[idx, "needs_llm"] = False
            out.at[idx, "llm_priority"] = max(0, int(out.at[idx, "llm_priority"]) - 2)
            append_trigger(idx, "embed_strong")
            stats["n_embed_llm_skipped"] += 1
        else:
            if strategy == "dense":
                out.at[idx, "needs_llm"] = True
                out.at[idx, "llm_priority"] = max(int(out.at[idx, "llm_priority"]), 2)
                append_trigger(idx, "embed_dense_medium")
                stats["n_embed_forced_llm"] += 1
            else:
                out.at[idx, "needs_llm"] = bool(out.at[idx, "needs_llm"])
                append_trigger(idx, "embed_medium")

        out.at[idx, "label_source"] = "embed_dense" if strategy == "dense" else "embed_rule"
        out.at[idx, "embed_applied"] = True
        out.at[idx, "llm_action"] = "NOT_CALLED"
        out.at[idx, "from_label"] = prev_label
        out.at[idx, "to_label"] = f"food_service|{top1_label}"
        out.at[idx, "action_validated"] = False
        out.at[idx, "evidence_terms"] = f"{out.at[idx, 'evidence_terms']}, embed:{top1_label}"[:500]
        rule_hit = str(out.at[idx, "rule_hit"])
        embed_rule_tag = "E1_embed_recall_dense" if strategy == "dense" else "E1_embed_recall"
        if embed_rule_tag not in rule_hit:
            out.at[idx, "rule_hit"] = f"{rule_hit}|{embed_rule_tag}"
        stats["n_embed_applied"] += 1

    print(
        f"[INFO] embedding recall strategy={strategy} applied={stats['n_embed_applied']}/{stats['n_embed_called']} "
        f"llm_skipped={stats['n_embed_llm_skipped']} forced_llm={stats['n_embed_forced_llm']} "
        f"cache_miss={stats['n_embed_cache_miss']} "
        f"model={actual_model_name} batch={actual_batch_size}"
    )
    return stats


def rule_relabel_one(
    row: pd.Series,
    allowed_l1: set[str],
    allowed_l2_service: set[str],
    service_terms: list[str],
    retail_terms: list[str],
    non_food_terms: list[str],
) -> dict[str, Any]:
    name_text = normalize_text(row.get("name", ""))
    categories_text = normalize_text(row.get("categories", ""))
    review_excerpt = ""
    if RULE_USE_REVIEW_SIGNAL:
        review_excerpt = normalize_text(str(row.get("text", "") or "")[:RULE_REVIEW_SIGNAL_CHARS])

    text = normalize_text(name_text, categories_text)
    service_hits = any_hits(text, service_terms)
    retail_hits = any_hits(text, retail_terms)
    non_food_hits = any_hits(text, non_food_terms)

    l1 = "uncertain"
    confidence = 1
    rule_hit = "R5_fallback_uncertain"

    if non_food_hits and not service_hits:
        l1 = "non_food"
        confidence = 5
        rule_hit = "R1_non_food_override"
    elif retail_hits and not service_hits:
        l1 = "food_retail"
        confidence = 5
        rule_hit = "R2_food_retail_override"
    elif service_hits and not retail_hits:
        l1 = "food_service"
        confidence = 4
        rule_hit = "R3_food_service_include"
    elif service_hits and retail_hits:
        retail_dominant_hits = any_hits(text, RETAIL_DOMINANT_TERMS)
        if retail_dominant_hits or (len(retail_hits) >= len(service_hits) + 1):
            l1 = "food_retail"
            confidence = 3
            rule_hit = "R4_conflict_retail_dominant"
        elif len(service_hits) >= len(retail_hits) + 2:
            l1 = "food_service"
            confidence = 3
            rule_hit = "R4_conflict_service_dominant"
        else:
            l1 = "uncertain"
            confidence = 2
            rule_hit = "R4_conflict_service_retail"
    elif "food" in text and "restaurants" not in text:
        l1 = "food_retail"
        confidence = 3
        rule_hit = "R4_food_only_not_service"
    elif "restaurants" in text:
        l1 = "food_service"
        confidence = 3
        rule_hit = "R3_restaurants_literal"

    if l1 == "uncertain" and should_rescue_uncertain_to_food_service(
        text=text,
        service_hits=service_hits,
        retail_hits=retail_hits,
        non_food_hits=non_food_hits,
    ):
        l1 = "food_service"
        confidence = max(confidence, 3)
        rule_hit = f"{rule_hit}|R4b_uncertain_rescue_service_dominant"

    if l1 not in allowed_l1:
        l1 = "uncertain"
        confidence = 1
        rule_hit = "R0_not_in_whitelist"

    l2_top1 = ""
    l2_top2 = ""
    l2_meta: dict[str, Any] = {
        "source": "",
        "top1_score": 0.0,
        "top2_score": 0.0,
        "score_gap": 0.0,
        "is_ambiguous": False,
        "top1_terms": [],
        "top2_terms": [],
        "general_score": 0.0,
        "decision_notes": "",
    }
    if l1 == "food_service":
        l2_top1, l2_top2, l2_meta = choose_l2_labels(
            name_text=name_text,
            categories_text=categories_text,
            review_excerpt=review_excerpt,
            allowed_l2=allowed_l2_service,
        )

    llm_trigger_reasons: list[str] = []
    if l1 == "food_service":
        if (
            LLM_TRIGGER_GENERAL_FALLBACK
            and str(l2_meta.get("source", "")) in {"fallback_general", "fallback_low_score_general"}
            and l2_top1 == "restaurants_general"
        ):
            llm_trigger_reasons.append("general_fallback")
            confidence = min(confidence, 3)
        if LLM_TRIGGER_L2_AMBIGUOUS and bool(l2_meta.get("is_ambiguous", False)):
            score_gap = float(l2_meta.get("score_gap", 999.0))
            top1_score = float(l2_meta.get("top1_score", 0.0))
            if (
                score_gap <= float(LLM_TRIGGER_L2_AMBIGUOUS_MAX_GAP)
                and top1_score <= float(LLM_TRIGGER_L2_AMBIGUOUS_MAX_TOP1_SCORE)
            ):
                llm_trigger_reasons.append("l2_ambiguous")
                confidence = min(confidence, 3)
        if LLM_TRIGGER_L2_LOW_SIGNAL and float(l2_meta.get("top1_score", 0.0)) < float(
            L2_LOW_SIGNAL_MIN_SCORE
        ):
            llm_trigger_reasons.append("low_l2_signal")
            confidence = min(confidence, 3)

    llm_priority = 0
    if l1 == "uncertain":
        llm_priority += 4
    if confidence <= 2:
        llm_priority += 3
    if "general_fallback" in llm_trigger_reasons:
        llm_priority += 3
    if "l2_ambiguous" in llm_trigger_reasons:
        llm_priority += 2
    if "low_l2_signal" in llm_trigger_reasons:
        llm_priority += 1

    needs_llm = RELABEL_USE_LLM and bool(
        (l1 == "uncertain")
        or (confidence <= int(LLM_REVIEW_CONFIDENCE_MAX))
        or (
            llm_trigger_reasons
            and confidence <= int(LLM_REVIEW_TRIGGER_CONFIDENCE_MAX)
        )
    )
    reason = (
        f"rule={rule_hit}; service_hits={len(service_hits)}; retail_hits={len(retail_hits)}; "
        f"l2_source={l2_meta.get('source', '')}; l2_gap={float(l2_meta.get('score_gap', 0.0)):.2f}; "
        f"notes={str(l2_meta.get('decision_notes', ''))[:120]}"
    )
    evidence_terms = ", ".join((service_hits + retail_hits + non_food_hits)[:8])

    return {
        "l1_label": l1,
        "l2_label_top1": l2_top1,
        "l2_label_top2": l2_top2,
        "label_confidence": confidence,
        "label_reason": reason,
        "evidence_terms": evidence_terms,
        "rule_hit": rule_hit,
        "l2_source": str(l2_meta.get("source", "")),
        "l2_score_top1": float(l2_meta.get("top1_score", 0.0)),
        "l2_score_top2": float(l2_meta.get("top2_score", 0.0)),
        "l2_score_gap": float(l2_meta.get("score_gap", 0.0)),
        "l2_top1_terms": ", ".join([str(x) for x in l2_meta.get("top1_terms", [])][:6]),
        "l2_top2_terms": ", ".join([str(x) for x in l2_meta.get("top2_terms", [])][:6]),
        "llm_trigger_reason": ",".join(llm_trigger_reasons),
        "llm_priority": int(llm_priority),
        "needs_llm": bool(needs_llm),
        "llm_action": "NOT_CALLED",
        "from_label": f"{l1}|{l2_top1}",
        "to_label": f"{l1}|{l2_top1}",
        "action_validated": False,
        "llm_status": "not_called",
        "llm_error_code": "",
        "llm_raw_head": "",
        "label_source": "rule",
    }


def ollama_generate(prompt: str, *, num_predict: int | None = None) -> str:
    predict_tokens = int(num_predict) if num_predict is not None else int(OLLAMA_NUM_PREDICT)
    options = {
        "temperature": 0,
        "num_ctx": int(OLLAMA_NUM_CTX),
        "num_predict": int(predict_tokens),
        # qwen3 in Ollama accepts both keys across versions; keep both for compatibility.
        "thinking": bool(OLLAMA_THINKING),
        "think": bool(OLLAMA_THINKING),
    }
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": "You are a business category labeling assistant. Return strict JSON only.",
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": options,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()

    def _extract_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "".join(str(x) for x in value if x is not None).strip()
        return ""

    # Different Ollama + model combos may place the useful body in different fields.
    # qwen3:4b can return JSON in "thinking" while leaving "response" empty.
    text = _extract_text(data.get("response"))
    if not text:
        text = _extract_text(data.get("thinking"))
    if not text:
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            text = _extract_text(msg.get("content"))
    return text


def try_parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def compact_text(text: Any, max_chars: int = 200) -> str:
    s = str(text or "").replace("\n", " ").replace("\r", " ").strip()
    return s[:max_chars]


def parse_embed_topk_pairs(row: pd.Series) -> list[tuple[str, float]]:
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    labels_raw = str(row.get("embed_topk_labels", "") or "").strip()
    sims_raw = str(row.get("embed_topk_sims", "") or "").strip()
    if (not labels_raw) and str(row.get("embed_top1_label", "") or "").strip():
        labels_raw = "|".join(
            [
                str(row.get("embed_top1_label", "") or "").strip(),
                str(row.get("embed_top2_label", "") or "").strip(),
            ]
        ).strip("|")
        sims_raw = "|".join(
            [
                str(_to_float(row.get("embed_top1_sim", 0.0))),
                str(_to_float(row.get("embed_top2_sim", 0.0))),
            ]
        ).strip("|")
    labels = [x.strip().lower() for x in labels_raw.split("|") if x.strip()]
    sims: list[float] = []
    for token in sims_raw.split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            sims.append(float(token))
        except Exception:
            sims.append(0.0)
    out: list[tuple[str, float]] = []
    for i, label in enumerate(labels):
        score = sims[i] if i < len(sims) else 0.0
        out.append((label, float(score)))
    return out


def choose_llm_candidate_topk(row: pd.Series, fallback: dict[str, Any]) -> int:
    conf = clamp_confidence(fallback.get("label_confidence", row.get("label_confidence", 2)))
    trigger = str(fallback.get("llm_trigger_reason", row.get("llm_trigger_reason", ""))).strip().lower()
    l1 = str(fallback.get("l1_label", row.get("l1_label", ""))).strip().lower()
    if (
        l1 == "uncertain"
        or conf <= 2
        or ("l2_ambiguous" in trigger)
        or ("general_fallback" in trigger)
        or ("low_l2_signal" in trigger)
    ):
        return int(LLM_CANDIDATE_TOPK_LOW_CONF)
    if conf <= 3:
        return int(LLM_CANDIDATE_TOPK_MID_CONF)
    return int(LLM_CANDIDATE_TOPK_HIGH_CONF)


def build_llm_l2_candidates(
    row: pd.Series,
    fallback: dict[str, Any],
    allowed_l2_service: set[str],
) -> list[str]:
    candidates: list[str] = []

    def push(label: Any) -> None:
        lab = str(label or "").strip().lower()
        if not lab:
            return
        if lab not in allowed_l2_service:
            return
        if lab not in candidates:
            candidates.append(lab)

    push(fallback.get("l2_label_top1", ""))
    push(fallback.get("l2_label_top2", ""))
    topk_target = max(2, min(int(choose_llm_candidate_topk(row, fallback)), int(LLM_CANDIDATE_HARD_LIMIT)))
    for lab, sim in parse_embed_topk_pairs(row):
        if sim < float(LLM_CANDIDATE_MIN_SIM):
            continue
        push(lab)
        if len(candidates) >= topk_target:
            break
    push("restaurants_general")

    if bool(LLM_CANDIDATE_BACKFILL_ALL) and len(candidates) < topk_target:
        for lab in sorted(list(allowed_l2_service)):
            if lab in {"other_service"}:
                continue
            push(lab)
            if len(candidates) >= topk_target:
                break
    return candidates[: int(LLM_CANDIDATE_HARD_LIMIT)]


def make_llm_fallback_keep(
    fallback: dict[str, Any],
    *,
    from_label: str,
    status: str,
    error_code: str,
    reason_note: str,
    raw_head: str = "",
) -> dict[str, Any]:
    return {
        **fallback,
        "label_reason": f"{fallback.get('label_reason', '')} | {reason_note}"[:500],
        "needs_llm": False,
        "llm_action": "KEEP",
        "from_label": from_label,
        "to_label": from_label,
        "action_validated": False,
        "label_source": fallback.get("label_source", "rule"),
        "llm_status": status,
        "llm_error_code": error_code,
        "llm_raw_head": raw_head[:200],
    }


def validate_llm_payload(
    parsed: dict[str, Any],
    fallback: dict[str, Any],
    allowed_l1: set[str],
    allowed_l2_service: set[str],
    candidate_l2_service: set[str] | None = None,
) -> tuple[bool, dict[str, Any], str]:
    action = str(parsed.get("action", "")).strip().upper()
    if action not in LLM_ACTIONS:
        return False, {}, "action_invalid"

    fallback_l1 = str(fallback.get("l1_label", "uncertain")).strip().lower()
    fallback_top1 = str(fallback.get("l2_label_top1", "")).strip().lower()
    fallback_top2 = str(fallback.get("l2_label_top2", "")).strip().lower()

    candidate_l1 = str(parsed.get("l1_label", fallback_l1)).strip().lower()
    candidate_top1 = str(parsed.get("l2_label_top1", fallback_top1)).strip().lower()
    candidate_top2 = str(parsed.get("l2_label_top2", fallback_top2)).strip().lower()
    confidence = clamp_confidence(parsed.get("confidence", fallback.get("label_confidence", 2)))
    reason_short = compact_text(parsed.get("reason_short", parsed.get("reason", "")), LLM_REASON_SHORT_MAX_CHARS)
    candidate_set = set(candidate_l2_service or set())

    if action == "KEEP":
        return True, {
            "action": "KEEP",
            "l1_label": fallback_l1,
            "l2_label_top1": fallback_top1,
            "l2_label_top2": fallback_top2,
            "confidence": confidence,
            "reason_short": reason_short,
        }, ""

    if candidate_l1 not in allowed_l1:
        return False, {}, "l1_out_of_whitelist"

    if action == "ADD":
        # ADD keeps current L1/top1 and only tries to add top2.
        add_candidate = ""
        if (
            candidate_top2
            and candidate_top2 in allowed_l2_service
            and candidate_top2 != fallback_top1
            and ((not candidate_set) or (candidate_top2 in candidate_set))
        ):
            add_candidate = candidate_top2
        elif (
            candidate_top1
            and candidate_top1 in allowed_l2_service
            and candidate_top1 != fallback_top1
            and ((not candidate_set) or (candidate_top1 in candidate_set))
        ):
            add_candidate = candidate_top1
        return True, {
            "action": "ADD",
            "l1_label": fallback_l1,
            "l2_label_top1": fallback_top1,
            "l2_label_top2": add_candidate,
            "confidence": confidence,
            "reason_short": reason_short,
        }, ""

    if action == "REJECT":
        return True, {
            "action": "REJECT",
            "l1_label": candidate_l1,
            "l2_label_top1": candidate_top1,
            "l2_label_top2": "",
            "confidence": confidence,
            "reason_short": reason_short,
        }, ""

    # MODIFY
    if candidate_l1 != "food_service":
        return True, {
            "action": "MODIFY",
            "l1_label": candidate_l1,
            "l2_label_top1": "",
            "l2_label_top2": "",
            "confidence": confidence,
            "reason_short": reason_short,
        }, ""

    if candidate_top1 not in allowed_l2_service:
        return False, {}, "modify_top1_invalid"
    if candidate_set and candidate_top1 not in candidate_set:
        return False, {}, "modify_top1_not_in_candidates"

    if candidate_top2 and (
        candidate_top2 not in allowed_l2_service
        or candidate_top2 == candidate_top1
        or (candidate_set and candidate_top2 not in candidate_set)
    ):
        candidate_top2 = ""

    return True, {
        "action": "MODIFY",
        "l1_label": candidate_l1,
        "l2_label_top1": candidate_top1,
        "l2_label_top2": candidate_top2,
        "confidence": confidence,
        "reason_short": reason_short,
    }, ""


def llm_relabel_one(
    row: pd.Series,
    allowed_l1: set[str],
    allowed_l2_service: set[str],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    allowed_l1_txt = ", ".join(sorted(list(allowed_l1)))
    l2_candidates = build_llm_l2_candidates(row, fallback, allowed_l2_service)
    l2_candidate_set = {str(x).strip().lower() for x in l2_candidates if str(x).strip()}
    allowed_l2_txt = ", ".join(l2_candidates)
    review_excerpt = str(row.get("text", "") or "")[: min(int(RELABEL_REVIEW_CHARS), int(LLM_PROMPT_REVIEW_CHARS))]
    embed_hint = ""
    embed_top1 = str(row.get("embed_top1_label", "")).strip()
    if embed_top1:
        embed_hint = (
            f"embed_top1={embed_top1}({float(row.get('embed_top1_sim', 0.0)):.3f}), "
            f"embed_top2={str(row.get('embed_top2_label', '')).strip()}({float(row.get('embed_top2_sim', 0.0)):.3f}), "
            f"gap={float(row.get('embed_score_gap', 0.0)):.3f}\n"
        )
    prompt = (
        "You are a strict labeling validator.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Required keys:\n"
        "action,l1_label,l2_label_top1,l2_label_top2,confidence\n"
        "Optional key:\n"
        "reason_short\n"
        "action must be one of KEEP,MODIFY,ADD,REJECT.\n"
        "KEEP: keep rule labels unchanged.\n"
        "MODIFY: replace labels.\n"
        "ADD: keep rule top1 and only add top2.\n"
        "REJECT: fallback to conservative label.\n"
        "For MODIFY and ADD, choose l2 labels only from allowed_l2_food_service_candidates.\n"
        "No markdown. No extra words.\n"
        f"allowed_l1: {allowed_l1_txt}\n"
        f"allowed_l2_food_service_candidates: {allowed_l2_txt}\n"
        f"name: {row.get('name', '')}\n"
        f"categories: {row.get('categories', '')}\n"
        f"avg_review_stars: {row.get('avg_review_stars', '')}\n"
        f"n_reviews: {row.get('n_reviews', '')}\n"
        f"rule_l1: {fallback.get('l1_label', '')}\n"
        f"rule_l2_top1: {fallback.get('l2_label_top1', '')}\n"
        f"rule_l2_top2: {fallback.get('l2_label_top2', '')}\n"
        f"rule_l2_score_top1: {fallback.get('l2_score_top1', 0)}\n"
        f"rule_l2_score_top2: {fallback.get('l2_score_top2', 0)}\n"
        f"rule_trigger_reason: {fallback.get('llm_trigger_reason', '')}\n"
        f"{embed_hint}"
        f"review_excerpt: {review_excerpt}\n"
    )

    from_label = f"{fallback.get('l1_label', '')}|{fallback.get('l2_label_top1', '')}"
    def fallback_keep(status: str, error_code: str, note: str, raw_head: str = "") -> dict[str, Any]:
        return make_llm_fallback_keep(
            fallback,
            from_label=from_label,
            status=status,
            error_code=error_code,
            reason_note=note,
            raw_head=raw_head,
        )

    raw_used = ""
    try:
        raw_used = ollama_generate(prompt)
        parsed = try_parse_json(raw_used)
    except Exception as exc:
        return fallback_keep(
            status="http_error",
            error_code="ollama_http_error",
            note=f"llm_error={str(exc)[:200]}",
        )

    if not parsed and OLLAMA_JSON_RETRY_ON_PARSE_FAIL:
        retry_prompt = (
            "Return one-line JSON object only.\n"
            "Keys exactly: action,l1_label,l2_label_top1,l2_label_top2,confidence,reason_short.\n"
            "action in {KEEP,MODIFY,ADD,REJECT}.\n"
            f"allowed_l1: {allowed_l1_txt}\n"
            f"allowed_l2_food_service_candidates: {allowed_l2_txt}\n"
            f"name: {row.get('name', '')}\n"
            f"rule_l1: {fallback.get('l1_label', '')}\n"
            f"rule_l2_top1: {fallback.get('l2_label_top1', '')}\n"
            f"rule_l2_top2: {fallback.get('l2_label_top2', '')}\n"
            f"review_excerpt: {review_excerpt[:220]}\n"
        )
        try:
            raw_used = ollama_generate(retry_prompt, num_predict=OLLAMA_JSON_RETRY_NUM_PREDICT)
            parsed = try_parse_json(raw_used)
        except Exception as exc:
            return fallback_keep(
                status="http_error",
                error_code="ollama_retry_http_error",
                note=f"llm_retry_error={str(exc)[:200]}",
            )

    if not parsed:
        return fallback_keep(
            status="parse_fail",
            error_code="json_parse_empty",
            note="llm_parse_empty",
            raw_head=compact_text(raw_used, 200),
        )

    is_valid, normalized, schema_error = validate_llm_payload(
        parsed=parsed,
        fallback=fallback,
        allowed_l1=allowed_l1,
        allowed_l2_service=allowed_l2_service,
        candidate_l2_service=l2_candidate_set,
    )

    if (not is_valid) and OLLAMA_JSON_RETRY_ON_PARSE_FAIL:
        schema_retry_prompt = (
            "Output JSON only. Fix schema error and retry.\n"
            f"schema_error: {schema_error}\n"
            "Keys exactly: action,l1_label,l2_label_top1,l2_label_top2,confidence,reason_short.\n"
            "action in {KEEP,MODIFY,ADD,REJECT}.\n"
            f"allowed_l1: {allowed_l1_txt}\n"
            f"allowed_l2_food_service_candidates: {allowed_l2_txt}\n"
            f"rule_l1: {fallback.get('l1_label', '')}\n"
            f"rule_l2_top1: {fallback.get('l2_label_top1', '')}\n"
            f"rule_l2_top2: {fallback.get('l2_label_top2', '')}\n"
            f"review_excerpt: {review_excerpt[:180]}\n"
        )
        try:
            raw_used = ollama_generate(schema_retry_prompt, num_predict=OLLAMA_JSON_RETRY_NUM_PREDICT)
            parsed_retry = try_parse_json(raw_used)
            if parsed_retry:
                is_valid, normalized, schema_error = validate_llm_payload(
                    parsed=parsed_retry,
                    fallback=fallback,
                    allowed_l1=allowed_l1,
                    allowed_l2_service=allowed_l2_service,
                    candidate_l2_service=l2_candidate_set,
                )
        except Exception:
            pass

    if not is_valid:
        return fallback_keep(
            status="schema_fail",
            error_code=schema_error or "schema_invalid",
            note=f"llm_schema_invalid={schema_error or 'unknown'}",
            raw_head=compact_text(raw_used, 200),
        )

    action = str(normalized.get("action", "KEEP")).strip().upper()
    candidate_l1 = str(normalized.get("l1_label", fallback["l1_label"])).strip().lower()
    candidate_top1 = str(normalized.get("l2_label_top1", fallback.get("l2_label_top1", ""))).strip().lower()
    candidate_top2 = str(normalized.get("l2_label_top2", fallback.get("l2_label_top2", ""))).strip().lower()

    final_l1 = str(fallback.get("l1_label", "uncertain"))
    final_top1 = str(fallback.get("l2_label_top1", ""))
    final_top2 = str(fallback.get("l2_label_top2", ""))
    fallback_l1_norm = final_l1.strip().lower()
    fallback_top1_norm = final_top1.strip().lower()
    fallback_top2_norm = final_top2.strip().lower()
    action_validated = False
    applied_note = "keep_rule"

    if action == "MODIFY":
        final_l1 = candidate_l1
        if final_l1 != "food_service":
            final_top1 = ""
            final_top2 = ""
            action_validated = True
            applied_note = "modify_l1_non_food_service"
        else:
            if candidate_top1 in allowed_l2_service:
                final_top1 = candidate_top1
                if candidate_top2 in allowed_l2_service and candidate_top2 != final_top1:
                    final_top2 = candidate_top2
                else:
                    final_top2 = ""
                action_validated = True
                applied_note = "modify_l2"
    elif action == "ADD":
        if final_l1 == "food_service":
            add_label = ""
            if candidate_top2 in allowed_l2_service and candidate_top2 != final_top1:
                add_label = candidate_top2
            elif candidate_top1 in allowed_l2_service and candidate_top1 != final_top1:
                add_label = candidate_top1
            if add_label:
                final_top2 = add_label
                action_validated = True
                applied_note = f"add_top2_{add_label}"
    elif action == "REJECT":
        if final_l1 == "food_service":
            if "restaurants_general" in allowed_l2_service:
                final_top1 = "restaurants_general"
            final_top2 = ""
        else:
            final_l1 = "uncertain" if "uncertain" in allowed_l1 else final_l1
            final_top1 = ""
            final_top2 = ""
        action_validated = True
        applied_note = "reject_to_conservative"
    else:
        action_validated = True
        applied_note = "keep_rule"

    # Normalize "MODIFY/ADD but no effective label change" to KEEP.
    # This prevents inflated MODIFY counts when model only restates rule output.
    final_l1_norm = str(final_l1).strip().lower()
    final_top1_norm = str(final_top1).strip().lower()
    final_top2_norm = str(final_top2).strip().lower()
    if action in {"MODIFY", "ADD"} and action_validated:
        if (
            final_l1_norm == fallback_l1_norm
            and final_top1_norm == fallback_top1_norm
            and final_top2_norm == fallback_top2_norm
        ):
            action = "KEEP"
            applied_note = "normalize_noop_keep"

    parsed_conf = clamp_confidence(normalized.get("confidence", fallback.get("label_confidence", 2)))
    base_conf = int(fallback.get("label_confidence", 2))
    if action == "KEEP":
        final_conf = base_conf
    elif action == "REJECT":
        final_conf = min(2, base_conf)
    else:
        final_conf = parsed_conf

    final_source = fallback.get("label_source", "rule")
    if action_validated and action in {"MODIFY", "ADD", "REJECT"}:
        final_source = "llm"

    short_reason = str(normalized.get("reason_short", "")).strip()
    if short_reason:
        final_reason = short_reason
    else:
        final_reason = str(fallback.get("label_reason", ""))
    final_reason = f"{final_reason} | llm_action={action}:{applied_note}"[:500]
    final_evidence = str(fallback.get("evidence_terms", ""))[:500]
    to_label = f"{final_l1}|{final_top1}"

    return {
        "l1_label": final_l1,
        "l2_label_top1": final_top1,
        "l2_label_top2": final_top2,
        "label_confidence": clamp_confidence(final_conf),
        "label_reason": final_reason,
        "evidence_terms": final_evidence,
        "rule_hit": fallback.get("rule_hit", ""),
        "l2_source": fallback.get("l2_source", ""),
        "l2_score_top1": float(fallback.get("l2_score_top1", 0.0)),
        "l2_score_top2": float(fallback.get("l2_score_top2", 0.0)),
        "l2_score_gap": float(fallback.get("l2_score_gap", 0.0)),
        "l2_top1_terms": str(fallback.get("l2_top1_terms", "")),
        "l2_top2_terms": str(fallback.get("l2_top2_terms", "")),
        "llm_trigger_reason": str(fallback.get("llm_trigger_reason", "")),
        "llm_priority": int(fallback.get("llm_priority", 0)),
        "needs_llm": False,
        "llm_action": action,
        "from_label": from_label,
        "to_label": to_label,
        "action_validated": bool(action_validated),
        "label_source": final_source,
        "llm_status": "ok",
        "llm_error_code": "",
        "llm_raw_head": compact_text(raw_used, 200),
    }


def build_layered_label_outputs(relabeled_pdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_df = relabeled_pdf.copy()
    review_snippet = audit_df.get("text", pd.Series("", index=audit_df.index))
    audit_df["review_snippet"] = review_snippet.fillna("").astype(str).str.slice(0, 320)
    audit_df["final_label_reason_short"] = (
        audit_df.get("label_reason", pd.Series("", index=audit_df.index))
        .fillna("")
        .astype(str)
        .str.slice(0, 180)
    )

    llm_action_series = (
        audit_df.get("llm_action", pd.Series("NOT_CALLED", index=audit_df.index))
        .fillna("NOT_CALLED")
        .astype(str)
        .str.upper()
    )
    llm_status_series = (
        audit_df.get("llm_status", pd.Series("not_called", index=audit_df.index))
        .fillna("not_called")
        .astype(str)
        .str.lower()
    )
    llm_error_code_series = (
        audit_df.get("llm_error_code", pd.Series("", index=audit_df.index))
        .fillna("")
        .astype(str)
    )
    llm_raw_head_series = (
        audit_df.get("llm_raw_head", pd.Series("", index=audit_df.index))
        .fillna("")
        .astype(str)
        .str.slice(0, 200)
    )
    action_validated_series = (
        audit_df.get("action_validated", pd.Series(False, index=audit_df.index))
        .fillna(False)
        .astype(bool)
    )
    l2_source_series = (
        audit_df.get("l2_source", pd.Series("", index=audit_df.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    label_conf_series = pd.to_numeric(
        audit_df.get("label_confidence", pd.Series(0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0)
    l1_series = audit_df.get("l1_label", pd.Series("", index=audit_df.index)).fillna("").astype(str)

    needs_review = (
        l1_series.eq("uncertain")
        | (label_conf_series <= int(LLM_REVIEW_CONFIDENCE_MAX))
        | l2_source_series.str.contains("fallback", na=False)
        | llm_action_series.eq("REJECT")
        | (llm_action_series.ne("NOT_CALLED") & (~action_validated_series))
        | (llm_action_series.isin(["MODIFY", "ADD", "REJECT"]) & (~action_validated_series))
    )
    audit_df["needs_review"] = needs_review.astype(bool)
    audit_df["llm_action"] = llm_action_series
    audit_df["llm_status"] = llm_status_series
    audit_df["llm_error_code"] = llm_error_code_series
    audit_df["llm_raw_head"] = llm_raw_head_series
    audit_df["action_validated"] = action_validated_series
    audit_df["from_label"] = (
        audit_df.get("from_label", pd.Series("", index=audit_df.index)).fillna("").astype(str)
    )
    audit_df["to_label"] = audit_df.get("to_label", pd.Series("", index=audit_df.index)).fillna("").astype(str)

    # Explicit naming to avoid confusion between final labels and rule evidence labels.
    audit_df["final_l1_label"] = audit_df.get("l1_label", pd.Series("", index=audit_df.index)).fillna("").astype(str)
    audit_df["final_l2_label_top1"] = (
        audit_df.get("l2_label_top1", pd.Series("", index=audit_df.index)).fillna("").astype(str)
    )
    audit_df["final_l2_label_top2"] = (
        audit_df.get("l2_label_top2", pd.Series("", index=audit_df.index)).fillna("").astype(str)
    )
    audit_df["final_label_confidence"] = pd.to_numeric(
        audit_df.get("label_confidence", pd.Series(0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    audit_df["final_label_source"] = (
        audit_df.get("label_source", pd.Series("rule", index=audit_df.index)).fillna("rule").astype(str)
    )
    audit_df["final_label_reason"] = (
        audit_df.get("label_reason", pd.Series("", index=audit_df.index)).fillna("").astype(str)
    )
    audit_df["final_evidence_terms"] = (
        audit_df.get("evidence_terms", pd.Series("", index=audit_df.index)).fillna("").astype(str)
    )

    audit_df["llm_action"] = llm_action_series
    audit_df["llm_action_validated"] = action_validated_series
    audit_df["llm_from_label"] = audit_df["from_label"]
    audit_df["llm_to_label"] = audit_df["to_label"]

    audit_df["rule_l1_label"] = (
        audit_df.get("rule_l1_label", audit_df.get("l1_label", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )
    audit_df["rule_l2_label_top1"] = (
        audit_df.get("rule_l2_label_top1", audit_df.get("l2_label_top1", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )
    audit_df["rule_l2_label_top2"] = (
        audit_df.get("rule_l2_label_top2", audit_df.get("l2_label_top2", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )
    audit_df["rule_label_confidence"] = pd.to_numeric(
        audit_df.get("rule_label_confidence", audit_df.get("label_confidence", pd.Series(0, index=audit_df.index))),
        errors="coerce",
    ).fillna(0).astype(int)
    audit_df["rule_l2_source"] = (
        audit_df.get("rule_l2_source", audit_df.get("l2_source", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )
    audit_df["rule_l2_top1_score"] = pd.to_numeric(
        audit_df.get("rule_l2_score_top1", audit_df.get("l2_score_top1", pd.Series(0.0, index=audit_df.index))),
        errors="coerce",
    ).fillna(0.0)
    audit_df["rule_l2_top2_score"] = pd.to_numeric(
        audit_df.get("rule_l2_score_top2", audit_df.get("l2_score_top2", pd.Series(0.0, index=audit_df.index))),
        errors="coerce",
    ).fillna(0.0)
    audit_df["rule_l2_score_gap"] = pd.to_numeric(
        audit_df.get("rule_l2_score_gap", audit_df.get("l2_score_gap", pd.Series(0.0, index=audit_df.index))),
        errors="coerce",
    ).fillna(0.0)
    audit_df["rule_top1_evidence_terms"] = (
        audit_df.get("rule_l2_top1_terms", audit_df.get("l2_top1_terms", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )
    audit_df["rule_top2_evidence_terms"] = (
        audit_df.get("rule_l2_top2_terms", audit_df.get("l2_top2_terms", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )
    audit_df["rule_label_reason"] = (
        audit_df.get("rule_label_reason", audit_df.get("label_reason", pd.Series("", index=audit_df.index)))
        .fillna("")
        .astype(str)
    )

    final_cols = [
        "business_id",
        "name",
        "city",
        "categories",
        "stars",
        "review_count",
        "avg_review_stars",
        "n_reviews",
        "final_l1_label",
        "final_l2_label_top1",
        "final_l2_label_top2",
        "final_label_confidence",
        "final_label_source",
        "llm_action",
        "llm_status",
        "llm_error_code",
        "needs_review",
        "rule_hit",
        "llm_trigger_reason",
        "final_evidence_terms",
        "final_label_reason_short",
    ]
    final_cols = [c for c in final_cols if c in audit_df.columns]
    final_df = audit_df[final_cols].copy()

    review_cols = [
        "business_id",
        "name",
        "city",
        "categories",
        "final_l1_label",
        "final_l2_label_top1",
        "final_l2_label_top2",
        "final_label_confidence",
        "final_label_source",
        "llm_action",
        "llm_status",
        "llm_error_code",
        "llm_raw_head",
        "llm_action_validated",
        "llm_from_label",
        "llm_to_label",
        "rule_l1_label",
        "rule_l2_label_top1",
        "rule_l2_label_top2",
        "rule_label_confidence",
        "rule_hit",
        "llm_trigger_reason",
        "rule_l2_source",
        "rule_l2_top1_score",
        "rule_l2_top2_score",
        "rule_l2_score_gap",
        "rule_top1_evidence_terms",
        "rule_top2_evidence_terms",
        "final_evidence_terms",
        "rule_label_reason",
        "final_label_reason",
        "review_snippet",
    ]
    review_cols = [c for c in review_cols if c in audit_df.columns]
    review_df = audit_df.loc[audit_df["needs_review"]].copy()
    sort_cols = [c for c in ["final_label_confidence", "n_reviews"] if c in review_df.columns]
    if sort_cols:
        review_df = review_df.sort_values(
            by=sort_cols,
            ascending=[True if c == "final_label_confidence" else False for c in sort_cols],
        )
    review_df = review_df[review_cols].copy()

    return final_df, review_df, audit_df


def add_final_label_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        "l1_label": "final_l1_label",
        "l2_label_top1": "final_l2_label_top1",
        "l2_label_top2": "final_l2_label_top2",
        "label_confidence": "final_label_confidence",
        "label_source": "final_label_source",
        "label_reason": "final_label_reason",
        "evidence_terms": "final_evidence_terms",
    }
    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    return out


def _pick_first_existing_series(df: pd.DataFrame, candidates: list[str], fill_value: Any = np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([fill_value] * len(df), index=df.index)


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    txt = series.fillna("").astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "yes", "y", "t"})


def build_cluster_input(relabeled_pdf: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = add_final_label_alias_columns(relabeled_pdf)
    mode = str(CLUSTER_INPUT_MODE).strip().lower()
    if mode not in {"strict", "food_service"}:
        mode = "strict"

    stats: dict[str, Any] = {
        "cluster_input_mode": mode,
        "n_source": int(len(df)),
        "n_after_food_service_filter": 0,
        "n_after_conf_filter": 0,
        "n_after_general_filter": 0,
        "n_after_review_filter": 0,
    }

    l1_series = _pick_first_existing_series(df, ["final_l1_label", "l1_label"], fill_value="")
    l1_norm = l1_series.fillna("").astype(str).str.strip().str.lower()
    mask_food_service = l1_norm.eq("food_service")
    if KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER:
        mask = mask_food_service.copy()
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    stats["n_after_food_service_filter"] = int(mask.sum())

    if mode == "strict":
        conf_series = _pick_first_existing_series(
            df, ["final_label_confidence", "label_confidence"], fill_value=0.0
        )
        conf_num = pd.to_numeric(conf_series, errors="coerce").fillna(0.0)
        mask = mask & (conf_num >= float(CLUSTER_STRICT_MIN_CONFIDENCE))
        stats["n_after_conf_filter"] = int(mask.sum())

        if CLUSTER_STRICT_EXCLUDE_GENERAL:
            l2_series = _pick_first_existing_series(
                df, ["final_l2_label_top1", "l2_label_top1"], fill_value=""
            )
            l2_norm = l2_series.fillna("").astype(str).str.strip().str.lower()
            mask = mask & (~l2_norm.eq("restaurants_general"))
        stats["n_after_general_filter"] = int(mask.sum())

        if CLUSTER_STRICT_EXCLUDE_REVIEW_QUEUE and "needs_review" in df.columns:
            needs_review = _as_bool_series(df["needs_review"])
            mask = mask & (~needs_review)
        stats["n_after_review_filter"] = int(mask.sum())
    else:
        stats["n_after_conf_filter"] = int(mask.sum())
        stats["n_after_general_filter"] = int(mask.sum())
        stats["n_after_review_filter"] = int(mask.sum())

    cluster_pdf = df.loc[mask].copy().reset_index(drop=True)
    stats["n_cluster_input"] = int(len(cluster_pdf))
    return cluster_pdf, stats


def relabel_businesses(
    pdf: pd.DataFrame,
    profile: str,
    progress_csv_path: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    allowed_l1, allowed_l2_service = load_label_whitelist(LABEL_CONFIG_DIR)
    brand_terms = load_brand_terms(LABEL_CONFIG_DIR)

    service_terms = sorted(list(set(BASE_SERVICE_TERMS + brand_terms["food_service"])))
    retail_terms = sorted(list(set(BASE_RETAIL_TERMS + brand_terms["food_retail"])))
    non_food_terms = sorted(list(set(BASE_NON_FOOD_TERMS + brand_terms["non_food"])))

    print(f"[CONFIG] relabel_use_llm={RELABEL_USE_LLM}")
    print(f"[CONFIG] relabel_embed_strategy={get_relabel_embed_strategy()}")
    print(f"[CONFIG] whitelist_l1={sorted(list(allowed_l1))}")
    print(f"[CONFIG] whitelist_l2_food_service_count={len(allowed_l2_service)}")

    rule_rows = []
    for _, row in pdf.iterrows():
        rule_rows.append(
            rule_relabel_one(
                row,
                allowed_l1=allowed_l1,
                allowed_l2_service=allowed_l2_service,
                service_terms=service_terms,
                retail_terms=retail_terms,
                non_food_terms=non_food_terms,
            )
        )
    out = pd.concat([pdf.reset_index(drop=True), pd.DataFrame(rule_rows)], axis=1)

    snapshot_cols = [
        "l1_label",
        "l2_label_top1",
        "l2_label_top2",
        "label_confidence",
        "label_reason",
        "evidence_terms",
        "rule_hit",
        "l2_source",
        "l2_score_top1",
        "l2_score_top2",
        "l2_score_gap",
        "l2_top1_terms",
        "l2_top2_terms",
        "llm_trigger_reason",
        "llm_priority",
        "needs_llm",
        "label_source",
    ]
    for col in snapshot_cols:
        if col in out.columns:
            out[f"rule_{col}"] = out[col]

    stats: dict[str, int] = {
        "n_rule": int(len(out)),
        "n_embed_candidate_pool": 0,
        "n_embed_called": 0,
        "n_embed_applied": 0,
        "n_embed_llm_skipped": 0,
        "n_embed_forced_llm": 0,
        "n_llm_called": 0,
        "n_llm_validated": 0,
        "n_llm_applied": 0,
        "n_llm_action_keep": 0,
        "n_llm_action_modify": 0,
        "n_llm_action_add": 0,
        "n_llm_action_reject": 0,
        "n_llm_parse_fail": 0,
        "n_llm_schema_fail": 0,
        "n_llm_http_error": 0,
        "n_llm_parallel_error": 0,
        "n_llm_fallback_keep": 0,
    }

    embed_stats = apply_embedding_recall_to_relabels(
        out=out,
        allowed_l2_service=allowed_l2_service,
        profile=profile,
        cache_dir=cache_dir,
    )
    for k, v in embed_stats.items():
        stats[k] = int(v)

    if RELABEL_WRITE_IN_PROGRESS and progress_csv_path is not None:
        try:
            out.to_csv(progress_csv_path, index=False)
            print(f"[INFO] wrote in-progress relabel snapshot: {progress_csv_path}")
        except Exception as exc:
            print(f"[WARN] failed to write in-progress relabel snapshot: {exc}")

    if not RELABEL_USE_LLM:
        return out, stats

    llm_budget = RELABEL_MAX_LLM_CALLS_SAMPLE if profile == "sample" else RELABEL_MAX_LLM_CALLS_FULL
    cand = out[out["needs_llm"]].copy()
    if cand.empty or llm_budget <= 0:
        return out, stats
    cand = cand.sort_values(
        by=["llm_priority", "label_confidence", "n_reviews"],
        ascending=[False, True, False],
    ).head(llm_budget)

    llm_workers_cfg = RELABEL_LLM_WORKERS_SAMPLE if profile == "sample" else RELABEL_LLM_WORKERS_FULL
    llm_workers = max(1, min(int(llm_workers_cfg), int(len(cand))))
    print(
        f"[STEP] llm relabel candidates={len(cand)} budget={llm_budget} "
        f"workers={llm_workers}"
    )

    job_args: list[tuple[int, pd.Series, dict[str, Any]]] = []
    for idx in cand.index.tolist():
        row = out.loc[idx]
        fallback = {
            "l1_label": row["l1_label"],
            "l2_label_top1": row["l2_label_top1"],
            "l2_label_top2": row["l2_label_top2"],
            "label_confidence": int(row["label_confidence"]),
            "label_reason": str(row["label_reason"]),
            "evidence_terms": str(row["evidence_terms"]),
            "rule_hit": str(row["rule_hit"]),
            "l2_source": str(row.get("l2_source", "")),
            "l2_score_top1": float(row.get("l2_score_top1", 0.0)),
            "l2_score_top2": float(row.get("l2_score_top2", 0.0)),
            "l2_score_gap": float(row.get("l2_score_gap", 0.0)),
            "l2_top1_terms": str(row.get("l2_top1_terms", "")),
            "l2_top2_terms": str(row.get("l2_top2_terms", "")),
            "llm_trigger_reason": str(row.get("llm_trigger_reason", "")),
            "llm_priority": int(row.get("llm_priority", 0)),
            "needs_llm": False,
            "llm_status": "not_called",
            "llm_error_code": "",
            "llm_raw_head": "",
            "label_source": str(row.get("label_source", "rule")),
        }
        job_args.append((idx, row, fallback))

    stats["n_llm_called"] = int(len(job_args))
    llm_total = int(len(job_args))
    progress_every = max(1, int(RELABEL_PROGRESS_EVERY_N))
    flush_every = max(1, int(RELABEL_FLUSH_EVERY_N))
    progress_state: dict[str, float] = {
        "completed": 0.0,
        "start_ts": time.time(),
    }

    def maybe_report_and_flush(force: bool = False) -> None:
        completed = int(progress_state["completed"])
        if completed <= 0 and not force:
            return
        elapsed = max(1e-6, time.time() - float(progress_state["start_ts"]))
        rate = completed / elapsed
        remaining = max(0, llm_total - completed)
        eta = (remaining / rate) if rate > 0 else float("inf")

        should_report = force or (completed % progress_every == 0)
        if should_report:
            eta_txt = "n/a" if eta == float("inf") else f"{eta:.1f}s"
            print(
                f"[PROGRESS] llm relabel {completed}/{llm_total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta_txt}"
            )

        if RELABEL_WRITE_IN_PROGRESS and progress_csv_path is not None:
            should_flush = force or (completed % flush_every == 0)
            if should_flush:
                try:
                    out.to_csv(progress_csv_path, index=False)
                    if force:
                        print(f"[INFO] wrote final in-progress relabel snapshot: {progress_csv_path}")
                except Exception as exc:
                    print(f"[WARN] failed to flush in-progress relabel snapshot: {exc}")

    def safe_default_keep(fallback: dict[str, Any], error_text: str) -> dict[str, Any]:
        from_label = f"{fallback.get('l1_label', '')}|{fallback.get('l2_label_top1', '')}"
        return {
            **fallback,
            "label_reason": f"{fallback.get('label_reason', '')} | llm_parallel_error={error_text[:200]}",
            "needs_llm": False,
            "llm_action": "KEEP",
            "from_label": from_label,
            "to_label": from_label,
            "action_validated": False,
            "label_source": fallback.get("label_source", "rule"),
            "llm_status": "parallel_error",
            "llm_error_code": "llm_parallel_exception",
            "llm_raw_head": "",
        }

    def apply_llm_result(idx: int, llm_out: dict[str, Any]) -> None:
        if llm_out.get("label_source") == "llm":
            stats["n_llm_applied"] += 1
        if bool(llm_out.get("action_validated", False)):
            stats["n_llm_validated"] += 1
        action_key = f"n_llm_action_{str(llm_out.get('llm_action', 'KEEP')).strip().lower()}"
        if action_key in stats:
            stats[action_key] += 1
        llm_status = str(llm_out.get("llm_status", "")).strip().lower()
        if llm_status == "parse_fail":
            stats["n_llm_parse_fail"] += 1
        elif llm_status == "schema_fail":
            stats["n_llm_schema_fail"] += 1
        elif llm_status == "http_error":
            stats["n_llm_http_error"] += 1
        elif llm_status == "parallel_error":
            stats["n_llm_parallel_error"] += 1
        if llm_status in {"parse_fail", "schema_fail", "http_error", "parallel_error"}:
            stats["n_llm_fallback_keep"] += 1
        for key in [
            "l1_label",
            "l2_label_top1",
            "l2_label_top2",
            "label_confidence",
            "label_reason",
            "evidence_terms",
            "l2_source",
            "l2_score_top1",
            "l2_score_top2",
            "l2_score_gap",
            "l2_top1_terms",
            "l2_top2_terms",
            "llm_trigger_reason",
            "llm_priority",
            "needs_llm",
            "label_source",
            "llm_action",
            "from_label",
            "to_label",
            "action_validated",
            "llm_status",
            "llm_error_code",
            "llm_raw_head",
        ]:
            out.at[idx, key] = llm_out[key]
        progress_state["completed"] = float(progress_state["completed"] + 1.0)
        maybe_report_and_flush(force=False)

    if llm_workers <= 1:
        for idx, row, fallback in job_args:
            llm_out = llm_relabel_one(
                row=row,
                allowed_l1=allowed_l1,
                allowed_l2_service=allowed_l2_service,
                fallback=fallback,
            )
            apply_llm_result(idx, llm_out)
    else:
        by_idx: dict[int, tuple[pd.Series, dict[str, Any]]] = {
            idx: (row, fallback) for idx, row, fallback in job_args
        }
        with ThreadPoolExecutor(max_workers=llm_workers) as ex:
            fut_map = {
                ex.submit(
                    llm_relabel_one,
                    row=row,
                    allowed_l1=allowed_l1,
                    allowed_l2_service=allowed_l2_service,
                    fallback=fallback,
                ): idx
                for idx, row, fallback in job_args
            }
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                row, fallback = by_idx[idx]
                try:
                    llm_out = fut.result()
                except Exception as exc:
                    if RELABEL_LLM_RETRY_SERIAL_ON_FAILURE:
                        try:
                            llm_out = llm_relabel_one(
                                row=row,
                                allowed_l1=allowed_l1,
                                allowed_l2_service=allowed_l2_service,
                                fallback=fallback,
                            )
                        except Exception as exc2:
                            llm_out = safe_default_keep(fallback, f"{exc}; retry={exc2}")
                    else:
                        llm_out = safe_default_keep(fallback, str(exc))
                apply_llm_result(idx, llm_out)

    maybe_report_and_flush(force=True)
    return out, stats


def run_cluster_label_stage(run_dir: Path) -> None:
    script_08 = Path(__file__).with_name("08_cluster_label_ollama.py")
    if not script_08.exists():
        print(f"[WARN] 08 script not found: {script_08}")
        return
    env = dict(os.environ)
    env["SOURCE_07_RUN_DIR"] = str(run_dir)
    print(f"[STEP] run 08 cluster label on {run_dir}")
    result = subprocess.run([sys.executable, str(script_08)], env=env, check=False)
    if result.returncode != 0:
        print(f"[WARN] 08 script exited with code={result.returncode}")


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = resolve_profile()
    paths = build_output_paths(run_id, cfg["profile"])
    model_name, model_source = choose_model_name()

    print(f"[CONFIG] profile={cfg['profile']}")
    print(f"[CONFIG] use_bge_m3={USE_BGE_M3}, model={model_name}")
    print(f"[CONFIG] model_source={model_source}")
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
    print(f"[COUNT] businesses_before_relabel={len(pdf)}")
    if len(pdf) == 0:
        print("[ERROR] no businesses collected")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return

    print("[STEP] relabel businesses (rule + llm)")
    relabeled_pdf, relabel_stats = relabel_businesses(
        pdf,
        profile=cfg["profile"],
        progress_csv_path=paths["relabels_inprogress_csv"],
        cache_dir=paths["cache_dir"],
    )
    relabeled_pdf = add_final_label_alias_columns(relabeled_pdf)
    relabeled_pdf.to_csv(paths["relabels_csv"], index=False)

    labels_final_df = relabeled_pdf
    labels_review_df = relabeled_pdf.iloc[0:0].copy()
    labels_audit_df = relabeled_pdf
    if WRITE_LAYERED_LABEL_OUTPUTS:
        labels_final_df, labels_review_df, labels_audit_df = build_layered_label_outputs(relabeled_pdf)
        labels_final_df.to_csv(paths["labels_final_csv"], index=False)
        labels_review_df.to_csv(paths["labels_review_csv"], index=False)
        labels_audit_df.to_csv(paths["labels_audit_csv"], index=False)

    label_stats = (
        relabeled_pdf.groupby(["l1_label", "label_source"], dropna=False)
        .size()
        .reset_index(name="n_businesses")
        .sort_values("n_businesses", ascending=False)
    )
    label_stats.to_csv(paths["label_stats_csv"], index=False)
    print("[COUNT] relabel distribution:")
    print(label_stats.to_string(index=False))

    cluster_pdf, cluster_input_stats = build_cluster_input(relabeled_pdf)
    print(
        "[CONFIG] cluster_input_mode="
        f"{cluster_input_stats.get('cluster_input_mode', 'strict')}"
        f", keep_only_food_service={KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER}"
        f", strict_min_conf={CLUSTER_STRICT_MIN_CONFIDENCE}"
    )
    print(
        "[COUNT] cluster input:"
        f" source={cluster_input_stats.get('n_source', 0)}"
        f" -> food_service={cluster_input_stats.get('n_after_food_service_filter', 0)}"
        f" -> conf={cluster_input_stats.get('n_after_conf_filter', 0)}"
        f" -> non_general={cluster_input_stats.get('n_after_general_filter', 0)}"
        f" -> no_review={cluster_input_stats.get('n_after_review_filter', 0)}"
    )
    print(f"[COUNT] businesses_for_cluster={len(cluster_pdf)}")
    if len(cluster_pdf) < 2:
        print("[ERROR] not enough businesses after relabel filter")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return
    cluster_pdf.drop(columns=["text"], errors="ignore").to_csv(paths["cluster_input_csv"], index=False)

    cache_file = make_cache_file(paths["cache_dir"], model_name, cfg)
    business_ids = cluster_pdf["business_id"].astype(str).tolist()
    embeddings = None
    actual_model_name = model_name
    actual_batch_size = MANUAL_BATCH_SIZE
    if REUSE_EMBEDDINGS:
        embeddings = try_load_cached_embeddings(cache_file, business_ids)

    if embeddings is None:
        print("[STEP] compute embeddings")
        embeddings, actual_model_name, actual_batch_size = compute_embeddings(
            cluster_pdf["text"].tolist(), model_name
        )
        save_cache_embeddings(cache_file, business_ids, embeddings)
        print(f"[INFO] saved embeddings cache to {cache_file}")

    print("[STEP] kmeans clustering")
    from sklearn.cluster import KMeans

    cluster_k = min(int(cfg["cluster_k"]), len(cluster_pdf))
    if cluster_k < 2:
        print("[ERROR] not enough businesses for clustering")
        reviews.unpersist()
        biz_text.unpersist()
        biz.unpersist()
        spark.stop()
        return

    cluster_features = np.asarray(embeddings, dtype=np.float32)
    hybrid_l2_feature_dim = 0
    if USE_HYBRID_CLUSTER_FEATURES:
        l2_mat, l2_feature_labels = build_l2_feature_matrix(cluster_pdf)
        hybrid_l2_feature_dim = int(l2_mat.shape[1])
        if hybrid_l2_feature_dim > 0:
            cluster_features = np.hstack([cluster_features, l2_mat]).astype(np.float32)
            print(
                "[INFO] hybrid cluster features enabled: "
                f"l2_dim={hybrid_l2_feature_dim}, l2_weight={HYBRID_L2_FEATURE_WEIGHT}, "
                f"l2_top2={HYBRID_USE_L2_TOP2}"
            )
            print(f"[INFO] hybrid l2 labels: {', '.join(l2_feature_labels)}")
        else:
            print("[INFO] hybrid cluster features skipped: no usable l2 labels")

    kmeans = KMeans(n_clusters=cluster_k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(cluster_features)
    cluster_pdf["cluster"] = labels

    print("[STEP] tf-idf keywords per cluster")
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=int(cfg["tfidf_max_features"]),
        min_df=int(cfg["tfidf_min_df"]),
        max_df=float(cfg["tfidf_max_df"]),
        ngram_range=tuple(cfg["tfidf_ngram_range"]),
        stop_words="english",
    )
    X = vectorizer.fit_transform(cluster_pdf["text"].tolist())
    terms = vectorizer.get_feature_names_out()

    keyword_rows = []
    summary_rows = []
    example_rows = []
    top_terms = int(cfg["top_terms_per_cluster"])
    for c in range(cluster_k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        cluster_df = cluster_pdf.iloc[idx]
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
                    "l1_label": row["l1_label"],
                    "l2_label_top1": row["l2_label_top1"],
                    "n_reviews": int(row["n_reviews"]),
                    "avg_review_stars": float(row["avg_review_stars"]),
                }
            )

    cluster_pdf.drop(columns=["text"]).to_csv(paths["assignments_csv"], index=False)
    pd.DataFrame(keyword_rows).to_csv(paths["cluster_keywords_csv"], index=False)
    pd.DataFrame(summary_rows).to_csv(paths["cluster_summary_csv"], index=False)
    pd.DataFrame(example_rows).to_csv(paths["cluster_examples_csv"], index=False)

    relabel_n_total = int(len(relabeled_pdf))
    relabel_uncertain_rate = float(
        (relabeled_pdf["l1_label"].fillna("").astype(str).str.lower() == "uncertain").mean()
    ) if relabel_n_total > 0 else 0.0
    relabel_review_rate = float(int(len(labels_review_df)) / max(1, relabel_n_total))
    llm_action_series = labels_audit_df.get("llm_action", pd.Series("NOT_CALLED", index=labels_audit_df.index))
    llm_called_mask = llm_action_series.fillna("").astype(str).str.upper() != "NOT_CALLED"
    llm_called_n = int(llm_called_mask.sum())
    llm_from_series = labels_audit_df.get("from_label", pd.Series("", index=labels_audit_df.index)).fillna("").astype(str)
    llm_to_series = labels_audit_df.get("to_label", pd.Series("", index=labels_audit_df.index)).fillna("").astype(str)
    llm_nochange_n = int((llm_called_mask & (llm_from_series == llm_to_series)).sum())

    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "profile": cfg["profile"],
                "model_requested": model_name,
                "model_source_requested": model_source,
                "model_used": actual_model_name,
                "batch_size_used": actual_batch_size,
                "cluster_k_used": cluster_k,
                "n_businesses_before_relabel": len(relabeled_pdf),
                "n_businesses_after_relabel_filter": len(cluster_pdf),
                "keep_only_food_service_for_cluster": KEEP_ONLY_FOOD_SERVICE_FOR_CLUSTER,
                "cluster_input_mode": cluster_input_stats.get("cluster_input_mode", "strict"),
                "cluster_strict_min_confidence": float(CLUSTER_STRICT_MIN_CONFIDENCE),
                "cluster_strict_exclude_review_queue": bool(CLUSTER_STRICT_EXCLUDE_REVIEW_QUEUE),
                "cluster_strict_exclude_general": bool(CLUSTER_STRICT_EXCLUDE_GENERAL),
                "cluster_input_source_n": int(cluster_input_stats.get("n_source", len(relabeled_pdf))),
                "cluster_input_after_food_service_n": int(
                    cluster_input_stats.get("n_after_food_service_filter", len(cluster_pdf))
                ),
                "cluster_input_after_conf_n": int(cluster_input_stats.get("n_after_conf_filter", len(cluster_pdf))),
                "cluster_input_after_general_n": int(
                    cluster_input_stats.get("n_after_general_filter", len(cluster_pdf))
                ),
                "cluster_input_after_review_n": int(
                    cluster_input_stats.get("n_after_review_filter", len(cluster_pdf))
                ),
                "relabel_use_llm": RELABEL_USE_LLM,
                "relabel_embed_recall_enabled": RELABEL_USE_EMBED_RECALL,
                "relabel_embed_strategy": get_relabel_embed_strategy(),
                "relabel_embed_candidate_pool": relabel_stats.get("n_embed_candidate_pool", 0),
                "relabel_embed_called": relabel_stats.get("n_embed_called", 0),
                "relabel_embed_applied": relabel_stats.get("n_embed_applied", 0),
                "relabel_embed_llm_skipped": relabel_stats.get("n_embed_llm_skipped", 0),
                "relabel_embed_forced_llm": relabel_stats.get("n_embed_forced_llm", 0),
                "relabel_embed_cache_miss": relabel_stats.get("n_embed_cache_miss", 0),
                "relabel_llm_called": relabel_stats["n_llm_called"],
                "relabel_llm_validated": relabel_stats.get("n_llm_validated", 0),
                "relabel_llm_applied": relabel_stats["n_llm_applied"],
                "relabel_llm_action_keep": relabel_stats.get("n_llm_action_keep", 0),
                "relabel_llm_action_modify": relabel_stats.get("n_llm_action_modify", 0),
                "relabel_llm_action_add": relabel_stats.get("n_llm_action_add", 0),
                "relabel_llm_action_reject": relabel_stats.get("n_llm_action_reject", 0),
                "relabel_llm_parse_fail": relabel_stats.get("n_llm_parse_fail", 0),
                "relabel_llm_schema_fail": relabel_stats.get("n_llm_schema_fail", 0),
                "relabel_llm_http_error": relabel_stats.get("n_llm_http_error", 0),
                "relabel_llm_parallel_error": relabel_stats.get("n_llm_parallel_error", 0),
                "relabel_llm_fallback_keep": relabel_stats.get("n_llm_fallback_keep", 0),
                "relabel_llm_valid_rate": float(
                    int(relabel_stats.get("n_llm_validated", 0)) / max(1, int(relabel_stats.get("n_llm_called", 0)))
                ),
                "relabel_llm_nochange": llm_nochange_n,
                "relabel_llm_nochange_rate": float(llm_nochange_n / max(1, llm_called_n)),
                "relabel_review_queue": int(len(labels_review_df)),
                "relabel_review_rate": relabel_review_rate,
                "relabel_uncertain_rate": relabel_uncertain_rate,
                "use_hybrid_cluster_features": USE_HYBRID_CLUSTER_FEATURES,
                "hybrid_l2_feature_weight": HYBRID_L2_FEATURE_WEIGHT,
                "hybrid_l2_feature_dim": hybrid_l2_feature_dim,
                "hybrid_use_l2_top2": HYBRID_USE_L2_TOP2,
                "hybrid_ignore_general_l2": HYBRID_IGNORE_GENERAL_L2,
                "cache_file": str(cache_file),
            }
        ]
    ).to_csv(paths["run_meta_csv"], index=False)

    print(f"[INFO] wrote {paths['relabels_csv']}")
    if RELABEL_WRITE_IN_PROGRESS:
        print(f"[INFO] in-progress relabel snapshot path: {paths['relabels_inprogress_csv']}")
    if WRITE_LAYERED_LABEL_OUTPUTS:
        print(f"[INFO] wrote {paths['labels_final_csv']}")
        print(f"[INFO] wrote {paths['labels_review_csv']}")
        print(f"[INFO] wrote {paths['labels_audit_csv']}")
    print(f"[INFO] wrote {paths['label_stats_csv']}")
    print(f"[INFO] wrote {paths['cluster_input_csv']}")
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

    if RUN_08_AFTER_CLUSTER:
        run_cluster_label_stage(paths["run_dir"])


if __name__ == "__main__":
    main()
