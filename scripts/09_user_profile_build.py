from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

# Keep sentence-transformers on PyTorch path in mixed TensorFlow/Keras setups.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


# Runtime
RUN_PROFILE = os.getenv("RUN_PROFILE_OVERRIDE", "full").strip().lower() or "full"  # "sample" | "full"
RUN_TAG = "stage09_user_profile_build"
RANDOM_SEED = 42

# Paths
PARQUET_BASE = Path(r"D:/5006 BDA project/data/parquet")
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/09_user_profiles")
CACHE_ROOT = Path(r"D:/5006 BDA project/data/cache/user_profile_embeddings")

# Scope
TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
SAMPLE_MAX_BUSINESSES = int(os.getenv("PROFILE_SAMPLE_MAX_BUSINESSES", "600").strip() or 600)

# Split (strict no-leak for profile side)
HOLDOUT_PER_USER = 2  # leave last 2 interactions for valid/test

# Time windows
PRIMARY_WINDOW_MONTHS = 12
FALLBACK_WINDOW_MONTHS = 24
PRIMARY_MIN_SENTENCES = 8

# Tier definition (agreed in discussion)
TIER_COLD_MAX = 1
TIER_LIGHT_MAX = 5

# High-recall text filter (remove obvious noise only)
TEXT_MIN_CHARS = 20
TEXT_MAX_CHARS = 1200
TEXT_MIN_WORDS = 6
TEXT_MAX_WORDS = 220
TEXT_MIN_ALPHA_RATIO = 0.45
MIN_SENTENCE_CHARS = 20
MIN_SENTENCE_WORDS = 6
MAX_SENTENCE_WORDS = 60

# Selection caps
MAX_REVIEWS_PER_USER = 24
MAX_SENTENCES_PER_REVIEW = 5
MAX_SENTENCES_PER_USER = 30
PROFILE_TEXT_MAX_CHARS = 2200
DIVERSITY_MAX_JACCARD = 0.88
SHORT_TERM_DAYS = 90
SHORT_TEXT_MAX_SENTENCES = 10
LONG_TEXT_MAX_SENTENCES = 20

# Confidence
INTERACTION_CONF_NORM = 20.0
RECENCY_HALF_LIFE_DAYS = 180.0
TEXT_CONF_TARGET_SENTENCES = 20.0
PROFILE_EVIDENCE_HALF_LIFE_DAYS = 180.0
TAG_SUPPORT_TARGET = 12.0
CONF_BLEND_USE_V2 = os.getenv("PROFILE_CONF_USE_V2", "true").strip().lower() == "true"

# Embedding
ENABLE_EMBEDDING = True
EMBED_SCOPE = "profile_text"  # "profile_text" only in v1 (stable on local hardware)
USE_BGE_M3 = True
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_LOCAL_MODEL_PATH = r"D:/hf_cache/hub/models--BAAI--bge-m3"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ALLOW_MODEL_FALLBACK = True
EMBED_NORMALIZE = True
EMBED_BATCH_SIZE_GPU_BGE = 8
EMBED_BATCH_SIZE_GPU_MINILM = 64
EMBED_BATCH_SIZE_CPU_BGE = 4
EMBED_BATCH_SIZE_CPU_MINILM = 32

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'&-]*")
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+|\n+")

# Keep keyword list explicit and maintainable.
PROFILE_KEYWORDS = {
    "cajun",
    "creole",
    "gumbo",
    "jambalaya",
    "etouffee",
    "po boy",
    "oyster",
    "crawfish",
    "shrimp",
    "seafood",
    "sushi",
    "ramen",
    "pizza",
    "burger",
    "taco",
    "burrito",
    "pho",
    "banh mi",
    "bbq",
    "barbecue",
    "breakfast",
    "brunch",
    "coffee",
    "espresso",
    "latte",
    "cafe",
    "cocktail",
    "bar",
    "nightlife",
    "wine",
    "pub",
    "gastropub",
    "steak",
    "deli",
    "sandwich",
    "salad",
    "vegan",
    "vegetarian",
    "dessert",
    "pastry",
    "beignet",
    "service",
    "staff",
    "price",
    "portion",
    "wait",
    "atmosphere",
    "music",
    "family",
    "date night",
}

# Canonical user-tag taxonomy (typed) for auditable typed matching.
# Keys are canonical tags, values include maintainable alias lists.
PROFILE_TAG_DEFS: dict[str, dict[str, Any]] = {
    "cajun_creole": {"tag_type": "cuisine", "aliases": ("cajun", "creole", "gumbo", "jambalaya", "etouffee", "po boy")},
    "seafood": {"tag_type": "cuisine", "aliases": ("seafood", "oyster", "crawfish", "shrimp", "crab")},
    "japanese": {"tag_type": "cuisine", "aliases": ("sushi", "ramen")},
    "mexican": {"tag_type": "cuisine", "aliases": ("taco", "burrito")},
    "vietnamese": {"tag_type": "cuisine", "aliases": ("pho", "banh mi")},
    "bbq": {"tag_type": "cuisine", "aliases": ("bbq", "barbecue")},
    "steakhouse": {"tag_type": "cuisine", "aliases": ("steak",)},
    "pizza": {"tag_type": "cuisine", "aliases": ("pizza",)},
    "deli_sandwich": {"tag_type": "cuisine", "aliases": ("deli", "sandwich")},
    "salad_light": {"tag_type": "meal", "aliases": ("salad",)},
    "breakfast_brunch": {"tag_type": "meal", "aliases": ("breakfast", "brunch")},
    "dessert": {"tag_type": "meal", "aliases": ("dessert", "pastry", "beignet")},
    "vegan_vegetarian": {"tag_type": "meal", "aliases": ("vegan", "vegetarian")},
    "coffee_tea": {"tag_type": "beverage", "aliases": ("coffee", "espresso", "latte", "cafe")},
    "cocktail_bar": {"tag_type": "beverage", "aliases": ("cocktail", "bar", "nightlife", "wine", "pub", "gastropub")},
    "spicy": {"tag_type": "taste", "aliases": ("spicy", "hot")},
    "sweet": {"tag_type": "taste", "aliases": ("sweet",)},
    "savory": {"tag_type": "taste", "aliases": ("savory",)},
    "service": {"tag_type": "service", "aliases": ("service", "staff")},
    "wait_long": {"tag_type": "service", "aliases": ("wait",)},
    "price_value": {"tag_type": "value", "aliases": ("price", "portion")},
    "atmosphere": {"tag_type": "scene", "aliases": ("atmosphere", "music")},
    "family_friendly": {"tag_type": "scene", "aliases": ("family",)},
    "date_night": {"tag_type": "scene", "aliases": ("date night",)},
}


def _compile_profile_tag_matchers() -> list[tuple[str, str, re.Pattern[str]]]:
    matcher: list[tuple[str, str, re.Pattern[str]]] = []
    for tag, spec in PROFILE_TAG_DEFS.items():
        aliases = tuple(spec.get("aliases", ()))
        for alias in aliases:
            a = str(alias).strip().lower()
            if not a:
                continue
            if " " in a:
                pat = re.compile(re.escape(a))
            else:
                pat = re.compile(rf"\b{re.escape(a)}\b")
            matcher.append((tag, a, pat))
    # Longer aliases first to avoid short-token shadowing.
    matcher.sort(key=lambda x: len(x[1]), reverse=True)
    return matcher


PROFILE_TAG_MATCHERS = _compile_profile_tag_matchers()
PROFILE_TAG_TYPE_BY_TAG = {
    tag: str(spec.get("tag_type", "other")).strip().lower() or "other"
    for tag, spec in PROFILE_TAG_DEFS.items()
}

WEAK_TERMS = {
    "good",
    "great",
    "nice",
    "bad",
    "place",
    "food",
    "service",
    "time",
}

POSITIVE_TERMS = {
    "love",
    "loved",
    "great",
    "amazing",
    "excellent",
    "awesome",
    "fantastic",
    "friendly",
    "fresh",
    "delicious",
    "tasty",
    "perfect",
    "favorite",
    "best",
    "quick",
    "cozy",
    "clean",
    "affordable",
    "worth",
}

NEGATIVE_TERMS = {
    "bad",
    "awful",
    "terrible",
    "bland",
    "salty",
    "cold",
    "rude",
    "slow",
    "dirty",
    "overpriced",
    "expensive",
    "worst",
    "disappointing",
    "mediocre",
    "wait",
    "late",
    "noisy",
}


def build_spark() -> SparkSession:
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage09-user-profile-build")
        .master("local[4]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.default.parallelism", "32")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_business_scope(spark: SparkSession) -> tuple[DataFrame, dict[str, Any]]:
    business = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
        .withColumn("business_id", F.col("business_id").cast("string"))
    )
    cat = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    biz = business.filter(F.col("state") == TARGET_STATE)
    cond = None
    if REQUIRE_RESTAURANTS:
        cond = cat.contains("restaurants")
    if REQUIRE_FOOD:
        cond = (cond | cat.contains("food")) if cond is not None else cat.contains("food")
    if cond is not None:
        biz = biz.filter(cond)
    biz = biz.select("business_id").distinct()

    stats: dict[str, Any] = {"n_scope_full": int(biz.count())}
    if RUN_PROFILE == "sample":
        biz = biz.orderBy(F.rand(RANDOM_SEED)).limit(int(SAMPLE_MAX_BUSINESSES))
    stats["n_scope_after_profile"] = int(biz.count())
    return biz.persist(StorageLevel.DISK_ONLY), stats


def load_reviews(spark: SparkSession, biz: DataFrame) -> DataFrame:
    return (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "date", "text")
        .withColumn("review_id", F.col("review_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(biz, on="business_id", how="inner")
        .persist(StorageLevel.DISK_ONLY)
    )


def split_train_only(reviews: DataFrame) -> tuple[DataFrame, DataFrame]:
    # Strictly use train side for profile feature generation.
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = reviews.withColumn("rn", F.row_number().over(w))
    train = ranked.filter(F.col("rn") > F.lit(int(HOLDOUT_PER_USER))).drop("rn")
    n_train = train.groupBy("user_id").agg(F.count("*").alias("n_train"))
    return train.persist(StorageLevel.DISK_ONLY), n_train.persist(StorageLevel.DISK_ONLY)


def build_train_text_pool(train: DataFrame) -> DataFrame:
    text_clean = F.trim(F.regexp_replace(F.coalesce(F.col("text"), F.lit("")), r"\s+", " "))
    char_len = F.length(text_clean)
    alpha_len = F.length(F.regexp_replace(text_clean, r"[^A-Za-z]", ""))
    alpha_ratio = F.when(char_len > 0, alpha_len / char_len).otherwise(F.lit(0.0))
    text_norm = F.lower(text_clean)
    word_count = F.size(F.split(text_norm, r"\s+"))

    pool = (
        train.select("user_id", "business_id", "review_id", "ts", "text")
        .withColumn("text_clean", text_clean)
        .withColumn("text_norm", text_norm)
        .withColumn("char_len", char_len)
        .withColumn("word_count", word_count)
        .withColumn("alpha_ratio", alpha_ratio)
        .filter(F.col("text_clean") != "")
        .filter((F.col("char_len") >= F.lit(TEXT_MIN_CHARS)) & (F.col("char_len") <= F.lit(TEXT_MAX_CHARS)))
        .filter((F.col("word_count") >= F.lit(TEXT_MIN_WORDS)) & (F.col("word_count") <= F.lit(TEXT_MAX_WORDS)))
        .filter(F.col("alpha_ratio") >= F.lit(float(TEXT_MIN_ALPHA_RATIO)))
        .withColumn("text_hash", F.xxhash64("text_norm"))
        .dropDuplicates(["user_id", "business_id", "text_hash"])
        .withColumn(
            "n_sent",
            F.size(F.expr(f"filter(split(text_clean, '[.!?]+'), x -> length(trim(x)) >= {int(MIN_SENTENCE_CHARS)})")),
        )
        .persist(StorageLevel.DISK_ONLY)
    )
    return pool


def user_tier_from_n_train(n_train: int) -> str:
    if n_train <= int(TIER_COLD_MAX):
        return "cold_lt2"
    if n_train <= int(TIER_LIGHT_MAX):
        return "light_2_5"
    return "warm_6p"


def _word_tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _normalize_space(text)
    if not cleaned:
        return []
    return [_normalize_space(x) for x in SENTENCE_SPLIT_RE.split(cleaned) if _normalize_space(x)]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _sentence_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _score_sentence(sentence: str, recency_score: float, window_priority: int) -> tuple[float, list[str], set[str]]:
    lower = sentence.lower()
    words = _word_tokens(lower)
    wset = set(words)
    if not words:
        return -999.0, [], wset

    matched_keywords: list[str] = []
    seen: set[str] = set()
    for tag, _alias, pat in PROFILE_TAG_MATCHERS:
        if tag in seen:
            continue
        if pat.search(lower):
            matched_keywords.append(tag)
            seen.add(tag)
    weak_hits = len([w for w in wset if w in WEAK_TERMS])
    spam_punct = max(0, sentence.count("!") - 2) + max(0, sentence.count("?") - 2)

    # Keep scoring simple and interpretable for maintenance.
    info = (1.4 * float(len(matched_keywords))) + (0.04 * float(min(len(wset), 30)))
    quality = (0.25 * float(recency_score)) + (0.12 * (1.0 if window_priority == 0 else 0.0))
    penalty = (0.22 * float(max(0, weak_hits - 2))) + (0.35 * float(spam_punct))
    score = info + quality - penalty
    return score, matched_keywords, wset


def _sentiment_score(token_set: set[str]) -> float:
    if not token_set:
        return 0.0
    pos = sum(1 for t in token_set if t in POSITIVE_TERMS)
    neg = sum(1 for t in token_set if t in NEGATIVE_TERMS)
    total = pos + neg
    if total <= 0:
        return 0.0
    return float(pos - neg) / float(total)


def _sentence_reliability(sentence_score: float, token_set: set[str], keyword_hits: list[str]) -> float:
    # Keep this simple and auditable.
    score_term = min(1.0, max(0.0, float(sentence_score)) / 4.0)
    token_term = min(1.0, float(len(token_set)) / 18.0)
    keyword_term = min(1.0, float(len(keyword_hits)) / 3.0)
    rel = (0.5 * score_term) + (0.3 * token_term) + (0.2 * keyword_term)
    return float(np.clip(rel, 0.0, 1.0))


def _time_decay_by_days(days_since_review: float) -> float:
    d = max(0.0, float(days_since_review))
    return float(math.exp(-d / float(PROFILE_EVIDENCE_HALF_LIFE_DAYS)))


def _profile_confidence(n_train: int, days_since_last: float, n_sentences: int) -> float:
    interaction_conf = min(1.0, math.log1p(float(max(0, n_train))) / math.log1p(float(INTERACTION_CONF_NORM)))
    recency_conf = math.exp(-max(0.0, float(days_since_last)) / float(RECENCY_HALF_LIFE_DAYS))
    text_conf = min(1.0, float(max(0, n_sentences)) / float(TEXT_CONF_TARGET_SENTENCES))
    conf = (0.5 * interaction_conf) + (0.3 * recency_conf) + (0.2 * text_conf)
    return round(float(conf), 6)


def _profile_confidence_v2(
    n_train: int,
    n_sentences: int,
    tag_support: float,
    freshness_conf: float,
    consistency_conf: float,
) -> tuple[float, dict[str, float]]:
    interaction_conf = min(1.0, math.log1p(float(max(0, n_train))) / math.log1p(float(INTERACTION_CONF_NORM)))
    text_cov_conf = min(1.0, float(max(0, n_sentences)) / float(TEXT_CONF_TARGET_SENTENCES))
    tag_support_conf = min(1.0, float(max(0.0, tag_support)) / float(TAG_SUPPORT_TARGET))
    text_conf = (0.6 * text_cov_conf) + (0.4 * tag_support_conf)
    conf = (
        0.35 * interaction_conf
        + 0.25 * text_conf
        + 0.25 * float(np.clip(freshness_conf, 0.0, 1.0))
        + 0.15 * float(np.clip(consistency_conf, 0.0, 1.0))
    )
    parts = {
        "profile_conf_interaction": round(float(interaction_conf), 6),
        "profile_conf_text": round(float(text_conf), 6),
        "profile_conf_freshness": round(float(np.clip(freshness_conf, 0.0, 1.0)), 6),
        "profile_conf_consistency": round(float(np.clip(consistency_conf, 0.0, 1.0)), 6),
    }
    return round(float(np.clip(conf, 0.0, 1.0)), 6), parts


def _resolve_bge_path(path_text: str) -> str | None:
    p = Path(str(path_text or "").strip())
    if not p:
        return None
    if p.exists():
        if (p / "config.json").exists():
            return str(p)
        snapshots = p / "snapshots"
        if snapshots.exists():
            subs = [x for x in snapshots.iterdir() if x.is_dir() and (x / "config.json").exists()]
            if subs:
                subs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(subs[0])
    return None


def _load_embedding_model() -> tuple[Any, str, str]:
    try:
        import torch  # type: ignore

        device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:
        device = "cpu"

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(f"sentence-transformers import failed: {e}") from e

    primary = MINILM_MODEL_NAME
    if USE_BGE_M3:
        local = _resolve_bge_path(BGE_LOCAL_MODEL_PATH)
        primary = local if local else BGE_MODEL_NAME

    try:
        model = SentenceTransformer(primary, device=device)
        return model, primary, device
    except Exception:
        if not ALLOW_MODEL_FALLBACK or primary == MINILM_MODEL_NAME:
            raise
        model = SentenceTransformer(MINILM_MODEL_NAME, device=device)
        return model, MINILM_MODEL_NAME, device


def _pick_batch_size(model_name: str, device: str) -> int:
    low_name = str(model_name).lower()
    is_bge = "bge" in low_name
    if device == "cuda":
        return int(EMBED_BATCH_SIZE_GPU_BGE if is_bge else EMBED_BATCH_SIZE_GPU_MINILM)
    return int(EMBED_BATCH_SIZE_CPU_BGE if is_bge else EMBED_BATCH_SIZE_CPU_MINILM)


def _cache_file_for_model(model_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model_name))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT / f"profile_text_cache_{safe}.npz"


def _load_npz_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([], dtype="<U40"), np.zeros((0, 0), dtype=np.float32)
    data = np.load(path, allow_pickle=False)
    hashes = data["text_hashes"].astype(str)
    vectors = data["embeddings"].astype(np.float32)
    return hashes, vectors


def _save_npz_cache(path: Path, hashes: np.ndarray, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, text_hashes=hashes, embeddings=vectors.astype(np.float32))


def _join_with_char_limit(sentences: list[str], max_chars: int) -> str:
    out = " ".join([_normalize_space(x) for x in sentences if _normalize_space(x)])
    if len(out) <= int(max_chars):
        return out
    return out[: int(max_chars)].rstrip()


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_PROFILE}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP] load business scope")
    biz, scope_stats = load_business_scope(spark)
    print(
        f"[COUNT] businesses_full={scope_stats['n_scope_full']} "
        f"after_profile={scope_stats['n_scope_after_profile']}"
    )

    print("[STEP] load reviews")
    reviews = load_reviews(spark, biz)
    n_reviews_raw = int(reviews.count())
    print(f"[COUNT] reviews_raw={n_reviews_raw}")

    print("[STEP] split train-only")
    train, n_train = split_train_only(reviews)
    n_users_train = int(n_train.count())
    n_train_events = int(train.count())
    print(f"[COUNT] users_train={n_users_train} train_events={n_train_events}")

    print("[STEP] build train text pool")
    pool = build_train_text_pool(train)
    print(f"[COUNT] pool_reviews={int(pool.count())}")

    print("[STEP] rank and select reviews by 12m priority + 24m fallback")
    last_train = train.groupBy("user_id").agg(F.max("ts").alias("last_train_ts"))
    ref_ts = last_train.agg(F.max("last_train_ts").alias("mx")).first()["mx"]

    cand = (
        pool.join(n_train, on="user_id", how="inner")
        .join(last_train, on="user_id", how="inner")
        .withColumn("m_diff", F.months_between(F.col("last_train_ts"), F.col("ts")))
        .filter((F.col("m_diff") >= F.lit(0.0)) & (F.col("m_diff") <= F.lit(float(FALLBACK_WINDOW_MONTHS))))
        .withColumn("window_priority", F.when(F.col("m_diff") <= F.lit(float(PRIMARY_WINDOW_MONTHS)), F.lit(0)).otherwise(F.lit(1)))
        .withColumn("recency_score", F.exp(-F.col("m_diff") / F.lit(6.0)))
        .withColumn("len_score", F.least(F.col("word_count") / F.lit(50.0), F.lit(1.0)))
        .withColumn("sent_score", F.least(F.col("n_sent") / F.lit(5.0), F.lit(1.0)))
        .withColumn(
            "review_score",
            (F.col("recency_score") * F.lit(0.55))
            + (F.col("len_score") * F.lit(0.20))
            + (F.col("sent_score") * F.lit(0.15))
            + (F.col("alpha_ratio") * F.lit(0.10)),
        )
    )
    w = Window.partitionBy("user_id").orderBy(
        F.col("window_priority").asc(),
        F.col("review_score").desc(),
        F.col("ts").desc(),
        F.col("review_id").desc(),
    )
    selected_reviews = (
        cand.withColumn("rnk", F.row_number().over(w))
        .filter(F.col("rnk") <= F.lit(int(MAX_REVIEWS_PER_USER)))
        .select(
            "user_id",
            "n_train",
            "last_train_ts",
            "review_id",
            "ts",
            "window_priority",
            "review_score",
            "text_clean",
        )
        .persist(StorageLevel.DISK_ONLY)
    )
    print(f"[COUNT] selected_reviews={int(selected_reviews.count())}")

    grouped = (
        selected_reviews.groupBy("user_id", "n_train", "last_train_ts")
        .agg(
            F.collect_list(
                F.struct("review_id", "ts", "window_priority", "review_score", "text_clean")
            ).alias("review_items")
        )
        .persist(StorageLevel.DISK_ONLY)
    )

    profile_rows: list[dict[str, Any]] = []
    tag_long_rows: list[dict[str, Any]] = []
    sentence_rows_path = out_dir / "user_profile_sentences.csv"
    evidence_rows_path = out_dir / "user_profile_evidence.csv"
    sent_fieldnames = [
        "user_id",
        "sentence_rank",
        "review_id",
        "review_ts",
        "days_since_review",
        "sentence_hash",
        "sentence_score",
        "sentiment_score",
        "window_priority",
        "keyword_hits",
        "sentence_text",
    ]
    evidence_fieldnames = [
        "user_id",
        "sentence_rank",
        "review_id",
        "review_ts",
        "days_since_review",
        "sentence_hash",
        "tag",
        "tag_type",
        "polarity",
        "p_tag",
        "reliability",
        "time_decay",
        "final_weight",
        "window_priority",
        "rule_id",
        "sentence_text",
    ]
    with (
        sentence_rows_path.open("w", newline="", encoding="utf-8") as sent_f,
        evidence_rows_path.open("w", newline="", encoding="utf-8") as ev_f,
    ):
        sent_writer = csv.DictWriter(sent_f, fieldnames=sent_fieldnames)
        ev_writer = csv.DictWriter(ev_f, fieldnames=evidence_fieldnames)
        sent_writer.writeheader()
        ev_writer.writeheader()

        for row in grouped.toLocalIterator():
            user_id = str(row["user_id"])
            n_train_i = int(row["n_train"] or 0)
            last_ts = row["last_train_ts"]
            review_items = list(row["review_items"] or [])

            candidates: list[dict[str, Any]] = []
            for rv in review_items:
                text = _normalize_space(rv["text_clean"])
                if not text:
                    continue
                window_priority = int(rv["window_priority"] or 1)
                review_recency = float(rv["review_score"] or 0.0)
                review_id = str(rv["review_id"]) if rv["review_id"] is not None else ""
                review_ts = rv["ts"]
                days_since_review = 0.0
                if last_ts is not None and review_ts is not None:
                    days_since_review = max(0.0, float((last_ts - review_ts).days))
                sents = _split_sentences(text)
                review_scored: list[dict[str, Any]] = []
                for s in sents:
                    s_clean = _normalize_space(s)
                    if len(s_clean) < int(MIN_SENTENCE_CHARS):
                        continue
                    words = _word_tokens(s_clean)
                    if len(words) < int(MIN_SENTENCE_WORDS) or len(words) > int(MAX_SENTENCE_WORDS):
                        continue
                    score, kw_hits, token_set = _score_sentence(s_clean, review_recency, window_priority)
                    if score <= -100:
                        continue
                    review_scored.append(
                        {
                            "sentence_text": s_clean,
                            "sentence_score": float(score),
                            "token_set": token_set,
                            "keyword_hits": kw_hits,
                            "window_priority": int(window_priority),
                            "review_id": review_id,
                            "review_ts": review_ts,
                            "days_since_review": float(days_since_review),
                        }
                    )
                review_scored.sort(key=lambda x: (-x["sentence_score"], x["window_priority"]))
                candidates.extend(review_scored[: int(MAX_SENTENCES_PER_REVIEW)])

            # User-level dedup + diversity
            dedup: dict[str, dict[str, Any]] = {}
            for c in candidates:
                key = c["sentence_text"].lower()
                if key not in dedup or c["sentence_score"] > dedup[key]["sentence_score"]:
                    dedup[key] = c
            all_scored = list(dedup.values())
            all_scored.sort(key=lambda x: (-x["sentence_score"], x["window_priority"]))

            chosen: list[dict[str, Any]] = []
            for c in all_scored:
                too_sim = False
                for s in chosen:
                    if _jaccard(c["token_set"], s["token_set"]) >= float(DIVERSITY_MAX_JACCARD):
                        too_sim = True
                        break
                if too_sim:
                    continue
                chosen.append(c)
                if len(chosen) >= int(MAX_SENTENCES_PER_USER):
                    break

            keyword_counter: Counter[str] = Counter()
            used_fallback = False
            tag_agg: dict[str, dict[str, Any]] = {}
            tag_support = 0.0
            decay_weight_num = 0.0
            decay_weight_den = 0.0

            for i, x in enumerate(chosen, start=1):
                sent_hash = _sentence_hash(x["sentence_text"])
                sentiment_score = _sentiment_score(x["token_set"])
                sent_writer.writerow(
                    {
                        "user_id": user_id,
                        "sentence_rank": int(i),
                        "review_id": x["review_id"],
                        "review_ts": str(x["review_ts"]) if x["review_ts"] is not None else "",
                        "days_since_review": round(float(x["days_since_review"]), 3),
                        "sentence_hash": sent_hash,
                        "sentence_score": round(float(x["sentence_score"]), 6),
                        "sentiment_score": round(float(sentiment_score), 6),
                        "window_priority": int(x["window_priority"]),
                        "keyword_hits": ",".join(x["keyword_hits"]),
                        "sentence_text": x["sentence_text"],
                    }
                )

                for kw in x["keyword_hits"]:
                    keyword_counter[kw] += 1
                if int(x["window_priority"]) > 0:
                    used_fallback = True

                if not x["keyword_hits"]:
                    continue

                polarity = 1.0 if sentiment_score >= 0.0 else -1.0
                reliability = _sentence_reliability(
                    sentence_score=float(x["sentence_score"]),
                    token_set=x["token_set"],
                    keyword_hits=x["keyword_hits"],
                )
                time_decay = _time_decay_by_days(float(x["days_since_review"]))
                p_tag = float(np.clip(0.40 + 0.12 * min(len(x["keyword_hits"]), 3) + 0.18 * abs(sentiment_score), 0.15, 0.98))

                for kw in x["keyword_hits"]:
                    tag = kw.replace(" ", "_")
                    tag_type = PROFILE_TAG_TYPE_BY_TAG.get(tag, "other")
                    final_weight = float(p_tag * time_decay * reliability * polarity)
                    st = tag_agg.setdefault(
                        tag,
                        {"tag_type": tag_type, "pos_w": 0.0, "neg_w": 0.0, "support": 0.0},
                    )
                    if str(st.get("tag_type", "other")) == "other" and tag_type != "other":
                        st["tag_type"] = tag_type
                    if final_weight >= 0.0:
                        st["pos_w"] += float(final_weight)
                    else:
                        st["neg_w"] += float(-final_weight)
                    st["support"] += 1.0
                    tag_support += abs(float(final_weight))
                    decay_weight_num += abs(float(final_weight)) * float(time_decay)
                    decay_weight_den += abs(float(final_weight))

                    ev_writer.writerow(
                        {
                            "user_id": user_id,
                            "sentence_rank": int(i),
                            "review_id": x["review_id"],
                            "review_ts": str(x["review_ts"]) if x["review_ts"] is not None else "",
                            "days_since_review": round(float(x["days_since_review"]), 3),
                            "sentence_hash": sent_hash,
                            "tag": tag,
                            "tag_type": tag_type,
                            "polarity": int(polarity),
                            "p_tag": round(float(p_tag), 6),
                            "reliability": round(float(reliability), 6),
                            "time_decay": round(float(time_decay), 6),
                            "final_weight": round(float(final_weight), 6),
                            "window_priority": int(x["window_priority"]),
                            "rule_id": "kw_sentiment_v1",
                            "sentence_text": x["sentence_text"],
                        }
                    )

            n_sent = len(chosen)
            days_since_last = 0.0
            if ref_ts is not None and last_ts is not None:
                days_since_last = float((ref_ts - last_ts).days)

            # Build short/long profile text to preserve temporal preference structure.
            chosen_sorted = sorted(
                chosen,
                key=lambda z: (z["days_since_review"], -z["sentence_score"], z["window_priority"]),
            )
            short_texts = [z["sentence_text"] for z in chosen_sorted if float(z["days_since_review"]) <= float(SHORT_TERM_DAYS)]
            long_texts = [z["sentence_text"] for z in chosen_sorted if float(z["days_since_review"]) > float(SHORT_TERM_DAYS)]
            if not short_texts:
                short_texts = [z["sentence_text"] for z in chosen_sorted[: max(1, int(SHORT_TEXT_MAX_SENTENCES // 2))]]
            profile_text_short = _join_with_char_limit(short_texts[: int(SHORT_TEXT_MAX_SENTENCES)], int(PROFILE_TEXT_MAX_CHARS // 2))
            profile_text_long = _join_with_char_limit(long_texts[: int(LONG_TEXT_MAX_SENTENCES)], int(PROFILE_TEXT_MAX_CHARS // 2))
            text_parts: list[str] = []
            if profile_text_short:
                text_parts.append("[SHORT] " + profile_text_short)
            if profile_text_long:
                text_parts.append("[LONG] " + profile_text_long)
            if not text_parts:
                text_parts.append(_join_with_char_limit([z["sentence_text"] for z in chosen_sorted], int(PROFILE_TEXT_MAX_CHARS)))
            profile_text = _join_with_char_limit(text_parts, int(PROFILE_TEXT_MAX_CHARS))

            # Confidence decomposition (auditable).
            conf_v1 = _profile_confidence(n_train_i, days_since_last, n_sent)
            if decay_weight_den > 0.0:
                freshness_conf = float(np.clip(decay_weight_num / decay_weight_den, 0.0, 1.0))
            else:
                freshness_conf = float(math.exp(-max(0.0, float(days_since_last)) / float(RECENCY_HALF_LIFE_DAYS)))

            conflict_num = 0.0
            conflict_den = 0.0
            for tag, st in tag_agg.items():
                pos_w = float(st["pos_w"])
                neg_w = float(st["neg_w"])
                net_w = pos_w - neg_w
                abs_net = abs(net_w)
                support = float(st["support"])
                denom = pos_w + neg_w
                conflict_num += min(pos_w, neg_w)
                conflict_den += max(denom, 1e-9)
                tag_conf = abs_net / max(denom, 1e-9)
                tag_long_rows.append(
                    {
                        "user_id": user_id,
                        "tag": tag,
                        "tag_type": str(st.get("tag_type", "other")),
                        "pos_w": round(pos_w, 6),
                        "neg_w": round(neg_w, 6),
                        "net_w": round(net_w, 6),
                        "abs_net_w": round(abs_net, 6),
                        "support": int(support),
                        "tag_confidence": round(float(tag_conf), 6),
                    }
                )
            consistency_conf = 1.0 - (conflict_num / conflict_den) if conflict_den > 0.0 else 0.5
            consistency_conf = float(np.clip(consistency_conf, 0.0, 1.0))

            conf_v2, conf_parts = _profile_confidence_v2(
                n_train=n_train_i,
                n_sentences=n_sent,
                tag_support=float(tag_support),
                freshness_conf=float(freshness_conf),
                consistency_conf=float(consistency_conf),
            )
            final_conf = float(conf_v2 if CONF_BLEND_USE_V2 else conf_v1)

            pos_sum = sum(float(v["pos_w"]) for v in tag_agg.values())
            neg_sum = sum(float(v["neg_w"]) for v in tag_agg.values())
            tag_sorted = sorted(
                ((tag, float(v["pos_w"]) - float(v["neg_w"])) for tag, v in tag_agg.items()),
                key=lambda t: t[1],
                reverse=True,
            )
            top_pos_tags = [t for t, w in tag_sorted if w > 0][:5]
            top_neg_tags = [t for t, w in sorted(tag_sorted, key=lambda t: t[1]) if w < 0][:5]
            pos_by_type: dict[str, list[str]] = {}
            neg_by_type: dict[str, list[str]] = {}
            for t, w in tag_sorted:
                t_type = str(tag_agg.get(t, {}).get("tag_type", "other"))
                if w > 0:
                    pos_by_type.setdefault(t_type, [])
                    if len(pos_by_type[t_type]) < 5:
                        pos_by_type[t_type].append(t)
            for t, w in sorted(tag_sorted, key=lambda z: z[1]):
                t_type = str(tag_agg.get(t, {}).get("tag_type", "other"))
                if w < 0:
                    neg_by_type.setdefault(t_type, [])
                    if len(neg_by_type[t_type]) < 5:
                        neg_by_type[t_type].append(t)

            profile_rows.append(
                {
                    "user_id": user_id,
                    "n_train": int(n_train_i),
                    "tier": user_tier_from_n_train(n_train_i),
                    "last_train_ts": str(last_ts) if last_ts is not None else "",
                    "days_since_last_train": round(days_since_last, 3),
                    "used_window_months": 24 if used_fallback else 12,
                    "profile_confidence": float(final_conf),
                    "profile_confidence_v1": float(conf_v1),
                    "profile_confidence_v2": float(conf_v2),
                    "n_reviews_selected": int(len(review_items)),
                    "n_sentences_selected": int(n_sent),
                    "profile_keywords": ",".join([k for k, _ in keyword_counter.most_common(15)]),
                    "profile_tag_support": round(float(tag_support), 6),
                    "profile_pos_weight_sum": round(float(pos_sum), 6),
                    "profile_neg_weight_sum": round(float(neg_sum), 6),
                    "profile_top_pos_tags": "|".join(top_pos_tags),
                    "profile_top_neg_tags": "|".join(top_neg_tags),
                    "profile_top_pos_tags_by_type": json.dumps(pos_by_type, ensure_ascii=True, sort_keys=True),
                    "profile_top_neg_tags_by_type": json.dumps(neg_by_type, ensure_ascii=True, sort_keys=True),
                    "profile_text_short": profile_text_short,
                    "profile_text_long": profile_text_long,
                    "profile_text": profile_text,
                    **conf_parts,
                }
            )

    # Profile table
    profile_fieldnames = [
        "user_id",
        "n_train",
        "tier",
        "last_train_ts",
        "days_since_last_train",
        "used_window_months",
        "profile_confidence",
        "profile_confidence_v1",
        "profile_confidence_v2",
        "profile_conf_interaction",
        "profile_conf_text",
        "profile_conf_freshness",
        "profile_conf_consistency",
        "n_reviews_selected",
        "n_sentences_selected",
        "profile_keywords",
        "profile_tag_support",
        "profile_pos_weight_sum",
        "profile_neg_weight_sum",
        "profile_top_pos_tags",
        "profile_top_neg_tags",
        "profile_top_pos_tags_by_type",
        "profile_top_neg_tags_by_type",
        "profile_text_short",
        "profile_text_long",
        "profile_text",
    ]
    with (out_dir / "user_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fieldnames)
        writer.writeheader()
        for row in profile_rows:
            writer.writerow({k: row.get(k, "") for k in profile_fieldnames})

    tag_long_fieldnames = [
        "user_id",
        "tag",
        "tag_type",
        "pos_w",
        "neg_w",
        "net_w",
        "abs_net_w",
        "support",
        "tag_confidence",
    ]
    with (out_dir / "user_profile_tag_profile_long.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tag_long_fieldnames)
        writer.writeheader()
        for row in tag_long_rows:
            writer.writerow({k: row.get(k, "") for k in tag_long_fieldnames})

    # Lightweight summary
    n_profiles = len(profile_rows)
    n_nonempty = len([x for x in profile_rows if int(x["n_sentences_selected"]) > 0])
    n_ge10 = len([x for x in profile_rows if int(x["n_sentences_selected"]) >= 10])
    n_ge15 = len([x for x in profile_rows if int(x["n_sentences_selected"]) >= 15])

    summary = {
        "run_id": run_id,
        "run_profile": RUN_PROFILE,
        "n_profiles": int(n_profiles),
        "n_profiles_nonempty": int(n_nonempty),
        "n_profiles_sent_ge10": int(n_ge10),
        "n_profiles_sent_ge15": int(n_ge15),
        "n_user_tag_rows": int(len(tag_long_rows)),
    }
    with (out_dir / "user_profiles_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Embedding on profile_text with persistent hash cache.
    embed_meta: dict[str, Any] = {"enabled": bool(ENABLE_EMBEDDING), "status": "skipped"}
    if ENABLE_EMBEDDING:
        try:
            if EMBED_SCOPE != "profile_text":
                raise RuntimeError(f"unsupported EMBED_SCOPE={EMBED_SCOPE}; only profile_text is supported in v1")
            model, model_name, device = _load_embedding_model()
            batch_size = _pick_batch_size(model_name, device)
            cache_file = _cache_file_for_model(model_name)
            cache_hashes, cache_vectors = _load_npz_cache(cache_file)
            cache_map = {str(h): i for i, h in enumerate(cache_hashes.tolist())}

            user_ids: list[str] = []
            text_hashes: list[str] = []
            text_values: list[str] = []
            for row in profile_rows:
                txt = _normalize_space(str(row.get("profile_text", "")))
                if not txt:
                    continue
                if int(row.get("n_sentences_selected", 0)) <= 0:
                    continue
                h = _sentence_hash(txt)
                user_ids.append(str(row["user_id"]))
                text_hashes.append(h)
                text_values.append(txt)

            unique_missing: list[str] = []
            unique_missing_texts: list[str] = []
            seen: set[str] = set()
            for h, txt in zip(text_hashes, text_values):
                if h in cache_map or h in seen:
                    continue
                seen.add(h)
                unique_missing.append(h)
                unique_missing_texts.append(txt)

            if unique_missing_texts:
                miss_emb = model.encode(
                    unique_missing_texts,
                    batch_size=int(batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=bool(EMBED_NORMALIZE),
                    show_progress_bar=True,
                ).astype(np.float32)
                if cache_vectors.shape[0] == 0:
                    cache_hashes = np.array(unique_missing, dtype="<U40")
                    cache_vectors = miss_emb
                else:
                    cache_hashes = np.concatenate([cache_hashes, np.array(unique_missing, dtype="<U40")], axis=0)
                    cache_vectors = np.concatenate([cache_vectors, miss_emb], axis=0)
                _save_npz_cache(cache_file, cache_hashes, cache_vectors)
                cache_map = {str(h): i for i, h in enumerate(cache_hashes.tolist())}

            # Build per-user vectors
            vec_user_ids: list[str] = []
            vec_rows: list[np.ndarray] = []
            for uid, h in zip(user_ids, text_hashes):
                idx = cache_map.get(h)
                if idx is None:
                    continue
                vec_user_ids.append(uid)
                vec_rows.append(cache_vectors[int(idx)])

            if vec_rows:
                mat = np.stack(vec_rows).astype(np.float32)
                np.savez_compressed(
                    (out_dir / "user_profile_vectors.npz").as_posix(),
                    user_ids=np.array(vec_user_ids, dtype="<U64"),
                    vectors=mat,
                    model_name=np.array([model_name]),
                    normalized=np.array([int(EMBED_NORMALIZE)], dtype=np.int32),
                )
            embed_meta = {
                "enabled": True,
                "status": "ok",
                "model_name": model_name,
                "device": device,
                "batch_size": int(batch_size),
                "cache_file": str(cache_file),
                "cache_size": int(cache_hashes.shape[0]),
                "vectors_written": int(len(vec_rows)),
                "missing_encoded": int(len(unique_missing_texts)),
            }
        except Exception as e:
            embed_meta = {
                "enabled": True,
                "status": "failed",
                "error": str(e),
            }

    write_json(
        out_dir / "run_meta.json",
        {
            "run_id": run_id,
            "run_profile": RUN_PROFILE,
            "run_tag": RUN_TAG,
            "scope": {
                "target_state": TARGET_STATE,
                "require_restaurants": bool(REQUIRE_RESTAURANTS),
                "require_food": bool(REQUIRE_FOOD),
                "sample_max_businesses": int(SAMPLE_MAX_BUSINESSES),
            },
            "split": {"holdout_per_user": int(HOLDOUT_PER_USER)},
            "windows": {
                "primary_months": int(PRIMARY_WINDOW_MONTHS),
                "fallback_months": int(FALLBACK_WINDOW_MONTHS),
                "primary_min_sentences": int(PRIMARY_MIN_SENTENCES),
            },
            "selection_caps": {
                "max_reviews_per_user": int(MAX_REVIEWS_PER_USER),
                "max_sentences_per_review": int(MAX_SENTENCES_PER_REVIEW),
                "max_sentences_per_user": int(MAX_SENTENCES_PER_USER),
                "short_term_days": int(SHORT_TERM_DAYS),
                "short_text_max_sentences": int(SHORT_TEXT_MAX_SENTENCES),
                "long_text_max_sentences": int(LONG_TEXT_MAX_SENTENCES),
            },
            "profile_confidence_config": {
                "interaction_conf_norm": float(INTERACTION_CONF_NORM),
                "recency_half_life_days": float(RECENCY_HALF_LIFE_DAYS),
                "text_conf_target_sentences": float(TEXT_CONF_TARGET_SENTENCES),
                "evidence_half_life_days": float(PROFILE_EVIDENCE_HALF_LIFE_DAYS),
                "tag_support_target": float(TAG_SUPPORT_TARGET),
                "use_v2_as_profile_confidence": bool(CONF_BLEND_USE_V2),
            },
            "embedding_config": {
                "enabled": bool(ENABLE_EMBEDDING),
                "scope": EMBED_SCOPE,
                "use_bge_m3": bool(USE_BGE_M3),
                "normalize": bool(EMBED_NORMALIZE),
            },
            "summary": summary,
            "embedding": embed_meta,
        },
    )

    print(f"[OK] wrote: {out_dir}")

    grouped.unpersist()
    selected_reviews.unpersist()
    pool.unpersist()
    n_train.unpersist()
    train.unpersist()
    reviews.unpersist()
    biz.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
