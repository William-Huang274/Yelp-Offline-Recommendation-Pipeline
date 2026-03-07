from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

# Keep sentence-transformers on torch path in mixed TensorFlow/Keras env.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


RUN_PROFILE = os.getenv("RUN_PROFILE_OVERRIDE", "full").strip().lower() or "full"  # "sample" | "full"
RUN_TAG = "stage09_item_semantic_build"
RANDOM_SEED = int(os.getenv("ITEM_SEM_RANDOM_SEED", "42").strip() or 42)

PARQUET_BASE = Path(r"D:/5006 BDA project/data/parquet")
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/09_item_semantics")

TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
SAMPLE_MAX_BUSINESSES = int(os.getenv("ITEM_SEM_SAMPLE_MAX_BUSINESSES", "600").strip() or 600)

TEXT_MIN_CHARS = int(os.getenv("ITEM_SEM_TEXT_MIN_CHARS", "20").strip() or 20)
TEXT_MAX_CHARS = int(os.getenv("ITEM_SEM_TEXT_MAX_CHARS", "800").strip() or 800)
TEXT_MIN_WORDS = int(os.getenv("ITEM_SEM_TEXT_MIN_WORDS", "5").strip() or 5)
TEXT_MAX_WORDS = int(os.getenv("ITEM_SEM_TEXT_MAX_WORDS", "60").strip() or 60)
TEXT_MIN_ALPHA_RATIO = float(os.getenv("ITEM_SEM_TEXT_MIN_ALPHA_RATIO", "0.45").strip() or 0.45)
MAX_SENTENCES_PER_REVIEW = int(os.getenv("ITEM_SEM_MAX_SENTENCES_PER_REVIEW", "6").strip() or 6)

HALF_LIFE_DAYS = float(os.getenv("ITEM_SEM_HALF_LIFE_DAYS", "180").strip() or 180.0)
SUPPORT_CONF_TARGET = float(os.getenv("ITEM_SEM_SUPPORT_CONF_TARGET", "8").strip() or 8.0)

ENABLE_EMBED_MATCH = os.getenv("ITEM_SEM_ENABLE_EMBED", "true").strip().lower() == "true"
EMBED_MAX_SENTENCES = int(os.getenv("ITEM_SEM_EMBED_MAX_SENTENCES", "140000").strip() or 140000)
EMBED_SIM_MIN = float(os.getenv("ITEM_SEM_EMBED_SIM_MIN", "0.58").strip() or 0.58)
EMBED_BATCH_SIZE = int(os.getenv("ITEM_SEM_EMBED_BATCH_SIZE", "64").strip() or 64)
USE_BGE_M3 = os.getenv("ITEM_SEM_USE_BGE_M3", "true").strip().lower() == "true"
BGE_LOCAL_MODEL_PATH = Path(os.getenv("BGE_LOCAL_MODEL_PATH", r"D:/hf_cache/hub/models--BAAI--bge-m3").strip())
MINILM_MODEL_NAME = os.getenv("ITEM_SEM_FALLBACK_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

ENABLE_LLM_MATCH = os.getenv("ITEM_SEM_ENABLE_LLM", "false").strip().lower() == "true"
LLM_MAX_SENTENCES = int(os.getenv("ITEM_SEM_LLM_MAX_SENTENCES", "1500").strip() or 1500)
LLM_MODEL = os.getenv("ITEM_SEM_LLM_MODEL", "qwen3:8b").strip() or "qwen3:8b"
LLM_BASE_URL = os.getenv("ITEM_SEM_LLM_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
LLM_SIM_LOW = float(os.getenv("ITEM_SEM_LLM_SIM_LOW", "0.48").strip() or 0.48)
LLM_SIM_HIGH = float(os.getenv("ITEM_SEM_LLM_SIM_HIGH", "0.62").strip() or 0.62)

SEMANTIC_ACTIVE_NET_MIN = float(os.getenv("ITEM_SEM_ACTIVE_NET_MIN", "0.03").strip() or 0.03)

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "perfect",
    "fresh",
    "friendly",
    "authentic",
    "fast",
    "love",
    "delicious",
    "tasty",
    "favorite",
    "clean",
    "cozy",
}
NEGATIVE_WORDS = {
    "bad",
    "awful",
    "terrible",
    "slow",
    "rude",
    "cold",
    "bland",
    "overpriced",
    "dirty",
    "stale",
    "worst",
    "disappointing",
    "burnt",
    "wait",
}
NEGATION_WORDS = {"not", "no", "never", "without", "hardly", "barely", "none", "isnt", "wasnt", "dont", "didnt"}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'&-]*")
SENT_SPLIT_RE = re.compile(r"[.!?]+\s+|\n+")


@dataclass(frozen=True)
class TagDef:
    tag: str
    facet: str
    keywords: tuple[str, ...]
    prototypes: tuple[str, ...]


TAG_DEFS: list[TagDef] = [
    TagDef("cajun_creole", "cuisine", ("cajun", "creole", "gumbo", "jambalaya", "etouffee"), ("cajun creole louisiana food",)),
    TagDef("seafood", "cuisine", ("seafood", "oyster", "crawfish", "shrimp", "crab"), ("fresh seafood oyster crawfish shrimp",)),
    TagDef("bbq", "cuisine", ("bbq", "barbecue", "smoked meat", "brisket", "ribs"), ("barbecue smoked brisket ribs",)),
    TagDef("mexican", "cuisine", ("mexican", "taco", "burrito", "quesadilla", "salsa"), ("mexican tacos burritos",)),
    TagDef("italian", "cuisine", ("italian", "pasta", "risotto", "gnocchi"), ("italian pasta dishes",)),
    TagDef("japanese", "cuisine", ("sushi", "ramen", "izakaya", "sashimi"), ("japanese sushi ramen",)),
    TagDef("vietnamese", "cuisine", ("pho", "banh mi", "vermicelli"), ("vietnamese pho banh mi",)),
    TagDef("breakfast_brunch", "meal", ("breakfast", "brunch", "egg benedict", "pancake", "waffle"), ("breakfast brunch cafe",)),
    TagDef("coffee_tea", "beverage", ("coffee", "espresso", "latte", "tea", "matcha"), ("specialty coffee tea",)),
    TagDef("cocktail_bar", "beverage", ("cocktail", "mixology", "bar", "happy hour", "wine list"), ("cocktail bar drinks",)),
    TagDef("dessert", "meal", ("dessert", "beignet", "cake", "pastry", "ice cream"), ("dessert pastry beignet",)),
    TagDef("spicy", "taste", ("spicy", "hot", "heat", "chili"), ("spicy flavorful dishes",)),
    TagDef("savory", "taste", ("savory", "rich flavor", "well seasoned"), ("savory seasoned food",)),
    TagDef("sweet", "taste", ("sweet", "sweetness", "sugary"), ("sweet dessert drinks",)),
    TagDef("large_portion", "value", ("large portion", "huge portion", "big portion", "filling"), ("large portions filling meal",)),
    TagDef("value_good", "value", ("worth it", "good value", "fair price", "reasonable price"), ("good value reasonable price",)),
    TagDef("value_bad", "value", ("overpriced", "too expensive", "not worth", "pricey"), ("overpriced expensive not worth",)),
    TagDef("service_good", "service", ("friendly staff", "great service", "attentive", "helpful"), ("friendly attentive service",)),
    TagDef("service_bad", "service", ("rude", "slow service", "bad service", "ignored"), ("bad slow rude service",)),
    TagDef("wait_long", "service", ("long wait", "waited", "line was long"), ("long wait line",)),
    TagDef("atmosphere_good", "ambience", ("cozy", "vibe", "atmosphere", "romantic", "nice music"), ("cozy atmosphere vibe",)),
    TagDef("atmosphere_noisy", "ambience", ("noisy", "too loud", "crowded", "chaotic"), ("noisy loud crowded",)),
    TagDef("cleanliness_good", "ambience", ("clean", "spotless", "well kept"), ("clean spotless place",)),
    TagDef("cleanliness_bad", "ambience", ("dirty", "unclean", "messy"), ("dirty unclean place",)),
    TagDef("family_friendly", "audience", ("family friendly", "kids", "kid friendly"), ("family friendly kids",)),
    TagDef("date_night", "audience", ("date night", "romantic dinner"), ("date night romantic dining",)),
]

_TAG_BY_NAME = {t.tag: t for t in TAG_DEFS}


def build_spark() -> SparkSession:
    local_dir = Path(r"D:/5006 BDA project/data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage09-item-semantic-build")
        .master("local[2]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for sent in SENT_SPLIT_RE.split(text):
        s = normalize_space(sent)
        if s:
            out.append(s)
    return out


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha_cnt = sum(1 for c in text if c.isalpha())
    return float(alpha_cnt) / float(max(1, len(text)))


def sentence_reliability(text: str, words: list[str]) -> float:
    n_chars = len(text)
    n_words = len(words)
    if n_chars < int(TEXT_MIN_CHARS) or n_chars > int(TEXT_MAX_CHARS):
        return 0.0
    if n_words < int(TEXT_MIN_WORDS) or n_words > int(TEXT_MAX_WORDS):
        return 0.0
    ar = alpha_ratio(text)
    if ar < float(TEXT_MIN_ALPHA_RATIO):
        return 0.0
    char_term = min(1.0, float(n_chars) / 120.0)
    word_term = min(1.0, float(n_words) / 18.0)
    alpha_term = min(1.0, ar / 0.85)
    uniq_term = float(len(set(words))) / float(max(1, n_words))
    return float(np.clip(0.35 * char_term + 0.25 * word_term + 0.25 * alpha_term + 0.15 * uniq_term, 0.0, 1.0))


def detect_polarity(words: list[str], neg_hint: bool) -> int:
    if neg_hint:
        return -1
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if neg > pos:
        return -1
    return 1


def _compile_rule_patterns() -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    out: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for tag in TAG_DEFS:
        tag_rules: list[tuple[str, re.Pattern[str]]] = []
        for kw in tag.keywords:
            k = kw.strip().lower()
            if not k:
                continue
            if " " in k:
                pat = re.compile(re.escape(k))
            else:
                pat = re.compile(rf"\b{re.escape(k)}\b")
            tag_rules.append((k, pat))
        out[tag.tag] = tag_rules
    return out


RULE_PATTERNS = _compile_rule_patterns()


def rule_match(sentence_lc: str, words: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    neg_hint = any(w in NEGATION_WORDS for w in words)
    for tag in TAG_DEFS:
        for kw, pat in RULE_PATTERNS[tag.tag]:
            if not pat.search(sentence_lc):
                continue
            p = 0.92 if (" " in kw or len(kw) >= 8) else 0.86
            pol = detect_polarity(words, neg_hint=neg_hint and tag.facet in {"service", "value", "ambience"})
            hits.append(
                {
                    "tag": tag.tag,
                    "facet": tag.facet,
                    "source": "rule",
                    "raw_score": 1.0,
                    "p_tag": float(p),
                    "polarity": int(pol),
                    "rule_id": f"kw:{kw}",
                }
            )
            break
    return hits


def pick_latest_snapshot(model_root: Path) -> Path | None:
    snaps = model_root / "snapshots"
    if not snaps.exists():
        return None
    candidates = [p for p in snaps.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if (p / "config.json").exists() and ((p / "modules.json").exists() or (p / "sentence_bert_config.json").exists()):
            return p
    return None


def resolve_embedding_model() -> str:
    if bool(USE_BGE_M3):
        snap = pick_latest_snapshot(BGE_LOCAL_MODEL_PATH)
        if snap is not None:
            return snap.as_posix()
    return MINILM_MODEL_NAME


def load_encoder() -> Any | None:
    if not bool(ENABLE_EMBED_MATCH):
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"[WARN] sentence-transformers unavailable: {e}")
        return None
    model_name = resolve_embedding_model()
    try:
        enc = SentenceTransformer(model_name, device="cpu")
        print(f"[INFO] embedding_model={model_name}")
        return enc
    except Exception as e:
        print(f"[WARN] embedding model load failed: {e}")
        return None


def encode_texts(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    arr = encoder.encode(
        texts,
        batch_size=int(max(1, batch_size)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(arr, dtype=np.float32)


def sim_to_prob(sim: float, sim_min: float) -> float:
    if sim <= sim_min:
        return 0.0
    x = (sim - sim_min) / max(1e-6, 1.0 - sim_min)
    return float(np.clip(0.35 + 0.60 * x, 0.0, 0.98))


def llm_extract_tag(sentence: str) -> dict[str, Any] | None:
    if not ENABLE_LLM_MATCH:
        return None
    try:
        import requests
    except Exception:
        return None

    prompt = (
        "You are a strict restaurant sentence tagger. "
        "Return JSON only: {\"tag\": string, \"polarity\": \"pos\"|\"neg\", \"confidence\": 0-1}. "
        f"Allowed tags: {', '.join(sorted(_TAG_BY_NAME.keys()))}. "
        f"Sentence: {sentence}"
    )
    body = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(f"{LLM_BASE_URL}/api/generate", json=body, timeout=40)
        resp.raise_for_status()
        payload = resp.json()
        txt = payload.get("response", "{}")
        data = json.loads(txt)
        tag = str(data.get("tag", "")).strip()
        if tag not in _TAG_BY_NAME:
            return None
        pol = str(data.get("polarity", "pos")).strip().lower()
        pol_sign = -1 if pol in {"neg", "negative", "-1"} else 1
        conf = float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0))
        return {"tag": tag, "facet": _TAG_BY_NAME[tag].facet, "polarity": pol_sign, "p_tag": conf}
    except Exception:
        return None


def load_business_scope(spark: SparkSession) -> DataFrame:
    business = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories")
        .withColumn("business_id", F.col("business_id").cast("string"))
    )
    cat = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    biz = business.filter(F.col("state") == F.lit(TARGET_STATE))
    cond = None
    if REQUIRE_RESTAURANTS:
        cond = cat.contains("restaurants")
    if REQUIRE_FOOD:
        cond = (cond | cat.contains("food")) if cond is not None else cat.contains("food")
    if cond is not None:
        biz = biz.filter(cond)
    biz = biz.select("business_id").distinct()
    if RUN_PROFILE == "sample":
        biz = biz.orderBy(F.rand(RANDOM_SEED)).limit(int(SAMPLE_MAX_BUSINESSES))
    return biz.persist(StorageLevel.DISK_ONLY)


def load_reviews(spark: SparkSession, biz: DataFrame) -> DataFrame:
    return (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "business_id", "date", "text")
        .withColumn("review_id", F.col("review_id").cast("string"))
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(biz, on="business_id", how="inner")
        .select("review_id", "business_id", "ts", "text")
        .persist(StorageLevel.DISK_ONLY)
    )


def now_ref_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def half_life_decay(days_since: int) -> float:
    d = max(0.0, float(days_since))
    hl = max(1.0, float(HALF_LIFE_DAYS))
    return float(np.power(0.5, d / hl))


def build_outputs() -> tuple[Path, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_PROFILE}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, run_id


def emit_evidence(
    writer: csv.DictWriter,
    agg: dict[tuple[str, str], dict[str, Any]],
    ref_dt: datetime,
    business_id: str,
    review_id: str,
    sent_id: int,
    sentence: str,
    review_dt: datetime,
    tag: str,
    facet: str,
    source: str,
    raw_score: float,
    p_tag: float,
    polarity: int,
    reliability: float,
    rule_id: str,
) -> None:
    days_since = max(0, int((ref_dt - review_dt).days))
    decay = half_life_decay(days_since)
    p = float(np.clip(float(p_tag), 0.0, 1.0))
    pol = -1 if int(polarity) < 0 else 1
    rel = float(np.clip(float(reliability), 0.0, 1.0))
    final_weight = float(p * decay * rel * pol)

    writer.writerow(
        {
            "business_id": business_id,
            "review_id": review_id,
            "sentence_id": int(sent_id),
            "sentence": sentence,
            "tag": tag,
            "facet": facet,
            "source": source,
            "rule_id": rule_id,
            "raw_score": float(raw_score),
            "p_tag": p,
            "polarity": int(pol),
            "reliability": rel,
            "days_since": int(days_since),
            "time_decay": float(decay),
            "final_weight": float(final_weight),
            "review_date": review_dt.strftime("%Y-%m-%d"),
        }
    )

    key = (business_id, tag)
    st = agg.get(key)
    if st is None:
        st = {
            "facet": facet,
            "support": 0,
            "pos_w": 0.0,
            "neg_w": 0.0,
            "p_sum": 0.0,
            "source_rule": 0,
            "source_embed": 0,
            "source_llm": 0,
            "last_date": review_dt,
        }
        agg[key] = st
    st["support"] += 1
    st["p_sum"] += p
    if final_weight >= 0.0:
        st["pos_w"] += float(final_weight)
    else:
        st["neg_w"] += float(-final_weight)
    if source == "rule":
        st["source_rule"] += 1
    elif source == "embed":
        st["source_embed"] += 1
    elif source == "llm":
        st["source_llm"] += 1
    if review_dt > st["last_date"]:
        st["last_date"] = review_dt


def main() -> None:
    out_dir, run_id = build_outputs()
    evidence_path = out_dir / "item_tag_evidence.csv"
    profile_long_path = out_dir / "item_tag_profile_long.csv"
    feature_path = out_dir / "item_semantic_features.csv"

    spark = build_spark()
    biz = load_business_scope(spark)
    reviews = load_reviews(spark, biz)
    n_reviews = int(reviews.count())
    n_biz = int(biz.count())
    print(f"[INFO] run_profile={RUN_PROFILE} businesses={n_biz} reviews={n_reviews}")

    encoder = load_encoder()
    enable_embed = encoder is not None

    agg: dict[tuple[str, str], dict[str, Any]] = {}
    embed_queue: list[dict[str, Any]] = []
    llm_queue: list[dict[str, Any]] = []

    ref_dt = now_ref_utc()
    processed = 0
    emitted_rule = 0
    queued_embed = 0

    with evidence_path.open("w", newline="", encoding="utf-8") as ef:
        evidence_writer = csv.DictWriter(
            ef,
            fieldnames=[
                "business_id",
                "review_id",
                "sentence_id",
                "sentence",
                "tag",
                "facet",
                "source",
                "rule_id",
                "raw_score",
                "p_tag",
                "polarity",
                "reliability",
                "days_since",
                "time_decay",
                "final_weight",
                "review_date",
            ],
        )
        evidence_writer.writeheader()

        # Process reviews in stream to control memory.
        for row in reviews.select("review_id", "business_id", "ts", "text").toLocalIterator():
            processed += 1
            if processed % 50000 == 0:
                print(f"[PROGRESS] processed_reviews={processed}/{n_reviews} rule_evidence={emitted_rule} queued_embed={queued_embed}")
            business_id = str(row["business_id"])
            review_id = str(row["review_id"])
            review_ts = row["ts"]
            if review_ts is None:
                continue
            review_dt = review_ts.to_pydatetime() if hasattr(review_ts, "to_pydatetime") else review_ts
            txt = normalize_space(str(row["text"] or ""))
            if not txt:
                continue
            sentences = split_sentences(txt)
            if not sentences:
                continue
            if len(sentences) > int(MAX_SENTENCES_PER_REVIEW):
                sentences = sentences[: int(MAX_SENTENCES_PER_REVIEW)]

            for sid, sent in enumerate(sentences, start=1):
                words = tokenize_words(sent)
                rel = sentence_reliability(sent, words)
                if rel <= 0.0:
                    continue
                sent_lc = sent.lower()
                rule_hits = rule_match(sent_lc, words)
                if rule_hits:
                    for hit in rule_hits:
                        emit_evidence(
                            writer=evidence_writer,
                            agg=agg,
                            ref_dt=ref_dt,
                            business_id=business_id,
                            review_id=review_id,
                            sent_id=sid,
                            sentence=sent,
                            review_dt=review_dt,
                            tag=hit["tag"],
                            facet=hit["facet"],
                            source="rule",
                            raw_score=float(hit["raw_score"]),
                            p_tag=float(hit["p_tag"]),
                            polarity=int(hit["polarity"]),
                            reliability=float(rel),
                            rule_id=str(hit["rule_id"]),
                        )
                        emitted_rule += 1
                    continue

                if enable_embed and len(embed_queue) < int(max(1, EMBED_MAX_SENTENCES)):
                    embed_queue.append(
                        {
                            "business_id": business_id,
                            "review_id": review_id,
                            "sentence_id": sid,
                            "sentence": sent,
                            "review_dt": review_dt,
                            "reliability": float(rel),
                        }
                    )
                    queued_embed += 1

        emitted_embed = 0
        emitted_llm = 0
        if enable_embed and embed_queue:
            proto_texts = [normalize_space(" ; ".join(t.prototypes)) for t in TAG_DEFS]
            proto_emb = encode_texts(encoder, proto_texts, batch_size=int(max(8, EMBED_BATCH_SIZE)))
            sent_texts = [r["sentence"] for r in embed_queue]
            sent_emb = encode_texts(encoder, sent_texts, batch_size=int(max(8, EMBED_BATCH_SIZE)))
            sims = np.matmul(sent_emb, proto_emb.T).astype(np.float32)
            best_idx = np.argmax(sims, axis=1)
            best_sim = sims[np.arange(sims.shape[0]), best_idx]

            for i, sim in enumerate(best_sim.tolist()):
                rec = embed_queue[i]
                tag = TAG_DEFS[int(best_idx[i])]
                words = tokenize_words(rec["sentence"])
                polarity = detect_polarity(words, neg_hint=False)
                if float(sim) >= float(EMBED_SIM_MIN):
                    p = sim_to_prob(float(sim), float(EMBED_SIM_MIN))
                    emit_evidence(
                        writer=evidence_writer,
                        agg=agg,
                        ref_dt=ref_dt,
                        business_id=rec["business_id"],
                        review_id=rec["review_id"],
                        sent_id=int(rec["sentence_id"]),
                        sentence=rec["sentence"],
                        review_dt=rec["review_dt"],
                        tag=tag.tag,
                        facet=tag.facet,
                        source="embed",
                        raw_score=float(sim),
                        p_tag=float(p),
                        polarity=int(polarity),
                        reliability=float(rec["reliability"]),
                        rule_id="embed_top1",
                    )
                    emitted_embed += 1
                    continue
                if ENABLE_LLM_MATCH and len(llm_queue) < int(max(1, LLM_MAX_SENTENCES)) and float(LLM_SIM_LOW) <= float(sim) <= float(LLM_SIM_HIGH):
                    llm_queue.append(rec)

        if ENABLE_LLM_MATCH and llm_queue:
            for rec in llm_queue:
                out = llm_extract_tag(rec["sentence"])
                if not out:
                    continue
                emit_evidence(
                    writer=evidence_writer,
                    agg=agg,
                    ref_dt=ref_dt,
                    business_id=rec["business_id"],
                    review_id=rec["review_id"],
                    sent_id=int(rec["sentence_id"]),
                    sentence=rec["sentence"],
                    review_dt=rec["review_dt"],
                    tag=str(out["tag"]),
                    facet=str(out["facet"]),
                    source="llm",
                    raw_score=float(out["p_tag"]),
                    p_tag=float(out["p_tag"]),
                    polarity=int(out["polarity"]),
                    reliability=float(rec["reliability"]),
                    rule_id=f"llm:{LLM_MODEL}",
                )
                emitted_llm += 1

        print(
            f"[INFO] evidence_emitted rule={emitted_rule} embed={emitted_embed} llm={emitted_llm} "
            f"embed_queued={len(embed_queue)} llm_queued={len(llm_queue)}"
        )

    # Aggregate rows to long profile and item-level feature table.
    long_rows: list[dict[str, Any]] = []
    biz_acc: dict[str, dict[str, Any]] = {}
    for (biz_id, tag), st in agg.items():
        support = int(st["support"])
        pos_w = float(st["pos_w"])
        neg_w = float(st["neg_w"])
        net_w = pos_w - neg_w
        conf = float(np.clip((float(st["p_sum"]) / max(1.0, float(support))) * min(1.0, float(support) / float(SUPPORT_CONF_TARGET)), 0.0, 1.0))
        long_rows.append(
            {
                "business_id": biz_id,
                "tag": tag,
                "facet": str(st["facet"]),
                "support_count": support,
                "pos_weight_sum": pos_w,
                "neg_weight_sum": neg_w,
                "net_weight_sum": net_w,
                "tag_confidence": conf,
                "source_rule_count": int(st["source_rule"]),
                "source_embed_count": int(st["source_embed"]),
                "source_llm_count": int(st["source_llm"]),
                "last_review_date": st["last_date"].strftime("%Y-%m-%d"),
            }
        )
        acc = biz_acc.get(biz_id)
        if acc is None:
            acc = {"pos": 0.0, "neg": 0.0, "support": 0, "conf_num": 0.0, "conf_den": 0.0, "tags": []}
            biz_acc[biz_id] = acc
        acc["pos"] += pos_w
        acc["neg"] += neg_w
        acc["support"] += support
        acc["conf_num"] += conf * float(support)
        acc["conf_den"] += float(support)
        acc["tags"].append((tag, net_w))

    with profile_long_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "business_id",
                "tag",
                "facet",
                "support_count",
                "pos_weight_sum",
                "neg_weight_sum",
                "net_weight_sum",
                "tag_confidence",
                "source_rule_count",
                "source_embed_count",
                "source_llm_count",
                "last_review_date",
            ],
        )
        wr.writeheader()
        for row in long_rows:
            wr.writerow(row)

    feature_rows: list[dict[str, Any]] = []
    for biz_id, acc in biz_acc.items():
        pos_total = float(acc["pos"])
        neg_total = float(acc["neg"])
        abs_total = pos_total + neg_total
        net_total = pos_total - neg_total
        semantic_score = float(net_total / max(1e-6, abs_total))
        semantic_conf = float(acc["conf_num"] / max(1e-6, acc["conf_den"]))
        support_total = int(acc["support"])

        tags_sorted = sorted(acc["tags"], key=lambda x: x[1], reverse=True)
        top_pos = [t for t, w in tags_sorted if w > 0][:3]
        top_neg = [t for t, w in sorted(acc["tags"], key=lambda x: x[1]) if w < 0][:3]
        rich = int(sum(1 for _, w in acc["tags"] if abs(float(w)) >= float(SEMANTIC_ACTIVE_NET_MIN)))
        feature_rows.append(
            {
                "business_id": biz_id,
                "semantic_score": semantic_score,
                "semantic_confidence": semantic_conf,
                "semantic_support": support_total,
                "semantic_tag_richness": rich,
                "semantic_pos_weight_sum": pos_total,
                "semantic_neg_weight_sum": neg_total,
                "top_pos_tags": "|".join(top_pos),
                "top_neg_tags": "|".join(top_neg),
            }
        )

    with feature_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "business_id",
                "semantic_score",
                "semantic_confidence",
                "semantic_support",
                "semantic_tag_richness",
                "semantic_pos_weight_sum",
                "semantic_neg_weight_sum",
                "top_pos_tags",
                "top_neg_tags",
            ],
        )
        wr.writeheader()
        for row in feature_rows:
            wr.writerow(row)

    run_meta = {
        "run_id": run_id,
        "run_profile": RUN_PROFILE,
        "run_tag": RUN_TAG,
        "output_dir": str(out_dir),
        "n_business_scope": int(n_biz),
        "n_reviews_scope": int(n_reviews),
        "n_item_tag_pairs": int(len(long_rows)),
        "n_item_features": int(len(feature_rows)),
        "enable_embed_match": bool(enable_embed),
        "enable_llm_match": bool(ENABLE_LLM_MATCH),
        "embed_model": resolve_embedding_model() if enable_embed else "",
        "llm_model": LLM_MODEL if ENABLE_LLM_MATCH else "",
        "half_life_days": float(HALF_LIFE_DAYS),
        "embed_sim_min": float(EMBED_SIM_MIN),
    }
    write_json(out_dir / "run_meta.json", run_meta)
    print(f"[INFO] wrote {out_dir}")

    reviews.unpersist()
    biz.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
