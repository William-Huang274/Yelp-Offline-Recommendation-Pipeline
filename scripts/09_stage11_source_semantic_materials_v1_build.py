from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession, Window, functions as F

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, write_latest_run_pointer


RUN_TAG = "stage09_stage11_source_semantic_materials_v1_build"

INPUT_10_ROOT = env_or_project_path("INPUT_10_ROOT_DIR", "data/output/p0_stage10_context_eval")
INPUT_10_RUN_DIR = os.getenv("INPUT_10_RUN_DIR", "").strip()
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()

RAW_REVIEW_ROOT = env_or_project_path("RAW_REVIEW_ROOT_DIR", "data/parquet/yelp_academic_dataset_review")
RAW_BUSINESS_ROOT = env_or_project_path("RAW_BUSINESS_ROOT_DIR", "data/parquet/yelp_academic_dataset_business")
RAW_USER_ROOT = env_or_project_path("RAW_USER_ROOT_DIR", "data/parquet/yelp_academic_dataset_user")
RAW_TIP_ROOT = env_or_project_path("RAW_TIP_ROOT_DIR", "data/parquet/yelp_academic_dataset_tip")
RAW_CHECKIN_ROOT = env_or_project_path("RAW_CHECKIN_ROOT_DIR", "data/parquet/yelp_academic_dataset_checkin")

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR",
    "data/output/09_stage11_source_semantic_materials_v1",
)

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "5").strip()
PAIRWISE_POOL_TOPN = int(os.getenv("QLORA_PAIRWISE_POOL_TOPN", "100").strip() or 100)
TARGET_TRUTH_RANK_MIN = int(os.getenv("STAGE11_SEM_TARGET_TRUTH_RANK_MIN", "11").strip() or 11)
TARGET_TRUTH_RANK_MAX = int(os.getenv("STAGE11_SEM_TARGET_TRUTH_RANK_MAX", "100").strip() or 100)

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[24]").strip() or "local[24]"
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "76g").strip() or "76g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "76g").strip() or "76g"
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "192").strip() or "192"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "192").strip() or "192"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "900s").strip() or "900s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", env_or_project_path("SPARK_LOCAL_DIR", "data/spark-tmp").as_posix()).strip()

USER_REVIEW_LIMIT_PER_USER = int(os.getenv("SOURCE_USER_REVIEW_LIMIT_PER_USER", "48").strip() or 48)
MERCHANT_REVIEW_LIMIT_PER_BUSINESS = int(os.getenv("SOURCE_MERCHANT_REVIEW_LIMIT_PER_BUSINESS", "64").strip() or 64)
USER_TIP_LIMIT_PER_USER = int(os.getenv("SOURCE_USER_TIP_LIMIT_PER_USER", "12").strip() or 12)
MERCHANT_TIP_LIMIT_PER_BUSINESS = int(os.getenv("SOURCE_MERCHANT_TIP_LIMIT_PER_BUSINESS", "16").strip() or 16)
USER_POS_SENT_LIMIT = int(os.getenv("SOURCE_USER_POS_SENT_LIMIT", "8").strip() or 8)
USER_NEG_SENT_LIMIT = int(os.getenv("SOURCE_USER_NEG_SENT_LIMIT", "6").strip() or 6)
USER_RECENT_SENT_LIMIT = int(os.getenv("SOURCE_USER_RECENT_SENT_LIMIT", "6").strip() or 6)
MERCHANT_POS_SENT_LIMIT = int(os.getenv("SOURCE_MERCHANT_POS_SENT_LIMIT", "8").strip() or 8)
MERCHANT_NEG_SENT_LIMIT = int(os.getenv("SOURCE_MERCHANT_NEG_SENT_LIMIT", "6").strip() or 6)
USER_TEXT_MAX_CHARS = int(os.getenv("SOURCE_USER_TEXT_MAX_CHARS", "720").strip() or 720)
MERCHANT_TEXT_MAX_CHARS = int(os.getenv("SOURCE_MERCHANT_TEXT_MAX_CHARS", "720").strip() or 720)
CHECKIN_PARSE_LIMIT = int(os.getenv("SOURCE_CHECKIN_PARSE_LIMIT", "200").strip() or 200)

POSITIVE_CUES = {
    "love",
    "loved",
    "favorite",
    "best",
    "great",
    "excellent",
    "amazing",
    "delicious",
    "fantastic",
    "recommend",
    "return",
    "come back",
    "worth",
}
NEGATIVE_CUES = {
    "awful",
    "worst",
    "disappointed",
    "not returning",
    "long wait",
    "wait in line",
    "waited",
    "rude",
    "overpriced",
    "over priced",
    "expensive",
    "cold",
    "bland",
    "bad",
    "poor",
    "dirty",
    "noisy",
    "burnt",
    "closed",
    "could not sit",
    "couldn't sit",
    "no reservations",
    "reservation",
    "parking",
    "not returning",
    "poor service",
    "slow service",
}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SPACE_RE = re.compile(r"\s+")
_SEMANTIC_HINTS = {
    "breakfast",
    "brunch",
    "lunch",
    "dinner",
    "late night",
    "coffee",
    "tea",
    "cocktail",
    "bar",
    "wine",
    "beer",
    "seafood",
    "oyster",
    "shrimp",
    "crawfish",
    "burger",
    "sandwich",
    "po boy",
    "pizza",
    "dessert",
    "bakery",
    "beignet",
    "steak",
    "fries",
    "salad",
    "sushi",
    "ramen",
    "pho",
    "bbq",
    "cajun",
    "creole",
    "service",
    "wait",
    "value",
    "clean",
    "noise",
    "parking",
    "reservation",
    "patio",
    "group",
    "family",
    "date night",
    "fast casual",
}
_DINING_CATEGORY_HINTS = {
    "restaurant",
    "food",
    "bar",
    "cafe",
    "coffee",
    "tea",
    "bakery",
    "dessert",
    "brunch",
    "breakfast",
    "nightlife",
    "sandwich",
    "pizza",
    "seafood",
    "sushi",
    "ramen",
    "pho",
    "bbq",
    "cajun",
    "creole",
    "burger",
    "cocktail",
    "brew",
    "wine",
}
_NOISY_META_PATTERNS = {
    "this place is",
    "nice place to",
    "a nice laid back",
    "probably my best dining experience",
    "best dining experience",
    "on my last",
    "of many trip",
    "owner",
    "i came",
    "we came",
    "yank",
    "so cute and homey",
    "cute and homey",
    "makes it happen",
    "that happens almost never",
    "favorite meal on my last",
    "love y'all food",
}
_NEGATIVE_FALSE_POSITIVE_PATTERNS = {
    "never disappointed",
    "not disappointed",
    "not bad",
    "worth the wait",
    "worth a visit",
    "better than i thought",
    "not overly expensive",
    "not too expensive",
    "friendly and have",
    "service is always fast and pleasant",
    "no issues with service",
    "seated immediately",
    "sat us anyway",
    "confirmed the reservation",
    "super inexpensive",
    "inexpensive",
    "very good food",
    "clean, inexpensive",
    "sorry i waited this long",
    "reservation was for",
    "made fresh as we waited",
    "dirty bird",
}
_FIRST_PERSON_STORY_RE = re.compile(
    r"\b(i|i'm|i’ve|ive|i'd|i’ll|we|we're|we’ve|we'd|we'll|my|our|me|us|wife|husband|fiance|friend and i)\b",
    flags=re.I,
)
_STORY_EVENT_PATTERNS = {
    "my last",
    "last trip",
    "last visit",
    "came here",
    "came in",
    "went here",
    "stopped in",
    "walked in",
    "walked out",
    "we came",
    "i came",
    "i ordered",
    "we ordered",
    "i decided",
    "we decided",
    "i wanted",
    "we wanted",
    "i stayed",
    "we stayed",
    "breakfast run",
    "for brunch this",
    "for dinner this",
    "for lunch this",
    "for the day",
    "the minute we walked",
    "with the girls",
    "i was in",
    "we were in",
    "i've been here",
    "ive been here",
    "my wife",
    "my fiance",
    "my friend and i",
}

_FIRST_PERSON_STORY_RE_EXTRA = re.compile(
    r"\b(i've|i'll|we've|weve|we'll)\b",
    flags=re.I,
)

_STORY_EVENT_PATTERNS_EXTRA = {
    "brought me here",
    "brought us here",
    "went with",
    "the other day",
    "last night",
    "first time trying",
    "on my birthday",
    "before catching my",
    "at first glance",
    "we held a",
}
_STORY_CONTAMINATION_PATTERNS = {
    "first off",
    "here goes my attempt",
    "on my first visit",
    "on my birthday",
    "we were super disappointed",
    "i know they have",
    "attempting to go",
    "what better food in a cold rainy day",
    "favorite sushi restaurant",
    "delicious food excellent service beautiful place",
    "well lit space and freret",
    "located in the",
    "bywater neighborhood",
    "the place to grab",
    "right up there",
    "this is",
    "absolutely beautiful",
    "service is consistently",
    "drinks are strong and tasty",
    "you truly can't go wrong",
}
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[:;!?]|(?<!\d)\.(?!\d)|\s+(?:but|although|though|while|except|however)\s+)",
    flags=re.I,
)
_NEGATIVE_FOCUS_PATTERNS = {
    "only negatives are",
    "only negative is",
    "the only negative is",
    "only bad thing is",
    "the bad thing is",
    "bad thing is",
    "negative is",
    "complaint is",
    "complaints are",
}
_NEGATIVE_SUPPORT_TOKENS = {
    "wait",
    "service",
    "rude",
    "value",
    "overpriced",
    "expensive",
    "noise",
    "loud",
    "parking",
    "clean",
    "dirty",
    "reservation",
    "cold",
    "stale",
    "soggy",
    "fresh",
    "wrong order",
    "small portion",
    "portion",
    "shipment",
}
_POSITIVE_SUPPORT_REJECT_PATTERNS = {
    "not expecting much",
    "come here for good",
    "our go to place",
    "go to place is",
    "when dining for",
    "ask yourself that question",
    "a staple in new orleans",
    "great place to get",
    "this company as a whole",
    "i do love this little",
    "we come here all the time",
    "it was love at first bite",
    "as someone from",
    "traveling with",
    "favorite sushi in the area",
    "doesn't disappoint",
    "doesnt disappoint",
    "i really wanted to love",
    "not my cup of tea",
    "not for everyone",
    "to be expected",
    "located inside",
    "located in the",
    "located inside the historic",
    "what we were celebrating",
    "our table was ready on time",
    "white tablecloth type of restaurant",
    "you must make a reservation",
    "tried to make a reservation",
    "made reservations",
}
_NEGATIVE_SUPPORT_REJECT_PATTERNS = {
    "we didn't die",
    "we didnt die",
    "there is basically nothing wrong with it",
    "it was well worth it",
    "glad we did",
    "off street parking alll the way",
    "only bad thing is it is difficult to find",
    "while talking in another language",
    "i went on a bad day",
    "good bar service",
    "friendly atmosphere",
    "we made reservations",
    "made reservations at",
    "what we were celebrating",
    "our table was ready on time",
    "white tablecloth type of restaurant",
    "you must make a reservation",
    "so i tried to make a reservation",
    "called to make reservations",
    "reservations for",
    "no issues with service",
    "seated immediately",
    "sat us anyway",
    "confirmed the reservation",
    "super inexpensive",
    "inexpensive",
    "very good food",
    "clean, inexpensive",
    "reservation was for",
    "made fresh as we waited",
    "dirty bird",
}
_NEGATIVE_SENTENCE_AVOID_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(long waits?|wait(?:ed|ing)? too long|line was long|queue was long|waited for (?:a )?table|waited for drinks|waited for (?:our|the) entr(?:e|é)e?s?)\b", flags=re.I), "long waits"),
    (re.compile(r"\b(slow service|service was slow|poor service|bad service|service immediately set a bad tone)\b", flags=re.I), "slow service"),
    (re.compile(r"\b(rude|unfriendly|not friendly|inattentive|ignored us|staff was rude|server was rude|bad tone)\b", flags=re.I), "rude or inattentive service"),
    (re.compile(r"\b(could not sit|couldn't sit|not seated|turned us away|no reservations? (?:were )? available|host stand|host was rude|rude upon arrival|not welcoming)\b", flags=re.I), "seating or reservation issues"),
    (re.compile(r"\b(never clean|not clean|dirty|filthy|unclean|messy)\b", flags=re.I), "cleanliness issues"),
    (re.compile(r"\b(noisy|too loud|very loud|noise|crowded|packed)\b", flags=re.I), "noise and overly loud rooms"),
    (re.compile(r"\b(overpriced|too expensive|pricey|not worth|weak value|bad value|expensive side|little expensive|bit on the expensive side|quite expensive)\b", flags=re.I), "weak value for money"),
    (re.compile(r"\b(cold food|food was cold|bad meal|not enjoyable|very disappointing|bland|bland food|dry food|stale|soggy|tasteless|not fresh|inconsistent|no flavor|greasy)\b", flags=re.I), "inconsistent food quality"),
    (re.compile(r"\b(wrong order|forgot our order|forgot the order|missing order|never brought)\b", flags=re.I), "service issues"),
    (re.compile(r"\b(small portions|tiny portions)\b", flags=re.I), "weak value for money"),
    (re.compile(r"\b(parking|hard to park|no parking)\b", flags=re.I), "parking trouble"),
]


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def parse_bucket_override(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} suffix={suffix}")
    return runs[0]


def resolve_optional_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = normalize_legacy_project_path(raw)
        if not p.exists():
            raise FileNotFoundError(f"run dir not found: {p}")
        return p
    return pick_latest_run(root, suffix)


def resolve_stage10_run() -> Path:
    return resolve_optional_run(INPUT_10_RUN_DIR, INPUT_10_ROOT, "_stage10_2_rank_infer_eval")


def resolve_source_stage09_run(stage10_run: Path) -> Path:
    if INPUT_09_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    meta_path = stage10_run / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"stage10 run meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = str(meta.get("source_stage09_run", "")).strip()
    if not raw:
        raise KeyError(f"source_stage09_run missing in {meta_path}")
    p = normalize_legacy_project_path(raw)
    if not p.exists():
        raise FileNotFoundError(f"source_stage09_run not found: {p}")
    return p


def choose_candidate_file(bucket_dir: Path) -> Path:
    for name in [
        "candidates_pretrim.parquet",
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates.parquet",
    ]:
        path = bucket_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no candidate parquet in {bucket_dir}")


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_spark() -> SparkSession:
    local_dir = Path(SPARK_LOCAL_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage09-stage11-source-semantic-materials-v1")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.python.worker.reuse", "true")
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_truth_11_100_pair_universe(stage10_run: Path, source_stage09_run: Path, buckets: list[int]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    all_rows: list[pd.DataFrame] = []
    bucket_stats: list[dict[str, Any]] = []
    for bucket in buckets:
        stage10_bucket = stage10_run / f"bucket_{int(bucket)}"
        source_bucket = source_stage09_run / f"bucket_{int(bucket)}"
        learned_path = stage10_bucket / "learned_scored.parquet"
        truth_path = source_bucket / "truth.parquet"
        cand_path = choose_candidate_file(source_bucket)
        if not learned_path.exists():
            raise FileNotFoundError(f"learned_scored.parquet missing: {learned_path}")
        if not truth_path.exists():
            raise FileNotFoundError(f"truth.parquet missing: {truth_path}")

        learned = pd.read_parquet(
            learned_path,
            columns=["user_idx", "item_idx", "pre_rank", "pre_score", "learned_blend_score", "learned_rank"],
        )
        truth = pd.read_parquet(truth_path, columns=["user_idx", "user_id", "true_item_idx"])
        cand = pd.read_parquet(cand_path, columns=["user_idx", "item_idx", "business_id"])

        truth_rank = truth.merge(
            learned.rename(columns={"item_idx": "true_item_idx"}),
            on=["user_idx", "true_item_idx"],
            how="left",
        ).rename(columns={"learned_rank": "truth_learned_rank"})
        target_users = truth_rank.loc[
            pd.to_numeric(truth_rank["truth_learned_rank"], errors="coerce").between(
                TARGET_TRUTH_RANK_MIN,
                TARGET_TRUTH_RANK_MAX,
                inclusive="both",
            )
        ][["user_idx", "user_id", "true_item_idx", "truth_learned_rank"]].drop_duplicates()
        pair_rows = learned.loc[
            pd.to_numeric(learned["learned_rank"], errors="coerce").fillna(PAIRWISE_POOL_TOPN + 1).le(PAIRWISE_POOL_TOPN)
        ].merge(target_users, on="user_idx", how="inner")
        pair_rows = pair_rows.merge(cand, on=["user_idx", "item_idx"], how="left")
        pair_rows["bucket"] = int(bucket)
        pair_rows = pair_rows.drop_duplicates(subset=["bucket", "user_idx", "item_idx"], keep="first")
        all_rows.append(pair_rows)
        bucket_stats.append(
            {
                "bucket": int(bucket),
                "target_users": int(target_users["user_idx"].nunique()),
                "target_pairs_topn": int(len(pair_rows)),
                "truth_rank_min": int(pd.to_numeric(target_users["truth_learned_rank"], errors="coerce").min()) if not target_users.empty else None,
                "truth_rank_max": int(pd.to_numeric(target_users["truth_learned_rank"], errors="coerce").max()) if not target_users.empty else None,
                "candidate_file": cand_path.name,
            }
        )
    if not all_rows:
        raise RuntimeError("no bucket rows produced for source semantic materials")
    universe = pd.concat(all_rows, ignore_index=True)
    universe["user_id"] = universe["user_id"].astype(str)
    universe["business_id"] = universe["business_id"].astype(str)
    return universe, bucket_stats


def clean_text(raw: Any, max_chars: int | None = None) -> str:
    text = SPACE_RE.sub(" ", str(raw or "").replace("\r", " ").replace("\n", " ")).strip()
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip(" ,;:.") + "..."
    return text


def split_sentences(raw: Any) -> list[str]:
    text = clean_text(raw)
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        sentence = clean_text(part)
        if len(sentence.split()) < 4:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
    return out


def has_any_cue(text: str, cues: set[str]) -> bool:
    low = text.lower()
    return any(cue in low for cue in cues)


def has_negative_material_cue(text: str) -> bool:
    low = clean_text(text).lower()
    if not low:
        return False
    if any(token in low for token in _NEGATIVE_FALSE_POSITIVE_PATTERNS):
        return False
    if extract_negative_material_labels(low):
        return True
    if "not returning" in low or "will not be returning" in low:
        return True
    return has_any_cue(low, NEGATIVE_CUES)


def extract_negative_material_labels(text: Any, limit: int = 4) -> list[str]:
    raw = clean_text(text, max_chars=240)
    if not raw:
        return []
    labels: list[str] = []
    for pattern, label in _NEGATIVE_SENTENCE_AVOID_PATTERNS:
        if pattern.search(raw):
            labels.append(label)
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= limit:
            break
    return out


def has_semantic_hint(text: str) -> bool:
    low = clean_text(text).lower()
    return any(token in low for token in _SEMANTIC_HINTS)


def is_dining_category_text(text: Any) -> bool:
    low = clean_text(text, max_chars=0).lower()
    if not low:
        return False
    return any(token in low for token in _DINING_CATEGORY_HINTS)


def is_noisy_meta_sentence(text: str) -> bool:
    low = clean_text(text).lower()
    return any(token in low for token in _NOISY_META_PATTERNS)


def is_first_person_story_sentence(text: str) -> bool:
    low = clean_text(text).lower()
    if not low:
        return False
    if not (_FIRST_PERSON_STORY_RE.search(low) or _FIRST_PERSON_STORY_RE_EXTRA.search(low)):
        return False
    return any(token in low for token in (_STORY_EVENT_PATTERNS | _STORY_EVENT_PATTERNS_EXTRA)) or not has_semantic_hint(low)


def _material_clauses(text: str) -> list[str]:
    cleaned = clean_text(text, max_chars=0)
    if not cleaned:
        return []
    clauses = [cleaned]
    clauses.extend(part.strip(" ,;:-") for part in _CLAUSE_SPLIT_RE.split(cleaned))
    out: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        item = clean_text(clause, max_chars=220).strip(" ,;:-")
        if len(item.split()) < 3:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_negative_clause(text: str) -> str:
    cleaned = clean_text(text, max_chars=220).strip(" ,;:-")
    if not cleaned:
        return ""
    low = cleaned.lower()
    for prefix in _NEGATIVE_FOCUS_PATTERNS:
        pos = low.find(prefix)
        if pos >= 0:
            cleaned = clean_text(cleaned[pos + len(prefix) :], max_chars=180).strip(" ,;:-")
            low = cleaned.lower()
            break
    cleaned = re.sub(r"^(while|but|however)\s+", "", cleaned, flags=re.I).strip(" ,;:-")
    if not cleaned:
        return ""
    return cleaned


def normalize_material_sentence(text: Any, role: str) -> str:
    raw = clean_text(text, max_chars=220)
    if not raw:
        return ""
    best = ""
    best_score = -1.0
    for clause in _material_clauses(raw):
        low = clause.lower()
        if any(token in low for token in _STORY_CONTAMINATION_PATTERNS):
            continue
        if role == "negative":
            clause = _normalize_negative_clause(clause)
            low = clause.lower()
            if not clause or len(clause.split()) < 3:
                continue
            if any(token in low for token in _NEGATIVE_FALSE_POSITIVE_PATTERNS):
                continue
            if any(token in low for token in _NEGATIVE_SUPPORT_REJECT_PATTERNS):
                continue
            avoid_labels = extract_negative_material_labels(low, limit=4)
            if not (avoid_labels or has_negative_material_cue(low)):
                continue
            if not (avoid_labels or any(token in low for token in _NEGATIVE_SUPPORT_TOKENS)):
                continue
            if has_any_cue(low, POSITIVE_CUES) and len(low.split()) > 8:
                continue
            score = float(len(avoid_labels)) * 18.0 + float(sum(token in low for token in _NEGATIVE_SUPPORT_TOKENS)) * 6.0 - float(len(low.split()))
        else:
            if any(token in low for token in _POSITIVE_SUPPORT_REJECT_PATTERNS):
                continue
            if has_negative_material_cue(low):
                continue
            if (" not " in f" {low} " or low.startswith("not ")) and not any(
                token in low for token in ("not bad", "not too expensive", "not overly expensive")
            ):
                continue
            if is_first_person_story_sentence(low):
                continue
            if (_FIRST_PERSON_STORY_RE.search(low) or _FIRST_PERSON_STORY_RE_EXTRA.search(low)) and len(low.split()) > 8:
                continue
            if is_noisy_meta_sentence(low) and not has_semantic_hint(low):
                continue
            if not has_semantic_hint(low):
                continue
            score = float(sum(token in low for token in _SEMANTIC_HINTS)) * 10.0 - float(len(low.split()))
        if score > best_score:
            best = clause
            best_score = score
    return best


def allow_material_sentence(text: Any, role: str) -> bool:
    return bool(normalize_material_sentence(text, role))


def trim_sentences(entries: list[tuple[float, str]], max_sentences: int, max_chars: int) -> tuple[list[str], str]:
    out: list[str] = []
    seen: set[str] = set()
    total = 0
    for _, sentence in sorted(entries, key=lambda x: x[0], reverse=True):
        text = clean_text(sentence)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        projected = total + len(text) + (1 if out else 0)
        if projected > max_chars and out:
            continue
        seen.add(key)
        out.append(text)
        total = projected
        if len(out) >= max_sentences:
            break
    return out, " ".join(out)


def parse_checkin_hints(raw: Any) -> str:
    text = clean_text(raw, max_chars=0)
    if not text:
        return ""
    bins = Counter()
    items = [part.strip() for part in text.split(",") if part.strip()]
    for item in items[:CHECKIN_PARSE_LIMIT]:
        try:
            ts = pd.Timestamp(item)
        except Exception:
            continue
        hour = int(ts.hour)
        if hour < 11:
            bins["breakfast"] += 1
        elif hour < 15:
            bins["lunch"] += 1
        elif hour < 18:
            bins["afternoon"] += 1
        elif hour < 23:
            bins["dinner"] += 1
        else:
            bins["late night"] += 1
        if int(ts.dayofweek) >= 5:
            bins["weekend"] += 1
    if not bins:
        return ""
    top = [label for label, _ in bins.most_common(3)]
    return clean_text(f"Check-in activity concentrates around {', '.join(top)}.", max_chars=180)


def _build_user_material_records(
    target_users_df: pd.DataFrame,
    user_reviews_df: pd.DataFrame,
    user_tips_df: pd.DataFrame,
    business_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    business_name_map = {
        str(row["business_id"]): clean_text(row.get("name", ""), max_chars=80)
        for _, row in business_meta_df.iterrows()
    }
    review_groups = {uid: grp.copy() for uid, grp in user_reviews_df.groupby("user_id", sort=False)}
    tip_groups = {uid: grp.copy() for uid, grp in user_tips_df.groupby("user_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for record in target_users_df[["bucket", "user_idx", "user_id", "truth_learned_rank"]].drop_duplicates().to_dict(orient="records"):
        user_id = str(record["user_id"])
        reviews = review_groups.get(user_id)
        tips = tip_groups.get(user_id)
        pos_entries: list[tuple[float, str]] = []
        neg_entries: list[tuple[float, str]] = []
        recent_entries: list[tuple[float, str]] = []
        history_business_counter: Counter[str] = Counter()
        history_anchor_texts: list[str] = []
        user_review_count = 0 if reviews is None else int(len(reviews))
        if reviews is not None and not reviews.empty:
            reviews = reviews.sort_values(["review_ts", "useful"], ascending=[False, False])
            for _, review in reviews.iterrows():
                stars = float(review.get("stars", 0.0) or 0.0)
                useful = float(review.get("useful", 0.0) or 0.0)
                business_id = str(review.get("business_id", "") or "")
                sentences = split_sentences(review.get("text", ""))
                positive_history_candidates: list[str] = []
                if stars >= 4.0:
                    history_business_counter[business_id] += 1
                for idx, sentence in enumerate(sentences[:4]):
                    base_score = useful + max(stars, 0.0) + max(0, 3 - idx) * 0.05
                    positive_text = ""
                    negative_text = ""
                    if stars >= 4.0 or has_any_cue(sentence, POSITIVE_CUES):
                        positive_text = normalize_material_sentence(sentence, "positive")
                    if stars <= 2.5 or has_any_cue(sentence, NEGATIVE_CUES):
                        negative_text = normalize_material_sentence(sentence, "negative")
                    positive_ok = bool(positive_text)
                    negative_ok = bool(negative_text)
                    if positive_ok:
                        pos_entries.append((base_score + 1.0, positive_text))
                        positive_history_candidates.append(positive_text)
                        if idx == 0:
                            recent_entries.append((base_score + 0.5, positive_text))
                    if negative_ok:
                        neg_entries.append((base_score + 0.8, negative_text))
                if stars >= 4.0 and positive_history_candidates:
                    top_sentence = clean_text(positive_history_candidates[0], max_chars=140)
                    business_name = business_name_map.get(business_id, "")
                    if business_name:
                        history_anchor_texts.append(clean_text(f"{business_name}: {top_sentence}", max_chars=180))
                    else:
                        history_anchor_texts.append(top_sentence)
        tip_entries: list[tuple[float, str]] = []
        user_tip_count = 0 if tips is None else int(len(tips))
        if tips is not None and not tips.empty:
            tips = tips.sort_values(["tip_ts", "likes"], ascending=[False, False])
            for _, tip in tips.iterrows():
                text = normalize_material_sentence(tip.get("text", ""), "tip")
                if len(text.split()) < 3:
                    continue
                tip_entries.append((float(tip.get("likes", 0.0) or 0.0) + 1.0, text))
                recent_entries.append((float(tip.get("likes", 0.0) or 0.0) + 0.6, text))
        pos_list, pos_text = trim_sentences(pos_entries, USER_POS_SENT_LIMIT, USER_TEXT_MAX_CHARS)
        neg_list, neg_text = trim_sentences(neg_entries, USER_NEG_SENT_LIMIT, USER_TEXT_MAX_CHARS)
        recent_list, recent_text = trim_sentences(recent_entries, USER_RECENT_SENT_LIMIT, USER_TEXT_MAX_CHARS)
        history_anchor_ids = [bid for bid, _ in history_business_counter.most_common(3) if bid]
        history_anchor_list, history_anchor_text = trim_sentences(
            [(float(len(history_anchor_texts) - i), text) for i, text in enumerate(history_anchor_texts)],
            3,
            USER_TEXT_MAX_CHARS,
        )
        tip_list, tip_text = trim_sentences(tip_entries, 4, 320)
        signal_count = int(bool(pos_text)) + int(bool(neg_text)) + int(bool(recent_text)) + int(bool(history_anchor_text)) + int(bool(tip_text))
        rows.append(
            {
                "bucket": int(record["bucket"]),
                "user_idx": int(record["user_idx"]),
                "user_id": user_id,
                "truth_learned_rank": int(record["truth_learned_rank"]),
                "user_source_positive_sentences_v1": json.dumps(pos_list, ensure_ascii=False),
                "user_source_negative_sentences_v1": json.dumps(neg_list, ensure_ascii=False),
                "user_source_recent_sentences_v1": json.dumps(recent_list, ensure_ascii=False),
                "user_source_tip_sentences_v1": json.dumps(tip_list, ensure_ascii=False),
                "user_history_anchor_business_ids_v1": json.dumps(history_anchor_ids, ensure_ascii=False),
                "user_history_anchor_texts_v1": json.dumps(history_anchor_list, ensure_ascii=False),
                "user_source_positive_text_v1": pos_text,
                "user_source_negative_text_v1": neg_text,
                "user_source_recent_text_v1": recent_text,
                "user_source_tip_text_v1": tip_text,
                "user_source_history_anchor_text_v1": history_anchor_text,
                "user_source_positive_sentence_count_v1": int(len(pos_list)),
                "user_source_negative_sentence_count_v1": int(len(neg_list)),
                "user_source_recent_sentence_count_v1": int(len(recent_list)),
                "user_source_tip_sentence_count_v1": int(len(tip_list)),
                "user_source_history_anchor_count_v1": int(len(history_anchor_ids)),
                "user_source_review_count_v1": int(user_review_count),
                "user_source_tip_count_v1": int(user_tip_count),
                "user_source_signal_count_v1": int(signal_count),
            }
        )
    return pd.DataFrame(rows)


def _build_merchant_material_records(
    target_business_df: pd.DataFrame,
    merchant_reviews_df: pd.DataFrame,
    merchant_tips_df: pd.DataFrame,
    checkin_df: pd.DataFrame,
    business_meta_df: pd.DataFrame,
) -> pd.DataFrame:
    review_groups = {bid: grp.copy() for bid, grp in merchant_reviews_df.groupby("business_id", sort=False)}
    tip_groups = {bid: grp.copy() for bid, grp in merchant_tips_df.groupby("business_id", sort=False)}
    checkin_map = {str(row["business_id"]): clean_text(row.get("date", ""), max_chars=0) for _, row in checkin_df.iterrows()}
    meta_map = {
        str(row["business_id"]): {
            "name": clean_text(row.get("name", ""), max_chars=80),
            "city": clean_text(row.get("city", ""), max_chars=48),
            "categories": clean_text(row.get("categories", ""), max_chars=180),
        }
        for _, row in business_meta_df.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for record in target_business_df[["bucket", "business_id"]].drop_duplicates().to_dict(orient="records"):
        business_id = str(record["business_id"])
        reviews = review_groups.get(business_id)
        tips = tip_groups.get(business_id)
        pos_entries: list[tuple[float, str]] = []
        neg_entries: list[tuple[float, str]] = []
        review_count = 0 if reviews is None else int(len(reviews))
        if reviews is not None and not reviews.empty:
            reviews = reviews.sort_values(["useful", "review_ts"], ascending=[False, False])
            for _, review in reviews.iterrows():
                stars = float(review.get("stars", 0.0) or 0.0)
                useful = float(review.get("useful", 0.0) or 0.0)
                sentences = split_sentences(review.get("text", ""))
                for idx, sentence in enumerate(sentences[:4]):
                    base_score = useful + max(stars, 0.0) + max(0, 3 - idx) * 0.05
                    positive_text = ""
                    negative_text = ""
                    if stars >= 4.0 or has_any_cue(sentence, POSITIVE_CUES):
                        positive_text = normalize_material_sentence(sentence, "positive")
                    if stars <= 2.5 or has_any_cue(sentence, NEGATIVE_CUES):
                        negative_text = normalize_material_sentence(sentence, "negative")
                    if positive_text:
                        pos_entries.append((base_score + 1.0, positive_text))
                    if negative_text:
                        neg_entries.append((base_score + 0.8, negative_text))
        tip_entries: list[tuple[float, str]] = []
        tip_count = 0 if tips is None else int(len(tips))
        if tips is not None and not tips.empty:
            tips = tips.sort_values(["likes", "tip_ts"], ascending=[False, False])
            for _, tip in tips.iterrows():
                text = normalize_material_sentence(tip.get("text", ""), "tip")
                if len(text.split()) < 3:
                    continue
                tip_entries.append((float(tip.get("likes", 0.0) or 0.0) + 1.0, text))
        strength_list, strength_text = trim_sentences(pos_entries, MERCHANT_POS_SENT_LIMIT, MERCHANT_TEXT_MAX_CHARS)
        risk_list, risk_text = trim_sentences(neg_entries, MERCHANT_NEG_SENT_LIMIT, MERCHANT_TEXT_MAX_CHARS)
        tip_list, tip_text = trim_sentences(tip_entries, 4, 320)
        checkin_text = parse_checkin_hints(checkin_map.get(business_id, ""))
        signal_count = int(bool(strength_text)) + int(bool(risk_text)) + int(bool(tip_text)) + int(bool(checkin_text))
        meta = meta_map.get(business_id, {})
        rows.append(
            {
                "bucket": int(record["bucket"]),
                "business_id": business_id,
                "name": meta.get("name", ""),
                "city": meta.get("city", ""),
                "categories": meta.get("categories", ""),
                "merchant_source_strength_sentences_v1": json.dumps(strength_list, ensure_ascii=False),
                "merchant_source_risk_sentences_v1": json.dumps(risk_list, ensure_ascii=False),
                "merchant_source_tip_sentences_v1": json.dumps(tip_list, ensure_ascii=False),
                "merchant_source_strength_text_v1": strength_text,
                "merchant_source_risk_text_v1": risk_text,
                "merchant_source_tip_text_v1": tip_text,
                "merchant_source_checkin_hint_text_v1": checkin_text,
                "merchant_source_strength_sentence_count_v1": int(len(strength_list)),
                "merchant_source_risk_sentence_count_v1": int(len(risk_list)),
                "merchant_source_tip_sentence_count_v1": int(len(tip_list)),
                "merchant_source_review_count_v1": int(review_count),
                "merchant_source_tip_count_v1": int(tip_count),
                "merchant_source_signal_count_v1": int(signal_count),
            }
        )
    return pd.DataFrame(rows)


def _quantiles(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    q = series.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).to_dict()
    return {str(k): float(v) for k, v in q.items()}


def _top_records(df: pd.DataFrame, cols: list[str], sort_cols: list[str], ascending: list[bool], n: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.sort_values(sort_cols, ascending=ascending).head(n)[cols].to_dict(orient="records")


def main() -> None:
    stage10_run = resolve_stage10_run()
    source_stage09_run = resolve_source_stage09_run(stage10_run)
    buckets = parse_bucket_override(BUCKETS_OVERRIDE)
    if not buckets:
        raise ValueError("BUCKETS_OVERRIDE resolved to empty")

    out_dir = OUTPUT_ROOT / now_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[SOURCE-SEM] load 11-100 pair universe", flush=True)
    pair_universe, bucket_stats = load_truth_11_100_pair_universe(stage10_run, source_stage09_run, buckets)
    print(
        f"[SOURCE-SEM] pair universe ready rows={len(pair_universe)} users={pair_universe['user_idx'].nunique()} businesses={pair_universe['business_id'].nunique()}",
        flush=True,
    )

    target_users_df = pair_universe[["bucket", "user_idx", "user_id", "truth_learned_rank"]].drop_duplicates()
    target_business_df = pair_universe[["bucket", "business_id"]].drop_duplicates()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        target_user_ids = pair_universe[["user_id"]].drop_duplicates().reset_index(drop=True)
        target_business_ids = pair_universe[["business_id"]].drop_duplicates().reset_index(drop=True)
        user_scope_sdf = spark.createDataFrame(target_user_ids)
        business_scope_sdf = spark.createDataFrame(target_business_ids)

        review_df = (
            spark.read.parquet(RAW_REVIEW_ROOT.as_posix())
            .select("review_id", "user_id", "business_id", "stars", "useful", "date", "text")
            .withColumn("review_ts", F.to_timestamp("date"))
        )
        business_meta_sdf = (
            spark.read.parquet(RAW_BUSINESS_ROOT.as_posix())
            .select("business_id", "name", "city", "categories")
            .withColumn("categories_norm", F.lower(F.coalesce(F.col("categories"), F.lit(""))))
        )
        dining_business_scope_sdf = business_meta_sdf.filter(
            F.col("categories_norm").rlike(
                "restaurant|food|bar|cafe|coffee|tea|bakery|dessert|brunch|breakfast|nightlife|sandwich|pizza|seafood|sushi|ramen|pho|bbq|cajun|creole|burger|cocktail|brew|wine"
            )
        ).select("business_id").distinct()
        user_df_exists = RAW_USER_ROOT.exists() and any(RAW_USER_ROOT.iterdir())
        tip_df_exists = RAW_TIP_ROOT.exists() and any(RAW_TIP_ROOT.iterdir())
        checkin_df_exists = RAW_CHECKIN_ROOT.exists() and any(RAW_CHECKIN_ROOT.iterdir())

        w_user = Window.partitionBy("user_id").orderBy(F.col("review_ts").desc(), F.col("useful").desc(), F.col("review_id").desc())
        user_reviews_limited = (
            review_df.join(F.broadcast(user_scope_sdf), on="user_id", how="inner")
            .join(F.broadcast(dining_business_scope_sdf), on="business_id", how="inner")
            .withColumn("rn", F.row_number().over(w_user))
            .filter(F.col("rn") <= F.lit(USER_REVIEW_LIMIT_PER_USER))
            .drop("rn")
        )
        w_biz = Window.partitionBy("business_id").orderBy(F.col("useful").desc(), F.col("review_ts").desc(), F.col("review_id").desc())
        merchant_reviews_limited = (
            review_df.join(F.broadcast(business_scope_sdf), on="business_id", how="inner")
            .withColumn("rn", F.row_number().over(w_biz))
            .filter(F.col("rn") <= F.lit(MERCHANT_REVIEW_LIMIT_PER_BUSINESS))
            .drop("rn")
        )

        user_tips_limited = spark.createDataFrame([], "user_id string, business_id string, date string, text string, likes long, tip_ts timestamp")
        merchant_tips_limited = spark.createDataFrame([], "business_id string, user_id string, date string, text string, likes long, tip_ts timestamp")
        if tip_df_exists:
            tip_raw = spark.read.parquet(RAW_TIP_ROOT.as_posix())
            tip_cols = set(tip_raw.columns)
            if "likes" in tip_cols:
                tip_weight_col = F.col("likes").cast("double")
            elif "compliment_count" in tip_cols:
                tip_weight_col = F.col("compliment_count").cast("double")
            else:
                tip_weight_col = F.lit(0.0)
            tip_df = (
                tip_raw
                .withColumn("likes", F.coalesce(tip_weight_col, F.lit(0.0)))
                .select("user_id", "business_id", "date", "text", "likes")
                .withColumn("tip_ts", F.to_timestamp("date"))
            )
            w_user_tip = Window.partitionBy("user_id").orderBy(F.col("tip_ts").desc(), F.col("likes").desc())
            user_tips_limited = (
                tip_df.join(F.broadcast(user_scope_sdf), on="user_id", how="inner")
                .join(F.broadcast(dining_business_scope_sdf), on="business_id", how="inner")
                .withColumn("rn", F.row_number().over(w_user_tip))
                .filter(F.col("rn") <= F.lit(USER_TIP_LIMIT_PER_USER))
                .drop("rn")
            )
            w_biz_tip = Window.partitionBy("business_id").orderBy(F.col("likes").desc(), F.col("tip_ts").desc())
            merchant_tips_limited = (
                tip_df.join(F.broadcast(business_scope_sdf), on="business_id", how="inner")
                .withColumn("rn", F.row_number().over(w_biz_tip))
                .filter(F.col("rn") <= F.lit(MERCHANT_TIP_LIMIT_PER_BUSINESS))
                .drop("rn")
            )

        checkin_limited = spark.createDataFrame([], "business_id string, date string")
        if checkin_df_exists:
            checkin_limited = spark.read.parquet(RAW_CHECKIN_ROOT.as_posix()).select("business_id", "date").join(
                F.broadcast(business_scope_sdf),
                on="business_id",
                how="inner",
            )

        print("[SOURCE-SEM] materialize limited raw subsets", flush=True)
        user_reviews_pdf = user_reviews_limited.toPandas()
        merchant_reviews_pdf = merchant_reviews_limited.toPandas()
        user_tips_pdf = user_tips_limited.toPandas()
        merchant_tips_pdf = merchant_tips_limited.toPandas()
        checkin_pdf = checkin_limited.toPandas()
        business_meta_pdf = (
            business_meta_sdf.join(F.broadcast(business_scope_sdf), on="business_id", how="inner")
            .drop("categories_norm")
            .toPandas()
        )
    finally:
        spark.stop()

    if not user_reviews_pdf.empty:
        user_reviews_pdf["user_id"] = user_reviews_pdf["user_id"].astype(str)
        user_reviews_pdf["business_id"] = user_reviews_pdf["business_id"].astype(str)
        user_reviews_pdf["review_ts"] = pd.to_datetime(user_reviews_pdf["review_ts"], errors="coerce")
        user_reviews_pdf["useful"] = pd.to_numeric(user_reviews_pdf["useful"], errors="coerce").fillna(0.0)
        user_reviews_pdf["stars"] = pd.to_numeric(user_reviews_pdf["stars"], errors="coerce").fillna(0.0)
    if not merchant_reviews_pdf.empty:
        merchant_reviews_pdf["user_id"] = merchant_reviews_pdf["user_id"].astype(str)
        merchant_reviews_pdf["business_id"] = merchant_reviews_pdf["business_id"].astype(str)
        merchant_reviews_pdf["review_ts"] = pd.to_datetime(merchant_reviews_pdf["review_ts"], errors="coerce")
        merchant_reviews_pdf["useful"] = pd.to_numeric(merchant_reviews_pdf["useful"], errors="coerce").fillna(0.0)
        merchant_reviews_pdf["stars"] = pd.to_numeric(merchant_reviews_pdf["stars"], errors="coerce").fillna(0.0)
    if not user_tips_pdf.empty:
        user_tips_pdf["user_id"] = user_tips_pdf["user_id"].astype(str)
        user_tips_pdf["business_id"] = user_tips_pdf["business_id"].astype(str)
        user_tips_pdf["tip_ts"] = pd.to_datetime(user_tips_pdf["tip_ts"], errors="coerce")
        user_tips_pdf["likes"] = pd.to_numeric(user_tips_pdf["likes"], errors="coerce").fillna(0.0)
    if not merchant_tips_pdf.empty:
        merchant_tips_pdf["user_id"] = merchant_tips_pdf["user_id"].astype(str)
        merchant_tips_pdf["business_id"] = merchant_tips_pdf["business_id"].astype(str)
        merchant_tips_pdf["tip_ts"] = pd.to_datetime(merchant_tips_pdf["tip_ts"], errors="coerce")
        merchant_tips_pdf["likes"] = pd.to_numeric(merchant_tips_pdf["likes"], errors="coerce").fillna(0.0)
    if not checkin_pdf.empty:
        checkin_pdf["business_id"] = checkin_pdf["business_id"].astype(str)
    if not business_meta_pdf.empty:
        business_meta_pdf["business_id"] = business_meta_pdf["business_id"].astype(str)

    print("[SOURCE-SEM] build user source materials", flush=True)
    user_materials_df = _build_user_material_records(target_users_df, user_reviews_pdf, user_tips_pdf, business_meta_pdf)
    print("[SOURCE-SEM] build merchant source materials", flush=True)
    merchant_materials_df = _build_merchant_material_records(target_business_df, merchant_reviews_pdf, merchant_tips_pdf, checkin_pdf, business_meta_pdf)

    target_pairs_path = out_dir / "stage11_target_pair_universe_v1.parquet"
    target_users_path = out_dir / "stage11_target_users_v1.parquet"
    user_materials_path = out_dir / "user_source_semantic_materials_v1.parquet"
    merchant_materials_path = out_dir / "merchant_source_semantic_materials_v1.parquet"
    print("[SOURCE-SEM] write parquet outputs", flush=True)
    pair_universe.to_parquet(target_pairs_path, index=False)
    target_users_df.to_parquet(target_users_path, index=False)
    user_materials_df.to_parquet(user_materials_path, index=False)
    merchant_materials_df.to_parquet(merchant_materials_path, index=False)

    audit = {
        "bucket_stats": bucket_stats,
        "target_users_total": int(target_users_df["user_idx"].nunique()),
        "target_businesses_total": int(target_business_df["business_id"].nunique()),
        "target_pairs_total": int(len(pair_universe)),
        "raw_user_table_present": bool(user_df_exists),
        "raw_tip_table_present": bool(tip_df_exists),
        "raw_checkin_table_present": bool(checkin_df_exists),
        "user_review_rows_limited": int(len(user_reviews_pdf)),
        "merchant_review_rows_limited": int(len(merchant_reviews_pdf)),
        "user_tip_rows_limited": int(len(user_tips_pdf)),
        "merchant_tip_rows_limited": int(len(merchant_tips_pdf)),
        "checkin_rows_limited": int(len(checkin_pdf)),
        "user_material_rows": int(len(user_materials_df)),
        "merchant_material_rows": int(len(merchant_materials_df)),
        "user_positive_rate": float(user_materials_df["user_source_positive_sentence_count_v1"].gt(0).mean()) if not user_materials_df.empty else 0.0,
        "user_negative_rate": float(user_materials_df["user_source_negative_sentence_count_v1"].gt(0).mean()) if not user_materials_df.empty else 0.0,
        "user_recent_rate": float(user_materials_df["user_source_recent_sentence_count_v1"].gt(0).mean()) if not user_materials_df.empty else 0.0,
        "user_tip_rate": float(user_materials_df["user_source_tip_sentence_count_v1"].gt(0).mean()) if not user_materials_df.empty else 0.0,
        "user_history_anchor_rate": float(user_materials_df["user_source_history_anchor_count_v1"].gt(0).mean()) if not user_materials_df.empty else 0.0,
        "user_signal_count_quantiles": _quantiles(user_materials_df["user_source_signal_count_v1"]) if not user_materials_df.empty else {},
        "merchant_strength_rate": float(merchant_materials_df["merchant_source_strength_sentence_count_v1"].gt(0).mean()) if not merchant_materials_df.empty else 0.0,
        "merchant_risk_rate": float(merchant_materials_df["merchant_source_risk_sentence_count_v1"].gt(0).mean()) if not merchant_materials_df.empty else 0.0,
        "merchant_tip_rate": float(merchant_materials_df["merchant_source_tip_sentence_count_v1"].gt(0).mean()) if not merchant_materials_df.empty else 0.0,
        "merchant_checkin_rate": float(merchant_materials_df["merchant_source_checkin_hint_text_v1"].fillna("").str.len().gt(0).mean()) if not merchant_materials_df.empty else 0.0,
        "merchant_signal_count_quantiles": _quantiles(merchant_materials_df["merchant_source_signal_count_v1"]) if not merchant_materials_df.empty else {},
        "sample_low_user_materials": _top_records(
            user_materials_df.assign(_sort=user_materials_df["user_source_signal_count_v1"]),
            ["user_id", "truth_learned_rank", "user_source_signal_count_v1", "user_source_positive_text_v1", "user_source_negative_text_v1", "user_source_recent_text_v1", "user_source_history_anchor_text_v1"],
            ["_sort", "truth_learned_rank"],
            [True, True],
        ) if not user_materials_df.empty else [],
        "sample_high_user_materials": _top_records(
            user_materials_df.assign(_sort=user_materials_df["user_source_signal_count_v1"]),
            ["user_id", "truth_learned_rank", "user_source_signal_count_v1", "user_source_positive_text_v1", "user_source_negative_text_v1", "user_source_recent_text_v1", "user_source_history_anchor_text_v1"],
            ["_sort", "truth_learned_rank"],
            [False, True],
        ) if not user_materials_df.empty else [],
        "sample_low_merchant_materials": _top_records(
            merchant_materials_df.assign(_sort=merchant_materials_df["merchant_source_signal_count_v1"]),
            ["business_id", "name", "merchant_source_signal_count_v1", "merchant_source_strength_text_v1", "merchant_source_risk_text_v1", "merchant_source_tip_text_v1", "merchant_source_checkin_hint_text_v1"],
            ["_sort", "business_id"],
            [True, True],
        ) if not merchant_materials_df.empty else [],
        "sample_high_merchant_materials": _top_records(
            merchant_materials_df.assign(_sort=merchant_materials_df["merchant_source_signal_count_v1"]),
            ["business_id", "name", "merchant_source_signal_count_v1", "merchant_source_strength_text_v1", "merchant_source_risk_text_v1", "merchant_source_tip_text_v1", "merchant_source_checkin_hint_text_v1"],
            ["_sort", "business_id"],
            [False, True],
        ) if not merchant_materials_df.empty else [],
    }
    safe_json_write(out_dir / "audit.json", audit)
    run_meta = {
        "run_tag": RUN_TAG,
        "stage10_run": str(stage10_run),
        "source_stage09_run": str(source_stage09_run),
        "target_users_total": int(target_users_df["user_idx"].nunique()),
        "target_businesses_total": int(target_business_df["business_id"].nunique()),
        "target_pairs_total": int(len(pair_universe)),
        "user_material_rows": int(len(user_materials_df)),
        "merchant_material_rows": int(len(merchant_materials_df)),
        "user_positive_rate": audit["user_positive_rate"],
        "user_negative_rate": audit["user_negative_rate"],
        "user_recent_rate": audit["user_recent_rate"],
        "merchant_strength_rate": audit["merchant_strength_rate"],
        "merchant_risk_rate": audit["merchant_risk_rate"],
        "raw_tip_table_present": bool(tip_df_exists),
        "raw_checkin_table_present": bool(checkin_df_exists),
        "output_dir": str(out_dir),
    }
    safe_json_write(out_dir / "run_meta.json", run_meta)
    write_latest_run_pointer(OUTPUT_ROOT, out_dir)
    print("[SOURCE-SEM] done", flush=True)


if __name__ == "__main__":
    main()
