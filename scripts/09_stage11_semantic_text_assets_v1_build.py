from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.stage11_text_features import (
    build_clean_user_evidence_text,
    build_profile_preference_evidence_text,
    naturalize_user_context_text,
    naturalize_user_long_pref_text,
    naturalize_user_negative_avoid_text,
    naturalize_user_recent_intent_text,
)
from pipeline.stage11_pairwise import classify_boundary_constructability
from pipeline.project_paths import (
    env_or_project_path,
    normalize_legacy_project_path,
    project_path,
    write_latest_run_pointer,
)


RUN_TAG = "stage09_stage11_semantic_text_assets_v1_build"

INPUT_10_ROOT = env_or_project_path("INPUT_10_ROOT_DIR", "data/output/p0_stage10_context_eval")
INPUT_10_RUN_DIR = os.getenv("INPUT_10_RUN_DIR", "").strip()
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()

INPUT_09_USER_PROFILE_TEXT_ROOT = env_or_project_path(
    "INPUT_09_USER_PROFILE_TEXT_ROOT_DIR",
    "data/output/09_user_profiles",
)
INPUT_09_USER_PROFILE_TEXT_RUN_DIR = os.getenv("INPUT_09_USER_PROFILE_TEXT_RUN_DIR", "").strip()
INPUT_09_USER_TEXT_VIEWS_ROOT = env_or_project_path(
    "INPUT_09_USER_TEXT_VIEWS_ROOT_DIR",
    "data/output/09_candidate_wise_text_views_v1",
)
INPUT_09_USER_TEXT_VIEWS_RUN_DIR = os.getenv("INPUT_09_USER_TEXT_VIEWS_RUN_DIR", "").strip()
INPUT_09_USER_SCHEMA_ROOT = env_or_project_path(
    "INPUT_09_USER_SCHEMA_PROJECTION_V2_ROOT_DIR",
    "data/output/09_user_schema_projection_v2",
)
INPUT_09_USER_SCHEMA_RUN_DIR = os.getenv("INPUT_09_USER_SCHEMA_PROJECTION_V2_RUN_DIR", "").strip()
INPUT_09_USER_INTENT_V3_ROOT = env_or_project_path(
    "INPUT_09_USER_INTENT_PROFILE_V3_ROOT_DIR",
    "data/output/09_user_intent_profile_v3_all_bucket_users",
)
INPUT_09_USER_INTENT_V3_RUN_DIR = os.getenv("INPUT_09_USER_INTENT_PROFILE_V3_RUN_DIR", "").strip()
INPUT_09_MERCHANT_CARD_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_CARD_ROOT_DIR",
    "data/output/09_merchant_semantic_card",
)
INPUT_09_MERCHANT_CARD_RUN_DIR = os.getenv("INPUT_09_MERCHANT_CARD_RUN_DIR", "").strip()
INPUT_09_MERCHANT_TEXT_V3_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_TEXT_VIEWS_V3_ROOT_DIR",
    "data/output/09_merchant_text_views_v3",
)
INPUT_09_MERCHANT_TEXT_V3_RUN_DIR = os.getenv("INPUT_09_MERCHANT_TEXT_VIEWS_V3_RUN_DIR", "").strip()
INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_ROOT_DIR",
    "data/output/09_merchant_structured_text_views_v2",
)
INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_RUN_DIR = os.getenv(
    "INPUT_09_MERCHANT_STRUCTURED_TEXT_VIEWS_V2_RUN_DIR", ""
).strip()
INPUT_09_MATCH_FEATURE_ROOT = env_or_project_path(
    "INPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_ROOT_DIR",
    "data/output/09_user_business_match_features_v2",
)
INPUT_09_MATCH_FEATURE_RUN_DIR = os.getenv("INPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_RUN_DIR", "").strip()
INPUT_09_MATCH_CHANNEL_ROOT = env_or_project_path(
    "INPUT_09_USER_BUSINESS_MATCH_CHANNELS_V2_ROOT_DIR",
    "data/output/09_user_business_match_channels_v2",
)
INPUT_09_MATCH_CHANNEL_RUN_DIR = os.getenv("INPUT_09_USER_BUSINESS_MATCH_CHANNELS_V2_RUN_DIR", "").strip()
INPUT_09_CANDIDATE_TEXT_MATCH_ROOT = env_or_project_path(
    "INPUT_09_CANDIDATE_TEXT_MATCH_FEATURES_V1_ROOT_DIR",
    "data/output/09_candidate_wise_text_match_features_v1",
)
INPUT_09_CANDIDATE_TEXT_MATCH_RUN_DIR = os.getenv(
    "INPUT_09_CANDIDATE_TEXT_MATCH_FEATURES_V1_RUN_DIR", ""
).strip()
INPUT_09_SOURCE_SEMANTIC_MATERIALS_V1_ROOT = env_or_project_path(
    "INPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR",
    "data/output/09_stage11_source_semantic_materials_v1",
)
INPUT_09_SOURCE_SEMANTIC_MATERIALS_V1_RUN_DIR = os.getenv(
    "INPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_RUN_DIR", ""
).strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_STAGE11_SEMANTIC_TEXT_ASSETS_V1_ROOT_DIR",
    "data/output/09_stage11_semantic_text_assets_v1",
)

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "5").strip()
PAIRWISE_POOL_TOPN = int(os.getenv("QLORA_PAIRWISE_POOL_TOPN", "100").strip() or 100)
TARGET_TRUTH_RANK_MIN = int(os.getenv("STAGE11_SEM_TARGET_TRUTH_RANK_MIN", "11").strip() or 11)
TARGET_TRUTH_RANK_MAX = int(os.getenv("STAGE11_SEM_TARGET_TRUTH_RANK_MAX", "100").strip() or 100)
STAGE09_SEM_THREADS = max(1, int(os.getenv("STAGE09_SEM_THREADS", "1").strip() or 1))
STAGE09_SEM_CHUNK_ROWS = max(200, int(os.getenv("STAGE09_SEM_CHUNK_ROWS", "2500").strip() or 2500))

_GENERIC_TERMS = {
    "food",
    "restaurant",
    "restaurants",
    "cuisine",
    "meal",
    "meals",
    "scene",
    "service",
    "property",
    "properties",
    "business",
    "option",
    "atmosphere",
}
_LOW_SIGNAL_TOKENS = {"good", "great", "nice", "place", "spot", "experience", "weak", "money", "long"}
_NEGATIVE_CUE_PATTERNS: list[tuple[str, str]] = [
    ("wait", "long waits"),
    ("service", "service issues"),
    ("value", "weak value for money"),
    ("noise", "noisy settings"),
    ("parking", "parking trouble"),
    ("clean", "cleanliness issues"),
    ("crowd", "crowded rooms"),
]
_SOURCE_SEMANTIC_HINTS = {
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
_FIRST_PERSON_STORY_RE = re.compile(
    r"\b(i|i'm|i’ve|ive|i'd|we|we're|we’ve|we'd|my|our|me|us|wife|husband|fiance|friend and i)\b",
    flags=re.I,
)
_RAW_STORY_PATTERNS = {
    "my last",
    "last trip",
    "last visit",
    "came here",
    "came in",
    "went here",
    "stopped in",
    "walked in",
    "walked out",
    "i came",
    "we came",
    "i ordered",
    "we ordered",
    "i decided",
    "we decided",
    "i wanted",
    "we wanted",
    "i stayed",
    "we stayed",
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
    "breakfast run",
    "i really enjoyed",
    "i was delighted",
    "i don't understand",
    "i love",
    "i usually go",
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

_FIRST_PERSON_STORY_RE_EXTRA = re.compile(
    r"\b(i've|i'll|we've|weve|we'll)\b",
    flags=re.I,
)

_RAW_STORY_PATTERNS_EXTRA = {
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
_RECENT_CONTEXT_ALIAS_GROUPS: dict[str, set[str]] = {
    "late night": {
        "late night",
        "late hours",
        "late night meals",
        "late night visits",
        "nightlife",
        "happy hour",
        "date night outings",
        "sit down dinner",
        "sit down meals",
    },
    "fast casual": {
        "fast casual",
        "fast casual meals",
        "quick casual meals",
        "weekday lunch",
        "family friendly settings",
    },
    "group dining": {
        "group dining",
        "family friendly settings",
        "celebration",
        "sit down meals",
    },
}
_CONTEXT_FIT_KEEP_TERMS = {
    "takeout",
    "delivery",
    "reservations",
    "late hours",
    "late night",
    "late night meals",
    "late night visits",
    "nightlife",
    "happy hour",
    "date night outings",
    "date night meals",
    "sit down dinner",
    "sit down meals",
    "fast casual",
    "fast casual meals",
    "quick casual meals",
    "group dining",
    "family friendly settings",
    "celebration",
}
_AVOID_TERM_NORMALIZATION = {
    "service": "service issues",
    "slow service": "slow service",
    "rude service": "rude or inattentive service",
    "value": "weak value for money",
    "noise": "noise and overly loud rooms",
    "parking": "parking trouble",
    "clean": "cleanliness issues",
    "crowd": "crowded rooms",
    "wait long": "long waits",
    "price value": "weak value for money",
    "cleanliness issue": "cleanliness issues",
    "noise issue": "noise and overly loud rooms",
}
_POSITIVE_SENTENCE_CUES = {
    "love",
    "loved",
    "favorite",
    "favourite",
    "best",
    "great",
    "excellent",
    "amazing",
    "delicious",
    "fantastic",
    "return",
    "come back",
    "recommend",
    "glorious",
    "must try",
}
_NEGATIVE_SENTENCE_CUES = {
    "awful",
    "worst",
    "burnt",
    "disappointed",
    "not returning",
    "long wait",
    "waited",
    "rude",
    "overpriced",
    "expensive",
    "cold",
    "bland",
    "bad",
    "poor",
    "dirty",
    "noise",
    "noisy",
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
}
_COMPLAINT_LABELS = {
    "noise_issue": "noise and overly loud rooms",
    "parking_trouble": "parking trouble",
    "rude_service": "rude or inattentive service",
    "value_issue": "weak value for money",
    "wait_long": "long waits",
    "cleanliness_issue": "cleanliness issues",
    "slow_service": "slow service",
}
_BAD_AVOID_TERMS = {
    "american",
    "seafood",
    "coffee tea",
    "dessert bakery",
    "cajun creole",
    "breakfast brunch",
}
_INTENT_TIME_PREFS: list[tuple[str, str]] = [
    ("breakfast_pref", "breakfast"),
    ("lunch_pref", "lunch"),
    ("dinner_pref", "dinner"),
    ("late_night_pref", "late-night meals"),
]
_INTENT_SCENE_PREFS: list[tuple[str, str]] = [
    ("family_scene_pref", "family-friendly settings"),
    ("group_scene_pref", "group dining"),
    ("date_scene_pref", "date-night meals"),
    ("nightlife_scene_pref", "nightlife"),
    ("fast_casual_pref", "quick casual meals"),
    ("sitdown_pref", "sit-down meals"),
]
_INTENT_PROPERTY_PREFS: list[tuple[str, str]] = [
    ("delivery_pref", "delivery"),
    ("takeout_pref", "takeout"),
    ("reservation_pref", "reservations"),
    ("weekend_pref", "weekend outings"),
]
_GENERIC_FIT_PHRASES = {
    "its core offering lines up with the user's broader dining taste.",
    "its service setup and practical details fit what this user usually values.",
    "it also matches the user's recent dining intent.",
    "its dining timing and occasion fit the user's usual outing pattern.",
    "reviews reinforce that fit with consistently positive evidence.",
    "its overall offering still overlaps with the user's broader preference profile.",
}
_CONCRETE_FIT_PATTERNS = {
    "serves the kinds of dishes the user comes back to most often",
    "the clearest overlap is around",
    "this business leans into",
    "the user's recent choices have drifted toward",
    "the business stays close to the user's stronger long-run preferences around",
    "the user's recent pattern still clusters around",
    "recent activity has leaned toward",
    "those same themes show up clearly in this business",
    "those details also appear in this business's offering",
    "the business shows the same pattern in its menu and review evidence",
    "older positive visits repeatedly highlighted",
    "older favorites kept returning to",
    "recent behavior points toward",
    "the user's recent outings keep clustering around",
    "this business supports that same pattern",
    "the same recent context also shows up in this business",
    "review evidence also suggests practical details that match this user's preferences",
    "it also matches practical preferences the user seems to care about",
}
_TIMING_ONLY_FIT_PATTERNS = {
    "the dining setting also lines up with this user's usual outing pattern",
    "it also matches the user's usual timing because this business is strongest around",
}
_GENERIC_SCENE_TERMS = {
    "breakfast",
    "lunch",
    "dinner",
    "weekend",
    "plans",
}
_BROAD_SCENE_REASON_TERMS = {
    "family friendly",
    "family friendly settings",
    "group dining",
    "nightlife",
    "date night",
    "date night outings",
    "date night meals",
    "fast casual spots",
    "quick casual meals",
    "sit down dining settings",
    "sit down meals",
    "late night meals",
}
_BROAD_SERVICE_REASON_TERMS = {
    "takeout",
    "delivery",
    "reservations",
    "friendly service",
    "attentive service",
    "fast service",
    "clean space",
    "good value",
    "large portions",
}
_BROAD_MERCHANT_SCENE_TERMS = {
    "breakfast",
    "lunch",
    "dinner",
    "weekend",
    "family friendly dining",
    "group dining",
    "sit down meals",
}
_BROAD_MERCHANT_PROPERTY_TERMS = {
    "takeout",
    "delivery",
    "reservations",
    "table service",
    "casual vibe",
}
_PRACTICAL_SIGNAL_TERMS = {
    "takeout",
    "delivery",
    "reservations",
    "late hours",
    "late night meals",
    "late night visits",
    "outdoor seating",
    "full bar",
    "beer and wine",
    "happy hour",
    "brunch spot",
}
_BAD_DISPLAY_TERMS = {
    "quick",
    "work",
    "coffee work",
    "older activity",
    "across older activity",
    "nightlife and quick",
    "quick lunch",
    "quick bites",
    "late night",
    "late-night",
}
_BROAD_CORE_TERMS = {
    "american",
    "seafood",
    "cajun",
    "cajun creole",
    "coffee tea",
    "dessert bakery",
    "nightlife",
    "family friendly",
    "group dining",
    "date night",
}
_BROAD_RECENT_TERMS = _BROAD_CORE_TERMS | {
    "late night",
    "late night meals",
    "breakfast brunch",
    "lunch",
    "dinner",
}
_BROAD_RECENT_PROPERTY_TERMS = {
    "takeout",
    "delivery",
    "reservations",
    "late hours",
    "weekend activity",
    "weekend outings",
}
_BROAD_MERCHANT_TIME_TERMS = {
    "breakfast",
    "lunch",
    "dinner",
    "weekend",
    "brunch",
}
_BAD_TERM_PHRASES = {
    "signature dishes or cuisines",
    "dining scenes",
    "strong time cues",
    "useful properties",
    "frequent strengths",
    "frequent complaints",
    "primary category",
    "primary geo area",
    "event planning and services",
    "event planning & services",
    "hotels and travel",
    "across older activity",
    "older activity",
    "most activity is around",
    "recent activity is concentrating around",
    "recent activity is clustering around",
    "working best around",
    "positive profile signals repeatedly center on",
    "earlier positive profile evidence repeatedly linked this user with",
    "review evidence repeatedly connects this user with",
    "coffee work",
    "nightlife and quick",
    "quick lunch",
    "quick bites",
}
_NON_FOOD_PRIMARY_CATEGORIES = {
    "event planning and services",
    "arts and entertainment",
    "hotels and travel",
    "hotels and travel",
    "active life",
    "shopping",
    "beauty and spas",
    "home services",
    "professional services",
    "financial services",
    "health and medical",
    "automotive",
    "local services",
    "public services and government",
    "religious organizations",
}

_THEME_LABELS = {
    "seafood": "seafood",
    "cajun_creole": "Cajun and Creole food",
    "coffee_tea": "coffee and tea",
    "dessert_bakery": "desserts and bakeries",
    "breakfast_brunch": "breakfast and brunch",
    "tacos_mexican": "Mexican and taco-style meals",
    "pizza": "pizza",
    "sandwich_deli": "sandwiches and deli-style meals",
    "burger_american": "burgers and casual American food",
    "asian_other": "Asian dishes",
    "vegan_veg": "vegetarian-friendly dishes",
    "bbq_smoked": "barbecue and smoked dishes",
    "cocktail_bar": "cocktails and bar-oriented dining",
}

_TERM_THEME_MAP = {
    "shrimp": "seafood",
    "oyster": "seafood",
    "oysters": "seafood",
    "crawfish": "seafood",
    "gumbo": "cajun_creole",
    "po boy": "sandwich_deli",
    "po boys": "sandwich_deli",
    "po-boy": "sandwich_deli",
    "po-boys": "sandwich_deli",
    "seafood": "seafood",
    "cajun and creole food": "cajun_creole",
    "cajun creole": "cajun_creole",
    "coffee": "coffee_tea",
    "coffee and tea": "coffee_tea",
    "coffee tea": "coffee_tea",
    "dessert": "dessert_bakery",
    "desserts": "dessert_bakery",
    "bakery": "dessert_bakery",
    "dessert bakery": "dessert_bakery",
    "breakfast": "breakfast_brunch",
    "brunch": "breakfast_brunch",
    "breakfast brunch": "breakfast_brunch",
    "taco": "tacos_mexican",
    "tacos": "tacos_mexican",
    "burrito": "tacos_mexican",
    "mexican": "tacos_mexican",
    "tacos mexican": "tacos_mexican",
    "pizza": "pizza",
    "sandwich": "sandwich_deli",
    "sandwiches": "sandwich_deli",
    "deli sandwich": "sandwich_deli",
    "burger": "burger_american",
    "burgers": "burger_american",
    "american": "burger_american",
    "asian other": "asian_other",
    "vegan and vegetarian dishes": "vegan_veg",
    "vegan vegetarian": "vegan_veg",
    "bbq": "bbq_smoked",
    "barbecue": "bbq_smoked",
    "smoked meats": "bbq_smoked",
    "cocktail bar": "cocktail_bar",
    "cocktails": "cocktail_bar",
}
_DISPLAY_TERM_REWRITE = {
    "coffee tea": "coffee and tea",
    "dessert bakery": "desserts and bakeries",
    "breakfast brunch": "breakfast and brunch",
    "cajun creole": "cajun and creole food",
    "burgers sandwiches": "burgers and sandwiches",
    "burger american": "burgers and American comfort food",
    "asian other": "Asian dishes",
    "tacos mexican": "Mexican tacos",
    "family friendly": "family-friendly settings",
    "date night": "date-night outings",
    "quick bite": "quick bites",
    "fast casual": "fast-casual spots",
    "sitdown": "sit-down dining",
    "salad light": "lighter salads",
    "vegan vegetarian": "vegan and vegetarian dishes",
    "wait long": "long waits",
    "noise issue": "noise and overly loud rooms",
    "service issue": "service issues",
    "value issues": "weak value for money",
    "value issue": "weak value for money",
    "price value": "good value for money",
    "beer bar": "beer bars",
    "rude service": "rude or inattentive service",
    "cleanliness issue": "cleanliness issues",
}
_TERM_CANONICAL_ALIAS = {
    "desserts and bakeries": "dessert bakery",
    "desserts and bakery": "dessert bakery",
    "dessert and bakery": "dessert bakery",
    "breakfast and brunch": "breakfast brunch",
    "family friendly settings": "family friendly",
    "family friendly dining": "family friendly",
    "family friendly meals": "family friendly",
    "date night outings": "date night",
    "date night meals": "date night",
    "date night dining": "date night",
    "quick casual meals": "fast casual",
    "quick casual spots": "fast casual",
    "fast casual meals": "fast casual",
    "fast casual spots": "fast casual",
    "sit down meals": "sitdown",
    "sit down dining": "sitdown",
    "sit down dinner": "sitdown",
    "sit down dining settings": "sitdown",
    "late night meals": "late night",
    "late night visits": "late night",
    "late night outings": "late night",
    "group dining plans": "group dining",
    "group dinners": "group dining",
    "weekend outings": "weekend",
    "weekend activity": "weekend",
}


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def parse_bucket_override(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except Exception:
            continue
    return sorted(set(out))


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_optional_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = normalize_legacy_project_path(raw)
        if not p.is_absolute():
            p = project_path(p)
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
    names = [
        "candidates_pretrim.parquet",
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
    ]
    for name in names:
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate file not found in {bucket_dir}")


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def clean_text(raw: Any, max_chars: int | None = None) -> str:
    txt = str(raw or "").replace("\r", " ").replace("\n", " ")
    txt = txt.replace("[SHORT]", " ").replace("[LONG]", " ")
    txt = re.sub(r"\bSHORT\b|\bLONG\b", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r'["\']?[a-z0-9_]+["\']?\s*:\s*\[[^\]]*\]', " ", txt, flags=re.I)
    txt = re.sub(r"[{}\[\]]", " ", txt)
    txt = re.sub(r"\s+([,.;:])", r"\1", txt)
    txt = re.sub(r"([,.;:])([^\s])", r"\1 \2", txt)
    txt = re.sub(r"\s+", " ", txt).strip(" ,.;:")
    txt = re.sub(r"\bweak value for(?:\s+money)?\b", "weak value for money", txt, flags=re.I)
    txt = re.sub(r"\bweak value for money(?:\s+money)+\b", "weak value for money", txt, flags=re.I)
    txt = re.sub(r"\bvalue for(?:\s+money)?\b", "value for money", txt, flags=re.I)
    txt = re.sub(r"\blong wait(?:s)?\b", "long waits", txt, flags=re.I)
    txt = re.sub(r"\b([a-z]+)(?:\s+\1){1,}\b", r"\1", txt, flags=re.I)
    txt = re.sub(
        r"\b(recent activity is concentrating around|recent activity is clustering around|review evidence repeatedly connects this user with|earlier positive profile evidence repeatedly linked this user with)(?:\s+\1)+\b",
        r"\1",
        txt,
        flags=re.I,
    )
    txt = re.sub(r"\band\s+and\b", "and", txt, flags=re.I)
    if txt.lower() in {"nan", "none", "null"}:
        return ""
    if max_chars and max_chars > 0 and len(txt) > max_chars:
        clipped = txt[: max_chars + 1]
        if " " in clipped[:-1]:
            txt = clipped[:max_chars].rsplit(" ", 1)[0]
        else:
            txt = clipped[:max_chars]
        txt = txt.rstrip(" ,.;:")
    return txt


def normalize_term(raw: Any, max_chars: int = 48) -> str:
    return clean_text(str(raw or "").replace("_", " ").replace("-", " ").replace("&", " and "), max_chars=max_chars).lower()


def is_bad_term_text(raw: Any) -> bool:
    text = normalize_term(raw, max_chars=80)
    if not text:
        return True
    if ":" in text:
        return True
    return any(phrase in text for phrase in _BAD_TERM_PHRASES)


def sanitize_primary_category(raw: Any) -> str:
    text = clean_text(str(raw or "").replace("_", " ").replace("&", " and "), max_chars=64).lower()
    if not text:
        return ""
    if text in _NON_FOOD_PRIMARY_CATEGORIES:
        return ""
    return text


def readable_term(raw: Any, max_chars: int = 48) -> str:
    text = clean_text(str(raw or "").replace("_", " ").replace("&", " and "), max_chars=0).lower()
    if not text:
        return ""
    text = _DISPLAY_TERM_REWRITE.get(text, text)
    text = re.sub(
        r"(?:\s*,\s*|\s+)(and|or|but|with|around|toward|towards|for|to|in|on|of)$",
        "",
        text,
        flags=re.I,
    ).strip(" ,.;:-")
    if not text or text in _BAD_DISPLAY_TERMS or text in _GENERIC_TERMS or text in _LOW_SIGNAL_TOKENS:
        return ""
    if is_bad_term_text(text):
        return ""
    return clean_text(text, max_chars=max_chars)


def ensure_terms(raw: Any, limit: int = 8) -> list[str]:
    if isinstance(raw, np.ndarray):
        values = raw.tolist()
    elif isinstance(raw, list):
        values = raw
    elif isinstance(raw, tuple):
        values = list(raw)
    elif raw is None or (isinstance(raw, float) and np.isnan(raw)):
        values = []
    else:
        text = str(raw or "").strip()
        values = re.split(r"[|,;/]+", text) if text else []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_term(value, max_chars=48)
        if not text or text in _GENERIC_TERMS or text in _LOW_SIGNAL_TOKENS or is_bad_term_text(text):
            continue
        if len(text.split()) == 1 and len(text) <= 2:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def dedupe_terms(*term_lists: list[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term_list in term_lists:
        for value in term_list:
            text = normalize_term(value, max_chars=48)
            if not text or text in seen or text in _GENERIC_TERMS or text in _LOW_SIGNAL_TOKENS or is_bad_term_text(text):
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                return out
    return out


def join_terms_natural(terms: list[str], limit: int = 4) -> str:
    clean: list[str] = []
    for term in terms:
        display = readable_term(term, max_chars=48)
        if not display:
            norm = normalize_term(term, max_chars=48)
            if norm in _CONTEXT_FIT_KEEP_TERMS:
                display = norm
        if display:
            clean.append(display)
    clean = clean[:limit]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def filter_terms_display(
    terms: list[str],
    *,
    limit: int = 4,
    broad: set[str] | None = None,
) -> list[str]:
    broad_terms = broad or set()
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        text = readable_term(term, max_chars=48)
        if not text or text in broad_terms:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def specific_terms(terms: list[str], *, limit: int = 4, broad: set[str] | None = None) -> list[str]:
    broad_terms = broad or _BROAD_CORE_TERMS
    out: list[str] = []
    for term in dedupe_terms(terms, limit=max(limit * 2, limit)):
        lower = normalize_term(term, max_chars=48)
        if not lower or lower in broad_terms:
            continue
        out.append(term)
        if len(out) >= limit:
            break
    return out


def canonical_term_key(raw: Any) -> str:
    txt = normalize_term(raw, max_chars=48)
    if not txt:
        return ""
    txt = _TERM_CANONICAL_ALIAS.get(txt, txt)
    if txt.endswith("ies") and len(txt) > 5:
        txt = txt[:-3] + "y"
    elif txt.endswith("s") and len(txt) > 4 and not txt.endswith(("ss", "ous")):
        txt = txt[:-1]
    txt = _TERM_CANONICAL_ALIAS.get(txt, txt)
    return txt


def term_theme_key(raw: Any) -> str:
    key = canonical_term_key(raw)
    if not key:
        return ""
    return _TERM_THEME_MAP.get(key, "")


def theme_label(theme_key: str) -> str:
    return _THEME_LABELS.get(theme_key, theme_key.replace("_", " "))


def collect_theme_examples(raw_terms: list[str] | set[str], *, limit_per_theme: int = 3) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for raw in raw_terms:
        theme = term_theme_key(raw)
        display = normalize_term(raw, max_chars=48)
        if not theme or not display:
            continue
        prev = out.get(theme, [])
        out[theme] = dedupe_terms(prev, [display], limit=limit_per_theme)
    return out


def best_theme_overlap(
    user_terms: list[str] | set[str],
    merchant_terms: list[str] | set[str],
) -> tuple[str, list[str], list[str]]:
    user_map = collect_theme_examples(user_terms)
    merchant_map = collect_theme_examples(merchant_terms)
    shared = [theme for theme in user_map.keys() if theme in merchant_map]
    if not shared:
        return "", [], []
    shared.sort(
        key=lambda theme: (
            len(user_map.get(theme, [])) + len(merchant_map.get(theme, [])),
            len(user_map.get(theme, [])),
            len(merchant_map.get(theme, [])),
        ),
        reverse=True,
    )
    theme = shared[0]
    return theme, user_map.get(theme, []), merchant_map.get(theme, [])


def intersect_terms_display(
    left_terms: list[str] | set[str],
    right_terms: list[str] | set[str],
    *,
    limit: int = 3,
    broad: set[str] | None = None,
) -> list[str]:
    broad_terms = broad or set()
    right_map: dict[str, str] = {}
    for term in right_terms:
        key = canonical_term_key(term)
        if key and key not in right_map:
            right_map[key] = normalize_term(term, max_chars=48)
    out: list[str] = []
    seen: set[str] = set()
    for term in left_terms:
        display = normalize_term(term, max_chars=48)
        key = canonical_term_key(term)
        if not key or key in broad_terms or key not in right_map:
            continue
        if display in seen:
            continue
        seen.add(display)
        out.append(display)
        if len(out) >= limit:
            break
    return out


def best_text(*values: Any, max_chars: int = 320) -> str:
    for value in values:
        text = clean_text(value, max_chars=max_chars)
        if text:
            return text
    return ""


def compact_terms_text(raw: Any, limit: int = 6) -> str:
    return join_terms_natural(ensure_terms(raw, limit=limit), limit=limit)


def stringify_terms(raw: Any, limit: int = 8) -> str:
    return "|".join(ensure_terms(raw, limit=limit))


def parse_terms_blob(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [clean_text(part, max_chars=48).lower() for part in text.split("|") if clean_text(part, max_chars=48)]


def parse_typed_text_terms(raw: Any, limit: int = 8) -> list[str]:
    text = clean_text(raw, max_chars=480)
    if not text:
        return []
    values: list[str] = []
    for part in re.split(r"[;]+", text):
        chunk = clean_text(part, max_chars=160)
        if not chunk:
            continue
        if ":" in chunk:
            chunk = chunk.split(":", 1)[1]
        values.extend(re.split(r"[,/|]+", chunk))
    return ensure_terms(values, limit=limit)


def parse_typed_text_facets(raw: Any, limit_per_facet: int = 4) -> dict[str, list[str]]:
    text = clean_text(raw, max_chars=480)
    out: dict[str, list[str]] = {}
    if not text:
        return out
    for part in text.split(";"):
        seg = clean_text(part, max_chars=180)
        if not seg:
            continue
        facet = "other"
        values = seg
        if ":" in seg:
            facet, values = seg.split(":", 1)
            facet = normalize_term(facet, max_chars=24) or "other"
        parsed = ensure_terms(re.split(r"[,/|]+", values), limit=limit_per_facet)
        if not parsed:
            continue
        prev = out.get(facet, [])
        out[facet] = dedupe_terms(prev, parsed, limit=limit_per_facet)
    return out


def extract_labeled_items_from_text(raw: Any, labels: list[str], limit: int = 4) -> list[str]:
    text = clean_text(raw, max_chars=700)
    if not text:
        return []
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*:\s*([^.;]+)", text, flags=re.I)
        if not match:
            continue
        values = ensure_terms(re.split(r"[,/|]+", match.group(1)), limit=limit)
        if values:
            return values
    return []


def extract_recent_intent_facets(raw: Any) -> dict[str, list[str]]:
    cuisine_terms = dedupe_terms(
        extract_labeled_items_from_text(raw, ["Latest known cuisine intent"], limit=4),
        extract_labeled_items_from_text(raw, ["Recent broader cuisine drift"], limit=4),
        limit=4,
    )
    meal_terms = dedupe_terms(
        extract_labeled_items_from_text(raw, ["Latest known meals"], limit=4),
        extract_labeled_items_from_text(raw, ["Recent meals"], limit=4),
        limit=4,
    )
    scene_terms = dedupe_terms(
        extract_labeled_items_from_text(raw, ["Latest known dining scenes"], limit=4),
        extract_labeled_items_from_text(raw, ["Recent dining scenes"], limit=4),
        limit=4,
    )
    property_terms = dedupe_terms(
        extract_labeled_items_from_text(raw, ["Latest known useful properties"], limit=4),
        extract_labeled_items_from_text(raw, ["Recent useful properties"], limit=4),
        limit=4,
    )
    return {
        "cuisine": cuisine_terms,
        "meal": meal_terms,
        "scene": scene_terms,
        "property": property_terms,
    }


def extract_terms_from_sentential_text(raw: Any, limit: int = 8) -> list[str]:
    text = clean_text(raw, max_chars=420)
    if not text:
        return []
    text = re.sub(
        r"\b(signature dishes or cuisines|categories|dining scenes|strong time cues|useful properties|frequent strengths|frequent complaints|primary geo area|primary category|meals|price range)\b\s*:\s*",
        " ",
        text,
        flags=re.I,
    )
    text = text.replace(".", ",")
    return ensure_terms(re.split(r"[,;]+", text), limit=limit)


def split_review_sentences(raw: Any) -> list[str]:
    text = clean_text(raw, max_chars=1200)
    if not text:
        return []
    text = text.replace("[SHORT]", " ").replace("[LONG]", " ")
    parts = re.split(r"(?<=[.!?])\s+|(?<=[a-z])\s+(?=[A-Z])", text)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        sent = clean_text(part, max_chars=180)
        if not sent:
            continue
        if len(sent) < 24:
            continue
        if sent.lower() in seen:
            continue
        seen.add(sent.lower())
        out.append(sent)
    return out


def parse_json_text_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [clean_text(v, max_chars=220) for v in raw if clean_text(v, max_chars=220)]
    if isinstance(raw, tuple):
        return [clean_text(v, max_chars=220) for v in raw if clean_text(v, max_chars=220)]
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, list):
            return [clean_text(v, max_chars=220) for v in payload if clean_text(v, max_chars=220)]
    return []


def pick_review_sentences(raw: Any, polarity: str, limit: int = 2) -> list[str]:
    out: list[str] = []
    for sent in split_review_sentences(raw):
        usable = usable_source_support_text(sent, negative=(polarity == "negative"), max_chars=220)
        if not usable:
            continue
        pos_hits, neg_hits = sentiment_counts(sent)
        if polarity == "positive" and pos_hits > 0 and neg_hits == 0:
            out.append(usable)
        elif polarity == "negative" and neg_hits > 0 and neg_hits >= pos_hits:
            out.append(usable)
        if len(out) >= limit:
            break
    return out


def top_numeric_labels(row: pd.Series, specs: list[tuple[str, str]], threshold: float, limit: int = 3) -> list[str]:
    pairs: list[tuple[float, str]] = []
    for key, label in specs:
        try:
            score = float(row.get(key, 0.0) or 0.0)
        except Exception:
            score = 0.0
        if score >= threshold:
            pairs.append((score, label))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in pairs[:limit]]


def join_blocks_with_budget(texts: list[str], max_chars: int) -> str:
    kept: list[str] = []
    total = 0
    for raw in texts:
        text = clean_text(raw)
        if not text:
            continue
        extra = len(text) + (1 if kept else 0)
        if kept and total + extra > max_chars:
            break
        if not kept and len(text) > max_chars:
            return clean_text(text, max_chars=max_chars)
        kept.append(text)
        total += extra
    return clean_text(" ".join(kept), max_chars=max_chars)


def narrative_text(raw: Any, max_chars: int | None = None) -> str:
    txt = clean_text(raw, max_chars=0).strip(" ,;:")
    if not txt:
        return ""
    txt = re.sub(r"\bweak value for money(?:\s+money)+\b", "weak value for money", txt, flags=re.I)
    txt = re.sub(r"\b([a-z]+)(?:\s+\1){1,}\b", r"\1", txt, flags=re.I)
    txt = re.sub(r"\s*,\s*\.", ".", txt)
    txt = re.sub(r"\s*;\s*\.", ".", txt)
    txt = re.sub(r"\s*:\s*\.", ".", txt)
    txt = re.sub(
        r"(?:\s*,\s*|\s+)(and|or|but|with|around|toward|towards|for|to|in|on|of)\s*\.$",
        ".",
        txt,
        flags=re.I,
    )
    txt = re.sub(
        r"(?:\s*,\s*|\s+)(and|or|but|with|around|toward|towards|for|to|in|on|of)$",
        "",
        txt,
        flags=re.I,
    ).strip(" ,;:")
    if not re.search(r"[.!?]$", txt):
        txt = f"{txt}."
    txt = re.sub(r"\s+", " ", txt).strip()
    if max_chars and max_chars > 0 and len(txt) > max_chars:
        clipped = txt[: max_chars + 1]
        if " " in clipped[:-1]:
            txt = clipped[:max_chars].rsplit(" ", 1)[0]
        else:
            txt = clipped[:max_chars]
        txt = txt.rstrip(" ,;:")
        txt = re.sub(r"\bweak value for\b", "weak value for money", txt, flags=re.I)
        txt = re.sub(r"\bvalue for\b", "value for money", txt, flags=re.I)
        if txt and not re.search(r"[.!?]$", txt):
            txt = f"{txt}."
    return txt


def join_narrative_blocks_with_budget(texts: list[str], max_chars: int) -> str:
    kept: list[str] = []
    total = 0
    for raw in texts:
        text = narrative_text(raw)
        if not text:
            continue
        extra = len(text) + (1 if kept else 0)
        if kept and total + extra > max_chars:
            break
        if not kept and len(text) > max_chars:
            return narrative_text(text, max_chars=max_chars)
        kept.append(text)
        total += extra
    return narrative_text(" ".join(kept), max_chars=max_chars)


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = clean_text(value, max_chars=0)
        if text:
            return text
    return ""


def select_complete_narrative_block(*values: Any, max_chars: int = 140, min_words: int = 5) -> str:
    for raw in values:
        text = clean_text(raw, max_chars=0)
        if not text:
            continue
        sentences = [
            clean_text(part, max_chars=0)
            for part in re.split(r"(?<=[.!?])\s+", text)
            if clean_text(part, max_chars=0)
        ]
        kept: list[str] = []
        total = 0
        for sentence in sentences:
            sent = narrative_text(sentence)
            if not sent or len(sent.split()) < int(min_words):
                continue
            extra = len(sent) + (1 if kept else 0)
            if kept and total + extra > max_chars:
                break
            if not kept and len(sent) > max_chars:
                trimmed = narrative_text(sent, max_chars=max_chars)
                if trimmed and len(trimmed.split()) >= int(min_words):
                    return trimmed
                break
            kept.append(sent)
            total += extra
        if kept:
            return narrative_text(" ".join(kept), max_chars=max_chars)
        trimmed = narrative_text(text, max_chars=max_chars)
        if trimmed:
            return trimmed
    return ""


def summarize_sentence_terms(raw: Any, limit: int = 4) -> list[str]:
    sentences = split_review_sentences(raw)
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        for term in extract_terms_from_sentential_text(sentence, limit=6):
            key = normalize_term(term, max_chars=48)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= max(1, int(limit)):
                return out
    return out


_NEGATIVE_SENTENCE_AVOID_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(long waits?|wait(?:ed|ing)? too long|line was long|queue was long)\b", flags=re.I), "long waits"),
    (re.compile(r"\b(slow service|service was slow)\b", flags=re.I), "slow service"),
    (re.compile(r"\b(rude|unfriendly|not friendly|inattentive|ignored us|staff was rude|server was rude)\b", flags=re.I), "rude or inattentive service"),
    (re.compile(r"\b(could not sit|couldn't sit|not seated|turned us away|no reservations? (?:were )? available|host stand|host was rude|rude upon arrival|not welcoming)\b", flags=re.I), "seating or reservation issues"),
    (re.compile(r"\b(never clean|not clean|dirty|filthy|unclean|messy)\b", flags=re.I), "cleanliness issues"),
    (re.compile(r"\b(noisy|too loud|very loud|noise|crowded|packed)\b", flags=re.I), "noise and overly loud rooms"),
    (re.compile(r"\b(overpriced|too expensive|pricey|not worth|weak value|bad value)\b", flags=re.I), "weak value for money"),
    (re.compile(r"\b(cold food|food was cold|bad meal|not enjoyable|very disappointing|bland|bland food|dry food|stale|soggy|tasteless|not fresh|inconsistent)\b", flags=re.I), "inconsistent food quality"),
    (re.compile(r"\b(wrong order|forgot our order|forgot the order|missing order|never brought)\b", flags=re.I), "service issues"),
    (re.compile(r"\b(small portions|tiny portions)\b", flags=re.I), "weak value for money"),
    (re.compile(r"\b(parking|hard to park|no parking)\b", flags=re.I), "parking trouble"),
]


def extract_negative_sentence_avoid_terms(sentences: list[str], limit: int = 4) -> list[str]:
    terms: list[str] = []
    for sentence in sentences:
        txt = clean_text(sentence, max_chars=240)
        if not txt:
            continue
        for pattern, label in _NEGATIVE_SENTENCE_AVOID_PATTERNS:
            if pattern.search(txt):
                terms.append(label)
    return dedupe_avoid_terms(terms, limit=limit)


def choose_distinct_blocks(
    values: list[str],
    *,
    max_chars: int,
    max_blocks: int,
    min_new_tokens: int = 2,
) -> str:
    kept: list[str] = []
    for raw in values:
        block = narrative_text(raw)
        if not block:
            continue
        if kept and not has_meaningful_addition(" ".join(kept), block, min_new_tokens=min_new_tokens):
            continue
        kept.append(block)
        if len(kept) >= max(1, int(max_blocks)):
            break
    return join_narrative_blocks_with_budget(kept, max_chars=max_chars)


def reason_token_key(text: Any) -> set[str]:
    raw = clean_text(text, max_chars=220).lower()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", raw)
        if len(token) >= 4 and token not in {"user", "business", "recent", "activity", "reviews", "matches", "match"}
    }


def append_reason_if_distinct(out: list[str], candidate: Any, *, max_chars: int = 190) -> None:
    text = narrative_text(candidate, max_chars=max_chars)
    if not text:
        return
    cand_tokens = reason_token_key(text)
    if not cand_tokens:
        return
    for existing in out:
        existing_tokens = reason_token_key(existing)
        if not existing_tokens:
            continue
        overlap = len(cand_tokens & existing_tokens)
        if overlap >= min(len(cand_tokens), len(existing_tokens)) * 0.7:
            return
    out.append(text)


def build_balanced_evidence_basis(user_basis: str, merchant_basis: str, *, total_max_chars: int = 420) -> str:
    side_budget = max(220, total_max_chars // 2 - 20)
    user_text = select_complete_narrative_block(user_basis, max_chars=side_budget, min_words=5)
    merchant_text = select_complete_narrative_block(merchant_basis, max_chars=side_budget, min_words=5)
    if user_text and merchant_text:
        return narrative_text(
            f"User side: {user_text} Business side: {merchant_text}",
            max_chars=total_max_chars,
        )
    if user_text:
        return narrative_text(f"User side: {user_text}", max_chars=total_max_chars)
    if merchant_text:
        return narrative_text(f"Business side: {merchant_text}", max_chars=total_max_chars)
    return ""


def build_semantic_fact(
    *,
    scope: str,
    polarity: str,
    text: Any,
    facet_type: str,
    evidence_source: str = "",
    terms: list[str] | None = None,
    specificity: int = 0,
    visible_for_prompt: bool = True,
    max_chars: int = 320,
) -> dict[str, Any] | None:
    fact_text = clean_text(text, max_chars=max_chars)
    if not fact_text:
        return None
    fact_terms = dedupe_terms(list(terms or []), limit=6)
    return {
        "scope": str(scope or "").strip().lower(),
        "polarity": str(polarity or "").strip().lower(),
        "facet_type": str(facet_type or "").strip().lower(),
        "text": fact_text,
        "terms": fact_terms,
        "specificity": int(max(0, specificity)),
        "evidence_source": clean_text(evidence_source, max_chars=64),
        "visible_for_prompt": bool(visible_for_prompt),
    }


def semantic_facts_json(*facts: dict[str, Any] | None) -> str:
    payload = [fact for fact in facts if fact]
    if not payload:
        return "[]"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def count_semantic_facts(facts: list[dict[str, Any] | None], *, scope: str | None = None, polarity: str | None = None) -> int:
    total = 0
    for fact in facts:
        if not fact:
            continue
        if scope and str(fact.get("scope", "")).strip().lower() != str(scope).strip().lower():
            continue
        if polarity and str(fact.get("polarity", "")).strip().lower() != str(polarity).strip().lower():
            continue
        total += 1
    return int(total)


def compose_user_preference_sentence(
    focus_terms: list[str],
    scene_terms: list[str],
    service_terms: list[str],
    *,
    max_chars: int,
) -> str:
    focus_terms = dedupe_terms(focus_terms, limit=4)
    scene_terms = filter_terms_display(scene_terms, limit=3, broad=_BROAD_SCENE_REASON_TERMS | _GENERIC_SCENE_TERMS)
    service_terms = filter_terms_display(service_terms, limit=3, broad=_BROAD_SERVICE_REASON_TERMS)
    blocks: list[str] = []
    if focus_terms:
        blocks.append(
            f"Across repeated visits, the clearest stable preference is for places built around {join_terms_natural(focus_terms, limit=4)}."
        )
    if scene_terms:
        blocks.append(
            f"Those choices most often land in {join_terms_natural(scene_terms, limit=3)} settings."
        )
    if service_terms:
        blocks.append(
            f"Useful practical details for this user usually include {join_terms_natural(service_terms, limit=3)}."
        )
    return choose_distinct_blocks(blocks, max_chars=max_chars, max_blocks=3)


def compose_recent_intent_sentence(
    recent_terms: list[str],
    recent_scene_terms: list[str],
    *,
    max_chars: int,
) -> str:
    recent_terms = dedupe_terms(recent_terms, limit=4)
    recent_scene_terms = filter_terms_display(
        recent_scene_terms,
        limit=3,
        broad=_BROAD_SCENE_REASON_TERMS | _GENERIC_SCENE_TERMS | {"late night meals"},
    )
    blocks: list[str] = []
    if recent_terms:
        blocks.append(
            f"More recent activity has tilted toward {join_terms_natural(recent_terms, limit=4)}."
        )
    if recent_scene_terms:
        blocks.append(
            f"Those recent outings most often happen in {join_terms_natural(recent_scene_terms, limit=3)} settings."
        )
    return choose_distinct_blocks(blocks, max_chars=max_chars, max_blocks=2)


def compose_recent_context_sentence(
    recent_scene_terms: list[str],
    recent_property_terms: list[str],
    *,
    max_chars: int,
) -> str:
    scene_terms = [
        term
        for term in dedupe_terms(recent_scene_terms, limit=4)
        if normalize_term(term, max_chars=48) not in _GENERIC_SCENE_TERMS
    ]
    property_terms = dedupe_terms(recent_property_terms, limit=3)
    blocks: list[str] = []
    if scene_terms:
        blocks.append(
            f"Recent outings keep clustering around {join_terms_natural(scene_terms, limit=3)}."
        )
    if property_terms:
        blocks.append(
            f"Those visits also lean toward places with {join_terms_natural(property_terms, limit=3)}."
        )
    return choose_distinct_blocks(blocks, max_chars=max_chars, max_blocks=2)


def _should_keep_supporting_text(
    base_texts: list[str],
    candidate_text: str,
    *,
    min_new_tokens: int = 1,
) -> bool:
    candidate = narrative_text(candidate_text, max_chars=260)
    if not candidate:
        return False
    base = " ".join(clean_text(text, max_chars=220) for text in base_texts if clean_text(text, max_chars=220))
    if not base:
        return True
    if has_meaningful_addition(base, candidate, min_new_tokens=min_new_tokens):
        return True
    lower = candidate.lower()
    evidence_role_cues = (
        "past ",
        "older ",
        "earlier ",
        "reviews ",
        "review ",
        "evidence ",
        "favorites ",
        "high-satisfaction",
    )
    return len(candidate.split()) >= 8 and any(cue in lower for cue in evidence_role_cues)


def compose_avoidance_sentence(avoid_terms: list[str], *, max_chars: int) -> str:
    avoid_terms = dedupe_avoid_terms(avoid_terms, limit=4)
    if not avoid_terms:
        return ""
    blocks = [
        f"This user is most likely to pull back when a place involves {join_terms_natural(avoid_terms, limit=4)}.",
    ]
    return choose_distinct_blocks(blocks, max_chars=max_chars, max_blocks=1)


def build_history_anchor_text(
    *,
    stable_text: str,
    recent_text: str,
    avoid_text: str,
    positive_terms: list[str],
    positive_sentences: list[str],
    profile_text_long: str,
    profile_pos_text: str = "",
    fallback_terms: list[str] | None = None,
) -> str:
    def _allow(candidate: str) -> str:
        text = narrative_text(candidate, max_chars=220)
        if not text:
            return ""
        return text

    positive_sentence = select_complete_narrative_block(*positive_sentences, max_chars=180, min_words=6)
    sentence_terms = summarize_semantic_review_terms(
        positive_sentence or profile_text_long,
        limit=4,
        broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
    )
    if sentence_terms:
        anchor = _allow(
            f"Past high-satisfaction visits repeatedly highlighted {join_terms_natural(sentence_terms, limit=4)}."
        )
        if anchor:
            return anchor
    if positive_terms:
        anchor = _allow(f"Past favorites repeatedly point back to {join_terms_natural(positive_terms, limit=4)}.")
        if anchor:
            return anchor
    if positive_sentence:
        sentence_terms = summarize_semantic_review_terms(
            positive_sentence,
            limit=4,
            broad=_BROAD_CORE_TERMS,
        )
        if sentence_terms:
            candidate = _allow(
                f"Earlier positive visits explicitly praised {join_terms_natural(sentence_terms, limit=4)}."
            )
        else:
            candidate = ""
        if candidate:
            return candidate
    fallback_anchor_terms = specific_terms(
        list(fallback_terms or []) + positive_terms + parse_typed_text_terms(profile_pos_text, limit=8),
        limit=4,
        broad=_BROAD_RECENT_TERMS,
    )
    if fallback_anchor_terms:
        candidate = _allow(
            f"Older favorites most often centered on {join_terms_natural(fallback_anchor_terms, limit=4)}."
        )
        if candidate:
            return candidate
    return ""


def build_user_basis_text(row: pd.Series, max_chars: int = 180) -> str:
    core_blocks = [
        select_complete_narrative_block(row.get("stable_preferences_text"), max_chars=120, min_words=5),
        select_complete_narrative_block(row.get("recent_intent_text_v2"), max_chars=100, min_words=5),
        select_complete_narrative_block(row.get("avoidance_text_v2"), max_chars=90, min_words=5),
        select_complete_narrative_block(row.get("history_anchor_hint_text"), max_chars=110, min_words=5),
    ]
    blocks = [b for b in core_blocks if b]
    if len(blocks) < 3:
        semantic_profile = select_complete_narrative_block(
            row.get("user_semantic_profile_text_v2"),
            max_chars=150,
            min_words=8,
        )
        if semantic_profile:
            blocks.append(semantic_profile)
    if len(blocks) < 3:
        user_evidence = select_complete_narrative_block(row.get("user_evidence_text_v2"), max_chars=120, min_words=5)
        if user_evidence:
            blocks.append(user_evidence)
    return choose_distinct_blocks(blocks, max_chars=max_chars, max_blocks=3)


def build_merchant_basis_text(row: pd.Series, max_chars: int = 180) -> str:
    name = clean_text(row.get("name"), max_chars=48)
    dish_terms = dedupe_terms(parse_terms_blob(row.get("merchant_dish_terms_v2")), limit=3)
    strength_terms = filter_terms_display(parse_terms_blob(row.get("merchant_strength_terms_v2")), limit=3, broad=_BROAD_SERVICE_REASON_TERMS)
    risk_terms = filter_terms_display(parse_terms_blob(row.get("merchant_risk_terms_v2")), limit=3)
    blocks: list[str] = []
    core = select_complete_narrative_block(row.get("core_offering_text"), max_chars=150, min_words=5)
    strength = select_complete_narrative_block(row.get("strengths_text"), max_chars=120, min_words=5)
    risk = select_complete_narrative_block(row.get("risk_points_text"), max_chars=110, min_words=5)
    core_lower = clean_text(core, max_chars=0).lower()
    generic_core = (
        bool(re.search(r"\bis a [a-z0-9 and]+ business in\b", core_lower))
        or "mainly centers on" in core_lower
        or "business under consideration" in core_lower
    )
    if dish_terms:
        if name:
            blocks.append(f"{name} is best known for {join_terms_natural(dish_terms, limit=3)}.")
        else:
            blocks.append(f"The menu and review evidence repeatedly center on {join_terms_natural(dish_terms, limit=3)}.")
    if strength:
        blocks.append(strength)
    elif strength_terms:
        blocks.append(f"Reviews repeatedly praise {join_terms_natural(strength_terms, limit=3)}.")
    if risk_terms:
        blocks.append(f"Common risks include {join_terms_natural(risk_terms, limit=3)}.")
    elif risk:
        blocks.append(risk)
    if core and (not generic_core or not blocks):
        blocks.append(core)
    elif name and not blocks:
        blocks.append(f"{name} is the business under consideration.")
    if len(blocks) < 3 and clean_text(row.get("scene_fit_text")):
        blocks.append(select_complete_narrative_block(row.get("scene_fit_text"), max_chars=110, min_words=5))
    if len(blocks) < 3 and clean_text(row.get("merchant_semantic_profile_text_v2")):
        semantic_profile = select_complete_narrative_block(
            row.get("merchant_semantic_profile_text_v2"),
            max_chars=140,
            min_words=8,
        )
        if semantic_profile:
            blocks.append(semantic_profile)
    return choose_distinct_blocks([b for b in blocks if b], max_chars=max_chars, max_blocks=5)


def build_terms_sentence(lead: str, terms: list[str], *, limit: int = 4, max_chars: int = 180) -> str:
    joined = join_terms_natural(terms, limit=limit)
    if not joined:
        return ""
    return narrative_text(f"{lead} {joined}", max_chars=max_chars)


def build_dish_preference_sentence(
    dish_terms: list[str],
    cuisine_terms: list[str],
    *,
    max_chars: int,
) -> str:
    dishes = specific_terms(dish_terms, limit=4)
    cuisines = dedupe_terms(cuisine_terms, limit=3)
    if dishes:
        return narrative_text(
            f"Repeated visits most often come back to dishes like {join_terms_natural(dishes, limit=4)}.",
            max_chars=max_chars,
        )
    if cuisines:
        return narrative_text(
            f"Across repeated visits, the strongest cuisine pull is toward {join_terms_natural(cuisines, limit=3)}.",
            max_chars=max_chars,
        )
    return ""


def text_token_set(raw: Any) -> set[str]:
    text = clean_text(raw, max_chars=240).lower()
    stop_tokens = {
        "across",
        "past",
        "older",
        "visit",
        "visits",
        "favorites",
        "favorite",
        "favorites",
        "repeated",
        "repeatedly",
        "keep",
        "keeps",
        "coming",
        "most",
        "often",
        "centered",
        "points",
        "point",
        "toward",
        "towards",
        "profile",
        "evidence",
        "cues",
        "suggest",
        "suggests",
        "responds",
        "likely",
        "pull",
        "back",
        "when",
        "involves",
        "places",
        "place",
        "lean",
        "leans",
        "recent",
        "activity",
        "outings",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) >= 4 and token not in stop_tokens
    }


def has_meaningful_addition(base_text: Any, candidate_text: Any, min_new_tokens: int = 2) -> bool:
    base_tokens = text_token_set(base_text)
    candidate_tokens = text_token_set(candidate_text)
    if not candidate_tokens:
        return False
    return len([tok for tok in candidate_tokens if tok not in base_tokens]) >= int(min_new_tokens)


def is_concrete_fit_text(raw: Any) -> bool:
    text = clean_text(raw, max_chars=0).lower()
    if not text:
        return False
    if any(pattern in text for pattern in _CONCRETE_FIT_PATTERNS):
        return True
    concrete_terms = {
        "seafood",
        "cajun",
        "creole",
        "taco",
        "pizza",
        "cocktail",
        "coffee",
        "tea",
        "sandwich",
        "brunch",
        "barbecue",
        "vegan",
        "dessert",
        "bakery",
        "shrimp",
        "oyster",
    }
    return any(term in text for term in concrete_terms) and any(
        cue in text
        for cue in {
            "comes back to most often",
            "clearest overlap",
            "leans into",
            "drifted toward",
            "practical preferences",
        }
    )


def is_timing_only_fit_text(raw: Any) -> bool:
    text = clean_text(raw, max_chars=0).lower()
    if not text:
        return False
    if is_concrete_fit_text(text):
        return False
    return any(pattern in text for pattern in _TIMING_ONLY_FIT_PATTERNS)


def sentiment_counts(text: str) -> tuple[int, int]:
    lower = clean_text(text, max_chars=220).lower()
    pos = sum(1 for cue in _POSITIVE_SENTENCE_CUES if cue in lower)
    neg = sum(1 for cue in _NEGATIVE_SENTENCE_CUES if cue in lower)
    if "not bad" in lower or "wasn't bad" in lower:
        pos += 1
        neg = max(0, neg - 1)
    return pos, neg


def summarize_semantic_review_terms(
    raw: Any,
    *,
    limit: int = 4,
    broad: set[str] | None = None,
) -> list[str]:
    broad_terms = (_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS) if broad is None else broad
    extracted: list[str] = []
    for sentence in split_review_sentences(raw):
        usable = usable_source_support_text(sentence, max_chars=220)
        if not usable:
            continue
        pos_hits, neg_hits = sentiment_counts(sentence)
        if pos_hits <= 0 or neg_hits > pos_hits:
            continue
        extracted.extend(
            [
                term
                for term in extract_terms_from_sentential_text(usable, limit=max(limit + 2, limit))
                if is_semantic_support_term(term)
            ]
        )
        extracted.extend(
            [
                term
                for term in summarize_sentence_terms(usable, limit=max(limit + 2, limit))
                if is_semantic_support_term(term)
            ]
        )
    return specific_terms(dedupe_terms(extracted, limit=max(limit * 2, limit)), limit=limit, broad=broad_terms)


def summarize_semantic_review_avoid_terms(raw: Any, *, limit: int = 4) -> list[str]:
    avoid_terms: list[str] = []
    for sentence in split_review_sentences(raw):
        usable = usable_source_support_text(sentence, negative=True, max_chars=220)
        if not usable:
            continue
        pos_hits, neg_hits = sentiment_counts(sentence)
        if neg_hits <= 0 or neg_hits < pos_hits:
            continue
        avoid_terms.extend(extract_negative_sentence_avoid_terms([usable], limit=limit))
        avoid_terms.extend(
            [
                normalize_avoid_term(term)
                for term in summarize_sentence_terms(usable, limit=max(limit + 2, limit))
                if is_experience_avoid_term(term) and is_semantic_support_term(term, negative=True)
            ]
        )
    return dedupe_avoid_terms(avoid_terms, limit=limit)


def build_review_evidence_summary(
    *,
    positive_sentences: list[str],
    negative_sentences: list[str],
    complaint_terms: list[str],
    positive_terms: list[str] | None = None,
    negative_terms: list[str] | None = None,
    max_chars: int = 260,
) -> str:
    positive_terms = specific_terms(
        list(positive_terms or []) + summarize_sentence_terms(" ".join(positive_sentences), limit=6),
        limit=4,
        broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
    )
    negative_terms = dedupe_avoid_terms(
        complaint_terms,
        list(negative_terms or []),
        [
            normalize_avoid_term(term)
            for term in summarize_sentence_terms(" ".join(negative_sentences), limit=6)
            if is_experience_avoid_term(term)
        ],
        limit=4,
    )
    blocks: list[str] = []
    if positive_terms:
        blocks.append(
            narrative_text(
                f"Past review notes repeatedly praise {join_terms_natural(positive_terms, limit=4)}.",
                max_chars=180,
            )
        )
    if negative_terms:
        blocks.append(
            narrative_text(
                f"Past complaints most often mention {join_terms_natural(negative_terms, limit=4)}.",
                max_chars=170,
            )
    )
    return choose_distinct_blocks(blocks, max_chars=max_chars, max_blocks=2, min_new_tokens=2)


_SUPPORT_TERM_BLOCKED_TOKENS = {
    "review",
    "reviews",
    "friend",
    "friends",
    "wife",
    "husband",
    "girls",
    "trip",
    "visit",
    "visits",
    "day",
    "today",
    "yesterday",
    "saturday",
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "came",
    "went",
    "walked",
    "ordered",
    "stayed",
    "looking",
    "minute",
    "write",
    "writing",
    "birthday",
    "flight",
    "commercials",
    "gps",
    "coordinates",
    "warehouse",
    "district",
    "quarter",
    "closed",
}

_SUPPORT_TERM_BLOCKED_PATTERNS = {
    "the other day",
    "first time trying",
    "on my birthday",
    "before catching my",
    "at first glance",
    "parking lot",
    "same parking lot",
    "next parking lot",
    "first off",
    "here goes my attempt",
    "favorite sushi restaurant",
    "what better food in a cold rainy day",
    "delicious food excellent service beautiful place",
    "well lit space and freret",
    "located in the",
    "bywater neighborhood",
    "right up there",
    "the place to grab",
    "service is consistently",
    "drinks are strong and tasty",
    "you truly can't go wrong",
}


def is_noisy_support_term(raw: Any) -> bool:
    text = clean_text(raw, max_chars=96).lower()
    if not text:
        return True
    if len(text.split()) > 4:
        return True
    if any(pattern in text for pattern in _STORY_CONTAMINATION_PATTERNS):
        return True
    if re.search(r"\b(i|i'm|i've|ive|my|we|we're|our|us|me)\b", text, flags=re.I):
        return True
    if re.search(r"\b(is|are|was|were|am|be|been|being)\b", text, flags=re.I):
        return True
    if any(ch in text for ch in {"!", "?", '"', ":"}):
        return True
    if any(pattern in text for pattern in _SUPPORT_TERM_BLOCKED_PATTERNS):
        return True
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9&+'-]{1,30}", text)
        if tok and len(tok) >= 2
    }
    return bool(tokens & _SUPPORT_TERM_BLOCKED_TOKENS)


def is_semantic_support_term(raw: Any, *, negative: bool = False) -> bool:
    text = normalize_term(raw, max_chars=80)
    if not text or is_noisy_support_term(text):
        return False
    if any(pattern in text for pattern in _STORY_CONTAMINATION_PATTERNS):
        return False
    if negative:
        if extract_negative_sentence_avoid_terms([text], limit=2):
            return True
        return any(
            token in text
            for token in {
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
            }
        )
    return has_source_semantic_hint(text)


def summarize_support_terms(
    raw: Any,
    *,
    limit: int = 4,
    broad: set[str] | None = None,
) -> list[str]:
    broad_terms = broad or (_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS)
    return specific_terms(
        dedupe_terms(
            [term for term in extract_terms_from_sentential_text(raw, limit=max(limit + 2, limit)) if is_semantic_support_term(term)],
            [term for term in summarize_sentence_terms(raw, limit=max(limit + 2, limit)) if is_semantic_support_term(term)],
            limit=max(limit * 2, limit),
        ),
        limit=limit,
        broad=broad_terms,
    )


def summarize_support_avoid_terms(raw: Any, *, limit: int = 4) -> list[str]:
    return dedupe_avoid_terms(
        extract_negative_sentence_avoid_terms(
            [usable for usable in (usable_source_support_text(sent, negative=True, max_chars=220) for sent in split_review_sentences(raw)) if usable],
            limit=limit,
        ),
        [
            normalize_avoid_term(term)
            for term in summarize_sentence_terms(raw, limit=max(limit + 2, limit))
            if is_experience_avoid_term(term) and is_semantic_support_term(term, negative=True)
        ],
        limit=limit,
    )


def summarize_support_terms_from_sentence_blob(
    raw: Any,
    *,
    limit: int = 4,
    broad: set[str] | None = None,
    negative: bool = False,
    merchant: bool = False,
) -> list[str]:
    broad_terms = broad or (_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS)
    extracted: list[str] = []
    for sentence in parse_json_text_list(raw):
        usable = (
            usable_merchant_support_text(sentence, negative=negative, max_chars=220)
            if merchant
            else usable_source_support_text(sentence, negative=negative, max_chars=220)
        )
        if not usable:
            continue
        extracted.extend(
            [
                term
                for term in extract_terms_from_sentential_text(usable, limit=max(limit + 2, limit))
                if is_semantic_support_term(term, negative=negative)
            ]
        )
        extracted.extend(
            [
                term
                for term in summarize_sentence_terms(usable, limit=max(limit + 2, limit))
                if is_semantic_support_term(term, negative=negative)
            ]
        )
    return specific_terms(dedupe_terms(extracted, limit=max(limit * 2, limit)), limit=limit, broad=broad_terms)


def summarize_support_avoid_terms_from_sentence_blob(
    raw: Any,
    *,
    limit: int = 4,
    merchant: bool = False,
) -> list[str]:
    avoid_terms: list[str] = []
    for sentence in parse_json_text_list(raw):
        usable = (
            usable_merchant_support_text(sentence, negative=True, max_chars=220)
            if merchant
            else usable_source_support_text(sentence, negative=True, max_chars=220)
        )
        if not usable:
            continue
        avoid_terms.extend(extract_negative_sentence_avoid_terms([usable], limit=limit))
        avoid_terms.extend(
            [
                normalize_avoid_term(term)
                for term in summarize_sentence_terms(usable, limit=max(limit + 2, limit))
                if is_experience_avoid_term(term) and is_semantic_support_term(term, negative=True)
            ]
        )
    return dedupe_avoid_terms(avoid_terms, limit=limit)


def _finite_float(raw: Any) -> float | None:
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def finite_or_zero(raw: Any) -> float:
    val = _finite_float(raw)
    return float(val) if val is not None else 0.0


def is_experience_avoid_term(term: str) -> bool:
    lower = normalize_term(term, max_chars=48)
    if not lower:
        return False
    keep_markers = ("wait", "service", "value", "noise", "parking", "clean", "slow", "crowd", "rude", "overprice")
    return any(marker in lower for marker in keep_markers)


def filter_positive_semantic_terms(*term_lists: list[str], limit: int = 8) -> list[str]:
    positive_terms: list[str] = []
    for term_list in term_lists:
        for term in term_list:
            norm = normalize_term(term, max_chars=48)
            if not norm or is_experience_avoid_term(norm):
                continue
            positive_terms.append(norm)
    return dedupe_terms(positive_terms, limit=limit)


def normalize_avoid_term(term: str) -> str:
    lower = normalize_term(term, max_chars=48)
    if not lower:
        return ""
    return _AVOID_TERM_NORMALIZATION.get(lower, lower)


def dedupe_avoid_terms(*term_lists: list[str], limit: int = 8) -> list[str]:
    normalized_terms: list[str] = []
    for term_list in term_lists:
        for term in term_list:
            text = normalize_avoid_term(term)
            if text:
                normalized_terms.append(text)
    out = dedupe_terms(normalized_terms, limit=max(limit * 2, limit))
    if "weak value for money" in out:
        out = [term for term in out if term != "good value for money"]
    if "long waits" in out:
        out = [term for term in out if normalize_term(term, max_chars=48) != "wait long"]
    return out[:limit]


def has_negative_cue(text: Any) -> bool:
    raw = clean_text(text, max_chars=240).lower()
    if not raw:
        return False
    if any(token in raw for token in _NEGATIVE_FALSE_POSITIVE_PATTERNS):
        return False
    if extract_negative_sentence_avoid_terms([raw], limit=2):
        return True
    strong_patterns = [
        r"\b(no reservations?(?: were)? available|could not sit|couldn't sit|not seated|turned us away|not welcoming)\b",
        r"\b(host was rude|ignored us|server was rude|staff was rude)\b",
        r"\b(wait(?:ed|ing)?(?: for)? (?:an |over |almost )?(?:hour|hours|\d+\s*(?:minutes?|mins?))|took forever)\b",
        r"\b(wrong order|forgot (?:our|my) order|missing order|never brought)\b",
        r"\b(small portions?|tiny portions?|dry food|not fresh)\b",
    ]
    if any(re.search(pattern, raw, flags=re.I) for pattern in strong_patterns):
        return True
    negative_support_tokens = {
        "wait",
        "rude",
        "overpriced",
        "expensive",
        "noise",
        "noisy",
        "dirty",
        "reservation",
        "cold",
        "stale",
        "soggy",
        "wrong order",
        "small portion",
        "parking",
    }
    neg_cue_count = sum(1 for cue in _NEGATIVE_SENTENCE_CUES if cue in raw)
    return neg_cue_count >= 2 and any(token in raw for token in negative_support_tokens)


def has_source_semantic_hint(text: Any) -> bool:
    raw = clean_text(text, max_chars=320).lower()
    if not raw:
        return False
    return any(term in raw for term in _SOURCE_SEMANTIC_HINTS)


def is_raw_story_text(text: Any) -> bool:
    raw = clean_text(text, max_chars=320).lower()
    if not raw:
        return False
    if any(token in raw for token in _STORY_CONTAMINATION_PATTERNS):
        return True
    if not (_FIRST_PERSON_STORY_RE.search(raw) or _FIRST_PERSON_STORY_RE_EXTRA.search(raw)):
        return False
    return any(token in raw for token in (_RAW_STORY_PATTERNS | _RAW_STORY_PATTERNS_EXTRA)) or not has_source_semantic_hint(raw)


def usable_source_support_text(text: Any, *, negative: bool = False, max_chars: int = 320) -> str:
    cleaned = clean_text(text, max_chars=max_chars)
    if not cleaned:
        return ""
    if is_raw_story_text(cleaned):
        return ""
    if any(token in cleaned.lower() for token in _STORY_CONTAMINATION_PATTERNS):
        return ""
    if negative:
        if not has_negative_cue(cleaned):
            return ""
        return cleaned
    if not has_source_semantic_hint(cleaned):
        return ""
    return cleaned


def usable_merchant_support_text(text: Any, *, negative: bool = False, max_chars: int = 240) -> str:
    cleaned = clean_text(text, max_chars=max_chars)
    if not cleaned:
        return ""
    if is_raw_story_text(cleaned):
        return ""
    if any(token in cleaned.lower() for token in _STORY_CONTAMINATION_PATTERNS):
        return ""
    if negative:
        if not has_negative_cue(cleaned):
            return ""
        return cleaned
    if not has_source_semantic_hint(cleaned):
        return ""
    return cleaned


def build_user_profile_texts(row: pd.Series) -> dict[str, Any]:
    profile_text_short = clean_text(row.get("profile_text_short"), max_chars=1200)
    profile_text_long = clean_text(row.get("profile_text_long"), max_chars=900)
    profile_pos_text = clean_text(row.get("profile_pos_text"), max_chars=320)
    profile_neg_text = clean_text(row.get("profile_neg_text"), max_chars=320)
    user_long_pref_text = clean_text(row.get("user_long_pref_text"), max_chars=400)
    user_recent_intent_text = clean_text(row.get("user_recent_intent_text"), max_chars=320)
    user_negative_avoid_text = clean_text(row.get("user_negative_avoid_text"), max_chars=220)
    user_context_text = clean_text(row.get("user_context_text"), max_chars=220)
    source_positive_text = clean_text(row.get("user_source_positive_text_v1"), max_chars=480)
    source_negative_text = clean_text(row.get("user_source_negative_text_v1"), max_chars=480)
    source_recent_text = clean_text(row.get("user_source_recent_text_v1"), max_chars=360)
    source_tip_text = clean_text(row.get("user_source_tip_text_v1"), max_chars=220)
    source_history_text = clean_text(row.get("user_source_history_anchor_text_v1"), max_chars=260)
    source_positive_blob = row.get("user_source_positive_sentences_v1")
    source_negative_blob = row.get("user_source_negative_sentences_v1")
    source_recent_blob = row.get("user_source_recent_sentences_v1")
    source_tip_blob = row.get("user_source_tip_sentences_v1")
    source_history_blob = row.get("user_history_anchor_texts_v1")
    source_positive_usable = usable_source_support_text(source_positive_text, max_chars=320)
    source_negative_usable = usable_source_support_text(source_negative_text, negative=True, max_chars=220)
    source_recent_usable = usable_source_support_text(source_recent_text, max_chars=220)
    source_tip_usable = usable_source_support_text(source_tip_text, max_chars=180)
    source_history_usable = usable_source_support_text(source_history_text, max_chars=220)
    source_positive_terms = dedupe_terms(
        summarize_support_terms_from_sentence_blob(source_positive_blob, limit=4),
        summarize_support_terms(source_positive_usable, limit=4),
        limit=4,
    )
    source_recent_terms = dedupe_terms(
        summarize_support_terms_from_sentence_blob(source_recent_blob, limit=4, broad=_BROAD_RECENT_TERMS),
        summarize_support_terms(source_recent_usable, limit=4, broad=_BROAD_RECENT_TERMS),
        limit=4,
    )
    source_tip_terms = dedupe_terms(
        summarize_support_terms_from_sentence_blob(source_tip_blob, limit=3),
        summarize_support_terms(source_tip_usable, limit=3),
        limit=3,
    )
    source_history_terms = dedupe_terms(
        summarize_support_terms_from_sentence_blob(
            source_history_blob,
            limit=4,
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        ),
        summarize_support_terms(
            source_history_usable,
            limit=4,
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        ),
        limit=4,
    )
    source_negative_terms = dedupe_avoid_terms(
        summarize_support_avoid_terms_from_sentence_blob(source_negative_blob, limit=4),
        summarize_support_avoid_terms(source_negative_usable, limit=4),
        limit=4,
    )

    pos_cuisine = ensure_terms(row.get("schema_pos_cuisine"), limit=3)
    pos_dish = ensure_terms(row.get("schema_pos_dish"), limit=3)
    pos_scene_schema = ensure_terms(row.get("schema_pos_scene"), limit=3)
    pos_service_schema = ensure_terms(row.get("schema_pos_service"), limit=3)
    pos_time_schema = ensure_terms(row.get("schema_pos_time"), limit=3)
    typed_pos_terms = filter_positive_semantic_terms(parse_typed_text_terms(profile_pos_text, limit=8), limit=6)
    typed_neg_terms = parse_typed_text_terms(profile_neg_text, limit=6)
    typed_pos_facets = parse_typed_text_facets(profile_pos_text, limit_per_facet=3)
    typed_neg_facets = parse_typed_text_facets(profile_neg_text, limit_per_facet=3)
    beverage_terms = dedupe_terms(typed_pos_facets.get("beverage", []), limit=3)
    taste_terms = dedupe_terms(typed_pos_facets.get("taste", []), limit=3)
    avoid_service_terms = dedupe_avoid_terms(typed_neg_facets.get("service", []), limit=3)
    avoid_value_terms = dedupe_avoid_terms(typed_neg_facets.get("value", []), limit=2)
    positive_anchor_sentences = [
        *pick_review_sentences(profile_text_short, "positive", limit=2),
        *pick_review_sentences(profile_text_long, "positive", limit=1),
    ]
    negative_anchor_sentences = [
        *pick_review_sentences(profile_text_short, "negative", limit=2),
        *pick_review_sentences(profile_text_long, "negative", limit=1),
        *pick_review_sentences(source_negative_usable, "negative", limit=2),
    ]
    positive_review_terms = filter_positive_semantic_terms(
        summarize_semantic_review_terms(profile_text_short, limit=4),
        summarize_semantic_review_terms(profile_text_long, limit=4),
        limit=5,
    )
    negative_review_terms = dedupe_avoid_terms(
        summarize_semantic_review_avoid_terms(profile_text_short, limit=4),
        summarize_semantic_review_avoid_terms(profile_text_long, limit=4),
        summarize_semantic_review_avoid_terms(source_negative_usable, limit=4),
        limit=4,
    )
    negative_sentence_avoid_terms = extract_negative_sentence_avoid_terms(negative_anchor_sentences, limit=4)
    complaint_terms = [
        _COMPLAINT_LABELS.get(term, term.replace("_", " "))
        for term in ensure_terms(row.get("schema_net_complaint"), limit=4)
        if term in _COMPLAINT_LABELS
    ]
    long_pref_natural = naturalize_user_long_pref_text(user_long_pref_text, max_chars=220)
    recent_natural = naturalize_user_recent_intent_text(user_recent_intent_text, max_chars=180)
    context_natural = naturalize_user_context_text(user_context_text, max_chars=120)
    typed_pref_evidence = build_profile_preference_evidence_text(
        profile_pos_text,
        "",
        max_chars=220,
    )
    review_evidence_summary = build_review_evidence_summary(
        positive_sentences=positive_anchor_sentences,
        negative_sentences=negative_anchor_sentences,
        complaint_terms=complaint_terms + negative_review_terms,
        positive_terms=positive_review_terms + pos_dish + taste_terms + beverage_terms + source_positive_terms + source_tip_terms,
        negative_terms=negative_review_terms + source_negative_terms,
        max_chars=240,
    )
    source_positive_summary = build_terms_sentence(
        "Additional review evidence also highlights",
        source_positive_terms + source_tip_terms,
        limit=4,
        max_chars=180,
    )
    source_negative_summary = ""
    if source_negative_terms:
        source_negative_summary = narrative_text(
            f"Additional complaints still point to {join_terms_natural(source_negative_terms, limit=4)}.",
            max_chars=170,
        )
    summary_only_user_evidence = choose_distinct_blocks(
        [review_evidence_summary, source_positive_summary, source_negative_summary],
        max_chars=260,
        max_blocks=2,
        min_new_tokens=1,
    )
    clean_user_evidence = build_clean_user_evidence_text(
        profile_text_short=profile_text_short,
        profile_text_long=profile_text_long,
        profile_pos_text=profile_pos_text,
        profile_neg_text=profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        max_chars=380,
    )
    clean_user_evidence = choose_distinct_blocks(
        [clean_user_evidence, review_evidence_summary, source_positive_summary, source_negative_summary],
        max_chars=320,
        max_blocks=2,
        min_new_tokens=2,
    )
    if clean_user_evidence:
        evidence_lower = f" {clean_user_evidence.lower()} "
        if " nan " in evidence_lower or re.search(r"\b(i|i'm|i’ve|we|we've|my|our|me|us)\b", clean_user_evidence, flags=re.I):
            clean_user_evidence = ""
    if not clean_user_evidence and summary_only_user_evidence:
        clean_user_evidence = summary_only_user_evidence

    stable_core_terms = filter_positive_semantic_terms(typed_pos_terms, pos_cuisine, pos_dish, limit=5)
    stable_scene_terms = dedupe_terms(
        top_numeric_labels(row, _INTENT_TIME_PREFS, threshold=0.42, limit=2),
        top_numeric_labels(row, _INTENT_SCENE_PREFS, threshold=0.42, limit=2),
        pos_scene_schema,
        pos_time_schema,
        limit=4,
    )
    stable_service_terms = dedupe_terms(
        top_numeric_labels(row, _INTENT_PROPERTY_PREFS, threshold=0.45, limit=3),
        pos_service_schema,
        limit=4,
    )

    stable_preference_terms = filter_positive_semantic_terms(stable_core_terms, positive_review_terms, limit=5)
    stable_specific_terms = specific_terms(
        filter_positive_semantic_terms(pos_dish, positive_review_terms, typed_pos_terms, pos_cuisine, limit=12),
        limit=5,
        broad=_BROAD_CORE_TERMS,
    )
    stable_specific_scene_terms = [
        term
        for term in stable_scene_terms
        if term not in _GENERIC_SCENE_TERMS and term not in _BROAD_SCENE_REASON_TERMS
    ]
    stable_bits: list[str] = []
    specific_cuisine_terms = dedupe_terms(pos_cuisine, [row.get("long_term_top_cuisine")], limit=3)
    specific_dish_sentence = build_dish_preference_sentence(
        pos_dish + positive_review_terms,
        specific_cuisine_terms,
        max_chars=220,
    )
    if specific_dish_sentence:
        stable_bits.append(specific_dish_sentence)
    elif stable_specific_terms:
        stable_bits.append(
            narrative_text(
                f"Repeated visits most clearly point to {join_terms_natural(stable_specific_terms, limit=4)}.",
                max_chars=220,
            )
        )
    specific_long_pref_terms = specific_terms(
        dedupe_terms(
            extract_labeled_items_from_text(user_long_pref_text, ["Long-term cuisines"], limit=4),
            extract_labeled_items_from_text(user_long_pref_text, ["Typical meal preferences"], limit=3),
            extract_labeled_items_from_text(user_long_pref_text, ["Typical dining scenes"], limit=3),
            extract_labeled_items_from_text(user_long_pref_text, ["Useful service properties"], limit=3),
            limit=6,
        ),
        limit=4,
        broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
    )
    if not stable_bits:
        stable_bits.append(
            compose_user_preference_sentence(
                specific_terms(stable_preference_terms, limit=4, broad=_BROAD_CORE_TERMS),
                stable_specific_scene_terms,
                [term for term in stable_service_terms if term not in _BROAD_SERVICE_REASON_TERMS],
                max_chars=240,
            )
        )
    stable_specific_service_terms = [term for term in stable_service_terms if term not in _BROAD_SERVICE_REASON_TERMS]
    if (
        stable_specific_service_terms
        and has_meaningful_addition(
            " ".join(stable_bits),
            join_terms_natural(stable_specific_service_terms, limit=3),
        )
    ):
        stable_bits.append(
            f"They also tend to favor places known for {join_terms_natural(stable_specific_service_terms, limit=3)}."
        )
    if beverage_terms and has_meaningful_addition(" ".join(stable_bits), join_terms_natural(beverage_terms, limit=3)):
        stable_bits.append(
            narrative_text(
                f"Positive review evidence also points to places with {join_terms_natural(beverage_terms, limit=3)}.",
                max_chars=180,
            )
        )
    if taste_terms and has_meaningful_addition(" ".join(stable_bits), join_terms_natural(taste_terms, limit=3)):
        stable_bits.append(
            narrative_text(
                f"The user also responds well to flavors that lean {join_terms_natural(taste_terms, limit=3)}.",
                max_chars=180,
            )
        )
    stable_preferences_text = choose_distinct_blocks(stable_bits, max_chars=440, max_blocks=3)

    recent_top_cuisine = normalize_term(row.get("recent_top_cuisine"), max_chars=48)
    latest_top_cuisine = normalize_term(row.get("latest_top_cuisine"), max_chars=48)
    recent_raw_facets = extract_recent_intent_facets(user_recent_intent_text)
    recent_scene_terms = dedupe_terms(
        top_numeric_labels(row, [
            ("latest_breakfast_pref", "breakfast"),
            ("latest_lunch_pref", "lunch"),
            ("latest_dinner_pref", "dinner"),
            ("latest_late_night_pref", "late-night meals"),
            ("latest_family_scene_pref", "family-friendly settings"),
            ("latest_group_scene_pref", "group dining"),
            ("latest_date_scene_pref", "date-night meals"),
            ("latest_nightlife_scene_pref", "nightlife"),
            ("latest_fast_casual_pref", "quick casual meals"),
            ("latest_sitdown_pref", "sit-down meals"),
        ], threshold=0.40, limit=3),
        recent_raw_facets.get("meal", []),
        recent_raw_facets.get("scene", []),
        limit=4,
    )
    recent_property_terms = dedupe_terms(
        top_numeric_labels(row, _INTENT_PROPERTY_PREFS, threshold=0.50, limit=2),
        recent_raw_facets.get("property", []),
        limit=3,
    )
    recent_terms = dedupe_terms([recent_top_cuisine, latest_top_cuisine], recent_raw_facets.get("cuisine", []), limit=4)
    recent_specific_terms = specific_terms(
        dedupe_terms(recent_terms, limit=6),
        limit=4,
        broad=_BROAD_RECENT_TERMS,
    )
    recent_specific_scene_terms = [
        term for term in dedupe_terms(recent_scene_terms, limit=4)
        if term not in _GENERIC_SCENE_TERMS and term not in _BROAD_SCENE_REASON_TERMS
    ]
    recent_context_scene_terms = [
        term
        for term in dedupe_terms(recent_scene_terms, limit=4)
        if term not in _GENERIC_SCENE_TERMS
    ]
    recent_context_property_terms = [
        term
        for term in dedupe_terms(recent_property_terms, limit=4)
        if (
            normalize_term(term, max_chars=48) in _PRACTICAL_SIGNAL_TERMS
            or normalize_term(term, max_chars=48) in _CONTEXT_FIT_KEEP_TERMS
            or normalize_term(term, max_chars=48) not in _BROAD_RECENT_PROPERTY_TERMS
        )
    ]
    recent_bits: list[str] = []
    if recent_specific_terms:
        recent_bits.append(
            narrative_text(
                f"Recent visits are leaning toward {join_terms_natural(recent_specific_terms, limit=3)}.",
                max_chars=200,
            )
        )
    source_recent_summary = build_terms_sentence(
        "Recent review evidence also leans toward",
        source_recent_terms,
        limit=3,
        max_chars=180,
    )
    if source_recent_summary and has_meaningful_addition(" ".join(recent_bits), source_recent_summary):
        recent_bits.append(source_recent_summary)
    if not recent_bits:
        fallback_recent = compose_recent_intent_sentence(recent_specific_terms, recent_specific_scene_terms, max_chars=220)
        if fallback_recent:
            recent_bits.append(fallback_recent)
    recent_context_sentence = compose_recent_context_sentence(
        recent_context_scene_terms,
        recent_context_property_terms,
        max_chars=220,
    )
    if recent_context_sentence and has_meaningful_addition(" ".join(recent_bits), recent_context_sentence):
        recent_bits.append(recent_context_sentence)
    if recent_specific_scene_terms and has_meaningful_addition(" ".join(recent_bits), join_terms_natural(recent_specific_scene_terms, limit=3)):
        recent_bits.append(
            narrative_text(
                f"The latest visits most often happen around {join_terms_natural(recent_specific_scene_terms, limit=3)}.",
                max_chars=160,
            )
        )
    recent_specific_property_terms = [
        term for term in dedupe_terms(recent_property_terms, limit=3)
        if normalize_term(term, max_chars=48) not in _BROAD_RECENT_PROPERTY_TERMS
    ]
    if recent_specific_property_terms and has_meaningful_addition(" ".join(recent_bits), join_terms_natural(recent_specific_property_terms, limit=3)):
        recent_bits.append(
            narrative_text(
                f"Recent outings also lean toward places with {join_terms_natural(recent_specific_property_terms, limit=3)}.",
                max_chars=160,
            )
        )
    recent_intent_text_v2 = choose_distinct_blocks(recent_bits, max_chars=360, max_blocks=3)
    if recent_intent_text_v2 and any(token in recent_intent_text_v2.lower() for token in _STORY_CONTAMINATION_PATTERNS):
        recent_intent_text_v2 = ""

    typed_neg_experience = [
        normalize_avoid_term(term)
        for term in typed_neg_terms
        if term not in stable_core_terms and is_experience_avoid_term(term)
    ]
    source_negative_fallback_terms = dedupe_avoid_terms(
        source_negative_terms,
        negative_sentence_avoid_terms,
        limit=4,
    )
    avoid_terms = dedupe_avoid_terms(
        complaint_terms,
        typed_neg_experience,
        avoid_service_terms,
        avoid_value_terms,
        negative_sentence_avoid_terms,
        negative_review_terms,
        source_negative_fallback_terms,
        limit=5,
    )
    avoid_terms = [term for term in avoid_terms if term not in _BAD_AVOID_TERMS and term != "good value for money"]
    avoidance_text_v2 = compose_avoidance_sentence(avoid_terms, max_chars=260)
    if not avoidance_text_v2 and source_negative_fallback_terms:
        avoidance_text_v2 = compose_avoidance_sentence(source_negative_fallback_terms, max_chars=260)
    negative_avoid_natural = naturalize_user_negative_avoid_text(user_negative_avoid_text, max_chars=150)
    if negative_avoid_natural and has_negative_cue(negative_avoid_natural) and has_meaningful_addition(avoidance_text_v2, negative_avoid_natural):
        avoidance_text_v2 = choose_distinct_blocks(
            [avoidance_text_v2, negative_avoid_natural],
            max_chars=260,
            max_blocks=2,
        )
    source_negative_summary = ""
    if source_negative_fallback_terms:
        source_negative_summary = narrative_text(
            f"Negative source evidence repeatedly points to {join_terms_natural(source_negative_fallback_terms, limit=4)}.",
            max_chars=160,
        )
    source_negative_clause = ""
    if source_negative_usable:
        source_negative_clause = select_complete_narrative_block(source_negative_usable, max_chars=150, min_words=4)
    if source_negative_summary and has_meaningful_addition(avoidance_text_v2, source_negative_summary, min_new_tokens=1):
        avoidance_text_v2 = choose_distinct_blocks(
            [avoidance_text_v2, source_negative_summary],
            max_chars=260,
            max_blocks=2,
            min_new_tokens=1,
        )
    if (
        not avoidance_text_v2
        and source_negative_clause
        and has_negative_cue(source_negative_clause)
    ):
        avoidance_text_v2 = choose_distinct_blocks(
            [
                avoidance_text_v2,
                narrative_text(
                    f"Negative source evidence repeatedly points to {source_negative_clause}.",
                    max_chars=180,
                ),
            ],
            max_chars=260,
            max_blocks=2,
            min_new_tokens=1,
        )
    elif (
        source_negative_clause
        and has_negative_cue(source_negative_clause)
        and has_meaningful_addition(avoidance_text_v2, source_negative_clause, min_new_tokens=1)
    ):
        avoidance_text_v2 = choose_distinct_blocks(
            [
                avoidance_text_v2,
                narrative_text(
                    f"Negative source evidence also flags {source_negative_clause}.",
                    max_chars=160,
                ),
            ],
            max_chars=260,
            max_blocks=2,
            min_new_tokens=1,
        )
    if avoidance_text_v2 and any(token in avoidance_text_v2.lower() for token in _STORY_CONTAMINATION_PATTERNS):
        avoidance_text_v2 = compose_avoidance_sentence(avoid_terms, max_chars=260)

    history_hint = build_history_anchor_text(
        stable_text=stable_preferences_text,
        recent_text=recent_intent_text_v2,
        avoid_text=avoidance_text_v2,
        positive_terms=positive_review_terms + source_history_terms,
        positive_sentences=positive_anchor_sentences,
        profile_text_long=profile_text_long,
        profile_pos_text=profile_pos_text,
        fallback_terms=stable_specific_terms + specific_long_pref_terms + positive_review_terms + source_history_terms,
    )
    source_history_summary = build_terms_sentence(
        "Older positive visits also keep pointing back to",
        source_history_terms,
        limit=4,
        max_chars=180,
    )
    if source_history_summary and has_meaningful_addition(history_hint, source_history_summary, min_new_tokens=1):
        history_hint = choose_distinct_blocks([history_hint, source_history_summary], max_chars=220, max_blocks=2, min_new_tokens=1)
    if history_hint and not _should_keep_supporting_text(
        [stable_preferences_text, recent_intent_text_v2, avoidance_text_v2],
        history_hint,
        min_new_tokens=1,
    ):
        history_hint = ""
    if history_hint and any(token in history_hint.lower() for token in _STORY_CONTAMINATION_PATTERNS):
        history_hint = ""
    history_anchor_source = "evidence" if history_hint else ""
    if clean_user_evidence and not _should_keep_supporting_text(
        [stable_preferences_text, recent_intent_text_v2, avoidance_text_v2, history_hint],
        clean_user_evidence,
        min_new_tokens=1,
    ):
        clean_user_evidence = ""
    if (
        not clean_user_evidence
        and summary_only_user_evidence
        and _should_keep_supporting_text(
            [stable_preferences_text, recent_intent_text_v2, avoidance_text_v2, history_hint],
            summary_only_user_evidence,
            min_new_tokens=1,
        )
    ):
        clean_user_evidence = summary_only_user_evidence
    if clean_user_evidence and any(token in clean_user_evidence.lower() for token in _STORY_CONTAMINATION_PATTERNS):
        clean_user_evidence = ""
    sections = [stable_preferences_text, recent_intent_text_v2, avoidance_text_v2, history_hint, clean_user_evidence]
    richness = sum(1 for text in sections if clean_text(text))
    has_stable = bool(clean_text(stable_preferences_text))
    has_recent = bool(clean_text(recent_intent_text_v2))
    has_avoid = bool(clean_text(avoidance_text_v2))
    has_history_evidence = bool(clean_text(history_hint))
    has_clean_evidence = bool(clean_text(clean_user_evidence))
    has_specific_focus = bool(specific_dish_sentence) or bool(stable_specific_terms) or bool(positive_anchor_sentences) or bool(specific_long_pref_terms)
    has_specific_recent = bool(recent_intent_text_v2) and (
        bool(recent_specific_terms)
        or bool(recent_specific_scene_terms)
        or bool(recent_specific_property_terms)
        or bool(recent_context_sentence)
    )
    has_strong_clean_evidence = has_clean_evidence and bool(
        review_evidence_summary or typed_pref_evidence or source_positive_usable or source_negative_usable or source_tip_usable or has_negative_cue(clean_user_evidence)
    )
    quality_signal_count = (
        int(has_specific_focus)
        + int(has_specific_recent)
        + int(has_avoid)
        + int(has_history_evidence)
        + int(has_strong_clean_evidence)
    )
    tier = "LOW_SIGNAL"
    if (
        has_specific_focus
        and has_specific_recent
        and has_recent
        and (has_avoid or has_history_evidence)
        and has_strong_clean_evidence
        and quality_signal_count >= 4
    ):
        tier = "FULL"
    elif (
        has_specific_focus
        and has_strong_clean_evidence
        and ((has_recent and has_specific_recent) or has_avoid)
        and (has_avoid or has_history_evidence)
        and quality_signal_count >= 3
    ):
        tier = "PARTIAL"

    user_semantic_profile_text_v2 = choose_distinct_blocks(
        [text for text in sections if text],
        max_chars=1320,
        max_blocks=5,
        min_new_tokens=1,
    )
    user_fact_records = [
        build_semantic_fact(
            scope="stable",
            polarity="positive",
            text=stable_preferences_text,
            facet_type="user_preference",
            evidence_source="profile",
            terms=stable_specific_terms or stable_preference_terms or stable_core_terms,
            specificity=int(bool(has_specific_focus)),
        ),
        build_semantic_fact(
            scope="recent",
            polarity="positive",
            text=recent_intent_text_v2,
            facet_type="recent_intent",
            evidence_source="recent_activity",
            terms=recent_specific_terms + recent_context_scene_terms + recent_context_property_terms,
            specificity=int(bool(has_specific_recent)),
        ),
        build_semantic_fact(
            scope="avoid",
            polarity="negative",
            text=avoidance_text_v2,
            facet_type="user_avoidance",
            evidence_source="negative_feedback",
            terms=avoid_terms,
            specificity=int(bool(has_avoid)),
        ),
        build_semantic_fact(
            scope="history",
            polarity="positive",
            text=history_hint,
            facet_type="history_anchor",
            evidence_source=history_anchor_source or "history",
            terms=positive_review_terms + stable_specific_terms,
            specificity=int(bool(has_history_evidence)),
        ),
        build_semantic_fact(
            scope="evidence",
            polarity="mixed",
            text=clean_user_evidence,
            facet_type="review_evidence",
            evidence_source="reviews",
            terms=positive_review_terms + avoid_terms,
            specificity=int(bool(has_strong_clean_evidence)),
            max_chars=420,
        ),
    ]
    return {
        "stable_preferences_text": stable_preferences_text,
        "recent_intent_text_v2": recent_intent_text_v2,
        "avoidance_text_v2": avoidance_text_v2,
        "history_anchor_hint_text": history_hint,
        "history_anchor_source_v2": history_anchor_source,
        "user_evidence_text_v2": clean_user_evidence,
        "user_semantic_profile_text_v2": user_semantic_profile_text_v2,
        "user_profile_richness_v2": int(richness),
        "user_profile_richness_tier_v2": tier,
        "user_quality_signal_count_v2": int(quality_signal_count),
        "user_has_avoid_signal_v2": int(has_avoid),
        "user_has_clean_evidence_v2": int(has_clean_evidence),
        "user_has_specific_focus_v2": int(has_specific_focus),
        "user_has_specific_recent_v2": int(has_specific_recent),
        "user_has_recent_text_v2": int(has_recent),
        "user_has_strong_clean_evidence_v2": int(has_strong_clean_evidence),
        "user_has_history_evidence_v2": int(has_history_evidence),
        "user_semantic_facts_v1": semantic_facts_json(*user_fact_records),
        "user_visible_fact_count_v1": int(count_semantic_facts(user_fact_records)),
        "user_focus_fact_count_v1": int(count_semantic_facts(user_fact_records, scope="stable")),
        "user_recent_fact_count_v1": int(count_semantic_facts(user_fact_records, scope="recent")),
        "user_avoid_fact_count_v1": int(count_semantic_facts(user_fact_records, scope="avoid")),
        "user_history_fact_count_v1": int(count_semantic_facts(user_fact_records, scope="history")),
        "user_evidence_fact_count_v1": int(count_semantic_facts(user_fact_records, scope="evidence")),
        "user_preference_core_terms_v2": stringify_terms(stable_core_terms, limit=8),
        "user_dish_terms_v2": stringify_terms(pos_dish + positive_review_terms, limit=8),
        "user_beverage_terms_v2": stringify_terms(beverage_terms, limit=6),
        "user_taste_terms_v2": stringify_terms(taste_terms, limit=6),
        "user_scene_terms_v2": stringify_terms(stable_scene_terms, limit=6),
        "user_service_terms_v2": stringify_terms(stable_service_terms, limit=6),
        "user_preference_terms_v2": stringify_terms(stable_core_terms + stable_scene_terms + stable_service_terms, limit=10),
        "user_recent_terms_v2": stringify_terms(
            recent_specific_terms + recent_context_scene_terms + recent_context_property_terms,
            limit=8,
        ),
        "user_recent_semantic_terms_v2": stringify_terms(recent_specific_terms, limit=6),
        "user_recent_context_terms_v2": stringify_terms(
            recent_context_scene_terms + recent_context_property_terms,
            limit=6,
        ),
        "user_source_positive_terms_v2": stringify_terms(source_positive_terms, limit=6),
        "user_source_recent_terms_v2": stringify_terms(source_recent_terms, limit=6),
        "user_source_tip_terms_v2": stringify_terms(source_tip_terms, limit=4),
        "user_source_history_terms_v2": stringify_terms(source_history_terms, limit=6),
        "user_avoid_terms_v2": stringify_terms(avoid_terms, limit=8),
    }


def _top_scene_labels(row: pd.Series) -> list[str]:
    candidates = [
        ("family friendly dining", float(row.get("family_scene_fit", 0.0) or 0.0)),
        ("group dining", float(row.get("group_scene_fit", 0.0) or 0.0)),
        ("date night", float(row.get("date_scene_fit", 0.0) or 0.0)),
        ("nightlife", float(row.get("nightlife_scene_fit", 0.0) or 0.0)),
        ("fast casual meals", float(row.get("fast_casual_fit", 0.0) or 0.0)),
        ("sit-down meals", float(row.get("sitdown_fit", 0.0) or 0.0)),
        ("breakfast", float(row.get("meal_breakfast_fit", 0.0) or 0.0)),
        ("lunch", float(row.get("meal_lunch_fit", 0.0) or 0.0)),
        ("dinner", float(row.get("meal_dinner_fit", 0.0) or 0.0)),
        ("late night visits", float(row.get("late_night_fit", 0.0) or 0.0)),
    ]
    candidates = [(label, score) for label, score in candidates if score >= 0.33]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [label for label, _ in candidates[:3]]


def prioritized_merchant_menu_terms(
    primary: str,
    secondary: str,
    *term_lists: list[str],
    limit: int = 8,
) -> list[str]:
    priority_terms: list[str] = []
    for raw in [primary, secondary]:
        term = normalize_term(raw, max_chars=48)
        if not term or term in _NON_FOOD_PRIMARY_CATEGORIES or term in _GENERIC_TERMS or term in _LOW_SIGNAL_TOKENS:
            continue
        priority_terms.append(term)
    return dedupe_terms(priority_terms, *term_lists, limit=limit)


def build_merchant_profile_texts(row: pd.Series) -> dict[str, Any]:
    name = clean_text(row.get("name"), max_chars=60)
    city = clean_text(row.get("city"), max_chars=48)
    primary = sanitize_primary_category(row.get("merchant_primary_cuisine"))
    secondary = sanitize_primary_category(row.get("merchant_secondary_cuisine"))
    primary_display = readable_term(primary, max_chars=48)
    secondary_display = readable_term(secondary, max_chars=48)
    if not primary:
        primary = secondary
        primary_display = secondary_display
        secondary = ""
        secondary_display = ""
    core_text = best_text(
        row.get("merchant_dish_text_v2"),
        row.get("merchant_core_text_v3"),
        row.get("merchant_semantic_text_v3"),
        max_chars=320,
    )
    if re.search(
        r"\b(signature dishes or cuisines|dining scenes|strong time cues|useful properties|frequent strengths|frequent complaints|primary category)\b\s*:",
        core_text,
        flags=re.I,
    ):
        core_text = ""
    pos_text = usable_merchant_support_text(
        best_text(row.get("merchant_service_text_v2"), row.get("merchant_pos_text_v3"), max_chars=260),
        max_chars=260,
    )
    neg_text = usable_merchant_support_text(
        best_text(row.get("merchant_complaint_text_v2"), row.get("merchant_neg_text_v3"), max_chars=240),
        negative=True,
        max_chars=240,
    )
    context_text = best_text(
        row.get("merchant_scene_text_v2"),
        row.get("merchant_time_text_v2"),
        row.get("merchant_property_text_v2"),
        row.get("merchant_context_text_v3"),
        max_chars=240,
    )
    source_strength_text = usable_merchant_support_text(row.get("merchant_source_strength_text_v1"), max_chars=260)
    source_risk_text = usable_merchant_support_text(row.get("merchant_source_risk_text_v1"), negative=True, max_chars=220)
    source_tip_text = usable_merchant_support_text(row.get("merchant_source_tip_text_v1"), max_chars=180)
    source_checkin_text = clean_text(row.get("merchant_source_checkin_hint_text_v1"), max_chars=180)
    source_strength_terms = dedupe_terms(
        summarize_support_terms_from_sentence_blob(
            row.get("merchant_source_strength_sentences_v1"),
            limit=4,
            merchant=True,
        ),
        summarize_support_terms(source_strength_text, limit=4),
        limit=4,
    )
    source_tip_terms = dedupe_terms(
        summarize_support_terms_from_sentence_blob(
            row.get("merchant_source_tip_sentences_v1"),
            limit=3,
            merchant=True,
        ),
        summarize_support_terms(source_tip_text, limit=3),
        limit=3,
    )
    source_risk_terms = dedupe_avoid_terms(
        summarize_support_avoid_terms_from_sentence_blob(
            row.get("merchant_source_risk_sentences_v1"),
            limit=4,
            merchant=True,
        ),
        summarize_support_avoid_terms(source_risk_text, limit=4),
        limit=4,
    )
    dish_terms = dedupe_terms(
        parse_terms_blob(row.get("top_dishes_v2")),
        extract_terms_from_sentential_text(row.get("merchant_dish_text_v2"), limit=6),
        summarize_semantic_review_terms(source_strength_text, limit=5, broad=set()),
        limit=6,
    )
    menu_terms = prioritized_merchant_menu_terms(
        primary,
        secondary,
        dish_terms,
        extract_terms_from_sentential_text(core_text, limit=6),
        limit=8,
    )
    scene_terms = dedupe_terms(
        parse_terms_blob(row.get("top_scenes_v2")),
        extract_terms_from_sentential_text(row.get("merchant_scene_text_v2"), limit=6),
        limit=5,
    )
    time_terms = dedupe_terms(
        parse_terms_blob(row.get("top_times_v2")),
        extract_terms_from_sentential_text(row.get("merchant_time_text_v2"), limit=5),
        limit=4,
    )
    property_terms = dedupe_terms(
        parse_terms_blob(row.get("top_properties_v2")),
        extract_terms_from_sentential_text(row.get("merchant_property_text_v2"), limit=5),
        limit=5,
    )
    strength_terms = dedupe_terms(
        parse_terms_blob(row.get("top_services_v2")),
        summarize_semantic_review_terms(pos_text, limit=6, broad=set()),
        summarize_semantic_review_terms(source_strength_text, limit=5, broad=set()),
        source_strength_terms,
        source_tip_terms,
        limit=4,
    )
    risk_terms = dedupe_terms(
        parse_terms_blob(row.get("top_complaints_v2")),
        summarize_semantic_review_avoid_terms(neg_text, limit=4),
        summarize_semantic_review_avoid_terms(source_risk_text, limit=4),
        source_risk_terms,
        limit=4,
    )
    specific_scene_terms = filter_terms_display(scene_terms, limit=4, broad=_BROAD_MERCHANT_SCENE_TERMS)
    specific_time_terms = filter_terms_display(time_terms, limit=3, broad=_BROAD_MERCHANT_TIME_TERMS)
    specific_property_terms = filter_terms_display(property_terms, limit=3, broad=_BROAD_MERCHANT_PROPERTY_TERMS)
    specific_strength_terms = filter_terms_display(strength_terms, limit=4, broad=_BROAD_SERVICE_REASON_TERMS)
    specific_risk_terms = filter_terms_display(risk_terms, limit=4)

    core_bits: list[str] = []
    if name and primary_display and city:
        core_bits.append(f"{name} is in {city} and mainly focuses on {primary_display}.")
    elif name and primary_display:
        core_bits.append(f"{name} mainly centers on {primary_display}.")
    elif name and city:
        core_bits.append(f"{name} is in {city}.")
    elif name:
        core_bits.append(f"{name} is the business under consideration.")
    if menu_terms:
        core_bits.append(f"Reviews most often mention {join_terms_natural(menu_terms, limit=4)}.")
    if secondary_display and secondary.lower() not in primary.lower():
        core_bits.append(f"It also overlaps with {secondary_display}.")
    elif core_text and core_text.lower() not in " ".join(core_bits).lower():
        core_bits.append(core_text)
    core_offering_text = choose_distinct_blocks(core_bits, max_chars=420, max_blocks=3)

    scene_labels = dedupe_terms(
        filter_terms_display(_top_scene_labels(row), limit=3, broad=_BROAD_MERCHANT_SCENE_TERMS),
        specific_scene_terms,
        limit=4,
    )
    scene_bits: list[str] = []
    if scene_labels:
        scene_bits.append(f"It is most naturally suited to {join_terms_natural(scene_labels, limit=3)}.")
    if specific_time_terms:
        scene_bits.append(f"Visits lean more toward {join_terms_natural(specific_time_terms, limit=3)}.")
    if specific_property_terms:
        scene_bits.append(f"Useful visit details include {join_terms_natural(specific_property_terms, limit=3)}.")
    if source_checkin_text and has_meaningful_addition(" ".join(scene_bits), source_checkin_text):
        scene_bits.append(select_complete_narrative_block(source_checkin_text, max_chars=120, min_words=4))
    scene_fit_text = choose_distinct_blocks(scene_bits, max_chars=260, max_blocks=3)

    strengths_text = build_terms_sentence(
        "Reviews repeatedly praise",
        specific_strength_terms,
        limit=4,
        max_chars=220,
    )
    source_strength_summary = build_terms_sentence(
        "Additional source evidence also highlights",
        source_strength_terms + source_tip_terms,
        limit=4,
        max_chars=180,
    )
    if source_strength_summary and has_meaningful_addition(strengths_text, source_strength_summary):
        strengths_text = choose_distinct_blocks([strengths_text, source_strength_summary], max_chars=220, max_blocks=2, min_new_tokens=1)
    risk_point_terms = dedupe_avoid_terms(
        specific_risk_terms,
        summarize_support_avoid_terms(neg_text, limit=4),
        source_risk_terms,
        limit=4,
    )
    risk_points_text = build_terms_sentence(
        "Potential friction points include",
        risk_point_terms,
        limit=4,
        max_chars=200,
    )
    source_risk_summary = ""
    if source_risk_terms:
        source_risk_summary = narrative_text(
            f"Additional review risk still centers on {join_terms_natural(source_risk_terms, limit=4)}.",
            max_chars=170,
        )
    if source_risk_summary and has_meaningful_addition(risk_points_text, source_risk_summary):
        risk_points_text = choose_distinct_blocks([risk_points_text, source_risk_summary], max_chars=200, max_blocks=2, min_new_tokens=1)
    sections = [core_offering_text, scene_fit_text, strengths_text, risk_points_text]
    richness = sum(1 for text in sections if clean_text(text))
    merchant_semantic_profile_text_v2 = choose_distinct_blocks(
        [text for text in sections if text],
        max_chars=760,
        max_blocks=4,
        min_new_tokens=3,
    )
    merchant_fact_records = [
        build_semantic_fact(
            scope="core",
            polarity="positive",
            text=core_offering_text,
            facet_type="business_profile",
            evidence_source="merchant_card",
            terms=menu_terms,
            specificity=int(bool(menu_terms or primary_display)),
            max_chars=420,
        ),
        build_semantic_fact(
            scope="scene",
            polarity="positive",
            text=scene_fit_text,
            facet_type="business_scene",
            evidence_source="merchant_context",
            terms=scene_labels + specific_time_terms + specific_property_terms,
            specificity=int(bool(scene_labels or specific_time_terms or specific_property_terms)),
            max_chars=260,
        ),
        build_semantic_fact(
            scope="strength",
            polarity="positive",
            text=strengths_text,
            facet_type="business_strength",
            evidence_source="reviews",
            terms=specific_strength_terms,
            specificity=int(bool(specific_strength_terms)),
            max_chars=220,
        ),
        build_semantic_fact(
            scope="risk",
            polarity="negative",
            text=risk_points_text,
            facet_type="business_risk",
            evidence_source="reviews",
            terms=specific_risk_terms,
            specificity=int(bool(specific_risk_terms)),
            max_chars=200,
        ),
    ]
    return {
        "core_offering_text": core_offering_text,
        "scene_fit_text": scene_fit_text,
        "strengths_text": strengths_text,
        "risk_points_text": risk_points_text,
        "merchant_semantic_profile_text_v2": merchant_semantic_profile_text_v2,
        "merchant_profile_richness_v2": int(richness),
        "merchant_semantic_facts_v1": semantic_facts_json(*merchant_fact_records),
        "merchant_visible_fact_count_v1": int(count_semantic_facts(merchant_fact_records)),
        "merchant_core_fact_count_v1": int(count_semantic_facts(merchant_fact_records, scope="core")),
        "merchant_scene_fact_count_v1": int(count_semantic_facts(merchant_fact_records, scope="scene")),
        "merchant_strength_fact_count_v1": int(count_semantic_facts(merchant_fact_records, scope="strength")),
        "merchant_risk_fact_count_v1": int(count_semantic_facts(merchant_fact_records, scope="risk")),
        "merchant_core_terms_v2": stringify_terms(menu_terms + scene_terms + time_terms + property_terms + extract_terms_from_sentential_text(core_text, limit=6), limit=10),
        "merchant_dish_terms_v2": stringify_terms(menu_terms, limit=8),
        "merchant_scene_terms_v2": stringify_terms(scene_labels + specific_scene_terms + specific_time_terms, limit=8),
        "merchant_time_terms_v2": stringify_terms(specific_time_terms, limit=6),
        "merchant_property_terms_v2": stringify_terms(specific_property_terms, limit=6),
        "merchant_strength_terms_v2": stringify_terms(specific_strength_terms, limit=8),
        "merchant_risk_terms_v2": stringify_terms(specific_risk_terms, limit=8),
    }


def _reason_candidate(
    *,
    score: float,
    text: str,
    scope: str,
    polarity: str,
    terms: list[str] | None = None,
    evidence_source: str = "",
    specificity: int = 1,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "text": str(text or ""),
        "scope": str(scope or "").strip().lower(),
        "polarity": str(polarity or "").strip().lower(),
        "terms": dedupe_terms(list(terms or []), limit=6),
        "evidence_source": str(evidence_source or "").strip().lower(),
        "specificity": int(max(0, specificity)),
    }


def _collect_fit_reason_facts(row: pd.Series) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    user_core_terms = set(parse_terms_blob(row.get("user_preference_core_terms_v2")))
    user_dish_terms = set(parse_terms_blob(row.get("user_dish_terms_v2")))
    user_beverage_terms = set(parse_terms_blob(row.get("user_beverage_terms_v2")))
    user_scene_terms = set(parse_terms_blob(row.get("user_scene_terms_v2")))
    user_service_terms = set(parse_terms_blob(row.get("user_service_terms_v2")))
    user_recent_semantic_terms = set(parse_terms_blob(row.get("user_recent_semantic_terms_v2")))
    user_recent_context_terms = set(parse_terms_blob(row.get("user_recent_context_terms_v2")))
    user_source_positive_terms = set(parse_terms_blob(row.get("user_source_positive_terms_v2")))
    user_source_recent_terms = set(parse_terms_blob(row.get("user_source_recent_terms_v2")))
    user_source_tip_terms = set(parse_terms_blob(row.get("user_source_tip_terms_v2")))
    user_source_history_terms = set(parse_terms_blob(row.get("user_source_history_terms_v2")))
    user_recent_terms = user_recent_semantic_terms | user_recent_context_terms
    merchant_core_terms = set(parse_terms_blob(row.get("merchant_core_terms_v2")))
    merchant_dish_terms = set(parse_terms_blob(row.get("merchant_dish_terms_v2")))
    merchant_scene_terms = set(parse_terms_blob(row.get("merchant_scene_terms_v2")))
    merchant_time_terms = set(parse_terms_blob(row.get("merchant_time_terms_v2")))
    merchant_property_terms = set(parse_terms_blob(row.get("merchant_property_terms_v2")))
    merchant_strength_terms = set(parse_terms_blob(row.get("merchant_strength_terms_v2")))
    merchant_source_strength_text = clean_text(row.get("merchant_source_strength_text_v1"), max_chars=0).lower()
    merchant_source_tip_text = clean_text(row.get("merchant_source_tip_text_v1"), max_chars=0).lower()
    merchant_source_strength_terms = set(
        specific_terms(
            extract_terms_from_sentential_text(merchant_source_strength_text, limit=6),
            limit=4,
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        )
    )
    merchant_source_tip_terms = set(
        specific_terms(
            extract_terms_from_sentential_text(merchant_source_tip_text, limit=5),
            limit=3,
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        )
    )
    merchant_basis_preview = build_merchant_basis_text(row, max_chars=260)
    visible_merchant_text = " ".join(
        text
        for text in [
            merchant_basis_preview.lower(),
            clean_text(row.get("core_offering_text"), max_chars=0).lower(),
            clean_text(row.get("strengths_text"), max_chars=0).lower(),
            clean_text(row.get("scene_fit_text"), max_chars=0).lower(),
            clean_text(row.get("merchant_semantic_profile_text_v2"), max_chars=0).lower(),
            merchant_source_strength_text,
            merchant_source_tip_text,
        ]
        if text
    )
    history_anchor_text = clean_text(row.get("history_anchor_hint_text"), max_chars=220)
    source_positive_text = clean_text(row.get("user_source_positive_text_v1"), max_chars=0).lower()
    source_recent_text = clean_text(row.get("user_source_recent_text_v1"), max_chars=0).lower()
    source_history_text = clean_text(row.get("user_source_history_anchor_text_v1"), max_chars=0).lower()
    history_anchor_terms = set(
        specific_terms(
            extract_terms_from_sentential_text(" ".join(text for text in [history_anchor_text, source_history_text] if text), limit=8),
            limit=4,
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        )
    )
    stable_user_text = clean_text(row.get("stable_preferences_text"), max_chars=0).lower()
    recent_user_text = clean_text(row.get("recent_intent_text_v2"), max_chars=0).lower()
    avoid_user_text = clean_text(row.get("avoidance_text_v2"), max_chars=0).lower()
    history_user_text = clean_text(row.get("history_anchor_hint_text"), max_chars=0).lower()
    evidence_user_text = clean_text(row.get("user_evidence_text_v2"), max_chars=0).lower()
    visible_user_text = " ".join(
        text
        for text in [
            stable_user_text,
            recent_user_text,
            avoid_user_text,
            history_user_text,
            evidence_user_text,
            source_positive_text,
            source_recent_text,
            source_history_text,
        ]
        if text
    )

    def _visible_user_subset(terms: set[str], visible_text: str) -> set[str]:
        out: set[str] = set()
        for term in terms:
            norm = normalize_term(term, max_chars=48)
            if norm and norm in visible_text:
                out.add(norm)
        return out

    def _visible_subset(terms: set[str]) -> set[str]:
        out: set[str] = set()
        for term in terms:
            norm = normalize_term(term, max_chars=48)
            if norm and norm in visible_merchant_text:
                out.add(norm)
        return out

    def _bridge_terms(
        left_terms: set[str],
        right_terms: set[str],
        *,
        broad: set[str] | None = None,
        limit: int = 3,
    ) -> list[str]:
        broad_terms = broad or set()
        direct_terms = specific_terms(
            intersect_terms_display(list(left_terms), list(right_terms), limit=max(limit + 1, limit), broad=broad_terms),
            limit=limit,
            broad=broad_terms,
        )
        if direct_terms:
            return direct_terms
        theme, user_examples, merchant_examples = best_theme_overlap(left_terms, right_terms)
        if not theme:
            return []
        bridged = specific_terms(
            dedupe_terms(
                user_examples,
                merchant_examples,
                [theme_label(theme)],
                limit=max(limit + 2, limit),
            ),
            limit=limit,
            broad=broad_terms,
        )
        if bridged:
            return bridged
        label = readable_term(theme_label(theme), max_chars=48)
        if label and normalize_term(label, max_chars=48) not in broad_terms:
            return [label]
        return []

    def _alias_bridge_terms(
        left_terms: set[str],
        right_terms: set[str],
        *,
        alias_groups: dict[str, set[str]] | None = None,
        limit: int = 3,
    ) -> list[str]:
        groups = alias_groups or {}
        if not groups or not left_terms or not right_terms:
            return []
        left_norm = {normalize_term(term, max_chars=48) for term in left_terms if normalize_term(term, max_chars=48)}
        right_norm = {normalize_term(term, max_chars=48) for term in right_terms if normalize_term(term, max_chars=48)}
        out: list[str] = []
        for canonical, variants in groups.items():
            canonical_norm = normalize_term(canonical, max_chars=48)
            variant_norms = {
                normalize_term(term, max_chars=48)
                for term in variants
                if normalize_term(term, max_chars=48)
            }
            if canonical_norm:
                variant_norms.add(canonical_norm)
            if left_norm.intersection(variant_norms) and right_norm.intersection(variant_norms):
                out.append(canonical)
            if len(out) >= limit:
                break
        return out

    def _clean_reason_terms(
        raw_terms: list[str] | set[str],
        *,
        broad: set[str] | None = None,
        limit: int = 3,
        keep_practical: bool = False,
        force_keep: set[str] | None = None,
    ) -> list[str]:
        broad_terms = broad or set()
        force_keep_terms = {
            normalize_term(term, max_chars=48)
            for term in (force_keep or set())
            if normalize_term(term, max_chars=48)
        }
        out: list[str] = []
        for term in dedupe_terms(list(raw_terms or []), limit=max(limit * 3, limit)):
            raw_norm = normalize_term(term, max_chars=48)
            display = readable_term(term, max_chars=48)
            if not display and raw_norm in force_keep_terms:
                display = raw_norm
            norm = normalize_term(display, max_chars=48)
            if not display or not norm:
                continue
            if norm in _LOW_SIGNAL_TOKENS or norm in _GENERIC_TERMS:
                continue
            if (
                broad_terms
                and norm in broad_terms
                and norm not in force_keep_terms
                and (not keep_practical or norm not in _PRACTICAL_SIGNAL_TERMS)
            ):
                continue
            out.append(display)
            if len(out) >= limit:
                break
        return out

    grounded_merchant_terms = _visible_subset(
        merchant_dish_terms
        | merchant_strength_terms
        | merchant_core_terms
        | merchant_source_strength_terms
        | merchant_source_tip_terms
    )
    grounded_merchant_scene_terms = _visible_subset(merchant_scene_terms)
    grounded_merchant_time_terms = _visible_subset(merchant_time_terms)
    grounded_merchant_property_terms = _visible_subset(merchant_property_terms)
    grounded_merchant_context_terms = grounded_merchant_scene_terms | grounded_merchant_time_terms | grounded_merchant_property_terms
    visible_user_core_terms = _visible_user_subset(user_core_terms, stable_user_text)
    visible_user_dish_terms = _visible_user_subset(
        user_dish_terms,
        " ".join(text for text in [stable_user_text, evidence_user_text, source_positive_text, source_history_text] if text),
    )
    visible_user_scene_terms = _visible_user_subset(
        user_scene_terms,
        " ".join(
            text
            for text in [stable_user_text, recent_user_text, source_positive_text, source_history_text, evidence_user_text]
            if text
        ),
    )
    visible_user_service_terms = _visible_user_subset(user_service_terms, " ".join(text for text in [stable_user_text, recent_user_text, evidence_user_text] if text))
    visible_user_recent_semantic_terms = _visible_user_subset(
        user_recent_semantic_terms,
        " ".join(text for text in [recent_user_text, source_recent_text] if text),
    )
    visible_user_recent_context_terms = _visible_user_subset(
        user_recent_context_terms,
        " ".join(
            text
            for text in [recent_user_text, source_recent_text, evidence_user_text, history_user_text]
            if text
        ),
    )
    visible_user_source_positive_terms = _visible_user_subset(
        user_source_positive_terms,
        " ".join(text for text in [source_positive_text, evidence_user_text, history_user_text] if text),
    )
    visible_user_source_recent_terms = _visible_user_subset(
        user_source_recent_terms,
        " ".join(text for text in [recent_user_text, source_recent_text, evidence_user_text] if text),
    )
    visible_user_source_tip_terms = _visible_user_subset(
        user_source_tip_terms,
        " ".join(text for text in [evidence_user_text, source_positive_text, recent_user_text] if text),
    )
    visible_user_source_history_terms = _visible_user_subset(
        user_source_history_terms,
        " ".join(text for text in [history_user_text, source_history_text, evidence_user_text] if text),
    )
    shared_dish_terms = _clean_reason_terms(
        intersect_terms_display(
            list(visible_user_dish_terms),
            list(grounded_merchant_terms),
            limit=4,
            broad=_BROAD_CORE_TERMS,
        ),
        broad=_BROAD_CORE_TERMS,
        limit=3,
    )
    shared_core_terms = _clean_reason_terms(
        intersect_terms_display(
            list(visible_user_core_terms),
            list(grounded_merchant_terms),
            limit=4,
            broad=_BROAD_CORE_TERMS,
        ),
        broad=_BROAD_CORE_TERMS,
        limit=3,
    )
    shared_scene_terms_raw = intersect_terms_display(
        list(visible_user_scene_terms),
        list(grounded_merchant_scene_terms | grounded_merchant_time_terms),
        limit=4,
    )
    specific_scene_terms = [
        term
        for term in shared_scene_terms_raw
        if term not in _GENERIC_SCENE_TERMS
        and (
            term not in _BROAD_SCENE_REASON_TERMS
            or normalize_term(term, max_chars=48) in _PRACTICAL_SIGNAL_TERMS
            or normalize_term(term, max_chars=48) in _CONTEXT_FIT_KEEP_TERMS
        )
    ]
    specific_scene_terms = _clean_reason_terms(
        specific_scene_terms,
        broad=_BROAD_SCENE_REASON_TERMS | _GENERIC_SCENE_TERMS,
        limit=3,
        keep_practical=True,
        force_keep=_CONTEXT_FIT_KEEP_TERMS,
    )
    shared_service_terms = intersect_terms_display(
        list(visible_user_service_terms),
        list(merchant_property_terms | merchant_strength_terms),
        limit=3,
    )
    shared_history_terms = _clean_reason_terms(
        intersect_terms_display(
            list(history_anchor_terms),
            list(grounded_merchant_terms),
            limit=4,
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        ),
        broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        limit=3,
    )
    beverage_service_terms = []
    if user_beverage_terms and merchant_property_terms.intersection({"full bar", "beer and wine"}):
        beverage_service_terms = ["full bar" if "full bar" in merchant_property_terms else "beer and wine"]
    specific_service_terms = [
        term for term in shared_service_terms
        if normalize_term(term, max_chars=48) not in _BROAD_SERVICE_REASON_TERMS
        or normalize_term(term, max_chars=48) in _PRACTICAL_SIGNAL_TERMS
    ]
    specific_service_terms = _clean_reason_terms(
        dedupe_terms(specific_service_terms, beverage_service_terms, limit=4),
        broad=_BROAD_SERVICE_REASON_TERMS,
        limit=3,
        keep_practical=True,
    )
    stable_bridge_terms = _clean_reason_terms(_bridge_terms(
        visible_user_dish_terms | visible_user_core_terms,
        grounded_merchant_terms,
        broad=_BROAD_CORE_TERMS,
        limit=3,
    ), broad=_BROAD_CORE_TERMS, limit=3)
    source_supported_stable_terms = _clean_reason_terms(_bridge_terms(
        visible_user_source_positive_terms | visible_user_source_history_terms | history_anchor_terms,
        grounded_merchant_terms | merchant_source_strength_terms | merchant_source_tip_terms,
        broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        limit=3,
    ), broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS, limit=3)
    recent_bridge_terms = _clean_reason_terms(_bridge_terms(
        visible_user_recent_semantic_terms,
        grounded_merchant_terms,
        broad=_BROAD_RECENT_TERMS,
        limit=3,
    ), broad=_BROAD_RECENT_TERMS, limit=3)
    source_recent_bridge_terms = _clean_reason_terms(_bridge_terms(
        visible_user_source_recent_terms | visible_user_source_tip_terms,
        grounded_merchant_terms | merchant_source_strength_terms | merchant_source_tip_terms,
        broad=_BROAD_RECENT_TERMS,
        limit=3,
    ), broad=_BROAD_RECENT_TERMS, limit=3)
    history_bridge_terms = _clean_reason_terms(_bridge_terms(
        history_anchor_terms,
        grounded_merchant_terms,
        broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
        limit=3,
    ), broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS, limit=3)
    practical_bridge_terms = _clean_reason_terms(_bridge_terms(
        visible_user_service_terms | visible_user_recent_context_terms | visible_user_scene_terms,
        grounded_merchant_property_terms | grounded_merchant_context_terms | merchant_strength_terms,
        broad=(
            (_BROAD_SERVICE_REASON_TERMS | _BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS)
            - _PRACTICAL_SIGNAL_TERMS
        ),
        limit=3,
    ), broad=(
        (_BROAD_SERVICE_REASON_TERMS | _BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS)
        - _PRACTICAL_SIGNAL_TERMS
    ), limit=3, keep_practical=True)
    source_practical_bridge_terms = _clean_reason_terms(_bridge_terms(
        visible_user_source_recent_terms
        | visible_user_source_tip_terms
        | visible_user_recent_context_terms
        | visible_user_service_terms
        | visible_user_scene_terms,
        grounded_merchant_property_terms
        | grounded_merchant_context_terms
        | merchant_strength_terms
        | merchant_source_strength_terms
        | merchant_source_tip_terms,
        broad=(
            (_BROAD_SERVICE_REASON_TERMS | _BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS)
            - _PRACTICAL_SIGNAL_TERMS
        ),
        limit=3,
    ), broad=(
        (_BROAD_SERVICE_REASON_TERMS | _BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS)
        - _PRACTICAL_SIGNAL_TERMS
    ), limit=3, keep_practical=True)
    prop = finite_or_zero(row.get("mean_channel_preference_property_v1", 0.0))
    total_match = finite_or_zero(row.get("mean_match_total_v1", 0.0))
    sim_long = finite_or_zero(row.get("sim_long_pref_core", 0.0))
    sim_recent_sem = finite_or_zero(row.get("sim_recent_intent_semantic", 0.0))
    sim_recent_pos = finite_or_zero(row.get("sim_recent_intent_pos", 0.0))
    sim_context = finite_or_zero(row.get("sim_context_merchant", 0.0))

    pref_core = finite_or_zero(row.get("mean_channel_preference_core_v1", 0.0))
    candidate_pref_support = max(pref_core, sim_long, total_match) >= 0.18
    if max(pref_core, sim_long) >= 0.11 and shared_dish_terms:
        reasons.append(
            _reason_candidate(
                score=max(pref_core, sim_long),
                text=f"The user's stable profile repeatedly points toward {join_terms_natural(shared_dish_terms, limit=3)}, and this business shows the same pattern in its menu and review evidence.",
                scope="stable",
                polarity="fit",
                terms=shared_dish_terms,
                evidence_source="stable_profile",
                specificity=2,
            )
        )
    elif max(pref_core, sim_long) >= 0.10 and stable_bridge_terms:
        reasons.append(
            _reason_candidate(
                score=max(pref_core, sim_long),
                text=f"The business stays close to the user's stronger long-run preferences around {join_terms_natural(stable_bridge_terms, limit=3)}.",
                scope="stable",
                polarity="fit",
                terms=stable_bridge_terms,
                evidence_source="stable_profile",
                specificity=2,
            )
        )
    elif max(pref_core, sim_long, total_match) >= 0.12 and source_supported_stable_terms:
        reasons.append(
            _reason_candidate(
                score=max(pref_core, sim_long, total_match),
                text=f"Source positives and older favorites repeatedly point toward {join_terms_natural(source_supported_stable_terms, limit=3)}, and this business shows the same pattern in its visible offering.",
                scope="stable",
                polarity="fit",
                terms=source_supported_stable_terms,
                evidence_source="source_profile",
                specificity=2,
            )
        )
    elif max(pref_core, sim_long) >= 0.14 and specific_terms(shared_core_terms, limit=3, broad=_BROAD_CORE_TERMS):
        concrete_core_terms = specific_terms(shared_core_terms, limit=3, broad=_BROAD_CORE_TERMS)
        reasons.append(
            _reason_candidate(
                score=max(pref_core, sim_long),
                text=f"The business stays close to the user's stronger long-run preferences around {join_terms_natural(concrete_core_terms, limit=3)}.",
                scope="stable",
                polarity="fit",
                terms=concrete_core_terms,
                evidence_source="stable_profile",
                specificity=1,
            )
        )
    elif total_match >= 0.08 and (shared_dish_terms or shared_core_terms or stable_bridge_terms):
        visible_overlap_terms = _clean_reason_terms(
            dedupe_terms(shared_dish_terms, shared_core_terms, stable_bridge_terms, limit=4),
            broad=_BROAD_CORE_TERMS,
            limit=3,
        )
        if visible_overlap_terms:
            reasons.append(
                _reason_candidate(
                    score=total_match,
                    text=f"Visible user preferences still overlap with this business around {join_terms_natural(visible_overlap_terms, limit=3)}.",
                    scope="stable",
                    polarity="fit",
                    terms=visible_overlap_terms,
                    evidence_source="visible_overlap",
                    specificity=1,
                )
            )
    if max(pref_core, sim_long, total_match) >= 0.11 and shared_history_terms:
        reasons.append(
            _reason_candidate(
                score=max(pref_core, sim_long, total_match),
                text=f"Older positive visits repeatedly highlighted {join_terms_natural(shared_history_terms, limit=3)}, and this business foregrounds those same choices.",
                scope="history",
                polarity="fit",
                terms=shared_history_terms,
                evidence_source="history_anchor",
                specificity=2,
            )
        )
    elif max(pref_core, sim_long, total_match) >= 0.09 and history_bridge_terms:
        reasons.append(
            _reason_candidate(
                score=max(pref_core, sim_long, total_match),
                text=f"Older positive visits repeatedly highlighted {join_terms_natural(history_bridge_terms, limit=3)}, and this business foregrounds those same choices.",
                scope="history",
                polarity="fit",
                terms=history_bridge_terms,
                evidence_source="history_anchor",
                specificity=2,
            )
        )
    if prop >= 0.10 and specific_service_terms:
        reasons.append(
            _reason_candidate(
                score=prop,
                text=f"It also fits practical details the user seems to care about, especially {join_terms_natural(specific_service_terms, limit=3)}.",
                scope="practical",
                polarity="fit",
                terms=specific_service_terms,
                evidence_source="practical_preference",
                specificity=1,
            )
        )
    elif prop >= 0.08 and practical_bridge_terms:
        reasons.append(
            _reason_candidate(
                score=prop,
                text=f"It also matches practical visit details the user tends to care about, especially {join_terms_natural(practical_bridge_terms, limit=3)}.",
                scope="practical",
                polarity="fit",
                terms=practical_bridge_terms,
                evidence_source="practical_preference",
                specificity=2,
            )
        )
    elif max(prop, sim_context, total_match) >= 0.08 and source_practical_bridge_terms:
        reasons.append(
            _reason_candidate(
                score=max(prop, sim_context, total_match),
                text=f"It also matches practical visit details that show up in source feedback, especially {join_terms_natural(source_practical_bridge_terms, limit=3)}.",
                scope="practical",
                polarity="fit",
                terms=source_practical_bridge_terms,
                evidence_source="source_practical",
                specificity=2,
            )
        )
    elif prop >= 0.10 and user_beverage_terms and beverage_service_terms:
        reasons.append(
            _reason_candidate(
                score=prop,
                text=f"It also fits the user's bar-oriented preference because the business offers {join_terms_natural(beverage_service_terms, limit=2)}.",
                scope="practical",
                polarity="fit",
                terms=beverage_service_terms,
                evidence_source="beverage_preference",
                specificity=1,
            )
        )

    recent = finite_or_zero(row.get("mean_channel_recent_intent_v1", 0.0))
    recent_shared_terms = _clean_reason_terms(specific_terms(
        intersect_terms_display(
            list(visible_user_recent_semantic_terms),
            list(grounded_merchant_terms),
            limit=4,
        ),
        limit=3,
        broad=_BROAD_RECENT_TERMS,
    ), broad=_BROAD_RECENT_TERMS, limit=3)
    recent_context_shared_terms = [
        term
        for term in intersect_terms_display(
            list(visible_user_recent_context_terms),
            list(grounded_merchant_context_terms),
            limit=4,
        )
        if normalize_term(term, max_chars=48) not in _GENERIC_SCENE_TERMS
        and (
            normalize_term(term, max_chars=48) in _PRACTICAL_SIGNAL_TERMS
            or (
                normalize_term(term, max_chars=48) not in _BROAD_SCENE_REASON_TERMS
                and normalize_term(term, max_chars=48) not in _BROAD_RECENT_PROPERTY_TERMS
            )
            or normalize_term(term, max_chars=48) in _CONTEXT_FIT_KEEP_TERMS
        )
    ]
    recent_context_shared_terms = _clean_reason_terms(
        recent_context_shared_terms,
        broad=_BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS,
        limit=3,
        keep_practical=True,
        force_keep=_CONTEXT_FIT_KEEP_TERMS,
    )
    recent_context_alias_terms = _clean_reason_terms(
        _alias_bridge_terms(
            visible_user_recent_context_terms
            | visible_user_scene_terms
            | visible_user_source_recent_terms
            | visible_user_source_tip_terms,
            grounded_merchant_context_terms | grounded_merchant_scene_terms | grounded_merchant_time_terms,
            alias_groups=_RECENT_CONTEXT_ALIAS_GROUPS,
            limit=4,
        ),
        broad=_BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS,
        limit=4,
        keep_practical=True,
        force_keep=_CONTEXT_FIT_KEEP_TERMS,
    )
    recent_context_shared_terms = _clean_reason_terms(
        dedupe_terms(recent_context_shared_terms, recent_context_alias_terms, limit=5),
        broad=_BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS,
        limit=4,
        keep_practical=True,
        force_keep=_CONTEXT_FIT_KEEP_TERMS,
    )
    recent_unique_terms = [term for term in recent_shared_terms if normalize_term(term, max_chars=48) not in {normalize_term(v, max_chars=48) for v in shared_dish_terms + shared_core_terms}]
    recent_alignment_terms = _clean_reason_terms(
        recent_unique_terms or recent_bridge_terms or source_recent_bridge_terms or recent_shared_terms,
        broad=_BROAD_RECENT_TERMS,
        limit=3,
        keep_practical=True,
    )
    candidate_recent_support = max(recent, sim_recent_sem, sim_recent_pos) >= 0.12
    if max(recent, sim_recent_sem, sim_recent_pos) >= 0.08 and recent_alignment_terms:
        reasons.append(
            _reason_candidate(
                score=max(recent, sim_recent_sem, sim_recent_pos),
                text=f"Recent activity has leaned toward {join_terms_natural(recent_alignment_terms, limit=3)}, and those same themes show up clearly in this business.",
                scope="recent",
                polarity="fit",
                terms=recent_alignment_terms,
                evidence_source="recent_intent",
                specificity=2,
            )
        )
    elif max(recent, sim_recent_sem, sim_recent_pos) >= 0.10 and recent_shared_terms and not shared_dish_terms:
        reasons.append(
            _reason_candidate(
                score=max(recent, sim_recent_sem, sim_recent_pos),
                text=f"The user's recent pattern still clusters around {join_terms_natural(recent_shared_terms, limit=3)}, and those details also appear in this business's offering.",
                scope="recent",
                polarity="fit",
                terms=recent_shared_terms,
                evidence_source="recent_intent",
                specificity=1,
            )
        )
    elif max(recent, sim_recent_sem, sim_recent_pos, total_match) >= 0.11 and source_recent_bridge_terms:
        reasons.append(
            _reason_candidate(
                score=max(recent, sim_recent_sem, sim_recent_pos, total_match),
                text=f"Recent source evidence still points toward {join_terms_natural(source_recent_bridge_terms, limit=3)}, and those same details appear in this business.",
                scope="recent",
                polarity="fit",
                terms=source_recent_bridge_terms,
                evidence_source="source_recent",
                specificity=2,
            )
        )
    recent_context_bridge_score = max(sim_context, recent, sim_recent_sem, prop, total_match)
    candidate_context_support = sim_context >= 0.14 or recent_context_bridge_score >= 0.16
    recent_context_has_practical_term = any(
        normalize_term(term, max_chars=48) in _CONTEXT_FIT_KEEP_TERMS
        for term in recent_context_shared_terms
    )
    if recent_context_shared_terms and (
        max(sim_context, recent, sim_recent_sem) >= 0.08
        or (recent_context_alias_terms and recent_context_bridge_score >= 0.05)
        or (recent_context_has_practical_term and recent_context_bridge_score >= 0.04)
    ):
        reasons.append(
            _reason_candidate(
                score=recent_context_bridge_score,
                text=f"The user's recent outings keep clustering around {join_terms_natural(recent_context_shared_terms, limit=3)}, and this business supports that same pattern.",
                scope="recent_context",
                polarity="fit",
                terms=recent_context_shared_terms,
                evidence_source="recent_context",
                specificity=2 if (recent_context_alias_terms or recent_context_has_practical_term) else 1,
            )
        )
    elif recent_context_bridge_score >= 0.09 and source_practical_bridge_terms:
        reasons.append(
            _reason_candidate(
                score=recent_context_bridge_score,
                text=f"Recent context and source support both point to {join_terms_natural(source_practical_bridge_terms, limit=3)}, and this business supports that same visit pattern.",
                scope="recent_context",
                polarity="fit",
                terms=source_practical_bridge_terms,
                evidence_source="source_context",
                specificity=2,
            )
        )
    evidence = finite_or_zero(row.get("mean_channel_evidence_support_v1", 0.0))
    if evidence > 0.04 and specific_service_terms:
        reasons.append(
            _reason_candidate(
                score=evidence,
                text=f"Review evidence also points to practical details that fit this user's preferences, especially {join_terms_natural(specific_service_terms, limit=3)}.",
                scope="evidence",
                polarity="fit",
                terms=specific_service_terms,
                evidence_source="review_evidence",
                specificity=1,
            )
        )
    elif evidence > 0.03 and history_bridge_terms and merchant_source_strength_terms:
        reasons.append(
            _reason_candidate(
                score=evidence,
                text=f"Source review evidence also supports this match around {join_terms_natural(history_bridge_terms, limit=3)}.",
                scope="evidence",
                polarity="fit",
                terms=history_bridge_terms,
                evidence_source="review_evidence",
                specificity=2,
            )
        )
    elif evidence > 0.03 and stable_bridge_terms and (merchant_source_strength_terms or merchant_source_tip_terms):
        reasons.append(
            _reason_candidate(
                score=evidence,
                text=f"Source review evidence also supports this match around {join_terms_natural(stable_bridge_terms, limit=3)}.",
                scope="evidence",
                polarity="fit",
                terms=stable_bridge_terms,
                evidence_source="review_evidence",
                specificity=2,
            )
        )
    else:
        evidence_bridge_terms = _clean_reason_terms(
            dedupe_terms(
                recent_alignment_terms,
                history_bridge_terms,
                stable_bridge_terms,
                shared_dish_terms,
                shared_core_terms,
                limit=5,
            ),
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
            limit=3,
            keep_practical=True,
        )
        if evidence > 0.02 and evidence_bridge_terms and (merchant_source_strength_terms or merchant_source_tip_terms or merchant_strength_terms):
            reasons.append(
                _reason_candidate(
                    score=evidence,
                    text=f"Source review evidence also reinforces this match around {join_terms_natural(evidence_bridge_terms, limit=3)}.",
                    scope="evidence",
                    polarity="fit",
                    terms=evidence_bridge_terms,
                    evidence_source="review_evidence",
                    specificity=2,
                )
            )
    if max(sim_context, recent, sim_recent_sem) >= 0.10 and recent_context_shared_terms and (shared_dish_terms or shared_core_terms or recent_unique_terms or specific_service_terms):
        reasons.append(
            _reason_candidate(
                score=max(sim_context, recent, sim_recent_sem),
                text=f"The setting also fits how this user's recent outings are structured, especially around {join_terms_natural(recent_context_shared_terms, limit=3)}.",
                scope="recent_context",
                polarity="fit",
                terms=recent_context_shared_terms,
                evidence_source="recent_context",
                specificity=1,
            )
        )
    elif max(sim_context, recent) >= 0.12 and specific_scene_terms and (shared_dish_terms or shared_core_terms or recent_unique_terms):
        reasons.append(
            _reason_candidate(
                score=max(sim_context, recent),
                text=f"The setting also fits the user's usual outing pattern around {join_terms_natural(specific_scene_terms, limit=3)}.",
                scope="recent_context",
                polarity="fit",
                terms=specific_scene_terms,
                evidence_source="recent_context",
                specificity=1,
            )
        )
    if not reasons:
        fallback_recent_terms = _clean_reason_terms(
            dedupe_terms(
                recent_context_alias_terms,
                recent_context_shared_terms,
                source_recent_bridge_terms,
                source_practical_bridge_terms,
                recent_alignment_terms,
                limit=5,
            ),
            broad=_BROAD_RECENT_TERMS | _BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS,
            limit=3,
            keep_practical=True,
            force_keep=_CONTEXT_FIT_KEEP_TERMS,
        )
        fallback_stable_terms = _clean_reason_terms(
            dedupe_terms(
                source_supported_stable_terms,
                history_bridge_terms,
                stable_bridge_terms,
                shared_dish_terms,
                shared_core_terms,
                limit=5,
            ),
            broad=_BROAD_CORE_TERMS | _BROAD_RECENT_TERMS,
            limit=3,
            keep_practical=True,
        )
        fallback_practical_terms = _clean_reason_terms(
            dedupe_terms(
                source_practical_bridge_terms,
                practical_bridge_terms,
                specific_service_terms,
                recent_context_alias_terms,
                limit=5,
            ),
            broad=_BROAD_SERVICE_REASON_TERMS | _BROAD_SCENE_REASON_TERMS | _BROAD_RECENT_PROPERTY_TERMS | _GENERIC_SCENE_TERMS,
            limit=3,
            keep_practical=True,
            force_keep=_CONTEXT_FIT_KEEP_TERMS,
        )
        if candidate_context_support and fallback_recent_terms:
            reasons.append(
                _reason_candidate(
                    score=max(recent_context_bridge_score, sim_context, recent, sim_recent_sem, total_match),
                    text=f"Recent visit context and stronger pair signals both point toward {join_terms_natural(fallback_recent_terms, limit=3)}, and this business supports that same pattern.",
                    scope="recent_context",
                    polarity="fit",
                    terms=fallback_recent_terms,
                    evidence_source="contrast_bridge",
                    specificity=2,
                )
            )
        elif candidate_recent_support and fallback_recent_terms:
            reasons.append(
                _reason_candidate(
                    score=max(recent, sim_recent_sem, sim_recent_pos, total_match),
                    text=f"Recent intent and source support still point toward {join_terms_natural(fallback_recent_terms, limit=3)}, and those same details are visible in this business.",
                    scope="recent",
                    polarity="fit",
                    terms=fallback_recent_terms,
                    evidence_source="contrast_bridge",
                    specificity=2,
                )
            )
        elif candidate_pref_support and fallback_stable_terms:
            reasons.append(
                _reason_candidate(
                    score=max(pref_core, sim_long, total_match),
                    text=f"Long-run preferences and older positive evidence still line up around {join_terms_natural(fallback_stable_terms, limit=3)}, and this business shows that same fit.",
                    scope="stable",
                    polarity="fit",
                    terms=fallback_stable_terms,
                    evidence_source="contrast_bridge",
                    specificity=2,
                )
            )
        elif max(prop, recent_context_bridge_score, total_match) >= 0.14 and fallback_practical_terms:
            reasons.append(
                _reason_candidate(
                    score=max(prop, recent_context_bridge_score, total_match),
                    text=f"Practical visit details also line up around {join_terms_natural(fallback_practical_terms, limit=3)}, and this business supports that same visit pattern.",
                    scope="practical",
                    polarity="fit",
                    terms=fallback_practical_terms,
                    evidence_source="contrast_bridge",
                    specificity=2,
                )
            )
    reasons.sort(key=lambda x: x["score"], reverse=True)
    out_texts: list[str] = []
    out_facts: list[dict[str, Any]] = []
    for cand in reasons:
        cleaned = clean_text(cand.get("text", ""), max_chars=190)
        if (
            not cleaned
            or cleaned.lower() in _GENERIC_FIT_PHRASES
            or is_timing_only_fit_text(cleaned)
        ):
            continue
        before = len(out_texts)
        append_reason_if_distinct(out_texts, cleaned, max_chars=190)
        if len(out_texts) == before:
            continue
        out_facts.append(
            build_semantic_fact(
                scope=str(cand.get("scope", "") or "fit"),
                polarity=str(cand.get("polarity", "") or "fit"),
                text=cleaned,
                facet_type="user_business_fit",
                evidence_source=str(cand.get("evidence_source", "") or "fit_reason"),
                terms=list(cand.get("terms", []) or []),
                specificity=int(cand.get("specificity", 0) or 0),
                max_chars=200,
            )
        )
        if len(out_facts) >= 3:
            break
    return [fact for fact in out_facts if fact]


def describe_fit_reason(row: pd.Series) -> list[str]:
    return [str(fact.get("text", "") or "") for fact in _collect_fit_reason_facts(row)]


def _collect_friction_reason_facts(row: pd.Series) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    conflict = finite_or_zero(row.get("mean_channel_conflict_v1", 0.0))
    avoid_text = clean_text(row.get("avoidance_text_v2"), max_chars=190)
    risk_points = clean_text(row.get("risk_points_text"), max_chars=190)
    user_avoid_terms = set(parse_terms_blob(row.get("user_avoid_terms_v2")))
    merchant_risk_terms = set(parse_terms_blob(row.get("merchant_risk_terms_v2")))
    shared_risk_terms = intersect_terms_display(list(user_avoid_terms), list(merchant_risk_terms), limit=3)
    neg_avoid_core = finite_or_zero(row.get("sim_negative_avoid_core", 0.0))
    neg_avoid_neg = finite_or_zero(row.get("sim_negative_avoid_neg", 0.0))
    conflict_gap = finite_or_zero(row.get("sim_conflict_gap", 0.0))
    if max(conflict, neg_avoid_core, neg_avoid_neg) >= 0.06 and shared_risk_terms:
        reasons.append(
            _reason_candidate(
                score=max(conflict, neg_avoid_core, neg_avoid_neg),
                text=f"This business shows the same kinds of friction the user has reacted badly to before, especially {join_terms_natural(shared_risk_terms, limit=3)}.",
                scope="avoid",
                polarity="friction",
                terms=shared_risk_terms,
                evidence_source="avoidance_overlap",
                specificity=2,
            )
        )
    elif max(conflict, neg_avoid_core, neg_avoid_neg) >= 0.06 and avoid_text and merchant_risk_terms:
        reasons.append(
            _reason_candidate(
                score=max(conflict, neg_avoid_core, neg_avoid_neg),
                text=f"This business may clash with the user's avoidances because reviews repeatedly mention {join_terms_natural(list(merchant_risk_terms), limit=3)}.",
                scope="avoid",
                polarity="friction",
                terms=list(merchant_risk_terms),
                evidence_source="merchant_risk",
                specificity=1,
            )
        )
    neg_conf = finite_or_zero(row.get("mean_match_negative_conflict", 0.0))
    if neg_conf >= 0.002:
        for pattern, phrase in _NEGATIVE_CUE_PATTERNS:
            if pattern in avoid_text.lower() and pattern in risk_points.lower():
                reasons.append(
                    _reason_candidate(
                        score=neg_conf,
                        text=f"The business may create friction around {phrase}.",
                        scope="avoid",
                        polarity="friction",
                        terms=[phrase],
                        evidence_source="negative_conflict",
                        specificity=1,
                    )
                )
                break
    if not reasons and min(conflict_gap, conflict) <= -0.03 and risk_points and avoid_text:
        reasons.append(
            _reason_candidate(
                score=abs(min(conflict_gap, conflict)),
                text="Its risk profile overlaps with issues this user has reacted poorly to before.",
                scope="avoid",
                polarity="friction",
                evidence_source="risk_overlap",
            )
        )
    if not reasons and risk_points and avoid_text and max(conflict, neg_conf, neg_avoid_core, neg_avoid_neg) >= 0.04:
        reasons.append(
            _reason_candidate(
                score=max(conflict, neg_conf, neg_avoid_core, neg_avoid_neg),
                text="The business also carries risks that overlap with this user's known friction points.",
                scope="avoid",
                polarity="friction",
                evidence_source="risk_overlap",
            )
        )
    if not reasons and merchant_risk_terms and max(conflict, neg_conf, neg_avoid_core, neg_avoid_neg) >= 0.03:
        specific_risk_terms = specific_terms(list(merchant_risk_terms), limit=3)
        if specific_risk_terms:
            reasons.append(
                _reason_candidate(
                    score=max(conflict, neg_conf, neg_avoid_core, neg_avoid_neg),
                    text=f"Potential friction is still worth watching because reviews repeatedly flag {join_terms_natural(specific_risk_terms, limit=3)} here.",
                    scope="risk",
                    polarity="friction",
                    terms=specific_risk_terms,
                    evidence_source="merchant_risk",
                    specificity=1,
                )
            )
    reasons.sort(key=lambda x: x["score"], reverse=True)
    out_texts: list[str] = []
    out_facts: list[dict[str, Any]] = []
    for cand in reasons:
        cleaned = clean_text(cand.get("text", ""), max_chars=200)
        if not cleaned:
            continue
        before = len(out_texts)
        append_reason_if_distinct(out_texts, cleaned, max_chars=200)
        if len(out_texts) == before:
            continue
        out_facts.append(
            build_semantic_fact(
                scope=str(cand.get("scope", "") or "friction"),
                polarity=str(cand.get("polarity", "") or "friction"),
                text=cleaned,
                facet_type="user_business_friction",
                evidence_source=str(cand.get("evidence_source", "") or "friction_reason"),
                terms=list(cand.get("terms", []) or []),
                specificity=int(cand.get("specificity", 0) or 0),
                max_chars=220,
            )
        )
        if len(out_facts) >= 2:
            break
    return [fact for fact in out_facts if fact]


def describe_friction_reason(row: pd.Series) -> list[str]:
    return [str(fact.get("text", "") or "") for fact in _collect_friction_reason_facts(row)]


def build_pair_alignment_assets(pair_df: pd.DataFrame) -> pd.DataFrame:
    return _parallel_build_pair_alignment_assets(pair_df)


def build_stage11_target_users_filtered_assets(
    target_users_df: pd.DataFrame,
    user_assets_df: pd.DataFrame,
    pair_assets_df: pd.DataFrame,
    pair_universe_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_cols = [
        "bucket",
        "user_idx",
        "user_id",
        "user_profile_richness_v2",
        "user_profile_richness_tier_v2",
        "user_quality_signal_count_v2",
        "user_has_avoid_signal_v2",
        "user_has_clean_evidence_v2",
        "user_has_specific_focus_v2",
        "user_has_specific_recent_v2",
        "user_has_recent_text_v2",
        "user_has_strong_clean_evidence_v2",
        "user_has_history_evidence_v2",
        "history_anchor_source_v2",
    ]
    pair_stats = (
        pair_assets_df.groupby(["bucket", "user_idx", "user_id"], dropna=False)
        .agg(
            pair_rows_top100=("item_idx", "count"),
            pair_full_count=("semantic_prompt_readiness_tier_v1", lambda s: int((s == "FULL").sum())),
            pair_partial_count=("semantic_prompt_readiness_tier_v1", lambda s: int((s == "PARTIAL").sum())),
            pair_low_signal_count=("semantic_prompt_readiness_tier_v1", lambda s: int((s == "LOW_SIGNAL").sum())),
            pair_blank_fit_count=("fit_reasons_text_v1", lambda s: int((s.fillna("").str.len() == 0).sum())),
            pair_concrete_fit_count=("pair_has_concrete_fit_v1", "sum"),
            pair_timing_only_fit_count=("pair_has_timing_only_fit_v1", "sum"),
            pair_alignment_richness_mean=("pair_alignment_richness_v1", "mean"),
            pair_alignment_richness_max=("pair_alignment_richness_v1", "max"),
            pair_fit_fact_count_sum=("pair_fit_fact_count_v1", "sum"),
            pair_friction_fact_count_sum=("pair_friction_fact_count_v1", "sum"),
            pair_evidence_fact_count_sum=("pair_evidence_fact_count_v1", "sum"),
            pair_stable_fit_fact_count_sum=("pair_stable_fit_fact_count_v1", "sum"),
            pair_recent_fit_fact_count_sum=("pair_recent_fit_fact_count_v1", "sum"),
            pair_history_fit_fact_count_sum=("pair_history_fit_fact_count_v1", "sum"),
            pair_practical_fit_fact_count_sum=("pair_practical_fit_fact_count_v1", "sum"),
            pair_fit_scope_count_sum=("pair_fit_scope_count_v1", "sum"),
            pair_rows_with_fit_facts=("pair_fit_fact_count_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_conflict=("pair_friction_fact_count_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_evidence=("pair_evidence_fact_count_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_recent_fit=("pair_recent_fit_fact_count_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_history_fit=("pair_history_fit_fact_count_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_practical_fit=("pair_practical_fit_fact_count_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_visible_user_fit=("pair_has_visible_user_fit_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_visible_conflict=("pair_has_visible_conflict_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_recent_context_bridge=("pair_has_recent_context_visible_bridge_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_detail_support=("pair_has_detail_support_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_multisource_fit=("pair_has_multisource_fit_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_candidate_fit_support=("pair_has_candidate_fit_support_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_candidate_strong_fit_support=("pair_has_candidate_strong_fit_support_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
            pair_rows_with_contrastive_support=("pair_has_contrastive_support_v1", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(float) > 0).sum())),
        )
        .reset_index()
    )
    pair_stats["pair_non_low_signal_count"] = (
        pair_stats["pair_full_count"].astype(int) + pair_stats["pair_partial_count"].astype(int)
    )

    users = (
        target_users_df.merge(user_assets_df[user_cols], on=["bucket", "user_idx", "user_id"], how="left")
        .merge(pair_stats, on=["bucket", "user_idx", "user_id"], how="left")
    )
    fill_int_cols = [
        "user_profile_richness_v2",
        "user_quality_signal_count_v2",
        "pair_rows_top100",
        "pair_full_count",
        "pair_partial_count",
        "pair_low_signal_count",
        "pair_non_low_signal_count",
        "pair_blank_fit_count",
        "pair_concrete_fit_count",
        "pair_timing_only_fit_count",
        "pair_alignment_richness_max",
        "pair_fit_fact_count_sum",
        "pair_friction_fact_count_sum",
        "pair_evidence_fact_count_sum",
        "pair_stable_fit_fact_count_sum",
        "pair_recent_fit_fact_count_sum",
        "pair_history_fit_fact_count_sum",
        "pair_practical_fit_fact_count_sum",
        "pair_fit_scope_count_sum",
        "pair_rows_with_fit_facts",
        "pair_rows_with_conflict",
        "pair_rows_with_evidence",
        "pair_rows_with_recent_fit",
        "pair_rows_with_history_fit",
        "pair_rows_with_practical_fit",
        "pair_rows_with_visible_user_fit",
        "pair_rows_with_visible_conflict",
        "pair_rows_with_recent_context_bridge",
        "pair_rows_with_detail_support",
        "pair_rows_with_multisource_fit",
        "pair_rows_with_candidate_fit_support",
        "pair_rows_with_candidate_strong_fit_support",
        "pair_rows_with_contrastive_support",
        "user_has_avoid_signal_v2",
        "user_has_clean_evidence_v2",
        "user_has_specific_focus_v2",
        "user_has_specific_recent_v2",
        "user_has_strong_clean_evidence_v2",
        "user_has_history_evidence_v2",
    ]
    for col in fill_int_cols:
        users[col] = pd.to_numeric(users.get(col), errors="coerce").fillna(0).astype(int)
    users["pair_alignment_richness_mean"] = pd.to_numeric(
        users.get("pair_alignment_richness_mean"), errors="coerce"
    ).fillna(0.0)
    users["user_profile_richness_tier_v2"] = users.get("user_profile_richness_tier_v2", "").fillna("").astype(str)
    users["history_anchor_source_v2"] = users.get("history_anchor_source_v2", "").fillna("").astype(str)

    boundary_users = target_users_df.loc[
        pd.to_numeric(target_users_df.get("truth_learned_rank"), errors="coerce").between(11, 30)
    ].copy()
    boundary_annotations = pd.DataFrame(
        columns=[
            "bucket",
            "user_idx",
            "user_id",
            "item_idx",
            "boundary_constructability_class_v1",
            "boundary_constructability_reason_codes_v1",
            "boundary_prompt_ready_v1",
            "boundary_rival_total_v1",
            "boundary_rival_head_or_boundary_v1",
        ]
    )
    if not boundary_users.empty:
        boundary_users["bucket"] = pd.to_numeric(boundary_users["bucket"], errors="coerce").astype("Int64")
        boundary_users["user_idx"] = pd.to_numeric(boundary_users["user_idx"], errors="coerce").astype("Int64")
        boundary_user_support_cols = [
            "bucket",
            "user_idx",
            "user_id",
            "user_has_specific_focus_v2",
            "user_has_specific_recent_v2",
            "user_has_avoid_signal_v2",
            "user_has_history_evidence_v2",
            "user_has_strong_clean_evidence_v2",
            "user_focus_fact_count_v1",
            "user_recent_fact_count_v1",
            "user_avoid_fact_count_v1",
            "user_history_fact_count_v1",
            "user_evidence_fact_count_v1",
        ]
        boundary_users = boundary_users.merge(
            users[[c for c in boundary_user_support_cols if c in users.columns]].drop_duplicates(
                subset=["bucket", "user_idx", "user_id"],
                keep="first",
            ),
            on=["bucket", "user_idx", "user_id"],
            how="left",
        )
        if "true_item_idx" not in boundary_users.columns:
            truth_item_lookup = (
                pair_universe_df[["bucket", "user_idx", "user_id", "true_item_idx"]]
                .drop_duplicates(subset=["bucket", "user_idx", "user_id"])
                .copy()
            )
            boundary_users = boundary_users.merge(
                truth_item_lookup,
                on=["bucket", "user_idx", "user_id"],
                how="left",
            )
        boundary_users["true_item_idx"] = pd.to_numeric(boundary_users["true_item_idx"], errors="coerce").astype("Int64")
        boundary_pair_assets = pair_assets_df.copy()
        boundary_pair_assets["bucket"] = pd.to_numeric(boundary_pair_assets["bucket"], errors="coerce").astype("Int64")
        boundary_pair_assets["user_idx"] = pd.to_numeric(boundary_pair_assets["user_idx"], errors="coerce").astype("Int64")
        boundary_pair_assets["item_idx"] = pd.to_numeric(boundary_pair_assets["item_idx"], errors="coerce").astype("Int64")
        boundary_pair_universe = pair_universe_df.copy()
        boundary_pair_universe["bucket"] = pd.to_numeric(boundary_pair_universe["bucket"], errors="coerce").astype("Int64")
        boundary_pair_universe["user_idx"] = pd.to_numeric(boundary_pair_universe["user_idx"], errors="coerce").astype("Int64")
        boundary_pair_universe["item_idx"] = pd.to_numeric(boundary_pair_universe["item_idx"], errors="coerce").astype("Int64")
        boundary_pair_universe["learned_rank_band"] = pd.to_numeric(
            boundary_pair_universe.get("learned_rank"), errors="coerce"
        ).map(
            lambda rank: (
                "head_guard"
                if pd.notna(rank) and int(rank) <= 10
                else "boundary_11_30"
                if pd.notna(rank) and int(rank) <= 30
                else "rescue_31_60"
                if pd.notna(rank) and int(rank) <= 60
                else "rescue_61_100"
                if pd.notna(rank) and int(rank) <= 100
                else "outside_top100"
            )
        )
        chosen_pairs = boundary_users.merge(
            boundary_pair_assets,
            left_on=["bucket", "user_idx", "user_id", "true_item_idx"],
            right_on=["bucket", "user_idx", "user_id", "item_idx"],
            how="left",
        )
        rivals = boundary_users[["bucket", "user_idx", "user_id", "true_item_idx"]].merge(
            boundary_pair_universe[["bucket", "user_idx", "user_id", "item_idx", "learned_rank_band"]],
            on=["bucket", "user_idx", "user_id"],
            how="left",
        )
        rivals = rivals.loc[rivals["item_idx"] != rivals["true_item_idx"]].copy()
        rival_stats = (
            rivals.groupby(["bucket", "user_idx", "user_id"], dropna=False)
            .agg(
                rival_total=("item_idx", "count"),
                rival_head=("learned_rank_band", lambda s: int((s == "head_guard").sum())),
                rival_boundary=("learned_rank_band", lambda s: int((s == "boundary_11_30").sum())),
            )
            .reset_index()
        )
        rival_stats["boundary_rival_head_or_boundary_v1"] = (
            pd.to_numeric(rival_stats["rival_head"], errors="coerce").fillna(0).astype(int)
            + pd.to_numeric(rival_stats["rival_boundary"], errors="coerce").fillna(0).astype(int)
        )
        rival_stats["boundary_rival_total_v1"] = pd.to_numeric(
            rival_stats["rival_total"], errors="coerce"
        ).fillna(0).astype(int)
        chosen_pairs = chosen_pairs.merge(
            rival_stats[
                [
                    "bucket",
                    "user_idx",
                    "user_id",
                    "boundary_rival_total_v1",
                    "boundary_rival_head_or_boundary_v1",
                ]
            ],
            on=["bucket", "user_idx", "user_id"],
            how="left",
        )
        chosen_pairs["boundary_rival_total_v1"] = pd.to_numeric(
            chosen_pairs.get("boundary_rival_total_v1"), errors="coerce"
        ).fillna(0).astype(int)
        chosen_pairs["boundary_rival_head_or_boundary_v1"] = pd.to_numeric(
            chosen_pairs.get("boundary_rival_head_or_boundary_v1"), errors="coerce"
        ).fillna(0).astype(int)
        constructability = chosen_pairs.apply(
            lambda row: classify_boundary_constructability(
                row.to_dict(),
                rival_total=row.get("boundary_rival_total_v1", 0),
                rival_head_or_boundary=row.get("boundary_rival_head_or_boundary_v1", 0),
            ),
            axis=1,
            result_type="expand",
        )
        chosen_pairs["boundary_constructability_class_v1"] = constructability[0]
        chosen_pairs["boundary_constructability_reason_codes_v1"] = constructability[1].map(
            lambda codes: json.dumps(list(codes or []), ensure_ascii=False)
        )
        chosen_pairs["boundary_prompt_ready_v1"] = chosen_pairs["boundary_constructability_class_v1"].isin(
            ["C2_USABLE", "C3_IDEAL"]
        ).astype(int)
        boundary_annotations = chosen_pairs[
            [
                "bucket",
                "user_idx",
                "user_id",
                "item_idx",
                "boundary_constructability_class_v1",
                "boundary_constructability_reason_codes_v1",
                "boundary_prompt_ready_v1",
                "boundary_rival_total_v1",
                "boundary_rival_head_or_boundary_v1",
            ]
        ].drop_duplicates(subset=["bucket", "user_idx", "user_id", "item_idx"], keep="first")

    users = users.merge(
        boundary_annotations[
            [
                "bucket",
                "user_idx",
                "user_id",
                "boundary_constructability_class_v1",
                "boundary_constructability_reason_codes_v1",
                "boundary_prompt_ready_v1",
                "boundary_rival_total_v1",
                "boundary_rival_head_or_boundary_v1",
            ]
        ].drop_duplicates(subset=["bucket", "user_idx", "user_id"], keep="first"),
        on=["bucket", "user_idx", "user_id"],
        how="left",
    )
    if "boundary_constructability_class_v1" not in users.columns:
        users["boundary_constructability_class_v1"] = ""
    users["boundary_constructability_class_v1"] = users["boundary_constructability_class_v1"].fillna("").astype(str)
    if "boundary_constructability_reason_codes_v1" not in users.columns:
        users["boundary_constructability_reason_codes_v1"] = "[]"
    users["boundary_constructability_reason_codes_v1"] = users["boundary_constructability_reason_codes_v1"].fillna("[]").astype(str)
    users["boundary_prompt_ready_v1"] = pd.to_numeric(users.get("boundary_prompt_ready_v1"), errors="coerce").fillna(0).astype(int)
    users["boundary_rival_total_v1"] = pd.to_numeric(users.get("boundary_rival_total_v1"), errors="coerce").fillna(0).astype(int)
    users["boundary_rival_head_or_boundary_v1"] = pd.to_numeric(
        users.get("boundary_rival_head_or_boundary_v1"), errors="coerce"
    ).fillna(0).astype(int)

    semantic_user_tier: list[str] = []
    for _, row in users.iterrows():
        profile_tier = str(row.get("user_profile_richness_tier_v2", "") or "")
        quality_signal_count = int(row.get("user_quality_signal_count_v2", 0) or 0)
        has_avoid = int(row.get("user_has_avoid_signal_v2", 0) or 0) > 0
        has_clean_evidence = int(row.get("user_has_clean_evidence_v2", 0) or 0) > 0
        has_specific_focus = int(row.get("user_has_specific_focus_v2", 0) or 0) > 0
        has_specific_recent = int(row.get("user_has_specific_recent_v2", 0) or 0) > 0
        has_recent_text = int(row.get("user_has_recent_text_v2", 0) or 0) > 0
        has_strong_clean_evidence = int(row.get("user_has_strong_clean_evidence_v2", 0) or 0) > 0
        has_history_evidence = int(row.get("user_has_history_evidence_v2", 0) or 0) > 0
        non_low = int(row.get("pair_non_low_signal_count", 0) or 0)
        full_cnt = int(row.get("pair_full_count", 0) or 0)
        partial_cnt = int(row.get("pair_partial_count", 0) or 0)
        blank_fit_cnt = int(row.get("pair_blank_fit_count", 0) or 0)
        concrete_fit_cnt = int(row.get("pair_concrete_fit_count", 0) or 0)
        timing_only_fit_cnt = int(row.get("pair_timing_only_fit_count", 0) or 0)
        fit_fact_cnt = int(row.get("pair_fit_fact_count_sum", 0) or 0)
        friction_fact_cnt = int(row.get("pair_friction_fact_count_sum", 0) or 0)
        evidence_fact_cnt = int(row.get("pair_evidence_fact_count_sum", 0) or 0)
        recent_fit_fact_cnt = int(row.get("pair_recent_fit_fact_count_sum", 0) or 0)
        history_fit_fact_cnt = int(row.get("pair_history_fit_fact_count_sum", 0) or 0)
        practical_fit_fact_cnt = int(row.get("pair_practical_fit_fact_count_sum", 0) or 0)
        rows_with_fit_facts = int(row.get("pair_rows_with_fit_facts", 0) or 0)
        rows_with_conflict = int(row.get("pair_rows_with_conflict", 0) or 0)
        rows_with_evidence = int(row.get("pair_rows_with_evidence", 0) or 0)
        rows_with_recent_fit = int(row.get("pair_rows_with_recent_fit", 0) or 0)
        rows_with_history_fit = int(row.get("pair_rows_with_history_fit", 0) or 0)
        rows_with_practical_fit = int(row.get("pair_rows_with_practical_fit", 0) or 0)
        rows_with_visible_user_fit = int(row.get("pair_rows_with_visible_user_fit", 0) or 0)
        rows_with_detail_support = int(row.get("pair_rows_with_detail_support", 0) or 0)
        rows_with_multisource_fit = int(row.get("pair_rows_with_multisource_fit", 0) or 0)
        rows_with_recent_context_bridge = int(row.get("pair_rows_with_recent_context_bridge", 0) or 0)
        rows_with_candidate_fit_support = int(row.get("pair_rows_with_candidate_fit_support", 0) or 0)
        rows_with_candidate_strong_fit_support = int(row.get("pair_rows_with_candidate_strong_fit_support", 0) or 0)
        rows_with_contrastive_support = int(row.get("pair_rows_with_contrastive_support", 0) or 0)
        pair_rows_top100 = int(row.get("pair_rows_top100", 0) or 0)
        richness_max = int(row.get("pair_alignment_richness_max", 0) or 0)
        richness_mean = float(row.get("pair_alignment_richness_mean", 0.0) or 0.0)
        blank_fit_rate = blank_fit_cnt / max(pair_rows_top100, 1)
        timing_only_fit_rate = timing_only_fit_cnt / max(pair_rows_top100, 1)
        non_low_rate = non_low / max(pair_rows_top100, 1)
        fit_row_rate = rows_with_fit_facts / max(pair_rows_top100, 1)
        evidence_row_rate = rows_with_evidence / max(pair_rows_top100, 1)
        non_low_detail_cnt = full_cnt + partial_cnt
        has_user_side_detail = has_avoid or has_history_evidence
        has_specific_pair_fit = (
            recent_fit_fact_cnt > 0
            or history_fit_fact_cnt > 0
            or practical_fit_fact_cnt > 0
            or friction_fact_cnt > 0
        )
        if profile_tier == "LOW_SIGNAL" or non_low <= 0 or fit_fact_cnt <= 0 or rows_with_fit_facts <= 0:
            semantic_user_tier.append("LOW_SIGNAL")
        elif (
            profile_tier == "FULL"
            and has_specific_focus
            and has_specific_recent
            and has_recent_text
            and (has_strong_clean_evidence or has_history_evidence)
            and quality_signal_count >= 4
            and has_user_side_detail
            and non_low >= 18
            and non_low_detail_cnt >= 14
            and full_cnt >= 6
            and rows_with_visible_user_fit >= 6
            and (rows_with_multisource_fit >= 4 or rows_with_contrastive_support >= 4)
            and has_specific_pair_fit
            and (rows_with_recent_fit >= 1 or rows_with_history_fit >= 2)
            and (rows_with_conflict >= 1 or rows_with_practical_fit >= 1 or rows_with_evidence >= 3 or rows_with_detail_support >= 3)
            and (rows_with_candidate_strong_fit_support >= 2 or rows_with_recent_context_bridge >= 2 or rows_with_contrastive_support >= 4)
            and timing_only_fit_rate <= 0.18
            and richness_max >= 2
            and richness_mean >= 1.6
            and fit_row_rate >= 0.24
            and evidence_row_rate >= 0.18
            and blank_fit_rate <= 0.60
        ):
            semantic_user_tier.append("FULL")
        elif (
            profile_tier in {"FULL", "PARTIAL"}
            and has_specific_focus
            and has_strong_clean_evidence
            and ((has_recent_text and has_specific_recent) or has_avoid or has_history_evidence)
            and has_user_side_detail
            and quality_signal_count >= 3
            and non_low >= 8
            and non_low_rate >= 0.08
            and non_low_detail_cnt >= 5
            and rows_with_visible_user_fit >= 2
            and (rows_with_multisource_fit >= 1 or rows_with_contrastive_support >= 1)
            and has_specific_pair_fit
            and (
                rows_with_recent_fit >= 1
                or rows_with_practical_fit >= 1
                or rows_with_conflict >= 1
                or rows_with_multisource_fit >= 2
                or rows_with_contrastive_support >= 1
            )
            and (rows_with_candidate_fit_support >= 1 or rows_with_recent_context_bridge >= 1 or rows_with_contrastive_support >= 1)
            and timing_only_fit_rate <= 0.35
            and richness_max >= 2
            and richness_mean >= 1.1
            and fit_row_rate >= 0.12
            and blank_fit_rate <= 0.82
        ):
            semantic_user_tier.append("PARTIAL")
        else:
            semantic_user_tier.append("LOW_SIGNAL")
    users["semantic_user_readiness_tier_v1"] = semantic_user_tier
    boundary_truth_mask = pd.to_numeric(users.get("truth_learned_rank"), errors="coerce").between(11, 30)
    users.loc[
        boundary_truth_mask
        & users["boundary_prompt_ready_v1"].gt(0)
        & users["semantic_user_readiness_tier_v1"].eq("LOW_SIGNAL")
        & users["boundary_constructability_class_v1"].eq("C3_IDEAL"),
        "semantic_user_readiness_tier_v1",
    ] = "FULL"
    users.loc[
        boundary_truth_mask
        & users["boundary_prompt_ready_v1"].gt(0)
        & users["semantic_user_readiness_tier_v1"].eq("LOW_SIGNAL")
        & users["boundary_constructability_class_v1"].isin(["C2_USABLE", "C3_IDEAL"]),
        "semantic_user_readiness_tier_v1",
    ] = "PARTIAL"
    users.loc[boundary_truth_mask & users["boundary_prompt_ready_v1"].le(0), "semantic_user_readiness_tier_v1"] = "LOW_SIGNAL"

    filtered_users = users.loc[
        users["semantic_user_readiness_tier_v1"].isin(["FULL", "PARTIAL"])
    ][
        [
            "bucket",
            "user_idx",
            "user_id",
            "truth_learned_rank",
            "boundary_constructability_class_v1",
            "boundary_constructability_reason_codes_v1",
            "boundary_prompt_ready_v1",
            "boundary_rival_total_v1",
            "boundary_rival_head_or_boundary_v1",
        ]
    ].drop_duplicates()
    return users, filtered_users, boundary_annotations


def _iter_record_batches(df: pd.DataFrame, chunk_rows: int) -> Any:
    total = int(len(df))
    for start in range(0, total, max(1, int(chunk_rows))):
        stop = min(start + max(1, int(chunk_rows)), total)
        yield df.iloc[start:stop].to_dict(orient="records")


def _build_user_profile_chunk(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_user_profile_texts(pd.Series(record)) for record in records]


def _build_merchant_profile_chunk(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_merchant_profile_texts(pd.Series(record)) for record in records]


def _build_pair_alignment_chunk(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        row = pd.Series(record)
        fit_facts = _collect_fit_reason_facts(row)
        friction_facts = _collect_friction_reason_facts(row)
        fit_reasons = [str(fact.get("text", "") or "") for fact in fit_facts]
        friction_reasons = [str(fact.get("text", "") or "") for fact in friction_facts]
        user_basis = build_user_basis_text(row, max_chars=380)
        merchant_basis = build_merchant_basis_text(row, max_chars=360)
        evidence_basis = build_balanced_evidence_basis(user_basis, merchant_basis, total_max_chars=760)
        evidence_fact = build_semantic_fact(
            scope="evidence",
            polarity="mixed",
            text=evidence_basis,
            facet_type="user_business_evidence",
            evidence_source="balanced_basis",
            max_chars=760,
        )
        fit_text = join_narrative_blocks_with_budget(fit_reasons, max_chars=520)
        friction_text = join_narrative_blocks_with_budget(friction_reasons, max_chars=320)
        evidence_text = join_narrative_blocks_with_budget([evidence_basis], max_chars=700)
        fit_is_concrete = int(bool(fit_facts))
        fit_is_timing_only = int(is_timing_only_fit_text(fit_text))
        richness = int(bool(fit_text)) + int(bool(friction_text)) + int(bool(evidence_text))
        pair_fit_fact_count = len(fit_facts)
        pair_friction_fact_count = len(friction_facts)
        pair_evidence_fact_count = int(bool(evidence_fact))
        pair_stable_fit_fact_count = sum(1 for fact in fit_facts if fact.get("scope") == "stable")
        pair_recent_fit_fact_count = sum(1 for fact in fit_facts if fact.get("scope") in {"recent", "recent_context"})
        pair_history_fit_fact_count = sum(1 for fact in fit_facts if fact.get("scope") == "history")
        pair_practical_fit_fact_count = sum(1 for fact in fit_facts if fact.get("scope") in {"practical", "evidence"})
        candidate_pref_support = int(finite_or_zero(row.get("sim_long_pref_core", 0.0)) >= 0.18)
        candidate_recent_support = int(
            finite_or_zero(row.get("sim_recent_intent_semantic", 0.0)) >= 0.12
            or finite_or_zero(row.get("sim_recent_intent_pos", 0.0)) >= 0.08
        )
        candidate_context_support = int(finite_or_zero(row.get("sim_context_merchant", 0.0)) >= 0.14)
        pair_candidate_fit_signal_count = int(candidate_pref_support + candidate_recent_support + candidate_context_support)
        pair_fit_scope_count = int(
            sum(
                1
                for count in [
                    pair_stable_fit_fact_count,
                    pair_recent_fit_fact_count,
                    pair_history_fit_fact_count,
                    pair_practical_fit_fact_count,
                ]
                if count > 0
            )
        )
        pair_has_visible_user_fit = int(pair_fit_fact_count > 0)
        pair_has_visible_conflict = int(pair_friction_fact_count > 0)
        pair_has_recent_context_visible_bridge = int(
            any(str(fact.get("scope", "") or "") == "recent_context" for fact in fit_facts)
        )
        pair_has_detail_support = int(
            pair_friction_fact_count > 0 or pair_recent_fit_fact_count > 0 or pair_practical_fit_fact_count > 0
        )
        pair_has_multisource_fit = int(
            pair_fit_scope_count >= 2
            or (
                pair_fit_fact_count > 0
                and (
                    pair_recent_fit_fact_count > 0
                    or pair_history_fit_fact_count > 0
                    or pair_practical_fit_fact_count > 0
                )
            )
        )
        pair_has_candidate_fit_support = int(pair_candidate_fit_signal_count >= 1)
        pair_has_candidate_strong_fit_support = int(pair_candidate_fit_signal_count >= 2)
        pair_has_contrastive_support = int(
            pair_has_visible_user_fit > 0
            and pair_has_visible_conflict > 0
            and pair_evidence_fact_count > 0
            and (
                pair_has_multisource_fit > 0
                or pair_has_candidate_strong_fit_support > 0
                or pair_has_recent_context_visible_bridge > 0
            )
        )
        pair_fact_signal_count = int(
            pair_has_visible_user_fit
            + pair_has_visible_conflict
            + int(pair_evidence_fact_count > 0)
            + int(pair_recent_fit_fact_count > 0)
            + int(pair_history_fit_fact_count > 0)
        )
        tier = "LOW_SIGNAL"
        if (
            int(row.get("user_profile_richness_v2", 0) or 0) >= 3
            and int(row.get("merchant_profile_richness_v2", 0) or 0) >= 3
            and pair_fit_fact_count >= 2
            and pair_fact_signal_count >= 4
            and (pair_has_multisource_fit > 0 or pair_has_contrastive_support > 0)
            and (pair_recent_fit_fact_count > 0 or pair_history_fit_fact_count > 0 or pair_practical_fit_fact_count > 0)
            and pair_evidence_fact_count > 0
            and fit_is_timing_only == 0
        ):
            tier = "FULL"
        elif (
            int(row.get("user_profile_richness_v2", 0) or 0) >= 2
            and int(row.get("merchant_profile_richness_v2", 0) or 0) >= 2
            and pair_fit_fact_count >= 1
            and pair_fact_signal_count >= 3
            and (pair_has_multisource_fit > 0 or pair_has_contrastive_support > 0)
            and (
                pair_recent_fit_fact_count > 0
                or pair_practical_fit_fact_count > 0
                or pair_friction_fact_count > 0
                or pair_fit_scope_count >= 2
                or pair_has_contrastive_support > 0
            )
            and fit_is_timing_only == 0
        ):
            tier = "PARTIAL"
        out.append(
            {
                "bucket": int(record.get("bucket", 0) or 0),
                "user_idx": int(record.get("user_idx", 0) or 0),
                "item_idx": int(record.get("item_idx", 0) or 0),
                "user_id": str(record.get("user_id", "") or ""),
                "business_id": str(record.get("business_id", "") or ""),
                "fit_reasons_text_v1": fit_text,
                "friction_reasons_text_v1": friction_text,
                "evidence_basis_text_v1": evidence_text,
                "pair_alignment_facts_v1": semantic_facts_json(*fit_facts, *friction_facts, evidence_fact),
                "pair_has_concrete_fit_v1": int(fit_is_concrete),
                "pair_has_timing_only_fit_v1": int(fit_is_timing_only),
                "pair_alignment_richness_v1": int(richness),
                "pair_fit_fact_count_v1": int(pair_fit_fact_count),
                "pair_friction_fact_count_v1": int(pair_friction_fact_count),
                "pair_evidence_fact_count_v1": int(pair_evidence_fact_count),
                "pair_stable_fit_fact_count_v1": int(pair_stable_fit_fact_count),
                "pair_recent_fit_fact_count_v1": int(pair_recent_fit_fact_count),
                "pair_history_fit_fact_count_v1": int(pair_history_fit_fact_count),
                "pair_practical_fit_fact_count_v1": int(pair_practical_fit_fact_count),
                "pair_fit_scope_count_v1": int(pair_fit_scope_count),
                "pair_has_visible_user_fit_v1": int(pair_has_visible_user_fit),
                "pair_has_visible_conflict_v1": int(pair_has_visible_conflict),
                "pair_has_recent_context_visible_bridge_v1": int(pair_has_recent_context_visible_bridge),
                "pair_has_detail_support_v1": int(pair_has_detail_support),
                "pair_has_multisource_fit_v1": int(pair_has_multisource_fit),
                "pair_has_candidate_fit_support_v1": int(pair_has_candidate_fit_support),
                "pair_has_candidate_strong_fit_support_v1": int(pair_has_candidate_strong_fit_support),
                "pair_has_contrastive_support_v1": int(pair_has_contrastive_support),
                "pair_fact_signal_count_v1": int(pair_fact_signal_count),
                "semantic_prompt_readiness_tier_v1": tier,
            }
        )
    return out


def _parallel_record_payloads(
    df: pd.DataFrame,
    *,
    worker: Any,
    threads: int,
    chunk_rows: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    total = int(len(df))
    if total <= 0:
        return []
    if threads <= 1:
        payloads: list[dict[str, Any]] = []
        processed = 0
        for batch in _iter_record_batches(df, chunk_rows):
            payloads.extend(worker(batch))
            processed += len(batch)
            print(f"[STAGE09-SEM] {progress_label} {processed}/{total}", flush=True)
        return payloads

    payloads: list[dict[str, Any]] = []
    processed = 0
    max_workers = max(1, int(threads))
    # Keep the process pool saturated on small/medium tables instead of
    # cutting the dataframe into only a couple of oversized chunks.
    target_batches = max(max_workers * 4, max_workers + 1)
    effective_chunk_rows = max(200, int(math.ceil(float(total) / float(target_batches))))
    effective_chunk_rows = min(int(chunk_rows), effective_chunk_rows)
    print(
        f"[STAGE09-SEM] {progress_label} using workers={max_workers} chunk_rows={effective_chunk_rows}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_payloads in executor.map(worker, _iter_record_batches(df, effective_chunk_rows), chunksize=1):
            payloads.extend(batch_payloads)
            processed += len(batch_payloads)
            print(f"[STAGE09-SEM] {progress_label} {processed}/{total}", flush=True)
    return payloads


def _parallel_build_pair_alignment_assets(pair_df: pd.DataFrame) -> pd.DataFrame:
    payloads = _parallel_record_payloads(
        pair_df,
        worker=_build_pair_alignment_chunk,
        threads=STAGE09_SEM_THREADS,
        chunk_rows=STAGE09_SEM_CHUNK_ROWS,
        progress_label="pair alignment chunk",
    )
    if not payloads:
        return pair_df[["bucket", "user_idx", "item_idx", "user_id", "business_id"]].iloc[0:0].copy()
    return pd.DataFrame(payloads)


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
        )
        truth_rank = truth_rank.rename(columns={"learned_rank": "truth_learned_rank"})
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
        raise RuntimeError("no bucket rows produced for stage11 semantic assets")
    universe = pd.concat(all_rows, ignore_index=True)
    universe["user_id"] = universe["user_id"].astype(str)
    universe["business_id"] = universe["business_id"].astype(str)
    return universe, bucket_stats


def main() -> None:
    stage10_run = resolve_stage10_run()
    source_stage09_run = resolve_source_stage09_run(stage10_run)
    buckets = parse_bucket_override(BUCKETS_OVERRIDE)
    if not buckets:
        raise ValueError("BUCKETS_OVERRIDE resolved to empty")

    profile_run = resolve_optional_run(
        INPUT_09_USER_PROFILE_TEXT_RUN_DIR,
        INPUT_09_USER_PROFILE_TEXT_ROOT,
        "_full_stage09_user_profile_build",
    )
    user_text_run = resolve_optional_run(
        INPUT_09_USER_TEXT_VIEWS_RUN_DIR,
        INPUT_09_USER_TEXT_VIEWS_ROOT,
        "_full_stage09_candidate_wise_text_views_v1_build",
    )
    user_schema_run = resolve_optional_run(
        INPUT_09_USER_SCHEMA_RUN_DIR,
        INPUT_09_USER_SCHEMA_ROOT,
        "_full_stage09_user_schema_projection_v2_build",
    )
    user_intent_v3_run = resolve_optional_run(
        INPUT_09_USER_INTENT_V3_RUN_DIR,
        INPUT_09_USER_INTENT_V3_ROOT,
        "_full_stage09_user_intent_profile_v3_all_bucket_users_build",
    )
    merchant_card_run = resolve_optional_run(
        INPUT_09_MERCHANT_CARD_RUN_DIR,
        INPUT_09_MERCHANT_CARD_ROOT,
        "_full_stage09_merchant_semantic_card_build",
    )
    merchant_text_run = resolve_optional_run(
        INPUT_09_MERCHANT_TEXT_V3_RUN_DIR,
        INPUT_09_MERCHANT_TEXT_V3_ROOT,
        "_full_stage09_merchant_text_views_v3_build",
    )
    merchant_structured_text_run = resolve_optional_run(
        INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_RUN_DIR,
        INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_ROOT,
        "_full_stage09_merchant_structured_text_views_v2_build",
    )
    match_feature_run = resolve_optional_run(
        INPUT_09_MATCH_FEATURE_RUN_DIR,
        INPUT_09_MATCH_FEATURE_ROOT,
        "_full_stage09_user_business_match_split_aware_v2_build",
    )
    match_channel_run = resolve_optional_run(
        INPUT_09_MATCH_CHANNEL_RUN_DIR,
        INPUT_09_MATCH_CHANNEL_ROOT,
        "_full_stage09_user_business_match_split_aware_v2_build",
    )
    candidate_text_match_run = resolve_optional_run(
        INPUT_09_CANDIDATE_TEXT_MATCH_RUN_DIR,
        INPUT_09_CANDIDATE_TEXT_MATCH_ROOT,
        "_full_stage09_candidate_wise_text_match_features_v1_build",
    )
    source_semantic_materials_run = resolve_optional_run(
        INPUT_09_SOURCE_SEMANTIC_MATERIALS_V1_RUN_DIR,
        INPUT_09_SOURCE_SEMANTIC_MATERIALS_V1_ROOT,
        "_full_stage09_stage11_source_semantic_materials_v1_build",
    )

    out_dir = OUTPUT_ROOT / now_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[STAGE09-SEM] load truth 11-100 pair universe", flush=True)
    pair_universe, bucket_stats = load_truth_11_100_pair_universe(stage10_run, source_stage09_run, buckets)
    print(
        f"[STAGE09-SEM] pair universe ready rows={len(pair_universe)} users={pair_universe['user_idx'].nunique()}",
        flush=True,
    )

    print("[STAGE09-SEM] load upstream assets", flush=True)
    profile_df = pd.read_csv(profile_run / "user_profile_multi_vector_texts.csv", dtype=str, keep_default_na=False)
    user_text_df = pd.read_parquet(user_text_run / "user_text_views_v1.parquet")
    user_schema_df = pd.read_parquet(user_schema_run / "user_schema_profile_v1.parquet")
    user_intent_v3_df = pd.read_parquet(user_intent_v3_run / "user_intent_profile_v2.parquet")
    merchant_card_df = pd.read_parquet(merchant_card_run / "merchant_semantic_card_v2.parquet")
    merchant_text_df = pd.read_parquet(merchant_text_run / "merchant_text_views_v3.parquet")
    merchant_structured_text_df = pd.read_parquet(
        merchant_structured_text_run / "merchant_structured_text_views_v2.parquet"
    )
    match_feature_df = pd.read_parquet(match_feature_run / "user_business_match_features_v2_user_item.parquet")
    match_channel_df = pd.read_parquet(match_channel_run / "user_business_match_channels_v2_user_item.parquet")
    candidate_text_match_df = pd.read_parquet(
        candidate_text_match_run / "candidate_text_match_features_v1.parquet",
        columns=[
            "user_idx",
            "item_idx",
            "business_id",
            "sim_long_pref_core",
            "sim_recent_intent_semantic",
            "sim_recent_intent_pos",
            "sim_negative_avoid_neg",
            "sim_negative_avoid_core",
            "sim_context_merchant",
            "sim_conflict_gap",
        ],
    )
    user_source_materials_df = pd.read_parquet(source_semantic_materials_run / "user_source_semantic_materials_v1.parquet")
    merchant_source_materials_df = pd.read_parquet(source_semantic_materials_run / "merchant_source_semantic_materials_v1.parquet")
    print("[STAGE09-SEM] upstream assets loaded", flush=True)

    profile_df["user_id"] = profile_df["user_id"].astype(str)
    user_text_df["user_id"] = user_text_df["user_id"].astype(str)
    user_schema_df["user_id"] = user_schema_df["user_id"].astype(str)
    user_intent_v3_df["user_id"] = user_intent_v3_df["user_id"].astype(str)
    user_source_materials_df["user_id"] = user_source_materials_df["user_id"].astype(str)
    merchant_card_df["business_id"] = merchant_card_df["business_id"].astype(str)
    merchant_text_df["business_id"] = merchant_text_df["business_id"].astype(str)
    merchant_structured_text_df["business_id"] = merchant_structured_text_df["business_id"].astype(str)
    merchant_source_materials_df["business_id"] = merchant_source_materials_df["business_id"].astype(str)
    match_feature_df["user_id"] = match_feature_df["user_id"].astype(str)
    match_feature_df["business_id"] = match_feature_df["business_id"].astype(str)
    match_channel_df["user_id"] = match_channel_df["user_id"].astype(str)
    match_channel_df["business_id"] = match_channel_df["business_id"].astype(str)
    candidate_text_match_df["business_id"] = candidate_text_match_df["business_id"].astype(str)

    user_base = pair_universe[["bucket", "user_idx", "user_id"]].drop_duplicates().merge(profile_df, on="user_id", how="left")
    user_base = user_base.merge(user_text_df, on="user_id", how="left")
    user_base = user_base.merge(user_schema_df, on="user_id", how="left")
    user_base = user_base.merge(user_intent_v3_df, on=["user_idx", "user_id"], how="left")
    user_base = user_base.merge(
        user_source_materials_df,
        on=["bucket", "user_idx", "user_id"],
        how="left",
    )
    print(f"[STAGE09-SEM] build user semantic assets rows={len(user_base)}", flush=True)
    user_rows = _parallel_record_payloads(
        user_base,
        worker=_build_user_profile_chunk,
        threads=STAGE09_SEM_THREADS,
        chunk_rows=STAGE09_SEM_CHUNK_ROWS,
        progress_label="user semantic chunk",
    )
    user_assets = pd.concat([user_base[["bucket", "user_idx", "user_id"]].reset_index(drop=True), pd.DataFrame(user_rows)], axis=1)
    user_assets = user_assets.drop_duplicates(subset=["bucket", "user_idx", "user_id"], keep="first")
    print("[STAGE09-SEM] user semantic assets ready", flush=True)

    merchant_base = (
        pair_universe[["bucket", "business_id"]]
        .drop_duplicates()
        .merge(merchant_card_df, on="business_id", how="left")
        .merge(merchant_text_df, on="business_id", how="left")
        .merge(merchant_structured_text_df, on="business_id", how="left")
        .merge(merchant_source_materials_df, on=["bucket", "business_id"], how="left")
    )
    print(f"[STAGE09-SEM] build merchant semantic assets rows={len(merchant_base)}", flush=True)
    merchant_rows = _parallel_record_payloads(
        merchant_base,
        worker=_build_merchant_profile_chunk,
        threads=STAGE09_SEM_THREADS,
        chunk_rows=STAGE09_SEM_CHUNK_ROWS,
        progress_label="merchant semantic chunk",
    )
    merchant_assets = pd.concat([merchant_base[["bucket", "business_id"]].reset_index(drop=True), pd.DataFrame(merchant_rows)], axis=1)
    merchant_assets = merchant_assets.drop_duplicates(subset=["bucket", "business_id"], keep="first")
    print("[STAGE09-SEM] merchant semantic assets ready", flush=True)

    pair_base = pair_universe.merge(user_assets, on=["bucket", "user_idx", "user_id"], how="left")
    pair_base = pair_base.merge(merchant_assets, on=["bucket", "business_id"], how="left")
    pair_base = pair_base.merge(candidate_text_match_df, on=["user_idx", "item_idx", "business_id"], how="left")
    pair_base = pair_base.merge(match_feature_df, on=["user_id", "business_id"], how="left")
    pair_base = pair_base.merge(match_channel_df, on=["user_id", "business_id"], how="left", suffixes=("", "_channel"))
    print(f"[STAGE09-SEM] build pair alignment assets rows={len(pair_base)}", flush=True)
    pair_assets = build_pair_alignment_assets(pair_base)
    print("[STAGE09-SEM] pair alignment assets ready", flush=True)
    target_users_df = pair_universe[["bucket", "user_idx", "user_id", "truth_learned_rank", "true_item_idx"]].drop_duplicates()
    print("[STAGE09-SEM] build filtered target user assets", flush=True)
    target_user_readiness_df, filtered_target_users_df, boundary_annotations_df = build_stage11_target_users_filtered_assets(
        target_users_df,
        user_assets,
        pair_assets,
        pair_universe,
    )
    pair_assets = pair_assets.merge(
        boundary_annotations_df,
        on=["bucket", "user_idx", "user_id", "item_idx"],
        how="left",
    )
    if "boundary_constructability_class_v1" not in pair_assets.columns:
        pair_assets["boundary_constructability_class_v1"] = ""
    pair_assets["boundary_constructability_class_v1"] = pair_assets["boundary_constructability_class_v1"].fillna("").astype(str)
    if "boundary_constructability_reason_codes_v1" not in pair_assets.columns:
        pair_assets["boundary_constructability_reason_codes_v1"] = "[]"
    pair_assets["boundary_constructability_reason_codes_v1"] = pair_assets["boundary_constructability_reason_codes_v1"].fillna("[]").astype(str)
    pair_assets["boundary_prompt_ready_v1"] = pd.to_numeric(pair_assets.get("boundary_prompt_ready_v1"), errors="coerce").fillna(0).astype(int)
    pair_assets["boundary_rival_total_v1"] = pd.to_numeric(pair_assets.get("boundary_rival_total_v1"), errors="coerce").fillna(0).astype(int)
    pair_assets["boundary_rival_head_or_boundary_v1"] = pd.to_numeric(
        pair_assets.get("boundary_rival_head_or_boundary_v1"), errors="coerce"
    ).fillna(0).astype(int)
    print(
        f"[STAGE09-SEM] filtered target users ready surviving={len(filtered_target_users_df)} "
        f"of total={len(target_users_df)}",
        flush=True,
    )
    filtered_pair_assets_df = pair_assets.merge(
        filtered_target_users_df[["bucket", "user_idx", "user_id"]],
        on=["bucket", "user_idx", "user_id"],
        how="inner",
    )
    filtered_pair_assets_df = filtered_pair_assets_df.loc[
        filtered_pair_assets_df["semantic_prompt_readiness_tier_v1"].isin(["FULL", "PARTIAL"])
    ].drop_duplicates(subset=["bucket", "user_idx", "item_idx"], keep="first")
    print(
        f"[STAGE09-SEM] filtered pair assets ready rows={len(filtered_pair_assets_df)} "
        f"users={filtered_pair_assets_df['user_idx'].nunique() if not filtered_pair_assets_df.empty else 0}",
        flush=True,
    )

    user_assets_path = out_dir / "user_semantic_profile_text_v2.parquet"
    merchant_assets_path = out_dir / "merchant_semantic_profile_text_v2.parquet"
    pair_assets_path = out_dir / "user_business_alignment_text_v1.parquet"
    target_pairs_path = out_dir / "stage11_target_pair_universe_v1.parquet"
    target_users_path = out_dir / "stage11_target_users_v1.parquet"
    target_user_readiness_path = out_dir / "stage11_target_user_readiness_v1.parquet"
    filtered_target_users_path = out_dir / "stage11_target_users_filtered_v1.parquet"
    filtered_pair_assets_path = out_dir / "stage11_target_pair_assets_filtered_v1.parquet"

    print("[STAGE09-SEM] write parquet outputs", flush=True)
    user_assets.to_parquet(user_assets_path, index=False)
    merchant_assets.to_parquet(merchant_assets_path, index=False)
    pair_assets.to_parquet(pair_assets_path, index=False)
    pair_universe.to_parquet(target_pairs_path, index=False)
    target_users_df.to_parquet(target_users_path, index=False)
    target_user_readiness_df.to_parquet(target_user_readiness_path, index=False)
    filtered_target_users_df.to_parquet(filtered_target_users_path, index=False)
    filtered_pair_assets_df.to_parquet(filtered_pair_assets_path, index=False)

    audit = {
        "buckets": bucket_stats,
        "target_users_total": int(pair_universe["user_idx"].nunique()),
        "target_pairs_total": int(len(pair_universe)),
        "target_users_filtered_total": int(filtered_target_users_df["user_idx"].nunique()),
        "target_pairs_filtered_total": int(len(filtered_pair_assets_df)),
        "user_profile_richness_tier_counts": user_assets["user_profile_richness_tier_v2"].value_counts(dropna=False).to_dict(),
        "pair_prompt_readiness_tier_counts": pair_assets["semantic_prompt_readiness_tier_v1"].value_counts(dropna=False).to_dict(),
        "pair_alignment_richness_distribution": pair_assets["pair_alignment_richness_v1"].value_counts(dropna=False).sort_index().to_dict(),
        "filtered_pair_prompt_readiness_tier_counts": filtered_pair_assets_df["semantic_prompt_readiness_tier_v1"].value_counts(dropna=False).to_dict(),
        "semantic_user_readiness_tier_counts": target_user_readiness_df["semantic_user_readiness_tier_v1"].value_counts(dropna=False).to_dict(),
        "boundary_constructability_counts": target_user_readiness_df["boundary_constructability_class_v1"].value_counts(dropna=False).to_dict(),
        "boundary_prompt_ready_rate": float(
            pd.to_numeric(target_user_readiness_df.get("boundary_prompt_ready_v1"), errors="coerce").fillna(0).astype(int).gt(0).mean()
        ),
    }
    run_meta = {
        "run_tag": RUN_TAG,
        "run_dir": str(out_dir),
        "stage10_run": str(stage10_run),
        "source_stage09_run": str(source_stage09_run),
        "profile_run": str(profile_run),
        "user_text_run": str(user_text_run),
        "user_schema_run": str(user_schema_run),
        "user_intent_v3_run": str(user_intent_v3_run),
        "merchant_card_run": str(merchant_card_run),
        "merchant_text_run": str(merchant_text_run),
        "merchant_structured_text_run": str(merchant_structured_text_run),
        "match_feature_run": str(match_feature_run),
        "match_channel_run": str(match_channel_run),
        "candidate_text_match_run": str(candidate_text_match_run),
        "target_truth_rank_min": int(TARGET_TRUTH_RANK_MIN),
        "target_truth_rank_max": int(TARGET_TRUTH_RANK_MAX),
        "pairwise_pool_topn": int(PAIRWISE_POOL_TOPN),
        "target_users_total": int(pair_universe["user_idx"].nunique()),
        "target_users_filtered_total": int(filtered_target_users_df["user_idx"].nunique()),
        "target_pairs_filtered_total": int(len(filtered_pair_assets_df)),
        "target_pairs_total": int(len(pair_universe)),
        "user_semantic_profile_text_v2_parquet": str(user_assets_path),
        "merchant_semantic_profile_text_v2_parquet": str(merchant_assets_path),
        "user_business_alignment_text_v1_parquet": str(pair_assets_path),
        "target_pairs_parquet": str(target_pairs_path),
        "target_users_parquet": str(target_users_path),
        "target_user_readiness_parquet": str(target_user_readiness_path),
        "target_users_filtered_parquet": str(filtered_target_users_path),
        "target_pair_assets_filtered_parquet": str(filtered_pair_assets_path),
    }
    safe_json_write(out_dir / "audit.json", audit)
    safe_json_write(out_dir / "run_meta.json", run_meta)
    write_latest_run_pointer(
        "stage09_stage11_semantic_text_assets_v1",
        out_dir,
        extra={
            "run_tag": RUN_TAG,
            "target_users_total": int(pair_universe["user_idx"].nunique()),
            "target_pairs_total": int(len(pair_universe)),
        },
    )
    print(f"[DONE] wrote {out_dir}")


if __name__ == "__main__":
    main()
