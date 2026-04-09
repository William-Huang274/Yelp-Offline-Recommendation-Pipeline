import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# Input from step 07
SOURCE_07_ROOT = Path(r"D:/5006 BDA project/data/output/07_embedding_cluster")
SOURCE_07_RUN_DIR = ""  # optional override, e.g. r"D:/.../20260208_180929_sample"
RUN_PROFILE = "full"  # "sample" or "full"; controls source-run picking + output folder

# Output for step 08
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/08_cluster_labels")
RUN_TAG = "labels"

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
REQUEST_TIMEOUT_SEC = 180
OLLAMA_NUM_CTX = 4096
OLLAMA_NUM_PREDICT = 128
OLLAMA_THINKING = False

# Minimal run defaults
QUICK_MODE = False
MAX_CLUSTERS_QUICK = 3
TOP_KEYWORDS = 12
PROMPT_MAX_BUSINESSES = 30
CITY_TOPK = 8
CATEGORY_TOPK = 12
FORCE_NON_GEO_LABEL = True
MAX_EXISTING_LABELS_IN_PROMPT = 8
MAX_LABEL_WORDS = 6
INTERPRETIVE_LABEL_MODE = True
MIXED_CLUSTER_THRESHOLD = 0.55
MIXED_CLUSTER_FORCE_EXPLICIT = True
MIXED_CLUSTER_FORCE_MAX_DOMINANT_SHARE = 0.45
MIXED_CLUSTER_FORCE_MIN_N_BUSINESSES = 120
MIXED_CLUSTER_LABEL_TOPK = 2
INCLUDE_L2_EVIDENCE = True
NIGHTLIFE_REWRITE_MAX_DOMINANT_L2_SHARE = 0.45
NIGHTLIFE_REWRITE_MIN_SIGNAL_HITS = 2

NIGHTLIFE_SIGNAL_TERMS = {
    "nightlife",
    "bar",
    "bars",
    "cocktail",
    "cocktail bar",
    "bartender",
    "beer",
    "drinks",
    "pub",
    "sports bar",
    "music",
    "dj",
    "nightclub",
    "club",
}

NIGHTLIFE_STRONG_TERMS = {
    "nightlife",
    "cocktail",
    "cocktail bar",
    "bartender",
    "sports bar",
    "nightclub",
}

GENERIC_KEYWORD_TERMS = {
    "said",
    "told",
    "asked",
    "didn",
    "didnt",
    "minutes",
    "manager",
    "customer",
    "service",
    "location",
    "called",
    "came",
    "table",
    "staff",
    "line",
    "phone",
    "restaurant",
    "restaurants",
    "food",
    "menu",
    "store",
    "shop",
    "ordered",
}

GENERIC_CATEGORY_TERMS = {
    "restaurants",
    "restaurant",
    "food",
    "shopping",
}

VAGUE_LABEL_TERMS = {
    "restaurant",
    "restaurants",
    "cuisine",
    "food",
    "store",
    "stores",
    "shop",
    "shops",
    "mix",
    "general",
    "dining",
    "businesses",
}

CUISINE_LABEL_TERMS = {
    "mexican",
    "italian",
    "vietnamese",
    "chinese",
    "japanese",
    "sushi",
    "seafood",
    "cajun",
    "creole",
    "burger",
    "burgers",
    "pizza",
    "bakery",
    "dessert",
    "brunch",
    "breakfast",
    "coffee",
    "tea",
    "bbq",
    "mediterranean",
    "middle eastern",
    "falafel",
    "pho",
    "taco",
    "tacos",
}

MIXED_CLUSTER_SKIP_L2_LABELS = {"restaurants_general", "other_service", "uncertain"}

THEME_LABEL_TERMS = set(CUISINE_LABEL_TERMS) | set(NIGHTLIFE_SIGNAL_TERMS) | {
    "breakfast",
    "brunch",
    "coffee",
    "cafe",
    "cafes",
    "bakery",
    "dessert",
    "donut",
    "seafood",
}

# Optional: write full business assignment table with labels joined
WRITE_LABELED_ASSIGNMENTS = True
LABELED_ASSIGNMENTS_NAME = "biz_cluster_assignments_labeled.csv"

def normalize_run_profile(profile: str) -> str:
    p = str(profile).strip().lower()
    if p not in {"sample", "full"}:
        raise ValueError(f"RUN_PROFILE must be 'sample' or 'full', got: {profile!r}")
    return p


def run_name_matches_profile(run_name: str, profile: str) -> bool:
    return re.search(rf"(^|_){re.escape(profile)}($|_)", run_name.lower()) is not None


def pick_source_run_dir(profile: str) -> Path:
    env_override = os.environ.get("SOURCE_07_RUN_DIR", "").strip()
    if env_override:
        run_dir = Path(env_override)
        if not run_dir.exists():
            raise FileNotFoundError(
                f"07 run dir not found from env SOURCE_07_RUN_DIR: {run_dir}"
            )
        return run_dir
    if SOURCE_07_RUN_DIR.strip():
        run_dir = Path(SOURCE_07_RUN_DIR.strip())
        if not run_dir.exists():
            raise FileNotFoundError(f"07 run dir not found: {run_dir}")
        return run_dir

    runs = [p for p in SOURCE_07_ROOT.iterdir() if p.is_dir() and p.name != "_cache"]
    if not runs:
        raise FileNotFoundError(f"No 07 output run found under: {SOURCE_07_ROOT}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    profile_runs = [p for p in runs if run_name_matches_profile(p.name, profile)]
    if not profile_runs:
        raise FileNotFoundError(
            f"No 07 '{profile}' run found under: {SOURCE_07_ROOT}. "
            "Set SOURCE_07_RUN_DIR to override."
        )
    return profile_runs[0]


def require_files(run_dir: Path) -> dict[str, Path]:
    files = {
        "summary": run_dir / "biz_cluster_summary.csv",
        "keywords": run_dir / "biz_cluster_keywords.csv",
        "examples": run_dir / "biz_cluster_examples.csv",
        "assignments": run_dir / "biz_cluster_assignments.csv",
        "meta": run_dir / "run_meta.csv",
    }
    required = [files["summary"], files["keywords"], files["examples"], files["meta"]]
    if WRITE_LABELED_ASSIGNMENTS:
        required.append(files["assignments"])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing 07 files:\n" + "\n".join(missing))
    return files


def clean_text(s: str, limit: int = 240) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s[:limit]


def top_counts_str(values: pd.Series, topk: int) -> str:
    cleaned = values.fillna("").astype(str).map(lambda x: clean_text(x, 120))
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return ""
    vc = cleaned.value_counts().head(topk)
    return ", ".join([f"{idx}({int(cnt)})" for idx, cnt in vc.items()])


def top_categories_str(categories: pd.Series, topk: int) -> str:
    raw = categories.fillna("").astype(str)
    exploded = (
        raw.str.split(",")
        .explode()
        .fillna("")
        .map(lambda x: clean_text(x.strip(), 120))
    )
    exploded = exploded[exploded != ""]
    if exploded.empty:
        return ""
    vc = exploded.value_counts().head(topk)
    return ", ".join([f"{idx}({int(cnt)})" for idx, cnt in vc.items()])


def clean_city_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).map(lambda x: clean_text(x, 120))
    return cleaned[cleaned != ""]


def strip_city_from_label(label: str, city_series: pd.Series) -> str:
    out = label
    cities = sorted(city_series.unique().tolist(), key=len, reverse=True)
    for city in cities:
        if len(city) < 3:
            continue
        out = re.sub(rf"\b{re.escape(city)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"^[,;:/\-\s]+|[,;:/\-\s]+$", "", out).strip()
    out = re.sub(r"\b(in|at|near|around|of)\s*$", "", out, flags=re.IGNORECASE).strip()
    return out


def clamp_confidence(value: Any) -> int:
    try:
        x = int(float(value))
    except Exception:
        x = 2
    return max(1, min(5, x))


def canonical_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def trim_label_words(label: str, max_words: int = MAX_LABEL_WORDS) -> str:
    words = [w for w in re.split(r"\s+", label.strip()) if w]
    return " ".join(words[:max_words]).strip()


def titleize_term(term: str) -> str:
    cleaned = clean_text(term, 40)
    cleaned = re.sub(r"[^a-zA-Z0-9&\-\s/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    return " ".join([w.capitalize() if w.islower() else w for w in cleaned.split(" ")])


def top_non_generic_terms(kw_df: pd.DataFrame, cluster: int, topk: int = 15) -> list[str]:
    subset = kw_df[kw_df["cluster"] == cluster].sort_values("rank").head(topk)
    out: list[str] = []
    for term in subset["term"].astype(str).tolist():
        t = clean_text(term, 40).lower()
        if not t or t in GENERIC_KEYWORD_TERMS:
            continue
        out.append(t)
    return out


def summarize_l2_distribution(cluster_assign: pd.DataFrame, topk: int = 3) -> tuple[str, float, bool]:
    if not INCLUDE_L2_EVIDENCE:
        return "", 0.0, False
    if "l2_label_top1" not in cluster_assign.columns:
        return "", 0.0, False

    l2 = (
        cluster_assign["l2_label_top1"]
        .fillna("")
        .astype(str)
        .map(lambda x: clean_text(x, 80).lower())
    )
    l2 = l2[l2 != ""]
    if l2.empty:
        return "", 0.0, False

    vc = l2.value_counts()
    total = int(vc.sum())
    top = vc.head(topk)
    parts = [
        f"{label}({int(cnt)},{(100.0 * float(cnt) / float(total)):.1f}%)"
        for label, cnt in top.items()
    ]
    dominant_share = float(top.iloc[0]) / float(total)
    is_mixed = dominant_share < float(MIXED_CLUSTER_THRESHOLD)
    return "; ".join(parts), float(dominant_share), bool(is_mixed)


def contains_token(text: str, token: str) -> bool:
    escaped = re.escape(token).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def nightlife_signal_info(
    cluster_assign: pd.DataFrame,
    kw_df: pd.DataFrame,
    cluster: int,
) -> tuple[int, bool, list[str]]:
    kws = set(top_non_generic_terms(kw_df, cluster, 30))
    cat_text = ",".join(
        cluster_assign.get("categories", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    ).lower()
    l2_text = ",".join(
        cluster_assign.get("l2_label_top1", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    ).lower()

    hits: set[str] = set()
    for token in NIGHTLIFE_SIGNAL_TERMS:
        t = token.lower().strip()
        if not t:
            continue
        if t in kws or contains_token(cat_text, t) or contains_token(l2_text, t):
            hits.add(t)
    has_strong = any(t in NIGHTLIFE_STRONG_TERMS for t in hits)
    return len(hits), bool(has_strong), sorted(hits)


def should_allow_nightlife_rewrite(
    cluster_assign: pd.DataFrame,
    kw_df: pd.DataFrame,
    cluster: int,
    dominant_l2_share: float,
) -> tuple[bool, list[str]]:
    n_hits, has_strong, hits = nightlife_signal_info(cluster_assign, kw_df, cluster)
    allow = bool(
        has_strong
        and n_hits >= int(NIGHTLIFE_REWRITE_MIN_SIGNAL_HITS)
        and float(dominant_l2_share) <= float(NIGHTLIFE_REWRITE_MAX_DOMINANT_L2_SHARE)
    )
    notes = [f"nightlife_hits={n_hits}", f"nightlife_strong={has_strong}"]
    if hits:
        notes.append(f"nightlife_terms={','.join(hits[:6])}")
    notes.append(f"dominant_l2_share={float(dominant_l2_share):.3f}")
    return allow, notes


def interpretive_fallback_label(
    cluster: int,
    cluster_assign: pd.DataFrame,
    kw_df: pd.DataFrame,
    is_mixed_cluster: bool,
    allow_nightlife_rewrite: bool = True,
) -> str:
    kws = set(top_non_generic_terms(kw_df, cluster, 20))
    cat_text = ",".join(cluster_assign.get("categories", pd.Series(dtype=str)).fillna("").astype(str).tolist()).lower()

    def has_any(tokens: list[str]) -> bool:
        for token in tokens:
            t = token.lower().strip()
            if not t:
                continue
            if t in kws:
                return True
            if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", cat_text):
                return True
        return False

    if allow_nightlife_rewrite and has_any(
        ["nightlife", "bars", "bar", "cocktail", "beer", "music", "drinks"]
    ):
        return "Nightlife Social Spots"
    if has_any(["breakfast", "brunch", "coffee", "cafe", "latte", "espresso"]):
        return "Morning Cafe Routine"
    if has_any(["fast food", "drive", "chain", "subway", "mcdonald", "quick"]):
        return "Quick-Service Chains"
    if has_any(["grocery", "pharmacy", "drugstore", "convenience", "market", "retail"]):
        return "Food Retail Convenience"
    if has_any(["dessert", "bakery", "ice cream", "frozen yogurt", "cake", "donut"]):
        return "Sweet Treat Stops"
    if is_mixed_cluster:
        return "Mixed Local Dining"
    return "Neighborhood Dining Mix"


def parse_dominant_l2_label(dominant_l2_top3: str) -> str:
    text = clean_text(dominant_l2_top3, 200)
    if not text:
        return ""
    part = text.split(";", 1)[0].strip()
    m = re.match(r"^([a-z0-9_]+)\(", part.lower())
    if not m:
        return ""
    label = m.group(1).strip().lower()
    if label in {"", "restaurants_general", "other_service"}:
        return ""
    return label


def parse_dominant_l2_labels(dominant_l2_top3: str, topk: int = 3) -> list[str]:
    text = clean_text(dominant_l2_top3, 600)
    if not text:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for part in text.split(";"):
        token = part.strip().lower()
        m = re.match(r"^([a-z0-9_]+)\(", token)
        if not m:
            continue
        label = m.group(1).strip().lower()
        if not label or label in MIXED_CLUSTER_SKIP_L2_LABELS or label in seen:
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= int(topk):
            break
    return out


def should_force_explicit_mixed_label(
    is_mixed_cluster: bool,
    dominant_l2_share: float,
    n_businesses: int,
) -> bool:
    return bool(
        MIXED_CLUSTER_FORCE_EXPLICIT
        and is_mixed_cluster
        and float(dominant_l2_share) <= float(MIXED_CLUSTER_FORCE_MAX_DOMINANT_SHARE)
        and int(n_businesses) >= int(MIXED_CLUSTER_FORCE_MIN_N_BUSINESSES)
    )


def build_explicit_mixed_label(
    dominant_l2_top3: str,
    kw_df: pd.DataFrame,
    cluster: int,
    max_components: int = MIXED_CLUSTER_LABEL_TOPK,
) -> tuple[str, list[str]]:
    components: list[str] = []
    for l2 in parse_dominant_l2_labels(dominant_l2_top3, topk=max_components + 1):
        comp = titleize_term(l2.replace("_", " "))
        if comp:
            components.append(comp)
        if len(components) >= int(max_components):
            break

    if len(components) < 2:
        for term in top_non_generic_terms(kw_df, cluster, 8):
            comp = titleize_term(term)
            if not comp:
                continue
            key = canonical_label(comp)
            if key in {canonical_label(x) for x in components}:
                continue
            components.append(comp)
            if len(components) >= int(max_components):
                break

    if len(components) >= 2:
        label = trim_label_words(f"Mixed {components[0]} & {components[1]}", MAX_LABEL_WORDS)
    elif len(components) == 1:
        label = trim_label_words(f"Mixed {components[0]} Dining", MAX_LABEL_WORDS)
    else:
        label = "Mixed Local Dining"
    return label, components


def semantic_suffix_candidates(
    cluster: int,
    kw_df: pd.DataFrame,
    dominant_l2_top3: str,
    is_mixed_cluster: bool,
) -> list[str]:
    candidates: list[str] = []
    top_l2 = parse_dominant_l2_label(dominant_l2_top3)
    if top_l2:
        l2_title = titleize_term(top_l2.replace("_", " "))
        if l2_title:
            candidates.append(l2_title)
    for term in top_non_generic_terms(kw_df, cluster, 8):
        t = titleize_term(term)
        if t:
            candidates.append(t)
    if is_mixed_cluster:
        candidates.append("Mixed")

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        key = canonical_label(c)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def make_label_unique(
    label: str,
    cluster: int,
    kw_df: pd.DataFrame,
    used_keys: set[str],
    dominant_l2_top3: str,
    is_mixed_cluster: bool,
) -> tuple[str, str]:
    base = trim_label_words(label, MAX_LABEL_WORDS) or f"Cluster {cluster}"
    key = canonical_label(base)
    if key and key not in used_keys:
        used_keys.add(key)
        return base, "as_is"

    for suffix in semantic_suffix_candidates(
        cluster=cluster,
        kw_df=kw_df,
        dominant_l2_top3=dominant_l2_top3,
        is_mixed_cluster=is_mixed_cluster,
    ):
        cand = trim_label_words(f"{base} {suffix}", MAX_LABEL_WORDS)
        cand_key = canonical_label(cand)
        if cand_key and cand_key not in used_keys:
            used_keys.add(cand_key)
            return cand, f"semantic_suffix:{canonical_label(suffix)}"

    i = 2
    while True:
        cand = trim_label_words(f"{base} Variant {i}", MAX_LABEL_WORDS)
        cand_key = canonical_label(cand)
        if cand_key and cand_key not in used_keys:
            used_keys.add(cand_key)
            return cand, f"variant_suffix:{i}"
        i += 1


def top_non_generic_category_share(cluster_assign: pd.DataFrame) -> float:
    categories = cluster_assign.get("categories", pd.Series(dtype=str)).fillna("").astype(str)
    n = int(len(categories))
    if n == 0:
        return 0.0

    counts: dict[str, int] = {}
    for cat_text in categories.tolist():
        seen: set[str] = set()
        for raw in cat_text.split(","):
            token = clean_text(raw, 80).lower()
            if not token or token in GENERIC_CATEGORY_TERMS:
                continue
            seen.add(token)
        for token in seen:
            counts[token] = counts.get(token, 0) + 1

    if not counts:
        return 0.0
    return max(counts.values()) / float(n)


def is_vague_label(label: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t]
    if not tokens:
        return True
    vague_hits = sum(1 for t in tokens if t in VAGUE_LABEL_TERMS)
    return vague_hits >= max(1, len(tokens) // 2)


def label_has_theme_terms(label: str) -> bool:
    text = clean_text(label, 160).lower()
    if not text:
        return False
    for term in THEME_LABEL_TERMS:
        if contains_token(text, term):
            return True
    return False


def confidence_cap(
    cluster_assign: pd.DataFrame,
    kw_df: pd.DataFrame,
    cluster: int,
    label: str,
    dominant_l2_share: float,
    is_mixed_cluster: bool,
) -> int:
    n_businesses = int(len(cluster_assign))
    top_cat_share = top_non_generic_category_share(cluster_assign)
    specific_kw_count = len(top_non_generic_terms(kw_df, cluster, 10))

    cap = 5
    if n_businesses < 70:
        cap = min(cap, 4)
    if top_cat_share < 0.24:
        cap = min(cap, 4)
    if top_cat_share < 0.16:
        cap = min(cap, 3)
    if specific_kw_count < 3:
        cap = min(cap, 4)
    if specific_kw_count < 2:
        cap = min(cap, 3)
    if is_vague_label(label):
        cap = min(cap, 4)
    if top_cat_share < 0.12 and specific_kw_count < 2:
        cap = min(cap, 2)
    if is_mixed_cluster:
        cap = min(cap, 4)
    if dominant_l2_share > 0 and dominant_l2_share < 0.40:
        cap = min(cap, 3)
    return cap


def build_prompt(
    cluster: int,
    cluster_assign: pd.DataFrame,
    kw_df: pd.DataFrame,
    existing_labels: list[str],
    dominant_l2_top3: str,
    dominant_l2_share: float,
    is_mixed_cluster: bool,
) -> str:
    keywords = kw_df[kw_df["cluster"] == cluster].sort_values("rank").head(TOP_KEYWORDS)
    assign_sorted = cluster_assign.sort_values(
        by="n_reviews", ascending=False, na_position="last"
    )
    examples = assign_sorted.head(PROMPT_MAX_BUSINESSES)
    n_businesses = int(len(cluster_assign))

    avg_review_stars = pd.to_numeric(
        cluster_assign.get("avg_review_stars", pd.Series(dtype=float)), errors="coerce"
    ).mean()
    avg_business_stars = pd.to_numeric(
        cluster_assign.get("stars", pd.Series(dtype=float)), errors="coerce"
    ).mean()
    avg_n_reviews = pd.to_numeric(
        cluster_assign.get("n_reviews", pd.Series(dtype=float)), errors="coerce"
    ).mean()
    if pd.isna(avg_review_stars):
        avg_review_stars = 0.0
    if pd.isna(avg_business_stars):
        avg_business_stars = 0.0
    if pd.isna(avg_n_reviews):
        avg_n_reviews = 0.0

    keyword_list = [clean_text(x) for x in keywords["term"].tolist()]
    ex_lines = []
    for _, r in examples.iterrows():
        ex_lines.append(
            f"- name={clean_text(r.get('name', ''))}; city={clean_text(r.get('city', ''))}; "
            f"categories={clean_text(r.get('categories', ''))}; n_reviews={int(r.get('n_reviews', 0))}; "
            f"avg_review_stars={float(r.get('avg_review_stars', 0.0)):.2f}"
        )

    city_mix = top_counts_str(cluster_assign.get("city", pd.Series(dtype=str)), CITY_TOPK)
    category_mix = top_categories_str(
        cluster_assign.get("categories", pd.Series(dtype=str)), CATEGORY_TOPK
    )
    existing_str = ", ".join(existing_labels[-MAX_EXISTING_LABELS_IN_PROMPT:]) or "(none yet)"

    return (
        "You are labeling a Yelp business cluster.\n"
        "Return JSON only with keys: label, short_reason, confidence.\n"
        "Rules:\n"
        "- label: short semantic name in English, <= 6 words.\n"
        "- label must NOT contain city/region names.\n"
        "- label must be distinct from existing labels.\n"
        "- avoid generic labels like 'Restaurants' without theme words.\n"
        "- label should reflect the most concrete dominant theme (cuisine, service mode, or scene).\n"
        "- cuisine words are allowed when clearly dominant in evidence.\n"
        "- short_reason: <= 30 words, mention behavioral or business pattern evidence.\n"
        "- confidence: integer 1-5.\n"
        "- confidence rubric: 5=very coherent niche theme, 4=mostly coherent with noise, 3=mixed broad cluster, 2=weak theme, 1=unclear.\n\n"
        f"Cluster id: {cluster}\n"
        f"Stats: n_businesses={n_businesses}, avg_review_stars={avg_review_stars:.3f}, "
        f"avg_business_stars={avg_business_stars:.3f}, avg_n_reviews={avg_n_reviews:.2f}\n"
        f"L2 evidence from step07 (reference only): {dominant_l2_top3 or 'n/a'}\n"
        f"L2 dominant_share={dominant_l2_share:.3f}, is_mixed_cluster={is_mixed_cluster}\n"
        f"Existing labels (avoid duplicates): {existing_str}\n"
        f"Top keywords: {', '.join(keyword_list)}\n"
        f"City mix: {city_mix}\n"
        f"Category mix: {category_mix}\n"
        "Top example businesses:\n"
        + "\n".join(ex_lines)
    )


def ollama_generate(prompt: str) -> str:
    options = {
        "temperature": 0,
        "num_ctx": int(OLLAMA_NUM_CTX),
        "num_predict": int(OLLAMA_NUM_PREDICT),
        # qwen3 in Ollama accepts both keys across versions; keep both for compatibility.
        "thinking": bool(OLLAMA_THINKING),
        "think": bool(OLLAMA_THINKING),
    }
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": "You are a concise clustering label assistant. Output strict JSON only.",
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": options,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    text = (data.get("response") or "").strip()
    if not text:
        msg = data.get("message") or {}
        if isinstance(msg, dict):
            text = (msg.get("content") or "").strip()
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

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def fallback_label(kw_df: pd.DataFrame, cluster: int) -> dict[str, Any]:
    kws = kw_df[kw_df["cluster"] == cluster].sort_values("rank").head(3)["term"].tolist()
    label = " / ".join([clean_text(x, 40) for x in kws]) if kws else f"cluster_{cluster}"
    return {
        "label": label,
        "short_reason": "Fallback label from top keywords.",
        "confidence": 2,
    }


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_profile = normalize_run_profile(RUN_PROFILE)
    source_dir = pick_source_run_dir(run_profile)
    files = require_files(source_dir)

    summary_df = pd.read_csv(files["summary"])
    keywords_df = pd.read_csv(files["keywords"])
    assignments_df = pd.read_csv(files["assignments"])

    clusters = sorted(assignments_df["cluster"].dropna().astype(int).unique().tolist())
    if QUICK_MODE:
        clusters = clusters[:MAX_CLUSTERS_QUICK]

    profile_out_root = OUTPUT_ROOT / run_profile
    if RUN_TAG.strip():
        run_name = f"{run_id}_{run_profile}_{RUN_TAG.strip()}"
    else:
        run_name = f"{run_id}_{run_profile}"
    out_dir = profile_out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "cluster_labels.csv"
    meta_csv = out_dir / "run_meta.csv"
    labeled_assignments_csv = out_dir / LABELED_ASSIGNMENTS_NAME

    print(f"[CONFIG] run_profile={run_profile}")
    print(f"[CONFIG] source_07_dir={source_dir}")
    print(f"[CONFIG] ollama_url={OLLAMA_URL}")
    print(f"[CONFIG] ollama_model={OLLAMA_MODEL}")
    print(f"[CONFIG] quick_mode={QUICK_MODE}, n_clusters={len(clusters)}")

    rows = []
    used_label_keys: set[str] = set()
    existing_labels: list[str] = []
    for c in clusters:
        cluster_assign = assignments_df[assignments_df["cluster"] == c]
        if cluster_assign.empty:
            continue
        city_series = clean_city_series(cluster_assign.get("city", pd.Series(dtype=str)))
        dominant_l2_top3, dominant_l2_share, is_mixed_cluster = summarize_l2_distribution(
            cluster_assign
        )
        prompt = build_prompt(
            c,
            cluster_assign,
            keywords_df,
            existing_labels,
            dominant_l2_top3,
            dominant_l2_share,
            is_mixed_cluster,
        )
        print(f"[STEP] labeling cluster={c}")
        raw = ""
        err = ""
        parsed: dict[str, Any] = {}
        try:
            raw = ollama_generate(prompt)
            parsed = try_parse_json(raw)
        except Exception as exc:
            err = str(exc)

        rewrite_notes: list[str] = []
        raw_label_text = ""
        if not parsed:
            parsed = fallback_label(keywords_df, c)
            rewrite_notes.append("fallback_parse_error_or_empty_json")

        raw_label_text = clean_text(parsed.get("label", ""))
        final_label_text = raw_label_text
        mixed_label_forced = False
        mixed_label_components: list[str] = []
        if FORCE_NON_GEO_LABEL:
            stripped = strip_city_from_label(final_label_text, city_series)
            if stripped and stripped != final_label_text:
                final_label_text = stripped
                rewrite_notes.append("strip_city_token")

        fallback_trigger = ""
        if not final_label_text:
            fallback_trigger = "empty_label"
        elif is_vague_label(final_label_text) and not label_has_theme_terms(final_label_text):
            fallback_trigger = "vague_label_no_theme"

        if fallback_trigger:
            allow_nightlife, nightlife_notes = should_allow_nightlife_rewrite(
                cluster_assign=cluster_assign,
                kw_df=keywords_df,
                cluster=c,
                dominant_l2_share=dominant_l2_share,
            )
            rewritten = interpretive_fallback_label(
                cluster=c,
                cluster_assign=cluster_assign,
                kw_df=keywords_df,
                is_mixed_cluster=is_mixed_cluster,
                allow_nightlife_rewrite=allow_nightlife,
            )
            rewritten = clean_text(rewritten, 120)
            if rewritten:
                final_label_text = rewritten
            rewrite_notes.append(fallback_trigger)
            rewrite_notes.append("interpretive_fallback")
            rewrite_notes.extend(nightlife_notes)
            rewrite_notes.append(f"nightlife_rewrite_allowed={allow_nightlife}")

        if should_force_explicit_mixed_label(
            is_mixed_cluster=is_mixed_cluster,
            dominant_l2_share=dominant_l2_share,
            n_businesses=len(cluster_assign),
        ):
            mixed_label, components = build_explicit_mixed_label(
                dominant_l2_top3=dominant_l2_top3,
                kw_df=keywords_df,
                cluster=c,
                max_components=MIXED_CLUSTER_LABEL_TOPK,
            )
            if mixed_label:
                if mixed_label != final_label_text:
                    rewrite_notes.append("force_explicit_mixed_label")
                final_label_text = mixed_label
                mixed_label_forced = True
                mixed_label_components = components

        pre_unique_label = final_label_text
        final_label_text, unique_strategy = make_label_unique(
            label=final_label_text,
            cluster=c,
            kw_df=keywords_df,
            used_keys=used_label_keys,
            dominant_l2_top3=dominant_l2_top3,
            is_mixed_cluster=is_mixed_cluster,
        )
        if final_label_text != pre_unique_label:
            rewrite_notes.append(f"dedup:{unique_strategy}")
        existing_labels.append(final_label_text)

        model_conf = clamp_confidence(parsed.get("confidence", 2))
        cap = confidence_cap(
            cluster_assign,
            keywords_df,
            c,
            final_label_text,
            dominant_l2_share,
            is_mixed_cluster,
        )
        final_conf = min(model_conf, cap)

        model_short_reason = clean_text(parsed.get("short_reason", ""), 400)
        rewrite_reason = ";".join([x for x in rewrite_notes if x])
        final_short_reason = model_short_reason
        if rewrite_reason and final_label_text != raw_label_text:
            final_short_reason = clean_text(
                (
                    f"Label rewritten from '{raw_label_text or 'empty'}' to '{final_label_text}' "
                    "based on cluster keywords, category mix, and L2 evidence."
                ),
                400,
            )

        if is_mixed_cluster and dominant_l2_share < 0.40:
            cluster_action_hint = "split_candidate"
        elif is_mixed_cluster:
            cluster_action_hint = "monitor_mixed"
        else:
            cluster_action_hint = "stable_cluster"

        rows.append(
            {
                "cluster": c,
                "label": final_label_text,
                "raw_label": raw_label_text,
                "final_label": final_label_text,
                "label_rewrite_reason": rewrite_reason,
                "short_reason": final_short_reason,
                "short_reason_raw": model_short_reason,
                "confidence": final_conf,
                "dominant_l2_top3": dominant_l2_top3,
                "dominant_l2_share": dominant_l2_share,
                "is_mixed_cluster": bool(is_mixed_cluster),
                "mixed_label_forced": bool(mixed_label_forced),
                "mixed_label_components": ", ".join(mixed_label_components),
                "cluster_action_hint": cluster_action_hint,
                "n_businesses": int(len(cluster_assign)),
                "avg_review_stars": float(
                    pd.to_numeric(cluster_assign["avg_review_stars"], errors="coerce").mean()
                ),
                "avg_business_stars": float(
                    pd.to_numeric(cluster_assign["stars"], errors="coerce").mean()
                ),
                "avg_n_reviews": float(
                    pd.to_numeric(cluster_assign["n_reviews"], errors="coerce").mean()
                ),
                "ollama_error": clean_text(err, 500),
                "raw_response": clean_text(raw, 1000),
            }
        )

    labels_df = pd.DataFrame(rows)
    labels_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "run_profile": run_profile,
                "source_07_dir": str(source_dir),
                "ollama_url": OLLAMA_URL,
                "ollama_model": OLLAMA_MODEL,
                "quick_mode": QUICK_MODE,
                "clusters_processed": len(clusters),
                "interpretive_label_mode": INTERPRETIVE_LABEL_MODE,
                "include_l2_evidence": INCLUDE_L2_EVIDENCE,
                "output_csv": str(out_csv),
            }
        ]
    ).to_csv(meta_csv, index=False, encoding="utf-8-sig")

    if WRITE_LABELED_ASSIGNMENTS:
        assignments_df = pd.read_csv(files["assignments"])
        merged = assignments_df.merge(
            labels_df[
                [
                    "cluster",
                    "label",
                    "raw_label",
                    "final_label",
                    "label_rewrite_reason",
                    "short_reason",
                    "short_reason_raw",
                    "confidence",
                    "dominant_l2_top3",
                    "dominant_l2_share",
                    "is_mixed_cluster",
                    "mixed_label_forced",
                    "mixed_label_components",
                    "cluster_action_hint",
                ]
            ],
            on="cluster",
            how="left",
        )
        merged.to_csv(labeled_assignments_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] wrote {labeled_assignments_csv}")

    print(f"[INFO] wrote {out_csv}")
    print(f"[INFO] wrote {meta_csv}")


if __name__ == "__main__":
    main()
