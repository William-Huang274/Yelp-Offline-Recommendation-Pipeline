import re
from collections import Counter
from typing import Any


THEME_KEYWORDS = {
    "cajun",
    "creole",
    "gumbo",
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
    "brunch",
    "breakfast",
    "coffee",
    "espresso",
    "latte",
    "cafe",
    "cocktail",
    "bartender",
    "nightlife",
    "bar",
    "pub",
    "gastropub",
    "smoothie",
    "juice",
    "deli",
    "sandwich",
}

SPAM_HINTS = {
    "follow us",
    "check out",
    "use code",
    "promo code",
    "discount",
    "sponsored",
    "click link",
    "subscribe",
    "ad",
}

WEAK_GENERIC_TERMS = {
    "good",
    "great",
    "nice",
    "bad",
    "terrible",
    "service",
    "place",
    "staff",
    "time",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
REVIEW_SPLIT_RE = re.compile(r"\n{2,}|<review_sep>", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'&-]*")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = _normalize_space(text)
    if not cleaned:
        return []
    parts = SENTENCE_SPLIT_RE.split(cleaned)
    out: list[str] = []
    for p in parts:
        s = _normalize_space(p)
        if s:
            out.append(s)
    return out


def _split_reviews(text: str) -> list[str]:
    cleaned = _normalize_space(text)
    if not cleaned:
        return []
    parts = REVIEW_SPLIT_RE.split(cleaned)
    out = [_normalize_space(p) for p in parts if _normalize_space(p)]
    if out:
        return out
    return [cleaned]


def _word_tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _token_set(text: str) -> set[str]:
    return set(_word_tokens(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _alpha_ratio(text: str) -> float:
    t = text or ""
    if not t:
        return 0.0
    alpha = sum(1 for c in t if c.isalpha())
    return float(alpha) / float(len(t))


def _excess_punct_score(text: str) -> float:
    t = text or ""
    exclam = t.count("!")
    quest = t.count("?")
    rep = len(re.findall(r"(.)\1{3,}", t))
    return float(max(0, exclam - 2) + max(0, quest - 2) + rep)


def _score_sentence(sentence: str, context_tokens: set[str]) -> tuple[float, float, float, list[str]]:
    lower = sentence.lower()
    words = _word_tokens(lower)
    word_set = set(words)

    matched_theme = [kw for kw in THEME_KEYWORDS if kw in lower]
    theme_score = float(len(matched_theme))

    context_hits = len([w for w in word_set if w in context_tokens])
    context_score = float(context_hits) * 0.4

    weak_hits = len([w for w in word_set if w in WEAK_GENERIC_TERMS])
    weak_penalty = float(max(0, weak_hits - 2)) * 0.15

    spam_hits = len([kw for kw in SPAM_HINTS if kw in lower])
    spam_score = float(spam_hits) + _excess_punct_score(sentence)

    info_score = theme_score + context_score
    final_score = info_score - (0.9 * spam_score) - weak_penalty
    return final_score, info_score, spam_score, matched_theme


def _score_review(review_text: str, context_tokens: set[str]) -> tuple[float, float, float, list[str]]:
    lower = (review_text or "").lower()
    words = _word_tokens(lower)
    word_set = set(words)

    matched_theme = [kw for kw in THEME_KEYWORDS if kw in lower]
    theme_score = float(len(matched_theme))
    context_hits = len([w for w in word_set if w in context_tokens])
    context_score = float(context_hits) * 0.35
    spam_hits = len([kw for kw in SPAM_HINTS if kw in lower])
    spam_score = float(spam_hits) + _excess_punct_score(review_text)
    generic_hits = len([w for w in word_set if w in WEAK_GENERIC_TERMS])
    generic_penalty = float(max(0, generic_hits - 4)) * 0.12

    info_score = theme_score + context_score
    final_score = info_score - (0.85 * spam_score) - generic_penalty
    return final_score, info_score, spam_score, matched_theme


def _sentence_is_valid(sentence: str, min_words: int, max_words: int) -> bool:
    words = _word_tokens(sentence)
    n = len(words)
    if n < int(min_words) or n > int(max_words):
        return False
    if _alpha_ratio(sentence) < 0.55:
        return False
    return True


def _review_is_valid(review_text: str, min_words: int = 12, max_words: int = 450) -> bool:
    words = _word_tokens(review_text)
    n = len(words)
    if n < int(min_words) or n > int(max_words):
        return False
    if _alpha_ratio(review_text) < 0.55:
        return False
    return True


def _select_diverse_sentences(
    candidates: list[dict[str, Any]],
    target_n: int,
    max_sim: float = 0.86,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for c in candidates:
        c_tokens = c["token_set"]
        too_similar = False
        for s in chosen:
            if _jaccard(c_tokens, s["token_set"]) >= float(max_sim):
                too_similar = True
                break
        if too_similar:
            continue
        chosen.append(c)
        if len(chosen) >= int(target_n):
            break
    return chosen


def _join_with_char_limit(sentences: list[str], max_chars: int) -> str:
    out = " ".join([_normalize_space(s) for s in sentences if _normalize_space(s)])
    if len(out) <= int(max_chars):
        return out
    return out[: int(max_chars)].rstrip()


def select_reviews_for_sentence_stage(
    reviews: list[str],
    categories: str = "",
    name: str = "",
    target_reviews: int = 6,
    min_review_words: int = 12,
    max_review_words: int = 450,
) -> dict[str, Any]:
    context_tokens = _token_set(f"{categories} {name}")
    candidates: list[dict[str, Any]] = []
    dropped = 0
    for rv in reviews:
        txt = _normalize_space(rv)
        if not txt:
            dropped += 1
            continue
        if not _review_is_valid(txt, min_words=min_review_words, max_words=max_review_words):
            dropped += 1
            continue
        final_score, info_score, spam_score, matched = _score_review(txt, context_tokens)
        candidates.append(
            {
                "review_text": txt,
                "final_score": float(final_score),
                "info_score": float(info_score),
                "spam_score": float(spam_score),
                "token_set": _token_set(txt),
                "matched_theme": matched,
            }
        )

    if not candidates:
        return {
            "selected_reviews": [],
            "review_total": int(len(reviews)),
            "review_kept": 0,
            "review_dropped": int(len(reviews)),
            "review_mean_info_score": 0.0,
            "review_mean_spam_score": 0.0,
            "review_selected_keywords": "",
        }

    candidates.sort(key=lambda x: (-x["final_score"], -x["info_score"], x["spam_score"]))
    selected = _select_diverse_sentences(candidates, target_n=target_reviews, max_sim=0.90)

    mean_info = sum(x["info_score"] for x in selected) / float(len(selected))
    mean_spam = sum(x["spam_score"] for x in selected) / float(len(selected))
    kw_counter: Counter[str] = Counter()
    for x in selected:
        for kw in x["matched_theme"]:
            kw_counter[kw] += 1

    return {
        "selected_reviews": [x["review_text"] for x in selected],
        "review_total": int(len(reviews)),
        "review_kept": int(len(selected)),
        "review_dropped": int(dropped),
        "review_mean_info_score": float(round(mean_info, 4)),
        "review_mean_spam_score": float(round(mean_spam, 4)),
        "review_selected_keywords": ",".join([k for k, _ in kw_counter.most_common(8)]),
    }


def build_text_views_from_reviews(
    reviews: list[str],
    categories: str = "",
    name: str = "",
    target_reviews: int = 6,
    relabel_sentences: int = 8,
    embed_sentences: int = 12,
    relabel_max_chars: int = 1500,
    embed_max_chars: int = 2500,
    min_words: int = 8,
    max_words: int = 35,
) -> dict[str, Any]:
    review_pick = select_reviews_for_sentence_stage(
        reviews=reviews,
        categories=categories,
        name=name,
        target_reviews=target_reviews,
    )
    selected_reviews = review_pick["selected_reviews"]
    sentence_pool: list[str] = []
    for rv in selected_reviews:
        sentence_pool.extend(_split_sentences(rv))

    raw_text = _normalize_space(" ".join(reviews))
    context_tokens = _token_set(f"{categories} {name}")
    candidates: list[dict[str, Any]] = []
    dropped_sentences = 0
    for s in sentence_pool:
        if not _sentence_is_valid(s, min_words=min_words, max_words=max_words):
            dropped_sentences += 1
            continue
        final_score, info_score, spam_score, matched = _score_sentence(s, context_tokens)
        candidates.append(
            {
                "sentence": s,
                "final_score": float(final_score),
                "info_score": float(info_score),
                "spam_score": float(spam_score),
                "token_set": _token_set(s),
                "matched_theme": matched,
            }
        )

    if not candidates:
        fallback_reviews = selected_reviews if selected_reviews else reviews
        fallback_text = _normalize_space(" ".join(fallback_reviews))
        if not fallback_text:
            fallback_text = raw_text
        return {
            "text_for_relabel": fallback_text[: int(relabel_max_chars)],
            "text_for_embed": fallback_text[: int(embed_max_chars)],
            "review_filter_total_reviews": int(review_pick["review_total"]),
            "review_filter_kept_reviews": int(review_pick["review_kept"]),
            "review_filter_dropped_reviews": int(review_pick["review_dropped"]),
            "review_filter_mean_info_score": float(review_pick["review_mean_info_score"]),
            "review_filter_mean_spam_score": float(review_pick["review_mean_spam_score"]),
            "review_filter_selected_keywords": str(review_pick["review_selected_keywords"]),
            "text_filter_total_sentences": int(len(sentence_pool)),
            "text_filter_kept_sentences": 0,
            "text_filter_dropped_sentences": int(len(sentence_pool)),
            "text_filter_mean_info_score": 0.0,
            "text_filter_mean_spam_score": 0.0,
            "text_filter_selected_keywords": "",
        }

    candidates.sort(key=lambda x: (-x["final_score"], -x["info_score"], x["spam_score"]))
    selected_embed = _select_diverse_sentences(candidates, target_n=embed_sentences)
    selected_relabel = selected_embed[: int(relabel_sentences)]

    relabel_sents = [x["sentence"] for x in selected_relabel]
    embed_sents = [x["sentence"] for x in selected_embed]
    text_for_relabel = _join_with_char_limit(relabel_sents, relabel_max_chars)
    text_for_embed = _join_with_char_limit(embed_sents, embed_max_chars)
    if not text_for_relabel:
        text_for_relabel = _join_with_char_limit([candidates[0]["sentence"]], relabel_max_chars)
    if not text_for_embed:
        text_for_embed = _join_with_char_limit([candidates[0]["sentence"]], embed_max_chars)

    all_selected = selected_embed if selected_embed else candidates[:1]
    mean_info = sum(x["info_score"] for x in all_selected) / float(len(all_selected))
    mean_spam = sum(x["spam_score"] for x in all_selected) / float(len(all_selected))
    kw_counter: Counter[str] = Counter()
    for x in all_selected:
        for kw in x["matched_theme"]:
            kw_counter[kw] += 1

    return {
        "text_for_relabel": text_for_relabel,
        "text_for_embed": text_for_embed,
        "review_filter_total_reviews": int(review_pick["review_total"]),
        "review_filter_kept_reviews": int(review_pick["review_kept"]),
        "review_filter_dropped_reviews": int(review_pick["review_dropped"]),
        "review_filter_mean_info_score": float(review_pick["review_mean_info_score"]),
        "review_filter_mean_spam_score": float(review_pick["review_mean_spam_score"]),
        "review_filter_selected_keywords": str(review_pick["review_selected_keywords"]),
        "text_filter_total_sentences": int(len(sentence_pool)),
        "text_filter_kept_sentences": int(len(all_selected)),
        "text_filter_dropped_sentences": int(dropped_sentences),
        "text_filter_mean_info_score": float(round(mean_info, 4)),
        "text_filter_mean_spam_score": float(round(mean_spam, 4)),
        "text_filter_selected_keywords": ",".join([k for k, _ in kw_counter.most_common(8)]),
    }


def build_text_views(
    text: str,
    categories: str = "",
    name: str = "",
    relabel_sentences: int = 8,
    embed_sentences: int = 12,
    relabel_max_chars: int = 1500,
    embed_max_chars: int = 2500,
    min_words: int = 8,
    max_words: int = 35,
) -> dict[str, Any]:
    reviews = _split_reviews(text)
    return build_text_views_from_reviews(
        reviews=reviews,
        categories=categories,
        name=name,
        target_reviews=6,
        relabel_sentences=relabel_sentences,
        embed_sentences=embed_sentences,
        relabel_max_chars=relabel_max_chars,
        embed_max_chars=embed_max_chars,
        min_words=min_words,
        max_words=max_words,
    )
