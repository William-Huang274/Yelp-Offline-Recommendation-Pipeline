from __future__ import annotations

import json
import math
import re
from typing import Any

from pipeline.stage11_text_features import (
    _looks_review_fragment,
    _normalize_summary_term,
    build_clean_user_evidence_text,
    build_candidate_compact_boundary_gap_hint,
    build_candidate_compact_channel_hint,
    build_candidate_compact_competition_hint,
    build_candidate_compact_head_guard_hint,
    build_candidate_compact_rescue_band_hint,
    build_candidate_compact_rank_role_hint,
    build_candidate_compact_risk_hint,
    build_candidate_compact_route_profile_hint,
    build_candidate_compact_route_diversity_hint,
    build_candidate_compact_stability_profile_hint,
    build_candidate_compact_stable_fit_hint,
    build_candidate_compact_support_profile_hint,
    build_candidate_compact_support_reliability_hint,
    build_candidate_competition_summary,
    build_candidate_fit_risk_summary,
    build_history_anchor_summary,
    build_item_semantic_summary,
    build_pair_alignment_summary,
    build_user_preference_summary,
    clean_text,
    merge_distinct_segments,
    naturalize_user_context_text,
    naturalize_user_long_pref_text,
    naturalize_user_negative_avoid_text,
    naturalize_user_recent_intent_text,
    split_tags,
)


def _clean(s: Any) -> str:
    return clean_text(s)


def _join_semantic_parts(parts: list[str], max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for raw in parts:
        seg = clean_text(raw)
        if not seg:
            continue
        if re.fullmatch(r"[a-z_ ]+:\s*", seg.strip(), flags=re.I):
            continue
        add_len = len(seg) if not out else len(seg) + 2
        if out and total + add_len > max_chars:
            break
        if not out and len(seg) > max_chars:
            return clean_text(seg, max_chars=max_chars)
        out.append(seg)
        total += add_len
    return clean_text("; ".join(out), max_chars=max_chars)


def _join_sentence_parts(parts: list[str], max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for raw in parts:
        seg = clean_text(raw)
        if not seg:
            continue
        add_len = len(seg) if not out else len(seg) + 1
        if out and total + add_len > max_chars:
            break
        if not out and len(seg) > max_chars:
            return clean_text(seg, max_chars=max_chars)
        out.append(seg)
        total += add_len
    return clean_text(" ".join(out), max_chars=max_chars)


def _line_if_text(label: str, text: Any, max_chars: int = 220) -> str:
    txt = clean_text(text, max_chars=max_chars)
    if not txt:
        return ""
    return f"{label}: {txt}"


def _strip_sentence_lead(text: str, *prefixes: str) -> str:
    cleaned = clean_text(text, max_chars=260).strip()
    lowered = cleaned.lower()
    for prefix in prefixes:
        pref = clean_text(prefix, max_chars=120).strip()
        if pref and lowered.startswith(pref.lower()):
            cleaned = cleaned[len(pref):].strip(" ,.;:")
            break
    return cleaned.strip(" ,.;:")


def _normalize_free_text(raw: Any, max_chars: int = 220) -> str:
    txt = clean_text(str(raw or "").replace("_", " "), max_chars=max_chars)
    txt = re.sub(r"\bPrimary geo area:\s*[^.;]+[.;]?", "", txt, flags=re.I)
    txt = re.sub(r'["\']?[a-z0-9_]+["\']?\s*:\s*\[[^\]]*\]', " ", txt, flags=re.I)
    txt = re.sub(r'[{}\[\]]', " ", txt)
    txt = re.sub(r"\s+\.\s*", ". ", txt)
    return clean_text(txt, max_chars=max_chars)


def _clean_item_evidence_text(raw: Any, max_chars: int = 180) -> str:
    merged = merge_distinct_segments(raw, max_segments=2, max_chars=max(240, int(max_chars) * 2))
    if not merged:
        return ""
    parts: list[str] = []
    for seg in str(merged).split("||"):
        txt = _normalize_free_text(seg, max_chars=140)
        if not txt:
            continue
        if _looks_review_fragment(txt):
            continue
        if any(ch in txt for ch in ("!", "?", "  ")):
            continue
        if len(txt.split()) < 4:
            continue
        parts.append(txt)
        if len(parts) >= 2:
            break
    return _join_sentence_parts(parts, max_chars=max_chars) if parts else ""


def _clean_user_evidence_asset(raw: Any, max_chars: int = 220) -> str:
    merged = merge_distinct_segments(raw, max_segments=2, max_chars=max(260, int(max_chars) * 2))
    if not merged:
        return ""
    parts: list[str] = []
    for seg in str(merged).split("||"):
        txt = _normalize_free_text(seg, max_chars=180)
        if not txt:
            continue
        parts.append(txt)
        if len(parts) >= 2:
            break
    return _join_semantic_parts(parts, max_chars=max_chars) if parts else ""


def _normalize_term_phrase(raw: Any, max_words: int = 4) -> str:
    txt = _normalize_summary_term(raw, max_words=max_words)
    if not txt:
        return ""
    lower = txt.lower()
    if lower in _GENERIC_CATEGORY_TERMS:
        return ""
    if len(txt.split()) == 1 and lower in _LOW_SIGNAL_USER_FOCUS_TOKENS:
        return ""
    if re.search(r"\b(profile|recent|history|preference|avoid|context|theme|themes)\b", lower):
        return ""
    return txt


def _join_terms_natural(values: list[str], limit: int = 4) -> str:
    vals = [clean_text(v, max_chars=40) for v in values if clean_text(v, max_chars=40)]
    if not vals:
        return ""
    vals = vals[: max(1, int(limit))]
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return f"{', '.join(vals[:-1])}, and {vals[-1]}"


def _take_tags(raw: Any, limit: int = 6) -> str:
    rewrite = {
        "service good": "good service",
        "service bad": "service issues",
        "value bad": "poor value",
        "value good": "good value",
        "atmosphere noisy": "noisy atmosphere",
        "cleanliness bad": "cleanliness issues",
        "wait long": "long waits",
    }
    tags = []
    for tag in split_tags(raw, limit=limit * 2):
        clean_tag = clean_text(tag, max_chars=40).lower()
        if not clean_tag:
            continue
        normalized = rewrite.get(clean_tag, clean_text(tag, max_chars=40))
        if normalized:
            tags.append(normalized)
        if len(tags) >= limit:
            break
    return ", ".join(tags) if tags else ""


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        out = float(v)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


_SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "good",
    "great",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "menu",
    "of",
    "on",
    "or",
    "place",
    "restaurant",
    "restaurants",
    "service",
    "spot",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "very",
    "was",
    "were",
    "with",
    "cuisine",
    "cuisines",
    "dish",
    "dishes",
    "dining",
    "experience",
    "experiences",
    "family",
    "flavor",
    "flavors",
    "food",
    "foods",
    "friendly",
    "local",
    "meal",
    "meals",
    "scene",
    "style",
    "styles",
    "vibe",
    "vibes",
}

_LOW_SIGNAL_USER_FOCUS_TOKENS = {
    "atmosphere",
    "cuisine",
    "dining",
    "experience",
    "family",
    "flavor",
    "food",
    "friendly",
    "meal",
    "scene",
    "service",
    "style",
    "vibe",
}
_GENERIC_CATEGORY_TERMS = {
    "american",
    "american (new)",
    "american new",
    "american (traditional)",
    "american traditional",
    "bar",
    "bars",
    "cafe",
    "cafes",
    "caterers",
    "event planning & services",
    "flowers & gifts",
    "food",
    "gift shops",
    "local flavor",
    "nightlife",
    "restaurant",
    "restaurants",
    "specialty food",
}
_EXPLICIT_NEGATIVE_TERMS = {
    "cleanliness issues",
    "crowds",
    "long waits",
    "noise",
    "poor value",
    "service issues",
}

_NO_MATCH_TEXT = "no clear direct match from available text"
_NO_CONFLICT_TEXT = "no clear direct conflict from available text"
_LOCAL_LISTWISE_ITEM_MAX_CHARS = 760
_JSONISH_TAG_RE = re.compile(r'["\']?([a-z0-9]+(?:_[a-z0-9]+)+)["\']?', re.I)

_NEGATIVE_CUE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(wait|waits|waiting|line|queue)\b", re.I), "long waits"),
    (re.compile(r"\b(rude|slow service|bad service|service bad|staff)\b", re.I), "service issues"),
    (re.compile(r"\b(expensive|overpriced|pricey|value bad|poor value|price value)\b", re.I), "poor value"),
    (re.compile(r"\b(dirty|unclean|cleanliness bad|cleanliness)\b", re.I), "cleanliness issues"),
    (re.compile(r"\b(loud|noisy)\b", re.I), "noise"),
    (re.compile(r"\b(crowded|packed|busy)\b", re.I), "crowds"),
]

_PREFERENCE_CONFLICT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bservice\b", re.I), "service issues"),
    (re.compile(r"\b(value|price)\b", re.I), "poor value"),
    (re.compile(r"\b(cleanliness|clean)\b", re.I), "cleanliness issues"),
    (re.compile(r"\b(wait|line|queue)\b", re.I), "long waits"),
    (re.compile(r"\b(noise|loud|atmosphere)\b", re.I), "noise"),
]


def _semantic_token_pool(*values: Any, limit: int = 18) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = clean_text(raw, max_chars=320)
        if not text:
            continue
        for piece in re.split(r"[|,;/]+", text):
            raw_piece = str(piece or "")
            seg = clean_text(
                re.sub(r"[\[\]{}\"']", " ", raw_piece).replace("_", " "),
                max_chars=48,
            )
            if ":" in raw_piece:
                seg = ""
            norm = seg.lower()
            if 2 <= len(norm) <= 48 and norm not in seen and norm not in _SEMANTIC_STOPWORDS:
                out.append(seg)
                seen.add(norm)
                if len(out) >= limit:
                    return out
        for token in re.findall(r"[a-z0-9][a-z0-9&+'-]{1,30}", text.lower()):
            if token in seen or token in _SEMANTIC_STOPWORDS:
                continue
            out.append(token)
            seen.add(token)
            if len(out) >= limit:
                return out
    return out


def _structured_semantic_terms(*values: Any, limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = clean_text(raw, max_chars=360)
        if not text:
            continue
        candidates: list[str] = []
        candidates.extend(_JSONISH_TAG_RE.findall(text))
        for piece in re.split(r"[|,;/]+", text):
            raw_piece = str(piece or "")
            seg = clean_text(
                re.sub(r"[\[\]{}\"']", " ", raw_piece).replace("_", " "),
                max_chars=48,
            )
            if not seg or ":" in raw_piece or len(seg.split()) > 4:
                continue
            if re.match(r"^(prefers|usually|often|common|typical|recent|latest|other|most|outings|near-term|likely)\b", seg, flags=re.I):
                continue
            candidates.append(seg)
        for raw_term in candidates:
            term = clean_text(str(raw_term).replace("_", " "), max_chars=48)
            if not term:
                continue
            tokens = _term_token_set(term)
            if not tokens:
                continue
            if len(tokens) == 1 and next(iter(tokens)) in _LOW_SIGNAL_USER_FOCUS_TOKENS:
                continue
            norm = term.lower()
            if norm in seen or norm in _SEMANTIC_STOPWORDS:
                continue
            seen.add(norm)
            out.append(term)
            if len(out) >= limit:
                return out
    return out


def _overlap_terms(left_terms: list[str], right_terms: list[str], limit: int = 4) -> list[str]:
    right_norm = {str(v).lower(): str(v) for v in right_terms}
    out: list[str] = []
    seen: set[str] = set()
    for term in left_terms:
        norm = str(term).lower()
        if norm in right_norm and norm not in seen:
            out.append(right_norm[norm])
            seen.add(norm)
            if len(out) >= limit:
                break
    return out


def _term_token_set(raw: Any) -> set[str]:
    txt = clean_text(raw, max_chars=80).lower()
    if not txt:
        return set()
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9&+'-]{1,30}", txt)
        if tok not in _SEMANTIC_STOPWORDS and len(tok) >= 3
    }
    return tokens


def _semantic_overlap_terms(
    left_terms: list[str],
    right_terms: list[str],
    limit: int = 4,
) -> list[str]:
    left_sets = [(_normalize_term_phrase(term, max_words=4), _term_token_set(term)) for term in left_terms]
    out: list[str] = []
    seen: set[str] = set()
    for raw in right_terms:
        term = _normalize_term_phrase(raw, max_words=4)
        if not term:
            continue
        norm = term.lower()
        if norm in seen:
            continue
        right_set = _term_token_set(term)
        if not right_set:
            continue
        for _, left_set in left_sets:
            if left_set and right_set & left_set:
                out.append(term)
                seen.add(norm)
                break
        if len(out) >= limit:
            break
    return out


def _dedupe_terms(values: list[str], limit: int = 5) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        term = clean_text(raw, max_chars=48)
        if not term:
            continue
        norm = term.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _clean_alignment_terms(values: list[str], limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    blocked = {
        "new",
        "service",
        "restaurant",
        "restaurants",
        "food",
        "bar",
        "bars",
        "place",
        "places",
    }
    for raw in values:
        term = _normalize_term_phrase(raw, max_words=4)
        if not term:
            continue
        if term.startswith("(") and term.endswith(")"):
            continue
        term = re.sub(r"^liked for\s+", "", term, flags=re.I)
        term = re.sub(r"^praised for\s+", "", term, flags=re.I)
        term = clean_text(term, max_chars=48)
        tokens = _term_token_set(term)
        if not tokens:
            continue
        if term.lower() in _GENERIC_CATEGORY_TERMS:
            continue
        if len(tokens) == 1 and next(iter(tokens)) in blocked:
            continue
        norm = "|".join(sorted(tokens)) if tokens else term.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(term)
        if len(out) >= max(limit * 2, 8):
            break
    pruned: list[str] = []
    token_sets = [(term, _term_token_set(term)) for term in out]
    for term, tokens in token_sets:
        if tokens and any(tokens < other_tokens for other_term, other_tokens in token_sets if other_term != term and other_tokens):
            continue
        pruned.append(term)
        if len(pruned) >= limit:
            break
    return pruned


def _item_category_terms(primary_category: Any, categories: Any, limit: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    raw_values: list[str] = []
    primary = clean_text(primary_category, max_chars=48)
    if primary:
        raw_values.append(primary)
    cat_text = clean_text(categories, max_chars=240)
    if cat_text:
        raw_values.extend([piece for piece in re.split(r"[,/|;]+", cat_text) if clean_text(piece)])
    for raw in raw_values:
        term = clean_text(str(raw).replace("_", " "), max_chars=48)
        if not term:
            continue
        norm = term.lower()
        if norm in _GENERIC_CATEGORY_TERMS or norm in seen:
            continue
        seen.add(norm)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _item_theme_terms(
    *,
    primary_category: Any = "",
    categories: Any = "",
    top_pos_tags: Any = "",
    item_review_summary: Any = "",
    limit: int = 8,
) -> list[str]:
    out: list[str] = []
    out.extend(split_tags(top_pos_tags, limit=max(limit, 6)))
    out.extend(_structured_semantic_terms(item_review_summary, limit=max(limit, 8)))
    out.extend(_item_category_terms(primary_category, categories, limit=max(2, limit // 2)))
    return _clean_alignment_terms(out, limit=limit)


def _filter_user_focus_terms(values: list[str], limit: int = 5) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    blocked_phrases = {
        "service",
        "service bad",
        "service good",
        "value bad",
        "value good",
        "atmosphere good",
        "atmosphere noisy",
        "cleanliness bad",
        "cleanliness good",
        "wait long",
        "yet",
    }
    for raw in values:
        term = clean_text(raw, max_chars=48)
        if not term:
            continue
        tokens = _term_token_set(term)
        if len(tokens) == 1 and next(iter(tokens)) in _LOW_SIGNAL_USER_FOCUS_TOKENS:
            continue
        norm = term.lower()
        if norm in blocked_phrases:
            continue
        if re.match(r"^(prefers|usually|often|common|typical|recent|latest|other|most|outings|near-term|likely)\b", norm):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _filter_user_avoid_terms(values: list[str], focus_terms: list[str] | None = None, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    focus_sets = [_term_token_set(term) for term in list(focus_terms or [])]
    for raw in values:
        term = clean_text(raw, max_chars=48)
        if not term:
            continue
        norm = term.lower()
        tokens = _term_token_set(term)
        if not tokens:
            continue
        if norm not in _EXPLICIT_NEGATIVE_TERMS:
            if any(tokens and tokens == focus_tokens for focus_tokens in focus_sets if focus_tokens):
                continue
            if len(tokens) == 1 and next(iter(tokens)) in _LOW_SIGNAL_USER_FOCUS_TOKENS:
                continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _novel_terms_for_summary(
    values: list[str],
    *,
    reference_texts: list[str],
    limit: int,
) -> list[str]:
    body = " ".join(
        clean_text(txt, max_chars=240).lower()
        for txt in reference_texts
        if clean_text(txt, max_chars=240)
    )
    out: list[str] = []
    for raw in values:
        term = clean_text(raw, max_chars=48)
        if not term:
            continue
        tokens = _term_token_set(term)
        if tokens and body and all(tok in body for tok in tokens):
            continue
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _history_focus_terms(raw: Any, limit: int = 6) -> list[str]:
    txt = clean_text(raw, max_chars=360)
    if not txt:
        return []
    out: list[str] = []
    for match in re.findall(r"(?:liked|praised) for ([^;|()]+)", txt, flags=re.I):
        out.extend(split_tags(match, limit=limit))
        if len(out) >= limit:
            break
    return _clean_alignment_terms(_dedupe_terms(out, limit=limit), limit=limit)


def _negative_cue_terms(*values: Any, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        txt = clean_text(raw, max_chars=360)
        if not txt:
            continue
        for pattern, label in _NEGATIVE_CUE_PATTERNS:
            if pattern.search(txt):
                norm = label.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                out.append(label)
                if len(out) >= limit:
                    return out
    return out


def _derived_avoid_terms_from_focus(focus_terms: list[str], limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in focus_terms:
        txt = clean_text(raw, max_chars=64)
        if not txt:
            continue
        for pattern, label in _PREFERENCE_CONFLICT_PATTERNS:
            if pattern.search(txt):
                norm = label.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                out.append(label)
                if len(out) >= limit:
                    return out
    return out


def _structured_avoid_terms(*values: Any, limit: int = 4) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in _structured_semantic_terms(*values, limit=max(limit * 3, 12)):
        txt = clean_text(term, max_chars=64)
        if not txt:
            continue
        for pattern, label in (_NEGATIVE_CUE_PATTERNS + _PREFERENCE_CONFLICT_PATTERNS):
            if pattern.search(txt):
                norm = label.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                out.append(label)
                if len(out) >= limit:
                    return out
    return out


def _build_user_focus_terms(
    *,
    top_pos_tags: Any = "",
    profile_text: Any = "",
    review_summary: Any = "",
    evidence_snippets: Any = "",
    history_anchors: Any = "",
    pair_evidence: Any = "",
    limit: int = 5,
) -> list[str]:
    terms = []
    terms.extend(split_tags(top_pos_tags, limit=limit))
    terms.extend(_history_focus_terms(history_anchors, limit=limit))
    terms.extend(
        _structured_semantic_terms(
            review_summary,
            evidence_snippets,
            profile_text,
            pair_evidence,
            limit=max(limit * 2, 10),
        )
    )
    return _filter_user_focus_terms(_dedupe_terms(terms, limit=max(limit * 2, 10)), limit=limit)


def _build_user_avoid_terms(
    *,
    top_neg_tags: Any = "",
    profile_text: Any = "",
    review_summary: Any = "",
    evidence_snippets: Any = "",
    pair_evidence: Any = "",
    focus_terms: list[str] | None = None,
    limit: int = 4,
) -> list[str]:
    terms = []
    terms.extend(split_tags(top_neg_tags, limit=limit))
    terms.extend(
        _negative_cue_terms(
            top_neg_tags,
            profile_text,
            review_summary,
            evidence_snippets,
            pair_evidence,
            limit=limit,
        )
    )
    terms.extend(
        _structured_avoid_terms(
            top_neg_tags,
            profile_text,
            review_summary,
            evidence_snippets,
            pair_evidence,
            limit=limit,
        )
    )
    terms.extend(_derived_avoid_terms_from_focus(list(focus_terms or []), limit=limit))
    return _filter_user_avoid_terms(
        _dedupe_terms(terms, limit=max(limit * 2, 8)),
        focus_terms=list(focus_terms or []),
        limit=limit,
    )


def _semantic_line(label: str, values: list[str], limit: int = 4) -> str:
    vals = [clean_text(v, max_chars=40) for v in values if clean_text(v, max_chars=40)]
    if not vals:
        return ""
    return f"{label}: {', '.join(vals[:limit])}"


def _semantic_line_with_fallback(label: str, values: list[str], fallback_text: str, limit: int = 4) -> str:
    line = _semantic_line(label, values, limit=limit)
    if line:
        return line
    return f"{label}: {fallback_text}"


def _semantic_sentence(label: str, prefix: str, values: list[str], fallback_text: str, limit: int = 4) -> str:
    vals = [clean_text(v, max_chars=40) for v in values if clean_text(v, max_chars=40)]
    if vals:
        return f"{label}: {prefix} {', '.join(vals[:limit])}"
    return f"{label}: {fallback_text}"


def _json_dict(raw: Any) -> dict[str, Any]:
    txt = clean_text(raw, max_chars=1200)
    if not txt:
        return {}
    try:
        obj = json.loads(txt)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _typed_terms(raw: Any, limit: int = 8) -> list[str]:
    obj = _json_dict(raw)
    out: list[str] = []
    for _, values in obj.items():
        if isinstance(values, list):
            for value in values:
                term = _normalize_term_phrase(str(value or "").replace("_", " "), max_words=4)
                if term:
                    out.append(term)
    return _dedupe_terms(out, limit=limit)


def _typed_focus_summary(raw: Any, limit_types: int = 3, limit_terms_per_type: int = 2) -> list[str]:
    obj = _json_dict(raw)
    out: list[str] = []
    for raw_key, values in obj.items():
        if not isinstance(values, list):
            continue
        key = clean_text(str(raw_key or "").replace("_", " "), max_chars=32)
        vals = _dedupe_terms(
            [clean_text(str(v or "").replace("_", " "), max_chars=32) for v in values],
            limit=limit_terms_per_type,
        )
        if key and vals:
            out.append(f"{key} ({', '.join(vals)})")
        if len(out) >= limit_types:
            break
    return out


def _usable_profile_note(raw: Any, max_chars: int = 180) -> str:
    txt = clean_text(raw, max_chars=max_chars)
    if not txt:
        return ""
    if re.search(r'"[^"]+"\s*:', txt):
        return ""
    if txt.count("{") or txt.count("["):
        return ""
    if len(txt.split()) < 6:
        return ""
    return txt


def _build_profile_summary(
    *,
    profile_text_short: Any = "",
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    top_pos_tags_by_type: Any = "",
    top_neg_tags_by_type: Any = "",
    confidence: float | None = None,
) -> str:
    parts: list[str] = []
    pos_typed = _typed_focus_summary(top_pos_tags_by_type, limit_types=3, limit_terms_per_type=2)
    neg_typed = _typed_focus_summary(top_neg_tags_by_type, limit_types=2, limit_terms_per_type=2)
    pos_simple = split_tags(top_pos_tags, limit=3)
    neg_simple = split_tags(top_neg_tags, limit=3)
    if pos_typed:
        parts.append(f"strongest preference themes are {'; '.join(pos_typed)}")
    elif pos_simple:
        parts.append(f"strongest preference themes include {', '.join(pos_simple)}")
    if neg_typed:
        parts.append(f"main avoidance themes are {'; '.join(neg_typed)}")
    elif neg_simple:
        parts.append(f"main avoidance themes include {', '.join(neg_simple)}")
    conf = _to_float(confidence)
    note = _usable_profile_note(profile_text_short, max_chars=160)
    if note:
        parts.append(note)
    if conf is not None and conf > 0.0:
        if conf >= 0.75:
            parts.append("The preference profile is stable across recent signals.")
        elif conf >= 0.45:
            parts.append("The preference profile is directionally consistent but not fully stable.")
    return _join_sentence_parts(parts, max_chars=260)


def build_user_item_match_text(
    *,
    user_top_pos_tags: Any = "",
    user_top_neg_tags: Any = "",
    user_profile_text: Any = "",
    user_top_pos_tags_by_type: Any = "",
    user_top_neg_tags_by_type: Any = "",
    user_evidence_text: Any = "",
    history_anchors: Any = "",
    user_profile_pos_text: Any = "",
    user_profile_neg_text: Any = "",
    user_long_pref_text: Any = "",
    user_recent_intent_text: Any = "",
    user_negative_avoid_text: Any = "",
    user_context_text: Any = "",
    categories: Any = "",
    primary_category: Any = "",
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
) -> str:
    user_focus_source = " || ".join(
        [
            _normalize_free_text(user_profile_pos_text, max_chars=180),
            _normalize_free_text(user_long_pref_text, max_chars=220),
            _normalize_free_text(user_recent_intent_text, max_chars=180),
            _normalize_free_text(user_context_text, max_chars=120),
        ]
    )
    user_avoid_source = " || ".join(
        [
            _normalize_free_text(user_profile_neg_text, max_chars=180),
            _normalize_free_text(user_negative_avoid_text, max_chars=160),
            _normalize_free_text(user_context_text, max_chars=120),
        ]
    )
    user_focus_terms = _build_user_focus_terms(
        top_pos_tags=user_top_pos_tags,
        profile_text=user_focus_source,
        review_summary=user_long_pref_text,
        evidence_snippets=user_recent_intent_text,
        history_anchors=history_anchors,
        limit=8,
    )
    user_focus_terms = _typed_terms(user_top_pos_tags_by_type, limit=8) + list(user_focus_terms)
    user_focus_terms = _filter_user_focus_terms(
        _dedupe_terms(user_focus_terms, limit=10),
        limit=6,
    )
    user_avoid_terms = _build_user_avoid_terms(
        top_neg_tags=user_top_neg_tags,
        profile_text=user_avoid_source,
        review_summary=user_negative_avoid_text,
        evidence_snippets=f"{user_profile_neg_text} || {user_negative_avoid_text}",
        focus_terms=user_focus_terms,
        limit=6,
    )
    user_avoid_terms = _typed_terms(user_top_neg_tags_by_type, limit=6) + list(user_avoid_terms)
    user_avoid_terms = _filter_user_avoid_terms(
        _dedupe_terms(user_avoid_terms, limit=8),
        focus_terms=user_focus_terms,
        limit=4,
    )
    item_match_terms = _item_theme_terms(
        primary_category=primary_category,
        categories=categories,
        top_pos_tags=top_pos_tags,
        item_review_summary=item_review_summary,
        limit=8,
    )
    item_conflict_terms = _clean_alignment_terms(
        split_tags(top_neg_tags, limit=6)
        + _structured_avoid_terms(top_neg_tags, item_review_summary, item_review_snippet, limit=4)
        + _negative_cue_terms(top_neg_tags, item_review_summary, item_review_snippet, limit=4),
        limit=6,
    )
    match_points = _clean_alignment_terms(
        _semantic_overlap_terms(item_match_terms, user_focus_terms, limit=4)
        + _semantic_overlap_terms(user_focus_terms, item_match_terms, limit=4),
        limit=4,
    )
    if len(match_points) == 1 and len(_term_token_set(match_points[0])) < 2:
        match_points = []
    item_conflict_cues = _negative_cue_terms(top_neg_tags, item_review_summary, item_review_snippet, limit=4)
    conflict_points = _clean_alignment_terms(
        _semantic_overlap_terms(item_match_terms, user_avoid_terms, limit=3)
        + _semantic_overlap_terms(user_avoid_terms, item_match_terms, limit=3)
        + _semantic_overlap_terms(item_conflict_terms, user_avoid_terms, limit=3)
        + _semantic_overlap_terms(item_conflict_cues, user_avoid_terms, limit=3),
        limit=4,
    )
    conflict_points = _filter_user_avoid_terms(conflict_points, focus_terms=user_focus_terms, limit=4)
    lines: list[str] = []
    if match_points:
        match_summary = _join_terms_natural(match_points, limit=4)
        if match_summary:
            lines.append(
                f"user_match_points: This business lines up with the user's recurring interests around {match_summary}."
            )
    if conflict_points:
        conflict_summary = _join_terms_natural(conflict_points, limit=4)
        if conflict_summary:
            lines.append(
                f"user_conflict_points: Potential friction comes from {conflict_summary}, which runs against the user's usual avoidances."
            )
    return "; ".join(lines)


def _build_user_focus_narrative(
    *,
    long_pref_text: str,
    recent_intent_text: str,
    focus_terms: list[str],
    context_text: str,
) -> str:
    parts: list[str] = []
    recent_clean = clean_text(recent_intent_text, max_chars=220)
    long_clean = clean_text(long_pref_text, max_chars=260)
    recent_body = _strip_sentence_lead(
        recent_clean,
        "Recent activity points to",
        "Recent outings lean toward",
        "Recent choices lean toward",
    )
    long_body = _strip_sentence_lead(
        long_clean,
        "Longer-term habits favor",
        "Long-term patterns lean toward",
    )
    focus_terms = _novel_terms_for_summary(
        focus_terms,
        reference_texts=[long_clean, recent_clean, context_text],
        limit=3,
    )
    focus_summary = _join_terms_natural(focus_terms, limit=4)
    if focus_summary:
        parts.append(f"The user consistently responds to {focus_summary}.")
    elif recent_body:
        parts.append(f"Recent choices lean toward {recent_body}.")
    elif long_body:
        parts.append(f"Longer-term habits favor {long_body}.")
    existing_body = " ".join(parts).lower()
    detail_candidates: list[tuple[str, str]] = []
    if recent_body:
        detail_candidates.append(("Recent choices lean toward", recent_body))
    if long_body:
        detail_candidates.append(("Longer-term habits favor", long_body))
    best_detail: tuple[str, str] | None = None
    best_novel = -1
    for prefix, body in detail_candidates:
        body_tokens = _term_token_set(body)
        if body_tokens and existing_body and all(tok in existing_body for tok in body_tokens):
            continue
        novel = sum(1 for tok in body_tokens if tok not in existing_body) if body_tokens else 0
        if novel > best_novel:
            best_novel = novel
            best_detail = (prefix, body)
    if best_detail:
        parts.append(f"{best_detail[0]} {best_detail[1]}.")
    if context_text and clean_text(context_text, max_chars=160).lower() not in " ".join(parts).lower():
        parts.append(context_text)
    return _join_sentence_parts(parts, max_chars=500)


def _build_user_avoid_narrative(
    *,
    negative_avoid_text: str,
    avoid_terms: list[str],
) -> str:
    parts: list[str] = []
    negative_avoid_lower = clean_text(negative_avoid_text, max_chars=240).lower()
    keep_negative_avoid_text = bool(
        negative_avoid_lower
        and (
            any(term in negative_avoid_lower for term in _EXPLICIT_NEGATIVE_TERMS)
            or any(
                cue in negative_avoid_lower
                for cue in ("wait", "service", "value", "clean", "noise", "crowd")
            )
        )
    )
    if negative_avoid_text and keep_negative_avoid_text:
        avoid_body = _strip_sentence_lead(
            negative_avoid_text,
            "The user tends to avoid",
            "The user usually steers away from",
        )
        if avoid_body:
            parts.append(f"The user tends to avoid {avoid_body}.")
    avoid_terms = _novel_terms_for_summary(
        avoid_terms,
        reference_texts=[negative_avoid_text],
        limit=3,
    )
    avoid_summary = _join_terms_natural(avoid_terms, limit=3)
    if avoid_summary:
        existing_body = " ".join(parts).lower()
        avoid_tokens = _term_token_set(avoid_summary)
        if not (avoid_tokens and existing_body and all(tok in existing_body for tok in avoid_tokens)):
            parts.append(f"Likely friction comes from {avoid_summary}.")
    return _join_sentence_parts(parts, max_chars=280)


def build_user_text(
    profile_text: Any,
    profile_text_short: Any = "",
    profile_text_long: Any = "",
    profile_pos_text: Any = "",
    profile_neg_text: Any = "",
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    top_pos_tags_by_type: Any = "",
    top_neg_tags_by_type: Any = "",
    confidence: float | None = None,
    review_summary: Any = "",
    review_raw_snippet: Any = "",
    evidence_snippets: Any = "",
    history_anchors: Any = "",
    pair_evidence: Any = "",
    user_long_pref_text: Any = "",
    user_recent_intent_text: Any = "",
    user_negative_avoid_text: Any = "",
    user_context_text: Any = "",
    stable_preferences_text: Any = "",
    recent_intent_text_v2: Any = "",
    avoidance_text_v2: Any = "",
    history_anchor_hint_text: Any = "",
    user_semantic_profile_text_v2: Any = "",
) -> str:
    parts: list[str] = []
    stable_pref_asset = clean_text(stable_preferences_text, max_chars=320)
    recent_pref_asset = clean_text(recent_intent_text_v2, max_chars=240)
    avoid_asset = clean_text(avoidance_text_v2, max_chars=220)
    history_asset = clean_text(history_anchor_hint_text, max_chars=220)
    profile_asset = clean_text(user_semantic_profile_text_v2, max_chars=500)
    long_pref_natural = naturalize_user_long_pref_text(user_long_pref_text, max_chars=220)
    recent_intent_natural = naturalize_user_recent_intent_text(user_recent_intent_text, max_chars=180)
    negative_avoid_natural = naturalize_user_negative_avoid_text(user_negative_avoid_text, max_chars=160)
    context_natural = naturalize_user_context_text(user_context_text, max_chars=120)
    hist_txt = build_history_anchor_summary(history_anchors, max_items=3, max_chars=220)
    focus_term_source = " || ".join(
        [
            recent_intent_natural,
            long_pref_natural,
            context_natural,
        ]
    )
    avoid_term_source = " || ".join(
        [
            negative_avoid_natural,
            context_natural,
            _normalize_free_text(profile_neg_text, max_chars=120),
        ]
    )
    pos_terms = _build_user_focus_terms(
        top_pos_tags=top_pos_tags,
        profile_text=focus_term_source,
        review_summary=f"{long_pref_natural} || {recent_intent_natural}",
        evidence_snippets="",
        history_anchors=history_anchors,
        pair_evidence="",
        limit=5,
    )
    pos_terms = _typed_terms(top_pos_tags_by_type, limit=6) + list(pos_terms)
    pos_terms = _filter_user_focus_terms(
        _dedupe_terms(pos_terms, limit=7),
        limit=4,
    )
    neg_terms = _build_user_avoid_terms(
        top_neg_tags=top_neg_tags,
        profile_text=avoid_term_source,
        review_summary=negative_avoid_natural,
        evidence_snippets="",
        pair_evidence="",
        focus_terms=pos_terms,
        limit=4,
    )
    neg_terms = _typed_terms(top_neg_tags_by_type, limit=5) + list(neg_terms)
    neg_terms = _filter_user_avoid_terms(
        _dedupe_terms(neg_terms, limit=6),
        focus_terms=pos_terms,
        limit=3,
    )
    rebuilt_evidence = build_clean_user_evidence_text(
        profile_text_short=profile_text_short or profile_text,
        profile_text_long=profile_text_long or profile_text,
        profile_pos_text=profile_pos_text,
        profile_neg_text=profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        max_chars=320,
    )
    evidence = _join_semantic_parts(
        [
            _clean_user_evidence_asset(evidence_snippets, max_chars=220),
            _clean_user_evidence_asset(pair_evidence, max_chars=180),
            _normalize_free_text(review_summary, max_chars=180),
            rebuilt_evidence,
        ],
        max_chars=320,
    )
    if stable_pref_asset:
        parts.append(f"user_focus: {stable_pref_asset}")
    elif profile_asset:
        parts.append(f"user_focus: {profile_asset}")
    focus_block = _build_user_focus_narrative(
        long_pref_text=long_pref_natural,
        recent_intent_text=recent_intent_natural,
        focus_terms=pos_terms,
        context_text=context_natural,
    )
    if focus_block and not stable_pref_asset:
        parts.append(f"user_focus: {focus_block}")
    if recent_pref_asset and recent_pref_asset.lower() not in " ".join(parts).lower():
        parts.append(f"recent_intent: {recent_pref_asset}")
    avoid_block = _build_user_avoid_narrative(
        negative_avoid_text=negative_avoid_natural,
        avoid_terms=neg_terms,
    )
    if avoid_asset:
        parts.append(f"user_avoid: {avoid_asset}")
    elif avoid_block:
        parts.append(f"user_avoid: {avoid_block}")
    if history_asset:
        parts.append(f"history_pattern: {history_asset}")
    elif hist_txt:
        parts.append(f"history_pattern: {hist_txt}")
    if evidence and len(evidence.split()) >= 5:
        parts.append(f"user_evidence: {evidence}")
    return _join_semantic_parts(parts, max_chars=900)


def build_item_text(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    source_set: Any = "",
    user_segment: Any = "",
    als_rank: float | None = None,
    cluster_rank: float | None = None,
    profile_rank: float | None = None,
    popular_rank: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
) -> str:
    parts: list[str] = []
    n = _clean(name)
    if n:
        parts.append(f"name: {n}")
    c = _clean(city)
    if c:
        parts.append(f"city: {c}")
    pc = _clean(primary_category)
    if pc:
        parts.append(f"primary_category: {pc}")
    cat = _clean(categories)
    if cat:
        parts.append(f"categories: {cat}")
    pos = _take_tags(top_pos_tags)
    if pos:
        parts.append(f"item_strengths: {pos}")
    neg = _take_tags(top_neg_tags)
    if neg:
        parts.append(f"item_weaknesses: {neg}")

    sem_summary = build_item_semantic_summary(top_pos_tags, top_neg_tags, semantic_score, semantic_confidence)
    if sem_summary:
        parts.append(f"item_semantics: {sem_summary}")

    src = _take_tags(source_set)
    if src:
        parts.append(f"candidate_sources: {src}")
    seg = _clean(user_segment)
    if seg:
        parts.append(f"user_segment: {seg}")

    als = _to_float(als_rank)
    if als is not None and als > 0:
        parts.append(f"als_rank: {int(round(als))}")
    cr = _to_float(cluster_rank)
    if cr is not None and cr > 0:
        parts.append(f"cluster_rank: {int(round(cr))}")
    pr = _to_float(profile_rank)
    if pr is not None and pr > 0:
        parts.append(f"profile_rank: {int(round(pr))}")
    pop = _to_float(popular_rank)
    if pop is not None and pop > 0:
        parts.append(f"popular_rank: {int(round(pop))}")

    sem_sup = _to_float(semantic_support)
    if sem_sup is not None and sem_sup > 0:
        parts.append(f"semantic_support: {sem_sup:.3f}")
    sem_rich = _to_float(semantic_tag_richness)
    if sem_rich is not None and sem_rich > 0:
        parts.append(f"semantic_tag_richness: {sem_rich:.3f}")

    tw = _to_float(tower_score)
    if tw is not None and abs(tw) > 1e-9:
        parts.append(f"tower_score: {tw:.3f}")
    seq = _to_float(seq_score)
    if seq is not None and abs(seq) > 1e-9:
        parts.append(f"seq_score: {seq:.3f}")

    clabel = _clean(cluster_label_for_recsys)
    if clabel:
        parts.append(f"cluster_label: {clabel}")
    cid = _clean(cluster_for_recsys)
    if cid:
        parts.append(f"cluster_id: {cid}")

    irs = clean_text(item_review_summary, max_chars=160)
    if irs and "recent_item_reviews=" not in irs:
        parts.append(f"item_signal: {irs}")
    irr = merge_distinct_segments(item_review_snippet, max_segments=2, max_chars=260)
    if irr:
        parts.append(f"item_evidence: {irr}")
    return "; ".join(parts)


def build_binary_prompt(user_text: str, item_text: str) -> str:
    u = _clean(user_text)
    i = _clean(item_text)
    return (
        "You are a recommendation assistant. "
        "Given user preference evidence and one candidate business, "
        "predict whether the user will positively interact with this business. "
        "Answer only YES or NO.\n"
        f"User: {u}\n"
        f"Candidate: {i}\n"
        "Answer:"
    )


def build_item_text_semantic(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
) -> str:
    parts: list[str] = []
    n = _clean(name)
    if n:
        parts.append(f"name: {n}")
    c = _clean(city)
    if c:
        parts.append(f"city: {c}")
    pc = _clean(primary_category)
    if pc:
        parts.append(f"primary_category: {pc}")
    cat = _clean(categories)
    if cat:
        parts.append(f"categories: {cat}")
    pos = _take_tags(top_pos_tags)
    if pos:
        parts.append(f"item_strengths: {pos}")
    neg = _take_tags(top_neg_tags)
    if neg:
        parts.append(f"item_weaknesses: {neg}")

    sem_summary = build_item_semantic_summary(top_pos_tags, top_neg_tags, semantic_score, semantic_confidence)
    if sem_summary:
        parts.append(f"item_semantics: {sem_summary}")

    sem_sup = _to_float(semantic_support)
    if sem_sup is not None and sem_sup > 0:
        parts.append(f"semantic_support: {sem_sup:.3f}")
    sem_rich = _to_float(semantic_tag_richness)
    if sem_rich is not None and sem_rich > 0:
        parts.append(f"semantic_tag_richness: {sem_rich:.3f}")
    tw = _to_float(tower_score)
    if tw is not None and abs(tw) > 1e-9:
        parts.append(f"tower_score: {tw:.3f}")
    seq = _to_float(seq_score)
    if seq is not None and abs(seq) > 1e-9:
        parts.append(f"seq_score: {seq:.3f}")
    clabel = _clean(cluster_label_for_recsys)
    if clabel:
        parts.append(f"cluster_label: {clabel}")
    cid = _clean(cluster_for_recsys)
    if cid:
        parts.append(f"cluster_id: {cid}")

    irs = clean_text(item_review_summary, max_chars=160)
    if irs and "recent_item_reviews=" not in irs:
        parts.append(f"item_signal: {irs}")
    irr = merge_distinct_segments(item_review_snippet, max_segments=2, max_chars=260)
    if irr:
        parts.append(f"item_evidence: {irr}")
    return "; ".join(parts)


def build_item_text_full_lite(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    pre_rank: Any = None,
    pre_score: Any = None,
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    source_set: Any = "",
    user_segment: Any = "",
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    pair_evidence_summary: Any = "",
) -> str:
    parts: list[str] = []
    n = _clean(name)
    if n:
        parts.append(f"name: {n}")
    c = _clean(city)
    if c:
        parts.append(f"city: {c}")
    pc = _clean(primary_category)
    if pc:
        parts.append(f"primary_category: {pc}")
    cat = _clean(categories)
    if cat:
        parts.append(f"categories: {cat}")
    pos = _take_tags(top_pos_tags)
    if pos:
        parts.append(f"item_strengths: {pos}")
    neg = _take_tags(top_neg_tags)
    if neg:
        parts.append(f"item_weaknesses: {neg}")

    sem_summary = build_item_semantic_summary(top_pos_tags, top_neg_tags, semantic_score, semantic_confidence)
    if sem_summary:
        parts.append(f"item_semantics: {sem_summary}")

    competition_summary = build_candidate_competition_summary(
        pre_rank,
        pre_score,
        group_gap_rank_pct=group_gap_rank_pct,
        group_gap_to_top3=group_gap_to_top3,
        group_gap_to_top10=group_gap_to_top10,
        net_score_rank_pct=net_score_rank_pct,
    )
    if competition_summary:
        parts.append(f"competition_context: {competition_summary}")

    fit_risk_summary = build_candidate_fit_risk_summary(
        pair_evidence_summary,
        item_neg_tags=top_neg_tags,
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        source_set=source_set,
        user_segment=user_segment,
    )
    if fit_risk_summary:
        parts.append(f"fit_risk: {fit_risk_summary}")

    sem_sup = _to_float(semantic_support)
    if sem_sup is not None and sem_sup > 0:
        parts.append(f"semantic_support: {sem_sup:.3f}")
    sem_rich = _to_float(semantic_tag_richness)
    if sem_rich is not None and sem_rich > 0:
        parts.append(f"semantic_tag_richness: {sem_rich:.3f}")
    clabel = _clean(cluster_label_for_recsys)
    if clabel:
        parts.append(f"cluster_label: {clabel}")

    irs = clean_text(item_review_summary, max_chars=160)
    if irs and "recent_item_reviews=" not in irs:
        parts.append(f"item_signal: {irs}")
    irr = merge_distinct_segments(item_review_snippet, max_segments=2, max_chars=260)
    if irr:
        parts.append(f"item_evidence: {irr}")
    return "; ".join(parts)


def build_item_text_semantic_compact(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    channel_preference_core_v1: Any = None,
    channel_recent_intent_v1: Any = None,
    channel_context_time_v1: Any = None,
    channel_conflict_v1: Any = None,
    channel_evidence_support_v1: Any = None,
) -> str:
    parts: list[str] = []
    base = build_item_text_semantic(
        name=name,
        city=city,
        categories=categories,
        primary_category=primary_category,
        top_pos_tags=top_pos_tags,
        top_neg_tags=top_neg_tags,
        semantic_score=semantic_score,
        semantic_confidence=semantic_confidence,
        semantic_support=semantic_support,
        semantic_tag_richness=semantic_tag_richness,
        tower_score=tower_score,
        seq_score=seq_score,
        cluster_for_recsys=cluster_for_recsys,
        cluster_label_for_recsys=cluster_label_for_recsys,
        item_review_summary=item_review_summary,
        item_review_snippet=item_review_snippet,
    )
    if base:
        parts.append(base)

    competition_hint = build_candidate_compact_competition_hint(
        group_gap_rank_pct=group_gap_rank_pct,
        group_gap_to_top3=group_gap_to_top3,
        group_gap_to_top10=group_gap_to_top10,
        net_score_rank_pct=net_score_rank_pct,
    )
    if competition_hint:
        parts.append(f"competition_hint: {competition_hint}")

    risk_hint = build_candidate_compact_risk_hint(
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
    )
    if risk_hint:
        parts.append(f"risk_hint: {risk_hint}")

    channel_hint = build_candidate_compact_channel_hint(
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
        evidence_support=channel_evidence_support_v1,
    )
    if channel_hint:
        parts.append(f"channel_hint: {channel_hint}")

    return "; ".join(parts)


def build_item_text_semantic_compact_preserve(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    channel_preference_core_v1: Any = None,
    channel_recent_intent_v1: Any = None,
    channel_context_time_v1: Any = None,
    channel_conflict_v1: Any = None,
    channel_evidence_support_v1: Any = None,
    source_set: Any = "",
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    context_rank: Any = None,
) -> str:
    parts: list[str] = []
    base = build_item_text_semantic_compact(
        name=name,
        city=city,
        categories=categories,
        primary_category=primary_category,
        top_pos_tags=top_pos_tags,
        top_neg_tags=top_neg_tags,
        semantic_score=semantic_score,
        semantic_confidence=semantic_confidence,
        semantic_support=semantic_support,
        semantic_tag_richness=semantic_tag_richness,
        tower_score=tower_score,
        seq_score=seq_score,
        cluster_for_recsys=cluster_for_recsys,
        cluster_label_for_recsys=cluster_label_for_recsys,
        item_review_summary=item_review_summary,
        item_review_snippet=item_review_snippet,
        group_gap_rank_pct=group_gap_rank_pct,
        group_gap_to_top3=group_gap_to_top3,
        group_gap_to_top10=group_gap_to_top10,
        net_score_rank_pct=net_score_rank_pct,
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        channel_preference_core_v1=channel_preference_core_v1,
        channel_recent_intent_v1=channel_recent_intent_v1,
        channel_context_time_v1=channel_context_time_v1,
        channel_conflict_v1=channel_conflict_v1,
        channel_evidence_support_v1=channel_evidence_support_v1,
    )
    if base:
        parts.append(base)

    route_diversity_hint = build_candidate_compact_route_diversity_hint(
        source_set=source_set,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        context_rank=context_rank,
    )
    if route_diversity_hint:
        parts.append(f"route_diversity_hint: {route_diversity_hint}")

    support_reliability_hint = build_candidate_compact_support_reliability_hint(
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
    )
    if support_reliability_hint:
        parts.append(f"support_reliability_hint: {support_reliability_hint}")

    stable_fit_hint = build_candidate_compact_stable_fit_hint(
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if stable_fit_hint:
        parts.append(f"stable_fit_hint: {stable_fit_hint}")

    return "; ".join(parts)


def build_item_text_semantic_compact_targeted(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    pre_rank: Any = None,
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    channel_preference_core_v1: Any = None,
    channel_recent_intent_v1: Any = None,
    channel_context_time_v1: Any = None,
    channel_conflict_v1: Any = None,
    channel_evidence_support_v1: Any = None,
    source_set: Any = "",
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    context_rank: Any = None,
) -> str:
    parts: list[str] = []
    base = build_item_text_semantic_compact(
        name=name,
        city=city,
        categories=categories,
        primary_category=primary_category,
        top_pos_tags=top_pos_tags,
        top_neg_tags=top_neg_tags,
        semantic_score=semantic_score,
        semantic_confidence=semantic_confidence,
        semantic_support=semantic_support,
        semantic_tag_richness=semantic_tag_richness,
        tower_score=tower_score,
        seq_score=seq_score,
        cluster_for_recsys=cluster_for_recsys,
        cluster_label_for_recsys=cluster_label_for_recsys,
        item_review_summary=item_review_summary,
        item_review_snippet=item_review_snippet,
        group_gap_rank_pct=group_gap_rank_pct,
        group_gap_to_top3=group_gap_to_top3,
        group_gap_to_top10=group_gap_to_top10,
        net_score_rank_pct=net_score_rank_pct,
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        channel_preference_core_v1=channel_preference_core_v1,
        channel_recent_intent_v1=channel_recent_intent_v1,
        channel_context_time_v1=channel_context_time_v1,
        channel_conflict_v1=channel_conflict_v1,
        channel_evidence_support_v1=channel_evidence_support_v1,
    )
    if base:
        parts.append(base)

    rank_role_hint = build_candidate_compact_rank_role_hint(
        pre_rank=pre_rank,
        group_gap_to_top10=group_gap_to_top10,
    )
    if rank_role_hint:
        parts.append(f"rank_role_hint: {rank_role_hint}")

    route_profile_hint = build_candidate_compact_route_profile_hint(
        source_set=source_set,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        context_rank=context_rank,
    )
    if route_profile_hint:
        parts.append(f"route_profile_hint: {route_profile_hint}")

    support_profile_hint = build_candidate_compact_support_profile_hint(
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
    )
    if support_profile_hint:
        parts.append(f"support_profile_hint: {support_profile_hint}")

    stability_profile_hint = build_candidate_compact_stability_profile_hint(
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if stability_profile_hint:
        parts.append(f"stability_profile_hint: {stability_profile_hint}")

    return "; ".join(parts)


def build_item_text_semantic_compact_boundary(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    learned_rank: Any = None,
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    channel_preference_core_v1: Any = None,
    channel_recent_intent_v1: Any = None,
    channel_context_time_v1: Any = None,
    channel_conflict_v1: Any = None,
    channel_evidence_support_v1: Any = None,
    source_set: Any = "",
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    context_rank: Any = None,
) -> str:
    parts: list[str] = []
    base = build_item_text_semantic_compact(
        name=name,
        city=city,
        categories=categories,
        primary_category=primary_category,
        top_pos_tags=top_pos_tags,
        top_neg_tags=top_neg_tags,
        semantic_score=semantic_score,
        semantic_confidence=semantic_confidence,
        semantic_support=semantic_support,
        semantic_tag_richness=semantic_tag_richness,
        tower_score=tower_score,
        seq_score=seq_score,
        cluster_for_recsys=cluster_for_recsys,
        cluster_label_for_recsys=cluster_label_for_recsys,
        item_review_summary=item_review_summary,
        item_review_snippet=item_review_snippet,
        group_gap_rank_pct=group_gap_rank_pct,
        group_gap_to_top3=group_gap_to_top3,
        group_gap_to_top10=group_gap_to_top10,
        net_score_rank_pct=net_score_rank_pct,
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        channel_preference_core_v1=channel_preference_core_v1,
        channel_recent_intent_v1=channel_recent_intent_v1,
        channel_context_time_v1=channel_context_time_v1,
        channel_conflict_v1=channel_conflict_v1,
        channel_evidence_support_v1=channel_evidence_support_v1,
    )
    if base:
        parts.append(base)

    rescue_band_hint = build_candidate_compact_rescue_band_hint(
        learned_rank=learned_rank,
    )
    if rescue_band_hint:
        parts.append(f"rescue_band_hint: {rescue_band_hint}")

    head_guard_hint = build_candidate_compact_head_guard_hint(
        learned_rank=learned_rank,
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if head_guard_hint:
        parts.append(f"head_guard_hint: {head_guard_hint}")

    boundary_gap_hint = build_candidate_compact_boundary_gap_hint(
        learned_rank=learned_rank,
        group_gap_to_top10=group_gap_to_top10,
        group_gap_to_top3=group_gap_to_top3,
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if boundary_gap_hint:
        parts.append(f"boundary_gap_hint: {boundary_gap_hint}")

    route_profile_hint = build_candidate_compact_route_profile_hint(
        source_set=source_set,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        context_rank=context_rank,
    )
    if route_profile_hint:
        parts.append(f"route_profile_hint: {route_profile_hint}")

    support_profile_hint = build_candidate_compact_support_profile_hint(
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
    )
    if support_profile_hint:
        parts.append(f"support_profile_hint: {support_profile_hint}")

    stability_profile_hint = build_candidate_compact_stability_profile_hint(
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if stability_profile_hint:
        parts.append(f"downside_profile_hint: {stability_profile_hint}")

    return "; ".join(parts)


def build_item_text_semantic_compact_boundary_compare(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
    cluster_for_recsys: Any = "",
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    learned_rank: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    channel_preference_core_v1: Any = None,
    channel_recent_intent_v1: Any = None,
    channel_context_time_v1: Any = None,
    channel_conflict_v1: Any = None,
    channel_evidence_support_v1: Any = None,
    source_set: Any = "",
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    context_rank: Any = None,
) -> str:
    parts: list[str] = []
    base = build_item_text_semantic(
        name=name,
        city=city,
        categories=categories,
        primary_category=primary_category,
        top_pos_tags=top_pos_tags,
        top_neg_tags=top_neg_tags,
        semantic_score=semantic_score,
        semantic_confidence=semantic_confidence,
        semantic_support=semantic_support,
        semantic_tag_richness=semantic_tag_richness,
        tower_score=tower_score,
        seq_score=seq_score,
        cluster_for_recsys=cluster_for_recsys,
        cluster_label_for_recsys=cluster_label_for_recsys,
        item_review_summary=item_review_summary,
        item_review_snippet=item_review_snippet,
    )
    if base:
        parts.append(base)

    head_guard_hint = build_candidate_compact_head_guard_hint(
        learned_rank=learned_rank,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if head_guard_hint:
        parts.append(f"head_guard_hint: {head_guard_hint}")

    boundary_gap_hint = build_candidate_compact_boundary_gap_hint(
        learned_rank=learned_rank,
        group_gap_to_top10=group_gap_to_top10,
        group_gap_to_top3=group_gap_to_top3,
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if boundary_gap_hint:
        parts.append(f"boundary_gap_hint: {boundary_gap_hint}")

    route_profile_hint = build_candidate_compact_route_profile_hint(
        source_set=source_set,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
        context_rank=context_rank,
    )
    if route_profile_hint:
        parts.append(f"route_profile_hint: {route_profile_hint}")

    support_profile_hint = build_candidate_compact_support_profile_hint(
        semantic_support=semantic_support,
        evidence_support=channel_evidence_support_v1,
        source_count=source_count,
        nonpopular_source_count=nonpopular_source_count,
        profile_cluster_source_count=profile_cluster_source_count,
    )
    if support_profile_hint:
        parts.append(f"support_profile_hint: {support_profile_hint}")

    stability_profile_hint = build_candidate_compact_stability_profile_hint(
        avoid_neg=avoid_neg,
        avoid_core=avoid_core,
        conflict_gap=conflict_gap,
        preference_core=channel_preference_core_v1,
        recent_intent=channel_recent_intent_v1,
        context_time=channel_context_time_v1,
        channel_conflict=channel_conflict_v1,
    )
    if stability_profile_hint:
        parts.append(f"downside_profile_hint: {stability_profile_hint}")

    return "; ".join(parts)


def build_blocker_comparison_prompt(
    user_text: str,
    candidate_a_text: str,
    candidate_b_text: str,
    *,
    ranking_context: str = "",
    comparison_summary: str = "",
    candidate_a_role: str = "",
    candidate_b_role: str = "",
) -> str:
    u = _clean(user_text)
    a = _clean(candidate_a_text)
    b = _clean(candidate_b_text)
    ctx = _clean(ranking_context)
    a_role = _clean(candidate_a_role)
    b_role = _clean(candidate_b_role)
    a_label = "Candidate A"
    b_label = "Candidate B"
    if a_role:
        a_label = f"{a_label} ({a_role})"
    if b_role:
        b_label = f"{b_label} ({b_role})"
    lines = [
        "You are a recommendation ranking judge.",
        "Compare two candidate businesses for the same user and decide whether Candidate A should rank above Candidate B.",
        "Candidate order is arbitrary. Base the decision on the user's focus and avoidance patterns, the user's history, and each business's semantic match, conflict, category, and review evidence.",
        "Current rank is context only and is not by itself the answer.",
        f"User: {u}",
    ]
    if ctx:
        lines.append(f"Ranking context: {ctx}")
    lines.extend(
        [
            f"{a_label}: {a}",
            f"{b_label}: {b}",
            "Question: Which candidate should rank higher for this user in this local rescue context?",
            "Answer:",
        ]
    )
    return "\n".join(lines)


def build_local_listwise_ranking_prompt(
    user_text: str,
    focus_candidate_text: str,
    rival_candidates: list[tuple[str, str]],
    *,
    ranking_context: str = "",
    local_slate_summary: str = "",
    focus_summary: str = "",
    focus_role: str = "",
) -> str:
    u = _clean(user_text)
    focus = _join_semantic_parts([p.strip() for p in str(focus_candidate_text or "").split(";")], max_chars=_LOCAL_LISTWISE_ITEM_MAX_CHARS)
    if not focus:
        focus = _normalize_free_text(focus_candidate_text, max_chars=min(220, _LOCAL_LISTWISE_ITEM_MAX_CHARS))
    ctx = _clean(ranking_context)
    slate = _clean(local_slate_summary)
    focus_cmp = _normalize_free_text(focus_summary, max_chars=260)
    role = _clean(focus_role)
    focus_label = "Focus candidate"
    if role:
        focus_label = f"{focus_label} ({role})"
    lines = [
        "You are a recommendation ranking judge.",
        "Evaluate one focus candidate inside a small local rescue slate for the same user.",
        "Base the judgment on the user's focus and avoidance patterns, the user's history, and each business's semantic match, conflict, category, and review evidence.",
        "Current rank is context only and is not by itself the answer.",
        f"User: {u}",
    ]
    if ctx:
        lines.append(f"Local ranking context: {ctx}")
    if slate:
        lines.append(f"Local slate summary: {slate}")
    lines.append(f"{focus_label}: {focus}")
    if focus_cmp:
        lines.append(f"Why the focus may beat nearby rivals: {focus_cmp}")
    for idx, (rival_role, rival_text) in enumerate(rival_candidates, start=1):
        rival_label = f"Rival {idx}"
        rr = _clean(rival_role)
        if rr:
            rival_label = f"{rival_label} ({rr})"
        rival_compact = _join_semantic_parts([p.strip() for p in str(rival_text or "").split(";")], max_chars=_LOCAL_LISTWISE_ITEM_MAX_CHARS)
        if not rival_compact:
            rival_compact = _normalize_free_text(rival_text, max_chars=min(220, _LOCAL_LISTWISE_ITEM_MAX_CHARS))
        if not rival_compact:
            continue
        lines.append(f"{rival_label}: {rival_compact}")
    lines.extend(
        [
            "Question: Should the focus candidate rank above these local blockers for this user?",
            "Answer:",
        ]
    )
    return "\n".join(lines)


def build_item_text_sft_clean(
    name: Any,
    city: Any,
    categories: Any,
    primary_category: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    semantic_score: float | None = None,
    semantic_confidence: float | None = None,
    cluster_label_for_recsys: Any = "",
    item_review_summary: Any = "",
    item_review_snippet: Any = "",
    user_profile_text: Any = "",
    user_top_pos_tags_by_type: Any = "",
    user_top_neg_tags_by_type: Any = "",
    user_top_pos_tags: Any = "",
    user_top_neg_tags: Any = "",
    user_evidence_text: Any = "",
    history_anchor_text: Any = "",
    user_profile_pos_text: Any = "",
    user_profile_neg_text: Any = "",
    user_long_pref_text: Any = "",
    user_recent_intent_text: Any = "",
    user_negative_avoid_text: Any = "",
    user_context_text: Any = "",
    core_offering_text: Any = "",
    scene_fit_text: Any = "",
    strengths_text: Any = "",
    risk_points_text: Any = "",
    merchant_semantic_profile_text_v2: Any = "",
    fit_reasons_text_v1: Any = "",
    friction_reasons_text_v1: Any = "",
    evidence_basis_text_v1: Any = "",
) -> str:
    parts: list[str] = []
    core_offering_asset = clean_text(core_offering_text, max_chars=220)
    scene_fit_asset = clean_text(scene_fit_text, max_chars=180)
    strengths_asset = clean_text(strengths_text, max_chars=180)
    risk_points_asset = clean_text(risk_points_text, max_chars=180)
    merchant_profile_asset = clean_text(merchant_semantic_profile_text_v2, max_chars=320)
    fit_reasons_asset = clean_text(fit_reasons_text_v1, max_chars=220)
    friction_reasons_asset = clean_text(friction_reasons_text_v1, max_chars=180)
    evidence_basis_asset = clean_text(evidence_basis_text_v1, max_chars=180)
    n = _clean(name)
    c = _clean(city)
    pc = _clean(primary_category)
    cat_terms = _item_category_terms(primary_category, categories, limit=3)
    profile_bits: list[str] = []
    if pc and pc.lower() in _GENERIC_CATEGORY_TERMS and cat_terms:
        pc = cat_terms[0]
        cat_terms = [t for t in cat_terms if t.lower() != pc.lower()]
    if n and pc and c:
        profile_bits.append(f"{n} is a {pc} option in {c}")
    elif n and pc:
        profile_bits.append(f"{n} is mainly known for {pc}")
    elif n and c:
        profile_bits.append(f"{n} is located in {c}")
    elif n:
        profile_bits.append(n)
    if cat_terms:
        broader = _join_terms_natural([t for t in cat_terms if t.lower() != pc.lower()], limit=2)
        if broader:
            profile_bits.append(f"It also leans toward {broader}")
    if core_offering_asset:
        parts.append(f"business_profile: {core_offering_asset}")
    elif profile_bits:
        parts.append(f"business_profile: {clean_text('. '.join(profile_bits), max_chars=160)}")
    elif merchant_profile_asset:
        parts.append(f"business_profile: {merchant_profile_asset}")

    match_text = build_user_item_match_text(
        user_top_pos_tags=user_top_pos_tags,
        user_top_neg_tags=user_top_neg_tags,
        user_top_pos_tags_by_type=user_top_pos_tags_by_type,
        user_top_neg_tags_by_type=user_top_neg_tags_by_type,
        user_profile_text=user_profile_text,
        user_evidence_text=user_evidence_text,
        history_anchors=history_anchor_text,
        user_profile_pos_text=user_profile_pos_text,
        user_profile_neg_text=user_profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        categories=categories,
        primary_category=primary_category,
        top_pos_tags=top_pos_tags,
        top_neg_tags=top_neg_tags,
        item_review_summary=item_review_summary,
        item_review_snippet=item_review_snippet,
    )
    if fit_reasons_asset:
        parts.append(f"user_match_points: {fit_reasons_asset}")
    elif match_text:
        parts.append(match_text)
    if friction_reasons_asset:
        parts.append(f"user_conflict_points: {friction_reasons_asset}")
    if scene_fit_asset:
        parts.append(f"business_scene: {scene_fit_asset}")

    pos = _take_tags(top_pos_tags)
    if strengths_asset:
        parts.append(f"item_strengths: {strengths_asset}")
    elif pos:
        parts.append(f"item_strengths: Reviews and tags repeatedly highlight {pos}.")
    neg = _take_tags(top_neg_tags)
    if risk_points_asset:
        parts.append(f"item_weaknesses: {risk_points_asset}")
    elif neg:
        parts.append(f"item_weaknesses: Available text also flags possible issues around {neg}.")

    irs = clean_text(item_review_summary, max_chars=160)
    if evidence_basis_asset:
        parts.append(f"evidence_basis: {evidence_basis_asset}")
    if irs and "recent_item_reviews=" not in irs:
        parts.append(f"item_signal: {irs}")
    irr = _clean_item_evidence_text(item_review_snippet, max_chars=180)
    if irr:
        parts.append(f"item_evidence: {irr}")
    return _join_semantic_parts(parts, max_chars=920)


def build_binary_prompt_semantic(user_text: str, item_text: str) -> str:
    u = _clean(user_text)
    i = _clean(item_text)
    return (
        "You are a recommendation assistant. "
        "Given user preference evidence and one candidate business, "
        "predict whether the user will positively interact with this business. "
        "Answer only YES or NO.\n"
        f"User: {u}\n"
        f"Candidate: {i}\n"
        "Answer:"
    )


def build_scoring_prompt(user_text: str, item_text: str) -> str:
    u = _clean(user_text)
    i = _clean(item_text)
    return (
        "You are a recommendation scoring model. "
        "Given user preference evidence and one candidate business, "
        "estimate how well this candidate matches the user's likely preference.\n"
        f"User: {u}\n"
        f"Candidate: {i}\n"
        "Preference score:"
    )


__all__ = [
    "build_binary_prompt",
    "build_binary_prompt_semantic",
    "build_scoring_prompt",
    "build_item_text",
    "build_item_text_full_lite",
    "build_item_text_semantic_compact",
    "build_item_text_semantic_compact_preserve",
    "build_item_text_semantic_compact_targeted",
    "build_item_text_semantic_compact_boundary",
    "build_item_text_semantic",
    "build_item_text_sft_clean",
    "build_pair_alignment_summary",
    "build_user_text",
]
