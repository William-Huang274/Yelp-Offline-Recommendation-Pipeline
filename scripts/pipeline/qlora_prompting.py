from __future__ import annotations

import math
from typing import Any

from pipeline.stage11_text_features import (
    build_history_anchor_summary,
    build_item_semantic_summary,
    build_pair_alignment_summary,
    build_user_preference_summary,
    clean_text,
    merge_distinct_segments,
    split_tags,
)


def _clean(s: Any) -> str:
    return clean_text(s)


def _take_tags(raw: Any, limit: int = 6) -> str:
    tags = split_tags(raw, limit=limit)
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


def build_user_text(
    profile_text: Any,
    top_pos_tags: Any = "",
    top_neg_tags: Any = "",
    confidence: float | None = None,
    review_summary: Any = "",
    review_raw_snippet: Any = "",
    evidence_snippets: Any = "",
    history_anchors: Any = "",
    pair_evidence: Any = "",
) -> str:
    parts: list[str] = []
    ptxt = clean_text(profile_text, max_chars=220)
    pref_summary = build_user_preference_summary(top_pos_tags, top_neg_tags, confidence)
    evidence = merge_distinct_segments(evidence_snippets or review_raw_snippet, max_segments=2, max_chars=260)
    hist_txt = build_history_anchor_summary(history_anchors, max_items=3, max_chars=180)
    pair_txt = clean_text(pair_evidence, max_chars=200)
    if ptxt:
        parts.append(f"profile_summary: {ptxt}")
    if pref_summary:
        parts.append(f"preference_summary: {pref_summary}")
    rs = clean_text(review_summary, max_chars=160)
    if rs and "recent_user_reviews=" not in rs:
        parts.append(f"user_signal: {rs}")
    if evidence:
        parts.append(f"user_evidence: {evidence}")
    if hist_txt:
        parts.append(f"history_anchors: {hist_txt}")
    if pair_txt:
        parts.append(f"pair_signal: {pair_txt}")
    return "; ".join(parts)


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
    source_set: Any = "",
    user_segment: Any = "",
    semantic_support: float | None = None,
    semantic_tag_richness: float | None = None,
    tower_score: float | None = None,
    seq_score: float | None = None,
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

    irs = clean_text(item_review_summary, max_chars=160)
    if irs and "recent_item_reviews=" not in irs:
        parts.append(f"item_signal: {irs}")
    irr = merge_distinct_segments(item_review_snippet, max_segments=2, max_chars=260)
    if irr:
        parts.append(f"item_evidence: {irr}")
    return "; ".join(parts)


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


__all__ = [
    "build_binary_prompt",
    "build_binary_prompt_semantic",
    "build_item_text",
    "build_item_text_full_lite",
    "build_item_text_semantic",
    "build_item_text_sft_clean",
    "build_pair_alignment_summary",
    "build_user_text",
]
