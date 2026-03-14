from __future__ import annotations

import math
import re
from typing import Any


def clean_text(raw: Any, max_chars: int = 0) -> str:
    txt = str(raw or "").replace("\n", " ").replace("\r", " ")
    txt = txt.replace("[SHORT]", " ").replace("[LONG]", " ")
    txt = txt.replace('""', '"')
    txt = re.sub(r"\s+", " ", txt).strip(" ;|")
    if int(max_chars) > 0 and len(txt) > int(max_chars):
        clipped = txt[: int(max_chars)].rsplit(" ", 1)[0].strip()
        txt = clipped if clipped else txt[: int(max_chars)].strip()
    return txt


def humanize_tag(raw: Any) -> str:
    txt = clean_text(raw)
    if not txt:
        return ""
    return clean_text(txt.replace("_", " "))


def split_tags(raw: Any, limit: int = 6) -> list[str]:
    txt = clean_text(raw)
    if not txt:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[|,;]", txt):
        item = humanize_tag(part)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _finite_float(raw: Any) -> float | None:
    try:
        val = float(raw)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def merge_distinct_segments(raw: Any, max_segments: int = 2, max_chars: int = 260) -> str:
    txt = clean_text(raw)
    if not txt:
        return ""
    out: list[str] = []
    seen: set[str] = set()
    for part in txt.split("||"):
        seg = clean_text(part)
        if not seg:
            continue
        key = seg.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
        if len(out) >= max(1, int(max_segments)):
            break
    return clean_text(" || ".join(out), max_chars=max_chars)


def build_user_preference_summary(
    top_pos_tags: Any,
    top_neg_tags: Any,
    confidence: Any,
) -> str:
    parts: list[str] = []
    pos = split_tags(top_pos_tags, limit=4)
    neg = split_tags(top_neg_tags, limit=4)
    conf = _finite_float(confidence)
    if pos:
        parts.append(f"likes {', '.join(pos)}")
    if neg:
        parts.append(f"dislikes {', '.join(neg)}")
    if conf is not None and conf > 0.0:
        parts.append(f"profile confidence {conf:.2f}")
    return clean_text("; ".join(parts), max_chars=220)


def build_item_semantic_summary(
    top_pos_tags: Any,
    top_neg_tags: Any,
    semantic_score: Any,
    semantic_confidence: Any,
) -> str:
    parts: list[str] = []
    pos = split_tags(top_pos_tags, limit=4)
    neg = split_tags(top_neg_tags, limit=4)
    sem = _finite_float(semantic_score)
    conf = _finite_float(semantic_confidence)
    if pos:
        parts.append(f"strengths {', '.join(pos)}")
    if neg:
        parts.append(f"weaknesses {', '.join(neg)}")
    if sem is not None:
        parts.append(f"semantic score {sem:.2f}")
    if conf is not None and conf > 0.0:
        parts.append(f"semantic confidence {conf:.2f}")
    return clean_text("; ".join(parts), max_chars=220)


def build_pair_alignment_summary(
    user_pos_tags: Any,
    user_neg_tags: Any,
    item_pos_tags: Any,
    item_neg_tags: Any,
) -> str:
    u_pos = {x.lower(): x for x in split_tags(user_pos_tags, limit=8)}
    u_neg = {x.lower(): x for x in split_tags(user_neg_tags, limit=8)}
    i_pos = {x.lower(): x for x in split_tags(item_pos_tags, limit=8)}
    i_neg = {x.lower(): x for x in split_tags(item_neg_tags, limit=8)}

    match = [u_pos[k] for k in sorted(set(u_pos) & set(i_pos))[:3]]
    dislike_present = [u_neg[k] for k in sorted(set(u_neg) & set(i_pos))[:3]]
    preferred_but_weak = [u_pos[k] for k in sorted(set(u_pos) & set(i_neg))[:3]]

    parts: list[str] = []
    if match:
        parts.append(f"match aspects: {', '.join(match)}")
    if dislike_present:
        parts.append(f"user-disliked aspects present: {', '.join(dislike_present)}")
    if preferred_but_weak:
        parts.append(f"possible weakness on preferred aspects: {', '.join(preferred_but_weak)}")
    return clean_text("; ".join(parts), max_chars=220)


def build_history_anchor_line(
    name: Any,
    city: Any = "",
    primary_category: Any = "",
    top_pos_tags: Any = "",
    rating: Any = None,
    max_chars: int = 120,
) -> str:
    parts: list[str] = []
    n = clean_text(name, max_chars=48)
    c = clean_text(city, max_chars=24)
    pc = clean_text(primary_category, max_chars=24)
    pos = split_tags(top_pos_tags, limit=2)
    if n:
        parts.append(n)
    meta: list[str] = []
    if c:
        meta.append(c)
    if pc:
        meta.append(pc)
    if meta:
        parts.append(f"({', '.join(meta)})")
    if pos:
        parts.append(f"liked for {', '.join(pos)}")
    rv = _finite_float(rating)
    if rv is not None:
        parts.append(f"rating {rv:.1f}")
    return clean_text("; ".join(parts), max_chars=max_chars)


def build_history_anchor_summary(
    raw: Any,
    max_items: int = 3,
    max_chars: int = 180,
) -> str:
    merged = merge_distinct_segments(raw, max_segments=max_items, max_chars=max_chars)
    return clean_text(merged, max_chars=max_chars)


def extract_user_evidence_text(profile_text_short: Any, profile_text_long: Any = "", max_chars: int = 260) -> str:
    short_txt = clean_text(profile_text_short)
    long_txt = clean_text(profile_text_long)
    if short_txt and long_txt and short_txt.lower() != long_txt.lower():
        return clean_text(f"{short_txt} || {long_txt}", max_chars=max_chars)
    if short_txt:
        return clean_text(short_txt, max_chars=max_chars)
    if long_txt:
        return clean_text(long_txt, max_chars=max_chars)
    return ""


def keyword_match_score(text: Any, pos_tags: Any, neg_tags: Any) -> float:
    body = clean_text(text).lower()
    if not body:
        return 0.0
    score = 0.0
    for tag in split_tags(pos_tags, limit=6) + split_tags(neg_tags, limit=6):
        tag_lc = tag.lower()
        if not tag_lc:
            continue
        if tag_lc in body:
            score += 1.0
    return float(score)
