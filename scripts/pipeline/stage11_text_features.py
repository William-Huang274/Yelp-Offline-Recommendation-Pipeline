from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

_JSONISH_KV_RE = re.compile(r'["\']?([a-z0-9_]+)["\']?\s*:\s*\[([^\]]+)\]', re.I)
_JSONISH_VALUE_RE = re.compile(r'["\']?([a-z0-9]+(?:_[a-z0-9]+)*)["\']?', re.I)
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
_BROAD_USER_SCENE_TERMS = {
    "family-friendly settings",
    "group dining",
    "sit-down meals",
    "sit-down dining settings",
    "fast-casual spots",
    "quick casual meals",
    "late-night meals",
    "date-night meals",
    "date-night outings",
    "weekend outings",
}
_BROAD_USER_PROPERTY_TERMS = {
    "delivery",
    "takeout",
    "reservations",
}


@lru_cache(maxsize=65536)
def _clean_text_cached(raw_text: str, max_chars: int) -> str:
    txt = raw_text.replace("\n", " ").replace("\r", " ")
    txt = txt.replace("[SHORT]", " ").replace("[LONG]", " ")
    txt = txt.replace('""', '"')
    txt = re.sub(r"\s+", " ", txt).strip(" ;|")
    txt = re.sub(r"\bweak value for(?:\s+money)?\b", "weak value for money", txt, flags=re.I)
    txt = re.sub(r"\bweak value for money(?:\s+money)+\b", "weak value for money", txt, flags=re.I)
    txt = re.sub(r"\bvalue for(?:\s+money)?\b", "value for money", txt, flags=re.I)
    txt = re.sub(r"\blong wait(?:s)?\b", "long waits", txt, flags=re.I)
    txt = re.sub(r"\b([a-z]+)(?:\s+\1){1,}\b", r"\1", txt, flags=re.I)
    txt = re.sub(r"\band\s+and\b", "and", txt, flags=re.I)
    if int(max_chars) > 0 and len(txt) > int(max_chars):
        clipped = txt[: int(max_chars)].rsplit(" ", 1)[0].strip()
        txt = clipped if clipped else txt[: int(max_chars)].strip()
    return txt


def clean_text(raw: Any, max_chars: int = 0) -> str:
    return _clean_text_cached(str(raw or ""), int(max_chars or 0))


def _polish_narrative_text(raw: Any, max_chars: int = 0) -> str:
    txt = clean_text(raw, max_chars=0)
    if not txt:
        return ""
    txt = re.sub(r"\b(Recent visits|Past behavior|Past positive notes|The latest visits|Recent activity)\.\s+(?=[A-Z])", "", txt, flags=re.I)
    txt = re.sub(r"\s*,\s*\.", ".", txt)
    txt = re.sub(r"\s*;\s*\.", ".", txt)
    txt = re.sub(r"\s*:\s*\.", ".", txt)
    txt = re.sub(r"\s+\.\s*", ". ", txt)
    txt = re.sub(r"\s+,", ",", txt)
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
    )
    txt = re.sub(r"\.\s*\.", ".", txt)
    txt = re.sub(r"\s+", " ", txt).strip(" ,;:-")
    if txt and not re.search(r"[.!?]$", txt):
        txt = f"{txt}."
    return clean_text(txt, max_chars=max_chars)


def humanize_tag(raw: Any) -> str:
    txt = clean_text(raw)
    if not txt:
        return ""
    txt = clean_text(txt.replace("_", " ").replace("&", " and ")).lower()
    txt = _DISPLAY_TERM_REWRITE.get(txt, txt)
    return clean_text(txt)


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


def _source_tokens(raw: Any) -> set[str]:
    txt = clean_text(raw)
    if not txt:
        return set()
    out: set[str] = set()
    for part in re.split(r"[|,;]", txt):
        token = clean_text(part).lower()
        if token:
            out.add(token)
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


def _join_natural_parts(parts: list[str], max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for raw in parts:
        seg = _polish_narrative_text(raw)
        if not seg:
            continue
        add_len = len(seg) if not out else len(seg) + 3
        if out and total + add_len > max_chars:
            break
        if not out and len(seg) > max_chars:
            return clean_text(seg, max_chars=max_chars)
        out.append(seg)
        total += add_len
    normalized: list[str] = []
    for seg in out:
        piece = _polish_narrative_text(seg)
        if not piece:
            continue
        if not re.search(r"[.!?]$", piece):
            piece = f"{piece}."
        normalized.append(piece)
    return _polish_narrative_text(" ".join(normalized), max_chars=max_chars)


def _join_sentence_parts(parts: list[str], max_chars: int) -> str:
    out: list[str] = []
    total = 0
    for raw in parts:
        seg = _polish_narrative_text(raw)
        if not seg:
            continue
        if not re.search(r"[.!?]$", seg):
            seg = f"{seg}."
        add_len = len(seg) if not out else len(seg) + 1
        if out and total + add_len > max_chars:
            break
        if not out and len(seg) > max_chars:
            return clean_text(seg, max_chars=max_chars)
        out.append(seg)
        total += add_len
    return _polish_narrative_text(" ".join(out), max_chars=max_chars)


def _join_terms_readable(values: list[str], limit: int = 4) -> str:
    vals = _dedupe_keep_order([humanize_tag(v) for v in values], limit=max(1, int(limit)))
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return f"{', '.join(vals[:-1])}, and {vals[-1]}"


def _extract_labeled_items(raw: Any, labels: list[str], limit: int = 4) -> list[str]:
    txt = clean_text(_strip_structured_markers(raw), max_chars=700)
    if not txt:
        return []
    for label in labels:
        match = re.search(rf"{re.escape(label)}\s*:\s*([^.;]+)", txt, flags=re.I)
        if not match:
            continue
        values = [
            _normalize_summary_term(v)
            for v in re.split(r"[,/|]", match.group(1))
            if _normalize_summary_term(v)
        ]
        out = _dedupe_keep_order(values, limit=limit)
        if out:
            return out
    return []


def _compose_preference_sentence(
    *,
    lead: str,
    cuisines: list[str],
    meals: list[str],
    scenes: list[str],
    props: list[str],
    max_chars: int,
) -> str:
    cuisines = _dedupe_keep_order([_normalize_summary_term(v) for v in cuisines if _normalize_summary_term(v)], limit=4)
    meals = _dedupe_keep_order([_normalize_summary_term(v) for v in meals if _normalize_summary_term(v)], limit=4)
    scenes = _dedupe_keep_order(
        [
            _normalize_summary_term(v)
            for v in scenes
            if _normalize_summary_term(v) and _normalize_summary_term(v) not in _BROAD_USER_SCENE_TERMS
        ],
        limit=4,
    )
    props = _dedupe_keep_order(
        [
            _normalize_summary_term(v)
            for v in props
            if _normalize_summary_term(v) and _normalize_summary_term(v) not in _BROAD_USER_PROPERTY_TERMS
        ],
        limit=4,
    )
    anchor = ""
    qualifiers: list[str] = []
    if cuisines:
        anchor = _join_terms_readable(cuisines, limit=3)
    elif meals:
        anchor = f"{_join_terms_readable(meals, limit=3)} meals"
    elif scenes:
        anchor = f"{_join_terms_readable(scenes, limit=3)} settings"
    elif props:
        anchor = f"places with {_join_terms_readable(props, limit=2)}"
    if not anchor:
        return ""
    if meals and not anchor.endswith(" meals"):
        qualifiers.append(f"most often around {_join_terms_readable(meals, limit=3)}")
    if scenes and not anchor.endswith(" settings"):
        qualifiers.append(f"usually in {_join_terms_readable(scenes, limit=3)} settings")
    if props and not anchor.startswith("places with ") and not qualifiers:
        qualifiers.append(f"often valuing {_join_terms_readable(props, limit=2)}")
    base_sentence = f"{lead} {anchor}".strip()
    if not base_sentence.endswith("."):
        base_sentence = f"{base_sentence}."
    best = _polish_narrative_text(base_sentence, max_chars=max_chars)
    remaining = list(qualifiers)
    while remaining:
        candidate = f"{lead} {anchor}, {', '.join(remaining)}."
        polished = _polish_narrative_text(candidate, max_chars=0)
        if len(polished) <= max_chars:
            return polished
        remaining = remaining[:-1]
    return best


def naturalize_user_long_pref_text(raw: Any, max_chars: int = 220) -> str:
    cuisines = _extract_labeled_items(raw, ["Long-term cuisines"], limit=4)
    meals = _extract_labeled_items(raw, ["Typical meal preferences"], limit=4)
    scenes = _extract_labeled_items(raw, ["Typical dining scenes"], limit=4)
    props = _extract_labeled_items(raw, ["Useful service properties"], limit=4)
    if cuisines or meals or scenes or props:
        summary = _compose_preference_sentence(
            lead="Over time, this user most often chooses",
            cuisines=cuisines,
            meals=meals,
            scenes=scenes,
            props=[],
            max_chars=max_chars,
        )
        blocks = [summary] if summary else []
        if props:
            prop_sentence = _polish_narrative_text(
                f"Practical details that tend to work well include {_join_terms_readable(props, limit=3)}.",
                max_chars=max(80, max_chars // 2),
            )
            if prop_sentence:
                blocks.append(prop_sentence)
        summary = _join_sentence_parts([b for b in blocks if b], max_chars=max_chars)
        if summary:
            return summary
    return _naturalize_text_view(raw, max_chars=max_chars)


def naturalize_user_recent_intent_text(raw: Any, max_chars: int = 180) -> str:
    cuisines = _extract_labeled_items(raw, ["Latest known cuisine intent", "Recent broader cuisine drift"], limit=3)
    meals = _extract_labeled_items(raw, ["Latest known meals", "Recent meals"], limit=4)
    scenes = _extract_labeled_items(raw, ["Latest known dining scenes", "Recent dining scenes"], limit=4)
    props = _extract_labeled_items(raw, ["Latest known useful properties", "Recent useful properties"], limit=4)
    props = [p for p in props if p not in {"takeout", "delivery", "reservations", "late hours", "weekend activity"}]
    if cuisines or meals or scenes or props:
        summary = _compose_preference_sentence(
            lead="Recent visits are leaning toward",
            cuisines=cuisines,
            meals=meals,
            scenes=scenes,
            props=[],
            max_chars=max_chars,
        )
        blocks = [summary] if summary else []
        if props:
            prop_sentence = _polish_narrative_text(
                f"Recent outings also line up with places offering {_join_terms_readable(props, limit=3)}.",
                max_chars=max(70, max_chars // 2),
            )
            if prop_sentence:
                blocks.append(prop_sentence)
        summary = _join_sentence_parts([b for b in blocks if b], max_chars=max_chars)
        if summary:
            return summary
    return _naturalize_text_view(raw, max_chars=max_chars)


def naturalize_user_negative_avoid_text(raw: Any, max_chars: int = 160) -> str:
    primary_avoid = _extract_labeled_items(raw, ["Often avoids"], limit=4)
    secondary_avoid = _extract_labeled_items(raw, ["Other negative cuisines"], limit=4)
    secondary_avoid_experience = [
        item for item in secondary_avoid
        if any(marker in item for marker in ["wait", "service", "value", "noise", "parking", "clean"])
    ]
    secondary_avoid_theme = [item for item in secondary_avoid if item not in secondary_avoid_experience]
    if primary_avoid or secondary_avoid:
        primary = _join_terms_readable(primary_avoid or secondary_avoid_experience or secondary_avoid_theme, limit=4)
        sentence = f"This user is less responsive to places marked by {primary}"
        if primary_avoid and secondary_avoid_experience:
            sentence = f"{sentence}, especially around {_join_terms_readable(secondary_avoid_experience, limit=3)}"
        if not sentence.endswith("."):
            sentence = f"{sentence}."
        blocks = [sentence]
        if secondary_avoid_theme and secondary_avoid_theme != primary_avoid:
            blocks.append(
                _polish_narrative_text(
                    f"Additional friction tends to show up around {_join_terms_readable(secondary_avoid_theme, limit=3)}.",
                    max_chars=max(80, max_chars // 2 + 20),
                )
            )
        return _join_sentence_parts([b for b in blocks if b], max_chars=max_chars)
    return _naturalize_text_view(raw, max_chars=max_chars)


def naturalize_user_context_text(raw: Any, max_chars: int = 140) -> str:
    txt = clean_text(_strip_structured_markers(raw), max_chars=500)
    if not txt:
        return ""
    outing_prefs = [
        clean_text(m.group(1).replace("-", " "), max_chars=32)
        for m in re.finditer(r"Prefers ([^.]+?) outings", txt, flags=re.I)
        if clean_text(m.group(1).replace("-", " "), max_chars=32)
    ]
    outing_prefs = [pref for pref in outing_prefs if pref.lower() not in _BROAD_USER_SCENE_TERMS]
    if outing_prefs:
        return _polish_narrative_text(
            f"Typical outings most often revolve around {_join_terms_readable(outing_prefs, limit=3)}.",
            max_chars=max_chars,
        )
    return ""


def _strip_structured_markers(raw: Any) -> str:
    txt = str(raw or "")
    txt = txt.replace("[SHORT]", " ").replace("[LONG]", " ")
    txt = txt.replace("||", ". ")
    txt = re.sub(r'["\']', " ", txt)
    txt = re.sub(r"[{}\[\]]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip(" ;|,.-")
    return txt


def _narrative_segments(raw: Any, max_segments: int = 2) -> list[str]:
    txt = clean_text(_strip_structured_markers(raw), max_chars=900)
    if not txt:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"(?<=[.!?])\s+|(?:\s*\|\|\s*)|(?:\s*;\s*)", txt):
        seg = clean_text(part, max_chars=180)
        if not seg or len(seg.split()) < 5:
            continue
        if _looks_structured_fragment(seg):
            continue
        if _looks_review_fragment(seg):
            continue
        key = seg.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
        if len(out) >= max_segments:
            break
    return out


def _naturalize_preference_text(raw: Any, *, positive: bool, max_chars: int = 180) -> str:
    txt = clean_text(_strip_structured_markers(raw), max_chars=500)
    if not txt:
        return ""
    facet_values: dict[str, list[str]] = {}
    freeform_bits: list[str] = []
    for piece in txt.split(";"):
        seg = clean_text(piece, max_chars=120)
        if not seg:
            continue
        seg = re.sub(r"^(likes|avoids)\s+", "", seg, flags=re.I)
        if ":" in seg:
            key, values = seg.split(":", 1)
            key = humanize_tag(key).lower()
            vals = _dedupe_keep_order(
                [humanize_tag(v) for v in re.split(r"[,/|]", values)],
                limit=4,
            )
            vals = [v for v in vals if v.lower() not in {"service", "food", "restaurant", "restaurants", "place", "places"}]
            if not vals:
                continue
            prev = facet_values.get(key, [])
            facet_values[key] = _dedupe_keep_order(prev + vals, limit=4)
        else:
            freeform_bits.append(seg.replace("_", " "))
    cuisines = facet_values.get("cuisine", []) + facet_values.get("cuisines", [])
    meals = facet_values.get("meal", []) + facet_values.get("meals", [])
    scenes = facet_values.get("scene", []) + facet_values.get("scenes", [])
    props = (
        facet_values.get("service", [])
        + facet_values.get("services", [])
        + facet_values.get("property", [])
        + facet_values.get("properties", [])
    )
    other = []
    for key, vals in facet_values.items():
        if key in {"cuisine", "cuisines", "meal", "meals", "scene", "scenes", "service", "services", "property", "properties"}:
            continue
        if vals:
            other.append(f"{humanize_tag(key)} like {_join_terms_readable(vals, limit=3)}")
    blocks: list[str] = []
    if positive:
        lead = _compose_preference_sentence(
            lead="Profile cues suggest this user responds well to",
            cuisines=cuisines,
            meals=meals,
            scenes=scenes,
            props=props,
            max_chars=max_chars,
        )
        if lead:
            blocks.append(lead)
        if other:
            blocks.append(_polish_narrative_text(f"Other positive cues include {_join_terms_readable(other, limit=2)}.", max_chars=max(80, max_chars // 2)))
    else:
        avoid_terms = _dedupe_keep_order([humanize_tag(v) for v in cuisines + meals + scenes + props], limit=4)
        if avoid_terms:
            blocks.append(
                _polish_narrative_text(
                    f"Profile cues suggest this user is less tolerant of {_join_terms_readable(avoid_terms, limit=4)}.",
                    max_chars=max_chars,
                )
            )
        if other:
            blocks.append(_polish_narrative_text(f"Other watch-outs include {_join_terms_readable(other, limit=2)}.", max_chars=max(80, max_chars // 2)))
    if freeform_bits:
        blocks.append(_polish_narrative_text("; ".join(freeform_bits[:2]), max_chars=max(90, max_chars // 2)))
    blocks = _dedupe_narrative_parts([b for b in blocks if b], limit=3, max_chars=max_chars)
    if not blocks:
        return ""
    return _join_sentence_parts(blocks, max_chars=max_chars)


def _preference_evidence_sentence(raw: Any, *, positive: bool, max_chars: int = 160) -> str:
    txt = clean_text(_strip_structured_markers(raw), max_chars=500)
    if not txt:
        return ""
    terms: list[str] = []
    for piece in txt.split(";"):
        seg = clean_text(piece, max_chars=140)
        if not seg:
            continue
        if ":" in seg:
            _, values = seg.split(":", 1)
            candidates = [humanize_tag(v) for v in re.split(r"[,/|]", values)]
        else:
            candidates = [seg]
        for val in candidates:
            norm = _normalize_summary_term(val, max_words=4)
            if not norm:
                continue
            terms.append(norm)
    terms = _dedupe_keep_order(terms, limit=3 if positive else 2)
    if not terms:
        return ""
    joined = clean_text(_join_terms_readable(terms, limit=4), max_chars=max_chars)
    joined = re.sub(r"(?:\s*,\s*)+$", "", joined).strip(" ,.;:-")
    if not joined:
        return ""
    if positive:
        sentence = _polish_narrative_text(
            f"Profile evidence most often mentions interest in {joined}.",
            max_chars=max_chars,
        )
        return re.sub(r",\s*\.$", ".", sentence)
    sentence = _polish_narrative_text(
        f"Profile evidence most often flags friction around {joined}.",
        max_chars=max_chars,
    )
    return re.sub(r",\s*\.$", ".", sentence)


def _looks_review_fragment(raw: Any) -> bool:
    txt = clean_text(raw, max_chars=220)
    if not txt:
        return False
    if re.search(r"\b(i|i'm|i’ve|we|we've|my|our|me|us)\b", txt, flags=re.I):
        return True
    if re.search(
        r"\b(ordered|had|got|came|went|tried|reviewing|visited|sat|returned|ate|drank|recommend|recommended|loved|liked|favorite)\b",
        txt,
        flags=re.I,
    ):
        return True
    if re.search(r"\b(location|both times|this time|last time)\b", txt, flags=re.I):
        return True
    if re.search(
        r"\b(was|were|is|are)\s+(great|good|excellent|amazing|fantastic|delicious)\b",
        txt,
        flags=re.I,
    ):
        return True
    if re.search(
        r"\b(served|head-on|portion|shrimp|fried|crispy|sauce|broth|noodles?)\b",
        txt,
        flags=re.I,
    ) and len(txt.split()) <= 20:
        return True
    return False


def _normalize_summary_term(raw: Any, max_words: int = 4) -> str:
    txt = clean_text(humanize_tag(raw), max_chars=48).strip(" ,.;:-")
    if not txt:
        return ""
    txt = re.sub(
        r"^(recent activity is concentrating around|recent activity is clustering around|most often around|usually in|working best around)\s+",
        "",
        txt,
        flags=re.I,
    ).strip(" ,.;:-")
    if not txt:
        return ""
    txt = re.sub(r"\b(and|or|with|to|for|in|on|at|of|a|an|the)$", "", txt, flags=re.I).strip(" ,.;:-")
    if not txt:
        return ""
    if _looks_review_fragment(txt) or _looks_structured_fragment(txt):
        return ""
    if len(txt.split()) > max(1, int(max_words)):
        return ""
    if txt.lower() in {
        "service",
        "food",
        "restaurant",
        "restaurants",
        "place",
        "places",
        "atmosphere",
        "attention",
        "activity",
        "outings",
        "weak",
        "long",
        "money",
        "but",
        "though",
        "however",
        "still",
        "dish",
        "dishes",
        "kept",
        "standing",
        "stood",
        "loved",
        "hated",
        "ordered",
        "tried",
        "visited",
    }:
        return ""
    return txt


def _combined_preference_evidence_sentence(
    profile_pos_text: Any,
    profile_neg_text: Any,
    *,
    max_chars: int = 180,
) -> str:
    pos = _preference_evidence_sentence(profile_pos_text, positive=True, max_chars=max(86, min(130, max_chars // 2 + 28)))
    neg = _preference_evidence_sentence(profile_neg_text, positive=False, max_chars=max(72, min(120, max_chars // 2 + 12)))
    if not pos:
        pos = _naturalize_preference_text(profile_pos_text, positive=True, max_chars=max(100, min(150, max_chars // 2 + 32)))
    if not neg:
        neg = _naturalize_preference_text(profile_neg_text, positive=False, max_chars=max(90, min(140, max_chars // 2 + 16)))
    if pos and neg:
        return _join_sentence_parts([pos, neg], max_chars=max_chars)
    if pos:
        return _polish_narrative_text(pos, max_chars=max_chars)
    if neg:
        return _polish_narrative_text(neg, max_chars=max_chars)
    return ""


def build_profile_preference_evidence_text(
    profile_pos_text: Any,
    profile_neg_text: Any,
    *,
    max_chars: int = 180,
) -> str:
    return _combined_preference_evidence_sentence(
        profile_pos_text,
        profile_neg_text,
        max_chars=max_chars,
    )


def _naturalize_text_view(raw: Any, max_chars: int = 220) -> str:
    txt = clean_text(_strip_structured_markers(raw), max_chars=500)
    if not txt:
        return ""
    txt = txt.replace("_", " ")
    txt = re.sub(r"\bPrimary geo area:\s*[^.]+\.?", "", txt, flags=re.I)
    txt = re.sub(r"\bLatest known\b", "Recent", txt, flags=re.I)
    txt = re.sub(r"\bOften avoids:\s*", "Often avoids ", txt, flags=re.I)
    txt = re.sub(r"\s+\.\s*", ". ", txt)
    return _polish_narrative_text(txt, max_chars=max_chars)


def _profile_review_evidence_from_raw(
    profile_text_short: Any,
    profile_text_long: Any = "",
    *,
    max_chars: int = 220,
) -> str:
    narrative_parts = _dedupe_narrative_parts(
        [
            *_narrative_segments(profile_text_short, max_segments=1),
            *_narrative_segments(profile_text_long, max_segments=1),
        ],
        limit=2,
        max_chars=max_chars,
    )
    if narrative_parts:
        joined = _join_sentence_parts(narrative_parts, max_chars=max_chars)
        if joined:
            return joined
    return ""


def _naturalize_review_evidence_segment(raw: Any, *, max_chars: int = 180) -> str:
    txt = clean_text(_strip_structured_markers(raw), max_chars=max_chars * 2)
    if not txt or _looks_structured_fragment(txt):
        return ""
    txt = txt.strip()
    txt = re.sub(r"\s+", " ", txt).strip(" ;|,.-")
    if not txt:
        return ""
    if re.match(r"(?i)^(past review|past reviews|review evidence|reviews?)\b", txt):
        sentence = txt
    elif re.match(r"(?i)^(they|it|there)\b", txt):
        sentence = f"Past reviews mention that {txt}."
    else:
        sentence = f"Past reviews mention {txt}."
    return _polish_narrative_text(sentence, max_chars=max_chars)


def build_clean_user_evidence_text(
    profile_text_short: Any,
    profile_text_long: Any = "",
    profile_pos_text: Any = "",
    profile_neg_text: Any = "",
    user_long_pref_text: Any = "",
    user_recent_intent_text: Any = "",
    user_negative_avoid_text: Any = "",
    user_context_text: Any = "",
    max_chars: int = 320,
) -> str:
    parts: list[str] = []
    review_evidence = _profile_review_evidence_from_raw(
        profile_text_short,
        profile_text_long,
        max_chars=220,
    )
    review_evidence = _naturalize_review_evidence_segment(review_evidence, max_chars=220)
    # Keep this field intentionally strict: if we do not have clean review-derived
    # evidence, prefer returning empty and letting upstream low-signal filtering handle
    # the user rather than backfilling with templated profile summaries.
    for seg in (review_evidence,):
        seg = clean_text(seg, max_chars=220)
        if not seg:
            continue
        seg_tokens = {t for t in re.findall(r"[a-z0-9]+", seg.lower()) if len(t) >= 4}
        existing_tokens = {
            t
            for base in parts
            for t in re.findall(r"[a-z0-9]+", str(base).lower())
            if len(t) >= 4
        }
        if seg_tokens and len([t for t in seg_tokens if t not in existing_tokens]) < 2:
            continue
        parts.append(seg)
        if len(parts) >= 3:
            break

    cleaned = _dedupe_narrative_parts(parts, limit=4, max_chars=220)
    return _join_sentence_parts(cleaned, max_chars=max_chars)


def _dedupe_keep_order(values: list[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        txt = clean_text(raw, max_chars=80)
        if not txt:
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _dedupe_narrative_parts(values: list[str], limit: int = 4, max_chars: int = 240) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        txt = _polish_narrative_text(raw, max_chars=max_chars)
        if not txt:
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _jsonish_value_terms(raw: Any, limit: int = 4) -> list[str]:
    text = clean_text(raw, max_chars=400)
    if not text:
        return []
    vals: list[str] = []
    for token in _JSONISH_VALUE_RE.findall(text):
        item = humanize_tag(token)
        if not item:
            continue
        vals.append(item)
    return _dedupe_keep_order(vals, limit=limit)


def _jsonish_key_phrase(raw_key: str, values: list[str]) -> str:
    key = clean_text(raw_key).lower().replace("_", " ")
    if not key or not values:
        return ""
    joined = ", ".join(values[:4])
    if key in {"cuisine", "food", "dish", "menu"}:
        return f"mentions cuisine preferences like {joined}"
    if key in {"service", "staff", "hospitality"}:
        return f"mentions service details like {joined}"
    if key in {"price", "value", "price value"}:
        return f"mentions value concerns like {joined}"
    if key in {"ambience", "atmosphere", "noise", "crowd", "crowds"}:
        return f"mentions atmosphere details like {joined}"
    if key in {"wait", "waiting", "queue", "line"}:
        return f"mentions wait-time concerns like {joined}"
    if key in {"location", "distance", "parking"}:
        return f"mentions location details like {joined}"
    return f"mentions {humanize_tag(key)} such as {joined}"


def _jsonish_evidence_segments(raw: Any, max_segments: int = 3) -> list[str]:
    text = clean_text(raw, max_chars=800)
    if not text:
        return []
    parts: list[str] = []
    for key, body in _JSONISH_KV_RE.findall(text):
        vals = _jsonish_value_terms(body, limit=4)
        phrase = _jsonish_key_phrase(key, vals)
        if phrase:
            parts.append(phrase)
    return _dedupe_keep_order(parts, limit=max_segments)


def _looks_structured_fragment(raw: Any) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    if re.search(r'"[^"]+"\s*:', text):
        return True
    if re.search(r"[{}\[\]]", text):
        return True
    if text.count(":") >= 2 and text.count(",") == 0:
        return True
    return False


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


def build_candidate_competition_summary(
    pre_rank: Any,
    pre_score: Any,
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
) -> str:
    parts: list[str] = []

    overlap_rank = _finite_float(group_gap_rank_pct)
    if overlap_rank is not None:
        if overlap_rank >= 0.90:
            parts.append("very strong schema fit within current candidate pool")
        elif overlap_rank >= 0.75:
            parts.append("above-average schema fit within current candidate pool")
        elif overlap_rank <= 0.25:
            parts.append("weak schema fit within current candidate pool")

    gap_top3 = _finite_float(group_gap_to_top3)
    if gap_top3 is not None:
        if gap_top3 >= -0.02:
            parts.append("close to the top-3 matched option in this pool")
        elif gap_top3 <= -0.15:
            parts.append("well below the top-3 matched option in this pool")

    gap_top10 = _finite_float(group_gap_to_top10)
    if gap_top10 is not None and not parts:
        if gap_top10 >= -0.02:
            parts.append("close to the top-10 matched options in this pool")
        elif gap_top10 <= -0.15:
            parts.append("well below the top-10 matched options in this pool")

    net_rank = _finite_float(net_score_rank_pct)
    if net_rank is not None:
        if net_rank >= 0.90:
            parts.append("strong net competition score")
        elif net_rank <= 0.25:
            parts.append("weak net competition score")

    rank_val = _finite_float(pre_rank)
    if rank_val is not None and rank_val > 0:
        rank_int = int(round(rank_val))
        if rank_int <= 10:
            parts.append(f"top-10 candidate in current pool (rank {rank_int})")
        elif rank_int <= 30:
            parts.append(f"top-30 candidate in current pool (rank {rank_int})")
        elif rank_int <= 80:
            parts.append(f"mid-ranked candidate in current pool (rank {rank_int})")
        else:
            parts.append(f"lower-ranked retained candidate (rank {rank_int})")

    score_val = _finite_float(pre_score)
    if score_val is not None:
        parts.append(f"candidate pre-score {score_val:.3f}")
    return clean_text("; ".join(parts), max_chars=220)


def build_candidate_fit_risk_summary(
    pair_evidence: Any,
    item_neg_tags: Any = "",
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    source_set: Any = "",
    user_segment: Any = "",
) -> str:
    parts: list[str] = []

    conflict_val = _finite_float(conflict_gap)
    if conflict_val is not None:
        if conflict_val >= 0.05:
            parts.append("lower semantic conflict risk")
        elif conflict_val <= -0.05:
            parts.append("higher semantic conflict risk")

    avoid_neg_val = _finite_float(avoid_neg)
    if avoid_neg_val is not None:
        if avoid_neg_val >= 0.05:
            parts.append("avoids user-disliked semantics")
        elif avoid_neg_val <= -0.05:
            parts.append("overlaps with user-disliked semantics")

    avoid_core_val = _finite_float(avoid_core)
    if avoid_core_val is not None:
        if avoid_core_val >= 0.05:
            parts.append("keeps distance from user core dislikes")
        elif avoid_core_val <= -0.05:
            parts.append("contains user core dislike signals")

    pair_txt = clean_text(pair_evidence, max_chars=220)
    if pair_txt:
        parts.append(pair_txt)

    neg = split_tags(item_neg_tags, limit=3)
    if neg:
        parts.append(f"watch-outs: {', '.join(neg)}")

    if not parts:
        src = split_tags(source_set, limit=3)
        if src:
            parts.append(f"surfaced by {', '.join(src)}")
        seg = clean_text(user_segment, max_chars=32)
        if seg:
            parts.append(f"user slice {seg}")

    return clean_text("; ".join(parts), max_chars=240)


def build_candidate_compact_competition_hint(
    group_gap_rank_pct: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    net_score_rank_pct: Any = None,
) -> str:
    parts: list[str] = []

    overlap_rank = _finite_float(group_gap_rank_pct)
    net_rank = _finite_float(net_score_rank_pct)
    strength = ""
    if overlap_rank is not None or net_rank is not None:
        score = max(v for v in [overlap_rank, net_rank] if v is not None)
        if score >= 0.90:
            strength = "strong contender"
        elif score >= 0.75:
            strength = "mid contender"
        elif score <= 0.25:
            strength = "weak contender"
    if strength:
        parts.append(strength)

    gap_top3 = _finite_float(group_gap_to_top3)
    gap_top10 = _finite_float(group_gap_to_top10)
    if gap_top3 is not None:
        if gap_top3 >= -0.02:
            parts.append("near top-3")
        elif gap_top3 <= -0.15:
            parts.append("far below top-3")
    elif gap_top10 is not None:
        if gap_top10 >= -0.02:
            parts.append("near top-10")
        elif gap_top10 <= -0.15:
            parts.append("deep in pool")

    return clean_text("; ".join(parts), max_chars=120)


def build_candidate_compact_risk_hint(
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
) -> str:
    parts: list[str] = []

    conflict_val = _finite_float(conflict_gap)
    if conflict_val is not None:
        if conflict_val >= 0.05:
            parts.append("low conflict")
        elif conflict_val <= -0.05:
            parts.append("high conflict")

    avoid_neg_val = _finite_float(avoid_neg)
    if avoid_neg_val is not None:
        if avoid_neg_val >= 0.05:
            parts.append("avoids disliked semantics")
        elif avoid_neg_val <= -0.05:
            parts.append("overlaps disliked semantics")

    avoid_core_val = _finite_float(avoid_core)
    if avoid_core_val is not None:
        if avoid_core_val >= 0.05:
            parts.append("keeps distance from core dislikes")
        elif avoid_core_val <= -0.05:
            parts.append("core dislike risk")

    return clean_text("; ".join(parts), max_chars=120)


def build_candidate_compact_channel_hint(
    preference_core: Any = None,
    recent_intent: Any = None,
    context_time: Any = None,
    channel_conflict: Any = None,
    evidence_support: Any = None,
    max_hints: int = 3,
) -> str:
    scored: list[tuple[float, str]] = []

    def _maybe_add(value: Any, threshold: float, text: str) -> None:
        val = _finite_float(value)
        if val is None:
            return
        if val >= threshold:
            scored.append((float(val), text))

    _maybe_add(preference_core, 0.05, "core preference aligned")
    _maybe_add(recent_intent, 0.05, "recent intent aligned")
    _maybe_add(context_time, 0.05, "time context aligned")
    _maybe_add(evidence_support, 0.05, "evidence supported")

    conflict_val = _finite_float(channel_conflict)
    if conflict_val is not None and conflict_val <= -0.05:
        scored.append((abs(float(conflict_val)), "channel conflict high"))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    hints: list[str] = []
    seen: set[str] = set()
    for _, text in scored:
        if text in seen:
            continue
        seen.add(text)
        hints.append(text)
        if len(hints) >= max(1, int(max_hints)):
            break

    return clean_text("; ".join(hints), max_chars=120)


def build_candidate_compact_route_diversity_hint(
    source_set: Any = "",
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    context_rank: Any = None,
    max_hints: int = 2,
) -> str:
    parts: list[str] = []
    sources = _source_tokens(source_set)
    src_count = _finite_float(source_count)
    nonpopular = _finite_float(nonpopular_source_count)
    profile_cluster = _finite_float(profile_cluster_source_count)
    context_rank_val = _finite_float(context_rank)

    has_incumbent = bool({"als", "popular"} & sources)
    has_personalized = bool({"profile", "cluster", "context"} & sources)

    if src_count is not None and src_count >= 3:
        parts.append("matched by multiple independent routes")
    elif src_count is not None and src_count >= 2 and has_incumbent and has_personalized:
        parts.append("supported by both incumbent and personalized routes")
    elif src_count is not None and src_count >= 2:
        parts.append("supported by multiple routes")

    if nonpopular is not None and nonpopular >= 1:
        parts.append("not driven only by popular recall")
    elif profile_cluster is not None and profile_cluster >= 2:
        parts.append("reinforced across personalized routes")
    elif context_rank_val is not None and context_rank_val > 0 and context_rank_val <= 20:
        parts.append("context route also supports it")

    if not parts:
        return ""
    return clean_text("; ".join(parts[: max(1, int(max_hints))]), max_chars=120)


def build_candidate_compact_support_reliability_hint(
    semantic_support: Any = None,
    evidence_support: Any = None,
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    max_hints: int = 2,
) -> str:
    parts: list[str] = []
    sem_support = _finite_float(semantic_support)
    ev_support = _finite_float(evidence_support)
    src_count = _finite_float(source_count)
    nonpopular = _finite_float(nonpopular_source_count)
    profile_cluster = _finite_float(profile_cluster_source_count)

    if sem_support is not None and sem_support >= 0.18 and ev_support is not None and ev_support >= 0.05:
        parts.append("broad evidence support")
    elif src_count is not None and src_count >= 2 and sem_support is not None and sem_support >= 0.10:
        parts.append("multiple signals agree")
    elif src_count is not None and src_count >= 2 and ev_support is not None and ev_support >= 0.05:
        parts.append("stable support despite limited public evidence")

    if nonpopular is not None and nonpopular >= 1 and src_count is not None and src_count >= 2:
        parts.append("support not driven only by popularity")
    elif profile_cluster is not None and profile_cluster >= 2:
        parts.append("personalized routes reinforce support")

    if not parts:
        return ""
    return clean_text("; ".join(parts[: max(1, int(max_hints))]), max_chars=120)


def build_candidate_compact_stable_fit_hint(
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    preference_core: Any = None,
    recent_intent: Any = None,
    context_time: Any = None,
    channel_conflict: Any = None,
    max_hints: int = 2,
) -> str:
    parts: list[str] = []
    avoid_neg_val = _finite_float(avoid_neg)
    avoid_core_val = _finite_float(avoid_core)
    conflict_val = _finite_float(conflict_gap)
    preference_val = _finite_float(preference_core)
    recent_val = _finite_float(recent_intent)
    context_val = _finite_float(context_time)
    channel_conflict_val = _finite_float(channel_conflict)

    positive_flags = sum(
        1
        for val in (avoid_neg_val, avoid_core_val, conflict_val, preference_val, recent_val, context_val)
        if val is not None and val >= 0.05
    )
    channel_conflict_bad = bool(channel_conflict_val is not None and channel_conflict_val <= -0.05)

    if positive_flags >= 3 and not channel_conflict_bad:
        parts.append("stable fit for this user")
    elif preference_val is not None and preference_val >= 0.05 and recent_val is not None and recent_val >= 0.05:
        parts.append("consistent with profile and recent intent")
    elif context_val is not None and context_val >= 0.05 and avoid_core_val is not None and avoid_core_val >= 0.05:
        parts.append("context fits with low downside")

    if (
        (avoid_neg_val is not None and avoid_neg_val >= 0.05)
        or (avoid_core_val is not None and avoid_core_val >= 0.05)
    ) and (conflict_val is not None and conflict_val >= 0.05):
        parts.append("low downside for this user")
    elif not channel_conflict_bad and positive_flags >= 2:
        parts.append("safer match than it looks")

    if not parts:
        return ""
    return clean_text("; ".join(parts[: max(1, int(max_hints))]), max_chars=120)


def build_candidate_compact_rank_role_hint(
    pre_rank: Any = None,
    group_gap_to_top10: Any = None,
    max_chars: int = 64,
) -> str:
    rank_val = _finite_float(pre_rank)
    top10_gap = _finite_float(group_gap_to_top10)
    text = ""
    if rank_val is not None:
        if rank_val <= 10:
            text = "head retention zone"
        elif rank_val <= 30:
            text = "top10 boundary zone"
        elif rank_val <= 80:
            text = "mid rescue zone"
        else:
            text = "deep long-shot zone"
    elif top10_gap is not None:
        if top10_gap <= 0.0:
            text = "head retention zone"
        elif top10_gap <= 0.08:
            text = "top10 boundary zone"
        elif top10_gap <= 0.20:
            text = "mid rescue zone"
        else:
            text = "deep long-shot zone"
    return clean_text(text, max_chars=max_chars)


def build_candidate_compact_rescue_band_hint(
    learned_rank: Any = None,
    max_chars: int = 64,
) -> str:
    rank_val = _finite_float(learned_rank)
    text = ""
    if rank_val is None:
        return ""
    if rank_val <= 10:
        text = "head guard zone"
    elif rank_val <= 30:
        text = "top10 boundary rescue"
    elif rank_val <= 60:
        text = "secondary rescue zone"
    elif rank_val <= 100:
        text = "light rescue zone"
    else:
        text = "out-of-scope deep candidate"
    return clean_text(text, max_chars=max_chars)


def build_candidate_compact_head_guard_hint(
    learned_rank: Any = None,
    semantic_support: Any = None,
    evidence_support: Any = None,
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    conflict_gap: Any = None,
    preference_core: Any = None,
    recent_intent: Any = None,
    context_time: Any = None,
    channel_conflict: Any = None,
    max_chars: int = 120,
) -> str:
    rank_val = _finite_float(learned_rank)
    if rank_val is None or rank_val > 10:
        return ""

    sem_support = _finite_float(semantic_support)
    ev_support = _finite_float(evidence_support)
    src_count = _finite_float(source_count)
    nonpopular = _finite_float(nonpopular_source_count)
    profile_cluster = _finite_float(profile_cluster_source_count)
    conflict_val = _finite_float(conflict_gap)
    preference_val = _finite_float(preference_core)
    recent_val = _finite_float(recent_intent)
    context_val = _finite_float(context_time)
    channel_conflict_val = _finite_float(channel_conflict)

    support_flags = 0
    if src_count is not None and src_count >= 2:
        support_flags += 1
    if (nonpopular or 0.0) >= 1.0 or (profile_cluster or 0.0) >= 1.0:
        support_flags += 1
    if (sem_support or 0.0) >= 0.15 or (ev_support or 0.0) >= 0.05:
        support_flags += 1
    if (conflict_val or 0.0) >= 0.05 or (preference_val or 0.0) >= 0.05 or (recent_val or 0.0) >= 0.05 or (context_val or 0.0) >= 0.05:
        support_flags += 1

    if channel_conflict_val is not None and channel_conflict_val <= -0.05:
        text = "already in current top10 but under clear displacement risk"
    elif support_flags >= 4:
        text = "already in current top10 with stable support"
    elif support_flags >= 2:
        text = "already in current top10 with defendable support"
    else:
        text = "already in current top10 but support is mixed"
    return clean_text(text, max_chars=max_chars)


def build_candidate_compact_boundary_gap_hint(
    learned_rank: Any = None,
    group_gap_to_top3: Any = None,
    group_gap_to_top10: Any = None,
    semantic_support: Any = None,
    evidence_support: Any = None,
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    conflict_gap: Any = None,
    preference_core: Any = None,
    recent_intent: Any = None,
    context_time: Any = None,
    channel_conflict: Any = None,
    max_chars: int = 120,
) -> str:
    rank_val = _finite_float(learned_rank)
    if rank_val is None or rank_val <= 10 or rank_val > 100:
        return ""

    top10_gap = _finite_float(group_gap_to_top10)
    top3_gap = _finite_float(group_gap_to_top3)
    sem_support = _finite_float(semantic_support)
    ev_support = _finite_float(evidence_support)
    src_count = _finite_float(source_count)
    nonpopular = _finite_float(nonpopular_source_count)
    profile_cluster = _finite_float(profile_cluster_source_count)
    conflict_val = _finite_float(conflict_gap)
    preference_val = _finite_float(preference_core)
    recent_val = _finite_float(recent_intent)
    context_val = _finite_float(context_time)
    channel_conflict_val = _finite_float(channel_conflict)

    support_flags = 0
    if src_count is not None and src_count >= 2:
        support_flags += 1
    if (nonpopular or 0.0) >= 1.0 or (profile_cluster or 0.0) >= 1.0:
        support_flags += 1
    if (sem_support or 0.0) >= 0.12 or (ev_support or 0.0) >= 0.05:
        support_flags += 1
    if (conflict_val or 0.0) >= 0.05 or (preference_val or 0.0) >= 0.05 or (recent_val or 0.0) >= 0.05 or (context_val or 0.0) >= 0.05:
        support_flags += 1
    conflict_bad = bool(channel_conflict_val is not None and channel_conflict_val <= -0.05)

    if rank_val <= 30:
        if not conflict_bad and support_flags >= 4 and top10_gap is not None and top10_gap <= 0.03:
            text = "just outside top10 with a credible rescue case"
        elif not conflict_bad and support_flags >= 3 and top10_gap is not None and top10_gap <= 0.08:
            text = "close behind top10 with enough support to challenge head results"
        elif top3_gap is not None and top3_gap <= 0.20 and not conflict_bad:
            text = "near head set but still behind stronger blockers"
        else:
            text = "top10 boundary candidate with mixed support"
    elif rank_val <= 60:
        if not conflict_bad and support_flags >= 3:
            text = "secondary rescue candidate with usable support"
        else:
            text = "secondary rescue candidate that still needs stronger evidence"
    else:
        if not conflict_bad and support_flags >= 4:
            text = "light rescue candidate with unusually clear support"
        else:
            text = "light rescue candidate, only if evidence is clear"
    return clean_text(text, max_chars=max_chars)


def build_candidate_compact_route_profile_hint(
    source_set: Any = "",
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    context_rank: Any = None,
    max_chars: int = 96,
) -> str:
    sources = _source_tokens(source_set)
    src_count = _finite_float(source_count)
    nonpopular = _finite_float(nonpopular_source_count)
    profile_cluster = _finite_float(profile_cluster_source_count)
    context_rank_val = _finite_float(context_rank)
    has_context = bool(context_rank_val is not None and context_rank_val > 0 and context_rank_val <= 20)
    has_incumbent = bool({"als", "popular"} & sources)
    has_personalized = bool({"profile", "cluster", "context"} & sources) or (profile_cluster or 0.0) >= 1.0 or has_context

    if src_count is not None and src_count >= 3 and has_personalized:
        text = "multi-route with personalized support"
    elif src_count is not None and src_count >= 2 and has_incumbent and has_personalized:
        text = "mixed incumbent and personalized routes"
    elif src_count is not None and src_count >= 2 and has_personalized:
        text = "multi-route, mostly personalized"
    elif src_count is not None and src_count >= 2:
        text = "multi-route, mostly generic"
    elif has_personalized:
        text = "single personalized route"
    else:
        text = "single-route or popularity-led"
    if nonpopular is not None and nonpopular >= 1 and "generic" in text:
        text = text.replace("generic", "nonpopular")
    return clean_text(text, max_chars=max_chars)


def build_candidate_compact_support_profile_hint(
    semantic_support: Any = None,
    evidence_support: Any = None,
    source_count: Any = None,
    nonpopular_source_count: Any = None,
    profile_cluster_source_count: Any = None,
    max_chars: int = 96,
) -> str:
    sem_support = _finite_float(semantic_support)
    ev_support = _finite_float(evidence_support)
    src_count = _finite_float(source_count)
    nonpopular = _finite_float(nonpopular_source_count)
    profile_cluster = _finite_float(profile_cluster_source_count)
    multi_route = bool(src_count is not None and src_count >= 2)
    personalized = bool((nonpopular or 0.0) >= 1.0 or (profile_cluster or 0.0) >= 1.0)
    strong_public = bool((sem_support or 0.0) >= 0.18 and (ev_support or 0.0) >= 0.05)

    if strong_public and multi_route:
        text = "public evidence plus route support"
    elif multi_route and personalized and (ev_support or 0.0) >= 0.05:
        text = "low public support but route-consistent"
    elif multi_route and personalized:
        text = "route support stronger than public evidence"
    elif strong_public:
        text = "surface support heavier than route support"
    elif (ev_support or 0.0) >= 0.05:
        text = "some evidence but limited route support"
    else:
        text = "thin public and route support"
    return clean_text(text, max_chars=max_chars)


def build_candidate_compact_stability_profile_hint(
    avoid_neg: Any = None,
    avoid_core: Any = None,
    conflict_gap: Any = None,
    preference_core: Any = None,
    recent_intent: Any = None,
    context_time: Any = None,
    channel_conflict: Any = None,
    max_chars: int = 96,
) -> str:
    avoid_neg_val = _finite_float(avoid_neg)
    avoid_core_val = _finite_float(avoid_core)
    conflict_val = _finite_float(conflict_gap)
    preference_val = _finite_float(preference_core)
    recent_val = _finite_float(recent_intent)
    context_val = _finite_float(context_time)
    channel_conflict_val = _finite_float(channel_conflict)
    positive_flags = sum(
        1
        for val in (avoid_neg_val, avoid_core_val, conflict_val, preference_val, recent_val, context_val)
        if val is not None and val >= 0.05
    )
    channel_conflict_bad = bool(channel_conflict_val is not None and channel_conflict_val <= -0.05)

    if channel_conflict_bad:
        text = "channel or context conflict present"
    elif positive_flags >= 4:
        text = "low-downside and context-consistent"
    elif (preference_val or 0.0) >= 0.05 and (recent_val or 0.0) >= 0.05:
        text = "profile-and-intent consistent"
    elif positive_flags >= 2:
        text = "mixed stability signals"
    else:
        text = "limited stability support"
    return clean_text(text, max_chars=max_chars)


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
    merged = merge_distinct_segments(raw, max_segments=max_items, max_chars=max_chars * 2)
    if not merged:
        return ""
    out: list[str] = []
    for seg in merged.split("||"):
        txt = clean_text(seg, max_chars=220)
        if not txt:
            continue
        parts = [clean_text(p, max_chars=80) for p in txt.split(";") if clean_text(p, max_chars=80)]
        if not parts:
            continue
        name = parts[0]
        city = ""
        liked = ""
        rating = ""
        for part in parts[1:]:
            p = clean_text(part, max_chars=80)
            if p.startswith("(") and p.endswith(")"):
                city = p.strip("() ")
            elif p.lower().startswith("liked for "):
                liked = p[10:].strip()
            elif p.lower().startswith("praised for "):
                liked = p[11:].strip()
            elif p.lower().startswith("rating "):
                rating = p
        if not liked and not rating:
            continue
        sentence_bits: list[str] = []
        if city:
            sentence_bits.append(f"{name} in {city}")
        else:
            sentence_bits.append(name)
        if liked:
            sentence_bits.append(f"praised for {liked}")
        if rating:
            sentence_bits.append(f"({rating})")
        out.append(" ".join(sentence_bits))
        if len(out) >= max(1, int(max_items)):
            break
    if out:
        return _join_natural_parts(out, max_chars=max_chars)
    return clean_text(merged, max_chars=max_chars)


def extract_user_evidence_text(
    profile_text_short: Any,
    profile_text_long: Any = "",
    profile_pos_text: Any = "",
    profile_neg_text: Any = "",
    user_long_pref_text: Any = "",
    user_recent_intent_text: Any = "",
    user_negative_avoid_text: Any = "",
    user_context_text: Any = "",
    max_chars: int = 260,
) -> str:
    natural = build_clean_user_evidence_text(
        profile_text_short=profile_text_short,
        profile_text_long=profile_text_long,
        profile_pos_text=profile_pos_text,
        profile_neg_text=profile_neg_text,
        user_long_pref_text=user_long_pref_text,
        user_recent_intent_text=user_recent_intent_text,
        user_negative_avoid_text=user_negative_avoid_text,
        user_context_text=user_context_text,
        max_chars=max_chars,
    )
    if natural:
        return natural

    short_txt = clean_text(profile_text_short)
    long_txt = clean_text(profile_text_long)
    short_segments = _jsonish_evidence_segments(short_txt, max_segments=2)
    long_segments = _jsonish_evidence_segments(long_txt, max_segments=3)
    parts: list[str] = []
    for seg in short_segments + long_segments:
        natural_seg = _naturalize_review_evidence_segment(seg, max_chars=160)
        if natural_seg:
            parts.append(natural_seg)
    if not parts and short_txt and not _looks_structured_fragment(short_txt):
        natural_short = _naturalize_review_evidence_segment(short_txt, max_chars=160)
        if natural_short:
            parts.append(natural_short)
    if not parts and long_txt and not _looks_structured_fragment(long_txt):
        natural_long = _naturalize_review_evidence_segment(
            merge_distinct_segments(long_txt, max_segments=2, max_chars=max_chars),
            max_chars=180,
        )
        if natural_long:
            parts.append(natural_long)
    cleaned_parts = _dedupe_narrative_parts(parts, limit=3, max_chars=max_chars)
    if cleaned_parts:
        return _join_sentence_parts(cleaned_parts, max_chars=max_chars)
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
