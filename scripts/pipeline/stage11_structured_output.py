from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any


SCHEMA_VERSION = "stage11_prompt_only_user_state_v2"

ALLOWED_CONFIDENCE_VALUES = {"high", "medium", "low"}
ALLOWED_HYPOTHESIS_TYPES = {
    "conditional_preference",
    "tolerance",
    "shift",
    "conflict",
    "latent_preference",
    "other",
}
ALLOWED_UNKNOWN_FIELDS = {
    "cuisine_preference",
    "scene_preference",
    "tolerance_threshold",
    "recent_shift",
}

REVIEW_REQUIRED_SECTIONS = {
    "stable_preferences",
    "avoid_signals",
    "discriminative_signals",
}

STANDARD_ALLOWED_SUPPORT_BASIS = {
    "direct_user_text",
    "review_pattern_inference",
    "mixed_support",
    "event_context_only",
}

CONTEXTUAL_ALLOWED_SUPPORT_BASIS = {
    "merchant_overlap",
    "contextual_inference",
    "mixed_context",
}

ALLOWED_CONTEXTUAL_AXES = {
    "localness_vs_touristiness",
    "crowding_and_relaxedness",
    "special_occasion_fit",
    "scene_and_ambiance_fit",
    "service_style_fit",
    "value_and_price_context",
    "family_and_group_context",
    "beverage_and_bar_context",
    "late_night_and_hours_fit",
    "geography_and_neighborhood_pattern",
    "cuisine_breadth_and_exploration",
    "dietary_and_lifestyle_context",
    "other_contextual_fit",
}

CONTEXTUAL_AXIS_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "localness_vs_touristiness",
        (
            "tourist",
            "touristy",
            "tourist trap",
            "local vibe",
            "local identity",
            "local spot",
            "local spots",
            "neighborhood gem",
            "hidden gem",
            "commercialized",
        ),
    ),
    (
        "crowding_and_relaxedness",
        (
            "crowded",
            "crowd",
            "relaxed atmosphere",
            "relaxed",
            "laid back",
            "calm",
            "quiet",
            "not overly crowded",
            "less crowded",
            "chill",
        ),
    ),
    (
        "special_occasion_fit",
        (
            "special occasion",
            "anniversary",
            "birthday",
            "celebration",
            "celebratory",
            "date night",
            "romantic night",
        ),
    ),
    (
        "scene_and_ambiance_fit",
        (
            "ambiance",
            "ambience",
            "decor",
            "atmosphere",
            "outdoor seating",
            "patio",
            "community atmosphere",
            "cozy",
            "romantic",
            "view",
        ),
    ),
    (
        "service_style_fit",
        (
            "service",
            "staff",
            "server",
            "waiter",
            "waitress",
            "hospitality",
            "attentive",
            "friendly",
            "knowledgeable",
            "rude",
        ),
    ),
    (
        "value_and_price_context",
        (
            "value",
            "price",
            "pricing",
            "overpriced",
            "expensive",
            "worth",
            "portion",
            "money",
            "fair value",
        ),
    ),
    (
        "family_and_group_context",
        (
            "family",
            "kid",
            "kids",
            "child",
            "children",
            "group",
            "large party",
            "sharing",
            "party",
        ),
    ),
    (
        "beverage_and_bar_context",
        (
            "cocktail",
            "cocktails",
            "wine",
            "beer",
            "brewery",
            "bar",
            "drinks",
            "drink program",
            "happy hour",
        ),
    ),
    (
        "late_night_and_hours_fit",
        (
            "late night",
            "open late",
            "late hours",
            "24/7",
            "24 7",
            "after hours",
            "hours",
        ),
    ),
    (
        "geography_and_neighborhood_pattern",
        (
            "mid city",
            "metairie",
            "marigny",
            "uptown",
            "bywater",
            "kenner",
            "french quarter",
            "neighborhood pattern",
            "west bank",
            "area",
        ),
    ),
    (
        "cuisine_breadth_and_exploration",
        (
            "diverse cuisines",
            "diverse cuisine",
            "broad culinary",
            "exploration",
            "exploring",
            "adventurous eater",
            "variety",
            "trying new",
            "broad palate",
        ),
    ),
    (
        "dietary_and_lifestyle_context",
        (
            "vegan",
            "vegetarian",
            "dairy free",
            "dairy-free",
            "gluten free",
            "gluten-free",
            "non dairy",
            "non-dairy",
            "plant based",
            "plant-based",
        ),
    ),
]

SECTION_REF_LIMITS = {
    "stable_preferences": 3,
    "avoid_signals": 3,
    "recent_signals": 3,
    "context_rules": 3,
    "state_hypotheses": 4,
    "discriminative_signals": 3,
    "contextual_inference_signals": 4,
    "unknowns": 4,
}

SECTION_ITEM_LIMITS = {
    "stable_preferences": 3,
    "avoid_signals": 3,
    "recent_signals": 2,
    "context_rules": 2,
    "state_hypotheses": 3,
    "discriminative_signals": 2,
    "contextual_inference_signals": 3,
    "unknowns": 3,
}

DIRECT_REVIEW_REF_LIMIT = 2
CONTEXTUAL_REF_LIMIT = 2


def normalize_match_text(text: Any) -> str:
    normalized = str(text or "").lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_contextual_axis(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    normalized = normalize_match_text(token).replace(" ", "_")
    if normalized in ALLOWED_CONTEXTUAL_AXES:
        return normalized
    return ""


def infer_contextual_axis(item: dict[str, Any]) -> str:
    explicit = normalize_contextual_axis(item.get("canonical_axis"))
    if explicit:
        return explicit
    text = normalize_match_text(
        " ".join(
            [
                str(item.get("claim", "") or ""),
                str(item.get("support_note", "") or ""),
            ]
        )
    )
    for axis, keywords in CONTEXTUAL_AXIS_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return axis
    return "other_contextual_fit"


def contextual_item_rank(item: dict[str, Any], *, item_index: int) -> tuple[int, int, int, int, int]:
    direct_review_refs = direct_review_ref_list(item)
    contextual_refs = contextual_ref_list(item)
    claim = str(item.get("claim", "") or "").strip()
    return (
        int(bool(contextual_refs)),
        int(bool(direct_review_refs)),
        len(direct_review_refs) + len(contextual_refs),
        len(claim),
        -item_index,
    )


def get_section_items(parsed: Any, *path: str) -> list[dict[str, Any]]:
    node = parsed
    for key in path:
        if not isinstance(node, dict):
            return []
        node = node.get(key)
    if not isinstance(node, list):
        return []
    return [item for item in node if isinstance(item, dict)]


def get_section_list_node(parsed: Any, *path: str) -> list[Any] | None:
    node = parsed
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if not isinstance(node, list):
        return None
    return node


def collect_claim_texts(items: list[dict[str, Any]], field: str = "claim") -> list[str]:
    out: list[str] = []
    for item in items:
        value = str(item.get(field, "") or "").strip()
        if value:
            out.append(value)
    return out


def collect_unknown_fields(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    unknowns = parsed.get("unknowns")
    if not isinstance(unknowns, list):
        return []
    fields: list[str] = []
    for item in unknowns:
        if isinstance(item, dict):
            field = str(item.get("field", "") or "").strip()
            if field:
                fields.append(field)
    return fields


def normalize_ref_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    refs: list[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            refs.append(text)
    return refs


def item_ref_list(item: dict[str, Any]) -> list[str]:
    evidence_refs = normalize_ref_list(item.get("evidence_refs"))
    direct_review_refs = normalize_ref_list(item.get("direct_review_refs"))
    contextual_refs = normalize_ref_list(item.get("contextual_refs"))
    merged: list[str] = []
    for ref in evidence_refs + direct_review_refs + contextual_refs:
        if ref not in merged:
            merged.append(ref)
    return merged


def direct_review_ref_list(item: dict[str, Any]) -> list[str]:
    direct_review_refs = normalize_ref_list(item.get("direct_review_refs"))
    if direct_review_refs:
        return direct_review_refs
    return [ref for ref in item_ref_list(item) if ref.startswith("rev_") or ref.startswith("tip_")]


def contextual_ref_list(item: dict[str, Any]) -> list[str]:
    contextual_refs = normalize_ref_list(item.get("contextual_refs"))
    if contextual_refs:
        return contextual_refs
    return [ref for ref in item_ref_list(item) if not (ref.startswith("rev_") or ref.startswith("tip_"))]


def has_direct_review_ref(refs: list[str]) -> bool:
    return any(ref.startswith("rev_") or ref.startswith("tip_") for ref in refs)


def collect_evidence_refs(node: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"evidence_refs", "direct_review_refs", "contextual_refs"} and isinstance(value, list):
                refs.extend(str(v) for v in value if v is not None)
            else:
                refs.extend(collect_evidence_refs(value))
    elif isinstance(node, list):
        for value in node:
            refs.extend(collect_evidence_refs(value))
    return refs


def collect_ref_groups(node: Any) -> list[list[str]]:
    groups: list[list[str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"evidence_refs", "direct_review_refs", "contextual_refs"} and isinstance(value, list):
                groups.append([str(v) for v in value if v is not None])
            else:
                groups.extend(collect_ref_groups(value))
    elif isinstance(node, list):
        for value in node:
            groups.extend(collect_ref_groups(value))
    return groups


def is_direct_review_ref(ref: str) -> bool:
    return ref.startswith("rev_") or ref.startswith("tip_")


def is_context_only_ref_group(refs: list[str]) -> bool:
    cleaned = [str(ref or "").strip() for ref in refs if str(ref or "").strip()]
    return bool(cleaned) and all(not is_direct_review_ref(ref) for ref in cleaned)


def dedupe_preserving_order(refs: list[str]) -> list[str]:
    out: list[str] = []
    for ref in refs:
        if ref not in out:
            out.append(ref)
    return out


def prioritize_representative_refs(
    refs: list[str],
    *,
    limit: int,
    prefer_direct: bool,
    require_direct: bool,
    allowed_refs: set[str] | None,
) -> tuple[list[str], dict[str, list[str]]]:
    deduped = dedupe_preserving_order(refs)
    removed_invalid = [ref for ref in deduped if allowed_refs is not None and ref not in allowed_refs]
    filtered = [ref for ref in deduped if allowed_refs is None or ref in allowed_refs]

    if prefer_direct:
        direct = [ref for ref in filtered if is_direct_review_ref(ref)]
        other = [ref for ref in filtered if not is_direct_review_ref(ref)]
        ordered = direct + other
    else:
        ordered = list(filtered)

    selected = ordered[: max(limit, 0)]
    if require_direct and selected and not any(is_direct_review_ref(ref) for ref in selected):
        direct = [ref for ref in filtered if is_direct_review_ref(ref)]
        if direct:
            anchor = direct[0]
            selected = [anchor] + [ref for ref in selected if ref != anchor]
            selected = selected[: max(limit, 0)]

    removed_trim = [ref for ref in filtered if ref not in selected]
    return selected, {
        "removed_invalid": removed_invalid,
        "removed_trim": removed_trim,
    }


def repair_stage11_output_refs(
    parsed: Any,
    *,
    allowed_refs: set[str] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(parsed, dict):
        return parsed, []

    repaired = copy.deepcopy(parsed)
    actions: list[dict[str, Any]] = []

    section_specs = [
        ("stable_preferences", get_section_list_node(repaired, "grounded_facts", "stable_preferences")),
        ("avoid_signals", get_section_list_node(repaired, "grounded_facts", "avoid_signals")),
        ("recent_signals", get_section_list_node(repaired, "grounded_facts", "recent_signals")),
        ("context_rules", get_section_list_node(repaired, "grounded_facts", "context_rules")),
        ("state_hypotheses", get_section_list_node(repaired, "state_hypotheses")),
        ("discriminative_signals", get_section_list_node(repaired, "discriminative_signals")),
    ]

    for section, node in section_specs:
        items = [item for item in (node or []) if isinstance(item, dict)]
        limit = SECTION_REF_LIMITS.get(section, 4)
        require_direct = section in REVIEW_REQUIRED_SECTIONS
        kept_items: list[dict[str, Any]] = []
        for item_index, item in enumerate(items, start=1):
            original_refs = item_ref_list(item)
            selected_refs, repair_meta = prioritize_representative_refs(
                original_refs,
                limit=limit,
                prefer_direct=True,
                require_direct=require_direct,
                allowed_refs=allowed_refs,
            )
            item["evidence_refs"] = selected_refs
            if repair_meta["removed_invalid"]:
                actions.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "action": "drop_invalid_refs",
                        "refs": repair_meta["removed_invalid"],
                    }
                )
            if repair_meta["removed_trim"]:
                actions.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "action": "trim_refs_to_representative_subset",
                        "refs": repair_meta["removed_trim"],
                        "ref_limit": limit,
                    }
                )
            if require_direct and not has_direct_review_ref(selected_refs):
                actions.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "action": "drop_item_missing_required_direct_review_ref",
                    }
                )
                continue
            if section not in REVIEW_REQUIRED_SECTIONS and is_context_only_ref_group(selected_refs):
                support_basis = str(item.get("support_basis", "") or "").strip()
                if support_basis and support_basis != "event_context_only":
                    item["support_basis"] = "event_context_only"
                    actions.append(
                        {
                            "section": section,
                            "item_index": item_index,
                            "action": "normalize_support_basis_to_event_context_only",
                            "previous_support_basis": support_basis,
                        }
                    )
            kept_items.append(item)
        if node is not None:
            node[:] = kept_items

    contextual_node = get_section_list_node(repaired, "contextual_inference_signals")
    contextual_items = [item for item in (contextual_node or []) if isinstance(item, dict)]
    discriminative_claims = {
        normalize_match_text(text)
        for text in collect_claim_texts(get_section_items(repaired, "discriminative_signals"))
        if normalize_match_text(text)
    }
    contextual_candidates: list[dict[str, Any]] = []
    for item_index, item in enumerate(contextual_items, start=1):
        claim_key = normalize_match_text(str(item.get("claim", "") or ""))
        if claim_key and claim_key in discriminative_claims:
            actions.append(
                {
                    "section": "contextual_inference_signals",
                    "item_index": item_index,
                    "action": "drop_cross_section_duplicate_claim",
                    "duplicate_with": "discriminative_signals",
                }
            )
            continue
        merged_refs = item_ref_list(item)
        direct_review_refs = direct_review_ref_list(item) or [ref for ref in merged_refs if is_direct_review_ref(ref)]
        contextual_refs = contextual_ref_list(item) or [ref for ref in merged_refs if not is_direct_review_ref(ref)]
        selected_direct, direct_meta = prioritize_representative_refs(
            direct_review_refs,
            limit=DIRECT_REVIEW_REF_LIMIT,
            prefer_direct=False,
            require_direct=False,
            allowed_refs=allowed_refs,
        )
        selected_contextual, contextual_meta = prioritize_representative_refs(
            contextual_refs,
            limit=CONTEXTUAL_REF_LIMIT,
            prefer_direct=False,
            require_direct=False,
            allowed_refs=allowed_refs,
        )
        item["direct_review_refs"] = selected_direct
        item["contextual_refs"] = selected_contextual
        item["evidence_refs"] = dedupe_preserving_order(selected_direct + selected_contextual)
        if direct_meta["removed_invalid"] or contextual_meta["removed_invalid"]:
            actions.append(
                {
                    "section": "contextual_inference_signals",
                    "item_index": item_index,
                    "action": "drop_invalid_refs",
                    "refs": direct_meta["removed_invalid"] + contextual_meta["removed_invalid"],
                }
            )
        trimmed_contextual = direct_meta["removed_trim"] + contextual_meta["removed_trim"]
        if trimmed_contextual:
            actions.append(
                {
                    "section": "contextual_inference_signals",
                    "item_index": item_index,
                    "action": "trim_refs_to_representative_subset",
                    "refs": trimmed_contextual,
                    "direct_review_ref_limit": DIRECT_REVIEW_REF_LIMIT,
                    "contextual_ref_limit": CONTEXTUAL_REF_LIMIT,
                }
            )
        if not selected_contextual:
            actions.append(
                {
                    "section": "contextual_inference_signals",
                    "item_index": item_index,
                    "action": "drop_item_missing_contextual_refs",
                }
            )
            continue
        previous_axis = str(item.get("canonical_axis", "") or "").strip()
        canonical_axis = infer_contextual_axis(item)
        item["canonical_axis"] = canonical_axis
        if normalize_contextual_axis(previous_axis) != canonical_axis:
            actions.append(
                {
                    "section": "contextual_inference_signals",
                    "item_index": item_index,
                    "action": "set_canonical_axis",
                    "canonical_axis": canonical_axis,
                    "previous_canonical_axis": previous_axis,
                }
            )
        contextual_candidates.append(
            {
                "item_index": item_index,
                "item": item,
                "canonical_axis": canonical_axis,
                "rank": contextual_item_rank(item, item_index=item_index),
            }
        )
    kept_contextual_items: list[dict[str, Any]] = []
    grouped_candidates: dict[str, list[dict[str, Any]]] = {}
    for candidate in contextual_candidates:
        grouped_candidates.setdefault(str(candidate["canonical_axis"]), []).append(candidate)
    for axis, group in grouped_candidates.items():
        best = max(group, key=lambda candidate: candidate["rank"])
        kept_contextual_items.append(best["item"])
        for candidate in group:
            if candidate is best:
                continue
            actions.append(
                {
                    "section": "contextual_inference_signals",
                    "item_index": candidate["item_index"],
                    "action": "drop_duplicate_contextual_axis",
                    "canonical_axis": axis,
                }
            )
    kept_contextual_items.sort(
        key=lambda item: next(
            candidate["item_index"]
            for candidate in contextual_candidates
            if candidate["item"] is item
        )
    )
    if contextual_node is not None:
        contextual_node[:] = kept_contextual_items

    unknown_node = get_section_list_node(repaired, "unknowns")
    unknown_items = [item for item in (unknown_node or []) if isinstance(item, dict)]
    unknown_limit = SECTION_REF_LIMITS.get("unknowns", 4)
    for item_index, item in enumerate(unknown_items, start=1):
        original_refs = item_ref_list(item)
        selected_refs, repair_meta = prioritize_representative_refs(
            original_refs,
            limit=unknown_limit,
            prefer_direct=True,
            require_direct=False,
            allowed_refs=allowed_refs,
        )
        item["evidence_refs"] = selected_refs
        if repair_meta["removed_invalid"]:
            actions.append(
                {
                    "section": "unknowns",
                    "item_index": item_index,
                    "action": "drop_invalid_refs",
                    "refs": repair_meta["removed_invalid"],
                }
            )
        if repair_meta["removed_trim"]:
            actions.append(
                {
                    "section": "unknowns",
                    "item_index": item_index,
                    "action": "trim_refs_to_representative_subset",
                    "refs": repair_meta["removed_trim"],
                    "ref_limit": unknown_limit,
                }
            )
    if unknown_node is not None:
        unknown_node[:] = [item for item in unknown_items if isinstance(item, dict)]

    return repaired, actions


def build_stage11_output_schema() -> dict[str, Any]:
    confidence_enum = sorted(ALLOWED_CONFIDENCE_VALUES)
    unknown_fields_enum = sorted(ALLOWED_UNKNOWN_FIELDS)
    hypothesis_types_enum = sorted(ALLOWED_HYPOTHESIS_TYPES)
    standard_support_basis_enum = sorted(STANDARD_ALLOWED_SUPPORT_BASIS)
    contextual_support_basis_enum = sorted(CONTEXTUAL_ALLOWED_SUPPORT_BASIS)
    contextual_axis_enum = sorted(ALLOWED_CONTEXTUAL_AXES)

    def ref_array() -> dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
        }

    def standard_item_schema(*, include_type: bool = False, include_why_not_generic: bool = False) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "claim": {"type": "string"},
            "confidence": {"type": "string", "enum": confidence_enum},
            "support_basis": {"type": "string", "enum": standard_support_basis_enum},
            "support_note": {"type": "string"},
            "evidence_refs": ref_array(),
        }
        required = ["claim", "confidence", "support_basis", "support_note", "evidence_refs"]
        if include_type:
            properties["type"] = {"type": "string", "enum": hypothesis_types_enum}
            required = ["type", *required]
        if include_why_not_generic:
            properties["why_not_generic"] = {"type": "string"}
            required = [*required, "why_not_generic"]
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    contextual_item_schema = {
        "type": "object",
        "properties": {
            "canonical_axis": {"type": "string", "enum": contextual_axis_enum},
            "claim": {"type": "string"},
            "confidence": {"type": "string", "enum": confidence_enum},
            "support_basis": {"type": "string", "enum": contextual_support_basis_enum},
            "support_note": {"type": "string"},
            "direct_review_refs": ref_array(),
            "contextual_refs": ref_array(),
            "evidence_refs": ref_array(),
        },
        "required": [
            "canonical_axis",
            "claim",
            "confidence",
            "support_basis",
            "support_note",
            "direct_review_refs",
            "contextual_refs",
        ],
        "additionalProperties": False,
    }

    unknown_item_schema = {
        "type": "object",
        "properties": {
            "field": {"type": "string", "enum": unknown_fields_enum},
            "reason": {"type": "string"},
            "evidence_refs": ref_array(),
        },
        "required": ["field", "reason", "evidence_refs"],
        "additionalProperties": False,
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://stage11.local/{SCHEMA_VERSION}.schema.json",
        "title": "Stage11 Prompt-Only User State Output",
        "type": "object",
        "properties": {
            "grounded_facts": {
                "type": "object",
                "properties": {
                    "stable_preferences": {
                        "type": "array",
                        "maxItems": SECTION_ITEM_LIMITS["stable_preferences"],
                        "items": standard_item_schema(),
                    },
                    "avoid_signals": {
                        "type": "array",
                        "maxItems": SECTION_ITEM_LIMITS["avoid_signals"],
                        "items": standard_item_schema(),
                    },
                    "recent_signals": {
                        "type": "array",
                        "maxItems": SECTION_ITEM_LIMITS["recent_signals"],
                        "items": standard_item_schema(),
                    },
                    "context_rules": {
                        "type": "array",
                        "maxItems": SECTION_ITEM_LIMITS["context_rules"],
                        "items": standard_item_schema(),
                    },
                },
                "additionalProperties": False,
            },
            "state_hypotheses": {
                "type": "array",
                "maxItems": SECTION_ITEM_LIMITS["state_hypotheses"],
                "items": standard_item_schema(include_type=True),
            },
            "discriminative_signals": {
                "type": "array",
                "maxItems": SECTION_ITEM_LIMITS["discriminative_signals"],
                "items": standard_item_schema(include_why_not_generic=True),
            },
            "contextual_inference_signals": {
                "type": "array",
                "maxItems": SECTION_ITEM_LIMITS["contextual_inference_signals"],
                "items": contextual_item_schema,
            },
            "unknowns": {
                "type": "array",
                "maxItems": SECTION_ITEM_LIMITS["unknowns"],
                "items": unknown_item_schema,
            },
            "confidence": {
                "type": "object",
                "properties": {
                    "overall": {"type": "string", "enum": confidence_enum},
                    "coverage": {"type": "string", "enum": confidence_enum},
                },
                "required": ["overall", "coverage"],
                "additionalProperties": False,
            },
        },
        "required": ["grounded_facts"],
        "additionalProperties": False,
    }


def _append_issue(
    issues: list[dict[str, Any]],
    *,
    issue: str,
    path: str,
    **extra: Any,
) -> None:
    payload = {"issue": issue, "path": path}
    payload.update(extra)
    issues.append(payload)


def _validate_string_enum(
    value: Any,
    *,
    path: str,
    allowed: set[str],
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(value, str):
        _append_issue(issues, issue="wrong_type", path=path, expected="string", actual=type(value).__name__)
        return
    token = value.strip().lower()
    if token not in allowed:
        _append_issue(issues, issue="invalid_enum", path=path, value=value, allowed=sorted(allowed))


def _validate_string_field(
    value: Any,
    *,
    path: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(value, str):
        _append_issue(issues, issue="wrong_type", path=path, expected="string", actual=type(value).__name__)
        return
    if not value.strip():
        _append_issue(issues, issue="empty_string", path=path)


def _validate_ref_list(
    value: Any,
    *,
    path: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(value, list):
        _append_issue(issues, issue="wrong_type", path=path, expected="array", actual=type(value).__name__)
        return
    for idx, ref in enumerate(value):
        if not isinstance(ref, str):
            _append_issue(
                issues,
                issue="wrong_type",
                path=f"{path}[{idx}]",
                expected="string",
                actual=type(ref).__name__,
            )
        elif not ref.strip():
            _append_issue(issues, issue="empty_string", path=f"{path}[{idx}]")


def _validate_item_keys(
    item: dict[str, Any],
    *,
    path: str,
    required: list[str],
    allowed: set[str],
    issues: list[dict[str, Any]],
) -> None:
    for key in required:
        if key not in item:
            _append_issue(issues, issue="missing_key", path=f"{path}.{key}")
    for key in item.keys():
        if key not in allowed:
            _append_issue(issues, issue="unexpected_key", path=f"{path}.{key}")


def _validate_standard_item(
    item: Any,
    *,
    path: str,
    issues: list[dict[str, Any]],
    include_type: bool = False,
    include_why_not_generic: bool = False,
) -> None:
    if not isinstance(item, dict):
        _append_issue(issues, issue="wrong_type", path=path, expected="object", actual=type(item).__name__)
        return
    required = ["claim", "confidence", "support_basis", "support_note", "evidence_refs"]
    allowed = set(required)
    if include_type:
        required.insert(0, "type")
        allowed.add("type")
    if include_why_not_generic:
        required.append("why_not_generic")
        allowed.add("why_not_generic")
    _validate_item_keys(item, path=path, required=required, allowed=allowed, issues=issues)
    if include_type and "type" in item:
        _validate_string_enum(item.get("type"), path=f"{path}.type", allowed=ALLOWED_HYPOTHESIS_TYPES, issues=issues)
    if "claim" in item:
        _validate_string_field(item.get("claim"), path=f"{path}.claim", issues=issues)
    if include_why_not_generic and "why_not_generic" in item:
        _validate_string_field(item.get("why_not_generic"), path=f"{path}.why_not_generic", issues=issues)
    if "confidence" in item:
        _validate_string_enum(item.get("confidence"), path=f"{path}.confidence", allowed=ALLOWED_CONFIDENCE_VALUES, issues=issues)
    if "support_basis" in item:
        _validate_string_enum(item.get("support_basis"), path=f"{path}.support_basis", allowed=STANDARD_ALLOWED_SUPPORT_BASIS, issues=issues)
    if "support_note" in item:
        _validate_string_field(item.get("support_note"), path=f"{path}.support_note", issues=issues)
    if "evidence_refs" in item:
        _validate_ref_list(item.get("evidence_refs"), path=f"{path}.evidence_refs", issues=issues)


def _validate_contextual_item(item: Any, *, path: str, issues: list[dict[str, Any]]) -> None:
    if not isinstance(item, dict):
        _append_issue(issues, issue="wrong_type", path=path, expected="object", actual=type(item).__name__)
        return
    required = [
        "canonical_axis",
        "claim",
        "confidence",
        "support_basis",
        "support_note",
        "direct_review_refs",
        "contextual_refs",
    ]
    allowed = set(required + ["evidence_refs"])
    _validate_item_keys(item, path=path, required=required, allowed=allowed, issues=issues)
    if "canonical_axis" in item:
        _validate_string_enum(item.get("canonical_axis"), path=f"{path}.canonical_axis", allowed=ALLOWED_CONTEXTUAL_AXES, issues=issues)
    if "claim" in item:
        _validate_string_field(item.get("claim"), path=f"{path}.claim", issues=issues)
    if "confidence" in item:
        _validate_string_enum(item.get("confidence"), path=f"{path}.confidence", allowed=ALLOWED_CONFIDENCE_VALUES, issues=issues)
    if "support_basis" in item:
        _validate_string_enum(item.get("support_basis"), path=f"{path}.support_basis", allowed=CONTEXTUAL_ALLOWED_SUPPORT_BASIS, issues=issues)
    if "support_note" in item:
        _validate_string_field(item.get("support_note"), path=f"{path}.support_note", issues=issues)
    if "direct_review_refs" in item:
        _validate_ref_list(item.get("direct_review_refs"), path=f"{path}.direct_review_refs", issues=issues)
    if "contextual_refs" in item:
        _validate_ref_list(item.get("contextual_refs"), path=f"{path}.contextual_refs", issues=issues)
    if "evidence_refs" in item:
        _validate_ref_list(item.get("evidence_refs"), path=f"{path}.evidence_refs", issues=issues)


def _validate_unknown_item(item: Any, *, path: str, issues: list[dict[str, Any]]) -> None:
    if not isinstance(item, dict):
        _append_issue(issues, issue="wrong_type", path=path, expected="object", actual=type(item).__name__)
        return
    required = ["field", "reason", "evidence_refs"]
    allowed = set(required)
    _validate_item_keys(item, path=path, required=required, allowed=allowed, issues=issues)
    if "field" in item:
        _validate_string_enum(item.get("field"), path=f"{path}.field", allowed=ALLOWED_UNKNOWN_FIELDS, issues=issues)
    if "reason" in item:
        _validate_string_field(item.get("reason"), path=f"{path}.reason", issues=issues)
    if "evidence_refs" in item:
        _validate_ref_list(item.get("evidence_refs"), path=f"{path}.evidence_refs", issues=issues)


def _validate_list_section(
    node: Any,
    *,
    path: str,
    issues: list[dict[str, Any]],
    item_limit: int,
    item_validator: Any,
) -> None:
    if not isinstance(node, list):
        _append_issue(issues, issue="wrong_type", path=path, expected="array", actual=type(node).__name__)
        return
    if len(node) > item_limit:
        _append_issue(issues, issue="too_many_items", path=path, count=len(node), limit=item_limit)
    for idx, item in enumerate(node):
        item_validator(item, path=f"{path}[{idx}]", issues=issues)


def validate_stage11_output_schema(parsed: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        _append_issue(issues, issue="wrong_type", path="$", expected="object", actual=type(parsed).__name__)
        return issues

    allowed_root_keys = {
        "grounded_facts",
        "state_hypotheses",
        "discriminative_signals",
        "contextual_inference_signals",
        "unknowns",
        "confidence",
    }
    for key in parsed.keys():
        if key not in allowed_root_keys:
            _append_issue(issues, issue="unexpected_key", path=f"$.{key}")

    if "grounded_facts" not in parsed:
        _append_issue(issues, issue="missing_key", path="$.grounded_facts")
    grounded = parsed.get("grounded_facts")
    if grounded is not None:
        if not isinstance(grounded, dict):
            _append_issue(issues, issue="wrong_type", path="$.grounded_facts", expected="object", actual=type(grounded).__name__)
        else:
            allowed_grounded_keys = {"stable_preferences", "avoid_signals", "recent_signals", "context_rules"}
            for key in grounded.keys():
                if key not in allowed_grounded_keys:
                    _append_issue(issues, issue="unexpected_key", path=f"$.grounded_facts.{key}")
            if "stable_preferences" in grounded:
                _validate_list_section(
                    grounded.get("stable_preferences"),
                    path="$.grounded_facts.stable_preferences",
                    issues=issues,
                    item_limit=SECTION_ITEM_LIMITS["stable_preferences"],
                    item_validator=_validate_standard_item,
                )
            if "avoid_signals" in grounded:
                _validate_list_section(
                    grounded.get("avoid_signals"),
                    path="$.grounded_facts.avoid_signals",
                    issues=issues,
                    item_limit=SECTION_ITEM_LIMITS["avoid_signals"],
                    item_validator=_validate_standard_item,
                )
            if "recent_signals" in grounded:
                _validate_list_section(
                    grounded.get("recent_signals"),
                    path="$.grounded_facts.recent_signals",
                    issues=issues,
                    item_limit=SECTION_ITEM_LIMITS["recent_signals"],
                    item_validator=_validate_standard_item,
                )
            if "context_rules" in grounded:
                _validate_list_section(
                    grounded.get("context_rules"),
                    path="$.grounded_facts.context_rules",
                    issues=issues,
                    item_limit=SECTION_ITEM_LIMITS["context_rules"],
                    item_validator=_validate_standard_item,
                )

    if "state_hypotheses" in parsed:
        _validate_list_section(
            parsed.get("state_hypotheses"),
            path="$.state_hypotheses",
            issues=issues,
            item_limit=SECTION_ITEM_LIMITS["state_hypotheses"],
            item_validator=lambda item, *, path, issues: _validate_standard_item(
                item,
                path=path,
                issues=issues,
                include_type=True,
            ),
        )
    if "discriminative_signals" in parsed:
        _validate_list_section(
            parsed.get("discriminative_signals"),
            path="$.discriminative_signals",
            issues=issues,
            item_limit=SECTION_ITEM_LIMITS["discriminative_signals"],
            item_validator=lambda item, *, path, issues: _validate_standard_item(
                item,
                path=path,
                issues=issues,
                include_why_not_generic=True,
            ),
        )
    if "contextual_inference_signals" in parsed:
        _validate_list_section(
            parsed.get("contextual_inference_signals"),
            path="$.contextual_inference_signals",
            issues=issues,
            item_limit=SECTION_ITEM_LIMITS["contextual_inference_signals"],
            item_validator=_validate_contextual_item,
        )
    if "unknowns" in parsed:
        _validate_list_section(
            parsed.get("unknowns"),
            path="$.unknowns",
            issues=issues,
            item_limit=SECTION_ITEM_LIMITS["unknowns"],
            item_validator=_validate_unknown_item,
        )
    if "confidence" in parsed:
        confidence = parsed.get("confidence")
        if not isinstance(confidence, dict):
            _append_issue(issues, issue="wrong_type", path="$.confidence", expected="object", actual=type(confidence).__name__)
        else:
            allowed_keys = {"overall", "coverage"}
            for key in confidence.keys():
                if key not in allowed_keys:
                    _append_issue(issues, issue="unexpected_key", path=f"$.confidence.{key}")
            for key in ["overall", "coverage"]:
                if key not in confidence:
                    _append_issue(issues, issue="missing_key", path=f"$.confidence.{key}")
                else:
                    _validate_string_enum(confidence.get(key), path=f"$.confidence.{key}", allowed=ALLOWED_CONFIDENCE_VALUES, issues=issues)
    return issues


def collect_cross_section_guardrail_issues(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    issues: list[dict[str, Any]] = []
    discriminative = [
        normalize_match_text(text)
        for text in collect_claim_texts(get_section_items(parsed, "discriminative_signals"))
        if normalize_match_text(text)
    ]
    contextual = [
        normalize_match_text(text)
        for text in collect_claim_texts(get_section_items(parsed, "contextual_inference_signals"))
        if normalize_match_text(text)
    ]
    overlap = sorted(set(discriminative) & set(contextual))
    for claim in overlap:
        issues.append(
            {
                "issue": "cross_section_duplicate_claim",
                "left_section": "discriminative_signals",
                "right_section": "contextual_inference_signals",
                "claim": claim,
            }
        )
    contextual_axes = [
        normalize_contextual_axis(item.get("canonical_axis"))
        for item in get_section_items(parsed, "contextual_inference_signals")
        if normalize_contextual_axis(item.get("canonical_axis"))
    ]
    duplicate_axes = [axis for axis, count in Counter(contextual_axes).items() if axis and count > 1]
    for axis in sorted(duplicate_axes):
        issues.append(
            {
                "issue": "duplicate_contextual_axis",
                "section": "contextual_inference_signals",
                "canonical_axis": axis,
            }
        )
    return issues


def schema_issue_counts(schema_issues: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(
            str(issue.get("issue", "") or "")
            for issue in schema_issues
            if isinstance(issue, dict) and issue.get("issue")
        )
    )


def clone_stage11_output_schema() -> dict[str, Any]:
    return copy.deepcopy(build_stage11_output_schema())
