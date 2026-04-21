from __future__ import annotations

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script}")
    print("This stage script is configured by environment variables and starts a data build job.")
    print("Set the required INPUT_/OUTPUT_ environment variables, then run without --help.")
    sys.exit(0)

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import (
    env_or_project_path,
    normalize_legacy_project_path,
    resolve_latest_run_pointer,
    write_latest_run_pointer,
)
from pipeline.stage11_structured_output import (
    ALLOWED_UNKNOWN_FIELDS,
    CONTEXTUAL_ALLOWED_SUPPORT_BASIS,
    REVIEW_REQUIRED_SECTIONS,
    SECTION_REF_LIMITS,
    STANDARD_ALLOWED_SUPPORT_BASIS,
    collect_unknown_fields as collect_unknown_fields_from_schema,
    contextual_ref_list as contextual_ref_list_from_schema,
    direct_review_ref_list as direct_review_ref_list_from_schema,
    get_section_items as get_section_items_from_schema,
    has_direct_review_ref as has_direct_review_ref_from_schema,
    item_ref_list as item_ref_list_from_schema,
    normalize_match_text as normalize_match_text_from_schema,
)


RUN_TAG = "stage11_pass2_signal_receiver_v1_build"
LATEST_POINTER_NAME = "stage11_pass2_signal_receiver_v1"
INPUT_POINTER_NAME = "stage11_prompt_only_user_state_audit"

INPUT_RUN_DIR_RAW = os.getenv("INPUT_11_PROMPT_ONLY_USER_STATE_AUDIT_RUN_DIR", "").strip()
INPUT_ROOT_DIR = env_or_project_path("INPUT_11_PROMPT_ONLY_USER_STATE_AUDIT_ROOT_DIR", "data/output/11_prompt_only_user_state_audit")
INPUT_ARTIFACT_PREFIX = os.getenv("INPUT_11_PROMPT_ONLY_ARTIFACT_PREFIX", "").strip()
OUTPUT_ROOT_DIR = env_or_project_path("OUTPUT_11_PASS2_SIGNAL_RECEIVER_V1_ROOT_DIR", "data/output/11_pass2_signal_receiver_v1")

SAMPLE_ROWS = int(os.getenv("PASS2_SIGNAL_RECEIVER_SAMPLE_ROWS", "10").strip() or 10)
MAX_SECTION_REFS = int(os.getenv("PASS2_SIGNAL_RECEIVER_MAX_SECTION_REFS", "4").strip() or 4)
QUALITY_FIRST_MIN_ITEMS = int(os.getenv("PASS2_SIGNAL_RECEIVER_QUALITY_FIRST_MIN_ITEMS", "2").strip() or 2)
BALANCED_MIN_ITEMS = int(os.getenv("PASS2_SIGNAL_RECEIVER_BALANCED_MIN_ITEMS", "2").strip() or 2)

QUALITY_FIRST_SECTIONS = {
    "stable_preferences",
    "avoid_signals",
    "discriminative_signals",
}

CONTEXTUAL_INFERENCE_SECTION = "contextual_inference_signals"

BALANCED_SECTIONS = {
    "stable_preferences",
    "avoid_signals",
    "recent_signals",
    "context_rules",
    "state_hypotheses",
    "discriminative_signals",
    CONTEXTUAL_INFERENCE_SECTION,
    "unknowns",
}

PROMOTED_MERCHANT_TERM_KEYS = (
    "family-friendly",
    "full bar",
    "nightlife",
    "casual vibe",
    "good for kids",
    "outdoor seating",
    "table service",
    "tourist",
    "visitor",
    "bar-seeking",
)

MERCHANT_TERM_PATTERNS = {
    "family-friendly": ["family-friendly", "family friendly", "family_friendly", "family-oriented", "family oriented"],
    "full bar": ["full bar", "full_bar"],
    "nightlife": ["nightlife"],
    "casual vibe": ["casual vibe", "casual_vibe"],
    "good for kids": ["good for kids", "good_for_kids"],
    "outdoor seating": ["outdoor seating", "outdoor_seating"],
    "table service": ["table service", "table_service"],
    "tourist": ["tourist"],
    "visitor": ["visitor"],
    "bar-seeking": ["bar-seeking", "bar seeking", "bar_seeking"],
}

SECTION_SPECS = [
    ("stable_preferences", ("grounded_facts", "stable_preferences")),
    ("avoid_signals", ("grounded_facts", "avoid_signals")),
    ("recent_signals", ("grounded_facts", "recent_signals")),
    ("context_rules", ("grounded_facts", "context_rules")),
    ("state_hypotheses", ("state_hypotheses",)),
    ("discriminative_signals", ("discriminative_signals",)),
    (CONTEXTUAL_INFERENCE_SECTION, (CONTEXTUAL_INFERENCE_SECTION,)),
    ("unknowns", ("unknowns",)),
]

AXIS_PATTERNS: dict[str, list[str]] = {
    "cuisine_dish": [
        "seafood", "oyster", "oysters", "gumbo", "sushi", "tataki", "brunch", "breakfast", "burger", "salad",
        "coffee", "tea", "dessert", "ramen", "pizza", "bbq", "barbecue", "cajun", "creole", "mexican",
        "italian", "japanese", "chinese", "thai", "indian", "mediterranean", "vegan", "vegetarian",
    ],
    "service": [
        "service", "server", "staff", "attentive", "inattention", "forgot", "forgetting", "rude", "friendly",
        "menu provision", "wait", "waiting",
    ],
    "value": [
        "price", "priced", "overpriced", "value", "worth", "portion", "expensive", "cheap", "cost",
    ],
    "time_mode": [
        "breakfast", "brunch", "lunch", "dinner", "late-night", "late night", "morning", "coffee", "tea",
    ],
    "scene": [
        "group", "family", "friends", "solo", "date", "quick bite", "sit-down", "sit down", "casual", "lively",
        "quiet", "upscale", "hip",
    ],
    "geo_context": [
        "new orleans", "travel", "trip", "visitor", "tourist", "local",
    ],
}


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_match_text(text: Any) -> str:
    return normalize_match_text_from_schema(text)


def read_json_payload(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json_payload(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def resolve_latest_run(root_dir: Path, suffix: str) -> Path | None:
    if not root_dir.exists():
        return None
    candidates = [
        path
        for path in root_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_run_dir() -> Path:
    if INPUT_RUN_DIR_RAW:
        run_dir = normalize_legacy_project_path(INPUT_RUN_DIR_RAW)
        if not run_dir.exists():
            raise FileNotFoundError(f"input run dir not found: {run_dir}")
        return run_dir
    pointed = resolve_latest_run_pointer(INPUT_POINTER_NAME)
    if pointed is not None and pointed.exists():
        return pointed
    latest = resolve_latest_run(INPUT_ROOT_DIR, "_full_stage11_prompt_only_user_state_audit")
    if latest is None:
        raise FileNotFoundError(f"no stage11 prompt-only audit run found under {INPUT_ROOT_DIR}")
    return latest


def choose_artifact_prefix(run_dir: Path) -> str:
    if INPUT_ARTIFACT_PREFIX:
        return INPUT_ARTIFACT_PREFIX
    preferred = [
        "prompt_only_pass2_generation_compact_v2_2048",
        "prompt_only_pass2_generation_compact_2048",
        "prompt_only_pass2_generation_2560",
        "prompt_only_pass2_generation",
        "prompt_only_generation",
    ]
    for prefix in preferred:
        if (run_dir / f"{prefix}_parsed.json").exists() or (run_dir / f"{prefix}_parsed.jsonl").exists():
            return prefix
    raise FileNotFoundError(f"no pass2 artifact found under {run_dir}")


def load_generation_rows(run_dir: Path, artifact_prefix: str, kind: str) -> list[dict[str, Any]]:
    json_path = run_dir / f"{artifact_prefix}_{kind}.json"
    jsonl_path = run_dir / f"{artifact_prefix}_{kind}.jsonl"
    if json_path.exists():
        return read_json_rows(json_path)
    if jsonl_path.exists():
        return read_jsonl_rows(jsonl_path)
    return []


def get_section_items(parsed: Any, *path: str) -> list[dict[str, Any]]:
    return get_section_items_from_schema(parsed, *path)


def collect_unknown_fields(parsed: Any) -> list[str]:
    return collect_unknown_fields_from_schema(parsed)


def find_term_hits(texts: list[str], allowed_keys: tuple[str, ...] | None = None) -> list[str]:
    normalized_texts = [normalize_match_text(text) for text in texts if str(text or "").strip()]
    allowed = set(allowed_keys) if allowed_keys else None
    hits: list[str] = []
    for label, variants in MERCHANT_TERM_PATTERNS.items():
        if allowed is not None and label not in allowed:
            continue
        normalized_variants = [normalize_match_text(variant) for variant in variants]
        if any(variant and variant in text for text in normalized_texts for variant in normalized_variants):
            hits.append(label)
    return sorted(set(hits))


def detect_axes(texts: list[str]) -> list[str]:
    joined = normalize_match_text(" ".join(texts))
    axes: list[str] = []
    for axis, variants in AXIS_PATTERNS.items():
        if any(normalize_match_text(variant) in joined for variant in variants):
            axes.append(axis)
    return sorted(set(axes))


def item_text(item: dict[str, Any], section: str) -> str:
    field = "reason" if section == "unknowns" else "claim"
    return str(item.get(field, "") or "").strip()


def item_ref_list(item: dict[str, Any]) -> list[str]:
    return item_ref_list_from_schema(item)


def direct_review_ref_list(item: dict[str, Any]) -> list[str]:
    return direct_review_ref_list_from_schema(item)


def contextual_ref_list(item: dict[str, Any]) -> list[str]:
    return contextual_ref_list_from_schema(item)


def has_direct_review_ref(refs: list[str]) -> bool:
    return has_direct_review_ref_from_schema(refs)


def build_text_key(section: str, item: dict[str, Any]) -> str:
    parts = [section, item_text(item, section)]
    if section == "unknowns":
        parts.append(str(item.get("field", "") or "").strip())
    if section == "discriminative_signals":
        parts.append(str(item.get("why_not_generic", "") or "").strip())
    return normalize_match_text(" | ".join(parts))


def signal_text_payload(item: dict[str, Any], section: str) -> list[str]:
    texts = [item_text(item, section)]
    why_not_generic = str(item.get("why_not_generic", "") or "").strip()
    if why_not_generic:
        texts.append(why_not_generic)
    return [text for text in texts if text]


def has_supported_cuisine_signal(item_rows: list[dict[str, Any]]) -> bool:
    for row in item_rows:
        if row["section"] not in {"stable_preferences", "discriminative_signals"}:
            continue
        if "cuisine_dish" in row["signal_axes"]:
            return True
    return False


def confidence_rank(value: Any) -> int:
    token = str(value or "").strip().lower()
    if token == "high":
        return 3
    if token == "medium":
        return 2
    if token == "low":
        return 1
    return 0


def build_item_rows(
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_by_index = {int(row.get("input_index", -1)): row for row in raw_rows}
    item_rows: list[dict[str, Any]] = []

    for parsed_row in parsed_rows:
        parsed = parsed_row.get("parsed_json")
        if not isinstance(parsed, dict):
            continue
        input_index = int(parsed_row.get("input_index", -1))
        raw_row = raw_by_index.get(input_index, {})
        allowed_refs = set(str(ref).strip() for ref in raw_row.get("allowed_evidence_refs", []) if str(ref).strip())
        section_counts: dict[str, Counter[str]] = {}

        for section, path in SECTION_SPECS:
            items = get_section_items(parsed, *path)
            counter = Counter(build_text_key(section, item) for item in items)
            section_counts[section] = counter
            for item_index, item in enumerate(items, start=1):
                refs = item_ref_list(item)
                texts = signal_text_payload(item, section)
                unknown_field = str(item.get("field", "") or "").strip() if section == "unknowns" else ""
                row = {
                    "user_id": str(parsed_row.get("user_id", "") or ""),
                    "density_band": str(parsed_row.get("density_band", "") or ""),
                    "selected_rank_in_band": int(parsed_row.get("selected_rank_in_band", 0) or 0),
                    "input_index": input_index,
                    "selection_score": float(parsed_row.get("selection_score", 0.0) or 0.0),
                    "section": section,
                    "item_index": item_index,
                    "text": item_text(item, section),
                    "why_not_generic": str(item.get("why_not_generic", "") or "").strip(),
                    "hypothesis_type": str(item.get("type", "") or "").strip(),
                    "unknown_field": unknown_field,
                    "confidence": str(item.get("confidence", "") or "").strip().lower(),
                    "confidence_rank": confidence_rank(item.get("confidence")),
                    "support_basis": str(item.get("support_basis", "") or "").strip(),
                    "support_note": str(item.get("support_note", "") or "").strip(),
                    "evidence_refs": refs,
                    "evidence_ref_count": len(refs),
                    "direct_review_refs": direct_review_ref_list(item),
                    "contextual_refs": contextual_ref_list(item),
                    "has_direct_review_ref": has_direct_review_ref(direct_review_ref_list(item)),
                    "invalid_evidence_refs": [ref for ref in refs if allowed_refs and ref not in allowed_refs],
                    "signal_axes": detect_axes(texts),
                    "merchant_term_hits": find_term_hits(
                        texts,
                        PROMOTED_MERCHANT_TERM_KEYS if section in QUALITY_FIRST_SECTIONS or section == "state_hypotheses" else None,
                    ),
                    "duplicate_in_section": section_counts[section][build_text_key(section, item)] > 1,
                    "json_valid": bool(raw_row.get("json_valid", True)),
                    "schema_valid": bool(raw_row.get("schema_valid", True)),
                    "guardrails_passed": bool(raw_row.get("guardrails_passed", True)),
                    "guardrail_failure_count": len(raw_row.get("guardrail_failures", []) or []),
                    "raw_ref_group_sizes": raw_row.get("ref_group_sizes", []),
                }
                item_rows.append(row)
    return item_rows


def classify_item_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        rows_by_user[row["user_id"]].append(row)

    out: list[dict[str, Any]] = []
    for user_id, user_rows in rows_by_user.items():
        cuisine_supported = has_supported_cuisine_signal(user_rows)
        for row in user_rows:
            flags: list[str] = []
            allowed_support_basis = (
                CONTEXTUAL_ALLOWED_SUPPORT_BASIS
                if row["section"] == CONTEXTUAL_INFERENCE_SECTION
                else STANDARD_ALLOWED_SUPPORT_BASIS
            )
            section_ref_limit = SECTION_REF_LIMITS.get(row["section"], MAX_SECTION_REFS)
            if not row["json_valid"]:
                flags.append("json_invalid")
            if not row["schema_valid"]:
                flags.append("schema_invalid")
            if not row["guardrails_passed"]:
                flags.append("guardrail_failed")
            if not row["text"]:
                flags.append("empty_text")
            if row["invalid_evidence_refs"]:
                flags.append("invalid_refs")
            if row["evidence_ref_count"] > section_ref_limit:
                flags.append("wide_ref_group")
            if row["duplicate_in_section"]:
                flags.append("duplicate_in_section")
            if row["merchant_term_hits"]:
                flags.append("merchant_term_leak")
            if row["section"] != "unknowns" and row["support_basis"] not in allowed_support_basis:
                flags.append("invalid_support_basis")
            if row["section"] != "unknowns" and not row["support_note"]:
                flags.append("missing_support_note")
            if row["section"] in REVIEW_REQUIRED_SECTIONS and not row["has_direct_review_ref"]:
                flags.append("missing_direct_review_ref")
            if row["section"] in REVIEW_REQUIRED_SECTIONS and row["support_basis"] == "event_context_only":
                flags.append("event_context_shortcut")
            if row["section"] == CONTEXTUAL_INFERENCE_SECTION and not row["contextual_refs"]:
                flags.append("missing_contextual_refs")
            if row["section"] == CONTEXTUAL_INFERENCE_SECTION and len(row["direct_review_refs"]) > 2:
                flags.append("too_many_direct_review_refs")
            if row["section"] == CONTEXTUAL_INFERENCE_SECTION and len(row["contextual_refs"]) > 2:
                flags.append("too_many_contextual_refs")
            if row["section"] == "unknowns" and row["unknown_field"] not in ALLOWED_UNKNOWN_FIELDS:
                flags.append("invalid_unknown_field")
            if row["section"] == "unknowns" and row["unknown_field"] == "cuisine_preference" and cuisine_supported:
                flags.append("redundant_cuisine_unknown")

            hard_fail = any(
                flag in {
                    "json_invalid",
                    "schema_invalid",
                    "guardrail_failed",
                    "empty_text",
                    "invalid_refs",
                    "wide_ref_group",
                    "duplicate_in_section",
                    "merchant_term_leak",
                    "invalid_support_basis",
                    "missing_support_note",
                    "missing_direct_review_ref",
                    "event_context_shortcut",
                    "missing_contextual_refs",
                    "too_many_direct_review_refs",
                    "too_many_contextual_refs",
                    "invalid_unknown_field",
                }
                for flag in flags
            )
            accepted_quality_first = (
                row["section"] in QUALITY_FIRST_SECTIONS
                and not hard_fail
                and row["confidence_rank"] >= 2
            )
            accepted_balanced = (
                row["section"] in BALANCED_SECTIONS
                and not hard_fail
                and "redundant_cuisine_unknown" not in flags
                and (
                    row["section"] == "unknowns"
                    or row["confidence_rank"] >= 1
                )
            )
            quality_tier = "reject"
            if accepted_quality_first:
                quality_tier = "gold"
            elif accepted_balanced:
                quality_tier = "silver"

            enriched = dict(row)
            enriched["flags"] = sorted(set(flags))
            enriched["hard_fail"] = hard_fail
            enriched["accepted_quality_first"] = accepted_quality_first
            enriched["accepted_balanced"] = accepted_balanced
            enriched["quality_tier"] = quality_tier
            out.append(enriched)
    return out


def build_teacher_payload(item_rows: list[dict[str, Any]], *, include_unknowns: bool) -> dict[str, Any]:
    grounded_facts = {
        "stable_preferences": [],
        "avoid_signals": [],
        "recent_signals": [],
        "context_rules": [],
    }
    payload: dict[str, Any] = {
        "grounded_facts": grounded_facts,
        "state_hypotheses": [],
        "discriminative_signals": [],
        CONTEXTUAL_INFERENCE_SECTION: [],
        "unknowns": [],
    }
    for row in item_rows:
        entry: dict[str, Any]
        if row["section"] == "unknowns":
            if not include_unknowns:
                continue
            entry = {
                "field": row["unknown_field"],
                "reason": row["text"],
                "evidence_refs": row["evidence_refs"],
            }
            payload["unknowns"].append(entry)
            continue

        entry = {
            "claim": row["text"],
            "confidence": row["confidence"] or "medium",
            "support_basis": row["support_basis"],
            "support_note": row["support_note"],
            "evidence_refs": row["evidence_refs"],
        }
        if row["section"] == "discriminative_signals" and row["why_not_generic"]:
            entry["why_not_generic"] = row["why_not_generic"]
        if row["section"] == "state_hypotheses" and row["hypothesis_type"]:
            entry["type"] = row["hypothesis_type"]
        if row["section"] == CONTEXTUAL_INFERENCE_SECTION:
            entry["direct_review_refs"] = row["direct_review_refs"]
            entry["contextual_refs"] = row["contextual_refs"]

        if row["section"] in grounded_facts:
            grounded_facts[row["section"]].append(entry)
        else:
            payload[row["section"]].append(entry)

    empty_grounded = {key: value for key, value in grounded_facts.items() if value}
    payload["grounded_facts"] = empty_grounded
    if not include_unknowns:
        payload.pop("unknowns", None)
    payload["meta"] = {
        "item_count": len(item_rows),
        "sections": dict(Counter(row["section"] for row in item_rows)),
        "axes": sorted(set(axis for row in item_rows for axis in row["signal_axes"])),
    }
    return payload


def build_unknown_mask_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in item_rows:
        if row["section"] != "unknowns" or not row["accepted_balanced"]:
            continue
        out.append(
            {
                "user_id": row["user_id"],
                "density_band": row["density_band"],
                "input_index": row["input_index"],
                "unknown_field": row["unknown_field"],
                "reason": row["text"],
                "evidence_refs_json": json.dumps(row["evidence_refs"], ensure_ascii=False),
            }
        )
    return out


def build_user_summary_rows(item_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        rows_by_user[row["user_id"]].append(row)

    out: list[dict[str, Any]] = []
    for user_id, rows in rows_by_user.items():
        rows.sort(key=lambda row: (row["input_index"], row["section"], row["item_index"]))
        quality_first_rows = [row for row in rows if row["accepted_quality_first"]]
        balanced_rows = [row for row in rows if row["accepted_balanced"]]
        quality_first_payload = build_teacher_payload(quality_first_rows, include_unknowns=False)
        balanced_payload = build_teacher_payload(balanced_rows, include_unknowns=True)
        flag_counts = Counter(flag for row in rows for flag in row["flags"])
        section_counts = Counter(row["section"] for row in rows)
        axes = sorted(set(axis for row in balanced_rows for axis in row["signal_axes"]))
        hard_fail_item_count = sum(1 for row in rows if row["hard_fail"])
        quality_first_payload_available = len(quality_first_rows) >= QUALITY_FIRST_MIN_ITEMS
        balanced_payload_available = len(balanced_rows) >= BALANCED_MIN_ITEMS
        strict_quality_first_ready = quality_first_payload_available and hard_fail_item_count == 0

        recommended_pool = "reject"
        if strict_quality_first_ready:
            recommended_pool = "quality_first"
        elif balanced_payload_available:
            recommended_pool = "balanced"

        out.append(
            {
                "user_id": user_id,
                "density_band": rows[0]["density_band"],
                "input_index": rows[0]["input_index"],
                "selection_score": rows[0]["selection_score"],
                "items_total": len(rows),
                "quality_first_item_count": len(quality_first_rows),
                "balanced_item_count": len(balanced_rows),
                "rejected_item_count": len(rows) - len(balanced_rows),
                "hard_fail_item_count": hard_fail_item_count,
                "quality_first_payload_available": quality_first_payload_available,
                "balanced_payload_available": balanced_payload_available,
                "strict_quality_first_ready": strict_quality_first_ready,
                "section_counts_json": json.dumps(dict(section_counts), ensure_ascii=False),
                "flag_counts_json": json.dumps(dict(flag_counts), ensure_ascii=False),
                "signal_axes_json": json.dumps(axes, ensure_ascii=False),
                "recommended_pool": recommended_pool,
                "quality_first_teacher_json": json.dumps(quality_first_payload, ensure_ascii=False),
                "balanced_teacher_json": json.dumps(balanced_payload, ensure_ascii=False),
            }
        )
    return out


def summarize_outputs(
    item_rows: list[dict[str, Any]],
    user_summary_rows: list[dict[str, Any]],
    *,
    input_run_dir: Path,
    artifact_prefix: str,
) -> dict[str, Any]:
    quality_first_items = [row for row in item_rows if row["accepted_quality_first"]]
    balanced_items = [row for row in item_rows if row["accepted_balanced"]]
    summary = {
        "run_tag": RUN_TAG,
        "input_run_dir": str(input_run_dir),
        "input_artifact_prefix": artifact_prefix,
        "users_total": len(user_summary_rows),
        "items_total": len(item_rows),
        "quality_first_items": len(quality_first_items),
        "balanced_items": len(balanced_items),
        "rejected_items": len(item_rows) - len(balanced_items),
        "quality_first_users": sum(1 for row in user_summary_rows if row["recommended_pool"] == "quality_first"),
        "balanced_users": sum(1 for row in user_summary_rows if row["recommended_pool"] == "balanced"),
        "rejected_users": sum(1 for row in user_summary_rows if row["recommended_pool"] == "reject"),
        "quality_first_payload_available_users": sum(1 for row in user_summary_rows if row["quality_first_payload_available"]),
        "section_counts": dict(Counter(row["section"] for row in item_rows)),
        "quality_first_section_counts": dict(Counter(row["section"] for row in quality_first_items)),
        "balanced_section_counts": dict(Counter(row["section"] for row in balanced_items)),
        "flag_counts": dict(Counter(flag for row in item_rows for flag in row["flags"])),
        "quality_first_axis_counts": dict(Counter(axis for row in quality_first_items for axis in row["signal_axes"])),
        "balanced_axis_counts": dict(Counter(axis for row in balanced_items for axis in row["signal_axes"])),
    }
    return summary


def build_sample_payload(
    item_rows: list[dict[str, Any]],
    user_summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    quality_first_items = [row for row in item_rows if row["accepted_quality_first"]][:SAMPLE_ROWS]
    rejected_items = [row for row in item_rows if row["quality_tier"] == "reject"][:SAMPLE_ROWS]
    users = sorted(
        user_summary_rows,
        key=lambda row: (
            {"quality_first": 0, "balanced": 1, "reject": 2}.get(row["recommended_pool"], 9),
            -int(row["quality_first_item_count"]),
            -int(row["balanced_item_count"]),
        ),
    )
    return {
        "quality_first_item_samples": quality_first_items,
        "rejected_item_samples": rejected_items,
        "user_samples": users[:SAMPLE_ROWS],
    }


def main() -> None:
    input_run_dir = resolve_run_dir()
    artifact_prefix = choose_artifact_prefix(input_run_dir)

    raw_rows = load_generation_rows(input_run_dir, artifact_prefix, "raw")
    parsed_rows = load_generation_rows(input_run_dir, artifact_prefix, "parsed")
    if not raw_rows or not parsed_rows:
        raise RuntimeError(f"missing generation artifacts for prefix={artifact_prefix} under {input_run_dir}")

    item_rows = classify_item_rows(build_item_rows(raw_rows, parsed_rows))
    user_summary_rows = build_user_summary_rows(item_rows)
    unknown_mask_rows = build_unknown_mask_rows(item_rows)

    output_run_dir = OUTPUT_ROOT_DIR / f"{now_run_id()}_full_stage11_pass2_signal_receiver_v1_build"
    output_run_dir.mkdir(parents=True, exist_ok=True)

    item_df = pd.DataFrame(item_rows)
    if not item_df.empty:
        item_df = item_df.assign(
            evidence_refs_json=item_df["evidence_refs"].apply(lambda value: json.dumps(value, ensure_ascii=False)),
            invalid_evidence_refs_json=item_df["invalid_evidence_refs"].apply(lambda value: json.dumps(value, ensure_ascii=False)),
            signal_axes_json=item_df["signal_axes"].apply(lambda value: json.dumps(value, ensure_ascii=False)),
            merchant_term_hits_json=item_df["merchant_term_hits"].apply(lambda value: json.dumps(value, ensure_ascii=False)),
            flags_json=item_df["flags"].apply(lambda value: json.dumps(value, ensure_ascii=False)),
            raw_ref_group_sizes_json=item_df["raw_ref_group_sizes"].apply(lambda value: json.dumps(value, ensure_ascii=False)),
        ).drop(columns=["evidence_refs", "invalid_evidence_refs", "signal_axes", "merchant_term_hits", "flags", "raw_ref_group_sizes"])

    quality_first_df = item_df[item_df["accepted_quality_first"]].copy() if not item_df.empty else pd.DataFrame()
    balanced_df = item_df[item_df["accepted_balanced"]].copy() if not item_df.empty else pd.DataFrame()
    unknown_mask_df = pd.DataFrame(unknown_mask_rows)
    user_summary_df = pd.DataFrame(user_summary_rows)
    model_target_columns = [
        "user_id",
        "density_band",
        "input_index",
        "selection_score",
        "quality_first_item_count",
        "balanced_item_count",
        "hard_fail_item_count",
        "quality_first_payload_available",
        "balanced_payload_available",
        "strict_quality_first_ready",
        "recommended_pool",
        "signal_axes_json",
        "quality_first_teacher_json",
        "balanced_teacher_json",
    ]
    model_targets_df = user_summary_df[model_target_columns].copy() if not user_summary_df.empty else pd.DataFrame(columns=model_target_columns)

    item_df.to_parquet(output_run_dir / "pass2_receiver_signal_items.parquet", index=False)
    quality_first_df.to_parquet(output_run_dir / "pass2_receiver_quality_first_items.parquet", index=False)
    balanced_df.to_parquet(output_run_dir / "pass2_receiver_balanced_items.parquet", index=False)
    unknown_mask_df.to_parquet(output_run_dir / "pass2_receiver_unknown_masks.parquet", index=False)
    user_summary_df.to_parquet(output_run_dir / "pass2_receiver_user_summary.parquet", index=False)
    model_targets_df.to_parquet(output_run_dir / "pass2_receiver_model_targets.parquet", index=False)

    summary = summarize_outputs(item_rows, user_summary_rows, input_run_dir=input_run_dir, artifact_prefix=artifact_prefix)
    sample_payload = build_sample_payload(item_rows, user_summary_rows)
    safe_json_write(output_run_dir / "pass2_receiver_summary.json", summary)
    safe_json_write(output_run_dir / "pass2_receiver_samples.json", sample_payload)
    safe_json_write(
        output_run_dir / "run_meta.json",
        {
            "run_tag": RUN_TAG,
            "input_run_dir": str(input_run_dir),
            "input_artifact_prefix": artifact_prefix,
            "input_rows_raw": len(raw_rows),
            "input_rows_parsed": len(parsed_rows),
        },
    )

    write_latest_run_pointer(
        LATEST_POINTER_NAME,
        output_run_dir,
        extra={
            "run_tag": RUN_TAG,
            "input_run_dir": str(input_run_dir),
            "input_artifact_prefix": artifact_prefix,
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_run_dir={output_run_dir}")


if __name__ == "__main__":
    main()
