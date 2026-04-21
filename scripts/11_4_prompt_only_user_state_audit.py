from __future__ import annotations

import copy
import gc
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd

from pipeline.stage11_structured_output import (
    ALLOWED_UNKNOWN_FIELDS,
    CONTEXTUAL_ALLOWED_SUPPORT_BASIS,
    REVIEW_REQUIRED_SECTIONS,
    SCHEMA_VERSION as STRUCTURED_OUTPUT_SCHEMA_VERSION,
    SECTION_REF_LIMITS,
    STANDARD_ALLOWED_SUPPORT_BASIS,
    build_stage11_output_schema,
    clone_stage11_output_schema,
    collect_cross_section_guardrail_issues,
    collect_evidence_refs as collect_evidence_refs_from_schema,
    collect_ref_groups as collect_ref_groups_from_schema,
    collect_unknown_fields as collect_unknown_fields_from_schema,
    contextual_ref_list as contextual_ref_list_from_schema,
    direct_review_ref_list as direct_review_ref_list_from_schema,
    get_section_items as get_section_items_from_schema,
    get_section_list_node as get_section_list_node_from_schema,
    has_direct_review_ref as has_direct_review_ref_from_schema,
    item_ref_list as item_ref_list_from_schema,
    normalize_match_text as normalize_match_text_from_schema,
    normalize_ref_list as normalize_ref_list_from_schema,
    repair_stage11_output_refs,
    validate_stage11_output_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_USER_STATE_RUN_DIR = PROJECT_ROOT / "data" / "output" / "09_user_state_training_assets_v1" / "20260415_191252_full_stage09_user_state_training_assets_v1_build"
DEFAULT_INPUT_MERCHANT_STATE_RUN_DIR = PROJECT_ROOT / "data" / "output" / "09_merchant_state_v2" / "20260415_195206_full_stage09_merchant_state_v2_build"
DEFAULT_PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "docs" / "stage11" / "qwen35_prompt_only_user_state_summary_v2_prompt.md"
DEFAULT_OUTPUT_ROOT_DIR = PROJECT_ROOT / "data" / "output" / "11_prompt_only_user_state_audit"


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


def env_optional_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def env_path_list(name: str) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


INPUT_USER_STATE_RUN_DIR = env_path("INPUT_09_USER_STATE_TRAINING_ASSETS_V1_RUN_DIR", DEFAULT_INPUT_USER_STATE_RUN_DIR)
INPUT_MERCHANT_STATE_RUN_DIR = env_path("INPUT_09_MERCHANT_STATE_V2_RUN_DIR", DEFAULT_INPUT_MERCHANT_STATE_RUN_DIR)
PROMPT_TEMPLATE_PATH = env_path("INPUT_QWEN35_PROMPT_TEMPLATE_PATH", DEFAULT_PROMPT_TEMPLATE_PATH)
PASS2_PROMPT_TEMPLATE_PATH = env_optional_path("INPUT_QWEN35_PASS2_PROMPT_TEMPLATE_PATH")
OUTPUT_ROOT_DIR = env_path("OUTPUT_11_PROMPT_ONLY_USER_STATE_AUDIT_ROOT_DIR", DEFAULT_OUTPUT_ROOT_DIR)
OUTPUT_RUN_DIR = env_optional_path("OUTPUT_11_PROMPT_ONLY_USER_STATE_AUDIT_RUN_DIR")
EXISTING_PASS1_RUN_DIR = env_optional_path("INPUT_11_PROMPT_ONLY_EXISTING_PASS1_RUN_DIR")
EXCLUDE_COMPLETED_PASS1_RUN_DIRS = env_path_list("PROMPT_ONLY_EXCLUDE_COMPLETED_PASS1_RUN_DIRS")
RESUME_FROM_EXISTING = os.environ.get("PROMPT_ONLY_RESUME_FROM_EXISTING", "false").strip().lower() == "true"
SINGLE_STAGE_ARTIFACT_PREFIX = os.environ.get("PROMPT_ONLY_SINGLE_STAGE_ARTIFACT_PREFIX", "").strip()

SAMPLE_PER_BAND = int(os.environ.get("PROMPT_ONLY_SAMPLE_PER_BAND", "1"))
SELECT_MODE = os.environ.get("PROMPT_ONLY_SELECT_MODE", "sample_per_band").strip().lower()
MAX_TOTAL_ROWS = int(os.environ.get("PROMPT_ONLY_MAX_TOTAL_ROWS", "0"))
MAX_REVIEW_EVIDENCE = int(os.environ.get("PROMPT_ONLY_MAX_REVIEW_EVIDENCE", "36"))
MAX_TIP_EVIDENCE = int(os.environ.get("PROMPT_ONLY_MAX_TIP_EVIDENCE", "6"))
MAX_RECENT_EVENTS = int(os.environ.get("PROMPT_ONLY_MAX_RECENT_EVENTS", "8"))
MAX_POS_ANCHORS = int(os.environ.get("PROMPT_ONLY_MAX_POS_ANCHORS", "4"))
MAX_NEG_ANCHORS = int(os.environ.get("PROMPT_ONLY_MAX_NEG_ANCHORS", "4"))
MAX_CONFLICT_EVENTS = int(os.environ.get("PROMPT_ONLY_MAX_CONFLICT_EVENTS", "4"))
SANITIZE_MERCHANT_CONTEXT = os.environ.get("PROMPT_ONLY_SANITIZE_MERCHANT_CONTEXT", "true").strip().lower() == "true"
MAX_EVENTS_PER_PRIMARY_CUISINE = int(os.environ.get("PROMPT_ONLY_MAX_EVENTS_PER_PRIMARY_CUISINE", "2"))
MAX_ANCHORS_PER_PRIMARY_CUISINE = int(os.environ.get("PROMPT_ONLY_MAX_ANCHORS_PER_PRIMARY_CUISINE", "1"))
STRIP_REVIEW_METADATA = os.environ.get("PROMPT_ONLY_STRIP_REVIEW_METADATA", "true").strip().lower() == "true"
STRIP_EVENT_METADATA = os.environ.get("PROMPT_ONLY_STRIP_EVENT_METADATA", "true").strip().lower() == "true"
MINIMAL_CONTEXT_NOTES = os.environ.get("PROMPT_ONLY_MINIMAL_CONTEXT_NOTES", "true").strip().lower() == "true"

RUN_INFERENCE = os.environ.get("PROMPT_ONLY_RUN_INFERENCE", "false").strip().lower() == "true"
RUN_PASS2 = os.environ.get("PROMPT_ONLY_RUN_PASS2", "false").strip().lower() == "true"
RUN_PASS2_COMPACT_INPUT = os.environ.get("PROMPT_ONLY_PASS2_COMPACT_INPUT", "false").strip().lower() == "true"
BASE_MODEL = os.environ.get("PROMPT_ONLY_BASE_MODEL", "/root/autodl-tmp/models/Qwen3.5-35B-A3B-Base").strip()
GEN_BATCH_SIZE = int(os.environ.get("PROMPT_ONLY_GEN_BATCH_SIZE", "2"))
GEN_MAX_NEW_TOKENS = int(os.environ.get("PROMPT_ONLY_GEN_MAX_NEW_TOKENS", "4096"))
PASS2_GEN_MAX_NEW_TOKENS = int(os.environ.get("PROMPT_ONLY_PASS2_GEN_MAX_NEW_TOKENS", str(GEN_MAX_NEW_TOKENS)))
ATTN_IMPL = os.environ.get("PROMPT_ONLY_ATTN_IMPL", "flash_attention_2").strip()
USE_BF16 = os.environ.get("PROMPT_ONLY_USE_BF16", "true").strip().lower() == "true"
DISABLE_GROUPED_MM = os.environ.get("PROMPT_ONLY_DISABLE_GROUPED_MM", "false").strip().lower() == "true"
SORT_BY_PROMPT_LENGTH = os.environ.get("PROMPT_ONLY_SORT_BY_PROMPT_LENGTH", "true").strip().lower() == "true"
MAX_BATCH_PROMPT_TOKENS = int(os.environ.get("PROMPT_ONLY_MAX_BATCH_PROMPT_TOKENS", "0"))
INFER_BACKEND = os.environ.get("PROMPT_ONLY_INFER_BACKEND", "hf").strip().lower() or "hf"
GUARDRAILS_HARD_FAIL = os.environ.get("PROMPT_ONLY_GUARDRAILS_HARD_FAIL", "true").strip().lower() == "true"
WRITE_STRUCTURED_OUTPUT_SCHEMA = os.environ.get("PROMPT_ONLY_WRITE_STRUCTURED_OUTPUT_SCHEMA", "true").strip().lower() == "true"
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("PROMPT_ONLY_VLLM_GPU_MEMORY_UTILIZATION", "0.95").strip() or 0.95)
VLLM_MAX_MODEL_LEN = int(os.environ.get("PROMPT_ONLY_VLLM_MAX_MODEL_LEN", "0").strip() or 0)
VLLM_MAX_NUM_SEQS = int(os.environ.get("PROMPT_ONLY_VLLM_MAX_NUM_SEQS", "0").strip() or 0)
VLLM_MAX_NUM_BATCHED_TOKENS = int(os.environ.get("PROMPT_ONLY_VLLM_MAX_NUM_BATCHED_TOKENS", "0").strip() or 0)
VLLM_TENSOR_PARALLEL_SIZE = int(os.environ.get("PROMPT_ONLY_VLLM_TENSOR_PARALLEL_SIZE", "1").strip() or 1)
VLLM_ENFORCE_EAGER = os.environ.get("PROMPT_ONLY_VLLM_ENFORCE_EAGER", "false").strip().lower() == "true"
VLLM_REQUIRE_GUIDED_DECODING = os.environ.get("PROMPT_ONLY_VLLM_REQUIRE_GUIDED_DECODING", "true").strip().lower() == "true"
PASS2_COMPACT_FALLBACK_USER_EVIDENCE = int(os.environ.get("PROMPT_ONLY_PASS2_COMPACT_FALLBACK_USER_EVIDENCE", "2"))
PASS2_COMPACT_FALLBACK_RECENT_EVENTS = int(os.environ.get("PROMPT_ONLY_PASS2_COMPACT_FALLBACK_RECENT_EVENTS", "2"))
PASS2_COMPACT_FALLBACK_POS_ANCHORS = int(os.environ.get("PROMPT_ONLY_PASS2_COMPACT_FALLBACK_POS_ANCHORS", "1"))
PASS2_COMPACT_FALLBACK_NEG_ANCHORS = int(os.environ.get("PROMPT_ONLY_PASS2_COMPACT_FALLBACK_NEG_ANCHORS", "1"))
PASS2_COMPACT_FALLBACK_CONFLICT_ANCHORS = int(os.environ.get("PROMPT_ONLY_PASS2_COMPACT_FALLBACK_CONFLICT_ANCHORS", "1"))
CHECKPOINT_EVERY_BATCHES = int(os.environ.get("PROMPT_ONLY_CHECKPOINT_EVERY_BATCHES", "0"))
CHECKPOINT_SAMPLE_ROWS = int(os.environ.get("PROMPT_ONLY_CHECKPOINT_SAMPLE_ROWS", "12"))
STOP_REQUEST_PATH = env_optional_path("PROMPT_ONLY_STOP_REQUEST_PATH")
WRITE_FULL_INPUTS = os.environ.get("PROMPT_ONLY_WRITE_FULL_INPUTS", "true").strip().lower() == "true"
INCLUDE_PROMPT_TEXT_IN_INPUTS = os.environ.get("PROMPT_ONLY_INCLUDE_PROMPT_TEXT_IN_INPUTS", "true").strip().lower() == "true"
INPUT_PREVIEW_ROWS = int(os.environ.get("PROMPT_ONLY_INPUT_PREVIEW_ROWS", str(CHECKPOINT_SAMPLE_ROWS)))


BAND_ORDER = [
    "b5_lower_visible_5_7",
    "b5_mid_visible_8_17",
    "b5_high_visible_18_plus",
]

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


@dataclass
class PromptRecord:
    input_index: int
    user_id: str
    density_band: str
    selected_rank_in_band: int
    user_meta: dict[str, Any]
    user_evidence_stream: list[dict[str, Any]]
    recent_event_sequence: list[dict[str, Any]]
    anchor_positive_events: list[dict[str, Any]]
    anchor_negative_events: list[dict[str, Any]]
    anchor_conflict_events: list[dict[str, Any]]
    context_notes: dict[str, Any]
    prompt_text: str
    selection_score: float


@dataclass
class GenerationRunResult:
    artifact_prefix: str
    raw_rows: list[dict[str, Any]]
    parsed_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    batch_metrics: list[dict[str, Any]]


@dataclass
class GenerationBackendState:
    tokenizer: Any
    model: Any
    infer_backend: str
    guided_decoding_mode: str
    tokenizer_load_seconds: float
    model_load_seconds: float


def parse_json_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except Exception:
        return []
    return value if isinstance(value, list) else []


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return normalize_value(value.tolist())
        except Exception:
            return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, str, bytes)) else False:
        return None
    return value


def load_merchant_state_map() -> dict[str, dict[str, Any]]:
    path = INPUT_MERCHANT_STATE_RUN_DIR / "merchant_state_v2.parquet"
    df = pd.read_parquet(path)
    keep_cols = [
        "business_id",
        "name",
        "city",
        "merchant_entity_scope_v2",
        "merchant_primary_cuisine_v2",
        "merchant_secondary_cuisine_v2",
        "merchant_scene_tags_v2",
        "merchant_meal_tags_v2",
        "merchant_property_tags_v2",
        "merchant_service_tags_v2",
        "merchant_complaint_tags_v2",
        "merchant_state_quality_band_v2",
        "merchant_state_text_v2",
    ]
    records = df[keep_cols].to_dict(orient="records")
    out: dict[str, dict[str, Any]] = {}
    for row in records:
        business_id = str(row["business_id"])
        out[business_id] = {k: normalize_value(v) for k, v in row.items()}
    return out


def enrich_event(event: dict[str, Any], merchant_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {k: normalize_value(v) for k, v in event.items()}
    business_id = str(out.get("business_id", "") or "")
    merchant_state = merchant_map.get(business_id)
    if merchant_state:
        out["merchant_state_v2"] = build_prompt_merchant_context_text(merchant_state)
        out["merchant_scope_v2"] = merchant_state.get("merchant_entity_scope_v2", "")
        out["merchant_primary_cuisine_v2"] = merchant_state.get("merchant_primary_cuisine_v2", "")
        out["merchant_secondary_cuisine_v2"] = merchant_state.get("merchant_secondary_cuisine_v2", "")
        out["merchant_quality_band_v2"] = merchant_state.get("merchant_state_quality_band_v2", "")
    else:
        out["merchant_state_v2"] = ""
    return out


def trim_event_for_prompt(event: dict[str, Any]) -> dict[str, Any]:
    if not STRIP_EVENT_METADATA:
        return event
    primary_cuisine = (
        normalize_value(event.get("merchant_primary_cuisine_v2"))
        or normalize_value(event.get("merchant_primary_cuisine"))
    )
    secondary_cuisine = (
        normalize_value(event.get("merchant_secondary_cuisine_v2"))
        or normalize_value(event.get("merchant_secondary_cuisine"))
    )
    trimmed = {
        "recent_rank": normalize_value(event.get("recent_rank")),
        "event_type": normalize_value(event.get("event_type")),
        "event_time": normalize_value(event.get("event_time")),
        "name": normalize_value(event.get("name")),
        "city": normalize_value(event.get("city")),
        "primary_cuisine": primary_cuisine,
        "secondary_cuisine": secondary_cuisine,
        "source_event_ref": normalize_value(event.get("source_event_ref")),
        "evidence_id": normalize_value(event.get("evidence_id")),
    }
    return {key: value for key, value in trimmed.items() if value not in (None, "", [])}


def sanitize_merchant_context_text(text: str) -> str:
    cleaned = str(text or "")
    if not SANITIZE_MERCHANT_CONTEXT or not cleaned:
        return cleaned
    for terms in MERCHANT_TERM_PATTERNS.values():
        for term in terms:
            cleaned = re.sub(rf"(?i)\b{re.escape(term)}\b", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:]){2,}", r"\1", cleaned)
    return cleaned.strip(" ,.;:")


def format_context_value_list(raw: Any) -> str:
    values = normalize_value(raw)
    if not isinstance(values, list):
        return ""
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return ", ".join(cleaned)


def build_prompt_merchant_context_text(merchant_state: dict[str, Any]) -> str:
    primary = str(merchant_state.get("merchant_primary_cuisine_v2", "") or "").strip()
    secondary = str(merchant_state.get("merchant_secondary_cuisine_v2", "") or "").strip()
    meals = format_context_value_list(merchant_state.get("merchant_meal_tags_v2"))
    service = format_context_value_list(merchant_state.get("merchant_service_tags_v2"))
    complaints = format_context_value_list(merchant_state.get("merchant_complaint_tags_v2"))
    quality = str(merchant_state.get("merchant_state_quality_band_v2", "") or "").strip()
    scope = str(merchant_state.get("merchant_entity_scope_v2", "") or "").strip()

    parts: list[str] = []
    if primary:
        cuisine = f"cuisine={primary}"
        if secondary:
            cuisine += f", secondary={secondary}"
        parts.append(cuisine)
    if meals:
        parts.append(f"meals={meals}")
    if service:
        parts.append(f"service={service}")
    if complaints:
        parts.append(f"complaints={complaints}")
    if scope:
        parts.append(f"scope={scope}")
    if quality:
        parts.append(f"quality={quality}")
    return sanitize_merchant_context_text(" | ".join(parts))


def assign_event_ids(events: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        row = dict(event)
        row["evidence_id"] = f"{prefix}_{idx}"
        out.append(row)
    return out


def select_diverse_events(
    events: list[dict[str, Any]],
    max_items: int,
    max_per_primary_cuisine: int,
) -> list[dict[str, Any]]:
    if max_items <= 0 or not events:
        return []

    deduped: list[dict[str, Any]] = []
    seen_business_ids: set[str] = set()
    for event in events:
        business_id = str(event.get("business_id", "") or "")
        if business_id and business_id in seen_business_ids:
            continue
        if business_id:
            seen_business_ids.add(business_id)
        deduped.append(event)

    if max_per_primary_cuisine <= 0:
        return deduped[:max_items]

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    cuisine_counts: Counter[str] = Counter()
    for event in deduped:
        if len(selected) >= max_items:
            break
        cuisine = str(event.get("merchant_primary_cuisine_v2") or event.get("merchant_primary_cuisine") or "").strip().lower()
        if cuisine and cuisine_counts[cuisine] >= max_per_primary_cuisine:
            continue
        business_id = str(event.get("business_id", "") or "")
        if business_id:
            selected_ids.add(business_id)
        if cuisine:
            cuisine_counts[cuisine] += 1
        selected.append(event)

    if len(selected) >= max_items:
        return selected[:max_items]

    for event in deduped:
        if len(selected) >= max_items:
            break
        business_id = str(event.get("business_id", "") or "")
        if business_id and business_id in selected_ids:
            continue
        if business_id:
            selected_ids.add(business_id)
        selected.append(event)
    return selected[:max_items]


def build_conflict_events(
    positive_events: list[dict[str, Any]],
    negative_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pos_cuisines = {
        str(ev.get("primary_cuisine") or ev.get("merchant_primary_cuisine_v2") or ev.get("merchant_primary_cuisine") or "")
        for ev in positive_events
    }
    neg_cuisines = {
        str(ev.get("primary_cuisine") or ev.get("merchant_primary_cuisine_v2") or ev.get("merchant_primary_cuisine") or "")
        for ev in negative_events
    }
    overlap = {c for c in pos_cuisines.intersection(neg_cuisines) if c}
    if not overlap:
        return []
    merged = []
    for ev in positive_events + negative_events:
        cuisine = str(ev.get("primary_cuisine") or ev.get("merchant_primary_cuisine_v2") or ev.get("merchant_primary_cuisine") or "")
        if cuisine in overlap:
            merged.append(ev)
    merged = merged[:MAX_CONFLICT_EVENTS]
    out: list[dict[str, Any]] = []
    for idx, ev in enumerate(merged, start=1):
        row = dict(ev)
        row["evidence_id"] = f"anchor_conflict_{idx}"
        source_ref = ev.get("evidence_id")
        if source_ref:
            row["source_event_ref"] = source_ref
        out.append(row)
    return out


def build_context_notes(
    row: dict[str, Any],
    recent_events: list[dict[str, Any]],
    positive_events: list[dict[str, Any]],
    negative_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if MINIMAL_CONTEXT_NOTES:
        return {
            "n_train_events_visible": int(row.get("n_train_max", 0) or 0),
            "sequence_event_count": int(row.get("sequence_event_count", 0) or 0),
            "review_evidence_count": int(row.get("review_evidence_count_v1", 0) or 0),
            "tip_evidence_count": int(row.get("tip_evidence_count_v1", 0) or 0),
            "positive_anchor_count": len(positive_events),
            "negative_anchor_count": len(negative_events),
        }
    city_counter = Counter(str(ev.get("city", "") or "") for ev in recent_events if ev.get("city"))
    event_counter = Counter(str(ev.get("event_type", "") or "") for ev in recent_events if ev.get("event_type"))
    cuisine_counter = Counter(
        str(ev.get("merchant_primary_cuisine_v2") or ev.get("merchant_primary_cuisine") or "")
        for ev in recent_events
        if ev.get("merchant_primary_cuisine_v2") or ev.get("merchant_primary_cuisine")
    )
    return {
        "n_train_events_visible": int(row.get("n_train_max", 0) or 0),
        "sequence_event_count": int(row.get("sequence_event_count", 0) or 0),
        "review_evidence_count": int(row.get("review_evidence_count_v1", 0) or 0),
        "tip_evidence_count": int(row.get("tip_evidence_count_v1", 0) or 0),
        "top_recent_cities": city_counter.most_common(3),
        "recent_event_type_counts": event_counter.most_common(),
        "recent_primary_cuisine_counts": cuisine_counter.most_common(5),
        "positive_anchor_count": len(positive_events),
        "negative_anchor_count": len(negative_events),
    }


def build_user_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": str(row["user_id"]),
        "density_band": str(row["density_band"]),
        "n_train_events_visible": int(row.get("n_train_max", 0) or 0),
        "sequence_event_count": int(row.get("sequence_event_count", 0) or 0),
        "review_evidence_count": int(row.get("review_evidence_count_v1", 0) or 0),
        "tip_evidence_count": int(row.get("tip_evidence_count_v1", 0) or 0),
    }


def maybe_limit_items(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if int(max_items) <= 0:
        return list(items)
    return list(items[: int(max_items)])


def build_user_evidence_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    review_items = parse_json_list(row.get("review_evidence_stream_json"))
    tip_items = parse_json_list(row.get("tip_evidence_stream_json"))
    review_items = [normalize_value(v) for v in maybe_limit_items(review_items, MAX_REVIEW_EVIDENCE)]
    tip_items = [normalize_value(v) for v in maybe_limit_items(tip_items, MAX_TIP_EVIDENCE)]
    evidence = []
    for item in review_items + tip_items:
        base_item = {
            "evidence_id": item.get("evidence_id", ""),
            "source": item.get("source", ""),
            "sentiment": item.get("sentiment", ""),
            "time_bucket": item.get("time_bucket", ""),
            "text": item.get("text", ""),
        }
        if not STRIP_REVIEW_METADATA:
            base_item.update(
                {
                    "weight": item.get("weight", None),
                    "tag": item.get("tag", ""),
                    "tags": item.get("tags", []),
                    "tag_types": item.get("tag_types", []),
                    "tag_count": item.get("tag_count", None),
                }
            )
        evidence.append(base_item)
    return evidence


def render_prompt(
    template_text: str,
    record: PromptRecord,
    extra_replacements: dict[str, str] | None = None,
) -> str:
    repl = {
        "{USER_META_JSON}": json.dumps(record.user_meta, ensure_ascii=False, indent=2),
        "{USER_EVIDENCE_STREAM_JSON}": json.dumps(record.user_evidence_stream, ensure_ascii=False, indent=2),
        "{RECENT_EVENT_SEQUENCE_JSON}": json.dumps(record.recent_event_sequence, ensure_ascii=False, indent=2),
        "{ANCHOR_POSITIVE_EVENTS_JSON}": json.dumps(record.anchor_positive_events, ensure_ascii=False, indent=2),
        "{ANCHOR_NEGATIVE_EVENTS_JSON}": json.dumps(record.anchor_negative_events, ensure_ascii=False, indent=2),
        "{ANCHOR_CONFLICT_EVENTS_JSON}": json.dumps(record.anchor_conflict_events, ensure_ascii=False, indent=2),
        "{CONTEXT_NOTES_JSON}": json.dumps(record.context_notes, ensure_ascii=False, indent=2),
    }
    if extra_replacements:
        repl.update(extra_replacements)
    out = template_text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def load_prompt_template_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    marker = "```text"
    if marker in raw:
        start = raw.find(marker)
        start = raw.find("\n", start)
        end = raw.find("```", start + 1)
        if start >= 0 and end > start:
            return raw[start + 1 : end].strip()
    return raw.strip()


def collect_allowed_evidence_ids(record: PromptRecord) -> set[str]:
    allowed: set[str] = set()
    for item in record.user_evidence_stream:
        evidence_id = str(item.get("evidence_id", "") or "")
        if evidence_id:
            allowed.add(evidence_id)
    for item in record.recent_event_sequence:
        evidence_id = str(item.get("evidence_id", "") or "")
        if evidence_id:
            allowed.add(evidence_id)
    for item in record.anchor_positive_events:
        evidence_id = str(item.get("evidence_id", "") or "")
        if evidence_id:
            allowed.add(evidence_id)
    for item in record.anchor_negative_events:
        evidence_id = str(item.get("evidence_id", "") or "")
        if evidence_id:
            allowed.add(evidence_id)
    for item in record.anchor_conflict_events:
        evidence_id = str(item.get("evidence_id", "") or "")
        if evidence_id:
            allowed.add(evidence_id)
    return allowed


def estimate_prompt_token_lengths(records: list[PromptRecord], tokenizer: Any) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for rec in records:
        encoded = tokenizer(
            rec.prompt_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        lengths[rec.input_index] = len(encoded["input_ids"])
    return lengths


def build_generation_batches(
    records: list[PromptRecord],
    prompt_token_lengths: dict[int, int],
) -> list[list[PromptRecord]]:
    ordered = list(records)
    if SORT_BY_PROMPT_LENGTH:
        ordered.sort(key=lambda rec: prompt_token_lengths.get(rec.input_index, 0))

    batches: list[list[PromptRecord]] = []
    current: list[PromptRecord] = []
    current_max_len = 0
    for rec in ordered:
        rec_len = prompt_token_lengths.get(rec.input_index, 0)
        next_max_len = max(current_max_len, rec_len)
        next_batch_size = len(current) + 1
        token_budget_ok = (
            MAX_BATCH_PROMPT_TOKENS <= 0
            or next_max_len * next_batch_size <= MAX_BATCH_PROMPT_TOKENS
        )
        if current and (
            next_batch_size > GEN_BATCH_SIZE
            or not token_budget_ok
        ):
            batches.append(current)
            current = [rec]
            current_max_len = rec_len
        else:
            current.append(rec)
            current_max_len = next_max_len
    if current:
        batches.append(current)
    return batches


def score_row(row: dict[str, Any]) -> float:
    review_count = float(row.get("review_evidence_count_v1", 0) or 0)
    tip_count = float(row.get("tip_evidence_count_v1", 0) or 0)
    seq_count = float(row.get("sequence_event_count", 0) or 0)
    pos_count = len(parse_json_list(row.get("positive_anchor_sequence_json")))
    neg_count = len(parse_json_list(row.get("negative_anchor_sequence_json")))
    both_bonus = 2.0 if pos_count > 0 and neg_count > 0 else 0.0
    tip_bonus = 1.0 if tip_count > 0 else 0.0
    return review_count + 0.75 * tip_count + 0.5 * seq_count + both_bonus + tip_bonus


def sort_band_df_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    band_df = df.copy()
    if band_df.empty:
        return band_df
    band_df["selection_score"] = band_df.apply(lambda r: score_row(r.to_dict()), axis=1)
    return band_df.sort_values(
        by=["selection_score", "tip_evidence_count_v1", "review_evidence_count_v1", "sequence_event_count"],
        ascending=False,
    )


def select_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_users: set[str] = set()
    for band in BAND_ORDER:
        band_df = sort_band_df_for_selection(df[df["density_band"] == band])
        if band_df.empty:
            continue
        rank = 0
        for _, row in band_df.iterrows():
            user_id = str(row["user_id"])
            if user_id in seen_users:
                continue
            selected.append({**row.to_dict(), "selected_rank_in_band": rank + 1})
            seen_users.add(user_id)
            rank += 1
            if SELECT_MODE != "all" and rank >= SAMPLE_PER_BAND:
                break
            if MAX_TOTAL_ROWS > 0 and len(selected) >= MAX_TOTAL_ROWS:
                return selected
    return selected


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""


def maybe_disable_transformers_grouped_mm() -> None:
    if not DISABLE_GROUPED_MM:
        return
    try:
        import transformers.integrations.moe as moe_integration
        import torch
    except Exception as exc:
        print(f"[WARN] failed to import transformers.integrations.moe for grouped_mm patch: {exc}")
        return

    def _always_false(*args: Any, **kwargs: Any) -> bool:
        return False

    def _grouped_mm_fallback_only(input_tensor: Any, weight_tensor: Any, offs: Any) -> Any:
        return torch.ops.transformers.grouped_mm_fallback(input_tensor, weight_tensor, offs=offs)

    moe_integration._can_use_grouped_mm = _always_false
    moe_integration._grouped_mm = _grouped_mm_fallback_only
    print("[INFO] PROMPT_ONLY_DISABLE_GROUPED_MM=true; forcing transformers MoE grouped_mm fallback path.")


def collect_evidence_refs(node: Any) -> list[str]:
    return collect_evidence_refs_from_schema(node)


def collect_ref_groups(node: Any) -> list[list[str]]:
    return collect_ref_groups_from_schema(node)


def collect_unknown_fields(parsed: Any) -> list[str]:
    return collect_unknown_fields_from_schema(parsed)


def collect_referenced_evidence_ids(parsed: Any) -> set[str]:
    return {ref for ref in collect_evidence_refs(parsed) if ref}


def select_items_for_compact_bundle(
    items: list[dict[str, Any]],
    referenced_ids: set[str],
    fallback_limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    selected = [item for item in items if str(item.get("evidence_id", "") or "") in referenced_ids]
    if selected:
        return selected, False
    if fallback_limit <= 0:
        return [], False
    return items[:fallback_limit], bool(items[:fallback_limit])


def build_pass2_compact_payload(
    record: PromptRecord,
    pass1_row: dict[str, Any] | None,
) -> dict[str, Any]:
    draft = pass1_row.get("parsed_json") if isinstance(pass1_row, dict) else None
    referenced_ids = collect_referenced_evidence_ids(draft)
    full_allowed_ids = sorted(collect_allowed_evidence_ids(record))
    draft_invalid_ids = sorted({ref for ref in referenced_ids if ref not in set(full_allowed_ids)})
    user_evidence, used_user_fallback = select_items_for_compact_bundle(
        record.user_evidence_stream,
        referenced_ids,
        PASS2_COMPACT_FALLBACK_USER_EVIDENCE,
    )
    recent_events, used_recent_fallback = select_items_for_compact_bundle(
        record.recent_event_sequence,
        referenced_ids,
        PASS2_COMPACT_FALLBACK_RECENT_EVENTS,
    )
    pos_anchors, used_pos_fallback = select_items_for_compact_bundle(
        record.anchor_positive_events,
        referenced_ids,
        PASS2_COMPACT_FALLBACK_POS_ANCHORS,
    )
    neg_anchors, used_neg_fallback = select_items_for_compact_bundle(
        record.anchor_negative_events,
        referenced_ids,
        PASS2_COMPACT_FALLBACK_NEG_ANCHORS,
    )
    conflict_anchors, used_conflict_fallback = select_items_for_compact_bundle(
        record.anchor_conflict_events,
        referenced_ids,
        PASS2_COMPACT_FALLBACK_CONFLICT_ANCHORS,
    )
    return {
        "pass1_json_valid": bool(draft is not None),
        "draft": draft,
        "draft_unknown_fields": collect_unknown_fields(draft),
        "draft_referenced_evidence_ids": sorted(referenced_ids),
        "full_allowed_evidence_ids": full_allowed_ids,
        "draft_invalid_evidence_ids": draft_invalid_ids,
        "compact_evidence_bundle": {
            "user_evidence_stream": user_evidence,
            "recent_event_sequence": recent_events,
            "anchor_positive_events": pos_anchors,
            "anchor_negative_events": neg_anchors,
            "anchor_conflict_events": conflict_anchors,
            "context_notes": record.context_notes,
        },
        "fallback_usage": {
            "user_evidence_stream": used_user_fallback,
            "recent_event_sequence": used_recent_fallback,
            "anchor_positive_events": used_pos_fallback,
            "anchor_negative_events": used_neg_fallback,
            "anchor_conflict_events": used_conflict_fallback,
        },
    }


def read_json_payload(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    payload = read_json_payload(path)
    return payload if isinstance(payload, list) else []


def load_generation_rows(output_dir: Path, artifact_prefix: str, row_kind: str) -> list[dict[str, Any]]:
    for suffix in [".jsonl", ".json"]:
        path = output_dir / f"{artifact_prefix}_{row_kind}{suffix}"
        rows = read_json_rows(path)
        if rows:
            return rows
    return []


def dedupe_rows_by_user_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_user: dict[str, dict[str, Any]] = {}
    for row in rows:
        user_id = str(row.get("user_id", "") or "")
        if not user_id:
            continue
        by_user[user_id] = row
    return list(by_user.values())


def remap_loaded_rows_to_records(
    rows: list[dict[str, Any]],
    records_by_user_id: dict[str, PromptRecord],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for row in dedupe_rows_by_user_id(rows):
        user_id = str(row.get("user_id", "") or "")
        record = records_by_user_id.get(user_id)
        if record is None:
            continue
        mapped = dict(row)
        mapped["user_id"] = record.user_id
        mapped["density_band"] = record.density_band
        mapped["selected_rank_in_band"] = record.selected_rank_in_band
        mapped["input_index"] = record.input_index
        mapped["selection_score"] = record.selection_score
        remapped.append(mapped)
    remapped.sort(key=lambda row: int(row["input_index"]))
    return remapped


def load_generation_result_from_dir(
    output_dir: Path,
    artifact_prefix: str,
    records: list[PromptRecord],
) -> GenerationRunResult | None:
    records_by_user_id = {record.user_id: record for record in records}
    raw_rows = remap_loaded_rows_to_records(
        load_generation_rows(output_dir, artifact_prefix, "raw"),
        records_by_user_id,
    )
    parsed_rows = remap_loaded_rows_to_records(
        load_generation_rows(output_dir, artifact_prefix, "parsed"),
        records_by_user_id,
    )
    if not raw_rows and not parsed_rows:
        return None
    summary_payload = read_json_payload(output_dir / f"{artifact_prefix}_summary.json")
    batch_metrics_payload = read_json_payload(output_dir / f"{artifact_prefix}_batch_metrics.json")
    return GenerationRunResult(
        artifact_prefix=artifact_prefix,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        summary=summary_payload if isinstance(summary_payload, dict) else {},
        batch_metrics=batch_metrics_payload if isinstance(batch_metrics_payload, list) else [],
    )


def collect_completed_user_ids_from_run_dir(
    run_dir: Path,
    artifact_prefix: str = "prompt_only_pass1_generation",
) -> set[str]:
    rows = load_generation_rows(run_dir, artifact_prefix, "raw")
    return {
        str(row.get("user_id", "") or "")
        for row in rows
        if str(row.get("user_id", "") or "")
    }


def filter_selected_rows_by_user_ids(
    rows: list[dict[str, Any]],
    excluded_user_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    if not excluded_user_ids:
        return rows, 0
    kept = [row for row in rows if str(row.get("user_id", "") or "") not in excluded_user_ids]
    return kept, len(rows) - len(kept)


def append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def normalize_match_text(text: Any) -> str:
    return normalize_match_text_from_schema(text)


def get_section_items(parsed: Any, *path: str) -> list[dict[str, Any]]:
    return get_section_items_from_schema(parsed, *path)


def collect_text_values(node: Any) -> list[str]:
    values: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            values.extend(collect_text_values(value))
    elif isinstance(node, list):
        for value in node:
            values.extend(collect_text_values(value))
    elif isinstance(node, str):
        text = node.strip()
        if text:
            values.append(text)
    return values


def collect_claim_texts(items: list[dict[str, Any]], field: str = "claim") -> list[str]:
    out: list[str] = []
    for item in items:
        value = str(item.get(field, "") or "").strip()
        if value:
            out.append(value)
    return out


def normalize_ref_list(value: Any) -> list[str]:
    return normalize_ref_list_from_schema(value)


def item_ref_list(item: dict[str, Any]) -> list[str]:
    return item_ref_list_from_schema(item)


def direct_review_ref_list(item: dict[str, Any]) -> list[str]:
    return direct_review_ref_list_from_schema(item)


def contextual_ref_list(item: dict[str, Any]) -> list[str]:
    return contextual_ref_list_from_schema(item)


def has_direct_review_ref(refs: list[str]) -> bool:
    return has_direct_review_ref_from_schema(refs)


def iter_section_items(parsed: Any) -> list[tuple[str, int, dict[str, Any]]]:
    section_specs = [
        ("stable_preferences", get_section_items(parsed, "grounded_facts", "stable_preferences")),
        ("avoid_signals", get_section_items(parsed, "grounded_facts", "avoid_signals")),
        ("recent_signals", get_section_items(parsed, "grounded_facts", "recent_signals")),
        ("context_rules", get_section_items(parsed, "grounded_facts", "context_rules")),
        ("state_hypotheses", get_section_items(parsed, "state_hypotheses")),
        ("discriminative_signals", get_section_items(parsed, "discriminative_signals")),
        ("contextual_inference_signals", get_section_items(parsed, "contextual_inference_signals")),
    ]
    out: list[tuple[str, int, dict[str, Any]]] = []
    for section, items in section_specs:
        for idx, item in enumerate(items, start=1):
            out.append((section, idx, item))
    return out


def collect_support_issues(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    issues: list[dict[str, Any]] = []
    for section, item_index, item in iter_section_items(parsed):
        refs = item_ref_list(item)
        direct_review_refs = direct_review_ref_list(item)
        contextual_refs = contextual_ref_list(item)
        support_basis = str(item.get("support_basis", "") or "").strip()
        support_note = str(item.get("support_note", "") or "").strip()
        has_direct_ref = has_direct_review_ref(direct_review_refs)
        context_only = bool(refs) and not has_direct_ref
        allowed_support_basis = (
            CONTEXTUAL_ALLOWED_SUPPORT_BASIS
            if section == "contextual_inference_signals"
            else STANDARD_ALLOWED_SUPPORT_BASIS
        )
        ref_limit = SECTION_REF_LIMITS.get(section, 4)

        if support_basis not in allowed_support_basis:
            issues.append(
                {
                    "section": section,
                    "item_index": item_index,
                    "issue": "invalid_support_basis",
                    "support_basis": support_basis,
                }
            )
        if len(refs) > ref_limit:
            issues.append(
                {
                    "section": section,
                    "item_index": item_index,
                    "issue": "too_many_refs_for_section",
                    "ref_count": len(refs),
                    "ref_limit": ref_limit,
                }
            )
        if not support_note:
            issues.append(
                {
                    "section": section,
                    "item_index": item_index,
                    "issue": "missing_support_note",
                    "support_basis": support_basis,
                }
            )
        if section in REVIEW_REQUIRED_SECTIONS and not has_direct_ref:
            issues.append(
                {
                    "section": section,
                    "item_index": item_index,
                    "issue": "missing_direct_review_ref",
                    "support_basis": support_basis,
                }
            )
        if section in REVIEW_REQUIRED_SECTIONS and support_basis == "event_context_only":
            issues.append(
                {
                    "section": section,
                    "item_index": item_index,
                    "issue": "event_context_shortcut",
                }
            )
        if section == "contextual_inference_signals":
            if not contextual_refs:
                issues.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "issue": "missing_contextual_refs",
                    }
                )
            if len(direct_review_refs) > 2:
                issues.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "issue": "too_many_direct_review_refs",
                        "ref_count": len(direct_review_refs),
                    }
                )
            if len(contextual_refs) > 2:
                issues.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "issue": "too_many_contextual_refs",
                        "ref_count": len(contextual_refs),
                    }
                )
        if context_only and support_basis != "event_context_only" and section != "contextual_inference_signals":
            issues.append(
                {
                    "section": section,
                    "item_index": item_index,
                    "issue": "event_support_basis_mismatch",
                    "support_basis": support_basis,
                }
            )
    return issues


def collect_promoted_merchant_guardrail_issues(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    promoted_sections = [
        ("stable_preferences", get_section_items(parsed, "grounded_facts", "stable_preferences"), False),
        ("avoid_signals", get_section_items(parsed, "grounded_facts", "avoid_signals"), False),
        ("state_hypotheses", get_section_items(parsed, "state_hypotheses"), False),
        ("discriminative_signals", get_section_items(parsed, "discriminative_signals"), True),
    ]
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for section, items, include_why_not_generic in promoted_sections:
        for item_index, item in enumerate(items, start=1):
            texts = [str(item.get("claim", "") or "").strip()]
            if include_why_not_generic:
                texts.append(str(item.get("why_not_generic", "") or "").strip())
            hits = find_term_hits(texts, PROMOTED_MERCHANT_TERM_KEYS)
            if not hits:
                continue
            support_basis = str(item.get("support_basis", "") or "").strip()
            has_direct_ref = has_direct_review_ref(item_ref_list(item))
            for term in hits:
                if (
                    term == "outdoor seating"
                    and support_basis in {"direct_user_text", "review_pattern_inference", "mixed_support"}
                    and has_direct_ref
                ):
                    continue
                key = (section, item_index, term)
                if key in seen:
                    continue
                seen.add(key)
                failures.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "issue": "merchant_term_leak",
                        "term": term,
                    }
                )
    return failures


def is_allowed_promoted_merchant_term(term: str, item: dict[str, Any]) -> bool:
    support_basis = str(item.get("support_basis", "") or "").strip()
    has_direct_ref = has_direct_review_ref(item_ref_list(item))
    return (
        term == "outdoor seating"
        and support_basis in {"direct_user_text", "review_pattern_inference", "mixed_support"}
        and has_direct_ref
    )


def repair_promoted_merchant_items(parsed: Any) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(parsed, dict):
        return parsed, []
    repaired = copy.deepcopy(parsed)
    actions: list[dict[str, Any]] = []
    promoted_sections = [
        ("stable_preferences", get_section_list_node_from_schema(repaired, "grounded_facts", "stable_preferences"), False),
        ("avoid_signals", get_section_list_node_from_schema(repaired, "grounded_facts", "avoid_signals"), False),
        ("state_hypotheses", get_section_list_node_from_schema(repaired, "state_hypotheses"), False),
        ("discriminative_signals", get_section_list_node_from_schema(repaired, "discriminative_signals"), True),
    ]
    for section, node, include_why_not_generic in promoted_sections:
        items = [item for item in (node or []) if isinstance(item, dict)]
        kept_items: list[dict[str, Any]] = []
        for item_index, item in enumerate(items, start=1):
            texts = [str(item.get("claim", "") or "").strip()]
            if include_why_not_generic:
                texts.append(str(item.get("why_not_generic", "") or "").strip())
            hits = find_term_hits(texts, PROMOTED_MERCHANT_TERM_KEYS)
            disallowed_terms = [term for term in hits if not is_allowed_promoted_merchant_term(term, item)]
            if disallowed_terms:
                actions.append(
                    {
                        "section": section,
                        "item_index": item_index,
                        "action": "drop_promoted_merchant_item",
                        "terms": sorted(set(disallowed_terms)),
                    }
                )
                continue
            kept_items.append(item)
        if node is not None:
            node[:] = kept_items
    return repaired, actions


def collect_guardrail_failures(
    parsed: Any,
    *,
    schema_issues: list[dict[str, Any]],
    support_issues: list[dict[str, Any]],
    invalid_refs: list[str],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for issue in schema_issues:
        failures.append({"issue": "schema_violation", **issue})
    if invalid_refs:
        failures.append({"issue": "invalid_evidence_refs", "refs": invalid_refs})
    failures.extend(support_issues)
    failures.extend(collect_cross_section_guardrail_issues(parsed))
    failures.extend(collect_promoted_merchant_guardrail_issues(parsed))
    return failures


def find_term_hits(texts: list[str], allowed_keys: tuple[str, ...] | None = None) -> list[str]:
    normalized_texts = [normalize_match_text(text) for text in texts if str(text or "").strip()]
    hits: list[str] = []
    allowed = set(allowed_keys) if allowed_keys else None
    for label, variants in MERCHANT_TERM_PATTERNS.items():
        if allowed is not None and label not in allowed:
            continue
        normalized_variants = [normalize_match_text(variant) for variant in variants]
        if any(variant and variant in text for text in normalized_texts for variant in normalized_variants):
            hits.append(label)
    return sorted(set(hits))


def extend_cuisine_terms_from_value(terms: set[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            extend_cuisine_terms_from_value(terms, item)
        return
    raw = normalize_match_text(value)
    if not raw:
        return
    terms.add(raw)
    split_text = raw
    for sep in ["/", ",", "&", "|", ";"]:
        split_text = split_text.replace(sep, "|")
    for piece in split_text.split("|"):
        piece = piece.strip()
        if len(piece) >= 3:
            terms.add(piece)


def collect_record_cuisine_terms(record: PromptRecord) -> set[str]:
    terms: set[str] = set()
    event_blocks = [
        record.recent_event_sequence,
        record.anchor_positive_events,
        record.anchor_negative_events,
        record.anchor_conflict_events,
    ]
    for block in event_blocks:
        for item in block:
            extend_cuisine_terms_from_value(terms, item.get("primary_cuisine"))
            extend_cuisine_terms_from_value(terms, item.get("secondary_cuisine"))
            extend_cuisine_terms_from_value(terms, item.get("merchant_primary_cuisine_v2"))
            extend_cuisine_terms_from_value(terms, item.get("merchant_secondary_cuisine_v2"))
            extend_cuisine_terms_from_value(terms, item.get("merchant_primary_cuisine"))
            extend_cuisine_terms_from_value(terms, item.get("merchant_secondary_cuisine"))
    return {term for term in terms if term}


def has_supported_cuisine_claim(parsed: Any, record: PromptRecord | None) -> bool:
    if not isinstance(parsed, dict) or record is None:
        return False
    cuisine_terms = collect_record_cuisine_terms(record)
    if not cuisine_terms:
        return False
    claim_texts = (
        collect_claim_texts(get_section_items(parsed, "grounded_facts", "stable_preferences"))
        + collect_claim_texts(get_section_items(parsed, "discriminative_signals"))
    )
    normalized_claims = [normalize_match_text(text) for text in claim_texts]
    return any(
        term in claim
        for claim in normalized_claims
        for term in cuisine_terms
        if len(term) >= 3
    )


def collect_invalid_unknown_fields(parsed: Any) -> list[str]:
    invalid_fields: list[str] = []
    if not isinstance(parsed, dict):
        return invalid_fields
    for field in collect_unknown_fields(parsed):
        if field not in ALLOWED_UNKNOWN_FIELDS:
            invalid_fields.append(field)
    return sorted(set(invalid_fields))


def collect_duplicate_sections(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    section_specs = {
        "stable_preferences": collect_claim_texts(get_section_items(parsed, "grounded_facts", "stable_preferences")),
        "avoid_signals": collect_claim_texts(get_section_items(parsed, "grounded_facts", "avoid_signals")),
        "recent_signals": collect_claim_texts(get_section_items(parsed, "grounded_facts", "recent_signals")),
        "context_rules": collect_claim_texts(get_section_items(parsed, "grounded_facts", "context_rules")),
        "state_hypotheses": collect_claim_texts(get_section_items(parsed, "state_hypotheses")),
        "discriminative_signals": collect_claim_texts(get_section_items(parsed, "discriminative_signals")),
        "contextual_inference_signals": collect_claim_texts(get_section_items(parsed, "contextual_inference_signals")),
    }
    duplicates: list[dict[str, Any]] = []
    for section, texts in section_specs.items():
        normalized = [normalize_match_text(text) for text in texts if text]
        duplicate_texts = [text for text, count in Counter(normalized).items() if text and count > 1]
        if duplicate_texts:
            duplicates.append({"section": section, "duplicate_claims": duplicate_texts[:3]})
    unknown_fields = [normalize_match_text(field) for field in collect_unknown_fields(parsed)]
    duplicate_unknowns = [field for field, count in Counter(unknown_fields).items() if field and count > 1]
    if duplicate_unknowns:
        duplicates.append({"section": "unknowns", "duplicate_claims": duplicate_unknowns[:3]})
    return duplicates


def build_row_preview(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    return {
        "stable_preferences": collect_claim_texts(get_section_items(parsed, "grounded_facts", "stable_preferences"))[:3],
        "avoid_signals": collect_claim_texts(get_section_items(parsed, "grounded_facts", "avoid_signals"))[:3],
        "recent_signals": collect_claim_texts(get_section_items(parsed, "grounded_facts", "recent_signals"))[:2],
        "context_rules": collect_claim_texts(get_section_items(parsed, "grounded_facts", "context_rules"))[:2],
        "state_hypotheses": collect_claim_texts(get_section_items(parsed, "state_hypotheses"))[:3],
        "discriminative_signals": collect_claim_texts(get_section_items(parsed, "discriminative_signals"))[:2],
        "contextual_inference_signals": collect_claim_texts(get_section_items(parsed, "contextual_inference_signals"))[:3],
        "unknown_fields": collect_unknown_fields(parsed),
    }


def build_semantic_audit(
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    records_by_index: dict[int, PromptRecord],
    *,
    sample_limit: int,
) -> dict[str, Any]:
    raw_by_index = {int(row["input_index"]): row for row in raw_rows}
    merchant_term_counts_anywhere: Counter[str] = Counter()
    promoted_rows: list[dict[str, Any]] = []
    invalid_unknown_rows: list[dict[str, Any]] = []
    cuisine_unknown_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    support_issue_rows: list[dict[str, Any]] = []
    contextual_inference_rows: list[dict[str, Any]] = []
    flagged_examples: list[dict[str, Any]] = []
    clean_examples: list[dict[str, Any]] = []

    for parsed_row in parsed_rows:
        input_index = int(parsed_row["input_index"])
        user_id = str(parsed_row["user_id"])
        raw_row = raw_by_index.get(input_index, {})
        parsed = parsed_row.get("parsed_json") or parsed_row.get("parsed_json_candidate")
        if not isinstance(parsed, dict):
            continue

        row_flags: list[str] = []
        all_texts = collect_text_values(parsed)
        merchant_hits_anywhere = find_term_hits(all_texts)
        merchant_term_counts_anywhere.update(set(merchant_hits_anywhere))

        promoted_hits: list[str] = []
        promoted_sections = [
            (get_section_items(parsed, "grounded_facts", "stable_preferences"), False),
            (get_section_items(parsed, "grounded_facts", "avoid_signals"), False),
            (get_section_items(parsed, "state_hypotheses"), False),
            (get_section_items(parsed, "discriminative_signals"), True),
        ]
        for section_items, include_why_not_generic in promoted_sections:
            for item in section_items:
                texts = [str(item.get("claim", "") or "").strip()]
                if include_why_not_generic:
                    texts.append(str(item.get("why_not_generic", "") or "").strip())
                for term in find_term_hits(texts, PROMOTED_MERCHANT_TERM_KEYS):
                    if is_allowed_promoted_merchant_term(term, item):
                        continue
                    promoted_hits.append(term)
        if promoted_hits:
            promoted_rows.append({"user_id": user_id, "terms": promoted_hits})
            row_flags.append("merchant_term_leak")

        contextual_items = get_section_items(parsed, "contextual_inference_signals")
        if contextual_items:
            contextual_inference_rows.append(
                {
                    "user_id": user_id,
                    "items": [
                        {
                            "canonical_axis": str(item.get("canonical_axis", "") or "").strip(),
                            "claim": str(item.get("claim", "") or "").strip(),
                            "support_basis": str(item.get("support_basis", "") or "").strip(),
                            "direct_review_refs": direct_review_ref_list(item),
                            "contextual_refs": contextual_ref_list(item),
                        }
                        for item in contextual_items
                    ],
                }
            )

        invalid_unknown_fields = collect_invalid_unknown_fields(parsed)
        if invalid_unknown_fields:
            invalid_unknown_rows.append({"user_id": user_id, "invalid_fields": invalid_unknown_fields})
            row_flags.append("invalid_unknown_field")

        if "cuisine_preference" in collect_unknown_fields(parsed) and has_supported_cuisine_claim(parsed, records_by_index.get(input_index)):
            cuisine_unknown_rows.append({"user_id": user_id})
            row_flags.append("cuisine_unknown_with_supported_cuisine")

        duplicate_sections = collect_duplicate_sections(parsed)
        if duplicate_sections:
            duplicate_rows.append({"user_id": user_id, "duplicate_sections": duplicate_sections})
            row_flags.append("duplicate_within_section")

        support_issues = parsed_row.get("support_issues", [])
        if support_issues:
            support_issue_rows.append({"user_id": user_id, "support_issues": support_issues})
            row_flags.append("support_grounding_issue")

        if any(size > 4 for size in raw_row.get("ref_group_sizes", [])):
            row_flags.append("wide_ref_group")

        preview = {
            "user_id": user_id,
            "density_band": parsed_row.get("density_band"),
            "flags": sorted(set(row_flags)),
            "preview": build_row_preview(parsed),
        }
        if row_flags and len(flagged_examples) < sample_limit:
            flagged_examples.append(preview)
        elif not row_flags and len(clean_examples) < max(1, sample_limit // 3):
            clean_examples.append(preview)

    return {
        "rows_with_promoted_merchant_terms": len(promoted_rows),
        "merchant_term_counts_anywhere": dict(merchant_term_counts_anywhere),
        "rows_with_invalid_unknown_fields": len(invalid_unknown_rows),
        "rows_with_cuisine_unknown_supported_cuisine": len(cuisine_unknown_rows),
        "rows_with_duplicate_within_section": len(duplicate_rows),
        "rows_with_support_grounding_issues": len(support_issue_rows),
        "rows_with_contextual_inference_signals": len(contextual_inference_rows),
        "promoted_merchant_term_examples": promoted_rows[:sample_limit],
        "invalid_unknown_examples": invalid_unknown_rows[:sample_limit],
        "cuisine_unknown_with_supported_cuisine_examples": cuisine_unknown_rows[:sample_limit],
        "duplicate_within_section_examples": duplicate_rows[:sample_limit],
        "support_grounding_examples": support_issue_rows[:sample_limit],
        "contextual_inference_examples": contextual_inference_rows[:sample_limit],
        "flagged_examples": flagged_examples,
        "clean_examples": clean_examples,
    }


def build_generation_summary(
    *,
    artifact_prefix: str,
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    batch_metrics: list[dict[str, Any]],
    planned_batch_sizes: list[int],
    max_new_tokens: int,
    tokenizer_load_seconds: float,
    model_load_seconds: float,
    prompt_length_seconds: float,
    generation_seconds: float,
    records_by_index: dict[int, PromptRecord],
    total_rows_planned: int,
    stop_requested: bool,
) -> dict[str, Any]:
    invalid_ref_rows = sum(1 for row in raw_rows if row.get("invalid_evidence_refs"))
    rows_with_schema_issues = sum(1 for row in raw_rows if row.get("schema_issues"))
    rows_with_unknowns = sum(1 for row in raw_rows if row.get("unknown_fields"))
    rows_with_support_issues = sum(1 for row in raw_rows if row.get("support_issues"))
    rows_with_guardrail_failures = sum(1 for row in raw_rows if row.get("guardrail_failures"))
    rows_passing_guardrails = sum(1 for row in raw_rows if row.get("guardrails_passed"))
    unknown_field_counts = Counter(
        field
        for row in raw_rows
        for field in row.get("unknown_fields", [])
    )
    schema_issue_counter = Counter(
        issue.get("issue", "")
        for row in raw_rows
        for issue in row.get("schema_issues", [])
        if isinstance(issue, dict) and issue.get("issue")
    )
    support_issue_counts = Counter(
        issue.get("issue", "")
        for row in raw_rows
        for issue in row.get("support_issues", [])
        if isinstance(issue, dict) and issue.get("issue")
    )
    repair_action_counts = Counter(
        action.get("action", "")
        for row in raw_rows
        for action in row.get("repair_actions", [])
        if isinstance(action, dict) and action.get("action")
    )
    rows_with_repairs = sum(1 for row in raw_rows if row.get("repair_actions"))
    rows_with_wide_ref_groups = sum(
        1
        for row in raw_rows
        if any(size > 4 for size in row.get("ref_group_sizes", []))
    )
    executed_batch_sizes = [
        int(metric["batch_size"])
        for metric in batch_metrics
        if metric.get("status") == "completed"
    ]
    avg_realized_batch_size = round(
        sum(executed_batch_sizes) / max(len(executed_batch_sizes), 1),
        4,
    )
    rows_completed = len(raw_rows)
    rows_per_second = round(rows_completed / generation_seconds, 6) if generation_seconds > 0 else 0.0
    remaining_rows = max(total_rows_planned - rows_completed, 0)
    est_remaining_seconds = round(remaining_rows / rows_per_second, 4) if rows_per_second > 0 else None
    semantic_audit = build_semantic_audit(
        raw_rows,
        parsed_rows,
        records_by_index,
        sample_limit=CHECKPOINT_SAMPLE_ROWS,
    )
    return {
        "artifact_prefix": artifact_prefix,
        "rows": rows_completed,
        "json_valid_rate": round(
            sum(1 for row in raw_rows if row["json_valid"]) / max(rows_completed, 1),
            4,
        ),
        "structured_output_valid_rate": round(
            sum(1 for row in raw_rows if row.get("schema_valid")) / max(rows_completed, 1),
            4,
        ),
        "guardrails_pass_rate": round(rows_passing_guardrails / max(rows_completed, 1), 4),
        "atomic_ref_valid_rate": round(
            sum(1 for row in raw_rows if row["json_valid"] and not row.get("invalid_evidence_refs")) / max(rows_completed, 1),
            4,
        ),
        "rows_with_invalid_refs": invalid_ref_rows,
        "rows_with_schema_issues": rows_with_schema_issues,
        "schema_issue_counts": dict(schema_issue_counter),
        "rows_with_unknowns": rows_with_unknowns,
        "unknown_field_counts": dict(unknown_field_counts),
        "rows_with_support_issues": rows_with_support_issues,
        "support_issue_counts": dict(support_issue_counts),
        "rows_with_repairs": rows_with_repairs,
        "repair_action_counts": dict(repair_action_counts),
        "rows_with_guardrail_failures": rows_with_guardrail_failures,
        "rows_with_wide_ref_groups": rows_with_wide_ref_groups,
        "bands": Counter(row["density_band"] for row in raw_rows),
        "model": BASE_MODEL,
        "infer_backend": INFER_BACKEND,
        "guided_decoding_modes": sorted(
            {
                str(metric.get("guided_decoding_mode", "") or "")
                for metric in batch_metrics
                if metric.get("guided_decoding_mode")
            }
        ),
        "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
        "guardrails_hard_fail": GUARDRAILS_HARD_FAIL,
        "gen_batch_size": GEN_BATCH_SIZE,
        "gen_max_new_tokens": max_new_tokens,
        "sort_by_prompt_length": SORT_BY_PROMPT_LENGTH,
        "max_batch_prompt_tokens": MAX_BATCH_PROMPT_TOKENS,
        "planned_batch_count": len(planned_batch_sizes),
        "planned_batch_size_hist": dict(Counter(planned_batch_sizes)),
        "realized_batch_count": len(executed_batch_sizes),
        "realized_batch_size_hist": dict(Counter(executed_batch_sizes)),
        "avg_realized_batch_size": avg_realized_batch_size,
        "phase_seconds": {
            "tokenizer_load": tokenizer_load_seconds,
            "model_load": model_load_seconds,
            "prompt_length_estimate": prompt_length_seconds,
            "generation_total": generation_seconds,
        },
        "progress": {
            "rows_completed": rows_completed,
            "rows_total_planned": total_rows_planned,
            "completion_rate": round(rows_completed / max(total_rows_planned, 1), 6),
            "rows_per_second": rows_per_second,
            "remaining_rows_estimate": remaining_rows,
            "eta_seconds_estimate": est_remaining_seconds,
        },
        "stopped_early": stop_requested,
        "semantic_audit": semantic_audit,
    }


def write_checkpoint_snapshot(
    output_dir: Path,
    artifact_prefix: str,
    summary: dict[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    checkpoint_root = output_dir / "checkpoints" / artifact_prefix
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_root / checkpoint_label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    (checkpoint_dir / "summary.json").write_text(payload, encoding="utf-8")
    (checkpoint_root / "latest_summary.json").write_text(payload, encoding="utf-8")


def write_prompt_record_dump(output_dir: Path, file_stem: str, records: list[PromptRecord]) -> None:
    if WRITE_FULL_INPUTS:
        serialized = serialize_prompt_records(records, include_prompt_text=INCLUDE_PROMPT_TEXT_IN_INPUTS)
        (output_dir / f"{file_stem}.json").write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return
    preview_count = min(INPUT_PREVIEW_ROWS, len(records))
    if preview_count <= 0:
        return
    preview = serialize_prompt_records(
        records[:preview_count],
        include_prompt_text=INCLUDE_PROMPT_TEXT_IN_INPUTS,
    )
    (output_dir / f"{file_stem}_preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_generation_artifacts(
    output_dir: Path,
    artifact_prefix: str,
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    batch_metrics: list[dict[str, Any]],
) -> None:
    (output_dir / f"{artifact_prefix}_raw.json").write_text(
        json.dumps(raw_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{artifact_prefix}_parsed.json").write_text(
        json.dumps(parsed_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{artifact_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / f"{artifact_prefix}_batch_metrics.json").write_text(
        json.dumps(batch_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_vllm_sampling_params_for_max_tokens(max_new_tokens: int) -> tuple[Any, str]:
    from vllm import SamplingParams

    base_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "skip_special_tokens": True,
    }
    schema = build_stage11_output_schema()
    attempts: list[tuple[str, dict[str, Any]]] = []
    try:
        from vllm.sampling_params import StructuredOutputsParams

        attempts.append(
            (
                "structured_outputs_json_schema",
                {
                    **base_kwargs,
                    "structured_outputs": StructuredOutputsParams(json=schema),
                },
            )
        )
        attempts.append(
            (
                "structured_outputs_json_text",
                {
                    **base_kwargs,
                    "structured_outputs": StructuredOutputsParams(json=json.dumps(schema, ensure_ascii=False)),
                },
            )
        )
    except Exception:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams

        attempts.append(
            (
                "guided_decoding_json_schema",
                {
                    **base_kwargs,
                    "guided_decoding": GuidedDecodingParams(json=schema),
                },
            )
        )
    except Exception:
        pass
    attempts.append(("guided_json_dict", {**base_kwargs, "guided_json": schema}))
    attempts.append(("guided_json_text", {**base_kwargs, "guided_json": json.dumps(schema, ensure_ascii=False)}))

    last_error: Exception | None = None
    for mode, kwargs in attempts:
        try:
            params = SamplingParams(**kwargs)
            return params, mode
        except Exception as exc:
            last_error = exc
    if VLLM_REQUIRE_GUIDED_DECODING:
        raise RuntimeError("vLLM guided decoding is unavailable for the installed vLLM build.") from last_error
    return SamplingParams(**base_kwargs), "unguided"


def create_generation_backend(max_new_tokens: int) -> GenerationBackendState:
    import torch
    from transformers import AutoTokenizer

    if INFER_BACKEND == "hf":
        maybe_disable_transformers_grouped_mm()

    tokenizer_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer_load_seconds = round(time.perf_counter() - tokenizer_load_start, 4)

    dtype = torch.bfloat16 if USE_BF16 else torch.float16
    model_load_start = time.perf_counter()
    guided_decoding_mode = "disabled"

    if INFER_BACKEND == "hf":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": 0},
            attn_implementation=ATTN_IMPL,
        )
        model.eval()
    elif INFER_BACKEND == "vllm":
        try:
            from vllm import LLM
        except Exception as exc:
            raise RuntimeError("PROMPT_ONLY_INFER_BACKEND=vllm but vllm is not importable.") from exc

        engine_kwargs: dict[str, Any] = {
            "model": BASE_MODEL,
            "trust_remote_code": True,
            "tensor_parallel_size": max(VLLM_TENSOR_PARALLEL_SIZE, 1),
            "dtype": "bfloat16" if USE_BF16 else "float16",
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "enforce_eager": VLLM_ENFORCE_EAGER,
            "max_num_seqs": VLLM_MAX_NUM_SEQS if VLLM_MAX_NUM_SEQS > 0 else max(GEN_BATCH_SIZE, 1),
            "max_num_batched_tokens": (
                VLLM_MAX_NUM_BATCHED_TOKENS if VLLM_MAX_NUM_BATCHED_TOKENS > 0 else max(MAX_BATCH_PROMPT_TOKENS, 0)
            ),
        }
        if VLLM_MAX_MODEL_LEN > 0:
            engine_kwargs["max_model_len"] = VLLM_MAX_MODEL_LEN
        if engine_kwargs["max_num_batched_tokens"] <= 0:
            engine_kwargs.pop("max_num_batched_tokens", None)
        model = LLM(**engine_kwargs)
        _, guided_decoding_mode = build_vllm_sampling_params_for_max_tokens(max_new_tokens)
    else:
        raise ValueError(f"unsupported PROMPT_ONLY_INFER_BACKEND={INFER_BACKEND}")

    model_load_seconds = round(time.perf_counter() - model_load_start, 4)
    return GenerationBackendState(
        tokenizer=tokenizer,
        model=model,
        infer_backend=INFER_BACKEND,
        guided_decoding_mode=guided_decoding_mode,
        tokenizer_load_seconds=tokenizer_load_seconds,
        model_load_seconds=model_load_seconds,
    )


def close_generation_backend(state: GenerationBackendState | None) -> None:
    if state is None:
        return
    try:
        engine = getattr(getattr(state, "model", None), "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass
    model = getattr(state, "model", None)
    tokenizer = getattr(state, "tokenizer", None)
    state.model = None
    state.tokenizer = None
    del model
    del tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_generation(
    records: list[PromptRecord],
    output_dir: Path,
    *,
    artifact_prefix: str,
    max_new_tokens: int,
    existing_state: GenerationRunResult | None = None,
    backend_state: GenerationBackendState | None = None,
) -> GenerationRunResult:
    import torch

    generation_start = time.perf_counter()
    structured_output_schema = clone_stage11_output_schema()
    if WRITE_STRUCTURED_OUTPUT_SCHEMA:
        (output_dir / f"{artifact_prefix}_structured_output_schema.json").write_text(
            json.dumps(structured_output_schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    owns_backend = backend_state is None
    state = backend_state or create_generation_backend(max_new_tokens)
    tokenizer = state.tokenizer
    model = state.model
    guided_decoding_mode = state.guided_decoding_mode
    tokenizer_load_seconds = state.tokenizer_load_seconds if owns_backend else 0.0
    model_load_seconds = state.model_load_seconds if owns_backend else 0.0
    prompt_length_start = time.perf_counter()
    prompt_token_lengths = estimate_prompt_token_lengths(records, tokenizer)
    prompt_length_seconds = round(time.perf_counter() - prompt_length_start, 4)

    raw_rows: list[dict[str, Any]] = list(existing_state.raw_rows) if existing_state else []
    parsed_rows: list[dict[str, Any]] = list(existing_state.parsed_rows) if existing_state else []
    batch_metrics: list[dict[str, Any]] = list(existing_state.batch_metrics) if existing_state else []
    records_by_index = {rec.input_index: rec for rec in records}
    completed_user_ids = {
        str(row.get("user_id", "") or "")
        for row in raw_rows
        if str(row.get("user_id", "") or "")
    }
    resume_loaded_rows = len(completed_user_ids)
    raw_jsonl_path = output_dir / f"{artifact_prefix}_raw.jsonl"
    parsed_jsonl_path = output_dir / f"{artifact_prefix}_parsed.jsonl"
    if existing_state is None:
        for path in [raw_jsonl_path, parsed_jsonl_path]:
            if path.exists():
                path.unlink()
    elif resume_loaded_rows:
        print(
            f"[INFO] resuming {artifact_prefix} with {resume_loaded_rows} completed rows already loaded.",
            flush=True,
        )

    def generate_batch(batch: list[PromptRecord], depth: int = 0) -> None:
        batch_start = time.perf_counter()
        prompts = [r.prompt_text for r in batch]
        prompt_lengths = [int(prompt_token_lengths.get(r.input_index, 0)) for r in batch]
        padded_width = max(prompt_lengths) if prompt_lengths else 0
        encoded = None
        input_ids = None
        attention_mask = None
        outputs = None
        generated_ids = None
        texts: list[str] = []
        generated_token_lengths: list[int] = []
        try:
            if INFER_BACKEND == "hf":
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                padded_width = int(input_ids.shape[1])
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        temperature=1.0,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                generated_ids = outputs[:, padded_width:]
                texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_token_lengths = [int(ids.shape[0]) for ids in generated_ids]
            else:
                sampling_params, _ = build_vllm_sampling_params_for_max_tokens(max_new_tokens)
                try:
                    request_outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
                except TypeError:
                    request_outputs = model.generate(prompts, sampling_params=sampling_params)
                texts = []
                generated_token_lengths = []
                for request_output in request_outputs:
                    if not getattr(request_output, "outputs", None):
                        texts.append("")
                        generated_token_lengths.append(0)
                        continue
                    candidate = request_output.outputs[0]
                    texts.append(str(getattr(candidate, "text", "") or ""))
                    token_ids = getattr(candidate, "token_ids", None)
                    if isinstance(token_ids, list):
                        generated_token_lengths.append(len(token_ids))
                    else:
                        generated_token_lengths.append(0)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and len(batch) > 1:
                torch.cuda.empty_cache()
                mid = len(batch) // 2
                batch_metrics.append(
                    {
                        "batch_size": len(batch),
                        "prompt_lengths": prompt_lengths,
                        "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
                        "prompt_token_budget": padded_width * len(batch),
                        "status": "split_on_oom",
                        "split_depth": depth,
                        "elapsed_seconds": round(time.perf_counter() - batch_start, 4),
                    }
                )
                generate_batch(batch[:mid], depth + 1)
                generate_batch(batch[mid:], depth + 1)
                return
            raise
        batch_elapsed = round(time.perf_counter() - batch_start, 4)
        batch_metrics.append(
            {
                "batch_size": len(batch),
                "prompt_lengths": prompt_lengths,
                "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
                "prompt_token_budget": padded_width * len(batch),
                "generated_token_lengths": generated_token_lengths,
                "elapsed_seconds": batch_elapsed,
                "status": "completed",
                "split_depth": depth,
                "infer_backend": INFER_BACKEND,
                "guided_decoding_mode": guided_decoding_mode,
            }
        )
        print(
            f"[BATCH] size={len(batch)} max_prompt_tokens={max(prompt_lengths) if prompt_lengths else 0} "
            f"prompt_budget={padded_width * len(batch)} generated_tokens={generated_token_lengths} "
            f"elapsed={batch_elapsed:.2f}s depth={depth} backend={INFER_BACKEND} guided={guided_decoding_mode}",
            flush=True,
        )
        raw_batch_rows: list[dict[str, Any]] = []
        parsed_batch_rows: list[dict[str, Any]] = []
        for rec, raw_text in zip(batch, texts):
            extracted = extract_first_json_object(raw_text)
            parsed = None
            valid = False
            schema_valid = False
            allowed_refs = sorted(collect_allowed_evidence_ids(rec))
            allowed_ref_set = set(allowed_refs)
            used_refs: list[str] = []
            invalid_refs: list[str] = []
            ref_groups: list[list[str]] = []
            unknown_fields: list[str] = []
            schema_issues: list[dict[str, Any]] = []
            support_issues: list[dict[str, Any]] = []
            guardrail_failures: list[dict[str, Any]] = []
            parsed_candidate = None
            parsed_repaired = None
            repair_actions: list[dict[str, Any]] = []
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    valid = True
                    parsed_candidate = parsed
                    parsed_repaired, repair_actions = repair_stage11_output_refs(
                        parsed_candidate,
                        allowed_refs=allowed_ref_set,
                    )
                    parsed_repaired, promoted_repair_actions = repair_promoted_merchant_items(parsed_repaired)
                    repair_actions.extend(promoted_repair_actions)
                    schema_issues = validate_stage11_output_schema(parsed_repaired)
                    schema_valid = not schema_issues
                    used_refs = collect_evidence_refs(parsed_repaired)
                    ref_groups = collect_ref_groups(parsed_repaired)
                    unknown_fields = collect_unknown_fields(parsed_repaired)
                    support_issues = collect_support_issues(parsed_repaired)
                    invalid_refs = sorted({ref for ref in used_refs if ref not in allowed_ref_set})
                    guardrail_failures = collect_guardrail_failures(
                        parsed_repaired,
                        schema_issues=schema_issues,
                        support_issues=support_issues,
                        invalid_refs=invalid_refs,
                    )
                except Exception:
                    parsed = None
                    parsed_candidate = None
                    parsed_repaired = None
                    repair_actions = []
            guardrails_passed = bool(valid and not guardrail_failures)
            accepted_parsed = parsed_repaired
            if GUARDRAILS_HARD_FAIL and not guardrails_passed:
                accepted_parsed = None
            raw_batch_rows.append(
                {
                    "user_id": rec.user_id,
                    "density_band": rec.density_band,
                    "selected_rank_in_band": rec.selected_rank_in_band,
                    "input_index": rec.input_index,
                    "selection_score": rec.selection_score,
                    "raw_text": raw_text,
                    "extracted_json_text": extracted,
                    "json_valid": valid,
                    "schema_valid": schema_valid,
                    "schema_issues": schema_issues,
                    "allowed_evidence_refs": allowed_refs,
                    "used_evidence_refs": used_refs,
                    "invalid_evidence_refs": invalid_refs,
                    "ref_group_sizes": [len(group) for group in ref_groups],
                    "unknown_fields": unknown_fields,
                    "support_issues": support_issues,
                    "repair_actions": repair_actions,
                    "guardrail_failures": guardrail_failures,
                    "guardrails_passed": guardrails_passed,
                    "infer_backend": INFER_BACKEND,
                    "guided_decoding_mode": guided_decoding_mode,
                    "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
                }
            )
            parsed_batch_rows.append(
                {
                    "user_id": rec.user_id,
                    "density_band": rec.density_band,
                    "selected_rank_in_band": rec.selected_rank_in_band,
                    "input_index": rec.input_index,
                    "selection_score": rec.selection_score,
                    "parsed_json": accepted_parsed,
                    "parsed_json_candidate": parsed_candidate,
                    "json_valid": valid,
                    "schema_valid": schema_valid,
                    "schema_issues": schema_issues,
                    "allowed_evidence_refs": allowed_refs,
                    "used_evidence_refs": used_refs,
                    "invalid_evidence_refs": invalid_refs,
                    "ref_group_sizes": [len(group) for group in ref_groups],
                    "unknown_fields": unknown_fields,
                    "support_issues": support_issues,
                    "repair_actions": repair_actions,
                    "guardrail_failures": guardrail_failures,
                    "guardrails_passed": guardrails_passed,
                    "infer_backend": INFER_BACKEND,
                    "guided_decoding_mode": guided_decoding_mode,
                    "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
                    "parsed_json_repaired": parsed_repaired,
                }
            )
        raw_rows.extend(raw_batch_rows)
        parsed_rows.extend(parsed_batch_rows)
        append_jsonl_rows(raw_jsonl_path, raw_batch_rows)
        append_jsonl_rows(parsed_jsonl_path, parsed_batch_rows)
        if outputs is not None:
            del outputs
        if generated_ids is not None:
            del generated_ids
        if input_ids is not None:
            del input_ids
        if attention_mask is not None:
            del attention_mask
        if encoded is not None:
            del encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generation_batches = build_generation_batches(records, prompt_token_lengths)
    planned_batch_sizes = [len(batch) for batch in generation_batches]
    remaining_records = [record for record in records if record.user_id not in completed_user_ids]
    remaining_batches = build_generation_batches(remaining_records, prompt_token_lengths)
    next_checkpoint_batch = CHECKPOINT_EVERY_BATCHES if CHECKPOINT_EVERY_BATCHES > 0 else None
    stop_requested = False

    def emit_checkpoint(checkpoint_label: str) -> None:
        generation_seconds = round(time.perf_counter() - generation_start, 4)
        checkpoint_summary = build_generation_summary(
            artifact_prefix=artifact_prefix,
            raw_rows=raw_rows,
            parsed_rows=parsed_rows,
            batch_metrics=batch_metrics,
            planned_batch_sizes=planned_batch_sizes,
            max_new_tokens=max_new_tokens,
            tokenizer_load_seconds=tokenizer_load_seconds,
            model_load_seconds=model_load_seconds,
            prompt_length_seconds=prompt_length_seconds,
            generation_seconds=generation_seconds,
            records_by_index=records_by_index,
            total_rows_planned=len(records),
            stop_requested=stop_requested,
        )
        checkpoint_summary["resume_loaded_rows"] = resume_loaded_rows
        checkpoint_summary["checkpoint_label"] = checkpoint_label
        write_checkpoint_snapshot(
            output_dir,
            artifact_prefix,
            checkpoint_summary,
            checkpoint_label=checkpoint_label,
        )

    for batch in remaining_batches:
        if STOP_REQUEST_PATH is not None and STOP_REQUEST_PATH.exists():
            stop_requested = True
            print(
                f"[INFO] stop request detected at {STOP_REQUEST_PATH.as_posix()}; finishing current artifacts and exiting early.",
                flush=True,
            )
            break
        generate_batch(batch)
        completed_batches = sum(1 for metric in batch_metrics if metric.get("status") == "completed")
        while next_checkpoint_batch is not None and completed_batches >= next_checkpoint_batch:
            emit_checkpoint(f"batch_{next_checkpoint_batch:05d}")
            next_checkpoint_batch += CHECKPOINT_EVERY_BATCHES

    raw_rows.sort(key=lambda row: row["input_index"])
    parsed_rows.sort(key=lambda row: row["input_index"])
    generation_seconds = round(time.perf_counter() - generation_start, 4)
    summary = build_generation_summary(
        artifact_prefix=artifact_prefix,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        batch_metrics=batch_metrics,
        planned_batch_sizes=planned_batch_sizes,
        max_new_tokens=max_new_tokens,
        tokenizer_load_seconds=tokenizer_load_seconds,
        model_load_seconds=model_load_seconds,
        prompt_length_seconds=prompt_length_seconds,
        generation_seconds=generation_seconds,
        records_by_index=records_by_index,
        total_rows_planned=len(records),
        stop_requested=stop_requested,
    )
    summary["resume_loaded_rows"] = resume_loaded_rows
    write_generation_artifacts(
        output_dir,
        artifact_prefix,
        raw_rows,
        parsed_rows,
        summary,
        batch_metrics,
    )
    write_checkpoint_snapshot(
        output_dir,
        artifact_prefix,
        summary,
        checkpoint_label="final",
    )
    if owns_backend:
        close_generation_backend(state)
    return GenerationRunResult(
        artifact_prefix=artifact_prefix,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        summary=summary,
        batch_metrics=batch_metrics,
    )


def build_pass2_draft_payload(pass1_row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(pass1_row, dict):
        return {"pass1_json_valid": False, "draft": None}
    return {
        "pass1_json_valid": bool(pass1_row.get("parsed_json") is not None),
        "draft": pass1_row.get("parsed_json"),
    }


def build_pass2_records(
    prompt_records: list[PromptRecord],
    pass1_result: GenerationRunResult,
    pass2_template_text: str,
    *,
    compact_mode: bool = False,
) -> list[PromptRecord]:
    pass1_by_index = {int(row["input_index"]): row for row in pass1_result.parsed_rows}
    pass2_records: list[PromptRecord] = []
    for rec in prompt_records:
        pass1_row = pass1_by_index.get(rec.input_index)
        pass1_payload = (
            build_pass2_compact_payload(rec, pass1_row)
            if compact_mode
            else build_pass2_draft_payload(pass1_row)
        )
        prompt_text = render_prompt(
            pass2_template_text,
            rec,
            extra_replacements={
                "{PASS1_DRAFT_JSON}": json.dumps(pass1_payload, ensure_ascii=False, indent=2),
            },
        )
        pass2_records.append(replace(rec, prompt_text=prompt_text))
    return pass2_records


def serialize_prompt_records(
    records: list[PromptRecord],
    *,
    include_prompt_text: bool = True,
) -> list[dict[str, Any]]:
    return [
        {
            "user_id": rec.user_id,
            "density_band": rec.density_band,
            "selected_rank_in_band": rec.selected_rank_in_band,
            "input_index": rec.input_index,
            "selection_score": rec.selection_score,
            "user_meta": rec.user_meta,
            "user_evidence_stream": rec.user_evidence_stream,
            "recent_event_sequence": rec.recent_event_sequence,
            "anchor_positive_events": rec.anchor_positive_events,
            "anchor_negative_events": rec.anchor_negative_events,
            "anchor_conflict_events": rec.anchor_conflict_events,
            "context_notes": rec.context_notes,
            **({"prompt_text": rec.prompt_text} if include_prompt_text else {}),
        }
        for rec in records
    ]


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_RUN_DIR if OUTPUT_RUN_DIR is not None else OUTPUT_ROOT_DIR / f"{timestamp}_full_stage11_prompt_only_user_state_audit"
    run_dir.mkdir(parents=True, exist_ok=True)

    if EXISTING_PASS1_RUN_DIR is not None and not RUN_PASS2:
        raise ValueError("INPUT_11_PROMPT_ONLY_EXISTING_PASS1_RUN_DIR requires PROMPT_ONLY_RUN_PASS2=true")

    df = pd.read_parquet(INPUT_USER_STATE_RUN_DIR / "user_state_labels_v1.parquet")
    selected_rows = select_rows(df)
    excluded_user_ids: set[str] = set()
    for completed_run_dir in EXCLUDE_COMPLETED_PASS1_RUN_DIRS:
        excluded_user_ids.update(collect_completed_user_ids_from_run_dir(completed_run_dir))
    selected_rows, excluded_completed_user_count = filter_selected_rows_by_user_ids(
        selected_rows,
        excluded_user_ids,
    )
    merchant_map = load_merchant_state_map()
    template_text = load_prompt_template_text(PROMPT_TEMPLATE_PATH)

    prompt_records: list[PromptRecord] = []
    for row in selected_rows:
        review_stream = build_user_evidence_stream(row)
        recent_events = [
            enrich_event(ev, merchant_map)
            for ev in parse_json_list(row.get("recent_event_sequence_json"))
        ]
        recent_events = select_diverse_events(
            recent_events,
            MAX_RECENT_EVENTS,
            MAX_EVENTS_PER_PRIMARY_CUISINE,
        )
        recent_events = assign_event_ids(recent_events, "event")
        recent_events = [trim_event_for_prompt(ev) for ev in recent_events]
        positive_events = [
            enrich_event(ev, merchant_map)
            for ev in parse_json_list(row.get("positive_anchor_sequence_json"))
        ]
        positive_events = select_diverse_events(
            positive_events,
            MAX_POS_ANCHORS,
            MAX_ANCHORS_PER_PRIMARY_CUISINE,
        )
        positive_events = assign_event_ids(positive_events, "anchor_pos")
        positive_events = [trim_event_for_prompt(ev) for ev in positive_events]
        negative_events = [
            enrich_event(ev, merchant_map)
            for ev in parse_json_list(row.get("negative_anchor_sequence_json"))
        ]
        negative_events = select_diverse_events(
            negative_events,
            MAX_NEG_ANCHORS,
            MAX_ANCHORS_PER_PRIMARY_CUISINE,
        )
        negative_events = assign_event_ids(negative_events, "anchor_neg")
        negative_events = [trim_event_for_prompt(ev) for ev in negative_events]
        conflict_events = build_conflict_events(positive_events, negative_events)
        conflict_events = [trim_event_for_prompt(ev) for ev in conflict_events]
        user_meta = build_user_meta(row)
        context_notes = build_context_notes(row, recent_events, positive_events, negative_events)
        record = PromptRecord(
            input_index=len(prompt_records),
            user_id=str(row["user_id"]),
            density_band=str(row["density_band"]),
            selected_rank_in_band=int(row["selected_rank_in_band"]),
            user_meta=user_meta,
            user_evidence_stream=review_stream,
            recent_event_sequence=recent_events,
            anchor_positive_events=positive_events,
            anchor_negative_events=negative_events,
            anchor_conflict_events=conflict_events,
            context_notes=context_notes,
            prompt_text="",
            selection_score=float(row["selection_score"]),
        )
        record.prompt_text = render_prompt(template_text, record)
        prompt_records.append(record)

    active_prompt_records = prompt_records
    existing_pass1_result: GenerationRunResult | None = None
    reused_pass1_user_count = 0
    if EXISTING_PASS1_RUN_DIR is not None:
        existing_pass1_result = load_generation_result_from_dir(
            EXISTING_PASS1_RUN_DIR,
            "prompt_only_pass1_generation",
            prompt_records,
        )
        if existing_pass1_result is None:
            raise ValueError(f"no reusable pass-1 artifacts found in {EXISTING_PASS1_RUN_DIR}")
        reused_user_ids = {
            str(row.get("user_id", "") or "")
            for row in existing_pass1_result.raw_rows
            if str(row.get("user_id", "") or "")
        }
        active_prompt_records = [record for record in prompt_records if record.user_id in reused_user_ids]
        reused_pass1_user_count = len(active_prompt_records)
        if not active_prompt_records:
            raise ValueError("existing pass-1 artifacts do not overlap current selected users")

    write_prompt_record_dump(run_dir, "prompt_only_inputs", active_prompt_records)

    summary = {
        "rows": len(active_prompt_records),
        "selected_rows_before_reuse_filter": len(prompt_records),
        "sample_per_band": SAMPLE_PER_BAND,
        "select_mode": SELECT_MODE,
        "max_total_rows": MAX_TOTAL_ROWS,
        "bands": Counter(rec.density_band for rec in active_prompt_records),
        "max_review_evidence": MAX_REVIEW_EVIDENCE,
        "max_tip_evidence": MAX_TIP_EVIDENCE,
        "max_recent_events": MAX_RECENT_EVENTS,
        "max_pos_anchors": MAX_POS_ANCHORS,
        "max_neg_anchors": MAX_NEG_ANCHORS,
        "max_conflict_events": MAX_CONFLICT_EVENTS,
        "sanitize_merchant_context": SANITIZE_MERCHANT_CONTEXT,
        "max_events_per_primary_cuisine": MAX_EVENTS_PER_PRIMARY_CUISINE,
        "max_anchors_per_primary_cuisine": MAX_ANCHORS_PER_PRIMARY_CUISINE,
        "run_inference": RUN_INFERENCE,
        "run_pass2": RUN_PASS2,
        "run_pass2_compact_input": RUN_PASS2_COMPACT_INPUT,
        "infer_backend": INFER_BACKEND,
        "guardrails_hard_fail": GUARDRAILS_HARD_FAIL,
        "write_structured_output_schema": WRITE_STRUCTURED_OUTPUT_SCHEMA,
        "structured_output_schema_version": STRUCTURED_OUTPUT_SCHEMA_VERSION,
        "vllm_runtime": {
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "max_num_batched_tokens": VLLM_MAX_NUM_BATCHED_TOKENS,
            "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
            "enforce_eager": VLLM_ENFORCE_EAGER,
            "require_guided_decoding": VLLM_REQUIRE_GUIDED_DECODING,
        },
        "prompt_template_path": str(PROMPT_TEMPLATE_PATH),
        "pass2_prompt_template_path": str(PASS2_PROMPT_TEMPLATE_PATH) if PASS2_PROMPT_TEMPLATE_PATH else "",
        "pass2_gen_max_new_tokens": PASS2_GEN_MAX_NEW_TOKENS if RUN_PASS2 else None,
        "checkpoint_every_batches": CHECKPOINT_EVERY_BATCHES,
        "checkpoint_sample_rows": CHECKPOINT_SAMPLE_ROWS,
        "stop_request_path": str(STOP_REQUEST_PATH) if STOP_REQUEST_PATH else "",
        "write_full_inputs": WRITE_FULL_INPUTS,
        "include_prompt_text_in_inputs": INCLUDE_PROMPT_TEXT_IN_INPUTS,
        "input_preview_rows": INPUT_PREVIEW_ROWS,
        "output_run_dir": str(run_dir),
        "resume_from_existing": RESUME_FROM_EXISTING,
        "single_stage_artifact_prefix": SINGLE_STAGE_ARTIFACT_PREFIX,
        "existing_pass1_run_dir": str(EXISTING_PASS1_RUN_DIR) if EXISTING_PASS1_RUN_DIR else "",
        "reuse_existing_pass1_user_count": reused_pass1_user_count,
        "exclude_completed_pass1_run_dirs": [str(path) for path in EXCLUDE_COMPLETED_PASS1_RUN_DIRS],
        "excluded_completed_user_count": excluded_completed_user_count,
    }
    (run_dir / "prompt_only_input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if RUN_INFERENCE:
        if RUN_PASS2 and PASS2_PROMPT_TEMPLATE_PATH is None:
            raise ValueError("PROMPT_ONLY_RUN_PASS2=true requires INPUT_QWEN35_PASS2_PROMPT_TEMPLATE_PATH")
        if RUN_PASS2:
            if existing_pass1_result is not None:
                pass1_result = existing_pass1_result
            else:
                existing_pass1_state = (
                    load_generation_result_from_dir(run_dir, "prompt_only_pass1_generation", active_prompt_records)
                    if RESUME_FROM_EXISTING
                    else None
                )
                pass1_result = run_generation(
                    active_prompt_records,
                    run_dir,
                    artifact_prefix="prompt_only_pass1_generation",
                    max_new_tokens=GEN_MAX_NEW_TOKENS,
                    existing_state=existing_pass1_state,
                )
            pass2_template_text = load_prompt_template_text(PASS2_PROMPT_TEMPLATE_PATH)
            pass2_records = build_pass2_records(
                active_prompt_records,
                pass1_result,
                pass2_template_text,
                compact_mode=RUN_PASS2_COMPACT_INPUT,
            )
            if existing_pass1_result is None and pass1_result.summary.get("stopped_early"):
                (run_dir / "prompt_only_two_stage_summary.json").write_text(
                    json.dumps(
                        {
                            "final_stage": "pass1",
                            "stopped_early": True,
                            "pass2_compact_input": RUN_PASS2_COMPACT_INPUT,
                            "pass1_summary": pass1_result.summary,
                            "pass2_summary": None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(json.dumps({"run_dir": run_dir.as_posix(), **summary, "stopped_early": True}, ensure_ascii=False, indent=2))
                return
            write_prompt_record_dump(run_dir, "prompt_only_pass2_inputs", pass2_records)
            existing_pass2_state = (
                load_generation_result_from_dir(run_dir, "prompt_only_pass2_generation", pass2_records)
                if RESUME_FROM_EXISTING
                else None
            )
            pass2_result = run_generation(
                pass2_records,
                run_dir,
                artifact_prefix="prompt_only_pass2_generation",
                max_new_tokens=PASS2_GEN_MAX_NEW_TOKENS,
                existing_state=existing_pass2_state,
            )
            write_generation_artifacts(
                run_dir,
                "prompt_only_generation",
                pass2_result.raw_rows,
                pass2_result.parsed_rows,
                {**pass2_result.summary, "final_stage": "pass2"},
                pass2_result.batch_metrics,
            )
            (run_dir / "prompt_only_two_stage_summary.json").write_text(
                json.dumps(
                    {
                        "final_stage": "pass2",
                        "pass2_compact_input": RUN_PASS2_COMPACT_INPUT,
                        "pass1_summary": pass1_result.summary,
                        "pass2_summary": pass2_result.summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            single_stage_artifact_prefix = SINGLE_STAGE_ARTIFACT_PREFIX or "prompt_only_generation"
            run_generation(
                active_prompt_records,
                run_dir,
                artifact_prefix=single_stage_artifact_prefix,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                existing_state=(
                    load_generation_result_from_dir(run_dir, single_stage_artifact_prefix, active_prompt_records)
                    if RESUME_FROM_EXISTING
                    else None
                ),
            )

    print(json.dumps({"run_dir": run_dir.as_posix(), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
