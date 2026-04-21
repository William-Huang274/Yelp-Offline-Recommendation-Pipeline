from __future__ import annotations

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script}")
    print("This stage script is configured by environment variables and starts a dataset build job.")
    print("Set the required INPUT_/OUTPUT_ environment variables, then run without --help.")
    sys.exit(0)

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path, write_latest_run_pointer

USER_PREF_ROOT = env_or_project_path(
    "INPUT_09_USER_PREFERENCE_SCHEMA_V1_ROOT_DIR",
    "data/output/09_user_preference_schema_v1",
)
USER_PREF_RUN_DIR = os.getenv("INPUT_09_USER_PREFERENCE_SCHEMA_V1_RUN_DIR", "").strip()

USER_PROFILES_ROOT = env_or_project_path(
    "INPUT_09_USER_PROFILES_ROOT_DIR",
    "data/output/09_user_profiles",
)
USER_PROFILES_RUN_DIR = os.getenv("INPUT_09_USER_PROFILES_RUN_DIR", "").strip()

TIP_SIGNAL_ROOT = env_or_project_path(
    "INPUT_09_TIP_SIGNALS_ROOT_DIR",
    "data/output/09_tip_signals",
)
TIP_SIGNAL_RUN_DIR = os.getenv("INPUT_09_TIP_SIGNALS_RUN_DIR", "").strip()

SEQUENCE_ROOT = env_or_project_path(
    "INPUT_09_SEQUENCE_VIEW_SCHEMA_V1_ROOT_DIR",
    "data/output/09_sequence_view_schema_v1",
)
SEQUENCE_RUN_DIR = os.getenv("INPUT_09_SEQUENCE_VIEW_SCHEMA_V1_RUN_DIR", "").strip()

TARGET_ROOT = env_or_project_path(
    "INPUT_09_USER_PREFERENCE_TARGET_SCHEMA_V1_ROOT_DIR",
    "data/output/09_user_preference_target_schema_v1",
)
TARGET_RUN_DIR = os.getenv("INPUT_09_USER_PREFERENCE_TARGET_SCHEMA_V1_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_11_PERSONA_SFT_DATA_ROOT_DIR",
    "data/output/11_persona_sft_data",
)
RUN_TAG = os.getenv("RUN_TAG", "stage11_1_persona_sft_build_dataset").strip() or "stage11_1_persona_sft_build_dataset"
PROMPT_VERSION = (os.getenv("PERSONA_SFT_PROMPT_VERSION", "v2").strip() or "v2").lower()

EVAL_USER_COHORT_PATH = Path(
    os.getenv(
        "PERSONA_SFT_EVAL_USER_COHORT_PATH",
        project_path("data/output/fixed_eval_cohorts/bucket5_accepted_test_users_1935_userid.csv").as_posix(),
    ).strip()
)
EVAL_USER_FRAC = float(os.getenv("PERSONA_SFT_EVAL_USER_FRAC", "0.15").strip() or 0.15)
MAX_TRAIN_ROWS = int(os.getenv("PERSONA_SFT_MAX_TRAIN_ROWS", "0").strip() or 0)
MAX_EVAL_ROWS = int(os.getenv("PERSONA_SFT_MAX_EVAL_ROWS", "0").strip() or 0)
SEED = int(os.getenv("PERSONA_SFT_RANDOM_SEED", "42").strip() or 42)
SAMPLE_ROWS = int(os.getenv("PERSONA_SFT_SAMPLE_ROWS", "8").strip() or 8)
PROFILE_TEXT_MAX_CHARS = int(os.getenv("PERSONA_SFT_PROFILE_TEXT_MAX_CHARS", "2400").strip() or 2400)
INCLUDE_DERIVED_NARRATIVE = os.getenv("PERSONA_SFT_INCLUDE_DERIVED_NARRATIVE", "false").strip().lower() == "true"
V3_MIN_TEXT_CHARS = int(os.getenv("PERSONA_SFT_V3_MIN_TEXT_CHARS", "12").strip() or 12)
V3_MIN_TEXT_WORDS = int(os.getenv("PERSONA_SFT_V3_MIN_TEXT_WORDS", "2").strip() or 2)
V3_REVIEW_MAX_ITEMS = int(os.getenv("PERSONA_SFT_V3_REVIEW_MAX_ITEMS", "0").strip() or 0)
V3_TIP_MAX_ITEMS = int(os.getenv("PERSONA_SFT_V3_TIP_MAX_ITEMS", "0").strip() or 0)
V3_INCLUDE_PROFILE_DIGEST = os.getenv("PERSONA_SFT_V3_INCLUDE_PROFILE_DIGEST", "true").strip().lower() == "true"
V3_PROFILE_DIGEST_MIN_ITEMS = int(os.getenv("PERSONA_SFT_V3_PROFILE_DIGEST_MIN_ITEMS", "8").strip() or 8)
V3_APPLY_PACKING = os.getenv("PERSONA_SFT_V3_APPLY_PACKING", "false").strip().lower() == "true"
V3_PACK_LOW_TOTAL_CAP = int(os.getenv("PERSONA_SFT_PACK_LOW_TOTAL_CAP", "18").strip() or 18)
V3_PACK_LOW_TIP_CAP = int(os.getenv("PERSONA_SFT_PACK_LOW_TIP_CAP", "4").strip() or 4)
V3_PACK_LOW_NEG_FLOOR = int(os.getenv("PERSONA_SFT_PACK_LOW_NEG_FLOOR", "2").strip() or 2)
V3_PACK_MID_TOTAL_CAP = int(os.getenv("PERSONA_SFT_PACK_MID_TOTAL_CAP", "24").strip() or 24)
V3_PACK_MID_TIP_CAP = int(os.getenv("PERSONA_SFT_PACK_MID_TIP_CAP", "6").strip() or 6)
V3_PACK_MID_NEG_FLOOR = int(os.getenv("PERSONA_SFT_PACK_MID_NEG_FLOOR", "3").strip() or 3)
V3_PACK_HIGH_TOTAL_CAP = int(os.getenv("PERSONA_SFT_PACK_HIGH_TOTAL_CAP", "36").strip() or 36)
V3_PACK_HIGH_TIP_CAP = int(os.getenv("PERSONA_SFT_PACK_HIGH_TIP_CAP", "12").strip() or 12)
V3_PACK_HIGH_NEG_FLOOR = int(os.getenv("PERSONA_SFT_PACK_HIGH_NEG_FLOOR", "4").strip() or 4)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"run dir not found: {path}")
        return path
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def clip_text(value: Any, max_chars: int) -> str:
    text = safe_text(value)
    if not text:
        return ""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    try:
        items = list(value)
    except TypeError:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = safe_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def safe_discovery_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_refs = safe_list(item.get("evidence_refs"))
        text_type = safe_text(item.get("type"))
        claim = safe_text(item.get("claim"))
        confidence = safe_text(item.get("confidence")) or "low"
        reason_short = safe_text(item.get("reason_short"))
        if not text_type or not claim:
            continue
        out.append(
            {
                "type": text_type,
                "claim": claim,
                "confidence": confidence,
                "evidence_refs": evidence_refs,
                "reason_short": reason_short,
            }
        )
    return out


def density_band(n_train_max: Any) -> str:
    try:
        numeric = int(n_train_max or 0)
    except Exception:
        return "unknown"
    if numeric <= 7:
        return "b5_lower_visible_5_7"
    if numeric <= 17:
        return "b5_mid_visible_8_17"
    return "b5_high_visible_18_plus"


def density_pack_cfg(n_train_max: Any) -> dict[str, int]:
    band = density_band(n_train_max)
    if band == "b5_high_visible_18_plus":
        return {"total_cap": V3_PACK_HIGH_TOTAL_CAP, "tip_cap": V3_PACK_HIGH_TIP_CAP, "neg_floor": V3_PACK_HIGH_NEG_FLOOR}
    if band == "b5_mid_visible_8_17":
        return {"total_cap": V3_PACK_MID_TOTAL_CAP, "tip_cap": V3_PACK_MID_TIP_CAP, "neg_floor": V3_PACK_MID_NEG_FLOOR}
    return {"total_cap": V3_PACK_LOW_TOTAL_CAP, "tip_cap": V3_PACK_LOW_TIP_CAP, "neg_floor": V3_PACK_LOW_NEG_FLOOR}


def load_json_list(raw: Any) -> list[dict[str, Any]]:
    text = safe_text(raw)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    out = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def is_usable_text(value: Any) -> bool:
    text = safe_text(value)
    if not text:
        return False
    lowered = text.lower()
    if len(text) < V3_MIN_TEXT_CHARS and len(text.split()) < V3_MIN_TEXT_WORDS:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if lowered in {"good", "great", "nice", "ok", "okay", "yummy", "delicious", "awesome"}:
        return False
    return True


def maybe_cap_items(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    return items[:max_items]


def pack_unified_evidence_items(items: list[dict[str, Any]], n_train_max: Any) -> list[dict[str, Any]]:
    cfg = density_pack_cfg(n_train_max)
    total_cap = cfg["total_cap"]
    tip_cap = cfg["tip_cap"]
    neg_floor = cfg["neg_floor"]
    if total_cap <= 0 or len(items) <= total_cap:
        return items

    negatives = [item for item in items if safe_text(item.get("source")) == "review" and safe_text(item.get("sentiment")) == "negative"]
    negatives = negatives[:neg_floor]
    selected_ids = {safe_text(item.get("evidence_id")) for item in negatives}
    packed: list[dict[str, Any]] = list(negatives)
    tip_count = sum(1 for item in packed if safe_text(item.get("source")) == "tip")

    for item in items:
        evidence_id = safe_text(item.get("evidence_id"))
        if evidence_id and evidence_id in selected_ids:
            continue
        if len(packed) >= total_cap:
            break
        if safe_text(item.get("source")) == "tip" and tip_count >= tip_cap:
            continue
        packed.append(item)
        if evidence_id:
            selected_ids.add(evidence_id)
        if safe_text(item.get("source")) == "tip":
            tip_count += 1
    return packed


def aggregate_review_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    sub = evidence.copy()
    sub["sentence_text"] = sub["sentence_text"].map(safe_text)
    sub = sub.loc[sub["sentence_text"].map(is_usable_text)].copy()
    sub["abs_weight"] = sub["final_weight"].abs()
    sub = sub.sort_values(
        ["user_id", "abs_weight", "days_since_review", "sentence_rank"],
        ascending=[True, False, True, True],
    )

    rows: list[dict[str, Any]] = []
    for user_id, grp in sub.groupby("user_id", sort=False):
        payload: list[dict[str, Any]] = []
        merged_by_text: dict[str, dict[str, Any]] = {}
        for item in grp.to_dict(orient="records"):
            text = safe_text(item.get("sentence_text"))
            if not text:
                continue
            bucket = merged_by_text.get(text)
            tag = safe_text(item.get("tag"))
            tag_type = safe_text(item.get("tag_type"))
            if bucket is None:
                polarity = float(item.get("polarity", 0) or 0)
                bucket = {
                    "evidence_id": "",
                    "source": "review",
                    "sentiment": "positive" if polarity > 0 else ("negative" if polarity < 0 else "neutral"),
                    "weight": round(float(item.get("abs_weight", 0.0) or 0.0), 4),
                    "time_bucket": (
                        "recent"
                        if float(item.get("days_since_review", 99999) or 99999) <= 180
                        else "older"
                    ),
                    "tag": tag,
                    "tags": [],
                    "tag_types": [],
                    "text": text,
                }
                merged_by_text[text] = bucket
                payload.append(bucket)
            if tag and tag not in bucket["tags"]:
                bucket["tags"].append(tag)
            if tag_type and tag_type not in bucket["tag_types"]:
                bucket["tag_types"].append(tag_type)
        for idx, bucket in enumerate(payload, start=1):
            bucket["evidence_id"] = f"rev_{idx}"
            bucket["tag_count"] = int(len(bucket["tags"]))
        payload = maybe_cap_items(payload, V3_REVIEW_MAX_ITEMS)
        rows.append(
            {
                "user_id": str(user_id),
                "review_evidence_stream_json": json.dumps(payload, ensure_ascii=False),
                "review_evidence_count_v3": int(len(payload)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_tip_evidence(tips: pd.DataFrame) -> pd.DataFrame:
    sub = tips.copy()
    sub["text"] = sub["text"].map(safe_text)
    sub = sub.loc[sub["text"].map(is_usable_text)].copy()
    sub = sub.sort_values(
        ["user_id", "tip_weight_v1", "tip_age_days", "text_len_chars"],
        ascending=[True, False, True, False],
    )

    rows: list[dict[str, Any]] = []
    for user_id, grp in sub.groupby("user_id", sort=False):
        seen: set[str] = set()
        payload: list[dict[str, Any]] = []
        for item in grp.to_dict(orient="records"):
            text = safe_text(item.get("text"))
            if not text or text in seen:
                continue
            seen.add(text)
            payload.append(
                {
                    "evidence_id": f"tip_{len(payload)+1}",
                    "source": "tip",
                    "sentiment": "unknown",
                    "weight": round(float(item.get("tip_weight_v1", 0.0) or 0.0), 4),
                    "time_bucket": (
                        "recent"
                        if float(item.get("tip_age_days", 99999) or 99999) <= 180
                        else "older"
                    ),
                    "recommend_cue": int(float(item.get("has_recommend_cue", 0.0) or 0.0)),
                    "time_cue": int(float(item.get("has_time_cue", 0.0) or 0.0)),
                    "dish_cue": int(float(item.get("has_dish_cue", 0.0) or 0.0)),
                    "text": text,
                }
            )
        payload = maybe_cap_items(payload, V3_TIP_MAX_ITEMS)
        rows.append(
            {
                "user_id": str(user_id),
                "tip_evidence_stream_json": json.dumps(payload, ensure_ascii=False),
                "tip_evidence_count_v3": int(len(payload)),
            }
        )
    return pd.DataFrame(rows)


def format_evidence_stream(payload: list[dict[str, Any]]) -> str:
    lines = ["User evidence stream:"]
    if not payload:
        lines.append("- none")
        return "\n".join(lines)
    for item in payload:
        meta = [
            f"id={safe_text(item.get('evidence_id')) or 'unknown'}",
            f"source={safe_text(item.get('source')) or 'unknown'}",
            f"sentiment={safe_text(item.get('sentiment')) or 'unknown'}",
            f"time_bucket={safe_text(item.get('time_bucket')) or 'unknown'}",
            f"weight={safe_text(item.get('weight')) or '0'}",
        ]
        tags = safe_list(item.get("tags"))
        if tags:
            meta.append(f"tags={','.join(tags)}")
        elif safe_text(item.get("tag")):
            meta.append(f"tag={safe_text(item.get('tag'))}")
        if item.get("source") == "tip":
            meta.append(f"recommend_cue={safe_text(item.get('recommend_cue')) or '0'}")
            meta.append(f"time_cue={safe_text(item.get('time_cue')) or '0'}")
            meta.append(f"dish_cue={safe_text(item.get('dish_cue')) or '0'}")
        lines.append(f"- {'; '.join(meta)}; text={safe_text(item.get('text'))}")
    return "\n".join(lines)


def format_evidence_block(prefix: str, values: list[str]) -> str:
    if not values:
        return f"{prefix}\n- none"
    lines = [prefix]
    for idx, value in enumerate(values, start=1):
        lines.append(f"- {prefix.split()[0].lower()}_{idx}: {value}")
    return "\n".join(lines)


def wrap_target_schema_v3(raw_json: str) -> str:
    payload = json.loads(raw_json)
    evidence_refs = payload.get("evidence_refs", {}) if isinstance(payload.get("evidence_refs", {}), dict) else {}
    ambivalent = payload.get("ambivalent_signals", {}) if isinstance(payload.get("ambivalent_signals", {}), dict) else {}
    open_discoveries = safe_discovery_list(payload.get("open_discoveries"))
    wrapped = {
        "core_profile": {
            "stable_preferences": payload.get("stable_preferences", {}),
            "avoid_signals": payload.get("avoid_signals", {}),
            "recent_preferences": payload.get("recent_preferences", {}),
            "behavior_summary": payload.get("behavior_summary", {}),
        },
        "open_discoveries": open_discoveries,
        "evidence_refs": {
            "positive_event_refs": safe_list(evidence_refs.get("positive_event_refs")),
            "negative_event_refs": safe_list(evidence_refs.get("negative_event_refs")),
            "positive_evidence_refs": safe_list(evidence_refs.get("positive_sentence_refs")),
            "negative_evidence_refs": safe_list(evidence_refs.get("negative_sentence_refs")),
        },
        "confidence": payload.get("confidence", {"overall": "unknown"}),
        "ambivalent_signals": {
            "cuisines": safe_list(ambivalent.get("cuisines")),
            "scenes": safe_list(ambivalent.get("scenes")),
        },
    }
    return json.dumps(wrapped, ensure_ascii=False)


def format_event_block(title: str, payload: list[dict[str, Any]]) -> str:
    lines = [title]
    if not payload:
        lines.append("- none")
        return "\n".join(lines)
    for idx, item in enumerate(payload, start=1):
        rank = safe_text(item.get("recent_rank")) or str(idx)
        event_type = safe_text(item.get("event_type")) or "unknown"
        cuisine = safe_text(item.get("merchant_primary_cuisine")) or "unknown"
        city = safe_text(item.get("city")) or "unknown"
        meal = ", ".join(safe_list(item.get("persona_meal_tags"))) or "unknown"
        scene = ", ".join(safe_list(item.get("persona_scene_tags"))) or "unknown"
        quality = safe_text(item.get("persona_quality_band")) or "unknown"
        reliability = safe_text(item.get("persona_reliability_band")) or "unknown"
        core = safe_text(item.get("merchant_core_text_short"))
        lines.append(
            f"- event_{rank}: type={event_type}; cuisine={cuisine}; city={city}; "
            f"meal={meal}; scene={scene}; quality={quality}; reliability={reliability}"
        )
        if core:
            lines.append(f"  text: {core}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    if PROMPT_VERSION == "v3":
        return (
            "You are a user preference extraction model for Louisiana restaurant recommendation on Yelp. "
            "Infer a structured user profile from training-period evidence only. "
            "Use cleaned but minimally processed evidence, not hidden labels. "
            "Return exactly one valid JSON object and nothing else. "
            "Start with '{' and end with '}'. "
            "Do not echo the prompt, do not repeat the JSON, and do not add prose outside the JSON. "
            "If evidence is insufficient, use empty arrays or \"unknown\". "
            "Only use open_discoveries when the evidence supports them."
        )
    return (
        "You are a user preference extraction model for Louisiana restaurant recommendation on Yelp. "
        "You must infer stable preferences, avoid signals, recent preferences, and behavior summary "
        "only from the provided training-period evidence. "
        "Return exactly one valid JSON object and nothing else. "
        "Start with '{' and end with '}'. "
        "Do not echo the prompt, do not repeat the JSON, and do not write natural-language explanations. "
        "If evidence is insufficient, use empty arrays or \"unknown\"."
    )


def build_prompt(row: dict[str, Any]) -> str:
    if PROMPT_VERSION == "v3":
        return build_prompt_v3(row)
    sequence_events = load_json_list(row.get("recent_event_sequence_json"))
    positive_anchor_events = load_json_list(row.get("positive_anchor_sequence_json"))
    negative_anchor_events = load_json_list(row.get("negative_anchor_sequence_json"))
    positive_sentences = safe_list(row.get("positive_evidence_sentences"))
    negative_sentences = safe_list(row.get("negative_evidence_sentences"))
    user_evidence_text = clip_text(row.get("profile_text") or row.get("profile_text_short"), PROFILE_TEXT_MAX_CHARS)

    stats_lines = [
        f"- training_interactions_max: {int(row.get('n_train_max', 0) or 0)}",
        f"- training_events_total: {int(row.get('n_events_trainable', 0) or 0)}",
        f"- distinct_businesses: {int(row.get('n_businesses_trainable', 0) or 0)}",
        f"- recent_sequence_count: {int(row.get('sequence_event_count', 0) or 0)}",
        f"- repeat_business_ratio: {round(float(row.get('repeat_business_ratio', 0.0) or 0.0), 4)}",
        f"- sequence_span_days: {int(row.get('sequence_span_days', 0) or 0)}",
    ]

    blocks = [
        "Task: extract a structured user preference profile from training-period evidence only.",
        "Output requirements:",
        "- valid JSON only",
        "- output exactly one JSON object",
        "- start with '{' and end with '}'",
        "- use the exact schema shown below",
        "- if evidence is weak, output empty arrays or \"unknown\"",
        "- do not invent unsupported cuisines, scenes, properties, or behavior styles",
        "- do not copy event lines or evidence text into the output",
        "- use ambivalent_signals when both positive and negative evidence exist for the same cuisine",
        "",
        "Required JSON schema:",
        json.dumps(
            {
                "stable_preferences": {
                    "preferred_cuisines": [],
                    "preferred_meals": [],
                    "preferred_scenes": [],
                    "preferred_properties": [],
                    "top_city": "",
                    "geo_style": "hyperlocal|metro|mixed|unknown",
                },
                "avoid_signals": {
                    "avoided_cuisines": [],
                    "avoided_scenes": [],
                    "service_risk_sensitivity": "low|medium|high|unknown",
                },
                "recent_preferences": {
                    "recent_focus_cuisines": [],
                    "recent_focus_meals": [],
                    "recent_focus_scenes": [],
                    "recent_focus_properties": [],
                    "recent_shift": "stable|switching|broadening|unknown",
                },
                "behavior_summary": {
                    "social_mode": [],
                    "time_mode": [],
                    "novelty_style": "low|medium|high|unknown",
                },
                "evidence_refs": {
                    "positive_event_refs": [],
                    "negative_event_refs": [],
                    "positive_sentence_refs": [],
                    "negative_sentence_refs": [],
                },
                "confidence": {"overall": "low|medium|high"},
                "ambivalent_signals": {"cuisines": []},
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "User review and tip evidence digest (cleaned training-period text):",
        user_evidence_text or "unknown",
        "",
        "Behavior stats:",
        "\n".join(stats_lines),
        "",
        format_evidence_block("Positive evidence sentences:", positive_sentences),
        "",
        format_evidence_block("Negative evidence sentences:", negative_sentences),
        "",
        format_event_block("Recent event sequence:", sequence_events),
        "",
        format_event_block("Positive anchor events:", positive_anchor_events),
        "",
        format_event_block("Negative anchor events:", negative_anchor_events),
    ]
    if INCLUDE_DERIVED_NARRATIVE:
        blocks.extend(
            [
                "",
                "Derived preference hints (use only as weak context, not as direct answer):",
                f"- long_term_hint: {safe_text(row.get('user_long_pref_text')) or 'unknown'}",
                f"- recent_hint: {safe_text(row.get('user_recent_intent_text')) or 'unknown'}",
                f"- avoid_hint: {safe_text(row.get('user_negative_avoid_text')) or 'unknown'}",
                f"- context_hint: {safe_text(row.get('user_context_text')) or 'unknown'}",
            ]
        )
    blocks.extend(
        [
            "",
            "Return the JSON object now.",
        ]
    )
    return "\n".join(blocks)


def build_prompt_v3(row: dict[str, Any]) -> str:
    sequence_events = load_json_list(row.get("recent_event_sequence_json"))
    positive_anchor_events = load_json_list(row.get("positive_anchor_sequence_json"))
    negative_anchor_events = load_json_list(row.get("negative_anchor_sequence_json"))
    review_items = load_json_list(row.get("review_evidence_stream_json"))
    tip_items = load_json_list(row.get("tip_evidence_stream_json"))
    unified_stream = review_items + tip_items
    if V3_APPLY_PACKING:
        unified_stream = pack_unified_evidence_items(unified_stream, row.get("n_train_max"))
    profile_digest = clip_text(row.get("profile_text") or row.get("profile_text_short"), PROFILE_TEXT_MAX_CHARS)

    stats_lines = [
        f"- training_interactions_max: {safe_int(row.get('n_train_max'))}",
        f"- training_events_total: {safe_int(row.get('n_events_trainable'))}",
        f"- distinct_businesses: {safe_int(row.get('n_businesses_trainable'))}",
        f"- recent_sequence_count: {safe_int(row.get('sequence_event_count'))}",
        f"- repeat_business_ratio: {round(float(row.get('repeat_business_ratio', 0.0) or 0.0), 4)}",
        f"- sequence_span_days: {safe_int(row.get('sequence_span_days'))}",
        f"- density_band_visible: {density_band(row.get('n_train_max'))}",
        f"- review_evidence_items: {safe_int(row.get('review_evidence_count_v3'))}",
        f"- tip_evidence_items: {safe_int(row.get('tip_evidence_count_v3'))}",
        f"- packed_evidence_items: {len(unified_stream)}",
    ]

    blocks = [
        "Task: infer a structured user preference profile from training-period evidence only.",
        "",
        "Output rules:",
        "- return exactly one valid JSON object",
        "- start with '{' and end with '}'",
        "- do not repeat the JSON",
        "- do not echo the prompt",
        "- if evidence is weak, use empty arrays or \"unknown\"",
        "- do not invent unsupported cuisines, scenes, properties, or behavior styles",
        "- every open discovery must include evidence_refs and confidence",
        "- use open_discoveries only when there is enough evidence",
        "",
        "Required output schema:",
        json.dumps(
            {
                "core_profile": {
                    "stable_preferences": {
                        "preferred_cuisines": [],
                        "preferred_meals": [],
                        "preferred_scenes": [],
                        "preferred_properties": [],
                        "top_city": "",
                        "geo_style": "hyperlocal|metro|mixed|unknown",
                    },
                    "avoid_signals": {
                        "avoided_cuisines": [],
                        "avoided_scenes": [],
                        "service_risk_sensitivity": "low|medium|high|unknown",
                    },
                    "recent_preferences": {
                        "recent_focus_cuisines": [],
                        "recent_focus_meals": [],
                        "recent_focus_scenes": [],
                        "recent_focus_properties": [],
                        "recent_shift": "stable|switching|broadening|unknown",
                    },
                    "behavior_summary": {
                        "social_mode": [],
                        "time_mode": [],
                        "novelty_style": "low|medium|high|unknown",
                    },
                },
                "open_discoveries": [
                    {
                        "type": "context_dependent_preference|tolerance_hypothesis|scenario_specific_preference|latent_preference|preference_conflict|uncertainty_note",
                        "claim": "",
                        "confidence": "low|medium|high",
                        "evidence_refs": [],
                        "reason_short": "",
                    }
                ],
                "evidence_refs": {
                    "positive_event_refs": [],
                    "negative_event_refs": [],
                    "positive_evidence_refs": [],
                    "negative_evidence_refs": [],
                },
                "confidence": {"overall": "low|medium|high"},
                "ambivalent_signals": {"cuisines": [], "scenes": []},
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        format_evidence_stream(unified_stream),
        "",
        "Behavior stats:",
        "\n".join(stats_lines),
    ]

    if V3_INCLUDE_PROFILE_DIGEST and len(unified_stream) < V3_PROFILE_DIGEST_MIN_ITEMS and profile_digest:
        blocks.extend(
            [
                "",
                "Supplemental cleaned archive digest:",
                profile_digest,
            ]
        )

    blocks.extend(
        [
            "",
            format_event_block("Recent event sequence:", sequence_events),
            "",
            format_event_block("Positive anchor events:", positive_anchor_events),
            "",
            format_event_block("Negative anchor events:", negative_anchor_events),
            "",
            "Return the JSON object now.",
        ]
    )
    return "\n".join(blocks)


def read_eval_users(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    for col in ("user_id", "userid", "userId"):
        if col in df.columns:
            return {safe_text(value) for value in df[col].tolist() if safe_text(value)}
    if len(df.columns) == 1:
        return {safe_text(value) for value in df.iloc[:, 0].tolist() if safe_text(value)}
    return set()


def split_users(user_ids: list[str]) -> tuple[set[str], set[str], str]:
    cohort_users = read_eval_users(EVAL_USER_COHORT_PATH)
    valid_cohort = cohort_users.intersection(set(user_ids))
    if valid_cohort:
        eval_users = valid_cohort
        train_users = set(user_ids) - eval_users
        return train_users, eval_users, "fixed_eval_cohort"
    rng = random.Random(SEED)
    shuffled = list(user_ids)
    rng.shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * EVAL_USER_FRAC))
    eval_users = set(shuffled[:eval_count])
    train_users = set(shuffled[eval_count:])
    return train_users, eval_users, "random_user_split"


def maybe_cap(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.head(max_rows).copy()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def main() -> None:
    user_pref_run = resolve_run(
        USER_PREF_RUN_DIR,
        USER_PREF_ROOT,
        "_full_stage09_user_preference_schema_v1_build",
    )
    user_profiles_run = resolve_run(
        USER_PROFILES_RUN_DIR,
        USER_PROFILES_ROOT,
        "_full_stage09_user_profile_build",
    )
    tip_signal_run = resolve_run(
        TIP_SIGNAL_RUN_DIR,
        TIP_SIGNAL_ROOT,
        "_full_stage09_tip_signal_build",
    )
    sequence_run = resolve_run(
        SEQUENCE_RUN_DIR,
        SEQUENCE_ROOT,
        "_full_stage09_sequence_view_schema_v1_build",
    )
    target_run = resolve_run(
        TARGET_RUN_DIR,
        TARGET_ROOT,
        "_full_stage09_user_preference_target_schema_v1_build",
    )

    user_pref = pd.read_parquet(user_pref_run / "user_preference_schema_v1.parquet")
    user_profile_evidence = pd.read_csv(user_profiles_run / "user_profile_evidence.csv")
    tip_signal_weights = pd.read_parquet(tip_signal_run / "tip_signal_weights.parquet")
    sequence = pd.read_parquet(sequence_run / "sequence_view_schema_v1.parquet")
    target = pd.read_parquet(target_run / "user_preference_target_schema_v1.parquet")

    for df in (user_pref, user_profile_evidence, tip_signal_weights, sequence, target):
        df["user_id"] = df["user_id"].astype(str)

    review_stream = aggregate_review_evidence(user_profile_evidence)
    tip_stream = aggregate_tip_evidence(tip_signal_weights)

    merged = (
        target.loc[target["sft_ready_v1"].fillna(False).eq(True)]
        .merge(user_pref, on="user_id", how="left", suffixes=("_target", ""))
        .merge(review_stream, on="user_id", how="left")
        .merge(tip_stream, on="user_id", how="left")
        .merge(sequence, on="user_id", how="left", suffixes=("", "_sequence"))
    )
    merged = merged.sort_values("user_id").reset_index(drop=True)

    system_prompt = build_system_prompt()
    records: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        prompt_text = build_prompt(row)
        target_text = safe_text(row.get("target_schema_v1_json"))
        if not target_text:
            continue
        if PROMPT_VERSION == "v3":
            target_text = wrap_target_schema_v3(target_text)
        records.append(
            {
                "user_id": row["user_id"],
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "prompt_chars": int(len(prompt_text)),
                "target_chars": int(len(target_text)),
                "sequence_event_count": safe_int(row.get("sequence_event_count")),
                "n_train_max": safe_int(row.get("n_train_max")),
                "density_band": density_band(row.get("n_train_max")),
                "review_evidence_count_v3": safe_int(row.get("review_evidence_count_v3")),
                "tip_evidence_count_v3": safe_int(row.get("tip_evidence_count_v3")),
                "unified_evidence_count_v3": safe_int(row.get("review_evidence_count_v3"))
                + safe_int(row.get("tip_evidence_count_v3")),
                "packing_enabled_v3": bool(V3_APPLY_PACKING),
                "confidence_overall": safe_text(row.get("confidence_overall")),
            }
        )

    dataset = pd.DataFrame(records)
    if dataset.empty:
        raise RuntimeError("no SFT-ready rows found")

    train_users, eval_users, split_strategy = split_users(dataset["user_id"].tolist())
    dataset["split"] = dataset["user_id"].apply(lambda user_id: "eval" if user_id in eval_users else "train")

    train_df = dataset.loc[dataset["split"].eq("train")].copy()
    eval_df = dataset.loc[dataset["split"].eq("eval")].copy()
    train_df = maybe_cap(train_df, MAX_TRAIN_ROWS)
    eval_df = maybe_cap(eval_df, MAX_EVAL_ROWS)

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(run_dir / "train.parquet", index=False)
    eval_df.to_parquet(run_dir / "eval.parquet", index=False)
    write_jsonl(run_dir / "train.jsonl", train_df.to_dict(orient="records"))
    write_jsonl(run_dir / "eval.jsonl", eval_df.to_dict(orient="records"))

    sample_payload = {
        "train": train_df.head(SAMPLE_ROWS).to_dict(orient="records"),
        "eval": eval_df.head(min(SAMPLE_ROWS, len(eval_df))).to_dict(orient="records"),
    }
    safe_json_write(run_dir / "sample.json", sample_payload)

    summary = {
        "prompt_version": PROMPT_VERSION,
        "include_derived_narrative": bool(INCLUDE_DERIVED_NARRATIVE),
        "include_profile_digest_v3": bool(V3_INCLUDE_PROFILE_DIGEST),
        "apply_packing_v3": bool(V3_APPLY_PACKING),
        "split_strategy": split_strategy,
        "rows_total": int(len(dataset)),
        "rows_train": int(len(train_df)),
        "rows_eval": int(len(eval_df)),
        "users_train": int(train_df["user_id"].nunique()),
        "users_eval": int(eval_df["user_id"].nunique()),
        "mean_prompt_chars_train": float(train_df["prompt_chars"].mean()) if len(train_df) else 0.0,
        "mean_prompt_chars_eval": float(eval_df["prompt_chars"].mean()) if len(eval_df) else 0.0,
        "p95_prompt_chars_train": float(train_df["prompt_chars"].quantile(0.95)) if len(train_df) else 0.0,
        "p95_prompt_chars_eval": float(eval_df["prompt_chars"].quantile(0.95)) if len(eval_df) else 0.0,
        "mean_target_chars_train": float(train_df["target_chars"].mean()) if len(train_df) else 0.0,
        "mean_target_chars_eval": float(eval_df["target_chars"].mean()) if len(eval_df) else 0.0,
        "sequence_event_mean_train": float(train_df["sequence_event_count"].mean()) if len(train_df) else 0.0,
        "sequence_event_mean_eval": float(eval_df["sequence_event_count"].mean()) if len(eval_df) else 0.0,
        "review_evidence_mean_train": float(train_df["review_evidence_count_v3"].mean()) if len(train_df) else 0.0,
        "tip_evidence_mean_train": float(train_df["tip_evidence_count_v3"].mean()) if len(train_df) else 0.0,
        "unified_evidence_mean_train": float(train_df["unified_evidence_count_v3"].mean()) if len(train_df) else 0.0,
        "review_evidence_mean_eval": float(eval_df["review_evidence_count_v3"].mean()) if len(eval_df) else 0.0,
        "tip_evidence_mean_eval": float(eval_df["tip_evidence_count_v3"].mean()) if len(eval_df) else 0.0,
        "unified_evidence_mean_eval": float(eval_df["unified_evidence_count_v3"].mean()) if len(eval_df) else 0.0,
    }
    dataset["has_user_evidence_stream"] = dataset["prompt_text"].str.contains("User evidence stream:", regex=False)
    dataset["has_user_evidence_digest"] = dataset["prompt_text"].str.contains("User review and tip evidence digest", regex=False)
    dataset["has_profile_digest_v3"] = dataset["prompt_text"].str.contains("Supplemental cleaned archive digest:", regex=False)
    dataset["has_recent_sequence"] = dataset["prompt_text"].str.contains("Recent event sequence:", regex=False)
    dataset["has_positive_anchor_events"] = dataset["prompt_text"].str.contains("Positive anchor events:", regex=False)
    dataset["has_negative_anchor_events"] = dataset["prompt_text"].str.contains("Negative anchor events:", regex=False)
    dataset["has_derived_narrative"] = dataset["prompt_text"].str.contains("Derived preference hints", regex=False)
    audit = {
        "prompt_version": PROMPT_VERSION,
        "include_derived_narrative": bool(INCLUDE_DERIVED_NARRATIVE),
        "include_profile_digest_v3": bool(V3_INCLUDE_PROFILE_DIGEST),
        "apply_packing_v3": bool(V3_APPLY_PACKING),
        "rows_total": int(len(dataset)),
        "has_user_evidence_stream_rate": rate(dataset["has_user_evidence_stream"]),
        "has_user_evidence_digest_rate": rate(dataset["has_user_evidence_digest"]),
        "has_profile_digest_v3_rate": rate(dataset["has_profile_digest_v3"]),
        "has_recent_sequence_rate": rate(dataset["has_recent_sequence"]),
        "has_positive_anchor_events_rate": rate(dataset["has_positive_anchor_events"]),
        "has_negative_anchor_events_rate": rate(dataset["has_negative_anchor_events"]),
        "has_derived_narrative_rate": rate(dataset["has_derived_narrative"]),
        "prompt_chars_mean": float(dataset["prompt_chars"].mean()),
        "prompt_chars_p95": float(dataset["prompt_chars"].quantile(0.95)),
        "prompt_chars_max": int(dataset["prompt_chars"].max()),
        "review_evidence_count_mean": float(dataset["review_evidence_count_v3"].mean()),
        "tip_evidence_count_mean": float(dataset["tip_evidence_count_v3"].mean()),
        "unified_evidence_count_mean": float(dataset["unified_evidence_count_v3"].mean()),
    }
    density_summary = []
    for band, sub in dataset.groupby("density_band", sort=True):
        density_summary.append(
            {
                "density_band": band,
                "rows": int(len(sub)),
                "mean_prompt_chars": float(sub["prompt_chars"].mean()),
                "p95_prompt_chars": float(sub["prompt_chars"].quantile(0.95)),
                "mean_sequence_event_count": float(sub["sequence_event_count"].mean()),
                "mean_review_evidence_count": float(sub["review_evidence_count_v3"].mean()),
                "mean_tip_evidence_count": float(sub["tip_evidence_count_v3"].mean()),
                "mean_unified_evidence_count": float(sub["unified_evidence_count_v3"].mean()),
            }
        )
    audit["density_summary"] = density_summary
    audit["audit_pass_v3"] = bool(
        audit["has_user_evidence_stream_rate"] >= 0.99
        and audit["has_recent_sequence_rate"] >= 0.99
        and audit["has_positive_anchor_events_rate"] >= 0.99
        and audit["has_negative_anchor_events_rate"] >= 0.99
        and audit["has_derived_narrative_rate"] == 0.0
        and min(
            (item["mean_unified_evidence_count"] for item in density_summary if item["density_band"] != "unknown"),
            default=0.0,
        )
        >= 8.0
    )
    safe_json_write(run_dir / "summary.json", summary)
    safe_json_write(run_dir / "audit.json", audit)

    run_meta = {
        "run_tag": RUN_TAG,
        "user_pref_run_dir": str(user_pref_run),
        "user_profiles_run_dir": str(user_profiles_run),
        "tip_signal_run_dir": str(tip_signal_run),
        "sequence_run_dir": str(sequence_run),
        "target_run_dir": str(target_run),
        "output_run_dir": str(run_dir),
        "summary": summary,
        "audit": audit,
    }
    safe_json_write(run_dir / "run_meta.json", run_meta)
    write_latest_run_pointer("11_persona_sft_data_latest", run_dir, {"run_tag": RUN_TAG, "summary": summary})
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
