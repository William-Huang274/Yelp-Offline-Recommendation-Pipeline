from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, project_path, write_latest_run_pointer


RUN_TAG = "stage12_semantic_match_assets_build"
DEFAULT_USER_PARSED_ROOT = env_or_project_path("INPUT_11_PROMPT_ONLY_QUEUE_INFER_ROOT_DIR", "data/output/11_prompt_only_queue_infer")
DEFAULT_MERCHANT_PARSED_ROOT = env_or_project_path("INPUT_12_MERCHANT_PROFILE_INFER_ROOT_DIR", "data/output/12_merchant_profile_infer")
DEFAULT_OUTPUT_ROOT = env_or_project_path("OUTPUT_12_SEMANTIC_MATCH_ASSETS_ROOT_DIR", "data/output/12_semantic_match_assets")

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.66, "low": 0.33}

FAMILY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "dish",
        (
            "gumbo",
            "oyster",
            "shrimp",
            "crawfish",
            "po-boy",
            "poboy",
            "burger",
            "sandwich",
            "taco",
            "pizza",
            "sushi",
            "dessert",
            "beignet",
            "coffee",
            "brunch",
            "chicken",
            "fish",
            "seafood",
            "bbq",
            "barbecue",
            "steak",
            "salad",
            "pasta",
            "cake",
            "pie",
            "rice noodles",
            "noodles",
            "biscuit",
            "sauce",
            "sauces",
            "flavor",
            "flavors",
            "food",
            "meal",
            "undercooked",
            "chewy",
            "bland",
            "cold",
            "warm",
            "lukewarm",
            "dry",
        ),
    ),
    (
        "cuisine",
        (
            "cajun",
            "creole",
            "mexican",
            "italian",
            "french",
            "american",
            "vietnamese",
            "thai",
            "chinese",
            "japanese",
            "seafood",
            "breakfast",
            "bakery",
            "barbecue",
            "bbq",
        ),
    ),
    (
        "scene",
        (
            "date",
            "family",
            "group",
            "celebration",
            "occasion",
            "fine-dining",
            "fine dining",
            "quick bite",
            "sit-down",
            "sit down",
            "nightlife",
            "brunch spot",
            "tourist",
            "local",
            "relaxed",
            "walking tour",
            "destination wedding",
            "snack",
            "food hall",
            "food halls",
            "market",
            "markets",
            "bucket list",
            "culturally significant",
            "conference",
            "solo traveler",
            "memorable",
        ),
    ),
    (
        "ambience",
        (
            "ambiance",
            "ambience",
            "atmosphere",
            "quiet",
            "loud",
            "patio",
            "outdoor",
            "view",
            "historic",
            "classic",
            "old-school",
            "casual",
            "upscale",
            "clean",
            "crowded",
            "courtyard",
            "seating",
            "dining room",
            "room",
            "bathroom",
            "bathrooms",
            "bed",
            "beds",
            "ceiling",
            "lighting",
            "cramped",
            "themed",
            "music",
            "jazz",
        ),
    ),
    (
        "service_ops",
        (
            "service",
            "staff",
            "server",
            "attentive",
            "friendly",
            "reservation",
            "takeout",
            "delivery",
            "parking",
            "check-in",
            "wait",
            "line",
            "hours",
            "late",
            "fast",
            "slow",
            "elevator",
            "water",
            "wifi",
            "valet",
        ),
    ),
    (
        "price_value",
        (
            "price",
            "value",
            "cheap",
            "expensive",
            "overpriced",
            "portion",
            "large portions",
            "good value",
        ),
    ),
    (
        "time_hours",
        (
            "breakfast",
            "brunch",
            "lunch",
            "dinner",
            "late night",
            "late-night",
            "weekend",
            "happy hour",
        ),
    ),
    ("beverage_bar", ("bar", "cocktail", "wine", "beer", "drink", "full bar", "jazz brunch")),
    (
        "geo",
        (
            "neighborhood",
            "quarter",
            "french quarter",
            "downtown",
            "metairie",
            "new orleans",
            "canal street",
            "bourbon street",
            "local",
            "tourist",
        ),
    ),
    ("reliability", ("consistent", "inconsistent", "reliable", "quality", "fresh", "well-done", "balanced")),
    (
        "risk",
        (
            "avoid",
            "rude",
            "dirty",
            "overpowering",
            "disappointing",
            "bad",
            "worst",
            "roach",
            "limited",
        ),
    ),
]

ATTRIBUTE_TYPE_TO_FAMILY = {
    "dish": "dish",
    "service": "service_ops",
    "ambience": "ambience",
    "scene": "scene",
    "price_value": "price_value",
    "operations": "service_ops",
    "other": "other",
}

USER_SECTION_TO_ROLE = {
    "stable_preferences": "user_positive_preference",
    "avoid_signals": "user_avoidance",
    "recent_signals": "user_recent_intent",
    "context_rules": "user_context_constraint",
    "state_hypotheses": "user_latent_preference",
    "discriminative_signals": "user_discriminative_signal",
    "contextual_inference_signals": "user_contextual_need",
}

MERCHANT_SECTION_TO_ROLE = {
    "direct_evidence_signals": "merchant_strength",
    "fine_grained_attributes": "merchant_attribute",
    "usage_scenes": "merchant_usage_scene",
    "risk_or_avoidance_notes": "merchant_risk",
    "cautious_inferences": "merchant_cautious_inference",
}

SECTION_TEXT_FIELD = {
    "direct_evidence_signals": "signal",
    "fine_grained_attributes": "attribute",
    "usage_scenes": "scene",
    "risk_or_avoidance_notes": "risk",
    "cautious_inferences": "inference",
}

MATCH_CHANNEL_BY_FAMILY = {
    "cuisine": "cuisine",
    "dish": "dish_attr",
    "scene": "scene_context",
    "ambience": "scene_context",
    "service_ops": "service_ops",
    "price_value": "price_value",
    "time_hours": "time_hours",
    "beverage_bar": "beverage_bar",
    "geo": "geo_context",
    "risk": "risk_conflict",
    "reliability": "service_ops",
    "other": "other",
}


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_full_{RUN_TAG}"


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9&/+\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_ref_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for item in value:
        ref = normalize_text(item)
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs


def normalize_snippets(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    snippets: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = normalize_text(item)
        if text and text not in seen:
            snippets.append(text)
            seen.add(text)
    return snippets


def item_hash(*parts: Any) -> str:
    raw = "||".join(normalize_text(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def discover_latest_file(root: Path, file_name: str) -> Path:
    root = normalize_legacy_project_path(root)
    if not root.exists():
        raise FileNotFoundError(f"input root not found: {root}")
    candidates = [path for path in root.rglob(file_name) if "checkpoint" not in path.as_posix().lower()]
    if not candidates:
        raise FileNotFoundError(f"no {file_name} found under {root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_optional_path(raw: str, default_root: Path, file_name: str) -> Path:
    if raw:
        path = normalize_legacy_project_path(raw)
        if path.is_dir():
            path = path / file_name
        if not path.exists():
            raise FileNotFoundError(f"input path not found: {path}")
        return path
    return discover_latest_file(default_root, file_name)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    path = normalize_legacy_project_path(path)
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list JSON records: {path}")
    return [row for row in payload if isinstance(row, dict)]


def load_universe_ids(raw_path: str, id_col: str) -> set[str]:
    if not raw_path:
        return set()
    path = normalize_legacy_project_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"universe path not found: {path}")
    suffix = path.suffix.lower()
    if path.is_dir() or suffix == ".parquet":
        df = pd.read_parquet(path, columns=[id_col])
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=[id_col])
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
        if id_col not in df.columns:
            raise ValueError(f"{id_col} not found in universe path: {path}")
        df = df[[id_col]]
    else:
        raise ValueError(f"unsupported universe path suffix: {path}")
    return {normalize_text(value) for value in df[id_col].dropna().astype(str).tolist() if normalize_text(value)}


def filter_records_by_universe(records: list[dict[str, Any]], id_col: str, universe_ids: set[str]) -> list[dict[str, Any]]:
    if not universe_ids:
        return records
    return [row for row in records if normalize_text(row.get(id_col)) in universe_ids]


def choose_parsed_json(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("parsed_json", "parsed_json_repaired", "parsed_json_candidate"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return None


def infer_family(text: str, *, source_section: str, item: dict[str, Any], entity_type: str) -> str:
    if entity_type == "merchant" and source_section == "fine_grained_attributes":
        attr_type = canonical_text(item.get("attribute_type"))
        attr_family = ATTRIBUTE_TYPE_TO_FAMILY.get(attr_type, "other")
        # The model sometimes assigns broad attribute_type values. Keep high-signal
        # beverage/price/geo cues available for route-specific ANN indexes.
        if attr_family in {"service_ops", "other"}:
            text_family = infer_family(text, source_section="", item={}, entity_type=entity_type)
            if text_family in {"beverage_bar", "price_value", "geo", "time_hours"}:
                return text_family
        return attr_family
    if entity_type == "user" and source_section == "contextual_inference_signals":
        axis = canonical_text(item.get("canonical_axis"))
        if "price" in axis or "value" in axis:
            return "price_value"
        if "service" in axis:
            return "service_ops"
        if "family" in axis or "group" in axis or "occasion" in axis or "scene" in axis:
            return "scene"
        if "crowding" in axis:
            return "ambience"
        if "geography" in axis:
            return "geo"
        if "beverage" in axis or "bar" in axis:
            return "beverage_bar"
        if "late_night" in axis or "hours" in axis:
            return "time_hours"
        if "cuisine" in axis:
            return "cuisine"
    canon = canonical_text(text)
    if "french quarter" in canon or "bourbon street" in canon or "canal street" in canon:
        return "geo"
    for family, keywords in FAMILY_KEYWORDS:
        if any(keyword in canon for keyword in keywords):
            return family
    return "other"


def infer_user_polarity(section: str) -> str:
    if section == "avoid_signals":
        return "negative"
    if section in {"stable_preferences", "recent_signals", "discriminative_signals"}:
        return "positive"
    return "neutral"


def infer_merchant_role(section: str, item: dict[str, Any]) -> str:
    role = MERCHANT_SECTION_TO_ROLE.get(section, "merchant_unknown")
    if section == "direct_evidence_signals" and canonical_text(item.get("polarity")) in {"negative", "mixed"}:
        return "merchant_risk"
    return role


def evidence_strength(confidence: str, evidence_refs: list[str], snippets: list[str]) -> float:
    conf_score = CONFIDENCE_WEIGHT.get(canonical_text(confidence), 0.66 if not confidence else 0.5)
    ref_score = min(len(evidence_refs), 3) / 3.0
    snippet_score = min(len(snippets), 2) / 2.0
    return round(0.45 * conf_score + 0.35 * ref_score + 0.20 * snippet_score, 6)


def ann_route_hint(entity_type: str, semantic_role: str, semantic_family: str, match_channel: str) -> str:
    if semantic_family == "other":
        return "not_ann_ready"
    if entity_type == "user":
        if semantic_role == "user_avoidance":
            return f"user_avoidance__{match_channel}"
        if semantic_role in {"user_positive_preference", "user_recent_intent", "user_latent_preference", "user_discriminative_signal"}:
            return f"user_positive__{match_channel}"
        return f"user_context__{match_channel}"
    if semantic_role == "merchant_risk":
        return f"merchant_risk__{match_channel}"
    if semantic_role in {"merchant_strength", "merchant_attribute", "merchant_usage_scene"}:
        return f"merchant_positive__{match_channel}"
    return f"merchant_context__{match_channel}"


def build_item_row(
    *,
    entity_type: str,
    entity_id: str,
    source_path: Path,
    source_row: dict[str, Any],
    source_section: str,
    source_item_index: int,
    item: dict[str, Any],
    raw_text: str,
    semantic_role: str,
    polarity: str,
    semantic_family: str,
) -> dict[str, Any]:
    evidence_refs = normalize_ref_list(item.get("evidence_refs"))
    direct_review_refs = normalize_ref_list(item.get("direct_review_refs"))
    contextual_refs = normalize_ref_list(item.get("contextual_refs"))
    if not evidence_refs:
        evidence_refs = [*direct_review_refs, *contextual_refs]
    snippets = normalize_snippets(item.get("evidence_snippets"))
    confidence = normalize_text(item.get("confidence")) or ("medium" if entity_type == "merchant" else "")
    support_basis = normalize_text(item.get("support_basis"))
    support_note = normalize_text(item.get("support_note") or item.get("reasoning"))
    text = normalize_text(raw_text)
    canonical = canonical_text(text)
    match_channel = MATCH_CHANNEL_BY_FAMILY.get(semantic_family, "other")
    strength = evidence_strength(confidence, evidence_refs, snippets)
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "item_id": item_hash(entity_type, entity_id, source_section, source_item_index, canonical, evidence_refs),
        "source_path": str(source_path),
        "source_section": source_section,
        "source_item_index": int(source_item_index),
        "raw_text": text,
        "canonical_text": canonical,
        "semantic_family": semantic_family,
        "semantic_role": semantic_role,
        "polarity": polarity,
        "match_channel": match_channel,
        "ann_route_hint": ann_route_hint(entity_type, semantic_role, semantic_family, match_channel),
        "match_ready": bool(semantic_family != "other" and evidence_refs),
        "confidence": confidence,
        "support_basis": support_basis,
        "support_note": support_note,
        "evidence_refs_json": compact_json(evidence_refs),
        "direct_review_refs_json": compact_json(direct_review_refs),
        "contextual_refs_json": compact_json(contextual_refs),
        "evidence_snippets_json": compact_json(snippets),
        "evidence_ref_count": int(len(evidence_refs)),
        "snippet_count": int(len(snippets)),
        "evidence_strength": strength,
        "guardrails_passed": bool(source_row.get("guardrails_passed", False)),
        "schema_valid": bool(source_row.get("schema_valid", True)),
        "input_index": source_row.get("input_index"),
        "queue_label": source_row.get("queue_label", ""),
        "density_band": source_row.get("density_band", ""),
        "merchant_sampling_segment": source_row.get("merchant_sampling_segment", ""),
    }


def user_items_from_row(row: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    user_id = normalize_text(row.get("user_id"))
    parsed = choose_parsed_json(row)
    if not user_id or not isinstance(parsed, dict):
        return []
    out: list[dict[str, Any]] = []
    grounded = parsed.get("grounded_facts") if isinstance(parsed.get("grounded_facts"), dict) else {}
    section_payloads = {
        "stable_preferences": grounded.get("stable_preferences", []),
        "avoid_signals": grounded.get("avoid_signals", []),
        "recent_signals": grounded.get("recent_signals", []),
        "context_rules": grounded.get("context_rules", []),
        "state_hypotheses": parsed.get("state_hypotheses", []),
        "discriminative_signals": parsed.get("discriminative_signals", []),
        "contextual_inference_signals": parsed.get("contextual_inference_signals", []),
    }
    for section, items in section_payloads.items():
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            text = normalize_text(item.get("claim"))
            if not text:
                continue
            family = infer_family(text, source_section=section, item=item, entity_type="user")
            out.append(
                build_item_row(
                    entity_type="user",
                    entity_id=user_id,
                    source_path=source_path,
                    source_row=row,
                    source_section=section,
                    source_item_index=idx,
                    item=item,
                    raw_text=text,
                    semantic_role=USER_SECTION_TO_ROLE.get(section, "user_unknown"),
                    polarity=infer_user_polarity(section),
                    semantic_family=family,
                )
            )
    return out


def merchant_items_from_row(row: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    business_id = normalize_text(row.get("business_id"))
    parsed = choose_parsed_json(row)
    if not business_id or not isinstance(parsed, dict):
        return []
    out: list[dict[str, Any]] = []
    for section, text_field in SECTION_TEXT_FIELD.items():
        items = parsed.get(section, [])
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            text = normalize_text(item.get(text_field))
            if not text:
                continue
            family = infer_family(text, source_section=section, item=item, entity_type="merchant")
            polarity = canonical_text(item.get("polarity")) or ("negative" if section == "risk_or_avoidance_notes" else "neutral")
            out.append(
                build_item_row(
                    entity_type="merchant",
                    entity_id=business_id,
                    source_path=source_path,
                    source_row=row,
                    source_section=section,
                    source_item_index=idx,
                    item=item,
                    raw_text=text,
                    semantic_role=infer_merchant_role(section, item),
                    polarity=polarity,
                    semantic_family=family,
                )
            )
    return out


def records_to_items(records: list[dict[str, Any]], source_path: Path, *, entity_type: str, include_failed: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not include_failed and record.get("guardrails_passed") is False:
            continue
        if entity_type == "user":
            rows.extend(user_items_from_row(record, source_path))
        elif entity_type == "merchant":
            rows.extend(merchant_items_from_row(record, source_path))
        else:
            raise ValueError(f"unknown entity_type={entity_type}")
    return rows


def group_counts(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return (
        df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="item_count")
        .sort_values(["item_count", *cols], ascending=[False, *([True] * len(cols))])
        .to_dict(orient="records")
    )


def build_alignment_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["entity_type", "semantic_role", "semantic_family", "match_channel"]
    for keys, group in df.groupby(group_cols, dropna=False):
        entity_type, role, family, channel = keys
        n = int(len(group))
        rows.append(
            {
                "entity_type": entity_type,
                "semantic_role": role,
                "semantic_family": family,
                "match_channel": channel,
                "item_count": n,
                "entity_count": int(group["entity_id"].nunique()),
                "evidence_ref_coverage": round(float((group["evidence_ref_count"] > 0).mean()), 6),
                "snippet_coverage": round(float((group["snippet_count"] > 0).mean()), 6),
                "avg_evidence_strength": round(float(group["evidence_strength"].mean()), 6),
                "sample_texts_json": compact_json(group["raw_text"].head(5).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["entity_type", "item_count"], ascending=[True, False])


def build_pairing_matrix(df: pd.DataFrame) -> list[dict[str, Any]]:
    match_ready = df[(df["match_ready"]) & (df["semantic_family"] != "other")]
    user = match_ready[match_ready["entity_type"] == "user"]
    merchant = match_ready[match_ready["entity_type"] == "merchant"]
    pairings = [
        ("positive_match", "user_positive_preference", ["merchant_strength", "merchant_attribute"], "same_family"),
        ("scene_match", "user_contextual_need", ["merchant_usage_scene", "merchant_attribute"], "scene_context"),
        ("recent_intent_match", "user_recent_intent", ["merchant_strength", "merchant_attribute", "merchant_usage_scene"], "same_family"),
        ("latent_match", "user_latent_preference", ["merchant_strength", "merchant_attribute", "merchant_cautious_inference"], "same_family"),
        ("discriminative_match", "user_discriminative_signal", ["merchant_strength", "merchant_attribute"], "same_family"),
        ("avoidance_risk_conflict", "user_avoidance", ["merchant_risk"], "same_family_or_risk"),
        ("context_constraint_conflict", "user_context_constraint", ["merchant_risk"], "same_family_or_risk"),
    ]
    rows: list[dict[str, Any]] = []
    for pairing_name, user_role, merchant_roles, rule in pairings:
        user_part = user[user["semantic_role"] == user_role]
        merchant_part = merchant[merchant["semantic_role"].isin(merchant_roles)]
        user_family_counts = Counter(user_part["semantic_family"].tolist())
        merchant_family_counts = Counter(merchant_part["semantic_family"].tolist())
        shared = sorted((set(user_family_counts) & set(merchant_family_counts)) - {"other"})
        rows.append(
            {
                "pairing_name": pairing_name,
                "user_role": user_role,
                "merchant_roles": merchant_roles,
                "family_rule": rule,
                "user_items": int(len(user_part)),
                "merchant_items": int(len(merchant_part)),
                "shared_families": shared,
                "shared_family_count": int(len(shared)),
                "user_family_counts": dict(user_family_counts),
                "merchant_family_counts": dict(merchant_family_counts),
            }
        )
    return rows


def write_outputs(run_dir: Path, items_df: pd.DataFrame, audit_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    items_path = run_dir / "semantic_atomic_items.parquet"
    audit_path = run_dir / "semantic_schema_alignment_audit.parquet"
    items_jsonl_path = run_dir / "semantic_atomic_items.jsonl"
    audit_json_path = run_dir / "semantic_schema_alignment_audit.json"
    summary_path = run_dir / "semantic_match_assets_summary.json"
    sample_path = run_dir / "semantic_atomic_items_sample.json"

    items_df.to_parquet(items_path, index=False)
    audit_df.to_parquet(audit_path, index=False)
    with items_jsonl_path.open("w", encoding="utf-8") as fh:
        for row in items_df.to_dict(orient="records"):
            fh.write(compact_json(row) + "\n")
    safe_json_write(audit_json_path, audit_df.to_dict(orient="records"))
    safe_json_write(sample_path, items_df.head(40).to_dict(orient="records"))
    safe_json_write(summary_path, summary)
    write_latest_run_pointer("12_semantic_match_assets", run_dir, {"run_tag": RUN_TAG, "rows": int(len(items_df))})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified user/merchant semantic atomic items for ANN and stage09/stage10 sidecar features.")
    parser.add_argument("--user-parsed-path", default=os.getenv("INPUT_USER_PARSED_PATH", "").strip())
    parser.add_argument("--merchant-parsed-path", default=os.getenv("INPUT_MERCHANT_PARSED_PATH", "").strip())
    parser.add_argument("--user-universe-path", default=os.getenv("INPUT_USER_UNIVERSE_PATH", "").strip())
    parser.add_argument("--merchant-universe-path", default=os.getenv("INPUT_MERCHANT_UNIVERSE_PATH", "").strip())
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-dir", default=os.getenv("OUTPUT_12_SEMANTIC_MATCH_ASSETS_RUN_DIR", "").strip())
    parser.add_argument("--include-failed-guardrails", action="store_true", default=os.getenv("SEMANTIC_MATCH_INCLUDE_FAILED_GUARDRAILS", "false").strip().lower() == "true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_path = resolve_optional_path(args.user_parsed_path, DEFAULT_USER_PARSED_ROOT, "prompt_only_generation_parsed.json")
    merchant_path = resolve_optional_path(args.merchant_parsed_path, DEFAULT_MERCHANT_PARSED_ROOT, "merchant_profile_generation_parsed.json")
    run_dir = normalize_legacy_project_path(args.run_dir) if args.run_dir else normalize_legacy_project_path(args.output_root) / now_run_id()

    user_records = load_json_records(user_path)
    merchant_records = load_json_records(merchant_path)
    user_rows_before_filter = len(user_records)
    merchant_rows_before_filter = len(merchant_records)
    user_universe_ids = load_universe_ids(args.user_universe_path, "user_id")
    merchant_universe_ids = load_universe_ids(args.merchant_universe_path, "business_id")
    user_records = filter_records_by_universe(user_records, "user_id", user_universe_ids)
    merchant_records = filter_records_by_universe(merchant_records, "business_id", merchant_universe_ids)
    user_items = records_to_items(user_records, user_path, entity_type="user", include_failed=bool(args.include_failed_guardrails))
    merchant_items = records_to_items(merchant_records, merchant_path, entity_type="merchant", include_failed=bool(args.include_failed_guardrails))
    items = user_items + merchant_items
    items_df = pd.DataFrame(items)
    if items_df.empty:
        raise RuntimeError("no semantic atomic items produced; check parsed input paths and guardrail filters")

    audit_df = build_alignment_audit(items_df)
    summary = {
        "run_tag": RUN_TAG,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_parsed_path": str(user_path),
        "merchant_parsed_path": str(merchant_path),
        "user_universe_path": str(normalize_legacy_project_path(args.user_universe_path)) if args.user_universe_path else "",
        "merchant_universe_path": str(normalize_legacy_project_path(args.merchant_universe_path)) if args.merchant_universe_path else "",
        "include_failed_guardrails": bool(args.include_failed_guardrails),
        "input_rows_before_universe_filter": {"user": int(user_rows_before_filter), "merchant": int(merchant_rows_before_filter)},
        "input_rows": {"user": int(len(user_records)), "merchant": int(len(merchant_records))},
        "universe_filter_counts": {
            "user_universe_ids": int(len(user_universe_ids)),
            "merchant_universe_ids": int(len(merchant_universe_ids)),
            "user_rows_dropped": int(user_rows_before_filter - len(user_records)),
            "merchant_rows_dropped": int(merchant_rows_before_filter - len(merchant_records)),
        },
        "semantic_atomic_items": int(len(items_df)),
        "entity_counts": {str(k): int(v) for k, v in items_df.groupby("entity_type")["entity_id"].nunique().to_dict().items()},
        "item_counts_by_entity": {str(k): int(v) for k, v in items_df["entity_type"].value_counts().sort_index().to_dict().items()},
        "item_counts_by_role": group_counts(items_df, ["entity_type", "semantic_role"]),
        "item_counts_by_family": group_counts(items_df, ["entity_type", "semantic_family"]),
        "item_counts_by_match_channel": group_counts(items_df, ["entity_type", "match_channel"]),
        "evidence_ref_coverage": round(float((items_df["evidence_ref_count"] > 0).mean()), 6),
        "snippet_coverage": round(float((items_df["snippet_count"] > 0).mean()), 6),
        "avg_evidence_strength": round(float(items_df["evidence_strength"].mean()), 6),
        "other_family_rate": round(float((items_df["semantic_family"] == "other").mean()), 6),
        "match_ready_rate": round(float(items_df["match_ready"].mean()), 6),
        "pairing_matrix": build_pairing_matrix(items_df),
        "output_files": {
            "semantic_atomic_items_parquet": str(run_dir / "semantic_atomic_items.parquet"),
            "semantic_atomic_items_jsonl": str(run_dir / "semantic_atomic_items.jsonl"),
            "semantic_schema_alignment_audit_parquet": str(run_dir / "semantic_schema_alignment_audit.parquet"),
            "semantic_schema_alignment_audit_json": str(run_dir / "semantic_schema_alignment_audit.json"),
            "summary": str(run_dir / "semantic_match_assets_summary.json"),
            "sample": str(run_dir / "semantic_atomic_items_sample.json"),
        },
    }
    write_outputs(run_dir, items_df, audit_df, summary)
    print(f"[OK] wrote {len(items_df)} semantic atomic items to {run_dir}")
    print(json.dumps({k: summary[k] for k in ["input_rows", "entity_counts", "semantic_atomic_items", "other_family_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
