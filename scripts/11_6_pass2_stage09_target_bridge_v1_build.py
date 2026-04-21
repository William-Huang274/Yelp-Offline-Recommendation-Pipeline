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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, project_path


RUN_TAG = "stage11_pass2_stage09_target_bridge_v1_build"

INPUT_RECEIVER_ROOT = env_or_project_path(
    "INPUT_11_PASS2_SIGNAL_RECEIVER_V1_ROOT_DIR",
    "data/output/11_pass2_signal_receiver_v1",
)
INPUT_RECEIVER_RUN_DIR = os.getenv("INPUT_11_PASS2_SIGNAL_RECEIVER_V1_RUN_DIR", "").strip()
OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_11_PASS2_STAGE09_TARGET_BRIDGE_V1_ROOT_DIR",
    "data/output/11_pass2_stage09_target_bridge_v1",
)
BASELINE_TARGET_ROOT = env_or_project_path(
    "INPUT_09_USER_PREFERENCE_TARGET_SCHEMA_V1_ROOT_DIR",
    "data/output/09_user_preference_target_schema_v1",
)
BASELINE_TARGET_RUN_DIR = os.getenv("INPUT_09_USER_PREFERENCE_TARGET_SCHEMA_V1_RUN_DIR", "").strip()

POOL_MODE = (os.getenv("PASS2_STAGE09_BRIDGE_POOL_MODE", "quality_first_then_balanced").strip().lower() or "quality_first_then_balanced")
SAMPLE_ROWS = int(os.getenv("PASS2_STAGE09_BRIDGE_SAMPLE_ROWS", "12").strip() or 12)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        path = normalize_legacy_project_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"run dir not found: {path}")
        return path
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.endswith(suffix)]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
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


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = safe_text(text)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = safe_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


TAG_PATTERNS: dict[str, list[str]] = {
    "seafood": ["seafood", "oyster", "oysters", "crab cake", "crab cakes", "gumbo", "shrimp", "crawfish"],
    "sushi": ["sushi", "tataki", "roll", "omakase", "nigiri", "sashimi"],
    "coffee_tea": ["coffee", "tea", "london fog", "dublin fog", "latte", "espresso"],
    "breakfast_brunch": ["breakfast", "brunch", "pastry", "pastries", "pancake", "waffle", "omelet", "omelette"],
    "burger_sandwich": ["burger", "sandwich", "turkey sandwich", "po boy", "po-boy"],
    "salad": ["salad", "salads"],
    "dessert": ["dessert", "cupcake", "cupcakes", "ice cream", "gelato", "sweet"],
    "pizza": ["pizza"],
    "bbq": ["bbq", "barbecue", "barbeque"],
    "cajun_creole": ["cajun", "creole", "new orleans"],
    "mexican": ["mexican", "taco", "tacos", "burrito", "quesadilla"],
    "italian": ["italian", "parmesan", "alfredo", "pasta"],
    "ramen_noodles": ["ramen", "noodle", "noodles"],
    "vegan_vegetarian": ["vegan", "vegetarian"],
    "service": ["service", "server", "staff", "attentive", "friendly", "wait", "waiting", "rude", "forgot", "forgetting", "menu provision"],
    "late_night": ["late night", "late-night"],
    "quick_bite": ["quick bite", "quick-bite"],
    "group_dining": ["group", "friends", "large party"],
    "family_dining": ["family", "kids", "kid"],
    "date_night": ["date", "romantic"],
}


SCENE_LABELS = {
    "late_night": "late_night",
    "quick_bite": "quick_bite",
    "group_dining": "group_dining",
    "family_dining": "family_friendly",
    "date_night": "date_night",
}


SERVICE_NEGATIVE_TERMS = (
    "rude",
    "forgot",
    "forgetting",
    "wait",
    "waiting",
    "inattention",
    "inattentive",
    "menu provision",
    "lack of menu",
    "service inconsistency",
)


def normalize_text(text: Any) -> str:
    normalized = safe_text(text).lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def find_tags(texts: list[str], allowed_keys: set[str] | None = None) -> list[str]:
    joined = " ".join(normalize_text(text) for text in texts if safe_text(text))
    tags: list[str] = []
    for label, variants in TAG_PATTERNS.items():
        if allowed_keys is not None and label not in allowed_keys:
            continue
        if any(normalize_text(variant) in joined for variant in variants):
            tags.append(label)
    return unique_keep_order(tags)


def collect_claim_texts(payload: dict[str, Any], *path: str) -> list[str]:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return []
        node = node.get(key)
    if not isinstance(node, list):
        return []
    texts: list[str] = []
    for item in node:
        if not isinstance(item, dict):
            continue
        claim = safe_text(item.get("claim"))
        if claim:
            texts.append(claim)
    return texts


def collect_unknown_fields(payload: dict[str, Any]) -> list[str]:
    node = payload.get("unknowns")
    if not isinstance(node, list):
        return []
    fields: list[str] = []
    for item in node:
        if not isinstance(item, dict):
            continue
        field = safe_text(item.get("field"))
        if field:
            fields.append(field)
    return unique_keep_order(fields)


def infer_service_risk_sensitivity(payload: dict[str, Any]) -> str:
    avoid_texts = collect_claim_texts(payload, "grounded_facts", "avoid_signals")
    disc_texts = collect_claim_texts(payload, "discriminative_signals")
    joined = " ".join(normalize_text(text) for text in avoid_texts + disc_texts)
    negative_hits = sum(1 for token in SERVICE_NEGATIVE_TERMS if token in joined)
    if negative_hits >= 2:
        return "high"
    if negative_hits >= 1:
        return "medium"
    if "service" in joined or "staff" in joined or "server" in joined:
        return "low"
    return "unknown"


def infer_recent_shift(
    *,
    stable_tags: list[str],
    recent_tags: list[str],
    unknown_fields: list[str],
) -> str:
    if "recent_shift" in set(unknown_fields):
        return "unknown"
    stable = set(stable_tags)
    recent = set(recent_tags)
    if not recent:
        return "unknown"
    if not stable:
        return "broadening"
    overlap = stable.intersection(recent)
    if overlap == recent:
        return "stable"
    if overlap:
        return "broadening"
    return "switching"


def build_open_discoveries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in payload.get("state_hypotheses", []):
        if not isinstance(item, dict):
            continue
        claim = safe_text(item.get("claim"))
        if not claim:
            continue
        out.append(
            {
                "type": safe_text(item.get("type")) or "hypothesis",
                "claim": claim,
                "confidence": safe_text(item.get("confidence")) or "low",
                "evidence_refs": safe_list(item.get("evidence_refs"))[:4],
                "reason_short": "derived from pass2 state_hypotheses",
            }
        )
        if len(out) >= 3:
            break
    return out


def choose_teacher_payload(row: pd.Series) -> tuple[str, dict[str, Any], bool]:
    quality_first = parse_json_object(row.get("quality_first_teacher_json"))
    balanced = parse_json_object(row.get("balanced_teacher_json"))
    strict_ready = bool(row.get("strict_quality_first_ready"))
    recommended_pool = safe_text(row.get("recommended_pool"))
    if POOL_MODE == "quality_first_only":
        return "quality_first", quality_first, strict_ready and bool(quality_first)
    if strict_ready and quality_first:
        return "quality_first", quality_first, True
    if POOL_MODE == "quality_first_then_balanced" and balanced:
        return "balanced", balanced, False
    if quality_first:
        return "quality_first", quality_first, strict_ready
    return "balanced", balanced, False


def build_target_schema(row: pd.Series) -> dict[str, Any]:
    source_pool, payload, sft_ready = choose_teacher_payload(row)
    grounded = payload.get("grounded_facts", {}) if isinstance(payload, dict) else {}
    stable_texts = collect_claim_texts(payload, "grounded_facts", "stable_preferences")
    avoid_texts = collect_claim_texts(payload, "grounded_facts", "avoid_signals")
    recent_texts = collect_claim_texts(payload, "grounded_facts", "recent_signals")
    context_texts = collect_claim_texts(payload, "grounded_facts", "context_rules")
    disc_texts = collect_claim_texts(payload, "discriminative_signals")
    unknown_fields = collect_unknown_fields(payload)

    stable_pref_tags = find_tags(stable_texts + disc_texts, {"seafood", "sushi", "coffee_tea", "breakfast_brunch", "burger_sandwich", "salad", "dessert", "pizza", "bbq", "cajun_creole", "mexican", "italian", "ramen_noodles", "vegan_vegetarian"})
    recent_pref_tags = find_tags(recent_texts, {"seafood", "sushi", "coffee_tea", "breakfast_brunch", "burger_sandwich", "salad", "dessert", "pizza", "bbq", "cajun_creole", "mexican", "italian", "ramen_noodles", "vegan_vegetarian"})
    avoided_tags = find_tags(avoid_texts, {"seafood", "sushi", "coffee_tea", "breakfast_brunch", "burger_sandwich", "salad", "dessert", "pizza", "bbq", "cajun_creole", "mexican", "italian", "ramen_noodles", "vegan_vegetarian"})
    scene_hits = find_tags(stable_texts + recent_texts + context_texts, {"late_night", "quick_bite", "group_dining", "family_dining", "date_night"})
    recent_scene_hits = find_tags(recent_texts + context_texts, {"late_night", "quick_bite", "group_dining", "family_dining", "date_night"})
    preferred_scenes = [SCENE_LABELS[tag] for tag in scene_hits if tag in SCENE_LABELS]
    recent_scenes = [SCENE_LABELS[tag] for tag in recent_scene_hits if tag in SCENE_LABELS]

    service_risk_sensitivity = infer_service_risk_sensitivity(payload)
    recent_shift_value = infer_recent_shift(
        stable_tags=stable_pref_tags,
        recent_tags=recent_pref_tags + recent_scenes,
        unknown_fields=unknown_fields,
    )
    confidence_overall = "high" if source_pool == "quality_first" and sft_ready else ("medium" if payload else "low")

    core_profile = {
        "stable_preferences": {
            "preferred_cuisines": stable_pref_tags,
            "preferred_meals": [],
            "preferred_scenes": unique_keep_order(preferred_scenes),
        },
        "avoid_signals": {
            "avoided_cuisines": avoided_tags,
            "service_risk_sensitivity": service_risk_sensitivity,
        },
        "recent_preferences": {
            "recent_cuisines": recent_pref_tags,
            "recent_scenes": unique_keep_order(recent_scenes),
            "recent_shift": recent_shift_value,
        },
        "behavior_summary": {
            "geo_style": "unknown",
            "novelty_style": "unknown",
            "confidence_overall": confidence_overall,
        },
        "open_discoveries": build_open_discoveries(payload),
    }

    return {
        "source_pool": source_pool,
        "sft_ready_v1": bool(sft_ready),
        "unknown_fields": unknown_fields,
        "target_schema_v1_json": json.dumps({"core_profile": core_profile}, ensure_ascii=False),
        "preferred_cuisine_count": len(stable_pref_tags),
        "preferred_scene_count": len(preferred_scenes),
        "avoided_cuisine_count": len(avoided_tags),
        "recent_shift_value": recent_shift_value,
        "service_risk_sensitivity": service_risk_sensitivity,
        "confidence_overall": confidence_overall,
    }


def read_baseline_target_frame() -> tuple[pd.DataFrame, Path | None]:
    try:
        baseline_run = resolve_run(
            BASELINE_TARGET_RUN_DIR,
            BASELINE_TARGET_ROOT,
            "_full_stage09_user_preference_target_schema_v1_build",
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["user_id", "target_schema_v1_json", "sft_ready_v1"]), None
    df = pd.read_parquet(baseline_run / "user_preference_target_schema_v1.parquet")
    df["user_id"] = df["user_id"].astype(str)
    return df[["user_id", "target_schema_v1_json", "sft_ready_v1"]].copy(), baseline_run


def parse_baseline_preferred_cuisines(text: Any) -> list[str]:
    payload = parse_json_object(text)
    core = payload.get("core_profile", {}) if isinstance(payload, dict) else {}
    stable = core.get("stable_preferences", {}) if isinstance(core, dict) else {}
    return safe_list(stable.get("preferred_cuisines"))


def main() -> None:
    receiver_run = resolve_run(
        INPUT_RECEIVER_RUN_DIR,
        INPUT_RECEIVER_ROOT,
        "_full_stage11_pass2_signal_receiver_v1_build",
    )
    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    user_summary = pd.read_parquet(receiver_run / "pass2_receiver_user_summary.parquet")
    if user_summary.empty:
        raise ValueError(f"receiver run has no user summary rows: {receiver_run}")
    user_summary["user_id"] = user_summary["user_id"].astype(str)

    mapped_rows: list[dict[str, Any]] = []
    for row in user_summary.to_dict(orient="records"):
        series = pd.Series(row)
        target = build_target_schema(series)
        mapped_rows.append(
            {
                "user_id": safe_text(series.get("user_id")),
                "density_band": safe_text(series.get("density_band")),
                "input_index": int(series.get("input_index", 0) or 0),
                "recommended_pool": safe_text(series.get("recommended_pool")),
                **target,
            }
        )

    mapped_df = pd.DataFrame(mapped_rows)
    mapped_df.to_parquet(run_dir / "user_preference_target_schema_v1.parquet", index=False)

    baseline_df, baseline_run = read_baseline_target_frame()
    baseline_df["baseline_preferred_cuisines"] = baseline_df["target_schema_v1_json"].map(parse_baseline_preferred_cuisines)
    compare_df = mapped_df.merge(
        baseline_df[["user_id", "sft_ready_v1", "baseline_preferred_cuisines"]].rename(columns={"sft_ready_v1": "baseline_sft_ready_v1"}),
        on="user_id",
        how="left",
    )
    compare_df["mapped_preferred_cuisines"] = compare_df["target_schema_v1_json"].map(parse_baseline_preferred_cuisines)
    compare_df["preferred_cuisine_overlap_count"] = compare_df.apply(
        lambda row: len(set(safe_list(row.get("mapped_preferred_cuisines"))).intersection(set(safe_list(row.get("baseline_preferred_cuisines"))))),
        axis=1,
    )

    sample_rows = compare_df.head(SAMPLE_ROWS).to_dict(orient="records")
    safe_json_write(run_dir / "pass2_stage09_target_bridge_sample.json", sample_rows)

    summary = {
        "run_tag": RUN_TAG,
        "receiver_run_dir": str(receiver_run),
        "rows_total": int(len(mapped_df)),
        "pool_mode": POOL_MODE,
        "sft_ready_users": int(mapped_df["sft_ready_v1"].fillna(False).sum()),
        "source_pool_counts": {str(k): int(v) for k, v in mapped_df["source_pool"].value_counts().to_dict().items()},
        "preferred_cuisine_nonempty_rate": float(mapped_df["preferred_cuisine_count"].gt(0).mean()),
        "preferred_scene_nonempty_rate": float(mapped_df["preferred_scene_count"].gt(0).mean()),
        "avoided_cuisine_nonempty_rate": float(mapped_df["avoided_cuisine_count"].gt(0).mean()),
        "recent_shift_counts": {str(k): int(v) for k, v in mapped_df["recent_shift_value"].value_counts().to_dict().items()},
        "service_risk_counts": {str(k): int(v) for k, v in mapped_df["service_risk_sensitivity"].value_counts().to_dict().items()},
        "confidence_overall_counts": {str(k): int(v) for k, v in mapped_df["confidence_overall"].value_counts().to_dict().items()},
        "baseline_overlap_users": int(compare_df["baseline_preferred_cuisines"].notna().sum()),
        "baseline_overlap_positive_rate": float(compare_df["preferred_cuisine_overlap_count"].gt(0).mean()),
        "baseline_sft_ready_overlap_rate": float(compare_df["baseline_sft_ready_v1"].fillna(False).mean()),
    }
    safe_json_write(run_dir / "pass2_stage09_target_bridge_summary.json", summary)

    run_meta = {
        "run_tag": RUN_TAG,
        "run_dir": str(run_dir),
        "receiver_run_dir": str(receiver_run),
        "baseline_target_root": str(BASELINE_TARGET_ROOT),
        "baseline_target_run_dir": str(baseline_run) if baseline_run is not None else "",
        "pool_mode": POOL_MODE,
        "rows_total": int(len(mapped_df)),
        "sft_ready_users": int(mapped_df["sft_ready_v1"].fillna(False).sum()),
    }
    safe_json_write(run_dir / "run_meta.json", run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
