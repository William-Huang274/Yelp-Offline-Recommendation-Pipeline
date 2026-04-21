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
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

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

MERCHANT_PERSONA_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_ROOT_DIR",
    "data/output/09_merchant_persona_schema_v1",
)
MERCHANT_PERSONA_RUN_DIR = os.getenv("INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_USER_STATE_TRAINING_ASSETS_V1_ROOT_DIR",
    "data/output/09_user_state_training_assets_v1",
)
RUN_TAG = "stage09_user_state_training_assets_v1_build"
SAMPLE_ROWS = int(os.getenv("USER_STATE_TRAINING_ASSETS_V1_SAMPLE_ROWS", "10").strip() or 10)
REVIEW_MAX_ITEMS = int(os.getenv("USER_STATE_TRAINING_REVIEW_MAX_ITEMS", "36").strip() or 36)
TIP_MAX_ITEMS = int(os.getenv("USER_STATE_TRAINING_TIP_MAX_ITEMS", "10").strip() or 10)
MAX_POS_ANCHORS = int(os.getenv("USER_STATE_TRAINING_MAX_POS_ANCHORS", "2").strip() or 2)
MAX_NEG_ANCHORS = int(os.getenv("USER_STATE_TRAINING_MAX_NEG_ANCHORS", "2").strip() or 2)
MAX_PAIRS_PER_USER = int(os.getenv("USER_STATE_TRAINING_MAX_PAIRS_PER_USER", "4").strip() or 4)
MIN_TEXT_CHARS = int(os.getenv("USER_STATE_TRAINING_MIN_TEXT_CHARS", "12").strip() or 12)
MIN_TEXT_WORDS = int(os.getenv("USER_STATE_TRAINING_MIN_TEXT_WORDS", "2").strip() or 2)


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
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(value)
    except Exception:
        return default


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


def density_band(n_train_max: Any) -> str:
    numeric = safe_int(n_train_max, 0)
    if numeric <= 7:
        return "b5_lower_visible_5_7"
    if numeric <= 17:
        return "b5_mid_visible_8_17"
    return "b5_high_visible_18_plus"


def is_usable_text(value: Any) -> bool:
    text = safe_text(value)
    if not text:
        return False
    lowered = text.lower()
    if len(text) < MIN_TEXT_CHARS and len(text.split()) < MIN_TEXT_WORDS:
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
                        "recent" if float(item.get("days_since_review", 99999) or 99999) <= 180 else "older"
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
        payload = maybe_cap_items(payload, REVIEW_MAX_ITEMS)
        rows.append(
            {
                "user_id": str(user_id),
                "review_evidence_stream_json": json.dumps(payload, ensure_ascii=False),
                "review_evidence_count_v1": int(len(payload)),
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
                    "time_bucket": "recent" if float(item.get("tip_age_days", 99999) or 99999) <= 180 else "older",
                    "recommend_cue": int(float(item.get("has_recommend_cue", 0.0) or 0.0)),
                    "time_cue": int(float(item.get("has_time_cue", 0.0) or 0.0)),
                    "dish_cue": int(float(item.get("has_dish_cue", 0.0) or 0.0)),
                    "text": text,
                }
            )
        payload = maybe_cap_items(payload, TIP_MAX_ITEMS)
        rows.append(
            {
                "user_id": str(user_id),
                "tip_evidence_stream_json": json.dumps(payload, ensure_ascii=False),
                "tip_evidence_count_v1": int(len(payload)),
            }
        )
    return pd.DataFrame(rows)


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
    return [item for item in payload if isinstance(item, dict)]


def parse_target_schema(raw: Any) -> dict[str, Any]:
    text = safe_text(raw)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    if "core_profile" in payload:
        return payload
    return {"core_profile": payload}


def merchant_state_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "business_id": safe_text(row.get("business_id")),
        "name": safe_text(row.get("name")),
        "city": safe_text(row.get("city")),
        "merchant_primary_cuisine": safe_text(row.get("merchant_primary_cuisine")),
        "merchant_secondary_cuisine": safe_text(row.get("merchant_secondary_cuisine")),
        "persona_meal_tags": safe_list(row.get("persona_meal_tags")),
        "persona_scene_tags": safe_list(row.get("persona_scene_tags")),
        "persona_property_tags": safe_list(row.get("persona_property_tags")),
        "persona_dish_tags": safe_list(row.get("persona_dish_tags")),
        "persona_service_tags": safe_list(row.get("persona_service_tags")),
        "persona_complaint_tags": safe_list(row.get("persona_complaint_tags")),
        "persona_time_tags": safe_list(row.get("persona_time_tags")),
        "persona_quality_band": safe_text(row.get("persona_quality_band")),
        "persona_reliability_band": safe_text(row.get("persona_reliability_band")),
        "merchant_core_text_v3": safe_text(row.get("merchant_core_text_v3")),
        "merchant_semantic_text_v3": safe_text(row.get("merchant_semantic_text_v3")),
        "merchant_pos_text_v3": safe_text(row.get("merchant_pos_text_v3")),
        "merchant_neg_text_v3": safe_text(row.get("merchant_neg_text_v3")),
        "merchant_context_text_v3": safe_text(row.get("merchant_context_text_v3")),
    }


def select_anchor_events(events: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in events:
        business_id = safe_text(item.get("business_id"))
        if not business_id or business_id in seen:
            continue
        seen.add(business_id)
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def build_pairs_for_user(row: pd.Series, merchant_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    user_id = safe_text(row.get("user_id"))
    pos_events = select_anchor_events(load_json_list(row.get("positive_anchor_sequence_json")), MAX_POS_ANCHORS)
    neg_events = select_anchor_events(load_json_list(row.get("negative_anchor_sequence_json")), MAX_NEG_ANCHORS)
    if not pos_events or not neg_events:
        return []

    target_schema = parse_target_schema(row.get("target_schema_v1_json"))
    core_profile = target_schema.get("core_profile", {}) if isinstance(target_schema, dict) else {}
    recent_preferences = core_profile.get("recent_preferences", {}) if isinstance(core_profile, dict) else {}
    avoid_signals = core_profile.get("avoid_signals", {}) if isinstance(core_profile, dict) else {}
    behavior_summary = core_profile.get("behavior_summary", {}) if isinstance(core_profile, dict) else {}

    pair_rows: list[dict[str, Any]] = []
    pair_idx = 0
    for pos in pos_events:
        for neg in neg_events:
            if pair_idx >= MAX_PAIRS_PER_USER:
                return pair_rows
            chosen_id = safe_text(pos.get("business_id"))
            rejected_id = safe_text(neg.get("business_id"))
            chosen_state = merchant_lookup.get(chosen_id)
            rejected_state = merchant_lookup.get(rejected_id)
            if not chosen_state or not rejected_state or chosen_id == rejected_id:
                continue
            pair_idx += 1
            pair_rows.append(
                {
                    "pair_id": f"{user_id}_pair_{pair_idx}",
                    "pair_source": "anchor_pos_vs_anchor_neg",
                    "user_id": user_id,
                    "density_band": density_band(row.get("n_train_max")),
                    "n_train_max": safe_int(row.get("n_train_max")),
                    "sequence_event_count": safe_int(row.get("sequence_event_count")),
                    "review_evidence_count_v1": safe_int(row.get("review_evidence_count_v1")),
                    "tip_evidence_count_v1": safe_int(row.get("tip_evidence_count_v1")),
                    "review_evidence_stream_json": safe_text(row.get("review_evidence_stream_json")),
                    "tip_evidence_stream_json": safe_text(row.get("tip_evidence_stream_json")),
                    "recent_event_sequence_json": safe_text(row.get("recent_event_sequence_json")),
                    "positive_anchor_sequence_json": safe_text(row.get("positive_anchor_sequence_json")),
                    "negative_anchor_sequence_json": safe_text(row.get("negative_anchor_sequence_json")),
                    "chosen_business_id": chosen_id,
                    "rejected_business_id": rejected_id,
                    "chosen_merchant_state_json": json.dumps(chosen_state, ensure_ascii=False),
                    "rejected_merchant_state_json": json.dumps(rejected_state, ensure_ascii=False),
                    "recent_shift_label": safe_text(recent_preferences.get("recent_shift")) or "unknown",
                    "service_risk_sensitivity_label": safe_text(avoid_signals.get("service_risk_sensitivity")) or "unknown",
                    "novelty_style_label": safe_text(behavior_summary.get("novelty_style")) or "unknown",
                    "target_core_profile_json": json.dumps(core_profile, ensure_ascii=False),
                }
            )
    return pair_rows


def main() -> None:
    user_pref_run = resolve_run(USER_PREF_RUN_DIR, USER_PREF_ROOT, "_full_stage09_user_preference_schema_v1_build")
    user_profiles_run = resolve_run(USER_PROFILES_RUN_DIR, USER_PROFILES_ROOT, "_full_stage09_user_profile_build")
    tip_signal_run = resolve_run(TIP_SIGNAL_RUN_DIR, TIP_SIGNAL_ROOT, "_full_stage09_tip_signal_build")
    sequence_run = resolve_run(SEQUENCE_RUN_DIR, SEQUENCE_ROOT, "_full_stage09_sequence_view_schema_v1_build")
    target_run = resolve_run(TARGET_RUN_DIR, TARGET_ROOT, "_full_stage09_user_preference_target_schema_v1_build")
    merchant_run = resolve_run(MERCHANT_PERSONA_RUN_DIR, MERCHANT_PERSONA_ROOT, "_full_stage09_merchant_persona_schema_v1_build")

    user_pref = pd.read_parquet(user_pref_run / "user_preference_schema_v1.parquet")
    user_profile_evidence = pd.read_csv(user_profiles_run / "user_profile_evidence.csv")
    tip_signal_weights = pd.read_parquet(tip_signal_run / "tip_signal_weights.parquet")
    sequence = pd.read_parquet(sequence_run / "sequence_view_schema_v1.parquet")
    target = pd.read_parquet(target_run / "user_preference_target_schema_v1.parquet")
    merchant = pd.read_parquet(merchant_run / "merchant_persona_schema_v1.parquet")

    for df in (user_pref, user_profile_evidence, tip_signal_weights, sequence, target):
        df["user_id"] = df["user_id"].astype(str)
    merchant["business_id"] = merchant["business_id"].astype(str)

    review_stream = aggregate_review_evidence(user_profile_evidence)
    tip_stream = aggregate_tip_evidence(tip_signal_weights)

    user_state_base = (
        target.loc[target["sft_ready_v1"].fillna(False).eq(True)]
        .merge(user_pref[["user_id", "n_train_max"]], on="user_id", how="left")
        .merge(
            sequence[
                [
                    "user_id",
                    "recent_event_sequence_json",
                    "positive_anchor_sequence_json",
                    "negative_anchor_sequence_json",
                ]
            ],
            on="user_id",
            how="left",
        )
        .merge(review_stream, on="user_id", how="left")
        .merge(tip_stream, on="user_id", how="left")
    )

    merchant_lookup = {
        row["business_id"]: merchant_state_payload(pd.Series(row))
        for row in merchant.to_dict(orient="records")
    }

    pair_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for row in user_state_base.to_dict(orient="records"):
        series = pd.Series(row)
        target_schema = parse_target_schema(series.get("target_schema_v1_json"))
        core_profile = target_schema.get("core_profile", {}) if isinstance(target_schema, dict) else {}
        recent_preferences = core_profile.get("recent_preferences", {}) if isinstance(core_profile, dict) else {}
        avoid_signals = core_profile.get("avoid_signals", {}) if isinstance(core_profile, dict) else {}
        behavior_summary = core_profile.get("behavior_summary", {}) if isinstance(core_profile, dict) else {}
        stable_preferences = core_profile.get("stable_preferences", {}) if isinstance(core_profile, dict) else {}

        label_rows.append(
            {
                "user_id": safe_text(series.get("user_id")),
                "density_band": density_band(series.get("n_train_max")),
                "n_train_max": safe_int(series.get("n_train_max")),
                "sequence_event_count": safe_int(series.get("sequence_event_count")),
                "review_evidence_count_v1": safe_int(series.get("review_evidence_count_v1")),
                "tip_evidence_count_v1": safe_int(series.get("tip_evidence_count_v1")),
                "recent_shift_label": safe_text(recent_preferences.get("recent_shift")) or "unknown",
                "service_risk_sensitivity_label": safe_text(avoid_signals.get("service_risk_sensitivity")) or "unknown",
                "novelty_style_label": safe_text(behavior_summary.get("novelty_style")) or "unknown",
                "preferred_cuisines_json": json.dumps(stable_preferences.get("preferred_cuisines", []), ensure_ascii=False),
                "preferred_scenes_json": json.dumps(stable_preferences.get("preferred_scenes", []), ensure_ascii=False),
                "review_evidence_stream_json": safe_text(series.get("review_evidence_stream_json")),
                "tip_evidence_stream_json": safe_text(series.get("tip_evidence_stream_json")),
                "recent_event_sequence_json": safe_text(series.get("recent_event_sequence_json")),
                "positive_anchor_sequence_json": safe_text(series.get("positive_anchor_sequence_json")),
                "negative_anchor_sequence_json": safe_text(series.get("negative_anchor_sequence_json")),
            }
        )
        pair_rows.extend(build_pairs_for_user(series, merchant_lookup))

    pair_df = pd.DataFrame(pair_rows)
    label_df = pd.DataFrame(label_rows)

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    pair_path = run_dir / "user_state_pairwise_train_v1.parquet"
    label_path = run_dir / "user_state_labels_v1.parquet"
    sample_path = run_dir / "user_state_training_assets_v1_sample.json"
    summary_path = run_dir / "user_state_training_assets_v1_summary.json"

    pair_df.to_parquet(pair_path, index=False)
    label_df.to_parquet(label_path, index=False)

    summary = {
        "users_total": int(len(label_df)),
        "users_with_pairs": int(pair_df["user_id"].nunique()) if not pair_df.empty else 0,
        "pair_rows": int(len(pair_df)),
        "pair_rows_per_user_mean": float(pair_df.groupby("user_id").size().mean()) if not pair_df.empty else 0.0,
        "density_band_counts": {k: int(v) for k, v in label_df["density_band"].value_counts().to_dict().items()},
        "pair_density_band_counts": {k: int(v) for k, v in pair_df["density_band"].value_counts().to_dict().items()} if not pair_df.empty else {},
        "recent_shift_counts": {k: int(v) for k, v in label_df["recent_shift_label"].value_counts().to_dict().items()},
        "service_risk_counts": {k: int(v) for k, v in label_df["service_risk_sensitivity_label"].value_counts().to_dict().items()},
        "novelty_style_counts": {k: int(v) for k, v in label_df["novelty_style_label"].value_counts().to_dict().items()},
        "review_evidence_nonempty_rate": float(label_df["review_evidence_stream_json"].fillna("").ne("").mean()),
        "tip_evidence_nonempty_rate": float(label_df["tip_evidence_stream_json"].fillna("").ne("").mean()),
        "sequence_nonempty_rate": float(label_df["recent_event_sequence_json"].fillna("").ne("").mean()),
    }

    sample_records = {
        "pair_samples": pair_df.head(SAMPLE_ROWS).to_dict(orient="records"),
        "label_samples": label_df.head(SAMPLE_ROWS).to_dict(orient="records"),
    }

    safe_json_write(summary_path, summary)
    safe_json_write(sample_path, sample_records)
    safe_json_write(
        run_dir / "run_meta.json",
        {
            "run_tag": RUN_TAG,
            "user_pref_run_dir": str(user_pref_run),
            "user_profiles_run_dir": str(user_profiles_run),
            "tip_signal_run_dir": str(tip_signal_run),
            "sequence_run_dir": str(sequence_run),
            "target_run_dir": str(target_run),
            "merchant_persona_run_dir": str(merchant_run),
            "output_run_dir": str(run_dir),
            "summary": summary,
        },
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
