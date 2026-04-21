from __future__ import annotations

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

SEQUENCE_ROOT = env_or_project_path(
    "INPUT_09_SEQUENCE_VIEW_SCHEMA_V1_ROOT_DIR",
    "data/output/09_sequence_view_schema_v1",
)
SEQUENCE_RUN_DIR = os.getenv("INPUT_09_SEQUENCE_VIEW_SCHEMA_V1_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_USER_PREFERENCE_TARGET_SCHEMA_V1_ROOT_DIR",
    "data/output/09_user_preference_target_schema_v1",
)
RUN_TAG = "stage09_user_preference_target_schema_v1_build"
SAMPLE_ROWS = int(os.getenv("USER_PREFERENCE_TARGET_SCHEMA_V1_SAMPLE_ROWS", "12").strip() or 12)


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
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


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


def overlap_resolve(preferred: list[str], avoided: list[str]) -> tuple[list[str], list[str], list[str]]:
    pref = unique_keep_order(preferred)
    avoid = unique_keep_order(avoided)
    overlap = [value for value in pref if value in set(avoid)]
    if not overlap:
        return pref, avoid, []
    overlap_set = set(overlap)
    pref = [value for value in pref if value not in overlap_set]
    avoid = [value for value in avoid if value not in overlap_set]
    return pref, avoid, overlap


def band_from_ratio(value: Any, *, low: float, high: float) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "unknown"
    if pd.isna(numeric):
        return "unknown"
    if numeric < low:
        return "low"
    if numeric < high:
        return "medium"
    return "high"


def geo_style(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "unknown"
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.70:
        return "hyperlocal"
    if numeric >= 0.40:
        return "metro"
    return "mixed"


def novelty_style(repeat_ratio: Any) -> str:
    try:
        numeric = float(repeat_ratio)
    except Exception:
        return "unknown"
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.55:
        return "low"
    if numeric >= 0.30:
        return "medium"
    return "high"


def recent_shift(stable_cuisines: list[str], recent_cuisines: list[str], flag_value: Any) -> str:
    flag = safe_text(flag_value).lower()
    if flag in {"switching", "broadening", "stable"}:
        return flag
    stable = set(stable_cuisines)
    recent = set(recent_cuisines)
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


def build_sentence_refs(sentences: list[str], prefix: str) -> list[str]:
    refs = []
    for idx, text in enumerate(sentences, start=1):
        if safe_text(text):
            refs.append(f"{prefix}_{idx}")
    return refs


def load_sequence_refs(raw_json: Any, prefix: str) -> list[str]:
    text = safe_text(raw_json)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    refs = []
    for idx, item in enumerate(payload, start=1):
        rank = None
        if isinstance(item, dict):
            rank = item.get("recent_rank")
        if rank is None:
            refs.append(f"{prefix}_{idx}")
        else:
            refs.append(f"{prefix}_{rank}")
    return refs


def take_refs(*groups: list[str], limit: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = safe_text(item)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
            if len(out) >= limit:
                return out
    return out


def build_open_discoveries(
    *,
    preferred_cuisines: list[str],
    avoided_cuisines: list[str],
    preferred_scenes: list[str],
    recent_cuisines: list[str],
    recent_scenes: list[str],
    ambivalent_cuisines: list[str],
    service_risk_sensitivity: str,
    confidence_overall: str,
    recent_shift_value: str,
    positive_event_refs: list[str],
    negative_event_refs: list[str],
    positive_sentence_refs: list[str],
    negative_sentence_refs: list[str],
) -> list[dict[str, Any]]:
    discoveries: list[dict[str, Any]] = []
    has_pos = bool(positive_sentence_refs or positive_event_refs)
    has_neg = bool(negative_sentence_refs or negative_event_refs)

    if (
        ambivalent_cuisines
        and has_pos
        and has_neg
        and (
            recent_shift_value in {"switching", "broadening"}
            or len(negative_sentence_refs) >= 2
            or len(negative_event_refs) >= 2
        )
    ):
        discoveries.append(
            {
                "type": "preference_conflict",
                "claim": f"the user shows mixed signals on {', '.join(ambivalent_cuisines[:3])}",
                "confidence": "medium",
                "evidence_refs": take_refs(
                    positive_sentence_refs,
                    negative_sentence_refs,
                    positive_event_refs,
                    negative_event_refs,
                    limit=4,
                ),
                "reason_short": "both positive and negative evidence exist for the same cuisine",
            }
        )

    if recent_shift_value in {"switching", "broadening"} and (recent_cuisines or recent_scenes):
        focus_bits = unique_keep_order(recent_cuisines[:2] + recent_scenes[:2])
        discoveries.append(
            {
                "type": "context_dependent_preference",
                "claim": (
                    "recent evidence suggests the user is "
                    f"{recent_shift_value} toward {', '.join(focus_bits[:3])}"
                    if focus_bits
                    else "recent evidence suggests the user is shifting behavior"
                ),
                "confidence": "medium" if recent_shift_value == "broadening" else "low",
                "evidence_refs": take_refs(
                    positive_event_refs,
                    positive_sentence_refs,
                    limit=4,
                ),
                "reason_short": "recent events differ from long-term stable profile",
            }
        )

    if service_risk_sensitivity in {"medium", "high"} and (negative_sentence_refs or negative_event_refs):
        discoveries.append(
            {
                "type": "tolerance_hypothesis",
                "claim": (
                    "the user appears sensitive to service inconsistency or wait-related issues"
                    if service_risk_sensitivity == "high"
                    else "the user may be moderately sensitive to service or wait-related issues"
                ),
                "confidence": service_risk_sensitivity,
                "evidence_refs": take_refs(
                    negative_sentence_refs,
                    negative_event_refs,
                    limit=4,
                ),
                "reason_short": "negative evidence is concentrated on service-risk dimensions",
            }
        )

    if preferred_scenes and recent_scenes:
        recent_only = [scene for scene in recent_scenes if scene not in set(preferred_scenes)]
        if recent_only:
            discoveries.append(
                {
                    "type": "scenario_specific_preference",
                    "claim": f"the user may behave differently in recent {', '.join(recent_only[:2])} contexts",
                    "confidence": "low",
                    "evidence_refs": take_refs(
                        positive_event_refs,
                        positive_sentence_refs,
                        limit=4,
                    ),
                    "reason_short": "recent scenes are not fully aligned with long-term top scenes",
                }
            )

    if confidence_overall == "low" or (not preferred_cuisines and not avoided_cuisines):
        discoveries.append(
            {
                "type": "uncertainty_note",
                "claim": "the current training-period evidence is insufficient for a fully stable preference profile",
                "confidence": "high",
                "evidence_refs": take_refs(
                    positive_event_refs,
                    negative_event_refs,
                    positive_sentence_refs,
                    negative_sentence_refs,
                    limit=4,
                ),
                "reason_short": "evidence density is limited or core preferences are sparse",
            }
        )

    return discoveries[:3]


def main() -> None:
    user_pref_run = resolve_run(
        USER_PREF_RUN_DIR,
        USER_PREF_ROOT,
        "_full_stage09_user_preference_schema_v1_build",
    )
    sequence_run = resolve_run(
        SEQUENCE_RUN_DIR,
        SEQUENCE_ROOT,
        "_full_stage09_sequence_view_schema_v1_build",
    )

    user_pref = pd.read_parquet(user_pref_run / "user_preference_schema_v1.parquet")
    sequence = pd.read_parquet(sequence_run / "sequence_view_schema_v1.parquet")

    user_pref["user_id"] = user_pref["user_id"].astype(str)
    sequence["user_id"] = sequence["user_id"].astype(str)

    merged = user_pref.merge(
        sequence[
            [
                "user_id",
                "sequence_event_count",
                "repeat_business_ratio",
                "recent_event_sequence_json",
                "positive_anchor_sequence_json",
                "negative_anchor_sequence_json",
            ]
        ],
        on="user_id",
        how="left",
    )

    records: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        preferred_cuisines = unique_keep_order(
            [row.get("long_term_top_cuisine")] + safe_list(row.get("schema_pos_cuisine"))
        )
        avoided_cuisines = unique_keep_order(
            [row.get("negative_top_cuisine")] + safe_list(row.get("schema_neg_cuisine"))
        )
        preferred_cuisines, avoided_cuisines, ambivalent_cuisines = overlap_resolve(
            preferred_cuisines,
            avoided_cuisines,
        )

        preferred_meals = safe_list(row.get("meal_tags"))
        preferred_scenes = safe_list(row.get("scene_tags"))
        preferred_properties = safe_list(row.get("property_tags"))
        recent_meals = safe_list(row.get("recent_meal_tags"))
        recent_scenes = safe_list(row.get("recent_scene_tags"))
        recent_properties = safe_list(row.get("recent_property_tags"))

        positive_sentences = safe_list(row.get("positive_evidence_sentences"))
        negative_sentences = safe_list(row.get("negative_evidence_sentences"))
        positive_event_refs = load_sequence_refs(row.get("positive_anchor_sequence_json"), "anchor_pos")
        negative_event_refs = load_sequence_refs(row.get("negative_anchor_sequence_json"), "anchor_neg")
        recent_focus_cuisines = unique_keep_order([row.get("recent_top_cuisine"), row.get("latest_top_cuisine")])
        service_risk_sensitivity = band_from_ratio(
            row.get("negative_pressure"),
            low=0.15,
            high=0.35,
        )
        recent_shift_value = recent_shift(
            preferred_cuisines,
            recent_focus_cuisines,
            row.get("cuisine_shift_flag"),
        )
        confidence_overall = (
            "high"
            if preferred_cuisines and preferred_meals and positive_sentences
            else "medium"
            if preferred_cuisines or preferred_meals or preferred_scenes
            else "low"
        )
        positive_sentence_refs = build_sentence_refs(positive_sentences, "pos")
        negative_sentence_refs = build_sentence_refs(negative_sentences, "neg")
        open_discoveries = build_open_discoveries(
            preferred_cuisines=preferred_cuisines,
            avoided_cuisines=avoided_cuisines,
            preferred_scenes=preferred_scenes,
            recent_cuisines=recent_focus_cuisines,
            recent_scenes=recent_scenes,
            ambivalent_cuisines=ambivalent_cuisines,
            service_risk_sensitivity=service_risk_sensitivity,
            confidence_overall=confidence_overall,
            recent_shift_value=recent_shift_value,
            positive_event_refs=positive_event_refs,
            negative_event_refs=negative_event_refs,
            positive_sentence_refs=positive_sentence_refs,
            negative_sentence_refs=negative_sentence_refs,
        )

        target = {
            "stable_preferences": {
                "preferred_cuisines": preferred_cuisines,
                "preferred_meals": preferred_meals,
                "preferred_scenes": preferred_scenes,
                "preferred_properties": preferred_properties,
                "top_city": safe_text(row.get("top_city")),
                "geo_style": geo_style(row.get("geo_concentration_ratio")),
            },
            "avoid_signals": {
                "avoided_cuisines": avoided_cuisines,
                "avoided_scenes": safe_list(row.get("schema_neg_scene")),
                "service_risk_sensitivity": service_risk_sensitivity,
            },
            "recent_preferences": {
                "recent_focus_cuisines": recent_focus_cuisines,
                "recent_focus_meals": recent_meals,
                "recent_focus_scenes": recent_scenes,
                "recent_focus_properties": recent_properties,
                "recent_shift": recent_shift_value,
            },
            "behavior_summary": {
                "social_mode": preferred_scenes[:3],
                "time_mode": preferred_meals[:3],
                "novelty_style": novelty_style(row.get("repeat_business_ratio")),
            },
            "evidence_refs": {
                "positive_event_refs": positive_event_refs,
                "negative_event_refs": negative_event_refs,
                "positive_sentence_refs": positive_sentence_refs,
                "negative_sentence_refs": negative_sentence_refs,
            },
            "confidence": {"overall": confidence_overall},
            "ambivalent_signals": {
                "cuisines": ambivalent_cuisines,
            },
            "open_discoveries": open_discoveries,
        }

        sft_ready = (
            bool(preferred_cuisines or preferred_meals or preferred_scenes)
            and bool(positive_sentences or positive_event_refs)
            and int(row.get("sequence_event_count", 0) or 0) >= 4
        )
        records.append(
            {
                "user_id": row["user_id"],
                "target_schema_v1_json": json.dumps(target, ensure_ascii=False),
                "sft_ready_v1": bool(sft_ready),
                "preferred_cuisine_count": int(len(preferred_cuisines)),
                "avoided_cuisine_count": int(len(avoided_cuisines)),
                "positive_sentence_count": int(len(positive_sentences)),
                "negative_sentence_count": int(len(negative_sentences)),
                "positive_event_ref_count": int(len(positive_event_refs)),
                "negative_event_ref_count": int(len(negative_event_refs)),
                "sequence_event_count": int(row.get("sequence_event_count", 0) or 0),
                "confidence_overall": confidence_overall,
                "open_discoveries_count": int(len(open_discoveries)),
            }
        )

    out_df = pd.DataFrame(records).sort_values("user_id").reset_index(drop=True)

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(run_dir / "user_preference_target_schema_v1.parquet", index=False)

    summary = {
        "rows": int(len(out_df)),
        "sft_ready_ratio_v1": float(out_df["sft_ready_v1"].mean()) if len(out_df) else 0.0,
        "mean_preferred_cuisine_count": float(out_df["preferred_cuisine_count"].mean()) if len(out_df) else 0.0,
        "mean_avoided_cuisine_count": float(out_df["avoided_cuisine_count"].mean()) if len(out_df) else 0.0,
        "mean_positive_sentence_count": float(out_df["positive_sentence_count"].mean()) if len(out_df) else 0.0,
        "mean_negative_sentence_count": float(out_df["negative_sentence_count"].mean()) if len(out_df) else 0.0,
        "mean_sequence_event_count": float(out_df["sequence_event_count"].mean()) if len(out_df) else 0.0,
        "mean_open_discoveries_count": float(out_df["open_discoveries_count"].mean()) if len(out_df) else 0.0,
        "confidence_counts": out_df["confidence_overall"].value_counts(dropna=False).to_dict(),
    }
    safe_json_write(run_dir / "summary.json", summary)

    sample_rows = []
    for row in out_df.head(SAMPLE_ROWS).to_dict(orient="records"):
        row = dict(row)
        row["target_schema_v1_json"] = json.loads(row["target_schema_v1_json"])
        sample_rows.append(row)
    safe_json_write(run_dir / "sample.json", sample_rows)

    run_meta = {
        "run_tag": RUN_TAG,
        "user_pref_run_dir": str(user_pref_run),
        "sequence_run_dir": str(sequence_run),
        "output_run_dir": str(run_dir),
        "summary": summary,
    }
    safe_json_write(run_dir / "run_meta.json", run_meta)
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
