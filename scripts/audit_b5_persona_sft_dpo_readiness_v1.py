from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

MERCHANT_PERSONA_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_ROOT_DIR",
    "data/output/09_merchant_persona_schema_v1",
)
MERCHANT_PERSONA_RUN_DIR = os.getenv("INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_RUN_DIR", "").strip()

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
    "OUTPUT_09_PERSONA_SFT_DPO_READINESS_AUDIT_V1_ROOT_DIR",
    "data/output/09_persona_sft_dpo_readiness_audit_v1",
)
RUN_TAG = "stage09_persona_sft_dpo_readiness_audit_v1"


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = project_path(raw)
        if not p.exists():
            raise FileNotFoundError(f"run dir not found: {p}")
        return p
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def safe_json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def has_values(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except Exception:
        return bool(value)


def list_nonempty_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(df[col].map(has_values).mean())


def text_nonempty_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(df[col].fillna("").astype(str).str.strip().ne("").mean())


def main() -> None:
    merchant_run = resolve_run(
        MERCHANT_PERSONA_RUN_DIR,
        MERCHANT_PERSONA_ROOT,
        "_full_stage09_merchant_persona_schema_v1_build",
    )
    user_run = resolve_run(
        USER_PREF_RUN_DIR,
        USER_PREF_ROOT,
        "_full_stage09_user_preference_schema_v1_build",
    )
    seq_run = resolve_run(
        SEQUENCE_RUN_DIR,
        SEQUENCE_ROOT,
        "_full_stage09_sequence_view_schema_v1_build",
    )

    merchant = pd.read_parquet(merchant_run / "merchant_persona_schema_v1.parquet")
    user_pref = pd.read_parquet(user_run / "user_preference_schema_v1.parquet")
    seq = pd.read_parquet(seq_run / "sequence_view_schema_v1.parquet")

    merchant["business_id"] = merchant["business_id"].astype(str)
    user_pref["user_id"] = user_pref["user_id"].astype(str)
    seq["user_id"] = seq["user_id"].astype(str)

    merged = user_pref.merge(seq, on="user_id", how="left", suffixes=("", "_seq"))

    merged["sft_ready_v1"] = (
        merged["long_term_top_cuisine"].fillna("").ne("")
        & merged["meal_tags"].map(has_values)
        & merged["scene_tags"].map(has_values)
        & merged["schema_pos_cuisine"].map(has_values)
        & merged["positive_evidence_sentences"].map(has_values)
        & merged["sequence_event_count"].fillna(0).ge(5)
    )
    merged["dpo_ready_v1"] = (
        merged["sft_ready_v1"]
        & merged["negative_top_cuisine"].fillna("").ne("")
        & merged["negative_evidence_sentences"].map(has_values)
        & merged["sequence_negative_count"].fillna(0).ge(1)
        & merged["sequence_positive_count"].fillna(0).ge(2)
    )

    merchant_metrics = {
        "rows": int(len(merchant)),
        "has_primary_cuisine": text_nonempty_ratio(merchant, "merchant_primary_cuisine"),
        "has_meal_tags": list_nonempty_ratio(merchant, "persona_meal_tags"),
        "has_scene_tags": list_nonempty_ratio(merchant, "persona_scene_tags"),
        "has_dish_tags": list_nonempty_ratio(merchant, "persona_dish_tags"),
        "has_service_tags": list_nonempty_ratio(merchant, "persona_service_tags"),
        "has_complaint_tags": list_nonempty_ratio(merchant, "persona_complaint_tags"),
        "has_time_tags": list_nonempty_ratio(merchant, "persona_time_tags"),
        "has_core_text": text_nonempty_ratio(merchant, "merchant_core_text_v3"),
        "has_semantic_text": text_nonempty_ratio(merchant, "merchant_semantic_text_v3"),
        "has_context_text": text_nonempty_ratio(merchant, "merchant_context_text_v3"),
    }
    user_metrics = {
        "rows": int(len(user_pref)),
        "has_long_term_top_cuisine": text_nonempty_ratio(user_pref, "long_term_top_cuisine"),
        "has_negative_top_cuisine": text_nonempty_ratio(user_pref, "negative_top_cuisine"),
        "has_recent_top_cuisine": text_nonempty_ratio(user_pref, "recent_top_cuisine"),
        "has_meal_tags": list_nonempty_ratio(user_pref, "meal_tags"),
        "has_scene_tags": list_nonempty_ratio(user_pref, "scene_tags"),
        "has_property_tags": list_nonempty_ratio(user_pref, "property_tags"),
        "has_schema_pos_cuisine": list_nonempty_ratio(user_pref, "schema_pos_cuisine"),
        "has_schema_neg_cuisine": list_nonempty_ratio(user_pref, "schema_neg_cuisine"),
        "has_profile_text_short": text_nonempty_ratio(user_pref, "profile_text_short"),
        "has_user_long_pref_text": text_nonempty_ratio(user_pref, "user_long_pref_text"),
        "has_recent_intent_text": text_nonempty_ratio(user_pref, "user_recent_intent_text"),
        "has_negative_avoid_text": text_nonempty_ratio(user_pref, "user_negative_avoid_text"),
    }
    sequence_metrics = {
        "rows": int(len(seq)),
        "users_with_recent_events": float(seq["sequence_event_count"].fillna(0).gt(0).mean()),
        "users_with_full_recent_k": float(seq["sequence_event_count"].fillna(0).ge(8).mean()),
        "users_with_positive_anchor": text_nonempty_ratio(seq, "positive_anchor_sequence_json"),
        "users_with_negative_anchor": text_nonempty_ratio(seq, "negative_anchor_sequence_json"),
        "mean_recent_event_count": float(seq["sequence_event_count"].fillna(0).mean()),
    }

    readiness = {
        "sft_ready_ratio_v1": float(merged["sft_ready_v1"].mean()),
        "dpo_ready_ratio_v1": float(merged["dpo_ready_v1"].mean()),
    }
    go_sft = (
        merchant_metrics["has_scene_tags"] >= 0.90
        and merchant_metrics["has_core_text"] >= 0.99
        and user_metrics["has_meal_tags"] >= 0.90
        and user_metrics["has_schema_pos_cuisine"] >= 0.85
        and sequence_metrics["users_with_full_recent_k"] >= 0.80
        and readiness["sft_ready_ratio_v1"] >= 0.45
    )
    go_dpo = go_sft and readiness["dpo_ready_ratio_v1"] >= 0.30

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "persona_sft_dpo_readiness_audit_v1.json"
    sample_path = run_dir / "persona_sft_dpo_readiness_sample_v1.json"
    run_meta_path = run_dir / "run_meta.json"

    safe_json_write(
        summary_path,
        {
            "merchant_metrics": merchant_metrics,
            "user_metrics": user_metrics,
            "sequence_metrics": sequence_metrics,
            "readiness": readiness,
            "decision": {
                "go_sft": bool(go_sft),
                "go_dpo": bool(go_dpo),
            },
        },
    )
    safe_json_write(
        sample_path,
        merged[
            [
                "user_id",
                "n_train_max",
                "long_term_top_cuisine",
                "negative_top_cuisine",
                "meal_tags",
                "scene_tags",
                "sequence_event_count",
                "sequence_positive_count",
                "sequence_negative_count",
                "sft_ready_v1",
                "dpo_ready_v1",
            ]
        ].head(20).to_dict(orient="records"),
    )
    safe_json_write(
        run_meta_path,
        {
            "run_tag": RUN_TAG,
            "inputs": {
                "merchant_persona_run": str(merchant_run),
                "user_preference_run": str(user_run),
                "sequence_view_run": str(seq_run),
            },
            "outputs": {
                "audit_json": str(summary_path),
                "sample_json": str(sample_path),
            },
        },
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
