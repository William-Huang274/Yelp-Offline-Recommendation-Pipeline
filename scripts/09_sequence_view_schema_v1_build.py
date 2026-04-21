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

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

INTERACTION_ROOT = env_or_project_path(
    "INPUT_09_INTERACTION_V2_WEIGHTED_ROOT_DIR",
    "data/output/09_interaction_v2_weighted",
)
INTERACTION_RUN_DIR = os.getenv("INPUT_09_INTERACTION_V2_WEIGHTED_RUN_DIR", "").strip()

USER_PREF_ROOT = env_or_project_path(
    "INPUT_09_USER_PREFERENCE_SCHEMA_V1_ROOT_DIR",
    "data/output/09_user_preference_schema_v1",
)
USER_PREF_RUN_DIR = os.getenv("INPUT_09_USER_PREFERENCE_SCHEMA_V1_RUN_DIR", "").strip()

MERCHANT_PERSONA_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_ROOT_DIR",
    "data/output/09_merchant_persona_schema_v1",
)
MERCHANT_PERSONA_RUN_DIR = os.getenv("INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_SEQUENCE_VIEW_SCHEMA_V1_ROOT_DIR",
    "data/output/09_sequence_view_schema_v1",
)
RUN_TAG = "stage09_sequence_view_schema_v1_build"
SAMPLE_ROWS = int(os.getenv("SEQUENCE_VIEW_SCHEMA_V1_SAMPLE_ROWS", "12").strip() or 12)
RECENT_K = int(os.getenv("SEQUENCE_VIEW_SCHEMA_V1_RECENT_K", "8").strip() or 8)
ANCHOR_POS_K = int(os.getenv("SEQUENCE_VIEW_SCHEMA_V1_ANCHOR_POS_K", "3").strip() or 3)
ANCHOR_NEG_K = int(os.getenv("SEQUENCE_VIEW_SCHEMA_V1_ANCHOR_NEG_K", "2").strip() or 2)


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


def safe_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        items = list(value)
    except TypeError:
        return []
    out = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def dominant_label(series) -> str:
    counts = {}
    for value in series:
        text = str(value or "").strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def main() -> None:
    interaction_run = resolve_run(
        INTERACTION_RUN_DIR,
        INTERACTION_ROOT,
        "_full_stage09_interaction_v2_weight_build",
    )
    user_pref_run = resolve_run(
        USER_PREF_RUN_DIR,
        USER_PREF_ROOT,
        "_full_stage09_user_preference_schema_v1_build",
    )
    merchant_persona_run = resolve_run(
        MERCHANT_PERSONA_RUN_DIR,
        MERCHANT_PERSONA_ROOT,
        "_full_stage09_merchant_persona_schema_v1_build",
    )

    user_pref = pd.read_parquet(user_pref_run / "user_preference_schema_v1.parquet")
    merchant = pd.read_parquet(merchant_persona_run / "merchant_persona_schema_v1.parquet")
    events = pd.read_parquet(
        interaction_run / "interaction_v2_weighted.parquet",
        columns=[
            "event_id",
            "event_type",
            "event_time",
            "user_id",
            "business_id",
            "sample_weight_v2",
            "is_trainable_cohort",
        ],
    )

    user_pref["user_id"] = user_pref["user_id"].astype(str)
    merchant["business_id"] = merchant["business_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)
    events["business_id"] = events["business_id"].astype(str)

    for col in [
        "meal_tags",
        "scene_tags",
        "recent_meal_tags",
        "recent_scene_tags",
        "positive_evidence_sentences",
        "negative_evidence_sentences",
    ]:
        if col in user_pref.columns:
            user_pref[col] = user_pref[col].apply(safe_list)

    users = set(user_pref["user_id"].tolist())
    events = events.loc[events["is_trainable_cohort"].fillna(0).eq(1) & events["user_id"].isin(users)].copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
    events["sample_weight_v2"] = pd.to_numeric(events["sample_weight_v2"], errors="coerce").fillna(0.0)
    events = events.merge(
        merchant[
            [
                "business_id",
                "name",
                "city",
                "merchant_primary_cuisine",
                "merchant_secondary_cuisine",
                "persona_meal_tags",
                "persona_scene_tags",
                "persona_property_tags",
                "persona_quality_band",
                "persona_reliability_band",
                "merchant_core_text_v3",
            ]
        ],
        on="business_id",
        how="left",
    )

    events = events.sort_values(["user_id", "event_time"], ascending=[True, False]).copy()
    events["recent_rank"] = events.groupby("user_id").cumcount() + 1
    recent = events.loc[events["recent_rank"] <= RECENT_K].copy()

    recent["merchant_core_text_short"] = recent["merchant_core_text_v3"].fillna("").astype(str).str.slice(0, 180)
    recent["persona_meal_tags"] = recent["persona_meal_tags"].apply(safe_list)
    recent["persona_scene_tags"] = recent["persona_scene_tags"].apply(safe_list)
    recent["persona_property_tags"] = recent["persona_property_tags"].apply(safe_list)

    recent_events_out = recent[
        [
            "user_id",
            "recent_rank",
            "event_id",
            "event_type",
            "event_time",
            "sample_weight_v2",
            "business_id",
            "name",
            "city",
            "merchant_primary_cuisine",
            "merchant_secondary_cuisine",
            "persona_meal_tags",
            "persona_scene_tags",
            "persona_property_tags",
            "persona_quality_band",
            "persona_reliability_band",
            "merchant_core_text_short",
        ]
    ].copy()

    group = recent.groupby("user_id", sort=False)
    seq_summary = group.agg(
        sequence_event_count=("event_id", "count"),
        sequence_positive_count=("event_type", lambda s: int((s == "review_positive").sum())),
        sequence_negative_count=("event_type", lambda s: int((s == "review_negative").sum())),
        sequence_tip_count=("event_type", lambda s: int((s == "tip_signal").sum())),
        sequence_neutral_count=("event_type", lambda s: int((s == "review_neutral").sum())),
        unique_business_count=("business_id", "nunique"),
        dominant_recent_cuisine=("merchant_primary_cuisine", dominant_label),
        dominant_recent_city=("city", dominant_label),
        latest_event_time=("event_time", "max"),
        earliest_recent_event_time=("event_time", "min"),
    ).reset_index()
    seq_summary["repeat_business_ratio"] = 1.0 - (
        seq_summary["unique_business_count"] / seq_summary["sequence_event_count"].clip(lower=1)
    )
    seq_summary["sequence_span_days"] = (
        seq_summary["latest_event_time"] - seq_summary["earliest_recent_event_time"]
    ).dt.days.fillna(0).astype(int)

    def agg_json(df: pd.DataFrame, topk: int | None = None) -> str:
        if topk is not None:
            df = df.head(topk)
        payload = df[
            [
                "recent_rank",
                "event_type",
                "event_time",
                "business_id",
                "name",
                "city",
                "merchant_primary_cuisine",
                "merchant_secondary_cuisine",
                "persona_meal_tags",
                "persona_scene_tags",
                "persona_property_tags",
                "persona_quality_band",
                "persona_reliability_band",
                "merchant_core_text_short",
            ]
        ].copy()
        payload["event_time"] = payload["event_time"].astype(str)
        return json.dumps(payload.to_dict(orient="records"), ensure_ascii=False)

    seq_json_rows = []
    for user_id, user_df in group:
        pos_df = user_df.loc[user_df["event_type"].eq("review_positive")].sort_values(
            ["sample_weight_v2", "recent_rank"], ascending=[False, True]
        )
        neg_df = user_df.loc[user_df["event_type"].eq("review_negative")].sort_values(
            ["sample_weight_v2", "recent_rank"], ascending=[False, True]
        )
        seq_json_rows.append(
            {
                "user_id": user_id,
                "recent_event_sequence_json": agg_json(user_df),
                "positive_anchor_sequence_json": agg_json(pos_df, topk=ANCHOR_POS_K),
                "negative_anchor_sequence_json": agg_json(neg_df, topk=ANCHOR_NEG_K),
            }
        )
    seq_json_df = pd.DataFrame(seq_json_rows)

    out = user_pref[
        [
            "user_id",
            "n_train_max",
            "n_events_trainable",
            "n_businesses_trainable",
            "long_term_top_cuisine",
            "recent_top_cuisine",
            "negative_top_cuisine",
            "top_city",
            "meal_tags",
            "scene_tags",
            "recent_meal_tags",
            "recent_scene_tags",
            "positive_evidence_sentences",
            "negative_evidence_sentences",
        ]
    ].merge(seq_summary, on="user_id", how="left").merge(seq_json_df, on="user_id", how="left")

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    wide_path = run_dir / "sequence_view_schema_v1.parquet"
    long_path = run_dir / "sequence_view_events_v1.parquet"
    summary_path = run_dir / "sequence_view_schema_v1_summary.json"
    sample_path = run_dir / "sequence_view_schema_v1_sample.json"
    run_meta_path = run_dir / "run_meta.json"

    out.to_parquet(wide_path, index=False)
    recent_events_out.to_parquet(long_path, index=False)

    coverage = {
        "rows": int(len(out)),
        "recent_event_rows": int(len(recent_events_out)),
        "users_with_recent_events": float(out["sequence_event_count"].fillna(0).gt(0).mean()),
        "users_with_full_recent_k": float(out["sequence_event_count"].fillna(0).ge(RECENT_K).mean()),
        "users_with_positive_anchor": float(out["positive_anchor_sequence_json"].fillna("[]").ne("[]").mean()),
        "users_with_negative_anchor": float(out["negative_anchor_sequence_json"].fillna("[]").ne("[]").mean()),
    }
    safe_json_write(summary_path, coverage)
    safe_json_write(sample_path, out.head(SAMPLE_ROWS).to_dict(orient="records"))
    safe_json_write(
        run_meta_path,
        {
            "run_tag": RUN_TAG,
            "inputs": {
                "interaction_run": str(interaction_run),
                "user_preference_run": str(user_pref_run),
                "merchant_persona_run": str(merchant_persona_run),
            },
            "outputs": {
                "sequence_view_schema_v1": str(wide_path),
                "sequence_view_events_v1": str(long_path),
                "summary_json": str(summary_path),
                "sample_json": str(sample_path),
            },
            "coverage": coverage,
            "recent_k": RECENT_K,
        },
    )
    print(json.dumps({"run_dir": str(run_dir), **coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
