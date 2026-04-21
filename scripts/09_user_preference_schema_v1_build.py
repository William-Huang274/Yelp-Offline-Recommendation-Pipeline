from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

USER_INTENT_ROOT = env_or_project_path(
    "INPUT_09_USER_INTENT_PROFILE_V2_ROOT_DIR",
    "data/output/09_user_intent_profile_v2",
)
USER_INTENT_RUN_DIR = os.getenv("INPUT_09_USER_INTENT_PROFILE_V2_RUN_DIR", "").strip()

USER_SCHEMA_ROOT = env_or_project_path(
    "INPUT_09_USER_SCHEMA_PROJECTION_V2_ROOT_DIR",
    "data/output/09_user_schema_projection_v2",
)
USER_SCHEMA_RUN_DIR = os.getenv("INPUT_09_USER_SCHEMA_PROJECTION_V2_RUN_DIR", "").strip()

USER_PROFILES_ROOT = env_or_project_path(
    "INPUT_09_USER_PROFILES_ROOT_DIR",
    "data/output/09_user_profiles",
)
USER_PROFILES_RUN_DIR = os.getenv("INPUT_09_USER_PROFILES_RUN_DIR", "").strip()

USER_TEXT_ROOT = env_or_project_path(
    "INPUT_09_CANDIDATE_WISE_TEXT_VIEWS_V1_ROOT_DIR",
    "data/output/09_candidate_wise_text_views_v1",
)
USER_TEXT_RUN_DIR = os.getenv("INPUT_09_CANDIDATE_WISE_TEXT_VIEWS_V1_RUN_DIR", "").strip()

REVIEW_EVIDENCE_ROOT = env_or_project_path(
    "INPUT_09_REVIEW_EVIDENCE_ROOT_DIR",
    "data/output/09_review_evidence",
)
REVIEW_EVIDENCE_RUN_DIR = os.getenv("INPUT_09_REVIEW_EVIDENCE_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_USER_PREFERENCE_SCHEMA_V1_ROOT_DIR",
    "data/output/09_user_preference_schema_v1",
)
RUN_TAG = "stage09_user_preference_schema_v1_build"
SAMPLE_ROWS = int(os.getenv("USER_PREFERENCE_SCHEMA_V1_SAMPLE_ROWS", "12").strip() or 12)
EVIDENCE_TOPK_POS = int(os.getenv("USER_PREFERENCE_SCHEMA_V1_EVIDENCE_TOPK_POS", "4").strip() or 4)
EVIDENCE_TOPK_NEG = int(os.getenv("USER_PREFERENCE_SCHEMA_V1_EVIDENCE_TOPK_NEG", "3").strip() or 3)

PREF_GROUPS = {
    "meal": [
        "breakfast_pref",
        "lunch_pref",
        "dinner_pref",
        "late_night_pref",
    ],
    "scene": [
        "family_scene_pref",
        "group_scene_pref",
        "date_scene_pref",
        "nightlife_scene_pref",
        "fast_casual_pref",
        "sitdown_pref",
    ],
    "property": [
        "delivery_pref",
        "takeout_pref",
        "reservation_pref",
        "weekend_pref",
    ],
    "recent_meal": [
        "recent_breakfast_pref",
        "recent_lunch_pref",
        "recent_dinner_pref",
        "recent_late_night_pref",
    ],
    "recent_scene": [
        "recent_family_scene_pref",
        "recent_group_scene_pref",
        "recent_date_scene_pref",
        "recent_nightlife_scene_pref",
        "recent_fast_casual_pref",
        "recent_sitdown_pref",
    ],
    "recent_property": [
        "recent_delivery_pref",
        "recent_takeout_pref",
        "recent_reservation_pref",
        "recent_weekend_pref",
    ],
}

PREF_LABELS = {
    "breakfast_pref": "breakfast",
    "lunch_pref": "lunch",
    "dinner_pref": "dinner",
    "late_night_pref": "late_night",
    "family_scene_pref": "family_friendly",
    "group_scene_pref": "group_dining",
    "date_scene_pref": "date_night",
    "nightlife_scene_pref": "nightlife",
    "fast_casual_pref": "quick_bite",
    "sitdown_pref": "sit_down",
    "delivery_pref": "delivery",
    "takeout_pref": "takeout",
    "reservation_pref": "reservations",
    "weekend_pref": "weekend",
    "recent_breakfast_pref": "breakfast",
    "recent_lunch_pref": "lunch",
    "recent_dinner_pref": "dinner",
    "recent_late_night_pref": "late_night",
    "recent_family_scene_pref": "family_friendly",
    "recent_group_scene_pref": "group_dining",
    "recent_date_scene_pref": "date_night",
    "recent_nightlife_scene_pref": "nightlife",
    "recent_fast_casual_pref": "quick_bite",
    "recent_sitdown_pref": "sit_down",
    "recent_delivery_pref": "delivery",
    "recent_takeout_pref": "takeout",
    "recent_reservation_pref": "reservations",
    "recent_weekend_pref": "weekend",
}


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


def top_pref_tags(row: pd.Series, cols: list[str], min_score: float = 0.35, topk: int = 4) -> list[str]:
    scored = []
    for col in cols:
        val = float(row.get(col, 0.0) or 0.0)
        if val >= min_score:
            scored.append((PREF_LABELS[col], val))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in scored[:topk]]


def collect_evidence(df: pd.DataFrame, polarity: int, topk: int) -> pd.DataFrame:
    sub = df.loc[df["polarity"].eq(polarity)].copy()
    sub["abs_weight"] = sub["final_weight"].abs()
    sub = sub.sort_values(["user_id", "abs_weight", "sentence_rank"], ascending=[True, False, True])
    sub = sub.drop_duplicates(["user_id", "sentence_text"], keep="first")
    sub["rn"] = sub.groupby("user_id").cumcount() + 1
    sub = sub.loc[sub["rn"] <= topk]
    agg = sub.groupby("user_id")["sentence_text"].agg(list).reset_index()
    col = "positive_evidence_sentences" if polarity > 0 else "negative_evidence_sentences"
    return agg.rename(columns={"sentence_text": col})


def main() -> None:
    user_intent_run = resolve_run(
        USER_INTENT_RUN_DIR,
        USER_INTENT_ROOT,
        "_full_stage09_user_intent_profile_split_aware_v2_build",
    )
    user_schema_run = resolve_run(
        USER_SCHEMA_RUN_DIR,
        USER_SCHEMA_ROOT,
        "_full_stage09_user_schema_projection_v2_build",
    )
    user_profiles_run = resolve_run(
        USER_PROFILES_RUN_DIR,
        USER_PROFILES_ROOT,
        "_full_stage09_user_profile_build",
    )
    user_text_run = resolve_run(
        USER_TEXT_RUN_DIR,
        USER_TEXT_ROOT,
        "_full_stage09_candidate_wise_text_views_v1_build",
    )
    review_run = resolve_run(
        REVIEW_EVIDENCE_RUN_DIR,
        REVIEW_EVIDENCE_ROOT,
        "_full_stage09_review_evidence_weight_build",
    )

    intent = pd.read_parquet(user_intent_run / "user_intent_profile_v2.parquet")
    schema = pd.read_parquet(user_schema_run / "user_schema_profile_v1.parquet")
    profiles = pd.read_csv(user_profiles_run / "user_profiles.csv")
    evidence = pd.read_csv(user_profiles_run / "user_profile_evidence.csv")
    user_text = pd.read_parquet(user_text_run / "user_text_views_v1.parquet")
    review_agg = pd.read_parquet(review_run / "user_review_evidence_agg.parquet")

    for df in (intent, schema, profiles, evidence, user_text, review_agg):
        df["user_id"] = df["user_id"].astype(str)

    # bucket5 split-aware users are anchored by the intent table.
    users = intent.merge(schema, on="user_id", how="left", suffixes=("", "_schema"))
    users = users.merge(profiles, on="user_id", how="left", suffixes=("", "_profile"))
    users = users.merge(user_text, on="user_id", how="left")
    users = users.merge(review_agg, on="user_id", how="left", suffixes=("", "_review"))

    pos_evidence = collect_evidence(evidence, polarity=1, topk=EVIDENCE_TOPK_POS)
    neg_evidence = collect_evidence(evidence, polarity=-1, topk=EVIDENCE_TOPK_NEG)
    users = users.merge(pos_evidence, on="user_id", how="left").merge(neg_evidence, on="user_id", how="left")

    for group_name, cols in PREF_GROUPS.items():
        users[f"{group_name}_tags"] = users.apply(lambda r: top_pref_tags(r, cols), axis=1)

    list_cols = [
        "schema_pos_cuisine",
        "schema_pos_dish",
        "schema_pos_scene",
        "schema_pos_service",
        "schema_pos_complaint",
        "schema_pos_time",
        "schema_pos_property",
        "schema_neg_cuisine",
        "schema_neg_dish",
        "schema_neg_scene",
        "schema_neg_service",
        "schema_neg_complaint",
        "schema_neg_time",
        "schema_neg_property",
    ]
    for col in list_cols:
        if col in users.columns:
            users[col] = users[col].apply(safe_list)
    for col in [
        "positive_evidence_sentences",
        "negative_evidence_sentences",
        "profile_keywords",
        "profile_top_pos_tags",
        "profile_top_neg_tags",
    ]:
        if col in users.columns:
            users[col] = users[col].apply(safe_list)

    out_cols = [
        "user_id",
        "n_events_trainable",
        "n_businesses_trainable",
        "n_train_max",
        "profile_confidence",
        "profile_conf_interaction",
        "profile_conf_text",
        "profile_conf_freshness",
        "profile_conf_consistency",
        "n_reviews_selected",
        "n_sentences_selected",
        "review_rows",
        "avg_review_stars",
        "high_vote_review_rows",
        "top_city",
        "top_geo_cell_3dp",
        "geo_concentration_ratio",
        "city_concentration_ratio",
        "long_term_top_cuisine",
        "recent_top_cuisine",
        "recent_top_cuisine_source",
        "negative_top_cuisine",
        "negative_pressure",
        "tip_aux_share",
        "cuisine_shift_flag",
        "is_negative_heavy",
        "is_geo_concentrated",
        "meal_tags",
        "scene_tags",
        "property_tags",
        "recent_meal_tags",
        "recent_scene_tags",
        "recent_property_tags",
        "schema_pos_cuisine",
        "schema_pos_dish",
        "schema_pos_scene",
        "schema_pos_service",
        "schema_pos_complaint",
        "schema_pos_time",
        "schema_pos_property",
        "schema_neg_cuisine",
        "schema_neg_dish",
        "schema_neg_scene",
        "schema_neg_service",
        "schema_neg_complaint",
        "schema_neg_time",
        "schema_neg_property",
        "profile_keywords",
        "profile_top_pos_tags",
        "profile_top_neg_tags",
        "profile_text_short",
        "profile_text",
        "user_long_pref_text",
        "user_recent_intent_text",
        "user_negative_avoid_text",
        "user_context_text",
        "positive_evidence_sentences",
        "negative_evidence_sentences",
    ]
    out = users[out_cols].copy()

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "user_preference_schema_v1.parquet"
    summary_path = run_dir / "user_preference_schema_v1_summary.json"
    sample_path = run_dir / "user_preference_schema_v1_sample.json"
    run_meta_path = run_dir / "run_meta.json"

    out.to_parquet(out_path, index=False)

    coverage = {
        "rows": int(len(out)),
        "has_long_term_top_cuisine": float(out["long_term_top_cuisine"].fillna("").ne("").mean()),
        "has_negative_top_cuisine": float(out["negative_top_cuisine"].fillna("").ne("").mean()),
        "has_recent_top_cuisine": float(out["recent_top_cuisine"].fillna("").ne("").mean()),
        "has_meal_tags": float(out["meal_tags"].map(bool).mean()),
        "has_scene_tags": float(out["scene_tags"].map(bool).mean()),
        "has_property_tags": float(out["property_tags"].map(bool).mean()),
        "has_schema_pos_cuisine": float(out["schema_pos_cuisine"].map(bool).mean()),
        "has_schema_neg_cuisine": float(out["schema_neg_cuisine"].map(bool).mean()),
        "has_profile_text_short": float(out["profile_text_short"].fillna("").ne("").mean()),
        "has_user_long_pref_text": float(out["user_long_pref_text"].fillna("").ne("").mean()),
        "has_user_recent_intent_text": float(out["user_recent_intent_text"].fillna("").ne("").mean()),
        "has_user_negative_avoid_text": float(out["user_negative_avoid_text"].fillna("").ne("").mean()),
        "has_positive_evidence_sentences": float(out["positive_evidence_sentences"].map(lambda x: bool(x) if isinstance(x, list) else False).mean()),
        "has_negative_evidence_sentences": float(out["negative_evidence_sentences"].map(lambda x: bool(x) if isinstance(x, list) else False).mean()),
    }
    safe_json_write(summary_path, coverage)
    safe_json_write(sample_path, out.head(SAMPLE_ROWS).to_dict(orient="records"))
    safe_json_write(
        run_meta_path,
        {
            "run_tag": RUN_TAG,
            "inputs": {
                "user_intent_run": str(user_intent_run),
                "user_schema_run": str(user_schema_run),
                "user_profiles_run": str(user_profiles_run),
                "user_text_run": str(user_text_run),
                "review_evidence_run": str(review_run),
            },
            "outputs": {
                "user_preference_schema_v1": str(out_path),
                "summary_json": str(summary_path),
                "sample_json": str(sample_path),
            },
            "coverage": coverage,
        },
    )
    print(json.dumps({"run_dir": str(run_dir), **coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
