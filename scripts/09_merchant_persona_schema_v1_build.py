from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

MERCHANT_CARD_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_CARD_ROOT_DIR",
    "data/output/09_merchant_semantic_card",
)
MERCHANT_CARD_RUN_DIR = os.getenv("INPUT_09_MERCHANT_CARD_RUN_DIR", "").strip()

MERCHANT_TEXT_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_TEXT_V3_ROOT_DIR",
    "data/output/09_merchant_text_views_v3",
)
MERCHANT_TEXT_RUN_DIR = os.getenv("INPUT_09_MERCHANT_TEXT_V3_RUN_DIR", "").strip()

MERCHANT_STRUCT_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_ROOT_DIR",
    "data/output/09_merchant_structured_text_views_v2",
)
MERCHANT_STRUCT_RUN_DIR = os.getenv("INPUT_09_MERCHANT_STRUCTURED_TEXT_V2_RUN_DIR", "").strip()

REVIEW_EVIDENCE_ROOT = env_or_project_path(
    "INPUT_09_REVIEW_EVIDENCE_ROOT_DIR",
    "data/output/09_review_evidence",
)
REVIEW_EVIDENCE_RUN_DIR = os.getenv("INPUT_09_REVIEW_EVIDENCE_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_MERCHANT_PERSONA_SCHEMA_V1_ROOT_DIR",
    "data/output/09_merchant_persona_schema_v1",
)
RUN_TAG = "stage09_merchant_persona_schema_v1_build"
SAMPLE_ROWS = int(os.getenv("MERCHANT_PERSONA_SCHEMA_V1_SAMPLE_ROWS", "12").strip() or 12)

MEAL_COLS = [
    "meal_breakfast_fit",
    "meal_lunch_fit",
    "meal_dinner_fit",
    "late_night_fit",
]
MEAL_LABELS = {
    "meal_breakfast_fit": "breakfast",
    "meal_lunch_fit": "lunch",
    "meal_dinner_fit": "dinner",
    "late_night_fit": "late_night",
}
SCENE_COLS = [
    "family_scene_fit",
    "group_scene_fit",
    "date_scene_fit",
    "nightlife_scene_fit",
    "fast_casual_fit",
    "sitdown_fit",
]
SCENE_LABELS = {
    "family_scene_fit": "family_friendly",
    "group_scene_fit": "group_dining",
    "date_scene_fit": "date_night",
    "nightlife_scene_fit": "nightlife",
    "fast_casual_fit": "quick_bite",
    "sitdown_fit": "sit_down",
}
PROPERTY_COLS = [
    "attr_delivery",
    "attr_takeout",
    "attr_reservations",
    "attr_groups",
    "attr_good_for_kids",
    "attr_outdoor",
    "attr_table_service",
    "attr_alcohol_full_bar",
    "attr_alcohol_beer_wine",
    "attr_attire_casual",
    "attr_attire_upscale",
    "parking_lot",
    "parking_street",
]
PROPERTY_LABELS = {
    "attr_delivery": "delivery",
    "attr_takeout": "takeout",
    "attr_reservations": "reservations",
    "attr_groups": "good_for_groups",
    "attr_good_for_kids": "good_for_kids",
    "attr_outdoor": "outdoor_seating",
    "attr_table_service": "table_service",
    "attr_alcohol_full_bar": "full_bar",
    "attr_alcohol_beer_wine": "beer_and_wine",
    "attr_attire_casual": "casual_vibe",
    "attr_attire_upscale": "upscale_vibe",
    "parking_lot": "parking_lot",
    "parking_street": "street_parking",
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


def top_labels(row: pd.Series, cols: list[str], labels: dict[str, str], topk: int, min_score: float) -> list[str]:
    scored = []
    for col in cols:
        val = float(row.get(col, 0.0) or 0.0)
        if val >= min_score:
            scored.append((labels[col], val))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in scored[:topk]]


def property_labels(row: pd.Series) -> list[str]:
    out = []
    for col in PROPERTY_COLS:
        val = row.get(col, 0)
        try:
            keep = float(val) > 0.5
        except Exception:
            keep = bool(val)
        if keep:
            out.append(PROPERTY_LABELS[col])
    return out


def reliability_band(row: pd.Series) -> str:
    neg = float(row.get("review_negative_pressure", 0.0) or 0.0)
    high_vote = float(row.get("review_high_vote_share", 0.0) or 0.0)
    if neg <= 0.08 and high_vote >= 0.2:
        return "high"
    if neg <= 0.16:
        return "medium"
    return "low"


def value_band(row: pd.Series) -> str:
    stars = float(row.get("avg_review_stars", 0.0) or 0.0)
    recommend = float(row.get("tip_recommend_share", 0.0) or 0.0)
    if stars >= 4.2 and recommend >= 0.22:
        return "high_value"
    if stars >= 3.7:
        return "mid_value"
    return "uncertain"


def main() -> None:
    card_run = resolve_run(MERCHANT_CARD_RUN_DIR, MERCHANT_CARD_ROOT, "_full_stage09_merchant_semantic_card_build")
    text_run = resolve_run(MERCHANT_TEXT_RUN_DIR, MERCHANT_TEXT_ROOT, "_full_stage09_merchant_text_views_v3_build")
    struct_run = resolve_run(
        MERCHANT_STRUCT_RUN_DIR,
        MERCHANT_STRUCT_ROOT,
        "_full_stage09_merchant_structured_text_views_v2_build",
    )
    review_run = resolve_run(
        REVIEW_EVIDENCE_RUN_DIR,
        REVIEW_EVIDENCE_ROOT,
        "_full_stage09_review_evidence_weight_build",
    )

    card = pd.read_parquet(card_run / "merchant_semantic_card_v2.parquet")
    text = pd.read_parquet(text_run / "merchant_text_views_v3.parquet")
    struct = pd.read_parquet(struct_run / "merchant_structured_text_views_v2.parquet")
    review = pd.read_parquet(review_run / "business_review_evidence_agg.parquet")

    card["business_id"] = card["business_id"].astype(str)
    text["business_id"] = text["business_id"].astype(str)
    struct["business_id"] = struct["business_id"].astype(str)
    review["business_id"] = review["business_id"].astype(str)

    df = (
        card.merge(text, on="business_id", how="left")
        .merge(struct, on="business_id", how="left")
        .merge(review, on="business_id", how="left", suffixes=("", "_review"))
    )

    df["persona_meal_tags"] = df.apply(
        lambda r: top_labels(r, MEAL_COLS, MEAL_LABELS, topk=3, min_score=0.35),
        axis=1,
    )
    df["persona_scene_tags"] = df.apply(
        lambda r: top_labels(r, SCENE_COLS, SCENE_LABELS, topk=4, min_score=0.35),
        axis=1,
    )
    df["persona_property_tags"] = df.apply(property_labels, axis=1)
    df["persona_dish_tags"] = df["top_dishes_v2"].apply(safe_list)
    df["persona_service_tags"] = df["top_services_v2"].apply(safe_list)
    df["persona_complaint_tags"] = df["top_complaints_v2"].apply(safe_list)
    df["persona_time_tags"] = df["top_times_v2"].apply(safe_list)
    df["persona_structured_property_tags"] = df["top_properties_v2"].apply(safe_list)
    df["persona_quality_band"] = df.apply(value_band, axis=1)
    df["persona_reliability_band"] = df.apply(reliability_band, axis=1)

    out = df[
        [
            "business_id",
            "name",
            "city",
            "state",
            "primary_category",
            "merchant_primary_cuisine",
            "merchant_secondary_cuisine",
            "geo_cell_3dp",
            "avg_review_stars",
            "review_high_vote_share",
            "review_negative_pressure",
            "tip_recommend_share",
            "semantic_support_per_review",
            "audit_review_count",
            "audit_tip_rows",
            "audit_checkin_rows",
            "persona_quality_band",
            "persona_reliability_band",
            "persona_meal_tags",
            "persona_scene_tags",
            "persona_property_tags",
            "persona_dish_tags",
            "persona_service_tags",
            "persona_complaint_tags",
            "persona_time_tags",
            "persona_structured_property_tags",
            "merchant_core_text_v3",
            "merchant_semantic_text_v3",
            "merchant_pos_text_v3",
            "merchant_neg_text_v3",
            "merchant_context_text_v3",
            "core_text_len_v3",
            "semantic_text_len_v3",
            "pos_text_len_v3",
            "neg_text_len_v3",
            "context_text_len_v3",
        ]
    ].copy()

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "merchant_persona_schema_v1.parquet"
    summary_path = run_dir / "merchant_persona_schema_v1_summary.json"
    sample_path = run_dir / "merchant_persona_schema_v1_sample.json"
    run_meta_path = run_dir / "run_meta.json"

    out.to_parquet(out_path, index=False)

    coverage = {
        "rows": int(len(out)),
        "has_primary_cuisine": float(out["merchant_primary_cuisine"].fillna("").ne("").mean()),
        "has_secondary_cuisine": float(out["merchant_secondary_cuisine"].fillna("").ne("").mean()),
        "has_meal_tags": float(out["persona_meal_tags"].map(bool).mean()),
        "has_scene_tags": float(out["persona_scene_tags"].map(bool).mean()),
        "has_dish_tags": float(out["persona_dish_tags"].map(bool).mean()),
        "has_service_tags": float(out["persona_service_tags"].map(bool).mean()),
        "has_complaint_tags": float(out["persona_complaint_tags"].map(bool).mean()),
        "has_time_tags": float(out["persona_time_tags"].map(bool).mean()),
        "has_property_tags": float(out["persona_property_tags"].map(bool).mean()),
        "has_core_text": float(out["merchant_core_text_v3"].fillna("").ne("").mean()),
        "has_semantic_text": float(out["merchant_semantic_text_v3"].fillna("").ne("").mean()),
        "has_context_text": float(out["merchant_context_text_v3"].fillna("").ne("").mean()),
    }
    safe_json_write(summary_path, coverage)
    safe_json_write(sample_path, out.head(SAMPLE_ROWS).to_dict(orient="records"))
    safe_json_write(
        run_meta_path,
        {
            "run_tag": RUN_TAG,
            "inputs": {
                "merchant_card_run": str(card_run),
                "merchant_text_run": str(text_run),
                "merchant_structured_run": str(struct_run),
                "review_evidence_run": str(review_run),
            },
            "outputs": {
                "merchant_persona_schema_v1": str(out_path),
                "summary_json": str(summary_path),
                "sample_json": str(sample_path),
            },
            "coverage": coverage,
        },
    )
    print(json.dumps({"run_dir": str(run_dir), **coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
