from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

MERCHANT_PERSONA_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_ROOT_DIR",
    "data/output/09_merchant_persona_schema_v1",
)
MERCHANT_PERSONA_RUN_DIR = os.getenv("INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_RUN_DIR", "").strip()

PHOTO_SUMMARY_PATH = env_or_project_path(
    "INPUT_YELP_PHOTO_BUSINESS_SUMMARY_PATH",
    "data/parquet/yelp_photo_business_summary/business_photo_summary.parquet",
)

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_MERCHANT_STATE_V2_AUDIT_ROOT_DIR",
    "data/output/09_merchant_state_v2_audit",
)
RUN_TAG = "stage09_merchant_state_v2_audit_sample"
SAMPLE_ROWS_PER_BUCKET = int(os.getenv("MERCHANT_STATE_V2_SAMPLE_ROWS_PER_BUCKET", "6").strip() or 6)


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


def safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    return [safe_text(item) for item in payload if safe_text(item)]
            except Exception:
                pass
        return [text]
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


def merchant_evidence_band(row: pd.Series) -> str:
    review_rows = float(row.get("audit_review_count", 0.0) or 0.0)
    tip_rows = float(row.get("audit_tip_rows", 0.0) or 0.0)
    semantic_len = float(row.get("semantic_text_len_v3", 0.0) or 0.0)
    if review_rows < 20 and tip_rows < 5 and semantic_len < 220:
        return "sparse"
    if review_rows < 100 and tip_rows < 20:
        return "mid"
    return "rich"


def merchant_risk_bucket(row: pd.Series) -> str:
    neg = float(row.get("review_negative_pressure", 0.0) or 0.0)
    if neg >= 0.18:
        return "high_risk"
    if neg >= 0.08:
        return "medium_risk"
    return "low_risk"


def photo_bucket(row: pd.Series) -> str:
    if pd.isna(row.get("photo_count")) or int(float(row.get("photo_count", 0) or 0)) <= 0:
        return "no_photo"
    if float(row.get("photo_caption_nonempty_ratio", 0.0) or 0.0) >= 0.35:
        return "photo_with_caption"
    return "photo_sparse_caption"


def prompt_input_payload(row: pd.Series) -> dict[str, Any]:
    photo_count = row.get("photo_count")
    review_count = row.get("audit_review_count")
    tip_rows = row.get("audit_tip_rows")
    return {
        "business_id": safe_text(row.get("business_id")),
        "name": safe_text(row.get("name")),
        "city": safe_text(row.get("city")),
        "state": safe_text(row.get("state")),
        "primary_category": safe_text(row.get("primary_category")),
        "merchant_primary_cuisine": safe_text(row.get("merchant_primary_cuisine")),
        "merchant_secondary_cuisine": safe_text(row.get("merchant_secondary_cuisine")),
        "merchant_core_text_v3": safe_text(row.get("merchant_core_text_v3")),
        "merchant_semantic_text_v3": safe_text(row.get("merchant_semantic_text_v3")),
        "merchant_pos_text_v3": safe_text(row.get("merchant_pos_text_v3")),
        "merchant_neg_text_v3": safe_text(row.get("merchant_neg_text_v3")),
        "merchant_context_text_v3": safe_text(row.get("merchant_context_text_v3")),
        "photo_caption_summary": safe_text(row.get("photo_caption_summary")),
        "photo_food_caption_summary": safe_text(row.get("photo_food_caption_summary")),
        "photo_inside_caption_summary": safe_text(row.get("photo_inside_caption_summary")),
        "photo_outside_caption_summary": safe_text(row.get("photo_outside_caption_summary")),
        "photo_menu_caption_summary": safe_text(row.get("photo_menu_caption_summary")),
        "photo_dominant_label": safe_text(row.get("photo_dominant_label")),
        "photo_count": 0 if pd.isna(photo_count) else int(float(photo_count or 0)),
        "audit_review_count": 0 if pd.isna(review_count) else int(float(review_count or 0)),
        "audit_tip_rows": 0 if pd.isna(tip_rows) else int(float(tip_rows or 0)),
        "persona_reliability_band": safe_text(row.get("persona_reliability_band")),
        "persona_quality_band": safe_text(row.get("persona_quality_band")),
    }


def reference_state_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "meal_tags": safe_list(row.get("persona_meal_tags")),
        "scene_tags": safe_list(row.get("persona_scene_tags")),
        "property_tags": safe_list(row.get("persona_property_tags")),
        "dish_tags": safe_list(row.get("persona_dish_tags")),
        "service_tags": safe_list(row.get("persona_service_tags")),
        "complaint_tags": safe_list(row.get("persona_complaint_tags")),
        "time_tags": safe_list(row.get("persona_time_tags")),
        "structured_property_tags": safe_list(row.get("persona_structured_property_tags")),
        "quality_band": safe_text(row.get("persona_quality_band")),
        "reliability_band": safe_text(row.get("persona_reliability_band")),
    }


def build_sample_frame(df: pd.DataFrame) -> pd.DataFrame:
    sample_frames: list[pd.DataFrame] = []
    bucket_specs = [
        ("sparse+photo", df.loc[df["evidence_band"].eq("sparse") & df["photo_bucket"].ne("no_photo")]),
        ("sparse+no_photo", df.loc[df["evidence_band"].eq("sparse") & df["photo_bucket"].eq("no_photo")]),
        ("rich+photo", df.loc[df["evidence_band"].eq("rich") & df["photo_bucket"].ne("no_photo")]),
        ("high_risk", df.loc[df["risk_bucket"].eq("high_risk")]),
        ("mid_text+caption", df.loc[df["evidence_band"].eq("mid") & df["photo_bucket"].eq("photo_with_caption")]),
    ]
    for bucket_name, sub in bucket_specs:
        if sub.empty:
            continue
        ranked = sub.sort_values(
            ["audit_review_count", "audit_tip_rows", "photo_count", "semantic_text_len_v3"],
            ascending=[False, False, False, False],
        ).head(SAMPLE_ROWS_PER_BUCKET).copy()
        ranked["audit_bucket"] = bucket_name
        sample_frames.append(ranked)
    if not sample_frames:
        return df.head(SAMPLE_ROWS_PER_BUCKET).copy()
    out = pd.concat(sample_frames, ignore_index=True)
    out = out.drop_duplicates(subset=["business_id"]).reset_index(drop=True)
    return out


def main() -> None:
    merchant_run = resolve_run(
        MERCHANT_PERSONA_RUN_DIR,
        MERCHANT_PERSONA_ROOT,
        "_full_stage09_merchant_persona_schema_v1_build",
    )

    merchant = pd.read_parquet(merchant_run / "merchant_persona_schema_v1.parquet")
    merchant["business_id"] = merchant["business_id"].astype(str)

    photo_path = Path(PHOTO_SUMMARY_PATH)
    if photo_path.exists():
        photo_df = pd.read_parquet(photo_path)
        photo_df["business_id"] = photo_df["business_id"].astype(str)
    else:
        photo_df = pd.DataFrame({"business_id": merchant["business_id"].astype(str)})

    df = merchant.merge(photo_df, on="business_id", how="left")
    df["evidence_band"] = df.apply(merchant_evidence_band, axis=1)
    df["risk_bucket"] = df.apply(merchant_risk_bucket, axis=1)
    df["photo_bucket"] = df.apply(photo_bucket, axis=1)
    df["state_signal_count_v1"] = (
        df["persona_meal_tags"].map(lambda v: len(safe_list(v)))
        + df["persona_scene_tags"].map(lambda v: len(safe_list(v)))
        + df["persona_property_tags"].map(lambda v: len(safe_list(v)))
        + df["persona_dish_tags"].map(lambda v: len(safe_list(v)))
        + df["persona_service_tags"].map(lambda v: len(safe_list(v)))
        + df["persona_complaint_tags"].map(lambda v: len(safe_list(v)))
        + df["persona_time_tags"].map(lambda v: len(safe_list(v)))
    )

    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(len(df)),
        "photo_cover_rate": float(df["photo_bucket"].ne("no_photo").mean()),
        "photo_caption_cover_rate": float(df["photo_caption_summary"].fillna("").ne("").mean()),
        "sparse_merchant_ratio": float(df["evidence_band"].eq("sparse").mean()),
        "photo_cover_on_sparse": float(
            df.loc[df["evidence_band"].eq("sparse"), "photo_bucket"].ne("no_photo").mean()
        ) if df["evidence_band"].eq("sparse").any() else 0.0,
        "avg_state_signal_count_v1": float(df["state_signal_count_v1"].mean()),
        "evidence_band_counts": {k: int(v) for k, v in df["evidence_band"].value_counts().to_dict().items()},
        "risk_bucket_counts": {k: int(v) for k, v in df["risk_bucket"].value_counts().to_dict().items()},
        "photo_bucket_counts": {k: int(v) for k, v in df["photo_bucket"].value_counts().to_dict().items()},
    }

    sample_df = build_sample_frame(df)
    sample_records = []
    for row in sample_df.to_dict(orient="records"):
        series = pd.Series(row)
        sample_records.append(
            {
                "audit_bucket": safe_text(row.get("audit_bucket")),
                "business_id": safe_text(row.get("business_id")),
                "name": safe_text(row.get("name")),
                "city": safe_text(row.get("city")),
                "evidence_band": safe_text(row.get("evidence_band")),
                "risk_bucket": safe_text(row.get("risk_bucket")),
                "photo_bucket": safe_text(row.get("photo_bucket")),
                "input_payload": prompt_input_payload(series),
                "reference_state_v1": reference_state_payload(series),
            }
        )

    prompt_jsonl_path = run_dir / "merchant_state_v2_prompt_audit.jsonl"
    with prompt_jsonl_path.open("w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    safe_json_write(run_dir / "merchant_state_v2_audit_summary.json", summary)
    safe_json_write(run_dir / "merchant_state_v2_sample.json", sample_records)
    safe_json_write(
        run_dir / "run_meta.json",
        {
            "run_tag": RUN_TAG,
            "merchant_persona_run_dir": str(merchant_run),
            "photo_summary_path": str(photo_path),
            "output_run_dir": str(run_dir),
            "summary": summary,
            "sample_rows": int(len(sample_records)),
        },
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
