from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path


RUN_TAG = "stage11_leak_clean_prompt_input_v1_build"

INPUT_ROUTE_EVAL_ROOT = env_or_project_path(
    "INPUT_11_PASS2_PROFILE_ROUTE_SUBSET_EVAL_ROOT_DIR",
    "data/output/11_pass2_profile_route_subset_eval",
)
INPUT_ROUTE_EVAL_RUN_DIR = os.getenv("INPUT_11_PASS2_PROFILE_ROUTE_SUBSET_EVAL_RUN_DIR", "").strip()

INPUT_BRIDGE_ROOT = env_or_project_path(
    "INPUT_11_PASS2_STAGE09_TARGET_BRIDGE_V1_ROOT_DIR",
    "data/output/11_pass2_stage09_target_bridge_v1",
)
INPUT_BRIDGE_RUN_DIR = os.getenv("INPUT_11_PASS2_STAGE09_TARGET_BRIDGE_V1_RUN_DIR", "").strip()

INPUT_MERCHANT_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_ROOT_DIR",
    "data/output/09_merchant_persona_schema_v1",
)
INPUT_MERCHANT_RUN_DIR = os.getenv("INPUT_09_MERCHANT_PERSONA_SCHEMA_V1_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_11_STAGE11_LEAK_CLEAN_PROMPT_INPUT_V1_ROOT_DIR",
    "data/output/11_stage11_leak_clean_prompt_input_v1",
)

RECENT_K = int(os.getenv("LEAK_CLEAN_RECENT_K", "8").strip() or 8)
ANCHOR_POS_K = int(os.getenv("LEAK_CLEAN_ANCHOR_POS_K", "3").strip() or 3)
ANCHOR_NEG_K = int(os.getenv("LEAK_CLEAN_ANCHOR_NEG_K", "2").strip() or 2)
REVIEW_EVIDENCE_TOP_K = int(os.getenv("LEAK_CLEAN_REVIEW_EVIDENCE_TOP_K", "0").strip() or 0)
KEEP_ALL_SEMANTIC_REVIEWS = os.getenv("LEAK_CLEAN_KEEP_ALL_SEMANTIC_REVIEWS", "true").strip().lower() == "true"
REVIEW_TEXT_MAX_CHARS = int(os.getenv("LEAK_CLEAN_REVIEW_TEXT_MAX_CHARS", "0").strip() or 0)
REVIEW_MIN_CHARS = int(os.getenv("LEAK_CLEAN_REVIEW_MIN_CHARS", "8").strip() or 8)
REVIEW_MIN_WORDS = int(os.getenv("LEAK_CLEAN_REVIEW_MIN_WORDS", "2").strip() or 2)
MIN_STARS_POS = float(os.getenv("LEAK_CLEAN_MIN_STARS_POS", "4.0").strip() or 4.0)
MAX_STARS_NEG = float(os.getenv("LEAK_CLEAN_MAX_STARS_NEG", "2.0").strip() or 2.0)
SAMPLE_ROWS = int(os.getenv("LEAK_CLEAN_SAMPLE_ROWS", "12").strip() or 12)

SPACE_RE = re.compile(r"\s+")
GENERIC_INVALID_SHORT_REVIEWS = {
    "good",
    "great",
    "nice",
    "ok",
    "okay",
    "awesome",
    "yummy",
    "delicious",
}


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
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return safe_list(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [safe_text(x) for x in value if safe_text(x)]
    if isinstance(value, list):
        return [safe_text(x) for x in value if safe_text(x)]
    raw = safe_text(value)
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]") and "'" in raw and "," not in raw:
        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw)
        parsed = [safe_text(a or b) for a, b in tokens if safe_text(a or b)]
        if parsed:
            return parsed
    try:
        payload = json.loads(raw)
    except Exception:
        return [raw]
    if isinstance(payload, list):
        return [safe_text(x) for x in payload if safe_text(x)]
    return [raw]


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def normalize_review_text(raw: Any, max_chars: int | None = None) -> str:
    text = SPACE_RE.sub(" ", safe_text(raw).replace("\r", " ").replace("\n", " ")).strip()
    if max_chars is not None and int(max_chars) > 0 and len(text) > int(max_chars):
        return text[: int(max_chars)].rstrip()
    return text


def is_semantic_review_text(raw: Any) -> bool:
    text = normalize_review_text(raw, max_chars=0)
    if not text:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    words = text.split()
    if len(text) < REVIEW_MIN_CHARS and len(words) < REVIEW_MIN_WORDS:
        return False
    if len(words) <= 3 and text.lower() in GENERIC_INVALID_SHORT_REVIEWS:
        return False
    return True


def event_type_from_stars(stars: float) -> str:
    if float(stars) >= float(MIN_STARS_POS):
        return "review_positive"
    if float(stars) <= float(MAX_STARS_NEG):
        return "review_negative"
    return "review_neutral"


def event_payload(row: pd.Series, rank: int) -> dict[str, Any]:
    return {
        "recent_rank": int(rank),
        "event_type": event_type_from_stars(float(row.get("stars", 0.0) or 0.0)),
        "event_time": safe_text(row.get("event_time")),
        "sample_weight_v2": round(float(row.get("sample_weight_v2", 0.0) or 0.0), 4),
        "business_id": safe_text(row.get("business_id")),
        "name": safe_text(row.get("name")),
        "city": safe_text(row.get("city")),
        "merchant_primary_cuisine": safe_text(row.get("merchant_primary_cuisine")),
        "merchant_secondary_cuisine": safe_text(row.get("merchant_secondary_cuisine")),
        "persona_meal_tags": safe_list(row.get("persona_meal_tags")),
        "persona_scene_tags": safe_list(row.get("persona_scene_tags")),
        "persona_property_tags": safe_list(row.get("persona_property_tags")),
        "persona_quality_band": safe_text(row.get("persona_quality_band")),
        "persona_reliability_band": safe_text(row.get("persona_reliability_band")),
        "merchant_core_text_short": safe_text(row.get("merchant_core_text_short")),
    }


def review_evidence_payload(row: pd.Series, idx: int) -> dict[str, Any]:
    stars = float(row.get("stars", 0.0) or 0.0)
    return {
        "evidence_id": f"rev_{idx}",
        "review_id": safe_text(row.get("review_id")),
        "source": "review",
        "sentiment": "positive" if stars >= MIN_STARS_POS else ("negative" if stars <= MAX_STARS_NEG else "neutral"),
        "weight": round(float(row.get("evidence_weight", 0.0) or 0.0), 4),
        "time_bucket": "recent" if int(row.get("days_from_latest", 99999) or 99999) <= 180 else "older",
        "business_id": safe_text(row.get("business_id")),
        "event_time": safe_text(row.get("event_time")),
        "stars": round(stars, 2),
        "tag": safe_text(row.get("merchant_primary_cuisine")) or "unknown",
        "text": normalize_review_text(row.get("text"), max_chars=REVIEW_TEXT_MAX_CHARS),
    }


def build_user_rows(events: pd.DataFrame, bridge_df: pd.DataFrame) -> pd.DataFrame:
    bridge_map = bridge_df.set_index("user_id")[["density_band"]].to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for input_index, (user_id, grp) in enumerate(events.groupby("user_id", sort=True)):
        grp = grp.sort_values(["event_time", "review_id"], ascending=[False, False]).copy()
        latest_ts = grp["event_time"].max()
        grp["days_from_latest"] = (latest_ts - grp["event_time"]).dt.days.fillna(0).astype(int)
        grp["sample_weight_v2"] = (grp["stars"].clip(lower=1.0, upper=5.0) / 5.0).astype(float)
        grp["merchant_core_text_short"] = grp["merchant_core_text_v3"].fillna("").astype(str).str.slice(0, 180)

        recent_payload = [event_payload(row, rank + 1) for rank, (_, row) in enumerate(grp.head(RECENT_K).iterrows())]
        pos_grp = grp[grp["stars"] >= MIN_STARS_POS].head(ANCHOR_POS_K)
        neg_grp = grp[grp["stars"] <= MAX_STARS_NEG].head(ANCHOR_NEG_K)
        pos_payload = [event_payload(row, rank + 1) for rank, (_, row) in enumerate(pos_grp.iterrows())]
        neg_payload = [event_payload(row, rank + 1) for rank, (_, row) in enumerate(neg_grp.iterrows())]

        grp["evidence_weight"] = grp["sample_weight_v2"] * (1.0 / (1.0 + grp["days_from_latest"].clip(lower=0)))
        semantic_reviews = grp.loc[grp["text"].map(is_semantic_review_text)].copy()
        semantic_reviews = semantic_reviews.sort_values(["event_time", "review_id"], ascending=[False, False])
        if not KEEP_ALL_SEMANTIC_REVIEWS and REVIEW_EVIDENCE_TOP_K > 0:
            semantic_reviews = semantic_reviews.sort_values(
                ["evidence_weight", "event_time", "review_id"],
                ascending=[False, False, False],
            ).head(REVIEW_EVIDENCE_TOP_K)
        review_payload = [
            review_evidence_payload(row, idx + 1)
            for idx, (_, row) in enumerate(semantic_reviews.iterrows())
        ]

        bridge_meta = bridge_map.get(str(user_id), {})
        rows.append(
            {
                "input_index": int(input_index),
                "user_id": str(user_id),
                "density_band": safe_text(bridge_meta.get("density_band")) or "unknown",
                "n_train_max": int(grp.shape[0]),
                "sequence_event_count": int(len(recent_payload)),
                "review_evidence_count_v1": int(len(review_payload)),
                "raw_review_count_v1": int(grp["review_id"].astype(str).nunique()),
                "semantic_review_count_v1": int(semantic_reviews["review_id"].astype(str).nunique()),
                "retained_review_count_v1": int(len(review_payload)),
                "tip_evidence_count_v1": 0,
                "review_evidence_stream_json": json.dumps(review_payload, ensure_ascii=False),
                "tip_evidence_stream_json": "[]",
                "recent_event_sequence_json": json.dumps(recent_payload, ensure_ascii=False),
                "positive_anchor_sequence_json": json.dumps(pos_payload, ensure_ascii=False),
                "negative_anchor_sequence_json": json.dumps(neg_payload, ensure_ascii=False),
                "leak_clean": True,
            }
        )
    return pd.DataFrame(rows).sort_values("input_index").reset_index(drop=True)


def explode_review_payloads(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in labels[["user_id", "review_evidence_stream_json"]].to_dict(orient="records"):
        user_id = safe_text(record.get("user_id"))
        raw = safe_text(record.get("review_evidence_stream_json"))
        try:
            payload = json.loads(raw) if raw else []
        except Exception:
            payload = []
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "review_id": safe_text(item.get("review_id")),
                    "business_id": safe_text(item.get("business_id")),
                    "event_time": safe_text(item.get("event_time")),
                    "stars": float(item.get("stars", 0.0) or 0.0),
                    "retained_text_norm": normalize_review_text(item.get("text"), max_chars=0),
                }
            )
    return pd.DataFrame(rows)


def build_review_retention_audit(train_events: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = train_events[["user_id", "review_id", "business_id", "event_time", "stars", "text"]].copy()
    raw["user_id"] = raw["user_id"].astype(str)
    raw["review_id"] = raw["review_id"].astype(str)
    raw["business_id"] = raw["business_id"].astype(str)
    raw["raw_stars"] = pd.to_numeric(raw["stars"], errors="coerce").fillna(0.0)
    raw["raw_text_norm"] = raw["text"].map(lambda x: normalize_review_text(x, max_chars=0))
    raw = raw.loc[raw["raw_text_norm"].map(is_semantic_review_text)].copy()
    raw = raw.drop(columns=["stars"])
    raw = raw.drop_duplicates(subset=["user_id", "review_id"], keep="first")

    retained = explode_review_payloads(labels)
    if retained.empty:
        merged = raw.assign(retained_text_norm="", retained_business_id="", retained_event_time="", text_exact_match=False)
    else:
        merged = raw.merge(
            retained.rename(columns={"business_id": "retained_business_id", "event_time": "retained_event_time"}),
            on=["user_id", "review_id"],
            how="left",
        )
        merged["retained_text_norm"] = merged["retained_text_norm"].fillna("")
        merged["retained_business_id"] = merged["retained_business_id"].fillna("")
        merged["retained_event_time"] = merged["retained_event_time"].fillna("")
        merged["text_exact_match"] = merged["raw_text_norm"].eq(merged["retained_text_norm"])
    merged["retained_flag"] = merged["retained_text_norm"].ne("")
    merged["business_match"] = merged["business_id"].eq(merged["retained_business_id"])

    missing = merged.loc[~merged["retained_flag"], ["user_id", "review_id", "business_id", "event_time", "raw_stars", "raw_text_norm"]]
    mismatched = merged.loc[
        merged["retained_flag"] & (~merged["text_exact_match"] | ~merged["business_match"]),
        ["user_id", "review_id", "business_id", "retained_business_id", "event_time", "retained_event_time", "raw_text_norm", "retained_text_norm"],
    ]
    summary = {
        "raw_train_review_rows_total": int(train_events["review_id"].astype(str).nunique()),
        "semantic_review_rows_total": int(len(raw)),
        "retained_review_rows_total": int(len(retained)),
        "semantic_review_retention_rate": float(merged["retained_flag"].mean()) if not merged.empty else 1.0,
        "text_exact_match_rate": float(merged.loc[merged["retained_flag"], "text_exact_match"].mean()) if merged["retained_flag"].any() else 1.0,
        "business_match_rate": float(merged.loc[merged["retained_flag"], "business_match"].mean()) if merged["retained_flag"].any() else 1.0,
        "users_with_full_semantic_retention_rate": float(
            merged.groupby("user_id")["retained_flag"].all().mean()
        ) if not merged.empty else 1.0,
        "missing_review_rows": int(len(missing)),
        "mismatched_review_rows": int(len(mismatched)),
        "missing_review_samples": missing.head(10).to_dict(orient="records"),
        "mismatched_review_samples": mismatched.head(10).to_dict(orient="records"),
    }
    return summary, merged


def main() -> None:
    route_run = resolve_run(
        INPUT_ROUTE_EVAL_RUN_DIR,
        INPUT_ROUTE_EVAL_ROOT,
        "_full_stage11_pass2_profile_route_subset_eval",
    )
    bridge_run = resolve_run(
        INPUT_BRIDGE_RUN_DIR,
        INPUT_BRIDGE_ROOT,
        "_full_stage11_pass2_stage09_target_bridge_v1_build",
    )
    merchant_run = resolve_run(
        INPUT_MERCHANT_RUN_DIR,
        INPUT_MERCHANT_ROOT,
        "_full_stage09_merchant_persona_schema_v1_build",
    )

    split_df = pd.read_parquet(route_run / "event_split.parquet")
    split_df["user_id"] = split_df["user_id"].astype(str)
    split_df["business_id"] = split_df["business_id"].astype(str)
    split_df["review_id"] = split_df["review_id"].astype(str)
    split_df["event_time"] = pd.to_datetime(split_df["ts"], errors="coerce")
    split_df["stars"] = pd.to_numeric(split_df["stars"], errors="coerce").fillna(0.0)

    merchant = pd.read_parquet(
        merchant_run / "merchant_persona_schema_v1.parquet",
        columns=[
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
        ],
    )
    merchant["business_id"] = merchant["business_id"].astype(str)

    bridge_df = pd.read_parquet(bridge_run / "user_preference_target_schema_v1.parquet", columns=["user_id", "density_band"])
    bridge_df["user_id"] = bridge_df["user_id"].astype(str)

    train_events = (
        split_df[split_df["split_label"].eq("train")]
        .merge(merchant, on="business_id", how="left", suffixes=("_event", "_merchant"))
        .copy()
    )
    for base_col in ["name", "city"]:
        event_col = f"{base_col}_event"
        merchant_col = f"{base_col}_merchant"
        if event_col in train_events.columns or merchant_col in train_events.columns:
            train_events[base_col] = ""
            if event_col in train_events.columns:
                train_events[base_col] = train_events[event_col].fillna("")
            if merchant_col in train_events.columns:
                train_events[base_col] = train_events[base_col].where(
                    train_events[base_col].astype(str).str.strip().ne(""),
                    train_events[merchant_col].fillna(""),
                )
    labels = build_user_rows(train_events, bridge_df)
    review_retention_summary, review_retention_rows = build_review_retention_audit(train_events, labels)

    run_id = now_run_id()
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    label_path = out_dir / "user_state_labels_v1.parquet"
    labels.to_parquet(label_path, index=False)
    review_retention_rows.to_parquet(out_dir / "review_retention_audit_rows.parquet", index=False)

    summary = {
        "run_id": run_id,
        "input_route_eval_run_dir": str(route_run),
        "input_bridge_run_dir": str(bridge_run),
        "input_merchant_run_dir": str(merchant_run),
        "users_total": int(labels["user_id"].nunique()),
        "rows_total": int(labels.shape[0]),
        "sequence_nonempty_rate": float(labels["recent_event_sequence_json"].fillna("").ne("[]").mean()) if not labels.empty else 0.0,
        "review_evidence_nonempty_rate": float(labels["review_evidence_stream_json"].fillna("").ne("[]").mean()) if not labels.empty else 0.0,
        "tip_evidence_nonempty_rate": float(labels["tip_evidence_stream_json"].fillna("").ne("[]").mean()) if not labels.empty else 0.0,
        "mean_sequence_event_count": float(labels["sequence_event_count"].mean()) if not labels.empty else 0.0,
        "mean_review_evidence_count": float(labels["review_evidence_count_v1"].mean()) if not labels.empty else 0.0,
        "mean_raw_review_count": float(labels["raw_review_count_v1"].mean()) if not labels.empty else 0.0,
        "mean_semantic_review_count": float(labels["semantic_review_count_v1"].mean()) if not labels.empty else 0.0,
        "mean_retained_review_count": float(labels["retained_review_count_v1"].mean()) if not labels.empty else 0.0,
        "review_retention_audit": review_retention_summary,
        "sample_rows": labels.head(SAMPLE_ROWS).to_dict(orient="records"),
    }
    safe_json_write(out_dir / "user_state_labels_v1_summary.json", summary)
    safe_json_write(
        out_dir / "run_meta.json",
        {
            "run_id": run_id,
            "run_tag": RUN_TAG,
            "user_state_labels_v1": str(label_path),
            "summary_json": str(out_dir / "user_state_labels_v1_summary.json"),
            "summary": summary,
        },
    )
    print(f"[OK] output_dir={out_dir}")
    print(f"[OK] users_total={int(labels['user_id'].nunique())} rows_total={int(labels.shape[0])}")


if __name__ == "__main__":
    main()
