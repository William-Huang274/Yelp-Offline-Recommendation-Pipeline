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
    "OUTPUT_09_MERCHANT_STATE_V2_ROOT_DIR",
    "data/output/09_merchant_state_v2",
)
RUN_TAG = "stage09_merchant_state_v2_build"
SAMPLE_ROWS = int(os.getenv("MERCHANT_STATE_V2_SAMPLE_ROWS", "16").strip() or 16)

CUISINE_CANONICAL_RULES = [
    ("dessert_bakery", ("bakeries", "bakery", "desserts", "dessert")),
    ("coffee_tea", ("coffee & tea", "coffee", "tea", "juice bars & smoothies", "juice bars", "smoothies")),
    ("pizza", ("pizza",)),
    ("burgers_sandwiches", ("burgers", "burger", "sandwiches", "sandwich", "deli")),
    ("seafood", ("seafood", "oyster", "oysters")),
    ("tacos_mexican", ("mexican", "tacos", "taqueria")),
    ("vegan_vegetarian", ("vegan", "vegetarian")),
    ("barbeque", ("bbq", "barbeque", "barbecue", "smokehouse")),
    ("cajun_creole", ("cajun/creole", "cajun", "creole", "southern", "soul food")),
    ("breakfast_brunch", ("breakfast & brunch", "breakfast", "brunch")),
    ("sushi", ("sushi", "japanese")),
    ("italian", ("italian", "pasta")),
]

PRIMARY_FOOD_CATEGORIES = {
    "restaurants",
    "bakeries",
    "coffee & tea",
    "cafes",
    "pizza",
    "sandwiches",
    "seafood",
    "mexican",
    "vegan",
    "vegetarian",
    "delis",
    "food trucks",
    "chicken wings",
    "breakfast & brunch",
    "sushi bars",
    "italian",
}
STRONG_FOOD_HINTS = (
    "restaurants",
    "restaurant",
    "bakery",
    "bakeries",
    "coffee",
    "tea",
    "juice",
    "pizza",
    "burger",
    "sandwich",
    "seafood",
    "mexican",
    "vegan",
    "vegetarian",
    "bbq",
    "barbeque",
    "barbecue",
    "cajun",
    "creole",
    "southern",
    "food trucks",
    "food truck",
    "chicken",
    "wings",
    "deli",
    "cafe",
    "cafes",
    "sushi",
    "italian",
)
NON_FOOD_HINTS = (
    "health & medical",
    "drugstores",
    "pharmacy",
    "shopping",
    "convenience stores",
    "department stores",
    "grocery",
    "groceries",
    "wholesale stores",
    "auction houses",
    "home & garden",
    "furniture stores",
    "post offices",
    "internet cafes",
)
FAST_FOOD_HINTS = (
    "fast food",
    "food trucks",
    "food truck",
    "chicken wings",
    "juice bars",
    "coffee & tea",
    "bakeries",
    "bakery",
    "delis",
)
SITDOWN_HINTS = (
    "wine bars",
    "cocktail bars",
    "bars",
    "nightlife",
    "venues & event spaces",
    "upscale",
)
ALLOWED_FAST_SCENES = {"quick_bite", "family_friendly", "group_dining"}
ALLOWED_MIXED_RETAIL_SCENES = {"quick_bite"}


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


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


def expand_pipe_items(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        for raw in item.split("|"):
            text = safe_text(raw).lower().replace(" ", "_")
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def classify_scope(primary_category: str, category_blob: str) -> str:
    primary = primary_category.lower().strip()
    lowered = category_blob.lower()
    food_hits = sum(1 for hint in STRONG_FOOD_HINTS if hint in lowered)
    non_food_hits = sum(1 for hint in NON_FOOD_HINTS if hint in lowered)
    if primary in PRIMARY_FOOD_CATEGORIES:
        return "restaurant_primary"
    if food_hits <= 0 and non_food_hits > 0:
        return "non_restaurant"
    if food_hits > 0 and non_food_hits > 0:
        return "mixed_food_retail"
    if food_hits > 0:
        return "restaurant_primary"
    return "mixed_food_retail" if "food" in lowered else "non_restaurant"


def infer_cuisine_from_categories(category_blob: str) -> str:
    lowered = category_blob.lower()
    for canonical, hints in CUISINE_CANONICAL_RULES:
        if any(hint in lowered for hint in hints):
            return canonical
    return ""


def maybe_override_cuisine(row: pd.Series, scope: str) -> tuple[str, str, list[str]]:
    flags: list[str] = []
    primary = safe_text(row.get("merchant_primary_cuisine"))
    secondary = safe_text(row.get("merchant_secondary_cuisine"))
    category_blob = " ".join(
        [
            safe_text(row.get("primary_category")),
            safe_text(row.get("merchant_core_text_v3")),
        ]
    )
    inferred = infer_cuisine_from_categories(category_blob)
    if scope == "non_restaurant":
        if primary or secondary:
            flags.append("non_restaurant_cuisine_cleared")
        return "", "", flags
    if primary in {"dessert_bakery", "coffee_tea"} and inferred not in {"", primary}:
        flags.append("suspicious_primary_cuisine_overridden")
        primary = inferred
    elif not primary and inferred:
        flags.append("primary_cuisine_filled_from_categories")
        primary = inferred
    if secondary == primary:
        secondary = ""
    return primary, secondary, flags


def canonicalize_scene_tags(row: pd.Series, scope: str) -> tuple[list[str], list[str]]:
    scenes = expand_pipe_items(safe_list(row.get("persona_scene_tags")))
    categories = " ".join(
        [safe_text(row.get("primary_category")), safe_text(row.get("merchant_core_text_v3"))]
    ).lower()
    flags: list[str] = []
    if scope == "non_restaurant":
        if scenes:
            flags.append("scene_tags_cleared_non_restaurant")
        return [], flags
    if scope == "mixed_food_retail":
        kept = [tag for tag in scenes if tag in ALLOWED_MIXED_RETAIL_SCENES]
        if len(kept) != len(scenes):
            flags.append("scene_tags_trimmed_mixed_retail")
        return kept, flags
    if any(hint in categories for hint in FAST_FOOD_HINTS):
        kept = [tag for tag in scenes if tag in ALLOWED_FAST_SCENES]
        if len(kept) != len(scenes):
            flags.append("scene_tags_trimmed_fast_food")
        return kept, flags
    if not any(hint in categories for hint in SITDOWN_HINTS):
        kept = [tag for tag in scenes if tag != "date_night"]
        if len(kept) != len(scenes):
            flags.append("date_night_trimmed_without_sitdown_signal")
        return kept, flags
    return scenes, flags


def photo_signal_tags(row: pd.Series) -> list[str]:
    out: list[str] = []
    photo_count = safe_float(row.get("photo_count"))
    if photo_count <= 0:
        return out
    if safe_text(row.get("photo_food_caption_summary")):
        out.append("food_photo")
    if safe_text(row.get("photo_inside_caption_summary")):
        out.append("inside_photo")
    if safe_text(row.get("photo_outside_caption_summary")):
        out.append("outside_photo")
    if safe_text(row.get("photo_menu_caption_summary")):
        out.append("menu_photo")
    dominant = safe_text(row.get("photo_dominant_label")).lower()
    if dominant and dominant not in out:
        out.append(f"dominant_{dominant}")
    return out


def quality_band_v2(scope: str, signal_count: int, review_count: int, reliability: str, noise_flags: list[str]) -> str:
    if scope == "non_restaurant":
        return "noisy"
    if "suspicious_primary_cuisine_overridden" in noise_flags or "scene_tags_trimmed_mixed_retail" in noise_flags:
        return "noisy" if review_count < 25 else "usable"
    if signal_count >= 8 and review_count >= 30 and reliability in {"high", "medium"}:
        return "strong"
    if signal_count >= 4:
        return "usable"
    return "noisy"


def build_state_text(row: pd.Series) -> str:
    parts: list[str] = []
    primary = safe_text(row.get("merchant_primary_cuisine_v2"))
    secondary = safe_text(row.get("merchant_secondary_cuisine_v2"))
    if primary:
        cuisine_text = primary
        if secondary:
            cuisine_text += f", secondary={secondary}"
        parts.append(f"cuisine={cuisine_text}")
    scenes = safe_list(row.get("merchant_scene_tags_v2"))
    if scenes:
        parts.append("scenes=" + ", ".join(scenes[:4]))
    meals = safe_list(row.get("merchant_meal_tags_v2"))
    if meals:
        parts.append("meals=" + ", ".join(meals[:3]))
    props = safe_list(row.get("merchant_property_tags_v2"))
    if props:
        parts.append("properties=" + ", ".join(props[:5]))
    risks = safe_list(row.get("merchant_complaint_tags_v2"))
    if risks:
        parts.append("complaints=" + ", ".join(risks[:4]))
    service = safe_list(row.get("merchant_service_tags_v2"))
    if service:
        parts.append("service=" + ", ".join(service[:4]))
    photo_tags = safe_list(row.get("merchant_photo_signal_tags_v2"))
    if photo_tags:
        parts.append("photo=" + ", ".join(photo_tags[:4]))
    if safe_text(row.get("merchant_entity_scope_v2")):
        parts.append(f"scope={safe_text(row.get('merchant_entity_scope_v2'))}")
    if safe_text(row.get("merchant_state_quality_band_v2")):
        parts.append(f"quality={safe_text(row.get('merchant_state_quality_band_v2'))}")
    return " | ".join(parts)


def sample_rows(df: pd.DataFrame) -> pd.DataFrame:
    bucket_specs = [
        df.loc[df["merchant_entity_scope_v2"].eq("mixed_food_retail")],
        df.loc[df["merchant_entity_scope_v2"].eq("non_restaurant")],
        df.loc[df["merchant_state_quality_band_v2"].eq("noisy")],
        df.loc[df["merchant_state_quality_band_v2"].eq("strong")],
    ]
    frames: list[pd.DataFrame] = []
    for sub in bucket_specs:
        if sub.empty:
            continue
        frames.append(
            sub.sort_values(
                ["audit_review_count", "merchant_state_signal_count_v2"],
                ascending=[False, False],
            ).head(max(1, SAMPLE_ROWS // len(bucket_specs))).copy()
        )
    if not frames:
        return df.head(SAMPLE_ROWS).copy()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["business_id"]).reset_index(drop=True)
    return out.head(SAMPLE_ROWS).copy()


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

    rows: list[dict[str, Any]] = []
    override_count = 0
    for raw in df.to_dict(orient="records"):
        row = pd.Series(raw)
        category_blob = " ".join([safe_text(row.get("primary_category")), safe_text(row.get("merchant_core_text_v3"))])
        scope = classify_scope(safe_text(row.get("primary_category")), category_blob)
        primary_v2, secondary_v2, cuisine_flags = maybe_override_cuisine(row, scope)
        scene_tags_v2, scene_flags = canonicalize_scene_tags(row, scope)
        meal_tags_v2 = expand_pipe_items(safe_list(row.get("persona_meal_tags")))
        property_tags_v2 = expand_pipe_items(safe_list(row.get("persona_property_tags")))
        dish_tags_v2 = expand_pipe_items(safe_list(row.get("persona_dish_tags")))
        service_tags_v2 = expand_pipe_items(safe_list(row.get("persona_service_tags")))
        complaint_tags_v2 = expand_pipe_items(safe_list(row.get("persona_complaint_tags")))
        time_tags_v2 = expand_pipe_items(safe_list(row.get("persona_time_tags")))
        photo_tags_v2 = photo_signal_tags(row)
        noise_flags = cuisine_flags + scene_flags
        if cuisine_flags:
            override_count += 1
        signal_count = sum(
            len(items)
            for items in (
                meal_tags_v2,
                scene_tags_v2,
                property_tags_v2,
                dish_tags_v2,
                service_tags_v2,
                complaint_tags_v2,
                time_tags_v2,
                photo_tags_v2,
            )
        )
        quality_v2 = quality_band_v2(
            scope=scope,
            signal_count=signal_count,
            review_count=int(safe_float(row.get("audit_review_count"))),
            reliability=safe_text(row.get("persona_reliability_band")),
            noise_flags=noise_flags,
        )

        out_row = dict(raw)
        out_row.update(
            {
                "merchant_entity_scope_v2": scope,
                "merchant_primary_cuisine_v2": primary_v2,
                "merchant_secondary_cuisine_v2": secondary_v2,
                "merchant_meal_tags_v2": meal_tags_v2,
                "merchant_scene_tags_v2": scene_tags_v2,
                "merchant_property_tags_v2": property_tags_v2,
                "merchant_dish_tags_v2": dish_tags_v2,
                "merchant_service_tags_v2": service_tags_v2,
                "merchant_complaint_tags_v2": complaint_tags_v2,
                "merchant_time_tags_v2": time_tags_v2,
                "merchant_photo_signal_tags_v2": photo_tags_v2,
                "merchant_noise_flags_v2": noise_flags,
                "merchant_state_signal_count_v2": int(signal_count),
                "merchant_state_quality_band_v2": quality_v2,
            }
        )
        out_row["merchant_state_text_v2"] = build_state_text(pd.Series(out_row))
        rows.append(out_row)

    out = pd.DataFrame(rows)
    run_dir = OUTPUT_ROOT / now_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / "merchant_state_v2.parquet"
    summary_path = run_dir / "merchant_state_v2_summary.json"
    sample_path = run_dir / "merchant_state_v2_sample.json"
    run_meta_path = run_dir / "run_meta.json"

    out.to_parquet(out_path, index=False)

    summary = {
        "rows": int(len(out)),
        "entity_scope_counts": {
            k: int(v) for k, v in out["merchant_entity_scope_v2"].value_counts().to_dict().items()
        },
        "quality_band_counts_v2": {
            k: int(v) for k, v in out["merchant_state_quality_band_v2"].value_counts().to_dict().items()
        },
        "photo_cover_rate": float(out["merchant_photo_signal_tags_v2"].map(bool).mean()),
        "cuisine_override_rate_v2": float(override_count / max(1, len(out))),
        "avg_signal_count_v2": float(out["merchant_state_signal_count_v2"].mean()),
        "non_restaurant_with_cuisine_rate_v2": float(
            out.loc[out["merchant_entity_scope_v2"].eq("non_restaurant"), "merchant_primary_cuisine_v2"].fillna("").ne("").mean()
        ) if out["merchant_entity_scope_v2"].eq("non_restaurant").any() else 0.0,
        "mixed_food_retail_rows": int(out["merchant_entity_scope_v2"].eq("mixed_food_retail").sum()),
    }

    sample_df = sample_rows(out)
    sample_records = sample_df[
        [
            "business_id",
            "name",
            "city",
            "primary_category",
            "merchant_entity_scope_v2",
            "merchant_primary_cuisine",
            "merchant_primary_cuisine_v2",
            "merchant_secondary_cuisine",
            "merchant_secondary_cuisine_v2",
            "merchant_scene_tags_v2",
            "merchant_property_tags_v2",
            "merchant_dish_tags_v2",
            "merchant_complaint_tags_v2",
            "merchant_photo_signal_tags_v2",
            "merchant_noise_flags_v2",
            "merchant_state_quality_band_v2",
            "merchant_state_text_v2",
            "merchant_core_text_v3",
            "merchant_semantic_text_v3",
        ]
    ].to_dict(orient="records")

    safe_json_write(summary_path, summary)
    safe_json_write(sample_path, sample_records)
    safe_json_write(
        run_meta_path,
        {
            "run_tag": RUN_TAG,
            "merchant_persona_run_dir": str(merchant_run),
            "photo_summary_path": str(photo_path),
            "output_run_dir": str(run_dir),
            "summary": summary,
        },
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
