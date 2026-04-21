from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path


RUN_TAG = "stage11_pass2_stage09_profile_tag_sidecar_v1_build"

INPUT_BRIDGE_ROOT = env_or_project_path(
    "INPUT_11_PASS2_STAGE09_TARGET_BRIDGE_V1_ROOT_DIR",
    "data/output/11_pass2_stage09_target_bridge_v1",
)
INPUT_BRIDGE_RUN_DIR = os.getenv("INPUT_11_PASS2_STAGE09_TARGET_BRIDGE_V1_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_11_PASS2_STAGE09_PROFILE_TAG_SIDECAR_V1_ROOT_DIR",
    "data/output/11_pass2_stage09_profile_tag_sidecar_v1",
)

POOL_MODE = (os.getenv("PASS2_PROFILE_TAG_SIDECAR_POOL_MODE", "quality_first_then_balanced").strip().lower() or "quality_first_then_balanced")
SAMPLE_ROWS = int(os.getenv("PASS2_PROFILE_TAG_SIDECAR_SAMPLE_ROWS", "16").strip() or 16)

INCLUDE_SCENE_TAGS = os.getenv("PASS2_PROFILE_TAG_SIDECAR_INCLUDE_SCENE_TAGS", "true").strip().lower() == "true"
INCLUDE_RECENT_TAGS = os.getenv("PASS2_PROFILE_TAG_SIDECAR_INCLUDE_RECENT_TAGS", "true").strip().lower() == "true"
INCLUDE_SERVICE_TAGS = os.getenv("PASS2_PROFILE_TAG_SIDECAR_INCLUDE_SERVICE_TAGS", "false").strip().lower() == "true"

WEIGHT_STABLE_CUISINE = float(os.getenv("PASS2_PROFILE_TAG_SIDECAR_WEIGHT_STABLE_CUISINE", "0.62").strip() or 0.62)
WEIGHT_STABLE_SCENE = float(os.getenv("PASS2_PROFILE_TAG_SIDECAR_WEIGHT_STABLE_SCENE", "0.34").strip() or 0.34)
WEIGHT_AVOID_CUISINE = float(os.getenv("PASS2_PROFILE_TAG_SIDECAR_WEIGHT_AVOID_CUISINE", "0.66").strip() or 0.66)
WEIGHT_RECENT_CUISINE = float(os.getenv("PASS2_PROFILE_TAG_SIDECAR_WEIGHT_RECENT_CUISINE", "0.26").strip() or 0.26)
WEIGHT_RECENT_SCENE = float(os.getenv("PASS2_PROFILE_TAG_SIDECAR_WEIGHT_RECENT_SCENE", "0.18").strip() or 0.18)
WEIGHT_SERVICE = float(os.getenv("PASS2_PROFILE_TAG_SIDECAR_WEIGHT_SERVICE", "0.14").strip() or 0.14)

CONFIDENCE_BY_LABEL = {
    "high": 1.00,
    "medium": 0.84,
    "low": 0.68,
    "unknown": 0.72,
    "": 0.72,
}
POOL_SCALE = {
    "quality_first": 1.00,
    "balanced": 0.78,
    "rejected": 0.0,
}
SCENE_TAG_MAP = {
    "family_friendly": "family_friendly",
    "date_night": "date_night",
}
SERVICE_WEIGHT_BY_SENSITIVITY = {
    "high": 1.00,
    "medium": 0.72,
    "low": 0.0,
    "unknown": 0.0,
    "": 0.0,
}
TAG_TYPE_OVERRIDES = {
    "breakfast_brunch": ("breakfast_brunch", "meal"),
    "coffee_tea": ("coffee_tea", "beverage"),
    "dessert": ("dessert", "meal"),
}


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


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = safe_text(text)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def normalize_tag(value: Any) -> str:
    text = safe_text(value).lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_tag(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def remap_tag_and_type(tag: str, default_tag_type: str) -> tuple[str, str]:
    norm_tag = normalize_tag(tag)
    if not norm_tag:
        return "", normalize_tag(default_tag_type) or "other"
    override = TAG_TYPE_OVERRIDES.get(norm_tag)
    if override is not None:
        mapped_tag, mapped_type = override
        return normalize_tag(mapped_tag), normalize_tag(mapped_type) or "other"
    return norm_tag, normalize_tag(default_tag_type) or "other"


def safe_list(node: Any) -> list[str]:
    if not isinstance(node, list):
        return []
    out: list[str] = []
    for item in node:
        text = normalize_tag(item)
        if text:
            out.append(text)
    return unique_keep_order(out)


def get_nested_list(payload: dict[str, Any], *path: str) -> list[str]:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict):
            return []
        node = node.get(key)
    return safe_list(node)


def pool_label_rank(*, pool_name: str, recommended_pool: str) -> int:
    if pool_name == "quality_first_only":
        return 1 if recommended_pool == "quality_first" else 0
    if pool_name == "balanced_only":
        return 1 if recommended_pool == "balanced" else 0
    if pool_name == "quality_first_then_balanced":
        return 1 if recommended_pool in {"quality_first", "balanced"} else 0
    return 1


def confidence_multiplier(pool: str, confidence_label: str) -> float:
    base = float(CONFIDENCE_BY_LABEL.get(confidence_label, CONFIDENCE_BY_LABEL["unknown"]))
    pool_scale = float(POOL_SCALE.get(pool, 0.0))
    return max(0.0, min(1.0, base * pool_scale))


def add_signal_row(
    rows: list[dict[str, Any]],
    *,
    user_id: str,
    density_band: str,
    recommended_pool: str,
    source_pool: str,
    confidence_overall: str,
    tag: str,
    tag_type: str,
    direction: str,
    base_weight: float,
    signal_scope: str,
    support: float,
) -> None:
    norm_tag = normalize_tag(tag)
    norm_tag, norm_tag_type = remap_tag_and_type(norm_tag, tag_type)
    if not user_id or not norm_tag or base_weight <= 0.0:
        return
    conf = confidence_multiplier(recommended_pool, confidence_overall)
    if conf <= 0.0:
        return
    pos_w = float(base_weight) if direction == "pos" else 0.0
    neg_w = float(base_weight) if direction == "neg" else 0.0
    rows.append(
        {
            "user_id": user_id,
            "tag": norm_tag,
            "tag_type": norm_tag_type,
            "pos_w": pos_w,
            "neg_w": neg_w,
            "net_w": pos_w - neg_w,
            "abs_net_w": abs(pos_w - neg_w),
            "support": float(max(support, 1.0)),
            "tag_confidence": float(conf),
            "signal_scope": signal_scope,
            "recommended_pool": recommended_pool,
            "source_pool": source_pool,
            "density_band": density_band,
            "confidence_overall": confidence_overall,
        }
    )


def build_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in df.itertuples(index=False):
        recommended_pool = safe_text(getattr(record, "recommended_pool", "")).lower()
        if pool_label_rank(pool_name=POOL_MODE, recommended_pool=recommended_pool) <= 0:
            continue
        user_id = safe_text(getattr(record, "user_id", ""))
        density_band = safe_text(getattr(record, "density_band", ""))
        source_pool = safe_text(getattr(record, "source_pool", "")).lower()
        confidence_overall = safe_text(getattr(record, "confidence_overall", "")).lower()
        service_risk = safe_text(getattr(record, "service_risk_sensitivity", "")).lower()

        payload = parse_json_object(getattr(record, "target_schema_v1_json", ""))
        stable_pref = get_nested_list(payload, "core_profile", "stable_preferences", "preferred_cuisines")
        stable_scene = get_nested_list(payload, "core_profile", "stable_preferences", "preferred_scenes")
        avoid_cuisine = get_nested_list(payload, "core_profile", "avoid_signals", "avoided_cuisines")
        recent_cuisine = get_nested_list(payload, "core_profile", "recent_preferences", "recent_cuisines")
        recent_scene = get_nested_list(payload, "core_profile", "recent_preferences", "recent_scenes")

        for tag in stable_pref:
            add_signal_row(
                rows,
                user_id=user_id,
                density_band=density_band,
                recommended_pool=recommended_pool,
                source_pool=source_pool,
                confidence_overall=confidence_overall,
                tag=tag,
                tag_type="cuisine",
                direction="pos",
                base_weight=WEIGHT_STABLE_CUISINE,
                signal_scope="stable_preferred_cuisine",
                support=2.0,
            )
        if INCLUDE_SCENE_TAGS:
            for tag in stable_scene:
                scene_tag = SCENE_TAG_MAP.get(tag, "")
                if not scene_tag:
                    continue
                add_signal_row(
                    rows,
                    user_id=user_id,
                    density_band=density_band,
                    recommended_pool=recommended_pool,
                    source_pool=source_pool,
                    confidence_overall=confidence_overall,
                    tag=scene_tag,
                    tag_type="scene",
                    direction="pos",
                    base_weight=WEIGHT_STABLE_SCENE,
                    signal_scope="stable_preferred_scene",
                    support=1.0,
                )
        for tag in avoid_cuisine:
            add_signal_row(
                rows,
                user_id=user_id,
                density_band=density_band,
                recommended_pool=recommended_pool,
                source_pool=source_pool,
                confidence_overall=confidence_overall,
                tag=tag,
                tag_type="cuisine",
                direction="neg",
                base_weight=WEIGHT_AVOID_CUISINE,
                signal_scope="avoid_cuisine",
                support=2.0,
            )
        if INCLUDE_RECENT_TAGS:
            for tag in recent_cuisine:
                add_signal_row(
                    rows,
                    user_id=user_id,
                    density_band=density_band,
                    recommended_pool=recommended_pool,
                    source_pool=source_pool,
                    confidence_overall=confidence_overall,
                    tag=tag,
                    tag_type="cuisine",
                    direction="pos",
                    base_weight=WEIGHT_RECENT_CUISINE,
                    signal_scope="recent_cuisine",
                    support=1.0,
                )
            if INCLUDE_SCENE_TAGS:
                for tag in recent_scene:
                    scene_tag = SCENE_TAG_MAP.get(tag, "")
                    if not scene_tag:
                        continue
                    add_signal_row(
                        rows,
                        user_id=user_id,
                        density_band=density_band,
                        recommended_pool=recommended_pool,
                        source_pool=source_pool,
                        confidence_overall=confidence_overall,
                        tag=scene_tag,
                        tag_type="scene",
                        direction="pos",
                        base_weight=WEIGHT_RECENT_SCENE,
                        signal_scope="recent_scene",
                        support=1.0,
                    )
        if INCLUDE_SERVICE_TAGS:
            service_scale = float(SERVICE_WEIGHT_BY_SENSITIVITY.get(service_risk, 0.0))
            if service_scale > 0.0:
                add_signal_row(
                    rows,
                    user_id=user_id,
                    density_band=density_band,
                    recommended_pool=recommended_pool,
                    source_pool=source_pool,
                    confidence_overall=confidence_overall,
                    tag="service",
                    tag_type="service",
                    direction="pos",
                    base_weight=WEIGHT_SERVICE * service_scale,
                    signal_scope="service_guard",
                    support=1.0,
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "tag",
                "tag_type",
                "pos_w",
                "neg_w",
                "net_w",
                "abs_net_w",
                "support",
                "tag_confidence",
                "signal_scope",
                "recommended_pool",
                "source_pool",
                "density_band",
                "confidence_overall",
            ]
        )
    out = out.sort_values(["user_id", "tag_type", "tag", "signal_scope"], ascending=[True, True, True, True]).reset_index(drop=True)
    return out


def build_summary(sidecar: pd.DataFrame, bridge_df: pd.DataFrame, run_id: str, bridge_run: Path) -> dict[str, Any]:
    sidecar_users = sidecar["user_id"].nunique() if not sidecar.empty else 0
    density_counts = (
        sidecar[["user_id", "density_band"]]
        .drop_duplicates()
        .groupby("density_band")["user_id"]
        .nunique()
        .sort_index()
        .to_dict()
        if not sidecar.empty
        else {}
    )
    scope_counts = sidecar["signal_scope"].value_counts().sort_index().to_dict() if not sidecar.empty else {}
    tag_type_counts = sidecar["tag_type"].value_counts().sort_index().to_dict() if not sidecar.empty else {}
    pool_counts = sidecar["recommended_pool"].value_counts().sort_index().to_dict() if not sidecar.empty else {}
    users_per_pool = (
        sidecar[["user_id", "recommended_pool"]]
        .drop_duplicates()
        .groupby("recommended_pool")["user_id"]
        .nunique()
        .sort_index()
        .to_dict()
        if not sidecar.empty
        else {}
    )
    mean_rows_per_user = float(sidecar.shape[0] / max(sidecar_users, 1))
    return {
        "run_id": run_id,
        "input_bridge_run_dir": str(bridge_run),
        "bridge_users": int(bridge_df["user_id"].nunique()),
        "bridge_rows": int(bridge_df.shape[0]),
        "sidecar_rows": int(sidecar.shape[0]),
        "sidecar_users": int(sidecar_users),
        "sidecar_users_coverage_vs_bridge": float(sidecar_users / max(int(bridge_df["user_id"].nunique()), 1)),
        "mean_rows_per_user": mean_rows_per_user,
        "density_band_user_counts": density_counts,
        "signal_scope_counts": scope_counts,
        "tag_type_counts": tag_type_counts,
        "recommended_pool_row_counts": pool_counts,
        "recommended_pool_user_counts": users_per_pool,
        "config": {
            "pool_mode": POOL_MODE,
            "include_scene_tags": bool(INCLUDE_SCENE_TAGS),
            "include_recent_tags": bool(INCLUDE_RECENT_TAGS),
            "include_service_tags": bool(INCLUDE_SERVICE_TAGS),
            "weight_stable_cuisine": float(WEIGHT_STABLE_CUISINE),
            "weight_stable_scene": float(WEIGHT_STABLE_SCENE),
            "weight_avoid_cuisine": float(WEIGHT_AVOID_CUISINE),
            "weight_recent_cuisine": float(WEIGHT_RECENT_CUISINE),
            "weight_recent_scene": float(WEIGHT_RECENT_SCENE),
            "weight_service": float(WEIGHT_SERVICE),
        },
    }


def main() -> None:
    bridge_run = resolve_run(
        INPUT_BRIDGE_RUN_DIR,
        INPUT_BRIDGE_ROOT,
        "_full_stage11_pass2_stage09_target_bridge_v1_build",
    )
    bridge_df = pd.read_parquet(bridge_run / "user_preference_target_schema_v1.parquet")
    for col in [
        "user_id",
        "density_band",
        "recommended_pool",
        "source_pool",
        "confidence_overall",
        "service_risk_sensitivity",
        "target_schema_v1_json",
    ]:
        if col not in bridge_df.columns:
            bridge_df[col] = ""
    bridge_df["user_id"] = bridge_df["user_id"].astype(str)

    sidecar = build_rows(bridge_df)

    run_id = now_run_id()
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "user_profile_tag_profile_long_pass2_sidecar.csv"
    sidecar.to_csv(csv_path, index=False, encoding="utf-8")

    sample_cols = [
        "user_id",
        "tag",
        "tag_type",
        "pos_w",
        "neg_w",
        "net_w",
        "support",
        "tag_confidence",
        "signal_scope",
        "recommended_pool",
        "density_band",
    ]
    sample_rows = sidecar[sample_cols].head(SAMPLE_ROWS).to_dict(orient="records") if not sidecar.empty else []
    safe_json_write(out_dir / "user_profile_tag_profile_long_pass2_sidecar_sample.json", sample_rows)

    summary = build_summary(sidecar, bridge_df, run_id, bridge_run)
    safe_json_write(out_dir / "user_profile_tag_profile_long_pass2_sidecar_summary.json", summary)
    safe_json_write(
        out_dir / "run_meta.json",
        {
            "run_id": run_id,
            "run_tag": RUN_TAG,
            "input_bridge_run_dir": str(bridge_run),
            "output_csv": str(csv_path),
            "summary_json": str(out_dir / "user_profile_tag_profile_long_pass2_sidecar_summary.json"),
            "sample_json": str(out_dir / "user_profile_tag_profile_long_pass2_sidecar_sample.json"),
            "summary": summary,
        },
    )

    print(f"[OK] sidecar_rows={int(sidecar.shape[0])} sidecar_users={int(sidecar['user_id'].nunique() if not sidecar.empty else 0)}")
    print(f"[OK] output_csv={csv_path}")


if __name__ == "__main__":
    main()
