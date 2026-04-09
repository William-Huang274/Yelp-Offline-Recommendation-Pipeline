from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LIGHT_MAX_TRAIN = 7
MID_MAX_TRAIN = 19
TOP_SLICE_K = 12
PREF_SLICE_MIN_SCORE = 0.05

MEAL_PREF_SPECS = [
    ("breakfast", "breakfast_pref"),
    ("lunch", "lunch_pref"),
    ("dinner", "dinner_pref"),
    ("late_night", "late_night_pref"),
]
SCENE_PREF_SPECS = [
    ("family", "family_scene_pref"),
    ("group", "group_scene_pref"),
    ("date", "date_scene_pref"),
    ("nightlife", "nightlife_scene_pref"),
    ("fast_casual", "fast_casual_pref"),
    ("sitdown", "sitdown_pref"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bucket5-focused slice eval from stage10 user diagnostics.")
    p.add_argument("--stage10-user-audit", required=True, help="Path to user_diagnostics.parquet")
    p.add_argument("--bucket-dir", required=True, help="Stage09 bucket dir containing train_history.parquet")
    p.add_argument("--profile-run-dir", required=True, help="Profile run dir containing user_intent_profile_v2.parquet")
    p.add_argument("--user-schema-run-dir", default="", help="Optional user schema run dir containing user_schema_profile_v1.parquet")
    p.add_argument("--output-dir", required=True, help="Output dir")
    return p.parse_args()


def _ndcg_from_rank(rank_series: pd.Series) -> np.ndarray:
    r = pd.to_numeric(rank_series, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    out = np.zeros(len(r), dtype=np.float64)
    mask = r > 0
    if mask.any():
        out[mask] = 1.0 / np.log2(r[mask] + 1.0)
    return out


def _activity_band(n_train: int) -> str:
    if n_train <= 0:
        return "unknown"
    if n_train <= LIGHT_MAX_TRAIN:
        return "light"
    if n_train <= MID_MAX_TRAIN:
        return "mid"
    return "heavy"


def _popularity_band(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    out = pd.Series(["unknown"] * len(arr), index=arr.index, dtype="object")
    nonnull = arr.dropna()
    if len(nonnull) < 10:
        return out
    q20 = float(nonnull.quantile(0.2))
    q80 = float(nonnull.quantile(0.8))
    out.loc[arr.notna() & (arr <= q20)] = "tail"
    out.loc[arr.notna() & (arr >= q80)] = "head"
    out.loc[arr.notna() & (arr > q20) & (arr < q80)] = "mid"
    return out


def _normalize_label(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return "unknown"
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "unknown"


def _compress_top_values(values: pd.Series, topk: int = TOP_SLICE_K) -> pd.Series:
    clean = values.fillna("").astype(str).map(_normalize_label)
    top = clean.loc[clean.ne("unknown")].value_counts().head(int(topk)).index.tolist()
    if not top:
        return clean
    return clean.where(clean.isin(top), np.where(clean.eq("unknown"), "unknown", "other"))


def _top_pref_label(df: pd.DataFrame, specs: list[tuple[str, str]], min_signal: float = PREF_SLICE_MIN_SCORE) -> pd.Series:
    out = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")
    available = [(label, col) for label, col in specs if col in df.columns]
    if not available:
        return out
    mats = [
        pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        for _label, col in available
    ]
    score_mat = np.column_stack(mats)
    if score_mat.size == 0:
        return out
    best_idx = score_mat.argmax(axis=1)
    best_val = score_mat.max(axis=1)
    labels = np.array([label for label, _col in available], dtype=object)
    picked = labels[best_idx]
    out.loc[best_val >= float(min_signal)] = picked[best_val >= float(min_signal)]
    return out


def _alignment_band(left: pd.Series, right: pd.Series) -> pd.Series:
    l = left.fillna("").astype(str).map(_normalize_label)
    r = right.fillna("").astype(str).map(_normalize_label)
    out = pd.Series(["unknown"] * len(l), index=l.index, dtype="object")
    both = l.ne("unknown") & r.ne("unknown")
    out.loc[both & l.eq(r)] = "aligned"
    out.loc[both & l.ne(r)] = "mismatch"
    return out


def _pick_negative_signal_flag(schema_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.Series:
    base = pd.Series(0, index=schema_df.index, dtype="int64")
    if "has_negative_avoidance_view" in schema_df.columns:
        base = np.maximum(base.to_numpy(dtype=np.int64), pd.to_numeric(schema_df["has_negative_avoidance_view"], errors="coerce").fillna(0).astype(int).to_numpy())
        base = pd.Series(base, index=schema_df.index, dtype="int64")

    complaint_cols = [c for c in schema_df.columns if "complaint" in c.lower() and c != "user_id"]
    for c in complaint_cols:
        flag = schema_df[c].fillna("").astype(str).str.len().gt(0).astype(int)
        base = np.maximum(base.to_numpy(dtype=np.int64), flag.to_numpy(dtype=np.int64))
        base = pd.Series(base, index=schema_df.index, dtype="int64")

    if "negative_top_cuisine" in profile_df.columns:
        prof_neg = profile_df["negative_top_cuisine"].fillna("").astype(str).str.len().gt(0).astype(int)
        base = np.maximum(base.to_numpy(dtype=np.int64), prof_neg.to_numpy(dtype=np.int64))
        base = pd.Series(base, index=schema_df.index, dtype="int64")
    return base


def _load_truth_user_map(bucket_dir: Path) -> pd.DataFrame:
    truth_path = bucket_dir / "truth.parquet"
    if not truth_path.exists():
        return pd.DataFrame(columns=["user_idx", "user_id"])
    truth_df = pd.read_parquet(truth_path)
    keep = [c for c in ("user_idx", "user_id") if c in truth_df.columns]
    if len(keep) < 2:
        return pd.DataFrame(columns=["user_idx", "user_id"])
    return truth_df[keep].dropna().drop_duplicates(["user_idx"]).reset_index(drop=True)


def slice_summary(pdf: pd.DataFrame, slice_name: str, slice_value: str) -> dict[str, Any]:
    n = int(len(pdf))
    if n <= 0:
        return {
            "slice_name": slice_name,
            "slice_value": slice_value,
            "n_users": 0,
            "pre_recall_at_k": 0.0,
            "learned_recall_at_k": 0.0,
            "recall_delta": 0.0,
            "pre_ndcg_at_k": 0.0,
            "learned_ndcg_at_k": 0.0,
            "ndcg_delta": 0.0,
            "improved_rate": 0.0,
            "degraded_rate": 0.0,
        }
    pre_recall = float(pd.to_numeric(pdf["pre_hit"], errors="coerce").fillna(0).mean())
    learned_recall = float(pd.to_numeric(pdf["learned_hit"], errors="coerce").fillna(0).mean())
    pre_ndcg = float(_ndcg_from_rank(pdf["pre_hit_rank"]).mean())
    learned_ndcg = float(_ndcg_from_rank(pdf["learned_hit_rank"]).mean())
    improved_rate = float(pd.to_numeric(pdf["improved"], errors="coerce").fillna(0).mean())
    degraded_rate = float(pd.to_numeric(pdf["degraded"], errors="coerce").fillna(0).mean())
    return {
        "slice_name": slice_name,
        "slice_value": slice_value,
        "n_users": n,
        "pre_recall_at_k": pre_recall,
        "learned_recall_at_k": learned_recall,
        "recall_delta": learned_recall - pre_recall,
        "pre_ndcg_at_k": pre_ndcg,
        "learned_ndcg_at_k": learned_ndcg,
        "ndcg_delta": learned_ndcg - pre_ndcg,
        "improved_rate": improved_rate,
        "degraded_rate": degraded_rate,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_dir = Path(args.bucket_dir)

    audit_df = pd.read_parquet(args.stage10_user_audit)
    hist_df = pd.read_parquet(bucket_dir / "train_history.parquet", columns=["user_idx"])
    prof_df = pd.read_parquet(Path(args.profile_run_dir) / "user_intent_profile_v2.parquet")
    truth_user_map = _load_truth_user_map(bucket_dir)
    schema_df = pd.DataFrame(columns=["user_id"])
    if args.user_schema_run_dir:
        schema_path = Path(args.user_schema_run_dir) / "user_schema_profile_v1.parquet"
        if schema_path.exists():
            schema_df = pd.read_parquet(schema_path)

    hist_counts = hist_df.groupby("user_idx").size().rename("n_train").reset_index()
    prof_keep = [
        "user_idx",
        "user_id",
        "has_profile_row_v2",
        "is_cold_start_profile_v3",
        "has_history_but_no_profile_v3",
        "has_recent_intent_view",
        "cuisine_shift_flag",
        "long_term_top_cuisine",
        "recent_top_cuisine",
        "recent_top_cuisine_source",
        "negative_top_cuisine",
        "typed_intent_primary_cuisine",
        "typed_intent_primary_source",
        "typed_intent_confidence",
        "typed_intent_confidence_band",
        "typed_intent_conflict_score",
        "typed_intent_conflict_band",
        "typed_intent_staleness_score",
        "typed_intent_staleness_band",
        "breakfast_pref",
        "lunch_pref",
        "dinner_pref",
        "late_night_pref",
        "family_scene_pref",
        "group_scene_pref",
        "date_scene_pref",
        "nightlife_scene_pref",
        "fast_casual_pref",
        "sitdown_pref",
    ]
    prof_keep = [c for c in prof_keep if c in prof_df.columns]
    prof_df = prof_df[prof_keep].copy()
    schema_keep_cols = ["user_id"]
    if "has_negative_avoidance_view" in schema_df.columns:
        schema_keep_cols.append("has_negative_avoidance_view")
    complaint_cols = [c for c in schema_df.columns if "complaint" in c.lower() and c != "user_id"]
    schema_keep_cols.extend(complaint_cols)
    schema_df = schema_df[[c for c in schema_keep_cols if c in schema_df.columns]].copy()

    merged = audit_df.merge(hist_counts, on="user_idx", how="left")
    if "user_id" not in merged.columns and not truth_user_map.empty:
        merged = merged.merge(truth_user_map, on="user_idx", how="left")

    if "user_idx" in prof_df.columns:
        merged = merged.merge(prof_df, on="user_idx", how="left")
    elif "user_id" in prof_df.columns and "user_id" in merged.columns:
        merged = merged.merge(prof_df, on="user_id", how="left")
    else:
        for c in prof_keep:
            if c not in {"user_idx", "user_id"} and c not in merged.columns:
                merged[c] = np.nan

    if "user_id" in merged.columns and "user_id" in schema_df.columns:
        merged = merged.merge(schema_df, on="user_id", how="left")
    else:
        merged["has_negative_avoidance_view"] = 0

    merged["n_train"] = pd.to_numeric(merged.get("n_train"), errors="coerce").fillna(0).astype(int)
    merged["activity_band"] = merged["n_train"].map(_activity_band)
    merged["popularity_band"] = _popularity_band(merged.get("item_train_pop_count"))
    if "is_cold_start_profile_v3" not in merged.columns:
        merged["is_cold_start_profile_v3"] = 0
    merged["is_cold_start_profile_v3"] = pd.to_numeric(merged["is_cold_start_profile_v3"], errors="coerce").fillna(0).astype(int)
    if "has_recent_intent_view" not in merged.columns:
        merged["has_recent_intent_view"] = 0
    if "cuisine_shift_flag" not in merged.columns:
        merged["cuisine_shift_flag"] = 0
    merged["has_recent_intent_view"] = pd.to_numeric(merged["has_recent_intent_view"], errors="coerce").fillna(0).astype(int)
    merged["cuisine_shift_flag"] = pd.to_numeric(merged["cuisine_shift_flag"], errors="coerce").fillna(0).astype(int)
    merged["profile_band"] = np.where(merged["is_cold_start_profile_v3"] > 0, "cold_start", "profiled")
    merged["recent_intent_view_band"] = np.where(merged["has_recent_intent_view"] > 0, "recent_intent", "no_recent_intent")
    merged["cuisine_shift_band"] = np.where(merged["cuisine_shift_flag"] > 0, "shifted", "stable")
    merged["truth_cuisine_slice"] = _compress_top_values(merged.get("primary_category", pd.Series(dtype="object")))
    merged["recent_top_cuisine_slice"] = _compress_top_values(merged.get("recent_top_cuisine", pd.Series(dtype="object")))
    merged["long_term_top_cuisine_slice"] = _compress_top_values(merged.get("long_term_top_cuisine", pd.Series(dtype="object")))
    merged["negative_top_cuisine_slice"] = _compress_top_values(merged.get("negative_top_cuisine", pd.Series(dtype="object")))
    merged["typed_intent_primary_cuisine_slice"] = _compress_top_values(merged.get("typed_intent_primary_cuisine", pd.Series(dtype="object")))
    if "recent_top_cuisine_source" not in merged.columns:
        merged["recent_top_cuisine_source"] = "none"
    merged["recent_top_cuisine_source"] = merged["recent_top_cuisine_source"].fillna("").astype(str).replace("", "none")
    if "typed_intent_primary_source" not in merged.columns:
        merged["typed_intent_primary_source"] = "unknown"
    merged["typed_intent_primary_source"] = merged["typed_intent_primary_source"].fillna("").astype(str).replace("", "unknown")
    if "typed_intent_confidence_band" not in merged.columns:
        merged["typed_intent_confidence_band"] = "unknown"
    merged["typed_intent_confidence_band"] = merged["typed_intent_confidence_band"].fillna("").astype(str).replace("", "unknown")
    if "typed_intent_conflict_band" not in merged.columns:
        merged["typed_intent_conflict_band"] = "unknown"
    merged["typed_intent_conflict_band"] = merged["typed_intent_conflict_band"].fillna("").astype(str).replace("", "unknown")
    if "typed_intent_staleness_band" not in merged.columns:
        merged["typed_intent_staleness_band"] = "unknown"
    merged["typed_intent_staleness_band"] = merged["typed_intent_staleness_band"].fillna("").astype(str).replace("", "unknown")
    merged["meal_focus_slice"] = _top_pref_label(merged, MEAL_PREF_SPECS)
    merged["scene_focus_slice"] = _top_pref_label(merged, SCENE_PREF_SPECS)
    merged["recent_truth_alignment"] = _alignment_band(merged.get("recent_top_cuisine", pd.Series(dtype="object")), merged.get("primary_category", pd.Series(dtype="object")))
    merged["long_term_truth_alignment"] = _alignment_band(merged.get("long_term_top_cuisine", pd.Series(dtype="object")), merged.get("primary_category", pd.Series(dtype="object")))
    merged["typed_intent_truth_alignment"] = _alignment_band(merged.get("typed_intent_primary_cuisine", pd.Series(dtype="object")), merged.get("primary_category", pd.Series(dtype="object")))

    schema_for_flag = merged[[c for c in merged.columns if c == "user_id" or c == "has_negative_avoidance_view" or "complaint" in c.lower()]].copy()
    profile_for_flag = merged[[c for c in merged.columns if c == "negative_top_cuisine"]].copy()
    merged["negative_signal_strong"] = _pick_negative_signal_flag(schema_for_flag, profile_for_flag)
    merged["negative_signal_band"] = np.where(merged["negative_signal_strong"] > 0, "negative_strong", "ordinary")

    rows: list[dict[str, Any]] = []
    for slice_name in (
        "activity_band",
        "popularity_band",
        "profile_band",
        "negative_signal_band",
        "recent_intent_view_band",
        "cuisine_shift_band",
        "truth_cuisine_slice",
        "recent_top_cuisine_slice",
        "long_term_top_cuisine_slice",
        "negative_top_cuisine_slice",
        "typed_intent_primary_cuisine_slice",
        "recent_top_cuisine_source",
        "typed_intent_primary_source",
        "typed_intent_confidence_band",
        "typed_intent_conflict_band",
        "typed_intent_staleness_band",
        "meal_focus_slice",
        "scene_focus_slice",
        "recent_truth_alignment",
        "long_term_truth_alignment",
        "typed_intent_truth_alignment",
    ):
        for slice_value, g in merged.groupby(slice_name, dropna=False):
            rows.append(slice_summary(g, slice_name, str(slice_value)))

    slice_df = pd.DataFrame(rows)
    slice_df.to_csv(out_dir / "focus_slice_metrics.csv", index=False, encoding="utf-8")
    merged.to_parquet(out_dir / "focus_user_table.parquet", index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage10_user_audit": str(args.stage10_user_audit),
        "bucket_dir": str(args.bucket_dir),
        "profile_run_dir": str(args.profile_run_dir),
        "user_schema_run_dir": str(args.user_schema_run_dir),
        "n_users": int(len(merged)),
        "outputs": {
            "focus_slice_metrics_csv": str(out_dir / "focus_slice_metrics.csv"),
            "focus_user_table_parquet": str(out_dir / "focus_user_table.parquet"),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
