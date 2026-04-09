from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

REL_CROSS_V2_ROOT = env_or_project_path(
    "INPUT_09_STAGE10_RELATIVE_CROSS_FEATURES_V2_ROOT_DIR",
    "data/output/09_stage10_relative_cross_features_v2",
)
REL_CROSS_V2_RUN_DIR = os.getenv("INPUT_09_STAGE10_RELATIVE_CROSS_FEATURES_V2_RUN_DIR", "").strip()
OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_STAGE10_GROUP_GAP_FEATURES_V1_ROOT_DIR",
    "data/output/09_stage10_group_gap_features_v1",
)
RUN_TAG = "stage09_stage10_group_gap_features_v1_build"

MAX_ROWS = int(os.getenv("STAGE10_GROUP_GAP_MAX_ROWS", "0").strip() or 0)
SAMPLE_ROWS = int(os.getenv("STAGE10_GROUP_GAP_SAMPLE_ROWS", "12").strip() or 12)

GAP_SOURCE_COLS = [
    "pre_score",
    "signal_score",
    "quality_score",
    "semantic_score",
    "item_train_pop_count",
    "source_count",
    "schema_overlap_total_ratio_v1",
    "schema_overlap_user_ratio_v1",
    "schema_net_score_v1",
    "schema_weighted_overlap_user_ratio_v2",
    "schema_weighted_net_score_v2",
]


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = project_path(raw)
        if not p.exists():
            raise FileNotFoundError(f"run dir not found: {p}")
        return p
    return pick_latest_run(root, suffix)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    try:
        val = a.astype(float).corr(b.astype(float))
    except Exception:
        return float("nan")
    return float(val) if pd.notna(val) else float("nan")


def topk_mean(arr: pd.Series, k: int) -> float:
    vals = np.asarray(arr, dtype=np.float32)
    if vals.size == 0:
        return float("nan")
    take = min(k, vals.size)
    idx = np.argpartition(vals, vals.size - take)[-take:]
    return float(vals[idx].mean())


def main() -> None:
    rel_v2_run = resolve_run(
        REL_CROSS_V2_RUN_DIR,
        REL_CROSS_V2_ROOT,
        "_full_stage09_stage10_relative_cross_features_v2_build",
    )
    feat_path = rel_v2_run / "candidate_relative_cross_features_v2.parquet"
    feat_df = pd.read_parquet(feat_path)
    if MAX_ROWS > 0:
        feat_df = feat_df.head(int(MAX_ROWS)).copy()

    group = feat_df.groupby("user_idx", sort=False)
    derived: dict[str, pd.Series] = {}
    for col in GAP_SOURCE_COLS:
        feat_df[col] = feat_df[col].astype(float)
        grp_max = group[col].transform("max")
        grp_mean = group[col].transform("mean")
        grp_top3 = group[col].transform(lambda s: topk_mean(s, 3))
        grp_top10 = group[col].transform(lambda s: topk_mean(s, 10))
        pct = group[col].rank(method="average", pct=True).astype(np.float32)

        derived[f"{col}_gap_to_max_v3"] = (feat_df[col] - grp_max).astype(np.float32)
        derived[f"{col}_gap_to_mean_v3"] = (feat_df[col] - grp_mean).astype(np.float32)
        derived[f"{col}_gap_to_top3_v3"] = (feat_df[col] - grp_top3).astype(np.float32)
        derived[f"{col}_gap_to_top10_v3"] = (feat_df[col] - grp_top10).astype(np.float32)
        derived[f"{col}_rank_pct_v3"] = pct

    derived_df = pd.DataFrame(derived, index=feat_df.index)
    derived_df["schema_rank_minus_pre_v3"] = (
        derived_df["schema_weighted_overlap_user_ratio_v2_rank_pct_v3"]
        - derived_df["pre_score_rank_pct_v3"]
    ).astype(np.float32)
    derived_df["schema_rank_minus_pop_v3"] = (
        derived_df["schema_weighted_overlap_user_ratio_v2_rank_pct_v3"]
        - derived_df["item_train_pop_count_rank_pct_v3"]
    ).astype(np.float32)
    derived_df["schema_rank_minus_signal_v3"] = (
        derived_df["schema_weighted_overlap_user_ratio_v2_rank_pct_v3"]
        - derived_df["signal_score_rank_pct_v3"]
    ).astype(np.float32)
    derived_df["schema_top_pop_low_flag_v3"] = (
        (derived_df["schema_weighted_overlap_user_ratio_v2_rank_pct_v3"] >= 0.95)
        & (derived_df["item_train_pop_count_rank_pct_v3"] <= 0.50)
    ).astype(np.int8)
    derived_df["schema_top_pre_low_flag_v3"] = (
        (derived_df["schema_weighted_overlap_user_ratio_v2_rank_pct_v3"] >= 0.95)
        & (derived_df["pre_score_rank_pct_v3"] <= 0.50)
    ).astype(np.int8)

    out_df = pd.concat([feat_df.reset_index(drop=True), derived_df.reset_index(drop=True)], axis=1)
    audit_features = [
        "schema_weighted_overlap_user_ratio_v2_rank_pct_v3",
        "schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3",
        "schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3",
        "schema_weighted_net_score_v2_rank_pct_v3",
        "schema_rank_minus_pre_v3",
        "schema_rank_minus_pop_v3",
        "schema_rank_minus_signal_v3",
        "schema_top_pop_low_flag_v3",
        "schema_top_pre_low_flag_v3",
        "pre_score_rank_pct_v3",
        "signal_score_rank_pct_v3",
        "item_train_pop_count_rank_pct_v3",
    ]
    truth_rows = out_df.loc[out_df["is_truth"] == 1]
    nontruth_rows = out_df.loc[out_df["is_truth"] == 0]
    truth_gap = {
        col: float(truth_rows[col].astype(float).mean() - nontruth_rows[col].astype(float).mean())
        for col in audit_features
    }
    corr_summary = {
        col: {
            "corr_pre_score": safe_corr(out_df[col], out_df["pre_score"]),
            "corr_item_train_pop_count": safe_corr(out_df[col], out_df["item_train_pop_count"]),
            "corr_semantic_score": safe_corr(out_df[col], out_df["semantic_score"]),
        }
        for col in audit_features
    }

    sample_cols = [
        "user_idx", "item_idx", "business_id", "pre_rank", "is_truth",
        "schema_weighted_overlap_user_ratio_v2",
        "schema_weighted_overlap_user_ratio_v2_rank_pct_v3",
        "schema_rank_minus_pre_v3",
        "schema_rank_minus_pop_v3",
        "schema_top_pop_low_flag_v3",
        "schema_top_pre_low_flag_v3",
    ]

    run_id = now_run_id()
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path_out = out_dir / "candidate_group_gap_features_v1.parquet"
    out_df.to_parquet(feat_path_out, index=False)
    sample = {
        "truth_top_rows": out_df.loc[out_df["is_truth"] == 1, sample_cols].head(SAMPLE_ROWS).to_dict(orient="records"),
        "nontruth_top_rows": out_df.loc[out_df["is_truth"] == 0, sample_cols].head(SAMPLE_ROWS).to_dict(orient="records"),
    }
    sample_path = out_dir / "candidate_group_gap_features_v1_sample.json"
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    run_meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "inputs": {
            "relative_cross_v2_run_dir": str(rel_v2_run),
        },
        "row_counts": {
            "candidate_rows": int(len(out_df)),
            "truth_rows": int(out_df["is_truth"].sum()),
            "truth_user_count": int(out_df.loc[out_df["is_truth"] == 1, "user_idx"].nunique()),
        },
        "truth_gap": truth_gap,
        "corr_summary": corr_summary,
        "outputs": {
            "candidate_group_gap_features_v1": str(feat_path_out),
            "sample_json": str(sample_path),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
