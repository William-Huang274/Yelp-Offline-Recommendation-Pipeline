from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from pipeline.project_paths import env_or_project_path, project_path

INDEX_MAPS_ROOT = env_or_project_path("INPUT_09_INDEX_MAPS_ROOT_DIR", "data/output/09_index_maps")
INDEX_MAPS_RUN_DIR = os.getenv("INPUT_09_INDEX_MAPS_RUN_DIR", "").strip()
USER_SCHEMA_ROOT = env_or_project_path(
    "INPUT_09_USER_SCHEMA_PROJECTION_V1_ROOT_DIR",
    "data/output/09_user_schema_projection_v1",
)
USER_SCHEMA_RUN_DIR = os.getenv("INPUT_09_USER_SCHEMA_PROJECTION_V1_RUN_DIR", "").strip()
MERCHANT_EVIDENCE_ROOT = env_or_project_path(
    "INPUT_09_MERCHANT_EVIDENCE_ASSETS_V1_ROOT_DIR",
    "data/output/09_merchant_evidence_assets_v1",
)
MERCHANT_EVIDENCE_RUN_DIR = os.getenv("INPUT_09_MERCHANT_EVIDENCE_ASSETS_V1_RUN_DIR", "").strip()
SOURCE_STAGE09_BUCKET_DIR = os.getenv("SOURCE_STAGE09_BUCKET_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_STAGE10_RELATIVE_CROSS_FEATURES_V1_ROOT_DIR",
    "data/output/09_stage10_relative_cross_features_v1",
)
RUN_TAG = "stage09_stage10_relative_cross_features_v1_build"

BUCKET = int(os.getenv("STAGE10_RELATIVE_CROSS_BUCKET", "5").strip() or 5)
TOPK = int(os.getenv("STAGE10_RELATIVE_CROSS_TOPK", "150").strip() or 150)
MAX_ROWS = int(os.getenv("STAGE10_RELATIVE_CROSS_MAX_ROWS", "0").strip() or 0)
SAMPLE_ROWS = int(os.getenv("STAGE10_RELATIVE_CROSS_SAMPLE_ROWS", "12").strip() or 12)

RELATIVE_NUMERIC_COLS = [
    "pre_score",
    "signal_score",
    "quality_score",
    "semantic_score",
    "item_train_pop_count",
    "source_count",
    "tower_score",
    "seq_score",
    "head_score",
]

POS_DOMAIN_MAP = {
    "cuisine": "schema_pos_cuisine",
    "dish": "schema_pos_dish",
    "scene": "schema_pos_scene",
    "time": "schema_pos_time",
    "property": "schema_pos_property",
    "service": "schema_pos_service",
}
NEG_DOMAIN_MAP = {
    "complaint": "schema_neg_complaint",
    "cuisine": "schema_neg_cuisine",
    "dish": "schema_neg_dish",
}
MERCHANT_DOMAIN_MAP = {
    "cuisine": "evidence_cuisine_tags_v1",
    "dish": "evidence_dish_tags_v1",
    "scene": "evidence_scene_tags_v1",
    "time": "evidence_time_tags_v1",
    "property": "evidence_property_tags_v1",
    "service": "evidence_service_tags_v1",
    "complaint": "evidence_complaint_tags_v1",
}


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


def resolve_bucket_dir(raw: str) -> Path:
    if not raw:
        raise FileNotFoundError("SOURCE_STAGE09_BUCKET_DIR is required")
    p = Path(raw)
    if not p.is_absolute():
        p = project_path(raw)
    if not p.exists():
        raise FileNotFoundError(f"bucket dir not found: {p}")
    return p


def pick_candidate_file(bucket_dir: Path) -> Path:
    for name in ("candidates_pretrim.parquet", "candidates_pretrim150.parquet"):
        path = bucket_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, np.ndarray):
        return [str(v).strip() for v in value.tolist() if str(v).strip()]
    return []


def popcount_u64(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.uint64, copy=False)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def build_vocab(tags_series_list: list[pd.Series]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    next_bit = 0
    for series in tags_series_list:
        for value in series.tolist():
            for tag in as_list(value):
                if tag not in vocab:
                    if next_bit >= 63:
                        raise ValueError("tag vocab exceeds uint64 capacity")
                    vocab[tag] = next_bit
                    next_bit += 1
    return vocab


def encode_tags_to_mask(value: object, vocab: dict[str, int]) -> np.uint64:
    mask = np.uint64(0)
    for tag in as_list(value):
        bit = vocab.get(tag)
        if bit is None:
            continue
        mask |= np.uint64(1) << np.uint64(bit)
    return mask


def make_mask_series(series: pd.Series, vocab: dict[str, int]) -> pd.Series:
    return series.apply(lambda v: encode_tags_to_mask(v, vocab)).astype("uint64")


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            val = a.astype(float).corr(b.astype(float))
    except Exception:
        return float("nan")
    return float(val) if pd.notna(val) else float("nan")


def main() -> None:
    index_run = resolve_run(INDEX_MAPS_RUN_DIR, INDEX_MAPS_ROOT, "_full_stage09_index_maps_build")
    user_schema_run = resolve_run(
        USER_SCHEMA_RUN_DIR,
        USER_SCHEMA_ROOT,
        "_full_stage09_user_schema_projection_v1_build",
    )
    merchant_evidence_run = resolve_run(
        MERCHANT_EVIDENCE_RUN_DIR,
        MERCHANT_EVIDENCE_ROOT,
        "_full_stage09_merchant_evidence_assets_v1_build",
    )
    bucket_dir = resolve_bucket_dir(SOURCE_STAGE09_BUCKET_DIR)

    cand_path = pick_candidate_file(bucket_dir)
    truth_path = bucket_dir / "truth.parquet"
    user_map_path = index_run / f"bucket_{BUCKET}" / "user_index_map.parquet"
    item_map_path = index_run / f"bucket_{BUCKET}" / "item_index_map.parquet"
    user_schema_path = user_schema_run / "user_schema_profile_v1.parquet"
    merchant_evidence_path = merchant_evidence_run / "merchant_evidence_assets_v1.parquet"

    cand_cols = [
        "user_idx",
        "item_idx",
        "business_id",
        "user_segment",
        "pre_rank",
        "pre_score",
        "signal_score",
        "quality_score",
        "semantic_score",
        "semantic_support",
        "item_train_pop_count",
        "source_count",
        "tower_score",
        "seq_score",
        "head_score",
    ]
    cand_tbl = ds.dataset(cand_path, format="parquet").to_table(
        columns=cand_cols,
        filter=pc.field("pre_rank") <= TOPK,
    )
    cand_df = cand_tbl.to_pandas(types_mapper=None)
    if MAX_ROWS > 0:
        cand_df = cand_df.head(int(MAX_ROWS)).copy()

    user_map_df = pd.read_parquet(user_map_path)
    item_map_df = pd.read_parquet(item_map_path)
    user_schema_df = pd.read_parquet(user_schema_path)
    merchant_df = pd.read_parquet(merchant_evidence_path)

    cand_df = cand_df.merge(user_map_df, on="user_idx", how="left")
    cand_df = cand_df.merge(item_map_df, on="item_idx", how="left", suffixes=("", "_map"))
    cand_df["business_id"] = cand_df["business_id"].fillna(cand_df["business_id_map"])
    cand_df = cand_df.drop(columns=["business_id_map"])
    cand_df = cand_df.merge(user_schema_df, on="user_id", how="left")
    cand_df = cand_df.merge(
        merchant_df[
            [
                "business_id",
                *MERCHANT_DOMAIN_MAP.values(),
                "audit_review_count",
            ]
        ],
        on="business_id",
        how="left",
    )

    cand_df["item_pop_log"] = np.log1p(cand_df["item_train_pop_count"].astype(float).clip(lower=0.0))

    for col in RELATIVE_NUMERIC_COLS:
        cand_df[col] = cand_df[col].astype(float)

    group = cand_df.groupby("user_idx", sort=False)
    cand_df["user_group_size_v1"] = group["item_idx"].transform("size").astype(np.int32)
    for col in RELATIVE_NUMERIC_COLS:
        mean = group[col].transform("mean")
        median = group[col].transform("median")
        std = group[col].transform("std").fillna(0.0)
        cand_df[f"{col}_rel_center_v1"] = (cand_df[col] - mean).astype(np.float32)
        cand_df[f"{col}_rel_median_gap_v1"] = (cand_df[col] - median).astype(np.float32)
        cand_df[f"{col}_rel_z_v1"] = ((cand_df[col] - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
        if col in {"pre_score", "signal_score", "quality_score", "semantic_score", "item_train_pop_count", "head_score"}:
            pct = group[col].rank(method="average", pct=True).astype(np.float32)
            cand_df[f"{col}_rel_pct_v1"] = pct

    pre_rank_pct = group["pre_rank"].rank(method="average", pct=True).astype(np.float32)
    cand_df["pre_rank_rel_inv_pct_v1"] = (1.0 - pre_rank_pct).astype(np.float32)

    # Build user/merchant masks in the same refined schema space.
    all_domains = sorted(set(POS_DOMAIN_MAP) | set(NEG_DOMAIN_MAP))
    user_mask_cols: list[str] = []
    merchant_mask_cols: list[str] = []
    for domain in all_domains:
        user_series_list: list[pd.Series] = []
        merchant_series_list: list[pd.Series] = []
        if domain in POS_DOMAIN_MAP:
            user_series_list.append(cand_df[POS_DOMAIN_MAP[domain]])
        if domain in NEG_DOMAIN_MAP:
            user_series_list.append(cand_df[NEG_DOMAIN_MAP[domain]])
        merchant_series_list.append(cand_df[MERCHANT_DOMAIN_MAP[domain]])
        vocab = build_vocab(user_series_list + merchant_series_list)

        if domain in POS_DOMAIN_MAP:
            pos_col = f"user_mask_pos_{domain}_v1"
            cand_df[pos_col] = make_mask_series(cand_df[POS_DOMAIN_MAP[domain]], vocab)
            user_mask_cols.append(pos_col)
        if domain in NEG_DOMAIN_MAP:
            neg_col = f"user_mask_neg_{domain}_v1"
            cand_df[neg_col] = make_mask_series(cand_df[NEG_DOMAIN_MAP[domain]], vocab)
            user_mask_cols.append(neg_col)
        merch_col = f"merchant_mask_{domain}_v1"
        cand_df[merch_col] = make_mask_series(cand_df[MERCHANT_DOMAIN_MAP[domain]], vocab)
        merchant_mask_cols.append(merch_col)

        if domain in POS_DOMAIN_MAP:
            overlap = popcount_u64(cand_df[f"user_mask_pos_{domain}_v1"].to_numpy(dtype=np.uint64) & cand_df[merch_col].to_numpy(dtype=np.uint64))
            cand_df[f"schema_overlap_{domain}_count_v1"] = overlap.astype(np.int16)
            cand_df[f"schema_overlap_{domain}_any_v1"] = (overlap > 0).astype(np.int8)
        if domain in NEG_DOMAIN_MAP:
            conflict = popcount_u64(cand_df[f"user_mask_neg_{domain}_v1"].to_numpy(dtype=np.uint64) & cand_df[merch_col].to_numpy(dtype=np.uint64))
            cand_df[f"schema_conflict_{domain}_count_v1"] = conflict.astype(np.int16)
            cand_df[f"schema_conflict_{domain}_any_v1"] = (conflict > 0).astype(np.int8)

    overlap_cols = [c for c in cand_df.columns if c.startswith("schema_overlap_") and c.endswith("_count_v1")]
    conflict_cols = [c for c in cand_df.columns if c.startswith("schema_conflict_") and c.endswith("_count_v1")]
    derived_cols: dict[str, pd.Series] = {}
    derived_cols["schema_overlap_total_count_v1"] = cand_df[overlap_cols].sum(axis=1).astype(np.int16)
    derived_cols["schema_conflict_total_count_v1"] = cand_df[conflict_cols].sum(axis=1).astype(np.int16)
    derived_cols["schema_net_score_v1"] = (derived_cols["schema_overlap_total_count_v1"] - derived_cols["schema_conflict_total_count_v1"]).astype(np.int16)
    merchant_pref_mask_cols = [f"merchant_mask_{d}_v1" for d in POS_DOMAIN_MAP]
    user_pref_mask_cols = [f"user_mask_pos_{d}_v1" for d in POS_DOMAIN_MAP]
    user_neg_mask_cols = [f"user_mask_neg_{d}_v1" for d in NEG_DOMAIN_MAP]
    merchant_total = np.zeros(len(cand_df), dtype=np.int16)
    user_pref_total = np.zeros(len(cand_df), dtype=np.int16)
    user_neg_total = np.zeros(len(cand_df), dtype=np.int16)
    for col in merchant_pref_mask_cols:
        merchant_total = merchant_total + popcount_u64(cand_df[col].to_numpy(dtype=np.uint64)).astype(np.int16)
    for col in user_pref_mask_cols:
        user_pref_total = user_pref_total + popcount_u64(cand_df[col].to_numpy(dtype=np.uint64)).astype(np.int16)
    for col in user_neg_mask_cols:
        user_neg_total = user_neg_total + popcount_u64(cand_df[col].to_numpy(dtype=np.uint64)).astype(np.int16)
    derived_cols["schema_merchant_pref_total_count_v1"] = merchant_total
    derived_cols["schema_user_pref_total_count_v1"] = user_pref_total
    derived_cols["schema_user_neg_total_count_v1"] = user_neg_total
    merchant_total_f = pd.Series(merchant_total, index=cand_df.index, dtype="float32").replace(0, np.nan)
    user_pref_total_f = pd.Series(user_pref_total, index=cand_df.index, dtype="float32").replace(0, np.nan)
    user_neg_total_f = pd.Series(user_neg_total, index=cand_df.index, dtype="float32").replace(0, np.nan)
    derived_cols["schema_overlap_total_ratio_v1"] = (derived_cols["schema_overlap_total_count_v1"] / merchant_total_f).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    derived_cols["schema_overlap_user_ratio_v1"] = (derived_cols["schema_overlap_total_count_v1"] / user_pref_total_f).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    derived_cols["schema_conflict_user_ratio_v1"] = (derived_cols["schema_conflict_total_count_v1"] / user_neg_total_f).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

    # Cross-difference and challenger/conflict indicators.
    derived_cols["semantic_minus_item_pop_log_v1"] = (cand_df["semantic_score"] - cand_df["item_pop_log"]).astype(np.float32)
    derived_cols["signal_minus_item_pop_log_v1"] = (cand_df["signal_score"] - cand_df["item_pop_log"]).astype(np.float32)
    derived_cols["quality_minus_item_pop_log_v1"] = (cand_df["quality_score"] - cand_df["item_pop_log"]).astype(np.float32)
    derived_cols["semantic_minus_quality_v1"] = (cand_df["semantic_score"] - cand_df["quality_score"]).astype(np.float32)
    derived_cols["schema_net_minus_pop_z_v1"] = (derived_cols["schema_net_score_v1"].astype(float) - cand_df["item_train_pop_count_rel_z_v1"]).astype(np.float32)
    derived_cols["schema_overlap_minus_pre_z_v1"] = (derived_cols["schema_overlap_total_count_v1"].astype(float) - cand_df["pre_score_rel_z_v1"]).astype(np.float32)
    derived_cols["schema_overlap_minus_semantic_z_v1"] = (derived_cols["schema_overlap_total_count_v1"].astype(float) - cand_df["semantic_score_rel_z_v1"]).astype(np.float32)
    derived_cols["high_source_low_semantic_flag_v1"] = (
        (cand_df["source_count_rel_z_v1"] > 0.5) & (cand_df["semantic_score_rel_z_v1"] < -0.5)
    ).astype(np.int8)
    derived_cols["challenger_low_pop_high_semantic_flag_v1"] = (
        (cand_df["item_train_pop_count_rel_z_v1"] < -0.5) & (cand_df["semantic_score_rel_z_v1"] > 0.5)
    ).astype(np.int8)
    derived_cols["quality_high_conflict_flag_v1"] = (
        (cand_df["quality_score_rel_z_v1"] > 0.5) & (derived_cols["schema_conflict_total_count_v1"] > 0)
    ).astype(np.int8)
    derived_cols["high_support_low_semantic_flag_v1"] = (
        (cand_df["semantic_support"].astype(float) > cand_df["semantic_support"].astype(float).median()) & (cand_df["semantic_score_rel_z_v1"] < -0.5)
    ).astype(np.int8)
    cand_df = pd.concat([cand_df, pd.DataFrame(derived_cols, index=cand_df.index)], axis=1)

    truth_df = pd.read_parquet(truth_path)[["user_idx", "true_item_idx"]].rename(columns={"true_item_idx": "item_idx"})
    truth_df["is_truth"] = 1
    cand_df = cand_df.merge(truth_df, on=["user_idx", "item_idx"], how="left")
    cand_df["is_truth"] = cand_df["is_truth"].fillna(0).astype(np.int8)

    feature_cols = [
        c for c in cand_df.columns
        if c.endswith("_v1")
        and ("mask_" not in c)
        or c in {
            "user_idx", "item_idx", "business_id", "user_id", "user_segment", "pre_rank",
            "pre_score", "signal_score", "quality_score", "semantic_score", "semantic_support",
            "item_train_pop_count", "item_pop_log", "source_count", "tower_score", "seq_score",
            "head_score", "is_truth",
        }
    ]
    feature_cols = list(dict.fromkeys(feature_cols))
    feat_df = cand_df[feature_cols].copy()

    audit_features = [
        "schema_overlap_total_count_v1",
        "schema_conflict_total_count_v1",
        "schema_net_score_v1",
        "schema_overlap_total_ratio_v1",
        "schema_overlap_user_ratio_v1",
        "schema_conflict_user_ratio_v1",
        "semantic_minus_item_pop_log_v1",
        "signal_minus_item_pop_log_v1",
        "quality_minus_item_pop_log_v1",
        "schema_net_minus_pop_z_v1",
        "schema_overlap_minus_pre_z_v1",
        "schema_overlap_minus_semantic_z_v1",
        "high_source_low_semantic_flag_v1",
        "challenger_low_pop_high_semantic_flag_v1",
        "quality_high_conflict_flag_v1",
        "high_support_low_semantic_flag_v1",
        "pre_score_rel_z_v1",
        "signal_score_rel_z_v1",
        "quality_score_rel_z_v1",
        "semantic_score_rel_z_v1",
        "item_train_pop_count_rel_z_v1",
        "source_count_rel_z_v1",
    ]

    truth_summary: dict[str, dict[str, float | int]] = {}
    for flag, sub in feat_df.groupby("is_truth"):
        truth_summary[str(int(flag))] = {
            "rows": int(len(sub)),
            **{f"{col}_mean": float(sub[col].astype(float).mean()) for col in audit_features},
        }

    truth_gap = {}
    truth_rows = feat_df.loc[feat_df["is_truth"] == 1]
    nontruth_rows = feat_df.loc[feat_df["is_truth"] == 0]
    for col in audit_features:
        truth_gap[col] = float(truth_rows[col].astype(float).mean() - nontruth_rows[col].astype(float).mean())

    segment_summary = {}
    for seg, sub in feat_df.groupby("user_segment"):
        segment_summary[str(seg)] = {
            "rows": int(len(sub)),
            "truth_rows": int(sub["is_truth"].sum()),
            **{f"{col}_mean": float(sub[col].astype(float).mean()) for col in audit_features},
        }

    corr_summary = {}
    for col in audit_features:
        corr_summary[col] = {
            "corr_pre_score": safe_corr(feat_df[col], feat_df["pre_score"]),
            "corr_item_train_pop_count": safe_corr(feat_df[col], feat_df["item_train_pop_count"]),
            "corr_semantic_score": safe_corr(feat_df[col], feat_df["semantic_score"]),
        }

    run_id = now_run_id()
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "candidate_relative_cross_features_v1.parquet"
    feat_df.to_parquet(feat_path, index=False)

    sample_cols = [
        "user_idx", "item_idx", "business_id", "user_segment", "pre_rank", "is_truth",
        "pre_score", "signal_score", "quality_score", "semantic_score", "item_train_pop_count",
        "schema_overlap_total_count_v1", "schema_conflict_total_count_v1", "schema_net_score_v1",
        "schema_overlap_total_ratio_v1", "semantic_minus_item_pop_log_v1", "schema_net_minus_pop_z_v1",
        "challenger_low_pop_high_semantic_flag_v1", "quality_high_conflict_flag_v1",
    ]
    sample = {
        "truth_top_rows": feat_df.loc[feat_df["is_truth"] == 1, sample_cols].head(SAMPLE_ROWS).to_dict(orient="records"),
        "nontruth_top_rows": feat_df.loc[feat_df["is_truth"] == 0, sample_cols].head(SAMPLE_ROWS).to_dict(orient="records"),
    }
    (out_dir / "candidate_relative_cross_features_v1_sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "bucket": int(BUCKET),
        "topk": int(TOPK),
        "max_rows": int(MAX_ROWS),
        "inputs": {
            "source_stage09_bucket_dir": str(bucket_dir),
            "index_maps_run_dir": str(index_run),
            "user_schema_run_dir": str(user_schema_run),
            "merchant_evidence_run_dir": str(merchant_evidence_run),
        },
        "row_counts": {
            "candidate_rows": int(len(feat_df)),
            "truth_rows": int(feat_df["is_truth"].sum()),
            "truth_user_count": int(feat_df.loc[feat_df["is_truth"] == 1, "user_idx"].nunique()),
        },
        "truth_summary": truth_summary,
        "truth_gap": truth_gap,
        "segment_summary": segment_summary,
        "corr_summary": corr_summary,
        "outputs": {
            "candidate_relative_cross_features_v1": str(feat_path),
            "sample_json": str(out_dir / "candidate_relative_cross_features_v1_sample.json"),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
