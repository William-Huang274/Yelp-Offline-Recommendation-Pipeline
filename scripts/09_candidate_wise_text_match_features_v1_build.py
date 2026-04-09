from __future__ import annotations

import json
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
SOURCE_STAGE09_BUCKET_DIR = os.getenv("SOURCE_STAGE09_BUCKET_DIR", "").strip()
TEXT_EMBED_ROOT = env_or_project_path(
    "INPUT_09_CANDIDATE_WISE_TEXT_EMBEDDINGS_V1_ROOT_DIR",
    "data/output/09_candidate_wise_text_embeddings_v1",
)
TEXT_EMBED_RUN_DIR = os.getenv("INPUT_09_CANDIDATE_WISE_TEXT_EMBEDDINGS_V1_RUN_DIR", "").strip()
OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_CANDIDATE_WISE_TEXT_MATCH_FEATURES_V1_ROOT_DIR",
    "data/output/09_candidate_wise_text_match_features_v1",
)
RUN_TAG = "stage09_candidate_wise_text_match_features_v1_build"
BUCKET = int(os.getenv("CANDIDATE_WISE_TEXT_MATCH_BUCKET", "5").strip() or 5)
TOPK = int(os.getenv("CANDIDATE_WISE_TEXT_MATCH_TOPK", "150").strip() or 150)
CHUNK_SIZE = int(os.getenv("CANDIDATE_WISE_TEXT_MATCH_CHUNK_SIZE", "50000").strip() or 50000)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_optional_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = project_path(raw)
        if not p.exists():
            raise FileNotFoundError(f"run dir not found: {p}")
        return p
    return pick_latest_run(root, suffix)


def resolve_bucket_dir(raw: str, bucket: int) -> Path:
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = project_path(raw)
        return p
    raise FileNotFoundError("SOURCE_STAGE09_BUCKET_DIR is required")


def pick_candidate_file(bucket_dir: Path) -> Path:
    for name in ("candidates_pretrim.parquet", "candidates_pretrim150.parquet"):
        path = bucket_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def chunk_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b, dtype=np.float32).astype(np.float32)


def build_pos_map(ids_df: pd.DataFrame, id_col: str) -> dict[str, int]:
    return {str(v): int(i) for i, v in enumerate(ids_df[id_col].astype(str).tolist())}


def main() -> None:
    index_run = resolve_optional_run(
        INDEX_MAPS_RUN_DIR,
        INDEX_MAPS_ROOT,
        "_full_stage09_index_maps_build",
    )
    embed_run = resolve_optional_run(
        TEXT_EMBED_RUN_DIR,
        TEXT_EMBED_ROOT,
        "_full_stage09_candidate_wise_text_embeddings_v1_build",
    )
    bucket_dir = resolve_bucket_dir(SOURCE_STAGE09_BUCKET_DIR, BUCKET)

    cand_path = pick_candidate_file(bucket_dir)
    truth_path = bucket_dir / "truth.parquet"
    user_map_path = index_run / f"bucket_{BUCKET}" / "user_index_map.parquet"

    cand_ds = ds.dataset(cand_path, format="parquet")
    cand_tbl = cand_ds.to_table(
        columns=["user_idx", "item_idx", "business_id", "user_segment", "pre_rank", "item_train_pop_count", "pre_score", "head_score"],
        filter=pc.field("pre_rank") <= TOPK,
    )
    cand_df = cand_tbl.to_pandas(types_mapper=None)

    user_map_df = pd.read_parquet(user_map_path)
    cand_df = cand_df.merge(user_map_df, on="user_idx", how="left")

    user_ids_df = pd.read_parquet(embed_run / "user_text_embedding_ids_v1.parquet")
    merchant_ids_df = pd.read_parquet(embed_run / "merchant_text_embedding_ids_v1.parquet")
    user_pos = build_pos_map(user_ids_df, "user_id")
    merchant_pos = build_pos_map(merchant_ids_df, "business_id")

    cand_df["user_embed_pos"] = cand_df["user_id"].astype(str).map(user_pos)
    cand_df["merchant_embed_pos"] = cand_df["business_id"].astype(str).map(merchant_pos)
    cand_df = cand_df.loc[cand_df["user_embed_pos"].notna() & cand_df["merchant_embed_pos"].notna()].copy()
    cand_df["user_embed_pos"] = cand_df["user_embed_pos"].astype(int)
    cand_df["merchant_embed_pos"] = cand_df["merchant_embed_pos"].astype(int)

    user_npz = np.load(embed_run / "user_text_embeddings_v1.npz")
    merchant_npz = np.load(embed_run / "merchant_text_embeddings_v1.npz")

    out = {
        "sim_long_pref_core": np.zeros(len(cand_df), dtype=np.float32),
        "sim_recent_intent_semantic": np.zeros(len(cand_df), dtype=np.float32),
        "sim_recent_intent_pos": np.zeros(len(cand_df), dtype=np.float32),
        "sim_negative_avoid_neg": np.zeros(len(cand_df), dtype=np.float32),
        "sim_negative_avoid_core": np.zeros(len(cand_df), dtype=np.float32),
        "sim_context_merchant": np.zeros(len(cand_df), dtype=np.float32),
        "sim_conflict_gap": np.zeros(len(cand_df), dtype=np.float32),
    }

    u_long = user_npz["user_long_pref_emb"]
    u_recent = user_npz["user_recent_intent_emb"]
    u_neg = user_npz["user_negative_avoid_emb"]
    u_ctx = user_npz["user_context_emb"]
    m_core = merchant_npz["merchant_core_emb"]
    m_sem = merchant_npz["merchant_semantic_emb"]
    m_pos = merchant_npz["merchant_pos_emb"]
    m_neg = merchant_npz["merchant_neg_emb"]
    m_ctx = merchant_npz["merchant_context_emb"]

    u_idx = cand_df["user_embed_pos"].to_numpy(dtype=np.int32)
    m_idx = cand_df["merchant_embed_pos"].to_numpy(dtype=np.int32)
    for start in range(0, len(cand_df), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(cand_df))
        sl = slice(start, end)
        out["sim_long_pref_core"][sl] = chunk_dot(u_long[u_idx[sl]], m_core[m_idx[sl]])
        out["sim_recent_intent_semantic"][sl] = chunk_dot(u_recent[u_idx[sl]], m_sem[m_idx[sl]])
        out["sim_recent_intent_pos"][sl] = chunk_dot(u_recent[u_idx[sl]], m_pos[m_idx[sl]])
        out["sim_negative_avoid_neg"][sl] = chunk_dot(u_neg[u_idx[sl]], m_neg[m_idx[sl]])
        out["sim_negative_avoid_core"][sl] = chunk_dot(u_neg[u_idx[sl]], m_core[m_idx[sl]])
        out["sim_context_merchant"][sl] = chunk_dot(u_ctx[u_idx[sl]], m_ctx[m_idx[sl]])
        out["sim_conflict_gap"][sl] = out["sim_negative_avoid_core"][sl] - out["sim_long_pref_core"][sl]

    feat_df = cand_df[["user_idx", "item_idx", "business_id", "user_segment", "pre_rank", "item_train_pop_count", "pre_score", "head_score"]].copy()
    for k, v in out.items():
        feat_df[k] = v

    truth_tbl = ds.dataset(truth_path, format="parquet").to_table(columns=["user_idx", "true_item_idx"])
    truth_df = truth_tbl.to_pandas().rename(columns={"true_item_idx": "item_idx"})
    truth_df["is_truth"] = 1
    feat_df = feat_df.merge(truth_df, on=["user_idx", "item_idx"], how="left")
    feat_df["is_truth"] = feat_df["is_truth"].fillna(0).astype(int)

    sim_cols = [
        "sim_long_pref_core",
        "sim_recent_intent_semantic",
        "sim_recent_intent_pos",
        "sim_negative_avoid_neg",
        "sim_negative_avoid_core",
        "sim_context_merchant",
        "sim_conflict_gap",
    ]
    overall = {}
    for col in sim_cols:
        series = feat_df[col].astype(float)
        overall[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "corr_item_train_pop_count": float(series.corr(feat_df["item_train_pop_count"].astype(float))),
            "corr_pre_score": float(series.corr(feat_df["pre_score"].astype(float))),
        }
    truth_summary = {}
    grouped = feat_df.groupby("is_truth")
    for flag, sub in grouped:
        truth_summary[str(int(flag))] = {
            "rows": int(len(sub)),
            **{f"{col}_mean": float(sub[col].mean()) for col in sim_cols},
        }
    segment_summary = {}
    for seg, sub in feat_df.groupby("user_segment"):
        segment_summary[str(seg)] = {
            "rows": int(len(sub)),
            "truth_rows": int(sub["is_truth"].sum()),
            **{f"{col}_mean": float(sub[col].mean()) for col in sim_cols},
        }

    run_id = now_run_id()
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_dir / "candidate_text_match_features_v1.parquet", index=False)
    sample = {
        "truth_top_rows": feat_df.loc[feat_df["is_truth"] == 1].head(10).to_dict(orient="records"),
        "nontruth_top_rows": feat_df.loc[feat_df["is_truth"] == 0].head(10).to_dict(orient="records"),
    }
    (out_dir / "candidate_text_match_features_v1_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    run_meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "bucket": int(BUCKET),
        "topk": int(TOPK),
        "chunk_size": int(CHUNK_SIZE),
        "inputs": {
            "source_stage09_bucket_dir": str(bucket_dir),
            "index_maps_run_dir": str(index_run),
            "text_embeddings_run_dir": str(embed_run),
        },
        "row_counts": {
            "candidate_rows_after_topk": int(len(feat_df)),
            "truth_rows_in_window": int(feat_df["is_truth"].sum()),
        },
        "overall_summary": overall,
        "truth_summary": truth_summary,
        "segment_summary": segment_summary,
        "outputs": {
            "candidate_text_match_features_v1": str(out_dir / "candidate_text_match_features_v1.parquet"),
            "sample_json": str(out_dir / "candidate_text_match_features_v1_sample.json"),
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
