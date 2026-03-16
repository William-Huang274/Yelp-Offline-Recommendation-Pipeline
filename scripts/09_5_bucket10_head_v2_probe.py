from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pipeline.project_paths import env_or_project_path

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from xgboost import XGBRanker
except Exception:
    XGBRanker = None


RUN_TAG = "stage09_bucket10_head_v2_probe"

INPUT_HEAD_V2_RUN_DIR = os.getenv("INPUT_HEAD_V2_RUN_DIR", "").strip()
INPUT_HEAD_V2_ROOT = env_or_project_path("INPUT_HEAD_V2_ROOT_DIR", "data/output/09_head_v2_table")
INPUT_HEAD_V2_SUFFIX = f"_{RUN_TAG.replace('_probe', '_table')}"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_HEAD_V2_PROBE_ROOT_DIR", "data/output/09_head_v2_probe")

TOPN_LIST = [int(x.strip()) for x in os.getenv("HEAD_V2_PROBE_TOPN_LIST", "80,150,250").split(",") if x.strip()]
N_FOLDS = int(os.getenv("HEAD_V2_PROBE_FOLDS", "3").strip() or 3)
RANDOM_SEED = int(os.getenv("HEAD_V2_PROBE_RANDOM_SEED", "42").strip() or 42)
MODEL_BACKENDS = [x.strip().lower() for x in os.getenv("HEAD_V2_PROBE_BACKENDS", "xgboost_ranker").split(",") if x.strip()]
PROBE_SCOPE = os.getenv("HEAD_V2_PROBE_SCOPE", "all_candidates").strip().lower() or "all_candidates"

XGB_N_ESTIMATORS = int(os.getenv("HEAD_V2_PROBE_XGB_N_ESTIMATORS", "180").strip() or 180)
XGB_MAX_DEPTH = int(os.getenv("HEAD_V2_PROBE_XGB_MAX_DEPTH", "6").strip() or 6)
XGB_LEARNING_RATE = float(os.getenv("HEAD_V2_PROBE_XGB_LEARNING_RATE", "0.05").strip() or 0.05)
XGB_SUBSAMPLE = float(os.getenv("HEAD_V2_PROBE_XGB_SUBSAMPLE", "0.8").strip() or 0.8)
XGB_COLSAMPLE = float(os.getenv("HEAD_V2_PROBE_XGB_COLSAMPLE", "0.8").strip() or 0.8)
XGB_REG_LAMBDA = float(os.getenv("HEAD_V2_PROBE_XGB_REG_LAMBDA", "2.0").strip() or 2.0)
XGB_MIN_CHILD_WEIGHT = float(os.getenv("HEAD_V2_PROBE_XGB_MIN_CHILD_WEIGHT", "8.0").strip() or 8.0)
XGB_OBJECTIVE = os.getenv("HEAD_V2_PROBE_XGB_OBJECTIVE", "rank:pairwise").strip() or "rank:pairwise"
XGB_N_JOBS = int(os.getenv("HEAD_V2_PROBE_XGB_N_JOBS", "48").strip() or 48)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} suffix={suffix}")
    return runs[0]


def resolve_input_run() -> Path:
    if INPUT_HEAD_V2_RUN_DIR:
        p = Path(INPUT_HEAD_V2_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_HEAD_V2_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_HEAD_V2_ROOT, INPUT_HEAD_V2_SUFFIX)


def _inv_rank(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.zeros((len(vals),), dtype=np.float64), index=series.index)
    mask = vals.notna() & (vals > 0)
    out.loc[mask] = 1.0 / np.log2(vals.loc[mask].astype(np.float64) + 1.0)
    return out


def load_probe_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = run_dir / "head_v2_training_table.parquet"
    truth_path = run_dir / "head_v2_positive_audit.parquet"
    if not train_path.exists() or not truth_path.exists():
        raise FileNotFoundError(f"missing head-v2 parquet under {run_dir}")
    cols = [
        "user_idx",
        "item_idx",
        "label",
        "user_segment",
        "baseline_pre_rank",
        "pre_score",
        "head_score",
        "signal_score",
        "quality_score",
        "semantic_score",
        "semantic_confidence",
        "semantic_support",
        "semantic_tag_richness",
        "semantic_effective_score",
        "als_backbone_score",
        "head_multisource_boost",
        "head_profile_boost",
        "head_cluster_boost",
        "head_popular_penalty",
        "tower_score",
        "seq_score",
        "tower_inv",
        "seq_inv",
        "user_train_count",
        "item_train_pop_count",
        "source_count",
        "nonpopular_source_count",
        "profile_cluster_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
        "als_rank",
        "cluster_rank",
        "profile_rank",
        "popular_rank",
        "als_rank_pct",
        "cluster_rank_pct",
        "profile_rank_pct",
        "popular_rank_pct",
        "popular_log1p",
        "head_shortfall_to_top150",
        "is_als_only",
        "is_als_popular_only",
        "als_source_score",
        "als_source_confidence",
        "als_source_norm",
        "als_signal_score",
        "cluster_source_score",
        "cluster_source_confidence",
        "cluster_source_norm",
        "cluster_signal_score",
        "profile_source_score",
        "profile_source_confidence",
        "profile_source_norm",
        "profile_signal_score",
        "popular_source_score",
        "popular_source_confidence",
        "popular_source_norm",
        "popular_signal_score",
        "als_evd_score",
        "als_evd_confidence",
        "als_evd_norm",
        "als_evd_signal",
        "cluster_evd_score",
        "cluster_evd_confidence",
        "cluster_evd_norm",
        "cluster_evd_signal",
        "profile_evd_score",
        "profile_evd_confidence",
        "profile_evd_norm",
        "profile_evd_signal",
        "popular_evd_score",
        "popular_evd_confidence",
        "popular_evd_norm",
        "popular_evd_signal",
        "profile_route_rows",
        "profile_active_detail_count",
        "profile_vector_rank",
        "profile_vector_score",
        "profile_vector_confidence",
        "profile_vector_norm",
        "profile_vector_signal",
        "profile_shared_rank",
        "profile_shared_score",
        "profile_shared_confidence",
        "profile_shared_norm",
        "profile_shared_signal",
        "profile_bridge_user_rank",
        "profile_bridge_user_score",
        "profile_bridge_user_confidence",
        "profile_bridge_user_norm",
        "profile_bridge_user_signal",
        "profile_bridge_type_rank",
        "profile_bridge_type_score",
        "profile_bridge_type_confidence",
        "profile_bridge_type_norm",
        "profile_bridge_type_signal",
    ]
    train_pdf = pd.read_parquet(train_path, columns=cols)
    truth_pdf = pd.read_parquet(truth_path)
    truth_pdf["baseline_pre_rank"] = pd.to_numeric(truth_pdf.get("baseline_pre_rank"), errors="coerce")
    truth_pdf["is_in_top150_baseline"] = pd.to_numeric(truth_pdf.get("is_in_top150_baseline"), errors="coerce").fillna(0).astype(np.int32)
    truth_pdf["is_in_top250_baseline"] = pd.to_numeric(truth_pdf.get("is_in_top250_baseline"), errors="coerce").fillna(0).astype(np.int32)
    truth_pdf["truth_in_table"] = truth_pdf["baseline_pre_rank"].notna().astype(np.int32)
    meta = {
        "train_rows": int(train_pdf.shape[0]),
        "truth_users_total": int(truth_pdf["user_idx"].nunique()),
        "truth_in_table_users": int(truth_pdf["truth_in_table"].sum()),
        "truth_in_top150_users": int(truth_pdf["is_in_top150_baseline"].sum()),
        "truth_in_top250_users": int(truth_pdf["is_in_top250_baseline"].sum()),
    }
    return train_pdf, truth_pdf, meta


def prepare_features(train_pdf: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = train_pdf.copy()
    numeric_cols = [c for c in out.columns if c not in {"user_segment"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["label"] = out["label"].fillna(0).astype(np.int32)
    out["user_idx"] = out["user_idx"].fillna(-1).astype(np.int32)
    out["item_idx"] = out["item_idx"].fillna(-1).astype(np.int32)
    out["user_segment"] = out["user_segment"].fillna("unknown").astype(str)

    rank_cols = [
        "baseline_pre_rank",
        "als_rank",
        "cluster_rank",
        "profile_rank",
        "popular_rank",
        "profile_vector_rank",
        "profile_shared_rank",
        "profile_bridge_user_rank",
        "profile_bridge_type_rank",
    ]
    rank_fill = float(max(1200.0, pd.to_numeric(out["baseline_pre_rank"], errors="coerce").max(skipna=True) or 0.0) + 50.0)
    for col in rank_cols:
        raw = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_missing"] = raw.isna().astype(np.float64)
        out[f"{col}_filled"] = raw.fillna(rank_fill).astype(np.float64)
        out[f"inv_{col}"] = _inv_rank(raw).astype(np.float64)

    fill_zero_cols = [c for c in out.columns if c not in {"user_segment"}]
    for col in fill_zero_cols:
        if col not in {"user_idx", "item_idx", "label"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float64)

    out["is_light"] = (out["user_segment"] == "light").astype(np.float64)
    out["is_mid"] = (out["user_segment"] == "mid").astype(np.float64)
    out["is_heavy"] = (out["user_segment"] == "heavy").astype(np.float64)
    out["user_train_log"] = np.log1p(np.maximum(out["user_train_count"], 0.0))
    out["item_pop_log"] = np.log1p(np.maximum(out["item_train_pop_count"], 0.0))
    out["semantic_support_log"] = np.log1p(np.maximum(out["semantic_support"], 0.0))
    out["has_als_profile"] = ((out["has_als"] > 0.5) & (out["has_profile"] > 0.5)).astype(np.float64)
    out["has_als_cluster"] = ((out["has_als"] > 0.5) & (out["has_cluster"] > 0.5)).astype(np.float64)
    out["has_cluster_profile"] = ((out["has_cluster"] > 0.5) & (out["has_profile"] > 0.5)).astype(np.float64)
    out["has_profile_or_cluster"] = ((out["has_profile"] > 0.5) | (out["has_cluster"] > 0.5)).astype(np.float64)
    out["head_minus_pre"] = out["head_score"] - out["pre_score"]
    out["pre_minus_signal"] = out["pre_score"] - out["signal_score"]
    out["pre_minus_semantic"] = out["pre_score"] - out["semantic_effective_score"]
    out["profile_route_density"] = out["profile_route_rows"] / np.maximum(out["source_count"], 1.0)
    out["profile_detail_count"] = np.maximum(out["profile_active_detail_count"], 0.0)

    detail_score_cols = [
        "profile_vector_score",
        "profile_shared_score",
        "profile_bridge_user_score",
        "profile_bridge_type_score",
    ]
    detail_conf_cols = [
        "profile_vector_confidence",
        "profile_shared_confidence",
        "profile_bridge_user_confidence",
        "profile_bridge_type_confidence",
    ]
    out["profile_detail_best_score"] = np.maximum.reduce([out[c].to_numpy(dtype=np.float64) for c in detail_score_cols])
    out["profile_detail_best_conf"] = np.maximum.reduce([out[c].to_numpy(dtype=np.float64) for c in detail_conf_cols])
    out["bridge_best_score"] = np.maximum(out["profile_bridge_user_score"], out["profile_bridge_type_score"])
    out["bridge_best_conf"] = np.maximum(out["profile_bridge_user_confidence"], out["profile_bridge_type_confidence"])
    out["shared_vs_vector_gap"] = out["profile_shared_score"] - out["profile_vector_score"]
    out["bridge_vs_vector_gap"] = out["bridge_best_score"] - out["profile_vector_score"]
    out["detail_score_sum"] = (
        out["profile_vector_score"] + out["profile_shared_score"] + out["profile_bridge_user_score"] + out["profile_bridge_type_score"]
    )
    out["detail_conf_sum"] = (
        out["profile_vector_confidence"]
        + out["profile_shared_confidence"]
        + out["profile_bridge_user_confidence"]
        + out["profile_bridge_type_confidence"]
    )
    out["profile_best_rank_inv"] = np.maximum.reduce(
        [
            out["inv_profile_vector_rank"].to_numpy(dtype=np.float64),
            out["inv_profile_shared_rank"].to_numpy(dtype=np.float64),
            out["inv_profile_bridge_user_rank"].to_numpy(dtype=np.float64),
            out["inv_profile_bridge_type_rank"].to_numpy(dtype=np.float64),
        ]
    )
    out["best_route_score"] = np.maximum.reduce(
        [
            out["als_source_score"].to_numpy(dtype=np.float64),
            out["cluster_source_score"].to_numpy(dtype=np.float64),
            out["profile_source_score"].to_numpy(dtype=np.float64),
            out["popular_source_score"].to_numpy(dtype=np.float64),
        ]
    )
    out["best_nonpopular_score"] = np.maximum.reduce(
        [
            out["als_source_score"].to_numpy(dtype=np.float64),
            out["cluster_source_score"].to_numpy(dtype=np.float64),
            out["profile_source_score"].to_numpy(dtype=np.float64),
        ]
    )
    out["profile_minus_als_score"] = out["profile_source_score"] - out["als_source_score"]
    out["cluster_minus_als_score"] = out["cluster_source_score"] - out["als_source_score"]
    out["profile_detail_x_heavy"] = out["profile_detail_best_score"] * out["is_heavy"]
    out["cluster_x_heavy"] = out["cluster_source_score"] * out["is_heavy"]
    out["head_x_heavy"] = out["head_score"] * out["is_heavy"]

    # User-local features matter more than cross-user absolute scale for head ranking.
    user_group = out.groupby("user_idx", sort=False)
    local_score_cols = [
        "pre_score",
        "head_score",
        "signal_score",
        "semantic_effective_score",
        "als_source_score",
        "cluster_source_score",
        "profile_source_score",
        "profile_detail_best_score",
    ]
    for col in local_score_cols:
        pct = user_group[col].rank(method="average", ascending=False, pct=True)
        maxv = user_group[col].transform("max")
        out[f"user_pct_{col}"] = pct.astype(np.float64)
        out[f"user_gap_{col}"] = (maxv - out[col]).astype(np.float64)

    score_missing_cols = [
        "als_source_score",
        "cluster_source_score",
        "profile_source_score",
        "popular_source_score",
        "profile_vector_score",
        "profile_shared_score",
        "profile_bridge_user_score",
        "profile_bridge_type_score",
    ]
    for col in score_missing_cols:
        raw = pd.to_numeric(train_pdf[col], errors="coerce")
        out[f"{col}_missing"] = raw.isna().astype(np.float64)
    out = out.copy()

    feature_cols = [
        "pre_score",
        "head_score",
        "head_minus_pre",
        "signal_score",
        "quality_score",
        "semantic_score",
        "semantic_confidence",
        "semantic_support_log",
        "semantic_tag_richness",
        "semantic_effective_score",
        "als_backbone_score",
        "head_multisource_boost",
        "head_profile_boost",
        "head_cluster_boost",
        "head_popular_penalty",
        "tower_score",
        "seq_score",
        "tower_inv",
        "seq_inv",
        "user_train_log",
        "item_pop_log",
        "source_count",
        "nonpopular_source_count",
        "profile_cluster_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
        "has_profile_or_cluster",
        "is_als_only",
        "is_als_popular_only",
        "is_light",
        "is_mid",
        "is_heavy",
        "has_als_profile",
        "has_als_cluster",
        "has_cluster_profile",
        "baseline_pre_rank_filled",
        "als_rank_filled",
        "cluster_rank_filled",
        "profile_rank_filled",
        "popular_rank_filled",
        "profile_vector_rank_filled",
        "profile_shared_rank_filled",
        "profile_bridge_user_rank_filled",
        "profile_bridge_type_rank_filled",
        "inv_baseline_pre_rank",
        "inv_als_rank",
        "inv_cluster_rank",
        "inv_profile_rank",
        "inv_popular_rank",
        "profile_best_rank_inv",
        "als_rank_pct",
        "cluster_rank_pct",
        "profile_rank_pct",
        "popular_rank_pct",
        "popular_log1p",
        "head_shortfall_to_top150",
        "pre_minus_signal",
        "pre_minus_semantic",
        "profile_route_rows",
        "profile_route_density",
        "profile_detail_count",
        "profile_detail_best_score",
        "profile_detail_best_conf",
        "bridge_best_score",
        "bridge_best_conf",
        "shared_vs_vector_gap",
        "bridge_vs_vector_gap",
        "detail_score_sum",
        "detail_conf_sum",
        "best_route_score",
        "best_nonpopular_score",
        "profile_minus_als_score",
        "cluster_minus_als_score",
        "profile_detail_x_heavy",
        "cluster_x_heavy",
        "head_x_heavy",
        "user_pct_pre_score",
        "user_gap_pre_score",
        "user_pct_head_score",
        "user_gap_head_score",
        "user_pct_signal_score",
        "user_gap_signal_score",
        "user_pct_semantic_effective_score",
        "user_gap_semantic_effective_score",
        "user_pct_als_source_score",
        "user_gap_als_source_score",
        "user_pct_cluster_source_score",
        "user_gap_cluster_source_score",
        "user_pct_profile_source_score",
        "user_gap_profile_source_score",
        "user_pct_profile_detail_best_score",
        "user_gap_profile_detail_best_score",
        "als_source_score",
        "als_source_confidence",
        "als_source_norm",
        "als_signal_score",
        "cluster_source_score",
        "cluster_source_confidence",
        "cluster_source_norm",
        "cluster_signal_score",
        "profile_source_score",
        "profile_source_confidence",
        "profile_source_norm",
        "profile_signal_score",
        "popular_source_score",
        "popular_source_confidence",
        "popular_source_norm",
        "popular_signal_score",
        "profile_vector_score",
        "profile_vector_confidence",
        "profile_vector_norm",
        "profile_vector_signal",
        "profile_shared_score",
        "profile_shared_confidence",
        "profile_shared_norm",
        "profile_shared_signal",
        "profile_bridge_user_score",
        "profile_bridge_user_confidence",
        "profile_bridge_user_norm",
        "profile_bridge_user_signal",
        "profile_bridge_type_score",
        "profile_bridge_type_confidence",
        "profile_bridge_type_norm",
        "profile_bridge_type_signal",
        "baseline_pre_rank_missing",
        "als_rank_missing",
        "cluster_rank_missing",
        "profile_rank_missing",
        "popular_rank_missing",
        "profile_vector_rank_missing",
        "profile_shared_rank_missing",
        "profile_bridge_user_rank_missing",
        "profile_bridge_type_rank_missing",
        "als_source_score_missing",
        "cluster_source_score_missing",
        "profile_source_score_missing",
        "popular_source_score_missing",
        "profile_vector_score_missing",
        "profile_shared_score_missing",
        "profile_bridge_user_score_missing",
        "profile_bridge_type_score_missing",
    ]
    feature_meta = {
        "rows": int(out.shape[0]),
        "users": int(out["user_idx"].nunique()),
        "positives": int(out["label"].sum()),
        "feature_count": int(len(feature_cols)),
        "rank_fill_value": float(rank_fill),
        "topn_list": [int(x) for x in TOPN_LIST],
    }
    return out, feature_cols, feature_meta


def make_user_folds(user_ids: np.ndarray, n_folds: int, seed: int) -> dict[int, int]:
    arr = np.array(sorted({int(x) for x in user_ids.tolist()}), dtype=np.int32)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(arr)
    return {int(uid): int(i % max(2, int(n_folds))) for i, uid in enumerate(arr)}


def rank_by_score(pdf: pd.DataFrame, score_col: str, out_rank_col: str) -> pd.DataFrame:
    out = pdf.sort_values(
        ["user_idx", score_col, "baseline_pre_rank", "item_idx"],
        ascending=[True, False, True, True],
        kind="stable",
    ).copy()
    out[out_rank_col] = out.groupby("user_idx", sort=False).cumcount() + 1
    return out


def summarize_truth_df(truth_df: pd.DataFrame, rank_col: str, topn_list: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rank = pd.to_numeric(truth_df[rank_col], errors="coerce")
    summary: dict[str, Any] = {
        "users": int(truth_df["user_idx"].nunique()),
        "truth_in_table_users": int(rank.notna().sum()),
        "truth_in_table_rate": round(float(rank.notna().mean()), 6),
    }
    if rank.notna().any():
        summary["rank_median"] = float(rank.dropna().median())
        summary["rank_p75"] = float(rank.dropna().quantile(0.75))
    for topn in topn_list:
        mask = rank.notna() & (rank <= int(topn))
        summary[f"truth_in_top{int(topn)}_users"] = int(mask.sum())
        summary[f"truth_in_top{int(topn)}_rate"] = round(float(mask.mean()), 6)
    seg_rows: list[dict[str, Any]] = []
    for seg, g in truth_df.groupby("user_segment", sort=True):
        g_rank = pd.to_numeric(g[rank_col], errors="coerce")
        row: dict[str, Any] = {
            "user_segment": str(seg),
            "users": int(g["user_idx"].nunique()),
            "truth_in_table_users": int(g_rank.notna().sum()),
            "truth_in_table_rate": round(float(g_rank.notna().mean()), 6),
        }
        if g_rank.notna().any():
            row["rank_median"] = float(g_rank.dropna().median())
        for topn in topn_list:
            row[f"truth_in_top{int(topn)}_users"] = int((g_rank.notna() & (g_rank <= int(topn))).sum())
            row[f"truth_in_top{int(topn)}_rate"] = round(float((g_rank.notna() & (g_rank <= int(topn))).mean()), 6)
        seg_rows.append(row)
    return summary, seg_rows


def fit_backend(
    backend: str,
    train_pdf: pd.DataFrame,
    test_pdf: pd.DataFrame,
    feature_cols: list[str],
    fold: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if backend != "xgboost_ranker":
        raise RuntimeError(f"unsupported backend: {backend}")
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    train_sorted = train_pdf.sort_values(["user_idx", "baseline_pre_rank", "item_idx"], kind="stable")
    test_sorted = test_pdf.sort_values(["user_idx", "baseline_pre_rank", "item_idx"], kind="stable")
    x_train = train_sorted[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_sorted["label"].to_numpy(dtype=np.float32, copy=False)
    x_test = test_sorted[feature_cols].to_numpy(dtype=np.float32, copy=False)
    group_train = train_sorted.groupby("user_idx", sort=False).size().to_numpy(dtype=np.int32)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_cols)
    dtrain.set_group(group_train.tolist())
    dtest = xgb.DMatrix(x_test, feature_names=feature_cols)
    booster = xgb.train(
        params={
            "objective": XGB_OBJECTIVE,
            "max_depth": int(XGB_MAX_DEPTH),
            "eta": float(XGB_LEARNING_RATE),
            "subsample": float(XGB_SUBSAMPLE),
            "colsample_bytree": float(XGB_COLSAMPLE),
            "lambda": float(XGB_REG_LAMBDA),
            "min_child_weight": float(XGB_MIN_CHILD_WEIGHT),
            "tree_method": "hist",
            "seed": int(RANDOM_SEED + fold),
            "nthread": int(XGB_N_JOBS),
        },
        dtrain=dtrain,
        num_boost_round=int(XGB_N_ESTIMATORS),
        verbose_eval=False,
    )
    pred = booster.predict(dtest).astype(np.float64)
    importances = {str(k): float(v) for k, v in booster.get_score(importance_type="gain").items() if float(v) > 0.0}
    return pred, importances


def run_backend_cv(
    backend: str,
    probe_pdf: pd.DataFrame,
    truth_audit_all: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_map = make_user_folds(probe_pdf["user_idx"].to_numpy(dtype=np.int32, copy=False), int(N_FOLDS), int(RANDOM_SEED))
    probe_pdf = probe_pdf.copy()
    probe_pdf["fold_id"] = probe_pdf["user_idx"].map(fold_map).astype(np.int32)
    all_pos_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold in sorted(probe_pdf["fold_id"].unique().tolist()):
        train_pdf = probe_pdf[probe_pdf["fold_id"] != int(fold)].copy()
        test_pdf = probe_pdf[probe_pdf["fold_id"] == int(fold)].copy()
        pred, importances = fit_backend(backend=backend, train_pdf=train_pdf, test_pdf=test_pdf, feature_cols=feature_cols, fold=int(fold))
        ranked = rank_by_score(test_pdf.assign(probe_score=pred), score_col="probe_score", out_rank_col="probe_rank")
        pos_rows = ranked[ranked["label"] == 1][["user_idx", "probe_rank"]].copy()
        pos_rows["fold_id"] = int(fold)
        pos_rows["backend"] = str(backend)
        all_pos_rows.append(pos_rows)
        fold_truth = truth_audit_all[truth_audit_all["user_idx"].isin(test_pdf["user_idx"].unique())].copy()
        fold_truth = fold_truth.merge(pos_rows[["user_idx", "probe_rank"]], on="user_idx", how="left")
        probe_summary, _ = summarize_truth_df(fold_truth, rank_col="probe_rank", topn_list=TOPN_LIST)
        baseline_summary, _ = summarize_truth_df(fold_truth, rank_col="baseline_pre_rank", topn_list=TOPN_LIST)
        row = {
            "backend": str(backend),
            "fold_id": int(fold),
            "users": int(probe_summary["users"]),
            "truth_in_table_rate": float(probe_summary["truth_in_table_rate"]),
            "baseline_truth_in_table_rate": float(baseline_summary["truth_in_table_rate"]),
        }
        for topn in TOPN_LIST:
            row[f"truth_in_top{int(topn)}_rate"] = float(probe_summary[f"truth_in_top{int(topn)}_rate"])
            row[f"baseline_truth_in_top{int(topn)}_rate"] = float(baseline_summary[f"truth_in_top{int(topn)}_rate"])
        fold_rows.append(row)
        top_features = sorted(importances.items(), key=lambda z: (-z[1], z[0]))[:40]
        for rank_idx, (feature_name, score) in enumerate(top_features, start=1):
            importance_rows.append(
                {
                    "backend": str(backend),
                    "fold_id": int(fold),
                    "feature": str(feature_name),
                    "importance": float(score),
                    "importance_rank": int(rank_idx),
                }
            )
    pos_all = pd.concat(all_pos_rows, ignore_index=True)
    truth_probe = truth_audit_all.merge(pos_all[["user_idx", "probe_rank"]], on="user_idx", how="left")
    probe_overall, probe_seg = summarize_truth_df(truth_probe, rank_col="probe_rank", topn_list=TOPN_LIST)
    baseline_overall, baseline_seg = summarize_truth_df(truth_audit_all, rank_col="baseline_pre_rank", topn_list=TOPN_LIST)
    in_table_mask = truth_audit_all["truth_in_table"] > 0
    probe_in_table, probe_seg_in_table = summarize_truth_df(
        truth_probe[in_table_mask].copy(),
        rank_col="probe_rank",
        topn_list=TOPN_LIST,
    )
    baseline_in_table, baseline_seg_in_table = summarize_truth_df(
        truth_audit_all[in_table_mask].copy(),
        rank_col="baseline_pre_rank",
        topn_list=TOPN_LIST,
    )
    summary = {
        "backend": str(backend),
        "overall_probe": probe_overall,
        "overall_baseline": baseline_overall,
        "overall_delta": {},
        "in_table_probe": probe_in_table,
        "in_table_baseline": baseline_in_table,
        "in_table_delta": {},
        "segment_overall_probe": probe_seg,
        "segment_overall_baseline": baseline_seg,
        "segment_in_table_probe": probe_seg_in_table,
        "segment_in_table_baseline": baseline_seg_in_table,
    }
    for topn in TOPN_LIST:
        summary["overall_delta"][f"truth_in_top{int(topn)}_users"] = int(
            probe_overall[f"truth_in_top{int(topn)}_users"] - baseline_overall[f"truth_in_top{int(topn)}_users"]
        )
        summary["overall_delta"][f"truth_in_top{int(topn)}_rate"] = round(
            float(probe_overall[f"truth_in_top{int(topn)}_rate"] - baseline_overall[f"truth_in_top{int(topn)}_rate"]),
            6,
        )
        summary["in_table_delta"][f"truth_in_top{int(topn)}_users"] = int(
            probe_in_table[f"truth_in_top{int(topn)}_users"] - baseline_in_table[f"truth_in_top{int(topn)}_users"]
        )
        summary["in_table_delta"][f"truth_in_top{int(topn)}_rate"] = round(
            float(probe_in_table[f"truth_in_top{int(topn)}_rate"] - baseline_in_table[f"truth_in_top{int(topn)}_rate"]),
            6,
        )
    return summary, pd.DataFrame(fold_rows), pd.DataFrame(importance_rows), truth_probe


def main() -> None:
    run_dir = resolve_input_run()
    out_dir = OUTPUT_ROOT / datetime.now().strftime(f"%Y%m%d_%H%M%S_{RUN_TAG}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pdf, truth_audit_all, load_meta = load_probe_inputs(run_dir)
    if PROBE_SCOPE == "pretrim_only":
        train_pdf = train_pdf[pd.to_numeric(train_pdf["baseline_pre_rank"], errors="coerce").notna()].copy()
        truth_audit_all = truth_audit_all.copy()
    elif PROBE_SCOPE != "all_candidates":
        raise RuntimeError(f"unsupported HEAD_V2_PROBE_SCOPE: {PROBE_SCOPE}")
    train_pdf, feature_cols, feature_meta = prepare_features(train_pdf)
    probe_pdf = train_pdf.groupby("user_idx")["label"].transform("sum") > 0
    probe_pdf = train_pdf[probe_pdf].copy()
    print(
        f"[PROBE] scope={PROBE_SCOPE} rows={len(probe_pdf)} users={probe_pdf['user_idx'].nunique()} "
        f"positives={int(probe_pdf['label'].sum())} all_users={truth_audit_all['user_idx'].nunique()}"
    )
    backend_summaries: dict[str, Any] = {}
    fold_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    truth_probe_frames: list[pd.DataFrame] = []
    for backend in MODEL_BACKENDS:
        if backend == "xgboost_ranker" and xgb is None:
            print("[WARN] skip backend=xgboost_ranker because xgboost is unavailable")
            continue
        print(f"[MODEL] backend={backend}")
        summary, fold_df, importance_df, truth_probe = run_backend_cv(
            backend=backend,
            probe_pdf=probe_pdf,
            truth_audit_all=truth_audit_all,
            feature_cols=feature_cols,
        )
        backend_summaries[str(backend)] = summary
        fold_frames.append(fold_df)
        truth_probe["backend"] = str(backend)
        truth_probe_frames.append(truth_probe)
        if not importance_df.empty:
            importance_frames.append(importance_df)
    if not backend_summaries:
        raise RuntimeError("no backend completed")
    payload = {
        "run_tag": RUN_TAG,
        "source_head_v2_run": run_dir.as_posix(),
        "load_meta": load_meta,
        "feature_meta": feature_meta,
        "model_backends": MODEL_BACKENDS,
        "topn_list": TOPN_LIST,
        "n_folds": int(N_FOLDS),
        "probe_scope": str(PROBE_SCOPE),
        "backend_summaries": backend_summaries,
    }
    write_json(out_dir / "probe_summary.json", payload)
    pd.DataFrame(
        [
            {
                "backend": k,
                **v["overall_probe"],
                **{f"overall_delta_{kk}": vv for kk, vv in v["overall_delta"].items()},
                **{f"in_table_delta_{kk}": vv for kk, vv in v["in_table_delta"].items()},
            }
            for k, v in backend_summaries.items()
        ]
    ).to_csv(out_dir / "probe_backend_summary.csv", index=False, encoding="utf-8")
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(out_dir / "probe_fold_metrics.csv", index=False, encoding="utf-8")
    if truth_probe_frames:
        pd.concat(truth_probe_frames, ignore_index=True).to_csv(out_dir / "probe_truth_ranks.csv", index=False, encoding="utf-8")
    if importance_frames:
        imp = pd.concat(importance_frames, ignore_index=True)
        imp.to_csv(out_dir / "probe_feature_importance.csv", index=False, encoding="utf-8")
        imp.groupby(["backend", "feature"], as_index=False)["importance"].mean().sort_values(
            ["backend", "importance", "feature"], ascending=[True, False, True], kind="stable"
        ).to_csv(out_dir / "probe_feature_importance_mean.csv", index=False, encoding="utf-8")
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
