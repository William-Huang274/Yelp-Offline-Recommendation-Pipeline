from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pipeline.project_paths import env_or_project_path


RUN_TAG = "stage09_bucket10_boundary_repair_simulator"

INPUT_HEAD_V2_RUN_DIR = os.getenv("INPUT_HEAD_V2_RUN_DIR", "").strip()
INPUT_HEAD_V2_ROOT = env_or_project_path("INPUT_09_HEAD_V2_ROOT_DIR", "data/output/09_head_v2_table")
INPUT_HEAD_V2_SUFFIX = "_stage09_bucket10_head_v2_table"
OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_09_BOUNDARY_REPAIR_ROOT_DIR", "data/output/09_boundary_repair_simulator"
)

COHORT_PATH = os.getenv("BOUNDARY_SIM_COHORT_PATH", "").strip()
TOP150 = int(os.getenv("BOUNDARY_SIM_TOP150", "150").strip() or 150)
TOP250 = int(os.getenv("BOUNDARY_SIM_TOP250", "250").strip() or 250)
TOP80 = int(os.getenv("BOUNDARY_SIM_TOP80", "80").strip() or 80)
MAX_TRAIN_ROWS = int(os.getenv("BOUNDARY_SIM_MAX_TRAIN_ROWS", "3200000").strip() or 3200000)

PROTECT_TOPK_LIST = [
    int(x.strip()) for x in os.getenv("BOUNDARY_SIM_PROTECT_TOPK", "100,110,120,130").split(",") if x.strip()
]
CANDIDATE_CAP_LIST = [
    int(x.strip()) for x in os.getenv("BOUNDARY_SIM_CANDIDATE_CAP", "180,200,220").split(",") if x.strip()
]
HEAVY_REPLACE_LIST = [
    int(x.strip()) for x in os.getenv("BOUNDARY_SIM_HEAVY_REPLACE", "6,10,14").split(",") if x.strip()
]
MID_REPLACE_LIST = [
    int(x.strip()) for x in os.getenv("BOUNDARY_SIM_MID_REPLACE", "4,8,12").split(",") if x.strip()
]
CHALLENGER_MIN_DETAIL_LIST = [
    int(x.strip()) for x in os.getenv("BOUNDARY_SIM_MIN_DETAIL", "1,2").split(",") if x.strip()
]
MIN_MARGIN_LIST = [
    float(x.strip()) for x in os.getenv("BOUNDARY_SIM_MIN_MARGIN", "0.00,0.03,0.06").split(",") if x.strip()
]
NUM_WORKERS = int(os.getenv("BOUNDARY_SIM_NUM_WORKERS", "1").strip() or 1)
POOL_CHUNK_SIZE = int(os.getenv("BOUNDARY_SIM_POOL_CHUNK_SIZE", "8").strip() or 8)

_WORKER_USER_CACHES: list[dict[str, Any]] | None = None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} suffix={suffix}")
    return runs[0]


def resolve_head_v2_run() -> Path:
    if INPUT_HEAD_V2_RUN_DIR:
        p = Path(INPUT_HEAD_V2_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_HEAD_V2_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_HEAD_V2_ROOT, INPUT_HEAD_V2_SUFFIX)


def resolve_cohort_filter() -> tuple[str, set[Any]] | None:
    path = str(COHORT_PATH or "").strip()
    if not path:
        return None
    cohort = pd.read_csv(path)
    if "user_idx" in cohort.columns:
        vals = pd.to_numeric(cohort["user_idx"], errors="coerce").dropna().astype(np.int32).tolist()
        return "user_idx", set(int(x) for x in vals)
    if "user_id" in cohort.columns:
        vals = cohort["user_id"].dropna().astype(str).tolist()
        return "user_id", set(vals)
    raise RuntimeError(f"cohort file missing user_idx/user_id column: {path}")


def ensure_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)


def rank_inv(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(np.float64)
    arr_np = arr.to_numpy(copy=False)
    return pd.Series(
        np.where(np.isfinite(arr_np) & (arr_np > 0.0), 1.0 / (1.0 + arr_np), 0.0),
        index=values.index,
        dtype=np.float64,
    )


def stable_desc_order(score: np.ndarray, baseline_rank: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
    return np.lexsort((item_idx.astype(np.int64), baseline_rank.astype(np.float64), -score.astype(np.float64)))


def stable_asc_order(score: np.ndarray, baseline_rank: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
    return np.lexsort((item_idx.astype(np.int64), -baseline_rank.astype(np.float64), score.astype(np.float64)))


def load_probe_inputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_path = run_dir / "head_v2_training_table.parquet"
    truth_path = run_dir / "head_v2_positive_audit.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"head_v2_training_table.parquet not found in {run_dir}")
    if not truth_path.exists():
        raise FileNotFoundError(f"head_v2_positive_audit.parquet not found in {run_dir}")

    truth_cols = [
        "user_idx",
        "user_id",
        "user_segment",
        "baseline_pre_rank",
        "baseline_rank_bucket",
        "is_in_pretrim",
        "is_in_top150_baseline",
        "is_in_top250_baseline",
        "head_shortfall_to_top150",
    ]
    truth_pdf = pd.read_parquet(truth_path, columns=truth_cols)
    cohort_filter = resolve_cohort_filter()
    if cohort_filter is not None:
        key, vals = cohort_filter
        truth_pdf = truth_pdf[truth_pdf[key].isin(vals)].copy()
    truth_pdf = truth_pdf.drop_duplicates("user_idx").reset_index(drop=True)

    train_cols = [
        "user_idx",
        "item_idx",
        "user_segment",
        "label",
        "baseline_pre_rank",
        "is_in_pretrim",
        "nonpopular_source_count",
        "profile_cluster_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
        "has_profile_or_cluster",
        "is_als_only",
        "is_als_popular_only",
        "popular_log1p",
        "head_shortfall_to_top150",
        "als_evd_rank",
        "als_evd_norm",
        "als_evd_signal",
        "cluster_evd_rank",
        "cluster_evd_norm",
        "cluster_evd_signal",
        "profile_evd_rank",
        "profile_evd_norm",
        "profile_evd_signal",
        "profile_vector_rank",
        "profile_vector_norm",
        "profile_vector_signal",
        "profile_shared_rank",
        "profile_shared_norm",
        "profile_shared_signal",
        "profile_bridge_user_rank",
        "profile_bridge_user_norm",
        "profile_bridge_user_signal",
        "profile_bridge_type_rank",
        "profile_bridge_type_norm",
        "profile_bridge_type_signal",
        "profile_active_detail_count",
    ]
    train_pdf = pd.read_parquet(train_path, columns=train_cols, filters=[("is_in_pretrim", "=", 1)])
    train_pdf = train_pdf[train_pdf["user_idx"].isin(truth_pdf["user_idx"])].copy()
    if len(train_pdf) > int(MAX_TRAIN_ROWS):
        raise RuntimeError(
            f"train rows exceed cap: rows={len(train_pdf)} max={MAX_TRAIN_ROWS}; narrow cohort or raise BOUNDARY_SIM_MAX_TRAIN_ROWS"
        )
    meta = {
        "run_dir": run_dir.as_posix(),
        "train_rows": int(train_pdf.shape[0]),
        "truth_users": int(truth_pdf.shape[0]),
        "cohort_path": str(COHORT_PATH or ""),
        "max_train_rows_cap": int(MAX_TRAIN_ROWS),
    }
    return train_pdf, truth_pdf, meta


def prepare_features(train_pdf: pd.DataFrame, truth_pdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = train_pdf.copy()
    truth = truth_pdf.copy()

    for col in (
        "label",
        "baseline_pre_rank",
        "nonpopular_source_count",
        "profile_cluster_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
        "has_profile_or_cluster",
        "is_als_only",
        "is_als_popular_only",
        "popular_log1p",
        "head_shortfall_to_top150",
        "als_evd_rank",
        "als_evd_norm",
        "als_evd_signal",
        "cluster_evd_rank",
        "cluster_evd_norm",
        "cluster_evd_signal",
        "profile_evd_rank",
        "profile_evd_norm",
        "profile_evd_signal",
        "profile_vector_rank",
        "profile_vector_norm",
        "profile_vector_signal",
        "profile_shared_rank",
        "profile_shared_norm",
        "profile_shared_signal",
        "profile_bridge_user_rank",
        "profile_bridge_user_norm",
        "profile_bridge_user_signal",
        "profile_bridge_type_rank",
        "profile_bridge_type_norm",
        "profile_bridge_type_signal",
        "profile_active_detail_count",
    ):
        ensure_numeric(out, col)

    truth["baseline_pre_rank"] = pd.to_numeric(truth["baseline_pre_rank"], errors="coerce")
    truth["head_shortfall_to_top150"] = pd.to_numeric(truth["head_shortfall_to_top150"], errors="coerce")
    for col in ("is_in_pretrim", "is_in_top150_baseline", "is_in_top250_baseline"):
        ensure_numeric(truth, col)

    out["user_segment"] = out["user_segment"].fillna("unknown").astype(str)
    truth["user_segment"] = truth["user_segment"].fillna("unknown").astype(str)
    truth["baseline_rank_bucket"] = truth["baseline_rank_bucket"].fillna("absent").astype(str)

    out["baseline_rank_inv"] = rank_inv(out["baseline_pre_rank"])
    out["als_rank_inv"] = rank_inv(out["als_evd_rank"])
    out["cluster_rank_inv"] = rank_inv(out["cluster_evd_rank"])
    out["profile_rank_inv"] = rank_inv(out["profile_evd_rank"])
    out["profile_vector_rank_inv"] = rank_inv(out["profile_vector_rank"])
    out["profile_shared_rank_inv"] = rank_inv(out["profile_shared_rank"])
    out["profile_bridge_user_rank_inv"] = rank_inv(out["profile_bridge_user_rank"])
    out["profile_bridge_type_rank_inv"] = rank_inv(out["profile_bridge_type_rank"])

    detail_cols: list[str] = []
    for prefix in ("profile_vector", "profile_shared", "profile_bridge_user", "profile_bridge_type"):
        score_col = f"{prefix}_support_score"
        out[score_col] = (
            0.55 * out.get(f"{prefix}_norm", 0.0).astype(np.float64)
            + 0.25 * out.get(f"{prefix}_signal", 0.0).astype(np.float64)
            + 0.20 * out.get(f"{prefix}_rank_inv", 0.0).astype(np.float64)
        )
        detail_cols.append(score_col)

    out["detail_best_support_score"] = out[detail_cols].max(axis=1, skipna=True).fillna(0.0).astype(np.float64)
    out["detail_mean_support_score"] = out[detail_cols].mean(axis=1, skipna=True).fillna(0.0).astype(np.float64)
    out["als_support_score"] = (
        0.60 * out["als_evd_norm"].astype(np.float64)
        + 0.25 * out["als_evd_signal"].astype(np.float64)
        + 0.15 * out["als_rank_inv"].astype(np.float64)
    )
    out["cluster_support_score"] = (
        0.60 * out["cluster_evd_norm"].astype(np.float64)
        + 0.25 * out["cluster_evd_signal"].astype(np.float64)
        + 0.15 * out["cluster_rank_inv"].astype(np.float64)
    )
    out["profile_route_support_score"] = (
        0.50 * out["profile_evd_norm"].astype(np.float64)
        + 0.20 * out["profile_evd_signal"].astype(np.float64)
        + 0.10 * out["profile_rank_inv"].astype(np.float64)
        + 0.20 * out["detail_best_support_score"].astype(np.float64)
    )
    out["detail_active_ge1"] = (out["profile_active_detail_count"] >= 1.0).astype(np.float32)
    out["detail_active_ge2"] = (out["profile_active_detail_count"] >= 2.0).astype(np.float32)
    out["multi_nonpopular"] = (out["nonpopular_source_count"] >= 2.0).astype(np.float32)
    out["consensus_count"] = (
        (out["has_profile"] > 0.5).astype(np.int8)
        + (out["has_cluster"] > 0.5).astype(np.int8)
        + out["detail_active_ge1"].astype(np.int8)
        + out["detail_active_ge2"].astype(np.int8)
        + out["multi_nonpopular"].astype(np.int8)
    ).astype(np.int8)
    out["personal_eligible"] = (
        (out["has_profile_or_cluster"] > 0.5)
        | (out["detail_active_ge1"] > 0.5)
        | (out["multi_nonpopular"] > 0.5)
    ).astype(np.int8)
    out["popular_penalty_score"] = (
        0.15 * out["popular_log1p"].clip(lower=0.0, upper=10.0).astype(np.float64)
        + 0.15 * out["has_popular"].astype(np.float64)
        + 0.20 * out["is_als_popular_only"].astype(np.float64)
        + 0.10 * out["is_als_only"].astype(np.float64)
    )
    out["head_shortfall_norm"] = (
        np.minimum(out["head_shortfall_to_top150"].clip(lower=0.0).astype(np.float64), 400.0) / 400.0
    )
    out["retain_score"] = (
        0.55 * out["baseline_rank_inv"]
        + 0.25 * out["als_support_score"]
        + 0.10 * out["cluster_support_score"]
        + 0.08 * out["detail_best_support_score"]
        - 0.12 * out["popular_penalty_score"]
    ).astype(np.float64)
    out["challenge_score"] = (
        0.72 * out["detail_best_support_score"]
        + 0.38 * out["detail_mean_support_score"]
        + 0.48 * out["cluster_support_score"]
        + 0.30 * out["profile_route_support_score"]
        + 0.12 * out["consensus_count"].astype(np.float64)
        + 0.15 * out["head_shortfall_norm"]
        + 0.05 * out["als_support_score"]
        - 0.10 * out["popular_penalty_score"]
    ).astype(np.float64)
    return out, truth


@dataclass(frozen=True)
class BoundaryConfig:
    protect_topk: int
    candidate_cap: int
    heavy_replace: int
    mid_replace: int
    challenger_min_detail: int
    min_margin: float

    def replace_quota_for(self, user_segment: str) -> int:
        if user_segment == "heavy":
            return int(self.heavy_replace)
        return int(self.mid_replace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protect_topk": int(self.protect_topk),
            "candidate_cap": int(self.candidate_cap),
            "heavy_replace": int(self.heavy_replace),
            "mid_replace": int(self.mid_replace),
            "challenger_min_detail": int(self.challenger_min_detail),
            "min_margin": float(self.min_margin),
        }


def build_grid() -> list[BoundaryConfig]:
    grid: list[BoundaryConfig] = []
    for protect_topk, candidate_cap, heavy_replace, mid_replace, min_detail, min_margin in itertools.product(
        PROTECT_TOPK_LIST,
        CANDIDATE_CAP_LIST,
        HEAVY_REPLACE_LIST,
        MID_REPLACE_LIST,
        CHALLENGER_MIN_DETAIL_LIST,
        MIN_MARGIN_LIST,
    ):
        if protect_topk >= TOP150:
            continue
        if candidate_cap <= TOP150:
            continue
        max_repl = TOP150 - protect_topk
        if heavy_replace > max_repl or mid_replace > max_repl:
            continue
        if candidate_cap < protect_topk:
            continue
        grid.append(
            BoundaryConfig(
                protect_topk=int(protect_topk),
                candidate_cap=int(candidate_cap),
                heavy_replace=int(heavy_replace),
                mid_replace=int(mid_replace),
                challenger_min_detail=int(min_detail),
                min_margin=float(min_margin),
            )
        )
    return grid


def build_user_truth_meta(truth_pdf: pd.DataFrame) -> dict[int, dict[str, Any]]:
    meta: dict[int, dict[str, Any]] = {}
    for row in truth_pdf.itertuples(index=False):
        meta[int(row.user_idx)] = {
            "user_idx": int(row.user_idx),
            "user_segment": str(row.user_segment),
            "baseline_truth_rank": int(row.baseline_pre_rank) if pd.notna(row.baseline_pre_rank) else 10**9,
            "baseline_rank_bucket": str(row.baseline_rank_bucket),
            "baseline_in_pretrim": int(row.is_in_pretrim),
            "baseline_in_top150": int(row.is_in_top150_baseline),
            "baseline_in_top250": int(row.is_in_top250_baseline),
        }
    return meta


def build_user_caches(train_pdf: pd.DataFrame, truth_pdf: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    truth_meta = build_user_truth_meta(truth_pdf)
    caches: list[dict[str, Any]] = []
    seen_users: set[int] = set()
    positive_users = 0

    for user_idx, group in train_pdf.groupby("user_idx", sort=False):
        user_id = int(user_idx)
        meta = truth_meta.get(user_id)
        seen_users.add(user_id)
        g = group.sort_values(["baseline_pre_rank", "item_idx"], ascending=[True, True], kind="stable").reset_index(drop=True)
        label = pd.to_numeric(g["label"], errors="coerce").fillna(0).to_numpy(dtype=np.int8, copy=False)
        truth_idx_arr = np.flatnonzero(label > 0)
        truth_idx = int(truth_idx_arr[0]) if truth_idx_arr.size > 0 else -1
        if truth_idx >= 0:
            positive_users += 1
        baseline_rank = pd.to_numeric(g["baseline_pre_rank"], errors="coerce").fillna(10**9).to_numpy(dtype=np.float64, copy=False)
        item_idx = pd.to_numeric(g["item_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
        challenge_score = pd.to_numeric(g["challenge_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False)
        retain_score = pd.to_numeric(g["retain_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False)
        baseline_order = np.lexsort((item_idx.astype(np.int64), baseline_rank.astype(np.float64))).astype(np.int32, copy=False)
        challenger_order = stable_desc_order(challenge_score, baseline_rank, item_idx).astype(np.int32, copy=False)
        incumbent_drop_order = stable_asc_order(retain_score, baseline_rank, item_idx).astype(np.int32, copy=False)

        caches.append(
            {
                "user_idx": user_id,
                "user_segment": str(meta["user_segment"] if meta is not None else g["user_segment"].iloc[0]),
                "n_rows": int(g.shape[0]),
                "truth_idx": int(truth_idx),
                "baseline_truth_rank": int(meta["baseline_truth_rank"]) if meta is not None else 10**9,
                "baseline_rank_bucket": str(meta["baseline_rank_bucket"]) if meta is not None else "absent",
                "baseline_in_pretrim": int(meta["baseline_in_pretrim"]) if meta is not None else 0,
                "baseline_in_top150": int(meta["baseline_in_top150"]) if meta is not None else 0,
                "baseline_in_top250": int(meta["baseline_in_top250"]) if meta is not None else 0,
                "baseline_order": baseline_order,
                "challenger_order": challenger_order,
                "incumbent_drop_order": incumbent_drop_order,
                "baseline_rank": baseline_rank,
                "item_idx": item_idx,
                "label": label,
                "challenge_score": challenge_score,
                "retain_score": retain_score,
                "personal_eligible": pd.to_numeric(g["personal_eligible"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False),
                "consensus_count": pd.to_numeric(g["consensus_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False),
                "detail_count": pd.to_numeric(g["profile_active_detail_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False),
            }
        )

    for user_id, meta in truth_meta.items():
        if user_id in seen_users:
            continue
        caches.append(
            {
                "user_idx": int(meta["user_idx"]),
                "user_segment": str(meta["user_segment"]),
                "n_rows": 0,
                "truth_idx": -1,
                "baseline_truth_rank": int(meta["baseline_truth_rank"]),
                "baseline_rank_bucket": str(meta["baseline_rank_bucket"]),
                "baseline_in_pretrim": int(meta["baseline_in_pretrim"]),
                "baseline_in_top150": int(meta["baseline_in_top150"]),
                "baseline_in_top250": int(meta["baseline_in_top250"]),
                "baseline_order": np.zeros((0,), dtype=np.int32),
                "challenger_order": np.zeros((0,), dtype=np.int32),
                "incumbent_drop_order": np.zeros((0,), dtype=np.int32),
                "baseline_rank": np.zeros((0,), dtype=np.float64),
                "item_idx": np.zeros((0,), dtype=np.int64),
                "label": np.zeros((0,), dtype=np.int8),
                "challenge_score": np.zeros((0,), dtype=np.float64),
                "retain_score": np.zeros((0,), dtype=np.float64),
                "personal_eligible": np.zeros((0,), dtype=np.int8),
                "consensus_count": np.zeros((0,), dtype=np.int8),
                "detail_count": np.zeros((0,), dtype=np.int8),
            }
        )

    caches.sort(key=lambda x: x["user_idx"])
    meta = {
        "users": int(len(caches)),
        "positive_users_in_pretrim": int(positive_users),
        "baseline_truth_in_pretrim": int(sum(int(x["baseline_in_pretrim"]) for x in caches)),
        "baseline_truth_in_top150": int(sum(int(x["baseline_in_top150"]) for x in caches)),
        "baseline_truth_in_top250": int(sum(int(x["baseline_in_top250"]) for x in caches)),
    }
    return caches, meta


def rank_truth_for_user(user_cache: dict[str, Any], cfg: BoundaryConfig) -> tuple[int, str, int]:
    truth_idx = int(user_cache["truth_idx"])
    n_rows = int(user_cache["n_rows"])
    if truth_idx < 0 or n_rows <= 0:
        return 10**9, "absent_pretrim", 0

    baseline_order = user_cache["baseline_order"]
    baseline_rank = user_cache["baseline_rank"]
    challenge_score = user_cache["challenge_score"]
    retain_score = user_cache["retain_score"]
    personal_eligible = user_cache["personal_eligible"]
    consensus_count = user_cache["consensus_count"]
    detail_count = user_cache["detail_count"]
    quota = int(cfg.replace_quota_for(str(user_cache["user_segment"])))

    fixed_mask = baseline_rank <= float(cfg.protect_topk)
    incumbent_mask = (baseline_rank > float(cfg.protect_topk)) & (baseline_rank <= float(TOP150))
    challenger_mask = (
        (baseline_rank > float(TOP150))
        & (baseline_rank <= float(cfg.candidate_cap))
        & (personal_eligible > 0)
        & ((detail_count >= int(cfg.challenger_min_detail)) | (consensus_count >= 3))
    )

    fixed_set = set(int(idx) for idx in baseline_order[fixed_mask[baseline_order]])
    incumbent_order = [int(idx) for idx in user_cache["incumbent_drop_order"] if incumbent_mask[int(idx)]]
    challenger_order = [int(idx) for idx in user_cache["challenger_order"] if challenger_mask[int(idx)]]

    dropped: set[int] = set()
    accepted: list[int] = []
    replacements = 0
    inc_ptr = 0
    for ch_idx in challenger_order:
        if replacements >= quota:
            break
        while inc_ptr < len(incumbent_order) and incumbent_order[inc_ptr] in dropped:
            inc_ptr += 1
        if inc_ptr >= len(incumbent_order):
            break
        inc_idx = int(incumbent_order[inc_ptr])
        if challenge_score[ch_idx] < retain_score[inc_idx] + float(cfg.min_margin):
            break
        dropped.add(inc_idx)
        accepted.append(int(ch_idx))
        replacements += 1
        inc_ptr += 1

    boundary_selected = [int(idx) for idx in baseline_order if incumbent_mask[int(idx)] and int(idx) not in dropped]
    boundary_selected.extend(accepted)
    boundary_selected.sort(
        key=lambda idx: (
            0 if idx in accepted else 1,
            -float(challenge_score[idx]) if idx in accepted else float(baseline_rank[idx]),
            int(user_cache["item_idx"][idx]),
        )
    )

    chosen = fixed_set.union(boundary_selected)
    final_order: list[int] = []
    for idx in baseline_order:
        idx_i = int(idx)
        if idx_i in fixed_set:
            final_order.append(idx_i)
    for idx_i in boundary_selected:
        if idx_i not in final_order:
            final_order.append(int(idx_i))
    for idx in baseline_order:
        idx_i = int(idx)
        if idx_i in chosen:
            continue
        final_order.append(idx_i)

    truth_rank = 10**9
    truth_action = "tail_unchanged"
    for pos, idx_i in enumerate(final_order, start=1):
        if idx_i != truth_idx:
            continue
        truth_rank = int(pos)
        if truth_idx in fixed_set:
            truth_action = "fixed_prefix"
        elif truth_idx in accepted:
            truth_action = "accepted_challenger"
        elif truth_idx in dropped:
            truth_action = "dropped_incumbent"
        elif incumbent_mask[truth_idx]:
            truth_action = "kept_boundary"
        elif challenger_mask[truth_idx]:
            truth_action = "unchosen_challenger"
        break
    return int(truth_rank), str(truth_action), int(replacements)


def summarize_truth_rows(truth_rows: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "users": int(truth_rows.shape[0]),
        "truth_in_top80_users": int((truth_rows["truth_rank"] <= TOP80).sum()),
        "truth_in_top150_users": int((truth_rows["truth_rank"] <= TOP150).sum()),
        "truth_in_top250_users": int((truth_rows["truth_rank"] <= TOP250).sum()),
        "recovered_from_151_250_users": int(
            ((truth_rows["baseline_truth_rank"] > TOP150) & (truth_rows["baseline_truth_rank"] <= TOP250) & (truth_rows["truth_rank"] <= TOP150)).sum()
        ),
        "recovered_from_251_plus_users": int(
            ((truth_rows["baseline_truth_rank"] > TOP250) & (truth_rows["truth_rank"] <= TOP150)).sum()
        ),
        "lost_from_top150_users": int(
            ((truth_rows["baseline_truth_rank"] <= TOP150) & (truth_rows["truth_rank"] > TOP150)).sum()
        ),
    }
    summary["truth_in_top80_rate"] = round(float(summary["truth_in_top80_users"] / max(1, summary["users"])), 6)
    summary["truth_in_top150_rate"] = round(float(summary["truth_in_top150_users"] / max(1, summary["users"])), 6)
    summary["truth_in_top250_rate"] = round(float(summary["truth_in_top250_users"] / max(1, summary["users"])), 6)
    seg_rows: list[dict[str, Any]] = []
    for seg, group in truth_rows.groupby("user_segment", sort=True):
        row: dict[str, Any] = {
            "user_segment": str(seg),
            "users": int(group.shape[0]),
            "truth_in_top150_users": int((group["truth_rank"] <= TOP150).sum()),
            "truth_in_top250_users": int((group["truth_rank"] <= TOP250).sum()),
        }
        row["truth_in_top150_rate"] = round(float(row["truth_in_top150_users"] / max(1, row["users"])), 6)
        row["truth_in_top250_rate"] = round(float(row["truth_in_top250_users"] / max(1, row["users"])), 6)
        seg_rows.append(row)
    return summary, seg_rows


def run_baseline(user_caches: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for user in user_caches:
        rows.append(
            {
                "user_idx": int(user["user_idx"]),
                "user_segment": str(user["user_segment"]),
                "baseline_truth_rank": int(user["baseline_truth_rank"]),
                "baseline_rank_bucket": str(user["baseline_rank_bucket"]),
                "truth_rank": int(user["baseline_truth_rank"]),
                "truth_action": "baseline",
                "replacement_count": 0,
            }
        )
    truth_rows = pd.DataFrame(rows)
    summary, _ = summarize_truth_rows(truth_rows)
    return summary, truth_rows


def run_config(
    user_caches: list[dict[str, Any]], cfg: BoundaryConfig
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for user in user_caches:
        truth_rank, truth_action, replacement_count = rank_truth_for_user(user, cfg)
        rows.append(
            {
                "user_idx": int(user["user_idx"]),
                "user_segment": str(user["user_segment"]),
                "baseline_truth_rank": int(user["baseline_truth_rank"]),
                "baseline_rank_bucket": str(user["baseline_rank_bucket"]),
                "truth_rank": int(truth_rank),
                "truth_action": str(truth_action),
                "replacement_count": int(replacement_count),
            }
        )
    truth_rows = pd.DataFrame(rows)
    summary, seg_rows = summarize_truth_rows(truth_rows)
    action_summary = (
        truth_rows.assign(is_in_top150=(truth_rows["truth_rank"] <= TOP150).astype(np.int8))
        .groupby(["user_segment", "baseline_rank_bucket", "truth_action"], dropna=False)
        .agg(users=("user_idx", "count"), truth_in_top150_users=("is_in_top150", "sum"))
        .reset_index()
        .sort_values(["user_segment", "baseline_rank_bucket", "truth_in_top150_users", "users"], ascending=[True, True, False, False], kind="stable")
    )
    payload = {**cfg.to_dict(), **summary}
    for item in seg_rows:
        seg = str(item["user_segment"])
        payload[f"{seg}_truth_in_top150_users"] = int(item["truth_in_top150_users"])
        payload[f"{seg}_truth_in_top150_rate"] = float(item["truth_in_top150_rate"])
        payload[f"{seg}_truth_in_top250_users"] = int(item["truth_in_top250_users"])
        payload[f"{seg}_truth_in_top250_rate"] = float(item["truth_in_top250_rate"])
    return payload, truth_rows, action_summary


def is_better_summary(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    cand_key = (
        int(candidate["truth_in_top150_users"]),
        int(candidate.get("heavy_truth_in_top150_users", 0)),
        -int(candidate["lost_from_top150_users"]),
        int(candidate["recovered_from_151_250_users"] + candidate["recovered_from_251_plus_users"]),
        int(candidate["truth_in_top250_users"]),
    )
    inc_key = (
        int(incumbent["truth_in_top150_users"]),
        int(incumbent.get("heavy_truth_in_top150_users", 0)),
        -int(incumbent["lost_from_top150_users"]),
        int(incumbent["recovered_from_151_250_users"] + incumbent["recovered_from_251_plus_users"]),
        int(incumbent["truth_in_top250_users"]),
    )
    return cand_key > inc_key


def _init_worker_user_caches(user_caches: list[dict[str, Any]]) -> None:
    global _WORKER_USER_CACHES
    _WORKER_USER_CACHES = user_caches


def _run_config_summary_worker(cfg: BoundaryConfig) -> dict[str, Any]:
    if _WORKER_USER_CACHES is None:
        raise RuntimeError("worker user caches not initialized")
    summary, _, _ = run_config(_WORKER_USER_CACHES, cfg)
    return summary


def main() -> None:
    run_dir = resolve_head_v2_run()
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_stage09_bucket10_boundary_repair_simulator")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pdf, truth_pdf, load_meta = load_probe_inputs(run_dir)
    print(
        f"[LOAD] train_rows={load_meta['train_rows']} truth_users={load_meta['truth_users']} "
        f"cohort={load_meta['cohort_path'] or 'ALL'}"
    )
    train_pdf, truth_pdf = prepare_features(train_pdf, truth_pdf)
    user_caches, cache_meta = build_user_caches(train_pdf, truth_pdf)
    print(
        f"[CACHE] users={cache_meta['users']} positives_in_pretrim={cache_meta['positive_users_in_pretrim']} "
        f"baseline_top150={cache_meta['baseline_truth_in_top150']}"
    )

    baseline_summary, baseline_truth_rows = run_baseline(user_caches)
    grid = build_grid()
    print(f"[GRID] configs={len(grid)}")

    rows: list[dict[str, Any]] = []
    best_cfg: BoundaryConfig | None = None
    best_summary: dict[str, Any] | None = None

    use_pool = bool(NUM_WORKERS > 1 and os.name != "nt")
    if use_pool:
        ctx = mp.get_context("fork")
        print(f"[POOL] workers={NUM_WORKERS} chunk_size={POOL_CHUNK_SIZE} start_method=fork")
        with ctx.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker_user_caches,
            initargs=(user_caches,),
        ) as pool:
            for idx, summary in enumerate(pool.imap(_run_config_summary_worker, grid, chunksize=max(1, POOL_CHUNK_SIZE)), start=1):
                cfg = grid[idx - 1]
                summary["config_idx"] = int(idx)
                summary["delta_truth_in_top150_users"] = int(
                    summary["truth_in_top150_users"] - baseline_summary["truth_in_top150_users"]
                )
                summary["delta_truth_in_top250_users"] = int(
                    summary["truth_in_top250_users"] - baseline_summary["truth_in_top250_users"]
                )
                summary["delta_recovered_from_151_plus_users"] = int(
                    summary["recovered_from_151_250_users"] + summary["recovered_from_251_plus_users"]
                )
                rows.append(summary)
                if is_better_summary(summary, best_summary):
                    best_cfg = cfg
                    best_summary = summary
                if idx % 40 == 0 or idx == len(grid):
                    top150_best = best_summary["truth_in_top150_users"] if best_summary is not None else -1
                    print(f"[SIM] {idx}/{len(grid)} best_top150={top150_best}")
    else:
        for idx, cfg in enumerate(grid, start=1):
            summary, _, _ = run_config(user_caches, cfg)
            summary["config_idx"] = int(idx)
            summary["delta_truth_in_top150_users"] = int(
                summary["truth_in_top150_users"] - baseline_summary["truth_in_top150_users"]
            )
            summary["delta_truth_in_top250_users"] = int(
                summary["truth_in_top250_users"] - baseline_summary["truth_in_top250_users"]
            )
            summary["delta_recovered_from_151_plus_users"] = int(
                summary["recovered_from_151_250_users"] + summary["recovered_from_251_plus_users"]
            )
            rows.append(summary)
            if is_better_summary(summary, best_summary):
                best_cfg = cfg
                best_summary = summary
            if idx % 40 == 0 or idx == len(grid):
                top150_best = best_summary["truth_in_top150_users"] if best_summary is not None else -1
                print(f"[SIM] {idx}/{len(grid)} best_top150={top150_best}")

    best_truth_rows: pd.DataFrame | None = None
    best_action_summary: pd.DataFrame | None = None
    if best_cfg is not None:
        _, best_truth_rows, best_action_summary = run_config(user_caches, best_cfg)

    result_df = pd.DataFrame(rows).sort_values(
        ["truth_in_top150_users", "heavy_truth_in_top150_users", "lost_from_top150_users", "delta_recovered_from_151_plus_users", "truth_in_top250_users", "config_idx"],
        ascending=[False, False, True, False, False, True],
        kind="stable",
    )
    result_df.to_csv(out_dir / "boundary_grid_results.csv", index=False, encoding="utf-8")
    baseline_truth_rows.to_csv(out_dir / "baseline_truth_ranks.csv", index=False, encoding="utf-8")
    if best_truth_rows is not None:
        best_truth_rows.to_csv(out_dir / "best_truth_ranks.csv", index=False, encoding="utf-8")
    if best_action_summary is not None:
        best_action_summary.to_csv(out_dir / "best_action_summary.csv", index=False, encoding="utf-8")

    payload = {
        "run_tag": RUN_TAG,
        "source_head_v2_run": run_dir.as_posix(),
        "load_meta": load_meta,
        "cache_meta": cache_meta,
        "baseline_summary": baseline_summary,
        "grid_size": int(len(grid)),
        "grid_values": {
            "protect_topk": PROTECT_TOPK_LIST,
            "candidate_cap": CANDIDATE_CAP_LIST,
            "heavy_replace": HEAVY_REPLACE_LIST,
            "mid_replace": MID_REPLACE_LIST,
            "challenger_min_detail": CHALLENGER_MIN_DETAIL_LIST,
            "min_margin": MIN_MARGIN_LIST,
        },
        "best_config": best_cfg.to_dict() if best_cfg is not None else None,
        "best_summary": best_summary,
    }
    write_json(out_dir / "boundary_sim_summary.json", payload)
    print(f"[INFO] wrote {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
