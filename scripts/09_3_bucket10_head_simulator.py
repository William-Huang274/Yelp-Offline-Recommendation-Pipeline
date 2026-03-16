from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pipeline.project_paths import env_or_project_path


RUN_TAG = "stage09_bucket10_head_simulator"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_HEAD_SIM_ROOT_DIR", "data/output/09_head_simulator")

PROBE_BUCKET = int(os.getenv("HEAD_SIM_BUCKET", "10").strip() or 10)
COHORT_PATH = os.getenv("HEAD_SIM_COHORT_PATH", "").strip()
MAX_CANDIDATE_ROWS = int(os.getenv("HEAD_SIM_MAX_CANDIDATE_ROWS", "3000000").strip() or 3000000)
TOPN_LIST = [int(x.strip()) for x in os.getenv("HEAD_SIM_TOPN_LIST", "80,150,250").split(",") if x.strip()]

GRID_FRONT_GUARDS = [int(x.strip()) for x in os.getenv("HEAD_SIM_FRONT_GUARDS", "80,100,120").split(",") if x.strip()]
GRID_PERSONAL_LANES = [int(x.strip()) for x in os.getenv("HEAD_SIM_PERSONAL_LANES", "20,30,40").split(",") if x.strip()]
GRID_HEAVY_PERSONAL_LANES = [
    int(x.strip()) for x in os.getenv("HEAD_SIM_HEAVY_PERSONAL_LANES", "40,60,80").split(",") if x.strip()
]
GRID_RECOVERY_LANES = [int(x.strip()) for x in os.getenv("HEAD_SIM_RECOVERY_LANES", "0,10,20").split(",") if x.strip()]
GRID_RECOVERY_MIN_RANKS = [
    int(x.strip()) for x in os.getenv("HEAD_SIM_RECOVERY_MIN_RANKS", "120,150,180").split(",") if x.strip()
]

PROFILE_BONUS = float(os.getenv("HEAD_SIM_PROFILE_BONUS", "0.060").strip() or 0.060)
CLUSTER_BONUS = float(os.getenv("HEAD_SIM_CLUSTER_BONUS", "0.050").strip() or 0.050)
MULTISOURCE_BONUS = float(os.getenv("HEAD_SIM_MULTISOURCE_BONUS", "0.025").strip() or 0.025)
PROFILE_CLUSTER_BONUS = float(os.getenv("HEAD_SIM_PROFILE_CLUSTER_BONUS", "0.020").strip() or 0.020)
POPULAR_PENALTY = float(os.getenv("HEAD_SIM_POPULAR_PENALTY", "0.030").strip() or 0.030)
ALS_ONLY_PENALTY = float(os.getenv("HEAD_SIM_ALS_ONLY_PENALTY", "0.020").strip() or 0.020)
TOWER_WEIGHT = float(os.getenv("HEAD_SIM_TOWER_WEIGHT", "0.020").strip() or 0.020)
SEQ_WEIGHT = float(os.getenv("HEAD_SIM_SEQ_WEIGHT", "0.015").strip() or 0.015)
SEM_WEIGHT = float(os.getenv("HEAD_SIM_SEM_WEIGHT", "0.015").strip() or 0.015)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def resolve_candidate_path(bucket_dir: Path) -> Path:
    for name in ("candidates_pretrim150.parquet", "candidates_pretrim.parquet"):
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no candidate parquet in {bucket_dir}")


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


def load_probe_pdf(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    bucket_dir = run_dir / f"bucket_{int(PROBE_BUCKET)}"
    cand_path = resolve_candidate_path(bucket_dir)
    truth = pd.read_parquet(bucket_dir / "truth.parquet", columns=["user_idx", "true_item_idx", "user_id"])
    truth = truth.drop_duplicates("user_idx")
    cohort_filter = resolve_cohort_filter()
    if cohort_filter is not None:
        key, vals = cohort_filter
        truth = truth[truth[key].isin(vals)].copy()
    user_idx_vals = set(pd.to_numeric(truth["user_idx"], errors="coerce").dropna().astype(np.int32).tolist())
    cols = [
        "user_idx",
        "item_idx",
        "user_segment",
        "user_train_count",
        "source_set",
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
        "pre_score",
        "pre_rank",
        "pre_rank_before_layered",
        "semantic_effective_score",
        "tower_inv",
        "seq_inv",
        "profile_rank_inv",
        "cluster_rank_inv",
    ]
    cand = pd.read_parquet(cand_path, columns=cols)
    cand = cand[cand["user_idx"].isin(user_idx_vals)].copy()
    if len(cand) > int(MAX_CANDIDATE_ROWS):
        raise RuntimeError(
            f"candidate rows exceed cap: rows={len(cand)} max={MAX_CANDIDATE_ROWS}; narrow cohort or raise HEAD_SIM_MAX_CANDIDATE_ROWS"
        )
    probe = cand.merge(
        truth[["user_idx", "true_item_idx", "user_id"]],
        on="user_idx",
        how="inner",
    )
    probe["label"] = (pd.to_numeric(probe["item_idx"], errors="coerce") == pd.to_numeric(probe["true_item_idx"], errors="coerce")).astype(
        np.int8
    )
    meta = {
        "candidate_path": cand_path.as_posix(),
        "rows": int(probe.shape[0]),
        "users": int(probe["user_idx"].nunique()),
        "positives": int(probe["label"].sum()),
        "max_candidate_rows_cap": int(MAX_CANDIDATE_ROWS),
        "cohort_path": str(COHORT_PATH or ""),
    }
    return probe, meta


def normalize_source_combo(val: Any) -> str:
    if isinstance(val, np.ndarray):
        return "+".join(sorted(str(x) for x in val.tolist()))
    if isinstance(val, (list, tuple)):
        return "+".join(sorted(str(x) for x in val))
    if pd.isna(val):
        return "NA"
    return str(val)


def prepare_features(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf.copy()
    numeric_cols = [
        "item_idx",
        "true_item_idx",
        "label",
        "user_train_count",
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
        "pre_score",
        "pre_rank",
        "pre_rank_before_layered",
        "semantic_effective_score",
        "tower_inv",
        "seq_inv",
        "profile_rank_inv",
        "cluster_rank_inv",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["user_segment"] = out["user_segment"].fillna("unknown").astype(str)
    out["source_combo"] = out["source_set"].apply(normalize_source_combo)
    out["has_profile_or_cluster"] = ((out["has_profile"] > 0.5) | (out["has_cluster"] > 0.5)).astype(np.float32)
    out["is_als_only"] = ((out["has_als"] > 0.5) & (out["source_count"] <= 1.0)).astype(np.float32)
    out["is_heavy"] = (out["user_segment"] == "heavy").astype(np.float32)
    out["is_mid"] = (out["user_segment"] == "mid").astype(np.float32)
    out["is_light"] = (out["user_segment"] == "light").astype(np.float32)
    out["multi_nonpopular"] = (out["nonpopular_source_count"] >= 2.0).astype(np.float32)
    out["personal_eligible"] = ((out["has_profile_or_cluster"] > 0.5) | (out["multi_nonpopular"] > 0.5)).astype(np.int8)
    out["personal_score"] = (
        out["pre_score"].fillna(0.0)
        + PROFILE_BONUS * out["has_profile"].fillna(0.0)
        + CLUSTER_BONUS * out["has_cluster"].fillna(0.0)
        + MULTISOURCE_BONUS * out["multi_nonpopular"].fillna(0.0)
        + PROFILE_CLUSTER_BONUS * out["profile_cluster_source_count"].fillna(0.0)
        + TOWER_WEIGHT * out["tower_inv"].fillna(0.0)
        + SEQ_WEIGHT * out["seq_inv"].fillna(0.0)
        + SEM_WEIGHT * out["semantic_effective_score"].fillna(0.0)
        + 0.050 * out["profile_rank_inv"].fillna(0.0)
        + 0.040 * out["cluster_rank_inv"].fillna(0.0)
        - POPULAR_PENALTY * out["has_popular"].fillna(0.0)
        - ALS_ONLY_PENALTY * out["is_als_only"].fillna(0.0)
    ).astype(np.float64)
    out["recovery_score"] = (
        out["personal_score"]
        + 0.020 * np.where(pd.to_numeric(out["pre_rank"], errors="coerce").fillna(10**9) > 150.0, 1.0, 0.0)
    ).astype(np.float64)
    return out


@dataclass(frozen=True)
class HeadConfig:
    front_guard: int
    personal_lane: int
    heavy_personal_lane: int
    recovery_lane: int
    recovery_min_rank: int

    def to_dict(self) -> dict[str, int]:
        return {
            "front_guard": int(self.front_guard),
            "personal_lane": int(self.personal_lane),
            "heavy_personal_lane": int(self.heavy_personal_lane),
            "recovery_lane": int(self.recovery_lane),
            "recovery_min_rank": int(self.recovery_min_rank),
        }


def _stable_desc_order(score: np.ndarray, pre_rank: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
    return np.lexsort((item_idx.astype(np.int64), pre_rank.astype(np.float64), -score.astype(np.float64)))


def build_user_cache(pdf: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    caches: list[dict[str, Any]] = []
    pos_count = 0
    for user_idx, g in pdf.groupby("user_idx", sort=False):
        g2 = g.sort_values(["pre_rank", "item_idx"], ascending=[True, True], kind="stable").reset_index(drop=True)
        label = g2["label"].to_numpy(dtype=np.int8, copy=False)
        truth_idx_arr = np.flatnonzero(label > 0)
        truth_idx = int(truth_idx_arr[0]) if truth_idx_arr.size > 0 else -1
        if truth_idx >= 0:
            pos_count += 1
        pre_rank = pd.to_numeric(g2["pre_rank"], errors="coerce").fillna(10**9).to_numpy(dtype=np.float64, copy=False)
        item_idx = pd.to_numeric(g2["item_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
        personal_eligible = pd.to_numeric(g2["personal_eligible"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False)
        personal_score = pd.to_numeric(g2["personal_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False)
        recovery_score = pd.to_numeric(g2["recovery_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False)
        baseline_order = np.arange(g2.shape[0], dtype=np.int32)
        personal_order = _stable_desc_order(personal_score, pre_rank, item_idx)
        personal_order = personal_order[personal_eligible[personal_order] > 0]
        recovery_order = _stable_desc_order(recovery_score, pre_rank, item_idx)
        recovery_order = recovery_order[personal_eligible[recovery_order] > 0]
        caches.append(
            {
                "user_idx": int(user_idx),
                "user_segment": str(g2["user_segment"].iloc[0]),
                "n_rows": int(g2.shape[0]),
                "truth_idx": int(truth_idx),
                "truth_pre_rank": int(pre_rank[truth_idx]) if truth_idx >= 0 and np.isfinite(pre_rank[truth_idx]) else 10**9,
                "baseline_order": baseline_order,
                "personal_order": personal_order.astype(np.int32, copy=False),
                "recovery_order": recovery_order.astype(np.int32, copy=False),
                "pre_rank": pre_rank,
                "label": label,
            }
        )
    meta = {"users": int(len(caches)), "positives": int(pos_count)}
    return caches, meta


def rank_truth_for_user(user_cache: dict[str, Any], cfg: HeadConfig) -> int:
    truth_idx = int(user_cache["truth_idx"])
    if truth_idx < 0:
        return 10**9
    baseline_order = user_cache["baseline_order"]
    personal_order = user_cache["personal_order"]
    recovery_order = user_cache["recovery_order"]
    pre_rank = user_cache["pre_rank"]
    n_rows = int(user_cache["n_rows"])
    chosen = np.zeros((n_rows,), dtype=bool)
    final_rank = 0
    front_guard = int(min(cfg.front_guard, n_rows))
    personal_lane = int(cfg.heavy_personal_lane if user_cache["user_segment"] == "heavy" else cfg.personal_lane)
    recovery_lane = int(cfg.recovery_lane)

    def take_idx(idx: int) -> bool:
        nonlocal final_rank
        if chosen[idx]:
            return False
        chosen[idx] = True
        final_rank += 1
        return idx == truth_idx

    for idx in baseline_order[:front_guard]:
        if take_idx(int(idx)):
            return final_rank

    personal_taken = 0
    if personal_lane > 0:
        for idx in personal_order:
            idx_i = int(idx)
            if chosen[idx_i]:
                continue
            if take_idx(idx_i):
                return final_rank
            personal_taken += 1
            if personal_taken >= personal_lane:
                break

    recovery_taken = 0
    if recovery_lane > 0:
        for idx in recovery_order:
            idx_i = int(idx)
            if chosen[idx_i]:
                continue
            if pre_rank[idx_i] <= float(cfg.recovery_min_rank):
                continue
            if take_idx(idx_i):
                return final_rank
            recovery_taken += 1
            if recovery_taken >= recovery_lane:
                break

    for idx in baseline_order:
        idx_i = int(idx)
        if chosen[idx_i]:
            continue
        if take_idx(idx_i):
            return final_rank
    return final_rank


def summarize_ranks(truth_rows: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "users": int(truth_rows.shape[0]),
        "rank_min": int(truth_rows["truth_rank"].min()),
        "rank_median": float(truth_rows["truth_rank"].median()),
        "rank_p75": float(truth_rows["truth_rank"].quantile(0.75)),
        "rank_max": int(truth_rows["truth_rank"].max()),
    }
    for topn in TOPN_LIST:
        mask = truth_rows["truth_rank"] <= int(topn)
        summary[f"truth_in_top{int(topn)}_users"] = int(mask.sum())
        summary[f"truth_in_top{int(topn)}_rate"] = round(float(mask.mean()), 6)
    seg_rows: list[dict[str, Any]] = []
    for seg, g in truth_rows.groupby("user_segment", sort=True):
        row: dict[str, Any] = {
            "user_segment": str(seg),
            "users": int(g.shape[0]),
            "rank_median": float(g["truth_rank"].median()),
        }
        for topn in TOPN_LIST:
            mask = g["truth_rank"] <= int(topn)
            row[f"truth_in_top{int(topn)}_users"] = int(mask.sum())
            row[f"truth_in_top{int(topn)}_rate"] = round(float(mask.mean()), 6)
        seg_rows.append(row)
    return summary, seg_rows


def run_baseline(user_caches: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for user in user_caches:
        rows.append(
            {
                "user_idx": int(user["user_idx"]),
                "user_segment": str(user["user_segment"]),
                "truth_rank": int(user["truth_pre_rank"] or 10**9),
            }
        )
    truth_rows = pd.DataFrame(rows)
    summary, _ = summarize_ranks(truth_rows)
    return summary, truth_rows


def run_config(user_caches: list[dict[str, Any]], cfg: HeadConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for user in user_caches:
        rank = rank_truth_for_user(user, cfg)
        rows.append(
            {
                "user_idx": int(user["user_idx"]),
                "user_segment": str(user["user_segment"]),
                "truth_rank": int(rank),
            }
        )
    truth_rows = pd.DataFrame(rows)
    summary, seg_rows = summarize_ranks(truth_rows)
    payload = {
        **cfg.to_dict(),
        **summary,
    }
    for item in seg_rows:
        seg = str(item["user_segment"])
        payload[f"{seg}_truth_in_top150_rate"] = float(item.get("truth_in_top150_rate", 0.0))
        payload[f"{seg}_truth_in_top250_rate"] = float(item.get("truth_in_top250_rate", 0.0))
    return payload, truth_rows


def build_grid() -> list[HeadConfig]:
    grid = []
    for front_guard, personal_lane, heavy_personal_lane, recovery_lane, recovery_min_rank in itertools.product(
        GRID_FRONT_GUARDS,
        GRID_PERSONAL_LANES,
        GRID_HEAVY_PERSONAL_LANES,
        GRID_RECOVERY_LANES,
        GRID_RECOVERY_MIN_RANKS,
    ):
        if personal_lane + recovery_lane > max(TOPN_LIST):
            continue
        if heavy_personal_lane + recovery_lane > max(TOPN_LIST):
            continue
        grid.append(
            HeadConfig(
                front_guard=int(front_guard),
                personal_lane=int(personal_lane),
                heavy_personal_lane=int(heavy_personal_lane),
                recovery_lane=int(recovery_lane),
                recovery_min_rank=int(recovery_min_rank),
            )
        )
    return grid


def main() -> None:
    run_dir = resolve_stage09_run()
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_stage09_bucket10_head_simulator")
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_pdf, load_meta = load_probe_pdf(run_dir=run_dir)
    print(f"[LOAD] rows={load_meta['rows']} users={load_meta['users']} positives={load_meta['positives']}")
    probe_pdf = prepare_features(probe_pdf)
    user_caches, cache_meta = build_user_cache(probe_pdf)
    print(f"[CACHE] users={cache_meta['users']} positives={cache_meta['positives']}")

    baseline_summary, baseline_truth_rows = run_baseline(user_caches)
    grid = build_grid()
    print(f"[GRID] configs={len(grid)}")
    rows: list[dict[str, Any]] = []
    best_cfg: HeadConfig | None = None
    best_summary: dict[str, Any] | None = None
    best_truth_rows: pd.DataFrame | None = None
    for idx, cfg in enumerate(grid, start=1):
        summary, truth_rows = run_config(user_caches, cfg)
        summary["config_idx"] = int(idx)
        summary["delta_truth_in_top150_users"] = int(
            summary["truth_in_top150_users"] - baseline_summary["truth_in_top150_users"]
        )
        summary["delta_truth_in_top250_users"] = int(
            summary["truth_in_top250_users"] - baseline_summary["truth_in_top250_users"]
        )
        rows.append(summary)
        if best_summary is None:
            best_cfg = cfg
            best_summary = summary
            best_truth_rows = truth_rows
        else:
            if (
                summary["truth_in_top150_users"],
                summary["truth_in_top250_users"],
                summary["truth_in_top80_users"],
            ) > (
                best_summary["truth_in_top150_users"],
                best_summary["truth_in_top250_users"],
                best_summary["truth_in_top80_users"],
            ):
                best_cfg = cfg
                best_summary = summary
                best_truth_rows = truth_rows
        if idx % 20 == 0 or idx == len(grid):
            print(
                f"[SIM] {idx}/{len(grid)} best_top150="
                f"{best_summary['truth_in_top150_users'] if best_summary else -1}"
            )

    result_df = pd.DataFrame(rows).sort_values(
        ["truth_in_top150_users", "truth_in_top250_users", "truth_in_top80_users", "config_idx"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    result_df.to_csv(out_dir / "head_sim_grid_results.csv", index=False, encoding="utf-8")
    baseline_truth_rows.to_csv(out_dir / "baseline_truth_ranks.csv", index=False, encoding="utf-8")
    if best_truth_rows is not None:
        best_truth_rows.to_csv(out_dir / "best_truth_ranks.csv", index=False, encoding="utf-8")

    payload = {
        "run_tag": RUN_TAG,
        "source_stage09_run": run_dir.as_posix(),
        "bucket": int(PROBE_BUCKET),
        "load_meta": load_meta,
        "cache_meta": cache_meta,
        "baseline_summary": baseline_summary,
        "grid_size": int(len(grid)),
        "grid_values": {
            "front_guards": GRID_FRONT_GUARDS,
            "personal_lanes": GRID_PERSONAL_LANES,
            "heavy_personal_lanes": GRID_HEAVY_PERSONAL_LANES,
            "recovery_lanes": GRID_RECOVERY_LANES,
            "recovery_min_ranks": GRID_RECOVERY_MIN_RANKS,
        },
        "score_params": {
            "profile_bonus": PROFILE_BONUS,
            "cluster_bonus": CLUSTER_BONUS,
            "multisource_bonus": MULTISOURCE_BONUS,
            "profile_cluster_bonus": PROFILE_CLUSTER_BONUS,
            "popular_penalty": POPULAR_PENALTY,
            "als_only_penalty": ALS_ONLY_PENALTY,
            "tower_weight": TOWER_WEIGHT,
            "seq_weight": SEQ_WEIGHT,
            "sem_weight": SEM_WEIGHT,
        },
        "best_config": best_cfg.to_dict() if best_cfg is not None else None,
        "best_summary": best_summary,
    }
    write_json(out_dir / "head_sim_summary.json", payload)
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
