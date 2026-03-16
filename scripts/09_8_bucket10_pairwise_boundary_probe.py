from __future__ import annotations

import importlib.util
import itertools
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pipeline.project_paths import env_or_project_path


RUN_TAG = "stage09_bucket10_pairwise_boundary_probe"

INPUT_HEAD_V2_RUN_DIR = os.getenv("INPUT_HEAD_V2_RUN_DIR", "").strip()
INPUT_HEAD_V2_ROOT = env_or_project_path("INPUT_09_HEAD_V2_ROOT_DIR", "data/output/09_head_v2_table")
INPUT_HEAD_V2_SUFFIX = "_stage09_bucket10_head_v2_table"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_PAIRWISE_BOUNDARY_ROOT_DIR", "data/output/09_pairwise_boundary_probe")

TOP150 = int(os.getenv("PAIRWISE_BOUNDARY_TOP150", "150").strip() or 150)
TOP250 = int(os.getenv("PAIRWISE_BOUNDARY_TOP250", "250").strip() or 250)
TOP80 = int(os.getenv("PAIRWISE_BOUNDARY_TOP80", "80").strip() or 80)

PROTECT_TOPK_LIST = [int(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_PROTECT_TOPK", "130").split(",") if x.strip()]
CANDIDATE_CAP_LIST = [int(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_CANDIDATE_CAP", "190,200").split(",") if x.strip()]
HEAVY_REPLACE_LIST = [int(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_HEAVY_REPLACE", "6,8").split(",") if x.strip()]
MID_REPLACE_LIST = [int(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_MID_REPLACE", "8,10").split(",") if x.strip()]
MIN_DETAIL_LIST = [int(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_MIN_DETAIL", "1,2").split(",") if x.strip()]
DETAIL_W_LIST = [float(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_DETAIL_W", "0.0,0.2,0.4").split(",") if x.strip()]
CLUSTER_W_LIST = [float(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_CLUSTER_W", "0.0,0.2,0.4").split(",") if x.strip()]
PROFILE_W_LIST = [float(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_PROFILE_W", "0.0,0.1,0.2").split(",") if x.strip()]
CONSENSUS_W_LIST = [float(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_CONSENSUS_W", "0.0,0.05").split(",") if x.strip()]
POPULAR_W_LIST = [float(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_POPULAR_W", "0.0,0.1").split(",") if x.strip()]
MIN_MARGIN_LIST = [float(x.strip()) for x in os.getenv("PAIRWISE_BOUNDARY_MIN_MARGIN", "0.0,0.02,0.04").split(",") if x.strip()]

NUM_WORKERS = int(os.getenv("PAIRWISE_BOUNDARY_NUM_WORKERS", "1").strip() or 1)
POOL_CHUNK_SIZE = int(os.getenv("PAIRWISE_BOUNDARY_POOL_CHUNK_SIZE", "4").strip() or 4)

_WORKER_USER_CACHES: list[dict[str, Any]] | None = None


def _load_base_module() -> Any:
    script_path = Path(__file__).with_name("09_7_bucket10_boundary_repair_simulator.py")
    spec = importlib.util.spec_from_file_location("stage09_boundary_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class PairwiseConfig:
    protect_topk: int
    candidate_cap: int
    heavy_replace: int
    mid_replace: int
    min_detail: int
    detail_w: float
    cluster_w: float
    profile_w: float
    consensus_w: float
    popular_w: float
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
            "min_detail": int(self.min_detail),
            "detail_w": float(self.detail_w),
            "cluster_w": float(self.cluster_w),
            "profile_w": float(self.profile_w),
            "consensus_w": float(self.consensus_w),
            "popular_w": float(self.popular_w),
            "min_margin": float(self.min_margin),
        }


def build_grid() -> list[PairwiseConfig]:
    grid: list[PairwiseConfig] = []
    for vals in itertools.product(
        PROTECT_TOPK_LIST,
        CANDIDATE_CAP_LIST,
        HEAVY_REPLACE_LIST,
        MID_REPLACE_LIST,
        MIN_DETAIL_LIST,
        DETAIL_W_LIST,
        CLUSTER_W_LIST,
        PROFILE_W_LIST,
        CONSENSUS_W_LIST,
        POPULAR_W_LIST,
        MIN_MARGIN_LIST,
    ):
        cfg = PairwiseConfig(*vals)
        max_repl = TOP150 - int(cfg.protect_topk)
        if int(cfg.protect_topk) >= TOP150:
            continue
        if int(cfg.candidate_cap) <= TOP150:
            continue
        if int(cfg.heavy_replace) > max_repl or int(cfg.mid_replace) > max_repl:
            continue
        grid.append(cfg)
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
        baseline_order = np.lexsort((item_idx.astype(np.int64), baseline_rank.astype(np.float64))).astype(np.int32, copy=False)
        challenger_order = _BASE.stable_desc_order(
            pd.to_numeric(g["challenge_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False),
            baseline_rank,
            item_idx,
        ).astype(np.int32, copy=False)
        incumbent_order = _BASE.stable_asc_order(
            pd.to_numeric(g["retain_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False),
            baseline_rank,
            item_idx,
        ).astype(np.int32, copy=False)
        caches.append(
            {
                "user_idx": user_id,
                "user_segment": str(meta["user_segment"] if meta is not None else g["user_segment"].iloc[0]),
                "truth_idx": int(truth_idx),
                "baseline_truth_rank": int(meta["baseline_truth_rank"]) if meta is not None else 10**9,
                "baseline_rank_bucket": str(meta["baseline_rank_bucket"]) if meta is not None else "absent",
                "baseline_in_pretrim": int(meta["baseline_in_pretrim"]) if meta is not None else 0,
                "baseline_in_top150": int(meta["baseline_in_top150"]) if meta is not None else 0,
                "baseline_in_top250": int(meta["baseline_in_top250"]) if meta is not None else 0,
                "baseline_order": baseline_order,
                "challenger_order": challenger_order,
                "incumbent_order": incumbent_order,
                "baseline_rank": baseline_rank,
                "item_idx": item_idx,
                "label": label,
                "challenge_score": pd.to_numeric(g["challenge_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False),
                "retain_score": pd.to_numeric(g["retain_score"], errors="coerce").fillna(-10**9).to_numpy(dtype=np.float64, copy=False),
                "detail_best_support_score": pd.to_numeric(g["detail_best_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False),
                "cluster_support_score": pd.to_numeric(g["cluster_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False),
                "profile_route_support_score": pd.to_numeric(g["profile_route_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False),
                "popular_penalty_score": pd.to_numeric(g["popular_penalty_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False),
                "consensus_count": pd.to_numeric(g["consensus_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False),
                "detail_count": pd.to_numeric(g["profile_active_detail_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False),
                "personal_eligible": pd.to_numeric(g["personal_eligible"], errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False),
            }
        )

    for user_id, meta in truth_meta.items():
        if user_id in seen_users:
            continue
        caches.append(
            {
                "user_idx": int(meta["user_idx"]),
                "user_segment": str(meta["user_segment"]),
                "truth_idx": -1,
                "baseline_truth_rank": int(meta["baseline_truth_rank"]),
                "baseline_rank_bucket": str(meta["baseline_rank_bucket"]),
                "baseline_in_pretrim": int(meta["baseline_in_pretrim"]),
                "baseline_in_top150": int(meta["baseline_in_top150"]),
                "baseline_in_top250": int(meta["baseline_in_top250"]),
                "baseline_order": np.zeros((0,), dtype=np.int32),
                "challenger_order": np.zeros((0,), dtype=np.int32),
                "incumbent_order": np.zeros((0,), dtype=np.int32),
                "baseline_rank": np.zeros((0,), dtype=np.float64),
                "item_idx": np.zeros((0,), dtype=np.int64),
                "label": np.zeros((0,), dtype=np.int8),
                "challenge_score": np.zeros((0,), dtype=np.float64),
                "retain_score": np.zeros((0,), dtype=np.float64),
                "detail_best_support_score": np.zeros((0,), dtype=np.float64),
                "cluster_support_score": np.zeros((0,), dtype=np.float64),
                "profile_route_support_score": np.zeros((0,), dtype=np.float64),
                "popular_penalty_score": np.zeros((0,), dtype=np.float64),
                "consensus_count": np.zeros((0,), dtype=np.int8),
                "detail_count": np.zeros((0,), dtype=np.int8),
                "personal_eligible": np.zeros((0,), dtype=np.int8),
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


def pair_margin(cache: dict[str, Any], ch_idx: int, inc_idx: int, cfg: PairwiseConfig) -> float:
    return (
        float(cache["challenge_score"][ch_idx] - cache["retain_score"][inc_idx])
        + float(cfg.detail_w) * float(cache["detail_best_support_score"][ch_idx] - cache["detail_best_support_score"][inc_idx])
        + float(cfg.cluster_w) * float(cache["cluster_support_score"][ch_idx] - cache["cluster_support_score"][inc_idx])
        + float(cfg.profile_w) * float(cache["profile_route_support_score"][ch_idx] - cache["profile_route_support_score"][inc_idx])
        + float(cfg.consensus_w) * float(cache["consensus_count"][ch_idx] - cache["consensus_count"][inc_idx])
        - float(cfg.popular_w) * float(cache["popular_penalty_score"][ch_idx] - cache["popular_penalty_score"][inc_idx])
    )


def rank_truth_for_user(cache: dict[str, Any], cfg: PairwiseConfig) -> tuple[int, str, int]:
    truth_idx = int(cache["truth_idx"])
    if truth_idx < 0 or cache["baseline_order"].size == 0:
        return 10**9, "absent_pretrim", 0

    baseline_rank = cache["baseline_rank"]
    fixed_mask = baseline_rank <= float(cfg.protect_topk)
    incumbent_mask = (baseline_rank > float(cfg.protect_topk)) & (baseline_rank <= float(TOP150))
    challenger_mask = (
        (baseline_rank > float(TOP150))
        & (baseline_rank <= float(cfg.candidate_cap))
        & (cache["personal_eligible"] > 0)
        & ((cache["detail_count"] >= int(cfg.min_detail)) | (cache["consensus_count"] >= 3))
    )

    fixed_set = set(int(idx) for idx in cache["baseline_order"][fixed_mask[cache["baseline_order"]]])
    incumbents = [int(idx) for idx in cache["incumbent_order"] if incumbent_mask[int(idx)]]
    challengers = [int(idx) for idx in cache["challenger_order"] if challenger_mask[int(idx)]]
    dropped: set[int] = set()
    accepted: list[int] = []
    quota = int(cfg.replace_quota_for(str(cache["user_segment"])))
    replacements = 0

    for ch_idx in challengers:
        if replacements >= quota:
            break
        best_inc = None
        best_margin = None
        for inc_idx in incumbents:
            if inc_idx in dropped:
                continue
            margin = pair_margin(cache, ch_idx, inc_idx, cfg)
            if best_margin is None or margin > best_margin:
                best_margin = margin
                best_inc = inc_idx
        if best_inc is None or best_margin is None or best_margin < float(cfg.min_margin):
            continue
        dropped.add(int(best_inc))
        accepted.append(int(ch_idx))
        replacements += 1

    boundary_selected = [int(idx) for idx in cache["baseline_order"] if incumbent_mask[int(idx)] and int(idx) not in dropped]
    boundary_selected.extend(accepted)
    boundary_selected.sort(
        key=lambda idx: (
            0 if idx in accepted else 1,
            -float(cache["challenge_score"][idx]) if idx in accepted else float(cache["baseline_rank"][idx]),
            int(cache["item_idx"][idx]),
        )
    )

    final_order: list[int] = [int(idx) for idx in cache["baseline_order"] if int(idx) in fixed_set]
    for idx in boundary_selected:
        if idx not in final_order:
            final_order.append(int(idx))
    chosen = fixed_set.union(boundary_selected)
    for idx in cache["baseline_order"]:
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


def run_config(user_caches: list[dict[str, Any]], cfg: PairwiseConfig) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
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
    summary, seg_rows = _BASE.summarize_truth_rows(truth_rows)
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


def _init_worker_user_caches(user_caches: list[dict[str, Any]]) -> None:
    global _WORKER_USER_CACHES
    _WORKER_USER_CACHES = user_caches


def _run_config_summary_worker(cfg: PairwiseConfig) -> dict[str, Any]:
    if _WORKER_USER_CACHES is None:
        raise RuntimeError("worker user caches not initialized")
    summary, _, _ = run_config(_WORKER_USER_CACHES, cfg)
    return summary


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


def main() -> None:
    run_dir = _BASE.resolve_head_v2_run()
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_stage09_bucket10_pairwise_boundary_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pdf, truth_pdf, load_meta = _BASE.load_probe_inputs(run_dir)
    print(f"[LOAD] train_rows={load_meta['train_rows']} truth_users={load_meta['truth_users']} cohort={load_meta['cohort_path'] or 'ALL'}")
    train_pdf, truth_pdf = _BASE.prepare_features(train_pdf, truth_pdf)
    user_caches, cache_meta = build_user_caches(train_pdf, truth_pdf)
    print(f"[CACHE] users={cache_meta['users']} positives_in_pretrim={cache_meta['positive_users_in_pretrim']} baseline_top150={cache_meta['baseline_truth_in_top150']}")

    baseline_summary, baseline_truth_rows = _BASE.run_baseline(user_caches)
    grid = build_grid()
    print(f"[GRID] configs={len(grid)}")

    rows: list[dict[str, Any]] = []
    best_cfg: PairwiseConfig | None = None
    best_summary: dict[str, Any] | None = None

    use_pool = bool(NUM_WORKERS > 1 and os.name != "nt")
    if use_pool:
        ctx = mp.get_context("fork")
        print(f"[POOL] workers={NUM_WORKERS} chunk_size={POOL_CHUNK_SIZE} start_method=fork")
        with ctx.Pool(processes=NUM_WORKERS, initializer=_init_worker_user_caches, initargs=(user_caches,)) as pool:
            for idx, summary in enumerate(pool.imap(_run_config_summary_worker, grid, chunksize=max(1, POOL_CHUNK_SIZE)), start=1):
                cfg = grid[idx - 1]
                summary["config_idx"] = int(idx)
                summary["delta_truth_in_top150_users"] = int(summary["truth_in_top150_users"] - baseline_summary["truth_in_top150_users"])
                summary["delta_truth_in_top250_users"] = int(summary["truth_in_top250_users"] - baseline_summary["truth_in_top250_users"])
                summary["delta_recovered_from_151_plus_users"] = int(summary["recovered_from_151_250_users"] + summary["recovered_from_251_plus_users"])
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
            summary["delta_truth_in_top150_users"] = int(summary["truth_in_top150_users"] - baseline_summary["truth_in_top150_users"])
            summary["delta_truth_in_top250_users"] = int(summary["truth_in_top250_users"] - baseline_summary["truth_in_top250_users"])
            summary["delta_recovered_from_151_plus_users"] = int(summary["recovered_from_151_250_users"] + summary["recovered_from_251_plus_users"])
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
    result_df.to_csv(out_dir / "pairwise_grid_results.csv", index=False, encoding="utf-8")
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
            "min_detail": MIN_DETAIL_LIST,
            "detail_w": DETAIL_W_LIST,
            "cluster_w": CLUSTER_W_LIST,
            "profile_w": PROFILE_W_LIST,
            "consensus_w": CONSENSUS_W_LIST,
            "popular_w": POPULAR_W_LIST,
            "min_margin": MIN_MARGIN_LIST,
            "num_workers": int(NUM_WORKERS),
            "pool_chunk_size": int(POOL_CHUNK_SIZE),
        },
        "best_config": best_cfg.to_dict() if best_cfg is not None else None,
        "best_summary": best_summary,
    }
    write_json(out_dir / "pairwise_summary.json", payload)
    print(f"[INFO] wrote {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
