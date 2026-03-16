from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE09_RUN_DIR = Path(os.getenv("INPUT_09_RUN_DIR", "").strip())
BUCKET = int(os.getenv("MV_ROUTE_PROBE_BUCKET", "10").strip() or 10)
OUTPUT_ROOT = Path(os.getenv("OUTPUT_09_MV_ROUTE_PROBE_ROOT_DIR", "").strip() or str(STAGE09_RUN_DIR.parent / "audits"))

BONUS_WEIGHTS = [float(x) for x in os.getenv("MV_ROUTE_PROBE_BONUS_WEIGHTS", "0.05,0.1,0.2,0.4,0.8").split(",") if x.strip()]
BOUNDARY_PROTECT_TOPK = int(os.getenv("MV_ROUTE_PROBE_BOUNDARY_PROTECT_TOPK", "130").strip() or 130)
BOUNDARY_REPLACE_TOPK = int(os.getenv("MV_ROUTE_PROBE_BOUNDARY_REPLACE_TOPK", "20").strip() or 20)
BOUNDARY_ROUTE_TOPK = [int(x) for x in os.getenv("MV_ROUTE_PROBE_BOUNDARY_ROUTE_TOPK", "20,40,80").split(",") if x.strip()]
BOUNDARY_SCALES = [float(x) for x in os.getenv("MV_ROUTE_PROBE_BOUNDARY_SCALES", "0.25,0.5,1.0,1.5").split(",") if x.strip()]
APPEND_TOPK = [int(x) for x in os.getenv("MV_ROUTE_PROBE_APPEND_TOPK", "40,80,120").split(",") if x.strip()]
APPEND_SCALES = [float(x) for x in os.getenv("MV_ROUTE_PROBE_APPEND_SCALES", "0.25,0.5,1.0").split(",") if x.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def truth_metrics(rank_df: pd.DataFrame) -> dict[str, Any]:
    if rank_df.empty:
        return {"n_users": 0, "truth_in_all": 0, "truth_in_top150": 0, "truth_in_top250": 0}
    r = pd.to_numeric(rank_df["rank"], errors="coerce").fillna(0).astype(int)
    return {
        "n_users": int(rank_df["user_idx"].nunique()),
        "truth_in_all": int((r > 0).sum()),
        "truth_in_top150": int((r.between(1, 150)).sum()),
        "truth_in_top250": int((r.between(1, 250)).sum()),
    }


def build_truth_join(base: pd.DataFrame, truth: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    joined = truth.merge(
        base[["user_idx", "item_idx", rank_col]].rename(columns={"item_idx": "pred_item", rank_col: "rank"}),
        left_on=["user_idx", "true_item_idx"],
        right_on=["user_idx", "pred_item"],
        how="left",
    )
    joined["rank"] = pd.to_numeric(joined["rank"], errors="coerce").fillna(0).astype(int)
    return joined[["user_idx", "true_item_idx", "rank"]]


def normalize_group_score(pdf: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    pdf = pdf.copy()
    g = pdf.groupby("user_idx")[score_col]
    min_v = g.transform("min")
    max_v = g.transform("max")
    denom = (max_v - min_v).replace(0.0, np.nan)
    pdf[out_col] = ((pdf[score_col] - min_v) / denom).fillna(0.0).astype(np.float32)
    return pdf


def rank_with_score(pdf: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    pdf = pdf.sort_values(["user_idx", score_col, "item_idx"], ascending=[True, False, True], kind="mergesort").copy()
    pdf[rank_col] = pdf.groupby("user_idx", sort=False).cumcount() + 1
    return pdf


def build_route_stats(route_df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    if route_df.empty:
        return pd.DataFrame(columns=["route_name", "n_rows", "n_users", "truth_in_top150", "truth_in_top250"])

    route_combined = (
        route_df.sort_values(["user_idx", "item_idx", "route_score", "route_confidence", "route_rank"], ascending=[True, True, False, False, True], kind="mergesort")
        .drop_duplicates(["user_idx", "item_idx"], keep="first")
    )
    route_combined = rank_with_score(route_combined.assign(route_mix_score=route_combined["route_score"] * route_combined["route_confidence"]), "route_mix_score", "route_mix_rank")
    joined_all = build_truth_join(route_combined.rename(columns={"route_mix_rank": "rank"}), truth, "rank")
    m_all = truth_metrics(joined_all)
    out.append(
        {
            "route_name": "combined_best",
            "n_rows": int(route_combined.shape[0]),
            "n_users": int(route_combined["user_idx"].nunique()),
            "truth_in_all": int(m_all["truth_in_all"]),
            "truth_in_top150": int(m_all["truth_in_top150"]),
            "truth_in_top250": int(m_all["truth_in_top250"]),
        }
    )
    for route_name, grp in route_df.groupby("route_name", sort=True):
        g = grp.copy()
        g["route_mix_score"] = pd.to_numeric(g["route_score"], errors="coerce").fillna(0.0) * pd.to_numeric(
            g["route_confidence"], errors="coerce"
        ).fillna(0.0)
        g = rank_with_score(g, "route_mix_score", "route_mix_rank")
        joined = build_truth_join(g.rename(columns={"route_mix_rank": "rank"}), truth, "rank")
        m = truth_metrics(joined)
        out.append(
            {
                "route_name": str(route_name),
                "n_rows": int(g.shape[0]),
                "n_users": int(g["user_idx"].nunique()),
                "truth_in_all": int(m["truth_in_all"]),
                "truth_in_top150": int(m["truth_in_top150"]),
                "truth_in_top250": int(m["truth_in_top250"]),
            }
        )
    return pd.DataFrame(out)


def run_bonus_existing(enriched: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = enriched.copy()
    work["route_bonus_score"] = pd.to_numeric(work.get("mv_route_best_score", 0.0), errors="coerce").fillna(0.0)
    for w in BONUS_WEIGHTS:
        trial = work.copy()
        trial["trial_score"] = pd.to_numeric(trial["head_score"], errors="coerce").fillna(0.0) + float(w) * trial["route_bonus_score"]
        trial = rank_with_score(trial, "trial_score", "trial_rank")
        joined = build_truth_join(trial, truth, "trial_rank")
        m = truth_metrics(joined)
        rows.append(
            {
                "method": "bonus_existing",
                "weight": float(w),
                "truth_in_all": int(m["truth_in_all"]),
                "truth_in_top150": int(m["truth_in_top150"]),
                "truth_in_top250": int(m["truth_in_top250"]),
            }
        )
    return pd.DataFrame(rows)


def run_boundary_replace(enriched: pd.DataFrame, route_df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = enriched.copy()
    base["head_score"] = pd.to_numeric(base["head_score"], errors="coerce").fillna(0.0)
    route_best = (
        route_df.assign(route_mix_score=pd.to_numeric(route_df["route_score"], errors="coerce").fillna(0.0) * pd.to_numeric(route_df["route_confidence"], errors="coerce").fillna(0.0))
        .sort_values(["user_idx", "item_idx", "route_mix_score", "route_rank"], ascending=[True, True, False, True], kind="mergesort")
        .drop_duplicates(["user_idx", "item_idx"], keep="first")
    )
    route_best = normalize_group_score(route_best, "route_mix_score", "route_mix_score_norm")
    for route_topk in BOUNDARY_ROUTE_TOPK:
        route_top = route_best[route_best["route_rank"] <= int(route_topk)].copy()
        missing = route_top.merge(base[["user_idx", "item_idx"]], on=["user_idx", "item_idx"], how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"].copy()
        if missing.empty:
            continue
        base_group = {uid: grp.copy() for uid, grp in base.groupby("user_idx", sort=False)}
        miss_group = {uid: grp.copy() for uid, grp in missing.groupby("user_idx", sort=False)}
        for scale in BOUNDARY_SCALES:
            results: list[pd.DataFrame] = []
            for uid, grp in base_group.items():
                grp = grp.copy()
                top_fixed = grp[grp["final_pre_rank"] <= int(BOUNDARY_PROTECT_TOPK)].copy()
                boundary = grp[(grp["final_pre_rank"] > int(BOUNDARY_PROTECT_TOPK)) & (grp["final_pre_rank"] <= int(BOUNDARY_PROTECT_TOPK + BOUNDARY_REPLACE_TOPK))].copy()
                if boundary.empty:
                    results.append(grp)
                    continue
                challengers = miss_group.get(uid)
                if challengers is None or challengers.empty:
                    results.append(grp)
                    continue
                boundary_floor = float(boundary["head_score"].min())
                boundary_ceiling = float(boundary["head_score"].max())
                boundary_gap = max(1e-6, abs(boundary_ceiling - boundary_floor))
                ch = challengers.copy()
                ch["trial_score"] = float(boundary_floor) + float(scale) * float(boundary_gap) * pd.to_numeric(
                    ch["route_mix_score_norm"], errors="coerce"
                ).fillna(0.0)
                ch["item_idx"] = ch["item_idx"].astype(int)
                ch["user_idx"] = ch["user_idx"].astype(int)
                inc = boundary[["user_idx", "item_idx", "head_score", "final_pre_rank"]].copy()
                inc["trial_score"] = inc["head_score"]
                inc["is_new_route"] = 0
                ch = ch[["user_idx", "item_idx", "trial_score"]].copy()
                ch["final_pre_rank"] = 999999
                ch["is_new_route"] = 1
                pool = pd.concat([inc, ch], ignore_index=True)
                pool = pool.sort_values(["trial_score", "final_pre_rank", "item_idx"], ascending=[False, True, True], kind="mergesort").head(
                    int(BOUNDARY_REPLACE_TOPK)
                )
                top_fixed = top_fixed[["user_idx", "item_idx"]].copy()
                top_fixed["trial_rank"] = np.arange(1, top_fixed.shape[0] + 1, dtype=np.int32)
                pool = pool[["user_idx", "item_idx"]].copy()
                pool["trial_rank"] = np.arange(
                    int(top_fixed.shape[0]) + 1,
                    int(top_fixed.shape[0]) + int(pool.shape[0]) + 1,
                    dtype=np.int32,
                )
                tail = grp[grp["final_pre_rank"] > int(BOUNDARY_PROTECT_TOPK + BOUNDARY_REPLACE_TOPK)][["user_idx", "item_idx"]].copy()
                tail["trial_rank"] = np.arange(
                    int(top_fixed.shape[0]) + int(pool.shape[0]) + 1,
                    int(top_fixed.shape[0]) + int(pool.shape[0]) + int(tail.shape[0]) + 1,
                    dtype=np.int32,
                )
                results.append(pd.concat([top_fixed, pool, tail], ignore_index=True))
            trial = pd.concat(results, ignore_index=True)
            joined = build_truth_join(trial, truth, "trial_rank")
            m = truth_metrics(joined)
            rows.append(
                {
                    "method": "boundary_replace_missing",
                    "route_topk": int(route_topk),
                    "scale": float(scale),
                    "truth_in_all": int(m["truth_in_all"]),
                    "truth_in_top150": int(m["truth_in_top150"]),
                    "truth_in_top250": int(m["truth_in_top250"]),
                }
            )
    return pd.DataFrame(rows)


def run_append_mix(enriched: pd.DataFrame, route_df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = enriched.copy()
    route_best = (
        route_df.assign(route_mix_score=pd.to_numeric(route_df["route_score"], errors="coerce").fillna(0.0) * pd.to_numeric(route_df["route_confidence"], errors="coerce").fillna(0.0))
        .sort_values(["user_idx", "item_idx", "route_mix_score", "route_rank"], ascending=[True, True, False, True], kind="mergesort")
        .drop_duplicates(["user_idx", "item_idx"], keep="first")
    )
    route_best = normalize_group_score(route_best, "route_mix_score", "route_mix_score_norm")
    base_keys = set(zip(base["user_idx"].astype(int), base["item_idx"].astype(int)))
    missing = route_best[[tuple(x) not in base_keys for x in route_best[["user_idx", "item_idx"]].itertuples(index=False, name=None)]].copy()
    if missing.empty:
        return pd.DataFrame(columns=["method", "route_topk", "scale", "truth_in_all", "truth_in_top150", "truth_in_top250"])

    anchor_map = (
        base[base["final_pre_rank"].between(150, 200)][["user_idx", "head_score"]]
        .groupby("user_idx", as_index=False)["head_score"]
        .mean()
        .rename(columns={"head_score": "anchor_head_score"})
    )
    gap_map = (
        base[base["final_pre_rank"].between(130, 200)][["user_idx", "head_score"]]
        .groupby("user_idx", as_index=False)["head_score"]
        .agg(lambda x: float(max(x) - min(x)) if len(x) > 0 else 1e-6)
        .rename(columns={"head_score": "anchor_gap"})
    )
    missing = missing.merge(anchor_map, on="user_idx", how="left").merge(gap_map, on="user_idx", how="left")
    missing["anchor_head_score"] = pd.to_numeric(missing["anchor_head_score"], errors="coerce").fillna(base["head_score"].min())
    missing["anchor_gap"] = pd.to_numeric(missing["anchor_gap"], errors="coerce").fillna(1.0).replace(0.0, 1.0)

    for route_topk in APPEND_TOPK:
        cand = missing[missing["route_rank"] <= int(route_topk)].copy()
        if cand.empty:
            continue
        for scale in APPEND_SCALES:
            aug = cand.copy()
            aug["trial_score"] = aug["anchor_head_score"] + float(scale) * aug["anchor_gap"] * aug["route_mix_score_norm"]
            aug = aug[["user_idx", "item_idx", "trial_score"]].copy()
            base_trial = base[["user_idx", "item_idx", "head_score", "final_pre_rank"]].copy()
            base_trial["trial_score"] = pd.to_numeric(base_trial["head_score"], errors="coerce").fillna(0.0)
            combo = pd.concat(
                [
                    base_trial[["user_idx", "item_idx", "trial_score", "final_pre_rank"]],
                    aug.assign(final_pre_rank=999999),
                ],
                ignore_index=True,
            )
            combo = rank_with_score(combo, "trial_score", "trial_rank")
            joined = build_truth_join(combo, truth, "trial_rank")
            m = truth_metrics(joined)
            rows.append(
                {
                    "method": "append_missing_mix",
                    "route_topk": int(route_topk),
                    "scale": float(scale),
                    "truth_in_all": int(m["truth_in_all"]),
                    "truth_in_top150": int(m["truth_in_top150"]),
                    "truth_in_top250": int(m["truth_in_top250"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if not STAGE09_RUN_DIR.exists():
        raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {STAGE09_RUN_DIR}")
    bucket_dir = STAGE09_RUN_DIR / f"bucket_{int(BUCKET)}"
    enriched_path = bucket_dir / "candidates_enriched_audit.parquet"
    truth_path = bucket_dir / "truth.parquet"
    route_path = bucket_dir / "profile_multivector_route_audit.parquet"
    if not enriched_path.exists():
        raise FileNotFoundError(f"missing enriched audit: {enriched_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"missing truth: {truth_path}")
    if not route_path.exists():
        raise FileNotFoundError(f"missing multivector route audit: {route_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_stage09_bucket10_multivector_integration_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = read_parquet(enriched_path)
    truth = read_parquet(truth_path)
    route_df = read_parquet(route_path)

    truth = truth[["user_idx", "true_item_idx"]].drop_duplicates(["user_idx"]).copy()
    enriched["user_idx"] = pd.to_numeric(enriched["user_idx"], errors="coerce").fillna(-1).astype(int)
    enriched["item_idx"] = pd.to_numeric(enriched["item_idx"], errors="coerce").fillna(-1).astype(int)
    enriched["final_pre_rank"] = pd.to_numeric(enriched["final_pre_rank"], errors="coerce").fillna(999999).astype(int)
    enriched["head_score"] = pd.to_numeric(enriched["head_score"], errors="coerce").fillna(0.0)
    route_df["user_idx"] = pd.to_numeric(route_df["user_idx"], errors="coerce").fillna(-1).astype(int)
    route_df["item_idx"] = pd.to_numeric(route_df["item_idx"], errors="coerce").fillna(-1).astype(int)
    route_df["route_rank"] = pd.to_numeric(route_df["route_rank"], errors="coerce").fillna(999999).astype(int)
    route_df["route_score"] = pd.to_numeric(route_df["route_score"], errors="coerce").fillna(0.0)
    route_df["route_confidence"] = pd.to_numeric(route_df["route_confidence"], errors="coerce").fillna(0.0)

    route_best = (
        route_df.assign(mv_route_best_score=route_df["route_score"] * route_df["route_confidence"])
        .sort_values(["user_idx", "item_idx", "mv_route_best_score", "route_rank"], ascending=[True, True, False, True], kind="mergesort")
        .drop_duplicates(["user_idx", "item_idx"], keep="first")[["user_idx", "item_idx", "mv_route_best_score", "route_name", "route_rank"]]
    )
    enriched = enriched.merge(route_best[["user_idx", "item_idx", "mv_route_best_score"]], on=["user_idx", "item_idx"], how="left")
    enriched["mv_route_best_score"] = pd.to_numeric(enriched["mv_route_best_score"], errors="coerce").fillna(0.0)

    baseline_join = build_truth_join(enriched, truth, "final_pre_rank")
    baseline_metrics = truth_metrics(baseline_join)

    route_stats = build_route_stats(route_df, truth)
    bonus_df = run_bonus_existing(enriched, truth)
    boundary_df = run_boundary_replace(enriched, route_df, truth)
    append_df = run_append_mix(enriched, route_df, truth)
    all_trials = pd.concat([bonus_df, boundary_df, append_df], ignore_index=True, sort=False)
    if not all_trials.empty:
        all_trials = all_trials.sort_values(["truth_in_top150", "truth_in_top250", "truth_in_all"], ascending=[False, False, False], kind="mergesort")

    route_stats.to_csv(out_dir / "route_stats.csv", index=False)
    all_trials.to_csv(out_dir / "integration_trials.csv", index=False)

    summary = {
        "input_09_run_dir": str(STAGE09_RUN_DIR),
        "bucket": int(BUCKET),
        "baseline": baseline_metrics,
        "route_stats_best_top150": (
            route_stats.sort_values(["truth_in_top150", "truth_in_top250"], ascending=[False, False]).iloc[0].to_dict()
            if not route_stats.empty
            else {}
        ),
        "best_trial": all_trials.iloc[0].to_dict() if not all_trials.empty else {},
        "n_route_rows": int(route_df.shape[0]),
        "n_enriched_rows": int(enriched.shape[0]),
        "n_truth_users": int(truth.shape[0]),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
