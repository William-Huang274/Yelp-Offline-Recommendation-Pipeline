import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


_TARGET_BAND_ALIASES = {
    "boundary_11_30": "boundary_11_30",
    "11-30": "boundary_11_30",
    "11_30": "boundary_11_30",
    "011_030": "boundary_11_30",
    "rescue_31_60": "rescue_31_60",
    "31-60": "rescue_31_60",
    "31_60": "rescue_31_60",
    "031_060": "rescue_31_60",
    "rescue_61_100": "rescue_61_100",
    "61-100": "rescue_61_100",
    "61_100": "rescue_61_100",
    "061_100": "rescue_61_100",
    "001_010": "head_guard",
    "151_plus": "outside_top100",
    "outside_top100": "outside_top100",
    "head_guard": "head_guard",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_str(value: Any) -> str:
    return str(value or "").strip()


def _band(rank: Any) -> str:
    rank_i = _to_int(rank, 10**9)
    if rank_i <= 10:
        return "head_guard"
    if rank_i <= 30:
        return "boundary_11_30"
    if rank_i <= 60:
        return "rescue_31_60"
    if rank_i <= 100:
        return "rescue_61_100"
    return "outside_top100"


def _canonical_target_band(raw: Any) -> str:
    token = _to_str(raw).lower()
    canonical = _TARGET_BAND_ALIASES.get(token)
    if canonical:
        return canonical
    raise ValueError(f"unsupported target band: {raw}")


def _normalize_band_value(raw: Any) -> str:
    token = _to_str(raw).lower()
    if not token:
        return ""
    canonical = _TARGET_BAND_ALIASES.get(token)
    if canonical:
        return canonical
    return _to_str(raw)


def _summarize_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def _stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"count": 0, "min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return {"count": 0, "min": 0.0, "p50": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p50": float(values.quantile(0.50)),
        "mean": float(values.mean()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stage11_1 pairwise pool readiness by chosen true band")
    parser.add_argument("--stage11-run-dir", required=True, help="stage11_1 run dir")
    parser.add_argument("--target-band", required=True, help="boundary_11_30 | rescue_31_60 | rescue_61_100")
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument("--output", default="", help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_band = _canonical_target_band(args.target_band)
    stage11_run_dir = Path(args.stage11_run_dir).expanduser()
    parquet_dir = stage11_run_dir / f"bucket_{int(args.bucket)}" / "pairwise_pool_all_parquet"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"pairwise_pool_all_parquet not found: {parquet_dir}")

    df = pd.read_parquet(parquet_dir)
    if "learned_rank_band" in df.columns:
        df["target_true_band"] = df["learned_rank_band"].map(_normalize_band_value)
    elif "learned_rank" in df.columns:
        df["target_true_band"] = df["learned_rank"].map(_band)
    elif "pre_rank" in df.columns:
        df["target_true_band"] = df["pre_rank"].map(_band)
    else:
        raise RuntimeError("stage11 parquet missing learned_rank/learned_rank_band/pre_rank")

    pos = df.loc[pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    pos = pos.loc[pos["target_true_band"].eq(target_band)].copy()
    target_users = pos[["user_idx", "split"]].drop_duplicates()
    scoped = df.merge(target_users, on=["user_idx", "split"], how="inner")

    out: dict[str, Any] = {
        "stage11_run_dir": str(stage11_run_dir),
        "target_band": target_band,
        "bucket": int(args.bucket),
        "splits": {},
    }

    for split_name in ["train", "eval"]:
        split_pos = pos.loc[pos["split"].astype(str).eq(split_name)].copy()
        split_scoped = scoped.loc[scoped["split"].astype(str).eq(split_name)].copy()
        user_counts = split_pos["user_idx"].value_counts(dropna=False)
        neg_counts = (
            split_scoped.loc[pd.to_numeric(split_scoped["label"], errors="coerce").fillna(0).astype(int).eq(0)]
            .groupby("user_idx", dropna=False)
            .size()
        )
        out["splits"][split_name] = {
            "positive_rows": int(len(split_pos)),
            "target_users": int(split_pos["user_idx"].nunique()),
            "scoped_rows_total": int(len(split_scoped)),
            "negative_rows_total": int(
                pd.to_numeric(split_scoped["label"], errors="coerce").fillna(0).astype(int).eq(0).sum()
            ),
            "positive_rows_per_user": _stats(user_counts),
            "negative_rows_per_user": _stats(neg_counts),
            "label_source_counts": _summarize_counts(split_scoped["label_source"].astype(str)),
            "neg_tier_counts": _summarize_counts(split_scoped["neg_tier"].astype(str)),
            "pre_rank_band_counts": _summarize_counts(split_scoped["pre_rank_band"].astype(str)),
            "positive_primary_reason_counts": _summarize_counts(split_pos.get("primary_reason", pd.Series(dtype=str)).astype(str)),
            "signal_rates_on_positive": {
                "user_avoid_present_rate": float(split_pos.get("avoidance_text_v2", pd.Series("", index=split_pos.index)).astype(str).str.strip().ne("").mean()) if len(split_pos) else 0.0,
                "user_evidence_present_rate": float(split_pos.get("user_evidence_text", pd.Series("", index=split_pos.index)).astype(str).str.strip().ne("").mean()) if len(split_pos) else 0.0,
                "pair_fit_rate": float(pd.to_numeric(split_pos.get("pair_fit_fact_count_v1"), errors="coerce").fillna(0).gt(0).mean()) if len(split_pos) else 0.0,
                "pair_conflict_rate": float(pd.to_numeric(split_pos.get("pair_friction_fact_count_v1"), errors="coerce").fillna(0).gt(0).mean()) if len(split_pos) else 0.0,
                "pair_evidence_rate": float(pd.to_numeric(split_pos.get("pair_evidence_fact_count_v1"), errors="coerce").fillna(0).gt(0).mean()) if len(split_pos) else 0.0,
                "multisource_fit_rate": float(pd.to_numeric(split_pos.get("pair_has_multisource_fit_v1"), errors="coerce").fillna(0).gt(0).mean()) if len(split_pos) else 0.0,
                "detail_support_rate": float(pd.to_numeric(split_pos.get("pair_has_detail_support_v1"), errors="coerce").fillna(0).gt(0).mean()) if len(split_pos) else 0.0,
                "boundary_prompt_ready_rate": float(pd.to_numeric(split_pos.get("boundary_prompt_ready_v1"), errors="coerce").fillna(0).gt(0).mean()) if len(split_pos) else 0.0,
            },
        }

    output_text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
