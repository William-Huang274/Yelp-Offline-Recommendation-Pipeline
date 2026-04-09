from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pipeline.stage11_pairwise import classify_boundary_constructability


RUN_TAG = "stage11_boundary_constructability_audit"
_TARGET_BAND_ALIASES = {
    "boundary_11_30": "boundary_11_30",
    "11-30": "boundary_11_30",
    "11_30": "boundary_11_30",
    "rescue_31_60": "rescue_31_60",
    "31-60": "rescue_31_60",
    "31_60": "rescue_31_60",
    "rescue_61_100": "rescue_61_100",
    "61-100": "rescue_61_100",
    "61_100": "rescue_61_100",
}
_SUSPECT_POSITIVE_AVOID_PHRASES = {
    "great service",
    "excellent service",
    "friendly service",
    "service prompt",
    "super clean",
    "very clean",
    "clean bathroom",
    "clean bathrooms",
    "great bartenders",
    "friendly staff",
}
_RECENT_CONTEXT_ALIAS_GROUPS: dict[str, set[str]] = {
    "late night": {
        "late night",
        "late hours",
        "late night meals",
        "late night visits",
        "nightlife",
        "happy hour",
        "date night outings",
        "sit down dinner",
        "sit down meals",
    },
    "fast casual": {
        "fast casual",
        "fast casual meals",
        "quick casual meals",
        "weekday lunch",
        "family friendly settings",
    },
    "group dining": {
        "group dining",
        "family friendly settings",
        "celebration",
        "sit down meals",
    },
}
_GENERIC_SCENE_TERMS = {
    "breakfast",
    "lunch",
    "dinner",
    "weekend",
    "plans",
}


def require_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")
    return path


def _read_parquet_from_runs(
    primary_run_dir: Path,
    source_run_dir: Path | None,
    filename: str,
    *,
    required: bool = True,
) -> pd.DataFrame:
    primary_path = primary_run_dir / filename
    if primary_path.exists():
        return pd.read_parquet(primary_path)
    if source_run_dir is not None:
        source_path = source_run_dir / filename
        if source_path.exists():
            return pd.read_parquet(source_path)
    if not required:
        return pd.DataFrame()
    fallback = source_run_dir / filename if source_run_dir is not None else primary_path
    raise FileNotFoundError(f"required parquet not found: {primary_path} (fallback: {fallback})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stage11 prompt constructability by target band")
    parser.add_argument("--stage09-run-dir", required=True, help="stage09 semantic assets run dir")
    parser.add_argument("--source-run-dir", default="", help="optional source semantic materials run dir")
    parser.add_argument("--export-run-dir", default="", help="optional export-only run dir")
    parser.add_argument(
        "--target-band",
        default="boundary_11_30",
        help="target chosen true band: boundary_11_30 | rescue_31_60 | rescue_61_100",
    )
    parser.add_argument("--output", default="", help="optional JSON output path")
    parser.add_argument("--details-output", default="", help="optional parquet output path for per-user details")
    return parser.parse_args()


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_str(value: Any) -> str:
    return str(value or "").strip()


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).gt(0)


def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


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


def _bucket_user_richness(row: pd.Series) -> str:
    profile_tier = _to_str(row.get("user_profile_richness_tier_v2"))
    has_focus = _to_int(row.get("user_has_specific_focus_v2")) > 0
    has_recent = (
        _to_int(row.get("user_has_specific_recent_v2")) > 0
        or _to_int(row.get("user_has_recent_text_v2")) > 0
    )
    has_avoid = _to_int(row.get("user_has_avoid_signal_v2")) > 0
    has_evidence = (
        _to_int(row.get("user_has_strong_clean_evidence_v2")) > 0
        or _to_int(row.get("user_has_clean_evidence_v2")) > 0
    )
    has_history = _to_int(row.get("user_has_history_evidence_v2")) > 0
    visible_facts = _to_int(row.get("user_visible_fact_count_v1"))
    quality_signals = _to_int(row.get("user_quality_signal_count_v2"))
    signal_count = sum([has_focus, has_recent, has_avoid, has_evidence, has_history])
    if profile_tier == "FULL" and signal_count >= 4 and visible_facts >= 5 and quality_signals >= 4:
        return "heavy"
    if signal_count >= 2 and visible_facts >= 3 and quality_signals >= 2:
        return "mid"
    return "light"


def _classify_constructability(row: pd.Series) -> tuple[str, list[str]]:
    return classify_boundary_constructability(
        row.to_dict(),
        rival_total=_to_int(row.get("rival_total")),
        rival_head_or_boundary=_to_int(row.get("rival_head_or_boundary")),
    )


def _summarize_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def _load_export_ready_users(export_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = export_run_dir / "pairwise_train.jsonl"
    eval_path = export_run_dir / "pairwise_eval.jsonl"

    def _read_jsonl(path: Path, split: str) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["user_idx", "chosen_learned_rank_band", "split"])
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(
                    {
                        "user_idx": _to_int(row.get("user_idx"), -1),
                        "chosen_learned_rank_band": _to_str(row.get("chosen_learned_rank_band")),
                        "split": split,
                    }
                )
        return pd.DataFrame(rows)

    train_df = _read_jsonl(train_path, "train")
    eval_df = _read_jsonl(eval_path, "eval")
    return train_df, eval_df


def _source_support_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "source_positive_rate": float(df["user_source_positive_sentence_count_v1"].gt(0).mean()),
        "source_negative_rate": float(df["user_source_negative_sentence_count_v1"].gt(0).mean()),
        "source_recent_rate": float(df["user_source_recent_sentence_count_v1"].gt(0).mean()),
        "source_tip_rate": float(df["user_source_tip_sentence_count_v1"].gt(0).mean()),
        "source_history_rate": float(df["user_source_history_anchor_count_v1"].gt(0).mean()),
        "has_avoid_rate": float(df["has_avoid"].mean()),
        "has_user_evidence_rate": float(df["has_user_evidence"].mean()),
        "has_fit_rate": float(df["has_fit"].mean()),
        "has_multisource_fit_rate": float(df["has_multisource_fit"].mean()),
    }


def _raw_pair_support_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "raw_pair_support_rate": float(df["has_raw_pair_support"].mean()),
        "raw_pair_support_strong_rate": float(df["has_strong_raw_pair_support"].mean()),
        "raw_preference_rate": float(df["raw_preference_support"].mean()),
        "raw_recent_rate": float(df["raw_recent_support"].mean()),
        "raw_context_rate": float(df["raw_context_support"].mean()),
        "raw_property_rate": float(df["raw_property_support"].mean()),
        "raw_positive_evidence_rate": float(df["raw_positive_evidence_support"].mean()),
    }


def _suspect_positive_avoid(text: Any) -> bool:
    raw = _to_str(text).lower()
    if not raw:
        return False
    return any(term in raw for term in _SUSPECT_POSITIVE_AVOID_PHRASES)


def _norm_term(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_term_blob(value: Any) -> list[str]:
    raw = _to_str(value)
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return [_norm_term(item) for item in items if _norm_term(item)]
        except Exception:
            pass
    if "|" in raw:
        return [_norm_term(part) for part in raw.split("|") if _norm_term(part)]
    return [_norm_term(part) for part in raw.split(",") if _norm_term(part)]


def _alias_terms_match(left_terms: list[str], right_terms: list[str]) -> list[str]:
    left = {_norm_term(term) for term in left_terms if _norm_term(term)}
    right = {_norm_term(term) for term in right_terms if _norm_term(term)}
    out: list[str] = []
    for canonical, aliases in _RECENT_CONTEXT_ALIAS_GROUPS.items():
        alias_norms = {_norm_term(canonical)} | {_norm_term(term) for term in aliases if _norm_term(term)}
        if left.intersection(alias_norms) and right.intersection(alias_norms):
            out.append(canonical)
    return out


def _exact_context_terms(left_terms: list[str], right_terms: list[str]) -> list[str]:
    left = {_norm_term(term) for term in left_terms if _norm_term(term)}
    right = {_norm_term(term) for term in right_terms if _norm_term(term)}
    out = sorted(
        term
        for term in left.intersection(right)
        if term not in _GENERIC_SCENE_TERMS
    )
    return out


def _fit_scope_signature(row: pd.Series) -> str:
    scopes: list[str] = []
    if _to_int(row.get("pair_stable_fit_fact_count_v1")) > 0:
        scopes.append("stable")
    if _to_int(row.get("pair_recent_fit_fact_count_v1")) > 0:
        scopes.append("recent")
    if _to_int(row.get("pair_history_fit_fact_count_v1")) > 0:
        scopes.append("history")
    if _to_int(row.get("pair_practical_fit_fact_count_v1")) > 0:
        scopes.append("practical")
    if not scopes:
        return "none"
    return "+".join(scopes)


def _count_true(df: pd.DataFrame, col: str) -> int:
    if df.empty:
        return 0
    if col not in df.columns:
        return 0
    series = df[col]
    if series.dtype == bool:
        return int(series.sum())
    return int(pd.to_numeric(series, errors="coerce").fillna(0).astype(int).gt(0).sum())


def main() -> None:
    args = parse_args()
    stage09_run_dir = require_path(args.stage09_run_dir)
    source_run_dir = require_path(args.source_run_dir) if args.source_run_dir else None
    export_run_dir = require_path(args.export_run_dir) if args.export_run_dir else None
    target_band = _canonical_target_band(args.target_band)

    target_users = _read_parquet_from_runs(stage09_run_dir, source_run_dir, "stage11_target_users_v1.parquet")
    user_assets = _read_parquet_from_runs(stage09_run_dir, source_run_dir, "user_semantic_profile_text_v2.parquet")
    pair_universe = _read_parquet_from_runs(stage09_run_dir, source_run_dir, "stage11_target_pair_universe_v1.parquet")
    pair_assets = _read_parquet_from_runs(stage09_run_dir, source_run_dir, "user_business_alignment_text_v1.parquet")
    user_readiness_path = stage09_run_dir / "stage11_target_user_readiness_v1.parquet"
    if user_readiness_path.exists():
        user_readiness = pd.read_parquet(user_readiness_path)
    elif source_run_dir is not None and (source_run_dir / "stage11_target_user_readiness_v1.parquet").exists():
        user_readiness = pd.read_parquet(source_run_dir / "stage11_target_user_readiness_v1.parquet")
    else:
        user_readiness = pd.DataFrame()

    selected_users = target_users.loc[
        pd.to_numeric(target_users["truth_learned_rank"], errors="coerce").map(_band).eq(target_band)
    ].copy()
    if "true_item_idx" not in selected_users.columns:
        truth_item_lookup = (
            pair_universe[["bucket", "user_idx", "user_id", "true_item_idx"]]
            .drop_duplicates(subset=["bucket", "user_idx", "user_id"])
            .copy()
        )
        selected_users = selected_users.merge(
            truth_item_lookup,
            on=["bucket", "user_idx", "user_id"],
            how="left",
        )
    selected_users["user_id"] = selected_users["user_id"].astype(str)
    selected_users["user_idx"] = pd.to_numeric(selected_users["user_idx"], errors="coerce").astype("Int64")
    selected_users["bucket"] = pd.to_numeric(selected_users["bucket"], errors="coerce").astype("Int64")
    selected_users["true_item_idx"] = pd.to_numeric(selected_users["true_item_idx"], errors="coerce").astype("Int64")

    user_join_cols = [
        "bucket",
        "user_idx",
        "user_id",
        "user_profile_richness_v2",
        "user_profile_richness_tier_v2",
        "user_quality_signal_count_v2",
        "user_has_avoid_signal_v2",
        "user_has_clean_evidence_v2",
        "user_has_specific_focus_v2",
        "user_has_specific_recent_v2",
        "user_has_recent_text_v2",
        "user_has_strong_clean_evidence_v2",
        "user_has_history_evidence_v2",
        "user_visible_fact_count_v1",
        "history_anchor_source_v2",
        "stable_preferences_text",
        "recent_intent_text_v2",
        "avoidance_text_v2",
        "history_anchor_hint_text",
        "user_evidence_text_v2",
        "user_semantic_profile_text_v2",
    ]
    user_join = user_assets[[c for c in user_join_cols if c in user_assets.columns]].drop_duplicates(
        subset=["bucket", "user_idx", "user_id"]
    )
    if not user_readiness.empty:
        readiness_cols = [c for c in ["bucket", "user_idx", "user_id", "semantic_user_readiness_tier_v1"] if c in user_readiness.columns]
        user_join = user_join.merge(
            user_readiness[readiness_cols].drop_duplicates(subset=["bucket", "user_idx", "user_id"]),
            on=["bucket", "user_idx", "user_id"],
            how="left",
        )

    selected_users = selected_users.merge(user_join, on=["bucket", "user_idx", "user_id"], how="left")
    selected_users["light_mid_heavy"] = selected_users.apply(_bucket_user_richness, axis=1)

    chosen_pairs = selected_users.merge(
        pair_assets,
        left_on=["bucket", "user_idx", "user_id", "true_item_idx"],
        right_on=["bucket", "user_idx", "user_id", "item_idx"],
        how="left",
    )

    chosen_pairs["has_focus"] = _bool_col(chosen_pairs, "user_has_specific_focus_v2")
    chosen_pairs["has_recent_or_history"] = _bool_col(chosen_pairs, "user_has_specific_recent_v2") | _bool_col(chosen_pairs, "user_has_history_evidence_v2")
    chosen_pairs["has_avoid"] = _bool_col(chosen_pairs, "user_has_avoid_signal_v2")
    chosen_pairs["has_user_evidence"] = _bool_col(chosen_pairs, "user_has_strong_clean_evidence_v2") | _bool_col(chosen_pairs, "user_has_clean_evidence_v2")
    chosen_pairs["has_fit"] = _bool_col(chosen_pairs, "pair_has_visible_user_fit_v1") | _num_col(chosen_pairs, "pair_fit_fact_count_v1").astype(int).gt(0)
    chosen_pairs["has_conflict"] = _bool_col(chosen_pairs, "pair_has_visible_conflict_v1") | _num_col(chosen_pairs, "pair_friction_fact_count_v1").astype(int).gt(0)
    chosen_pairs["has_pair_evidence"] = _num_col(chosen_pairs, "pair_evidence_fact_count_v1").astype(int).gt(0)
    chosen_pairs["has_multisource_fit"] = _bool_col(chosen_pairs, "pair_has_multisource_fit_v1") | _num_col(chosen_pairs, "pair_fit_scope_count_v1").astype(int).ge(2)
    chosen_pairs["has_contrastive_support"] = _bool_col(chosen_pairs, "pair_has_contrastive_support_v1")
    chosen_pairs["has_detail_support"] = _bool_col(chosen_pairs, "pair_has_detail_support_v1")

    pair_universe["bucket"] = pd.to_numeric(pair_universe["bucket"], errors="coerce").astype("Int64")
    pair_universe["user_idx"] = pd.to_numeric(pair_universe["user_idx"], errors="coerce").astype("Int64")
    pair_universe["item_idx"] = pd.to_numeric(pair_universe["item_idx"], errors="coerce").astype("Int64")
    pair_universe["learned_rank_band"] = pair_universe.get("learned_rank", pd.Series(index=pair_universe.index)).map(_band)

    rivals = selected_users[["bucket", "user_idx", "user_id", "true_item_idx"]].merge(
        pair_universe[["bucket", "user_idx", "user_id", "item_idx", "learned_rank_band"]],
        on=["bucket", "user_idx", "user_id"],
        how="left",
    )
    rivals = rivals.loc[rivals["item_idx"] != rivals["true_item_idx"]].copy()
    rival_stats = (
        rivals.groupby(["bucket", "user_idx", "user_id"], dropna=False)
        .agg(
            rival_total=("item_idx", "count"),
            rival_head=("learned_rank_band", lambda s: int((s == "head_guard").sum())),
            rival_boundary=("learned_rank_band", lambda s: int((s == "boundary_11_30").sum())),
            rival_mid=("learned_rank_band", lambda s: int((s == "rescue_31_60").sum())),
            rival_deep=("learned_rank_band", lambda s: int((s == "rescue_61_100").sum())),
        )
        .reset_index()
    )
    rival_stats["rival_head_or_boundary"] = rival_stats["rival_head"] + rival_stats["rival_boundary"]
    chosen_pairs = chosen_pairs.merge(rival_stats, on=["bucket", "user_idx", "user_id"], how="left")
    for col in ["rival_total", "rival_head", "rival_boundary", "rival_mid", "rival_deep", "rival_head_or_boundary"]:
        chosen_pairs[col] = _num_col(chosen_pairs, col).astype(int)

    constructability = chosen_pairs.apply(_classify_constructability, axis=1, result_type="expand")
    chosen_pairs["constructability_class"] = constructability[0]
    chosen_pairs["reason_codes"] = constructability[1]
    chosen_pairs["first_issue_code"] = chosen_pairs["reason_codes"].map(lambda xs: xs[0] if xs else "PASS")
    chosen_pairs["failure_primary_reason_code"] = chosen_pairs.apply(
        lambda row: row["reason_codes"][0]
        if row["constructability_class"] in {"C0_FAIL", "C1_WEAK"} and row["reason_codes"]
        else "PASS",
        axis=1,
    )
    # Backward-compatible alias. Historically this field meant "first issue seen", not strictly "failure primary reason".
    chosen_pairs["primary_reason_code"] = chosen_pairs["first_issue_code"]

    summary: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "stage09_run_dir": str(stage09_run_dir),
        "target_band": target_band,
        "target_user_total": int(len(selected_users)),
        "light_mid_heavy_counts": _summarize_counts(chosen_pairs["light_mid_heavy"]),
        "constructability_counts": _summarize_counts(chosen_pairs["constructability_class"]),
        "constructability_by_bucket": {},
        "first_issue_counts": _summarize_counts(chosen_pairs["first_issue_code"]),
        "primary_reason_counts": _summarize_counts(chosen_pairs["primary_reason_code"]),
        "failure_primary_reason_counts": _summarize_counts(chosen_pairs["failure_primary_reason_code"]),
        "reason_code_counts": {},
    }

    for bucket_name, bucket_df in chosen_pairs.groupby("light_mid_heavy", dropna=False):
        bucket_key = str(bucket_name)
        summary["constructability_by_bucket"][bucket_key] = {
            "user_count": int(len(bucket_df)),
            "constructability_counts": _summarize_counts(bucket_df["constructability_class"]),
            "first_issue_counts": _summarize_counts(bucket_df["first_issue_code"]),
            "primary_reason_counts": _summarize_counts(bucket_df["primary_reason_code"]),
            "failure_primary_reason_counts": _summarize_counts(bucket_df["failure_primary_reason_code"]),
        }

    all_reason_counts: dict[str, int] = {}
    reason_counts_by_bucket: dict[str, dict[str, int]] = {}
    for bucket_name, bucket_df in chosen_pairs.groupby("light_mid_heavy", dropna=False):
        bucket_counter: dict[str, int] = {}
        for codes in bucket_df["reason_codes"]:
            for code in codes:
                bucket_counter[code] = bucket_counter.get(code, 0) + 1
                all_reason_counts[code] = all_reason_counts.get(code, 0) + 1
        reason_counts_by_bucket[str(bucket_name)] = dict(sorted(bucket_counter.items(), key=lambda kv: (-kv[1], kv[0])))
    summary["reason_code_counts"] = dict(sorted(all_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    summary["reason_code_counts_by_bucket"] = reason_counts_by_bucket

    summary["key_signal_rates"] = {
        "has_avoid_rate": float(chosen_pairs["has_avoid"].mean()),
        "has_user_evidence_rate": float(chosen_pairs["has_user_evidence"].mean()),
        "has_fit_rate": float(chosen_pairs["has_fit"].mean()),
        "has_conflict_rate": float(chosen_pairs["has_conflict"].mean()),
        "has_pair_evidence_rate": float(chosen_pairs["has_pair_evidence"].mean()),
        "has_multisource_fit_rate": float(chosen_pairs["has_multisource_fit"].mean()),
        "has_detail_support_rate": float(chosen_pairs["has_detail_support"].mean()),
        "rival_ge_3_rate": float(chosen_pairs["rival_total"].ge(3).mean()),
        "rival_head_or_boundary_ge_1_rate": float(chosen_pairs["rival_head_or_boundary"].ge(1).mean()),
    }

    examples: dict[str, list[dict[str, Any]]] = {}
    for bucket_name in ["heavy", "mid", "light"]:
        bucket_df = chosen_pairs.loc[chosen_pairs["light_mid_heavy"] == bucket_name].copy()
        if bucket_df.empty:
            continue
        for cls_name in ["C0_FAIL", "C1_WEAK", "C2_USABLE", "C3_IDEAL"]:
            sample_df = bucket_df.loc[bucket_df["constructability_class"] == cls_name].copy()
            if sample_df.empty:
                continue
            sample_df["_sort_pair_fit_fact_count_v1"] = _num_col(sample_df, "pair_fit_fact_count_v1")
            sample_df["_sort_pair_evidence_fact_count_v1"] = _num_col(sample_df, "pair_evidence_fact_count_v1")
            sample_df = sample_df.sort_values(
                by=["rival_total", "_sort_pair_fit_fact_count_v1", "_sort_pair_evidence_fact_count_v1"],
                ascending=[True, False, False],
            ).head(3)
            example_cols = [
                "user_id",
                "user_idx",
                "truth_learned_rank",
                "semantic_user_readiness_tier_v1",
                "light_mid_heavy",
                "constructability_class",
                "primary_reason_code",
                "reason_codes",
                "rival_total",
                "rival_head",
                "rival_boundary",
                "rival_mid",
                "rival_deep",
                "user_profile_richness_tier_v2",
                "user_quality_signal_count_v2",
                "user_visible_fact_count_v1",
                "stable_preferences_text",
                "recent_intent_text_v2",
                "avoidance_text_v2",
                "history_anchor_hint_text",
                "user_evidence_text_v2",
                "fit_reasons_text_v1",
                "friction_reasons_text_v1",
                "evidence_basis_text_v1",
                "pair_fit_fact_count_v1",
                "pair_friction_fact_count_v1",
                "pair_evidence_fact_count_v1",
                "pair_fit_scope_count_v1",
                "pair_has_detail_support_v1",
                "pair_has_multisource_fit_v1",
            ]
            available_example_cols = [col for col in example_cols if col in sample_df.columns]
            examples[f"{bucket_name}:{cls_name}"] = sample_df[available_example_cols].to_dict(orient="records")

    bucket_x_constructability: dict[str, dict[str, int]] = {}
    first_issue_by_bucket: dict[str, dict[str, int]] = {}
    primary_reason_by_bucket: dict[str, dict[str, int]] = {}
    failure_primary_reason_by_bucket: dict[str, dict[str, int]] = {}
    for bucket_name, bucket_df in chosen_pairs.groupby("light_mid_heavy", dropna=False):
        bucket_key = str(bucket_name)
        bucket_x_constructability[bucket_key] = _summarize_counts(bucket_df["constructability_class"])
        first_issue_by_bucket[bucket_key] = _summarize_counts(bucket_df["first_issue_code"])
        primary_reason_by_bucket[bucket_key] = _summarize_counts(bucket_df["primary_reason_code"])
        failure_primary_reason_by_bucket[bucket_key] = _summarize_counts(bucket_df["failure_primary_reason_code"])

    payload: dict[str, Any] = {
        "summary": summary,
        "bucket_x_constructability": bucket_x_constructability,
        "first_issue_by_bucket": first_issue_by_bucket,
        "primary_reason_by_bucket": primary_reason_by_bucket,
        "failure_primary_reason_by_bucket": failure_primary_reason_by_bucket,
        "examples": examples,
    }

    if source_run_dir is not None:
        source_user_path = source_run_dir / "user_source_semantic_materials_v1.parquet"
        if source_user_path.exists():
            source_user = pd.read_parquet(source_user_path)
            source_user["bucket"] = pd.to_numeric(source_user["bucket"], errors="coerce").astype("Int64")
            source_user["user_idx"] = pd.to_numeric(source_user["user_idx"], errors="coerce").astype("Int64")
            source_join_cols = [
                "bucket",
                "user_idx",
                "user_id",
                "user_source_positive_sentence_count_v1",
                "user_source_negative_sentence_count_v1",
                "user_source_recent_sentence_count_v1",
                "user_source_tip_sentence_count_v1",
                "user_source_history_anchor_count_v1",
            ]
            chosen_pairs = chosen_pairs.merge(
                source_user[source_join_cols].drop_duplicates(subset=["bucket", "user_idx", "user_id"]),
                on=["bucket", "user_idx", "user_id"],
                how="left",
            )
            for col in source_join_cols[3:]:
                chosen_pairs[col] = pd.to_numeric(chosen_pairs[col], errors="coerce").fillna(0).astype(int)

            heavy_c0 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C0_FAIL")
            ].copy()
            heavy_c1 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C1_WEAK")
            ].copy()
            mid_fail = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "mid")
                & (chosen_pairs["constructability_class"].isin(["C0_FAIL", "C1_WEAK"]))
            ].copy()
            mid_no_avoid = mid_fail.loc[mid_fail["reason_codes"].map(lambda xs: "U_NO_AVOID" in xs)].copy()
            mid_no_avoid_with_source_negative = mid_no_avoid.loc[
                mid_no_avoid["user_source_negative_sentence_count_v1"].gt(0)
            ].copy()
            mid_no_user_evidence = mid_fail.loc[
                mid_fail["reason_codes"].map(lambda xs: "U_NO_USER_EVIDENCE" in xs)
            ].copy()
            mid_fit_missing = mid_fail.loc[
                mid_fail["reason_codes"].map(lambda xs: "P_NO_VISIBLE_FIT" in xs)
            ].copy()

            payload["source_support_audit"] = {
                "source_run_dir": str(source_run_dir),
                "heavy_c0_support": _source_support_summary(heavy_c0),
                "heavy_c1_support": _source_support_summary(heavy_c1),
                "mid_fail_support": _source_support_summary(mid_fail),
                "mid_no_avoid_support": _source_support_summary(mid_no_avoid),
                "mid_no_avoid_with_source_negative": _source_support_summary(mid_no_avoid_with_source_negative),
                "mid_no_user_evidence_support": _source_support_summary(mid_no_user_evidence),
                "mid_fit_missing_support": _source_support_summary(mid_fit_missing),
                "source_only_recoverable_upper_bound": {
                    "heavy_c0_fit_missing": int(len(heavy_c0)),
                    "mid_no_avoid_with_source_negative": int(len(mid_no_avoid_with_source_negative)),
                    "mid_fit_missing": int(len(mid_fit_missing)),
                    "total": int(len(heavy_c0) + len(mid_no_avoid_with_source_negative) + len(mid_fit_missing)),
                    "fail_total_mid_heavy": int(len(mid_fail) + len(heavy_c0) + len(heavy_c1)),
                },
            }
        else:
            payload["source_support_audit"] = {
                "source_run_dir": str(source_run_dir),
                "available": False,
                "missing_file": str(source_user_path),
            }

        merchant_assets = _read_parquet_from_runs(
            stage09_run_dir, source_run_dir, "merchant_semantic_profile_text_v2.parquet"
        )
        merchant_join_cols = [
            "bucket",
            "business_id",
            "merchant_core_terms_v2",
            "merchant_dish_terms_v2",
            "merchant_scene_terms_v2",
            "merchant_time_terms_v2",
            "merchant_property_terms_v2",
            "merchant_strength_terms_v2",
            "merchant_risk_terms_v2",
            "merchant_semantic_profile_text_v2",
            "core_offering_text",
            "scene_fit_text",
            "strengths_text",
            "risk_points_text",
        ]
        user_fact_cols = [
            "bucket",
            "user_idx",
            "user_id",
            "user_preference_core_terms_v2",
            "user_dish_terms_v2",
            "user_beverage_terms_v2",
            "user_recent_semantic_terms_v2",
            "user_recent_context_terms_v2",
            "user_source_recent_terms_v2",
            "user_source_tip_terms_v2",
            "user_source_history_terms_v2",
            "user_avoid_terms_v2",
        ]
        recoverability_pairs = (
            chosen_pairs.merge(
                user_assets[[c for c in user_fact_cols if c in user_assets.columns]].drop_duplicates(
                    subset=["bucket", "user_idx", "user_id"]
                ),
                on=["bucket", "user_idx", "user_id"],
                how="left",
            ).merge(
                merchant_assets[[c for c in merchant_join_cols if c in merchant_assets.columns]].drop_duplicates(
                    subset=["bucket", "business_id"]
                ),
                on=["bucket", "business_id"],
                how="left",
            )
        )

        def _visible_overlap_count(df: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
            count = 0
            examples: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                stable_text = " ".join(
                    part
                    for part in [
                        _to_str(row.get("stable_preferences_text")).lower(),
                        _to_str(row.get("user_evidence_text_v2")).lower(),
                    ]
                    if part
                )
                recent_text = " ".join(
                    part
                    for part in [
                        _to_str(row.get("recent_intent_text_v2")).lower(),
                        _to_str(row.get("user_evidence_text_v2")).lower(),
                    ]
                    if part
                )
                history_text = " ".join(
                    part
                    for part in [
                        _to_str(row.get("history_anchor_hint_text")).lower(),
                        _to_str(row.get("user_evidence_text_v2")).lower(),
                    ]
                    if part
                )
                avoid_text = _to_str(row.get("avoidance_text_v2")).lower()
                merchant_text = " ".join(
                    part
                    for part in [
                        _to_str(row.get("merchant_semantic_profile_text_v2")).lower(),
                        _to_str(row.get("core_offering_text")).lower(),
                        _to_str(row.get("scene_fit_text")).lower(),
                        _to_str(row.get("strengths_text")).lower(),
                    ]
                    if part
                )
                risk_text = " ".join(
                    part
                    for part in [
                        _to_str(row.get("risk_points_text")).lower(),
                        _to_str(row.get("merchant_semantic_profile_text_v2")).lower(),
                    ]
                    if part
                )
                stable_terms = {
                    term
                    for term in _parse_term_blob(row.get("user_preference_core_terms_v2"))
                    + _parse_term_blob(row.get("user_dish_terms_v2"))
                    + _parse_term_blob(row.get("user_beverage_terms_v2"))
                    if term
                }
                recent_terms = {
                    term
                    for term in _parse_term_blob(row.get("user_recent_semantic_terms_v2"))
                    + _parse_term_blob(row.get("user_recent_context_terms_v2"))
                    + _parse_term_blob(row.get("user_source_recent_terms_v2"))
                    + _parse_term_blob(row.get("user_source_tip_terms_v2"))
                    if term
                }
                history_terms = {
                    term for term in _parse_term_blob(row.get("user_source_history_terms_v2")) if term
                }
                avoid_terms = {term for term in _parse_term_blob(row.get("user_avoid_terms_v2")) if term}
                merchant_terms = {
                    term
                    for term in _parse_term_blob(row.get("merchant_core_terms_v2"))
                    + _parse_term_blob(row.get("merchant_dish_terms_v2"))
                    + _parse_term_blob(row.get("merchant_scene_terms_v2"))
                    + _parse_term_blob(row.get("merchant_time_terms_v2"))
                    + _parse_term_blob(row.get("merchant_property_terms_v2"))
                    + _parse_term_blob(row.get("merchant_strength_terms_v2"))
                    if term
                }
                risk_terms = {term for term in _parse_term_blob(row.get("merchant_risk_terms_v2")) if term}
                stable_overlap = sorted(
                    [
                        term
                        for term in stable_terms
                        if term in stable_text and term in merchant_text and term in merchant_terms
                    ]
                )
                recent_overlap = sorted(
                    [
                        term
                        for term in recent_terms
                        if term in recent_text and term in merchant_text and term in merchant_terms
                    ]
                )
                history_overlap = sorted(
                    [
                        term
                        for term in history_terms
                        if term in history_text and term in merchant_text and term in merchant_terms
                    ]
                )
                avoid_overlap = sorted(
                    [
                        term
                        for term in avoid_terms
                        if term in avoid_text and term in risk_text and term in risk_terms
                    ]
                )
                if stable_overlap or recent_overlap or history_overlap or avoid_overlap:
                    count += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "user_id": _to_str(row.get("user_id")),
                                "stable_overlap": stable_overlap[:4],
                                "recent_overlap": recent_overlap[:4],
                                "history_overlap": history_overlap[:4],
                                "avoid_overlap": avoid_overlap[:4],
                            }
                        )
            return count, examples

        heavy_c0_visible_count, heavy_c0_examples = _visible_overlap_count(heavy_c0)

        def _context_alias_frame(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["recent_context_exact_terms"] = out.apply(
                lambda row: _exact_context_terms(
                    _parse_term_blob(row.get("user_recent_context_terms_v2"))
                    + _parse_term_blob(row.get("user_scene_terms_v2"))
                    + _parse_term_blob(row.get("user_source_recent_terms_v2"))
                    + _parse_term_blob(row.get("user_source_tip_terms_v2")),
                    _parse_term_blob(row.get("merchant_scene_terms_v2"))
                    + _parse_term_blob(row.get("merchant_time_terms_v2"))
                    + _parse_term_blob(row.get("merchant_property_terms_v2")),
                ),
                axis=1,
            )
            out["recent_context_alias_terms"] = out.apply(
                lambda row: _alias_terms_match(
                    _parse_term_blob(row.get("user_recent_context_terms_v2"))
                    + _parse_term_blob(row.get("user_scene_terms_v2"))
                    + _parse_term_blob(row.get("user_source_recent_terms_v2"))
                    + _parse_term_blob(row.get("user_source_tip_terms_v2")),
                    _parse_term_blob(row.get("merchant_scene_terms_v2"))
                    + _parse_term_blob(row.get("merchant_time_terms_v2"))
                    + _parse_term_blob(row.get("merchant_property_terms_v2")),
                ),
                axis=1,
            )
            return out

        recoverability_pairs = _context_alias_frame(recoverability_pairs)
        recoverability_pairs["has_recent_context_exact_bridge"] = recoverability_pairs[
            "recent_context_exact_terms"
        ].map(lambda xs: len(xs) > 0)
        recoverability_pairs["has_recent_context_alias_bridge"] = recoverability_pairs[
            "recent_context_alias_terms"
        ].map(lambda xs: len(xs) > 0)
        recoverability_pairs["has_recent_context_visible_bridge"] = (
            recoverability_pairs["has_recent_context_exact_bridge"]
            | recoverability_pairs["has_recent_context_alias_bridge"]
        )
        chosen_pairs = chosen_pairs.merge(
            recoverability_pairs[
                [
                    "bucket",
                    "user_idx",
                    "user_id",
                    "item_idx",
                    "recent_context_exact_terms",
                    "recent_context_alias_terms",
                    "has_recent_context_exact_bridge",
                    "has_recent_context_alias_bridge",
                    "has_recent_context_visible_bridge",
                ]
            ].drop_duplicates(subset=["bucket", "user_idx", "user_id", "item_idx"]),
            on=["bucket", "user_idx", "user_id", "item_idx"],
            how="left",
        )
        chosen_pairs["has_recent_context_exact_bridge"] = _bool_col(
            chosen_pairs, "has_recent_context_exact_bridge"
        )
        chosen_pairs["has_recent_context_alias_bridge"] = _bool_col(
            chosen_pairs, "has_recent_context_alias_bridge"
        )
        chosen_pairs["has_recent_context_visible_bridge"] = _bool_col(
            chosen_pairs, "has_recent_context_visible_bridge"
        )

        heavy_c0_alias_matches = recoverability_pairs.loc[
            (recoverability_pairs["light_mid_heavy"] == "heavy")
            & (recoverability_pairs["constructability_class"] == "C0_FAIL")
        ].copy()
        heavy_c1_alias_matches = recoverability_pairs.loc[
            (recoverability_pairs["light_mid_heavy"] == "heavy")
            & (recoverability_pairs["constructability_class"] == "C1_WEAK")
        ].copy()
        heavy_c0_exact_count = int(
            heavy_c0_alias_matches["recent_context_exact_terms"].map(lambda xs: len(xs) > 0).sum()
        )
        heavy_c0_alias_count = int(
            heavy_c0_alias_matches["recent_context_alias_terms"].map(lambda xs: len(xs) > 0).sum()
        )
        heavy_c0_alias_example_cols = [
            c
            for c in [
                "user_id",
                "business_id",
                "user_recent_context_terms_v2",
                "user_scene_terms_v2",
                "user_source_recent_terms_v2",
                "merchant_scene_terms_v2",
                "merchant_time_terms_v2",
                "merchant_property_terms_v2",
                "recent_context_exact_terms",
                "recent_context_alias_terms",
                "fit_reasons_text_v1",
            ]
            if c in heavy_c0_alias_matches.columns
        ]
        heavy_c0_alias_examples = heavy_c0_alias_matches.loc[
            heavy_c0_alias_matches["recent_context_alias_terms"].map(lambda xs: len(xs) > 0)
        ][heavy_c0_alias_example_cols].head(8).to_dict(orient="records")
        heavy_c1_alias_matches["recent_context_alias_terms"] = heavy_c1_alias_matches["recent_context_alias_terms"]
        heavy_c1_exact_count = int(
            heavy_c1_alias_matches["recent_context_exact_terms"].map(lambda xs: len(xs) > 0).sum()
        )
        heavy_c1_alias_count = int(
            heavy_c1_alias_matches["recent_context_alias_terms"].map(lambda xs: len(xs) > 0).sum()
        )
        heavy_c1_alias_example_cols = [
            c
            for c in [
                "user_id",
                "business_id",
                "user_recent_context_terms_v2",
                "merchant_scene_terms_v2",
                "merchant_time_terms_v2",
                "recent_context_exact_terms",
                "recent_context_alias_terms",
                "fit_reasons_text_v1",
            ]
            if c in heavy_c1_alias_matches.columns
        ]
        heavy_c1_alias_examples = heavy_c1_alias_matches.loc[
            heavy_c1_alias_matches["recent_context_alias_terms"].map(lambda xs: len(xs) > 0)
        ][heavy_c1_alias_example_cols].head(8).to_dict(orient="records")

        payload["recoverability_audit"] = {
            "heavy_c0_visible_overlap": {
                "n": int(len(heavy_c0)),
                "visible_overlap_count": int(heavy_c0_visible_count),
                "visible_overlap_rate": float(heavy_c0_visible_count / len(heavy_c0)) if len(heavy_c0) else 0.0,
                "examples": heavy_c0_examples,
            },
            "heavy_c0_context_alias": {
                "n": int(len(heavy_c0)),
                "exact_match_count": int(heavy_c0_exact_count),
                "alias_match_count": int(heavy_c0_alias_count),
                "alias_match_rate": float(heavy_c0_alias_count / len(heavy_c0)) if len(heavy_c0) else 0.0,
                "examples": heavy_c0_alias_examples,
            },
            "heavy_c1_context_alias": {
                "n": int(len(heavy_c1)),
                "exact_match_count": int(heavy_c1_exact_count),
                "alias_match_count": int(heavy_c1_alias_count),
                "alias_match_rate": float(heavy_c1_alias_count / len(heavy_c1)) if len(heavy_c1) else 0.0,
                "examples": heavy_c1_alias_examples,
            },
            "mid_no_avoid_source_negative": {
                "n": int(len(mid_no_avoid)),
                "with_source_negative_count": int(len(mid_no_avoid_with_source_negative)),
                "with_source_negative_rate": float(len(mid_no_avoid_with_source_negative) / len(mid_no_avoid)) if len(mid_no_avoid) else 0.0,
            },
        }

    run_meta_path = stage09_run_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            run_meta = {}
    else:
        run_meta = {}

    match_feature_run = _to_str(run_meta.get("match_feature_run"))
    match_channel_run = _to_str(run_meta.get("match_channel_run"))
    candidate_text_match_run = _to_str(run_meta.get("candidate_text_match_run"))
    if match_feature_run and match_channel_run:
        feature_path = Path(match_feature_run) / "user_business_match_features_v2_user_item.parquet"
        channel_path = Path(match_channel_run) / "user_business_match_channels_v2_user_item.parquet"
        if feature_path.exists() and channel_path.exists():
            feature_df = pd.read_parquet(feature_path)
            channel_df = pd.read_parquet(channel_path)
            for frame in [feature_df, channel_df]:
                frame["user_id"] = frame["user_id"].astype(str)
                frame["business_id"] = frame["business_id"].astype(str)
            chosen_pairs["user_id"] = chosen_pairs["user_id"].astype(str)
            chosen_pairs["business_id"] = chosen_pairs["business_id"].astype(str)
            chosen_pairs = chosen_pairs.merge(
                feature_df[
                    [
                        "user_id",
                        "business_id",
                        "mean_match_total_v1",
                        "mean_match_positive_evidence",
                        "mean_match_recent_cuisine",
                        "mean_match_negative_conflict",
                    ]
                ].drop_duplicates(subset=["user_id", "business_id"]),
                on=["user_id", "business_id"],
                how="left",
            )
            chosen_pairs = chosen_pairs.merge(
                channel_df[
                    [
                        "user_id",
                        "business_id",
                        "mean_channel_preference_core_v1",
                        "mean_channel_recent_intent_v1",
                        "mean_channel_evidence_support_v1",
                        "mean_channel_conflict_v1",
                        "mean_channel_context_time_v1",
                        "mean_channel_preference_property_v1",
                    ]
                ].drop_duplicates(subset=["user_id", "business_id"]),
                on=["user_id", "business_id"],
                how="left",
            )
            chosen_pairs["raw_preference_support"] = (
                _num_col(chosen_pairs, "mean_match_total_v1").ge(0.33)
                | _num_col(chosen_pairs, "mean_channel_preference_core_v1").ge(0.33)
            )
            chosen_pairs["raw_recent_support"] = (
                _num_col(chosen_pairs, "mean_channel_recent_intent_v1").ge(0.08)
                | _num_col(chosen_pairs, "mean_match_recent_cuisine").ge(0.08)
            )
            chosen_pairs["raw_context_support"] = (
                _num_col(chosen_pairs, "mean_channel_context_time_v1").ge(0.40)
            )
            chosen_pairs["raw_property_support"] = (
                _num_col(chosen_pairs, "mean_channel_preference_property_v1").ge(0.40)
            )
            chosen_pairs["raw_positive_evidence_support"] = (
                _num_col(chosen_pairs, "mean_match_positive_evidence").ge(0.00003)
                | _num_col(chosen_pairs, "mean_channel_evidence_support_v1").ge(0.00003)
            )
            chosen_pairs["has_raw_pair_support"] = (
                chosen_pairs["raw_preference_support"]
                | chosen_pairs["raw_recent_support"]
                | chosen_pairs["raw_context_support"]
                | chosen_pairs["raw_property_support"]
            )
            chosen_pairs["has_strong_raw_pair_support"] = (
                chosen_pairs["raw_preference_support"].astype(int)
                + chosen_pairs["raw_recent_support"].astype(int)
                + chosen_pairs["raw_context_support"].astype(int)
                + chosen_pairs["raw_property_support"].astype(int)
            ).ge(2)

            heavy_c0 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C0_FAIL")
            ].copy()
            heavy_c1 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C1_WEAK")
            ].copy()
            heavy_c3 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C3_IDEAL")
            ].copy()
            mid_fail = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "mid")
                & (chosen_pairs["constructability_class"].isin(["C0_FAIL", "C1_WEAK"]))
            ].copy()
            mid_c2 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "mid")
                & (chosen_pairs["constructability_class"] == "C2_USABLE")
            ].copy()

            payload["raw_pair_support_audit"] = {
                "match_feature_run": match_feature_run,
                "match_channel_run": match_channel_run,
                "heavy_c0": _raw_pair_support_summary(heavy_c0),
                "heavy_c1": _raw_pair_support_summary(heavy_c1),
                "heavy_c3": _raw_pair_support_summary(heavy_c3),
                "mid_fail": _raw_pair_support_summary(mid_fail),
                "mid_c2": _raw_pair_support_summary(mid_c2),
            }

    if candidate_text_match_run:
        candidate_text_path = Path(candidate_text_match_run) / "candidate_text_match_features_v1.parquet"
        if candidate_text_path.exists():
            candidate_text_df = pd.read_parquet(
                candidate_text_path,
                columns=[
                    "user_idx",
                    "item_idx",
                    "sim_long_pref_core",
                    "sim_recent_intent_semantic",
                    "sim_recent_intent_pos",
                    "sim_negative_avoid_neg",
                    "sim_negative_avoid_core",
                    "sim_context_merchant",
                    "sim_conflict_gap",
                ],
            )
            candidate_text_df["user_idx"] = pd.to_numeric(candidate_text_df["user_idx"], errors="coerce").astype("Int64")
            candidate_text_df["item_idx"] = pd.to_numeric(candidate_text_df["item_idx"], errors="coerce").astype("Int64")
            chosen_pairs["user_idx"] = pd.to_numeric(chosen_pairs["user_idx"], errors="coerce").astype("Int64")
            chosen_pairs["item_idx"] = pd.to_numeric(chosen_pairs["item_idx"], errors="coerce").astype("Int64")
            chosen_pairs = chosen_pairs.merge(
                candidate_text_df.drop_duplicates(subset=["user_idx", "item_idx"]),
                on=["user_idx", "item_idx"],
                how="left",
            )
            chosen_pairs["candidate_pref_support"] = (
                _num_col(chosen_pairs, "sim_long_pref_core").ge(0.18)
            )
            chosen_pairs["candidate_recent_support"] = (
                _num_col(chosen_pairs, "sim_recent_intent_semantic").ge(0.12)
                | _num_col(chosen_pairs, "sim_recent_intent_pos").ge(0.08)
            )
            chosen_pairs["candidate_context_support"] = (
                _num_col(chosen_pairs, "sim_context_merchant").ge(0.14)
            )
            chosen_pairs["candidate_conflict_support"] = (
                _num_col(chosen_pairs, "sim_negative_avoid_core").ge(0.08)
                | _num_col(chosen_pairs, "sim_negative_avoid_neg").ge(0.08)
                | _num_col(chosen_pairs, "sim_conflict_gap").ge(0.08)
            )
            chosen_pairs["has_candidate_text_fit_support"] = (
                chosen_pairs["candidate_pref_support"]
                | chosen_pairs["candidate_recent_support"]
                | chosen_pairs["candidate_context_support"]
            )
            chosen_pairs["has_candidate_text_strong_fit_support"] = (
                chosen_pairs["candidate_pref_support"].astype(int)
                + chosen_pairs["candidate_recent_support"].astype(int)
                + chosen_pairs["candidate_context_support"].astype(int)
            ).ge(2)
            for col in [
                "has_recent_context_exact_bridge",
                "has_recent_context_alias_bridge",
                "has_recent_context_visible_bridge",
                "user_source_positive_sentence_count_v1",
                "user_source_negative_sentence_count_v1",
                "user_source_recent_sentence_count_v1",
                "user_source_tip_sentence_count_v1",
                "user_source_history_anchor_count_v1",
            ]:
                if col not in chosen_pairs.columns:
                    chosen_pairs[col] = 0
            chosen_pairs["has_recent_context_exact_bridge"] = _bool_col(
                chosen_pairs, "has_recent_context_exact_bridge"
            )
            chosen_pairs["has_recent_context_alias_bridge"] = _bool_col(
                chosen_pairs, "has_recent_context_alias_bridge"
            )
            chosen_pairs["has_recent_context_visible_bridge"] = _bool_col(
                chosen_pairs, "has_recent_context_visible_bridge"
            )

            def _candidate_support_summary(df: pd.DataFrame) -> dict[str, Any]:
                if df.empty:
                    return {"n": 0}
                return {
                    "n": int(len(df)),
                    "fit_support_rate": float(df["has_candidate_text_fit_support"].mean()),
                    "strong_fit_support_rate": float(df["has_candidate_text_strong_fit_support"].mean()),
                    "pref_support_rate": float(df["candidate_pref_support"].mean()),
                    "recent_support_rate": float(df["candidate_recent_support"].mean()),
                    "context_support_rate": float(df["candidate_context_support"].mean()),
                    "conflict_support_rate": float(df["candidate_conflict_support"].mean()),
                }

            heavy_c0 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C0_FAIL")
            ].copy()
            heavy_c1 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C1_WEAK")
            ].copy()
            heavy_c3 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "heavy") & (chosen_pairs["constructability_class"] == "C3_IDEAL")
            ].copy()
            mid_fail = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "mid")
                & (chosen_pairs["constructability_class"].isin(["C0_FAIL", "C1_WEAK"]))
            ].copy()
            mid_c2 = chosen_pairs.loc[
                (chosen_pairs["light_mid_heavy"] == "mid")
                & (chosen_pairs["constructability_class"] == "C2_USABLE")
            ].copy()

            payload["candidate_text_support_audit"] = {
                "candidate_text_match_run": candidate_text_match_run,
                "heavy_c0": _candidate_support_summary(heavy_c0),
                "heavy_c1": _candidate_support_summary(heavy_c1),
                "heavy_c3": _candidate_support_summary(heavy_c3),
                "mid_fail": _candidate_support_summary(mid_fail),
                "mid_c2": _candidate_support_summary(mid_c2),
            }

            for frame in [heavy_c0, heavy_c1, mid_fail]:
                if frame.empty:
                    frame["fit_scope_signature"] = pd.Series(dtype=str)
                else:
                    frame["fit_scope_signature"] = frame.apply(_fit_scope_signature, axis=1)

            heavy_c0_candidate_supported = heavy_c0.loc[heavy_c0["has_candidate_text_fit_support"]].copy()
            heavy_c0_candidate_unsupported = heavy_c0.loc[~heavy_c0["has_candidate_text_fit_support"]].copy()
            heavy_c0_alias_supported = heavy_c0.loc[heavy_c0["has_recent_context_visible_bridge"]].copy()
            heavy_c0_likely_recoverable = heavy_c0.loc[
                heavy_c0["has_candidate_text_fit_support"] | heavy_c0["has_recent_context_visible_bridge"]
            ].copy()

            heavy_c1_stable_only = heavy_c1.loc[heavy_c1["fit_scope_signature"] == "stable"].copy()
            heavy_c1_recent_only = heavy_c1.loc[heavy_c1["fit_scope_signature"] == "recent"].copy()
            heavy_c1_history_only = heavy_c1.loc[heavy_c1["fit_scope_signature"] == "history"].copy()
            heavy_c1_practical_only = heavy_c1.loc[heavy_c1["fit_scope_signature"] == "practical"].copy()
            heavy_c1_likely_recoverable = heavy_c1.loc[
                heavy_c1["has_candidate_text_strong_fit_support"]
                | heavy_c1["has_recent_context_alias_bridge"]
            ].copy()

            mid_fail_full = mid_fail.loc[
                mid_fail["semantic_user_readiness_tier_v1"].astype(str).eq("FULL")
            ].copy()
            mid_fail_low_signal = mid_fail.loc[
                mid_fail["semantic_user_readiness_tier_v1"].astype(str).eq("LOW_SIGNAL")
            ].copy()
            mid_no_avoid = mid_fail.loc[mid_fail["reason_codes"].map(lambda xs: "U_NO_AVOID" in xs)].copy()
            mid_no_user_evidence = mid_fail.loc[
                mid_fail["reason_codes"].map(lambda xs: "U_NO_USER_EVIDENCE" in xs)
            ].copy()
            mid_fit_missing = mid_fail.loc[
                mid_fail["reason_codes"].map(lambda xs: "P_NO_VISIBLE_FIT" in xs)
            ].copy()
            mid_no_avoid_with_source_negative = mid_no_avoid.loc[
                mid_no_avoid["user_source_negative_sentence_count_v1"].gt(0)
            ].copy()
            mid_no_avoid_no_source_negative = mid_no_avoid.loc[
                ~mid_no_avoid["user_source_negative_sentence_count_v1"].gt(0)
            ].copy()
            mid_fit_missing_with_candidate_support = mid_fit_missing.loc[
                mid_fit_missing["has_candidate_text_fit_support"]
            ].copy()
            mid_fit_missing_without_candidate_support = mid_fit_missing.loc[
                ~mid_fit_missing["has_candidate_text_fit_support"]
            ].copy()
            mid_no_user_evidence_with_source_support = mid_no_user_evidence.loc[
                mid_no_user_evidence["user_source_positive_sentence_count_v1"].gt(0)
                | mid_no_user_evidence["user_source_recent_sentence_count_v1"].gt(0)
                | mid_no_user_evidence["user_source_history_anchor_count_v1"].gt(0)
            ].copy()
            mid_likely_fallback = mid_fail.loc[
                mid_fail["semantic_user_readiness_tier_v1"].astype(str).isin(["LOW_SIGNAL", ""])
                & ~mid_fail["has_candidate_text_fit_support"]
                & ~mid_fail["user_source_negative_sentence_count_v1"].gt(0)
            ].copy()

            def _example_records(df: pd.DataFrame, cols: list[str], limit: int = 6) -> list[dict[str, Any]]:
                if df.empty:
                    return []
                keep = [c for c in cols if c in df.columns]
                return df[keep].head(limit).to_dict(orient="records")

            payload["remaining_fail_actionability"] = {
                "heavy_c0": {
                    "n": int(len(heavy_c0)),
                    "with_candidate_fit_support": int(len(heavy_c0_candidate_supported)),
                    "without_candidate_fit_support": int(len(heavy_c0_candidate_unsupported)),
                    "with_context_alias_bridge": int(len(heavy_c0_alias_supported)),
                    "likely_recoverable": int(len(heavy_c0_likely_recoverable)),
                    "likely_fallback": int(len(heavy_c0) - len(heavy_c0_likely_recoverable)),
                    "candidate_support_rate": float(heavy_c0["has_candidate_text_fit_support"].mean()) if len(heavy_c0) else 0.0,
                    "context_alias_rate": float(heavy_c0["has_recent_context_alias_bridge"].mean()) if len(heavy_c0) else 0.0,
                "examples": _example_records(
                    heavy_c0.loc[
                            heavy_c0["has_candidate_text_fit_support"] | heavy_c0["has_recent_context_visible_bridge"]
                        ].copy(),
                        [
                            "user_id",
                            "business_id",
                            "truth_learned_rank",
                            "primary_reason_code",
                            "stable_preferences_text",
                            "recent_intent_text_v2",
                            "history_anchor_hint_text",
                            "fit_reasons_text_v1",
                            "evidence_basis_text_v1",
                            "has_recent_context_exact_bridge",
                            "has_candidate_text_fit_support",
                            "has_recent_context_alias_bridge",
                        ],
                    ),
                },
                "heavy_c1": {
                    "n": int(len(heavy_c1)),
                    "fit_scope_signature_counts": _summarize_counts(heavy_c1["fit_scope_signature"]) if len(heavy_c1) else {},
                    "stable_only": int(len(heavy_c1_stable_only)),
                    "recent_only": int(len(heavy_c1_recent_only)),
                    "history_only": int(len(heavy_c1_history_only)),
                    "practical_only": int(len(heavy_c1_practical_only)),
                    "with_candidate_strong_support": int(_count_true(heavy_c1, "has_candidate_text_strong_fit_support")),
                    "with_context_alias_bridge": int(_count_true(heavy_c1, "has_recent_context_alias_bridge")),
                    "likely_recoverable": int(len(heavy_c1_likely_recoverable)),
                    "likely_fallback": int(len(heavy_c1) - len(heavy_c1_likely_recoverable)),
                    "examples": _example_records(
                        heavy_c1,
                        [
                            "user_id",
                            "business_id",
                            "truth_learned_rank",
                            "fit_scope_signature",
                            "fit_reasons_text_v1",
                            "friction_reasons_text_v1",
                            "evidence_basis_text_v1",
                            "has_recent_context_exact_bridge",
                            "has_candidate_text_strong_fit_support",
                            "has_recent_context_alias_bridge",
                        ],
                    ),
                },
                "mid_fail": {
                    "n": int(len(mid_fail)),
                    "readiness_tier_counts": _summarize_counts(mid_fail["semantic_user_readiness_tier_v1"]) if len(mid_fail) else {},
                    "full_readiness_fail": int(len(mid_fail_full)),
                    "low_signal_fail": int(len(mid_fail_low_signal)),
                    "mid_no_avoid_with_source_negative": int(len(mid_no_avoid_with_source_negative)),
                    "mid_no_avoid_without_source_negative": int(len(mid_no_avoid_no_source_negative)),
                    "mid_no_user_evidence_with_source_support": int(len(mid_no_user_evidence_with_source_support)),
                    "mid_fit_missing_with_candidate_support": int(len(mid_fit_missing_with_candidate_support)),
                    "mid_fit_missing_without_candidate_support": int(len(mid_fit_missing_without_candidate_support)),
                    "likely_fallback": int(len(mid_likely_fallback)),
                    "likely_recoverable_lower_bound": int(
                        len(mid_no_avoid_with_source_negative)
                        + len(mid_no_user_evidence_with_source_support)
                        + len(mid_fit_missing_with_candidate_support)
                    ),
                    "examples_recoverable": _example_records(
                        pd.concat(
                            [
                                mid_no_avoid_with_source_negative,
                                mid_no_user_evidence_with_source_support,
                                mid_fit_missing_with_candidate_support,
                            ],
                            ignore_index=True,
                        ).drop_duplicates(subset=["user_id"]),
                        [
                            "user_id",
                            "truth_learned_rank",
                            "semantic_user_readiness_tier_v1",
                            "primary_reason_code",
                            "stable_preferences_text",
                            "recent_intent_text_v2",
                            "avoidance_text_v2",
                            "user_evidence_text_v2",
                            "fit_reasons_text_v1",
                            "evidence_basis_text_v1",
                            "has_candidate_text_fit_support",
                            "user_source_negative_sentence_count_v1",
                        ],
                    ),
                    "examples_likely_fallback": _example_records(
                        mid_likely_fallback,
                        [
                            "user_id",
                            "truth_learned_rank",
                            "semantic_user_readiness_tier_v1",
                            "primary_reason_code",
                            "stable_preferences_text",
                            "recent_intent_text_v2",
                            "avoidance_text_v2",
                            "user_evidence_text_v2",
                            "fit_reasons_text_v1",
                            "has_candidate_text_fit_support",
                            "user_source_negative_sentence_count_v1",
                        ],
                    ),
                },
            }

            heavy_c0_stage09_bug_likely = heavy_c0_alias_supported.copy()
            heavy_c0_latent_numeric_only = heavy_c0.loc[
                heavy_c0["has_candidate_text_fit_support"] & ~heavy_c0["has_recent_context_alias_bridge"]
            ].copy()
            heavy_c0_no_obvious_support = heavy_c0.loc[
                ~heavy_c0["has_candidate_text_fit_support"] & ~heavy_c0["has_recent_context_alias_bridge"]
            ].copy()

            heavy_c1_needs_stage11_contrast = heavy_c1.loc[
                heavy_c1["fit_scope_signature"].eq("stable")
                & heavy_c1["has_candidate_text_strong_fit_support"]
            ].copy()
            heavy_c1_stage09_bug_likely = heavy_c1.loc[
                heavy_c1["has_recent_context_visible_bridge"]
            ].copy()
            heavy_c1_no_obvious_support = heavy_c1.loc[
                ~heavy_c1["has_candidate_text_strong_fit_support"]
                & ~heavy_c1["has_recent_context_alias_bridge"]
            ].copy()

            mid_stage09_bug_no_avoid = mid_no_avoid_with_source_negative.copy()
            mid_stage09_bug_no_evidence = mid_no_user_evidence_with_source_support.copy()
            mid_latent_numeric_only = mid_fit_missing_with_candidate_support.copy()
            mid_unresolved_other = mid_fail.loc[
                ~mid_fail["user_id"].isin(
                    pd.concat(
                        [
                            mid_stage09_bug_no_avoid[["user_id"]],
                            mid_stage09_bug_no_evidence[["user_id"]],
                            mid_latent_numeric_only[["user_id"]],
                            mid_likely_fallback[["user_id"]],
                        ],
                        ignore_index=True,
                    )["user_id"].drop_duplicates()
                )
            ].copy()

            payload["resolution_layer_audit"] = {
                "heavy_c0": {
                    "n": int(len(heavy_c0)),
                    "stage09_bug_likely": int(len(heavy_c0_stage09_bug_likely)),
                    "latent_numeric_only": int(len(heavy_c0_latent_numeric_only)),
                    "no_obvious_support": int(len(heavy_c0_no_obvious_support)),
                },
                "heavy_c1": {
                    "n": int(len(heavy_c1)),
                    "needs_stage11_contrast": int(len(heavy_c1_needs_stage11_contrast)),
                    "stage09_bug_likely": int(len(heavy_c1_stage09_bug_likely)),
                    "no_obvious_support": int(len(heavy_c1_no_obvious_support)),
                },
                "mid_fail": {
                    "n": int(len(mid_fail)),
                    "stage09_bug_no_avoid": int(len(mid_stage09_bug_no_avoid)),
                    "stage09_bug_no_evidence": int(len(mid_stage09_bug_no_evidence)),
                    "latent_numeric_only_fit_missing": int(len(mid_latent_numeric_only)),
                    "likely_fallback": int(len(mid_likely_fallback)),
                    "unresolved_other": int(len(mid_unresolved_other)),
                },
            }

    chosen_pairs["avoid_positive_suspect"] = chosen_pairs.get("avoidance_text_v2", pd.Series("", index=chosen_pairs.index)).map(_suspect_positive_avoid)
    payload["text_quality_audit"] = {
        "avoid_positive_suspect_rate": float(chosen_pairs["avoid_positive_suspect"].mean()),
        "avoid_positive_suspect_by_bucket": {
            str(bucket): float(bucket_df["avoid_positive_suspect"].mean())
            for bucket, bucket_df in chosen_pairs.groupby("light_mid_heavy", dropna=False)
        },
        "avoid_positive_suspect_by_constructability": {
            str(cls): float(cls_df["avoid_positive_suspect"].mean())
            for cls, cls_df in chosen_pairs.groupby("constructability_class", dropna=False)
        },
    }

    if export_run_dir is not None:
        train_df, eval_df = _load_export_ready_users(export_run_dir)
        actual_users = pd.concat([train_df, eval_df], ignore_index=True)
        actual_users = actual_users.loc[actual_users["chosen_learned_rank_band"] == target_band].copy()
        actual_users = actual_users.loc[actual_users["user_idx"] >= 0].drop_duplicates(subset=["user_idx", "split"])
        chosen_pairs["user_idx"] = pd.to_numeric(chosen_pairs["user_idx"], errors="coerce").fillna(-1).astype(int)
        actual_join = actual_users.merge(
            chosen_pairs[["user_idx", "light_mid_heavy", "constructability_class", "primary_reason_code"]],
            on="user_idx",
            how="left",
        )
        users_with_pairs_by_bucket = {
            split: _summarize_counts(actual_join.loc[actual_join["split"] == split, "light_mid_heavy"])
            for split in ["train", "eval"]
        }
        users_with_pairs_by_constructability = {
            split: _summarize_counts(actual_join.loc[actual_join["split"] == split, "constructability_class"])
            for split in ["train", "eval"]
        }
        payload["export_only_lower_bound"] = {
            "export_run_dir": str(export_run_dir),
            "target_band": target_band,
            "train_users_with_pairs": int(train_df.loc[train_df["chosen_learned_rank_band"] == target_band, "user_idx"].nunique()),
            "eval_users_with_pairs": int(eval_df.loc[eval_df["chosen_learned_rank_band"] == target_band, "user_idx"].nunique()),
            "users_with_pairs_by_bucket": users_with_pairs_by_bucket,
            "users_with_pairs_by_constructability": users_with_pairs_by_constructability,
        }
        payload["realized_train_by_bucket"] = users_with_pairs_by_bucket.get("train", {})
        payload["realized_eval_by_bucket"] = users_with_pairs_by_bucket.get("eval", {})
        payload["realized_train_by_constructability"] = users_with_pairs_by_constructability.get("train", {})
        payload["realized_eval_by_constructability"] = users_with_pairs_by_constructability.get("eval", {})

    if args.details_output:
        details_cols = [
            "bucket",
            "user_idx",
            "user_id",
            "business_id",
            "item_idx",
            "true_item_idx",
            "truth_learned_rank",
            "light_mid_heavy",
            "constructability_class",
            "primary_reason_code",
            "reason_codes",
            "semantic_user_readiness_tier_v1",
            "user_profile_richness_tier_v2",
            "user_quality_signal_count_v2",
            "user_visible_fact_count_v1",
            "has_focus",
            "has_recent_or_history",
            "has_avoid",
            "has_user_evidence",
            "has_fit",
            "has_conflict",
            "has_pair_evidence",
            "has_multisource_fit",
            "has_detail_support",
            "rival_total",
            "rival_head",
            "rival_boundary",
            "rival_mid",
            "rival_deep",
            "rival_head_or_boundary",
            "stable_preferences_text",
            "recent_intent_text_v2",
            "avoidance_text_v2",
            "history_anchor_hint_text",
            "user_evidence_text_v2",
            "fit_reasons_text_v1",
            "friction_reasons_text_v1",
            "evidence_basis_text_v1",
            "pair_fit_fact_count_v1",
            "pair_friction_fact_count_v1",
            "pair_evidence_fact_count_v1",
            "pair_fit_scope_count_v1",
            "pair_has_detail_support_v1",
            "pair_has_multisource_fit_v1",
            "pair_has_visible_user_fit_v1",
            "pair_has_visible_conflict_v1",
            "mean_match_total_v1",
            "mean_match_positive_evidence",
            "mean_match_recent_cuisine",
            "mean_match_negative_conflict",
            "mean_channel_preference_core_v1",
            "mean_channel_recent_intent_v1",
            "mean_channel_evidence_support_v1",
            "mean_channel_conflict_v1",
            "mean_channel_context_time_v1",
            "mean_channel_preference_property_v1",
            "raw_preference_support",
            "raw_recent_support",
            "raw_context_support",
            "raw_property_support",
            "raw_positive_evidence_support",
            "has_raw_pair_support",
            "has_strong_raw_pair_support",
            "sim_long_pref_core",
            "sim_recent_intent_semantic",
            "sim_recent_intent_pos",
            "sim_negative_avoid_neg",
            "sim_negative_avoid_core",
            "sim_context_merchant",
            "sim_conflict_gap",
            "candidate_pref_support",
            "candidate_recent_support",
            "candidate_context_support",
            "candidate_conflict_support",
            "has_candidate_text_fit_support",
            "has_candidate_text_strong_fit_support",
            "has_recent_context_exact_bridge",
            "has_recent_context_alias_bridge",
            "has_recent_context_visible_bridge",
            "recent_context_exact_terms",
            "recent_context_alias_terms",
            "user_source_positive_sentence_count_v1",
            "user_source_negative_sentence_count_v1",
            "user_source_recent_sentence_count_v1",
            "user_source_tip_sentence_count_v1",
            "user_source_history_anchor_count_v1",
        ]
        details_cols = [c for c in details_cols if c in chosen_pairs.columns]
        details_df = chosen_pairs[details_cols].copy()
        details_path = Path(args.details_output).expanduser()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_df.to_parquet(details_path, index=False)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
