"""Stage 11-2 DPO pairwise export + audit.

Build pairwise preference data from an existing stage11_1 dataset run
without starting model training.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path
from pipeline.qlora_prompting import (
    build_binary_prompt,
    build_binary_prompt_semantic,
    build_item_text,
    build_item_text_full_lite,
    build_item_text_semantic,
    build_item_text_sft_clean,
    build_user_text,
)
from pipeline.stage11_pairwise import (
    build_dpo_pairs,
    build_pointwise_audit,
    build_rerank_pool_pairs,
    build_rich_sft_dpo_pairs,
)
from pipeline.stage11_text_features import build_pair_alignment_summary, extract_user_evidence_text


RUN_TAG = "stage11_2_dpo_export_pairs"

INPUT_11_RUN_DIR = os.getenv("INPUT_11_RUN_DIR", "").strip()
INPUT_11_ROOT = env_or_project_path("INPUT_11_ROOT_DIR", "data/output/11_qlora_data")
INPUT_11_SUFFIX = "_stage11_1_qlora_build_dataset"
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_PAIRWISE_ROOT_DIR", "data/output/11_qlora_pairwise")
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "false").strip().lower() == "true"
EVAL_USER_FRAC = float(os.getenv("QLORA_EVAL_USER_FRAC", "0.1").strip() or 0.1)
PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "semantic").strip().lower() or "semantic"

DPO_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_MAX_PAIRS", "8").strip() or 8)
DPO_HARD_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_HARD_MAX_PAIRS", "2").strip() or 2)
DPO_TRUE_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_TRUE_MAX_PAIRS", "2").strip() or 2)
DPO_VALID_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_VALID_MAX_PAIRS", "1").strip() or 1)
DPO_HIST_MAX_PAIRS_PER_USER = int(os.getenv("QLORA_DPO_HIST_MAX_PAIRS", "1").strip() or 1)
DPO_ALLOW_MID_NEG = os.getenv("QLORA_DPO_ALLOW_MID_NEG", "true").strip().lower() == "true"
DPO_PREFER_EASY_NEG = os.getenv("QLORA_DPO_PREFER_EASY_NEG", "true").strip().lower() == "true"
DPO_FILTER_INVERTED = os.getenv("QLORA_DPO_FILTER_INVERTED", "false").strip().lower() == "true"
DPO_STRIP_RANK_FEATURES = os.getenv("QLORA_DPO_STRIP_RANK_FEATURES", "false").strip().lower() == "true"
PAIRWISE_SOURCE_MODE = os.getenv("QLORA_PAIRWISE_SOURCE_MODE", "auto").strip().lower() or "auto"

_RANK_FEATURE_RE = re.compile(
    r"; (?:candidate_sources|user_segment|als_rank|cluster_rank|profile_rank|popular_rank): [^;]*"
)


def strip_rank_features(prompt: str) -> str:
    return _RANK_FEATURE_RE.sub("", prompt)


def parse_bucket_override(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return sorted(list(set(out)))


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage11_dataset_run() -> Path:
    if INPUT_11_RUN_DIR:
        p = Path(INPUT_11_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_11_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_11_ROOT, INPUT_11_SUFFIX)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _source_set_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _pre_rank_band(value: Any) -> str:
    r = _safe_int(value, 999999)
    if r <= 10:
        return "001_010"
    if r <= 30:
        return "011_030"
    if r <= 80:
        return "031_080"
    if r <= 150:
        return "081_150"
    return "151_plus"


def _normalize_text_key(value: Any) -> str:
    return _to_text(value).strip().lower()


def choose_stage09_candidate_file(bucket_dir: Path) -> Path:
    candidates = [
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates_pretrim.parquet",
    ]
    for name in candidates:
        path = bucket_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_stage09_run(source_11: Path) -> Path:
    if INPUT_09_RUN_DIR:
        path = Path(INPUT_09_RUN_DIR)
        if not path.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {path}")
        return path
    meta_path = source_11 / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"dataset run_meta.json missing: {meta_path}. Provide INPUT_09_RUN_DIR for stage09_direct mode."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    src = _to_text(meta.get("source_stage09_run"))
    if not src:
        raise RuntimeError(
            f"source_stage09_run missing in {meta_path}. Provide INPUT_09_RUN_DIR for stage09_direct mode."
        )
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"source_stage09_run not found: {path}")
    return path


def _resolve_eval_user_frac(source_11: Path) -> float:
    meta_path = source_11 / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("eval_user_frac", None)
        try:
            if raw is not None:
                val = float(raw)
                if 0.0 < val < 1.0:
                    return float(val)
        except Exception:
            pass
    return float(EVAL_USER_FRAC)


def _load_stage09_support_tables(source_09: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    meta_path = source_09 / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"stage09 run_meta.json missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    user_profile_path = normalize_legacy_project_path(_to_text(meta.get("user_profile_table")))
    item_sem_path = normalize_legacy_project_path(_to_text(meta.get("item_semantic_features")))
    cluster_profile_path = normalize_legacy_project_path(_to_text(meta.get("cluster_profile_csv")))

    user_df = _read_csv_if_exists(user_profile_path)
    if not user_df.empty:
        user_base = pd.DataFrame()
        user_base["user_id"] = user_df["user_id"] if "user_id" in user_df.columns else ""
        if "profile_text_short" in user_df.columns:
            user_base["profile_text"] = user_df["profile_text_short"]
        elif "profile_text" in user_df.columns:
            user_base["profile_text"] = user_df["profile_text"]
        else:
            user_base["profile_text"] = ""
        if "profile_text_long" in user_df.columns:
            user_base["profile_text_evidence"] = user_df["profile_text_long"]
        elif "profile_text" in user_df.columns:
            user_base["profile_text_evidence"] = user_df["profile_text"]
        elif "profile_text_short" in user_df.columns:
            user_base["profile_text_evidence"] = user_df["profile_text_short"]
        else:
            user_base["profile_text_evidence"] = ""
        user_base["profile_top_pos_tags"] = user_df["profile_top_pos_tags"] if "profile_top_pos_tags" in user_df.columns else ""
        user_base["profile_top_neg_tags"] = user_df["profile_top_neg_tags"] if "profile_top_neg_tags" in user_df.columns else ""
        user_base["profile_confidence"] = user_df["profile_confidence"] if "profile_confidence" in user_df.columns else ""
        user_df = user_base.drop_duplicates(subset=["user_id"])
        user_lookup = user_df.set_index("user_id").to_dict(orient="index")
    else:
        user_lookup = {}

    item_df = _read_csv_if_exists(item_sem_path)
    if not item_df.empty:
        for col in ["business_id", "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence"]:
            if col not in item_df.columns:
                item_df[col] = ""
        item_df = item_df[
            ["business_id", "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence"]
        ].drop_duplicates(subset=["business_id"])
        item_lookup = item_df.set_index("business_id").to_dict(orient="index")
    else:
        item_lookup = {}

    cluster_df = _read_csv_if_exists(cluster_profile_path)
    if not cluster_df.empty:
        for col in ["business_id", "cluster_for_recsys", "cluster_label_for_recsys"]:
            if col not in cluster_df.columns:
                cluster_df[col] = ""
        cluster_df = cluster_df[
            ["business_id", "cluster_for_recsys", "cluster_label_for_recsys"]
        ].drop_duplicates(subset=["business_id"])
        cluster_lookup = cluster_df.set_index("business_id").to_dict(orient="index")
    else:
        cluster_lookup = {}

    return user_lookup, item_lookup, cluster_lookup


def build_stage09_pool_rows(source_11: Path, buckets: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[tuple[int, int], dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_09 = _resolve_stage09_run(source_11)
    eval_user_frac = _resolve_eval_user_frac(source_11)
    user_lookup, item_lookup, cluster_lookup = _load_stage09_support_tables(source_09)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    row_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    file_summary: dict[str, Any] = {"source_stage09_run": str(source_09), "buckets": []}

    for bucket in buckets:
        bdir = source_09 / f"bucket_{bucket}"
        truth_path = bdir / "truth.parquet"
        if not bdir.exists() or not truth_path.exists():
            continue
        cand_path = choose_stage09_candidate_file(bdir)
        truth_df = pd.read_parquet(truth_path, columns=["user_idx", "user_id", "true_item_idx"])
        truth_df = truth_df.drop_duplicates(subset=["user_idx"])
        cand_df = pd.read_parquet(cand_path)
        if truth_df.empty or cand_df.empty:
            file_summary["buckets"].append(
                {"bucket": int(bucket), "candidate_file": str(cand_path), "rows_total": 0}
            )
            continue

        cand_df = cand_df[cand_df["pre_rank"].fillna(999999).astype(float) <= 150].copy()
        cand_df = cand_df.merge(truth_df, on="user_idx", how="inner")
        truth_in_pool = cand_df.loc[cand_df["item_idx"] == cand_df["true_item_idx"], ["user_idx"]].drop_duplicates()
        cand_df = cand_df.merge(truth_in_pool, on="user_idx", how="inner")
        if cand_df.empty:
            file_summary["buckets"].append(
                {
                    "bucket": int(bucket),
                    "candidate_file": str(cand_path),
                    "rows_total": 0,
                    "truth_in_pool_users": 0,
                }
            )
            continue

        pos_ctx = (
            cand_df.loc[cand_df["item_idx"] == cand_df["true_item_idx"], ["user_idx", "city", "primary_category", "pre_score"]]
            .drop_duplicates(subset=["user_idx"])
            .rename(
                columns={
                    "city": "pos_city",
                    "primary_category": "pos_primary_category",
                    "pre_score": "pos_pre_score",
                }
            )
        )
        cand_df = cand_df.merge(pos_ctx, on="user_idx", how="left")
        cand_df["label"] = (cand_df["item_idx"] == cand_df["true_item_idx"]).astype(int)
        cand_df["label_source"] = cand_df["label"].map({1: "true", 0: "pair_pool"})
        cand_df["sample_weight"] = 1.0
        cand_df["neg_pick_rank"] = 0
        cand_df["pre_rank_band"] = cand_df["pre_rank"].map(_pre_rank_band)
        cand_df["neg_is_near"] = (
            (cand_df["label"] == 0)
            & (
                (cand_df["city"].map(_normalize_text_key) == cand_df["pos_city"].map(_normalize_text_key))
                | (
                    cand_df["primary_category"].map(_normalize_text_key)
                    == cand_df["pos_primary_category"].map(_normalize_text_key)
                )
            )
        )
        cand_df["neg_is_hard"] = (cand_df["label"] == 0) & (cand_df["pre_rank"].fillna(999999).astype(float) <= 10)
        cand_df["neg_tier"] = "easy"
        cand_df.loc[cand_df["label"] == 1, "neg_tier"] = "pos"
        cand_df.loc[(cand_df["label"] == 0) & cand_df["neg_is_near"], "neg_tier"] = "near"
        cand_df.loc[(cand_df["label"] == 0) & (~cand_df["neg_is_near"]) & (cand_df["pre_rank"].fillna(999999).astype(float) <= 10), "neg_tier"] = "hard"
        cand_df.loc[
            (cand_df["label"] == 0)
            & (~cand_df["neg_is_near"])
            & (cand_df["pre_rank"].fillna(999999).astype(float) > 10)
            & (cand_df["pre_rank"].fillna(999999).astype(float) <= 80),
            "neg_tier",
        ] = "mid"
        cand_df["split"] = cand_df["user_idx"].astype(int).map(
            lambda x: "eval" if (int(x) % 100) < int(100 * eval_user_frac) else "train"
        )
        cand_df["bucket"] = int(bucket)
        cand_df["source_set_text"] = cand_df["source_set"].map(_source_set_text) if "source_set" in cand_df.columns else ""
        for col in [
            "user_segment",
            "als_rank",
            "cluster_rank",
            "profile_rank",
            "popular_rank",
            "semantic_support",
            "semantic_tag_richness",
            "tower_score",
            "seq_score",
            "name",
            "city",
            "categories",
            "primary_category",
        ]:
            if col not in cand_df.columns:
                cand_df[col] = ""

        selected_cols = [
            "bucket",
            "user_idx",
            "user_id",
            "item_idx",
            "business_id",
            "label",
            "sample_weight",
            "label_source",
            "neg_tier",
            "neg_pick_rank",
            "neg_is_near",
            "neg_is_hard",
            "pre_rank",
            "pre_rank_band",
            "pre_score",
            "split",
            "name",
            "city",
            "categories",
            "primary_category",
            "source_set_text",
            "user_segment",
            "als_rank",
            "cluster_rank",
            "profile_rank",
            "popular_rank",
            "semantic_support",
            "semantic_tag_richness",
            "tower_score",
            "seq_score",
        ]
        rows = cand_df[selected_cols].to_dict(orient="records")
        for row in rows:
            key = (_safe_int(row.get("user_idx"), -1), _safe_int(row.get("item_idx"), -1))
            row_lookup[key] = row
            if str(row.get("split", "")) == "eval":
                eval_rows.append(row)
            else:
                train_rows.append(row)

        file_summary["buckets"].append(
            {
                "bucket": int(bucket),
                "candidate_file": str(cand_path),
                "rows_total": int(len(rows)),
                "truth_in_pool_users": int(len(truth_in_pool)),
                "train_rows": int(sum(1 for row in rows if row.get("split") == "train")),
                "eval_rows": int(sum(1 for row in rows if row.get("split") == "eval")),
                "eval_user_frac": float(eval_user_frac),
            }
        )

    return train_rows, eval_rows, file_summary, row_lookup, user_lookup, item_lookup, cluster_lookup


def _build_prompt_from_row(
    row: dict[str, Any],
    user_lookup: dict[str, dict[str, Any]],
    item_lookup: dict[str, dict[str, Any]],
    cluster_lookup: dict[str, dict[str, Any]],
) -> str:
    user_id = _to_text(row.get("user_id"))
    business_id = _to_text(row.get("business_id"))
    u = user_lookup.get(user_id, {})
    i = item_lookup.get(business_id, {})
    c = cluster_lookup.get(business_id, {})

    profile_text = _to_text(u.get("profile_text"))
    profile_text_evidence = _to_text(u.get("profile_text_evidence"))
    profile_top_pos_tags = _to_text(u.get("profile_top_pos_tags"))
    profile_top_neg_tags = _to_text(u.get("profile_top_neg_tags"))
    profile_confidence = _safe_float(u.get("profile_confidence"), 0.0)
    top_pos_tags = _to_text(i.get("top_pos_tags"))
    top_neg_tags = _to_text(i.get("top_neg_tags"))
    semantic_score = _safe_float(i.get("semantic_score"), 0.0)
    semantic_confidence = _safe_float(i.get("semantic_confidence"), 0.0)
    cluster_for_recsys = _to_text(c.get("cluster_for_recsys"))
    cluster_label_for_recsys = _to_text(c.get("cluster_label_for_recsys"))

    user_evidence = extract_user_evidence_text(profile_text, profile_text_evidence, max_chars=260)
    pair_signal = build_pair_alignment_summary(
        profile_top_pos_tags,
        profile_top_neg_tags,
        top_pos_tags,
        top_neg_tags,
    )
    user_text = build_user_text(
        profile_text=profile_text,
        top_pos_tags=profile_top_pos_tags,
        top_neg_tags=profile_top_neg_tags,
        confidence=profile_confidence,
        evidence_snippets=user_evidence,
        pair_evidence=pair_signal,
    )
    if PROMPT_MODE == "semantic":
        item_text = build_item_text_semantic(
            name=row.get("name", ""),
            city=row.get("city", ""),
            categories=row.get("categories", ""),
            primary_category=row.get("primary_category", ""),
            top_pos_tags=top_pos_tags,
            top_neg_tags=top_neg_tags,
            semantic_score=semantic_score,
            semantic_confidence=semantic_confidence,
            semantic_support=row.get("semantic_support", 0.0),
            semantic_tag_richness=row.get("semantic_tag_richness", 0.0),
            tower_score=row.get("tower_score", 0.0),
            seq_score=row.get("seq_score", 0.0),
            cluster_for_recsys=cluster_for_recsys,
            cluster_label_for_recsys=cluster_label_for_recsys,
            item_review_snippet="",
        )
        prompt = build_binary_prompt_semantic(user_text, item_text)
    elif PROMPT_MODE == "full_lite":
        item_text = build_item_text_full_lite(
            name=row.get("name", ""),
            city=row.get("city", ""),
            categories=row.get("categories", ""),
            primary_category=row.get("primary_category", ""),
            top_pos_tags=top_pos_tags,
            top_neg_tags=top_neg_tags,
            semantic_score=semantic_score,
            semantic_confidence=semantic_confidence,
            source_set=row.get("source_set_text", ""),
            user_segment=row.get("user_segment", ""),
            semantic_support=row.get("semantic_support", 0.0),
            semantic_tag_richness=row.get("semantic_tag_richness", 0.0),
            tower_score=row.get("tower_score", 0.0),
            seq_score=row.get("seq_score", 0.0),
            cluster_label_for_recsys=cluster_label_for_recsys,
            item_review_snippet="",
        )
        prompt = build_binary_prompt(user_text, item_text)
    elif PROMPT_MODE == "sft_clean":
        item_text = build_item_text_sft_clean(
            name=row.get("name", ""),
            city=row.get("city", ""),
            categories=row.get("categories", ""),
            primary_category=row.get("primary_category", ""),
            top_pos_tags=top_pos_tags,
            top_neg_tags=top_neg_tags,
            semantic_score=semantic_score,
            semantic_confidence=semantic_confidence,
            cluster_label_for_recsys=cluster_label_for_recsys,
            item_review_snippet="",
        )
        prompt = build_binary_prompt(user_text, item_text)
    else:
        item_text = build_item_text(
            name=row.get("name", ""),
            city=row.get("city", ""),
            categories=row.get("categories", ""),
            primary_category=row.get("primary_category", ""),
            top_pos_tags=top_pos_tags,
            top_neg_tags=top_neg_tags,
            semantic_score=semantic_score,
            semantic_confidence=semantic_confidence,
            source_set=row.get("source_set_text", ""),
            user_segment=row.get("user_segment", ""),
            als_rank=row.get("als_rank", 0.0),
            cluster_rank=row.get("cluster_rank", 0.0),
            profile_rank=row.get("profile_rank", 0.0),
            popular_rank=row.get("popular_rank", 0.0),
            semantic_support=row.get("semantic_support", 0.0),
            semantic_tag_richness=row.get("semantic_tag_richness", 0.0),
            tower_score=row.get("tower_score", 0.0),
            seq_score=row.get("seq_score", 0.0),
            cluster_for_recsys=cluster_for_recsys,
            cluster_label_for_recsys=cluster_label_for_recsys,
            item_review_snippet="",
        )
        prompt = build_binary_prompt(user_text, item_text)
    if DPO_STRIP_RANK_FEATURES:
        prompt = strip_rank_features(prompt)
    return prompt


def hydrate_pair_prompts(
    pairs: list[dict[str, Any]],
    row_lookup: dict[tuple[int, int], dict[str, Any]],
    user_lookup: dict[str, dict[str, Any]],
    item_lookup: dict[str, dict[str, Any]],
    cluster_lookup: dict[str, dict[str, Any]],
) -> None:
    prompt_cache: dict[tuple[int, int], str] = {}
    for pair in pairs:
        chosen_key = (_safe_int(pair.get("user_idx"), -1), _safe_int(pair.get("chosen_item_idx"), -1))
        rejected_key = (_safe_int(pair.get("user_idx"), -1), _safe_int(pair.get("rejected_item_idx"), -1))
        chosen_row = row_lookup.get(chosen_key)
        rejected_row = row_lookup.get(rejected_key)
        if chosen_row is None or rejected_row is None:
            pair["prompt"] = ""
            pair["chosen"] = ""
            pair["rejected"] = ""
            continue
        chosen_prompt = prompt_cache.get(chosen_key)
        if chosen_prompt is None:
            chosen_prompt = _build_prompt_from_row(chosen_row, user_lookup, item_lookup, cluster_lookup)
            prompt_cache[chosen_key] = chosen_prompt
        rejected_prompt = prompt_cache.get(rejected_key)
        if rejected_prompt is None:
            rejected_prompt = _build_prompt_from_row(rejected_row, user_lookup, item_lookup, cluster_lookup)
            prompt_cache[rejected_key] = rejected_prompt
        common_prompt = chosen_prompt.split("Candidate:")[0] + "Candidate:" if "Candidate:" in chosen_prompt else ""
        pair["prompt"] = common_prompt
        pair["chosen"] = chosen_prompt + " YES"
        pair["rejected"] = rejected_prompt + " YES"

def collect_json_files(
    source_11: Path,
    buckets: list[int],
    train_dir_name: str,
    eval_dir_name: str,
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    train_files: list[Path] = []
    eval_files: list[Path] = []
    summary: dict[str, Any] = {"buckets": [], "train_dir_name": train_dir_name, "eval_dir_name": eval_dir_name}
    for b in buckets:
        bdir = source_11 / f"bucket_{b}"
        train_dir = bdir / train_dir_name
        eval_dir = bdir / eval_dir_name
        if not bdir.exists():
            continue
        b_train = sorted(train_dir.glob("*.json")) if train_dir.exists() else []
        b_eval = sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []
        train_files.extend(b_train)
        eval_files.extend(b_eval)
        summary["buckets"].append(
            {"bucket": int(b), "train_files": len(b_train), "eval_files": len(b_eval), "bucket_dir": str(bdir)}
        )
    return train_files, eval_files, summary


def resolve_pairwise_source_mode(source_11: Path, buckets: list[int]) -> str:
    valid_modes = {"auto", "pointwise", "pool", "rich_sft", "stage09_direct"}
    if PAIRWISE_SOURCE_MODE not in valid_modes:
        raise ValueError(f"unsupported QLORA_PAIRWISE_SOURCE_MODE={PAIRWISE_SOURCE_MODE}; expected one of {sorted(valid_modes)}")
    if PAIRWISE_SOURCE_MODE in ("pointwise", "pool", "rich_sft", "stage09_direct"):
        return PAIRWISE_SOURCE_MODE

    for b in buckets:
        bdir = source_11 / f"bucket_{b}"
        if (bdir / "rich_sft_train_json").exists() and list((bdir / "rich_sft_train_json").glob("*.json")):
            return "rich_sft"
        if (bdir / "pairwise_pool_train_json").exists() and list((bdir / "pairwise_pool_train_json").glob("*.json")):
            return "pool"
    if (source_11 / f"bucket_{buckets[0]}" / "train_json").exists():
        return "pointwise"
    return "stage09_direct"


def load_jsonl_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    source_11 = resolve_stage11_dataset_run()
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
    source_mode = resolve_pairwise_source_mode(source_11, buckets)

    if ENFORCE_STAGE09_GATE:
        ds_meta_path = source_11 / "run_meta.json"
        if not ds_meta_path.exists():
            raise FileNotFoundError(
                f"dataset run_meta.json missing: {ds_meta_path}. "
                "Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
            )
        ds_meta = json.loads(ds_meta_path.read_text(encoding="utf-8"))
        gate = ds_meta.get("stage09_gate_result", {})
        if not isinstance(gate, dict) or not gate:
            raise RuntimeError(
                "stage09 gate evidence missing in dataset run_meta. "
                "Rebuild 11_1 with gate enabled or set QLORA_ENFORCE_STAGE09_GATE=false."
            )
        missing = [str(int(b)) for b in buckets if str(int(b)) not in gate]
        if missing:
            raise RuntimeError(
                f"stage09 gate evidence missing for buckets={missing}. "
                "Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
            )

    row_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    user_lookup: dict[str, dict[str, Any]] = {}
    item_lookup: dict[str, dict[str, Any]] = {}
    cluster_lookup: dict[str, dict[str, Any]] = {}
    train_files: list[Path] = []
    eval_files: list[Path] = []
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    if source_mode == "stage09_direct":
        train_rows, eval_rows, file_summary, row_lookup, user_lookup, item_lookup, cluster_lookup = build_stage09_pool_rows(
            source_11,
            buckets,
        )
    elif source_mode == "rich_sft":
        train_files, eval_files, file_summary = collect_json_files(
            source_11,
            buckets,
            train_dir_name="rich_sft_train_json",
            eval_dir_name="rich_sft_eval_json",
        )
    elif source_mode == "pool":
        train_files, eval_files, file_summary = collect_json_files(
            source_11,
            buckets,
            train_dir_name="pairwise_pool_train_json",
            eval_dir_name="pairwise_pool_eval_json",
        )
    else:
        train_files, eval_files, file_summary = collect_json_files(
            source_11,
            buckets,
            train_dir_name="train_json",
            eval_dir_name="eval_json",
        )
    if source_mode != "stage09_direct" and not train_files:
        raise RuntimeError(
            f"no train json files found under {source_11} for buckets={buckets} source_mode={source_mode}"
        )
    if source_mode != "stage09_direct":
        train_rows = load_jsonl_rows(train_files)
        eval_rows = load_jsonl_rows(eval_files)
    if not train_rows:
        raise RuntimeError(f"no train rows resolved under {source_11} for buckets={buckets} source_mode={source_mode}")

    if DPO_STRIP_RANK_FEATURES and source_mode != "stage09_direct":
        for row in train_rows:
            row["prompt"] = strip_rank_features(str(row.get("prompt", "")))
        for row in eval_rows:
            row["prompt"] = strip_rank_features(str(row.get("prompt", "")))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_out = out_dir / "audit.json"
    meta_out = out_dir / "run_meta.json"

    if source_mode in ("pool", "stage09_direct"):
        conservative_train_pairs, conservative_train_audit = build_rerank_pool_pairs(
            train_rows,
            DPO_MAX_PAIRS_PER_USER,
            SEED,
            mode="conservative",
        )
        conservative_eval_pairs, conservative_eval_audit = (
            build_rerank_pool_pairs(
                eval_rows,
                max(2, DPO_MAX_PAIRS_PER_USER // 2),
                SEED + 1,
                mode="conservative",
            )
            if eval_rows
            else ([], {})
        )
        hard_train_pairs, hard_train_audit = build_rerank_pool_pairs(
            train_rows,
            max(1, DPO_HARD_MAX_PAIRS_PER_USER),
            SEED + 7,
            mode="hard",
        )
        hard_eval_pairs, hard_eval_audit = (
            build_rerank_pool_pairs(
                eval_rows,
                max(1, DPO_HARD_MAX_PAIRS_PER_USER),
                SEED + 8,
                mode="hard",
            )
            if eval_rows
            else ([], {})
        )

        if source_mode == "stage09_direct":
            hydrate_pair_prompts(conservative_train_pairs, row_lookup, user_lookup, item_lookup, cluster_lookup)
            hydrate_pair_prompts(conservative_eval_pairs, row_lookup, user_lookup, item_lookup, cluster_lookup)
            hydrate_pair_prompts(hard_train_pairs, row_lookup, user_lookup, item_lookup, cluster_lookup)
            hydrate_pair_prompts(hard_eval_pairs, row_lookup, user_lookup, item_lookup, cluster_lookup)

        conservative_train_out = out_dir / "conservative_train.jsonl"
        conservative_eval_out = out_dir / "conservative_eval.jsonl"
        hard_train_out = out_dir / "hard_train.jsonl"
        hard_eval_out = out_dir / "hard_eval.jsonl"
        write_jsonl(conservative_train_out, conservative_train_pairs)
        write_jsonl(conservative_eval_out, conservative_eval_pairs)
        write_jsonl(hard_train_out, hard_train_pairs)
        write_jsonl(hard_eval_out, hard_eval_pairs)

        audit = {
            "source_stage11_dataset_run": str(source_11),
            "source_mode": source_mode,
            "buckets": buckets,
            "pool_audit": {
                "train": build_pointwise_audit(train_rows),
                "eval": build_pointwise_audit(eval_rows),
                "all": build_pointwise_audit(train_rows + eval_rows),
            },
            "pairwise_audit": {
                "conservative": {
                    "train": conservative_train_audit,
                    "eval": conservative_eval_audit,
                },
                "hard": {
                    "train": hard_train_audit,
                    "eval": hard_eval_audit,
                },
            },
            "pairwise_examples": {
                "conservative_train": conservative_train_pairs[:3],
                "conservative_eval": conservative_eval_pairs[:3],
                "hard_train": hard_train_pairs[:3],
                "hard_eval": hard_eval_pairs[:3],
            },
        }
        payload = {
            "run_id": run_id,
            "run_tag": RUN_TAG,
            "source_stage11_dataset_run": str(source_11),
            "source_mode": source_mode,
            "buckets": buckets,
            "file_summary": file_summary,
            "config": {
                "conservative_max_pairs_per_user": int(DPO_MAX_PAIRS_PER_USER),
                "hard_max_pairs_per_user": int(DPO_HARD_MAX_PAIRS_PER_USER),
                "strip_rank_features": bool(DPO_STRIP_RANK_FEATURES),
                "seed": int(SEED),
                "prompt_mode": PROMPT_MODE,
            },
            "outputs": {
                "conservative_train": str(conservative_train_out),
                "conservative_eval": str(conservative_eval_out),
                "hard_train": str(hard_train_out),
                "hard_eval": str(hard_eval_out),
                "audit": str(audit_out),
            },
            "pair_counts": {
                "conservative_train": int(len(conservative_train_pairs)),
                "conservative_eval": int(len(conservative_eval_pairs)),
                "hard_train": int(len(hard_train_pairs)),
                "hard_eval": int(len(hard_eval_pairs)),
            },
        }
        print(f"[DATA] source_mode={source_mode} raw_train rows={len(train_rows)}, raw_eval rows={len(eval_rows)}")
        print(
            f"[DATA] conservative_train rows={len(conservative_train_pairs)}, "
            f"conservative_eval rows={len(conservative_eval_pairs)}, "
            f"hard_train rows={len(hard_train_pairs)}, hard_eval rows={len(hard_eval_pairs)}"
        )
    elif source_mode == "rich_sft":
        train_pairs, train_pair_audit = build_rich_sft_dpo_pairs(
            train_rows,
            DPO_MAX_PAIRS_PER_USER,
            SEED,
            true_max_pairs_per_user=DPO_TRUE_MAX_PAIRS_PER_USER,
            valid_max_pairs_per_user=DPO_VALID_MAX_PAIRS_PER_USER,
            hist_max_pairs_per_user=DPO_HIST_MAX_PAIRS_PER_USER,
            allow_mid_neg=DPO_ALLOW_MID_NEG,
        )
        eval_pairs, eval_pair_audit = (
            build_rich_sft_dpo_pairs(
                eval_rows,
                DPO_MAX_PAIRS_PER_USER,
                SEED + 1,
                true_max_pairs_per_user=max(1, DPO_TRUE_MAX_PAIRS_PER_USER),
                valid_max_pairs_per_user=max(0, DPO_VALID_MAX_PAIRS_PER_USER),
                hist_max_pairs_per_user=max(0, DPO_HIST_MAX_PAIRS_PER_USER),
                allow_mid_neg=DPO_ALLOW_MID_NEG,
            )
            if eval_rows
            else ([], {})
        )
        train_out = out_dir / "pairwise_train.jsonl"
        eval_out = out_dir / "pairwise_eval.jsonl"
        write_jsonl(train_out, train_pairs)
        write_jsonl(eval_out, eval_pairs)

        audit = {
            "source_stage11_dataset_run": str(source_11),
            "source_mode": source_mode,
            "buckets": buckets,
            "pointwise_audit": {
                "train": build_pointwise_audit(train_rows),
                "eval": build_pointwise_audit(eval_rows),
                "all": build_pointwise_audit(train_rows + eval_rows),
            },
            "pairwise_audit": {
                "train": train_pair_audit,
                "eval": eval_pair_audit,
            },
            "pairwise_examples": {
                "train": train_pairs[:3],
                "eval": eval_pairs[:3],
            },
        }
        payload = {
            "run_id": run_id,
            "run_tag": RUN_TAG,
            "source_stage11_dataset_run": str(source_11),
            "source_mode": source_mode,
            "buckets": buckets,
            "file_summary": file_summary,
            "config": {
                "max_pairs_per_user": int(DPO_MAX_PAIRS_PER_USER),
                "true_max_pairs_per_user": int(DPO_TRUE_MAX_PAIRS_PER_USER),
                "valid_max_pairs_per_user": int(DPO_VALID_MAX_PAIRS_PER_USER),
                "hist_max_pairs_per_user": int(DPO_HIST_MAX_PAIRS_PER_USER),
                "allow_mid_neg": bool(DPO_ALLOW_MID_NEG),
                "strip_rank_features": bool(DPO_STRIP_RANK_FEATURES),
                "seed": int(SEED),
                "prompt_mode": PROMPT_MODE,
            },
            "outputs": {
                "pairwise_train": str(train_out),
                "pairwise_eval": str(eval_out),
                "audit": str(audit_out),
            },
            "pair_counts": {
                "train": int(len(train_pairs)),
                "eval": int(len(eval_pairs)),
            },
        }
        print(f"[DATA] source_mode=rich_sft raw_train rows={len(train_rows)}, raw_eval rows={len(eval_rows)}")
        print(f"[DATA] pairwise_train rows={len(train_pairs)}, pairwise_eval rows={len(eval_pairs)}")
    else:
        train_pairs, train_pair_audit = build_dpo_pairs(
            train_rows,
            DPO_MAX_PAIRS_PER_USER,
            SEED,
            prefer_easy_neg=DPO_PREFER_EASY_NEG,
            filter_inverted=DPO_FILTER_INVERTED,
        )
        eval_pairs, eval_pair_audit = (
            build_dpo_pairs(
                eval_rows,
                max(2, DPO_MAX_PAIRS_PER_USER // 2),
                SEED + 1,
                prefer_easy_neg=DPO_PREFER_EASY_NEG,
                filter_inverted=DPO_FILTER_INVERTED,
            )
            if eval_rows
            else ([], {})
        )
        train_out = out_dir / "pairwise_train.jsonl"
        eval_out = out_dir / "pairwise_eval.jsonl"
        write_jsonl(train_out, train_pairs)
        write_jsonl(eval_out, eval_pairs)

        audit = {
            "source_stage11_dataset_run": str(source_11),
            "source_mode": source_mode,
            "buckets": buckets,
            "pointwise_audit": {
                "train": build_pointwise_audit(train_rows),
                "eval": build_pointwise_audit(eval_rows),
                "all": build_pointwise_audit(train_rows + eval_rows),
            },
            "pairwise_audit": {
                "train": train_pair_audit,
                "eval": eval_pair_audit,
            },
            "pairwise_examples": {
                "train": train_pairs[:3],
                "eval": eval_pairs[:3],
            },
        }
        payload = {
            "run_id": run_id,
            "run_tag": RUN_TAG,
            "source_stage11_dataset_run": str(source_11),
            "source_mode": source_mode,
            "buckets": buckets,
            "file_summary": file_summary,
            "config": {
                "max_pairs_per_user": int(DPO_MAX_PAIRS_PER_USER),
                "prefer_easy_neg": bool(DPO_PREFER_EASY_NEG),
                "filter_inverted": bool(DPO_FILTER_INVERTED),
                "strip_rank_features": bool(DPO_STRIP_RANK_FEATURES),
                "seed": int(SEED),
            },
            "outputs": {
                "pairwise_train": str(train_out),
                "pairwise_eval": str(eval_out),
                "audit": str(audit_out),
            },
            "pair_counts": {
                "train": int(len(train_pairs)),
                "eval": int(len(eval_pairs)),
            },
        }
        print(f"[DATA] source_mode=pointwise raw_train rows={len(train_rows)}, raw_eval rows={len(eval_rows)}")
        print(f"[DATA] pairwise_train rows={len(train_pairs)}, pairwise_eval rows={len(eval_pairs)}")

    audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[DONE] audit: {audit_out}")
    print(f"[DONE] run_meta: {meta_out}")


if __name__ == "__main__":
    main()
