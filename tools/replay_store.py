from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUEST_ID_TEMPLATE = "stage11_b5_u{user_idx:06d}"
REQUEST_ID_RE = re.compile(r"^stage11_b5_u(?P<user_idx>\d+)$")

PRIMARY_SCORE_PATHS = [
    REPO_ROOT
    / "data"
    / "output"
    / "_prod_runs"
    / "stage11_freeze_pack_20260409"
    / "stage11_freeze_pack_20260409"
    / "bucket5"
    / "v124_triband_joint12_gate_top100"
    / "bucket_5_scores.csv",
    REPO_ROOT
    / "data"
    / "output"
    / "cloud_stage11"
    / "stage11_v124_eval_full"
    / "20260409_011908_stage11_3_qlora_sidecar_eval"
    / "bucket_5_scores.csv",
]

PRIMARY_RUN_META_PATHS = [
    REPO_ROOT
    / "data"
    / "output"
    / "_prod_runs"
    / "stage11_freeze_pack_20260409"
    / "stage11_freeze_pack_20260409"
    / "bucket5"
    / "v124_triband_joint12_gate_top100"
    / "run_meta.json",
    REPO_ROOT
    / "data"
    / "output"
    / "cloud_stage11"
    / "stage11_v124_eval_full"
    / "20260409_011908_stage11_3_qlora_sidecar_eval"
    / "run_meta.json",
]

PRIMARY_ITEM_MAP_PATHS = [
    REPO_ROOT
    / "data"
    / "output"
    / "09_index_maps"
    / "20260321_194022_full_stage09_index_maps_build"
    / "bucket_5"
    / "item_index_map.parquet",
]

PRIMARY_USER_MAP_PATHS = [
    REPO_ROOT
    / "data"
    / "output"
    / "09_index_maps"
    / "20260321_194022_full_stage09_index_maps_build"
    / "bucket_5"
    / "user_index_map.parquet",
]

BUSINESS_ROOT = REPO_ROOT / "data" / "parquet" / "yelp_academic_dataset_business"
USER_PROFILE_ROOT = REPO_ROOT / "data" / "output" / "09_user_profiles"
USER_TEXT_VIEWS_ROOT = REPO_ROOT / "data" / "output" / "12_user_nonllm_text_views_v1"


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"no existing path found in candidates: {[str(path) for path in paths]}")


def _resolve_item_map_path() -> Path:
    for path in PRIMARY_ITEM_MAP_PATHS:
        if path.exists():
            return path
    matches = sorted((REPO_ROOT / "data" / "output" / "09_index_maps").rglob("item_index_map.parquet"))
    if not matches:
        raise FileNotFoundError("item_index_map.parquet not found under data/output/09_index_maps")
    return matches[-1]


def _resolve_user_map_path() -> Path:
    for path in PRIMARY_USER_MAP_PATHS:
        if path.exists():
            return path
    matches = sorted((REPO_ROOT / "data" / "output" / "09_index_maps").rglob("user_index_map.parquet"))
    if not matches:
        raise FileNotFoundError("user_index_map.parquet not found under data/output/09_index_maps")
    return matches[-1]


def _resolve_user_profile_table_path() -> Path:
    matches = sorted(USER_PROFILE_ROOT.rglob("user_profiles.csv"))
    if not matches:
        raise FileNotFoundError("user_profiles.csv not found under data/output/09_user_profiles")
    return matches[-1]


def _resolve_user_text_views_path() -> Path:
    matches = sorted(USER_TEXT_VIEWS_ROOT.rglob("user_text_views_v1.parquet"))
    if not matches:
        raise FileNotFoundError("user_text_views_v1.parquet not found under data/output/12_user_nonllm_text_views_v1")
    return matches[-1]


def build_request_id(user_idx: int) -> str:
    return REQUEST_ID_TEMPLATE.format(user_idx=int(user_idx))


def parse_request_id(request_id: str) -> int:
    match = REQUEST_ID_RE.match(str(request_id).strip())
    if match is None:
        raise ValueError(f"unsupported replay request_id: {request_id!r}")
    return int(match.group("user_idx"))


@dataclass(frozen=True)
class ReplayPaths:
    score_csv: Path
    run_meta_json: Path
    item_map_parquet: Path
    user_map_parquet: Path
    user_profile_csv: Path
    user_text_views_parquet: Path
    business_parquet_root: Path


@dataclass(frozen=True)
class ReplayStore:
    paths: ReplayPaths
    score_frame: pd.DataFrame
    request_ids: list[str]
    request_index: dict[str, int]
    sample_request_id: str
    cohort_note: str
    remote_score_csv: str
    remote_pairwise_dir: str
    remote_base_model: str
    remote_adapter_dir_11_30: str
    remote_adapter_dir_31_60: str
    remote_adapter_dir_61_100: str

    def get_request_frame(self, request_id: str) -> pd.DataFrame:
        user_idx = self.request_index.get(str(request_id).strip())
        if user_idx is None:
            user_idx = parse_request_id(request_id)
        user_pdf = self.score_frame[self.score_frame["user_idx"].eq(int(user_idx))].copy()
        if user_pdf.empty:
            raise KeyError(f"request_id not found in replay store: {request_id}")
        return user_pdf


def _sample_paths() -> ReplayPaths:
    support_root = REPO_ROOT / "data" / "output" / "current_release" / "test_support"
    return ReplayPaths(
        score_csv=support_root / "embedded_stage11_replay_sample",
        run_meta_json=REPO_ROOT
        / "data"
        / "output"
        / "current_release"
        / "stage11"
        / "eval"
        / "bucket5_tri_band_freeze_v124_alpha036"
        / "summary.json",
        item_map_parquet=support_root / "embedded_item_map",
        user_map_parquet=support_root / "embedded_user_map",
        user_profile_csv=support_root / "embedded_user_profile",
        user_text_views_parquet=support_root / "embedded_user_text_views",
        business_parquet_root=support_root / "embedded_business",
    )


def _build_sample_replay_frame() -> pd.DataFrame:
    users = [
        (97, "demo_user_97", "New Orleans", "ramen|japanese|seafood", 58),
        (1072, "demo_user_1072", "New Orleans", "ramen|japanese|asian fusion", 58),
        (1940, "demo_user_1940", "New Orleans", "brunch|southern|coffee", 72),
        (5508, "demo_user_5508", "New Orleans", "seafood|cajun|sandwich", 83),
        (6562, "demo_user_6562", "New Orleans", "pizza|italian|small plates", 91),
    ]
    names = [
        "Katie's Restaurant & Bar",
        "High Hat Cafe",
        "Pizza Delicious",
        "Domenica",
        "Willa Jean",
        "Bayona",
        "Cochon",
        "Mid-City Ramen Lab",
        "Bywater American Bistro",
        "Napoleon House",
    ]
    rows: list[dict[str, Any]] = []
    for user_idx, user_id, city, keywords, truth_item_idx in users:
        for rank in range(1, 101):
            item_idx = truth_item_idx if rank == 17 else user_idx * 1000 + rank
            learned_rank = 17 if item_idx == truth_item_idx else rank if rank < 17 else rank + 1
            blend_rank = 8 if item_idx == truth_item_idx else rank if rank < 8 else rank + 1
            pre_rank = learned_rank
            is_truth = item_idx == truth_item_idx
            category = keywords.split("|")[(rank - 1) % len(keywords.split("|"))]
            rows.append(
                {
                    "user_idx": user_idx,
                    "item_idx": item_idx,
                    "pre_rank": pre_rank,
                    "learned_rank": learned_rank,
                    "blend_rank": blend_rank,
                    "label_true": 1 if is_truth else 0,
                    "sidecar_score_present": 1 if 11 <= pre_rank <= 100 else 0,
                    "pre_score": round(max(0.05, 1.0 - pre_rank * 0.006), 4),
                    "baseline_score": round(max(0.05, 1.0 - pre_rank * 0.006), 4),
                    "learned_blend_score": round(max(0.05, 1.0 - learned_rank * 0.007), 4),
                    "reward_score": 13.4375 if is_truth else round(8.0 - min(pre_rank, 100) * 0.025, 4),
                    "rescue_bonus": 0.7579 if is_truth else round(max(0.0, 0.24 - pre_rank * 0.002), 4),
                    "final_score": round(max(0.05, 1.0 - blend_rank * 0.007), 4),
                    "joint_bonus_effective": 0.7579 if is_truth else 0.0,
                    "joint_gate_pass": bool(is_truth or pre_rank <= 30),
                    "business_id": f"sample_biz_{item_idx}",
                    "user_id": user_id,
                    "tier": "mid",
                    "n_train": 12,
                    "profile_confidence": 0.82,
                    "profile_keywords": keywords,
                    "profile_top_pos_tags": keywords.replace("|", ","),
                    "profile_top_neg_tags": "slow service",
                    "profile_text_short": f"User prefers {keywords.replace('|', ', ')} restaurants around {city}.",
                    "user_view_quality_tier": "sample",
                    "user_quality_signal_count": 8,
                    "user_long_pref_text": f"Long-term preference for {keywords.replace('|', ', ')} with strong local relevance.",
                    "user_recent_intent_text": f"Recently interacted with {category} restaurants.",
                    "user_negative_avoid_text": "Avoids slow service and weak food quality.",
                    "user_context_text": f"Homepage recommendation replay in {city}.",
                    "user_pos_tags_json": json.dumps(keywords.split("|")),
                    "user_neg_tags_json": json.dumps(["slow service"]),
                    "long_term_top_cuisine": keywords.split("|")[0],
                    "recent_top_cuisine": category,
                    "negative_top_cuisine": "",
                    "negative_pressure": 0.1,
                    "top_city": city,
                    "top_geo_cell_3dp": "29.951,-90.072",
                    "nonllm_source": "embedded_sample_replay",
                    "name": names[(rank - 1) % len(names)] if not is_truth else "Mid-City Ramen Lab",
                    "city": city,
                    "categories": f"Restaurants, {category.title()}",
                    "route_band": _route_band_label(pre_rank),
                }
            )
    sample = pd.DataFrame(rows)
    sample["request_id"] = sample["user_idx"].map(build_request_id)
    sample["stage11_final_rank"] = sample["blend_rank"].astype(int)
    sample["stage11_band_label"] = sample["pre_rank"].map(_route_band_label)
    return sample.sort_values(["user_idx", "stage11_final_rank", "item_idx"], kind="stable").reset_index(drop=True)


def _load_sample_replay_store() -> ReplayStore:
    score_frame = _build_sample_replay_frame()
    request_ids = sorted(str(item) for item in score_frame["request_id"].drop_duplicates().tolist())
    request_index = {request_id: parse_request_id(request_id) for request_id in request_ids}
    sample_request_id = build_request_id(1072)
    return ReplayStore(
        paths=_sample_paths(),
        score_frame=score_frame,
        request_ids=request_ids,
        request_index=request_index,
        sample_request_id=sample_request_id,
        cohort_note="embedded sample replay cohort for public demo tests; real frozen replay pack is used when local artifacts are available",
        remote_score_csv="",
        remote_pairwise_dir="",
        remote_base_model="",
        remote_adapter_dir_11_30="",
        remote_adapter_dir_31_60="",
        remote_adapter_dir_61_100="",
    )


def _load_business_metadata(path: Path) -> pd.DataFrame:
    business_pdf = pd.read_parquet(
        path,
        columns=["business_id", "name", "city", "categories"],
    )
    business_pdf["business_id"] = business_pdf["business_id"].astype(str)
    business_pdf["name"] = business_pdf["name"].fillna("").astype(str)
    business_pdf["city"] = business_pdf["city"].fillna("").astype(str)
    business_pdf["categories"] = business_pdf["categories"].fillna("").astype(str)
    return business_pdf.drop_duplicates(["business_id"])


def _route_band_label(pre_rank: Any) -> str:
    try:
        rank = int(pre_rank)
    except Exception:
        return "outside_window"
    if 11 <= rank <= 30:
        return "11-30"
    if 31 <= rank <= 40:
        return "31-40"
    if 41 <= rank <= 60:
        return "41-60"
    if 61 <= rank <= 100:
        return "61-100"
    if rank <= 10:
        return "top10_anchor"
    return "outside_window"


def _sample_request_id(score_pdf: pd.DataFrame) -> str:
    ranked = score_pdf.copy()
    ranked["rescued_into_top5"] = (
        ranked["blend_rank"].lt(ranked["learned_rank"]) & ranked["blend_rank"].le(5)
    ).astype(int)
    by_user = (
        ranked.groupby("user_idx", as_index=False)
        .agg(
            rescued_into_top5=("rescued_into_top5", "sum"),
            best_blend_rank=("blend_rank", "min"),
        )
        .sort_values(["rescued_into_top5", "best_blend_rank", "user_idx"], ascending=[False, True, True])
    )
    if by_user.empty:
        raise RuntimeError("stage11 replay store contains no users")
    return build_request_id(int(by_user.iloc[0]["user_idx"]))


def resolve_replay_paths() -> ReplayPaths:
    return ReplayPaths(
        score_csv=_first_existing(PRIMARY_SCORE_PATHS),
        run_meta_json=_first_existing(PRIMARY_RUN_META_PATHS),
        item_map_parquet=_resolve_item_map_path(),
        user_map_parquet=_resolve_user_map_path(),
        user_profile_csv=_resolve_user_profile_table_path(),
        user_text_views_parquet=_resolve_user_text_views_path(),
        business_parquet_root=BUSINESS_ROOT,
    )


@lru_cache(maxsize=1)
def load_replay_store() -> ReplayStore:
    try:
        paths = resolve_replay_paths()
    except FileNotFoundError:
        return _load_sample_replay_store()
    score_pdf = pd.read_csv(paths.score_csv)
    item_map_pdf = pd.read_parquet(paths.item_map_parquet)
    user_map_pdf = pd.read_parquet(paths.user_map_parquet)
    user_profile_pdf = pd.read_csv(paths.user_profile_csv)
    user_text_views_pdf = pd.read_parquet(
        paths.user_text_views_parquet,
        columns=[
            "user_id",
            "user_view_quality_tier",
            "user_quality_signal_count",
            "user_long_pref_text",
            "user_recent_intent_text",
            "user_negative_avoid_text",
            "user_context_text",
            "user_pos_tags_json",
            "user_neg_tags_json",
            "long_term_top_cuisine",
            "recent_top_cuisine",
            "negative_top_cuisine",
            "negative_pressure",
            "top_city",
            "top_geo_cell_3dp",
            "nonllm_source",
        ],
    )
    business_pdf = _load_business_metadata(paths.business_parquet_root)

    item_map_pdf["item_idx"] = pd.to_numeric(item_map_pdf["item_idx"], errors="coerce").fillna(-1).astype(int)
    item_map_pdf["business_id"] = item_map_pdf["business_id"].astype(str)
    user_map_pdf["user_idx"] = pd.to_numeric(user_map_pdf["user_idx"], errors="coerce").fillna(-1).astype(int)
    user_map_pdf["user_id"] = user_map_pdf["user_id"].astype(str)
    profile_cols = [
        "user_id",
        "n_train",
        "tier",
        "profile_confidence",
        "profile_keywords",
        "profile_top_pos_tags",
        "profile_top_neg_tags",
        "profile_text_short",
    ]
    user_profile_pdf = user_profile_pdf[profile_cols].copy()
    user_profile_pdf["user_id"] = user_profile_pdf["user_id"].astype(str)
    user_profile_pdf["n_train"] = pd.to_numeric(user_profile_pdf["n_train"], errors="coerce")
    user_profile_pdf["profile_confidence"] = pd.to_numeric(user_profile_pdf["profile_confidence"], errors="coerce")
    for col in ("tier", "profile_keywords", "profile_top_pos_tags", "profile_top_neg_tags", "profile_text_short"):
        user_profile_pdf[col] = user_profile_pdf[col].fillna("").astype(str)
    user_text_views_pdf["user_id"] = user_text_views_pdf["user_id"].astype(str)
    user_text_views_pdf["user_quality_signal_count"] = pd.to_numeric(
        user_text_views_pdf["user_quality_signal_count"], errors="coerce"
    )
    user_text_views_pdf["negative_pressure"] = pd.to_numeric(user_text_views_pdf["negative_pressure"], errors="coerce")
    for col in (
        "user_view_quality_tier",
        "user_long_pref_text",
        "user_recent_intent_text",
        "user_negative_avoid_text",
        "user_context_text",
        "user_pos_tags_json",
        "user_neg_tags_json",
        "long_term_top_cuisine",
        "recent_top_cuisine",
        "negative_top_cuisine",
        "top_city",
        "top_geo_cell_3dp",
        "nonllm_source",
    ):
        user_text_views_pdf[col] = user_text_views_pdf[col].fillna("").astype(str)

    score_pdf["user_idx"] = pd.to_numeric(score_pdf["user_idx"], errors="coerce").fillna(-1).astype(int)
    score_pdf["item_idx"] = pd.to_numeric(score_pdf["item_idx"], errors="coerce").fillna(-1).astype(int)
    for col in (
        "pre_rank",
        "learned_rank",
        "blend_rank",
        "label_true",
        "sidecar_score_present",
    ):
        if col in score_pdf.columns:
            score_pdf[col] = pd.to_numeric(score_pdf[col], errors="coerce").fillna(-1).astype(int)
    for col in (
        "pre_score",
        "baseline_score",
        "learned_blend_score",
        "reward_score",
        "rescue_bonus",
        "final_score",
        "joint_bonus_effective",
    ):
        if col in score_pdf.columns:
            score_pdf[col] = pd.to_numeric(score_pdf[col], errors="coerce")
    if "joint_gate_pass" in score_pdf.columns:
        score_pdf["joint_gate_pass"] = score_pdf["joint_gate_pass"].fillna(False).astype(bool)

    merged = (
        score_pdf.merge(item_map_pdf, on="item_idx", how="left")
        .merge(user_map_pdf, on="user_idx", how="left")
        .merge(user_profile_pdf, on="user_id", how="left")
        .merge(user_text_views_pdf, on="user_id", how="left")
        .merge(business_pdf, on="business_id", how="left")
        .copy()
    )
    merged["business_id"] = merged["business_id"].fillna("").astype(str)
    merged["user_id"] = merged["user_id"].fillna("").astype(str)
    merged["tier"] = merged["tier"].fillna("").astype(str)
    merged["profile_keywords"] = merged["profile_keywords"].fillna("").astype(str)
    merged["profile_top_pos_tags"] = merged["profile_top_pos_tags"].fillna("").astype(str)
    merged["profile_top_neg_tags"] = merged["profile_top_neg_tags"].fillna("").astype(str)
    merged["profile_text_short"] = merged["profile_text_short"].fillna("").astype(str)
    merged["user_view_quality_tier"] = merged["user_view_quality_tier"].fillna("").astype(str)
    merged["user_long_pref_text"] = merged["user_long_pref_text"].fillna("").astype(str)
    merged["user_recent_intent_text"] = merged["user_recent_intent_text"].fillna("").astype(str)
    merged["user_negative_avoid_text"] = merged["user_negative_avoid_text"].fillna("").astype(str)
    merged["user_context_text"] = merged["user_context_text"].fillna("").astype(str)
    merged["user_pos_tags_json"] = merged["user_pos_tags_json"].fillna("").astype(str)
    merged["user_neg_tags_json"] = merged["user_neg_tags_json"].fillna("").astype(str)
    merged["long_term_top_cuisine"] = merged["long_term_top_cuisine"].fillna("").astype(str)
    merged["recent_top_cuisine"] = merged["recent_top_cuisine"].fillna("").astype(str)
    merged["negative_top_cuisine"] = merged["negative_top_cuisine"].fillna("").astype(str)
    merged["top_city"] = merged["top_city"].fillna("").astype(str)
    merged["top_geo_cell_3dp"] = merged["top_geo_cell_3dp"].fillna("").astype(str)
    merged["nonllm_source"] = merged["nonllm_source"].fillna("").astype(str)
    merged["name"] = merged["name"].fillna("").astype(str)
    merged["city"] = merged["city"].fillna("").astype(str)
    merged["categories"] = merged["categories"].fillna("").astype(str)
    merged["request_id"] = merged["user_idx"].map(build_request_id)
    merged["stage11_final_rank"] = pd.to_numeric(merged.get("blend_rank"), errors="coerce").fillna(999999).astype(int)
    merged["stage11_band_label"] = merged["pre_rank"].map(_route_band_label)
    merged = merged.sort_values(["user_idx", "stage11_final_rank", "item_idx"], kind="stable").reset_index(drop=True)

    run_meta = json.loads(paths.run_meta_json.read_text(encoding="utf-8"))
    metrics_path = Path(str(run_meta.get("metrics_file", "")).strip())
    remote_score_csv = ""
    if metrics_path.name:
        remote_score_csv = metrics_path.with_name("bucket_5_scores.csv").as_posix()
    remote_pairwise_dir = (
        Path(str(run_meta.get("source_run_11_1_data", "")).strip()) / "bucket_5" / "pairwise_pool_all_parquet"
    ).as_posix()

    request_ids = sorted(str(item) for item in merged["request_id"].drop_duplicates().tolist())
    request_index = {request_id: parse_request_id(request_id) for request_id in request_ids}
    sample_request_id = _sample_request_id(merged)
    cohort_users = int(run_meta.get("bucket_run_summaries", [{}])[0].get("n_users", 0) or 0)
    cohort_note = (
        f"bucket5 rescue replay cohort from frozen v124 tri-band eval ({cohort_users} users, stage10 true-rank 11-100 subset)"
    )

    return ReplayStore(
        paths=paths,
        score_frame=merged,
        request_ids=request_ids,
        request_index=request_index,
        sample_request_id=sample_request_id,
        cohort_note=cohort_note,
        remote_score_csv=remote_score_csv,
        remote_pairwise_dir=remote_pairwise_dir,
        remote_base_model=str(run_meta.get("base_model", "")).strip(),
        remote_adapter_dir_11_30=str(run_meta.get("adapter_dir_11_30", run_meta.get("adapter_dir", ""))).strip(),
        remote_adapter_dir_31_60=str(run_meta.get("adapter_dir_31_60", "")).strip(),
        remote_adapter_dir_61_100=str(run_meta.get("adapter_dir_61_100", "")).strip(),
    )
