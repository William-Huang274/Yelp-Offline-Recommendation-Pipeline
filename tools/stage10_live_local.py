from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


REPO_ROOT = Path(__file__).resolve().parents[1]

PRIMARY_STAGE09_BUCKET_DIRS = [
    REPO_ROOT
    / "data"
    / "output"
    / "09_candidate_fusion_structural_v5_sourceparity"
    / "20260324_030511_full_stage09_candidate_fusion"
    / "bucket_5",
    REPO_ROOT
    / "data"
    / "output"
    / "phase0_repair"
    / "09_candidate_fusion"
    / "20260320_165951_full_stage09_candidate_fusion"
    / "bucket_5",
]

PRIMARY_STAGE10_MODEL_JSON_PATHS = [
    REPO_ROOT
    / "data"
    / "output"
    / "10_rank_models"
    / "20260320_173236_stage10_1_rank_train"
    / "rank_model.json",
]

BACKEND_COMPARE_SUMMARY_PATH = (
    REPO_ROOT / "data" / "output" / "current_release" / "stage10" / "bucket5_mainline" / "backend_compare_summary.csv"
)

CURRENT_STAGE09_RELEASE_BUCKET_DIR = (
    REPO_ROOT
    / "data"
    / "output"
    / "09_candidate_fusion_structural_v5_sourceparity"
    / "20260324_030511_full_stage09_candidate_fusion"
    / "bucket_5"
)

ROUTE_KEYS = ("als", "cluster", "profile", "popular")
CANDIDATE_TEXT_COLUMNS = ("business_id", "name", "city", "categories", "primary_category", "user_segment")
CANDIDATE_NUMERIC_COLUMNS = (
    "item_idx",
    "pre_rank",
    "als_rank",
    "cluster_rank",
    "popular_rank",
    "pre_score",
    "signal_score",
    "quality_score",
    "semantic_score",
    "semantic_confidence",
    "semantic_support",
    "semantic_support_adj",
    "semantic_tag_richness",
    "tower_score",
    "seq_score",
    "tower_inv",
    "seq_inv",
    "item_train_pop_count",
    "user_train_count",
)


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"no existing path found in candidates: {[str(path) for path in paths]}")


def _safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _normalize_candidate_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    for col in CANDIDATE_TEXT_COLUMNS:
        if col in pdf.columns:
            pdf[col] = pdf[col].fillna("").astype(str)
    for col in CANDIDATE_NUMERIC_COLUMNS:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
        else:
            pdf[col] = np.nan
    pdf["pre_score"] = pdf["pre_score"].fillna(0.0)
    pdf["signal_score"] = pdf["signal_score"].fillna(0.0)
    pdf["quality_score"] = pdf["quality_score"].fillna(0.0)
    pdf["semantic_score"] = pdf["semantic_score"].fillna(0.0)
    pdf["semantic_confidence"] = pdf["semantic_confidence"].fillna(0.0)
    pdf["semantic_support"] = pdf["semantic_support"].fillna(0.0)
    pdf["semantic_support_adj"] = pdf["semantic_support_adj"].fillna(pdf["semantic_support"]).fillna(0.0)
    pdf["semantic_tag_richness"] = pdf["semantic_tag_richness"].fillna(0.0)
    pdf["tower_score"] = pdf["tower_score"].fillna(0.0)
    pdf["seq_score"] = pdf["seq_score"].fillna(0.0)
    pdf["tower_inv"] = pdf["tower_inv"].fillna(0.0)
    pdf["seq_inv"] = pdf["seq_inv"].fillna(0.0)
    pdf["item_train_pop_count"] = pdf["item_train_pop_count"].fillna(0.0)
    pdf["user_train_count"] = pdf["user_train_count"].fillna(0.0)
    pdf["semantic_support_log"] = np.log1p(pdf["semantic_support_adj"].clip(lower=0.0).to_numpy(dtype=np.float64))
    pdf["item_pop_log"] = np.log1p(pdf["item_train_pop_count"].clip(lower=0.0).to_numpy(dtype=np.float64))
    pdf["user_train_log"] = np.log1p(pdf["user_train_count"].clip(lower=0.0).to_numpy(dtype=np.float64))
    pdf["inv_pre_rank"] = _inv_rank(pdf["pre_rank"])
    pdf["inv_als_rank"] = _inv_rank(pdf["als_rank"])
    pdf["inv_cluster_rank"] = _inv_rank(pdf["cluster_rank"])
    pdf["inv_popular_rank"] = _inv_rank(pdf["popular_rank"])
    source_series = pdf["source_set"] if "source_set" in pdf.columns else pd.Series([()] * len(pdf), index=pdf.index)
    pdf["has_als"] = source_series.map(lambda value: _contains_route(value, "als"))
    pdf["has_cluster"] = source_series.map(lambda value: _contains_route(value, "cluster"))
    pdf["has_profile"] = source_series.map(lambda value: _contains_route(value, "profile"))
    pdf["has_popular"] = source_series.map(lambda value: _contains_route(value, "popular"))
    pdf["source_count"] = source_series.map(_route_count)
    segment_series = pdf["user_segment"] if "user_segment" in pdf.columns else pd.Series([""] * len(pdf), index=pdf.index)
    pdf["is_light"] = segment_series.eq("light").astype(float)
    pdf["is_mid"] = segment_series.eq("mid").astype(float)
    pdf["is_heavy"] = segment_series.eq("heavy").astype(float)
    return pdf


@dataclass(frozen=True)
class Stage09LivePaths:
    requested_release_bucket_dir: Path
    effective_bucket_dir: Path
    candidate_parquet: Path
    history_parquet: Path
    source_alignment: str


@dataclass(frozen=True)
class Stage10LiveAssets:
    stage09_paths: Stage09LivePaths
    model_json: Path
    model_path: Path
    model_backend: str
    feature_columns: tuple[str, ...]
    blend_alpha: float
    blend_mode: str
    route_weights: dict[str, float]
    route_gamma: float
    history_short_days: int
    history_mid_days: int
    history_max_short: int
    history_max_mid: int
    history_max_old: int
    model: Any


def resolve_stage09_live_paths() -> Stage09LivePaths:
    effective_bucket_dir: Path | None = None
    candidate_parquet: Path | None = None
    history_parquet: Path | None = None
    for bucket_dir in PRIMARY_STAGE09_BUCKET_DIRS:
        candidate_probe = bucket_dir / "candidates_pretrim150.parquet"
        if candidate_probe.exists():
            effective_bucket_dir = bucket_dir
            candidate_parquet = candidate_probe
            history_parquet = bucket_dir / "train_history.parquet"
            break
    if effective_bucket_dir is None or candidate_parquet is None or history_parquet is None:
        probe_paths = [str(path / "candidates_pretrim150.parquet") for path in PRIMARY_STAGE09_BUCKET_DIRS]
        raise FileNotFoundError(f"stage09 live candidate parquet missing in candidates: {probe_paths}")
    source_alignment = (
        "release_sourceparity"
        if effective_bucket_dir.resolve() == CURRENT_STAGE09_RELEASE_BUCKET_DIR.resolve()
        else "fallback_local_candidate_pack"
    )
    return Stage09LivePaths(
        requested_release_bucket_dir=CURRENT_STAGE09_RELEASE_BUCKET_DIR,
        effective_bucket_dir=effective_bucket_dir,
        candidate_parquet=candidate_parquet,
        history_parquet=history_parquet,
        source_alignment=source_alignment,
    )


def _resolve_stage10_model_json() -> Path:
    return _first_existing(PRIMARY_STAGE10_MODEL_JSON_PATHS)


def _resolve_bucket5_blend_alpha(default: float = 0.15) -> float:
    if not BACKEND_COMPARE_SUMMARY_PATH.exists():
        return float(default)
    try:
        summary_pdf = pd.read_csv(BACKEND_COMPARE_SUMMARY_PATH)
    except Exception:
        return float(default)
    if summary_pdf.empty:
        return float(default)
    bucket5 = summary_pdf[summary_pdf["backend"].astype(str).eq("xgboost_cls")]
    if bucket5.empty:
        return float(default)
    try:
        return float(bucket5.iloc[0]["alpha"])
    except Exception:
        return float(default)


@lru_cache(maxsize=1)
def load_stage10_live_assets() -> Stage10LiveAssets:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed for stage10 xgb_live")
    stage09_paths = resolve_stage09_live_paths()
    model_json = _resolve_stage10_model_json()
    model_data = json.loads(model_json.read_text(encoding="utf-8"))
    model_bucket = dict(model_data.get("models_by_bucket", {}).get("5", {}))
    if not model_bucket:
        raise RuntimeError(f"bucket5 model metadata missing in {model_json}")
    model_file = str(model_bucket.get("model_file", "")).strip()
    if not model_file:
        raise RuntimeError(f"bucket5 model_file missing in {model_json}")
    model_path = model_json.parent / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"bucket5 model file missing: {model_path}")
    model = XGBClassifier()
    model.load_model(model_path.as_posix())
    route_quality = model_bucket.get("route_quality_calibration", {})
    bucket_policy = model_bucket.get("bucket_policy_applied", {})
    return Stage10LiveAssets(
        stage09_paths=stage09_paths,
        model_json=model_json,
        model_path=model_path,
        model_backend=str(model_bucket.get("model_backend", "xgboost_cls")).strip().lower(),
        feature_columns=tuple(str(item) for item in model_bucket.get("feature_columns", [])),
        blend_alpha=_resolve_bucket5_blend_alpha(),
        blend_mode="prob",
        route_weights={str(key): float(route_quality.get("weights", {}).get(key, 1.0)) for key in ROUTE_KEYS},
        route_gamma=float(route_quality.get("gamma", 0.25)),
        history_short_days=int(bucket_policy.get("train_window_short_days", 90) or 90),
        history_mid_days=int(bucket_policy.get("train_window_mid_days", 365) or 365),
        history_max_short=int(bucket_policy.get("train_max_short_pos_per_user", 3) or 3),
        history_max_mid=int(bucket_policy.get("train_max_mid_pos_per_user", 4) or 4),
        history_max_old=int(bucket_policy.get("train_max_old_pos_per_user", 3) or 3),
        model=model,
    )


@lru_cache(maxsize=256)
def _load_stage09_user_candidates_cached(user_idx: int) -> pd.DataFrame:
    assets = load_stage10_live_assets()
    pdf = pd.read_parquet(
        assets.stage09_paths.candidate_parquet,
        filters=[[("user_idx", "==", int(user_idx))]],
    )
    if pdf.empty:
        raise KeyError(f"user_idx not found in stage09 live candidate parquet: {user_idx}")
    return _normalize_candidate_frame(pdf)


@lru_cache(maxsize=256)
def _load_history_user_frame_cached(user_idx: int) -> pd.DataFrame:
    assets = load_stage10_live_assets()
    history_path = assets.stage09_paths.history_parquet
    if not history_path.exists():
        return pd.DataFrame(columns=["user_idx", "item_idx", "days_to_test", "hist_ts", "hist_rating"])
    pdf = pd.read_parquet(
        history_path,
        filters=[[("user_idx", "==", int(user_idx))]],
    )
    if pdf.empty:
        return pd.DataFrame(columns=["user_idx", "item_idx", "days_to_test", "hist_ts", "hist_rating"])
    keep_cols = [col for col in ("user_idx", "item_idx", "days_to_test", "hist_ts", "hist_rating") if col in pdf.columns]
    pdf = pdf[keep_cols].copy()
    pdf["days_to_test"] = pd.to_numeric(pdf.get("days_to_test"), errors="coerce")
    pdf["hist_rating"] = pd.to_numeric(pdf.get("hist_rating"), errors="coerce")
    return pdf


def _load_stage09_live_candidates_real(user_idx: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    candidates_pdf = _load_stage09_user_candidates_cached(int(user_idx)).copy()
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    assets = load_stage10_live_assets()
    return candidates_pdf, {
        "timings_ms": {"candidate_lookup": elapsed_ms, "total": elapsed_ms},
        "data_source": _safe_relative(assets.stage09_paths.candidate_parquet),
        "history_source": _safe_relative(assets.stage09_paths.history_parquet),
        "source_alignment": assets.stage09_paths.source_alignment,
        "requested_release_stage09_bucket_dir": _safe_relative(assets.stage09_paths.requested_release_bucket_dir),
        "effective_stage09_bucket_dir": _safe_relative(assets.stage09_paths.effective_bucket_dir),
    }


def _sample_live_candidates(user_idx: int) -> pd.DataFrame:
    from replay_store import build_request_id, load_replay_store

    store = load_replay_store()
    request_id = build_request_id(int(user_idx))
    pdf = store.get_request_frame(request_id).copy()
    pdf["source_set"] = [["profile", "popular"] for _ in range(len(pdf))]
    pdf["signal_score"] = pd.to_numeric(pdf.get("pre_score"), errors="coerce").fillna(0.0)
    pdf["quality_score"] = pd.to_numeric(pdf.get("pre_score"), errors="coerce").fillna(0.0)
    pdf["semantic_score"] = pd.to_numeric(pdf.get("reward_score"), errors="coerce").fillna(0.0) / 20.0
    pdf["semantic_confidence"] = 0.75
    pdf["semantic_support"] = 1.0
    pdf["semantic_support_adj"] = 1.0
    pdf["semantic_tag_richness"] = 0.5
    pdf["tower_score"] = 0.0
    pdf["seq_score"] = 0.0
    pdf["item_train_pop_count"] = 10
    pdf["user_train_count"] = pd.to_numeric(pdf.get("n_train"), errors="coerce").fillna(1)
    pdf["user_segment"] = "mid"
    return _normalize_candidate_frame(pdf)


def load_stage09_live_candidates(user_idx: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        return _load_stage09_live_candidates_real(int(user_idx))
    except Exception:
        started = time.perf_counter()
        sample = _sample_live_candidates(int(user_idx))
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return sample, {
            "timings_ms": {"candidate_lookup": elapsed_ms, "total": elapsed_ms},
            "data_source": "embedded_sample_replay_store",
            "history_source": "embedded_sample_replay_store",
            "source_alignment": "embedded_sample_fixture",
            "requested_release_stage09_bucket_dir": _safe_relative(CURRENT_STAGE09_RELEASE_BUCKET_DIR),
            "effective_stage09_bucket_dir": "embedded_sample_replay_store",
        }


def _cap_history_rows(pdf: pd.DataFrame, *, min_days: int, max_days: int | None, max_rows: int, label: str) -> pd.DataFrame:
    if max_rows <= 0 or pdf.empty:
        return pd.DataFrame(columns=["user_idx", "item_idx", label])
    mask = pdf["days_to_test"].notna() & pdf["days_to_test"].ge(int(min_days))
    if max_days is not None:
        mask = mask & pdf["days_to_test"].le(int(max_days))
    scoped = pdf[mask].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["user_idx", "item_idx", label])
    scoped = scoped.sort_values(
        ["user_idx", "days_to_test", "hist_ts", "hist_rating", "item_idx"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    scoped["_hist_rank"] = scoped.groupby("user_idx").cumcount() + 1
    scoped = scoped[scoped["_hist_rank"].le(int(max_rows))][["user_idx", "item_idx"]].drop_duplicates()
    scoped[label] = 1
    return scoped


def _build_history_flags(history_pdf: pd.DataFrame, assets: Stage10LiveAssets) -> pd.DataFrame:
    if history_pdf.empty:
        return pd.DataFrame(columns=["user_idx", "item_idx", "label_hist_short", "label_hist_mid", "label_hist_old"])
    hist_raw = history_pdf.drop_duplicates(["user_idx", "item_idx"]).copy()
    hist_raw = hist_raw.sort_values(
        ["user_idx", "days_to_test", "hist_ts", "hist_rating", "item_idx"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )

    def build_scope(min_days: int, max_days: int | None, max_rows: int, label: str) -> pd.DataFrame:
        if max_rows <= 0:
            return pd.DataFrame(columns=["user_idx", "item_idx", label])
        mask = hist_raw["days_to_test"].notna() & hist_raw["days_to_test"].ge(int(min_days))
        if max_days is not None:
            mask = mask & hist_raw["days_to_test"].le(int(max_days))
        scoped = hist_raw.loc[mask, ["user_idx", "item_idx"]].copy()
        if scoped.empty:
            return pd.DataFrame(columns=["user_idx", "item_idx", label])
        scoped["_hist_rank"] = scoped.groupby("user_idx").cumcount() + 1
        scoped = scoped[scoped["_hist_rank"].le(int(max_rows))][["user_idx", "item_idx"]].drop_duplicates()
        scoped[label] = 1
        return scoped

    hist_short = build_scope(1, assets.history_short_days, assets.history_max_short, "label_hist_short")
    hist_mid = build_scope(
        assets.history_short_days + 1,
        assets.history_mid_days,
        assets.history_max_mid,
        "label_hist_mid",
    )
    hist_old = build_scope(
        assets.history_mid_days + 1,
        None,
        assets.history_max_old,
        "label_hist_old",
    )
    flags = hist_short.merge(hist_mid, on=["user_idx", "item_idx"], how="outer")
    flags = flags.merge(hist_old, on=["user_idx", "item_idx"], how="outer")
    if flags.empty:
        return pd.DataFrame(columns=["user_idx", "item_idx", "label_hist_short", "label_hist_mid", "label_hist_old"])
    for col in ("label_hist_short", "label_hist_mid", "label_hist_old"):
        if col not in flags.columns:
            flags[col] = 0
        flags[col] = pd.to_numeric(flags[col], errors="coerce").fillna(0).astype(int)
    return flags


def _inv_rank(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    out = np.zeros(len(series), dtype=np.float64)
    mask = numeric.notna() & numeric.gt(0)
    if mask.any():
        out[mask.to_numpy()] = 1.0 / np.log2(numeric[mask].to_numpy(dtype=np.float64) + 1.0)
    return out


def _contains_route(value: Any, route: str) -> float:
    if isinstance(value, np.ndarray):
        return 1.0 if route in value.tolist() else 0.0
    if isinstance(value, (list, tuple, set)):
        return 1.0 if route in value else 0.0
    return 0.0


def _route_count(value: Any) -> float:
    if isinstance(value, np.ndarray):
        return float(len(value.tolist()))
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    return 0.0


@lru_cache(maxsize=256)
def _load_history_flag_frame_cached(user_idx: int) -> pd.DataFrame:
    assets = load_stage10_live_assets()
    history_pdf = _load_history_user_frame_cached(int(user_idx))
    return _build_history_flags(history_pdf, assets)


def _prepare_stage10_features(
    candidates_pdf: pd.DataFrame,
    history_flags_pdf: pd.DataFrame,
    assets: Stage10LiveAssets,
) -> pd.DataFrame:
    pdf = candidates_pdf.copy()
    flags = history_flags_pdf.copy()
    pdf = pdf.merge(flags, on=["user_idx", "item_idx"], how="left")
    for col in ("label_hist_short", "label_hist_mid", "label_hist_old"):
        pdf[col] = pd.to_numeric(pdf.get(col), errors="coerce").fillna(0).astype(int)
    pdf["hist_feat_short"] = pdf["label_hist_short"].astype(float)
    pdf["hist_feat_mid"] = pdf["label_hist_mid"].astype(float)
    pdf["hist_feat_old"] = pdf["label_hist_old"].astype(float)
    pdf["hist_feat_any"] = (
        (pdf["label_hist_short"] + pdf["label_hist_mid"] + pdf["label_hist_old"]).gt(0).astype(float)
    )
    pdf["hist_feat_recency"] = np.select(
        [pdf["label_hist_short"].eq(1), pdf["label_hist_mid"].eq(1), pdf["label_hist_old"].eq(1)],
        [1.0, 0.5, 0.2],
        default=0.0,
    )
    pdf["pre_score_x_profile"] = pdf["pre_score"] * pdf["has_profile"]
    pdf["pre_score_x_cluster"] = pdf["pre_score"] * pdf["has_cluster"]
    pdf["pre_score_x_als"] = pdf["pre_score"] * pdf["has_als"]
    pdf["signal_x_profile"] = pdf["signal_score"] * pdf["has_profile"]
    pdf["quality_x_profile"] = pdf["quality_score"] * pdf["has_profile"]
    pdf["semantic_x_profile"] = pdf["semantic_score"] * pdf["has_profile"]
    pdf["semantic_x_heavy"] = pdf["semantic_score"] * pdf["is_heavy"]
    pdf["profile_x_heavy"] = pdf["has_profile"] * pdf["is_heavy"]
    pdf["cluster_x_heavy"] = pdf["has_cluster"] * pdf["is_heavy"]
    pdf["inv_pre_minus_best_route"] = pdf["inv_pre_rank"] - pdf[["inv_als_rank", "inv_cluster_rank", "inv_popular_rank"]].max(axis=1)
    pdf["pre_minus_signal"] = pdf["pre_score"] - pdf["signal_score"]
    pdf["pre_minus_semantic"] = pdf["pre_score"] - pdf["semantic_score"]
    for feature in assets.feature_columns:
        if feature not in pdf.columns:
            pdf[feature] = 0.0
        pdf[feature] = pd.to_numeric(pdf[feature], errors="coerce").fillna(0.0)
    return pdf


def _score_stage10_live_real(user_idx: int, *, top_k: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    asset_hits_before = load_stage10_live_assets.cache_info().hits
    asset_started = time.perf_counter()
    assets = load_stage10_live_assets()
    asset_elapsed_ms = round((time.perf_counter() - asset_started) * 1000.0, 3)

    candidate_started = time.perf_counter()
    candidates_pdf = _load_stage09_user_candidates_cached(int(user_idx)).copy()
    candidate_elapsed_ms = round((time.perf_counter() - candidate_started) * 1000.0, 3)

    history_started = time.perf_counter()
    history_pdf = _load_history_user_frame_cached(int(user_idx)).copy()
    history_elapsed_ms = round((time.perf_counter() - history_started) * 1000.0, 3)

    history_flag_started = time.perf_counter()
    history_flags_pdf = _load_history_flag_frame_cached(int(user_idx)).copy()
    history_flag_elapsed_ms = round((time.perf_counter() - history_flag_started) * 1000.0, 3)

    feature_started = time.perf_counter()
    prepared = _prepare_stage10_features(candidates_pdf, history_flags_pdf, assets)
    feature_elapsed_ms = round((time.perf_counter() - feature_started) * 1000.0, 3)

    predict_started = time.perf_counter()
    x = prepared[list(assets.feature_columns)].to_numpy(dtype=np.float32)
    prepared["learned_score"] = assets.model.predict_proba(x)[:, 1].astype(np.float64)
    prepared["learned_score_for_blend"] = prepared["learned_score"].to_numpy(dtype=np.float64)
    prepared["pre_prior"] = 1.0 / np.log2(prepared["pre_rank"].to_numpy(dtype=np.float64) + 1.0)
    denom = np.maximum(prepared["source_count"].to_numpy(dtype=np.float64), 1.0)
    route_total = np.zeros(len(prepared), dtype=np.float64)
    for route in ROUTE_KEYS:
        route_total += prepared[f"has_{route}"].to_numpy(dtype=np.float64) * float(assets.route_weights.get(route, 1.0))
    route_factor = np.clip(route_total / denom, 0.5, 1.5)
    prepared["route_factor"] = route_factor
    prepared["pre_prior_route"] = prepared["pre_prior"] * np.power(route_factor, float(assets.route_gamma))
    prepared["learned_model_signal"] = prepared["learned_score_for_blend"].to_numpy(dtype=np.float64)
    prepared["learned_blend_score"] = (
        (1.0 - float(assets.blend_alpha)) * prepared["pre_prior_route"]
        + float(assets.blend_alpha) * prepared["learned_model_signal"]
    )
    predict_elapsed_ms = round((time.perf_counter() - predict_started) * 1000.0, 3)

    rank_started = time.perf_counter()
    prepared = prepared.sort_values(
        ["learned_blend_score", "pre_score", "item_idx"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    prepared["learned_rank"] = np.arange(1, len(prepared) + 1, dtype=np.int32)
    rank_elapsed_ms = round((time.perf_counter() - rank_started) * 1000.0, 3)
    total_elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    topn = max(int(top_k), 10)
    meta = {
        "mode": "xgb_live",
        "model_backend": assets.model_backend,
        "blend_alpha": float(assets.blend_alpha),
        "blend_mode": str(assets.blend_mode),
        "feature_count": len(assets.feature_columns),
        "topn_preview": int(topn),
        "data_source": _safe_relative(assets.stage09_paths.candidate_parquet),
        "history_source": _safe_relative(assets.stage09_paths.history_parquet),
        "model_json": _safe_relative(assets.model_json),
        "model_file": _safe_relative(assets.model_path),
        "source_alignment": assets.stage09_paths.source_alignment,
        "requested_release_stage09_bucket_dir": _safe_relative(assets.stage09_paths.requested_release_bucket_dir),
        "effective_stage09_bucket_dir": _safe_relative(assets.stage09_paths.effective_bucket_dir),
        "timings_ms": {
            "assets_load": asset_elapsed_ms,
            "candidate_lookup": candidate_elapsed_ms,
            "history_lookup": history_elapsed_ms,
            "history_flag_lookup": history_flag_elapsed_ms,
            "feature_build": feature_elapsed_ms,
            "model_predict": predict_elapsed_ms,
            "rank_sort": rank_elapsed_ms,
            "total": total_elapsed_ms,
        },
        "warm_model_cache": load_stage10_live_assets.cache_info().hits > asset_hits_before,
    }
    return prepared, meta


def score_stage10_live(user_idx: int, *, top_k: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        return _score_stage10_live_real(int(user_idx), top_k=top_k)
    except Exception:
        started = time.perf_counter()
        prepared = _sample_live_candidates(int(user_idx)).copy()
        prepared["learned_score"] = pd.to_numeric(prepared.get("learned_blend_score"), errors="coerce").fillna(0.0)
        prepared = prepared.sort_values(["learned_rank", "item_idx"], kind="stable").reset_index(drop=True)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return prepared, {
            "mode": "xgb_live",
            "model_backend": "embedded_sample_xgb_replay",
            "blend_alpha": _resolve_bucket5_blend_alpha(),
            "blend_mode": "replay_sample",
            "feature_count": 0,
            "topn_preview": max(int(top_k), 10),
            "data_source": "embedded_sample_replay_store",
            "history_source": "embedded_sample_replay_store",
            "model_json": "embedded_sample_replay_store",
            "model_file": "embedded_sample_replay_store",
            "source_alignment": "embedded_sample_fixture",
            "requested_release_stage09_bucket_dir": _safe_relative(CURRENT_STAGE09_RELEASE_BUCKET_DIR),
            "effective_stage09_bucket_dir": "embedded_sample_replay_store",
            "timings_ms": {
                "assets_load": 0.0,
                "candidate_lookup": elapsed_ms,
                "history_lookup": 0.0,
                "history_flag_lookup": 0.0,
                "feature_build": 0.0,
                "model_predict": 0.0,
                "rank_sort": 0.0,
                "total": elapsed_ms,
            },
            "warm_model_cache": False,
        }
