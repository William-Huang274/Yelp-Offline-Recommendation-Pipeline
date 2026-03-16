from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pipeline.project_paths import env_or_project_path, project_path
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from xgboost import XGBRanker
except Exception:
    XGBRanker = None


RUN_TAG = "stage09_bucket10_prerank_probe"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_PRERANK_PROBE_ROOT_DIR", "data/output/09_prerank_probe")

PROBE_BUCKET = int(os.getenv("PRERANK_PROBE_BUCKET", "10").strip() or 10)
COHORT_PATH = os.getenv("PRERANK_PROBE_COHORT_PATH", "").strip()
MAX_CANDIDATE_ROWS = int(os.getenv("PRERANK_PROBE_MAX_CANDIDATE_ROWS", "2000000").strip() or 2000000)
N_FOLDS = int(os.getenv("PRERANK_PROBE_FOLDS", "5").strip() or 5)
RANDOM_SEED = int(os.getenv("PRERANK_PROBE_RANDOM_SEED", "42").strip() or 42)
MODEL_BACKENDS = [x.strip().lower() for x in os.getenv("PRERANK_PROBE_BACKENDS", "numpy_lr").split(",") if x.strip()]
TOPN_LIST = [int(x.strip()) for x in os.getenv("PRERANK_PROBE_TOPN_LIST", "80,150,250").split(",") if x.strip()]

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "8g").strip() or "8g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "8g").strip() or "8g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[8]").strip() or "local[8]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "32").strip() or "32"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "32").strip() or "32"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "900s").strip() or "900s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = (
    os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
)
SPARK_SQL_ADAPTIVE_ENABLED = os.getenv("SPARK_SQL_ADAPTIVE_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
PY_TEMP_DIR = os.getenv("PY_TEMP_DIR", "").strip()

XGB_N_ESTIMATORS = int(os.getenv("PRERANK_PROBE_XGB_N_ESTIMATORS", "240").strip() or 240)
XGB_MAX_DEPTH = int(os.getenv("PRERANK_PROBE_XGB_MAX_DEPTH", "6").strip() or 6)
XGB_LEARNING_RATE = float(os.getenv("PRERANK_PROBE_XGB_LEARNING_RATE", "0.05").strip() or 0.05)
XGB_SUBSAMPLE = float(os.getenv("PRERANK_PROBE_XGB_SUBSAMPLE", "0.8").strip() or 0.8)
XGB_COLSAMPLE = float(os.getenv("PRERANK_PROBE_XGB_COLSAMPLE", "0.8").strip() or 0.8)
XGB_REG_LAMBDA = float(os.getenv("PRERANK_PROBE_XGB_REG_LAMBDA", "2.0").strip() or 2.0)
XGB_OBJECTIVE = os.getenv("PRERANK_PROBE_XGB_OBJECTIVE", "rank:pairwise").strip() or "rank:pairwise"
XGB_MIN_CHILD_WEIGHT = float(os.getenv("PRERANK_PROBE_XGB_MIN_CHILD_WEIGHT", "8.0").strip() or 8.0)
XGB_N_JOBS = int(os.getenv("PRERANK_PROBE_XGB_N_JOBS", "16").strip() or 16)
LR_POS_WEIGHT = float(os.getenv("PRERANK_PROBE_LR_POS_WEIGHT", "12.0").strip() or 12.0)
NUMPY_LR_MAXITER = int(os.getenv("PRERANK_PROBE_NUMPY_LR_MAXITER", "180").strip() or 180)
NUMPY_LR_L2 = float(os.getenv("PRERANK_PROBE_NUMPY_LR_L2", "0.08").strip() or 0.08)
NUMPY_LR_CLIP = float(os.getenv("PRERANK_PROBE_NUMPY_LR_CLIP", "18.0").strip() or 18.0)

_SPARK_TMP_CTX: SparkTmpContext | None = None


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        py_temp_root_override=PY_TEMP_DIR,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=True,
    )
    local_dir = _SPARK_TMP_CTX.spark_local_dir
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} spark_local_dir={local_dir} py_temp={_SPARK_TMP_CTX.py_temp_dir} "
        f"auto_clean={SPARK_TMP_AUTOCLEAN_ENABLED} retention_h={SPARK_TMP_RETENTION_HOURS} "
        f"cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    return (
        SparkSession.builder.appName(RUN_TAG)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.network.timeout", SPARK_NETWORK_TIMEOUT)
        .config("spark.executor.heartbeatInterval", SPARK_EXECUTOR_HEARTBEAT_INTERVAL)
        .config("spark.sql.adaptive.enabled", str(SPARK_SQL_ADAPTIVE_ENABLED).lower())
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


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


def resolve_cohort_df(spark: SparkSession, truth: Any) -> Any:
    path = str(COHORT_PATH or "").strip()
    if not path:
        return truth
    cohort = spark.read.option("header", True).csv(path)
    if "user_idx" in cohort.columns:
        return truth.join(cohort.select(F.col("user_idx").cast("int").alias("user_idx")).dropDuplicates(["user_idx"]), on="user_idx", how="inner")
    if "user_id" in cohort.columns:
        return truth.join(cohort.select(F.col("user_id").cast("string").alias("user_id")).dropDuplicates(["user_id"]), on="user_id", how="inner")
    raise RuntimeError(f"cohort file missing user_idx/user_id column: {path}")


def resolve_candidate_path(bucket_dir: Path) -> Path:
    candidates = [
        bucket_dir / "candidates_pretrim150.parquet",
        bucket_dir / "candidates_pretrim.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"no candidate parquet in {bucket_dir}")


def load_probe_pdf(spark: SparkSession, run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    bucket_dir = run_dir / f"bucket_{int(PROBE_BUCKET)}"
    truth = (
        spark.read.parquet((bucket_dir / "truth.parquet").as_posix())
        .select("user_idx", F.col("true_item_idx").cast("int").alias("true_item_idx"), "user_id")
        .dropDuplicates(["user_idx"])
    )
    truth = resolve_cohort_df(spark, truth).cache()
    users = truth.select("user_idx").dropDuplicates(["user_idx"])
    cand_path = resolve_candidate_path(bucket_dir)
    cand = (
        spark.read.parquet(cand_path.as_posix())
        .join(users, on="user_idx", how="inner")
        .join(truth.select("user_idx", "true_item_idx"), on="user_idx", how="inner")
        .withColumn("label", F.when(F.col("item_idx").cast("int") == F.col("true_item_idx"), F.lit(1)).otherwise(F.lit(0)))
        .withColumn("source_combo", F.concat_ws("+", F.sort_array(F.col("source_set"))))
        .withColumn("source_count", F.coalesce(F.size(F.col("source_set")).cast("double"), F.lit(0.0)))
        .withColumn("has_als", F.when(F.array_contains(F.col("source_set"), F.lit("als")), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("has_cluster", F.when(F.array_contains(F.col("source_set"), F.lit("cluster")), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("has_profile", F.when(F.array_contains(F.col("source_set"), F.lit("profile")), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("has_popular", F.when(F.array_contains(F.col("source_set"), F.lit("popular")), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("nonpopular_source_count", F.col("has_als") + F.col("has_cluster") + F.col("has_profile"))
        .withColumn(
            "is_als_only",
            F.when((F.col("source_count") <= F.lit(1.0)) & (F.col("has_als") > F.lit(0.5)), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "is_als_popular_only",
            F.when(
                (F.col("source_count") <= F.lit(2.0))
                & (F.col("has_als") > F.lit(0.5))
                & (F.col("has_popular") > F.lit(0.5))
                & (F.col("has_cluster") <= F.lit(0.5))
                & (F.col("has_profile") <= F.lit(0.5)),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "has_profile_or_cluster",
            F.when((F.col("has_profile") > F.lit(0.5)) | (F.col("has_cluster") > F.lit(0.5)), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    )
    row_probe = int(cand.limit(int(MAX_CANDIDATE_ROWS) + 1).count())
    if row_probe > int(MAX_CANDIDATE_ROWS):
        raise RuntimeError(
            f"probe rows exceed cap: rows>{MAX_CANDIDATE_ROWS}; narrow cohort or raise PRERANK_PROBE_MAX_CANDIDATE_ROWS"
        )
    selected = cand.select(
        "user_idx",
        "item_idx",
        "true_item_idx",
        "label",
        "user_segment",
        "user_train_count",
        "pre_rank",
        "pre_score",
        "signal_score",
        "quality_score",
        "semantic_score",
        "semantic_confidence",
        "semantic_support",
        "semantic_tag_richness",
        "semantic_effective_score",
        "als_backbone_score",
        "tower_score",
        "seq_score",
        "tower_inv",
        "seq_inv",
        "als_rank",
        "cluster_rank",
        "profile_rank",
        "popular_rank",
        "source_count",
        "nonpopular_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
        "has_profile_or_cluster",
        "is_als_only",
        "is_als_popular_only",
        "source_combo",
    )
    pdf = selected.toPandas()
    truth.unpersist(blocking=False)
    meta = {
        "candidate_path": cand_path.as_posix(),
        "rows": int(pdf.shape[0]),
        "users": int(pdf["user_idx"].nunique()),
        "positives": int(pd.to_numeric(pdf["label"], errors="coerce").fillna(0).astype(np.int32).sum()),
        "max_candidate_rows_cap": int(MAX_CANDIDATE_ROWS),
    }
    return pdf, meta


def _inv_rank(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.zeros((len(vals),), dtype=np.float64), index=series.index)
    mask = vals.notna() & (vals > 0)
    out.loc[mask] = 1.0 / np.log2(vals.loc[mask].astype(np.float64) + 1.0)
    return out


def prepare_features(pdf: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    out = pdf.copy()
    numeric_cols = [
        "user_idx",
        "item_idx",
        "true_item_idx",
        "label",
        "user_train_count",
        "pre_rank",
        "pre_score",
        "signal_score",
        "quality_score",
        "semantic_score",
        "semantic_confidence",
        "semantic_support",
        "semantic_tag_richness",
        "semantic_effective_score",
        "als_backbone_score",
        "tower_score",
        "seq_score",
        "tower_inv",
        "seq_inv",
        "als_rank",
        "cluster_rank",
        "profile_rank",
        "popular_rank",
        "source_count",
        "nonpopular_source_count",
        "has_als",
        "has_cluster",
        "has_profile",
        "has_popular",
        "has_profile_or_cluster",
        "is_als_only",
        "is_als_popular_only",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["label"] = out["label"].fillna(0).astype(np.int32)
    out["user_idx"] = out["user_idx"].fillna(-1).astype(np.int32)
    out["item_idx"] = out["item_idx"].fillna(-1).astype(np.int32)
    out["true_item_idx"] = out["true_item_idx"].fillna(-1).astype(np.int32)
    out["user_segment"] = out["user_segment"].fillna("unknown").astype(str)
    out["source_combo"] = out["source_combo"].fillna("NA").astype(str)
    for col in numeric_cols:
        if col not in {"user_idx", "item_idx", "true_item_idx", "label"}:
            out[col] = out[col].fillna(0.0).astype(np.float64)

    rank_cols = ["pre_rank", "als_rank", "cluster_rank", "profile_rank", "popular_rank"]
    rank_fill = float(max(1000.0, pd.to_numeric(out["pre_rank"], errors="coerce").max(skipna=True) or 0.0) + 50.0)
    for col in rank_cols:
        raw = pd.to_numeric(pdf[col], errors="coerce")
        out[f"{col}_missing"] = raw.isna().astype(np.float64)
        out[f"{col}_filled"] = raw.fillna(rank_fill).astype(np.float64)
        out[f"inv_{col}"] = _inv_rank(raw).astype(np.float64)

    out["user_train_log"] = np.log1p(np.maximum(out["user_train_count"].astype(np.float64), 0.0))
    out["semantic_support_log"] = np.log1p(np.maximum(out["semantic_support"].astype(np.float64), 0.0))
    out["is_light"] = (out["user_segment"] == "light").astype(np.float64)
    out["is_mid"] = (out["user_segment"] == "mid").astype(np.float64)
    out["is_heavy"] = (out["user_segment"] == "heavy").astype(np.float64)
    out["has_als_profile"] = ((out["has_als"] > 0.5) & (out["has_profile"] > 0.5)).astype(np.float64)
    out["has_als_cluster"] = ((out["has_als"] > 0.5) & (out["has_cluster"] > 0.5)).astype(np.float64)
    out["has_cluster_profile"] = ((out["has_cluster"] > 0.5) & (out["has_profile"] > 0.5)).astype(np.float64)
    out["has_all4"] = (
        (out["has_als"] > 0.5)
        & (out["has_cluster"] > 0.5)
        & (out["has_profile"] > 0.5)
        & (out["has_popular"] > 0.5)
    ).astype(np.float64)
    out["pre_minus_signal"] = out["pre_score"] - out["signal_score"]
    out["pre_minus_semantic"] = out["pre_score"] - out["semantic_effective_score"]
    out["quality_x_profile"] = out["quality_score"] * out["has_profile"]
    out["semantic_x_profile"] = out["semantic_effective_score"] * out["has_profile"]
    out["semantic_x_cluster"] = out["semantic_effective_score"] * out["has_cluster"]
    out["semantic_x_heavy"] = out["semantic_effective_score"] * out["is_heavy"]
    out["profile_x_heavy"] = out["has_profile"] * out["is_heavy"]
    out["cluster_x_heavy"] = out["has_cluster"] * out["is_heavy"]
    out["als_profile_x_heavy"] = out["has_als_profile"] * out["is_heavy"]
    out["pre_x_profile"] = out["pre_score"] * out["has_profile"]
    out["pre_x_cluster"] = out["pre_score"] * out["has_cluster"]
    out["pre_x_popular"] = out["pre_score"] * out["has_popular"]
    out["pre_x_heavy"] = out["pre_score"] * out["is_heavy"]
    out["inv_best_route"] = np.maximum.reduce(
        [
            out["inv_als_rank"].to_numpy(dtype=np.float64),
            out["inv_cluster_rank"].to_numpy(dtype=np.float64),
            out["inv_profile_rank"].to_numpy(dtype=np.float64),
            out["inv_popular_rank"].to_numpy(dtype=np.float64),
        ]
    )
    out["inv_best_nonpopular"] = np.maximum.reduce(
        [
            out["inv_als_rank"].to_numpy(dtype=np.float64),
            out["inv_cluster_rank"].to_numpy(dtype=np.float64),
            out["inv_profile_rank"].to_numpy(dtype=np.float64),
        ]
    )
    out["inv_pre_minus_best_route"] = out["inv_pre_rank"] - out["inv_best_route"]
    out["inv_pre_minus_best_nonpopular"] = out["inv_pre_rank"] - out["inv_best_nonpopular"]
    out["profile_rank_gap_vs_als"] = out["als_rank_filled"] - out["profile_rank_filled"]
    out["cluster_rank_gap_vs_als"] = out["als_rank_filled"] - out["cluster_rank_filled"]

    feature_cols = [
        "pre_score",
        "signal_score",
        "quality_score",
        "semantic_score",
        "semantic_confidence",
        "semantic_support_log",
        "semantic_tag_richness",
        "semantic_effective_score",
        "als_backbone_score",
        "tower_score",
        "seq_score",
        "tower_inv",
        "seq_inv",
        "user_train_log",
        "pre_rank_filled",
        "als_rank_filled",
        "cluster_rank_filled",
        "profile_rank_filled",
        "popular_rank_filled",
        "inv_pre_rank",
        "inv_als_rank",
        "inv_cluster_rank",
        "inv_profile_rank",
        "inv_popular_rank",
        "source_count",
        "nonpopular_source_count",
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
        "has_all4",
        "pre_minus_signal",
        "pre_minus_semantic",
        "quality_x_profile",
        "semantic_x_profile",
        "semantic_x_cluster",
        "semantic_x_heavy",
        "profile_x_heavy",
        "cluster_x_heavy",
        "als_profile_x_heavy",
        "pre_x_profile",
        "pre_x_cluster",
        "pre_x_popular",
        "pre_x_heavy",
        "inv_best_route",
        "inv_best_nonpopular",
        "inv_pre_minus_best_route",
        "inv_pre_minus_best_nonpopular",
        "profile_rank_gap_vs_als",
        "cluster_rank_gap_vs_als",
        "pre_rank_missing",
        "als_rank_missing",
        "cluster_rank_missing",
        "profile_rank_missing",
        "popular_rank_missing",
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
        ["user_idx", score_col, "pre_rank", "item_idx"],
        ascending=[True, False, True, True],
        kind="stable",
    ).copy()
    out[out_rank_col] = out.groupby("user_idx", sort=False).cumcount() + 1
    return out


def summarize_truth_rows(
    ranked_pdf: pd.DataFrame,
    rank_col: str,
    topn_list: list[int],
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    pos = ranked_pdf[ranked_pdf["label"] == 1].copy()
    if pos.empty:
        raise RuntimeError(f"no positives found for rank column {rank_col}")
    summary: dict[str, Any] = {
        "users": int(pos["user_idx"].nunique()),
        "rank_min": int(pos[rank_col].min()),
        "rank_median": float(pos[rank_col].median()),
        "rank_p75": float(pos[rank_col].quantile(0.75)),
        "rank_max": int(pos[rank_col].max()),
    }
    for topn in topn_list:
        summary[f"truth_in_top{int(topn)}_users"] = int((pos[rank_col] <= int(topn)).sum())
        summary[f"truth_in_top{int(topn)}_rate"] = round(
            float((pos[rank_col] <= int(topn)).mean()),
            6,
        )
    seg_rows: list[dict[str, Any]] = []
    for seg, g in pos.groupby("user_segment", sort=True):
        row: dict[str, Any] = {
            "user_segment": str(seg),
            "users": int(g["user_idx"].nunique()),
            "rank_median": float(g[rank_col].median()),
        }
        for topn in topn_list:
            row[f"truth_in_top{int(topn)}_users"] = int((g[rank_col] <= int(topn)).sum())
            row[f"truth_in_top{int(topn)}_rate"] = round(
                float((g[rank_col] <= int(topn)).mean()),
                6,
            )
        seg_rows.append(row)
    return summary, pos, seg_rows


def fit_backend(
    backend: str,
    train_pdf: pd.DataFrame,
    test_pdf: pd.DataFrame,
    feature_cols: list[str],
    fold: int,
) -> tuple[np.ndarray, dict[str, float]]:
    x_train = train_pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_pdf["label"].to_numpy(dtype=np.float32, copy=False)
    x_test = test_pdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
    if backend == "xgboost_ranker":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        train_sorted = train_pdf.sort_values(["user_idx", "pre_rank", "item_idx"], kind="stable")
        x_train = train_sorted[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_train = train_sorted["label"].to_numpy(dtype=np.float32, copy=False)
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
        raw_importances = booster.get_score(importance_type="gain")
        importances = {str(k): float(v) for k, v in raw_importances.items() if float(v) > 0.0}
        return pred, importances
    if backend == "numpy_lr":
        x_train64 = train_pdf[feature_cols].to_numpy(dtype=np.float64, copy=False)
        y_train64 = train_pdf["label"].to_numpy(dtype=np.float64, copy=False)
        x_test64 = test_pdf[feature_cols].to_numpy(dtype=np.float64, copy=False)
        sample_weight = np.where(y_train64 > 0.5, float(LR_POS_WEIGHT), 1.0).astype(np.float64)
        mean = np.nanmean(x_train64, axis=0)
        std = np.nanstd(x_train64, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        x_train_scaled = (x_train64 - mean) / std
        x_test_scaled = (x_test64 - mean) / std
        x_train_aug = np.concatenate(
            [x_train_scaled, np.ones((x_train_scaled.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        x_test_aug = np.concatenate(
            [x_test_scaled, np.ones((x_test_scaled.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        denom = float(max(sample_weight.sum(), 1.0))

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            z = np.clip(x_train_aug @ params, -float(NUMPY_LR_CLIP), float(NUMPY_LR_CLIP))
            loss_vec = np.logaddexp(0.0, z) - y_train64 * z
            loss = float(np.dot(sample_weight, loss_vec) / denom)
            probs = 1.0 / (1.0 + np.exp(-z))
            diff = (sample_weight * (probs - y_train64)) / denom
            grad = x_train_aug.T @ diff
            grad[:-1] += float(NUMPY_LR_L2) * params[:-1]
            loss += 0.5 * float(NUMPY_LR_L2) * float(np.dot(params[:-1], params[:-1]))
            return loss, grad

        init = np.zeros((x_train_aug.shape[1],), dtype=np.float64)
        if minimize is not None:
            result = minimize(
                fun=lambda w: objective(w)[0],
                x0=init,
                jac=lambda w: objective(w)[1],
                method="L-BFGS-B",
                options={"maxiter": int(NUMPY_LR_MAXITER), "disp": False},
            )
            coef = result.x.astype(np.float64, copy=False)
        else:
            coef = init
            lr = 0.25
            for _ in range(int(NUMPY_LR_MAXITER)):
                _, grad = objective(coef)
                coef = coef - lr * grad
                lr *= 0.985
        logits = np.clip(x_test_aug @ coef, -float(NUMPY_LR_CLIP), float(NUMPY_LR_CLIP))
        pred = (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
        importances = {feature_cols[i]: abs(float(v)) for i, v in enumerate(coef[:-1]) if abs(float(v)) > 0.0}
        return pred, importances
    if backend == "sklearn_lr":
        if LogisticRegression is None:
            raise RuntimeError("sklearn is not installed")
        sample_weight = np.where(y_train > 0.5, float(LR_POS_WEIGHT), 1.0).astype(np.float64)
        clf = LogisticRegression(solver="lbfgs", max_iter=500, random_state=int(RANDOM_SEED + fold))
        clf.fit(x_train, y_train, sample_weight=sample_weight)
        pred = clf.predict_proba(x_test)[:, 1].astype(np.float64)
        importances = {feature_cols[i]: abs(float(v)) for i, v in enumerate(np.ravel(clf.coef_))}
        return pred, importances
    raise RuntimeError(f"unsupported backend: {backend}")


def run_backend_cv(
    backend: str,
    probe_pdf: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
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
        ranked = test_pdf.copy()
        ranked["probe_score"] = pred
        ranked = rank_by_score(ranked, score_col="probe_score", out_rank_col="probe_rank")
        fold_summary, pos_rows, seg_rows = summarize_truth_rows(ranked, rank_col="probe_rank", topn_list=TOPN_LIST)
        baseline_ranked = rank_by_score(test_pdf.copy(), score_col="pre_score", out_rank_col="baseline_score_rank")
        baseline_summary, baseline_pos, _ = summarize_truth_rows(baseline_ranked, rank_col="pre_rank", topn_list=TOPN_LIST)
        pos_rows = pos_rows.merge(
            baseline_pos[["user_idx", "pre_rank"]].rename(columns={"pre_rank": "baseline_pre_rank"}),
            on="user_idx",
            how="left",
        )
        pos_rows["fold_id"] = int(fold)
        pos_rows["backend"] = str(backend)
        all_pos_rows.append(pos_rows)
        row = {
            "backend": str(backend),
            "fold_id": int(fold),
            "users": int(fold_summary["users"]),
        }
        for topn in TOPN_LIST:
            row[f"truth_in_top{int(topn)}_rate"] = float(fold_summary[f"truth_in_top{int(topn)}_rate"])
            row[f"baseline_truth_in_top{int(topn)}_rate"] = float(baseline_summary[f"truth_in_top{int(topn)}_rate"])
        fold_rows.append(row)
        for item in seg_rows:
            item["backend"] = str(backend)
            item["fold_id"] = int(fold)
        top_features = sorted(importances.items(), key=lambda z: (-z[1], z[0]))[:30]
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
    overall_summary, _, seg_summary = summarize_truth_rows(pos_all, rank_col="probe_rank", topn_list=TOPN_LIST)
    baseline_summary, _, baseline_seg_summary = summarize_truth_rows(pos_all, rank_col="baseline_pre_rank", topn_list=TOPN_LIST)
    summary = {
        "backend": str(backend),
        "users": int(overall_summary["users"]),
        "baseline": baseline_summary,
        "probe": overall_summary,
        "delta": {},
        "segment_probe": seg_summary,
        "segment_baseline": baseline_seg_summary,
    }
    for topn in TOPN_LIST:
        summary["delta"][f"truth_in_top{int(topn)}_users"] = int(
            overall_summary[f"truth_in_top{int(topn)}_users"] - baseline_summary[f"truth_in_top{int(topn)}_users"]
        )
        summary["delta"][f"truth_in_top{int(topn)}_rate"] = round(
            float(overall_summary[f"truth_in_top{int(topn)}_rate"] - baseline_summary[f"truth_in_top{int(topn)}_rate"]),
            6,
        )
    importance_df = pd.DataFrame(importance_rows)
    return summary, pd.DataFrame(fold_rows), importance_df


def main() -> None:
    run_dir = resolve_stage09_run()
    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_stage09_bucket10_prerank_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    spark = build_spark()
    try:
        probe_pdf, load_meta = load_probe_pdf(spark=spark, run_dir=run_dir)
    finally:
        spark.stop()
    print(f"[PROBE] rows={load_meta['rows']} users={load_meta['users']} positives={load_meta['positives']}")
    probe_pdf, feature_cols, feature_meta = prepare_features(probe_pdf)
    backend_summaries: dict[str, Any] = {}
    fold_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for backend in MODEL_BACKENDS:
        if backend == "xgboost_ranker" and xgb is None:
            print("[WARN] skip backend=xgboost_ranker because xgboost is unavailable")
            continue
        if backend == "sklearn_lr" and LogisticRegression is None:
            print("[WARN] skip backend=sklearn_lr because sklearn is unavailable")
            continue
        print(f"[MODEL] backend={backend}")
        summary, fold_df, importance_df = run_backend_cv(backend=backend, probe_pdf=probe_pdf, feature_cols=feature_cols)
        backend_summaries[str(backend)] = summary
        fold_frames.append(fold_df)
        if not importance_df.empty:
            importance_frames.append(importance_df)
    if not backend_summaries:
        raise RuntimeError("no backend completed")
    baseline_summary, pos_baseline, baseline_seg = summarize_truth_rows(probe_pdf, rank_col="pre_rank", topn_list=TOPN_LIST)
    baseline_payload = {
        "baseline": baseline_summary,
        "segment_baseline": baseline_seg,
        "oracle_truth_in_pretrim_users": int(pos_baseline["user_idx"].nunique()),
        "oracle_truth_in_pretrim_rate": round(float((pos_baseline["pre_rank"] > 0).mean()), 6),
    }
    payload = {
        "run_tag": RUN_TAG,
        "source_stage09_run": run_dir.as_posix(),
        "bucket": int(PROBE_BUCKET),
        "cohort_path": str(COHORT_PATH or ""),
        "load_meta": load_meta,
        "feature_meta": feature_meta,
        "baseline": baseline_payload,
        "backend_summaries": backend_summaries,
        "model_backends": MODEL_BACKENDS,
        "topn_list": TOPN_LIST,
    }
    write_json(out_dir / "probe_summary.json", payload)
    pd.DataFrame(
        [
            {"backend": k, **v["probe"], **{f"delta_{kk}": vv for kk, vv in v["delta"].items()}}
            for k, v in backend_summaries.items()
        ]
    ).to_csv(out_dir / "probe_backend_summary.csv", index=False, encoding="utf-8")
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(out_dir / "probe_fold_metrics.csv", index=False, encoding="utf-8")
    if importance_frames:
        imp = pd.concat(importance_frames, ignore_index=True)
        imp.to_csv(out_dir / "probe_feature_importance.csv", index=False, encoding="utf-8")
        imp.groupby(["backend", "feature"], as_index=False)["importance"].mean().sort_values(
            ["backend", "importance", "feature"], ascending=[True, False, True], kind="stable"
        ).to_csv(out_dir / "probe_feature_importance_mean.csv", index=False, encoding="utf-8")
    print(f"[INFO] wrote {out_dir}")


if __name__ == "__main__":
    main()
