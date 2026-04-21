from __future__ import annotations

import sys

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    script = __file__.replace("\\", "/").split("/")[-1]
    print(f"Usage: python scripts/{script}")
    print("This stage script is configured by environment variables and starts a Spark evaluation job.")
    print("Set the required INPUT_/OUTPUT_/SPARK_* environment variables, then run without --help.")
    sys.exit(0)

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, project_path
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context


RUN_TAG = "stage11_pass2_profile_route_subset_eval"

PARQUET_BASE = env_or_project_path("PARQUET_BASE_DIR", "data/parquet")
INPUT_COHORT_CSV = os.getenv("INPUT_PASS2_USER_COHORT_CSV", "").strip()
INPUT_PASS2_SIDECAR_CSV = os.getenv("INPUT_PASS2_PROFILE_TAG_SIDECAR_CSV", "").strip()

USER_PROFILE_ROOT = env_or_project_path("INPUT_09_USER_PROFILES_ROOT_DIR", "data/output/09_user_profiles")
USER_PROFILE_RUN_DIR = os.getenv("INPUT_09_USER_PROFILES_RUN_DIR", "").strip()
ITEM_SEMANTIC_ROOT = env_or_project_path("INPUT_09_ITEM_SEMANTICS_ROOT_DIR", "data/output/09_item_semantics")
ITEM_SEMANTIC_RUN_DIR = os.getenv("INPUT_09_ITEM_SEMANTICS_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path(
    "OUTPUT_11_PASS2_PROFILE_ROUTE_SUBSET_EVAL_ROOT_DIR",
    "data/output/11_pass2_profile_route_subset_eval",
)

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "8").strip() or "8"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "8").strip() or "8"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)

TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
MIN_USER_REVIEWS_OFFSET = 2
FILTER_POLICY: dict[str, Any] = {
    "require_is_open": True,
    "min_business_stars": 3.0,
    "min_business_review_count": 20,
    "stale_cutoff_date": "2020-01-01",
}
BUCKETS = [2, 5, 10]
PROFILE_TOP_K_BY_BUCKET = {2: 240, 5: 280, 10: 400}
PROFILE_TAG_SHARED_TOP_K = 120
PROFILE_SHARED_SCORE_MIN = 0.0
PROFILE_CONFIDENCE_FLOOR = 0.25

ITEM_TAG_TYPE_ALIAS = {"ambience": "scene", "audience": "scene"}
SAMPLE_ROWS = int(os.getenv("PASS2_PROFILE_ROUTE_SUBSET_SAMPLE_ROWS", "12").strip() or 12)

_SPARK_TMP_CTX: SparkTmpContext | None = None


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        session_isolation=True,
        auto_clean_enabled=True,
        clean_on_exit=True,
        retention_hours=8,
        clean_max_entries=3000,
        set_env_temp=True,
    )
    local_dir = _SPARK_TMP_CTX.spark_local_dir
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} spark_local_dir={local_dir} py_temp={_SPARK_TMP_CTX.py_temp_dir} "
        f"cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    return (
        SparkSession.builder.appName(RUN_TAG)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", SPARK_DEFAULT_PARALLELISM)
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def safe_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def resolve_run(raw: str, root: Path, suffix: str) -> Path:
    if raw:
        path = normalize_legacy_project_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"run dir not found: {path}")
        return path
    runs = [path for path in root.iterdir() if path.is_dir() and path.name.endswith(suffix)]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def normalize_tag(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = pd.Series([text]).str.replace(r"[^a-z0-9]+", "_", regex=True).iloc[0]
    text = pd.Series([text]).str.replace(r"_+", "_", regex=True).iloc[0]
    return text.strip("_")


def load_user_cohort() -> pd.DataFrame:
    if not INPUT_COHORT_CSV:
        raise RuntimeError("INPUT_PASS2_USER_COHORT_CSV is required")
    path = normalize_legacy_project_path(INPUT_COHORT_CSV)
    if not path.exists():
        raise FileNotFoundError(f"cohort csv not found: {path}")
    pdf = pd.read_csv(path)
    if "user_id" not in pdf.columns:
        raise RuntimeError(f"cohort csv missing user_id: {path}")
    pdf["user_id"] = pdf["user_id"].astype(str).str.strip()
    pdf = pdf[pdf["user_id"] != ""].copy()
    return pdf[["user_id"]].drop_duplicates()


def load_scoped_reviews(spark: SparkSession, cohort_pdf: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    business = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "name", "state", "city", "categories", "is_open", "stars", "review_count")
        .withColumn("business_id", F.col("business_id").cast("string"))
    )
    cat = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    biz = business.filter(F.col("state") == TARGET_STATE)
    cond = None
    if REQUIRE_RESTAURANTS:
        cond = cat.contains("restaurants")
    if REQUIRE_FOOD:
        cond = (cond | cat.contains("food")) if cond is not None else cat.contains("food")
    if cond is not None:
        biz = biz.filter(cond)
    before_count = int(biz.count())
    if FILTER_POLICY["require_is_open"]:
        biz = biz.filter(F.col("is_open") == 1)
    biz = biz.filter(F.col("stars") >= F.lit(float(FILTER_POLICY["min_business_stars"])))
    biz = biz.filter(F.col("review_count") >= F.lit(int(FILTER_POLICY["min_business_review_count"])))

    review_ts = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("business_id", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
    )
    last_review = review_ts.groupBy("business_id").agg(F.max("ts").alias("last_review_ts"))
    biz = biz.join(last_review, on="business_id", how="left")
    biz = biz.filter(F.col("last_review_ts") >= F.to_timestamp(F.lit(str(FILTER_POLICY["stale_cutoff_date"]))))
    after_count = int(biz.count())

    cohort_sp = spark.createDataFrame(cohort_pdf)
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "stars", "date", "text")
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(F.broadcast(cohort_sp), on="user_id", how="inner")
        .join(biz.select("business_id", "name", "city", "categories"), on="business_id", how="inner")
    )
    pdf = rvw.toPandas()
    pdf["user_id"] = pdf["user_id"].astype(str)
    pdf["business_id"] = pdf["business_id"].astype(str)
    pdf["review_id"] = pdf["review_id"].astype(str)
    pdf["ts"] = pd.to_datetime(pdf["ts"], errors="coerce")
    pdf["stars"] = pd.to_numeric(pdf["stars"], errors="coerce").fillna(0.0)
    pdf["text"] = pdf["text"].fillna("").astype(str)
    return pdf, {
        "scope_business_before_hard_filter": before_count,
        "scope_business_after_hard_filter": after_count,
        "scoped_review_rows": int(pdf.shape[0]),
        "scoped_users": int(pdf["user_id"].nunique()),
    }


def resolve_user_profile_run() -> Path:
    return resolve_run(USER_PROFILE_RUN_DIR, USER_PROFILE_ROOT, "_full_stage09_user_profile_build")


def resolve_item_semantic_run() -> Path:
    return resolve_run(ITEM_SEMANTIC_RUN_DIR, ITEM_SEMANTIC_ROOT, "_full_stage09_item_semantic_build")


def load_profile_confidence(profile_run: Path) -> dict[str, float]:
    use_cols = [
        "user_id",
        "profile_confidence_v1",
        "profile_confidence",
        "profile_confidence_v2",
        "profile_conf_consistency",
        "profile_tag_support",
        "n_sentences_selected",
    ]
    prof = pd.read_csv(profile_run / "user_profiles.csv", usecols=lambda c: c in set(use_cols))
    prof["user_id"] = prof["user_id"].astype(str)
    if "profile_confidence_v1" not in prof.columns:
        prof["profile_confidence_v1"] = pd.to_numeric(prof.get("profile_confidence", 0.0), errors="coerce").fillna(0.0)
    if "profile_confidence_v2" not in prof.columns:
        prof["profile_confidence_v2"] = pd.to_numeric(prof["profile_confidence_v1"], errors="coerce").fillna(0.0)
    if "profile_conf_consistency" not in prof.columns:
        prof["profile_conf_consistency"] = 0.5
    if "profile_tag_support" not in prof.columns:
        prof["profile_tag_support"] = 0.0
    if "n_sentences_selected" not in prof.columns:
        prof["n_sentences_selected"] = 0
    conf_raw = (
        0.4 * pd.to_numeric(prof["profile_confidence_v1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        + 0.6 * pd.to_numeric(prof["profile_confidence_v2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    )
    sentence_factor = np.minimum(1.0, pd.to_numeric(prof["n_sentences_selected"], errors="coerce").fillna(0).to_numpy(dtype=np.float32) / 8.0)
    consistency = np.clip(pd.to_numeric(prof["profile_conf_consistency"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float32), 0.0, 1.0)
    support_factor = np.minimum(
        1.0,
        np.log1p(pd.to_numeric(prof["profile_tag_support"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)) / np.log(5.0),
    )
    prof["profile_confidence"] = np.clip(conf_raw, 0.0, 1.0) * sentence_factor * (0.8 + 0.2 * consistency) * (0.85 + 0.15 * support_factor)
    return dict(zip(prof["user_id"].tolist(), prof["profile_confidence"].tolist()))


def load_user_tags(profile_run: Path, extra_csv: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support", "abs_net_w"]
    base = pd.read_csv(profile_run / "user_profile_tag_profile_long.csv", usecols=lambda c: c in set(cols))
    for df in (base,):
        df["user_id"] = df["user_id"].astype(str)
        df["tag"] = df["tag"].astype(str).map(normalize_tag)
        df["tag_type"] = df["tag_type"].astype(str).str.strip().str.lower()
        df["tag_type"] = df["tag_type"].replace(ITEM_TAG_TYPE_ALIAS)
        df["net_w"] = pd.to_numeric(df["net_w"], errors="coerce").fillna(0.0)
        df["tag_confidence"] = pd.to_numeric(df["tag_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        df["support"] = pd.to_numeric(df["support"], errors="coerce").fillna(1.0).clip(lower=0.0)
    base = base[(base["user_id"] != "") & (base["tag"] != "")]
    treat = base.copy()
    if extra_csv:
        path = normalize_legacy_project_path(extra_csv)
        extra = pd.read_csv(path)
        extra["user_id"] = extra["user_id"].astype(str)
        extra["tag"] = extra["tag"].astype(str).map(normalize_tag)
        extra["tag_type"] = extra["tag_type"].astype(str).str.strip().str.lower()
        extra["tag_type"] = extra["tag_type"].replace(ITEM_TAG_TYPE_ALIAS)
        extra["net_w"] = pd.to_numeric(extra["net_w"], errors="coerce").fillna(0.0)
        extra["tag_confidence"] = pd.to_numeric(extra["tag_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        extra["support"] = pd.to_numeric(extra["support"], errors="coerce").fillna(1.0).clip(lower=0.0)
        extra = extra[(extra["user_id"] != "") & (extra["tag"] != "")]
        treat = pd.concat([treat, extra[["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"]]], ignore_index=True)
    return (
        base[["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"]].copy(),
        treat[["user_id", "tag", "tag_type", "net_w", "tag_confidence", "support"]].copy(),
    )


def load_item_tags(item_run: Path, scoped_business_ids: set[str]) -> pd.DataFrame:
    item = pd.read_csv(item_run / "item_tag_profile_long.csv")
    item["business_id"] = item["business_id"].astype(str)
    item = item[item["business_id"].isin(scoped_business_ids)].copy()
    item["tag"] = item["tag"].astype(str).map(normalize_tag)
    item["tag_type"] = item.get("tag_type", item.get("facet", "other")).astype(str).str.strip().str.lower()
    item["tag_type"] = item["tag_type"].replace(ITEM_TAG_TYPE_ALIAS)
    item["net_weight_sum"] = pd.to_numeric(item["net_weight_sum"], errors="coerce").fillna(0.0)
    item["tag_confidence"] = pd.to_numeric(item["tag_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    item["support_count"] = pd.to_numeric(item["support_count"], errors="coerce").fillna(1.0).clip(lower=0.0)
    item = item[(item["tag"] != "") & (item["tag_type"] != "")]
    item["item_tag_weight"] = (
        item["net_weight_sum"].to_numpy(dtype=np.float64)
        * item["tag_confidence"].to_numpy(dtype=np.float64)
        * np.log1p(np.maximum(item["support_count"].to_numpy(dtype=np.float64), 0.0))
    )
    item = item[np.isfinite(item["item_tag_weight"])].copy()
    return item[["business_id", "tag", "tag_type", "item_tag_weight"]]


def build_global_split(events: pd.DataFrame) -> pd.DataFrame:
    counts = events.groupby("user_id", sort=False).size().rename("n_reviews").reset_index()
    eligible = counts[counts["n_reviews"] >= 4]["user_id"].astype(str)
    out = events[events["user_id"].isin(set(eligible.tolist()))].copy()
    out = out.sort_values(["user_id", "ts", "review_id"], ascending=[True, False, False]).reset_index(drop=True)
    out["rn"] = out.groupby("user_id", sort=False).cumcount() + 1
    out["split_label"] = np.where(out["rn"] == 1, "test", np.where(out["rn"] == 2, "valid", "train"))
    out = out.merge(counts, on="user_id", how="left")
    return out


def compute_profile_shared(
    split_df: pd.DataFrame,
    user_tags: pd.DataFrame,
    item_tags: pd.DataFrame,
    profile_conf: dict[str, float],
    min_train_reviews: int,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = split_df[["user_id", "n_reviews"]].drop_duplicates()
    eligible_users = set(counts.loc[counts["n_reviews"] >= min_train_reviews, "user_id"].astype(str).tolist())
    train_df = split_df[(split_df["split_label"] == "train") & (split_df["user_id"].isin(eligible_users))].copy()
    truth_df = (
        split_df[(split_df["split_label"] == "test") & (split_df["user_id"].isin(eligible_users))][["user_id", "business_id"]]
        .rename(columns={"business_id": "true_business_id"})
        .drop_duplicates(["user_id"])
        .copy()
    )
    if truth_df.empty:
        return pd.DataFrame(columns=["user_id", "business_id", "source_rank", "source_score"]), {
            "eligible_users": 0,
            "test_users": 0,
            "truth_in_pool_at_top_k": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "recall_at_top_k": 0.0,
        }

    use_user_tags = user_tags[user_tags["user_id"].isin(set(truth_df["user_id"].tolist()))].copy()
    if use_user_tags.empty:
        return pd.DataFrame(columns=["user_id", "business_id", "source_rank", "source_score"]), {
            "eligible_users": int(len(eligible_users)),
            "test_users": int(truth_df.shape[0]),
            "truth_in_pool_at_top_k": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "recall_at_top_k": 0.0,
        }
    use_user_tags["profile_conf"] = use_user_tags["user_id"].map(profile_conf).fillna(0.0).astype(float)
    conf_gate = float(PROFILE_CONFIDENCE_FLOOR) + (1.0 - float(PROFILE_CONFIDENCE_FLOOR)) * np.clip(
        use_user_tags["profile_conf"].to_numpy(dtype=np.float32), 0.0, 1.0
    )
    use_user_tags["user_tag_weight"] = (
        use_user_tags["net_w"].to_numpy(dtype=np.float64)
        * use_user_tags["tag_confidence"].to_numpy(dtype=np.float64)
        * np.log1p(np.maximum(use_user_tags["support"].to_numpy(dtype=np.float64), 0.0))
        * conf_gate.astype(np.float64)
    )
    use_user_tags = use_user_tags[np.isfinite(use_user_tags["user_tag_weight"])][["user_id", "tag", "tag_type", "user_tag_weight"]]
    if use_user_tags.empty:
        return pd.DataFrame(columns=["user_id", "business_id", "source_rank", "source_score"]), {
            "eligible_users": int(len(eligible_users)),
            "test_users": int(truth_df.shape[0]),
            "truth_in_pool_at_top_k": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "recall_at_top_k": 0.0,
        }

    key_to_items: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in item_tags.itertuples(index=False):
        key_to_items.setdefault((str(row.tag_type), str(row.tag)), []).append((str(row.business_id), float(row.item_tag_weight)))
    user_tag_map: dict[str, list[tuple[str, str, float]]] = {}
    for row in use_user_tags.itertuples(index=False):
        user_tag_map.setdefault(str(row.user_id), []).append((str(row.tag_type), str(row.tag), float(row.user_tag_weight)))

    rows: list[dict[str, Any]] = []
    for user_id in truth_df["user_id"].astype(str).tolist():
        tags = user_tag_map.get(user_id, [])
        if not tags:
            continue
        accum: dict[str, float] = {}
        user_norm = 0.0
        for tag_type, tag, user_w in tags:
            user_norm += abs(float(user_w))
            for business_id, item_w in key_to_items.get((tag_type, tag), []):
                accum[business_id] = accum.get(business_id, 0.0) + float(user_w) * float(item_w)
        if user_norm <= 1e-6 or not accum:
            continue
        scored = [
            (business_id, float(score / user_norm))
            for business_id, score in accum.items()
            if float(score / user_norm) > float(PROFILE_SHARED_SCORE_MIN)
        ]
        if not scored:
            continue
        top = sorted(scored, key=lambda x: (-x[1], x[0]))[:top_k]
        for rank, (business_id, score) in enumerate(top, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "business_id": business_id,
                    "source_rank": rank,
                    "source_score": score,
                }
            )
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand, {
            "eligible_users": int(len(eligible_users)),
            "test_users": int(truth_df.shape[0]),
            "truth_in_pool_at_top_k": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "recall_at_top_k": 0.0,
        }
    merged = truth_df.merge(cand, left_on=["user_id", "true_business_id"], right_on=["user_id", "business_id"], how="left")
    rank = pd.to_numeric(merged["source_rank"], errors="coerce")
    metrics = {
        "eligible_users": int(len(eligible_users)),
        "test_users": int(truth_df.shape[0]),
        "users_with_candidates": int(cand["user_id"].nunique()),
        "mean_candidates_per_user": float(cand.groupby("user_id").size().mean()),
        "truth_in_pool_at_top_k": float(rank.notna().mean()),
        "recall_at_10": float((rank <= 10).fillna(False).mean()),
        "recall_at_20": float((rank <= 20).fillna(False).mean()),
        "recall_at_50": float((rank <= 50).fillna(False).mean()),
        "recall_at_top_k": float((rank <= top_k).fillna(False).mean()),
        "median_truth_rank_when_hit": float(rank.dropna().median()) if rank.notna().any() else None,
    }
    return cand, metrics


def main() -> None:
    cohort_pdf = load_user_cohort()
    user_profile_run = resolve_user_profile_run()
    item_run = resolve_item_semantic_run()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    scoped_reviews, scope_meta = load_scoped_reviews(spark, cohort_pdf)
    scoped_reviews = scoped_reviews.sort_values(["user_id", "ts", "review_id"], ascending=[True, False, False]).reset_index(drop=True)
    split_df = build_global_split(scoped_reviews)

    profile_conf = load_profile_confidence(user_profile_run)
    base_tags, treat_tags = load_user_tags(user_profile_run, INPUT_PASS2_SIDECAR_CSV or None)
    item_tags = load_item_tags(item_run, set(scoped_reviews["business_id"].astype(str).unique().tolist()))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_full_" + RUN_TAG
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    split_df.to_parquet(out_dir / "event_split.parquet", index=False)
    cohort_pdf.to_csv(out_dir / "user_cohort.csv", index=False)

    bucket_rows: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        min_user_reviews = int(bucket + MIN_USER_REVIEWS_OFFSET)
        top_k = int(min(PROFILE_TOP_K_BY_BUCKET.get(bucket, 120), PROFILE_TAG_SHARED_TOP_K))
        bucket_dir = out_dir / f"bucket_{bucket}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        baseline_cand, baseline_metrics = compute_profile_shared(
            split_df=split_df,
            user_tags=base_tags,
            item_tags=item_tags,
            profile_conf=profile_conf,
            min_train_reviews=min_user_reviews,
            top_k=top_k,
        )
        treatment_cand, treatment_metrics = compute_profile_shared(
            split_df=split_df,
            user_tags=treat_tags,
            item_tags=item_tags,
            profile_conf=profile_conf,
            min_train_reviews=min_user_reviews,
            top_k=top_k,
        )
        baseline_cand.to_parquet(bucket_dir / "baseline_profile_shared_candidates.parquet", index=False)
        treatment_cand.to_parquet(bucket_dir / "treatment_profile_shared_candidates.parquet", index=False)
        truth = (
            split_df[(split_df["split_label"] == "test") & (split_df["n_reviews"] >= min_user_reviews)][["user_id", "business_id"]]
            .rename(columns={"business_id": "true_business_id"})
            .drop_duplicates(["user_id"])
            .copy()
        )
        truth.to_parquet(bucket_dir / "truth.parquet", index=False)
        delta = {
            "truth_in_pool_at_top_k_delta": float(treatment_metrics["truth_in_pool_at_top_k"] - baseline_metrics["truth_in_pool_at_top_k"]),
            "recall_at_10_delta": float(treatment_metrics["recall_at_10"] - baseline_metrics["recall_at_10"]),
            "recall_at_20_delta": float(treatment_metrics["recall_at_20"] - baseline_metrics["recall_at_20"]),
            "recall_at_50_delta": float(treatment_metrics["recall_at_50"] - baseline_metrics["recall_at_50"]),
            "recall_at_top_k_delta": float(treatment_metrics["recall_at_top_k"] - baseline_metrics["recall_at_top_k"]),
        }
        bucket_row = {
            "bucket": int(bucket),
            "min_user_reviews": int(min_user_reviews),
            "top_k": int(top_k),
            "baseline": baseline_metrics,
            "treatment": treatment_metrics,
            "delta": delta,
        }
        bucket_rows.append(bucket_row)
        safe_json_write(bucket_dir / "profile_shared_eval_summary.json", bucket_row)

    summary = {
        "run_id": run_id,
        "input_cohort_csv": str(normalize_legacy_project_path(INPUT_COHORT_CSV)),
        "input_pass2_sidecar_csv": str(normalize_legacy_project_path(INPUT_PASS2_SIDECAR_CSV)) if INPUT_PASS2_SIDECAR_CSV else "",
        "user_profile_run": str(user_profile_run),
        "item_semantic_run": str(item_run),
        "scope_meta": scope_meta,
        "eligible_users_ge4": int(split_df["user_id"].nunique()),
        "split_rows": int(split_df.shape[0]),
        "bucket_results": bucket_rows,
        "sample_event_rows": split_df.head(SAMPLE_ROWS).to_dict(orient="records"),
    }
    safe_json_write(out_dir / "profile_route_subset_eval_summary.json", summary)
    print(f"[OK] output_dir={out_dir}")
    print(f"[OK] eligible_users_ge4={int(split_df['user_id'].nunique())} split_rows={int(split_df.shape[0])}")


if __name__ == "__main__":
    main()
