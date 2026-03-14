from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from pipeline.project_paths import env_or_project_path


RUN_TAG = "stage09_profile_calibration"

INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = env_or_project_path("OUTPUT_09_PROFILE_CALIB_ROOT_DIR", "data/output/09_profile_calibration")

NEG_POS_RATIO = float(os.getenv("CALIB_NEG_POS_RATIO", "20").strip() or 20.0)
HOLDOUT_USER_FRAC = float(os.getenv("CALIB_HOLDOUT_USER_FRAC", "0.2").strip() or 0.2)
RANDOM_SEED = int(os.getenv("CALIB_RANDOM_SEED", "42").strip() or 42)
MIN_POS_TRAIN = int(os.getenv("CALIB_MIN_POS_TRAIN", "50").strip() or 50)
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g").strip() or "4g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g").strip() or "4g"

FEATURE_NAMES = [
    "profile_raw_score",
    "profile_rank",
    "profile_confidence",
    "profile_raw_x_conf",
    "profile_raw_x_ranknorm",
]


def build_spark() -> SparkSession:
    local_dir = env_or_project_path("SPARK_LOCAL_DIR", "data/spark-tmp")
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage09-profile-calibration-train")
        .master("local[2]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run found in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def _rank_metrics(pdf: pd.DataFrame, rank_col: str) -> tuple[float, float]:
    if pdf.empty:
        return 0.0, 0.0
    g = pdf.groupby("user_idx", dropna=False)
    rows = []
    for uid, dfu in g:
        pos = dfu[dfu["label"] == 1]
        if pos.empty:
            rows.append((uid, 0))
            continue
        r = int(pos[rank_col].iloc[0])
        rows.append((uid, r))
    rank_df = pd.DataFrame(rows, columns=["user_idx", "rank"])
    recall = float((rank_df["rank"] > 0).mean())
    valid = rank_df["rank"] > 0
    if valid.any():
        ndcg = float((1.0 / np.log2(rank_df.loc[valid, "rank"].to_numpy(dtype=np.float64) + 1.0)).mean())
    else:
        ndcg = 0.0
    return recall, ndcg


def _build_features(pdf: pd.DataFrame) -> np.ndarray:
    rank = pdf["profile_rank"].to_numpy(dtype=np.float32)
    raw = pdf["profile_raw_score"].to_numpy(dtype=np.float32)
    conf = pdf["profile_confidence"].to_numpy(dtype=np.float32)
    rank_norm = 1.0 / np.log2(rank + 1.0)
    x = np.stack(
        [
            raw,
            rank,
            conf,
            raw * conf,
            raw * rank_norm,
        ],
        axis=1,
    ).astype(np.float32)
    return x


def _standardize(x_train: np.ndarray, x_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_valid - mean) / std, mean, std


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    source_run = resolve_stage09_run()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_dirs = sorted([p for p in source_run.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs under {source_run}")

    models: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []

    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        cand_path = bdir / "candidates_all.parquet"
        truth_path = bdir / "truth.parquet"
        if not cand_path.exists() or not truth_path.exists():
            continue

        print(f"[BUCKET] {bucket}")
        cand = (
            spark.read.parquet(cand_path.as_posix())
            .filter(F.col("source") == F.lit("profile"))
            .select(
                "user_idx",
                "item_idx",
                F.col("source_rank").cast("int").alias("profile_rank"),
                F.col("source_score").cast("double").alias("profile_raw_score"),
                F.coalesce(F.col("source_confidence").cast("double"), F.lit(1.0)).alias("profile_confidence"),
            )
        )
        truth = spark.read.parquet(truth_path.as_posix()).select("user_idx", "true_item_idx").dropDuplicates(["user_idx"])
        ds = (
            cand.join(truth, on="user_idx", how="left")
            .withColumn("label", F.when(F.col("item_idx") == F.col("true_item_idx"), F.lit(1)).otherwise(F.lit(0)))
            .drop("true_item_idx")
        )

        users = ds.select("user_idx").distinct().withColumn("is_valid", F.rand(seed=RANDOM_SEED + bucket) < F.lit(HOLDOUT_USER_FRAC))
        ds = ds.join(users, on="user_idx", how="left")
        train_df = ds.filter(~F.col("is_valid"))
        valid_df = ds.filter(F.col("is_valid"))

        # Downsample negatives for train stability/speed.
        pos_train = int(train_df.filter(F.col("label") == 1).count())
        neg_train = int(train_df.filter(F.col("label") == 0).count())
        if pos_train < MIN_POS_TRAIN:
            print(f"[WARN] bucket={bucket} skip: train positives={pos_train} < {MIN_POS_TRAIN}")
            summaries.append(
                {
                    "bucket": bucket,
                    "status": "skip_low_positive",
                    "train_pos": pos_train,
                    "train_neg": neg_train,
                }
            )
            continue
        keep_prob = 1.0
        if neg_train > 0 and NEG_POS_RATIO > 0:
            keep_prob = min(1.0, (NEG_POS_RATIO * float(pos_train)) / float(neg_train))
        train_df = train_df.withColumn("rnd", F.rand(seed=RANDOM_SEED + bucket + 100))
        train_df = train_df.filter((F.col("label") == 1) | (F.col("rnd") <= F.lit(keep_prob))).drop("rnd")

        train_pdf = train_df.toPandas()
        valid_pdf = valid_df.toPandas()
        if train_pdf.empty or valid_pdf.empty:
            summaries.append({"bucket": bucket, "status": "skip_empty_split"})
            continue

        x_train = _build_features(train_pdf)
        y_train = train_pdf["label"].to_numpy(dtype=np.int32)
        x_valid = _build_features(valid_pdf)
        y_valid = valid_pdf["label"].to_numpy(dtype=np.int32)
        x_train_z, x_valid_z, mean, std = _standardize(x_train, x_valid)

        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_SEED + bucket,
            max_iter=300,
        )
        clf.fit(x_train_z, y_train)
        p_valid = clf.predict_proba(x_valid_z)[:, 1].astype(np.float64)
        valid_pdf = valid_pdf.copy()
        valid_pdf["calib_score"] = p_valid

        # Baseline: profile original rank.
        baseline = valid_pdf[["user_idx", "label", "profile_rank"]].copy()
        base_recall, base_ndcg = _rank_metrics(baseline, "profile_rank")

        # Calibrated: rerank by predicted probability within user.
        valid_pdf = valid_pdf.sort_values(["user_idx", "calib_score"], ascending=[True, False]).copy()
        valid_pdf["calib_rank"] = valid_pdf.groupby("user_idx").cumcount() + 1
        calib_eval = valid_pdf[["user_idx", "label", "calib_rank"]].copy()
        calib_recall, calib_ndcg = _rank_metrics(calib_eval, "calib_rank")

        auc = float("nan")
        ll = float("nan")
        try:
            if len(np.unique(y_valid)) > 1:
                auc = float(roc_auc_score(y_valid, p_valid))
            ll = float(log_loss(y_valid, p_valid, labels=[0, 1]))
        except Exception:
            pass

        model_payload = {
            "feature_names": FEATURE_NAMES,
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "coef": clf.coef_[0].astype(float).tolist(),
            "intercept": float(clf.intercept_[0]),
            "metrics": {
                "baseline_recall_at10": float(base_recall),
                "baseline_ndcg_at10": float(base_ndcg),
                "calibrated_recall_at10": float(calib_recall),
                "calibrated_ndcg_at10": float(calib_ndcg),
                "delta_ndcg_at10": float(calib_ndcg - base_ndcg),
                "valid_auc": None if (isinstance(auc, float) and math.isnan(auc)) else float(auc),
                "valid_logloss": None if (isinstance(ll, float) and math.isnan(ll)) else float(ll),
                "train_rows": int(len(train_pdf)),
                "valid_rows": int(len(valid_pdf)),
                "train_pos": int((y_train == 1).sum()),
                "valid_pos": int((y_valid == 1).sum()),
                "neg_keep_prob": float(keep_prob),
            },
        }
        models[str(bucket)] = model_payload
        summaries.append({"bucket": bucket, "status": "ok", **model_payload["metrics"]})
        print(
            f"[METRIC] bucket={bucket} base_ndcg={base_ndcg:.4f} "
            f"calib_ndcg={calib_ndcg:.4f} delta={calib_ndcg - base_ndcg:+.4f}"
        )

    payload = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage09_run": str(source_run),
        "feature_names": FEATURE_NAMES,
        "models_by_bucket": models,
        "summaries": summaries,
    }
    out_json = out_dir / "profile_calibration.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] wrote calibration json: {out_json}")

    spark.stop()


if __name__ == "__main__":
    main()
