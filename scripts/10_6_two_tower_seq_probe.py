from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window


RUN_TAG = "stage10_6_two_tower_seq_probe"
PARQUET_BASE = Path(r"D:/5006 BDA project/data/parquet")
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/10_6_two_tower_seq_probe")
METRICS_PATH = Path(r"D:/5006 BDA project/data/metrics/stage10_6_two_tower_seq_probe_results.csv")

RUN_PROFILE = os.getenv("RUN_PROFILE_OVERRIDE", "sample").strip().lower() or "sample"
RANDOM_SEED = int(os.getenv("PROBE_RANDOM_SEED", "42").strip() or 42)
SAMPLE_FRAC = float(os.getenv("PROBE_SAMPLE_FRAC", "0.15").strip() or 0.15)
TOP_K = int(os.getenv("PROBE_TOP_K", "10").strip() or 10)
MIN_USER_REVIEWS_OFFSET = int(os.getenv("PROBE_MIN_USER_REVIEWS_OFFSET", "2").strip() or 2)
BUCKETS_DEFAULT = [2, 5]

TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
FILTER_POLICY: dict[str, Any] = {
    "require_is_open": True,
    "min_business_stars": 3.0,
    "min_business_review_count": 20,
    "stale_cutoff_date": "2020-01-01",
}

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", "D:/5006 BDA project/data/spark-tmp").strip() or "D:/5006 BDA project/data/spark-tmp"

ENABLE_ALS_BASELINE = os.getenv("PROBE_ENABLE_ALS_BASELINE", "true").strip().lower() == "true"
TOWER_DIM = int(os.getenv("TOWER_DIM", "64").strip() or 64)
TOWER_EPOCHS = int(os.getenv("TOWER_EPOCHS", "8").strip() or 8)
TOWER_LR = float(os.getenv("TOWER_LR", "0.04").strip() or 0.04)
TOWER_REG = float(os.getenv("TOWER_REG", "0.0001").strip() or 0.0001)
TOWER_NEG_TRIES = int(os.getenv("TOWER_NEG_TRIES", "6").strip() or 6)
TOWER_STEPS_PER_EPOCH = int(os.getenv("TOWER_STEPS_PER_EPOCH", "0").strip() or 0)

SEQ_RECENT_LEN = int(os.getenv("SEQ_RECENT_LEN", "10").strip() or 10)
SEQ_RECENT_MIN_LEN = int(os.getenv("SEQ_RECENT_MIN_LEN", "2").strip() or 2)
SEQ_DECAY = float(os.getenv("SEQ_DECAY", "0.85").strip() or 0.85)
ALPHA_GRID = [
    float(np.clip(float(x.strip()), 0.0, 1.0))
    for x in os.getenv("SEQ_BLEND_ALPHA_GRID", "0.0,0.1,0.2,0.35,0.5,0.7,1.0").split(",")
    if x.strip()
]
if not ALPHA_GRID:
    ALPHA_GRID = [0.0, 0.3, 0.6, 1.0]
ALPHA_GRID = sorted(set(ALPHA_GRID))

RESULT_FIELDS = [
    "run_id",
    "run_profile",
    "bucket_min_train_reviews",
    "split",
    "model",
    "recall_at_k",
    "ndcg_at_k",
    "k",
    "n_users",
    "n_items",
    "n_train",
    "density",
    "seq_ready_rate",
    "best_alpha",
]


def parse_buckets(raw: str) -> list[int]:
    out: list[int] = []
    for p in (raw or "").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return sorted(set(out))


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDS).writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def build_spark() -> SparkSession:
    ws = Path(__file__).resolve().parents[1]
    temp_dir = Path(os.getenv("PY_TEMP_DIR", "").strip() or str(ws / "tmp_py_runtime"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    local_dir = Path(SPARK_LOCAL_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage10-6-two-tower-seq-probe")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_business_pool(spark: SparkSession) -> tuple[DataFrame, dict[str, Any]]:
    business = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories", "is_open", "stars", "review_count")
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
    before = int(biz.count())
    if FILTER_POLICY["require_is_open"]:
        biz = biz.filter(F.col("is_open") == 1)
    biz = biz.filter(F.col("stars") >= F.lit(float(FILTER_POLICY["min_business_stars"])))
    biz = biz.filter(F.col("review_count") >= F.lit(int(FILTER_POLICY["min_business_review_count"])))
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("business_id", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
    )
    last_review = rvw.groupBy("business_id").agg(F.max("ts").alias("last_review_ts"))
    cutoff = str(FILTER_POLICY["stale_cutoff_date"]).strip()
    biz = biz.join(last_review, on="business_id", how="left").filter(F.col("last_review_ts") >= F.to_timestamp(F.lit(cutoff)))
    biz = biz.persist(StorageLevel.DISK_ONLY)
    return biz, {"n_scope_before_hard_filter": before, "n_scope_after_hard_filter": int(biz.count()), "stale_cutoff_date": cutoff}


def load_interactions(spark: SparkSession, biz: DataFrame) -> DataFrame:
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "stars", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(biz.select("business_id"), on="business_id", how="inner")
    )
    if RUN_PROFILE == "sample":
        rvw = rvw.sample(False, SAMPLE_FRAC, RANDOM_SEED)
    return rvw.persist(StorageLevel.DISK_ONLY)


def leave_two_out(rvw: DataFrame, min_user_reviews: int) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    users = rvw.groupBy("user_id").agg(F.count("*").alias("n")).filter(F.col("n") >= min_user_reviews).select("user_id")
    rvw2 = rvw.join(users, on="user_id", how="inner")
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = rvw2.withColumn("rn", F.row_number().over(w))
    return rvw2, ranked.filter("rn>2"), ranked.filter("rn=2"), ranked.filter("rn=1")


def index_ids(full_df: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    uidx = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip").fit(full_df)
    iidx = StringIndexer(inputCol="business_id", outputCol="item_idx", handleInvalid="skip").fit(full_df)
    def t(df: DataFrame) -> DataFrame:
        out = iidx.transform(uidx.transform(df))
        return out.withColumn("user_idx", F.col("user_idx").cast("int")).withColumn("item_idx", F.col("item_idx").cast("int"))
    return t(train), t(valid), t(test)


def metrics(topk: dict[int, list[int]], truth: dict[int, int], k: int) -> tuple[float, float]:
    if not truth:
        return 0.0, 0.0
    hit, ndcg = 0, 0.0
    for u, t in truth.items():
        r = 0
        for i, x in enumerate(topk.get(u, [])[:k], 1):
            if int(x) == int(t):
                r = i
                break
        if r > 0:
            hit += 1
            ndcg += 1.0 / math.log2(r + 1.0)
    n = float(len(truth))
    return hit / n, ndcg / n


def topk_scores(user_ids: np.ndarray, user_vecs: np.ndarray, item_vecs: np.ndarray, seen: dict[int, set[int]], k: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    n_items = item_vecs.shape[0]
    for uid, uvec in zip(user_ids.tolist(), user_vecs):
        s = item_vecs @ uvec
        st = seen.get(int(uid), set())
        if st:
            idx = np.fromiter(st, dtype=np.int32, count=len(st))
            idx = idx[(idx >= 0) & (idx < n_items)]
            if idx.size > 0:
                s[idx] = -1e12
        part = np.argpartition(s, -k)[-k:] if k < n_items else np.arange(n_items)
        rank = part[np.argsort(s[part])[::-1]]
        out[int(uid)] = [int(x) for x in rank[:k].tolist()]
    return out


def train_bpr(n_users: int, n_items: int, pairs: np.ndarray, pos: dict[int, set[int]]) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    u = (rng.normal(0, 0.1, size=(n_users, TOWER_DIM)) / math.sqrt(TOWER_DIM)).astype(np.float32)
    v = (rng.normal(0, 0.1, size=(n_items, TOWER_DIM)) / math.sqrt(TOWER_DIM)).astype(np.float32)
    if pairs.shape[0] == 0:
        return u, v, 0
    steps = int(TOWER_STEPS_PER_EPOCH) if TOWER_STEPS_PER_EPOCH > 0 else int(np.clip(pairs.shape[0] * 6, 5000, 60000))
    updates = 0
    for e in range(TOWER_EPOCHS):
        lr = TOWER_LR * (0.95 ** e)
        for _ in range(steps):
            idx = int(rng.integers(0, pairs.shape[0]))
            uid, iid = int(pairs[idx, 0]), int(pairs[idx, 1])
            pset = pos.get(uid)
            if not pset or len(pset) >= n_items:
                continue
            j = int(rng.integers(0, n_items))
            t = 0
            while j in pset and t < TOWER_NEG_TRIES:
                j = int(rng.integers(0, n_items))
                t += 1
            if j in pset:
                continue
            u0, i0, j0 = u[uid].copy(), v[iid].copy(), v[j].copy()
            x = float(np.clip(np.dot(u0, i0) - np.dot(u0, j0), -20.0, 20.0))
            g = 1.0 - (1.0 / (1.0 + math.exp(-x)))
            u[uid] = u0 + lr * (g * (i0 - j0) - TOWER_REG * u0)
            v[iid] = i0 + lr * (g * u0 - TOWER_REG * i0)
            v[j] = j0 + lr * (-g * u0 - TOWER_REG * j0)
            updates += 1
    return u, v, updates


def seq_vectors(train_pdf: pd.DataFrame, item_vecs: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    seq = np.zeros((n_users, item_vecs.shape[1]), dtype=np.float32)
    ready = np.zeros((n_users,), dtype=np.int32)
    if train_pdf.empty:
        return seq, ready
    gdf = train_pdf.sort_values(["user_idx", "ts"], ascending=[True, False]).groupby("user_idx", sort=False)
    for uid, g in gdf:
        ids = g["item_idx"].astype(np.int32).tolist()[:SEQ_RECENT_LEN]
        if len(ids) < SEQ_RECENT_MIN_LEN:
            continue
        arr = np.array(ids, dtype=np.int32)
        arr = arr[(arr >= 0) & (arr < item_vecs.shape[0])]
        if arr.size < SEQ_RECENT_MIN_LEN:
            continue
        w = np.array([SEQ_DECAY ** i for i in range(arr.size)], dtype=np.float32)
        w = w / max(1e-6, float(w.sum()))
        seq[int(uid)] = (item_vecs[arr] * w[:, None]).sum(axis=0).astype(np.float32)
        ready[int(uid)] = 1
    return seq, ready


def als_eval(spark: SparkSession, train_idx: DataFrame, truth_idx: DataFrame, k: int) -> tuple[float, float]:
    model = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="rating", implicitPrefs=True, rank=20, regParam=0.1, alpha=20.0, coldStartStrategy="drop", nonnegative=True).fit(
        train_idx.select("user_idx", "item_idx").withColumn("rating", F.lit(1.0))
    )
    rec = (
        model.recommendForUserSubset(truth_idx.select("user_idx").distinct(), k)
        .select("user_idx", F.posexplode("recommendations").alias("pos", "rec"))
        .select(F.col("user_idx").cast("int").alias("user_idx"), F.col("rec.item_idx").cast("int").alias("item_idx"), (F.col("pos") + 1).alias("rank"))
    ).toPandas()
    truth = truth_idx.select("user_idx", "item_idx").toPandas()
    tmap = dict(zip(truth["user_idx"].astype(int), truth["item_idx"].astype(int)))
    pmap: dict[int, list[int]] = {}
    if not rec.empty:
        rec = rec.sort_values(["user_idx", "rank"], ascending=[True, True])
        for uid, g in rec.groupby("user_idx", sort=False):
            pmap[int(uid)] = g["item_idx"].astype(int).head(k).tolist()
    return metrics(pmap, tmap, k)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    buckets = parse_buckets(os.getenv("MIN_TRAIN_BUCKETS_OVERRIDE", "").strip()) or BUCKETS_DEFAULT
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_PROFILE}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_results_file(METRICS_PATH)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    print(f"[CONFIG] profile={RUN_PROFILE} buckets={buckets} topk={TOP_K} dim={TOWER_DIM} epochs={TOWER_EPOCHS} alpha_grid={ALPHA_GRID}")
    biz, biz_stats = load_business_pool(spark)
    rvw = load_interactions(spark, biz)
    print(f"[COUNT] interactions={rvw.count()} users={rvw.select('user_id').distinct().count()}")

    summaries: list[dict[str, Any]] = []
    for b in buckets:
        min_user = int(b + MIN_USER_REVIEWS_OFFSET)
        print(f"\n[BUCKET] min_train={b} min_user_reviews={min_user}")
        full_df, train, valid, test = leave_two_out(rvw, min_user)
        n_train, n_valid, n_test = int(train.count()), int(valid.count()), int(test.count())
        if min(n_train, n_valid, n_test) <= 0:
            summaries.append({"bucket": b, "status": "empty"})
            continue
        train_idx, valid_idx, test_idx = index_ids(full_df, train, valid, test)
        train_pdf = train_idx.select("user_idx", "item_idx", "ts").toPandas()
        valid_pdf = valid_idx.select("user_idx", "item_idx").toPandas()
        test_pdf = test_idx.select("user_idx", "item_idx").toPandas()
        for df in (train_pdf, valid_pdf, test_pdf):
            df["user_idx"] = df["user_idx"].astype(int)
            df["item_idx"] = df["item_idx"].astype(int)

        n_users = int(max(train_pdf["user_idx"].max(), valid_pdf["user_idx"].max(), test_pdf["user_idx"].max()) + 1)
        n_items = int(max(train_pdf["item_idx"].max(), valid_pdf["item_idx"].max(), test_pdf["item_idx"].max()) + 1)
        density = float(n_train) / float(max(1, n_users * n_items))

        pos: dict[int, set[int]] = {}
        for r in train_pdf.itertuples(index=False):
            pos.setdefault(int(r.user_idx), set()).add(int(r.item_idx))
        pairs = train_pdf[["user_idx", "item_idx"]].to_numpy(dtype=np.int32, copy=True)
        valid_truth = dict(zip(valid_pdf["user_idx"].tolist(), valid_pdf["item_idx"].tolist()))
        test_truth = dict(zip(test_pdf["user_idx"].tolist(), test_pdf["item_idx"].tolist()))
        valid_users = np.array(sorted(valid_truth.keys()), dtype=np.int32)
        test_users = np.array(sorted(test_truth.keys()), dtype=np.int32)

        als_valid = als_test = (0.0, 0.0)
        if ENABLE_ALS_BASELINE:
            als_valid = als_eval(spark, train_idx, valid_idx, TOP_K)
            als_test = als_eval(spark, train_idx, test_idx, TOP_K)

        uvec, ivec, updates = train_bpr(n_users, n_items, pairs, pos)
        tt_valid = metrics(topk_scores(valid_users, uvec[valid_users], ivec, pos, TOP_K), valid_truth, TOP_K)
        tt_test = metrics(topk_scores(test_users, uvec[test_users], ivec, pos, TOP_K), test_truth, TOP_K)

        svec, ready = seq_vectors(train_pdf, ivec, n_users)
        seq_ready_rate = float(ready.sum()) / float(max(1, n_users))
        seq_valid = metrics(topk_scores(valid_users, svec[valid_users], ivec, pos, TOP_K), valid_truth, TOP_K)
        seq_test = metrics(topk_scores(test_users, svec[test_users], ivec, pos, TOP_K), test_truth, TOP_K)

        best_alpha, best_valid = 0.0, (-1.0, -1.0)
        alpha_rows = []
        mask = ready.astype(bool)
        for a in ALPHA_GRID:
            bvec = uvec.copy()
            bvec[mask] = (1.0 - a) * uvec[mask] + a * svec[mask]
            vm = metrics(topk_scores(valid_users, bvec[valid_users], ivec, pos, TOP_K), valid_truth, TOP_K)
            alpha_rows.append({"alpha": float(a), "valid_recall": vm[0], "valid_ndcg": vm[1]})
            if vm[1] > best_valid[1] or (abs(vm[1] - best_valid[1]) <= 1e-12 and vm[0] > best_valid[0]):
                best_alpha, best_valid = float(a), vm
        bvec = uvec.copy()
        bvec[mask] = (1.0 - best_alpha) * uvec[mask] + best_alpha * svec[mask]
        blend_test = metrics(topk_scores(test_users, bvec[test_users], ivec, pos, TOP_K), test_truth, TOP_K)

        rows = []
        if ENABLE_ALS_BASELINE:
            rows += [
                {"split": "valid", "model": "ALS", "recall_at_k": als_valid[0], "ndcg_at_k": als_valid[1]},
                {"split": "test", "model": "ALS", "recall_at_k": als_test[0], "ndcg_at_k": als_test[1]},
            ]
        rows += [
            {"split": "valid", "model": "TwoTowerID", "recall_at_k": tt_valid[0], "ndcg_at_k": tt_valid[1]},
            {"split": "test", "model": "TwoTowerID", "recall_at_k": tt_test[0], "ndcg_at_k": tt_test[1]},
            {"split": "valid", "model": "SeqOnly", "recall_at_k": seq_valid[0], "ndcg_at_k": seq_valid[1]},
            {"split": "test", "model": "SeqOnly", "recall_at_k": seq_test[0], "ndcg_at_k": seq_test[1]},
            {"split": "valid", "model": "TwoTowerSeqBlend", "recall_at_k": best_valid[0], "ndcg_at_k": best_valid[1]},
            {"split": "test", "model": "TwoTowerSeqBlend", "recall_at_k": blend_test[0], "ndcg_at_k": blend_test[1]},
        ]
        for r in rows:
            row = {
                "run_id": run_id,
                "run_profile": RUN_PROFILE,
                "bucket_min_train_reviews": int(b),
                "split": r["split"],
                "model": r["model"],
                "recall_at_k": r["recall_at_k"],
                "ndcg_at_k": r["ndcg_at_k"],
                "k": int(TOP_K),
                "n_users": int(n_users),
                "n_items": int(n_items),
                "n_train": int(n_train),
                "density": float(density),
                "seq_ready_rate": float(seq_ready_rate),
                "best_alpha": best_alpha if r["model"] == "TwoTowerSeqBlend" else "",
            }
            append_result(METRICS_PATH, row)

        bout = out_dir / f"bucket_{int(b)}"
        bout.mkdir(parents=True, exist_ok=True)
        summary = {
            "bucket_min_train_reviews": int(b),
            "min_user_reviews": int(min_user),
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_train": int(n_train),
            "n_valid": int(n_valid),
            "n_test": int(n_test),
            "density": float(density),
            "seq_ready_rate": float(seq_ready_rate),
            "tower_updates": int(updates),
            "alpha_grid": ALPHA_GRID,
            "alpha_valid": alpha_rows,
            "best_alpha": float(best_alpha),
            "test_metrics": {
                "als": {"recall": als_test[0], "ndcg": als_test[1]} if ENABLE_ALS_BASELINE else None,
                "two_tower": {"recall": tt_test[0], "ndcg": tt_test[1]},
                "seq_only": {"recall": seq_test[0], "ndcg": seq_test[1]},
                "blend": {"recall": blend_test[0], "ndcg": blend_test[1]},
            },
        }
        (bout / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(bout / "metrics_rows.csv", index=False, encoding="utf-8-sig")
        summaries.append(summary)
        print(f"[METRIC] bucket={b} test TwoTower NDCG@{TOP_K}={tt_test[1]:.6f} Blend NDCG@{TOP_K}={blend_test[1]:.6f} best_alpha={best_alpha:.2f}")

    run_meta = {
        "run_id": run_id,
        "run_profile": RUN_PROFILE,
        "run_tag": RUN_TAG,
        "output_dir": str(out_dir),
        "metrics_path": str(METRICS_PATH),
        "buckets": buckets,
        "biz_stats": biz_stats,
        "summaries": summaries,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")
    spark.stop()
    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
