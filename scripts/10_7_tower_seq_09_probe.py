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
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window


RUN_TAG = "stage10_7_tower_seq_09_probe"
PARQUET_BASE = Path(r"D:/5006 BDA project/data/parquet")
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = Path(r"D:/5006 BDA project/data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/10_7_tower_seq_09_probe")
METRICS_PATH = Path(r"D:/5006 BDA project/data/metrics/stage10_7_tower_seq_09_probe_results.csv")

RANDOM_SEED = int(os.getenv("PROBE_RANDOM_SEED", "42").strip() or 42)
TOP_K = int(os.getenv("PROBE_TOP_K", "10").strip() or 10)
POOL_K = int(os.getenv("PROBE_POOL_K", "150").strip() or 150)
MIN_USER_REVIEWS_OFFSET = int(os.getenv("PROBE_MIN_USER_REVIEWS_OFFSET", "2").strip() or 2)
RUN_PROFILE_OVERRIDE = os.getenv("RUN_PROFILE_OVERRIDE", "").strip().lower()

TARGET_STATE = "LA"
REQUIRE_RESTAURANTS = True
REQUIRE_FOOD = True
SAMPLE_FRAC = float(os.getenv("PROBE_SAMPLE_FRAC", "0.15").strip() or 0.15)
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

TOWER_DIM = int(os.getenv("TOWER_DIM", "64").strip() or 64)
TOWER_EPOCHS = int(os.getenv("TOWER_EPOCHS", "8").strip() or 8)
TOWER_LR = float(os.getenv("TOWER_LR", "0.04").strip() or 0.04)
TOWER_REG = float(os.getenv("TOWER_REG", "0.0001").strip() or 0.0001)
TOWER_NEG_TRIES = int(os.getenv("TOWER_NEG_TRIES", "6").strip() or 6)
TOWER_STEPS_PER_EPOCH = int(os.getenv("TOWER_STEPS_PER_EPOCH", "0").strip() or 0)

SEQ_RECENT_LEN = int(os.getenv("SEQ_RECENT_LEN", "10").strip() or 10)
SEQ_RECENT_MIN_LEN = int(os.getenv("SEQ_RECENT_MIN_LEN", "2").strip() or 2)
SEQ_DECAY = float(os.getenv("SEQ_DECAY", "0.85").strip() or 0.85)

WT_GRID = [float(x.strip()) for x in os.getenv("TOWER_WEIGHT_GRID", "0.0,0.05,0.1,0.15,0.2,0.3,0.4").split(",") if x.strip()]
WS_GRID = [float(x.strip()) for x in os.getenv("SEQ_WEIGHT_GRID", "0.0,0.05,0.1,0.15,0.2").split(",") if x.strip()]
if not WT_GRID:
    WT_GRID = [0.0, 0.1, 0.2, 0.3]
if not WS_GRID:
    WS_GRID = [0.0, 0.05, 0.1]
WT_GRID = sorted(set([float(np.clip(x, 0.0, 2.0)) for x in WT_GRID]))
WS_GRID = sorted(set([float(np.clip(x, 0.0, 2.0)) for x in WS_GRID]))

RESULT_FIELDS = [
    "run_id",
    "source_run_09",
    "bucket",
    "split",
    "model",
    "recall_at_k",
    "ndcg_at_k",
    "n_users",
    "n_items",
    "n_train_pairs",
    "density",
    "seq_ready_rate",
    "w_t",
    "w_s",
    "eval_users",
]


def ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDS).writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = Path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def parse_bucket_dirs(stage09_run: Path) -> list[Path]:
    dirs = [p for p in stage09_run.iterdir() if p.is_dir() and p.name.startswith("bucket_")]
    dirs.sort(key=lambda x: int(x.name.split("_")[-1]))
    raw = os.getenv("MIN_TRAIN_BUCKETS_OVERRIDE", "").strip()
    if not raw:
        return dirs
    keep = set()
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            keep.add(int(p))
        except ValueError:
            continue
    return [d for d in dirs if int(d.name.split("_")[-1]) in keep]


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
        SparkSession.builder.appName("stage10-7-tower-seq-09-probe")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_business_pool(spark: SparkSession) -> DataFrame:
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
    cutoff = str(FILTER_POLICY["stale_cutoff_date"]).strip()
    last_review = rvw.groupBy("business_id").agg(F.max("ts").alias("last_review_ts"))
    return biz.join(last_review, on="business_id", how="left").filter(F.col("last_review_ts") >= F.to_timestamp(F.lit(cutoff)))


def load_interactions(spark: SparkSession, biz: DataFrame, run_profile: str) -> DataFrame:
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("review_id", "user_id", "business_id", "stars", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
        .join(biz.select("business_id"), on="business_id", how="inner")
    )
    if run_profile == "sample":
        rvw = rvw.sample(False, SAMPLE_FRAC, RANDOM_SEED)
    return rvw.persist(StorageLevel.DISK_ONLY)


def leave_two_out(rvw: DataFrame, min_user_reviews: int) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    users = rvw.groupBy("user_id").agg(F.count("*").alias("n")).filter(F.col("n") >= min_user_reviews).select("user_id")
    rvw2 = rvw.join(users, on="user_id", how="inner")
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("review_id").desc())
    ranked = rvw2.withColumn("rn", F.row_number().over(w))
    return rvw2, ranked.filter("rn>2"), ranked.filter("rn=2"), ranked.filter("rn=1")


def inv_from_scores(scores: np.ndarray) -> np.ndarray:
    n = int(scores.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    order = np.argsort(scores)[::-1]
    ranks = np.empty((n,), dtype=np.int32)
    ranks[order] = np.arange(1, n + 1, dtype=np.int32)
    return (1.0 / np.log2(ranks.astype(np.float64) + 1.0)).astype(np.float32)


def train_bpr(n_users: int, n_items: int, pairs: np.ndarray, user_pos: dict[int, set[int]]) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    user_vec = (rng.normal(0, 0.1, size=(n_users, TOWER_DIM)) / math.sqrt(TOWER_DIM)).astype(np.float32)
    item_vec = (rng.normal(0, 0.1, size=(n_items, TOWER_DIM)) / math.sqrt(TOWER_DIM)).astype(np.float32)
    if pairs.shape[0] == 0:
        return user_vec, item_vec, 0
    steps = int(TOWER_STEPS_PER_EPOCH) if TOWER_STEPS_PER_EPOCH > 0 else int(np.clip(pairs.shape[0] * 6, 5000, 60000))
    updates = 0
    for e in range(TOWER_EPOCHS):
        lr = TOWER_LR * (0.95**e)
        for _ in range(steps):
            ridx = int(rng.integers(0, pairs.shape[0]))
            uid = int(pairs[ridx, 0]); iid = int(pairs[ridx, 1])
            pset = user_pos.get(uid)
            if not pset or len(pset) >= n_items:
                continue
            j = int(rng.integers(0, n_items))
            t = 0
            while j in pset and t < TOWER_NEG_TRIES:
                j = int(rng.integers(0, n_items)); t += 1
            if j in pset:
                continue
            u0 = user_vec[uid].copy(); i0 = item_vec[iid].copy(); j0 = item_vec[j].copy()
            x = float(np.clip(np.dot(u0, i0) - np.dot(u0, j0), -20.0, 20.0))
            g = 1.0 - (1.0 / (1.0 + math.exp(-x)))
            user_vec[uid] = u0 + lr * (g * (i0 - j0) - TOWER_REG * u0)
            item_vec[iid] = i0 + lr * (g * u0 - TOWER_REG * i0)
            item_vec[j] = j0 + lr * (-g * u0 - TOWER_REG * j0)
            updates += 1
    return user_vec, item_vec, updates


def seq_vectors(train_pdf: pd.DataFrame, item_vec: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    seq = np.zeros((n_users, item_vec.shape[1]), dtype=np.float32)
    ready = np.zeros((n_users,), dtype=np.int32)
    if train_pdf.empty:
        return seq, ready
    grouped = train_pdf.sort_values(["user_idx", "ts"], ascending=[True, False]).groupby("user_idx", sort=False)
    for uid, g in grouped:
        ids = g["item_idx"].astype(np.int32).tolist()[:SEQ_RECENT_LEN]
        if len(ids) < SEQ_RECENT_MIN_LEN:
            continue
        arr = np.array(ids, dtype=np.int32)
        arr = arr[(arr >= 0) & (arr < item_vec.shape[0])]
        if arr.size < SEQ_RECENT_MIN_LEN:
            continue
        w = np.array([SEQ_DECAY**i for i in range(arr.size)], dtype=np.float32)
        w = w / max(1e-6, float(w.sum()))
        seq[int(uid)] = (item_vec[arr] * w[:, None]).sum(axis=0)
        ready[int(uid)] = 1
    return seq, ready


def eval_blocks(blocks: dict[int, dict[str, Any]], truth: dict[int, int], w_t: float, w_s: float) -> tuple[float, float]:
    hits, ndcg = 0, 0.0
    n = int(len(truth))
    if n <= 0:
        return 0.0, 0.0
    for uid, t_item in truth.items():
        b = blocks.get(int(uid))
        if b is None:
            continue
        score = b["pre_inv"] + float(w_t) * b["tower_inv"] + float(w_s) * b["seq_inv"]
        if score.size <= TOP_K:
            order = np.argsort(score)[::-1]
        else:
            idx = np.argpartition(score, -TOP_K)[-TOP_K:]
            order = idx[np.argsort(score[idx])[::-1]]
        top = b["items"][order]
        pos = np.where(top == int(t_item))[0]
        if pos.size > 0:
            r = int(pos[0]) + 1
            hits += 1
            ndcg += 1.0 / math.log2(r + 1.0)
    return float(hits / n), float(ndcg / n)


def main() -> None:
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    stage09_run = resolve_stage09_run()
    run_meta_09 = json.loads((stage09_run / "run_meta.json").read_text(encoding="utf-8"))
    run_profile = RUN_PROFILE_OVERRIDE or str(run_meta_09.get("run_profile", "full")).strip().lower()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{run_profile}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_results_file(METRICS_PATH)

    spark = build_spark(); spark.sparkContext.setLogLevel("WARN")
    biz = load_business_pool(spark)
    rvw = load_interactions(spark, biz, run_profile=run_profile)
    print(f"[CONFIG] stage09={stage09_run.name} run_profile={run_profile} top_k={TOP_K} pool_k={POOL_K} buckets={[p.name for p in parse_bucket_dirs(stage09_run)]}")
    print(f"[COUNT] interactions={rvw.count()} users={rvw.select('user_id').distinct().count()}")

    all_summary: list[dict[str, Any]] = []
    for bdir in parse_bucket_dirs(stage09_run):
        bucket = int(bdir.name.split("_")[-1]); min_user = int(bucket + MIN_USER_REVIEWS_OFFSET)
        print(f"\n[BUCKET] {bucket}")
        cand_pre_df = spark.read.parquet((bdir / "candidates_pretrim150.parquet").as_posix())
        cand_pre = cand_pre_df.select("user_idx", "item_idx", "pre_rank", "als_rank").toPandas()
        cand_all_map = cand_pre_df.select("item_idx", "business_id").dropDuplicates(["item_idx"]).toPandas()
        truth_map_pdf = (
            spark.read.parquet((bdir / "truth.parquet").as_posix())
            .select("user_idx", "user_id", "true_item_idx")
            .dropDuplicates(["user_idx"])
            .toPandas()
        )
        user_map = truth_map_pdf[["user_idx", "user_id"]].drop_duplicates("user_idx").copy()
        user_map["user_idx"] = user_map["user_idx"].astype(int)
        item_map = cand_all_map.drop_duplicates("item_idx").copy()
        item_map["item_idx"] = item_map["item_idx"].astype(int)

        full_df, train, valid, test = leave_two_out(rvw, min_user_reviews=min_user)
        train_pdf = train.select("user_id", "business_id", "ts").toPandas()
        valid_pdf = valid.select("user_id", "business_id").toPandas()
        test_pdf = test.select("user_id", "business_id").toPandas()

        train_idx = train_pdf.merge(user_map, on="user_id", how="inner").merge(item_map, on="business_id", how="inner")
        valid_idx = valid_pdf.merge(user_map, on="user_id", how="inner").merge(item_map, on="business_id", how="inner")
        test_idx = test_pdf.merge(user_map, on="user_id", how="inner").merge(item_map, on="business_id", how="inner")
        for df in (train_idx, valid_idx, test_idx, cand_pre, truth_map_pdf):
            df["user_idx"] = pd.to_numeric(df["user_idx"], errors="coerce").fillna(-1).astype(int)
        for df in (train_idx, valid_idx, test_idx, cand_pre):
            df["item_idx"] = pd.to_numeric(df["item_idx"], errors="coerce").fillna(-1).astype(int)
        truth_map_pdf["true_item_idx"] = pd.to_numeric(truth_map_pdf["true_item_idx"], errors="coerce").fillna(-1).astype(int)
        cand_pre = cand_pre[(cand_pre["user_idx"] >= 0) & (cand_pre["item_idx"] >= 0)].copy()
        truth_map_pdf = truth_map_pdf[(truth_map_pdf["user_idx"] >= 0) & (truth_map_pdf["true_item_idx"] >= 0)].copy()

        if train_idx.empty or valid_idx.empty or test_idx.empty or cand_pre.empty:
            print(f"[WARN] bucket={bucket} empty mapped data")
            continue

        n_users = int(max(cand_pre["user_idx"].max(), train_idx["user_idx"].max()) + 1)
        n_items = int(max(cand_pre["item_idx"].max(), train_idx["item_idx"].max()) + 1)
        density = float(train_idx.shape[0]) / float(max(1, n_users * n_items))
        user_pos: dict[int, set[int]] = {}
        for r in train_idx[["user_idx", "item_idx"]].itertuples(index=False):
            user_pos.setdefault(int(r.user_idx), set()).add(int(r.item_idx))

        pairs = train_idx[["user_idx", "item_idx"]].to_numpy(dtype=np.int32, copy=True)
        uvec, ivec, updates = train_bpr(n_users, n_items, pairs, user_pos)
        svec, sready = seq_vectors(train_idx[["user_idx", "item_idx", "ts"]].copy(), ivec, n_users)
        seq_ready_rate = float(sready.sum()) / float(max(1, n_users))

        cand_pre["pre_rank"] = pd.to_numeric(cand_pre["pre_rank"], errors="coerce").fillna(9999.0).astype(np.float32)
        cand_pre["pre_inv"] = (1.0 / np.log2(cand_pre["pre_rank"].to_numpy(np.float64) + 1.0)).astype(np.float32)
        cand_pre["als_rank"] = pd.to_numeric(cand_pre["als_rank"], errors="coerce").fillna(0).astype(np.int32)

        blocks: dict[int, dict[str, Any]] = {}
        for uid, g in cand_pre.groupby("user_idx", sort=False):
            items = g["item_idx"].to_numpy(np.int32)
            pre_inv = g["pre_inv"].to_numpy(np.float32)
            tv = ivec[items] @ uvec[int(uid)]
            sv = ivec[items] @ svec[int(uid)]
            blocks[int(uid)] = {
                "items": items,
                "pre_inv": pre_inv,
                "tower_inv": inv_from_scores(tv),
                "seq_inv": inv_from_scores(sv),
                "als_rank": g["als_rank"].to_numpy(np.int32),
            }

        valid_truth = dict(zip(valid_idx["user_idx"].astype(int), valid_idx["item_idx"].astype(int)))
        # Keep test truth aligned with stage09 output indices to avoid business-id remap loss.
        test_truth = dict(zip(truth_map_pdf["user_idx"].astype(int), truth_map_pdf["true_item_idx"].astype(int)))
        valid_eval_users = int(len(valid_truth))
        test_eval_users = int(len(test_truth))

        best_t = (0.0, 0.0, -1.0, -1.0)
        for wt in WT_GRID:
            rec, nd = eval_blocks(blocks, valid_truth, wt, 0.0)
            if nd > best_t[3] or (abs(nd - best_t[3]) <= 1e-12 and rec > best_t[2]):
                best_t = (wt, 0.0, rec, nd)

        best_ts = (0.0, 0.0, -1.0, -1.0)
        for wt in WT_GRID:
            for ws in WS_GRID:
                rec, nd = eval_blocks(blocks, valid_truth, wt, ws)
                if nd > best_ts[3] or (abs(nd - best_ts[3]) <= 1e-12 and rec > best_ts[2]):
                    best_ts = (wt, ws, rec, nd)

        rows = [
            ("valid", "PreScore", 0.0, 0.0, *eval_blocks(blocks, valid_truth, 0.0, 0.0)),
            ("valid", "PlusTowerFeature", best_t[0], 0.0, best_t[2], best_t[3]),
            ("valid", "PlusTowerSeqFeature", best_ts[0], best_ts[1], best_ts[2], best_ts[3]),
            ("test", "PreScore", 0.0, 0.0, *eval_blocks(blocks, test_truth, 0.0, 0.0)),
            ("test", "PlusTowerFeature", best_t[0], 0.0, *eval_blocks(blocks, test_truth, best_t[0], 0.0)),
            ("test", "PlusTowerSeqFeature", best_ts[0], best_ts[1], *eval_blocks(blocks, test_truth, best_ts[0], best_ts[1])),
        ]
        for split, model, wt, ws, rec, nd in rows:
            append_result(
                METRICS_PATH,
                {
                    "run_id": run_id,
                    "source_run_09": stage09_run.name,
                    "bucket": bucket,
                    "split": split,
                    "model": model,
                    "recall_at_k": rec,
                    "ndcg_at_k": nd,
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_train_pairs": int(train_idx.shape[0]),
                    "density": density,
                    "seq_ready_rate": seq_ready_rate,
                    "w_t": wt,
                    "w_s": ws,
                    "eval_users": valid_eval_users if split == "valid" else test_eval_users,
                },
            )

        # Diagnostics on test users
        t_users = list(test_truth.keys())
        unique_hit_tower_not_als = 0
        overlap_j_sum = 0.0
        overlap_cnt = 0
        pre_pool_hit = 0
        tower_pool_hit = 0
        union_pool_hit = 0
        for uid in t_users:
            b = blocks.get(int(uid))
            if b is None:
                continue
            true_item = int(test_truth[uid])
            score_t = b["tower_inv"]
            idx_t = np.argpartition(score_t, -TOP_K)[-TOP_K:]
            top_t = set([int(x) for x in b["items"][idx_t].tolist()])
            als_items = set([int(x) for x, r in zip(b["items"], b["als_rank"]) if int(r) > 0 and int(r) <= TOP_K])
            hit_t = true_item in top_t
            hit_als = true_item in als_items
            if hit_t and (not hit_als):
                unique_hit_tower_not_als += 1
            if len(top_t | als_items) > 0:
                overlap_j_sum += len(top_t & als_items) / len(top_t | als_items)
                overlap_cnt += 1

            in_pre_pool = true_item in set([int(x) for x in b["items"].tolist()])
            pre_pool_hit += int(in_pre_pool)

            scores_full = ivec @ uvec[int(uid)]
            seen = user_pos.get(int(uid), set())
            if seen:
                seen_idx = np.fromiter(seen, dtype=np.int32, count=len(seen))
                seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx < n_items)]
                if seen_idx.size > 0:
                    scores_full[seen_idx] = -1e12
            idx_pool = np.argpartition(scores_full, -POOL_K)[-POOL_K:]
            in_tower_pool = true_item in set([int(x) for x in idx_pool.tolist()])
            tower_pool_hit += int(in_tower_pool)
            union_pool_hit += int(in_pre_pool or in_tower_pool)

        diag = {
            "tower_unique_hit_rate_vs_als_top10": float(unique_hit_tower_not_als / max(1, len(t_users))),
            "tower_als_jaccard_top10": float(overlap_j_sum / max(1, overlap_cnt)),
            "pre_pool_hit_rate_at_150": float(pre_pool_hit / max(1, len(t_users))),
            "tower_pool_hit_rate_at_150": float(tower_pool_hit / max(1, len(t_users))),
            "union_pool_hit_rate_at_150": float(union_pool_hit / max(1, len(t_users))),
            "tower_pool_increment_over_pre": float((union_pool_hit - pre_pool_hit) / max(1, len(t_users))),
        }

        b_out = out_dir / f"bucket_{bucket}"
        b_out.mkdir(parents=True, exist_ok=True)
        summary = {
            "bucket": bucket,
            "n_users": n_users,
            "n_items": n_items,
            "n_train_pairs": int(train_idx.shape[0]),
            "user_map_users": int(user_map["user_idx"].nunique()),
            "valid_truth_users": int(valid_eval_users),
            "test_truth_users_mapped": int(test_idx["user_idx"].nunique()),
            "test_truth_users": int(test_eval_users),
            "valid_truth_coverage_vs_users": float(valid_eval_users / max(1, n_users)),
            "test_truth_coverage_vs_users_mapped": float(test_idx["user_idx"].nunique() / max(1, n_users)),
            "test_truth_coverage_vs_users": float(test_eval_users / max(1, n_users)),
            "density": density,
            "seq_ready_rate": seq_ready_rate,
            "tower_updates": int(updates),
            "best_tower_weight": float(best_t[0]),
            "best_tower_seq_weight": {"w_t": float(best_ts[0]), "w_s": float(best_ts[1])},
            "metrics_rows": [
                {"split": s, "model": m, "w_t": wt, "w_s": ws, "recall": rec, "ndcg": nd}
                for s, m, wt, ws, rec, nd in rows
            ],
            "diagnostics": diag,
        }
        (b_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        pd.DataFrame(summary["metrics_rows"]).to_csv(b_out / "metrics_rows.csv", index=False, encoding="utf-8-sig")
        all_summary.append(summary)
        print(
            f"[METRIC] bucket={bucket} test pre_ndcg={rows[3][5]:.6f} tower_ndcg={rows[4][5]:.6f} "
            f"tower_seq_ndcg={rows[5][5]:.6f} pool_inc={diag['tower_pool_increment_over_pre']:.6f}"
        )

    run_meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "run_profile": run_profile,
        "source_run_09": stage09_run.name,
        "output_dir": str(out_dir),
        "metrics_path": str(METRICS_PATH),
        "top_k": TOP_K,
        "pool_k": POOL_K,
        "weight_grid": {"tower": WT_GRID, "seq": WS_GRID},
        "summaries": all_summary,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")
    spark.stop()
    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
