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
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

try:
    from xgboost import XGBRanker
except Exception:
    XGBRanker = None

RUN_TAG = "stage10_8_xgb_tower_seq_eval"
PARQUET_BASE = Path(r"D:/5006 BDA project/data/parquet")
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = Path(r"D:/5006 BDA project/data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"
OUTPUT_ROOT = Path(r"D:/5006 BDA project/data/output/10_8_xgb_tower_seq_eval")
METRICS_PATH = Path(r"D:/5006 BDA project/data/metrics/stage10_8_xgb_tower_seq_eval_results.csv")

RANDOM_SEED = int(os.getenv("EVAL_RANDOM_SEED", "42") or 42)
TOP_K = int(os.getenv("EVAL_TOP_K", "10") or 10)
MIN_USER_REVIEWS_OFFSET = int(os.getenv("EVAL_MIN_USER_REVIEWS_OFFSET", "2") or 2)
RUN_PROFILE_OVERRIDE = os.getenv("RUN_PROFILE_OVERRIDE", "").strip().lower()
SAMPLE_FRAC = float(os.getenv("EVAL_SAMPLE_FRAC", "0.15") or 0.15)

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]") or "local[2]"
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g") or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g") or "6g"
SPARK_LOCAL_DIR = os.getenv("SPARK_LOCAL_DIR", "D:/5006 BDA project/data/spark-tmp") or "D:/5006 BDA project/data/spark-tmp"

TOWER_DIM = int(os.getenv("TOWER_DIM", "64") or 64)
TOWER_EPOCHS = int(os.getenv("TOWER_EPOCHS", "8") or 8)
TOWER_LR = float(os.getenv("TOWER_LR", "0.04") or 0.04)
TOWER_REG = float(os.getenv("TOWER_REG", "0.0001") or 0.0001)
TOWER_NEG_TRIES = int(os.getenv("TOWER_NEG_TRIES", "6") or 6)

SEQ_RECENT_LEN = int(os.getenv("SEQ_RECENT_LEN", "10") or 10)
SEQ_RECENT_MIN_LEN = int(os.getenv("SEQ_RECENT_MIN_LEN", "2") or 2)
SEQ_DECAY = float(os.getenv("SEQ_DECAY", "0.85") or 0.85)

WT_GRID = [float(x.strip()) for x in os.getenv("TOWER_WEIGHT_GRID", "0.0,0.05,0.1,0.15,0.2,0.3,0.4").split(",") if x.strip()]
WS_GRID = [float(x.strip()) for x in os.getenv("SEQ_WEIGHT_GRID", "0.0,0.05,0.1,0.15,0.2").split(",") if x.strip()]
if not WT_GRID: WT_GRID = [0.0, 0.1, 0.2]
if not WS_GRID: WS_GRID = [0.0, 0.05, 0.1]
XGB_ALPHA_GRID = [float(x.strip()) for x in os.getenv("XGB_ALPHA_GRID", "0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5").split(",") if x.strip()]
if not XGB_ALPHA_GRID:
    XGB_ALPHA_GRID = [0.0, 0.1, 0.2, 0.3]

TRAIN_HARD_NEG_TOPK = int(os.getenv("XGB_TRAIN_HARD_NEG_TOPK", "40") or 40)
TRAIN_HARD_NEG_PER_USER = int(os.getenv("XGB_TRAIN_HARD_NEG_PER_USER", "24") or 24)
TRAIN_RANDOM_NEG_PER_USER = int(os.getenv("XGB_TRAIN_RANDOM_NEG_PER_USER", "12") or 12)

XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", "300") or 300)
XGB_MAX_DEPTH = int(os.getenv("XGB_MAX_DEPTH", "6") or 6)
XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", "0.06") or 0.06)
XGB_SUBSAMPLE = float(os.getenv("XGB_SUBSAMPLE", "0.8") or 0.8)
XGB_COLSAMPLE = float(os.getenv("XGB_COLSAMPLE", "0.8") or 0.8)
XGB_REG_LAMBDA = float(os.getenv("XGB_REG_LAMBDA", "2.0") or 2.0)

RESULT_FIELDS = [
    "run_id", "source_run_09", "bucket", "split", "model", "recall_at_k", "ndcg_at_k",
    "n_users", "n_items", "n_train_pairs", "density", "seq_ready_rate", "w_t", "w_s",
    "eval_users", "train_rows", "train_pos",
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
    for t in raw.split(","):
        t = t.strip()
        if not t:
            continue
        try:
            keep.add(int(t))
        except ValueError:
            pass
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
        SparkSession.builder.appName("stage10-8-xgb-tower-seq-eval")
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
    biz = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_business").as_posix())
        .select("business_id", "state", "categories", "is_open", "stars", "review_count")
        .withColumn("business_id", F.col("business_id").cast("string"))
    )
    cat = F.lower(F.coalesce(F.col("categories"), F.lit("")))
    cond = (cat.contains("restaurants") | cat.contains("food"))
    biz = biz.filter((F.col("state") == F.lit("LA")) & cond)
    biz = biz.filter((F.col("is_open") == 1) & (F.col("stars") >= 3.0) & (F.col("review_count") >= 20))
    rvw = (
        spark.read.parquet((PARQUET_BASE / "yelp_academic_dataset_review").as_posix())
        .select("business_id", "date")
        .withColumn("business_id", F.col("business_id").cast("string"))
        .withColumn("ts", F.to_timestamp("date"))
        .filter(F.col("ts").isNotNull())
    )
    cutoff = "2020-01-01"
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

def inv_from_rank(rank: np.ndarray) -> np.ndarray:
    rr = np.nan_to_num(rank.astype(np.float64), nan=9999.0, posinf=9999.0, neginf=9999.0)
    rr = np.clip(rr, 1.0, 1e9)
    return (1.0 / np.log2(rr + 1.0)).astype(np.float32)

def train_bpr(n_users: int, n_items: int, pairs: np.ndarray, user_pos: dict[int, set[int]]) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    user_vec = (rng.normal(0, 0.1, size=(n_users, TOWER_DIM)) / math.sqrt(TOWER_DIM)).astype(np.float32)
    item_vec = (rng.normal(0, 0.1, size=(n_items, TOWER_DIM)) / math.sqrt(TOWER_DIM)).astype(np.float32)
    if pairs.shape[0] == 0:
        return user_vec, item_vec, 0
    steps = int(np.clip(pairs.shape[0] * 6, 5000, 60000))
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

def eval_blocks(blocks: dict[int, dict[str, Any]], truth: dict[int, int], score_fn: Any) -> tuple[float, float]:
    hits, ndcg = 0, 0.0
    n = int(len(truth))
    if n <= 0:
        return 0.0, 0.0
    for uid, t_item in truth.items():
        b = blocks.get(int(uid))
        if b is None:
            continue
        score = np.asarray(score_fn(b), dtype=np.float64)
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

def build_rank_train(blocks: dict[int, dict[str, Any]], valid_truth: dict[int, int]) -> tuple[np.ndarray, np.ndarray, list[int], int, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    groups: list[int] = []
    pos_total = 0
    for uid, t_item in valid_truth.items():
        b = blocks.get(int(uid))
        if b is None:
            continue
        items = b["items"]
        y = (items == int(t_item)).astype(np.float32)
        pos_idx = np.where(y > 0.5)[0]
        if pos_idx.size == 0:
            continue
        pidx = int(pos_idx[0])
        all_idx = np.where(np.arange(items.size) != pidx)[0]
        hard_idx = all_idx[b["pre_rank"][all_idx] <= float(TRAIN_HARD_NEG_TOPK)]
        if hard_idx.size > TRAIN_HARD_NEG_PER_USER:
            hard_idx = rng.choice(hard_idx, size=TRAIN_HARD_NEG_PER_USER, replace=False)
        rest_idx = np.setdiff1d(all_idx, hard_idx, assume_unique=False)
        if rest_idx.size > TRAIN_RANDOM_NEG_PER_USER:
            rest_idx = rng.choice(rest_idx, size=TRAIN_RANDOM_NEG_PER_USER, replace=False)
        take = np.concatenate([np.array([pidx], dtype=np.int32), hard_idx.astype(np.int32), rest_idx.astype(np.int32)])
        x_list.append(b["feat"][take])
        y_list.append(y[take])
        groups.append(int(take.size))
        pos_total += 1
    if not x_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), [], 0, 0
    x = np.vstack(x_list).astype(np.float32, copy=False)
    y = np.concatenate(y_list).astype(np.float32, copy=False)
    return x, y, groups, int(x.shape[0]), int(pos_total)

def main() -> None:
    if XGBRanker is None:
        raise RuntimeError("xgboost not installed")
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
    print(f"[CONFIG] stage09={stage09_run.name} run_profile={run_profile} buckets={[p.name for p in parse_bucket_dirs(stage09_run)]}")
    print(f"[COUNT] interactions={rvw.count()} users={rvw.select('user_id').distinct().count()}")

    all_summary: list[dict[str, Any]] = []
    for bdir in parse_bucket_dirs(stage09_run):
        bucket = int(bdir.name.split("_")[-1]); min_user = int(bucket + MIN_USER_REVIEWS_OFFSET)
        print(f"\n[BUCKET] {bucket}")
        cand_raw_df = spark.read.parquet((bdir / "candidates_pretrim150.parquet").as_posix())
        cols = set(cand_raw_df.columns)
        def dcol(name: str, default: float = 0.0):
            return (F.coalesce(F.col(name).cast("double"), F.lit(float(default))).alias(name) if name in cols else F.lit(float(default)).cast("double").alias(name))
        def icol(name: str, default: int = 0):
            return (F.coalesce(F.col(name).cast("int"), F.lit(int(default))).alias(name) if name in cols else F.lit(int(default)).cast("int").alias(name))
        source_count = (F.when(F.col("source_set").isNull(), F.lit(0)).otherwise(F.size(F.col("source_set"))).cast("int").alias("source_count") if "source_set" in cols else F.lit(0).cast("int").alias("source_count"))
        cand_df = cand_raw_df.select(
            F.col("business_id").cast("string").alias("business_id"),
            icol("user_idx", -1), icol("item_idx", -1), icol("pre_rank", 9999),
            dcol("pre_score"), dcol("signal_score"), dcol("quality_score"), dcol("semantic_score"),
            dcol("semantic_confidence"), dcol("semantic_support"), dcol("semantic_tag_richness"),
            dcol("item_train_pop_count"), dcol("user_train_count"),
            icol("als_rank", 0), icol("cluster_rank", 0), icol("profile_rank", 0), icol("popular_rank", 0),
            source_count,
        )
        cand_pdf = cand_df.toPandas()
        truth_pdf = spark.read.parquet((bdir / "truth.parquet").as_posix()).select("user_idx", "user_id", "true_item_idx").dropDuplicates(["user_idx"]).toPandas()

        user_map = truth_pdf[["user_idx", "user_id"]].drop_duplicates("user_idx").copy()
        user_map["user_idx"] = pd.to_numeric(user_map["user_idx"], errors="coerce").fillna(-1).astype(int)
        item_map = cand_pdf[["item_idx", "business_id"]].drop_duplicates("item_idx").copy()
        item_map["item_idx"] = pd.to_numeric(item_map["item_idx"], errors="coerce").fillna(-1).astype(int)

        full_df, train, valid, test = leave_two_out(rvw, min_user_reviews=min_user)
        train_pdf = train.select("user_id", "business_id", "ts").toPandas()
        valid_pdf = valid.select("user_id", "business_id").toPandas()
        test_pdf = test.select("user_id", "business_id").toPandas()

        train_idx = train_pdf.merge(user_map, on="user_id", how="inner").merge(item_map, on="business_id", how="inner")
        valid_idx = valid_pdf.merge(user_map, on="user_id", how="inner").merge(item_map, on="business_id", how="inner")
        test_idx = test_pdf.merge(user_map, on="user_id", how="inner").merge(item_map, on="business_id", how="inner")

        for df in (cand_pdf, train_idx, valid_idx, test_idx, truth_pdf):
            df["user_idx"] = pd.to_numeric(df["user_idx"], errors="coerce").fillna(-1).astype(int)
        for df in (cand_pdf, train_idx, valid_idx, test_idx):
            df["item_idx"] = pd.to_numeric(df["item_idx"], errors="coerce").fillna(-1).astype(int)
        truth_pdf["true_item_idx"] = pd.to_numeric(truth_pdf["true_item_idx"], errors="coerce").fillna(-1).astype(int)
        cand_pdf = cand_pdf[(cand_pdf["user_idx"] >= 0) & (cand_pdf["item_idx"] >= 0)].copy()
        train_idx = train_idx[(train_idx["user_idx"] >= 0) & (train_idx["item_idx"] >= 0)].copy()
        valid_idx = valid_idx[(valid_idx["user_idx"] >= 0) & (valid_idx["item_idx"] >= 0)].copy()
        truth_pdf = truth_pdf[(truth_pdf["user_idx"] >= 0) & (truth_pdf["true_item_idx"] >= 0)].copy()

        if train_idx.empty or valid_idx.empty or cand_pdf.empty:
            print(f"[WARN] bucket={bucket} empty mapped data")
            continue
        n_users = int(max(cand_pdf["user_idx"].max(), train_idx["user_idx"].max()) + 1)
        n_items = int(max(cand_pdf["item_idx"].max(), train_idx["item_idx"].max()) + 1)
        density = float(train_idx.shape[0]) / float(max(1, n_users * n_items))
        user_pos: dict[int, set[int]] = {}
        for r in train_idx[["user_idx", "item_idx"]].itertuples(index=False):
            user_pos.setdefault(int(r.user_idx), set()).add(int(r.item_idx))

        pairs = train_idx[["user_idx", "item_idx"]].to_numpy(dtype=np.int32, copy=True)
        uvec, ivec, updates = train_bpr(n_users, n_items, pairs, user_pos)
        svec, sready = seq_vectors(train_idx[["user_idx", "item_idx", "ts"]].copy(), ivec, n_users)
        seq_ready_rate = float(sready.sum()) / float(max(1, n_users))

        blocks: dict[int, dict[str, Any]] = {}
        cand_pdf = cand_pdf.sort_values(["user_idx", "pre_rank", "item_idx"], ascending=[True, True, True]).copy()
        for uid, g in cand_pdf.groupby("user_idx", sort=False):
            uid_i = int(uid)
            items = g["item_idx"].to_numpy(np.int32)
            pre_rank = np.clip(g["pre_rank"].to_numpy(np.float64), 1.0, 1e9)
            pre_inv = inv_from_rank(pre_rank)
            tv = ivec[items] @ uvec[uid_i]
            sv = ivec[items] @ svec[uid_i]
            tower_inv = inv_from_scores(tv)
            seq_inv = inv_from_scores(sv)
            has_als = (g["als_rank"].to_numpy(np.int32) > 0).astype(np.float32)
            has_cluster = (g["cluster_rank"].to_numpy(np.int32) > 0).astype(np.float32)
            has_profile = (g["profile_rank"].to_numpy(np.int32) > 0).astype(np.float32)
            has_popular = (g["popular_rank"].to_numpy(np.int32) > 0).astype(np.float32)
            feat = np.column_stack([
                g["pre_score"].to_numpy(np.float32), g["signal_score"].to_numpy(np.float32), g["quality_score"].to_numpy(np.float32),
                g["semantic_score"].to_numpy(np.float32), g["semantic_confidence"].to_numpy(np.float32),
                np.log1p(np.maximum(g["semantic_support"].to_numpy(np.float32), 0.0)), g["semantic_tag_richness"].to_numpy(np.float32),
                np.log1p(np.maximum(g["item_train_pop_count"].to_numpy(np.float32), 0.0)), np.log1p(np.maximum(g["user_train_count"].to_numpy(np.float32), 0.0)),
                pre_inv.astype(np.float32), inv_from_rank(g["als_rank"].to_numpy(np.float64)), inv_from_rank(g["cluster_rank"].to_numpy(np.float64)),
                inv_from_rank(g["profile_rank"].to_numpy(np.float64)), inv_from_rank(g["popular_rank"].to_numpy(np.float64)),
                has_als, has_cluster, has_profile, has_popular,
                g["source_count"].to_numpy(np.float32), tower_inv.astype(np.float32), seq_inv.astype(np.float32),
                (g["pre_score"].to_numpy(np.float32) * has_profile), (g["semantic_score"].to_numpy(np.float32) * has_profile),
                (tower_inv.astype(np.float32) * has_profile), (seq_inv.astype(np.float32) * has_profile),
            ]).astype(np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            blocks[uid_i] = {"items": items, "pre_rank": pre_rank.astype(np.float32), "pre_inv": pre_inv.astype(np.float32), "tower_inv": tower_inv.astype(np.float32), "seq_inv": seq_inv.astype(np.float32), "feat": feat}

        valid_truth = dict(zip(valid_idx["user_idx"].astype(int), valid_idx["item_idx"].astype(int)))
        test_truth = dict(zip(truth_pdf["user_idx"].astype(int), truth_pdf["true_item_idx"].astype(int)))
        valid_eval_users = int(len(valid_truth)); test_eval_users = int(len(test_truth))

        best_ts = (0.0, 0.0, -1.0, -1.0)
        for wt in WT_GRID:
            for ws in WS_GRID:
                rec, nd = eval_blocks(blocks, valid_truth, lambda b, wt=wt, ws=ws: b["pre_inv"] + float(wt) * b["tower_inv"] + float(ws) * b["seq_inv"])
                if nd > best_ts[3] or (abs(nd - best_ts[3]) <= 1e-12 and rec > best_ts[2]):
                    best_ts = (wt, ws, rec, nd)

        x_train, y_train, g_train, train_rows, train_pos = build_rank_train(blocks, valid_truth)
        if train_rows <= 0 or train_pos <= 0:
            print(f"[WARN] bucket={bucket} no train rows")
            continue

        ranker = XGBRanker(
            objective="rank:pairwise", n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE,
            reg_lambda=XGB_REG_LAMBDA, random_state=RANDOM_SEED + bucket, tree_method="hist", n_jobs=max(1, os.cpu_count() // 2),
        )
        ranker.fit(x_train, y_train, group=g_train, verbose=False)

        for uid in (set(valid_truth.keys()) | set(test_truth.keys())):
            b = blocks.get(int(uid))
            if b is None:
                continue
            b["xgb_score"] = ranker.predict(b["feat"]).astype(np.float32)
            b["xgb_inv"] = inv_from_scores(b["xgb_score"])

        best_alpha = (0.0, -1.0, -1.0)
        for a in XGB_ALPHA_GRID:
            rec, nd = eval_blocks(
                blocks,
                valid_truth,
                lambda b, a=a: b["pre_inv"] + float(best_ts[0]) * b["tower_inv"] + float(best_ts[1]) * b["seq_inv"] + float(a) * b["xgb_inv"],
            )
            if nd > best_alpha[2] or (abs(nd - best_alpha[2]) <= 1e-12 and rec > best_alpha[1]):
                best_alpha = (float(a), rec, nd)

        rows = [
            ("valid", "PreScore", 0.0, 0.0, *eval_blocks(blocks, valid_truth, lambda b: b["pre_inv"])),
            ("valid", "TowerSeqBlend", best_ts[0], best_ts[1], *eval_blocks(blocks, valid_truth, lambda b: b["pre_inv"] + float(best_ts[0]) * b["tower_inv"] + float(best_ts[1]) * b["seq_inv"])),
            ("valid", "XGBRawRanker", 0.0, 0.0, *eval_blocks(blocks, valid_truth, lambda b: b["xgb_score"])),
            ("valid", "XGBBlendRanker", best_alpha[0], 0.0, *eval_blocks(blocks, valid_truth, lambda b: b["pre_inv"] + float(best_ts[0]) * b["tower_inv"] + float(best_ts[1]) * b["seq_inv"] + float(best_alpha[0]) * b["xgb_inv"])),
            ("test", "PreScore", 0.0, 0.0, *eval_blocks(blocks, test_truth, lambda b: b["pre_inv"])),
            ("test", "TowerSeqBlend", best_ts[0], best_ts[1], *eval_blocks(blocks, test_truth, lambda b: b["pre_inv"] + float(best_ts[0]) * b["tower_inv"] + float(best_ts[1]) * b["seq_inv"])),
            ("test", "XGBRawRanker", 0.0, 0.0, *eval_blocks(blocks, test_truth, lambda b: b["xgb_score"])),
            ("test", "XGBBlendRanker", best_alpha[0], 0.0, *eval_blocks(blocks, test_truth, lambda b: b["pre_inv"] + float(best_ts[0]) * b["tower_inv"] + float(best_ts[1]) * b["seq_inv"] + float(best_alpha[0]) * b["xgb_inv"])),
        ]

        for split, model, wt, ws, rec, nd in rows:
            append_result(METRICS_PATH, {
                "run_id": run_id, "source_run_09": stage09_run.name, "bucket": bucket, "split": split, "model": model,
                "recall_at_k": rec, "ndcg_at_k": nd, "n_users": n_users, "n_items": n_items, "n_train_pairs": int(train_idx.shape[0]),
                "density": density, "seq_ready_rate": seq_ready_rate, "w_t": wt, "w_s": ws,
                "eval_users": valid_eval_users if split == "valid" else test_eval_users,
                "train_rows": train_rows, "train_pos": train_pos,
            })

        b_out = out_dir / f"bucket_{bucket}"
        b_out.mkdir(parents=True, exist_ok=True)
        summary = {
            "bucket": bucket, "n_users": n_users, "n_items": n_items, "n_train_pairs": int(train_idx.shape[0]),
            "valid_truth_users": int(valid_eval_users), "test_truth_users_mapped": int(test_idx["user_idx"].nunique()),
            "test_truth_users": int(test_eval_users),
            "valid_truth_coverage_vs_users": float(valid_eval_users / max(1, n_users)),
            "test_truth_coverage_vs_users_mapped": float(test_idx["user_idx"].nunique() / max(1, n_users)),
            "test_truth_coverage_vs_users": float(test_eval_users / max(1, n_users)),
            "density": density, "seq_ready_rate": seq_ready_rate, "tower_updates": int(updates),
            "best_tower_seq_weight": {"w_t": float(best_ts[0]), "w_s": float(best_ts[1])},
            "best_xgb_alpha": float(best_alpha[0]),
            "xgb_train_rows": int(train_rows), "xgb_train_pos": int(train_pos),
            "metrics_rows": [{"split": s, "model": m, "w_t": wt, "w_s": ws, "recall": rec, "ndcg": nd} for s, m, wt, ws, rec, nd in rows],
        }
        (b_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
        pd.DataFrame(summary["metrics_rows"]).to_csv(b_out / "metrics_rows.csv", index=False, encoding="utf-8-sig")
        all_summary.append(summary)
        test_pre = float([r[-1] for r in rows if r[0] == "test" and r[1] == "PreScore"][0])
        test_blend = float([r[-1] for r in rows if r[0] == "test" and r[1] == "TowerSeqBlend"][0])
        test_xgb_raw = float([r[-1] for r in rows if r[0] == "test" and r[1] == "XGBRawRanker"][0])
        test_xgb_blend = float([r[-1] for r in rows if r[0] == "test" and r[1] == "XGBBlendRanker"][0])
        print(f"[METRIC] bucket={bucket} test pre_ndcg={test_pre:.6f} blend_ndcg={test_blend:.6f} xgb_raw_ndcg={test_xgb_raw:.6f} xgb_blend_ndcg={test_xgb_blend:.6f}")

    run_meta = {
        "run_id": run_id, "run_tag": RUN_TAG, "run_profile": run_profile, "source_run_09": stage09_run.name,
        "output_dir": str(out_dir), "metrics_path": str(METRICS_PATH), "top_k": TOP_K,
        "weight_grid": {"tower": WT_GRID, "seq": WS_GRID, "xgb_alpha": XGB_ALPHA_GRID},
        "xgb": {"n_estimators": XGB_N_ESTIMATORS, "max_depth": XGB_MAX_DEPTH, "learning_rate": XGB_LEARNING_RATE, "subsample": XGB_SUBSAMPLE, "colsample_bytree": XGB_COLSAMPLE, "reg_lambda": XGB_REG_LAMBDA},
        "summaries": all_summary,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")
    spark.stop()
    print(f"\n[DONE] {out_dir}")

if __name__ == "__main__":
    main()
