from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, project_path
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

# Keep sentence-transformers on torch path in mixed TensorFlow/Keras env.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

RUN_TAG = "stage10_5_transformer_rerank_eval"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"

PROFILE_TABLE_PATH = os.getenv("USER_PROFILE_TABLE", "").strip()
MODEL_NAME_OR_PATH = os.getenv("TRANSFORMER_MODEL", "").strip()
BGE_LOCAL_MODEL_PATH = Path(os.getenv("BGE_LOCAL_MODEL_PATH", r"D:/hf_cache/hub/models--BAAI--bge-m3").strip())
MINILM_MODEL_NAME = os.getenv("FALLBACK_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
ITEM_SEMANTIC_FEATURE_PATH = os.getenv("TF_ITEM_SEMANTIC_FEATURES", "").strip()

OUTPUT_ROOT = env_or_project_path("OUTPUT_10_5_ROOT_DIR", "data/output/10_5_transformer_rerank_eval")
METRICS_PATH = env_or_project_path("STAGE10_TRANSFORMER_RESULTS_METRICS_PATH", "data/metrics/recsys_stage10_transformer_results.csv")

TOP_K = int(os.getenv("RANK_EVAL_TOP_K", "10").strip() or 10)
RERANK_TOPN = int(os.getenv("TF_RERANK_TOPN", "150").strip() or 150)
BLEND_ALPHA = float(os.getenv("TF_BLEND_ALPHA", "0.25").strip() or 0.25)
PRE_PRIOR_POWER = float(os.getenv("TF_PRE_PRIOR_POWER", "1.0").strip() or 1.0)
SIM_POWER = float(os.getenv("TF_SIM_POWER", "1.0").strip() or 1.0)
USE_PROFILE_TAGS = os.getenv("TF_USE_PROFILE_TAGS", "true").strip().lower() == "true"
USE_ITEM_TAGS = os.getenv("TF_USE_ITEM_TAGS", "true").strip().lower() == "true"
USE_DYNAMIC_ALPHA = os.getenv("TF_DYNAMIC_ALPHA", "true").strip().lower() == "true"
FALLBACK_OUTSIDE_TOPN = os.getenv("TF_FALLBACK_OUTSIDE_TOPN", "true").strip().lower() == "true"
BASE_ALPHA_MIN = float(os.getenv("TF_ALPHA_MIN", "0.10").strip() or 0.10)
BASE_ALPHA_MAX = float(os.getenv("TF_ALPHA_MAX", "0.70").strip() or 0.70)
USE_TAG_ALIGN = os.getenv("TF_USE_TAG_ALIGN", "true").strip().lower() == "true"
TAG_ALIGN_WEIGHT = float(os.getenv("TF_TAG_ALIGN_WEIGHT", "0.30").strip() or 0.30)
TAG_ALIGN_NEG_PENALTY = float(os.getenv("TF_TAG_ALIGN_NEG_PENALTY", "1.0").strip() or 1.0)
TAG_ALIGN_POS_GAIN = float(os.getenv("TF_TAG_ALIGN_POS_GAIN", "2.0").strip() or 2.0)
TAG_ALIGN_MIN_PROFILE_CONF = float(os.getenv("TF_TAG_ALIGN_MIN_PROFILE_CONF", "0.10").strip() or 0.10)
SIM_CHUNK_ROWS = int(os.getenv("TF_SIM_CHUNK_ROWS", "12000").strip() or 12000)
EMBED_BATCH_SIZE = int(os.getenv("TF_EMBED_BATCH_SIZE", "8").strip() or 8)
BUCKETS_OVERRIDE = os.getenv("RANK_BUCKETS_OVERRIDE", "2,5,10").strip()
TAIL_QUANTILE = 0.8
ENABLE_LEARNED_HEAD = os.getenv("TF_ENABLE_LEARNED_HEAD", "true").strip().lower() == "true"
LEARN_TRAIN_TOPN = int(os.getenv("TF_LEARN_TRAIN_TOPN", "300").strip() or 300)
LEARN_NEG_PRE_PER_USER = int(os.getenv("TF_LEARN_NEG_PRE_PER_USER", "16").strip() or 16)
LEARN_NEG_SIM_PER_USER = int(os.getenv("TF_LEARN_NEG_SIM_PER_USER", "12").strip() or 12)
LEARN_NEG_RANDOM_PER_USER = int(os.getenv("TF_LEARN_NEG_RANDOM_PER_USER", "8").strip() or 8)
LEARN_MAX_ROWS = int(os.getenv("TF_LEARN_MAX_ROWS", "500000").strip() or 500000)
LEARN_MODEL_C = float(os.getenv("TF_LEARN_MODEL_C", "0.5").strip() or 0.5)
LEARN_ALPHA = float(os.getenv("TF_LEARN_ALPHA", "0.35").strip() or 0.35)

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_LOCAL_DIR = env_or_project_path("SPARK_LOCAL_DIR", "data/spark-tmp")
SPARK_DRIVER_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_DRIVER_EXTRA_JAVA_OPTIONS",
    "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=256m -XX:MaxMetaspaceSize=512m -Xss512k",
).strip()
SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS = os.getenv(
    "SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS",
    "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=256m -XX:MaxMetaspaceSize=512m -Xss512k",
).strip()


def build_spark() -> SparkSession:
    temp_root = os.getenv("PY_TEMP_DIR", "").strip()
    if not temp_root:
        temp_root = str(Path(__file__).resolve().parents[1])
    os.environ["TEMP"] = temp_root
    os.environ["TMP"] = temp_root
    os.environ["TMPDIR"] = temp_root
    tempfile.tempdir = temp_root
    local_dir = SPARK_LOCAL_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName("stage10-5-transformer-rerank")
        .master("local[2]")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.extraJavaOptions", SPARK_DRIVER_EXTRA_JAVA_OPTIONS)
        .config("spark.executor.extraJavaOptions", SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS)
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def pick_latest_run(root: Path, suffix: str) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no run in {root} with suffix={suffix}")
    return runs[0]


def resolve_stage09_run() -> Path:
    if INPUT_09_RUN_DIR:
        p = normalize_legacy_project_path(INPUT_09_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    return pick_latest_run(INPUT_09_ROOT, INPUT_09_SUFFIX)


def parse_bucket_override(raw: str) -> set[int]:
    out: set[int] = set()
    text = (raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except ValueError:
            continue
    return out


def resolve_profile_table(source_09: Path) -> Path:
    if PROFILE_TABLE_PATH:
        p = normalize_legacy_project_path(PROFILE_TABLE_PATH)
        if not p.exists():
            raise FileNotFoundError(f"USER_PROFILE_TABLE not found: {p}")
        return p
    meta_path = source_09 / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found under {source_09}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    p = normalize_legacy_project_path(str(meta.get("user_profile_table", "")).strip())
    if not p.exists():
        raise FileNotFoundError(f"user_profile_table missing from run_meta or not found: {p}")
    return p


def resolve_item_semantic_table(source_09: Path) -> Path | None:
    if ITEM_SEMANTIC_FEATURE_PATH:
        p = normalize_legacy_project_path(ITEM_SEMANTIC_FEATURE_PATH)
        if not p.exists() or p.is_dir():
            raise FileNotFoundError(f"TF_ITEM_SEMANTIC_FEATURES not found: {p}")
        return p
    meta_path = source_09 / "run_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    p = normalize_legacy_project_path(str(meta.get("item_semantic_features", "")).strip())
    if str(p).strip() in {"", "."}:
        return None
    if p.exists() and p.is_file():
        return p
    return None


def pick_candidate_file(bucket_dir: Path) -> Path:
    candidates = [
        str(os.getenv("TF_CANDIDATE_FILE", "").strip() or ""),
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates_pretrim.parquet",
        "candidates.parquet",
    ]
    for name in candidates:
        if not name:
            continue
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def resolve_model_name() -> str:
    if MODEL_NAME_OR_PATH:
        return MODEL_NAME_OR_PATH
    if BGE_LOCAL_MODEL_PATH.exists():
        snaps = BGE_LOCAL_MODEL_PATH / "snapshots"
        if snaps.exists():
            cands = [p for p in snaps.iterdir() if p.is_dir()]
            if cands:
                cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                # Some cache snapshots may be partial (e.g., only safetensors).
                for c in cands:
                    if (c / "config.json").exists() and ((c / "modules.json").exists() or (c / "sentence_bert_config.json").exists()):
                        return c.as_posix()
                return cands[0].as_posix()
        return BGE_LOCAL_MODEL_PATH.as_posix()
    return MINILM_MODEL_NAME


def normalize_space(s: str) -> str:
    return " ".join(str(s or "").split())


def split_tags(raw: str) -> set[str]:
    text = str(raw or "").strip().lower()
    if not text:
        return set()
    text = text.replace("|", ",").replace(";", ",")
    toks: list[str] = []
    for part in text.split(","):
        p = normalize_space(part).replace(" ", "_")
        p = re.sub(r"[^a-z0-9_]+", "", p)
        if len(p) >= 2:
            toks.append(p)
    return set(toks)


def compute_tag_align(
    user_idx_arr: np.ndarray,
    item_idx_arr: np.ndarray,
    user_tags: dict[int, tuple[set[str], set[str]]],
    item_tags: dict[int, tuple[set[str], set[str]]],
) -> np.ndarray:
    out = np.zeros(len(user_idx_arr), dtype=np.float32)
    empty: set[str] = set()
    neg_pen = float(max(0.0, TAG_ALIGN_NEG_PENALTY))
    pos_gain = float(max(0.0, TAG_ALIGN_POS_GAIN))
    for j in range(len(user_idx_arr)):
        u = int(user_idx_arr[j])
        i = int(item_idx_arr[j])
        u_pos, u_neg = user_tags.get(u, (empty, empty))
        i_pos, i_neg = item_tags.get(i, (empty, empty))
        if (not u_pos and not u_neg) or (not i_pos and not i_neg):
            continue
        pos_hit = float(len(u_pos & i_pos))
        neg_hit = float(len(u_neg & i_pos) + len(u_pos & i_neg))
        denom = float(max(1, len(u_pos)) + max(1, len(i_pos)))
        score = (pos_gain * pos_hit - neg_pen * neg_hit) / denom
        score = max(-1.0, min(1.0, score))
        out[j] = np.float32(score)
    return out


def build_item_text(row: pd.Series) -> str:
    def as_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return 0.0

    parts: list[str] = []
    name = normalize_space(row.get("name", ""))
    cat1 = normalize_space(row.get("primary_category", ""))
    catn = normalize_space(row.get("categories", ""))
    city = normalize_space(row.get("city", ""))
    top_pos = normalize_space(str(row.get("top_pos_tags", "")).replace("|", ", "))
    top_neg = normalize_space(str(row.get("top_neg_tags", "")).replace("|", ", "))
    sem_conf = as_float(row.get("semantic_confidence", 0.0))
    sem_sup = as_float(row.get("semantic_support", 0.0))
    sem_rich = as_float(row.get("semantic_tag_richness", 0.0))
    if name:
        parts.append(f"name: {name}")
    if cat1:
        parts.append(f"primary_category: {cat1}")
    if catn:
        parts.append(f"categories: {catn}")
    if city:
        parts.append(f"city: {city}")
    if USE_ITEM_TAGS and top_pos:
        parts.append(f"positive_tags: {top_pos}")
    if USE_ITEM_TAGS and top_neg:
        parts.append(f"negative_tags: {top_neg}")
    if sem_conf > 0.0:
        parts.append(f"semantic_confidence: {sem_conf:.3f}")
    if sem_sup > 0.0:
        parts.append(f"semantic_support: {sem_sup:.1f}")
    if sem_rich > 0.0:
        parts.append(f"semantic_richness: {sem_rich:.1f}")
    if not parts:
        parts.append("restaurant")
    return "; ".join(parts)


def load_profiles(profile_table: Path) -> pd.DataFrame:
    use_cols = [
        "user_id",
        "profile_text",
        "profile_text_short",
        "profile_text_long",
        "profile_confidence",
        "profile_top_pos_tags",
        "profile_top_neg_tags",
        "profile_keywords",
    ]
    df = pd.read_csv(profile_table, usecols=lambda c: c in use_cols)
    if "user_id" not in df.columns:
        raise RuntimeError(f"user_profile table missing user_id: {profile_table}")
    if "profile_text" not in df.columns:
        df["profile_text"] = ""
    if "profile_text_short" not in df.columns:
        df["profile_text_short"] = ""
    if "profile_text_long" not in df.columns:
        df["profile_text_long"] = ""
    if "profile_top_pos_tags" not in df.columns:
        df["profile_top_pos_tags"] = ""
    if "profile_top_neg_tags" not in df.columns:
        df["profile_top_neg_tags"] = ""
    if "profile_keywords" not in df.columns:
        df["profile_keywords"] = ""
    if "profile_confidence" not in df.columns:
        df["profile_confidence"] = 0.0
    df["user_id"] = df["user_id"].astype(str)
    df["profile_text"] = df["profile_text"].fillna("").astype(str)
    df["profile_text_short"] = df["profile_text_short"].fillna("").astype(str)
    df["profile_text_long"] = df["profile_text_long"].fillna("").astype(str)
    df["profile_top_pos_tags"] = df["profile_top_pos_tags"].fillna("").astype(str)
    df["profile_top_neg_tags"] = df["profile_top_neg_tags"].fillna("").astype(str)
    df["profile_keywords"] = df["profile_keywords"].fillna("").astype(str)
    df["profile_confidence"] = pd.to_numeric(df["profile_confidence"], errors="coerce").fillna(0.0).astype(float)
    return df


def load_item_semantic_pdf(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["business_id", "top_pos_tags", "top_neg_tags", "semantic_confidence", "semantic_support", "semantic_tag_richness"])
    use_cols = [
        "business_id",
        "top_pos_tags",
        "top_neg_tags",
        "semantic_confidence",
        "semantic_support",
        "semantic_tag_richness",
    ]
    sem = pd.read_csv(path, usecols=lambda c: c in set(use_cols))
    if "business_id" not in sem.columns:
        return pd.DataFrame(columns=["business_id", "top_pos_tags", "top_neg_tags", "semantic_confidence", "semantic_support", "semantic_tag_richness"])
    sem["business_id"] = sem["business_id"].astype(str)
    for c in ["top_pos_tags", "top_neg_tags"]:
        if c not in sem.columns:
            sem[c] = ""
        sem[c] = sem[c].fillna("").astype(str)
    for c in ["semantic_confidence", "semantic_support", "semantic_tag_richness"]:
        if c not in sem.columns:
            sem[c] = 0.0
        sem[c] = pd.to_numeric(sem[c], errors="coerce").fillna(0.0).astype(float)
    sem = sem.drop_duplicates("business_id")
    return sem


def build_user_text(profile_row: dict[str, Any], fallback_segment: str) -> str:
    parts: list[str] = []
    if USE_PROFILE_TAGS:
        pos = normalize_space(str(profile_row.get("profile_top_pos_tags", "")).replace("|", ", "))
        neg = normalize_space(str(profile_row.get("profile_top_neg_tags", "")).replace("|", ", "))
        kws = normalize_space(str(profile_row.get("profile_keywords", "")).replace(",", ", "))
        if pos:
            parts.append(f"likes: {pos}")
        if neg:
            parts.append(f"dislikes: {neg}")
        if kws:
            parts.append(f"keywords: {kws}")

    short_txt = normalize_space(str(profile_row.get("profile_text_short", "")))
    long_txt = normalize_space(str(profile_row.get("profile_text_long", "")))
    full_txt = normalize_space(str(profile_row.get("profile_text", "")))
    if short_txt:
        parts.append(short_txt[:420])
    if long_txt:
        parts.append(long_txt[:420])
    elif full_txt:
        parts.append(full_txt[:760])
    if not parts:
        seg = normalize_space(fallback_segment or "unknown")
        parts.append(f"user segment: {seg}; food preference")
    return normalize_space(" ; ".join(parts))


def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, trust_remote_code=True)


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=int(max(1, batch_size)),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def eval_from_pred_pdf_compact(
    pred_topk_pdf: pd.DataFrame,
    truth_pdf: pd.DataFrame,
    n_items_pool: int,
    item_pop_pdf: pd.DataFrame,
    total_train_events: int,
) -> tuple[float, float, float, float, float, float]:
    if truth_pdf.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    t = truth_pdf[["user_idx", "true_item_idx"]].drop_duplicates("user_idx").copy()
    p = pred_topk_pdf[["user_idx", "item_idx", "rank"]].copy()
    m = p.merge(t, on="user_idx", how="right")
    m["hit_rank"] = np.where(m["item_idx"] == m["true_item_idx"], m["rank"], np.nan)
    rank_df = m.groupby("user_idx", as_index=False)["hit_rank"].min()
    rank = rank_df["hit_rank"].fillna(0).to_numpy(dtype=np.int32)
    recall = float((rank > 0).mean())
    if (rank > 0).any():
        ndcg = float((1.0 / np.log2(rank[rank > 0] + 1.0)).sum() / max(1, len(rank)))
    else:
        ndcg = 0.0

    n_users = int(t["user_idx"].nunique())
    pred_users = int(p["user_idx"].nunique())
    user_cov = float(pred_users / max(1, n_users))
    n_items_pred = int(p["item_idx"].nunique())
    item_cov = float(n_items_pred / max(1, int(n_items_pool)))

    ip = item_pop_pdf[["item_idx", "item_train_pop_count"]].drop_duplicates("item_idx").copy()
    ip["item_train_pop_count"] = ip["item_train_pop_count"].fillna(1.0).astype(float)
    tail_cutoff = float(ip["item_train_pop_count"].quantile(TAIL_QUANTILE)) if not ip.empty else 0.0
    pred_diag = p.merge(ip, on="item_idx", how="left")
    pred_diag["item_train_pop_count"] = pred_diag["item_train_pop_count"].fillna(1.0).astype(float)
    pred_diag["pop_prob"] = np.maximum(
        pred_diag["item_train_pop_count"].to_numpy(dtype=np.float64) / max(1.0, float(total_train_events)),
        1e-12,
    )
    pred_diag["is_tail"] = (pred_diag["item_train_pop_count"] <= tail_cutoff).astype(float)
    pred_diag["novelty"] = -np.log2(pred_diag["pop_prob"])
    tail_cov = float(pred_diag["is_tail"].mean()) if not pred_diag.empty else 0.0
    novelty = float(pred_diag["novelty"].mean()) if not pred_diag.empty else 0.0
    return recall, ndcg, user_cov, item_cov, tail_cov, novelty


def append_metrics(rows: list[dict[str, Any]]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")


def _inv_rank(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    out = np.zeros_like(a, dtype=np.float64)
    ok = np.isfinite(a) & (a > 0.0)
    out[ok] = 1.0 / np.log2(a[ok] + 1.0)
    return out


def _norm_by_user(pdf: pd.DataFrame, col: str, out_col: str) -> None:
    pdf[out_col] = pdf.groupby("user_idx")[col].transform(
        lambda s: (s - s.min()) / max(1e-9, (s.max() - s.min()))
    )


def build_learn_features(cand_pdf: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feat = pd.DataFrame(index=cand_pdf.index)
    feat["pre_norm"] = cand_pdf["pre_norm"].astype(np.float64)
    feat["tf_norm"] = cand_pdf["tf_norm"].astype(np.float64)
    feat["tf_signal_norm"] = pd.to_numeric(cand_pdf.get("tf_signal_norm", cand_pdf["tf_norm"]), errors="coerce").fillna(0.0).astype(np.float64)
    feat["tag_norm"] = pd.to_numeric(cand_pdf.get("tag_norm", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["tag_align01"] = pd.to_numeric(cand_pdf.get("tag_align01", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["signal_score"] = pd.to_numeric(cand_pdf.get("signal_score", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["quality_score"] = pd.to_numeric(cand_pdf.get("quality_score", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["semantic_score"] = pd.to_numeric(cand_pdf.get("semantic_score", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["semantic_effective_score"] = pd.to_numeric(cand_pdf.get("semantic_effective_score", 0.0), errors="coerce").fillna(0.0).astype(
        np.float64
    )
    feat["semantic_confidence"] = pd.to_numeric(cand_pdf.get("semantic_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["semantic_support_log"] = np.log1p(
        pd.to_numeric(cand_pdf.get("semantic_support", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    )
    feat["semantic_tag_richness"] = pd.to_numeric(cand_pdf.get("semantic_tag_richness", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["profile_confidence"] = pd.to_numeric(cand_pdf.get("profile_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
    feat["inv_pre_rank"] = _inv_rank(cand_pdf["pre_rank"].to_numpy(np.float64))
    feat["inv_als_rank"] = _inv_rank(pd.to_numeric(cand_pdf.get("als_rank", np.nan), errors="coerce").to_numpy(np.float64))
    feat["inv_cluster_rank"] = _inv_rank(pd.to_numeric(cand_pdf.get("cluster_rank", np.nan), errors="coerce").to_numpy(np.float64))
    feat["inv_profile_rank"] = _inv_rank(pd.to_numeric(cand_pdf.get("profile_rank", np.nan), errors="coerce").to_numpy(np.float64))
    feat["inv_popular_rank"] = _inv_rank(pd.to_numeric(cand_pdf.get("popular_rank", np.nan), errors="coerce").to_numpy(np.float64))
    feat["has_als"] = np.isfinite(pd.to_numeric(cand_pdf.get("als_rank", np.nan), errors="coerce")).astype(np.float64)
    feat["has_cluster"] = np.isfinite(pd.to_numeric(cand_pdf.get("cluster_rank", np.nan), errors="coerce")).astype(np.float64)
    feat["has_profile"] = np.isfinite(pd.to_numeric(cand_pdf.get("profile_rank", np.nan), errors="coerce")).astype(np.float64)
    feat["has_popular"] = np.isfinite(pd.to_numeric(cand_pdf.get("popular_rank", np.nan), errors="coerce")).astype(np.float64)
    feat["pre_x_tf"] = feat["pre_norm"] * feat["tf_norm"]
    feat["pre_x_tf_signal"] = feat["pre_norm"] * feat["tf_signal_norm"]
    feat["tf_signal_x_sem"] = feat["tf_signal_norm"] * np.clip(feat["semantic_confidence"], 0.0, 1.0)
    feat["tag_x_profile"] = feat["tag_norm"] * np.clip(feat["profile_confidence"], 0.0, 1.0)
    feat["tf_x_sem"] = feat["tf_norm"] * np.clip(feat["semantic_confidence"], 0.0, 1.0)
    feat["pre_x_profile"] = feat["pre_norm"] * np.clip(feat["profile_confidence"], 0.0, 1.0)
    cols = list(feat.columns)
    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feat, cols


def sample_learn_rows(cand_pdf: pd.DataFrame) -> pd.DataFrame:
    pool = cand_pdf[cand_pdf["pre_rank"] <= int(max(1, LEARN_TRAIN_TOPN))].copy()
    if pool.empty:
        return pool
    pos = pool[pool["label"] > 0].copy()
    if pos.empty:
        return pos
    neg = pool[pool["label"] <= 0].copy()
    out = [pos]

    if int(LEARN_NEG_PRE_PER_USER) > 0 and not neg.empty:
        x = neg.sort_values(["user_idx", "pre_rank", "item_idx"], ascending=[True, True, True]).groupby("user_idx", as_index=False).head(
            int(LEARN_NEG_PRE_PER_USER)
        )
        out.append(x)
    if int(LEARN_NEG_SIM_PER_USER) > 0 and not neg.empty:
        x = neg.sort_values(["user_idx", "tf_norm", "item_idx"], ascending=[True, False, True]).groupby("user_idx", as_index=False).head(
            int(LEARN_NEG_SIM_PER_USER)
        )
        out.append(x)
    if int(LEARN_NEG_RANDOM_PER_USER) > 0 and not neg.empty:
        rng = np.random.default_rng(20260221)
        neg = neg.copy()
        neg["_r"] = rng.random(len(neg), dtype=np.float32)
        x = neg.sort_values(["user_idx", "_r"], ascending=[True, True]).groupby("user_idx", as_index=False).head(
            int(LEARN_NEG_RANDOM_PER_USER)
        )
        out.append(x.drop(columns=["_r"], errors="ignore"))

    train = pd.concat(out, ignore_index=True).drop_duplicates(["user_idx", "item_idx"])
    if int(LEARN_MAX_ROWS) > 0 and len(train) > int(LEARN_MAX_ROWS):
        pos_train = train[train["label"] > 0]
        neg_train = train[train["label"] <= 0]
        keep_neg = max(0, int(LEARN_MAX_ROWS) - len(pos_train))
        if keep_neg > 0 and len(neg_train) > keep_neg:
            neg_train = neg_train.sample(n=keep_neg, random_state=20260221)
        train = pd.concat([pos_train, neg_train], ignore_index=True)
    return train


def main() -> None:
    source_09 = resolve_stage09_run()
    profile_table = resolve_profile_table(source_09)
    item_semantic_table = resolve_item_semantic_table(source_09)
    model_name = resolve_model_name()
    print(f"[CONFIG] source_09={source_09}")
    print(f"[CONFIG] profile_table={profile_table}")
    print(f"[CONFIG] item_semantic_table={item_semantic_table if item_semantic_table else 'none'}")
    print(f"[CONFIG] transformer_model={model_name}")
    print(
        f"[CONFIG] top_k={TOP_K} rerank_topn={RERANK_TOPN} "
        f"alpha={BLEND_ALPHA} pre_power={PRE_PRIOR_POWER} sim_power={SIM_POWER} "
        f"dynamic_alpha={USE_DYNAMIC_ALPHA} fallback_topn={FALLBACK_OUTSIDE_TOPN}"
    )
    print(
        f"[CONFIG] tag_align={USE_TAG_ALIGN} weight={TAG_ALIGN_WEIGHT} "
        f"pos_gain={TAG_ALIGN_POS_GAIN} neg_penalty={TAG_ALIGN_NEG_PENALTY} "
        f"min_profile_conf={TAG_ALIGN_MIN_PROFILE_CONF}"
    )
    print(
        f"[CONFIG] learned_head={ENABLE_LEARNED_HEAD} learn_alpha={LEARN_ALPHA} "
        f"learn_train_topn={LEARN_TRAIN_TOPN} neg(pre/sim/rand)=({LEARN_NEG_PRE_PER_USER}/"
        f"{LEARN_NEG_SIM_PER_USER}/{LEARN_NEG_RANDOM_PER_USER}) learn_max_rows={LEARN_MAX_ROWS}"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    profiles = load_profiles(profile_table)
    profile_rows_by_user: dict[str, dict[str, Any]] = {}
    for r in profiles.itertuples(index=False):
        profile_rows_by_user[str(getattr(r, "user_id", ""))] = {
            "profile_text": str(getattr(r, "profile_text", "")),
            "profile_text_short": str(getattr(r, "profile_text_short", "")),
            "profile_text_long": str(getattr(r, "profile_text_long", "")),
            "profile_top_pos_tags": str(getattr(r, "profile_top_pos_tags", "")),
            "profile_top_neg_tags": str(getattr(r, "profile_top_neg_tags", "")),
            "profile_keywords": str(getattr(r, "profile_keywords", "")),
            "profile_confidence": float(getattr(r, "profile_confidence", 0.0) or 0.0),
        }
    item_sem_pdf = load_item_semantic_pdf(item_semantic_table)
    print(f"[COUNT] profiles={len(profile_rows_by_user)} item_semantic_rows={len(item_sem_pdf)}")

    encoder = load_encoder(model_name)
    metrics_rows: list[dict[str, Any]] = []

    bucket_dirs = sorted([p for p in source_09.iterdir() if p.is_dir() and p.name.startswith("bucket_")], key=lambda p: p.name)
    wanted = parse_bucket_override(BUCKETS_OVERRIDE)
    if wanted:
        bucket_dirs = [p for p in bucket_dirs if int(p.name.split("_")[-1]) in wanted]
    if not bucket_dirs:
        raise RuntimeError(f"no bucket dirs under {source_09}")

    for bdir in bucket_dirs:
        bucket = int(bdir.name.split("_")[-1])
        print(f"[BUCKET] {bucket}")
        cand_file = pick_candidate_file(bdir)
        print(f"[INFO] bucket={bucket} candidate_file={cand_file.name}")
        cand = spark.read.parquet(cand_file.as_posix())
        truth = spark.read.parquet((bdir / "truth.parquet").as_posix()).select("user_idx", "true_item_idx", "user_id")
        meta = json.loads((bdir / "bucket_meta.json").read_text(encoding="utf-8"))
        total_train_events = int(meta.get("total_train_events", 1))

        item_df = (
            cand.select("business_id", "item_idx", "name", "primary_category", "categories", "city", "item_train_pop_count", "semantic_confidence", "semantic_support", "semantic_tag_richness")
            .dropDuplicates(["item_idx"])
            .toPandas()
        )
        if len(item_sem_pdf) > 0 and "business_id" in item_df.columns:
            item_df["business_id"] = item_df["business_id"].astype(str)
            item_df = item_df.merge(item_sem_pdf, on="business_id", how="left", suffixes=("", "_sem"))
            for c in ["top_pos_tags", "top_neg_tags", "semantic_confidence_sem", "semantic_support_sem", "semantic_tag_richness_sem"]:
                if c not in item_df.columns:
                    item_df[c] = 0.0 if c.endswith("_sem") else ""
            # Prefer stronger semantic table values if available.
            for c, sem_c in [
                ("semantic_confidence", "semantic_confidence_sem"),
                ("semantic_support", "semantic_support_sem"),
                ("semantic_tag_richness", "semantic_tag_richness_sem"),
            ]:
                item_df[c] = pd.to_numeric(item_df[c], errors="coerce").fillna(0.0)
                item_df[sem_c] = pd.to_numeric(item_df[sem_c], errors="coerce").fillna(0.0)
                item_df[c] = np.where(item_df[sem_c] > 0.0, item_df[sem_c], item_df[c])
            item_df["top_pos_tags"] = item_df["top_pos_tags"].fillna("").astype(str)
            item_df["top_neg_tags"] = item_df["top_neg_tags"].fillna("").astype(str)
        else:
            item_df["top_pos_tags"] = ""
            item_df["top_neg_tags"] = ""

        item_df["item_text"] = [build_item_text(r._asdict()) for r in item_df.itertuples(index=False)]
        item_df["item_idx"] = item_df["item_idx"].astype(np.int32)
        item_texts = item_df["item_text"].fillna("").astype(str).tolist()
        item_emb = encode_texts(encoder, item_texts, batch_size=EMBED_BATCH_SIZE)
        item_idx_arr = item_df["item_idx"].to_numpy(np.int32)
        item_pos = {int(item_idx_arr[i]): int(i) for i in range(len(item_idx_arr))}

        truth_pdf = truth.toPandas()
        user_map = truth_pdf[["user_idx", "user_id"]].drop_duplicates("user_idx").copy()
        user_map["user_idx"] = user_map["user_idx"].astype(np.int32)
        user_map["profile_row"] = user_map["user_id"].astype(str).map(profile_rows_by_user)

        seg_map = cand.select("user_idx", "user_segment").dropDuplicates(["user_idx"]).toPandas()
        seg_map["user_idx"] = seg_map["user_idx"].astype(np.int32)
        seg_lookup = dict(zip(seg_map["user_idx"].tolist(), seg_map["user_segment"].fillna("").astype(str).tolist()))

        user_map["profile_confidence"] = user_map["profile_row"].map(
            lambda x: float(x.get("profile_confidence", 0.0)) if isinstance(x, dict) else 0.0
        )
        user_map["profile_confidence"] = pd.to_numeric(user_map["profile_confidence"], errors="coerce").fillna(0.0).astype(float)
        user_map["profile_text"] = user_map.apply(
            lambda r: build_user_text(
                r["profile_row"] if isinstance(r["profile_row"], dict) else {},
                seg_lookup.get(int(r["user_idx"]), "unknown"),
            ),
            axis=1,
        )
        user_texts = user_map["profile_text"].fillna("").astype(str).tolist()
        user_emb = encode_texts(encoder, user_texts, batch_size=EMBED_BATCH_SIZE)
        user_idx_arr = user_map["user_idx"].to_numpy(np.int32)
        user_pos = {int(user_idx_arr[i]): int(i) for i in range(len(user_idx_arr))}
        user_conf = dict(zip(user_map["user_idx"].tolist(), user_map["profile_confidence"].tolist()))
        user_tag_map: dict[int, tuple[set[str], set[str]]] = {}
        for ur in user_map[["user_idx", "profile_row"]].itertuples(index=False):
            pr = ur.profile_row if isinstance(ur.profile_row, dict) else {}
            u_pos = split_tags(str(pr.get("profile_top_pos_tags", "")))
            u_neg = split_tags(str(pr.get("profile_top_neg_tags", "")))
            user_tag_map[int(ur.user_idx)] = (u_pos, u_neg)
        item_tag_map: dict[int, tuple[set[str], set[str]]] = {}
        for ir in item_df[["item_idx", "top_pos_tags", "top_neg_tags"]].itertuples(index=False):
            i_pos = split_tags(str(ir.top_pos_tags))
            i_neg = split_tags(str(ir.top_neg_tags))
            item_tag_map[int(ir.item_idx)] = (i_pos, i_neg)

        cand_pdf = cand.select(
            "user_idx",
            "item_idx",
            "signal_score",
            "quality_score",
            "semantic_score",
            "semantic_effective_score",
            "pre_score",
            "pre_rank",
            "als_rank",
            "cluster_rank",
            "profile_rank",
            "popular_rank",
            "item_train_pop_count",
            "semantic_confidence",
            "semantic_support",
            "semantic_tag_richness",
        ).toPandas()
        cand_pdf["user_idx"] = cand_pdf["user_idx"].astype(np.int32)
        cand_pdf["item_idx"] = cand_pdf["item_idx"].astype(np.int32)
        cand_pdf["pre_rank"] = cand_pdf["pre_rank"].astype(np.int32)
        cand_pdf["pre_score"] = pd.to_numeric(cand_pdf["pre_score"], errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["item_train_pop_count"] = pd.to_numeric(cand_pdf["item_train_pop_count"], errors="coerce").fillna(1.0).astype(np.float64)
        cand_pdf["signal_score"] = pd.to_numeric(cand_pdf.get("signal_score", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["quality_score"] = pd.to_numeric(cand_pdf.get("quality_score", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["semantic_score"] = pd.to_numeric(cand_pdf.get("semantic_score", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["semantic_effective_score"] = pd.to_numeric(cand_pdf.get("semantic_effective_score", 0.0), errors="coerce").fillna(0.0).astype(
            np.float64
        )
        cand_pdf["semantic_confidence"] = pd.to_numeric(cand_pdf.get("semantic_confidence", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["semantic_support"] = pd.to_numeric(cand_pdf.get("semantic_support", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["semantic_tag_richness"] = pd.to_numeric(cand_pdf.get("semantic_tag_richness", 0.0), errors="coerce").fillna(0.0).astype(np.float64)
        cand_pdf["als_rank"] = pd.to_numeric(cand_pdf.get("als_rank", np.nan), errors="coerce").astype(np.float64)
        cand_pdf["cluster_rank"] = pd.to_numeric(cand_pdf.get("cluster_rank", np.nan), errors="coerce").astype(np.float64)
        cand_pdf["profile_rank"] = pd.to_numeric(cand_pdf.get("profile_rank", np.nan), errors="coerce").astype(np.float64)
        cand_pdf["popular_rank"] = pd.to_numeric(cand_pdf.get("popular_rank", np.nan), errors="coerce").astype(np.float64)
        cand_pdf["profile_confidence"] = cand_pdf["user_idx"].map(user_conf).fillna(0.0).astype(np.float64)
        cand_pdf["u_pos"] = cand_pdf["user_idx"].map(user_pos).astype(np.int32)
        cand_pdf["i_pos"] = cand_pdf["item_idx"].map(item_pos).astype(np.int32)

        u = cand_pdf["u_pos"].to_numpy(np.int32)
        i = cand_pdf["i_pos"].to_numpy(np.int32)
        sims = np.zeros(len(cand_pdf), dtype=np.float32)
        for st in range(0, len(cand_pdf), int(max(1, SIM_CHUNK_ROWS))):
            ed = min(st + int(max(1, SIM_CHUNK_ROWS)), len(cand_pdf))
            uc = user_emb[u[st:ed]]
            ic = item_emb[i[st:ed]]
            sims[st:ed] = np.sum(uc * ic, axis=1, dtype=np.float32)
        cand_pdf["tf_sim"] = sims
        cand_pdf["tf_sim01"] = np.clip((cand_pdf["tf_sim"].to_numpy(np.float64) + 1.0) * 0.5, 0.0, 1.0)

        truth_map = truth_pdf[["user_idx", "true_item_idx"]].drop_duplicates("user_idx")
        truth_lookup = dict(zip(truth_map["user_idx"].astype(np.int32).tolist(), truth_map["true_item_idx"].astype(np.int32).tolist()))
        cand_pdf["true_item_idx"] = cand_pdf["user_idx"].map(truth_lookup).astype(np.int32)
        cand_pdf["label"] = (cand_pdf["item_idx"].to_numpy(np.int32) == cand_pdf["true_item_idx"].to_numpy(np.int32)).astype(np.int32)

        # User-wise normalization keeps profile and semantic signals on comparable scales.
        cand_pdf["pre_prior"] = np.power(
            1.0 / np.log2(cand_pdf["pre_rank"].to_numpy(np.float64) + 1.0),
            float(max(1e-6, PRE_PRIOR_POWER)),
        )
        _norm_by_user(cand_pdf, "pre_prior", "pre_norm")
        cand_pdf["tf_norm"] = np.power(
            cand_pdf.groupby("user_idx")["tf_sim01"].rank(method="average", pct=True).astype(np.float64),
            float(max(1e-6, SIM_POWER)),
        )
        cand_pdf["tag_align"] = 0.0
        cand_pdf["tag_align01"] = 0.0
        cand_pdf["tag_norm"] = 0.0
        if USE_TAG_ALIGN:
            tag_align = compute_tag_align(
                cand_pdf["user_idx"].to_numpy(np.int32),
                cand_pdf["item_idx"].to_numpy(np.int32),
                user_tag_map,
                item_tag_map,
            )
            cand_pdf["tag_align"] = tag_align.astype(np.float64)
            cand_pdf["tag_align01"] = np.clip((cand_pdf["tag_align"].to_numpy(np.float64) + 1.0) * 0.5, 0.0, 1.0)
            _norm_by_user(cand_pdf, "tag_align01", "tag_norm")

        sem_conf_norm = np.clip(cand_pdf["semantic_confidence"].to_numpy(np.float64), 0.0, 1.0)
        sem_support_norm = np.clip(np.log1p(cand_pdf["semantic_support"].to_numpy(np.float64)) / np.log(12.0), 0.0, 1.0)
        item_sem_factor = np.clip(0.7 + 0.3 * (0.6 * sem_conf_norm + 0.4 * sem_support_norm), 0.6, 1.05)
        user_conf_norm = np.clip(cand_pdf["profile_confidence"].to_numpy(np.float64), 0.0, 1.0)
        has_profile_signal = np.isfinite(cand_pdf["profile_rank"].to_numpy(np.float64)).astype(np.float64)
        tag_gate = (user_conf_norm >= float(TAG_ALIGN_MIN_PROFILE_CONF)).astype(np.float64) * has_profile_signal
        tag_weight_eff = np.clip(float(max(0.0, TAG_ALIGN_WEIGHT)) * tag_gate, 0.0, 0.80)
        cand_pdf["tag_weight_eff"] = tag_weight_eff
        cand_pdf["tf_signal_norm"] = (
            (1.0 - cand_pdf["tag_weight_eff"].to_numpy(np.float64)) * cand_pdf["tf_norm"].to_numpy(np.float64)
            + cand_pdf["tag_weight_eff"].to_numpy(np.float64) * cand_pdf["tag_norm"].to_numpy(np.float64)
        )
        if USE_DYNAMIC_ALPHA:
            alpha_eff = np.clip(
                float(BLEND_ALPHA) * (0.6 + 0.4 * user_conf_norm) * item_sem_factor,
                float(BASE_ALPHA_MIN),
                float(BASE_ALPHA_MAX),
            )
        else:
            alpha_eff = np.clip(np.full(len(cand_pdf), float(BLEND_ALPHA), dtype=np.float64), float(BASE_ALPHA_MIN), float(BASE_ALPHA_MAX))
        cand_pdf["alpha_eff"] = alpha_eff
        cand_pdf["blend_raw_bienc"] = (
            (1.0 - cand_pdf["alpha_eff"]) * cand_pdf["pre_norm"]
            + cand_pdf["alpha_eff"] * cand_pdf["tf_signal_norm"]
        )
        cand_pdf["blend_raw"] = cand_pdf["blend_raw_bienc"]

        learned_enabled = bool(ENABLE_LEARNED_HEAD and LogisticRegression is not None)
        learned_train_rows = 0
        learned_pos_rows = 0
        learn_feature_cols: list[str] = []
        if learned_enabled:
            learn_feat, learn_feature_cols = build_learn_features(cand_pdf)
            train_df = sample_learn_rows(pd.concat([cand_pdf[["user_idx", "item_idx", "pre_rank", "label"]], learn_feat], axis=1))
            learned_train_rows = int(len(train_df))
            learned_pos_rows = int((train_df["label"] > 0).sum()) if learned_train_rows > 0 else 0
            if learned_train_rows > 0 and learned_pos_rows > 0 and learned_pos_rows < learned_train_rows:
                x_train = train_df[learn_feature_cols].to_numpy(dtype=np.float32, copy=False)
                y_train = train_df["label"].to_numpy(dtype=np.int32, copy=False)
                clf = LogisticRegression(
                    max_iter=500,
                    C=float(max(1e-4, LEARN_MODEL_C)),
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=20260221,
                )
                try:
                    clf.fit(x_train, y_train)
                    x_all = learn_feat[learn_feature_cols].to_numpy(dtype=np.float32, copy=False)
                    learned_prob = clf.predict_proba(x_all)[:, 1].astype(np.float64)
                    cand_pdf["learned_prob"] = np.clip(learned_prob, 0.0, 1.0)
                    _norm_by_user(cand_pdf, "learned_prob", "learned_norm")
                    alpha_learn = np.clip(
                        float(LEARN_ALPHA) * (0.7 + 0.3 * user_conf_norm) * item_sem_factor,
                        float(BASE_ALPHA_MIN),
                        float(BASE_ALPHA_MAX),
                    )
                    cand_pdf["alpha_learn"] = alpha_learn
                    cand_pdf["blend_raw_learn"] = (1.0 - cand_pdf["alpha_learn"]) * cand_pdf["pre_norm"] + cand_pdf["alpha_learn"] * cand_pdf[
                        "learned_norm"
                    ]
                except Exception:
                    learned_enabled = False
            else:
                learned_enabled = False
        if not learned_enabled:
            cand_pdf["blend_raw_learn"] = cand_pdf["blend_raw_bienc"]
            cand_pdf["alpha_learn"] = cand_pdf["alpha_eff"]

        if int(RERANK_TOPN) > 0:
            use_model = cand_pdf["pre_rank"].to_numpy(np.int32) <= int(RERANK_TOPN)
            if FALLBACK_OUTSIDE_TOPN:
                cand_pdf["final_score"] = np.where(use_model, cand_pdf["blend_raw_bienc"], cand_pdf["pre_norm"])
                cand_pdf["final_score_learn"] = np.where(use_model, cand_pdf["blend_raw_learn"], cand_pdf["pre_norm"])
            else:
                cand_pdf["final_score"] = cand_pdf["blend_raw_bienc"]
                cand_pdf["final_score_learn"] = cand_pdf["blend_raw_learn"]
        else:
            cand_pdf["final_score"] = cand_pdf["blend_raw_bienc"]
            cand_pdf["final_score_learn"] = cand_pdf["blend_raw_learn"]

        cand_pdf = cand_pdf.sort_values(["user_idx", "final_score", "pre_score", "item_idx"], ascending=[True, False, False, True])
        cand_pdf["rank"] = cand_pdf.groupby("user_idx").cumcount() + 1
        tf_topk = cand_pdf[cand_pdf["rank"] <= int(TOP_K)][["user_idx", "item_idx", "rank"]].copy()
        cand_pdf = cand_pdf.sort_values(["user_idx", "final_score_learn", "pre_score", "item_idx"], ascending=[True, False, False, True])
        cand_pdf["rank_learn"] = cand_pdf.groupby("user_idx").cumcount() + 1
        tf_learn_topk = cand_pdf[cand_pdf["rank_learn"] <= int(TOP_K)][["user_idx", "item_idx", "rank_learn"]].copy()
        tf_learn_topk = tf_learn_topk.rename(columns={"rank_learn": "rank"})
        pre_topk = cand_pdf[cand_pdf["pre_rank"] <= int(TOP_K)][["user_idx", "item_idx", "pre_rank"]].copy()
        pre_topk = pre_topk.rename(columns={"pre_rank": "rank"})

        als_topk = cand_pdf[cand_pdf["als_rank"].notna() & (cand_pdf["als_rank"] <= int(TOP_K))][["user_idx", "item_idx", "als_rank"]].copy()
        if als_topk.empty:
            als_topk = pd.DataFrame(columns=["user_idx", "item_idx", "rank"])
        else:
            als_topk = als_topk.rename(columns={"als_rank": "rank"})

        n_items_pool = int(item_df["item_idx"].nunique())
        item_pop_pdf = item_df[["item_idx", "item_train_pop_count"]].copy()
        n_users = int(truth_pdf["user_idx"].nunique())
        n_items = int(item_df["item_idx"].nunique())
        n_candidates = int(len(cand_pdf))

        als_metrics = eval_from_pred_pdf_compact(als_topk, truth_pdf, n_items_pool, item_pop_pdf, total_train_events)
        pre_metrics = eval_from_pred_pdf_compact(pre_topk, truth_pdf, n_items_pool, item_pop_pdf, total_train_events)
        tf_metrics = eval_from_pred_pdf_compact(tf_topk, truth_pdf, n_items_pool, item_pop_pdf, total_train_events)
        tf_learn_metrics = eval_from_pred_pdf_compact(tf_learn_topk, truth_pdf, n_items_pool, item_pop_pdf, total_train_events)

        print(
            f"[METRIC] bucket={bucket} ALS_ndcg={als_metrics[1]:.6f} "
            f"pre_ndcg={pre_metrics[1]:.6f} tf_ndcg={tf_metrics[1]:.6f} "
            f"tf_learn_ndcg={tf_learn_metrics[1]:.6f} "
            f"delta_tf={tf_metrics[1]-pre_metrics[1]:.6f} "
            f"delta_tf_learn={tf_learn_metrics[1]-pre_metrics[1]:.6f} "
            f"learn_rows={learned_train_rows} learn_pos={learned_pos_rows} "
            f"learn_enabled={learned_enabled}"
        )

        bucket_out = out_dir / f"bucket_{bucket}"
        bucket_out.mkdir(parents=True, exist_ok=True)
        tf_topk.to_csv(bucket_out / "transformer_top10.csv", index=False, encoding="utf-8-sig")
        tf_learn_topk.to_csv(bucket_out / "transformer_learned_top10.csv", index=False, encoding="utf-8-sig")
        pre_topk.to_csv(bucket_out / "pre_top10.csv", index=False, encoding="utf-8-sig")

        for model_name_out, m in [
            ("ALS@10_from_candidates", als_metrics),
            ("PreScore@10", pre_metrics),
            ("TransformerBiEnc@10", tf_metrics),
            ("TransformerLearned@10", tf_learn_metrics),
        ]:
            metrics_rows.append(
                {
                    "run_id_10": run_id,
                    "source_run_09": source_09.name,
                    "bucket_min_train_reviews": int(bucket),
                    "model": model_name_out,
                    "recall_at_k": float(m[0]),
                    "ndcg_at_k": float(m[1]),
                    "user_coverage_at_k": float(m[2]),
                    "item_coverage_at_k": float(m[3]),
                    "tail_coverage_at_k": float(m[4]),
                    "novelty_at_k": float(m[5]),
                    "n_users": int(n_users),
                    "n_items": int(n_items),
                    "n_candidates": int(n_candidates),
                    "transformer_model": model_name,
                    "blend_alpha": float(BLEND_ALPHA),
                    "rerank_topn": int(RERANK_TOPN),
                    "pre_prior_power": float(PRE_PRIOR_POWER),
                    "sim_power": float(SIM_POWER),
                    "dynamic_alpha": bool(USE_DYNAMIC_ALPHA),
                    "fallback_outside_topn": bool(FALLBACK_OUTSIDE_TOPN),
                    "candidate_file": cand_file.name,
                    "learned_head_enabled": bool(learned_enabled),
                    "learned_train_rows": int(learned_train_rows),
                    "learned_pos_rows": int(learned_pos_rows),
                    "learn_feature_count": int(len(learn_feature_cols)),
                }
            )

    append_metrics(metrics_rows)
    meta = {
        "run_id_10": run_id,
        "run_tag": RUN_TAG,
        "source_stage09_run": str(source_09),
        "profile_table": str(profile_table),
        "transformer_model": model_name,
        "top_k": int(TOP_K),
        "blend_alpha": float(BLEND_ALPHA),
        "rerank_topn": int(RERANK_TOPN),
        "pre_prior_power": float(PRE_PRIOR_POWER),
        "sim_power": float(SIM_POWER),
        "dynamic_alpha": bool(USE_DYNAMIC_ALPHA),
        "fallback_outside_topn": bool(FALLBACK_OUTSIDE_TOPN),
        "enable_learned_head": bool(ENABLE_LEARNED_HEAD),
        "learn_model_available": bool(LogisticRegression is not None),
        "learn_train_topn": int(LEARN_TRAIN_TOPN),
        "learn_neg_pre_per_user": int(LEARN_NEG_PRE_PER_USER),
        "learn_neg_sim_per_user": int(LEARN_NEG_SIM_PER_USER),
        "learn_neg_random_per_user": int(LEARN_NEG_RANDOM_PER_USER),
        "learn_max_rows": int(LEARN_MAX_ROWS),
        "learn_model_c": float(LEARN_MODEL_C),
        "learn_alpha": float(LEARN_ALPHA),
        "alpha_min": float(BASE_ALPHA_MIN),
        "alpha_max": float(BASE_ALPHA_MAX),
        "use_tag_align": bool(USE_TAG_ALIGN),
        "tag_align_weight": float(TAG_ALIGN_WEIGHT),
        "tag_align_pos_gain": float(TAG_ALIGN_POS_GAIN),
        "tag_align_neg_penalty": float(TAG_ALIGN_NEG_PENALTY),
        "tag_align_min_profile_conf": float(TAG_ALIGN_MIN_PROFILE_CONF),
        "use_profile_tags": bool(USE_PROFILE_TAGS),
        "use_item_tags": bool(USE_ITEM_TAGS),
        "sim_chunk_rows": int(SIM_CHUNK_ROWS),
        "embed_batch_size": int(EMBED_BATCH_SIZE),
        "buckets_override": sorted(parse_bucket_override(BUCKETS_OVERRIDE)),
        "metrics_path": str(METRICS_PATH),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] output_dir={out_dir}")
    print(f"[INFO] metrics_appended={METRICS_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
