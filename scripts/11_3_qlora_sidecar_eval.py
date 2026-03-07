from __future__ import annotations

import gc
import json
import math
import os
import inspect
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

# Keep transformers on torch path in mixed TensorFlow/Keras env.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Reduce fragmentation risk on small VRAM GPUs.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    os.getenv("QLORA_EVAL_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128").strip()
    or "expandable_segments:True,max_split_size_mb:128",
)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.modeling_utils as _tf_modeling_utils


from pipeline.project_paths import env_or_project_path, normalize_legacy_project_path, project_path
from pipeline.qlora_prompting import (
    build_binary_prompt,
    build_binary_prompt_semantic,
    build_item_text,
    build_item_text_semantic,
    build_user_text,
)
from pipeline.spark_tmp_manager import SparkTmpContext, build_spark_tmp_context


RUN_TAG = "stage11_3_qlora_sidecar_eval"
INPUT_09_RUN_DIR = os.getenv("INPUT_09_RUN_DIR", "").strip()
INPUT_09_ROOT = env_or_project_path("INPUT_09_ROOT_DIR", "data/output/09_candidate_fusion")
INPUT_09_SUFFIX = "_stage09_candidate_fusion"

INPUT_11_2_RUN_DIR = os.getenv("INPUT_11_2_RUN_DIR", "").strip()
INPUT_11_2_ROOT = env_or_project_path("INPUT_11_2_ROOT_DIR", "data/output/11_qlora_models")
INPUT_11_2_SUFFIX = "_stage11_2_qlora_train"
INPUT_11_DATA_RUN_DIR = os.getenv("INPUT_11_DATA_RUN_DIR", "").strip()

OUTPUT_ROOT = env_or_project_path("OUTPUT_11_SIDECAR_ROOT_DIR", "data/output/11_qlora_sidecar_eval")
METRICS_PATH = env_or_project_path("METRICS_STAGE11_SIDECAR_PATH", "data/metrics/recsys_stage11_qlora_sidecar_results.csv")

BUCKETS_OVERRIDE = os.getenv("BUCKETS_OVERRIDE", "10").strip()
TOP_K = int(os.getenv("RANK_EVAL_TOP_K", "10").strip() or 10)
RERANK_TOPN = int(os.getenv("QLORA_RERANK_TOPN", "80").strip() or 80)
BLEND_ALPHA = float(os.getenv("QLORA_BLEND_ALPHA", "0.12").strip() or 0.12)
MAX_USERS_PER_BUCKET = int(os.getenv("QLORA_EVAL_MAX_USERS_PER_BUCKET", "1200").strip() or 1200)
MAX_ROWS_PER_BUCKET = int(os.getenv("QLORA_EVAL_MAX_ROWS_PER_BUCKET", "200000").strip() or 200000)
ROW_CAP_ORDERED = os.getenv("QLORA_EVAL_ROW_CAP_ORDERED", "false").strip().lower() == "true"
MAX_SEQ_LEN = int(os.getenv("QLORA_EVAL_MAX_SEQ_LEN", "768").strip() or 768)
INFER_BATCH_SIZE = int(os.getenv("QLORA_EVAL_BATCH_SIZE", "12").strip() or 12)
PROMPT_CHUNK_ROWS = int(os.getenv("QLORA_EVAL_PROMPT_CHUNK_ROWS", "4096").strip() or 4096)
ITER_COALESCE_PARTITIONS = int(os.getenv("QLORA_EVAL_ITER_COALESCE", "8").strip() or 8)
INTERMEDIATE_FLUSH_ROWS = int(os.getenv("QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS", "8192").strip() or 8192)
RANDOM_SEED = int(os.getenv("QLORA_RANDOM_SEED", "42").strip() or 42)
USE_STAGE11_EVAL_SPLIT = os.getenv("QLORA_EVAL_USE_STAGE11_SPLIT", "true").strip().lower() == "true"

ENFORCE_STAGE09_GATE = os.getenv("QLORA_ENFORCE_STAGE09_GATE", "true").strip().lower() == "true"
STAGE09_GATE_METRICS_PATH = Path(
    os.getenv(
        "QLORA_STAGE09_GATE_METRICS_PATH",
        project_path("data/metrics/stage09_recall_audit_summary_latest.csv").as_posix(),
    ).strip()
)
STAGE09_GATE_MIN_TRUTH_IN_PRETRIM = float(os.getenv("QLORA_GATE_MIN_TRUTH_IN_PRETRIM", "0.82").strip() or 0.82)
STAGE09_GATE_MAX_PRETRIM_CUT_LOSS = float(os.getenv("QLORA_GATE_MAX_PRETRIM_CUT_LOSS", "0.08").strip() or 0.08)
STAGE09_GATE_MAX_HARD_MISS = float(os.getenv("QLORA_GATE_MAX_HARD_MISS", "0.10").strip() or 0.10)

ENABLE_SHORT_TEXT_SUMMARY = os.getenv("QLORA_ENABLE_SHORT_TEXT_SUMMARY", "true").strip().lower() == "true"
ENABLE_RAW_REVIEW_TEXT = os.getenv("QLORA_ENABLE_RAW_REVIEW_TEXT", "true").strip().lower() == "true"
REVIEW_TABLE_PATH = Path(
    os.getenv(
        "QLORA_REVIEW_TABLE_PATH",
        project_path("data/parquet/yelp_academic_dataset_review").as_posix(),
    ).strip()
)
USER_REVIEW_TOPN = int(os.getenv("QLORA_USER_REVIEW_TOPN", "2").strip() or 2)
ITEM_REVIEW_TOPN = int(os.getenv("QLORA_ITEM_REVIEW_TOPN", "2").strip() or 2)
REVIEW_SNIPPET_MAX_CHARS = int(os.getenv("QLORA_REVIEW_SNIPPET_MAX_CHARS", "220").strip() or 220)

USE_4BIT = os.getenv("QLORA_EVAL_USE_4BIT", "true").strip().lower() == "true"
USE_BF16 = os.getenv("QLORA_EVAL_USE_BF16", "true").strip().lower() == "true"
QWEN35_MAMBA_SSM_DTYPE = os.getenv("QLORA_QWEN35_MAMBA_SSM_DTYPE", "auto").strip().lower()
TRUST_REMOTE_CODE = os.getenv("QLORA_TRUST_REMOTE_CODE", "true").strip().lower() == "true"
DISABLE_ALLOC_WARMUP = os.getenv("QLORA_DISABLE_ALLOC_WARMUP", "true").strip().lower() == "true"
EVAL_DEVICE_MAP = os.getenv("QLORA_EVAL_DEVICE_MAP", "auto").strip() or "auto"
EVAL_MAX_MEMORY_CUDA = os.getenv("QLORA_EVAL_MAX_MEMORY_CUDA", "").strip()
EVAL_MAX_MEMORY_CPU = os.getenv("QLORA_EVAL_MAX_MEMORY_CPU", "").strip()
EVAL_OFFLOAD_FOLDER = os.getenv("QLORA_EVAL_OFFLOAD_FOLDER", "").strip()
EVAL_OFFLOAD_STATE_DICT = os.getenv("QLORA_EVAL_OFFLOAD_STATE_DICT", "true").strip().lower() == "true"
DISABLE_PARALLEL_LOADING = os.getenv("QLORA_EVAL_DISABLE_PARALLEL_LOADING", "true").strip().lower() == "true"
PARALLEL_LOADING_WORKERS = int(os.getenv("QLORA_EVAL_PARALLEL_LOADING_WORKERS", "1").strip() or 1)
LOAD_MODEL_BEFORE_SPARK = os.getenv("QLORA_EVAL_LOAD_MODEL_BEFORE_SPARK", "true").strip().lower() == "true"
INVERT_PROB = os.getenv("QLORA_INVERT_PROB", "false").strip().lower() == "true"
PROMPT_MODE = os.getenv("QLORA_PROMPT_MODE", "full").strip().lower() or "full"  # "full" | "semantic"
QWEN35_NO_THINK = os.getenv("QLORA_EVAL_QWEN35_NO_THINK", "false").strip().lower() == "true"

SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip() or "6g"
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "6g").strip() or "6g"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]").strip() or "local[2]"
SPARK_LOCAL_DIR = (
    os.getenv("SPARK_LOCAL_DIR", project_path("data/spark-tmp").as_posix()).strip()
    or project_path("data/spark-tmp").as_posix()
)
SPARK_SQL_SHUFFLE_PARTITIONS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "12").strip() or "12"
SPARK_DEFAULT_PARALLELISM = os.getenv("SPARK_DEFAULT_PARALLELISM", "12").strip() or "12"
SPARK_NETWORK_TIMEOUT = os.getenv("SPARK_NETWORK_TIMEOUT", "600s").strip() or "600s"
SPARK_EXECUTOR_HEARTBEAT_INTERVAL = os.getenv("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "60s").strip() or "60s"
SPARK_DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
SPARK_DRIVER_BIND_ADDRESS = os.getenv("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1").strip() or "127.0.0.1"
SPARK_TMP_SESSION_ISOLATION = os.getenv("SPARK_TMP_SESSION_ISOLATION", "true").strip().lower() == "true"
SPARK_TMP_AUTOCLEAN_ENABLED = os.getenv("SPARK_TMP_AUTOCLEAN_ENABLED", "true").strip().lower() == "true"
SPARK_TMP_CLEAN_ON_EXIT = os.getenv("SPARK_TMP_CLEAN_ON_EXIT", "true").strip().lower() == "true"
SPARK_TMP_RETENTION_HOURS = int(os.getenv("SPARK_TMP_RETENTION_HOURS", "8").strip() or 8)
SPARK_TMP_CLEAN_MAX_ENTRIES = int(os.getenv("SPARK_TMP_CLEAN_MAX_ENTRIES", "3000").strip() or 3000)
SPARK_RPC_MESSAGE_MAX_SIZE = os.getenv("SPARK_RPC_MESSAGE_MAX_SIZE", "1024").strip() or "1024"
SPARK_PYTHON_WORKER_REUSE = os.getenv("SPARK_PYTHON_WORKER_REUSE", "true").strip().lower() == "true"

_SPARK_TMP_CTX: SparkTmpContext | None = None


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


def is_qwen35_model_type(model_type: str) -> bool:
    mt = str(model_type or "").strip().lower()
    return mt.startswith("qwen3_5")


def detect_final_token_logits_arg(model: Any) -> str:
    seen: set[int] = set()
    candidates: list[Any] = [model]

    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        try:
            base_obj = get_base()
            if base_obj is not None:
                candidates.append(base_obj)
        except Exception:
            pass

    base_model_obj = getattr(model, "base_model", None)
    if base_model_obj is not None:
        candidates.append(base_model_obj)
        nested_model = getattr(base_model_obj, "model", None)
        if nested_model is not None:
            candidates.append(nested_model)

    for obj in candidates:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            sig = inspect.signature(obj.forward)
        except Exception:
            continue
        if "num_logits_to_keep" in sig.parameters:
            return "num_logits_to_keep"
        if "logits_to_keep" in sig.parameters:
            return "logits_to_keep"

    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
    if is_qwen35_model_type(model_type):
        # PeftModelForCausalLM.forward hides this kwarg behind **kwargs,
        # but it is forwarded to the underlying Qwen3.5 base model.
        return "logits_to_keep"

    return ""


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


def resolve_stage11_2_run() -> Path:
    def _is_complete_run(run_dir: Path) -> bool:
        return run_dir.is_dir() and (run_dir / "run_meta.json").exists() and (run_dir / "adapter").is_dir()

    if INPUT_11_2_RUN_DIR:
        p = Path(INPUT_11_2_RUN_DIR)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_11_2_RUN_DIR not found: {p}")
        if not _is_complete_run(p):
            raise FileNotFoundError(
                f"INPUT_11_2_RUN_DIR is incomplete (need run_meta.json + adapter dir): {p}"
            )
        return p
    runs = [p for p in INPUT_11_2_ROOT.iterdir() if p.is_dir() and p.name.endswith(INPUT_11_2_SUFFIX)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if _is_complete_run(r):
            return r
    if not runs:
        raise FileNotFoundError(f"no run found in {INPUT_11_2_ROOT} with suffix={INPUT_11_2_SUFFIX}")
    checked = ", ".join([r.name for r in runs[:3]])
    raise FileNotFoundError(
        "no complete stage11_2 run found "
        f"(need run_meta.json + adapter dir). checked_latest={checked}"
    )


def resolve_stage11_data_run(stage11_meta: dict[str, Any]) -> Path | None:
    raw = INPUT_11_DATA_RUN_DIR.strip() or str(stage11_meta.get("source_stage11_dataset_run", "")).strip()
    if not raw:
        if USE_STAGE11_EVAL_SPLIT:
            raise FileNotFoundError(
                "stage11 dataset run missing; set INPUT_11_DATA_RUN_DIR or ensure "
                "source_stage11_dataset_run exists in stage11_2 run_meta.json."
            )
        return None
    p = Path(raw)
    if not p.exists():
        if USE_STAGE11_EVAL_SPLIT:
            raise FileNotFoundError(f"stage11 dataset run not found: {p}")
        return None
    return p


def enforce_stage09_gate(source_09: Path, buckets: list[int]) -> dict[str, dict[str, float]]:
    if not ENFORCE_STAGE09_GATE:
        return {}
    if not STAGE09_GATE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "stage09 gate metrics csv not found: "
            f"{STAGE09_GATE_METRICS_PATH}. Set QLORA_ENFORCE_STAGE09_GATE=false to bypass."
        )
    mdf = pd.read_csv(STAGE09_GATE_METRICS_PATH)
    if mdf.empty:
        raise RuntimeError(f"stage09 gate metrics csv is empty: {STAGE09_GATE_METRICS_PATH}")

    src_key = source_09.name
    if "source_run_09" in mdf.columns:
        s = mdf["source_run_09"].astype(str)
        scoped = mdf[(s == src_key) | (s == str(source_09)) | s.str.endswith(src_key)].copy()
    else:
        scoped = mdf.copy()
    if scoped.empty:
        raise RuntimeError(
            "no stage09 gate rows matched current source run: "
            f"source={source_09}. csv={STAGE09_GATE_METRICS_PATH}"
        )

    out: dict[str, dict[str, float]] = {}
    for b in sorted(list(set([int(x) for x in buckets]))):
        rows = scoped[scoped["bucket"].astype(int) == int(b)].copy()
        if rows.empty:
            raise RuntimeError(f"stage09 gate missing bucket={b} in {STAGE09_GATE_METRICS_PATH}")
        if "run_id" in rows.columns:
            rows = rows.sort_values("run_id")
        row = rows.iloc[-1]

        truth_in_pretrim = float(row.get("truth_in_pretrim", float("nan")))
        pretrim_cut_loss = float(row.get("pretrim_cut_loss", row.get("cut_loss", float("nan"))))
        hard_miss = float(row.get("hard_miss", float("nan")))
        if not (math.isfinite(truth_in_pretrim) and math.isfinite(pretrim_cut_loss) and math.isfinite(hard_miss)):
            raise RuntimeError(f"stage09 gate has non-finite metrics for bucket={b}: {dict(row)}")

        if truth_in_pretrim < float(STAGE09_GATE_MIN_TRUTH_IN_PRETRIM):
            raise RuntimeError(
                f"stage09 gate fail bucket={b}: truth_in_pretrim={truth_in_pretrim:.6f} "
                f"< {float(STAGE09_GATE_MIN_TRUTH_IN_PRETRIM):.6f}"
            )
        if pretrim_cut_loss > float(STAGE09_GATE_MAX_PRETRIM_CUT_LOSS):
            raise RuntimeError(
                f"stage09 gate fail bucket={b}: pretrim_cut_loss={pretrim_cut_loss:.6f} "
                f"> {float(STAGE09_GATE_MAX_PRETRIM_CUT_LOSS):.6f}"
            )
        if hard_miss > float(STAGE09_GATE_MAX_HARD_MISS):
            raise RuntimeError(
                f"stage09 gate fail bucket={b}: hard_miss={hard_miss:.6f} "
                f"> {float(STAGE09_GATE_MAX_HARD_MISS):.6f}"
            )
        out[str(b)] = {
            "truth_in_pretrim": truth_in_pretrim,
            "pretrim_cut_loss": pretrim_cut_loss,
            "hard_miss": hard_miss,
        }
    print(
        f"[GATE] stage09 pass source={source_09.name} buckets={sorted(list(out.keys()))} "
        f"metrics_csv={STAGE09_GATE_METRICS_PATH}"
    )
    return out


def load_eval_users_from_stage11_data(spark: SparkSession, stage11_data_run: Path, bucket: int) -> DataFrame:
    bdir = stage11_data_run / f"bucket_{int(bucket)}" / "all_parquet"
    if not bdir.exists():
        raise FileNotFoundError(f"stage11 all_parquet missing for bucket={bucket}: {bdir}")
    sdf = spark.read.parquet(bdir.as_posix())
    cols = set(sdf.columns)
    if "user_idx" not in cols or "split" not in cols:
        raise RuntimeError(f"stage11 all_parquet missing required columns user_idx/split: {bdir}")
    return sdf.filter(F.col("split") == F.lit("eval")).select("user_idx").dropDuplicates(["user_idx"])


def resolve_stage09_meta_path(raw_path: str) -> Path:
    p = normalize_legacy_project_path(str(raw_path or "").strip())
    if p.exists():
        return p
    return p


def build_spark() -> SparkSession:
    global _SPARK_TMP_CTX
    _SPARK_TMP_CTX = build_spark_tmp_context(
        script_tag=RUN_TAG,
        spark_local_dir=SPARK_LOCAL_DIR,
        session_isolation=SPARK_TMP_SESSION_ISOLATION,
        auto_clean_enabled=SPARK_TMP_AUTOCLEAN_ENABLED,
        clean_on_exit=SPARK_TMP_CLEAN_ON_EXIT,
        retention_hours=SPARK_TMP_RETENTION_HOURS,
        clean_max_entries=SPARK_TMP_CLEAN_MAX_ENTRIES,
        set_env_temp=True,
    )
    print(
        f"[TMP] base={_SPARK_TMP_CTX.base_dir} local={_SPARK_TMP_CTX.spark_local_dir} "
        f"py_temp={_SPARK_TMP_CTX.py_temp_dir} cleanup={_SPARK_TMP_CTX.cleanup_summary}"
    )
    return (
        SparkSession.builder.appName(RUN_TAG)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.driver.host", SPARK_DRIVER_HOST)
        .config("spark.driver.bindAddress", SPARK_DRIVER_BIND_ADDRESS)
        .config("spark.local.dir", str(_SPARK_TMP_CTX.spark_local_dir))
        .config("spark.sql.shuffle.partitions", SPARK_SQL_SHUFFLE_PARTITIONS)
        .config("spark.default.parallelism", "4")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.python.worker.reuse", "true")
        .config("spark.network.timeout", "3600s")
        .config("spark.executor.heartbeatInterval", "600s")
        .config("spark.rpc.message.maxSize", "2046")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
        .config("spark.python.worker.memory", "2g")
        .config("spark.task.maxFailures", "4")
        .config("spark.io.compression.codec", "lz4")
        .config("spark.file.transferTo", "false")
        .config("spark.shuffle.file.buffer", "32k")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def pick_candidate_file(bucket_dir: Path) -> Path:
    candidates = [
        "candidates_pretrim150.parquet",
        "candidates_pretrim250.parquet",
        "candidates_pretrim300.parquet",
        "candidates_pretrim360.parquet",
        "candidates_pretrim500.parquet",
        "candidates_pretrim.parquet",
        "candidates.parquet",
    ]
    for name in candidates:
        p = bucket_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"candidate parquet not found under {bucket_dir}")


def _read_csv_subset(path: str, wanted_cols: list[str]) -> pd.DataFrame:
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=wanted_cols)
    cols = list(pd.read_csv(path, nrows=0).columns)
    usecols = [c for c in wanted_cols if c in cols]
    if not usecols:
        return pd.DataFrame(columns=wanted_cols)
    out = pd.read_csv(path, usecols=usecols)
    for c in wanted_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[wanted_cols]


def _str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and not np.isfinite(v):
        return ""
    s = str(v)
    return s if s.lower() != "nan" else ""


def _float_or_zero(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
        return float(f) if np.isfinite(f) else 0.0
    except Exception:
        return 0.0


def build_profile_lookup(profile_pdf: pd.DataFrame) -> dict[str, tuple[str, str, str, float]]:
    out: dict[str, tuple[str, str, str, float]] = {}
    if profile_pdf.empty:
        return out
    for r in profile_pdf.itertuples(index=False):
        uid = _str_or_empty(getattr(r, "user_id", ""))
        if not uid:
            continue
        out[uid] = (
            _str_or_empty(getattr(r, "profile_text", "")),
            _str_or_empty(getattr(r, "profile_top_pos_tags", "")),
            _str_or_empty(getattr(r, "profile_top_neg_tags", "")),
            _float_or_zero(getattr(r, "profile_confidence", 0.0)),
        )
    return out


def build_item_lookup(item_sem_pdf: pd.DataFrame) -> dict[str, tuple[str, str, float, float]]:
    out: dict[str, tuple[str, str, float, float]] = {}
    if item_sem_pdf.empty:
        return out
    for r in item_sem_pdf.itertuples(index=False):
        bid = _str_or_empty(getattr(r, "business_id", ""))
        if not bid:
            continue
        out[bid] = (
            _str_or_empty(getattr(r, "top_pos_tags", "")),
            _str_or_empty(getattr(r, "top_neg_tags", "")),
            _float_or_zero(getattr(r, "semantic_score", 0.0)),
            _float_or_zero(getattr(r, "semantic_confidence", 0.0)),
        )
    return out


def load_profile_item_tables(stage09_meta: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    item_sem_path = str(stage09_meta.get("item_semantic_features", "")).strip()

    pcols = ["user_id", "profile_text_short", "profile_text", "profile_top_pos_tags", "profile_top_neg_tags", "profile_confidence"]
    icols = ["business_id", "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence"]

    profile = _read_csv_subset(profile_path, pcols)
    profile["profile_text"] = profile["profile_text_short"].fillna(profile["profile_text"]).fillna("")
    profile = profile[["user_id", "profile_text", "profile_top_pos_tags", "profile_top_neg_tags", "profile_confidence"]].drop_duplicates(
        subset=["user_id"]
    )

    item_sem = _read_csv_subset(item_sem_path, icols)
    item_sem = item_sem.drop_duplicates(subset=["business_id"])
    return profile, item_sem


def build_user_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("user_profile_table", "")).strip()
    schema = "user_id string, profile_text string, profile_top_pos_tags string, profile_top_neg_tags string, profile_confidence double"
    if not profile_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "user_id",
            F.coalesce(F.col("profile_text_short"), F.col("profile_text"), F.lit("")).alias("profile_text"),
            F.coalesce(F.col("profile_top_pos_tags"), F.lit("")).alias("profile_top_pos_tags"),
            F.coalesce(F.col("profile_top_neg_tags"), F.lit("")).alias("profile_top_neg_tags"),
            F.col("profile_confidence").cast("double").alias("profile_confidence"),
        )
        .dropDuplicates(["user_id"])
    )


def build_item_sem_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    sem_path = str(stage09_meta.get("item_semantic_features", "")).strip()
    schema = "business_id string, top_pos_tags string, top_neg_tags string, semantic_score double, semantic_confidence double"
    if not sem_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(sem_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("top_pos_tags"), F.lit("")).alias("top_pos_tags"),
            F.coalesce(F.col("top_neg_tags"), F.lit("")).alias("top_neg_tags"),
            F.col("semantic_score").cast("double").alias("semantic_score"),
            F.col("semantic_confidence").cast("double").alias("semantic_confidence"),
        )
        .dropDuplicates(["business_id"])
    )


def build_temporal_review_features(
    spark: SparkSession,
    bucket_dir: Path,
    truth_df: DataFrame,
    cand_df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    user_schema = "user_idx int, user_review_summary string, user_review_raw string"
    item_schema = "user_idx int, business_id string, item_review_summary string, item_review_raw string"
    empty_user = spark.createDataFrame([], user_schema)
    empty_item = spark.createDataFrame([], item_schema)
    if not (ENABLE_SHORT_TEXT_SUMMARY or ENABLE_RAW_REVIEW_TEXT):
        return empty_user, empty_item
    if not REVIEW_TABLE_PATH.exists():
        print(f"[WARN] review table not found, skip review text features: {REVIEW_TABLE_PATH}")
        return empty_user, empty_item
    hist_path = bucket_dir / "train_history.parquet"
    if not hist_path.exists():
        print(f"[WARN] train_history.parquet missing, skip review text features: {hist_path}")
        return empty_user, empty_item

    user_cutoff = (
        spark.read.parquet(hist_path.as_posix())
        .select("user_idx", "test_ts")
        .filter(F.col("test_ts").isNotNull())
        .groupBy("user_idx")
        .agg(F.max("test_ts").alias("test_ts"))
    )

    rvw = (
        spark.read.parquet(REVIEW_TABLE_PATH.as_posix())
        .select("user_id", "business_id", "date", "text", "stars")
        .withColumn("review_ts", F.to_timestamp(F.col("date")))
        .withColumn("text_clean", F.regexp_replace(F.regexp_replace(F.coalesce(F.col("text"), F.lit("")), r"[\r\n]+", " "), r"\s+", " "))
        .filter(F.col("review_ts").isNotNull() & (F.length(F.col("text_clean")) > F.lit(0)))
    )
    u_topn = max(1, int(USER_REVIEW_TOPN))
    i_topn = max(1, int(ITEM_REVIEW_TOPN))
    max_chars = max(40, int(REVIEW_SNIPPET_MAX_CHARS))

    user_base = truth_df.select("user_idx", "user_id").dropDuplicates(["user_idx"]).join(user_cutoff, on="user_idx", how="inner")
    user_rows = (
        user_base.join(rvw.select("user_id", "review_ts", "text_clean", "stars"), on="user_id", how="inner")
        .filter(F.col("review_ts") < F.col("test_ts"))
        .withColumn("snippet", F.substring(F.col("text_clean"), 1, int(max_chars)))
    )
    w_user = Window.partitionBy("user_idx").orderBy(F.col("review_ts").desc())
    user_top = user_rows.withColumn("rn", F.row_number().over(w_user)).filter(F.col("rn") <= F.lit(int(u_topn)))
    user_feat = (
        user_top.groupBy("user_idx")
        .agg(
            F.count("*").alias("n_reviews"),
            F.avg(F.col("stars").cast("double")).alias("avg_stars"),
            F.concat_ws(" || ", F.collect_list("snippet")).alias("user_review_raw"),
        )
        .withColumn(
            "user_review_summary",
            F.concat(
                F.lit("recent_user_reviews="),
                F.col("n_reviews").cast("string"),
                F.lit(", avg_stars="),
                F.format_number(F.coalesce(F.col("avg_stars"), F.lit(0.0)), 2),
            ),
        )
        .select("user_idx", "user_review_summary", "user_review_raw")
    )

    item_base = (
        cand_df.select("user_idx", "business_id")
        .dropDuplicates(["user_idx", "business_id"])
        .join(user_cutoff, on="user_idx", how="inner")
    )
    item_rows = (
        item_base.join(rvw.select("business_id", "review_ts", "text_clean", "stars"), on="business_id", how="inner")
        .filter(F.col("review_ts") < F.col("test_ts"))
        .withColumn("snippet", F.substring(F.col("text_clean"), 1, int(max_chars)))
    )
    w_item = Window.partitionBy("user_idx", "business_id").orderBy(F.col("review_ts").desc())
    item_top = item_rows.withColumn("rn", F.row_number().over(w_item)).filter(F.col("rn") <= F.lit(int(i_topn)))
    item_feat = (
        item_top.groupBy("user_idx", "business_id")
        .agg(
            F.count("*").alias("n_reviews"),
            F.avg(F.col("stars").cast("double")).alias("avg_stars"),
            F.concat_ws(" || ", F.collect_list("snippet")).alias("item_review_raw"),
        )
        .withColumn(
            "item_review_summary",
            F.concat(
                F.lit("recent_item_reviews="),
                F.col("n_reviews").cast("string"),
                F.lit(", avg_stars="),
                F.format_number(F.coalesce(F.col("avg_stars"), F.lit(0.0)), 2),
            ),
        )
        .select("user_idx", "business_id", "item_review_summary", "item_review_raw")
    )

    if not ENABLE_SHORT_TEXT_SUMMARY:
        user_feat = user_feat.withColumn("user_review_summary", F.lit(""))
        item_feat = item_feat.withColumn("item_review_summary", F.lit(""))
    if not ENABLE_RAW_REVIEW_TEXT:
        user_feat = user_feat.withColumn("user_review_raw", F.lit(""))
        item_feat = item_feat.withColumn("item_review_raw", F.lit(""))
    return user_feat, item_feat


def build_prompt_udf() -> Any:
    def _mk(
        profile_text: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        user_review_summary: Any,
        user_review_raw: Any,
        name: Any,
        city: Any,
        categories: Any,
        primary_category: Any,
        top_pos_tags: Any,
        top_neg_tags: Any,
        semantic_score: Any,
        semantic_confidence: Any,
        source_set_text: Any,
        user_segment: Any,
        als_rank: Any,
        cluster_rank: Any,
        profile_rank: Any,
        popular_rank: Any,
        semantic_support: Any,
        semantic_tag_richness: Any,
        tower_score: Any,
        seq_score: Any,
        cluster_for_recsys: Any,
        cluster_label_for_recsys: Any,
        item_review_summary: Any,
        item_review_raw: Any,
    ) -> str:
        return build_binary_prompt(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                review_summary=user_review_summary,
                review_raw_snippet=user_review_raw,
            ),
            build_item_text(
                name,
                city,
                categories,
                primary_category,
                top_pos_tags,
                top_neg_tags,
                semantic_score,
                semantic_confidence,
                source_set=source_set_text,
                user_segment=user_segment,
                als_rank=als_rank,
                cluster_rank=cluster_rank,
                profile_rank=profile_rank,
                popular_rank=popular_rank,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_summary=item_review_summary,
                item_review_snippet=item_review_raw,
            ),
        )

    return F.udf(_mk, "string")


def build_prompt_udf_semantic() -> Any:
    """Prompt UDF that drops ranking-position features for DPO eval."""
    def _mk(
        profile_text: Any,
        profile_top_pos_tags: Any,
        profile_top_neg_tags: Any,
        profile_confidence: Any,
        user_review_summary: Any,
        user_review_raw: Any,
        name: Any,
        city: Any,
        categories: Any,
        primary_category: Any,
        top_pos_tags: Any,
        top_neg_tags: Any,
        semantic_score: Any,
        semantic_confidence: Any,
        semantic_support: Any,
        semantic_tag_richness: Any,
        tower_score: Any,
        seq_score: Any,
        cluster_for_recsys: Any,
        cluster_label_for_recsys: Any,
        item_review_summary: Any,
        item_review_raw: Any,
    ) -> str:
        return build_binary_prompt_semantic(
            build_user_text(
                profile_text,
                profile_top_pos_tags,
                profile_top_neg_tags,
                profile_confidence,
                review_summary=user_review_summary,
                review_raw_snippet=user_review_raw,
            ),
            build_item_text_semantic(
                name,
                city,
                categories,
                primary_category,
                top_pos_tags,
                top_neg_tags,
                semantic_score,
                semantic_confidence,
                semantic_support=semantic_support,
                semantic_tag_richness=semantic_tag_richness,
                tower_score=tower_score,
                seq_score=seq_score,
                cluster_for_recsys=cluster_for_recsys,
                cluster_label_for_recsys=cluster_label_for_recsys,
                item_review_summary=item_review_summary,
                item_review_snippet=item_review_raw,
            ),
        )

    return F.udf(_mk, "string")


def build_cluster_profile_df(spark: SparkSession, stage09_meta: dict[str, Any]) -> DataFrame:
    profile_path = str(stage09_meta.get("cluster_profile_csv", "")).strip()
    schema = "business_id string, cluster_for_recsys string, cluster_label_for_recsys string"
    if not profile_path:
        return spark.createDataFrame([], schema)
    p = resolve_stage09_meta_path(profile_path)
    if not p.exists():
        return spark.createDataFrame([], schema)
    return (
        spark.read.option("header", "true").csv(p.as_posix())
        .select(
            "business_id",
            F.coalesce(F.col("cluster_for_recsys"), F.lit("")).alias("cluster_for_recsys"),
            F.coalesce(F.col("cluster_label_for_recsys"), F.lit("")).alias("cluster_label_for_recsys"),
        )
        .dropDuplicates(["business_id"])
    )


def build_yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    def _pick(cands: list[str], fallback: str) -> int:
        for c in cands:
            ids = tokenizer(c, add_special_tokens=False).input_ids
            if len(ids) == 1:
                return int(ids[0])
        ids = tokenizer(fallback, add_special_tokens=False).input_ids
        if not ids:
            raise RuntimeError(f"failed to build token id for fallback={fallback}")
        return int(ids[0])

    yes_id = _pick([" YES", "Yes", " yes", "YES"], "Y")
    no_id = _pick([" NO", "No", " no", "NO"], "N")
    return yes_id, no_id


def score_yes_probability(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_seq_len: int,
    yes_id: int,
    no_id: int,
) -> tuple[np.ndarray, int]:
    def _next_fallback_bs(cur_bs: int) -> int:
        b = int(cur_bs)
        if b <= 1:
            return 1
        # User-requested path: if batch=3 fails, fallback to 2 first.
        if b == 3:
            return 2
        return max(1, b // 2)

    def _safe_cuda_empty_cache() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception as cache_exc:
            short = str(cache_exc).splitlines()[0][:180]
            print(f"[WARN] torch.cuda.empty_cache failed: {short}")

    out: list[np.ndarray] = []
    template_enable_thinking_supported = False
    template_use_chat = bool(QWEN35_NO_THINK and hasattr(tokenizer, "apply_chat_template"))
    if template_use_chat:
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)  # type: ignore[attr-defined]
            template_enable_thinking_supported = "enable_thinking" in sig.parameters
        except Exception:
            template_enable_thinking_supported = False

    model.eval()
    forward_kwargs_base: dict[str, Any] = {"use_cache": False}
    final_token_logits_arg = detect_final_token_logits_arg(model)
    if final_token_logits_arg:
        # Only score the final token position; avoids allocating full seq_len logits.
        forward_kwargs_base[final_token_logits_arg] = 1
        print(f"[CONFIG] final_token_logits_arg={final_token_logits_arg}")
    else:
        print("[CONFIG] final_token_logits_arg=<unsupported>")
    effective_bs = max(1, int(batch_size))
    t0 = time.monotonic()
    scored_total = 0
    last_log_time = t0
    with torch.inference_mode():
        i = 0
        n = len(prompts)
        while i < n:
            bs = min(effective_bs, n - i)
            batch = prompts[i : i + bs]
            try:
                if template_use_chat:
                    wrapped: list[str] = []
                    for p in batch:
                        messages = [{"role": "user", "content": str(p)}]
                        kwargs: dict[str, Any] = {
                            "tokenize": False,
                            "add_generation_prompt": True,
                        }
                        if template_enable_thinking_supported:
                            kwargs["enable_thinking"] = False
                        txt = tokenizer.apply_chat_template(messages, **kwargs)  # type: ignore[attr-defined]
                        wrapped.append(str(txt))
                    batch = wrapped
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}
                outputs = model(**enc, **forward_kwargs_base)
                logits = outputs.logits
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                pair = logits[:, [yes_id, no_id]]
                probs = torch.softmax(pair, dim=1)[:, 0]
                out.append(probs.detach().float().cpu().numpy())
                i += bs
                scored_total += bs
                now = time.monotonic()
                if now - last_log_time >= 30.0 or i >= n:
                    elapsed = now - t0
                    speed = scored_total / elapsed if elapsed > 0 else 0
                    remaining = (n - i) / speed if speed > 0 else 0
                    print(
                        f"[PROGRESS] scored={scored_total}/{n} "
                        f"speed={speed:.1f} rows/s "
                        f"elapsed={elapsed:.0f}s "
                        f"ETA={remaining:.0f}s "
                        f"batch_size={effective_bs}"
                    )
                    last_log_time = now
            except torch.OutOfMemoryError as oom:
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError(
                        "QLoRA sidecar inference OOM at batch_size=1. "
                        "Try lowering QLORA_EVAL_MAX_SEQ_LEN or disabling 4bit eval fallback."
                    ) from oom
                effective_bs = _next_fallback_bs(bs)
                print(f"[WARN] QLoRA eval OOM: batch={bs}, fallback_batch={effective_bs}")
            except RuntimeError as rte:
                msg = str(rte).lower()
                if "cuda" not in msg:
                    raise
                # These failures usually poison CUDA context; restarting process is safer.
                if "illegal memory access" in msg or "unspecified launch failure" in msg:
                    raise RuntimeError(
                        "QLoRA sidecar inference hit CUDA kernel failure. "
                        "Please restart Python process and retry with smaller "
                        "QLORA_EVAL_BATCH_SIZE (e.g. 12) and/or QLORA_EVAL_MAX_SEQ_LEN."
                    ) from rte
                is_retryable = ("cublas" in msg) or ("cudnn" in msg) or ("out of memory" in msg)
                if not is_retryable:
                    raise
                _safe_cuda_empty_cache()
                if bs <= 1:
                    raise RuntimeError(
                        "QLoRA sidecar inference CUDA runtime failure at batch_size=1. "
                        "Try lowering QLORA_EVAL_MAX_SEQ_LEN or setting QLORA_EVAL_USE_4BIT=false."
                    ) from rte
                effective_bs = _next_fallback_bs(bs)
                short = str(rte).splitlines()[0][:180]
                print(
                    f"[WARN] QLoRA eval CUDA failure: batch={bs}, "
                    f"fallback_batch={effective_bs}, err={short}"
                )
    total_time = time.monotonic() - t0
    if n > 0:
        print(f"[TIMING] inference done: {n} rows in {total_time:.1f}s ({n / total_time:.1f} rows/s)")
    merged = np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.float32)
    return merged, int(effective_bs)


def _ndcg_from_rank(rank_1_based: int) -> float:
    return float(1.0 / math.log2(rank_1_based + 1.0))


def evaluate_topk(pdf: pd.DataFrame, score_col: str, top_k: int) -> tuple[float, float]:
    hits: list[float] = []
    ndcgs: list[float] = []
    for _, g in pdf.groupby("user_idx", sort=False):
        s = g.sort_values(score_col, ascending=False)
        s = s.head(int(top_k))
        pos = np.where(s["label_true"].to_numpy(dtype=np.int32) == 1)[0]
        if len(pos) == 0:
            hits.append(0.0)
            ndcgs.append(0.0)
            continue
        rank = int(pos[0]) + 1
        hits.append(1.0)
        ndcgs.append(_ndcg_from_rank(rank))
    if not hits:
        return 0.0, 0.0
    return float(np.mean(hits)), float(np.mean(ndcgs))


def attach_truth_labels(cand: DataFrame, truth: DataFrame) -> DataFrame:
    return cand.join(
        truth.select("user_idx", F.col("true_item_idx").alias("label_item_idx")),
        on="user_idx",
        how="inner",
    ).withColumn("label_true", F.when(F.col("item_idx") == F.col("label_item_idx"), F.lit(1)).otherwise(F.lit(0)))


def normalize_pre_score(pdf: pd.DataFrame) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        a = s.min()
        b = s.max()
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
        return (s - a) / (b - a + 1e-9)

    return pdf.groupby("user_idx", sort=False)["pre_score"].transform(_norm).astype(np.float32)


def main() -> None:
    stage09_run = resolve_stage09_run()
    stage11_2_run = resolve_stage11_2_run()
    if int(MAX_USERS_PER_BUCKET) > 800:
        print(
            f"[WARN] QLORA_EVAL_MAX_USERS_PER_BUCKET={int(MAX_USERS_PER_BUCKET)} is large for local runs; "
            "prefer <=600 for faster iteration."
        )
    if int(MAX_ROWS_PER_BUCKET) > 120000:
        print(
            f"[WARN] QLORA_EVAL_MAX_ROWS_PER_BUCKET={int(MAX_ROWS_PER_BUCKET)} may lead to long eval time; "
            "prefer <=80000 for debug cycles."
        )
    if int(MAX_ROWS_PER_BUCKET) > 0 and ROW_CAP_ORDERED:
        print("[WARN] QLORA_EVAL_ROW_CAP_ORDERED=true uses global orderBy before limit and can be slower.")
    if int(MAX_SEQ_LEN) > 512:
        print(
            f"[WARN] QLORA_EVAL_MAX_SEQ_LEN={int(MAX_SEQ_LEN)} increases latency and OOM risk; "
            "prefer 512 unless validated."
        )
    adapter_dir = stage11_2_run / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")

    stage11_meta = json.loads((stage11_2_run / "run_meta.json").read_text(encoding="utf-8"))
    base_model = str(stage11_meta.get("base_model", "")).strip()
    if not base_model:
        raise RuntimeError(f"base_model missing in {stage11_2_run / 'run_meta.json'}")
    stage11_data_run = resolve_stage11_data_run(stage11_meta)
    print(
        f"[CONFIG] source_stage11_2={stage11_2_run} "
        f"source_stage11_data={stage11_data_run if stage11_data_run is not None else '<none>'} "
        f"use_eval_split={USE_STAGE11_EVAL_SPLIT} invert_prob={INVERT_PROB}"
    )

    stage09_meta = json.loads((stage09_run / "run_meta.json").read_text(encoding="utf-8"))
    buckets = parse_bucket_override(BUCKETS_OVERRIDE) or [10]
    gate_result = enforce_stage09_gate(stage09_run, buckets)
    spark: SparkSession | None = None
    user_profile_df: DataFrame | None = None
    item_sem_df: DataFrame | None = None
    cluster_profile_df: DataFrame | None = None

    def ensure_spark_runtime() -> tuple[SparkSession, DataFrame, DataFrame, DataFrame]:
        nonlocal spark, user_profile_df, item_sem_df, cluster_profile_df
        if spark is None:
            spark = build_spark()
            spark.sparkContext.setLogLevel("WARN")
            user_profile_df = build_user_profile_df(spark, stage09_meta)
            item_sem_df = build_item_sem_df(spark, stage09_meta)
            cluster_profile_df = build_cluster_profile_df(spark, stage09_meta)
            print(f"[CONFIG] PROMPT_MODE={PROMPT_MODE} (driver-side prompt generation, no Spark UDF)")
        return spark, user_profile_df, item_sem_df, cluster_profile_df


    has_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if (has_cuda and USE_BF16 and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = None
    if has_cuda and USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": TRUST_REMOTE_CODE,
        "low_cpu_mem_usage": True,
    }
    if has_cuda:
        model_kwargs["device_map"] = EVAL_DEVICE_MAP
        max_memory: dict[Any, str] = {}
        if EVAL_MAX_MEMORY_CUDA:
            max_memory[0] = str(EVAL_MAX_MEMORY_CUDA)
        if EVAL_MAX_MEMORY_CPU:
            max_memory["cpu"] = str(EVAL_MAX_MEMORY_CPU)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if EVAL_OFFLOAD_FOLDER:
            offload_dir = Path(EVAL_OFFLOAD_FOLDER).expanduser()
            offload_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = offload_dir.as_posix()
            if EVAL_OFFLOAD_STATE_DICT:
                # Keep final placement unchanged while spilling transient load-time state dict tensors.
                model_kwargs["offload_state_dict"] = True
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = compute_dtype

    # Align qwen3.5 eval with train-time dtype path to avoid float32-heavy load/memory spikes.
    try:
        model_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=TRUST_REMOTE_CODE)
        model_type = str(getattr(model_cfg, "model_type", "")).strip().lower()
        if is_qwen35_model_type(model_type):
            target_mamba_dtype = str(QWEN35_MAMBA_SSM_DTYPE or "auto").strip().lower()
            if target_mamba_dtype in {"", "auto"} and compute_dtype == torch.bfloat16:
                target_mamba_dtype = "bfloat16"
            if target_mamba_dtype not in {"", "auto"}:
                txt_cfg = getattr(model_cfg, "text_config", None)
                if isinstance(txt_cfg, dict):
                    txt_cfg["mamba_ssm_dtype"] = target_mamba_dtype
                elif txt_cfg is not None:
                    setattr(txt_cfg, "mamba_ssm_dtype", target_mamba_dtype)
                model_kwargs["config"] = model_cfg
                print(f"[CONFIG] qwen3.5 mamba_ssm_dtype={target_mamba_dtype}")
    except Exception as cfg_exc:
        short = str(cfg_exc).splitlines()[0][:180]
        print(f"[WARN] qwen3.5 dtype config override skipped: {short}")

    if DISABLE_ALLOC_WARMUP and hasattr(_tf_modeling_utils, "caching_allocator_warmup"):
        _tf_modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None  # type: ignore[assignment]
        print("[CONFIG] disable_transformers_alloc_warmup=true")
    if DISABLE_PARALLEL_LOADING:
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(max(1, int(PARALLEL_LOADING_WORKERS)))
    print(
        f"[CONFIG] model_load device_map={EVAL_DEVICE_MAP} "
        f"max_memory_cuda={EVAL_MAX_MEMORY_CUDA or '<none>'} "
        f"max_memory_cpu={EVAL_MAX_MEMORY_CPU or '<none>'} "
        f"offload_folder={EVAL_OFFLOAD_FOLDER or '<none>'} "
        f"offload_state_dict={bool(model_kwargs.get('offload_state_dict', False))} "
        f"disable_parallel_loading={DISABLE_PARALLEL_LOADING}"
    )
    print(f"[CONFIG] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    print(f"[CONFIG] qwen35_no_think={QWEN35_NO_THINK}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "left"

    if has_cuda and LOAD_MODEL_BEFORE_SPARK:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except OSError as exc:
        msg = str(exc).lower()
        if "1455" in msg or "paging file" in msg:
            raise RuntimeError(
                "model load failed with Windows os error 1455 (virtual memory/pagefile too small). "
                "Increase pagefile size (recommend >= 40GB), reboot to apply, then rerun stage11_3."
            ) from exc
        raise
    model = PeftModel.from_pretrained(base, adapter_dir.as_posix())
    # This eval path only needs final-token logits, not generation KV cache.
    try:
        if getattr(getattr(model, "config", None), "use_cache", None):
            model.config.use_cache = False
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    yes_id, no_id = build_yes_no_token_ids(tokenizer)

    resume_target = os.getenv("QLORA_EVAL_RESUME_DIR", "").strip()
    if resume_target:
        out_dir = Path(resume_target)
        if not out_dir.exists():
            raise FileNotFoundError(f"QLORA_EVAL_RESUME_DIR not found: {out_dir}")
        run_id = out_dir.name.replace(f"_{RUN_TAG}", "")
        print(f"[RESUME] Resuming eval in {out_dir}")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
        out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    for bucket in buckets:
        final_scores_path = out_dir / f"bucket_{bucket}_scores.csv"
        if final_scores_path.exists() and os.getenv("QLORA_EVAL_RESUME_DIR", "").strip():
            print(f"[RESUME] skip bucket={bucket} inference, final scores already exist. Recalculating metrics.")
            pdf = pd.read_csv(final_scores_path.as_posix())
            pre_recall, pre_ndcg = evaluate_topk(pdf, "pre_score", TOP_K)
            q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)
            n_users = int(pdf["user_idx"].nunique())
            n_items = int(pdf["item_idx"].nunique())
            n_rows = int(len(pdf))

            metrics_rows.append(
                {
                    "run_id_11": run_id,
                    "source_run_09": str(stage09_run),
                    "source_run_11_2": str(stage11_2_run),
                    "bucket_min_train_reviews": int(bucket),
                    "model": "PreScore@10",
                    "recall_at_k": float(pre_recall),
                    "ndcg_at_k": float(pre_ndcg),
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_candidates": n_rows,
                }
            )
            metrics_rows.append(
                {
                    "run_id_11": run_id,
                    "source_run_09": str(stage09_run),
                    "source_run_11_2": str(stage11_2_run),
                    "bucket_min_train_reviews": int(bucket),
                    "model": "QLoRASidecar@10",
                    "recall_at_k": float(q_recall),
                    "ndcg_at_k": float(q_ndcg),
                    "n_users": n_users,
                    "n_items": n_items,
                    "n_candidates": n_rows,
                }
            )
            continue

        bdir = stage09_run / f"bucket_{bucket}"
        truth_path = bdir / "truth.parquet"
        if not truth_path.exists():
            print(f"[WARN] skip bucket={bucket}: missing truth.parquet")
            continue
        cand_path = pick_candidate_file(bdir)
        spark, user_profile_df, item_sem_df, cluster_profile_df = ensure_spark_runtime()

        truth = (
            spark.read.parquet(truth_path.as_posix())
            .select("user_idx", "user_id", "true_item_idx")
            .dropDuplicates(["user_idx"])
        )
        users = truth.select("user_idx").dropDuplicates(["user_idx"])
        if USE_STAGE11_EVAL_SPLIT:
            if stage11_data_run is None:
                raise RuntimeError("USE_STAGE11_EVAL_SPLIT=true but stage11_data_run is not resolved")
            eval_users = load_eval_users_from_stage11_data(spark, stage11_data_run, int(bucket))
            users = users.join(eval_users, on="user_idx", how="inner")
            truth = truth.join(users, on="user_idx", how="inner")
        if MAX_USERS_PER_BUCKET > 0:
            users = users.orderBy(F.rand(int(RANDOM_SEED + bucket))).limit(int(MAX_USERS_PER_BUCKET))
            truth = truth.join(users, on="user_idx", how="inner")

        cand_raw = (
            spark.read.parquet(cand_path.as_posix())
            .join(users, on="user_idx", how="inner")
            .join(truth.select("user_idx", "user_id"), on="user_idx", how="left")
            .filter(F.col("pre_rank") <= F.lit(int(RERANK_TOPN)))
        )
        cand_cols = set(cand_raw.columns)

        def _num_col(name: str) -> Any:
            if name in cand_cols:
                return F.col(name).cast("double").alias(name)
            return F.lit(None).cast("double").alias(name)

        source_set_text = (
            F.when(F.col("source_set").isNull(), F.lit(""))
            .otherwise(F.concat_ws("|", F.col("source_set")))
            .alias("source_set_text")
            if "source_set" in cand_cols
            else F.lit("").alias("source_set_text")
        )
        user_segment_col = (
            F.coalesce(F.col("user_segment"), F.lit("")).alias("user_segment")
            if "user_segment" in cand_cols
            else F.lit("").alias("user_segment")
        )

        cand = (
            cand_raw.select(
                "user_idx",
                "user_id",
                "item_idx",
                "business_id",
                "pre_rank",
                "pre_score",
                "name",
                "city",
                "categories",
                "primary_category",
                source_set_text,
                user_segment_col,
                _num_col("als_rank"),
                _num_col("cluster_rank"),
                _num_col("profile_rank"),
                _num_col("popular_rank"),
                _num_col("semantic_support"),
                _num_col("semantic_tag_richness"),
                _num_col("tower_score"),
                _num_col("seq_score"),
            )
            .join(cluster_profile_df, on="business_id", how="left")
        )
        cand = attach_truth_labels(cand, truth)
        if MAX_ROWS_PER_BUCKET > 0:
            if ROW_CAP_ORDERED:
                cand = cand.orderBy(F.col("user_idx").asc(), F.col("pre_rank").asc()).limit(int(MAX_ROWS_PER_BUCKET))
            else:
                cand = cand.limit(int(MAX_ROWS_PER_BUCKET))

        user_review_df, item_review_df = build_temporal_review_features(
            spark=spark,
            bucket_dir=bdir,
            truth_df=truth.select("user_idx", "user_id"),
            cand_df=cand.select("user_idx", "business_id"),
        )
        cand = (
            cand.join(user_profile_df, on="user_id", how="left")
            .join(item_sem_df, on="business_id", how="left")
            .join(user_review_df, on="user_idx", how="left")
            .join(item_review_df, on=["user_idx", "business_id"], how="left")
            .fillna(
                {
                    "profile_text": "",
                    "profile_top_pos_tags": "",
                    "profile_top_neg_tags": "",
                    "profile_confidence": 0.0,
                    "top_pos_tags": "",
                    "top_neg_tags": "",
                    "semantic_score": 0.0,
                    "semantic_confidence": 0.0,
                    "source_set_text": "",
                    "user_segment": "",
                    "als_rank": 0.0,
                    "cluster_rank": 0.0,
                    "profile_rank": 0.0,
                    "popular_rank": 0.0,
                    "semantic_support": 0.0,
                    "semantic_tag_richness": 0.0,
                    "tower_score": 0.0,
                    "seq_score": 0.0,
                    "cluster_for_recsys": "",
                    "cluster_label_for_recsys": "",
                    "user_review_summary": "",
                    "user_review_raw": "",
                    "item_review_summary": "",
                    "item_review_raw": "",
                }
            )
        )

        partial_scores_path = out_dir / f"bucket_{bucket}_scores_partial.csv"
        flushed_rows = 0
        scored_pairs_df = None

        if partial_scores_path.exists() and os.getenv("QLORA_EVAL_RESUME_DIR", "").strip():
            print(f"[RESUME] Loading partial scores from {partial_scores_path}")
            try:
                partial_df = pd.read_csv(partial_scores_path.as_posix(), usecols=["user_idx", "item_idx"]).dropna()
                if not partial_df.empty:
                    flushed_rows = len(partial_df)
                    scored_pairs_df = spark.createDataFrame(
                        partial_df[["user_idx", "item_idx"]].astype(int)
                    )
            except Exception as e:
                print(f"[WARN] Failed to read partial scores: {e}. Will overwrite.")
                partial_scores_path.unlink()
        elif partial_scores_path.exists():
            partial_scores_path.unlink()

        pending_user_idx: list[int] = []
        pending_item_idx: list[int] = []
        pending_pre_rank: list[int] = []
        pending_label_true: list[int] = []
        pending_pre_score: list[float] = []
        pending_qlora_prob: list[float] = []

        prompt_buf: list[str] = []
        user_idx_buf: list[int] = []
        item_idx_buf: list[int] = []
        pre_rank_buf: list[int] = []
        label_true_buf: list[int] = []
        pre_score_buf: list[float] = []
        adaptive_batch_size = max(1, int(INFER_BATCH_SIZE))

        def _append_partial_scores() -> None:
            nonlocal flushed_rows
            if not pending_user_idx:
                return
            chunk = pd.DataFrame(
                {
                    "user_idx": np.asarray(pending_user_idx, dtype=np.int64),
                    "item_idx": np.asarray(pending_item_idx, dtype=np.int64),
                    "pre_rank": np.asarray(pending_pre_rank, dtype=np.int32),
                    "label_true": np.asarray(pending_label_true, dtype=np.int8),
                    "pre_score": np.asarray(pending_pre_score, dtype=np.float32),
                    "qlora_prob": np.asarray(pending_qlora_prob, dtype=np.float32),
                }
            )
            write_header = not partial_scores_path.exists()
            chunk.to_csv(partial_scores_path.as_posix(), mode="a", header=write_header, index=False)
            flushed_rows += int(len(chunk))
            pending_user_idx.clear()
            pending_item_idx.clear()
            pending_pre_rank.clear()
            pending_label_true.clear()
            pending_pre_score.clear()
            pending_qlora_prob.clear()

        def _flush_prompt_buf() -> None:
            nonlocal adaptive_batch_size
            if not prompt_buf:
                return
            prev_bs = int(adaptive_batch_size)
            q_prob_local, adaptive_batch_size = score_yes_probability(
                model=model,
                tokenizer=tokenizer,
                prompts=prompt_buf,
                batch_size=adaptive_batch_size,
                max_seq_len=MAX_SEQ_LEN,
                yes_id=yes_id,
                no_id=no_id,
            )
            q_prob_local = q_prob_local.astype(np.float32)
            if int(adaptive_batch_size) != int(prev_bs):
                print(
                    f"[CONFIG] adaptive_eval_batch_size update: "
                    f"{int(prev_bs)} -> {int(adaptive_batch_size)}"
                )
            if INVERT_PROB:
                q_prob_local = np.clip(1.0 - q_prob_local, 0.0, 1.0)
            if len(q_prob_local) != len(user_idx_buf):
                raise RuntimeError(
                    "qlora score size mismatch: "
                    f"probs={len(q_prob_local)} rows={len(user_idx_buf)}"
                )
            pending_user_idx.extend(user_idx_buf)
            pending_item_idx.extend(item_idx_buf)
            pending_pre_rank.extend(pre_rank_buf)
            pending_label_true.extend(label_true_buf)
            pending_pre_score.extend(pre_score_buf)
            pending_qlora_prob.extend([float(x) for x in q_prob_local.tolist()])
            if len(pending_user_idx) >= int(max(1, INTERMEDIATE_FLUSH_ROWS)):
                _append_partial_scores()
            prompt_buf.clear()
            user_idx_buf.clear()
            item_idx_buf.clear()
            pre_rank_buf.clear()
            label_true_buf.clear()
            pre_score_buf.clear()

        t_spark_start = time.monotonic()
        # ------------------------------------------------------------------
        # Driver-side collection: avoid Spark Python worker UDFs entirely.
        # Collect all needed columns to pandas, build prompts on the driver.
        # ------------------------------------------------------------------
        _collect_cols = [
            "user_idx", "item_idx", "pre_rank", "label_true", "pre_score",
            "profile_text", "profile_top_pos_tags", "profile_top_neg_tags",
            "profile_confidence", "user_review_summary", "user_review_raw",
            "name", "city", "categories", "primary_category",
            "top_pos_tags", "top_neg_tags", "semantic_score", "semantic_confidence",
            "source_set_text", "user_segment",
            "als_rank", "cluster_rank", "profile_rank", "popular_rank",
            "semantic_support", "semantic_tag_richness", "tower_score", "seq_score",
            "cluster_for_recsys", "cluster_label_for_recsys",
            "item_review_summary", "item_review_raw",
        ]
        cand_for_collect = cand.select(*_collect_cols)

        if scored_pairs_df is not None:
            cand_for_collect = cand_for_collect.join(
                F.broadcast(scored_pairs_df),
                on=["user_idx", "item_idx"],
                how="left_anti"
            )
            print(f"[RESUME] Skipping already evaluated rows. Pre-computed count = {flushed_rows}")

        print("[COLLECT] Collecting candidate data to driver (toPandas) ...")
        cand_pdf = cand_for_collect.toPandas()
        t_collect = time.monotonic() - t_spark_start
        print(f"[TIMING] toPandas done: {len(cand_pdf)} rows in {t_collect:.1f}s")

        # Build prompts on driver side (pure string ops, no Spark workers)
        def _build_prompt_for_row(r: pd.Series) -> str:
            user_text = build_user_text(
                r.get("profile_text", ""),
                r.get("profile_top_pos_tags", ""),
                r.get("profile_top_neg_tags", ""),
                r.get("profile_confidence", 0.0),
                review_summary=r.get("user_review_summary", ""),
                review_raw_snippet=r.get("user_review_raw", ""),
            )
            if PROMPT_MODE == "semantic":
                item_text = build_item_text_semantic(
                    r.get("name", ""),
                    r.get("city", ""),
                    r.get("categories", ""),
                    r.get("primary_category", ""),
                    r.get("top_pos_tags", ""),
                    r.get("top_neg_tags", ""),
                    r.get("semantic_score", 0.0),
                    r.get("semantic_confidence", 0.0),
                    semantic_support=r.get("semantic_support", 0.0),
                    semantic_tag_richness=r.get("semantic_tag_richness", 0.0),
                    tower_score=r.get("tower_score", 0.0),
                    seq_score=r.get("seq_score", 0.0),
                    cluster_for_recsys=r.get("cluster_for_recsys", ""),
                    cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                    item_review_summary=r.get("item_review_summary", ""),
                    item_review_snippet=r.get("item_review_raw", ""),
                )
                return build_binary_prompt_semantic(user_text, item_text)
            else:
                item_text = build_item_text(
                    r.get("name", ""),
                    r.get("city", ""),
                    r.get("categories", ""),
                    r.get("primary_category", ""),
                    r.get("top_pos_tags", ""),
                    r.get("top_neg_tags", ""),
                    r.get("semantic_score", 0.0),
                    r.get("semantic_confidence", 0.0),
                    source_set=r.get("source_set_text", ""),
                    user_segment=r.get("user_segment", ""),
                    als_rank=r.get("als_rank", 0.0),
                    cluster_rank=r.get("cluster_rank", 0.0),
                    profile_rank=r.get("profile_rank", 0.0),
                    popular_rank=r.get("popular_rank", 0.0),
                    semantic_support=r.get("semantic_support", 0.0),
                    semantic_tag_richness=r.get("semantic_tag_richness", 0.0),
                    tower_score=r.get("tower_score", 0.0),
                    seq_score=r.get("seq_score", 0.0),
                    cluster_for_recsys=r.get("cluster_for_recsys", ""),
                    cluster_label_for_recsys=r.get("cluster_label_for_recsys", ""),
                    item_review_summary=r.get("item_review_summary", ""),
                    item_review_snippet=r.get("item_review_raw", ""),
                )
                return build_binary_prompt(user_text, item_text)

        t_prompt_start = time.monotonic()
        cand_pdf["prompt"] = cand_pdf.apply(_build_prompt_for_row, axis=1)
        t_prompt = time.monotonic() - t_prompt_start
        print(f"[TIMING] driver-side prompt generation done: {len(cand_pdf)} rows in {t_prompt:.1f}s")

        # Iterate over driver-side pandas DataFrame in chunks
        iter_row_count = 0
        for idx in range(len(cand_pdf)):
            row = cand_pdf.iloc[idx]
            prompt_buf.append(_str_or_empty(row["prompt"]))
            user_idx_buf.append(int(row["user_idx"]))
            item_idx_buf.append(int(row["item_idx"]))
            pre_rank_buf.append(int(row["pre_rank"]))
            label_true_buf.append(int(row["label_true"]))
            pre_score_buf.append(_float_or_zero(row["pre_score"]))
            iter_row_count += 1
            if len(prompt_buf) >= int(PROMPT_CHUNK_ROWS):
                print(
                    f"[TIMING] prompt chunk ready: {iter_row_count}/{len(cand_pdf)} rows"
                )
                _flush_prompt_buf()

        t_spark_total = time.monotonic() - t_spark_start
        print(f"[TIMING] Spark iter done: {iter_row_count} total rows in {t_spark_total:.1f}s")
        _flush_prompt_buf()
        _append_partial_scores()
        if int(flushed_rows) <= 0 or not partial_scores_path.exists():
            print(f"[WARN] skip bucket={bucket}: empty candidate after filters")
            continue
        pdf = pd.read_csv(partial_scores_path.as_posix())
        if pdf.empty:
            print(f"[WARN] skip bucket={bucket}: empty partial scores")
            continue
        pdf["pre_norm"] = normalize_pre_score(pdf)
        pdf["blend_score"] = (1.0 - float(BLEND_ALPHA)) * pdf["pre_norm"] + float(BLEND_ALPHA) * pdf["qlora_prob"]

        pre_recall, pre_ndcg = evaluate_topk(pdf, "pre_score", TOP_K)
        q_recall, q_ndcg = evaluate_topk(pdf, "blend_score", TOP_K)

        n_users = int(pdf["user_idx"].nunique())
        n_items = int(pdf["item_idx"].nunique())
        n_rows = int(len(pdf))

        metrics_rows.append(
            {
                "run_id_11": run_id,
                "source_run_09": str(stage09_run),
                "source_run_11_2": str(stage11_2_run),
                "bucket_min_train_reviews": int(bucket),
                "model": "PreScore@10",
                "recall_at_k": float(pre_recall),
                "ndcg_at_k": float(pre_ndcg),
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_rows,
            }
        )
        metrics_rows.append(
            {
                "run_id_11": run_id,
                "source_run_09": str(stage09_run),
                "source_run_11_2": str(stage11_2_run),
                "bucket_min_train_reviews": int(bucket),
                "model": "QLoRASidecar@10",
                "recall_at_k": float(q_recall),
                "ndcg_at_k": float(q_ndcg),
                "n_users": n_users,
                "n_items": n_items,
                "n_candidates": n_rows,
            }
        )
        print(
            f"[METRIC] bucket={bucket} pre_ndcg={pre_ndcg:.6f} qlora_ndcg={q_ndcg:.6f} "
            f"pre_recall={pre_recall:.6f} qlora_recall={q_recall:.6f}"
        )

        pd.DataFrame(
            {
                "user_idx": pdf["user_idx"],
                "item_idx": pdf["item_idx"],
                "pre_rank": pdf["pre_rank"],
                "label_true": pdf["label_true"],
                "pre_score": pdf["pre_score"],
                "qlora_prob": pdf["qlora_prob"],
                "blend_score": pdf["blend_score"],
            }
        ).to_csv((out_dir / f"bucket_{bucket}_scores.csv").as_posix(), index=False)
        print(f"[CHECKPOINT] bucket={bucket} flushed_rows={int(flushed_rows)} partial_file={partial_scores_path}")

    if not metrics_rows:
        raise RuntimeError("no bucket metrics produced")

    df_metrics = pd.DataFrame(metrics_rows)
    out_csv = out_dir / "qlora_sidecar_metrics.csv"
    df_metrics.to_csv(out_csv.as_posix(), index=False)

    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        merged = pd.concat([old, df_metrics], ignore_index=True)
        merged.to_csv(METRICS_PATH, index=False)
    else:
        df_metrics.to_csv(METRICS_PATH, index=False)

    meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_run_09": str(stage09_run),
        "source_run_11_2": str(stage11_2_run),
        "source_run_11_1_data": str(stage11_data_run) if stage11_data_run is not None else "",
        "enforce_stage09_gate": bool(ENFORCE_STAGE09_GATE),
        "stage09_gate_result": gate_result,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "buckets": buckets,
        "top_k": int(TOP_K),
        "rerank_topn": int(RERANK_TOPN),
        "blend_alpha": float(BLEND_ALPHA),
        "invert_prob": bool(INVERT_PROB),
        "use_stage11_eval_split": bool(USE_STAGE11_EVAL_SPLIT),
        "structured_feature_prompt_enabled": True,
        "short_text_summary_enabled": bool(ENABLE_SHORT_TEXT_SUMMARY),
        "raw_review_text_enabled": bool(ENABLE_RAW_REVIEW_TEXT),
        "review_snippet_max_chars": int(REVIEW_SNIPPET_MAX_CHARS),
        "max_users_per_bucket": int(MAX_USERS_PER_BUCKET),
        "max_rows_per_bucket": int(MAX_ROWS_PER_BUCKET),
        "row_cap_ordered": bool(ROW_CAP_ORDERED),
        "prompt_chunk_rows": int(PROMPT_CHUNK_ROWS),
        "iter_coalesce_partitions": int(ITER_COALESCE_PARTITIONS),
        "intermediate_flush_rows": int(INTERMEDIATE_FLUSH_ROWS),
        "metrics_file": str(out_csv),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    if spark is not None:
        spark.stop()
    print(f"[DONE] qlora sidecar metrics: {out_csv}")
    print(f"[DONE] merged metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()


