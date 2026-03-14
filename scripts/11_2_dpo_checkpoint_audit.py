from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.project_paths import env_or_project_path

RUN_TAG = os.getenv("RUN_TAG", "stage11_2_dpo_checkpoint_audit").strip() or "stage11_2_dpo_checkpoint_audit"
INPUT_11_2_RUN_DIR = Path(os.getenv("INPUT_11_2_RUN_DIR", "").strip())
OUTPUT_ROOT = env_or_project_path("OUTPUT_11_DPO_CKPT_AUDIT_ROOT_DIR", "data/output/11_qlora_dpo_checkpoint_audit")
EVAL_SCRIPT = env_or_project_path("QLORA_DPO_CKPT_EVAL_SCRIPT", "scripts/11_3_qlora_sidecar_eval.py")
METRICS_PATH = env_or_project_path(
    "METRICS_STAGE11_DPO_CKPT_AUDIT_PATH",
    "data/metrics/recsys_stage11_dpo_checkpoint_audit_results.csv",
)
CHECKPOINTS_OVERRIDE = os.getenv("QLORA_DPO_CKPT_NAMES", "").strip()
ALPHA_GRID_START = float(os.getenv("QLORA_DPO_ALPHA_SWEEP_START", "0.0").strip() or 0.0)
ALPHA_GRID_END = float(os.getenv("QLORA_DPO_ALPHA_SWEEP_END", "0.60").strip() or 0.60)
ALPHA_GRID_STEP = float(os.getenv("QLORA_DPO_ALPHA_SWEEP_STEP", "0.02").strip() or 0.02)
TARGET_ALPHA = float(os.getenv("QLORA_DPO_ALPHA_TARGET", "0.42").strip() or 0.42)
TOP_K = int(os.getenv("RANK_EVAL_TOP_K", "10").strip() or 10)
KEEP_RUNS = os.getenv("QLORA_DPO_CKPT_AUDIT_KEEP_RUNS", "true").strip().lower() == "true"


def _ndcg_from_rank(rank: int) -> float:
    return 1.0 / np.log2(rank + 1.0)


def normalize_pre_score(pdf: pd.DataFrame) -> pd.Series:
    def _norm(s: pd.Series) -> pd.Series:
        a = s.min()
        b = s.max()
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
        return (s - a) / (b - a + 1e-9)

    return pdf.groupby("user_idx", sort=False)["pre_score"].transform(_norm).astype(np.float32)


def evaluate_topk(pdf: pd.DataFrame, score_col: str, top_k: int) -> tuple[float, float]:
    hits: list[float] = []
    ndcgs: list[float] = []
    for _, g in pdf.groupby("user_idx", sort=False):
        s = g.sort_values(score_col, ascending=False).head(int(top_k))
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


def alpha_grid() -> list[float]:
    steps = int(round((ALPHA_GRID_END - ALPHA_GRID_START) / ALPHA_GRID_STEP))
    values = [ALPHA_GRID_START + idx * ALPHA_GRID_STEP for idx in range(steps + 1)]
    return [round(v, 6) for v in values]


def collect_checkpoints(run_dir: Path) -> list[str]:
    if CHECKPOINTS_OVERRIDE:
        out = []
        for raw in CHECKPOINTS_OVERRIDE.split(","):
            name = raw.strip()
            if name:
                out.append(name)
        if "final" not in out:
            out.append("final")
        return out

    trainer_output = run_dir / "trainer_output"
    ckpts = []
    for path in sorted(trainer_output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        ckpts.append(path.name)
    ckpts.append("final")
    return ckpts


def wrap_checkpoint(root: Path, source_run_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name == "final":
        return source_run_dir
    ckpt_dir = source_run_dir / "trainer_output" / checkpoint_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")
    out = root / f"evalrun_{checkpoint_name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_run_dir / "run_meta.json", out / "run_meta.json")
    os.symlink(ckpt_dir.as_posix(), (out / "adapter").as_posix(), target_is_directory=True)
    return out


def build_eval_env(input_run_dir: Path, audit_run_root: Path, checkpoint_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["INPUT_11_2_RUN_DIR"] = input_run_dir.as_posix()
    env["OUTPUT_11_SIDECAR_ROOT_DIR"] = (audit_run_root / "runs").as_posix()
    env["METRICS_STAGE11_SIDECAR_PATH"] = (audit_run_root / "recsys_stage11_qlora_sidecar_results.csv").as_posix()
    env.setdefault("PYSPARK_PYTHON", "/root/miniconda3/bin/python")
    env.setdefault("PYSPARK_DRIVER_PYTHON", "/root/miniconda3/bin/python")
    env.setdefault("PYTHONPATH", "/root/5006_BDA_project/scripts:/root/5006_BDA_project")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("TMP", "/root/autodl-tmp/tmp")
    env.setdefault("TEMP", "/root/autodl-tmp/tmp")
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env["SPARK_LOCAL_DIR"] = "/root/autodl-tmp/data/spark-tmp"
    env.setdefault("QLORA_PROMPT_MODE", "full_lite")
    env.setdefault("QLORA_INVERT_PROB", "false")
    env.setdefault("QLORA_ENFORCE_STAGE09_GATE", "false")
    env.setdefault("QLORA_EVAL_PROFILE", "smoke")
    env.setdefault("QLORA_EVAL_USE_STAGE11_SPLIT", "true")
    env.setdefault("QLORA_RERANK_TOPN", "150")
    env.setdefault("QLORA_EVAL_MAX_USERS_PER_BUCKET", "80")
    env.setdefault("QLORA_EVAL_MAX_ROWS_PER_BUCKET", "0")
    env.setdefault("QLORA_EVAL_BATCH_SIZE", "60")
    env.setdefault("QLORA_EVAL_MAX_SEQ_LEN", "448")
    env.setdefault("QLORA_EVAL_ATTN_IMPLEMENTATION", "sdpa")
    env.setdefault("QLORA_EVAL_PAD_TO_MULTIPLE_OF", "64")
    env.setdefault("QLORA_EVAL_PROMPT_CHUNK_ROWS", "4096")
    env.setdefault("QLORA_EVAL_STREAM_LOG_ROWS", "2048")
    env.setdefault("QLORA_EVAL_ITER_COALESCE", "8")
    env.setdefault("QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS", "4096")
    env.setdefault("QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK", "true")
    env.setdefault("QLORA_EVAL_PIN_MEMORY", "true")
    env.setdefault("QLORA_EVAL_NON_BLOCKING_H2D", "true")
    env.setdefault("QLORA_EVAL_DRIVER_PROMPT_IMPL", "itertuples")
    env.setdefault("QLORA_EVAL_ARROW_TO_PANDAS", "true")
    env.setdefault("QLORA_EVAL_ARROW_FALLBACK", "false")
    env.setdefault("QLORA_ENABLE_RAW_REVIEW_TEXT", "true")
    env.setdefault("QLORA_REVIEW_TABLE_PATH", "/root/autodl-tmp/project_data/data/parquet/yelp_academic_dataset_review")
    env.setdefault("QLORA_ITEM_EVIDENCE_SCORE_UDF_MODE", "pandas")
    env.setdefault("QLORA_EVAL_REVIEW_BASE_CACHE_ENABLED", "true")
    env.setdefault("QLORA_EVAL_REVIEW_BASE_CACHE_ROOT", "/root/autodl-tmp/stage11_fs/tmp/review_base_cache")
    env.setdefault("QLORA_EVAL_QWEN35_NO_THINK", "false")
    env.setdefault("SPARK_MASTER", "local[32]")
    env.setdefault("SPARK_DRIVER_MEMORY", "48g")
    env.setdefault("SPARK_EXECUTOR_MEMORY", "48g")
    env.setdefault("SPARK_SQL_SHUFFLE_PARTITIONS", "128")
    env.setdefault("SPARK_DEFAULT_PARALLELISM", "64")
    env.setdefault("SPARK_NETWORK_TIMEOUT", "1200s")
    env.setdefault("SPARK_EXECUTOR_HEARTBEAT_INTERVAL", "120s")
    env.setdefault("SPARK_PYTHON_WORKER_MEMORY", "4g")
    env["RUN_TAG"] = f"{RUN_TAG}_{checkpoint_name}"
    return env


def latest_eval_dir(runs_root: Path, before: set[str]) -> Path:
    candidates = sorted(
        [p for p in runs_root.glob("*_stage11_3_qlora_sidecar_eval") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    for path in reversed(candidates):
        if path.name not in before:
            return path
    if not candidates:
        raise FileNotFoundError(f"no eval runs found in {runs_root}")
    return candidates[-1]


def run_checkpoint_eval(root: Path, checkpoint_name: str, input_run_dir: Path) -> Path:
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in runs_root.glob("*_stage11_3_qlora_sidecar_eval")}
    env = build_eval_env(input_run_dir, root, checkpoint_name)
    log_path = root / f"{checkpoint_name}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [os.sys.executable, EVAL_SCRIPT.as_posix()],
            check=True,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return latest_eval_dir(runs_root, before)


def summarize_scores(scores_csv: Path) -> dict[str, float | int]:
    pdf = pd.read_csv(scores_csv)
    pdf["pre_norm"] = normalize_pre_score(pdf)
    rows = []
    for alpha in alpha_grid():
        pdf["blend_score"] = (1.0 - float(alpha)) * pdf["pre_norm"] + float(alpha) * pdf["qlora_prob"]
        recall, ndcg = evaluate_topk(pdf, "blend_score", TOP_K)
        rows.append(
            {
                "alpha": float(alpha),
                "recall_at_10": float(recall),
                "ndcg_at_10": float(ndcg),
            }
        )
    best = max(rows, key=lambda r: (r["ndcg_at_10"], r["recall_at_10"], -abs(r["alpha"] - TARGET_ALPHA)))
    return {
        "n_users": int(pdf["user_idx"].nunique()),
        "n_rows": int(len(pdf)),
        "qprob_mean": float(pdf["qlora_prob"].mean()),
        "qprob_std": float(pdf["qlora_prob"].std()),
        "grid": rows,
        "best_alpha": float(best["alpha"]),
        "best_recall_at_10": float(best["recall_at_10"]),
        "best_ndcg_at_10": float(best["ndcg_at_10"]),
    }


def main() -> None:
    if not INPUT_11_2_RUN_DIR.exists():
        raise FileNotFoundError(f"INPUT_11_2_RUN_DIR not found: {INPUT_11_2_RUN_DIR}")
    if not (INPUT_11_2_RUN_DIR / "run_meta.json").exists():
        raise FileNotFoundError(f"run_meta.json not found under: {INPUT_11_2_RUN_DIR}")
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"eval script not found: {EVAL_SCRIPT}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{run_id}_{RUN_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    wrappers_root = out_dir / "wrappers"
    wrappers_root.mkdir(parents=True, exist_ok=True)

    for checkpoint_name in collect_checkpoints(INPUT_11_2_RUN_DIR):
        audit_input = wrap_checkpoint(wrappers_root, INPUT_11_2_RUN_DIR, checkpoint_name)
        eval_out = run_checkpoint_eval(out_dir, checkpoint_name, audit_input)
        summary = summarize_scores(eval_out / "bucket_10_scores.csv")
        result = {
            "checkpoint": checkpoint_name,
            "input_run_dir": audit_input.as_posix(),
            "eval_out_dir": eval_out.as_posix(),
            **summary,
        }
        results.append(result)
        if not KEEP_RUNS:
            shutil.rmtree(eval_out, ignore_errors=True)

    summary_jsonl = out_dir / "checkpoint_alpha_sweep_summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_csv = out_dir / "checkpoint_alpha_sweep_summary.csv"
    pd.DataFrame(
        [
            {
                "checkpoint": row["checkpoint"],
                "input_run_dir": row["input_run_dir"],
                "eval_out_dir": row["eval_out_dir"],
                "n_users": row["n_users"],
                "n_rows": row["n_rows"],
                "qprob_mean": row["qprob_mean"],
                "qprob_std": row["qprob_std"],
                "best_alpha": row["best_alpha"],
                "best_recall_at_10": row["best_recall_at_10"],
                "best_ndcg_at_10": row["best_ndcg_at_10"],
            }
            for row in results
        ]
    ).to_csv(summary_csv.as_posix(), index=False)

    best = max(results, key=lambda r: (r["best_ndcg_at_10"], r["best_recall_at_10"], -abs(r["best_alpha"] - TARGET_ALPHA)))
    best_json = out_dir / "best_checkpoint.json"
    best_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    run_meta = {
        "run_id": run_id,
        "run_tag": RUN_TAG,
        "source_stage11_2_run": INPUT_11_2_RUN_DIR.as_posix(),
        "checkpoints": [row["checkpoint"] for row in results],
        "alpha_grid_start": ALPHA_GRID_START,
        "alpha_grid_end": ALPHA_GRID_END,
        "alpha_grid_step": ALPHA_GRID_STEP,
        "target_alpha": TARGET_ALPHA,
        "top_k": TOP_K,
        "best_checkpoint": best["checkpoint"],
        "best_alpha": best["best_alpha"],
        "best_ndcg_at_10": best["best_ndcg_at_10"],
        "best_recall_at_10": best["best_recall_at_10"],
        "summary_jsonl": summary_jsonl.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "best_checkpoint_json": best_json.as_posix(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        merged = pd.concat([old, pd.read_csv(summary_csv)], ignore_index=True)
        merged.to_csv(METRICS_PATH, index=False)
    else:
        pd.read_csv(summary_csv).to_csv(METRICS_PATH, index=False)

    print(json.dumps(run_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
