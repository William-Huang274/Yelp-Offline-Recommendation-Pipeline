from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"D:/5006 BDA project")
SOURCE_09 = PROJECT_ROOT / r"data/output/09_candidate_fusion/20260215_110013_full_stage09_candidate_fusion"
METRICS_CSV = PROJECT_ROOT / r"data/metrics/recsys_stage10_results.csv"
TRAIN_SCRIPT = PROJECT_ROOT / r"scripts/10_1_rank_train.py"
EVAL_SCRIPT = PROJECT_ROOT / r"scripts/10_2_rank_infer_eval.py"
MODEL_ROOT = PROJECT_ROOT / r"data/output/10_rank_models"
RUN_TAG = "stage10_4_autotune_seq"
MAX_ROUNDS = 20
MAX_SECONDS = 4 * 3600
NO_IMPROVE_PATIENCE = 5


COMMON_ENV = {
    "INPUT_09_RUN_DIR": str(SOURCE_09),
    "PY_TEMP_DIR": str(PROJECT_ROOT),
    "SPARK_LOCAL_DIR": str(PROJECT_ROOT / "data/spark-tmp"),
    "SPARK_DRIVER_EXTRA_JAVA_OPTIONS": "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=256m -XX:MaxMetaspaceSize=512m -Xss512k",
    "SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS": "-XX:+UseSerialGC -XX:TieredStopAtLevel=1 -XX:CICompilerCount=2 -XX:ReservedCodeCacheSize=256m -XX:MaxMetaspaceSize=512m -Xss512k",
    "TRAIN_MODEL_BACKEND": "xgboost_ranker",
}


SEARCH_SPACE: dict[int, list[dict[str, Any]]] = {
    5: [
        {
            "name": "b5_a08_t0_g08",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "2",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "6",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "3",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "100",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.20",
                "TRAIN_ROUTE_BLEND_GAMMA": "0.8",
                "TRAIN_BLEND_ALPHA": "0.08",
                "TRAIN_MAX_VALID_USERS": "2000",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.08", "RANK_RERANK_TOPN": "0"},
        },
        {
            "name": "b5_a08_t20_g10",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "3",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "6",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "3",
                "TRAIN_TOTAL_NEG_PER_USER": "12",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "120",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.25",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.0",
                "TRAIN_BLEND_ALPHA": "0.08",
                "TRAIN_MAX_VALID_USERS": "2000",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.08", "RANK_RERANK_TOPN": "20"},
        },
        {
            "name": "b5_a10_t30_g12",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "3",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "6",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "3",
                "TRAIN_TOTAL_NEG_PER_USER": "12",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "120",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.30",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.2",
                "TRAIN_BLEND_ALPHA": "0.10",
                "TRAIN_MAX_VALID_USERS": "1800",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.10", "RANK_RERANK_TOPN": "30"},
        },
        {
            "name": "b5_a06_t15_g14",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "3",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "2",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "7",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "3",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "90",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.35",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.4",
                "TRAIN_BLEND_ALPHA": "0.06",
                "TRAIN_MAX_VALID_USERS": "1600",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.06", "RANK_RERANK_TOPN": "15"},
        },
        {
            "name": "b5_a12_t30_globalcal",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "3",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "6",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "3",
                "TRAIN_TOTAL_NEG_PER_USER": "12",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "120",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.25",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.0",
                "TRAIN_BLEND_ALPHA": "0.12",
                "TRAIN_ENABLE_GLOBAL_CALIBRATION": "true",
                "TRAIN_MAX_VALID_USERS": "1800",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.12", "RANK_RERANK_TOPN": "30"},
        },
    ],
    2: [
        {
            "name": "b2_a08_t0_sparse",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "2",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "4",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "2",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_BLEND_ALPHA": "0.08",
                "TRAIN_MAX_VALID_USERS": "1200",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.08", "RANK_RERANK_TOPN": "0"},
        },
        {
            "name": "b2_a10_t30_sparse",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "2",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "4",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "2",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_BLEND_ALPHA": "0.10",
                "TRAIN_MAX_VALID_USERS": "1200",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.10", "RANK_RERANK_TOPN": "30"},
        },
        {
            "name": "b2_a12_t20_route6",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "2",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "6",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "3",
                "TRAIN_TOTAL_NEG_PER_USER": "12",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "120",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.25",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.0",
                "TRAIN_BLEND_ALPHA": "0.12",
                "TRAIN_MAX_VALID_USERS": "1000",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.12", "RANK_RERANK_TOPN": "20"},
        },
        {
            "name": "b2_a06_t0_routecal",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "4",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "2",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "5",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "2",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "90",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.30",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.2",
                "TRAIN_BLEND_ALPHA": "0.06",
                "TRAIN_MAX_VALID_USERS": "1000",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.06", "RANK_RERANK_TOPN": "0"},
        },
    ],
    10: [
        {
            "name": "b10_a03_t20_pos",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "2",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "1",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "8",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "4",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_ENABLE_GLOBAL_CALIBRATION": "true",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "80",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.35",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.2",
                "TRAIN_BLEND_ALPHA": "0.03",
                "TRAIN_MAX_VALID_USERS": "2500",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.03", "RANK_RERANK_TOPN": "20"},
        },
        {
            "name": "b10_a02_t20_pos",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "2",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "1",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "8",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "4",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_ENABLE_GLOBAL_CALIBRATION": "true",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "80",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.35",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.2",
                "TRAIN_BLEND_ALPHA": "0.02",
                "TRAIN_MAX_VALID_USERS": "2500",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.02", "RANK_RERANK_TOPN": "20"},
        },
        {
            "name": "b10_a05_t0_route",
            "train": {
                "TRAIN_HARD_NEG_PER_USER": "2",
                "TRAIN_MAX_NEAR_POS_NEG_PER_USER": "1",
                "TRAIN_MAX_ROUTE_MIX_NEG_PER_USER": "9",
                "TRAIN_ROUTE_NEG_FLOOR_PER_USER": "5",
                "TRAIN_TOTAL_NEG_PER_USER": "10",
                "TRAIN_ENABLE_GLOBAL_CALIBRATION": "true",
                "TRAIN_ROUTE_CALIB_PRIOR_STRENGTH": "70",
                "TRAIN_ROUTE_CALIB_LIFT_MAX": "1.40",
                "TRAIN_ROUTE_BLEND_GAMMA": "1.4",
                "TRAIN_BLEND_ALPHA": "0.05",
                "TRAIN_MAX_VALID_USERS": "2200",
            },
            "infer": {"RANK_BLEND_ALPHA": "0.05", "RANK_RERANK_TOPN": "0"},
        },
    ],
}


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def metrics_run_ids() -> set[str]:
    if not METRICS_CSV.exists():
        return set()
    df = pd.read_csv(METRICS_CSV)
    if "run_id_10" not in df.columns:
        return set()
    return set(df["run_id_10"].astype(str).unique().tolist())


def latest_model_json() -> Path | None:
    if not MODEL_ROOT.exists():
        return None
    dirs = [d for d in MODEL_ROOT.iterdir() if d.is_dir() and d.name.endswith("_stage10_1_rank_train")]
    if not dirs:
        return None
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    cand = dirs[0] / "rank_model.json"
    return cand if cand.exists() else None


def read_bucket_metrics(run_id: str, bucket: int) -> dict[str, float] | None:
    if not METRICS_CSV.exists():
        return None
    df = pd.read_csv(METRICS_CSV)
    s = df[(df["run_id_10"].astype(str) == str(run_id)) & (df["bucket_min_train_reviews"] == int(bucket))].copy()
    if s.empty:
        return None
    pre = s[s["model"] == "PreScore@10"]
    learned = s[s["model"].str.contains("LearnedBlendXGBRanker@10", na=False)]
    if pre.empty or learned.empty:
        return None
    pre_r = float(pre.iloc[0]["recall_at_k"])
    pre_n = float(pre.iloc[0]["ndcg_at_k"])
    l_r = float(learned.iloc[0]["recall_at_k"])
    l_n = float(learned.iloc[0]["ndcg_at_k"])
    return {
        "pre_recall": pre_r,
        "pre_ndcg": pre_n,
        "learned_recall": l_r,
        "learned_ndcg": l_n,
        "delta_recall": l_r - pre_r,
        "delta_ndcg": l_n - pre_n,
    }


def run_cmd(cmd: list[str], env: dict[str, str], log_path: Path, timeout_sec: int) -> int:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, capture_output=True, timeout=timeout_sec)
    log_path.write_text((proc.stdout or "") + "\n\n[STDERR]\n" + (proc.stderr or ""), encoding="utf-8")
    return int(proc.returncode)


def make_env(bucket: int, train_env: dict[str, str], infer_env: dict[str, str], model_json: Path | None = None) -> tuple[dict[str, str], dict[str, str]]:
    train = dict(os.environ)
    train.update(COMMON_ENV)
    train.update(
        {
            "TRAIN_BUCKETS_OVERRIDE": str(bucket),
            "RANK_BUCKETS_OVERRIDE": str(bucket),
        }
    )
    train.update({k: str(v) for k, v in train_env.items()})

    infer = dict(os.environ)
    infer.update(COMMON_ENV)
    infer.update(
        {
            "RANK_BUCKETS_OVERRIDE": str(bucket),
        }
    )
    infer.update({k: str(v) for k, v in infer_env.items()})
    if model_json is not None:
        infer["RANK_MODEL_JSON"] = str(model_json)
    return train, infer


def main() -> None:
    if not SOURCE_09.exists():
        raise FileNotFoundError(f"stage09 run dir not found: {SOURCE_09}")
    if not TRAIN_SCRIPT.exists() or not EVAL_SCRIPT.exists():
        raise FileNotFoundError("train/eval script not found")

    session_dir = PROJECT_ROOT / "tmp" / f"{now_str()}_{RUN_TAG}"
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "rounds.csv"
    summary_path = session_dir / "summary.json"

    rounds: list[dict[str, Any]] = []
    best_by_bucket: dict[int, dict[str, Any]] = {}

    start_ts = time.time()
    total_rounds = 0

    for bucket in [5, 2, 10]:
        bucket_best = None
        no_improve = 0
        configs = SEARCH_SPACE.get(bucket, [])
        cfg_idx = 0

        while True:
            elapsed = time.time() - start_ts
            if elapsed >= MAX_SECONDS:
                break
            if total_rounds >= MAX_ROUNDS:
                break
            if no_improve >= NO_IMPROVE_PATIENCE:
                break
            if cfg_idx >= len(configs):
                break

            cfg = configs[cfg_idx]
            cfg_idx += 1
            total_rounds += 1

            round_id = total_rounds
            run_row: dict[str, Any] = {
                "round": round_id,
                "bucket": bucket,
                "cfg_name": cfg["name"],
                "status": "started",
                "start_time": now_str(),
            }

            train_env, _ = make_env(bucket=bucket, train_env=cfg["train"], infer_env=cfg["infer"], model_json=None)
            train_log = logs_dir / f"round_{round_id:02d}_bucket{bucket}_train.log"

            before_model = latest_model_json()
            try:
                rc_train = run_cmd(["python", str(TRAIN_SCRIPT)], train_env, train_log, timeout_sec=3600)
            except subprocess.TimeoutExpired:
                rc_train = 124
                train_log.write_text((train_log.read_text(encoding="utf-8") if train_log.exists() else "") + "\n[ERROR]\ntrain timeout", encoding="utf-8")

            run_row["train_log"] = str(train_log)
            run_row["rc_train"] = rc_train
            if rc_train != 0:
                run_row["status"] = "train_fail"
                run_row["train_env"] = json.dumps(cfg["train"], ensure_ascii=True)
                rounds.append(run_row)
                no_improve += 1
                pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue

            model_json = latest_model_json()
            if model_json is None:
                run_row["status"] = "no_model_json"
                run_row["train_env"] = json.dumps(cfg["train"], ensure_ascii=True)
                rounds.append(run_row)
                no_improve += 1
                pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue
            if before_model is not None and model_json == before_model:
                run_row["status"] = "model_not_updated"
                run_row["train_env"] = json.dumps(cfg["train"], ensure_ascii=True)
                run_row["model_json"] = str(model_json)
                rounds.append(run_row)
                no_improve += 1
                pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue

            pre_ids = metrics_run_ids()
            _, infer_env = make_env(bucket=bucket, train_env=cfg["train"], infer_env=cfg["infer"], model_json=model_json)
            eval_log = logs_dir / f"round_{round_id:02d}_bucket{bucket}_eval.log"
            try:
                rc_eval = run_cmd(["python", str(EVAL_SCRIPT)], infer_env, eval_log, timeout_sec=3600)
            except subprocess.TimeoutExpired:
                rc_eval = 124
                eval_log.write_text((eval_log.read_text(encoding="utf-8") if eval_log.exists() else "") + "\n[ERROR]\neval timeout", encoding="utf-8")

            run_row["eval_log"] = str(eval_log)
            run_row["rc_eval"] = rc_eval
            run_row["model_json"] = str(model_json)
            run_row["train_env"] = json.dumps(cfg["train"], ensure_ascii=True)
            run_row["infer_env"] = json.dumps(cfg["infer"], ensure_ascii=True)

            if rc_eval != 0:
                run_row["status"] = "eval_fail"
                rounds.append(run_row)
                no_improve += 1
                pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue

            post_ids = metrics_run_ids()
            new_ids = sorted([x for x in post_ids if x not in pre_ids])
            if not new_ids:
                run_row["status"] = "no_new_metric"
                rounds.append(run_row)
                no_improve += 1
                pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue

            run_id_10 = new_ids[-1]
            m = read_bucket_metrics(run_id=run_id_10, bucket=bucket)
            if m is None:
                run_row["status"] = "metric_parse_fail"
                run_row["run_id_10"] = run_id_10
                rounds.append(run_row)
                no_improve += 1
                pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")
                continue

            run_row.update(m)
            run_row["run_id_10"] = run_id_10
            run_row["status"] = "ok"

            improved = False
            if bucket_best is None:
                improved = True
            elif float(m["delta_ndcg"]) > float(bucket_best["delta_ndcg"]) + 1e-12:
                improved = True

            if improved:
                bucket_best = {
                    "bucket": bucket,
                    "round": round_id,
                    "run_id_10": run_id_10,
                    "cfg_name": cfg["name"],
                    "model_json": str(model_json),
                    "delta_ndcg": float(m["delta_ndcg"]),
                    "delta_recall": float(m["delta_recall"]),
                    "pre_ndcg": float(m["pre_ndcg"]),
                    "learned_ndcg": float(m["learned_ndcg"]),
                    "train_env": cfg["train"],
                    "infer_env": cfg["infer"],
                }
                no_improve = 0
            else:
                no_improve += 1

            rounds.append(run_row)
            pd.DataFrame(rounds).to_csv(csv_path, index=False, encoding="utf-8-sig")

        if bucket_best is not None:
            best_by_bucket[bucket] = bucket_best
        elapsed = time.time() - start_ts
        if elapsed >= MAX_SECONDS or total_rounds >= MAX_ROUNDS:
            break

    summary: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "session_dir": str(session_dir),
        "source_09": str(SOURCE_09),
        "max_rounds": MAX_ROUNDS,
        "max_seconds": MAX_SECONDS,
        "no_improve_patience": NO_IMPROVE_PATIENCE,
        "total_rounds_executed": total_rounds,
        "elapsed_seconds": round(time.time() - start_ts, 2),
        "best_by_bucket": best_by_bucket,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print("[DONE] autotune finished")
    print(f"[PATH] rounds_csv={csv_path}")
    print(f"[PATH] summary_json={summary_path}")
    for b in [5, 2, 10]:
        best = best_by_bucket.get(b)
        if not best:
            print(f"[BEST] bucket={b} none")
            continue
        print(
            f"[BEST] bucket={b} run={best['run_id_10']} cfg={best['cfg_name']} "
            f"delta_ndcg={best['delta_ndcg']:.6f} delta_recall={best['delta_recall']:.6f}"
        )


if __name__ == "__main__":
    main()
