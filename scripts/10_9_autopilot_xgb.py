from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"D:/5006 BDA project")
SOURCE_09_ROOT = PROJECT_ROOT / "data/output/09_candidate_fusion"
METRICS_CSV = PROJECT_ROOT / "data/metrics/recsys_stage10_results.csv"
PITFALLS_JSONL = PROJECT_ROOT / "data/metrics/training_pitfalls_memory.jsonl"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/10_1_rank_train.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts/10_2_rank_infer_eval.py"
MODEL_ROOT = PROJECT_ROOT / "data/output/10_rank_models"
RUN_TAG = "stage10_9_autopilot_xgb"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _bucket_order() -> list[int]:
    raw = os.getenv("AUTOTUNE_BUCKET_ORDER", "5,2,10").strip()
    out: list[int] = []
    for p in raw.split(","):
        t = p.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or [5, 2, 10]


MAX_ROUNDS = _env_int("AUTOTUNE_MAX_ROUNDS", 20)
MAX_SECONDS = _env_int("AUTOTUNE_MAX_SECONDS", 4 * 3600)
NO_IMPROVE_PATIENCE = _env_int("AUTOTUNE_PATIENCE", 5)
TRAIN_TIMEOUT_SEC = _env_int("AUTOTUNE_TRAIN_TIMEOUT_SEC", 5400)
EVAL_TIMEOUT_SEC = _env_int("AUTOTUNE_EVAL_TIMEOUT_SEC", 3600)
BUCKET_ORDER = _bucket_order()


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = dict(row)
    line.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=True) + "\n")


def record_training_memory(
    *,
    session_dir: Path,
    source_09: Path,
    bucket: int,
    round_id: int | None,
    cfg_name: str,
    event: str,
    severity: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "run_tag": RUN_TAG,
        "session_dir": str(session_dir),
        "source_09": str(source_09),
        "bucket": int(bucket),
        "round": None if round_id is None else int(round_id),
        "cfg_name": str(cfg_name),
        "event": str(event),
        "severity": str(severity),
        "message": str(message),
    }
    if extra:
        payload.update(extra)
    append_jsonl(PITFALLS_JSONL, payload)


def pick_latest_stage09_run() -> Path:
    forced = os.getenv("INPUT_09_RUN_DIR", "").strip()
    if forced:
        p = Path(forced)
        if not p.exists():
            raise FileNotFoundError(f"INPUT_09_RUN_DIR not found: {p}")
        return p
    runs = [p for p in SOURCE_09_ROOT.iterdir() if p.is_dir() and p.name.endswith("_stage09_candidate_fusion")]
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no stage09 runs under {SOURCE_09_ROOT}")
    return runs[0]


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
    p = dirs[0] / "rank_model.json"
    return p if p.exists() else None


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


def read_bucket_alpha(model_json: Path, bucket: int) -> float | None:
    try:
        obj = json.loads(model_json.read_text(encoding="utf-8"))
        m = obj.get("models_by_bucket", {}).get(str(bucket), {})
        return float(m.get("metrics", {}).get("blend_alpha", 0.0))
    except Exception:
        return None


def run_cmd(cmd: list[str], env: dict[str, str], log_path: Path, timeout_sec: int) -> int:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, capture_output=True, timeout=timeout_sec)
    elapsed = time.time() - t0
    body = [
        f"[CMD] {' '.join(cmd)}",
        f"[RET] {proc.returncode}",
        f"[ELAPSED_SEC] {elapsed:.1f}",
        "",
        "[STDOUT]",
        proc.stdout or "",
        "",
        "[STDERR]",
        proc.stderr or "",
    ]
    log_path.write_text("\n".join(body), encoding="utf-8")
    return int(proc.returncode)


def make_bucket_policy(bucket: int, overrides: dict[str, Any]) -> str:
    return json.dumps({str(int(bucket)): overrides}, ensure_ascii=True)


def build_search_space() -> dict[int, list[dict[str, Any]]]:
    # Configs are ordered from conservative to aggressive.
    return {
        5: [
            {
                "name": "b5_base_route",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        5,
                        {
                            "total_neg_per_user": 28,
                            "max_route_mix_neg_per_user": 11,
                            "route_neg_floor_per_user": 6,
                            "max_near_pos_neg_per_user": 6,
                            "max_easy_neg_per_user": 10,
                            "max_random_neg_per_user": 6,
                        },
                    ),
                },
                "infer": {},
            },
            {
                "name": "b5_tail_weight_mild",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        5,
                        {
                            "total_neg_per_user": 30,
                            "max_route_mix_neg_per_user": 12,
                            "route_neg_floor_per_user": 7,
                            "max_near_pos_neg_per_user": 5,
                            "max_easy_neg_per_user": 8,
                            "max_random_neg_per_user": 4,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "2.2",
                    "TRAIN_POS_W_MID": "2.0",
                    "TRAIN_POS_W_TAIL": "2.8",
                    "TRAIN_RANK_PRIOR_POWER": "0.20",
                },
                "infer": {},
            },
            {
                "name": "b5_tail_weight_strong",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        5,
                        {
                            "total_neg_per_user": 32,
                            "max_route_mix_neg_per_user": 14,
                            "route_neg_floor_per_user": 8,
                            "max_near_pos_neg_per_user": 4,
                            "max_easy_neg_per_user": 7,
                            "max_random_neg_per_user": 3,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "1.8",
                    "TRAIN_POS_W_MID": "2.2",
                    "TRAIN_POS_W_TAIL": "3.0",
                    "TRAIN_RANK_PRIOR_POWER": "0.12",
                    "TRAIN_BLEND_ALPHA_GRID": "0.0,0.05,0.1,0.15,0.25",
                },
                "infer": {},
            },
            {
                "name": "b5_force_learn_probe",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        5,
                        {
                            "total_neg_per_user": 30,
                            "max_route_mix_neg_per_user": 12,
                            "route_neg_floor_per_user": 7,
                            "max_near_pos_neg_per_user": 5,
                            "max_easy_neg_per_user": 8,
                            "max_random_neg_per_user": 4,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "2.0",
                    "TRAIN_POS_W_MID": "2.0",
                    "TRAIN_POS_W_TAIL": "2.6",
                    "TRAIN_RANK_PRIOR_POWER": "0.15",
                    "TRAIN_BLEND_ALPHA_GRID": "0.05,0.1,0.15,0.25",
                },
                "infer": {},
            },
        ],
        2: [
            {
                "name": "b2_base_route",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        2,
                        {
                            "total_neg_per_user": 20,
                            "max_route_mix_neg_per_user": 6,
                            "route_neg_floor_per_user": 3,
                            "max_near_pos_neg_per_user": 8,
                        },
                    ),
                },
                "infer": {},
            },
            {
                "name": "b2_tail_weight_mild",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        2,
                        {
                            "total_neg_per_user": 22,
                            "max_route_mix_neg_per_user": 8,
                            "route_neg_floor_per_user": 4,
                            "max_near_pos_neg_per_user": 6,
                            "max_easy_neg_per_user": 10,
                            "max_random_neg_per_user": 6,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "2.3",
                    "TRAIN_POS_W_MID": "2.0",
                    "TRAIN_POS_W_TAIL": "2.6",
                    "TRAIN_RANK_PRIOR_POWER": "0.22",
                },
                "infer": {},
            },
            {
                "name": "b2_tail_weight_strong",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        2,
                        {
                            "total_neg_per_user": 24,
                            "max_route_mix_neg_per_user": 9,
                            "route_neg_floor_per_user": 5,
                            "max_near_pos_neg_per_user": 5,
                            "max_easy_neg_per_user": 8,
                            "max_random_neg_per_user": 4,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "1.9",
                    "TRAIN_POS_W_MID": "2.2",
                    "TRAIN_POS_W_TAIL": "2.8",
                    "TRAIN_RANK_PRIOR_POWER": "0.14",
                    "TRAIN_BLEND_ALPHA_GRID": "0.0,0.05,0.1,0.15,0.25",
                },
                "infer": {},
            },
        ],
        10: [
            {
                "name": "b10_base_route",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        10,
                        {
                            "total_neg_per_user": 36,
                            "max_route_mix_neg_per_user": 16,
                            "route_neg_floor_per_user": 9,
                            "max_near_pos_neg_per_user": 4,
                            "max_easy_neg_per_user": 8,
                            "max_random_neg_per_user": 4,
                        },
                    ),
                },
                "infer": {},
            },
            {
                "name": "b10_tail_weight_mild",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        10,
                        {
                            "total_neg_per_user": 40,
                            "max_route_mix_neg_per_user": 20,
                            "route_neg_floor_per_user": 12,
                            "max_near_pos_neg_per_user": 3,
                            "max_easy_neg_per_user": 6,
                            "max_random_neg_per_user": 2,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "2.0",
                    "TRAIN_POS_W_MID": "2.3",
                    "TRAIN_POS_W_TAIL": "3.0",
                    "TRAIN_RANK_PRIOR_POWER": "0.12",
                },
                "infer": {},
            },
            {
                "name": "b10_force_learn_probe",
                "train": {
                    "TRAIN_BUCKET_POLICY_JSON": make_bucket_policy(
                        10,
                        {
                            "total_neg_per_user": 42,
                            "max_route_mix_neg_per_user": 22,
                            "route_neg_floor_per_user": 12,
                            "max_near_pos_neg_per_user": 2,
                            "max_easy_neg_per_user": 6,
                            "max_random_neg_per_user": 2,
                        },
                    ),
                    "TRAIN_POS_W_TOP": "1.8",
                    "TRAIN_POS_W_MID": "2.3",
                    "TRAIN_POS_W_TAIL": "3.2",
                    "TRAIN_RANK_PRIOR_POWER": "0.10",
                    "TRAIN_BLEND_ALPHA_GRID": "0.05,0.1,0.15,0.25",
                },
                "infer": {},
            },
        ],
    }


def make_env(source_09: Path, bucket: int, train_env: dict[str, Any], infer_env: dict[str, Any], model_json: Path | None = None) -> tuple[dict[str, str], dict[str, str]]:
    base = dict(os.environ)
    base.update(
        {
            "INPUT_09_RUN_DIR": str(source_09),
            "PY_TEMP_DIR": str(PROJECT_ROOT),
            "SPARK_LOCAL_DIR": str(PROJECT_ROOT / "data/spark-tmp"),
            "TRAIN_MODEL_BACKEND": "xgboost_ranker",
            "TRAIN_BUCKETS_OVERRIDE": str(bucket),
            "RANK_BUCKETS_OVERRIDE": str(bucket),
        }
    )

    train = dict(base)
    train.update({k: str(v) for k, v in train_env.items()})

    infer = dict(base)
    infer.update({k: str(v) for k, v in infer_env.items()})
    if model_json is not None:
        infer["RANK_MODEL_JSON"] = str(model_json)
    return train, infer


def diagnose_bucket(rows: list[dict[str, Any]], bucket: int) -> dict[str, Any]:
    br = [r for r in rows if int(r.get("bucket", -1)) == int(bucket) and r.get("status") == "ok"]
    if not br:
        return {"bucket": int(bucket), "status": "no_success_rounds"}
    best = max(br, key=lambda x: (float(x.get("delta_ndcg", -1e9)), float(x.get("delta_recall", -1e9))))
    alpha_zero_rate = sum(1 for r in br if abs(float(r.get("selected_alpha", 0.0))) < 1e-9) / float(len(br))
    no_gain = all(float(r.get("delta_ndcg", -1.0)) <= 1e-9 for r in br)
    diagnosis = []
    if no_gain and alpha_zero_rate >= 0.8:
        diagnosis.append("learned_signal_weaker_than_pre")
        diagnosis.append("likely_label_sparsity_or_feature_homogeneity")
    elif no_gain:
        diagnosis.append("no_stable_gain")
    else:
        diagnosis.append("has_positive_gain")
    return {
        "bucket": int(bucket),
        "rounds": int(len(br)),
        "best_round": int(best.get("round", -1)),
        "best_cfg": str(best.get("cfg_name", "")),
        "best_delta_ndcg": float(best.get("delta_ndcg", 0.0)),
        "best_delta_recall": float(best.get("delta_recall", 0.0)),
        "alpha_zero_rate": float(alpha_zero_rate),
        "diagnosis": diagnosis,
    }


def main() -> None:
    source_09 = pick_latest_stage09_run()
    if not TRAIN_SCRIPT.exists() or not EVAL_SCRIPT.exists():
        raise FileNotFoundError("missing stage10 train/eval scripts")
    search_space = build_search_space()

    session_dir = PROJECT_ROOT / "tmp" / f"{now_str()}_{RUN_TAG}"
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rounds_csv = session_dir / "rounds.csv"
    summary_json = session_dir / "summary.json"

    rounds: list[dict[str, Any]] = []
    best_by_bucket: dict[int, dict[str, Any]] = {}

    start = time.time()
    total_rounds = 0

    print(f"[SESSION] dir={session_dir}")
    print(f"[SOURCE_09] {source_09}")
    print(
        f"[CONFIG] buckets={BUCKET_ORDER} max_rounds={MAX_ROUNDS} "
        f"max_seconds={MAX_SECONDS} patience={NO_IMPROVE_PATIENCE}"
    )
    append_jsonl(
        PITFALLS_JSONL,
        {
            "run_tag": RUN_TAG,
            "session_dir": str(session_dir),
            "source_09": str(source_09),
            "event": "session_start",
            "severity": "info",
            "message": "autopilot session started",
            "bucket_order": BUCKET_ORDER,
            "max_rounds": int(MAX_ROUNDS),
            "max_seconds": int(MAX_SECONDS),
            "patience": int(NO_IMPROVE_PATIENCE),
        },
    )

    for bucket in BUCKET_ORDER:
        cfgs = search_space.get(int(bucket), [])
        if not cfgs:
            continue
        no_improve = 0
        bucket_best: dict[str, Any] | None = None

        for cfg in cfgs:
            if total_rounds >= MAX_ROUNDS or (time.time() - start) >= MAX_SECONDS:
                break
            if no_improve >= NO_IMPROVE_PATIENCE:
                break

            total_rounds += 1
            rid = total_rounds
            row: dict[str, Any] = {
                "round": int(rid),
                "bucket": int(bucket),
                "cfg_name": str(cfg.get("name", f"cfg_{rid}")),
                "status": "started",
                "time_start": now_str(),
            }
            print(f"[ROUND] {rid} bucket={bucket} cfg={row['cfg_name']}")

            train_env, _ = make_env(source_09, bucket, cfg.get("train", {}), cfg.get("infer", {}), None)
            train_log = logs_dir / f"round_{rid:02d}_bucket{bucket}_train.log"
            before_model = latest_model_json()
            try:
                rc_train = run_cmd([sys.executable, str(TRAIN_SCRIPT)], train_env, train_log, TRAIN_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                rc_train = 124
                train_log.write_text("[ERROR] train timeout", encoding="utf-8")
            row["train_log"] = str(train_log)
            row["rc_train"] = int(rc_train)
            if rc_train != 0:
                row["status"] = "train_fail"
                rounds.append(row)
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="train_fail",
                    severity="error",
                    message=f"train failed with rc={rc_train}",
                    extra={"train_log": str(train_log)},
                )
                no_improve += 1
                pd.DataFrame(rounds).to_csv(rounds_csv, index=False, encoding="utf-8-sig")
                print(f"[FAIL] train bucket={bucket} rc={rc_train}")
                continue

            model_json = latest_model_json()
            if model_json is None or (before_model is not None and model_json == before_model):
                row["status"] = "model_not_updated"
                rounds.append(row)
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="model_not_updated",
                    severity="warn",
                    message="rank_model.json not updated after training",
                    extra={"train_log": str(train_log)},
                )
                no_improve += 1
                pd.DataFrame(rounds).to_csv(rounds_csv, index=False, encoding="utf-8-sig")
                print(f"[FAIL] model not updated bucket={bucket}")
                continue

            pre_ids = metrics_run_ids()
            _, infer_env = make_env(source_09, bucket, cfg.get("train", {}), cfg.get("infer", {}), model_json)
            eval_log = logs_dir / f"round_{rid:02d}_bucket{bucket}_eval.log"
            try:
                rc_eval = run_cmd([sys.executable, str(EVAL_SCRIPT)], infer_env, eval_log, EVAL_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                rc_eval = 124
                eval_log.write_text("[ERROR] eval timeout", encoding="utf-8")
            row["eval_log"] = str(eval_log)
            row["rc_eval"] = int(rc_eval)
            row["model_json"] = str(model_json)
            if rc_eval != 0:
                row["status"] = "eval_fail"
                rounds.append(row)
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="eval_fail",
                    severity="error",
                    message=f"eval failed with rc={rc_eval}",
                    extra={"eval_log": str(eval_log), "model_json": str(model_json)},
                )
                no_improve += 1
                pd.DataFrame(rounds).to_csv(rounds_csv, index=False, encoding="utf-8-sig")
                print(f"[FAIL] eval bucket={bucket} rc={rc_eval}")
                continue

            post_ids = metrics_run_ids()
            new_ids = sorted([x for x in post_ids if x not in pre_ids])
            if not new_ids:
                row["status"] = "no_new_metric"
                rounds.append(row)
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="no_new_metric",
                    severity="warn",
                    message="eval finished but no new run_id_10 found in metrics csv",
                    extra={"eval_log": str(eval_log), "model_json": str(model_json)},
                )
                no_improve += 1
                pd.DataFrame(rounds).to_csv(rounds_csv, index=False, encoding="utf-8-sig")
                print(f"[FAIL] no new metric bucket={bucket}")
                continue
            run_id_10 = str(new_ids[-1])
            m = read_bucket_metrics(run_id_10, bucket)
            if m is None:
                row["status"] = "metric_parse_fail"
                row["run_id_10"] = run_id_10
                rounds.append(row)
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="metric_parse_fail",
                    severity="warn",
                    message="new run_id exists but bucket metrics cannot be parsed",
                    extra={"run_id_10": run_id_10, "eval_log": str(eval_log)},
                )
                no_improve += 1
                pd.DataFrame(rounds).to_csv(rounds_csv, index=False, encoding="utf-8-sig")
                print(f"[FAIL] metric parse bucket={bucket} run={run_id_10}")
                continue

            alpha = read_bucket_alpha(model_json, bucket)
            row.update(m)
            row["selected_alpha"] = None if alpha is None else float(alpha)
            row["run_id_10"] = run_id_10
            row["status"] = "ok"
            rounds.append(row)
            pd.DataFrame(rounds).to_csv(rounds_csv, index=False, encoding="utf-8-sig")
            if float(row["delta_ndcg"]) <= 1e-12:
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="no_gain",
                    severity="warn",
                    message="learned blend has no ndcg gain over prescore",
                    extra={
                        "run_id_10": run_id_10,
                        "delta_ndcg": float(row["delta_ndcg"]),
                        "delta_recall": float(row["delta_recall"]),
                        "selected_alpha": row["selected_alpha"],
                    },
                )
            else:
                record_training_memory(
                    session_dir=session_dir,
                    source_09=source_09,
                    bucket=int(bucket),
                    round_id=int(rid),
                    cfg_name=str(row["cfg_name"]),
                    event="positive_gain",
                    severity="info",
                    message="learned blend improved ndcg over prescore",
                    extra={
                        "run_id_10": run_id_10,
                        "delta_ndcg": float(row["delta_ndcg"]),
                        "delta_recall": float(row["delta_recall"]),
                        "selected_alpha": row["selected_alpha"],
                    },
                )

            improved = False
            if bucket_best is None:
                improved = True
            else:
                cur = float(row["delta_ndcg"])
                best_cur = float(bucket_best["delta_ndcg"])
                if cur > best_cur + 1e-12:
                    improved = True
                elif abs(cur - best_cur) <= 1e-12 and float(row["delta_recall"]) > float(bucket_best["delta_recall"]) + 1e-12:
                    improved = True

            if improved:
                bucket_best = dict(row)
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"[METRIC] bucket={bucket} run={run_id_10} "
                f"pre_ndcg={row['pre_ndcg']:.6f} learned_ndcg={row['learned_ndcg']:.6f} "
                f"delta={row['delta_ndcg']:+.6f} alpha={row['selected_alpha']}"
            )

        if bucket_best is not None:
            best_by_bucket[int(bucket)] = bucket_best
        if total_rounds >= MAX_ROUNDS or (time.time() - start) >= MAX_SECONDS:
            break

    diag_rows = [diagnose_bucket(rounds, b) for b in BUCKET_ORDER]
    summary = {
        "run_tag": RUN_TAG,
        "session_dir": str(session_dir),
        "source_09": str(source_09),
        "max_rounds": int(MAX_ROUNDS),
        "max_seconds": int(MAX_SECONDS),
        "patience": int(NO_IMPROVE_PATIENCE),
        "total_rounds_executed": int(total_rounds),
        "elapsed_seconds": round(time.time() - start, 2),
        "best_by_bucket": best_by_bucket,
        "diagnostics": diag_rows,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    append_jsonl(
        PITFALLS_JSONL,
        {
            "run_tag": RUN_TAG,
            "session_dir": str(session_dir),
            "source_09": str(source_09),
            "event": "session_end",
            "severity": "info",
            "message": "autopilot session finished",
            "total_rounds_executed": int(total_rounds),
            "elapsed_seconds": round(time.time() - start, 2),
            "diagnostics": diag_rows,
            "best_by_bucket": best_by_bucket,
            "rounds_csv": str(rounds_csv),
            "summary_json": str(summary_json),
        },
    )

    print("[DONE] autopilot finished")
    print(f"[PATH] rounds_csv={rounds_csv}")
    print(f"[PATH] summary_json={summary_json}")
    for d in diag_rows:
        print(
            f"[DIAG] bucket={d.get('bucket')} best_delta_ndcg={d.get('best_delta_ndcg')} "
            f"alpha_zero_rate={d.get('alpha_zero_rate')} diagnosis={d.get('diagnosis')}"
        )


if __name__ == "__main__":
    main()
