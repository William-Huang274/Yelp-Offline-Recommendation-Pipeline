from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    os.getenv("QLORA_EVAL_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128").strip()
    or "expandable_segments:True,max_split_size_mb:128",
)

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


WORKER_VERSION = "stage11_remote_worker_v1"
STATE: dict[str, Any] = {}
SCORE_LOCK = threading.Lock()


class ReuseAddressHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _load_stage11_eval(repo_root: Path):
    for module_path in (repo_root / "scripts", repo_root):
        module_path_str = str(module_path)
        if module_path_str not in sys.path:
            sys.path.insert(0, module_path_str)
    spec = importlib.util.spec_from_file_location(
        "stage11_eval_module",
        repo_root / "scripts" / "11_3_qlora_sidecar_eval.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load 11_3_qlora_sidecar_eval.py on remote host")
    stage11_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage11_eval)
    return stage11_eval


def _load_assets(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    repo_root = Path(str(config["remote_repo_root"])).resolve()
    stage11_eval = _load_stage11_eval(repo_root)

    score_pdf = pd.read_csv(str(config["remote_score_csv"]))
    score_pdf["user_idx"] = pd.to_numeric(score_pdf["user_idx"], errors="coerce").fillna(-1).astype(int)
    score_pdf["item_idx"] = pd.to_numeric(score_pdf["item_idx"], errors="coerce").fillna(-1).astype(int)
    score_pdf["pre_rank"] = pd.to_numeric(score_pdf["pre_rank"], errors="coerce").fillna(999999).astype(int)
    score_pdf["learned_rank"] = pd.to_numeric(score_pdf["learned_rank"], errors="coerce").fillna(999999).astype(int)
    score_pdf["reward_score"] = pd.to_numeric(score_pdf.get("reward_score"), errors="coerce")

    base_model = str(config["base_model"]).strip()
    dtype = torch.float16
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
        except Exception:
            dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    tokenizer.padding_side = "right"

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    setattr(cfg, "num_labels", 1)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        config=cfg,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "base_model") and getattr(model.base_model.config, "pad_token_id", None) is None:
        model.base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(
        model,
        str(config["adapter_dir_11_30"]).strip(),
        adapter_name="band_11_30",
    )
    adapter_dir_31_60 = str(config.get("adapter_dir_31_60", "")).strip()
    adapter_dir_61_100 = str(config.get("adapter_dir_61_100", "")).strip()
    if adapter_dir_31_60:
        model.load_adapter(adapter_dir_31_60, adapter_name="band_31_60")
    if adapter_dir_61_100:
        model.load_adapter(adapter_dir_61_100, adapter_name="band_61_100")
    model.eval()

    STATE.update(
        {
            "ready": True,
            "config": config,
            "stage11_eval": stage11_eval,
            "score_pdf": score_pdf,
            "pairwise_dir": Path(str(config["remote_pairwise_dir"])),
            "tokenizer": tokenizer,
            "model": model,
            "load_elapsed_s": round(time.perf_counter() - started, 3),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
        }
    )


def _route_adapter(pre_rank: int, adapter_dir_61_100: str) -> str:
    if 11 <= int(pre_rank) <= 30:
        return "band_11_30"
    if 31 <= int(pre_rank) <= 60:
        return "band_31_60"
    if 61 <= int(pre_rank) <= 100 and adapter_dir_61_100.strip():
        return "band_61_100"
    return ""


def _score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request_started = time.perf_counter()
    config = STATE["config"]
    stage11_eval = STATE["stage11_eval"]
    score_store = STATE["score_pdf"]
    pairwise_dir = STATE["pairwise_dir"]
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]

    user_idx = int(payload["user_idx"])
    max_pre_rank = int(payload.get("max_pre_rank", 100) or 100)
    max_rivals = int(payload.get("max_rivals", 11) or 11)
    preview_rows = int(payload.get("preview_rows", 12) or 12)
    batch_size = int(payload.get("batch_size", 8) or 8)
    max_seq_len = int(payload.get("max_seq_len", 1280) or 1280)

    prepare_started = time.perf_counter()
    score_pdf = score_store[score_store["user_idx"].eq(user_idx)].copy()
    score_pdf = score_pdf.sort_values(["pre_rank", "item_idx"], kind="stable")
    score_pdf = score_pdf[score_pdf["pre_rank"].le(max_pre_rank)].copy()
    if score_pdf.empty:
        raise RuntimeError(f"no replay rows found for user_idx={user_idx}")

    pair_pdf = pd.read_parquet(pairwise_dir, filters=[("user_idx", "==", user_idx)])
    pair_pdf["item_idx"] = pd.to_numeric(pair_pdf["item_idx"], errors="coerce").fillna(-1).astype(int)
    pair_pdf = pair_pdf[pair_pdf["item_idx"].isin(set(score_pdf["item_idx"].tolist()))].copy()
    if pair_pdf.empty:
        raise RuntimeError(f"no pairwise rows found for user_idx={user_idx}")

    score_merge_cols = ["user_idx", "item_idx", "pre_rank", "learned_rank", "reward_score", "route_band"]
    overlap_cols = [col for col in score_merge_cols if col not in {"user_idx", "item_idx"} and col in pair_pdf.columns]
    if overlap_cols:
        pair_pdf = pair_pdf.drop(columns=overlap_cols)
    pair_pdf = pair_pdf.merge(
        score_pdf[score_merge_cols].drop_duplicates(["user_idx", "item_idx"]),
        on=["user_idx", "item_idx"],
        how="inner",
    )
    pair_pdf = pair_pdf.sort_values(["pre_rank", "item_idx"], kind="stable").reset_index(drop=True)
    prepare_elapsed_ms = round((time.perf_counter() - prepare_started) * 1000.0, 3)

    prompt_started = time.perf_counter()
    prompts = stage11_eval.build_driver_local_listwise_prompts(pair_pdf, max_rivals=max_rivals)
    pair_pdf["prompt"] = prompts.reindex(pair_pdf.index).fillna("").astype(str)
    adapter_dir_61_100 = str(config.get("adapter_dir_61_100", "") or "")
    pair_pdf["sidecar_model_id"] = pair_pdf["pre_rank"].map(lambda rank: _route_adapter(int(rank), adapter_dir_61_100))
    score_target_pdf = pair_pdf[pair_pdf["sidecar_model_id"].ne("")].copy()
    if score_target_pdf.empty:
        raise RuntimeError(f"no live-score eligible rows found for user_idx={user_idx}")
    prompt_elapsed_ms = round((time.perf_counter() - prompt_started) * 1000.0, 3)

    inference_started = time.perf_counter()
    rows = []
    for adapter_name in sorted(str(x) for x in score_target_pdf["sidecar_model_id"].drop_duplicates().tolist()):
        subset = score_target_pdf[score_target_pdf["sidecar_model_id"].eq(adapter_name)].copy()
        if subset.empty:
            continue
        stage11_eval.set_active_adapter(model, adapter_name)
        live_scores, _ = stage11_eval.score_reward_model(
            model=model,
            tokenizer=tokenizer,
            prompts=subset["prompt"].fillna("").astype(str).tolist(),
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        subset["live_reward_score"] = pd.Series(live_scores.astype(np.float32), index=subset.index)
        rows.append(subset)
    if not rows:
        raise RuntimeError("no live scores produced")
    inference_elapsed_ms = round((time.perf_counter() - inference_started) * 1000.0, 3)

    result_pdf = pd.concat(rows, ignore_index=True)
    result_pdf["abs_diff_vs_replay"] = (
        pd.to_numeric(result_pdf["live_reward_score"], errors="coerce")
        - pd.to_numeric(result_pdf["reward_score"], errors="coerce")
    ).abs()
    result_pdf = result_pdf.sort_values(["abs_diff_vs_replay", "pre_rank", "item_idx"], ascending=[False, True, True])

    preview_pdf = result_pdf.sort_values(["pre_rank", "item_idx"], kind="stable").head(preview_rows)
    scored_rows = []
    for row in preview_pdf.to_dict("records"):
        scored_rows.append(
            {
                "item_idx": int(row.get("item_idx", -1)),
                "name": str(row.get("name", "") or ""),
                "pre_rank": int(row.get("pre_rank", -1)),
                "learned_rank": int(row.get("learned_rank", -1)),
                "route_band": str(row.get("route_band", "") or ""),
                "sidecar_model_id": str(row.get("sidecar_model_id", "") or ""),
                "replay_reward_score": None if pd.isna(row.get("reward_score")) else float(row.get("reward_score")),
                "live_reward_score": None
                if pd.isna(row.get("live_reward_score"))
                else float(row.get("live_reward_score")),
                "abs_diff_vs_replay": None
                if pd.isna(row.get("abs_diff_vs_replay"))
                else float(row.get("abs_diff_vs_replay")),
                "prompt_preview": str(row.get("prompt", "") or "")[:220],
            }
        )

    total_elapsed_ms = round((time.perf_counter() - request_started) * 1000.0, 3)
    return {
        "status": "ok",
        "worker_version": WORKER_VERSION,
        "request_id": str(payload.get("request_id", "")),
        "user_idx": user_idx,
        "n_candidates_total": int(pair_pdf.shape[0]),
        "n_scored": int(result_pdf.shape[0]),
        "mean_abs_diff_vs_replay": float(result_pdf["abs_diff_vs_replay"].fillna(0.0).mean()),
        "max_abs_diff_vs_replay": float(result_pdf["abs_diff_vs_replay"].fillna(0.0).max()),
        "timings_ms": {
            "prepare": prepare_elapsed_ms,
            "prompt_build": prompt_elapsed_ms,
            "inference": inference_elapsed_ms,
            "total": total_elapsed_ms,
        },
        "scored_rows": scored_rows,
    }


class Stage11WorkerHandler(BaseHTTPRequestHandler):
    server_version = "Stage11RemoteWorker/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path != "/health":
            _json_response(self, {"status": "not_found", "path": self.path}, status=404)
            return
        _json_response(
            self,
            {
                "status": "ok" if STATE.get("ready") else "loading",
                "ready": bool(STATE.get("ready")),
                "worker_version": WORKER_VERSION,
                "pid": STATE.get("pid", os.getpid()),
                "load_elapsed_s": STATE.get("load_elapsed_s"),
                "started_at": STATE.get("started_at"),
            },
        )

    def do_POST(self) -> None:
        if self.path != "/score":
            _json_response(self, {"status": "not_found", "path": self.path}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            _json_response(self, {"status": "error", "detail": f"invalid json payload: {exc}"}, status=400)
            return
        if not STATE.get("ready"):
            _json_response(self, {"status": "error", "detail": "worker is not ready"}, status=503)
            return
        try:
            with SCORE_LOCK:
                result = _score_payload(payload)
            _json_response(self, result)
        except Exception as exc:
            _json_response(
                self,
                {
                    "status": "error",
                    "worker_version": WORKER_VERSION,
                    "detail": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                },
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote Stage11 reward-model worker.")
    parser.add_argument("--config", required=True, help="worker config JSON path")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    host = str(config.get("worker_host", "127.0.0.1"))
    port = int(config.get("worker_port", 18080) or 18080)
    _load_assets(config)
    server = ReuseAddressHTTPServer((host, port), Stage11WorkerHandler)
    print(
        json.dumps(
            {
                "status": "ready",
                "worker_version": WORKER_VERSION,
                "host": host,
                "port": port,
                "pid": os.getpid(),
                "load_elapsed_s": STATE.get("load_elapsed_s"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
