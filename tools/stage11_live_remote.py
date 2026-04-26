from __future__ import annotations

import base64
import json
import os
import shlex
import textwrap
import time
from pathlib import Path
from typing import Any

from replay_store import ReplayStore


DEFAULT_HOST = "connect.westb.seetacloud.com"
DEFAULT_PORT = 20804
DEFAULT_USER = "root"
DEFAULT_REMOTE_REPO_ROOT = "/root/autodl-tmp/5006_BDA_project"
DEFAULT_REMOTE_PYTHON_BIN = "/root/miniconda3/bin/python"
DEFAULT_WORKER_PORT = 18080
REMOTE_WORKER_VERSION = "stage11_remote_worker_v1"
LOCAL_WORKER_SCRIPT_PATH = Path(__file__).resolve().with_name("stage11_remote_worker.py")

LIVE_VERIFY_MODES = {"off", "remote_dry_run", "remote_verify", "remote_worker"}


REMOTE_VERIFY_SCRIPT = textwrap.dedent(
    """
    import base64
    import importlib.util
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    payload = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    repo_root = Path(payload["remote_repo_root"]).resolve()
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

    user_idx = int(payload["user_idx"])
    score_csv = Path(payload["remote_score_csv"])
    pairwise_dir = Path(payload["remote_pairwise_dir"])
    score_pdf = pd.read_csv(score_csv)
    score_pdf = score_pdf[score_pdf["user_idx"].eq(user_idx)].copy()
    score_pdf["item_idx"] = pd.to_numeric(score_pdf["item_idx"], errors="coerce").fillna(-1).astype(int)
    score_pdf["pre_rank"] = pd.to_numeric(score_pdf["pre_rank"], errors="coerce").fillna(999999).astype(int)
    score_pdf["learned_rank"] = pd.to_numeric(score_pdf["learned_rank"], errors="coerce").fillna(999999).astype(int)
    score_pdf["reward_score"] = pd.to_numeric(score_pdf.get("reward_score"), errors="coerce")
    score_pdf = score_pdf.sort_values(["pre_rank", "item_idx"], kind="stable")
    max_pre_rank = int(payload.get("max_pre_rank", 100) or 100)
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

    prompts = stage11_eval.build_driver_local_listwise_prompts(
        pair_pdf,
        max_rivals=int(payload.get("max_rivals", 11) or 11),
    )
    pair_pdf["prompt"] = prompts.reindex(pair_pdf.index).fillna("").astype(str)

    def route_adapter(pre_rank: int) -> str:
        if 11 <= int(pre_rank) <= 30:
            return "band_11_30"
        if 31 <= int(pre_rank) <= 60:
            return "band_31_60"
        if 61 <= int(pre_rank) <= 100 and str(payload.get("adapter_dir_61_100", "")).strip():
            return "band_61_100"
        return ""

    pair_pdf["sidecar_model_id"] = pair_pdf["pre_rank"].map(route_adapter)
    score_target_pdf = pair_pdf[pair_pdf["sidecar_model_id"].ne("")].copy()
    if score_target_pdf.empty:
        raise RuntimeError(f"no live-score eligible rows found for user_idx={user_idx}")

    base_model = str(payload["base_model"]).strip()
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
        str(payload["adapter_dir_11_30"]).strip(),
        adapter_name="band_11_30",
    )
    adapter_dir_31_60 = str(payload.get("adapter_dir_31_60", "")).strip()
    adapter_dir_61_100 = str(payload.get("adapter_dir_61_100", "")).strip()
    if adapter_dir_31_60:
        model.load_adapter(adapter_dir_31_60, adapter_name="band_31_60")
    if adapter_dir_61_100:
        model.load_adapter(adapter_dir_61_100, adapter_name="band_61_100")

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
            batch_size=int(payload.get("batch_size", 8) or 8),
            max_seq_len=int(payload.get("max_seq_len", 1280) or 1280),
        )
        subset["live_reward_score"] = pd.Series(live_scores.astype(np.float32), index=subset.index)
        rows.append(subset)

    if not rows:
        raise RuntimeError("no live scores produced")

    result_pdf = pd.concat(rows, ignore_index=True)
    result_pdf["abs_diff_vs_replay"] = (
        pd.to_numeric(result_pdf["live_reward_score"], errors="coerce")
        - pd.to_numeric(result_pdf["reward_score"], errors="coerce")
    ).abs()
    result_pdf = result_pdf.sort_values(["abs_diff_vs_replay", "pre_rank", "item_idx"], ascending=[False, True, True])

    summary = {
        "status": "ok",
        "request_id": str(payload.get("request_id", "")),
        "user_idx": user_idx,
        "n_candidates_total": int(pair_pdf.shape[0]),
        "n_scored": int(result_pdf.shape[0]),
        "mean_abs_diff_vs_replay": float(result_pdf["abs_diff_vs_replay"].fillna(0.0).mean()),
        "max_abs_diff_vs_replay": float(result_pdf["abs_diff_vs_replay"].fillna(0.0).max()),
        "scored_rows": [],
    }
    preview_pdf = result_pdf.sort_values(["pre_rank", "item_idx"], kind="stable").head(int(payload.get("preview_rows", 12) or 12))
    for row in preview_pdf.to_dict("records"):
        summary["scored_rows"].append(
            {
                "item_idx": int(row.get("item_idx", -1)),
                "name": str(row.get("name", "") or ""),
                "pre_rank": int(row.get("pre_rank", -1)),
                "learned_rank": int(row.get("learned_rank", -1)),
                "route_band": str(row.get("route_band", "") or ""),
                "sidecar_model_id": str(row.get("sidecar_model_id", "") or ""),
                "replay_reward_score": None if pd.isna(row.get("reward_score")) else float(row.get("reward_score")),
                "live_reward_score": None if pd.isna(row.get("live_reward_score")) else float(row.get("live_reward_score")),
                "abs_diff_vs_replay": None if pd.isna(row.get("abs_diff_vs_replay")) else float(row.get("abs_diff_vs_replay")),
                "prompt_preview": str(row.get("prompt", "") or "")[:220],
            }
        )
    print(json.dumps(summary, ensure_ascii=False))
    """
).strip()


def _maybe_import_paramiko():
    try:
        import paramiko  # type: ignore
    except ImportError as exc:
        raise RuntimeError("paramiko is required for remote Stage11 live verification") from exc
    return paramiko


def resolve_live_mode(payload: dict[str, Any], serving_config: dict[str, Any]) -> str:
    mode = str(
        payload.get(
            "stage11_live_mode",
            serving_config.get("stage11_live_default_mode", "off"),
        )
    ).strip() or "off"
    if mode not in LIVE_VERIFY_MODES:
        raise ValueError(f"unsupported stage11_live_mode: {mode}")
    return mode


def build_stage11_live_plan(
    *,
    replay_result: dict[str, Any],
    payload: dict[str, Any],
    serving_config: dict[str, Any],
    replay_store: ReplayStore,
) -> dict[str, Any]:
    request_context = replay_result.get("request_context", {})
    user_idx = int(request_context.get("user_idx", -1))
    if user_idx < 0:
        raise RuntimeError("replay result missing request_context.user_idx")

    topn = int(payload.get("stage11_live_topn", serving_config.get("stage11_live_topn", 100)) or 100)
    max_pre_rank = int(payload.get("stage11_live_max_pre_rank", topn) or topn)
    max_rivals = int(payload.get("stage11_live_max_rivals", serving_config.get("stage11_live_max_rivals", 11)) or 11)
    preview_rows = int(payload.get("stage11_live_preview_rows", 12) or 12)
    batch_size = int(payload.get("stage11_live_batch_size", serving_config.get("stage11_live_batch_size", 8)) or 8)
    max_seq_len = int(payload.get("stage11_live_max_seq_len", serving_config.get("stage11_live_max_seq_len", 1280)) or 1280)
    timeout_s = int(payload.get("stage11_live_timeout_s", serving_config.get("stage11_live_timeout_s", 1800)) or 1800)
    remote_repo_root = str(
        payload.get("stage11_live_remote_repo_root", os.environ.get("BDA_CLOUD_REPO_ROOT", DEFAULT_REMOTE_REPO_ROOT))
    ).strip()
    remote_python_bin = str(
        payload.get("stage11_live_remote_python_bin", os.environ.get("BDA_CLOUD_PYTHON_BIN", DEFAULT_REMOTE_PYTHON_BIN))
    ).strip()
    host = str(payload.get("stage11_live_host", os.environ.get("BDA_CLOUD_HOST", DEFAULT_HOST))).strip()
    port = int(payload.get("stage11_live_port", os.environ.get("BDA_CLOUD_PORT", DEFAULT_PORT)) or DEFAULT_PORT)
    user = str(payload.get("stage11_live_user", os.environ.get("BDA_CLOUD_USER", DEFAULT_USER))).strip()
    worker_port = int(
        payload.get("stage11_worker_port", serving_config.get("stage11_worker_port", DEFAULT_WORKER_PORT))
        or DEFAULT_WORKER_PORT
    )
    worker_remote_script = str(
        payload.get(
            "stage11_worker_remote_script",
            serving_config.get("stage11_worker_remote_script", "/tmp/bda_stage11_remote_worker.py"),
        )
    ).strip()
    worker_config_path = str(
        payload.get(
            "stage11_worker_config_path",
            serving_config.get("stage11_worker_config_path", f"/tmp/bda_stage11_worker_{worker_port}.json"),
        )
    ).strip()
    worker_pid_path = str(
        payload.get(
            "stage11_worker_pid_path",
            serving_config.get("stage11_worker_pid_path", f"/tmp/bda_stage11_worker_{worker_port}.pid"),
        )
    ).strip()
    worker_log_path = str(
        payload.get(
            "stage11_worker_log_path",
            serving_config.get("stage11_worker_log_path", f"/tmp/bda_stage11_worker_{worker_port}.log"),
        )
    ).strip()

    remote_payload = {
        "request_id": replay_result.get("request_id", ""),
        "user_idx": user_idx,
        "remote_repo_root": remote_repo_root,
        "remote_score_csv": replay_store.remote_score_csv,
        "remote_pairwise_dir": replay_store.remote_pairwise_dir,
        "base_model": replay_store.remote_base_model,
        "adapter_dir_11_30": replay_store.remote_adapter_dir_11_30,
        "adapter_dir_31_60": replay_store.remote_adapter_dir_31_60,
        "adapter_dir_61_100": replay_store.remote_adapter_dir_61_100,
        "max_pre_rank": max_pre_rank,
        "max_rivals": max_rivals,
        "preview_rows": preview_rows,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
    }
    worker_config = {
        "worker_version": REMOTE_WORKER_VERSION,
        "worker_host": "127.0.0.1",
        "worker_port": worker_port,
        **remote_payload,
    }
    payload_b64 = base64.b64encode(json.dumps(remote_payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
    command = "\n".join(
        [
            "set -euo pipefail",
            f"{remote_python_bin} - '{payload_b64}' <<'PY'",
            REMOTE_VERIFY_SCRIPT,
            "PY",
        ]
    )
    return {
        "host": host,
        "port": port,
        "user": user,
        "timeout_s": timeout_s,
        "remote_python_bin": remote_python_bin,
        "remote_repo_root": remote_repo_root,
        "remote_payload": remote_payload,
        "worker_config": worker_config,
        "worker_port": worker_port,
        "worker_remote_script": worker_remote_script,
        "worker_config_path": worker_config_path,
        "worker_pid_path": worker_pid_path,
        "worker_log_path": worker_log_path,
        "command": command,
        "command_preview": f"{remote_python_bin} <inline stage11 live verify> user_idx={user_idx} max_pre_rank={max_pre_rank}",
    }


def _remote_password(payload: dict[str, Any]) -> str:
    password = str(payload.get("stage11_live_password", os.environ.get("BDA_CLOUD_PASSWORD", ""))).strip()
    if not password:
        raise RuntimeError("stage11_live_password or BDA_CLOUD_PASSWORD is required for remote Stage11 live mode")
    return password


def _connect_ssh(plan: dict[str, Any], payload: dict[str, Any]):
    paramiko = _maybe_import_paramiko()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=str(plan["host"]),
        port=int(plan["port"]),
        username=str(plan["user"]),
        password=_remote_password(payload),
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def _exec_ssh_json(client: Any, command: str, timeout_s: int) -> dict[str, Any]:
    stdin, stdout, stderr = client.exec_command(command, timeout=int(timeout_s))
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(err or out or f"remote command failed with exit_code={code}")
    if not out:
        raise RuntimeError("remote command produced empty stdout")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        for line in reversed(out.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"remote command returned non-json stdout: {out[:800]}")


def _build_worker_http_command(
    *,
    remote_python_bin: str,
    worker_port: int,
    path: str,
    payload: dict[str, Any] | None,
    timeout_s: int,
) -> str:
    request = {
        "url": f"http://127.0.0.1:{int(worker_port)}{path}",
        "payload": payload,
        "timeout_s": int(timeout_s),
    }
    request_b64 = base64.b64encode(json.dumps(request, ensure_ascii=False).encode("utf-8")).decode("ascii")
    script = textwrap.dedent(
        """
        import base64
        import json
        import sys
        import urllib.error
        import urllib.request

        request = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
        data = None
        method = "GET"
        headers = {}
        if request.get("payload") is not None:
            data = json.dumps(request["payload"], ensure_ascii=False).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(request["url"], data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=float(request.get("timeout_s", 30))) as response:
                print(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(body or json.dumps({"status": "error", "detail": str(exc)}))
        """
    ).strip()
    return "\n".join([f"{remote_python_bin} - '{request_b64}' <<'PY'", script, "PY"])


def _query_stage11_worker(plan: dict[str, Any], payload: dict[str, Any], *, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    command = _build_worker_http_command(
        remote_python_bin=str(plan["remote_python_bin"]),
        worker_port=int(plan["worker_port"]),
        path=path,
        payload=body,
        timeout_s=int(payload.get("stage11_worker_http_timeout_s", 120)),
    )
    client = _connect_ssh(plan, payload)
    try:
        return _exec_ssh_json(client, command, timeout_s=int(payload.get("stage11_worker_ssh_timeout_s", 180)))
    finally:
        client.close()


def _upload_stage11_worker_files(client: Any, plan: dict[str, Any]) -> None:
    if not LOCAL_WORKER_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"local stage11 worker script missing: {LOCAL_WORKER_SCRIPT_PATH}")
    sftp = client.open_sftp()
    try:
        sftp.put(str(LOCAL_WORKER_SCRIPT_PATH), str(plan["worker_remote_script"]))
        with sftp.file(str(plan["worker_config_path"]), "w") as handle:
            handle.write(json.dumps(plan["worker_config"], ensure_ascii=False, indent=2))
    finally:
        sftp.close()


def ensure_stage11_worker(plan: dict[str, Any], payload: dict[str, Any], serving_config: dict[str, Any]) -> dict[str, Any]:
    expected_version = REMOTE_WORKER_VERSION
    try:
        health = _query_stage11_worker(plan, payload, path="/health")
        if health.get("status") == "ok" and health.get("worker_version") == expected_version:
            health["worker_started"] = False
            return health
    except Exception:
        health = {}

    startup_timeout_s = int(
        payload.get("stage11_worker_startup_timeout_s", serving_config.get("stage11_worker_startup_timeout_s", 300))
        or 300
    )
    poll_interval_s = float(
        payload.get("stage11_worker_poll_interval_s", serving_config.get("stage11_worker_poll_interval_s", 5.0))
        or 5.0
    )
    client = _connect_ssh(plan, payload)
    try:
        _upload_stage11_worker_files(client, plan)
        stop_command = (
            f"if [ -f {shlex.quote(str(plan['worker_pid_path']))} ]; then "
            f"kill $(cat {shlex.quote(str(plan['worker_pid_path']))}) >/dev/null 2>&1 || true; "
            "sleep 2; fi"
        )
        client.exec_command(stop_command, timeout=30)[1].channel.recv_exit_status()
        start_command = " ".join(
            [
                "nohup",
                shlex.quote(str(plan["remote_python_bin"])),
                shlex.quote(str(plan["worker_remote_script"])),
                "--config",
                shlex.quote(str(plan["worker_config_path"])),
                ">",
                shlex.quote(str(plan["worker_log_path"])),
                "2>&1",
                "&",
                "echo",
                "$!",
                ">",
                shlex.quote(str(plan["worker_pid_path"])),
            ]
        )
        stdin, stdout, stderr = client.exec_command(start_command, timeout=30)
        stdout.channel.recv_exit_status()
    finally:
        client.close()

    deadline = time.perf_counter() + startup_timeout_s
    last_error = ""
    while time.perf_counter() < deadline:
        time.sleep(poll_interval_s)
        try:
            health = _query_stage11_worker(plan, payload, path="/health")
            if health.get("status") == "ok" and health.get("worker_version") == expected_version:
                health["worker_started"] = True
                health["log_path"] = str(plan["worker_log_path"])
                return health
            last_error = json.dumps(health, ensure_ascii=False)
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(
        f"stage11 remote worker did not become ready within {startup_timeout_s}s; "
        f"last_error={last_error}; log_path={plan['worker_log_path']}"
    )


def execute_stage11_worker_plan(plan: dict[str, Any], payload: dict[str, Any], serving_config: dict[str, Any]) -> dict[str, Any]:
    health = ensure_stage11_worker(plan, payload, serving_config)
    result = _query_stage11_worker(plan, payload, path="/score", body=plan["remote_payload"])
    result["mode"] = "remote_worker"
    result["worker_health"] = health
    result["command_preview"] = (
        f"remote worker http://127.0.0.1:{int(plan['worker_port'])}/score "
        f"user_idx={plan['remote_payload'].get('user_idx')}"
    )
    result["host"] = str(plan["host"])
    result["port"] = int(plan["port"])
    result["worker_port"] = int(plan["worker_port"])
    return result


def execute_stage11_live_plan(plan: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    client = _connect_ssh(plan, payload)
    try:
        stdin, stdout, stderr = client.exec_command(str(plan["command"]), timeout=int(plan["timeout_s"]))
        out = stdout.read().decode("utf-8", errors="replace").strip()
        err = stderr.read().decode("utf-8", errors="replace").strip()
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()

    if code != 0:
        raise RuntimeError(f"remote stage11 live verify failed: {err or out or f'exit_code={code}'}")
    if not out:
        raise RuntimeError("remote stage11 live verify produced empty stdout")

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        parsed = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            raise RuntimeError(f"remote stage11 live verify returned non-json stdout: {out[:800]}")
    parsed["mode"] = "remote_verify"
    parsed["command_preview"] = str(plan["command_preview"])
    parsed["host"] = str(plan["host"])
    parsed["port"] = int(plan["port"])
    return parsed


def maybe_run_stage11_live_verify(
    *,
    replay_result: dict[str, Any],
    payload: dict[str, Any],
    serving_config: dict[str, Any],
    replay_store: ReplayStore,
) -> dict[str, Any]:
    mode = resolve_live_mode(payload, serving_config)
    if mode == "off":
        return {"enabled": False, "mode": "off", "status": "skipped"}
    if str(replay_result.get("strategy_used", "")).strip() != "reward_rerank":
        return {
            "enabled": True,
            "mode": mode,
            "status": "skipped",
            "reason": "stage11 live verify only applies when strategy_used=reward_rerank",
        }

    plan = build_stage11_live_plan(
        replay_result=replay_result,
        payload=payload,
        serving_config=serving_config,
        replay_store=replay_store,
    )
    if mode == "remote_dry_run":
        return {
            "enabled": True,
            "mode": mode,
            "status": "dry_run",
            "command_preview": plan["command_preview"],
            "remote_payload": plan["remote_payload"],
            "host": plan["host"],
            "port": plan["port"],
            "user": plan["user"],
            "timeout_s": plan["timeout_s"],
        }
    if mode == "remote_worker":
        return execute_stage11_worker_plan(plan, payload, serving_config)
    return execute_stage11_live_plan(plan, payload)
