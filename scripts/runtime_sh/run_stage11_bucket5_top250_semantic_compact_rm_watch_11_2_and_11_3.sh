#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage11_bucket5_watch.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/../launchers/_path_contract.sh"
PYTHON_BIN="${PYTHON_BIN:-${BDA_REMOTE_PYTHON_BIN}}"

WATCH_11_2_RUN_DIR="${WATCH_11_2_RUN_DIR:-}"
FAST_ROOT_11_3="${FAST_ROOT_11_3:-${BDA_REMOTE_FAST_ROOT}/bucket5_top250_semantic_compact_boundary_reason_rm_eval_westd}"
OUTPUT_11_SIDECAR_ROOT_DIR="${OUTPUT_11_SIDECAR_ROOT_DIR:-${FAST_ROOT_11_3}/output/11_qlora_sidecar_eval}"
METRICS_STAGE11_SIDECAR_PATH="${METRICS_STAGE11_SIDECAR_PATH:-${FAST_ROOT_11_3}/metrics/recsys_stage11_bucket5_top250_semantic_compact_boundary_rm_eval.csv}"
OUTPUT_11_MODELS_ROOT_DIR="${OUTPUT_11_MODELS_ROOT_DIR:-${BDA_REMOTE_FAST_ROOT}/bucket5_top250_semantic_compact_boundary_reason_rm_train/output/11_qlora_models}"
INPUT_11_DATA_RUN_DIR="${INPUT_11_DATA_RUN_DIR:-}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/run_stage11_bucket5_top250_semantic_compact_rm_eval_only.sh}"
WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-30}"
WATCH_GATE_ENABLE="${WATCH_GATE_ENABLE:-auto}"
WATCH_GATE_31_40_MIN_COUNT="${WATCH_GATE_31_40_MIN_COUNT:-0}"
WATCH_GATE_31_40_MIN_WIN_RATE="${WATCH_GATE_31_40_MIN_WIN_RATE:-0}"
WATCH_GATE_41_60_MIN_COUNT="${WATCH_GATE_41_60_MIN_COUNT:-0}"
WATCH_GATE_41_60_MIN_WIN_RATE="${WATCH_GATE_41_60_MIN_WIN_RATE:-0}"

echo "[WATCH] bucket5 semantic_compact_boundary_rm 11_2 -> 11_3 watcher"
echo "[WATCH] repo=${REPO_ROOT}"
echo "[WATCH] watch_11_2_run_dir=${WATCH_11_2_RUN_DIR:-<latest>}"
echo "[WATCH] input_11_data_run_dir=${INPUT_11_DATA_RUN_DIR:-<unset>}"
echo "[WATCH] output_11_sidecar_root_dir=${OUTPUT_11_SIDECAR_ROOT_DIR}"
echo "[WATCH] gate_enable=${WATCH_GATE_ENABLE}"

resolve_run_dir() {
  if [[ -n "${WATCH_11_2_RUN_DIR}" ]]; then
    echo "${WATCH_11_2_RUN_DIR}"
    return
  fi
  find "${OUTPUT_11_MODELS_ROOT_DIR}" -mindepth 1 -maxdepth 1 -type d -name '*_stage11_2_rm_train' | sort | tail -n 1
}

gate_run_meta() {
  local run_meta_path="$1"
  "${PYTHON_BIN}" - "${run_meta_path}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

enable = str(os.getenv("WATCH_GATE_ENABLE", "auto")).strip().lower()
target_true_bands = str(os.getenv("QLORA_TARGET_TRUE_BANDS", "")).strip()

if enable == "false":
    print("[WATCH] gate disabled")
    raise SystemExit(0)

gate_keys = [
    "eval_rescue_31_40_true_count",
    "eval_rescue_31_40_true_win_rate",
    "eval_rescue_41_60_true_count",
    "eval_rescue_41_60_true_win_rate",
]

if enable == "auto" and target_true_bands != "rescue_31_60":
    print(f"[WATCH] gate skipped target_true_bands={target_true_bands or '<unset>'}")
    raise SystemExit(0)

for key in gate_keys:
    print(f"[WATCH] {key}={data.get(key)!r}")

missing = [key for key in gate_keys if data.get(key) is None]
if missing:
    print(f"[WATCH] gate missing metrics={','.join(missing)}", file=sys.stderr)
    raise SystemExit(2)

def _as_float(name: str) -> float:
    value = data.get(name)
    if value is None:
        raise ValueError(name)
    return float(value)

checks = [
    ("WATCH_GATE_31_40_MIN_COUNT", "eval_rescue_31_40_true_count"),
    ("WATCH_GATE_31_40_MIN_WIN_RATE", "eval_rescue_31_40_true_win_rate"),
    ("WATCH_GATE_41_60_MIN_COUNT", "eval_rescue_41_60_true_count"),
    ("WATCH_GATE_41_60_MIN_WIN_RATE", "eval_rescue_41_60_true_win_rate"),
]

for env_name, metric_name in checks:
    threshold = float(str(os.getenv(env_name, "0")).strip() or 0.0)
    if threshold <= 0.0:
        continue
    value = _as_float(metric_name)
    if value < threshold:
        print(
            f"[WATCH] gate failed {metric_name}={value:.6f} < {threshold:.6f}",
            file=sys.stderr,
        )
        raise SystemExit(3)

print("[WATCH] gate passed")
raise SystemExit(0)
PY
}

while true; do
  run_dir="$(resolve_run_dir || true)"
  if [[ -n "${run_dir}" && -f "${run_dir}/run_meta.json" ]]; then
    echo "[WATCH] detected completed 11_2 output at $(date '+%F %T')"
    echo "[WATCH] input_11_2_run_dir=${run_dir}"
    if ! gate_run_meta "${run_dir}/run_meta.json"; then
      rc=$?
      if [[ "${rc}" == "2" || "${rc}" == "3" ]]; then
        echo "[WATCH] gate failed; skip 11_3"
        exit 0
      fi
      echo "[WATCH][ERROR] gate check failed rc=${rc}" >&2
      exit "${rc}"
    fi
    if pgrep -f '11_3_qlora_sidecar_eval.py' >/dev/null; then
      echo "[WATCH] 11_3 already running; exit"
      exit 0
    fi
    mkdir -p "${FAST_ROOT_11_3}" "${OUTPUT_11_SIDECAR_ROOT_DIR}" "$(dirname "${METRICS_STAGE11_SIDECAR_PATH}")"
    export REPO_ROOT
    export PYTHON_BIN
    export INPUT_11_2_RUN_DIR="${run_dir}"
    export INPUT_11_DATA_RUN_DIR
    export FAST_ROOT="${FAST_ROOT_11_3}"
    export OUTPUT_11_SIDECAR_ROOT_DIR
    export METRICS_STAGE11_SIDECAR_PATH
    export QLORA_PROMPT_MODE="${QLORA_PROMPT_MODE:-local_listwise_compare_rm}"
    export QLORA_EVAL_MODEL_TYPE="${QLORA_EVAL_MODEL_TYPE:-rm}"
    export QLORA_EVAL_TARGET_TRUE_BANDS="${QLORA_EVAL_TARGET_TRUE_BANDS:-boundary_11_30}"
    export QLORA_RERANK_TOPN="${QLORA_RERANK_TOPN:-50}"
    export QLORA_EVAL_PROMPT_CHUNK_ROWS="${QLORA_EVAL_PROMPT_CHUNK_ROWS:-8192}"
    export QLORA_EVAL_STREAM_LOG_ROWS="${QLORA_EVAL_STREAM_LOG_ROWS:-8192}"
    export QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS="${QLORA_EVAL_INTERMEDIATE_FLUSH_ROWS:-8192}"
    export QLORA_EVAL_DRIVER_PROMPT_IMPL="${QLORA_EVAL_DRIVER_PROMPT_IMPL:-itertuples}"
    export QLORA_EVAL_ARROW_TO_PANDAS="${QLORA_EVAL_ARROW_TO_PANDAS:-false}"
    export QLORA_EVAL_ARROW_FALLBACK="${QLORA_EVAL_ARROW_FALLBACK:-false}"
    export QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK="${QLORA_EVAL_PRETOKENIZE_PROMPT_CHUNK:-true}"
    export QLORA_EVAL_GPU_PRELOAD_PROMPT_CHUNK="${QLORA_EVAL_GPU_PRELOAD_PROMPT_CHUNK:-true}"
    export QLORA_EVAL_BUCKET_SORT_PROMPT_CHUNK="${QLORA_EVAL_BUCKET_SORT_PROMPT_CHUNK:-true}"
    export QLORA_EVAL_LOCAL_TRIM_PROMPT_BATCH="${QLORA_EVAL_LOCAL_TRIM_PROMPT_BATCH:-true}"
    echo "[WATCH] launch 11_3 RM eval at $(date '+%F %T')"
    exec "${EVAL_SCRIPT}"
  fi

  if ! pgrep -f '11_2_rm_train.py' >/dev/null; then
    echo "[WATCH][ERROR] 11_2 exited before any completed output appeared at $(date '+%F %T')" >&2
    exit 1
  fi

  sleep "${WATCH_INTERVAL_SECONDS}"
done
