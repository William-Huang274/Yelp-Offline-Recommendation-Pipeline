#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/runtime_sh/run_stage11_persona_sft_v3_moe_handoff.sh"
  echo "Runs the Stage11 persona SFT v3 model handoff controller. This is a cloud/GPU orchestration job."
  echo "Set remote model and project paths, then run without --help."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${SCRIPT_DIR}/../launchers/_path_contract.sh"

LOG_DIR="${LOG_DIR:-${BDA_REMOTE_PROJECT_ROOT}/tmp}"
mkdir -p "${LOG_DIR}"
CTRL_LOG="${LOG_DIR}/run_stage11_persona_sft_v3_moe_handoff.log"
SENTINEL_35_READY="${LOG_DIR}/qwen35_35b_a3b_ready.flag"
WAIT_FOR_35_PID=""

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${CTRL_LOG}"
}

model_ready_30() {
  local model_dir="/root/autodl-tmp/models/Qwen3-30B-A3B-Base"
  [[ -d "${model_dir}" ]] || return 1
  local shard_count
  shard_count=$(find "${model_dir}" -maxdepth 1 -type f -name 'model*.safetensors' | wc -l)
  [[ "${shard_count}" -ge 16 ]] && [[ -f "${model_dir}/config.json" ]]
}

model_ready_35() {
  local model_dir="/root/autodl-tmp/models/Qwen3.5-35B-A3B-Base"
  [[ -d "${model_dir}" ]] || return 1
  local shard_count
  shard_count=$(find "${model_dir}" -maxdepth 1 -type f -name 'model*.safetensors' | wc -l)
  [[ "${shard_count}" -ge 14 ]] && [[ -f "${model_dir}/config.json" ]]
}

wait_for_35_in_background() {
  rm -f "${SENTINEL_35_READY}"
  (
    while true; do
      if model_ready_35; then
        touch "${SENTINEL_35_READY}"
        log "Qwen3.5-35B-A3B-Base download is complete."
        exit 0
      fi
      sleep 60
    done
  ) >> "${CTRL_LOG}" 2>&1 &
  WAIT_FOR_35_PID="$!"
}

stop_current_persona_train() {
  pkill -f '/root/autodl-tmp/5006_BDA_project/scripts/11_2_persona_sft_train.py' >/dev/null 2>&1 || true
  sleep 5
}

run_probe_with_backoff() {
  local label="$1"
  local launcher="$2"
  local max_steps="$3"
  local seq_candidates="$4"
  local lora_r="$5"
  local base_log_prefix="${LOG_DIR}/${label}"

  IFS=',' read -r -a seqs <<< "${seq_candidates}"
  for seq in "${seqs[@]}"; do
    local run_log="${base_log_prefix}_seq${seq}.log"
    log "Starting ${label} with max_seq_len=${seq}, lora_r=${lora_r}, max_steps=${max_steps}"
    set +e
    env \
      PERSONA_SFT_MAX_SEQ_LEN="${seq}" \
      PERSONA_SFT_MAX_STEPS="${max_steps}" \
      PERSONA_SFT_LORA_R="${lora_r}" \
      PERSONA_SFT_LORA_ALPHA="$((lora_r * 2))" \
      bash "${launcher}" > "${run_log}" 2>&1
    local rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      log "${label} finished successfully at max_seq_len=${seq}"
      return 0
    fi
    if grep -Eiq 'CUDA out of memory|OutOfMemoryError|out of memory' "${run_log}"; then
      log "${label} hit OOM at max_seq_len=${seq}; retrying with a safer context length."
      continue
    fi
    log "${label} failed with non-OOM exit code ${rc}. Check ${run_log}"
    return ${rc}
  done
  log "${label} exhausted all seq candidates without a successful run."
  return 1
}

main() {
  log "MOE handoff controller starting."
  stop_current_persona_train

  if ! model_ready_30; then
    log "Qwen3-30B-A3B-Base is not fully present; aborting."
    exit 1
  fi

  wait_for_35_in_background
  log "Spawned 35B download monitor pid=${WAIT_FOR_35_PID}"

  run_probe_with_backoff \
    "persona_sft_qwen3_30b_a3b_probe" \
    "${SCRIPT_DIR}/run_stage11_persona_sft_v3_qwen3_30b_a3b_probe.sh" \
    "75" \
    "6144,5120,4096" \
    "16"

  if [[ ! -f "${SENTINEL_35_READY}" ]]; then
    log "Waiting for Qwen3.5-35B-A3B-Base download to finish."
    wait "${WAIT_FOR_35_PID}"
  fi

  run_probe_with_backoff \
    "persona_sft_qwen35_35b_a3b_probe" \
    "${SCRIPT_DIR}/run_stage11_persona_sft_v3_qwen35_35b_a3b_probe.sh" \
    "75" \
    "6144,5120,4096" \
    "16"

  log "MOE handoff controller completed."
}

main "$@"
