#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage11_bucket5_pool_audit.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/../launchers/_path_contract.sh"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [[ -x "${BDA_REMOTE_PYTHON_BIN}" ]]; then
    PYTHON_BIN="${BDA_REMOTE_PYTHON_BIN}"
  else
    echo "[ERROR] python executable not found" >&2
    exit 1
  fi
fi

TARGET_BAND="${TARGET_BAND:-rescue_31_60}"
STAGE11_RUN_DIR="${STAGE11_RUN_DIR:-${BDA_REMOTE_FAST_ROOT}/bucket5_top250_semantic_compact_boundary_reason_rm_11_1/output/11_qlora_data/20260406_034859_stage11_1_qlora_build_dataset}"
OUTPUT_JSON="${OUTPUT_JSON:-${BDA_REMOTE_FAST_ROOT}/_audits/stage11_1_pool_${TARGET_BAND}_v88.json}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/audit_stage11_1_band_pool.py" \
  --stage11-run-dir "${STAGE11_RUN_DIR}" \
  --target-band "${TARGET_BAND}" \
  --output "${OUTPUT_JSON}"
