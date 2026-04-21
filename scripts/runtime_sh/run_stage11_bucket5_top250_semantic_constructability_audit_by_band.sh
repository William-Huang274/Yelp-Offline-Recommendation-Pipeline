#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage11_bucket5_constructability_audit.sh.

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
STAGE09_RUN_DIR="${STAGE09_RUN_DIR:-}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-${BDA_REMOTE_FAST_ROOT}/_audits/constructability_${TARGET_BAND}.json}"

if [[ -z "${STAGE09_RUN_DIR}" ]]; then
  echo "[ERROR] STAGE09_RUN_DIR is required" >&2
  exit 1
fi

ARGS=(
  --stage09-run-dir "${STAGE09_RUN_DIR}"
  --target-band "${TARGET_BAND}"
  --output "${OUTPUT_JSON}"
)
if [[ -n "${SOURCE_RUN_DIR}" ]]; then
  ARGS+=(--source-run-dir "${SOURCE_RUN_DIR}")
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/audit_stage11_boundary_constructability.py" "${ARGS[@]}"
