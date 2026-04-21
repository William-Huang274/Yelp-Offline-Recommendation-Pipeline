#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage09_bucket5_typed_intent_assets.sh.

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

STAGE09_RUN_DIR="${STAGE09_RUN_DIR:-${REPO_ROOT}/data/output/09_candidate_fusion_structural_v5_sourceparity/20260324_030511_full_stage09_candidate_fusion}"
STAGE09_BUCKET="${STAGE09_BUCKET:-5}"
STAGE09_BUCKET_DIR="${STAGE09_BUCKET_DIR:-${STAGE09_RUN_DIR}/bucket_${STAGE09_BUCKET}}"

USER_INTENT_OUTPUT_ROOT="${USER_INTENT_OUTPUT_ROOT:-${REPO_ROOT}/data/output/09_user_intent_profile_v2_bucket5_sourceparity}"
MATCH_FEATURE_OUTPUT_ROOT="${MATCH_FEATURE_OUTPUT_ROOT:-${REPO_ROOT}/data/output/09_user_business_match_features_v2_bucket5_sourceparity}"
MATCH_CHANNEL_OUTPUT_ROOT="${MATCH_CHANNEL_OUTPUT_ROOT:-${REPO_ROOT}/data/output/09_user_business_match_channels_v2_bucket5_sourceparity}"

mkdir -p "${USER_INTENT_OUTPUT_ROOT}" "${MATCH_FEATURE_OUTPUT_ROOT}" "${MATCH_CHANNEL_OUTPUT_ROOT}"

if [[ ! -d "${STAGE09_BUCKET_DIR}" ]]; then
  echo "[ERROR] stage09 bucket dir not found: ${STAGE09_BUCKET_DIR}" >&2
  exit 1
fi
if [[ ! -e "${STAGE09_BUCKET_DIR}/train_history.parquet" ]]; then
  echo "[ERROR] train_history.parquet missing under ${STAGE09_BUCKET_DIR}" >&2
  exit 1
fi

latest_run_dir() {
  local root="$1"
  local latest
  if [[ ! -d "${root}" ]]; then
    return 1
  fi
  latest="$(find "${root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '%s\n' "${latest}"
}

export BDA_PROJECT_ROOT="${REPO_ROOT}"
export INPUT_09_STAGE09_BUCKET_DIR="${STAGE09_BUCKET_DIR}"
export USER_INTENT_PROFILE_V2_SPLIT_AWARE_HISTORY_ONLY="${USER_INTENT_PROFILE_V2_SPLIT_AWARE_HISTORY_ONLY:-true}"
export USER_INTENT_PROFILE_V2_SPLIT_AWARE_BUCKET="${USER_INTENT_PROFILE_V2_SPLIT_AWARE_BUCKET:-${STAGE09_BUCKET}}"
export OUTPUT_09_USER_INTENT_PROFILE_V2_ROOT_DIR="${USER_INTENT_OUTPUT_ROOT}"

echo "[INFO] bucket5 typed-intent asset build"
echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] stage09_bucket_dir=${STAGE09_BUCKET_DIR}"
echo "[INFO] user_intent_output_root=${USER_INTENT_OUTPUT_ROOT}"
echo "[INFO] match_feature_output_root=${MATCH_FEATURE_OUTPUT_ROOT}"
echo "[INFO] match_channel_output_root=${MATCH_CHANNEL_OUTPUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/09_user_intent_profile_v2_build.py"

USER_INTENT_RUN_DIR="$(latest_run_dir "${USER_INTENT_OUTPUT_ROOT}")"
if [[ -z "${USER_INTENT_RUN_DIR}" || ! -f "${USER_INTENT_RUN_DIR}/user_intent_profile_v2.parquet" ]]; then
  echo "[ERROR] user intent output not found under ${USER_INTENT_OUTPUT_ROOT}" >&2
  exit 1
fi

export INPUT_09_USER_INTENT_PROFILE_V2_RUN_DIR="${USER_INTENT_RUN_DIR}"
export USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_HISTORY_ONLY="${USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_HISTORY_ONLY:-true}"
export USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_BUCKET="${USER_BUSINESS_MATCH_FEATURES_V2_SPLIT_AWARE_BUCKET:-${STAGE09_BUCKET}}"
export OUTPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_ROOT_DIR="${MATCH_FEATURE_OUTPUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/09_user_business_match_features_v2_build.py"

MATCH_FEATURE_RUN_DIR="$(latest_run_dir "${MATCH_FEATURE_OUTPUT_ROOT}")"
if [[ -z "${MATCH_FEATURE_RUN_DIR}" || ! -f "${MATCH_FEATURE_RUN_DIR}/user_business_match_features_v2_user_item.parquet" ]]; then
  echo "[ERROR] match feature output not found under ${MATCH_FEATURE_OUTPUT_ROOT}" >&2
  exit 1
fi

export INPUT_09_USER_BUSINESS_MATCH_FEATURES_V2_RUN_DIR="${MATCH_FEATURE_RUN_DIR}"
export USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_HISTORY_ONLY="${USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_HISTORY_ONLY:-true}"
export USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_BUCKET="${USER_BUSINESS_MATCH_CHANNELS_V2_SPLIT_AWARE_BUCKET:-${STAGE09_BUCKET}}"
export OUTPUT_09_USER_BUSINESS_MATCH_CHANNELS_V2_ROOT_DIR="${MATCH_CHANNEL_OUTPUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/09_user_business_match_channels_v2_build.py"

MATCH_CHANNEL_RUN_DIR="$(latest_run_dir "${MATCH_CHANNEL_OUTPUT_ROOT}")"
if [[ -z "${MATCH_CHANNEL_RUN_DIR}" || ! -f "${MATCH_CHANNEL_RUN_DIR}/user_business_match_channels_v2_user_item.parquet" ]]; then
  echo "[ERROR] match channel output not found under ${MATCH_CHANNEL_OUTPUT_ROOT}" >&2
  exit 1
fi

echo "[INFO] user_intent_run_dir=${USER_INTENT_RUN_DIR}"
echo "[INFO] match_feature_run_dir=${MATCH_FEATURE_RUN_DIR}"
echo "[INFO] match_channel_run_dir=${MATCH_CHANNEL_RUN_DIR}"
