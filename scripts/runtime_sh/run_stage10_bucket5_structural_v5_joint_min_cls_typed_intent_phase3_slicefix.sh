#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage10_bucket5_mainline.sh.

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

STAGE09_RUN_DIR="${STAGE09_RUN_DIR:-${REPO_ROOT}/data/output/09_candidate_fusion_structural_v5_sourceparity/20260324_030511_full_stage09_candidate_fusion}"
TEXT_MATCH_RUN_DIR="${TEXT_MATCH_RUN_DIR:-${REPO_ROOT}/data/output/09_candidate_wise_text_match_features_v1/20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build}"
GROUP_GAP_RUN_DIR="${GROUP_GAP_RUN_DIR:-${REPO_ROOT}/data/output/09_stage10_group_gap_features_v1/20260323_174757_full_stage09_stage10_group_gap_features_v1_build}"
USER_INTENT_ROOT="${USER_INTENT_ROOT:-${REPO_ROOT}/data/output/09_user_intent_profile_v2_bucket5_sourceparity}"
MATCH_CHANNEL_ROOT="${MATCH_CHANNEL_ROOT:-${REPO_ROOT}/data/output/09_user_business_match_channels_v2_bucket5_sourceparity}"
PROFILE_RUN_DIR="${PROFILE_RUN_DIR:-}"
MATCH_CHANNEL_RUN_DIR="${MATCH_CHANNEL_RUN_DIR:-}"
USER_SCHEMA_RUN_DIR="${USER_SCHEMA_RUN_DIR:-}"

if [[ -z "${PROFILE_RUN_DIR}" ]]; then
  PROFILE_RUN_DIR="$(latest_run_dir "${USER_INTENT_ROOT}")"
fi
if [[ -z "${MATCH_CHANNEL_RUN_DIR}" ]]; then
  MATCH_CHANNEL_RUN_DIR="$(latest_run_dir "${MATCH_CHANNEL_ROOT}")"
fi
if [[ -z "${PROFILE_RUN_DIR}" || ! -f "${PROFILE_RUN_DIR}/user_intent_profile_v2.parquet" ]]; then
  echo "[ERROR] profile run dir missing or invalid: ${PROFILE_RUN_DIR}" >&2
  exit 1
fi
if [[ -z "${MATCH_CHANNEL_RUN_DIR}" || ! -f "${MATCH_CHANNEL_RUN_DIR}/user_business_match_channels_v2_user_item.parquet" ]]; then
  echo "[ERROR] match channel run dir missing or invalid: ${MATCH_CHANNEL_RUN_DIR}" >&2
  exit 1
fi

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${REPO_ROOT}/data/output/10_rank_models_joint_min_cls_v5_typed_intent_phase3_slicefix}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-${REPO_ROOT}/data/output/10_2_rank_infer_eval_joint_min_cls_v5_typed_intent_phase3_slicefix}"
FOCUS_EVAL_OUTPUT_ROOT="${FOCUS_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/output/10_4_bucket5_focus_slice_eval_typed_phase3_slicefix}"
METRICS_PATH="${METRICS_PATH:-${REPO_ROOT}/data/metrics/recsys_stage10_results_joint_min_cls_v5_typed_intent_phase3_slicefix.csv}"

SPARK_LOCAL_DIR_DEFAULT="${REPO_ROOT}/data/spark-tmp"
mkdir -p "${TRAIN_OUTPUT_ROOT}" "${INFER_OUTPUT_ROOT}" "${FOCUS_EVAL_OUTPUT_ROOT}" "$(dirname "${METRICS_PATH}")" "${SPARK_LOCAL_DIR_DEFAULT}"

export BDA_PROJECT_ROOT="${REPO_ROOT}"
export INPUT_09_RUN_DIR="${STAGE09_RUN_DIR}"
export INPUT_09_TEXT_MATCH_RUN_DIR="${TEXT_MATCH_RUN_DIR}"
export INPUT_09_GROUP_GAP_RUN_DIR="${GROUP_GAP_RUN_DIR}"
export INPUT_09_MATCH_CHANNELS_RUN_DIR="${MATCH_CHANNEL_RUN_DIR}"
export OUTPUT_10_1_ROOT_DIR="${TRAIN_OUTPUT_ROOT}"
export OUTPUT_10_2_ROOT_DIR="${INFER_OUTPUT_ROOT}"
export STAGE10_RESULTS_METRICS_PATH="${METRICS_PATH}"
export RANK_EVAL_USER_COHORT_PATH="${RANK_EVAL_USER_COHORT_PATH:-${REPO_ROOT}/data/output/fixed_eval_cohorts/bucket5_accepted_test_users_1935_userid.csv}"
export RANK_BLEND_ALPHA="${RANK_BLEND_ALPHA:-0.15}"

export TRAIN_BUCKETS_OVERRIDE="${TRAIN_BUCKETS_OVERRIDE:-5}"
export RANK_BUCKETS_OVERRIDE="${RANK_BUCKETS_OVERRIDE:-5}"
export TRAIN_MODEL_BACKEND="${TRAIN_MODEL_BACKEND:-xgboost_cls}"
export STAGE10_CLOUD_FAST_MODE="${STAGE10_CLOUD_FAST_MODE:-true}"
export TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS="${TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS:-false}"
export RANK_DIAGNOSTICS_ENABLE="${RANK_DIAGNOSTICS_ENABLE:-true}"
export SPARK_SQL_ADAPTIVE_ENABLED="${SPARK_SQL_ADAPTIVE_ENABLED:-true}"
export SPARK_SQL_PARQUET_ENABLE_VECTORIZED="${SPARK_SQL_PARQUET_ENABLE_VECTORIZED:-true}"
export SPARK_SQL_FILES_MAX_PARTITION_BYTES="${SPARK_SQL_FILES_MAX_PARTITION_BYTES:-128m}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export ENABLE_STAGE10_V2_MATCH_CHANNELS="${ENABLE_STAGE10_V2_MATCH_CHANNELS:-true}"
export STAGE10_V2_MATCH_CHANNEL_BUCKETS="${STAGE10_V2_MATCH_CHANNEL_BUCKETS:-5}"
export STAGE10_V2_MATCH_CHANNEL_FEATURES="${STAGE10_V2_MATCH_CHANNEL_FEATURES:-channel_preference_core_uaware_v2,channel_recent_intent_uaware_v2,channel_context_time_uaware_v2,channel_conflict_v1,channel_evidence_support_v1,channel_typed_exact_match_uaware_v1,channel_typed_prior_trust_v2,channel_uncertainty_confidence_v1,channel_uncertainty_conflict_v1,channel_uncertainty_freshness_v1}"

export ENABLE_STAGE10_V2_TEXT_MATCH="${ENABLE_STAGE10_V2_TEXT_MATCH:-true}"
export STAGE10_V2_TEXT_MATCH_BUCKETS="${STAGE10_V2_TEXT_MATCH_BUCKETS:-5}"
export STAGE10_V2_TEXT_MATCH_FEATURES="${STAGE10_V2_TEXT_MATCH_FEATURES:-sim_negative_avoid_neg,sim_negative_avoid_core,sim_conflict_gap}"

export ENABLE_STAGE10_V2_GROUP_GAP="${ENABLE_STAGE10_V2_GROUP_GAP:-true}"
export STAGE10_V2_GROUP_GAP_BUCKETS="${STAGE10_V2_GROUP_GAP_BUCKETS:-5}"
export STAGE10_V2_GROUP_GAP_FEATURES="${STAGE10_V2_GROUP_GAP_FEATURES:-schema_weighted_overlap_user_ratio_v2_rank_pct_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3,schema_weighted_net_score_v2_rank_pct_v3,schema_top_pop_low_flag_v3}"

export RANK_WRITE_USER_AUDIT="${RANK_WRITE_USER_AUDIT:-true}"
export RUN_BUCKET5_TYPED_FOCUS_EVAL="${RUN_BUCKET5_TYPED_FOCUS_EVAL:-true}"

export SPARK_MASTER="${SPARK_MASTER:-local[4]}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-6g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-6g}"
export SPARK_LOCAL_DIR="${SPARK_LOCAL_DIR:-${SPARK_LOCAL_DIR_DEFAULT}}"
export SPARK_SQL_SHUFFLE_PARTITIONS="${SPARK_SQL_SHUFFLE_PARTITIONS:-24}"
export SPARK_DEFAULT_PARALLELISM="${SPARK_DEFAULT_PARALLELISM:-24}"
export SPARK_NETWORK_TIMEOUT="${SPARK_NETWORK_TIMEOUT:-600s}"
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL="${SPARK_EXECUTOR_HEARTBEAT_INTERVAL:-60s}"
export TRAIN_CACHE_MODE="${TRAIN_CACHE_MODE:-disk}"

echo "[INFO] bucket5 structural v5 joint min cls typed-intent phase3-slicefix train+infer"
echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] stage09=${INPUT_09_RUN_DIR}"
echo "[INFO] match_channels=${INPUT_09_MATCH_CHANNELS_RUN_DIR}"
echo "[INFO] profile_run_dir=${PROFILE_RUN_DIR}"
echo "[INFO] eval_user_cohort=${RANK_EVAL_USER_COHORT_PATH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/10_1_rank_train.py"

LATEST_TRAIN_DIR="$(latest_run_dir "${TRAIN_OUTPUT_ROOT}")"
if [[ -z "${LATEST_TRAIN_DIR}" ]]; then
  echo "[ERROR] No train output found under ${TRAIN_OUTPUT_ROOT}" >&2
  exit 1
fi
RANK_MODEL_JSON="${LATEST_TRAIN_DIR}/rank_model.json"
if [[ ! -f "${RANK_MODEL_JSON}" ]]; then
  echo "[ERROR] rank_model.json not found: ${RANK_MODEL_JSON}" >&2
  exit 1
fi
export RANK_MODEL_JSON

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/10_2_rank_infer_eval.py"

if [[ "${RUN_BUCKET5_TYPED_FOCUS_EVAL}" != "true" ]]; then
  exit 0
fi

LATEST_INFER_DIR="$(latest_run_dir "${INFER_OUTPUT_ROOT}")"
if [[ -z "${LATEST_INFER_DIR}" ]]; then
  echo "[ERROR] No infer output found under ${INFER_OUTPUT_ROOT}" >&2
  exit 1
fi

USER_AUDIT_PATH="${LATEST_INFER_DIR}/bucket_5/user_diagnostics.parquet"
if [[ ! -f "${USER_AUDIT_PATH}" ]]; then
  echo "[ERROR] user_diagnostics.parquet not found: ${USER_AUDIT_PATH}" >&2
  exit 1
fi

FOCUS_EVAL_OUT_DIR="${FOCUS_EVAL_OUTPUT_ROOT}/$(basename "${LATEST_INFER_DIR}")"
mkdir -p "${FOCUS_EVAL_OUT_DIR}"

FOCUS_ARGS=(
  --stage10-user-audit "${USER_AUDIT_PATH}"
  --bucket-dir "${STAGE09_RUN_DIR}/bucket_5"
  --profile-run-dir "${PROFILE_RUN_DIR}"
  --output-dir "${FOCUS_EVAL_OUT_DIR}"
)
if [[ -n "${USER_SCHEMA_RUN_DIR}" && -d "${USER_SCHEMA_RUN_DIR}" ]]; then
  FOCUS_ARGS+=(--user-schema-run-dir "${USER_SCHEMA_RUN_DIR}")
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/10_4_bucket5_focus_slice_eval.py" "${FOCUS_ARGS[@]}"

echo "[INFO] focus_eval_output_dir=${FOCUS_EVAL_OUT_DIR}"
