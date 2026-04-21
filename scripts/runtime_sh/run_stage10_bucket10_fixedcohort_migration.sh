#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage10_bucket10_mainline.sh.

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

STAGE09_RUN_DIR="${STAGE09_RUN_DIR:-${REPO_ROOT}/data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion}"
TEXT_MATCH_RUN_DIR="${TEXT_MATCH_RUN_DIR:-${REPO_ROOT}/data/output/09_candidate_wise_text_match_features_v1/20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build}"
GROUP_GAP_RUN_DIR="${GROUP_GAP_RUN_DIR:-${REPO_ROOT}/data/output/09_stage10_group_gap_features_v1/20260323_174757_full_stage09_stage10_group_gap_features_v1_build}"
FIXED_EVAL_COHORT_PATH="${FIXED_EVAL_COHORT_PATH:-${REPO_ROOT}/data/output/fixed_eval_cohorts/bucket10_accepted_eval_users_738_useridx.csv}"

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${REPO_ROOT}/data/output/10_rank_models_bucket10_fixedcohort}"
INFER_OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-${REPO_ROOT}/data/output/10_2_rank_infer_eval_bucket10_fixedcohort}"
METRICS_PATH="${METRICS_PATH:-${REPO_ROOT}/data/metrics/recsys_stage10_results_bucket10_fixedcohort.csv}"

SPARK_LOCAL_DIR_DEFAULT="${REPO_ROOT}/data/spark-tmp"
mkdir -p "${TRAIN_OUTPUT_ROOT}" "${INFER_OUTPUT_ROOT}" "$(dirname "${METRICS_PATH}")" "${SPARK_LOCAL_DIR_DEFAULT}"

export BDA_PROJECT_ROOT="${REPO_ROOT}"
export INPUT_09_RUN_DIR="${STAGE09_RUN_DIR}"
export INPUT_09_TEXT_MATCH_RUN_DIR="${TEXT_MATCH_RUN_DIR}"
export INPUT_09_GROUP_GAP_RUN_DIR="${GROUP_GAP_RUN_DIR}"
export OUTPUT_10_1_ROOT_DIR="${TRAIN_OUTPUT_ROOT}"
export OUTPUT_10_2_ROOT_DIR="${INFER_OUTPUT_ROOT}"
export STAGE10_RESULTS_METRICS_PATH="${METRICS_PATH}"
export RANK_EVAL_USER_COHORT_PATH="${FIXED_EVAL_COHORT_PATH}"

export TRAIN_BUCKETS_OVERRIDE="${TRAIN_BUCKETS_OVERRIDE:-10}"
export RANK_BUCKETS_OVERRIDE="${RANK_BUCKETS_OVERRIDE:-10}"
export TRAIN_MODEL_BACKEND="${TRAIN_MODEL_BACKEND:-xgboost_cls}"
export STAGE10_CLOUD_FAST_MODE="${STAGE10_CLOUD_FAST_MODE:-true}"
export TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS="${TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS:-false}"
export RANK_DIAGNOSTICS_ENABLE="${RANK_DIAGNOSTICS_ENABLE:-false}"
export SPARK_SQL_ADAPTIVE_ENABLED="${SPARK_SQL_ADAPTIVE_ENABLED:-true}"
export SPARK_SQL_PARQUET_ENABLE_VECTORIZED="${SPARK_SQL_PARQUET_ENABLE_VECTORIZED:-true}"
export SPARK_SQL_FILES_MAX_PARTITION_BYTES="${SPARK_SQL_FILES_MAX_PARTITION_BYTES:-128m}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export ENABLE_STAGE10_V2_TEXT_MATCH="${ENABLE_STAGE10_V2_TEXT_MATCH:-true}"
export STAGE10_V2_TEXT_MATCH_BUCKETS="${STAGE10_V2_TEXT_MATCH_BUCKETS:-10}"
export STAGE10_V2_TEXT_MATCH_FEATURES="${STAGE10_V2_TEXT_MATCH_FEATURES:-sim_negative_avoid_neg,sim_negative_avoid_core,sim_conflict_gap}"

export ENABLE_STAGE10_V2_GROUP_GAP="${ENABLE_STAGE10_V2_GROUP_GAP:-true}"
export STAGE10_V2_GROUP_GAP_BUCKETS="${STAGE10_V2_GROUP_GAP_BUCKETS:-10}"
export STAGE10_V2_GROUP_GAP_FEATURES="${STAGE10_V2_GROUP_GAP_FEATURES:-schema_weighted_overlap_user_ratio_v2_rank_pct_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3,schema_weighted_net_score_v2_rank_pct_v3,schema_top_pop_low_flag_v3}"

export SPARK_MASTER="${SPARK_MASTER:-local[4]}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-6g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-6g}"
export SPARK_LOCAL_DIR="${SPARK_LOCAL_DIR:-${SPARK_LOCAL_DIR_DEFAULT}}"
export SPARK_SQL_SHUFFLE_PARTITIONS="${SPARK_SQL_SHUFFLE_PARTITIONS:-24}"
export SPARK_DEFAULT_PARALLELISM="${SPARK_DEFAULT_PARALLELISM:-24}"
export SPARK_NETWORK_TIMEOUT="${SPARK_NETWORK_TIMEOUT:-600s}"
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL="${SPARK_EXECUTOR_HEARTBEAT_INTERVAL:-60s}"
export TRAIN_CACHE_MODE="${TRAIN_CACHE_MODE:-disk}"

echo "[INFO] bucket10 fixed-cohort migration audit"
echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] stage09=${INPUT_09_RUN_DIR}"
echo "[INFO] cohort=${RANK_EVAL_USER_COHORT_PATH}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/10_1_rank_train.py"

LATEST_TRAIN_DIR="$(find "${TRAIN_OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
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
