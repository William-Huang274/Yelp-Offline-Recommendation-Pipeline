#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher kept for ablation history; prefer scripts/launchers/stage10_bucket10_mainline.sh for outward docs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/launchers/_path_contract.sh"
FAST_ROOT="${FAST_ROOT:-${BDA_REMOTE_TMP_ROOT}/bucket10_rerank_v1}"
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

mkdir -p "${FAST_ROOT}/logs" "${FAST_ROOT}/spark"

resolve_first_existing() {
  local p=""
  for p in "$@"; do
    if [[ -n "${p}" && -e "${p}" ]]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  return 1
}

BASE_STAGE09_RUN_DIR="${BASE_STAGE09_RUN_DIR:-}"
if [[ -z "${BASE_STAGE09_RUN_DIR}" ]]; then
  BASE_STAGE09_RUN_DIR="$(resolve_first_existing \
    "${REPO_ROOT}/data/output/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion" \
    "${BDA_REMOTE_AUTODL_FS_OUTPUT_ROOT}/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion" \
    "${BDA_REMOTE_PROJECT_OUTPUT_ROOT}/09_candidate_fusion/20260311_005450_full_stage09_candidate_fusion" \
  )" || true
fi
BASE_TEXT_MATCH_RUN_DIR="${BASE_TEXT_MATCH_RUN_DIR:-}"
if [[ -z "${BASE_TEXT_MATCH_RUN_DIR}" ]]; then
  BASE_TEXT_MATCH_RUN_DIR="$(resolve_first_existing \
    "${REPO_ROOT}/data/output/09_candidate_wise_text_match_features_v1/20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build" \
    "${BDA_REMOTE_AUTODL_FS_OUTPUT_ROOT}/09_candidate_wise_text_match_features_v1/20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build" \
    "${BDA_REMOTE_PROJECT_OUTPUT_ROOT}/09_candidate_wise_text_match_features_v1/20260323_174614_full_stage09_candidate_wise_text_match_features_v1_build" \
  )" || true
fi
BASE_GROUP_GAP_RUN_DIR="${BASE_GROUP_GAP_RUN_DIR:-}"
if [[ -z "${BASE_GROUP_GAP_RUN_DIR}" ]]; then
  BASE_GROUP_GAP_RUN_DIR="$(resolve_first_existing \
    "${REPO_ROOT}/data/output/09_stage10_group_gap_features_v1/20260323_174757_full_stage09_stage10_group_gap_features_v1_build" \
    "${BDA_REMOTE_AUTODL_FS_OUTPUT_ROOT}/09_stage10_group_gap_features_v1/20260323_174757_full_stage09_stage10_group_gap_features_v1_build" \
    "${BDA_REMOTE_PROJECT_OUTPUT_ROOT}/09_stage10_group_gap_features_v1/20260323_174757_full_stage09_stage10_group_gap_features_v1_build" \
  )" || true
fi
BASE_COHORT_PATH="${BASE_COHORT_PATH:-}"
if [[ -z "${BASE_COHORT_PATH}" ]]; then
  BASE_COHORT_PATH="$(resolve_first_existing \
    "${REPO_ROOT}/data/output/fixed_eval_cohorts/bucket10_accepted_eval_users_738_useridx.csv" \
    "${BDA_REMOTE_AUTODL_FS_OUTPUT_ROOT}/fixed_eval_cohorts/bucket10_accepted_eval_users_738_useridx.csv" \
    "${BDA_REMOTE_PROJECT_OUTPUT_ROOT}/fixed_eval_cohorts/bucket10_accepted_eval_users_738_useridx.csv" \
  )" || true
fi
if [[ -z "${BASE_STAGE09_RUN_DIR}" || ! -d "${BASE_STAGE09_RUN_DIR}" ]]; then
  echo "[ERROR] bucket10 stage09 run dir not found" >&2
  exit 1
fi
if [[ -z "${BASE_TEXT_MATCH_RUN_DIR}" || ! -d "${BASE_TEXT_MATCH_RUN_DIR}" ]]; then
  echo "[ERROR] bucket10 text_match run dir not found" >&2
  exit 1
fi
if [[ -z "${BASE_GROUP_GAP_RUN_DIR}" || ! -d "${BASE_GROUP_GAP_RUN_DIR}" ]]; then
  echo "[ERROR] bucket10 group_gap run dir not found" >&2
  exit 1
fi
if [[ -z "${BASE_COHORT_PATH}" || ! -f "${BASE_COHORT_PATH}" ]]; then
  echo "[ERROR] bucket10 fixed cohort path not found" >&2
  exit 1
fi
BASE_LAUNCHER_PATH="${BASE_LAUNCHER_PATH:-}"
if [[ -z "${BASE_LAUNCHER_PATH}" ]]; then
  for candidate in \
    "${REPO_ROOT}/scripts/run_stage10_bucket10_fixedcohort_migration.sh" \
    "${BDA_REMOTE_AUTODL_FS_SCRIPTS_ROOT}/run_stage10_bucket10_fixedcohort_migration.sh" \
    "${BDA_REMOTE_PROJECT_ROOT}/scripts/run_stage10_bucket10_fixedcohort_migration.sh"
  do
    if [[ -f "${candidate}" ]]; then
      BASE_LAUNCHER_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BASE_LAUNCHER_PATH}" || ! -f "${BASE_LAUNCHER_PATH}" ]]; then
  echo "[ERROR] bucket10 fixed-cohort base launcher not found" >&2
  exit 1
fi

export REPO_ROOT
export PYTHON_BIN
export STAGE09_RUN_DIR="${STAGE09_RUN_DIR:-${BASE_STAGE09_RUN_DIR}}"
export TEXT_MATCH_RUN_DIR="${TEXT_MATCH_RUN_DIR:-${BASE_TEXT_MATCH_RUN_DIR}}"
export GROUP_GAP_RUN_DIR="${GROUP_GAP_RUN_DIR:-${BASE_GROUP_GAP_RUN_DIR}}"
export FIXED_EVAL_COHORT_PATH="${FIXED_EVAL_COHORT_PATH:-${BASE_COHORT_PATH}}"

export TRAIN_BUCKETS_OVERRIDE="${TRAIN_BUCKETS_OVERRIDE:-10}"
export RANK_BUCKETS_OVERRIDE="${RANK_BUCKETS_OVERRIDE:-10}"
export TRAIN_MODEL_BACKEND="${TRAIN_MODEL_BACKEND:-xgboost_cls}"
export STAGE10_CLOUD_FAST_MODE="${STAGE10_CLOUD_FAST_MODE:-true}"
export TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS="${TRAIN_ENABLE_EXPENSIVE_DIAGNOSTICS:-false}"
export RANK_DIAGNOSTICS_ENABLE="${RANK_DIAGNOSTICS_ENABLE:-false}"
export SPARK_MASTER="${SPARK_MASTER:-local[24]}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-64g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-64g}"
export SPARK_SQL_SHUFFLE_PARTITIONS="${SPARK_SQL_SHUFFLE_PARTITIONS:-96}"
export SPARK_DEFAULT_PARALLELISM="${SPARK_DEFAULT_PARALLELISM:-96}"
export SPARK_NETWORK_TIMEOUT="${SPARK_NETWORK_TIMEOUT:-1200s}"
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL="${SPARK_EXECUTOR_HEARTBEAT_INTERVAL:-120s}"
export SPARK_SQL_ADAPTIVE_ENABLED="${SPARK_SQL_ADAPTIVE_ENABLED:-true}"
export SPARK_SQL_PARQUET_ENABLE_VECTORIZED="${SPARK_SQL_PARQUET_ENABLE_VECTORIZED:-true}"
export SPARK_SQL_FILES_MAX_PARTITION_BYTES="${SPARK_SQL_FILES_MAX_PARTITION_BYTES:-128m}"
export TRAIN_CACHE_MODE="${TRAIN_CACHE_MODE:-disk}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

run_case() {
  local case_name="$1"
  local enable_text="$2"
  local enable_gap="$3"
  local feature_override="$4"

  local case_root="${FAST_ROOT}/${case_name}"
  mkdir -p "${case_root}/spark" "${case_root}/logs"

  export ENABLE_STAGE10_V2_TEXT_MATCH="${enable_text}"
  export STAGE10_V2_TEXT_MATCH_BUCKETS="10"
  export STAGE10_V2_TEXT_MATCH_FEATURES="sim_negative_avoid_neg,sim_negative_avoid_core,sim_conflict_gap"

  export ENABLE_STAGE10_V2_GROUP_GAP="${enable_gap}"
  export STAGE10_V2_GROUP_GAP_BUCKETS="10"
  export STAGE10_V2_GROUP_GAP_FEATURES="schema_weighted_overlap_user_ratio_v2_rank_pct_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3,schema_weighted_net_score_v2_rank_pct_v3,schema_top_pop_low_flag_v3"

  export STAGE10_FEATURE_COLUMNS_OVERRIDE="${feature_override}"
  export SPARK_LOCAL_DIR="${case_root}/spark"
  export TRAIN_OUTPUT_ROOT="${case_root}/output/10_rank_models"
  export INFER_OUTPUT_ROOT="${case_root}/output/10_infer_eval"
  export METRICS_PATH="${case_root}/metrics/recsys_stage10_results_${case_name}.csv"

  echo "[CASE] ${case_name}"
  echo "[CASE] feature_override=${STAGE10_FEATURE_COLUMNS_OVERRIDE}"
  echo "[CASE] base_launcher=${BASE_LAUNCHER_PATH}"
  bash "${BASE_LAUNCHER_PATH}" \
    > "${case_root}/logs/${case_name}.log" 2>&1
}

run_case \
  "gap_only" \
  "false" \
  "true" \
  "pre_score,schema_weighted_overlap_user_ratio_v2_rank_pct_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3,schema_weighted_net_score_v2_rank_pct_v3,schema_top_pop_low_flag_v3"

run_case \
  "conflict_only" \
  "true" \
  "false" \
  "pre_score,sim_negative_avoid_neg,sim_negative_avoid_core,sim_conflict_gap"

run_case \
  "gap_conflict_combo" \
  "true" \
  "true" \
  "pre_score,sim_negative_avoid_neg,sim_negative_avoid_core,sim_conflict_gap,schema_weighted_overlap_user_ratio_v2_rank_pct_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top3_v3,schema_weighted_overlap_user_ratio_v2_gap_to_top10_v3,schema_weighted_net_score_v2_rank_pct_v3,schema_top_pop_low_flag_v3"

echo "[DONE] bucket10 rerank v1 ablation root=${FAST_ROOT}"
