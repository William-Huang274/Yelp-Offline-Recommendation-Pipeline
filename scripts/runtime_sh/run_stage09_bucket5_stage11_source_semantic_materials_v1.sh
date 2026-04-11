#!/usr/bin/env bash
set -euo pipefail
# Legacy compatibility launcher. Prefer scripts/launchers/stage09_bucket5_stage11_assets.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/launchers/_path_contract.sh"
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

export PYSPARK_PYTHON="${PYSPARK_PYTHON:-${PYTHON_BIN}}"
export PYSPARK_DRIVER_PYTHON="${PYSPARK_DRIVER_PYTHON:-${PYTHON_BIN}}"

INPUT_10_ROOT_DIR="${INPUT_10_ROOT_DIR:-${BDA_REMOTE_PROJECT_OUTPUT_ROOT}/p0_stage10_context_eval}"
INPUT_10_RUN_DIR="${INPUT_10_RUN_DIR:-${BDA_REMOTE_FAST_ROOT}/bucket5_stage10_phase3_fullcohort_learned/output/10_2_rank_infer_eval_joint_min_cls_v5_typed_intent_phase3_slicefix_fullcohort/20260330_024735_stage10_2_rank_infer_eval}"
OUTPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR="${OUTPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR:-${REPO_ROOT}/data/output/09_stage11_source_semantic_materials_v1}"
PY_TEMP_DIR="${PY_TEMP_DIR:-${REPO_ROOT}/data/py-tmp}"

mkdir -p "${OUTPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR}" "${PY_TEMP_DIR}"

export BDA_PROJECT_ROOT="${REPO_ROOT}"
export INPUT_10_ROOT_DIR
export INPUT_10_RUN_DIR
export OUTPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR
export BUCKETS_OVERRIDE="${BUCKETS_OVERRIDE:-5}"
export QLORA_PAIRWISE_POOL_TOPN="${QLORA_PAIRWISE_POOL_TOPN:-100}"
export STAGE11_SEM_TARGET_TRUTH_RANK_MIN="${STAGE11_SEM_TARGET_TRUTH_RANK_MIN:-11}"
export STAGE11_SEM_TARGET_TRUTH_RANK_MAX="${STAGE11_SEM_TARGET_TRUTH_RANK_MAX:-100}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TMPDIR="${TMPDIR:-${PY_TEMP_DIR}}"
export TMP="${TMP:-${PY_TEMP_DIR}}"
export TEMP="${TEMP:-${PY_TEMP_DIR}}"

export SPARK_MASTER="${SPARK_MASTER:-local[24]}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-76g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-76g}"
export SPARK_SQL_SHUFFLE_PARTITIONS="${SPARK_SQL_SHUFFLE_PARTITIONS:-192}"
export SPARK_DEFAULT_PARALLELISM="${SPARK_DEFAULT_PARALLELISM:-192}"
export SPARK_NETWORK_TIMEOUT="${SPARK_NETWORK_TIMEOUT:-900s}"
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL="${SPARK_EXECUTOR_HEARTBEAT_INTERVAL:-60s}"

echo "[INFO] stage09 stage11 source semantic materials v1 launcher"
echo "[INFO] repo=${REPO_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] stage10_root=${INPUT_10_ROOT_DIR}"
echo "[INFO] stage10_run=${INPUT_10_RUN_DIR:-<latest under root>}"
echo "[INFO] output_root=${OUTPUT_09_STAGE11_SOURCE_SEMANTIC_MATERIALS_V1_ROOT_DIR}"
echo "[INFO] buckets=${BUCKETS_OVERRIDE} truth_rank_range=${STAGE11_SEM_TARGET_TRUTH_RANK_MIN}-${STAGE11_SEM_TARGET_TRUTH_RANK_MAX} topn=${QLORA_PAIRWISE_POOL_TOPN}"
echo "[INFO] spark_master=${SPARK_MASTER} driver=${SPARK_DRIVER_MEMORY} executor=${SPARK_EXECUTOR_MEMORY} shuffle=${SPARK_SQL_SHUFFLE_PARTITIONS} parallelism=${SPARK_DEFAULT_PARALLELISM}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/09_stage11_source_semantic_materials_v1_build.py"
