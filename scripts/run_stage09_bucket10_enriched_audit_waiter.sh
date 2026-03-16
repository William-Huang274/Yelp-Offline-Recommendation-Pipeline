#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/5006_BDA_project}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RECOVERY_ROOT="${RECOVERY_ROOT:-/root/autodl-tmp/project_data/data/output/09_user_profiles}"
PARQUET_BASE_DIR="${PARQUET_BASE_DIR:-/root/autodl-tmp/project_data/data/parquet}"
USER_PROFILE_ROOT_DIR="${USER_PROFILE_ROOT_DIR:-/root/autodl-tmp/project_data/data/output/09_user_profiles}"
ITEM_SEMANTIC_ROOT_DIR="${ITEM_SEMANTIC_ROOT_DIR:-/root/autodl-tmp/project_data/data/output/09_item_semantics}"
ITEM_SEMANTIC_RUN_DIR="${ITEM_SEMANTIC_RUN_DIR:-/root/autodl-tmp/project_data/data/output/09_item_semantics/20260305_000408_full_stage09_item_semantic_build}"
CLUSTER_PROFILE_ROOT_DIR="${CLUSTER_PROFILE_ROOT_DIR:-/root/autodl-tmp/project_data/data/output/08_cluster_labels/full}"
OUTPUT_ROOT_DIR="${OUTPUT_ROOT_DIR:-/root/autodl-tmp/stage09_fs}"
SPARK_LOCAL_DIR="${SPARK_LOCAL_DIR:-/root/autodl-tmp/spark-tmp}"
PY_TEMP_DIR="${PY_TEMP_DIR:-/root/autodl-tmp/py-tmp}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

mkdir -p "${SPARK_LOCAL_DIR}" "${PY_TEMP_DIR}" "$(dirname "${OUTPUT_ROOT_DIR}")"

find_latest_vector_run() {
  find "${RECOVERY_ROOT}" \
    -maxdepth 2 \
    -type f \
    -name "user_profile_vectors.npz" \
    -path "*stage09_user_profile_vector_recovery/user_profile_vectors.npz" \
    -printf "%T@ %h\n" 2>/dev/null | sort -nr | awk 'NR==1 {print $2}'
}

RUN_DIR="$(find_latest_vector_run || true)"
while [[ -z "${RUN_DIR}" || ! -f "${RUN_DIR}/user_profile_vectors.npz" ]]; do
  printf '[WAIT] %s waiting for recovered user_profile_vectors.npz\n' "$(date '+%F %T')"
  sleep "${WAIT_SECONDS}"
  RUN_DIR="$(find_latest_vector_run || true)"
done

printf '[INFO] %s using USER_PROFILE_RUN_DIR=%s\n' "$(date '+%F %T')" "${RUN_DIR}"

export PARQUET_BASE_DIR
export USER_PROFILE_ROOT_DIR
export USER_PROFILE_RUN_DIR="${RUN_DIR}"
export ITEM_SEMANTIC_ROOT_DIR
export ITEM_SEMANTIC_RUN_DIR
export CLUSTER_PROFILE_ROOT_DIR
export OUTPUT_ROOT_DIR
export RUN_PROFILE_OVERRIDE="${RUN_PROFILE_OVERRIDE:-full}"
export RECALL_PROFILE_OVERRIDE="${RECALL_PROFILE_OVERRIDE:-coverage_stage2}"
export MIN_TRAIN_BUCKETS_OVERRIDE="${MIN_TRAIN_BUCKETS_OVERRIDE:-10}"
export WRITE_ENRICHED_AUDIT_EXPORT="${WRITE_ENRICHED_AUDIT_EXPORT:-true}"
export ENRICHED_AUDIT_BUCKETS="${ENRICHED_AUDIT_BUCKETS:-10}"
export SKIP_INTERNAL_METRICS="${SKIP_INTERNAL_METRICS:-true}"
export SPARK_MASTER="${SPARK_MASTER:-local[32]}"
export SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-48g}"
export SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-48g}"
export SPARK_SQL_SHUFFLE_PARTITIONS="${SPARK_SQL_SHUFFLE_PARTITIONS:-128}"
export SPARK_DEFAULT_PARALLELISM="${SPARK_DEFAULT_PARALLELISM:-64}"
export SPARK_LOCAL_DIR
export TMPDIR="${PY_TEMP_DIR}"
export TMP="${PY_TEMP_DIR}"
export TEMP="${PY_TEMP_DIR}"
export PY_TEMP_DIR
export SPARK_NETWORK_TIMEOUT="${SPARK_NETWORK_TIMEOUT:-1200s}"
export SPARK_EXECUTOR_HEARTBEAT_INTERVAL="${SPARK_EXECUTOR_HEARTBEAT_INTERVAL:-120s}"
export OUTPUT_COALESCE_PARTITIONS="${OUTPUT_COALESCE_PARTITIONS:-16}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/09_candidate_fusion.py
