#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../runtime_sh/run_stage10_bucket10_rerank_v1_ablation.sh" "$@"
