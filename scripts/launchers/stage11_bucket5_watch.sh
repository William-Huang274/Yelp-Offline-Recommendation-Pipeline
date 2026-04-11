#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../runtime_sh/run_stage11_bucket5_top250_semantic_compact_rm_watch_11_2_and_11_3.sh" "$@"
