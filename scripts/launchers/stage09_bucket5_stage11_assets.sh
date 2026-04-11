#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/../runtime_sh/run_stage09_bucket5_stage11_source_semantic_materials_v1.sh" "$@"
"${SCRIPT_DIR}/../runtime_sh/run_stage09_bucket5_stage11_semantic_text_assets_v1.sh" "$@"
