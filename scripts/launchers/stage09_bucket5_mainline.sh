#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../runtime_sh/run_stage09_bucket5_structural_v5_sourceparity.sh" "$@"
