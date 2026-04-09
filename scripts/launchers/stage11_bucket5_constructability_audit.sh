#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../run_stage11_bucket5_top250_semantic_constructability_audit_by_band.sh" "$@"
