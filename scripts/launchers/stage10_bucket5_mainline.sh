#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../runtime_sh/run_stage10_bucket5_structural_v5_joint_min_cls_typed_intent_phase3_slicefix.sh" "$@"
