#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: tools/monitor_stage11_training.sh <log_path>"
  exit 1
fi

python "$PROJECT_ROOT/tools/monitor_stage11_training.py" "$1"
