#!/usr/bin/env bash
set -euo pipefail
python "$(dirname "$0")/pipeline/internal_pilot_runner.py" "$@"
