#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBODOJO_UV=1
exec bash "$SCRIPT_DIR/download_data.sh" "$@"
