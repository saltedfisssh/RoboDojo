#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBODOJO_UV=1
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  exec bash "$SCRIPT_DIR/download_ckpt.sh"
fi
exec bash "$SCRIPT_DIR/download_ckpt.sh" "$@"
