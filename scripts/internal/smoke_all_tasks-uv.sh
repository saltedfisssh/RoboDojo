#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/uv_env.sh"
for arg in "$@"; do
  if [[ "$arg" == -h || "$arg" == --help ]]; then
    exec bash "$SCRIPT_DIR/smoke_all_tasks.sh" "$@"
  fi
done
robodojo_use_uv
cd "$ROBODOJO_ROOT"
exec bash "$SCRIPT_DIR/smoke_all_tasks.sh" "$@"
