#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/internal/uv_env.sh"

if [[ "${1:-}" == --venv ]]; then
  [[ $# -ge 3 && -n "$2" && "$2" != -* ]] || { echo 'Usage: robodojo-uv.sh [--venv PATH] COMMAND [options]' >&2; exit 2; }
  export ROBODOJO_VENV="$2"
  shift 2
fi
if [[ $# -eq 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then
  printf 'Usage: bash scripts/robodojo-uv.sh [--venv PATH] COMMAND [options]\n'
  printf 'Defaults to <repo>/.venv; ROBODOJO_VENV also selects the environment.\n\n'
  exec bash "$SCRIPT_DIR/robodojo.sh" --help
fi
# Help remains usable before installing the environment.
for arg in "$@"; do
  if [[ "$arg" == -h || "$arg" == --help ]]; then
    export ROBODOJO_SIM_ENV="${ROBODOJO_VENV:-$ROBODOJO_ROOT/.venv}"
    exec bash "$SCRIPT_DIR/robodojo.sh" "$@"
  fi
done
robodojo_use_uv
cd "$ROBODOJO_ROOT"
exec bash "$SCRIPT_DIR/robodojo.sh" "$@"
