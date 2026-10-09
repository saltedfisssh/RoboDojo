#!/usr/bin/env bash
# Unified expert data generation. See docs/expert/README.md.
set -euo pipefail
GENERATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
cd "${GENERATION_ROOT}"
if [[ "${1:-}" == "export" ]]; then
  exec bash "$GENERATION_ROOT/scripts/run-lerobot.sh" -m src.expert "$@"
elif [[ "${1:-}" == "legacy" ]]; then
  shift
  exec bash "$GENERATION_ROOT/scripts/datagen-uv.sh" "$@"
fi
source "$GENERATION_ROOT/scripts/internal/uv_env.sh"
robodojo_use_uv
exec "$ROBODOJO_PYTHON" -m src.expert "$@"
