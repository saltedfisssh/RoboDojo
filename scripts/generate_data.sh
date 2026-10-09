#!/usr/bin/env bash
# Unified expert data generation. See docs/expert/README.md.
set -euo pipefail
GENERATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${GENERATION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${GENERATION_ROOT}"
GENERATION_PYTHON="${ROBODOJO_VENV:-${GENERATION_ROOT}/.venv}/bin/python"
if [[ "${1:-}" == "export" ]]; then
  GENERATION_PYTHON="${LEROBOT_PYTHON:-${GENERATION_ROOT}/../lerobot/.venv/bin/python}"
  if [[ -d "${GENERATION_ROOT}/.cache/datagen-export-deps" ]]; then
    export PYTHONPATH="${GENERATION_ROOT}/.cache/datagen-export-deps:${PYTHONPATH}"
  fi
fi
exec "${GENERATION_PYTHON}" -m src.expert "$@"
