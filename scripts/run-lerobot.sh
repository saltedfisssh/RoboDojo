#!/usr/bin/env bash
# Run Python in this checkout's export environment, from any working directory.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: bash scripts/run-lerobot.sh SCRIPT.py [arguments...]
       bash scripts/run-lerobot.sh -m MODULE [arguments...]

Run Python in this checkout's isolated LeRobot export environment. Arguments
are passed to Python unchanged; SCRIPT.py --help shows that script's own help.

Environment selection:
  LEROBOT_VENV     Export venv (default: <repo>/.venv-lerobot)
  LEROBOT_PYTHON   Explicit Python executable; takes precedence over LEROBOT_VENV
Relative interpreter, script and data paths are resolved from the repository root.
The active shell's simulator/Conda environment is not used for export.

Examples:
  bash scripts/run-lerobot.sh -m src.expert export --help
  bash scripts/run-lerobot.sh scripts/internal/validate_expert_dataset.py \
    --dataset data/expert/lerobot/stack_blocks --report data/expert/validation.json

Setup: bash scripts/install-lerobot-uv.sh
For normal exports, use `bash scripts/generate_data.sh export --help`.
EOF
  exit 0
fi
EXPORT_VENV="${LEROBOT_VENV:-.venv-lerobot}"
[[ "$EXPORT_VENV" == /* ]] || EXPORT_VENV="$ROOT_DIR/$EXPORT_VENV"
EXPORT_PYTHON="${LEROBOT_PYTHON:-$EXPORT_VENV/bin/python}"
[[ "$EXPORT_PYTHON" == /* ]] || EXPORT_PYTHON="$ROOT_DIR/$EXPORT_PYTHON"
if [[ ! -x "$EXPORT_PYTHON" ]]; then
  printf '[lerobot] Missing Python: %s\nRun: bash "%s/scripts/install-lerobot-uv.sh"\n' "$EXPORT_PYTHON" "$ROOT_DIR" >&2
  exit 1
fi
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export VIRTUAL_ENV="$(cd "$(dirname "$EXPORT_PYTHON")/.." && pwd)"
export PATH="$VIRTUAL_ENV/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# NumPy 1/2 and Python 3.11/3.12 wheels must never leak across environments.
export PYTHONPATH="$ROOT_DIR" LEROBOT_PYTHON="$EXPORT_PYTHON"
cd "$ROOT_DIR"
exec "$EXPORT_PYTHON" "$@"
