#!/usr/bin/env bash
# CPU-only export environment owned by this checkout; never upgrade Isaac's NumPy.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROBODOJO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${LEROBOT_VENV:-.venv-lerobot}"
VERIFY_ONLY=0
info() { printf '[install_lerobot] %s\n' "$*"; }
error() { printf '[install_lerobot] ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Usage: bash scripts/install-lerobot-uv.sh [--venv PATH] [--verify-only]

Create this checkout's .venv-lerobot using Python 3.12, LeRobot 0.6.1 and
CPU PyTorch. Installs the dataset extra, HDF5, video and test dependencies.
Requires Linux x86_64 and system FFmpeg shared libraries (e.g. apt install ffmpeg).
Missing uv/Python are downloaded automatically. No simulator or assets required.

  --venv PATH     Export venv; relative paths are resolved against the repo root
  --verify-only   Check an existing export environment without installing packages
  -h, --help      Show this help

LEROBOT_VENV sets the default export venv; ROBODOJO_VENV identifies the simulator
venv, which this script refuses to modify. LEROBOT_PYTHON only overrides launchers,
not this installer. The active shell environment is never an installation target.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || error '--venv requires a path'
      VENV_DIR="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) error "Unknown argument: $1 (see --help)" ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || error 'Supported platform: Linux x86_64.'
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
[[ "$VENV_DIR" == /* ]] || VENV_DIR="$ROOT_DIR/$VENV_DIR"
VENV_DIR="$(realpath -m "$VENV_DIR")"
SIM_DIR="${ROBODOJO_VENV:-.venv}"
[[ "$SIM_DIR" == /* ]] || SIM_DIR="$ROOT_DIR/$SIM_DIR"
SIM_DIR="$(realpath -m "$SIM_DIR")"
[[ "$VENV_DIR" != "$SIM_DIR" && "$VENV_DIR" != "$SIM_DIR/"* && "$SIM_DIR" != "$VENV_DIR/"* ]] \
  || error 'Export and simulator environments must be separate (including symlink targets).'
[[ -f "$ROOT_DIR/scripts/requirements-lerobot.txt" ]] || error 'Export requirements are missing from this checkout.'
PYTHON="$VENV_DIR/bin/python"
export PYTHONNOUSERSITE=1 PIP_USER=0
# Do not import wheels from an active simulator or the old export overlay.
unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
if [[ -e "$VENV_DIR" ]]; then
  [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$PYTHON" ]] || error "$VENV_DIR exists but is not a venv."
  "$PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys

assert sys.version_info[:2] == (3, 12), 'The export environment requires Python 3.12; choose a new --venv path.'
packages = {d.metadata['Name'].lower() for d in metadata.distributions()}
assert not any(p.startswith(('isaacsim', 'isaaclab')) for p in packages), 'Refusing to modify an Isaac environment.'
PY
fi
if [[ "$VERIFY_ONLY" == 0 ]]; then
  if ! command -v uv >/dev/null 2>&1; then export PATH="$ROOT_DIR/.cache/uv-bin:$PATH"; fi
  if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || error 'Install curl or uv first.'
    curl --fail --location --silent --show-error https://astral.sh/uv/install.sh \
      | env UV_UNMANAGED_INSTALL="$ROOT_DIR/.cache/uv-bin" sh
  fi
  [[ "$(uv pip install --help)" == *--torch-backend* ]] \
    || error 'This uv is too old for CPU wheel selection; upgrade uv and rerun.'
  if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then uv venv --python 3.12 "$VENV_DIR"; fi
  info "Installing locked export dependencies into $VENV_DIR"
  uv pip install --python "$PYTHON" --torch-backend cpu -r "$ROOT_DIR/scripts/requirements-lerobot.txt"
  uv pip check --python "$PYTHON"
fi
[[ -x "$PYTHON" ]] || error "Missing environment: $VENV_DIR. Run without --verify-only first."
export PYTHONPATH="$ROOT_DIR" VIRTUAL_ENV="$VENV_DIR" PATH="$VENV_DIR/bin:$PATH"
"$PYTHON" "$ROOT_DIR/scripts/internal/verify_lerobot.py"
info 'Ready. generate_data.sh export and dev qualify select this environment automatically.'
if [[ "$VENV_DIR" != "$ROOT_DIR/.venv-lerobot" ]]; then
  printf 'For subsequent commands: export LEROBOT_VENV=%q\n' "$VENV_DIR"
fi
