#!/usr/bin/env bash
# Run from any directory; ROBODOJO_ROOT optionally selects another checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROBODOJO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="${ROBODOJO_VENV:-.venv}"
FROM_STEP=system
SKIP_SYSTEM=0
STEPS=(system venv base_deps submodules isaacsim isaaclab curobo verify)

info() { printf '[install_uv] %s\n' "$*"; }
error() { printf '[install_uv] ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Usage: bash scripts/install-uv.sh [options]

Install Python 3.11, PyTorch 2.7.0 (cu128), Isaac Sim 5.1, IsaacLab,
and CuRobo into .venv. Requires Linux x86_64 and an NVIDIA GPU at runtime.
Missing uv is installed into the repository's .cache/uv-bin directory.

Options:
  -i, --install      Full install (also the default with no arguments)
  --venv PATH        Environment directory (relative to the repository root)
  --from STEP        Resume: system, venv, base_deps, submodules, isaacsim,
                     isaaclab, curobo, verify; an existing env is required
                     when resuming after venv
  --skip-system      Skip apt system packages (already installed / no sudo)
  -h, --help         Show this help

Environment:
  ROBODOJO_ROOT     Repository root (default: parent of this script)
  ROBODOJO_VENV     Default environment path (default: .venv)
  UV_DEFAULT_INDEX Optional PyPI mirror; PyTorch uses its official cu128 index

This installs the simulation client, without optional RL training frameworks.
Download assets with scripts/init_assets.sh and install policy-server
requirements separately through XPolicyLab. After activating the environment,
use scripts/robodojo.sh client to connect to a running policy server.
EOF
}
need_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || error "$1 requires a value"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--install) FROM_STEP=system; shift ;;
    --venv) need_value "$@"; VENV_DIR="$2"; shift 2 ;;
    --from) need_value "$@"; FROM_STEP="$2"; shift 2 ;;
    --skip-system) SKIP_SYSTEM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) error "Unknown argument: $1 (see --help)" ;;
  esac
done
START=-1
for i in "${!STEPS[@]}"; do
  if [[ "${STEPS[$i]}" == "$FROM_STEP" ]]; then START=$i; break; fi
done
[[ "$START" -ge 0 ]] || error "Unknown step '$FROM_STEP'. Valid: ${STEPS[*]}"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
  || error "This dependency stack supports Linux x86_64 only."
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
[[ -f "$ROOT_DIR/scripts/requirements.txt" && -f "$ROOT_DIR/scripts/install.sh" ]] \
  || error "Run from the RoboDojo repository root, or set ROBODOJO_ROOT."
[[ "$VENV_DIR" == /* ]] || VENV_DIR="$ROOT_DIR/$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
export PIP_USER=0 PYTHONNOUSERSITE=1
unset PYTHONHOME

# Match install.sh's runtime pins and preserve CUDA-enabled PyTorch wheels.
PIN_DIR="$(mktemp -d)"
trap 'rm -rf -- "$PIN_DIR"' EXIT
cat >"$PIN_DIR/constraints.txt" <<'EOF'
numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
psutil==5.9.8
wheel==0.45.1
stable-baselines3<2.8
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
setuptools<82
opencv-python-headless==4.11.0.86
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
EOF
# IsaacLab requests 0.49.1, but install.sh downgrades it for Isaac Sim.
# uv needs an override for that metadata conflict. uv pip check will still
# report the known mismatch; runtime compatibility needs a GPU smoke test.
printf '%s\n' 'starlette==0.45.3' >"$PIN_DIR/overrides.txt"

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    export PATH="$ROOT_DIR/.cache/uv-bin:$PATH"
  fi
  if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || error "Install curl or uv first."
    info "Installing uv into $ROOT_DIR/.cache/uv-bin..."
    # pipefail propagates download failures; no shell profile is modified.
    curl --fail --location --silent --show-error https://astral.sh/uv/install.sh \
      | env UV_UNMANAGED_INSTALL="$ROOT_DIR/.cache/uv-bin" sh
  fi
  command -v uv >/dev/null 2>&1 || error "uv installation failed."
}
activate_env() {
  [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$PYTHON" ]] \
    || error "Missing venv at $VENV_DIR. Run from the venv step first."
  "$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
  export VIRTUAL_ENV="$VENV_DIR"
  export PATH="$VENV_DIR/bin:$PATH"
  unset CONDA_PREFIX CONDA_DEFAULT_ENV
}
uv_install() {
  # Explicit targeting also protects an unrelated active Conda environment.
  uv pip install --python "$PYTHON" \
    --constraint "$PIN_DIR/constraints.txt" \
    --override "$PIN_DIR/overrides.txt" "$@"
}
setup_system() {
  if [[ "$SKIP_SYSTEM" == 1 ]]; then
    info "Skipping system packages."
    return
  fi
  command -v apt-get >/dev/null 2>&1 \
    || error "Install the system dependencies listed in setup_system, then use --skip-system."
  local packages=(build-essential cmake ninja-build pkg-config git git-lfs curl ca-certificates
    ffmpeg libgl1 libglib2.0-0 libegl1 libvulkan1 libxt6 libsm6 libxrender1 libxext6)
  local missing=() pkg
  for pkg in "${packages[@]}"; do
    if [[ "$(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null || true)" != "install ok installed" ]]; then
      missing+=("$pkg")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    local sudo_cmd=()
    if [[ "$(id -u)" != 0 ]]; then
      command -v sudo >/dev/null 2>&1 || error "Install system packages as root or use --skip-system."
      sudo_cmd=(sudo)
    fi
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y --no-install-recommends "${missing[@]}"
  fi
}
setup_venv() {
  ensure_uv
  if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
    [[ ! -e "$VENV_DIR" ]] || error "$VENV_DIR exists but is not a venv; choose another --venv path."
    uv venv --python 3.11 --seed "$VENV_DIR"
  fi
  activate_env
}
setup_base_deps() {
  # toml/setuptools_scm are needed by the editable, non-isolated builds below.
  uv_install -r "$ROOT_DIR/scripts/requirements.txt" \
    pip 'setuptools<82' setuptools-scm wheel toml ninja \
    opencv-python-headless pillow matplotlib scipy scikit-learn \
    numpy packaging typing_extensions filelock websockets click psutil \
    starlette warp-lang onnx 'ipython<9' virtualenv
}
setup_submodules() {
  # Use the superproject's recorded commits, not moving remote branch heads.
  git -C "$ROOT_DIR" submodule update --init --recursive -- \
    third_party/IsaacLab third_party/curobo XPolicyLab
  [[ -f "$ROOT_DIR/XPolicyLab/client_server/ws/model_client.py" ]] \
    || error "XPolicyLab client is missing. Check submodule access."
}
setup_isaacsim() {
  uv_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  # Match the existing pip installer's cross-index resolution for this call.
  uv_install 'isaacsim[all,extscache]==5.1.0' \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
}
setup_isaaclab() {
  local extension
  local editable=()
  [[ -f "$ROOT_DIR/third_party/IsaacLab/source/isaaclab/setup.py" ]] \
    || error "IsaacLab is missing. Run from the submodules step first."
  # Resolve core extensions together, without optional RL training frameworks.
  for extension in "$ROOT_DIR"/third_party/IsaacLab/source/*; do
    if [[ -f "$extension/setup.py" || -f "$extension/pyproject.toml" ]]; then
      editable+=(-e "$extension")
    fi
  done
  uv_install --no-build-isolation "${editable[@]}"
}
setup_curobo() {
  [[ -f "$ROOT_DIR/third_party/curobo/pyproject.toml" ]] \
    || error "CuRobo is missing. Run from the submodules step first."
  uv_install --no-build-isolation -e "$ROOT_DIR/third_party/curobo[cu12]"
}
setup_verify() {
  # Check metadata and the APIs RoboDojo uses without launching SimulationApp.
  "$PYTHON" - <<'PY'
from importlib.metadata import version

import numpy
import torch
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria

expected = {
    "isaacsim": "5.1.0.0",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "torchaudio": "2.7.0+cu128",
    "numpy": "1.26.0",
    "starlette": "0.45.3",
}
for name, wanted in expected.items():
    actual = version(name)
    assert actual == wanted, f"{name}: expected {wanted}, got {actual}"
    print(f"{name}=={actual}")
for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic", "nvidia-curobo"):
    print(f"{name}=={version(name)}")
assert torch.version.cuda == "12.8", torch.version.cuda
print(f"CUDA available: {torch.cuda.is_available()}")
print("Package/API checks passed; Isaac Sim rendering still requires a GPU smoke run.")
PY
}

cd "$ROOT_DIR"
if [[ "$START" -gt 1 ]]; then
  activate_env
  ensure_uv
fi
for i in "${!STEPS[@]}"; do
  if [[ "$i" -ge "$START" ]]; then
    info "[$((i + 1))/${#STEPS[@]}] ${STEPS[$i]}"
    "setup_${STEPS[$i]}"
  fi
done
info "Environment setup completed. Activate it with:"
printf '  source %q\n' "$VENV_DIR/bin/activate"
