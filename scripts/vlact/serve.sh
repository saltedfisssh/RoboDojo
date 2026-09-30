#!/usr/bin/env bash
# Serve an explicit locally trained checkpoint through RoboDojo's runtime contract.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/vlact/serve.sh /absolute/run/checkpoints/steps_N_pytorch_model.pt [port]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STARVLA_CKPT_PATH="$1"
if [[ "$STARVLA_CKPT_PATH" != /* || ! -f "$STARVLA_CKPT_PATH" ]]; then
  echo "Checkpoint must be an existing absolute file path: $STARVLA_CKPT_PATH" >&2
  exit 2
fi
VLACT_ROOT="${VLACT_ROOT:-${ROOT}/../VLAct}"
export STARVLA_BASE_VLM="${STARVLA_BASE_VLM:-${VLACT_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
export STARVLA_INCLUDE_STATE=false
export STARVLA_UNNORM_KEY=arx_x5
export STARVLA_EXECUTE_HORIZON="${STARVLA_EXECUTE_HORIZON:-16}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/XPolicyLab/policy/starVLA"
exec bash setup_eval_policy_server.sh RoboDojo "${VLACT_TASK:-build_tower}" \
  "$(dirname "$(dirname "$STARVLA_CKPT_PATH")")" arx_x5 joint 0 \
  "${VLACT_GPU:-0}" "${VLACT_ENV:-${VLACT_ROOT}/.venv}" "${2:-6000}" \
  "${VLACT_BIND_HOST:-127.0.0.1}"
