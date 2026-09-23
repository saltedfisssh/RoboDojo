#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help || $# -eq 0 ]]; then
  echo 'Usage: bash scripts/eval_policy-uv.sh --root_dir ROOT --task_name TASK --env_cfg_type CFG --device_id GPU --policy_name POLICY --port PORT [options]'
  echo 'Set ROBODOJO_VENV to select a venv (default: <repo>/.venv).'
  exit 0
fi
source "$SCRIPT_DIR/internal/uv_env.sh"
robodojo_use_uv
exec bash "$SCRIPT_DIR/eval_policy.sh" "$@"
