#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 || ! "$2" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Usage: bash scripts/vlact/run_task.sh TASK CHECKPOINT_LABEL PORT [EVAL_NUM]" >&2
  exit 2
fi
case "$1" in
  push_T|plug_in_charger|insert_tubes|pour_balls_into_vase) ;;
  *) echo "No state oracle implemented for task: $1" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROBODOJO_SUBTASK_ORACLE="${1}_state_v1"
export EVAL_NUM="${4:-1}"
export ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-1}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-${2}_${1}_oracle_$(date -u +%Y%m%dT%H%M%SZ)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
cd "$ROOT"
exec bash XPolicyLab/policy/starVLA/scripts/run_hf_robodojo_env_client.sh \
  "${ROBODOJO_VENV:-${ROOT}/.venv}" "$3" RoboDojo "$1" arx_x5 starVLA \
  "ckpt_name=${2},action_type=joint,conditioning=state_oracle" "$ROOT" \
  "${VLACT_LAYOUT_SEED:-0}" "${VLACT_ENV_GPU:-0}" "${VLACT_POLICY_HOST:-127.0.0.1}"
