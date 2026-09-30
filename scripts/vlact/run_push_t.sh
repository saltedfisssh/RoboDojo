#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Usage: bash scripts/vlact/run_push_t.sh CHECKPOINT_LABEL PORT [EVAL_NUM]" >&2
  echo "Start the matching checkpoint with serve.sh before running this client." >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/scripts/vlact/run_task.sh" push_T "$@"
