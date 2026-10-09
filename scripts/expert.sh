#!/usr/bin/env bash
# Compatibility alias; use scripts/generate_data.sh for new workflows.
set -euo pipefail
EXPERT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${EXPERT_ROOT}/scripts/generate_data.sh" "$@"
