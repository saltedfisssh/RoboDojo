#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/init_assets-uv.sh'
  echo 'Download assets using Git LFS; no Python/Conda activation is required.'
  exit 0
fi
[[ $# -eq 0 ]] || { echo 'init_assets-uv.sh takes no arguments (see --help).' >&2; exit 2; }
export ROBODOJO_UV=1
exec bash "$SCRIPT_DIR/init_assets.sh"
