#!/usr/bin/env bash
# RoboTwin-style expert data generation (src/expert). Usage:
#   bash scripts/expert.sh collect --task stack_blocks --num 100 --out data/expert/stack_blocks
#   bash scripts/expert.sh export  --input data/expert/stack_blocks --output data/expert_lerobot/stack_blocks
#   bash scripts/expert.sh restore --episode data/expert/stack_blocks/episode_0100000.h5 --frame 120
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
mode="${1:-help}"
[[ $# -gt 0 ]] && shift
case "$mode" in
  collect|restore|probe|batch)
    exec "${ROBODOJO_VENV:-${ROOT}/.venv}/bin/python" -m "src.expert.${mode}" "$@" ;;
  export)
    # h5py overlay for the LeRobot venv (see docs/datagen/README.md).
    [[ -d "${ROOT}/.cache/datagen-export-deps" ]] && export PYTHONPATH="${ROOT}/.cache/datagen-export-deps:${PYTHONPATH}"
    exec "${LEROBOT_PYTHON:-${ROOT}/../lerobot/.venv/bin/python}" -m "src.expert.${mode}" "$@" ;;
  *)
    sed -n '2,5p' "$0"; exit 2 ;;
esac
