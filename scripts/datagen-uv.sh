#!/usr/bin/env bash
set -euo pipefail
DATAGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DATAGEN_ROOT"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi
mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  collect|batch|replay|inventory|campaign|dispatch|state_probe|render_episode|camera_probe|promote|planner_probe)
    source "$DATAGEN_ROOT/scripts/internal/uv_env.sh"
    robodojo_use_uv
    exec "$ROBODOJO_PYTHON" -m "src.datagen.${mode}" "$@"
    ;;
  export|review)
    exec bash "$DATAGEN_ROOT/scripts/run-lerobot.sh" -m "src.datagen.${mode}" "$@"
    ;;
  *)
    echo 'Usage: bash scripts/datagen-uv.sh {collect|batch|campaign|dispatch|export|review|replay|inventory|state_probe|render_episode|camera_probe|promote|planner_probe} --help'
    exit 2
    ;;
esac
