#!/usr/bin/env bash
set -euo pipefail
DATAGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DATAGEN_ROOT"
usage() {
  cat <<'EOF'
Usage: bash scripts/generate_data.sh legacy COMMAND [options]
       bash scripts/datagen-uv.sh COMMAND [options]

Legacy fixed-layout experiments. For official-distribution expert collection,
use `bash scripts/generate_data.sh --help`; the data formats and task coverage differ.

Commands:
  collect, batch, campaign, dispatch   Collect or coordinate legacy experiments
  export, review                      Export or review data in the LeRobot environment
  replay, render_episode              Replay state or render a recorded episode
  inventory                           Inspect legacy task coverage
  state_probe, camera_probe           Inspect simulator state or cameras
  planner_probe                       Probe planner behavior
  promote                             Add RGB to successful state-only episodes

Run `bash scripts/generate_data.sh legacy COMMAND --help` for command options.
Legacy subcommand help may require the corresponding environment's dependencies.

Environments:
  ROBODOJO_VENV    Simulator venv (default: <repo>/.venv)
  LEROBOT_VENV     Export/review venv (default: <repo>/.venv-lerobot)
  LEROBOT_PYTHON   Explicit export interpreter; overrides LEROBOT_VENV
Relative paths are resolved from the repository root. See docs/datagen/README.md.
EOF
}
if [[ $# -eq 0 || "${1:-}" == -h || "${1:-}" == --help || "${1:-}" == help ]]; then
  usage
  exit 0
fi
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
    printf 'Unknown legacy command: %s\n\n' "$mode" >&2
    usage >&2
    exit 2
    ;;
esac
