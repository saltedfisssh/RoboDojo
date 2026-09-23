#!/usr/bin/env bash
# Shared by the uv entry points; source this file, then call robodojo_use_uv.
ROBODOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

robodojo_resolve_venv() {
  local path="${1:-${ROBODOJO_VENV:-.venv}}"
  [[ "$path" != uv ]] || path="${ROBODOJO_VENV:-.venv}"
  [[ "$path" != uv ]] || path=.venv
  [[ "$path" == /* ]] || path="$ROBODOJO_ROOT/$path"
  if [[ ! -f "$path/pyvenv.cfg" && -f "$path/.venv/pyvenv.cfg" ]]; then
    path="$path/.venv"
  fi
  if [[ ! -f "$path/pyvenv.cfg" || ! -x "$path/bin/python" ]]; then
    printf '[uv] Environment missing: %s. Run bash scripts/install-uv.sh first.\n' "$path" >&2
    return 1
  fi
  (cd "$path" && pwd)
}

robodojo_use_uv() {
  local path
  path="$(robodojo_resolve_venv "${1:-${ROBODOJO_VENV:-.venv}}")" || return 1
  unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
  export VIRTUAL_ENV="$path" PATH="$path/bin:$PATH" PYTHONNOUSERSITE=1
  export ROBODOJO_UV=1 ROBODOJO_VENV="$path" ROBODOJO_SIM_ENV="$path"
  export ROBODOJO_PYTHON="$path/bin/python"
  export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
  hash -r
}
