# RoboDojo scripts

## Public entry points

| Script | Purpose |
| --- | --- |
| [robodojo.sh](robodojo.sh) | Main CLI: `doctor`, `eval`, `client`, `smoke`, `benchmark`, `dimensions`, `summarize`, `tasks` |
| [install.sh](install.sh) | One-time environment setup (conda, Isaac Sim, submodules) |
| [init_assets.sh](init_assets.sh) | Download robot/object assets |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `robodojo.sh client` and XPolicyLab) |

## uv entry points

The uv scripts default to the repository's `.venv`; set `ROBODOJO_VENV` or use
`robodojo-uv.sh --venv PATH` to select another environment. They run its Python
directly, without `uv sync` changing the simulator's pinned dependencies.

| Script | Purpose |
| --- | --- |
| [install-uv.sh](install-uv.sh) | Install the simulator environment; supports `--from STEP` and `--skip-system` |
| [robodojo-uv.sh](robodojo-uv.sh) | All CLI commands, including doctor, eval, server, client, smoke, benchmark and summaries |
| [eval_policy-uv.sh](eval_policy-uv.sh) | Direct simulator client, with the same arguments as `eval_policy.sh` |
| [init_assets-uv.sh](init_assets-uv.sh) | Download assets using Git LFS, without Conda activation |
| [download_data-uv.sh](RoboDojo/download_data-uv.sh) | Download demonstration datasets |
| [download_ckpt-uv.sh](RoboDojo/download_ckpt-uv.sh) | Download policy checkpoints |
| [verify_install-uv.sh](internal/verify_install-uv.sh) | Direct environment/config/asset checks |
| [smoke_all_tasks-uv.sh](internal/smoke_all_tasks-uv.sh) | Direct batch runner |

```bash
bash scripts/install-uv.sh --skip-system
bash scripts/init_assets-uv.sh
bash scripts/robodojo-uv.sh doctor --skip-policy
bash scripts/robodojo-uv.sh dimensions

# Connect the simulator to an existing policy server.
bash scripts/robodojo-uv.sh client --task stack_bowls \
  --policy-name ACT --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1
```

Policy dependencies belong in a separate environment: the simulator pins
`websockets==12.0`, while the current XPolicyLab server requires `websockets>=14`.
Pass an absolute policy venv/project path, or `--policy-env uv` to use the
policy's `deploy.yml` `policy_uv_env_path` (falling back to its own `.venv`).
Per-policy installation instructions still apply; these launchers do not install
model-specific dependencies or checkpoints.

```bash
# One task: policy server + simulator, in their respective environments.
bash scripts/robodojo-uv.sh eval --policy-dir XPolicyLab/policy/ACT \
  --task stack_bowls --ckpt YOUR_CHECKPOINT \
  --policy-env /absolute/path/to/act/.venv --eval-num 1

# Batch run; --dry-run prints commands without launching either process.
bash scripts/robodojo-uv.sh benchmark --policy-dir XPolicyLab/policy/ACT \
  --ckpt YOUR_CHECKPOINT --policy-env /absolute/path/to/act/.venv \
  --only stack_bowls,push_T --gpu-ids 0,1 --eval-num 1 --dry-run

# On a policy-only machine, select an installed policy env for the CLI too.
bash scripts/robodojo-uv.sh --venv /absolute/path/to/act/.venv server \
  --policy-dir XPolicyLab/policy/ACT --task stack_bowls --ckpt YOUR_CHECKPOINT \
  --policy-env /absolute/path/to/act/.venv --policy-port 9999 --bind-host 0.0.0.0

bash scripts/RoboDojo/download_data-uv.sh huggingface demo
bash scripts/RoboDojo/download_ckpt-uv.sh huggingface ACT
```

The XPolicyLab changes under `utils/activate_env.sh`, shared client launchers,
and policy server launchers are part of this uv support. Keep those submodule
changes with the RoboDojo changes. They preserve policy-specific checkpoint
and runtime setup, and still accept Conda names through the original entry
points. The Docker image and Docker scripts retain their existing installation
flow.

The installer preserves `install.sh`'s Starlette compatibility downgrade via
an explicit uv override, so `uv pip check` can report that known IsaacLab
metadata mismatch. GPU rendering still needs an actual simulator run. Isaac Sim
may ask for EULA acceptance on first import. See [UV.md](UV.md) for a Chinese
walkthrough and a demo debug run that needs no GPU or checkpoint.

## Typical eval flow

```text
robodojo.sh eval
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine (see docs/SPLIT_EVAL.md):

robodojo.sh server  ->  scripts/internal/run_policy_server.sh  ->  policy server (bind 0.0.0.0)
robodojo.sh client  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

Run one or more official capability dimensions in a benchmark sweep:

```bash
bash scripts/robodojo.sh dimensions
bash scripts/robodojo.sh benchmark \
  --dimension memory,long-horizon \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CHECKPOINT> \
  --policy-env <ENV> \
  --eval-num native
```

Available dimensions are `generalization`, `memory`, `precision`,
`long-horizon`, and `open`. Generalization includes both the 12 standard tasks
and their 12 runnable `_random` layout variants. Combine `--dimension` with
`--only` or `--tasks-file` to narrow a dimension further.

## Auto multi-GPU grouping

`robodojo.sh smoke` and `robodojo.sh benchmark` now support balanced
multi-GPU execution driven by the runtime table in `../optimal_8group.txt`.
Pass a concrete GPU id list and RoboDojo will partition the selected tasks
online instead of relying on a hard-coded task group.

Dry-run example:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num 1 \
  --gpu-ids 0,2,5,7 \
  --dry-run
```

Launch only a subset of tasks:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num native \
  --only imitate_sorting_sequence,pour_by_language,play_tic_tac_toe \
  --gpu-ids 0,1,3
```

## Internal (`internal/`)

Not intended for direct daily use. Called by `robodojo.sh` or policy utilities.

| File | Called by |
| --- | --- |
| [verify_install.sh](internal/verify_install.sh) | `robodojo.sh doctor` |
| [task_inventory.py](internal/task_inventory.py) | `robodojo.sh tasks` |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | `robodojo.sh smoke` / `benchmark` |
| [summarize_result.py](internal/summarize_result.py) | `robodojo.sh summarize` |
| [stat_score_distribution.py](internal/stat_score_distribution.py) | Offline score histogram analysis (manual) |

## Docker

Container install and smoke tests live under [../docker/](../docker/), not here.

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).
