# Task planner development

Read `docs/expert/DEVELOPMENT.md` and the task's `HANDOFF.md` before editing.

- Keep task-specific code, sampler adapters and evidence in its own directory.
  Discovery is automatic through `planner.json`; no central registry edits are needed.
- Use a separate checkout/worktree for each contributor. Shared-core fixes should
  have their own regression coverage and must preserve existing task behavior.
- Read the official instruction, reward, failure queries and step budget. Preserve
  them and the complete configured layout/asset distribution. Do not teleport
  objects, loosen success checks, enlarge the budget or cherry-pick validation seeds.
- All actions go through `context.motion` / `context.skills`. Let `EpisodeEnd`
  propagate. Log subtask/stage semantics and query live state after movement.
- Iterate with `generate_data.sh dev run --split development`. Read report.json,
  raw/diagnostics, failed HDF5 and worker.log before choosing the next change.
- Use a fresh output directory for each version. Freeze source for validation,
  RGB and qualification runs. Reused validation seeds are regression evidence,
  not a new held-out estimate. Report all attempted seeds and failed runs.
- Keep the plugin experimental until its physical coverage and qualification
  bundle have been reviewed. CPU tests alone never establish physical success.
- Never stop unrelated simulator jobs or bypass a held development GPU lock.
  The lock coordinates these development commands, not arbitrary external jobs.
