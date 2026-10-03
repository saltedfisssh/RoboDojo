"""Physically check state capture/restore and a short continuation before task collection."""

import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    from src.datagen.frozen import install

    revision = install()
    from src.datagen.layouts import sample_layout
    from src.datagen.lease import acquire

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--priority", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    args.kit_args += " --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    if args.task == "match_and_pick_from_conveyor":
        args.kit_args += " --enable isaacsim.asset.gen.conveyor"
    lease = acquire(priority=args.priority)
    app = AppLauncher(args).app
    report = {"task": args.task, "seed": args.seed, "source_snapshot": revision, "passed": False}
    env = None
    try:
        import numpy as np

        from src.datagen.runtime import ActionAdapter, create_env
        from src.datagen.state import RigidStateAdapter

        env = create_env(app, args.task, args.seed, sample_layout(args.task, args.seed), cameras=False, planning=False)
        actions = ActionAdapter(env)
        adapter = RigidStateAdapter(env, actions)
        report["state_schema"] = adapter.schema
        report["scene_adaptations"] = env.datagen_scene_adaptations
        for _ in range(5):
            actions.apply(actions.command)
        from src.datagen.articulations import diagnostics

        report["articulations_before_restore"] = {k: diagnostics(obj) for k, obj in adapter.articulations.items()}
        arrays, context = adapter.capture()
        report["secondary_state_diagnostics"] = adapter.secondary.diagnostics()
        command = actions.command.copy()
        reference = []
        for _ in range(10):
            actions.apply(command)
            reference.append(adapter.capture()[0])
        report["evolution_before_restore"] = adapter.rollout_errors(arrays)
        report["articulations_reference_end"] = {k: diagnostics(obj) for k, obj in adapter.articulations.items()}
        adapter.restore(arrays, context)
        errors = adapter.errors(arrays)
        report["restore_errors"] = errors
        report["restore_max_abs_error"] = max(errors.values(), default=0)
        report["restoration_verified"] = report["restore_max_abs_error"] <= 1e-4
        report["rollout"] = []
        for i in range(10):
            actions.apply(command)
            metrics = adapter.rollout_errors(reference[i])
            metrics["step"] = i + 1
            report["rollout"].append(metrics)
        report["passed"] = report["restoration_verified"] and all(
            np.isfinite(value) for row in report["rollout"] for value in row.values()
        )
        report["scope"] = "Initial scene after settling; does not validate restoration during grasping/folding/pouring."
    except Exception as exc:
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("rollout", "restore_errors")}), flush=True)
        if env is not None:
            env.close()
        app.close()
        lease.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
