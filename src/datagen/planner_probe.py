"""Run a short planner program from a recorded checkpoint; never publish a full episode."""

import argparse
import importlib
import json
import os
from pathlib import Path
import traceback


def main():
    from src.datagen.frozen import install

    revision = install()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--frame", type=int)
    selector.add_argument("--time", type=float)
    selector.add_argument("--subtask", type=int)
    parser.add_argument("--program", required=True, help="module:function(planner); a short diagnostic suffix")
    parser.add_argument("--output", type=Path, required=True, help="Diagnostic HDF5, never accepted as a full episode")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--priority", type=int, default=0, help="GPU queue priority; lower values run first")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 600:
        parser.error("A short planner probe must use 1..600 frames; use collect for complete episodes")
    report_path = args.output.with_suffix(".probe.json")
    if any(path.exists() for path in (args.output, args.output.with_suffix(".partial.h5"), report_path)):
        parser.error("Output exists; choose a fresh path")
    import h5py

    from src.datagen.recording import read_snapshot, select_frame

    with h5py.File(args.input) as source:
        metadata = json.loads(source.attrs["metadata"])
        frame = select_frame(source, frame=args.frame, time_s=args.time, subtask=args.subtask)
        arrays, context = read_snapshot(source, frame)
        annotation = json.loads(source["annotations"][frame])
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    args.kit_args += " --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    if metadata["task"] == "match_and_pick_from_conveyor":
        args.kit_args += " --enable isaacsim.asset.gen.conveyor"
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    from src.datagen.lease import acquire

    # Bounded suffix probes take the same exclusive lease before any Isaac
    # initialization. They let us check a correction before another full run.
    lease = acquire(priority=args.priority)
    app = AppLauncher(args).app
    env = writer = states = None
    report = {
        "source": str(args.input.resolve()),
        "source_frame": frame,
        "source_subtask": annotation,
        "program": args.program,
        "source_snapshot": revision,
        "scope": "Checkpoint-resumed diagnostic suffix; not a complete independently collected episode.",
        "restore_passed": False,
        "program_completed": False,
        "task_after_suffix_passed": False,
    }
    try:
        from src.datagen.planner import RigidPlanner
        from src.datagen.provenance import sha256, source_manifest, verify_provenance
        from src.datagen.recording import EpisodeWriter
        from src.datagen.runtime import ActionAdapter, create_env
        from src.datagen.state import RigidStateAdapter, plain, terminal_checks

        verify_provenance(metadata)
        env = create_env(
            app,
            metadata["task"],
            metadata["seed"],
            metadata["layout"],
            cameras=False,
            env_config=metadata.get("env_config"),
            scene_adaptations=metadata.get("scene_adaptations", []),
            planning=True,
        )
        actions = ActionAdapter(env, metadata["fps"])
        states = RigidStateAdapter(env, actions)
        original_terminal_checks = terminal_checks(env.reward_manager)
        states.restore(arrays, context)
        report["restore_max_abs_error"] = max(states.errors(arrays).values(), default=0)
        report["restore_passed"] = report["restore_max_abs_error"] <= 1e-4
        if not report["restore_passed"]:
            raise RuntimeError("Checkpoint restore exceeded 1e-4")
        if metadata["task"] == "imitate_sorting_sequence":
            from src.datagen.support_demo import recorded_suffix

            with h5py.File(args.input) as source:
                env.support_arm_action = [recorded_suffix(source, frame)]
        diagnostic_metadata = {
            **metadata,
            "generation_mode": "checkpoint_resumed_planner_diagnostic",
            "source_snapshot": revision,
            "source_sha256": source_manifest(),
            "parent_source": str(args.input.resolve()),
            "parent_sha256": sha256(args.input),
            "source_start_frame": frame,
            "benchmark_action_budget": None,
            "collection_frame_limit": args.max_frames,
        }
        writer = EpisodeWriter(args.output, diagnostic_metadata)
        planner = RigidPlanner(env, actions, writer, states, max_frames=args.max_frames)
        planner.terminal_checks = original_terminal_checks
        module, function = args.program.split(":", 1)
        program_metrics = getattr(importlib.import_module(module), function)(planner)
        if program_metrics is not None:
            report["program_metrics"] = plain(program_metrics)
        report["program_completed"] = True
        report["task_after_suffix_passed"] = planner.finish_task()
        report["failure_reason"] = planner.failure_reason
    except Exception as exc:
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            report["frames"] = writer.count
            writer.file.attrs["diagnostic_program_completed"] = report["program_completed"]
            writer.file.attrs["diagnostic_task_passed"] = report["task_after_suffix_passed"]
            # Existing batch/export/promotion accept complete successful
            # episodes. Keep suffix diagnostics explicitly outside that set.
            try:
                writer.finish(False, "Diagnostic suffix only; see .probe.json", states.capture())
            except Exception as exc:
                traceback.print_exc()
                if writer.file.id.valid:
                    writer.abort()
                report["recording_error"] = f"{type(exc).__name__}: {exc}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report), flush=True)
        if report.get("error"):
            print("[datagen] Attempt failed; artifacts finalized; closing Isaac.", flush=True)
        if env is not None:
            env.close()
        app.close()
        lease.close()
    return 0 if report["program_completed"] and not report.get("error") and not report.get("recording_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
