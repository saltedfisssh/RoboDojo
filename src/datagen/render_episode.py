"""Render a verified physical plan by action replay or per-frame checkpoint restoration."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

import h5py


def main():
    from src.datagen.frozen import install

    revision = install()
    from src.datagen.lease import acquire
    from src.datagen.provenance import sha256

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--priority", type=int, default=10, help="GPU queue priority; lower values run first")
    parser.add_argument(
        "--mode",
        choices=("physical", "state"),
        default="physical",
        help="physical: execute actions again; state: render the already verified physical states",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".partial.h5").exists():
        parser.error("Output already exists")
    with h5py.File(args.input) as source:
        if not source.attrs.get("complete") or not source.attrs.get("success") or "terminal" not in source:
            parser.error("Input must be a complete physically successful trajectory with terminal state")
        metadata = json.loads(source.attrs["metadata"])
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    args.kit_args += " --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    if metadata["task"] == "match_and_pick_from_conveyor":
        args.kit_args += " --enable isaacsim.asset.gen.conveyor"
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    lease = acquire(priority=args.priority)
    if args.output.exists() or args.output.with_suffix(".partial.h5").exists():
        lease.close()
        parser.error("Output was created while waiting for the GPU")
    app = AppLauncher(args).app
    env = writer = adapter = None
    report = {
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "mode": args.mode,
        "success": False,
    }
    try:
        from src.datagen.provenance import source_manifest, verify_provenance
        from src.datagen.recording import AsyncEpisodeWriter, read_snapshot
        from src.datagen.runtime import ActionAdapter, create_env
        from src.datagen.state import RigidStateAdapter, terminal_checks

        verify_provenance(metadata)
        env = create_env(
            app,
            metadata["task"],
            metadata["seed"],
            metadata["layout"],
            cameras=True,
            env_config=metadata.get("env_config"),
            scene_adaptations=metadata.get("scene_adaptations", []),
            planning=False,
        )
        checks = terminal_checks(env.reward_manager)
        actions = ActionAdapter(env)
        adapter = RigidStateAdapter(env, actions)
        with h5py.File(args.input) as source:
            arrays, context = read_snapshot(source, 0)
            adapter.restore(arrays, context)
            if metadata["task"] == "imitate_sorting_sequence":
                from src.datagen.support_demo import recorded_suffix

                env.support_arm_action = [recorded_suffix(source, 0)]
            report["initial_restore_error"] = max(adapter.errors(arrays).values(), default=0)
            if report["initial_restore_error"] > 1e-4:
                raise RuntimeError("Initial physical checkpoint did not restore accurately")
            metadata = {
                **metadata,
                "planned_source_snapshot": metadata.get("source_snapshot"),
                "source_snapshot": revision,
                "source_sha256": source_manifest(),
                "generation_mode": (
                    "physical_replay_of_successful_plan"
                    if args.mode == "physical"
                    else "checkpoint_render_of_successful_physical_plan"
                ),
                "plan_source": str(args.input.resolve()),
                "plan_sha256": sha256(args.input),
                "state_schema": adapter.schema,
            }
            writer = AsyncEpisodeWriter(args.output, metadata)
            previous_segment = None
            stable = []
            timings = dict(observation=0.0, images=0.0, checkpoint=0.0, write=0.0, physics=0.0)
            if args.mode == "state":
                report["max_frame_restore_error"] = 0.0
                report["max_visual_position_error_m"] = 0.0
                report["max_visual_angle_error_rad"] = 0.0
            count = int(source.attrs["num_frames"])
            for i in range(count):
                if args.mode == "state":
                    stamp = perf_counter()
                    arrays, context = read_snapshot(source, i)
                    adapter.restore(arrays, context, warmup_render=(i == 0))
                    error = max(adapter.errors(arrays).values(), default=0)
                    report["max_frame_restore_error"] = max(report["max_frame_restore_error"], error)
                    if error > 1e-4:
                        raise RuntimeError(f"Frame {i} did not restore accurately: {error}")
                    timings["checkpoint"] += perf_counter() - stamp
                annotation = json.loads(source["annotations"][i])
                if annotation["segment_id"] != previous_segment:
                    print(f"[render] frame={i}/{count}: {annotation['text']}", flush=True)
                    previous_segment = annotation["segment_id"]
                action = source["action"][i]
                stamp = perf_counter()
                observed, measured = actions.observation_state(), actions.measured_state()
                now = perf_counter()
                timings["observation"] += now - stamp
                stamp = now
                images = actions.images()
                if args.mode == "state":
                    visual_errors = adapter.visual_state.errors()
                    report["max_visual_position_error_m"] = max(
                        report["max_visual_position_error_m"], visual_errors["position_m"]
                    )
                    report["max_visual_angle_error_rad"] = max(
                        report["max_visual_angle_error_rad"], visual_errors["angle_rad"]
                    )
                    if visual_errors["position_m"] > 1e-4 or visual_errors["angle_rad"] > 1e-4:
                        raise RuntimeError(f"Rendered frame {i} differs from articulation physics: {visual_errors}")
                now = perf_counter()
                timings["images"] += now - stamp
                stamp = now
                snapshot = adapter.capture()
                now = perf_counter()
                timings["checkpoint"] += now - stamp
                stamp = now
                writer.append(observed, measured, action, images, snapshot, annotation)
                now = perf_counter()
                timings["write"] += now - stamp
                stamp = now
                if args.mode == "physical":
                    actions.apply(action)
                    timings["physics"] += perf_counter() - stamp
                if (i + 1) % 100 == 0:
                    print(f"[timing] frames={i + 1}: {json.dumps(timings)}", flush=True)
                if (args.mode == "physical" and i >= count - 10) or (args.mode == "state" and i >= count - 9):
                    stable.append(all(env.reward_manager.check_once(check, 0) for check in checks))
            terminal = {}
            source["terminal"].visititems(
                lambda name, obj: terminal.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None
            )
            if args.mode == "state":
                adapter.restore(terminal, json.loads(source.attrs["terminal_context"]))
                error = max(adapter.errors(terminal).values(), default=0)
                report["terminal_restore_error"] = error
                if error > 1e-4:
                    raise RuntimeError(f"Terminal state did not restore accurately: {error}")
                stable.append(all(env.reward_manager.check_once(check, 0) for check in checks))
            report["terminal_deviation"] = adapter.rollout_errors(terminal)
            report["failed_terminal_predicates"] = [
                repr(check) for check in checks if not env.reward_manager.check_once(check, 0)
            ]
            report["success"] = (
                not report["failed_terminal_predicates"]
                and all(stable)
                and bool(env.reward_manager.get_reward()[0])
                and all(env.success)
            )
            report["frames"] = count
            report["frame_timings_seconds"] = timings
            report["action_timings_seconds"] = actions.timings
            writer.file.attrs["frame_timings_seconds"] = json.dumps(timings)
            report["last_ten_frames_stable"] = all(stable)
            report["stability_frames_checked"] = len(stable)
            writer.finish(
                report["success"],
                None if report["success"] else "Rendered episode did not pass the original task predicates",
                adapter.capture(),
            )
            writer = None
    except Exception as exc:
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
        if writer is not None:
            try:
                writer.finish(False, report["error"], adapter.capture())
            except Exception:
                writer.abort()
                traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report), flush=True)
        if env is not None:
            env.close()
        app.close()
        lease.close()
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
