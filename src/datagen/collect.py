"""Collect one physical demonstration; batch.py supervises isolated seed attempts."""

import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    from src.datagen.frozen import install

    source_revision = install()
    from src.datagen.lease import acquire

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if Path("/etc/vulkan/icd.d/nvidia_icd.json").is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-images", action="store_true", help="State-only debug episode; not exportable for VLA training"
    )
    parser.add_argument("--max-frames", type=int, help="Default: 5000 for the ten-piece stacking toy, otherwise 3600")
    parser.add_argument("--priority", type=int, help="GPU queue priority; defaults to 0 for debug and 10 for RGB")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    from src.datagen.layouts import sample_layout
    from src.datagen.planner import SUPPORTED_TASKS

    if args.task not in SUPPORTED_TASKS:
        parser.error(f"Supported task planners: {SUPPORTED_TASKS}")
    if args.max_frames is None:
        args.max_frames = 5000 if args.task == "play_stacking_toy" else 3600
    if args.max_frames < 1:
        parser.error("max-frames must be positive")
    if args.output.exists() or args.output.with_suffix(".partial.h5").exists():
        parser.error("Output already exists; choose a fresh path")
    layout = sample_layout(args.task, args.seed)
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    args.kit_args += " --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    if args.task == "match_and_pick_from_conveyor":
        args.kit_args += " --enable isaacsim.asset.gen.conveyor"
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    priority = args.priority if args.priority is not None else (0 if args.no_images else 10)
    gpu_lease = acquire(priority=priority)
    if args.output.exists() or args.output.with_suffix(".partial.h5").exists():
        gpu_lease.close()
        parser.error("Output was created while waiting for the GPU; choose a fresh path")
    app = AppLauncher(args).app
    env = writer = states = planner = None
    status = 2
    try:
        from src.datagen.planner import RigidPlanner
        from src.datagen.provenance import asset_manifest, code_revision, runtime_versions, source_manifest
        from src.datagen.recording import AsyncEpisodeWriter, EpisodeWriter
        from src.datagen.runtime import ActionAdapter, create_env
        from src.datagen.state import RigidStateAdapter

        env = create_env(app, args.task, args.seed, layout, cameras=not args.no_images)
        actions = ActionAdapter(env)
        states = RigidStateAdapter(env, actions)
        metadata = {
            "task": args.task,
            "seed": args.seed,
            "layout": layout,
            "fps": 25,
            "instruction": str(env.obs_manager.instruction[0]),
            "dt": env.dt,
            "action_semantics": "absolute_joint_position; eval_80pct_interpolation; gripper_open_fraction",
            "planner_batch_capacity": env.datagen_planner_batch_capacity,
            "curobo_cuda_graphs": env.datagen_curobo_cuda_graphs,
            "state_timing": "before_action",
            "benchmark_action_budget": int(env.step_lim) if env.datagen_task_definition == "official" else None,
            "collection_frame_limit": args.max_frames,
            "task_definition": env.datagen_task_definition,
            "support_demonstration": "generated_curobo"
            if args.task == "imitate_sorting_sequence"
            else "shipped_trajectory"
            if getattr(env, "interact", False)
            else None,
            "state_gripper": "previous_command",
            "versions": runtime_versions(),
            "git_revision": code_revision(),
            "source_sha256": source_manifest(),
            "source_snapshot": source_revision,
            "asset_sha256": asset_manifest(
                layout, robot_names=[r.robot_name for r in env.robot_manager.robot_list], task=args.task
            ),
            "env_config": env.datagen_env_config,
            "state_schema": states.schema,
            "scene_adaptations": env.datagen_scene_adaptations,
            "sampler": "asset_metadata_v1",
        }
        writer = (EpisodeWriter if args.no_images else AsyncEpisodeWriter)(args.output, metadata)
        planner = RigidPlanner(env, actions, writer, states, max_frames=args.max_frames)
        success = planner.run(args.task)
        writer.file.attrs["frame_timings_seconds"] = json.dumps(planner.timings)
        writer.file.attrs["action_timings_seconds"] = json.dumps(actions.timings)
        writer.finish(success, planner.failure_reason, states.capture())
        writer = None
        print(json.dumps({"success": success, "output": str(args.output)}), flush=True)
        status = 0 if success else 2
    except Exception as exc:
        traceback.print_exc()
        if writer is not None and writer.file.id.valid:
            try:
                if planner is not None:
                    writer.file.attrs["frame_timings_seconds"] = json.dumps(planner.timings)
                terminal = None
                if states is not None:
                    try:
                        terminal = states.capture()
                    except Exception:
                        traceback.print_exc()
                writer.finish(False, f"{type(exc).__name__}: {exc}", terminal)
            except Exception:
                writer.abort()
                traceback.print_exc()
            writer = None
        print("[datagen] Attempt failed; artifacts finalized; closing Isaac.", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()
        gpu_lease.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
