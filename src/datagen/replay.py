"""Restore a raw/LeRobot episode frame and replay actions or evaluate a local VLA adapter."""

import argparse
import importlib
import json
import os
from pathlib import Path
import traceback

import h5py
import numpy as np

from src.datagen.recording import read_snapshot, select_frame


def resolve_episode(path, episode_index=None):
    path = Path(path)
    if path.is_file():
        return path, 0, None
    if episode_index is None:
        raise ValueError("--episode-index is required for a LeRobot directory")
    manifest = path / "meta/replay.jsonl"
    if not manifest.exists():
        raise ValueError("Dataset has no simulator checkpoints; images and robot joints cannot restore object state")
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    entry = next((r for r in records if r["episode_index"] == episode_index), None)
    if entry is None:
        raise ValueError(f"Episode {episode_index} is absent from replay manifest")
    state_path = (path / entry["state_path"]).resolve()
    if not state_path.is_relative_to(path.resolve()):
        raise ValueError("State path must stay inside the dataset")
    return state_path, entry["source_start_frame"], entry["length"]


def main():
    from src.datagen.lease import acquire

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if Path("/etc/vulkan/icd.d/nvidia_icd.json").is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--episode-index", type=int)
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--frame", type=int)
    selector.add_argument("--time", type=float)
    selector.add_argument("--subtask", type=int, help="Planner segment id")
    p.add_argument("--steps", type=int, default=0, help="Actions after restore (0 verifies restoration only)")
    p.add_argument("--policy", help="module:factory; factory() -> callable(obs), returns 14D action or Nx14 chunk")
    p.add_argument("--success-hook", help="module:function(env, context) -> JSON metrics for custom atomic success")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--image-dir", type=Path, help="Save restored camera observations as PNGs (no physics step)")
    p.add_argument("--restore-tolerance", type=float, default=1e-4)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    if args.steps < 0 or args.restore_tolerance <= 0:
        p.error("steps must be nonnegative and restore-tolerance positive")
    if args.image_dir and args.no_images:
        p.error("--image-dir requires cameras")
    path, offset, length = resolve_episode(args.input, args.episode_index)
    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs["metadata"])
        available = int(f.attrs["num_frames"]) - offset if length is None else length
        if args.subtask is not None:
            frame = select_frame(f, subtask=args.subtask)
            if not offset <= frame < offset + available:
                p.error("Selected subtask is outside this exported episode")
        else:
            local_frame = args.frame
            if args.time is not None:
                if not np.isfinite(args.time) or not 0 <= args.time < available / metadata["fps"]:
                    p.error("Time outside selected episode")
                local_frame = int(np.floor(args.time * metadata["fps"] + 1e-8))
            if not 0 <= local_frame < available:
                p.error("Frame outside selected episode")
            frame = offset + local_frame
        arrays, context = read_snapshot(f, frame)
        annotation = json.loads(f["annotations"][frame])
        goal_frame = frame
        for i in range(frame + 1, int(f.attrs["num_frames"])):
            candidate = json.loads(f["annotations"][i])
            if candidate["segment_id"] != annotation["segment_id"]:
                break
            annotation["goal"] = candidate.get("goal", {})
            goal_frame = i
        annotation["goal_source_frame"] = goal_frame
        end = min(frame + args.steps, offset + available)
        replay_actions = f["action"][frame:end]
        support_suffix = None
        if metadata["task"] == "imitate_sorting_sequence":
            from src.datagen.support_demo import recorded_suffix

            support_suffix = recorded_suffix(f, frame)
        # Atomic slices still have a reference state at their exclusive end.
        source_count = int(f.attrs["num_frames"])
        reference = [read_snapshot(f, i)[0] for i in range(frame + 1, min(end + 1, source_count))]
        if end == source_count and "terminal" in f and len(replay_actions):
            terminal = {}
            f["terminal"].visititems(
                lambda name, obj: terminal.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None
            )
            reference.append(terminal)
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    args.kit_args += " --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    if metadata["task"] == "match_and_pick_from_conveyor":
        args.kit_args += " --enable isaacsim.asset.gen.conveyor"
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    gpu_lease = acquire()
    app = AppLauncher(args).app
    env = policy = None
    report = {"source": str(path), "source_frame": frame, "restore_passed": False}
    status = 1
    try:
        from src.datagen.provenance import verify_provenance
        from src.datagen.runtime import ActionAdapter, create_env
        from src.datagen.state import RigidStateAdapter, plain

        verify_provenance(metadata)
        env = create_env(
            app,
            metadata["task"],
            metadata["seed"],
            metadata["layout"],
            cameras=not args.no_images,
            env_config=metadata.get("env_config"),
            scene_adaptations=metadata.get("scene_adaptations", []),
            planning=False,  # Recorded/VLA joint actions do not use a motion planner.
        )
        actions = ActionAdapter(env, metadata["fps"])
        state = RigidStateAdapter(env, actions)
        state.restore(arrays, context)
        if support_suffix is not None:
            env.support_arm_action = [support_suffix]
        errors = state.errors(arrays)
        maximum = max(errors.values(), default=0)
        report = {
            "source": str(path),
            "source_frame": frame,
            "local_frame": frame - offset,
            "time_s": (frame - offset) / metadata["fps"],
            "subtask": annotation,
            "restore_max_abs_error": maximum,
            "restore_errors": errors,
            "restore_passed": maximum <= args.restore_tolerance,
            "rollout": [],
        }
        if not report["restore_passed"]:
            raise RuntimeError(f"Physical restore error {maximum} exceeds {args.restore_tolerance}")
        if args.image_dir:
            from PIL import Image

            args.image_dir.mkdir(parents=True, exist_ok=True)
            report["restored_images"] = {}
            for name, rgb in actions.images().items():
                output = args.image_dir / f"{name}.png"
                Image.fromarray(rgb).save(output)
                report["restored_images"][name] = str(output)
            report["after_render_max_abs_error"] = max(state.errors(arrays).values(), default=0)
        policy = None
        if args.policy:
            module, factory = args.policy.split(":", 1)
            policy = getattr(importlib.import_module(module), factory)()
        pending = []
        for step in range(args.steps if policy else len(replay_actions)):
            if policy:
                if not pending or hasattr(policy, "observe"):
                    obs = {
                        "observation.state": actions.observation_state(),
                        "images": actions.images(),
                        "task": metadata["instruction"],
                        "current_subtask": annotation["text"],
                        "subtask_id": annotation["segment_id"],
                        "step": step,
                    }
                    if hasattr(policy, "observe"):
                        policy.observe(obs)
                if not pending:
                    chunk = np.asarray(policy(obs), dtype=np.float32)
                    if chunk.shape == (14,):
                        chunk = chunk[None]
                    if chunk.ndim != 2 or chunk.shape[1] != 14 or len(chunk) == 0:
                        raise ValueError("Policy must return a (14,) action or nonempty (N,14) action chunk")
                    pending.extend(chunk)
                action = pending.pop(0)
            else:
                action = replay_actions[step]
            actions.apply(action)
            if step < len(reference):
                error = state.errors(reference[step])
                report["rollout"].append(
                    {
                        "step": step + 1,
                        **state.rollout_errors(reference[step]),
                        "max_abs_error": max(error.values()),
                        "object_pose_error": max(
                            (v for k, v in error.items() if k.startswith("objects/") and k.endswith("/pose")), default=0
                        ),
                    }
                )
        report["executed_steps"] = args.steps if policy else len(replay_actions)
        report["task_definition"] = metadata.get("task_definition", "official")
        report["task_predicate_success"] = bool(env.reward_manager.get_reward()[0])
        report["environment_success"] = bool(all(env.success))
        report["official_task_success"] = (
            report["task_predicate_success"] and report["environment_success"]
            if report["task_definition"] == "official"
            else None
        )
        if getattr(env, "interact", False):
            from src.datagen.interaction import turn_metrics

            start_turn = context["interaction"]["query_support_times"][0]
            report["opponent_turn_metrics"] = turn_metrics(env, max(1, start_turn))
        from src.datagen.atomic import evaluate_atomic

        report["atomic_metrics"] = evaluate_atomic(env, annotation)
        if args.success_hook:
            module, function = args.success_hook.split(":", 1)
            report["atomic_metrics"] = plain(
                getattr(importlib.import_module(module), function)(
                    env, {"metadata": metadata, "annotation": annotation, "start_frame": frame}
                )
            )
        report["requested_steps"] = args.steps
        report["reference_truncated"] = not args.policy and len(replay_actions) < args.steps
        status = 0
    except Exception as exc:
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("rollout", "restore_errors")}), flush=True)
        if policy is not None and hasattr(policy, "close"):
            policy.close()
        if env is not None:
            env.close()
        app.close()
        gpu_lease.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
