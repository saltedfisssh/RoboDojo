#!/usr/bin/env python3
"""Shorten existing offline demonstrations only after native Isaac replay passes."""

import argparse
from datetime import UTC, datetime
import faulthandler
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_sorting_eval_layouts import TASK, owner_matches
from src.eval_layouts.lease import TaskLock
from src.eval_layouts.storage import DEFAULT_ROOT, atomic_json, existing_records, paths, sha256


def run(args):
    scratch = ROOT / "data/eval_layout_generation" / str(args.set_seed) / TASK
    from src.expert.sim import launch_app
    from utils.load_file import load_object_metadata  # noqa: F401  (preload before Kit)

    app = launch_app(headless=True, cameras=True, device_id=args.device)
    import torch

    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    from src.eval_layouts.support_trajectory import DemoRecorder, compact_commands
    from src.eval_layouts.validation import initialize_and_render, make_env

    env = None
    shadow = scratch / "compact_candidate_root"
    directory = shadow / "RoboDojo/arx_x5" / str(args.set_seed)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / "candidate.json"
    for i in range(50):
        alias = directory / f"{TASK}_{i}.json"
        alias.unlink(missing_ok=True)
        alias.symlink_to(candidate)
    for index in args.indices:
        with TaskLock(scratch / "parallel" / str(index)) as lock:
            owner_matches(args.coordinator, args.token)
            record = next(
                r
                for r in existing_records(args.output, TASK, args.set_seed, require_contiguous=False)
                if r["index"] == index
            )
            check = record["validation"]["support_trajectory"]
            if check.get("trajectory_export_stride", 1) >= args.stride:
                print("TRAJECTORY_ALREADY_COMPACT", index, flush=True)
                continue
            original = args.output / record["trajectory_path"]
            data = pickle.loads(original.read_bytes())
            raw = [step for chunk in data["joint_path"] for step in chunk]
            exported = compact_commands(raw, args.stride)
            data["joint_path"] = [exported]
            trajectory = shadow / original.relative_to(args.output)
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            trajectory.write_bytes(pickle.dumps(data, protocol=4))
            scene, _, meta = paths(args.output, TASK, index, args.set_seed)
            shutil.copyfile(scene, candidate)
            started = time.monotonic()
            if env is None:
                env = make_env(TASK, app, shadow, args.set_seed, 16, "fast", skip_policy_planners=True)
            frame, initialization = initialize_and_render(env, index)
            env.run_reward()
            env.query_support_arm_traj(0)
            replay = DemoRecorder(env, record=False)
            while env.support_arm_action[0]:
                replay.hold(1)
            replay.hold(20)
            env.check_support_arm_stable(0)
            inside = [
                env.reward_manager.call_func_parser(
                    env.reward_manager.is_A_in_B(label_A=f"aim{i}", label_B="basket1"), 0
                )
                for i in range(5)
            ]
            if min(inside) < 0.5 or env.unstable_envs:
                raise ValueError(f"Shortened demonstration failed native replay: index={index}, inside={inside}")
            if sha256(scene) != record["json_sha256"]:
                raise RuntimeError("Original layout changed during replay")
            lock.ensure_owned()
            owner_matches(args.coordinator, args.token)
            shutil.copyfile(trajectory, original.with_suffix(".pkl.tmp"))
            os.replace(original.with_suffix(".pkl.tmp"), original)
            record["trajectory_sha256"] = sha256(original)
            check.update(
                physics_steps=len(exported),
                planning_physics_steps=len(raw),
                trajectory_export_stride=args.stride,
                native_replay=True,
                aim_in_basket=inside,
                gripper_timing_preserved=True,
                compact_replay_at=datetime.now(UTC).isoformat(),
            )
            record["validation"]["trajectory_recheck"] = initialization
            atomic_json(meta, record)
            print(
                "TRAJECTORY_COMPACT_REPLAY_PASSED",
                index,
                len(raw),
                len(exported),
                round(time.monotonic() - started, 3),
                inside,
                flush=True,
            )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--set-seed", type=int, default=1000)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stride", type=float, default=2)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    if args.stride <= 1 or args.stride > 4:
        parser.error("Stride must be in (1, 4]")
    args.output = args.output.resolve()
    faulthandler.enable()
    try:
        code = run(args)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
