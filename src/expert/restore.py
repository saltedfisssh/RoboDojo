"""Restore a recorded frame in Isaac and continue (replay, hold, or a VLA policy).

Examples:
  # restore frame 120 of a raw episode, check the error, replay 60 recorded actions
  bash scripts/expert.sh restore --episode data/expert/stack_blocks/episode_0100000.h5 --frame 120 --steps 60

  # atomic test from the start of subtask 1 in an exported dataset, driven by a policy server
  bash scripts/expert.sh restore --dataset data/expert_lerobot/stack_blocks --episode-index 3 --subtask 1 \\
      --policy starVLA --policy-url ws://127.0.0.1:6000 --steps 300 --report out.json

Selection: exactly one of --frame / --time (s) / --subtask (planner subtask id).
The policy is driven by its own ``XPolicyLab/policy/<name>/deploy.py::eval_one_episode``
through an env that exposes the same get_obs/take_action/is_episode_end as EvalEnv.
The episode ends when the atomic goal (subtask) or the full task succeeds, or after --steps.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from src.expert.arguments import nonnegative_int


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--episode", type=Path, help="raw expert episode .h5")
    src.add_argument("--dataset", type=Path, help="exported LeRobot root (uses states/ sidecars)")
    ap.add_argument("--episode-index", type=int, default=0)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--frame", type=int)
    sel.add_argument("--time", type=float)
    sel.add_argument("--subtask", type=int)
    ap.add_argument("--steps", type=int, default=0, help="actions to run after restoring")
    ap.add_argument("--policy", help="XPolicyLab policy name; omit to replay recorded actions")
    ap.add_argument("--policy-url", default=os.environ.get("ROBODOJO_POLICY_URL", "ws://127.0.0.1:6000"))
    ap.add_argument("--instruction", choices=("subtask", "task"), default="subtask")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--video", type=Path, help="write cam_high mp4 of the continuation")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--gpu", type=nonnegative_int, default=0)
    return ap.parse_args(argv)


def load_episode(args):
    import h5py

    if args.episode is not None:
        path = args.episode
    else:
        rows = [json.loads(lbl) for lbl in (args.dataset / "meta/replay.jsonl").read_text().splitlines()]
        path = args.dataset / next(r for r in rows if r["episode_index"] == args.episode_index)["state_path"]
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        segments = json.loads(f.attrs["segments"])
        subtasks = json.loads(f.attrs["subtasks"])
    n = meta["num_frames"]
    if args.frame is not None:
        frame = args.frame
    elif args.time is not None:
        frame = int(np.floor(args.time * meta["fps"] + 1e-6))
    else:
        starts = [s["start_frame"] for s in segments if s["subtask"] == args.subtask and not s.get("background")]
        if not starts:
            raise SystemExit(f"subtask {args.subtask} not in episode (have {[s['id'] for s in subtasks]})")
        frame = min(starts)
    if not 0 <= frame < n:
        raise SystemExit(f"frame {frame} outside [0, {n})")
    return path, meta, segments, subtasks, frame


def checkpoint(f, frame):
    import h5py

    g = f["checkpoint"]
    arrays = {}
    g["arrays"].visititems(
        lambda name, obj: arrays.__setitem__(name, np.asarray(obj[frame])) if isinstance(obj, h5py.Dataset) else None
    )
    context = json.loads(g["context"][frame])
    return arrays, context


def replay_recording(task_env, env, f, adapter, *, frame, budget, video=False):
    """Replay actions and compare state only with a checkpoint at the same time.

    Checkpoints precede actions. There is no checkpoint after the final action,
    so a complete replay measures drift immediately before that action instead.
    """
    recorded = f["action"][()]
    report = {}

    def compare(index):
        report["replay_drift_frame"] = index
        report["replay_drift"] = adapter.rollout_errors(checkpoint(f, index)[0])

    for index in range(frame, min(budget, len(recorded))):
        if video:
            task_env.get_obs()
        if index == len(recorded) - 1:
            compare(index)
        vec = recorded[index]
        task_env.take_action(
            {
                "left_arm_joint_state": vec[0:6],
                "left_ee_joint_state": vec[6:7],
                "right_arm_joint_state": vec[7:13],
                "right_ee_joint_state": vec[13:14],
            }
        )
        if env.end_flag[0]:
            break
    next_frame = env.take_action_cnt[0]
    if frame < next_frame < len(recorded):
        compare(next_frame)
    return report


class Shim:
    """What RigidStateAdapter expects from the action side: command and home."""

    def __init__(self, command, home):
        self.command, self.home = np.asarray(command, float), np.asarray(home, float)


def subtask_at(segments, subtasks, frame):
    sid = -1
    for s in segments:
        if not s.get("background") and s["start_frame"] <= frame < s["end_frame"]:
            sid = s["subtask"]
    return next((t for t in subtasks if t["id"] == sid), None)


def atomic_goal(segments, subtask):
    """Object-origin target for a pick-and-place subtask, from the planner's transfer goal."""
    if subtask is None:
        return None
    for s in segments:
        if s["subtask"] == subtask["id"] and s["primitive"] == "transfer" and "object_position" in s["goal"]:
            return {"object": s["object"], "position": s["goal"]["object_position"]}
    return None


def main(argv=None):
    args = parse(argv)
    path, meta, segments, subtasks, frame = load_episode(args)
    import h5py

    from src.expert.runtime import simulation_session

    with simulation_session(meta["task"], images=not args.no_images, device_id=args.gpu) as env:
        with h5py.File(path, "r") as f:
            restore_episode(args, env, path, f, meta, segments, subtasks, frame)


def restore_episode(args, env, path, f, meta, segments, subtasks, frame):
    from src.datagen.state import RigidStateAdapter

    layout = json.loads(f.attrs["layout"])
    if not env.reset_layout(meta["seed"], layout):
        raise SystemExit("recorded layout failed the stability check on reload")
    arrays, context = checkpoint(f, frame)
    shim = Shim(arrays["command"], arrays["home"])
    env.datagen_render_warmup = False
    adapter = RigidStateAdapter(env, shim)
    adapter.restore(arrays, context)
    env.take_action_cnt = [frame]
    env.end_flag = [False]
    errors = adapter.rollout_errors(arrays)
    print("[restore] frame", frame, "errors", json.dumps(errors), flush=True)

    sub = subtask_at(segments, subtasks, frame)
    goal = atomic_goal(segments, sub)
    instruction = sub["text"] if (sub and args.instruction == "subtask") else meta["instruction"]
    report = {
        "episode": str(path),
        "task": meta["task"],
        "seed": meta["seed"],
        "frame": frame,
        "time_s": frame / meta["fps"],
        "subtask": sub,
        "instruction": instruction,
        "restore_errors": errors,
        "mode": "policy" if args.policy else "replay",
    }

    lm = env.scene_manager.layout_manager

    def goal_met():
        if goal is None:
            return False
        pos, _ = lm.get_instance_pose(0, label=goal["object"])
        pos = np.asarray(pos, float).reshape(3)
        target = np.asarray(goal["position"], float)
        held = any(env.robot_manager.get_end_effector_real_val(r)[0][0] < r.gripper_scale[1] - 0.01 for r in env.arms)
        return bool(np.linalg.norm(pos[:2] - target[:2]) < 0.02 and abs(pos[2] - target[2]) < 0.02 and not held)

    frames = []
    budget = frame + args.steps
    state = {"atomic_success": False, "steps": 0}

    class TaskEnv:
        """EvalEnv-compatible surface for XPolicyLab deploy.eval_one_episode."""

        def get_obs(self):
            if not args.no_images:
                env.obs_manager.render_for_capture()
                obs = env.obs_manager.get_obs([0])[0]
            else:
                cm = env.obs_manager.capture_manager
                env.obs_manager.capture_manager = None
                obs = env.obs_manager.get_obs([0])[0]
                env.obs_manager.capture_manager = cm
            obs = deepcopy(obs)
            obs["instruction"] = instruction
            obs["env_idx"] = 0
            if args.video and "cam_head" in obs["vision"]:
                frames.append(np.asarray(obs["vision"]["cam_head"]["color"]).copy())
            return obs

        def take_action(self, action):
            vec = []
            for side in ("left", "right"):
                vec.extend(np.asarray(action[f"{side}_arm_joint_state"], float).reshape(-1))
                vec.extend(np.asarray(action[f"{side}_ee_joint_state"], float).reshape(-1)[:1])
            env.take_joint_action(np.asarray(vec))
            state["steps"] += 1
            if goal_met():
                state["atomic_success"] = True

        def is_episode_end(self):
            return env.end_flag[0] or state["atomic_success"] or env.take_action_cnt[0] >= budget

    task_env = TaskEnv()
    t0 = time.time()
    if args.steps and args.policy:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "XPolicyLab"))
        from client_server.ws.model_client import WsModelClient

        from src.eval_client.eval_env import _patch_websockets_proxy_compat

        _patch_websockets_proxy_compat()
        run = f"restore-{meta['task']}-{meta['seed']}-{frame}"
        client = WsModelClient(
            url=args.policy_url, evaluation_id=run, trial_id=run, action_case_id=f"{meta['task']}_atomic"
        )
        deploy = __import__(f"XPolicyLab.policy.{args.policy}.deploy", fromlist=["eval_one_episode"])
        deploy.eval_one_episode(TASK_ENV=task_env, model_client=client)
    elif args.steps:
        report.update(replay_recording(task_env, env, f, adapter, frame=frame, budget=budget, video=bool(args.video)))
    report.update(
        steps=state["steps"],
        seconds=round(time.time() - t0, 1),
        atomic_goal=goal,
        atomic_success=state["atomic_success"],
        task_success=bool(env.end_flag[0] and env.success[0]),
    )
    print("[restore] report", json.dumps(report), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    if args.video and frames:
        import imageio.v2 as imageio

        imageio.mimwrite(args.video, frames, fps=25)


if __name__ == "__main__":
    main()
