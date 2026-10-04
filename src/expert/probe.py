"""Smoke probe: boot Isaac, reset sampled layouts, time stepping."""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="stack_blocks")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
parser.add_argument("--no-cameras", action="store_true")
args = parser.parse_args()

from src.expert.sim import launch_app

t0 = time.time()
app = launch_app(cameras=not args.no_cameras)
print(f"[probe] app {time.time() - t0:.1f}s", flush=True)

import numpy as np

from src.expert.layout import sample_layout
from src.expert.sim import make_env

t0 = time.time()
env = make_env(args.task, app, cameras=not args.no_cameras)
print(f"[probe] env built {time.time() - t0:.1f}s", flush=True)
for seed in args.seeds:
    layout = sample_layout(args.task, seed)
    t0 = time.time()
    ok = env.reset_layout(seed, layout)
    print(f"[probe] seed {seed} reset ok={ok} {time.time() - t0:.1f}s", flush=True)
    obs = env.observe(images=False)
    print("[probe] state", np.round(obs["state"], 3), {k: np.round(v, 3) for k, v in obs["ee_pose"].items()})
    print("[probe] instruction", obs["instruction"])
    lm = env.scene_manager.layout_manager
    for rec in lm.get_layout_records(0, "Rigid"):
        pos, rot = lm.get_instance_pose(0, label=rec["label"])
        print(
            "[probe]", rec["label"], np.round(np.asarray(pos).reshape(-1), 3), np.round(np.asarray(rot).reshape(-1), 3)
        )
    action = obs["state"].copy()
    t0 = time.time()
    for _ in range(25):
        env.take_joint_action(action)
    print(f"[probe] 25 steps no images {time.time() - t0:.2f}s", flush=True)
    if not args.no_cameras:
        t0 = time.time()
        for _ in range(25):
            o = env.observe(images=True)
            env.take_joint_action(action)
        print(f"[probe] 25 steps w/ images {time.time() - t0:.2f}s", {k: v.shape for k, v in o["images"].items()})
    env.run_reward()
    print("[probe] reward", env.reward_manager.get_reward(final_check=False), "steps", env.take_action_cnt)
env.close()
app.close()
