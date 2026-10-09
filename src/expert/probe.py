"""Smoke probe: boot Isaac, reset sampled layouts, time stepping."""

import argparse
import time

from src.expert.arguments import nonnegative_int


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--gpu", type=nonnegative_int, default=0)
    args = parser.parse_args(argv)
    from src.expert.registry import official_tasks

    if args.task not in official_tasks():
        parser.error(f"Unknown official task: {args.task}")
    import numpy as np

    from src.expert.layout import sample_layout
    from src.expert.runtime import simulation_session

    with simulation_session(args.task, images=not args.no_cameras, device_id=args.gpu) as env:
        for seed in args.seeds:
            layout = sample_layout(args.task, seed)
            started = time.monotonic()
            stable = env.reset_layout(seed, layout)
            print(f"[probe] seed {seed} reset ok={stable} {time.monotonic() - started:.1f}s", flush=True)
            obs = env.observe(images=False)
            print("[probe] state", np.round(obs["state"], 3))
            print("[probe] instruction", obs["instruction"])
            manager = env.scene_manager.layout_manager
            for record in manager.get_layout_records(0, "Rigid"):
                position, rotation = manager.get_instance_pose(0, label=record["label"])
                print("[probe]", record["label"], np.round(position, 3), np.round(rotation, 3))
            for images in [False] if args.no_cameras else [False, True]:
                started = time.monotonic()
                for _ in range(25):
                    if images:
                        env.observe(images=True)
                    env.take_joint_action(obs["state"].copy())
                print(f"[probe] 25 steps images={images}: {time.monotonic() - started:.2f}s")
            env.run_reward()
            print("[probe] reward", env.reward_manager.get_reward(final_check=False), "steps", env.take_action_cnt)


if __name__ == "__main__":
    main()
