#!/usr/bin/env python3
"""Generate sorting layouts and native replay-verified teacher trajectories in parallel."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import faulthandler
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_layouts.lease import TaskLock
from src.eval_layouts.storage import (
    DEFAULT_ROOT,
    atomic_json,
    existing_records,
    official_payload,
    paths,
    sha256,
    task_names,
)

TASK = "imitate_sorting_sequence"


def owner_matches(path, token):
    if json.loads((path / "owner.json").read_text())["token"] != token:
        raise RuntimeError("Sorting coordinator lease has been lost")


def worker(args):
    task = args.task
    scratch = ROOT / "data/eval_layout_generation" / str(args.set_seed) / task / "parallel" / str(args.index)
    scratch.mkdir(parents=True, exist_ok=True)
    with TaskLock(scratch) as lock:
        records = existing_records(args.output, task, args.set_seed, require_contiguous=False)
        previous = next((r for r in records if r["index"] == args.index), None)
        if previous is not None and not args.replace:
            return 0
        owner_matches(args.coordinator, args.token)
        journal = scratch / "attempts.jsonl"
        from src.eval_layouts.watchdog import start

        start(journal, scratch.parents[2] / "logs" / f"{task}.{args.index}.log")
        last = args.seed_start + args.index - args.count
        already_started = 0
        if journal.exists():
            attempts = [json.loads(line) for line in journal.read_text().splitlines()]
            last = max(last, *(a["sampling_seed"] for a in attempts))
            already_started = sum(a.get("phase") == "started" for a in attempts)
        remaining = args.max_attempts - already_started
        if remaining <= 0:
            return 2
        from src.eval_layouts.devices import choose_device
        from src.eval_layouts.sampler import TaskLayoutSampler
        from src.expert.layout import LayoutError
        from src.expert.sim import launch_app
        from utils.load_file import load_object_metadata  # noqa: F401  (preload before Kit)

        args.device = choose_device(args.device)
        app = launch_app(headless=True, cameras=True, device_id=args.device)
        import torch

        torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        from PIL import Image

        from src.datagen.planner import PlanningFailure
        from src.eval_layouts.support_trajectory import generate_and_verify, prepare_planner
        from src.eval_layouts.validation import initialize_and_render, make_env
        from utils.cluttered_generator import UnStableError

        sampler = TaskLayoutSampler(task)
        shadow = scratch / "candidate_root"
        directory = shadow / "RoboDojo/arx_x5" / str(args.set_seed)
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / "candidate.json"
        # SeedManager enumerates files rather than using their suffix as the
        # episode ID. Private aliases preserve the final episode ID even when
        # other parallel slots have not been published yet.
        for index in range(args.count):
            alias = directory / f"{task}_{index}.json"
            if not alias.is_symlink():
                alias.unlink(missing_ok=True)
                alias.symlink_to(candidate)
        env = None
        for attempt_number in range(min(10, remaining)):
            source_seed = last + args.count * (attempt_number + 1)
            started = time.monotonic()
            with journal.open("a") as f:
                f.write(json.dumps({"index": args.index, "sampling_seed": source_seed, "phase": "started"}) + "\n")
            attempt = {"index": args.index, "sampling_seed": source_seed}
            layout = {}
            try:
                layout = official_payload(sampler.sample(source_seed))
                atomic_json(candidate, layout)
                if env is not None and layout.get("Articulation"):
                    env.close()
                    env = None
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                if env is None:
                    torch.cuda.set_device(args.device)
                    env = make_env(task, app, shadow, args.set_seed, 16, args.render_profile, skip_policy_planners=True)
                    env.layout_device_id = args.device
                    env.layout_activity_path = journal
                    if task == TASK:
                        prepare_planner(env)
                else:
                    env.seed_manager.init_eval()
                frame, validation = initialize_and_render(env, args.index)
                trajectory = None
                if task == TASK:
                    trajectory = shadow / "Traj/RoboDojo" / task / str(args.set_seed) / f"{args.index}.pkl"
                    validation["support_trajectory"] = generate_and_verify(env, source_seed, trajectory, args.index)
                else:
                    env.run_reward()
                validation["reward_initialization"] = "passed"
                scene, image, meta = paths(args.output, task, args.index, args.set_seed)
                scene.parent.mkdir(parents=True, exist_ok=True)
                image_tmp = scratch / "image.tmp.png"
                Image.fromarray(frame.astype("uint8")).save(image_tmp)
                lock.ensure_owned()
                owner_matches(args.coordinator, args.token)
                if previous is not None:
                    backup = scratch / "replaced" / previous["json_sha256"]
                    backup.mkdir(parents=True, exist_ok=True)
                    for old in (scene, image, meta):
                        shutil.copyfile(old, backup / old.name)
                    if previous.get("trajectory_path"):
                        old = args.output / previous["trajectory_path"]
                        shutil.copyfile(old, backup / old.name)
                shutil.copyfile(candidate, scene.with_suffix(".json.tmp"))
                os.replace(scene.with_suffix(".json.tmp"), scene)
                os.replace(image_tmp, image)
                record = {
                    "task": task,
                    "index": args.index,
                    "set_seed": args.set_seed,
                    "sampling_seed": source_seed,
                    "environment_seed": args.index,
                    "json_sha256": sha256(scene),
                    "image_sha256": sha256(image),
                    "sampling": sampler.sampling_info,
                    "validation": validation,
                    "validated_at": datetime.now(UTC).isoformat(),
                }
                if previous is not None:
                    record["replacement"] = {
                        "sampling_seed": previous["sampling_seed"],
                        "json_sha256": previous["json_sha256"],
                        "reason": args.replace_reason,
                    }
                if trajectory is not None:
                    destination = args.output / trajectory.relative_to(shadow)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(trajectory, destination.with_suffix(".pkl.tmp"))
                    os.replace(destination.with_suffix(".pkl.tmp"), destination)
                    record.update(
                        trajectory_path=str(destination.relative_to(args.output)), trajectory_sha256=sha256(destination)
                    )
                atomic_json(meta, record)
                attempt["accepted"] = True
            except (LayoutError, UnStableError, ValueError, PlanningFailure) as exc:
                attempt.update(accepted=False, reason=type(exc).__name__, detail=str(exc))
                if isinstance(exc, PlanningFailure) and already_started + attempt_number < 3 and env is not None:
                    names = env.camera_manager.camera_names[0]
                    camera = next(i for i, name in enumerate(names) if name in ("cam_head", "cam_high"))
                    env.obs_manager.render_for_capture()
                    pixels = env.capture_manager.step(env_ids=[0], cam_ids=[camera])[0]["rgb"][0]["data"]
                    Image.fromarray(pixels[..., :3].astype("uint8")).save(scratch / f"failure_{source_seed}.png")
            attempt["seconds"] = round(time.monotonic() - started, 3)
            with journal.open("a") as f:
                lock.ensure_owned()
                f.write(json.dumps(attempt, allow_nan=False) + "\n")
            print("SORTING_LAYOUT", json.dumps(attempt), flush=True)
            if attempt["accepted"]:
                return 0
            if layout.get("Articulation") and attempt_number >= 2:
                return 75
        return 75 if remaining > 10 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", choices=task_names(), default=TASK)
    parser.add_argument("--set-seed", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--ssh-host", help="Run on a server sharing the same checkout path; defaults to local execution"
    )
    parser.add_argument("--render-profile", choices=["fast", "official"], default="fast")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--replace-indices", type=int, nargs="+")
    parser.add_argument(
        "--replace-reason", default="Regenerated an early sample after a task-level native recheck failure"
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--coordinator", type=Path)
    parser.add_argument("--token")
    args = parser.parse_args()
    if args.replace_indices and any(i < 0 or i >= args.count for i in args.replace_indices):
        parser.error("Replacement indices must be within the collection")
    args.output = args.output.resolve()
    if args.worker:
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1)
        faulthandler.dump_traceback_later(900, repeat=True)
        print("SORTING_WORKER_STARTED", os.getpid(), args.index, args.device, flush=True)
        try:
            code = worker(args)
        except BaseException:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    scratch = ROOT / "data/eval_layout_generation" / str(args.set_seed) / args.task
    scratch.mkdir(parents=True, exist_ok=True)
    logs = scratch.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with TaskLock(scratch) as coordinator:
        records = existing_records(args.output, args.task, args.set_seed, require_contiguous=False)
        if not args.replace_indices and [r["index"] for r in records] == list(range(args.count)):
            return 0

        def run(index):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--index",
                str(index),
                "--task",
                args.task,
                "--output",
                str(args.output),
                "--set-seed",
                str(args.set_seed),
                "--seed-start",
                str(args.seed_start),
                "--count",
                str(args.count),
                "--max-attempts",
                str(args.max_attempts),
                "--device",
                str(args.devices[index % len(args.devices)]),
                "--coordinator",
                str(coordinator.path),
                "--token",
                coordinator.token,
                "--render-profile",
                args.render_profile,
            ]
            if args.replace_indices and index in args.replace_indices:
                command.extend(["--replace", "--replace-reason", args.replace_reason])
            if args.ssh_host:
                remote = (
                    "cd "
                    + shlex.quote(str(ROOT))
                    + " && "
                    + shlex.join(
                        ["env", "PYTHONUNBUFFERED=1", "EXPERT_CPU_THREADS=4", "EXPERT_PHYSX_THREADS=4", *command]
                    )
                )
                command = ["ssh", "-o", "BatchMode=yes", args.ssh_host, remote]
            retries = 0
            while retries < 8:
                with (logs / f"{args.task}.{index}.log").open("a") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    return index, 0
                if result.returncode != 75:
                    retries += 1
                time.sleep(1)
            return index, result.returncode

        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, i) for i in (args.replace_indices or range(args.count))]
            for future in as_completed(futures):
                index, code = future.result()
                print("SORTING_FINISHED", index, code, flush=True)
                if code:
                    failures.append(index)
        return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
