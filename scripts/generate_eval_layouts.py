#!/usr/bin/env python3
"""Generate additional official-format JSON layouts and matching Isaac PNGs."""

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

from src.eval_layouts.storage import (
    DEFAULT_ROOT,
    atomic_json,
    existing_records,
    manifest,
    official_payload,
    paths,
    sha256,
    task_names,
)


def worker(args):
    from src.eval_layouts.lease import TaskLock

    task = args.tasks[0]
    scratch = ROOT / "data/eval_layout_generation" / str(args.set_seed) / task
    scratch.mkdir(parents=True, exist_ok=True)
    with TaskLock(scratch) as lock:
        records = existing_records(args.output, task, args.set_seed)
        if len(records) >= args.count:
            return 0
        journal = scratch / "attempts.jsonl"
        from src.eval_layouts.watchdog import start

        start(journal, scratch.parent / "logs" / f"{task}.log")
        last_seed = max([args.seed_start - 1, *(r["sampling_seed"] for r in records)])
        if journal.exists():
            last_seed = max(
                last_seed, *(json.loads(line)["sampling_seed"] for line in journal.read_text().splitlines())
            )
        from src.eval_layouts.sampler import TaskLayoutSampler
        from src.expert.layout import LayoutError
        from src.expert.sim import launch_app

        # Match the official entrypoint: cache RoboDojo's I/O module before
        # Kit adds extensions with their own top-level `utils` packages.
        from utils.load_file import load_object_metadata  # noqa: F401

        sampler = TaskLayoutSampler(task)
        shadow = scratch / "candidate_root"
        directory = shadow / "RoboDojo/arx_x5" / str(args.set_seed)
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob(task + "_*.json"):
            old.unlink()
        for r in records:
            scene, _, _ = paths(args.output, task, r["index"], args.set_seed)
            (directory / scene.name).symlink_to(scene)
        from src.eval_layouts.devices import choose_device

        args.device = choose_device(args.device)
        app = launch_app(headless=True, cameras=True, device_id=args.device)
        import torch

        torch.cuda.set_device(args.device)
        torch.set_num_threads(int(os.environ.get("EXPERT_CPU_THREADS", "4")))
        from PIL import Image

        from src.datagen.planner import PlanningFailure
        from src.eval_layouts.validation import initialize_and_render, make_env
        from utils.cluttered_generator import UnStableError

        count = len(records)
        cold_worlds = 0
        scene_resets = 0
        env = None
        planner_cache = {}
        articulation_signature = None
        for sampling_seed in range(last_seed + 1, last_seed + 1 + args.max_attempts):
            started = time.monotonic()
            with journal.open("a") as handle:
                handle.write(json.dumps({"sampling_seed": sampling_seed, "phase": "started"}) + "\n")
            candidate = directory / f"{task}_{count}.json"
            layout, signature = {}, []
            try:
                layout = official_payload(sampler.sample(sampling_seed))
                atomic_json(candidate, layout)
                signature = [
                    (category, item["category_idx"], item.get("label"))
                    for category, items in layout.get("Articulation", {}).items()
                    for item in items
                ]
                # The native CPU wrapper respawns rigid objects on reset, while
                # articulation models are created before the first PhysX view.
                if env is not None and (signature or signature != articulation_signature):
                    env.close()
                    env = None
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                if env is None:
                    # Kit updates can change the current CUDA device when the
                    # native scene uses CPU views. Select it before planners.
                    torch.cuda.set_device(args.device)
                    env = make_env(
                        task,
                        app,
                        shadow,
                        args.set_seed,
                        args.warmup_spp,
                        args.render_profile,
                        planner_cache=planner_cache,
                        skip_policy_planners=True,
                    )
                    env.layout_device_id = args.device
                    env.layout_activity_path = journal
                    if task == "imitate_sorting_sequence":
                        from src.eval_layouts.support_trajectory import prepare_planner

                        prepare_planner(env)
                    cold_worlds += 1
                    articulation_signature = signature
                else:
                    env.seed_manager.init_eval()
                scene_resets += 1
                frame, validation = initialize_and_render(env, count)
                trajectory = None
                if task == "imitate_sorting_sequence":
                    from src.eval_layouts.support_trajectory import generate_and_verify

                    trajectory = shadow / "Traj/RoboDojo" / task / str(args.set_seed) / f"{count}.pkl"
                    validation["support_trajectory"] = generate_and_verify(env, sampling_seed, trajectory, count)
                else:
                    env.run_reward()
                validation["reward_initialization"] = "passed"
                scene, image, meta = paths(args.output, task, count, args.set_seed)
                scene.parent.mkdir(parents=True, exist_ok=True)
                image_tmp = scratch / "image.tmp.png"
                Image.fromarray(frame.astype("uint8")).save(image_tmp)
                lock.ensure_owned()
                shutil.copyfile(candidate, scene.with_suffix(".json.tmp"))
                os.replace(scene.with_suffix(".json.tmp"), scene)
                os.replace(image_tmp, image)
                record = {
                    "task": task,
                    "index": count,
                    "set_seed": args.set_seed,
                    "sampling_seed": sampling_seed,
                    "environment_seed": count,
                    "json_sha256": sha256(scene),
                    "image_sha256": sha256(image),
                    "sampling": sampler.sampling_info,
                    "validation": validation,
                    "validated_at": datetime.now(UTC).isoformat(),
                }
                if trajectory is not None:
                    final_trajectory = args.output / trajectory.relative_to(shadow)
                    final_trajectory.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(trajectory, final_trajectory.with_suffix(".pkl.tmp"))
                    os.replace(final_trajectory.with_suffix(".pkl.tmp"), final_trajectory)
                    record["trajectory_path"] = str(final_trajectory.relative_to(args.output))
                    record["trajectory_sha256"] = sha256(final_trajectory)
                atomic_json(meta, record)
                candidate.unlink()
                candidate.symlink_to(scene)
                count += 1
                attempt = {"sampling_seed": sampling_seed, "accepted": True, "count": count}
            except (LayoutError, UnStableError, ValueError, PlanningFailure) as exc:
                attempt = {
                    "sampling_seed": sampling_seed,
                    "accepted": False,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
                candidate.unlink(missing_ok=True)
            attempt["seconds"] = round(time.monotonic() - started, 3)
            with journal.open("a") as handle:
                lock.ensure_owned()
                handle.write(json.dumps(attempt, allow_nan=False) + "\n")
            print("EVAL_LAYOUT", task, json.dumps(attempt), flush=True)
            if count >= args.count:
                if env is not None:
                    env.close()
                return 0
            reset_limit = 3 if layout.get("Fluid") or layout.get("Garment") else 15
            if (cold_worlds >= 3 and signature) or scene_resets >= reset_limit:
                # A process exit releases Kit/CUDA graph resources retained
                # across cold worlds. Published files are already durable.
                print("EVAL_WORKER_RECYCLE", task, "cold_worlds", cold_worlds, "scene_resets", scene_resets, flush=True)
                return 75
        if env is not None:
            env.close()
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--set-seed", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup-spp", type=int, default=16)
    parser.add_argument("--render-profile", choices=["fast", "official"], default="fast")
    parser.add_argument("--ssh-host")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.tasks = args.tasks or task_names()
    if (
        args.count < 1
        or args.max_attempts < 1
        or args.workers < 1
        or args.warmup_spp < 1
        or set(args.tasks) - set(task_names())
    ):
        parser.error("Invalid count, workers, attempts, or task names")
    if args.worker:
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1)
        faulthandler.dump_traceback_later(900, repeat=True)
        print("EVAL_WORKER_STARTED", os.getpid(), args.tasks[0], args.render_profile, flush=True)
        try:
            code = worker(args)
        except BaseException:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    if args.manifest_only:
        print(json.dumps(manifest(args.output, set_seed=args.set_seed, count=args.count), indent=2))
        return 0
    logs = ROOT / "data/eval_layout_generation" / str(args.set_seed) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def run(task, device):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--tasks",
            task,
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
            str(device),
        ]
        command += ["--warmup-spp", str(args.warmup_spp)]
        command += ["--render-profile", args.render_profile]
        if task == "imitate_sorting_sequence":
            command = [
                sys.executable,
                str(ROOT / "scripts/generate_sorting_eval_layouts.py"),
                "--output",
                str(args.output),
                "--set-seed",
                str(args.set_seed),
                "--seed-start",
                str(args.seed_start),
                "--count",
                str(args.count),
                "--workers",
                str(min(4, args.workers)),
                "--devices",
                *map(str, args.devices),
                "--render-profile",
                args.render_profile,
                "--ssh-host",
                args.ssh_host or "",
            ]
        elif args.ssh_host:
            remote = (
                "cd "
                + shlex.quote(str(ROOT))
                + " && "
                + shlex.join(["env", "PYTHONUNBUFFERED=1", "EXPERT_CPU_THREADS=4", "EXPERT_PHYSX_THREADS=4", *command])
            )
            command = ["ssh", "-o", "BatchMode=yes", args.ssh_host, remote]
        retry = 0
        while retry < 8:
            with (logs / f"{task}.log").open("a") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if result.returncode == 75:
                continue
            retry += 1
            time.sleep(1 + retry)
        return task, result.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task, args.devices[i % len(args.devices)]) for i, task in enumerate(args.tasks)]
        for future in as_completed(futures):
            task, code = future.result()
            if code:
                failures.append(task)
            print("TASK_FINISHED", task, "exit", code, flush=True)
    summary = manifest(args.output, set_seed=args.set_seed, count=args.count)
    print(json.dumps({"failed_tasks": failures, "layout_count": summary["layout_count"]}), flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
