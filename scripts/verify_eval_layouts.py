#!/usr/bin/env python3
"""Recheck published JSONs missing native reward-initialization evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import faulthandler
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_layouts.storage import DEFAULT_ROOT, atomic_json, existing_records, paths, sha256, task_names


def worker(args):
    from src.eval_layouts.lease import TaskLock

    task = args.tasks[0]
    scratch = ROOT / "data/eval_layout_generation" / str(args.set_seed) / task
    scratch.mkdir(parents=True, exist_ok=True)
    with TaskLock(scratch) as lock:
        pending = [
            r
            for r in existing_records(args.output, task, args.set_seed)
            if r["validation"].get("reward_initialization") != "passed"
        ]
        if not pending:
            return 0
        if task == "imitate_sorting_sequence":
            raise ValueError("Sorting samples must include native teacher replay at generation")
        journal = scratch / "reward_rechecks.jsonl"
        from src.eval_layouts.watchdog import start

        start(journal, scratch.parent / "logs" / f"{task}.reward.log")
        from src.expert.sim import launch_app
        from utils.load_file import load_object_metadata  # noqa: F401  (preload before Kit)

        app = launch_app(headless=True, cameras=True, device_id=args.device)
        import torch

        torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        from src.eval_layouts.validation import initialize_and_render, make_env

        env, cache = None, {}
        cold_worlds = 0
        for ordinal, record in enumerate(pending):
            scene, _, meta = paths(args.output, task, record["index"], args.set_seed)
            layout = json.loads(scene.read_text())
            started = time.monotonic()
            with journal.open("a") as handle:
                handle.write(json.dumps({"index": record["index"], "phase": "started"}) + "\n")
            if env is not None and any(layout.get("Articulation", {}).values()):
                env.close()
                env = None
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            if env is None:
                torch.cuda.set_device(args.device)
                env = make_env(
                    task, app, args.output, args.set_seed, 16, "fast", planner_cache=cache, skip_policy_planners=True
                )
                cold_worlds += 1
            frame, check = initialize_and_render(env, record["index"])
            env.run_reward()
            if sha256(scene) != record["json_sha256"]:
                raise RuntimeError(f"JSON changed during verification: {scene}")
            lock.ensure_owned()
            record["validation"]["reward_initialization"] = "passed"
            record["validation"]["reward_recheck"] = {
                "validated_at": datetime.now(UTC).isoformat(),
                "initialization": check,
                "reward_entrypoint": "TaskEnv.run_reward",
                "status": "passed",
            }
            atomic_json(meta, record)
            result = {"index": record["index"], "phase": "passed", "seconds": round(time.monotonic() - started, 3)}
            with journal.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print("REWARD_RECHECK", task, json.dumps(result), flush=True)
            if ordinal + 1 < len(pending) and (ordinal >= 14 or cold_worlds >= 3):
                return 75
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--set-seed", type=int, default=1000)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ssh-host")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.tasks = args.tasks or task_names()
    if args.worker:
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1)
        try:
            code = worker(args)
        except BaseException:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
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
            "--device",
            str(device),
        ]
        if args.ssh_host:
            remote = (
                "cd "
                + shlex.quote(str(ROOT))
                + " && "
                + shlex.join(["env", "PYTHONUNBUFFERED=1", "EXPERT_CPU_THREADS=4", "EXPERT_PHYSX_THREADS=4", *command])
            )
            command = ["ssh", "-o", "BatchMode=yes", args.ssh_host, remote]
        failures = 0
        while failures < 4:
            with (logs / f"{task}.reward.log").open("a") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                return task, 0
            if result.returncode != 75:
                failures += 1
            time.sleep(1)
        return task, result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, t, args.devices[i % len(args.devices)]) for i, t in enumerate(args.tasks)]
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print("RECHECK_FINISHED", *result, flush=True)
    return int(any(code for _, code in results))


if __name__ == "__main__":
    raise SystemExit(main())
