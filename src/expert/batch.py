"""Collect to a shared success target with persistent, GPU-bound workers."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from src.expert.arguments import add_collection_arguments, help_parser, nonnegative_int, positive_int
from src.expert.registry import load_planner, planner_spec
from src.expert.scheduling import WorkQueue, batch_lock, serve_queue
from src.expert.storage import episode_count, prepare_output, read_attempts


def worker_commands(args, have):
    """Launch bounded GPU slots; dynamic workers receive seeds from the parent."""
    remaining = max(0, args.num - have)
    quotient, remainder = divmod(remaining, args.workers)
    dynamic = args.scheduling == "dynamic"
    commands = []
    for worker in range(args.workers):
        quota = remaining if dynamic and worker < remaining else quotient + int(worker < remainder)
        if quota == 0:
            continue
        command = [
            sys.executable,
            "-m",
            "src.expert",
            "collect",
            "--task",
            args.task,
            "--out",
            str(args.out),
            "--num",
            str(quota),
            "--additional",
            "--seed-start",
            str(args.seed_start if dynamic else args.seed_start + worker),
            "--seed-stride",
            str(args.workers),
            "--max-attempts",
            str(args.attempts_per_episode * quota),
            "--worker",
            str(worker),
            "--gpu",
            str(args.gpu_ids[worker % len(args.gpu_ids)] if args.gpu_ids else args.gpu),
            "--table-randomization",
            str(args.table_randomization),
        ]
        if args.planner:
            command += ["--planner", args.planner]
        if args.no_images:
            command.append("--no-images")
        if getattr(args, "no_checkpoints", False):
            command.append("--no-checkpoints")
        if getattr(args, "background_export", False):
            command += ["--background-export", "--max-pending-episodes", str(args.max_pending_episodes)]
        commands.append((worker, command))
    return commands


def summarize(out):
    rows = read_attempts(out)
    tried = [row for row in rows if row["result"] in ("success", "fail", "error")]
    successful = [row for row in tried if row["result"] == "success"]
    reasons = {}
    for row in tried:
        if row["result"] != "success":
            key = (row.get("reason") or "").split(":")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
    steps = sorted(row["steps"] for row in successful)
    return {
        "attempts": len(tried),
        "success": len(successful),
        "success_rate": round(len(successful) / len(tried), 3) if tried else None,
        "unstable_layouts": sum(row["result"] == "unstable" for row in rows),
        "layout_errors": sum(row["result"] == "layout_error" for row in rows),
        "median_steps": steps[len(steps) // 2] if steps else None,
        "failure_reasons": reasons,
        "episodes_on_disk": episode_count(out),
    }


def run_workers(args, commands, queue=None):
    processes = []
    with ExitStack() as stack:
        try:
            for index, (worker, command) in enumerate(commands):
                if index:
                    time.sleep(args.startup_delay)
                log = stack.enter_context((args.out / "logs" / f"worker{worker}.log").open("a"))
                processes.append(
                    (
                        worker,
                        subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            # Keep the output locked until children exit even
                            # if the coordinator is killed without cleanup.
                            pass_fds=tuple(
                                fd
                                for fd in (
                                    getattr(args, "batch_lock_fd", None),
                                    getattr(args, "pipeline_lock_fd", None),
                                )
                                if fd is not None
                            ),
                        ),
                    )
                )
            codes = {}
            stopping = {}
            while len(codes) < len(processes):
                for worker, process in processes:
                    if worker not in codes and (code := process.poll()) is not None:
                        codes[worker] = code
                        if queue is not None:
                            queue.worker_exited(worker, code)
                    elif worker not in codes and queue is not None:
                        if queue.stop_unused(worker):
                            stopping[worker] = time.monotonic()
                            process.terminate()
                        elif worker in stopping and time.monotonic() - stopping[worker] > 10:
                            process.kill()
                if len(codes) < len(processes):
                    time.sleep(0.1)
            return [codes[worker] for worker, _ in processes]
        finally:
            if queue is not None:
                queue.cancel()
            # Interrupt/startup failure must not leave orphan Isaac workers.
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def main(argv=None):
    parser = help_parser(
        "batch",
        "Collect to a shared success target using multiple persistent Isaac worker processes.",
        epilog="Examples:\n"
        "  %(prog)s --task stack_blocks --num 100 --out data/expert/raw/stack_blocks\n"
        "  %(prog)s --task stack_blocks --num 100 --format lerobot-v3 --out data/expert/lerobot/stack_blocks\n"
        "  %(prog)s --task stack_blocks --num 100 --gpu-ids 0 1 --workers 4 --out data/expert/raw/multi_gpu\n"
        "\n--workers is the total process count, not the count per GPU. GPU IDs are\n"
        "space-separated; workers are assigned round-robin. Without --gpu-ids, all\n"
        "workers use --gpu. Fewer workers may start when fewer successes remain.\n"
        "\nResume by rerunning with the same --out and recording settings. --num is the\n"
        "final total, not the number to add. Dynamic scheduling shares the remaining\n"
        "target; static scheduling gives each worker a fixed quota.\n"
        "Raw output: episode_*.h5, attempts_w*.jsonl, logs/worker*.log; dynamic runs also\n"
        "write scheduling_*.jsonl. --format lerobot-v3 converts and validates each success\n"
        "in the background, then deletes its HDF5; RGB is required, checkpoints are disabled.\n"
        "Exit status: 0 if the target is reached without worker\n"
        "errors, otherwise 1. See docs/expert/README.md for multi-GPU setup.",
    )
    add_collection_arguments(parser)
    workers = parser.add_argument_group("Workers and scheduling")
    workers.add_argument(
        "--workers", type=positive_int, help="total worker processes (default: one per --gpu-ids entry, otherwise 3)"
    )
    workers.add_argument(
        "--gpu-ids",
        nargs="+",
        type=nonnegative_int,
        metavar="ID",
        help="physical GPU IDs, e.g. 0 1 2 3; assigned round-robin (default: all workers on --gpu)",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR", help="dataset directory to create or resume (--format)"
    )
    workers.add_argument(
        "--attempts-per-episode",
        type=positive_int,
        default=4,
        help="attempt budget multiplier: remaining successes x this value; layout failures also consume attempts",
    )
    workers.add_argument(
        "--scheduling",
        choices=("dynamic", "static"),
        default="dynamic",
        help="dynamic: idle workers request another seed; static: fixed quotas and interleaved seeds",
    )
    workers.add_argument(
        "--startup-delay",
        type=float,
        default=20,
        metavar="SECONDS",
        help="delay between Isaac worker starts, from 0 to 60 seconds",
    )
    parser.add_argument("--pipeline-lock-fd", type=nonnegative_int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.workers is None:
        args.workers = len(args.gpu_ids) if args.gpu_ids else 3
    if not 0 <= args.startup_delay <= 60:
        parser.error("--startup-delay must be between 0 and 60")
    try:
        spec = planner_spec(args.task, args.planner)
        if args.format == "lerobot-v3":
            from src.expert.training import collect_training

            return collect_training("batch", args, sys.argv[1:] if argv is None else argv, spec)
        load_planner(spec)  # reject missing/broken plugins before starting any workers
        with batch_lock(args.out) as lock_fd:
            args.batch_lock_fd = lock_fd
            return run_batch(args, spec)
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        parser.error(str(exc))


def run_batch(args, spec):
    prepare_output(
        args.out,
        task=args.task,
        planner=spec.entrypoint,
        sampler=spec.sampler,
        images=not args.no_images,
        table_randomization=args.table_randomization,
        checkpoints=not getattr(args, "no_checkpoints", False),
    )
    read_attempts(args.out)  # fail early on corrupt resume logs
    (args.out / "logs").mkdir(exist_ok=True)
    have = episode_count(args.out)
    commands = worker_commands(args, have)
    worker_gpus = {worker: int(command[command.index("--gpu") + 1]) for worker, command in commands}
    started = time.monotonic()
    queue = None
    if args.scheduling == "dynamic" and commands:
        queue = WorkQueue(args, worker_gpus, spec.entrypoint)
        with serve_queue(queue) as address:
            for _, command in commands:
                command += ["--scheduler", address]
            codes = run_workers(args, commands, queue)
    else:
        codes = run_workers(args, commands)
    summary = summarize(args.out)
    summary.update(
        worker_exit_codes=codes,
        target_reached=summary["episodes_on_disk"] >= args.num,
        wall_seconds=round(time.monotonic() - started, 3),
        worker_gpus=worker_gpus,
        scheduling=args.scheduling,
        new_successes=summary["episodes_on_disk"] - have,
    )
    if queue is not None:
        summary.update(
            attempt_budget=queue.budget,
            dispatched=queue.assigned,
            completed=queue.completed,
            scheduling_log=str(queue.events),
            stopped_unused_workers=sorted(queue.stopped_unused),
        )
    print(json.dumps(summary), flush=True)
    unexpected = [
        code
        for (worker, _), code in zip(commands, codes)
        if code and (queue is None or worker not in queue.stopped_unused)
    ]
    return 0 if summary["target_reached"] and not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
