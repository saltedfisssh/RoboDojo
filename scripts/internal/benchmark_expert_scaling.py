"""Measure fixed-workload expert collection scaling on a multi-GPU host.

Run from the RoboDojo root with its simulator Python. Every scale attempts the
same seed set once, with RGB and checkpoints. Failures remain in the denominator;
no cherry-picking or retries change the workload between GPU counts.
"""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def command_output(command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def cpu_usage():
    rows = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    return {key: int(value) for key, value in rows.items()}


def sample_resources():
    return {
        "unix_time": time.time(),
        "gpus": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "processes": command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cpu": cpu_usage(),
        "memory_bytes": int(Path("/sys/fs/cgroup/memory.current").read_text()),
    }


def scale_directory(args, count):
    if args.gpu_ids:
        return args.out / f"workers_{count}_gpus_{'_'.join(map(str, args.gpu_ids))}"
    return args.out / f"gpus_{count}"


def run_scale(args, count):
    gpu_ids = args.gpu_ids or list(range(count))
    directory = scale_directory(args, count)
    directory.mkdir()  # never mix a fresh timing run with resumed files
    command = [
        sys.executable,
        "-m",
        "src.expert",
        "batch",
        "--scheduling",
        args.scheduling,
        "--task",
        args.task,
        "--num",
        str(args.episodes),
        "--workers",
        str(count),
        "--gpu-ids",
        *[str(index) for index in gpu_ids],
        "--seed-start",
        str(args.seed_start),
        "--attempts-per-episode",
        "1",
        "--startup-delay",
        "0",
        "--out",
        str(directory / "raw"),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    before = cpu_usage()
    start = time.monotonic()
    with (directory / "batch.log").open("w") as log, (directory / "resources.jsonl").open("w") as resources:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                resources.write(json.dumps(sample_resources()) + "\n")
                resources.flush()
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError(f"{count}-worker collection exceeded {args.timeout}s")
                time.sleep(5)
            code = process.wait()
        finally:
            if process.poll() is None:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    wall = time.monotonic() - start
    rows = []
    for path in sorted((directory / "raw").glob("attempts_w*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    successful = [row for row in rows if row["result"] == "success"]
    try:
        batch_summary = json.loads((directory / "batch.log").read_text().splitlines()[-1])
    except (ValueError, IndexError):
        batch_summary = {}
    tracebacks = [
        path.name
        for path in sorted((directory / "raw/logs").glob("worker*.log"))
        if "Traceback (most recent call last):" in path.read_text(errors="replace")
    ]
    after = cpu_usage()
    summary = {
        "gpus": len(set(gpu_ids[worker % len(gpu_ids)] for worker in range(count))),
        "workers": count,
        "gpu_ids": gpu_ids,
        "scheduling": args.scheduling,
        "task": args.task,
        "exit_code": code,
        "started_utc": datetime.fromtimestamp(time.time() - wall, UTC).isoformat(),
        "wall_seconds": wall,
        "attempts": len(rows),
        "successes": len(successful),
        "seeds": sorted(row["seed"] for row in rows),
        "saved_frames": sum(row.get("steps", 0) for row in successful),
        "total_steps": sum(row.get("steps", 0) for row in rows),
        "episodes_per_hour": 3600 * len(successful) / wall,
        "saved_frames_per_second": sum(row.get("steps", 0) for row in successful) / wall,
        "cpu_cores_mean": (after["usage_usec"] - before["usage_usec"]) / 1e6 / wall,
        "throttled_seconds": (after.get("throttled_usec", 0) - before.get("throttled_usec", 0)) / 1e6,
        "attempt_seconds": [row["seconds"] for row in rows],
        "worker_exit_codes": batch_summary.get("worker_exit_codes", []),
        "stopped_unused_workers": batch_summary.get("stopped_unused_workers", []),
        "worker_tracebacks": tracebacks,
        "raw_bytes": sum(path.stat().st_size for path in (directory / "raw").glob("episode_*.h5")),
    }
    expected = list(range(args.seed_start, args.seed_start + args.episodes))
    summary["fixed_workload_complete"] = summary["seeds"] == expected
    summary["benchmark_valid"] = (
        summary["fixed_workload_complete"]
        and not tracebacks
        and not any(row["result"] in ("error", "layout_error") for row in rows)
        and len(summary["worker_exit_codes"]) == count
        and all(
            code in (0, 1) or worker in summary["stopped_unused_workers"]
            for worker, code in enumerate(summary["worker_exit_codes"])
        )
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--gpu-ids", type=int, nargs="+", help="optional fixed GPU pool; e.g. --counts 8 --gpu-ids 0 for a control"
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument(
        "--scheduling",
        choices=("static", "dynamic"),
        default="static",
        help="static preserves the original scaling baseline; use dynamic for a shared queue",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per scale")
    args = parser.parse_args()
    if args.episodes < 1 or any(count < 1 or count > args.episodes for count in args.counts):
        parser.error("each worker count must be between 1 and the episode count")
    if args.gpu_ids and any(index < 0 for index in args.gpu_ids):
        parser.error("GPU ids must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(scale_directory(args, count).exists() for count in args.counts):
        parser.error("timing directories already exist; choose a fresh output or different counts")
    summaries = []
    for count in args.counts:
        summaries.append(run_scale(args, count))
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        if not summaries[-1]["benchmark_valid"]:
            raise SystemExit("Incomplete workload or runtime failure; inspect worker logs before continuing")


if __name__ == "__main__":
    main()
