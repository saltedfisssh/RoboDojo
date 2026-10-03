"""Bounded, resumable collection with a fresh Isaac process for each seed."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import h5py

from src.datagen.layouts import ROOT


class PlannerStallMonitor:
    """Recognize repeated native graph-replay stacks without trajectory progress."""

    def __init__(self):
        self.pending = b""
        self.replays = 0
        self.first_replay = None

    def feed(self, output, now):
        lines = (self.pending + output).split(b"\n")
        self.pending = lines.pop()[-4096:]
        for line in lines:
            if any(marker in line for marker in (b"[planner]", b"[timing] frames=", b"[render] frame=")):
                self.replays, self.first_replay = 0, None
            if b'torch/cuda/graphs.py"' in line and b" in replay" in line:
                self.replays += 1
                if self.first_replay is None:
                    self.first_replay = now
        # Five 60-second faulthandler dumps span at least four minutes. Requiring
        # elapsed time also avoids treating repeated historic log text as a stall.
        return self.replays >= 5 and now - self.first_replay >= 240


def supervise(process, log_path, timeout):
    """Limit active execution, excluding time waiting for the shared GPU lease."""
    deadline = time.monotonic() + timeout
    reason = None
    offset = 0
    tail = b""
    waiting = False
    acquired = False
    cleanup_deadline = None
    cleanup_reason = "Failed attempt stalled during Isaac shutdown"
    planner_stall = PlannerStallMonitor()
    while process.poll() is None:
        with log_path.open("rb") as stream:
            stream.seek(offset)
            new_output = stream.read()
            offset = stream.tell()
        tail = tail + new_output
        if not acquired and b"[datagen] Waiting for the Isaac GPU lease." in tail:
            waiting = True
        if not acquired and b"[datagen] Acquired the Isaac GPU lease." in tail:
            acquired = True
            waiting = False
            deadline = time.monotonic() + timeout
        if planner_stall.feed(new_output, time.monotonic()) and acquired:
            reason = "cuRobo CUDA graph replay stalled across five stack dumps without trajectory progress"
            break
        if cleanup_deadline is None and b"[datagen] Attempt failed; artifacts finalized; closing Isaac." in tail:
            cleanup_deadline = time.monotonic() + 60
        if cleanup_deadline is None and (
            b"PhysX Internal CUDA error. Simulation cannot continue!" in tail
            or b"CUDA error: an illegal memory access was encountered" in tail
        ):
            # The broken CUDA context may stall checkpoint/Kit teardown. Give
            # the worker time to finalize its failure artifacts, then release
            # only this worker's process group and the exclusive GPU lease.
            cleanup_deadline = time.monotonic() + 60
            cleanup_reason = "Isaac CUDA failure did not exit within 60 seconds"
        if cleanup_deadline is not None and time.monotonic() >= cleanup_deadline:
            reason = cleanup_reason
            break
        if b"GPU crash is detected" in tail or b"VkResult: ERROR_DEVICE_LOST" in tail:
            reason = "Isaac Vulkan device lost"
            break
        tail = tail[-65536:]
        if not waiting and time.monotonic() >= deadline:
            reason = "Attempt active execution timeout"
            break
        try:
            return process.wait(timeout=2), None
        except subprocess.TimeoutExpired:
            continue
    if reason is None:
        return process.returncode, None
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return (86 if "Vulkan" in reason else 124), reason


def episode_status(path):
    try:
        with h5py.File(path, "r") as file:
            return bool(
                file.attrs.get("complete", False)
                and file.attrs.get("success", False)
                and file.attrs.get("num_frames", 0) > 0
                and "images" in file
                and set(file["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
                and all(len(ds) == file.attrs["num_frames"] for ds in file["images"].values())
                and "terminal" in file
            )
    except OSError:
        return False


def accepted_by_seed(paths):
    """Re-rendering a physical plan does not create another independent episode."""
    accepted = {}
    for path in sorted(paths):
        if path.name.endswith(".partial.h5") or not episode_status(path):
            continue
        with h5py.File(path) as file:
            seed = int(json.loads(file.attrs["metadata"])["seed"])
        accepted.setdefault(seed, path)
    return accepted


def main():
    from src.datagen.planner import SUPPORTED_TASKS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=["stack_blocks"])
    p.add_argument(
        "--episodes", type=int, default=100, help="Desired total successful episodes per task, including existing"
    )
    p.add_argument("--max-attempts", type=int, default=500, help="Seed budget per task, including previous attempts")
    p.add_argument("--seed-start", type=int, default=10000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, help="Skip seeds already successful in other raw collection shards")
    p.add_argument(
        "--timeout", type=float, default=7200, help="Active seconds per attempt, excluding the GPU lease queue"
    )
    p.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    args = p.parse_args()
    if args.episodes < 1 or args.max_attempts < 1 or args.timeout <= 0:
        p.error("episodes, max-attempts and timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Kernel releases this lock even when a supervisor is interrupted/killed.
    lock = (args.output / ".collection.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("Another collector holds this output directory")
    failed_tasks = []
    for task in dict.fromkeys(args.tasks):
        folder = args.output / task
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / "collection.json"
        config = {"task": task, "seed_start": args.seed_start, "sampler": "asset_metadata_v1", "fps": 25}
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            p.error(f"Collection configuration changed for {folder}; choose a fresh directory")
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        own = accepted_by_seed(folder.glob("episode_*.h5"))
        external = (
            accepted_by_seed(
                path
                for path in args.reference_root.glob(f"**/{task}/episode_*.h5")
                if folder not in path.resolve().parents
            )
            if args.reference_root is not None
            else {}
        )
        success = len(own.keys() - external.keys())
        target_total = args.episodes + len(external)
        for seed in range(args.seed_start, args.seed_start + args.max_attempts):
            if args.reference_root is not None:
                external = accepted_by_seed(
                    path
                    for path in args.reference_root.glob(f"**/{task}/episode_*.h5")
                    if folder not in path.resolve().parents
                )
            own = accepted_by_seed(folder.glob("episode_*.h5"))
            success = len(own.keys() - external.keys())
            if len(own.keys() | external.keys()) >= target_total:
                break
            if seed in external:
                continue
            episode = folder / f"episode_{seed:010d}.h5"
            result = folder / f"attempt_{seed:010d}.json"
            if episode.exists() or result.exists():
                continue
            partial = episode.with_suffix(".partial.h5")
            if partial.exists():
                # Preserve interrupted data; never treat it as a complete trajectory.
                partial.rename(partial.with_name(partial.name + f".interrupted-{time.time_ns()}"))
            command = [
                args.python,
                "-m",
                "src.datagen.collect",
                "--task",
                task,
                "--seed",
                str(seed),
                "--output",
                str(episode),
                "--headless",
            ]
            start = time.monotonic()
            print(f"[batch] {task} seed={seed} successful={success}/{args.episodes}", flush=True)
            log_path = folder / f"attempt_{seed:010d}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                rc, failure = supervise(process, log_path, args.timeout)
            ok = episode_status(episode)
            result.write_text(
                json.dumps(
                    {
                        "task": task,
                        "seed": seed,
                        "returncode": rc,
                        "process_failure": failure,
                        "success": ok,
                        "elapsed_s": time.monotonic() - start,
                    },
                    indent=2,
                )
                + "\n"
            )
            success += int(ok)
        own = accepted_by_seed(folder.glob("episode_*.h5"))
        if args.reference_root is not None:
            external = accepted_by_seed(
                path
                for path in args.reference_root.glob(f"**/{task}/episode_*.h5")
                if folder not in path.resolve().parents
            )
        total = len(own.keys() | external.keys())
        if total < target_total:
            failed_tasks.append(task)
        print(
            f"[batch] {task}: {total}/{target_total} distinct successful episodes across collection shards", flush=True
        )
    return int(bool(failed_tasks))


if __name__ == "__main__":
    sys.exit(main())
