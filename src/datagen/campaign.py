"""Smoke-test each new task before spending RGB collection time; inspect artifacts, not Kit's exit code."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import h5py

from src.datagen.batch import accepted_by_seed, supervise
from src.datagen.layouts import ROOT
from src.datagen.planner import SUPPORTED_TASKS


def smoke_passed(path):
    try:
        with h5py.File(path) as f:
            return bool(f.attrs.get("complete") and f.attrs.get("success") and "terminal" in f)
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Unique tag; previous attempts are preserved")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--priority", type=int, help="GPU queue priority for physical smoke validation")
    parser.add_argument(
        "--smoke-only", action="store_true", help="Validate all listed task programs before RGB batches"
    )
    args = parser.parse_args()
    if not args.revision.replace("_", "").isalnum():
        parser.error("revision must contain only letters, numbers or underscores")
    root = args.root.resolve()
    debug = root / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    results = []
    for task in args.tasks:
        existing = len(accepted_by_seed((root / "raw").glob(f"**/{task}/episode_*.h5")))
        own = len(accepted_by_seed((root / "raw" / args.revision / task).glob("episode_*.h5")))
        if existing >= args.episodes:
            continue
        smoke = debug / f"{task}_{args.revision}.h5"
        log_path = smoke.with_suffix(".log")
        if not smoke.exists() and not smoke.with_suffix(".partial.h5").exists():
            command = [
                str(ROOT / "scripts/datagen-uv.sh"),
                "collect",
                "--task",
                task,
                "--seed",
                str(args.seed_start + 1),
                "--output",
                str(smoke),
                "--headless",
                "--no-images",
            ]
            if args.priority is not None:
                command.extend(["--priority", str(args.priority)])
            print(f"[campaign] smoke {task}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    ["bash", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                rc, failure = supervise(process, log_path, args.timeout)
            smoke.with_suffix(".process.json").write_text(json.dumps({"returncode": rc, "failure": failure}) + "\n")
        passed = smoke_passed(smoke)
        results.append({"task": task, "smoke_passed": passed, "time": time.time()})
        (root / f"campaign_{args.revision}.json").write_text(json.dumps(results, indent=2) + "\n")
        if not passed:
            print(f"[campaign] {task}: physical smoke failed; RGB batch skipped", flush=True)
            continue
        if args.smoke_only:
            print(f"[campaign] {task}: physical smoke passed", flush=True)
            continue
        print(f"[campaign] {task}: smoke passed; collect {args.episodes - existing} more RGB episodes", flush=True)
        subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/datagen-uv.sh"),
                "batch",
                "--tasks",
                task,
                "--episodes",
                str(args.episodes - existing + own),
                "--max-attempts",
                str(args.max_attempts),
                "--seed-start",
                str(args.seed_start),
                "--timeout",
                str(args.timeout),
                "--output",
                str(root / "raw" / args.revision),
                "--reference-root",
                str(root / "raw"),
            ],
            cwd=ROOT,
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
