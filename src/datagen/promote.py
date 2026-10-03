"""Render successful state-only physical episodes without counting duplicate seeds."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import h5py

from src.datagen.batch import accepted_by_seed, episode_status, supervise
from src.datagen.campaign import smoke_passed
from src.datagen.layouts import ROOT


def candidates(root, required, exclude=()):
    accepted = {}
    selected = set()
    exclude = {path.resolve() for path in exclude}
    for source in sorted((root / "debug").glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True):
        if source.resolve() in exclude or source.name.endswith(".partial.h5") or not smoke_passed(source):
            continue
        with h5py.File(source) as file:
            meta = json.loads(file.attrs["metadata"])
            if "images" in file:
                continue
        task, seed = meta["task"], meta["seed"]
        if task not in accepted:
            accepted[task] = accepted_by_seed((root / "raw").glob(f"**/{task}/episode_*.h5"))
        if len(accepted[task]) >= required or seed in accepted[task] or (task, seed) in selected:
            continue
        selected.add((task, seed))
        yield source, task, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--follow", action="store_true", help="Wait for new physically successful plans")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--exclude-source", nargs="*", type=Path, default=[], help="Plans handled by another worker")
    args = parser.parse_args()
    if args.episodes < 1 or args.timeout <= 0:
        parser.error("episodes and timeout must be positive")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".promotion.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another promotion worker holds this review directory")
    while True:
        for source, task, seed in candidates(root, args.episodes, args.exclude_source):
            output = root / "raw" / "promoted" / task / f"episode_{seed:010d}.h5"
            report = output.with_suffix(".process.json")
            if output.exists() or report.exists() or output.with_suffix(".partial.h5").exists():
                continue
            # A preceding render may have filled the task while this iterator
            # was suspended. Recheck counts immediately before dispatching.
            accepted = accepted_by_seed((root / "raw").glob(f"**/{task}/episode_*.h5"))
            if seed in accepted or len(accepted) >= args.episodes:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = output.with_suffix(".log")
            print(f"[promote] {task} seed={seed} source={source.name}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [
                        "bash",
                        str(ROOT / "scripts/datagen-uv.sh"),
                        "render_episode",
                        "--mode",
                        "state",
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--headless",
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                returncode, failure = supervise(process, log_path, args.timeout)
            result = {
                "source": str(source),
                "returncode": returncode,
                "failure": failure,
                "success": episode_status(output),
            }
            report.write_text(json.dumps(result, indent=2) + "\n")
            print(f"[promote] result: {json.dumps(result)}", flush=True)
        if not args.follow:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
