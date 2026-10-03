"""Collect RGB episodes as another campaign finishes its physical smoke tests."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from src.datagen.batch import accepted_by_seed
from src.datagen.campaign import smoke_passed
from src.datagen.layouts import ROOT


def latest_smoke(root, task, revision):
    if revision != "all":
        path = root / "debug" / f"{task}_{revision}.h5"
        return path if path.exists() or path.with_suffix(".process.json").exists() else None
    attempts = [p for p in (root / "debug").glob(f"{task}_*.h5") if not p.name.endswith(".partial.h5")]
    attempts.extend(
        p.with_name(p.name.removesuffix(".process.json") + ".h5")
        for p in (root / "debug").glob(f"{task}_*.process.json")
        if not p.with_name(p.name.removesuffix(".process.json") + ".h5").exists()
    )
    return max(
        attempts,
        key=lambda p: (p if p.exists() else p.with_suffix(".process.json")).stat().st_mtime_ns,
        default=None,
    )


def smoke_fingerprint(path):
    if path is None:
        return None
    artifact = path if path.exists() else path.with_suffix(".process.json")
    return f"{artifact.name}:{artifact.stat().st_mtime_ns}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--smoke-revision", default="all", help="A specific revision, or all for the newest attempt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--output-shard", help="RGB output folder name; defaults to <smoke-revision>rgb")
    parser.add_argument("--follow", action="store_true", help="Revisit incomplete tasks after a new smoke test")
    args = parser.parse_args()
    root = args.root.resolve()
    results = {}
    attempted = {}
    while len(results) < len(set(args.tasks)):
        progressed = False
        for task in args.tasks:
            if task in results:
                continue
            accepted = len(accepted_by_seed((root / "raw").glob(f"**/{task}/episode_*.h5")))
            if accepted >= args.episodes:
                results[task] = {"successful_rgb_episodes": accepted, "ready": True}
                continue
            smoke = latest_smoke(root, task, args.smoke_revision)
            fingerprint = smoke_fingerprint(smoke)
            if args.follow and task in attempted and attempted[task] == fingerprint:
                continue
            if not accepted and (
                smoke is None or not smoke.exists() and not smoke.with_suffix(".process.json").exists()
            ):
                continue
            progressed = True
            attempted[task] = fingerprint
            if (smoke is not None and not smoke_passed(smoke)) or (not accepted and smoke is None):
                if not args.follow:
                    results[task] = "physical_smoke_failed"
                else:
                    print(f"[dispatch] {task}: waiting for a new successful smoke test", flush=True)
            else:
                output = root / "raw" / (args.output_shard or f"{args.smoke_revision}rgb")
                if args.follow:
                    # A corrected planner needs a fresh bounded seed budget;
                    # preserve failed attempts from its preceding revision.
                    output /= smoke.stem if smoke is not None else "baseline"
                other = len(
                    accepted_by_seed(
                        p for p in (root / "raw").glob(f"**/{task}/episode_*.h5") if output not in p.parents
                    )
                )
                if other < args.episodes:
                    print(f"[dispatch] {task}: collect {args.episodes - other} RGB episodes", flush=True)
                    subprocess.run(
                        [
                            "bash",
                            str(ROOT / "scripts/datagen-uv.sh"),
                            "batch",
                            "--tasks",
                            task,
                            "--episodes",
                            str(args.episodes - other),
                            "--max-attempts",
                            "6",
                            "--seed-start",
                            str(args.seed_start),
                            "--timeout",
                            "14400",
                            "--output",
                            str(output),
                            "--reference-root",
                            str(root / "raw"),
                        ],
                        cwd=ROOT,
                        check=False,
                    )
                count = len(accepted_by_seed((root / "raw").glob(f"**/{task}/episode_*.h5")))
                if count >= args.episodes or not args.follow:
                    results[task] = {"successful_rgb_episodes": count, "ready": count >= args.episodes}
        (root / f"dispatch_{args.smoke_revision}.json").write_text(json.dumps(results, indent=2) + "\n")
        if not progressed and len(results) < len(set(args.tasks)):
            time.sleep(30)


if __name__ == "__main__":
    main()
