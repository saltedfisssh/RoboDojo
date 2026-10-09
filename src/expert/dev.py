"""Task handoff and evidence-producing iteration commands; no Isaac on import."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from src.expert.arguments import nonnegative_int, positive_int
from src.expert.development import (
    archive_sources,
    asset_fingerprint,
    create_plugin,
    gpu_lock,
    load_protocol,
    official_fingerprint,
    runtime_info,
    source_fingerprint,
    supervise,
    utc_now,
)
from src.expert.diagnostics import write_json
from src.expert.registry import load_planner, load_sampler, official_tasks, planner_spec


def inspect_task(task, seeds=(0, 1, 2)):
    from src.expert.layout_audit import audit_layout

    if task not in official_tasks():
        raise ValueError(f"Unknown official task: {task}")
    audit = audit_layout(task)
    result = {
        **audit,
        "task_source": f"task/RoboDojo/tasks/{task}.py",
        "task_config": f"task/RoboDojo/config/{task}.yml",
        "layout_samples": [],
        "errors": [],
    }
    try:
        spec = planner_spec(task)
        load_planner(spec)
        result.update(planner=spec.entrypoint, sampler=spec.sampler, status=spec.status)
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        result["errors"].append(str(exc))
        result["ready"] = False
        return result
    if audit["issues"] and spec.sampler is None:
        result["errors"].extend(audit["issues"])
    else:
        sampler = load_sampler(spec)(task)
        for seed in seeds:
            try:
                layout = sampler.sample(seed)
                objects = [
                    {
                        "kind": kind,
                        "category": category,
                        "model": obj["category_idx"],
                        "label": obj["label"],
                        "position": obj["default_pos"],
                    }
                    for kind in ("Rigid", "Geometry", "Articulation", "Garment", "Fluid", "Dynamic")
                    for category, items in layout.get(kind, {}).items()
                    for obj in items
                ]
                labels = [obj["label"] for obj in objects]
                if len(set(labels)) != len(labels):
                    raise ValueError("Duplicate object labels")
                result["layout_samples"].append(dict(seed=seed, objects=objects))
            except Exception as exc:
                result["errors"].append(f"seed {seed}: {type(exc).__name__}: {exc}")
    result["ready"] = not result["errors"]
    return result


def report_run(directory):
    from src.expert.storage import read_attempts

    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text())
    rows = read_attempts(directory / "raw")
    by_seed = {row["seed"]: row for row in rows}
    duplicates = len(rows) != len(by_seed)
    requested = manifest["seeds"]
    missing = [seed for seed in requested if seed not in by_seed]
    unexpected = sorted(set(by_seed) - set(requested))
    counts = Counter(row["result"] for row in rows)
    missing_files = [
        row["seed"]
        for row in rows
        if row["result"] == "success" and not (directory / "raw" / f"episode_{row['seed']:07d}.h5").is_file()
    ]
    execution = (
        json.loads((directory / "execution.json").read_text()) if (directory / "execution.json").exists() else None
    )
    current = source_fingerprint(manifest["task"])
    unchanged = current["sha256"] == manifest["source"]["sha256"]
    healthy = (
        execution is not None
        and not execution["timed_out"]
        and execution["exit_code"] in (0, 1)
        and not missing
        and not unexpected
        and not duplicates
        and not missing_files
        and not counts["error"]
        and not counts["layout_error"]
        and unchanged
    )
    successes = sum(by_seed.get(seed, {}).get("result") == "success" for seed in requested)
    success_rate = successes / len(requested)
    threshold_met = (
        successes >= manifest["protocol"]["minimum_rgb_successes"]
        if manifest["split"] == "rgb"
        else success_rate >= manifest["minimum_success_rate"]
    )
    report = {
        "schema": 1,
        "task": manifest["task"],
        "split": manifest["split"],
        "seeds": requested,
        "counts": dict(counts),
        "attempts": len(rows),
        "successes": successes,
        "success_rate_all_seeds": success_rate,
        "missing_seeds": missing,
        "unexpected_seeds": unexpected,
        "duplicate_attempts": duplicates,
        "missing_success_files": missing_files,
        "source_unchanged": unchanged,
        "execution": execution,
        "run_complete": healthy,
        "meets_threshold": healthy and threshold_met,
        "failures": [row for row in rows if row["result"] != "success"],
        "attempts_file_pattern": str(directory / "raw/attempts_w*.jsonl"),
        "progress": json.loads((directory / "progress.json").read_text())
        if (directory / "progress.json").exists()
        else None,
        "source_sha256": manifest["source"]["sha256"],
    }
    write_json(directory / "report.json", report)
    return report


def run(args):
    protocol = load_protocol(args.task)
    if official_fingerprint(args.task) != protocol["official_files"]:
        raise ValueError("Official task/config/reward changed since handoff; review it before creating a new protocol")
    seeds = args.seeds if args.seeds is not None else protocol["seeds"][args.split]
    if args.seeds is not None and args.split != "development":
        raise ValueError("Custom --seeds are only allowed for development")
    reserved = set(protocol["seeds"]["validation"] + protocol["seeds"]["rgb"])
    if len(set(seeds)) != len(seeds) or (args.split == "development" and reserved.intersection(seeds)):
        raise ValueError("Development seeds must be unique and disjoint from validation/RGB seeds")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use --gpu with physical GPU indices; remove CUDA_VISIBLE_DEVICES for Isaac/Vulkan")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    preflight = inspect_task(args.task, seeds)
    write_json(args.out / "preflight.json", preflight)
    if not preflight["ready"]:
        raise ValueError(f"Preflight failed; see {args.out / 'preflight.json'}")
    spec = planner_spec(args.task)
    source = source_fingerprint(args.task)
    archive_sources(args.out / "source.zip", source)
    manifest = {
        "schema": 1,
        "task": args.task,
        "split": args.split,
        "seeds": seeds,
        "gpu": args.gpu,
        "images": args.split == "rgb" or args.images,
        "source": source,
        "runtime": runtime_info(),
        "protocol": protocol,
        "minimum_success_rate": protocol["minimum_success_rate"],
        "created_utc": utc_now(),
        "planner": spec.entrypoint,
        "sampler": spec.sampler,
        "assets": asset_fingerprint(preflight["layout_samples"]),
    }
    write_json(args.out / "run.json", manifest)
    command = [
        sys.executable,
        "-m",
        "src.expert",
        "collect",
        "--task",
        args.task,
        "--num",
        str(len(seeds)),
        "--seeds",
        *map(str, seeds),
        "--gpu",
        str(args.gpu),
        "--out",
        str(args.out / "raw"),
        "--diagnostics",
        "--save-failures",
        "--progress",
        str(args.out / "progress.json"),
    ]
    if not manifest["images"]:
        command.append("--no-images")
    with gpu_lock(args.gpu, args.wait_timeout) as lock_fd:
        if source_fingerprint(args.task)["sha256"] != source["sha256"]:
            raise ValueError("Source changed while waiting for GPU; start a fresh run")
        print(json.dumps({"run": str(args.out), "seeds": seeds, "gpu": args.gpu}), flush=True)
        execution = supervise(
            command,
            args.out / "worker.log",
            timeout=args.timeout,
            progress_path=args.out / "progress.json",
            lock_fd=lock_fd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    write_json(args.out / "execution.json", execution)
    report = report_run(args.out)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["meets_threshold"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create an isolated task plugin and contributor handoff")
    init.add_argument("--task", required=True)
    inspect = sub.add_parser("inspect", help="audit default layout support and sample layouts without Isaac")
    inspect.add_argument("--task", required=True)
    inspect.add_argument("--seeds", nargs="+", type=nonnegative_int, default=[0, 1, 2])
    trial = sub.add_parser("run", help="run a fixed seed split with diagnostics and per-episode timeout")
    trial.add_argument("--task", required=True)
    trial.add_argument("--split", choices=("development", "validation", "rgb"), default="development")
    trial.add_argument("--seeds", nargs="+", type=nonnegative_int)
    trial.add_argument("--out", type=Path, required=True)
    trial.add_argument("--gpu", type=nonnegative_int, default=0)
    trial.add_argument("--images", action="store_true", help="also record RGB on development/validation splits")
    trial.add_argument("--timeout", type=positive_int, default=900, help="startup and each episode timeout in seconds")
    trial.add_argument("--wait-timeout", type=positive_int, default=600, help="wait for cooperative GPU slot")
    report = sub.add_parser("report", help="summarize failures, missing seeds and code changes")
    report.add_argument("--run", type=Path, required=True)
    qualify = sub.add_parser("qualify", help="validate RGB, export, decode videos and restore an intermediate frame")
    qualify.add_argument("--validation", type=Path, required=True)
    qualify.add_argument("--rgb", type=Path, required=True)
    qualify.add_argument("--out", type=Path, required=True)
    qualify.add_argument("--gpu", type=nonnegative_int, default=0)
    qualify.add_argument("--timeout", type=positive_int, default=1200)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(create_plugin(args.task))
            return 0
        if args.command == "inspect":
            result = inspect_task(args.task, args.seeds)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ready"] else 1
        if args.command == "run":
            return run(args)
        if args.command == "report":
            print(json.dumps(report_run(args.run), indent=2, ensure_ascii=False))
            return 0
        from src.expert.qualification import qualify

        return qualify(args)
    except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
