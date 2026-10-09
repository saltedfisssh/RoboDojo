"""Collect successful demonstrations in one long-lived Isaac process."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

from src.expert.arguments import add_collection_arguments, nonnegative_int, positive_int
from src.expert.registry import load_planner, planner_spec
from src.expert.storage import attempted_seeds, episode_files, log_attempt, prepare_output


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_collection_arguments(parser)
    parser.add_argument("--seed-stride", type=positive_int, default=1)
    parser.add_argument("--max-attempts", type=positive_int, help="default: 3 x num")
    parser.add_argument("--worker", type=nonnegative_int, default=0)
    parser.add_argument("--out", type=Path, help="omit to only measure success rate")
    parser.add_argument("--additional", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scheduler", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def collect_attempt(args, spec, factory, env, sampler, seed):
    from src.expert.layout import LayoutError
    from src.expert.runtime import run_episode

    started = time.monotonic()
    row = dict(seed=seed, task=args.task, planner=spec.entrypoint, gpu=args.gpu)
    try:
        layout = sampler.sample(seed)
        recorder = None
        if args.out is not None:
            from src.expert.recorder import Recorder

            recorder = Recorder(
                env, args.task, seed, layout, images=not args.no_images, planner_entrypoint=spec.entrypoint
            )
        result = run_episode(env, factory, task=args.task, seed=seed, layout=layout, recorder=recorder)
        row.update(asdict(result))
        if result.success and recorder is not None:
            row["file"] = str(recorder.save(args.out / f"episode_{seed:07d}.h5"))
    except LayoutError as exc:
        row.update(result="layout_error", reason=str(exc))
    except Exception as exc:
        traceback.print_exc()
        row.update(result="error", reason=f"{type(exc).__name__}: {exc}")
    row["seconds"] = round(time.monotonic() - started, 1)
    log_attempt(args.out, args.worker, row)
    return row


def collect(args, spec, factory):
    from src.expert.layout import LayoutSampler
    from src.expert.runtime import simulation_session
    from src.expert.scheduling import WorkerClient

    prepare_output(
        args.out,
        task=args.task,
        planner=spec.entrypoint,
        images=not args.no_images,
        table_randomization=args.table_randomization,
    )
    done = attempted_seeds(args.out)
    # Static batches assign new-success quotas; dynamic batches use the parent
    # coordinator's target and never stop based on a worker-local success count.
    collected = 0 if args.additional else len(episode_files(args.out))
    stats = dict(success=0, fail=0, unstable=0, layout_error=0, error=0)
    if not args.scheduler and collected >= args.num:
        summary = {**stats, "success_rate": None, "collected": collected, "target_reached": True}
        print("[expert] summary", json.dumps(summary), flush=True)
        return summary
    sampler = LayoutSampler(args.task, table_randomization=args.table_randomization)
    seed = args.seed_start
    with simulation_session(
        args.task,
        images=not args.no_images,
        exit_code=lambda: 0 if args.scheduler else int(collected < args.num),
        device_id=args.gpu,
    ) as env:
        connection = WorkerClient(args.scheduler, args.worker) if args.scheduler else nullcontext()
        with connection as jobs:
            previous = None
            while True:
                if jobs is not None:
                    seed = jobs.acquire(previous)
                    if seed is None:
                        break
                else:
                    if collected >= args.num or sum(stats.values()) >= (args.max_attempts or 3 * args.num):
                        break
                    while seed in done:
                        seed += args.seed_stride
                previous = collect_attempt(args, spec, factory, env, sampler, seed)
                stats[previous["result"]] += 1
                collected += int(previous["result"] == "success")
                done.add(seed)
                seed += args.seed_stride
            target_reached = jobs.target_reached if jobs is not None else collected >= args.num
        total = stats["success"] + stats["fail"] + stats["error"]
        summary = {
            **stats,
            "success_rate": stats["success"] / total if total else None,
            "collected": collected,
            "target_reached": target_reached,
        }
        # Kit fast shutdown may terminate inside the session's __exit__.
        print("[expert] summary", json.dumps(summary), flush=True)
        return summary


def main(argv=None):
    args = parse(argv)
    try:
        spec = planner_spec(args.task, args.planner)
        factory = load_planner(spec)
        summary = collect(args, spec, factory)
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0 if args.scheduler or summary["target_reached"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
