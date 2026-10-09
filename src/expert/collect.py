"""Collect successful demonstrations in one long-lived Isaac process."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

from src.expert.arguments import add_collection_arguments, help_parser, nonnegative_int, positive_int
from src.expert.registry import load_planner, load_sampler, planner_spec
from src.expert.storage import attempted_seeds, episode_files, log_attempt, prepare_output


def parse(argv=None):
    parser = help_parser(
        "collect",
        "Collect successful demonstrations in one Isaac process, or measure success without saving.",
        epilog="Examples:\n"
        "  %(prog)s --task stack_blocks --num 20 --no-images --seed-start 0\n"
        "  %(prog)s --task stack_blocks --num 100 --out data/expert/raw/stack_blocks\n"
        "  %(prog)s --task stack_blocks --num 3 --seeds 0 1 2 --no-images\n"
        "\nResume: reuse --out with the same task, planner and recording settings.\n"
        "Existing successes count toward --num; previously attempted seeds are skipped.\n"
        "Fixed --seeds are each tried once, ignoring --seed-start, --seed-stride and\n"
        "--max-attempts. All listed seeds run even if --num is reached early.\n"
        "Exit status: 0 when --num successes are reached, 1 if the target is unmet.\n"
        "Use 'dev run --help' for fixed splits, source snapshots and failure reports.",
    )
    add_collection_arguments(parser)
    attempts = parser.add_argument_group("Attempts and reproducibility")
    attempts.add_argument(
        "--seed-stride", type=positive_int, default=1, help="increment between candidate layout seeds"
    )
    attempts.add_argument(
        "--max-attempts",
        type=positive_int,
        help="maximum new attempts in this invocation, including layout failures (default: 3 x --num)",
    )
    attempts.add_argument(
        "--worker", type=nonnegative_int, default=0, help="worker ID used in attempts_w<ID>.jsonl logs"
    )
    output = parser.add_argument_group("Output and diagnostics")
    output.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="save successful episode_*.h5 and attempt logs here; omit for a trial without saved data",
    )
    parser.add_argument("--additional", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scheduler", help=argparse.SUPPRESS)
    attempts.add_argument(
        "--seeds",
        nargs="+",
        type=nonnegative_int,
        metavar="SEED",
        help="space-separated, unique seeds to try once each; requires no prior attempts in --out",
    )
    output.add_argument(
        "--diagnostics",
        action="store_true",
        help="with --out, save layouts, terminal states and planner stages in diagnostics/",
    )
    output.add_argument(
        "--save-failures",
        action="store_true",
        help="with --out, save recorded failed trajectories separately in failures/",
    )
    parser.add_argument("--progress", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def collect_attempt(args, spec, factory, env, sampler, seed):
    from src.expert.layout import LayoutError
    from src.expert.runtime import run_episode

    started = time.monotonic()
    row = dict(seed=seed, task=args.task, planner=spec.entrypoint, gpu=args.gpu)
    from src.expert.diagnostics import progress, write_json

    if args.diagnostics or args.progress is not None:
        env.expert_diagnostics = args.diagnostics
        env.expert_progress = lambda **values: progress(args.progress, seed, **values)
    progress(args.progress, seed, step=0, stage="sampling/reset")
    try:
        layout = sampler.sample(seed)
        if args.diagnostics and args.out is not None:
            write_json(args.out / "diagnostics" / f"layout_{seed:07d}.json", layout)
        recorder = None
        if args.out is not None:
            from src.expert.recorder import Recorder

            recorder = Recorder(
                env, args.task, seed, layout, images=not args.no_images, planner_entrypoint=spec.entrypoint
            )
        result = run_episode(env, factory, task=args.task, seed=seed, layout=layout, recorder=recorder)
        row.update(asdict(result))
        if args.diagnostics and args.out is not None:
            path = args.out / "diagnostics" / f"episode_{seed:07d}.json"
            write_json(path, row.pop("diagnostics", {}))
            row["diagnostics_file"] = str(path)
        if result.success and recorder is not None:
            row["file"] = str(recorder.save(args.out / f"episode_{seed:07d}.h5"))
        elif args.save_failures and recorder is not None and recorder.rows:
            directory = args.out / "failures"
            directory.mkdir(exist_ok=True)
            row["failure_file"] = str(recorder.save(directory / f"failed_{seed:07d}.h5", outcome=result.result))
    except LayoutError as exc:
        row.update(result="layout_error", reason=str(exc))
    except Exception as exc:
        traceback.print_exc()
        row.update(result="error", reason=f"{type(exc).__name__}: {exc}")
    row["seconds"] = round(time.monotonic() - started, 1)
    log_attempt(args.out, args.worker, row)
    return row


def collect(args, spec, factory):
    from src.expert.runtime import simulation_session
    from src.expert.scheduling import WorkerClient

    prepare_output(
        args.out,
        task=args.task,
        planner=spec.entrypoint,
        sampler=spec.sampler,
        images=not args.no_images,
        table_randomization=args.table_randomization,
    )
    done = attempted_seeds(args.out)
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds) or args.scheduler or args.additional:
            raise ValueError("--seeds must be unique and cannot be combined with a batch scheduler")
        if done:
            raise ValueError("Fixed-seed runs require fresh output; use a new run directory")
    fixed = iter(args.seeds) if args.seeds is not None else None
    # Static batches assign new-success quotas; dynamic batches use the parent
    # coordinator's target and never stop based on a worker-local success count.
    collected = 0 if args.additional else len(episode_files(args.out))
    stats = dict(success=0, fail=0, unstable=0, layout_error=0, error=0)
    if fixed is None and not args.scheduler and collected >= args.num:
        summary = {**stats, "success_rate": None, "collected": collected, "target_reached": True}
        print("[expert] summary", json.dumps(summary), flush=True)
        return summary
    sampler = load_sampler(spec)(args.task, table_randomization=args.table_randomization)
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
                elif fixed is not None:
                    seed = next(fixed, None)
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
