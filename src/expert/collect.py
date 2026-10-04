"""Collect expert episodes for one task in a single long-lived Isaac process.

Example:
    bash scripts/expert.sh collect --task stack_blocks --num 100 --out data/expert/stack_blocks
"""

import argparse
import json
from pathlib import Path
import time
import traceback


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--num", type=int, default=10, help="successful episodes to collect")
    ap.add_argument("--seed-start", type=int, default=100000)
    ap.add_argument("--seed-stride", type=int, default=1, help="seed step (parallel workers interleave seeds)")
    ap.add_argument("--max-attempts", type=int, default=None, help="default: 3 x num")
    ap.add_argument("--worker", type=int, default=0, help="worker id (separate attempt log)")
    ap.add_argument("--out", type=Path, default=None, help="output dir; omit to only measure success rate")
    ap.add_argument("--no-images", action="store_true", help="skip rendering (success-rate runs)")
    ap.add_argument(
        "--table-randomization", type=float, default=0.0, help="probability of random table/ground material"
    )
    return ap.parse_args()


def failing_predicates(env):
    """Names (and key args) of the official final predicates that do not hold now."""
    from src.datagen.state import terminal_checks

    rm = env.reward_manager
    out = []
    try:
        for check in terminal_checks(rm):
            if not rm.check_once(check, 0):
                name, args = check if isinstance(check, tuple) else (str(check), {})
                brief = {
                    k: v
                    for k, v in (args or {}).items()
                    if k in ("label", "label_A", "label_B", "label_list", "threshold")
                }
                out.append(f"{name}{brief}")
    except Exception as exc:  # diagnostics must never break collection
        out.append(f"<unavailable: {exc}>")
    return "; ".join(out) or "none failing now (sequential stage pending)"


def main():
    args = parse()
    from src.expert.sim import launch_app

    app = launch_app(cameras=not args.no_images)
    from src.expert.layout import LayoutError, LayoutSampler
    from src.expert.planner import EpisodeEnd, Planner, PlanningError
    from src.expert.sim import make_env
    from src.expert.skills import Skills
    from src.expert.tasks import TASKS

    if args.task not in TASKS:
        raise SystemExit(f"No expert for {args.task}; available: {sorted(TASKS)}")
    env = make_env(args.task, app, cameras=not args.no_images)
    sampler = LayoutSampler(args.task, table_randomization=args.table_randomization)
    out = args.out
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    log = (out / f"attempts_w{args.worker}.jsonl").open("a") if out is not None else None
    done_seeds = set()
    if out is not None:
        done_seeds = {int(f.stem.split("_")[-1]) for f in out.glob("episode_*.h5") if ".partial" not in f.name}
        tried = set()
        for f in out.glob(f"attempts_w{args.worker}.jsonl"):
            tried |= {json.loads(lbl)["seed"] for lbl in f.read_text().splitlines() if lbl.strip()}
        done_seeds |= tried
    ok = sum(1 for f in out.glob("episode_*.h5") if ".partial" not in f.name) if out is not None else 0
    max_attempts = args.max_attempts or 3 * args.num
    stats = {"success": 0, "fail": 0, "unstable": 0}
    seed = args.seed_start
    attempts = 0
    while ok < args.num and attempts < max_attempts:
        if seed in done_seeds:
            seed += args.seed_stride
            continue
        attempts += 1
        t0 = time.time()
        rec = {"seed": seed, "task": args.task}
        try:
            layout = sampler.sample(seed)
        except LayoutError as exc:
            rec.update(result="layout_error", reason=str(exc))
            seed += args.seed_stride
            continue
        if out is not None:
            from src.expert.recorder import Recorder
        recorder = Recorder(env, args.task, seed, layout, images=not args.no_images) if out is not None else None
        try:
            if not env.reset_layout(seed, layout):
                rec.update(result="unstable")
                stats["unstable"] += 1
                raise StopIteration
            planner = Planner(env, recorder)
            skills = Skills(planner)
            try:
                TASKS[args.task](planner, skills)
                # Program finished without the official end firing: wait briefly.
                for _ in range(40):
                    planner.step(planner.command.copy())
                reason = "program finished but task not successful: " + failing_predicates(env)
            except EpisodeEnd:
                reason = None if env.success[0] else f"step limit {env.step_lim} reached"
            except PlanningError as exc:
                reason = f"planning: {exc}"
            success = bool(env.success[0]) and env.end_flag[0] and reason is None
            durations = {}
            for seg in planner.segments:
                if seg.background:
                    continue
                end = seg.end_step if seg.end_step is not None else int(env.take_action_cnt[0])
                durations[seg.primitive] = durations.get(seg.primitive, 0) + end - seg.start_step
            rec["primitive_steps"] = durations
            rec.update(result="success" if success else "fail", reason=reason, steps=int(env.take_action_cnt[0]))
            if recorder is not None:
                recorder.annotate(planner)
                if success:
                    rec["file"] = str(recorder.save(out / f"episode_{seed:07d}.h5"))
            stats["success" if success else "fail"] += 1
            ok += int(success)
        except StopIteration:
            pass
        except Exception as exc:  # keep the process alive for later seeds
            traceback.print_exc()
            rec.update(result="error", reason=f"{type(exc).__name__}: {exc}")
            stats["fail"] += 1
        rec["seconds"] = round(time.time() - t0, 1)
        print("[expert]", json.dumps(rec), flush=True)
        if log is not None:
            log.write(json.dumps(rec) + "\n")
            log.flush()
        seed += args.seed_stride
    total = stats["success"] + stats["fail"]
    summary = {**stats, "success_rate": stats["success"] / total if total else None, "collected": ok}
    print("[expert] summary", json.dumps(summary), flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
