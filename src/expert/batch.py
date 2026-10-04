"""Run several collect workers per task in parallel and report success rates.

Example (4 workers, 100 successful episodes with RGB):
    bash scripts/expert.sh batch --task stack_blocks --num 100 --workers 4 --out data/expert/raw/stack_blocks

Workers interleave seeds (seed_start + k, + k + K, ...) and write into the same
output directory; the run resumes where it stopped. ``--num`` is the total
target, split evenly. Each worker is an independent Isaac process.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=100000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--table-randomization", type=float, default=0.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    have = len([f for f in args.out.glob("episode_*.h5") if ".partial" not in f.name])
    need = max(0, args.num - have)
    per = -(-need // args.workers) if need else 0
    procs = []
    for k in range(args.workers if need else 0):
        cmd = [
            sys.executable,
            "-m",
            "src.expert.collect",
            "--task",
            args.task,
            "--out",
            str(args.out),
            "--num",
            str(have + per),
            "--seed-start",
            str(args.seed_start + k),
            "--seed-stride",
            str(args.workers),
            "--max-attempts",
            str(4 * per),
            "--worker",
            str(k),
            "--table-randomization",
            str(args.table_randomization),
        ]
        if args.no_images:
            cmd.append("--no-images")
        log = (args.out / "logs" / f"worker{k}.log").open("a")
        procs.append(
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        )
        time.sleep(20)  # stagger Kit startup
    for proc in procs:
        proc.wait()
    print(json.dumps(summarize(args.out)), flush=True)


def summarize(out):
    rows = []
    for f in Path(out).glob("attempts_w*.jsonl"):
        rows += [json.loads(lbl) for lbl in f.read_text().splitlines() if lbl.strip()]
    tried = [r for r in rows if r["result"] in ("success", "fail", "error")]
    ok = [r for r in tried if r["result"] == "success"]
    reasons = {}
    for r in tried:
        if r["result"] != "success":
            key = (r.get("reason") or "").split(":")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
    steps = sorted(r["steps"] for r in ok)
    return {
        "attempts": len(tried),
        "success": len(ok),
        "success_rate": round(len(ok) / len(tried), 3) if tried else None,
        "unstable_layouts": sum(r["result"] == "unstable" for r in rows),
        "median_steps": steps[len(steps) // 2] if steps else None,
        "failure_reasons": reasons,
        "episodes_on_disk": len([f for f in Path(out).glob("episode_*.h5") if ".partial" not in f.name]),
    }


if __name__ == "__main__":
    main()
