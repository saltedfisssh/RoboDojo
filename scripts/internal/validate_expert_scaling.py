"""Validate collection benchmarks and summarize their sampled resources.

Run after benchmark_expert_scaling.py completes, so validation I/O does not affect
timings. Checks every saved episode's state/checkpoints and three RGB frames per
camera. Supports fixed seeds or success targets, with static or dynamic dispatch;
use validate_expert_dataset.py for full video decoding after export.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import median
import time


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resources(directory):
    samples = read_jsonl(directory / "resources.jsonl")
    assert samples, f"No resource samples in {directory}"
    peaks = {}
    for sample in samples:
        for line in sample["gpus"].splitlines():
            index, _uuid, _utilization, memory, _power = line.split(",")
            peaks[index.strip()] = max(peaks.get(index.strip(), 0), float(memory) / 1024)
    cpu_first, cpu_last = samples[0]["cpu"], samples[-1]["cpu"]
    periods = cpu_last["nr_periods"] - cpu_first["nr_periods"]
    throttled = cpu_last["nr_throttled"] - cpu_first["nr_throttled"]
    return {
        "gpu_peak_gib": peaks,
        "max_gpu_peak_gib": max(peaks.values()),
        "host_memory_peak_gib": max(sample["memory_bytes"] for sample in samples) / 1024**3,
        "cpu_throttled_period_fraction": throttled / periods if periods else 0,
        "resource_samples": len(samples),
    }


def validate_schedule(events, target):
    active, last_completed, assigned, gaps = {}, {}, {}, []
    successes = 0
    for event in events:
        worker = event.get("worker")
        if event["event"] == "assigned":
            assert worker not in active and successes + len(active) < target
            active[worker] = event["seed"]
            assigned.setdefault(worker, []).append(event["seed"])
            if worker in last_completed:
                gaps.append(event["elapsed"] - last_completed[worker])
        elif event["event"] == "completed":
            assert active.pop(worker) == event["seed"]
            successes += int(event["result"] == "success")
            assert event["successes"] == successes <= target
            last_completed[worker] = event["elapsed"]
        elif event["event"] == "worker_exit":
            assert worker not in active
    assert not active, "Unfinished queue reservations"
    return assigned, {
        "queue_reservations_valid": True,
        "dispatch_gap_median_seconds": median(gaps) if gaps else None,
        "dispatch_gap_max_seconds": max(gaps) if gaps else None,
    }


def validate_raw(directory, summary):
    import h5py
    import numpy as np

    raw = directory / "raw"
    command = json.loads((directory / "command.json").read_text())
    seed_start = int(command[command.index("--seed-start") + 1])
    episodes = int(command[command.index("--num") + 1])
    workers = summary["workers"]
    batch = json.loads((directory / "batch.log").read_text().splitlines()[-1])
    dynamic = batch.get("scheduling", "static") == "dynamic"
    goal = summary.get("workload") == "success_target"
    assigned, queue_validation = {}, {}
    if dynamic:
        events = read_jsonl(raw / Path(batch["scheduling_log"]).name)
        assigned, queue_validation = validate_schedule(events, episodes)
    attempts = []
    for worker in range(workers):
        path = raw / f"attempts_w{worker}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        # Check allocation from actual completed work, including failed seeds.
        if dynamic:
            expected = sorted(assigned.get(worker, []))
        elif goal:
            expected = [seed_start + worker + index * workers for index in range(len(rows))]
        else:
            expected = list(range(seed_start + worker, seed_start + episodes, workers))
        assert sorted(row["seed"] for row in rows) == expected
        attempts.extend(rows)
    seeds = sorted(row["seed"] for row in attempts)
    assert seeds == summary["seeds"] and len(set(seeds)) == len(seeds)
    if not goal or dynamic:
        assert seeds == list(range(seed_start, seed_start + (len(attempts) if goal else episodes)))
    assert all(row["result"] in ("success", "fail", "unstable") for row in attempts)
    successful = {row["seed"]: row for row in attempts if row["result"] == "success"}
    files = sorted(raw.glob("episode_*.h5"))
    assert not list(raw.glob("*.partial.h5")), f"Incomplete writes in {raw}"
    assert len(files) == len(successful) == summary["successes"]
    if goal:
        assert len(successful) == episodes == summary["success_target"] and batch["target_reached"]
    assignment = {str(worker): summary["gpu_ids"][worker % len(summary["gpu_ids"])] for worker in range(workers)}
    assert batch["worker_gpus"] == assignment
    for worker, gpu in assignment.items():
        log = (raw / "logs" / f"worker{worker}.log").read_text(errors="replace")
        if int(worker) not in batch.get("stopped_unused_workers", []):
            assert f"[expert] devices: renderer={gpu}, curobo=cuda:{gpu}" in log, (directory, worker)
        else:
            assert not assigned.get(int(worker)), (directory, worker)
        assert "Traceback (most recent call last):" not in log, (directory, worker)
    assert len(batch["worker_exit_codes"]) == workers
    assert all(
        code in (0, 1) or worker in batch.get("stopped_unused_workers", [])
        for worker, code in enumerate(batch["worker_exit_codes"])
    )
    seen, frames, rgb_samples = set(), 0, 0
    for path in files:
        with h5py.File(path, "r") as episode:
            meta = json.loads(episode.attrs["meta"])
            seed, count = meta["seed"], meta["num_frames"]
            assert seed in successful and seed not in seen, path
            seen.add(seed)
            assert meta["task"] == summary["task"] and meta["success"], path
            assert 0 < count == successful[seed]["steps"] <= meta["step_lim"], path
            assert episode["action"].shape == episode["state"].shape == (count, 14), path
            for key in ("action", "state"):
                assert np.isfinite(episode[key][()]).all(), (path, key)
            contexts = episode["checkpoint/context"]
            assert contexts.shape == (count,), path
            for context in contexts:
                assert isinstance(json.loads(context), dict), path
            arrays = {}
            episode["checkpoint/arrays"].visititems(
                lambda key, item: arrays.update({key: item}) if isinstance(item, h5py.Dataset) else None
            )
            assert arrays, path
            for key, array in arrays.items():
                assert array.shape[0] == count and np.isfinite(array[()]).all(), (path, key)
            segments = json.loads(episode.attrs["segments"])
            assert segments and all(0 <= part["start_frame"] < part["end_frame"] <= count for part in segments), path
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                images = episode[f"images/{camera}"]
                assert images.shape == (count, 480, 640, 3) and images.dtype == np.uint8, (path, camera)
                for index in sorted({0, count // 2, count - 1}):
                    assert images[index].std() > 1, (path, camera, index)
                    rgb_samples += 1
            frames += count
    assert seen == set(successful) and frames == summary["saved_frames"]
    return {
        **queue_validation,
        "raw_validation_passed": True,
        "validated_frames": frames,
        "rgb_frames_sampled": rgb_samples,
        "workers_per_gpu": dict(Counter(assignment.values())),
        "worker_attempts": {str(worker): len(assigned.get(worker, [])) for worker in range(workers)}
        if dynamic
        else None,
        "outcomes": {str(row["seed"]): row["result"] for row in sorted(attempts, key=lambda row: row["seed"])},
        "failures": [row for row in attempts if row["result"] != "success"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed benchmark root directory")
    args = parser.parse_args()
    started = time.monotonic()
    summaries = json.loads((args.input / "summary.json").read_text())
    directories = {}
    for path in args.input.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        key = (summary["workers"], tuple(summary["gpu_ids"]), summary.get("scheduling", "static"))
        assert key not in directories, f"Duplicate scale {key}"
        directories[key] = path.parent
    rows = []
    for summary in summaries:
        assert summary["benchmark_valid"], summary
        if summary.get("workload") == "success_target":
            assert summary["success_target"] == summaries[0]["success_target"], "Unequal success targets"
        else:
            assert summary["seeds"] == summaries[0]["seeds"], "Unequal workloads"
        directory = directories[(summary["workers"], tuple(summary["gpu_ids"]), summary.get("scheduling", "static"))]
        assert json.loads((directory / "summary.json").read_text()) == summary
        row = {**summary, "directory": str(directory.resolve())}
        row.update(resources(directory))
        row.update(validate_raw(directory, summary))
        row["wall_speedup"] = summaries[0]["wall_seconds"] / summary["wall_seconds"]
        baseline_rate = summaries[0]["episodes_per_hour"]
        row["success_throughput_speedup"] = summary["episodes_per_hour"] / baseline_rate if baseline_rate else None
        rows.append(row)
        print(
            json.dumps({key: row[key] for key in ("workers", "successes", "wall_seconds", "raw_validation_passed")}),
            flush=True,
        )
    report = {
        "runs": rows,
        "same_seed_outcomes": all(row["outcomes"] == rows[0]["outcomes"] for row in rows),
        "validation_seconds": time.monotonic() - started,
        "rgb_validation": "first, middle, last frame per episode and camera; remaining RGB frames not decoded",
        "resource_caveat": "5-second samples; host memory includes filesystem cache; CPU usage covers the whole cgroup",
    }
    (args.input / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "gpus",
        "workers",
        "scheduling",
        "attempts",
        "successes",
        "wall_seconds",
        "episodes_per_hour",
        "wall_speedup",
        "saved_frames_per_second",
        "max_gpu_peak_gib",
        "host_memory_peak_gib",
        "cpu_cores_mean",
        "raw_validation_passed",
    ]
    with (args.input / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
