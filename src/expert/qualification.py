"""Check real data and produce a reviewable qualification bundle."""

import json
import os
from pathlib import Path
import sys

from src.expert.development import (
    ROOT,
    digest,
    gpu_lock,
    load_protocol,
    official_fingerprint,
    runtime_info,
    source_fingerprint,
    supervise,
)
from src.expert.diagnostics import write_json
from src.expert.storage import episode_files, read_attempts


def validate_raw(path, *, images=True):
    import h5py
    import numpy as np

    with h5py.File(path, "r") as episode:
        meta = json.loads(episode.attrs["meta"])
        n = meta["num_frames"]
        if episode.attrs["schema"] != "robodojo.expert-episode.v1" or not meta["success"]:
            raise ValueError(f"Not a successful expert episode: {path}")
        if not 0 < n == meta["steps"] <= meta["step_lim"]:
            raise ValueError(f"Invalid frame count/budget: {path}")
        for key in ("action", "state"):
            array = episode[key][()]
            if array.shape != (n, 14) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key}: {path}")
        segments = json.loads(episode.attrs["segments"])
        if not segments or any(not 0 <= row["start_frame"] < row["end_frame"] <= n for row in segments):
            raise ValueError(f"Invalid annotation spans: {path}")
        if episode["checkpoint/context"].shape != (n,):
            raise ValueError(f"Missing per-frame checkpoint: {path}")

        def check_array(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.shape[0] != n or not np.isfinite(obj[()]).all():
                    raise ValueError(f"Invalid checkpoint {name}: {path}")

        episode["checkpoint/arrays"].visititems(check_array)
        for raw in episode["checkpoint/context"]:
            json.loads(raw)
        if images:
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                frames = episode[f"images/{camera}"]
                if frames.shape != (n, 480, 640, 3) or frames.dtype != np.uint8:
                    raise ValueError(f"Invalid camera shape/type {camera}: {path}")
                for index in {0, n // 2, n - 1}:
                    if frames[index].std() < 0.25:
                        raise ValueError(f"Blank camera {camera} frame {index}: {path}")
        foreground = [row for row in segments if not row.get("background") and row["subtask"] > 0]
        frame = foreground[0]["start_frame"] if foreground else n // 2
        return {
            "path": str(path),
            "seed": meta["seed"],
            "frames": n,
            "restore_frame": frame,
            "step_limit": meta["step_lim"],
            "task": meta["task"],
        }


def check_evidence(validation, rgb):
    from src.expert.dev import report_run

    validation, rgb = Path(validation).resolve(), Path(rgb).resolve()
    runs = [json.loads((path / "run.json").read_text()) for path in (validation, rgb)]
    task = runs[0]["task"]
    protocol = load_protocol(task)
    fingerprint = source_fingerprint(task)["sha256"]
    if official_fingerprint(task) != protocol["official_files"]:
        raise ValueError("Official task/config/reward changed")
    runtime = runtime_info()
    for manifest in runs:
        for key in ("versions", "environment"):
            if manifest["runtime"][key] != runtime[key]:
                raise ValueError(f"Runtime {key} differs from collection; use the recorded environment")
    checked_runs = []
    for path, manifest, split in zip((validation, rgb), runs, ("validation", "rgb"), strict=True):
        if manifest["task"] != task or manifest["split"] != split or manifest["seeds"] != protocol["seeds"][split]:
            raise ValueError(f"Wrong task or seed split: {path}")
        if manifest["source"]["sha256"] != fingerprint or manifest["protocol"] != protocol:
            raise ValueError(f"Source/protocol changed: {path}; rerun the frozen revision")
        result = report_run(path)
        if not result["run_complete"]:
            raise ValueError(f"Incomplete or unhealthy run: {path}")
        if split == "validation" and not result["meets_threshold"]:
            raise ValueError(f"Validation success rate below {protocol['minimum_success_rate']}: {path}")
        for name, expected in manifest["assets"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Asset changed: {name}")
        successful = {row["seed"] for row in read_attempts(path / "raw") if row["result"] == "success"}
        episodes = episode_files(path / "raw")
        checked = [validate_raw(file, images=manifest["images"]) for file in episodes]
        if len(checked) != len(successful) or {row["seed"] for row in checked} != successful:
            raise ValueError(f"Saved episodes differ from successful attempt receipts: {path}")
        if any(row["task"] != task for row in checked):
            raise ValueError(f"Wrong task in saved episode: {path}")
        checked_runs.append(checked)
    if not runs[1]["images"]:
        raise ValueError("RGB run did not enable images")
    raw = checked_runs[1]
    if len(raw) < protocol["minimum_rgb_successes"]:
        raise ValueError("Not enough successful RGB episodes")
    if any(row["task"] != task or row["seed"] not in runs[1]["seeds"] for row in raw):
        raise ValueError("RGB episode task/seed mismatch")
    return task, fingerprint, raw


def qualify(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use --gpu with physical GPU indices; remove CUDA_VISIBLE_DEVICES for Isaac/Vulkan")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "validation_run": str(args.validation.resolve()),
        "rgb_run": str(args.rgb.resolve()),
        "steps": [],
        "limitations": [
            "Physical restore is not bitwise deterministic.",
            "Qualification does not prove collision-free motion or VLA performance.",
        ],
    }
    try:
        task, fingerprint, raw = check_evidence(args.validation, args.rgb)
        report.update(task=task, source_sha256=fingerprint, raw_validation=raw)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)}
        dataset = args.out / "lerobot"

        def execute(name, command, lock_fd=None):
            result = supervise(command, args.out / f"{name}.log", timeout=args.timeout, env=env, lock_fd=lock_fd)
            report["steps"].append({"name": name, **result})
            write_json(args.out / "qualification.json", report)
            if result["timed_out"] or result["exit_code"] != 0:
                raise RuntimeError(f"{name} failed; inspect {args.out / (name + '.log')}")

        execute(
            "export",
            [
                "bash",
                str(ROOT / "scripts/generate_data.sh"),
                "export",
                "--input",
                str(args.rgb.resolve() / "raw"),
                "--output",
                str(dataset),
            ],
        )
        execute(
            "dataset_validation",
            [
                "bash",
                str(ROOT / "scripts/run-lerobot.sh"),
                str(ROOT / "scripts/internal/validate_expert_dataset.py"),
                "--dataset",
                str(dataset),
                "--report",
                str(args.out / "dataset_validation.json"),
            ],
        )
        first = raw[0]
        with gpu_lock(args.gpu, args.timeout) as lock_fd:
            execute(
                "restore",
                [
                    sys.executable,
                    "-m",
                    "src.expert",
                    "restore",
                    "--dataset",
                    str(dataset),
                    "--episode-index",
                    "0",
                    "--frame",
                    str(first["restore_frame"]),
                    "--steps",
                    str(first["step_limit"]),
                    "--gpu",
                    str(args.gpu),
                    "--report",
                    str(args.out / "restore.json"),
                    "--video",
                    str(args.out / "restore.mp4"),
                ],
                lock_fd,
            )
        restore = json.loads((args.out / "restore.json").read_text())
        if not restore["task_success"] or any(value > 1e-4 for value in restore["restore_errors"].values()):
            raise ValueError("Restored rollout did not complete the official task or reset error exceeded tolerance")
        if source_fingerprint(task)["sha256"] != fingerprint:
            raise ValueError("Source changed during qualification")
        report.update(passed=True, dataset=str(dataset), restore=restore)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write_json(args.out / "qualification.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1
