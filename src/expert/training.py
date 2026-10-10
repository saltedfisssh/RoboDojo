"""Resumable training-only collection, with export in the isolated LeRobot venv."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from src.expert.diagnostics import write_json
from src.expert.scheduling import batch_lock, reserved_seeds
from src.expert.storage import (
    converted_files,
    converted_metadata,
    episode_count,
    episode_files,
    episode_metadata,
    prepare_output,
    read_attempts,
    success_seeds,
)

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "local/robodojo_expert"
WORK_PATHS = (
    "raw",
    "converted",
    "export",
    "export.partial",
    "merged",
    "previous",
    "conversion.done",
    "conversion.ready",
    "conversion.error",
)


def start_process(command, lock_fd, **kwargs):
    return subprocess.Popen(command, pass_fds=(lock_fd,), start_new_session=True, **kwargs)


def stop_process(process):
    # Children share this process's dedicated group. Stop orphan Isaac workers
    # as well when a coordinator exits before cleaning up its own children.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_stage(command, lock_fd):
    """Keep the pipeline locked until its subprocess exits, including on interruption."""
    process = start_process(command, lock_fd)
    try:
        return process.wait()
    finally:
        stop_process(process)


def clean_work(work):
    for name in WORK_PATHS:
        path = work / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def run_lerobot(lock_fd, *arguments):
    return run_stage(["bash", str(ROOT / "scripts/run-lerobot.sh"), *arguments], lock_fd)


def converted_datasets(raw):
    roots = []
    for path in converted_files(raw):
        seed = int(path.stem.split("_")[-1])
        meta = converted_metadata(raw, seed)
        dataset = raw.parent / meta["dataset"]
        if dataset.resolve() != (raw.parent / "converted" / f"episode_{seed:07d}").resolve():
            raise ValueError(f"Unexpected dataset path in {path}")
        info = json.loads((dataset / "meta/info.json").read_text())
        if info["total_episodes"] != 1 or info["total_frames"] != meta["num_frames"]:
            raise ValueError(f"Converted dataset differs from {path}")
        roots.append(dataset)
    return roots


def merge_candidate(raw, out, work, have, lock_fd):
    """Merge validated v3 shards; no HDF5 is needed for final publication."""
    roots = ([out] if have else []) + converted_datasets(raw)
    candidate = work / "merged"
    if candidate.exists():
        shutil.rmtree(candidate)
    code = run_lerobot(lock_fd, "-m", "src.expert.training", "--merge", *map(str, roots), "--output", str(candidate))
    if not code:
        validation = [
            "-m",
            "src.expert.training",
            "--validate",
            str(candidate),
            "--datasets",
            *map(str, roots),
            "--expected-episodes",
            str(have + len(converted_files(raw))),
        ]
        code = run_lerobot(lock_fd, *validation)
    return candidate, code


def collection_attempts(raw, task):
    """Preserve seed history, including assignments and saves without a worker receipt."""
    attempts = {row["seed"]: row for row in read_attempts(raw)}
    for seed in reserved_seeds(raw) - attempts.keys():
        attempts[seed] = dict(seed=seed, task=task, result="interrupted")
    # A serial worker can exit after atomic save but before logging. Recover its
    # actual frame count as well: batch summaries need steps on every success.
    for seed in success_seeds(raw):
        meta = episode_metadata(raw, seed)
        row = attempts.setdefault(seed, dict(seed=seed, task=task))
        row.update(result="success", steps=int(meta["steps"]))
    return [
        {key: value for key, value in row.items() if key not in ("file", "failure_file", "diagnostics_file")}
        for _, row in sorted(attempts.items())
    ]


def collect_and_convert(child, raw, work, workers, lock_fd):
    """Overlap Isaac collection with a persistent, isolated CPU converter."""
    done, ready, error = (work / name for name in ("conversion.done", "conversion.ready", "conversion.error"))
    for path in (done, ready, error):
        path.unlink(missing_ok=True)
    command = [
        "bash",
        str(ROOT / "scripts/run-lerobot.sh"),
        "-m",
        "src.expert.conversion",
        "--raw",
        str(raw),
        "--converted",
        str(work / "converted"),
        "--done",
        str(done),
        "--ready",
        str(ready),
        "--workers",
        str(workers),
    ]
    collector = None
    with (work / "conversion.log").open("a") as log:
        converter = start_process(
            command,
            lock_fd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
        )
        try:
            while not ready.exists():
                if converter.poll() is not None:
                    return converter.returncode or 1
                time.sleep(0.1)
            code = 0
            if child is not None:
                collector = start_process(child, lock_fd)
                while collector.poll() is None:
                    if converter.poll() is not None or error.exists():
                        stop_process(collector)
                        # An error can be reported while another conversion
                        # thread is still running. Do not wait indefinitely
                        # for thread-pool shutdown before stopping the pipeline.
                        if converter.poll() is None:
                            stop_process(converter)
                            return 1
                        return converter.returncode or 1
                    time.sleep(0.1)
                code = collector.returncode
                stop_process(collector)
            done.write_text("done\n")
            # Continue supervising errors while draining the queue, including
            # resume runs with no collector. A failing thread must not leave
            # the parent blocked on other conversion threads during shutdown.
            while converter.poll() is None:
                if error.exists():
                    stop_process(converter)
                    return 1
                time.sleep(0.1)
            return converter.returncode or code
        finally:
            if collector is not None:
                stop_process(collector)
            stop_process(converter)
            if error.exists():
                print(f"[expert] Background conversion failed: {error.read_text().strip()}", flush=True)
            if error.exists() or converter.returncode:
                print(f"[expert] Retained pending data; inspect {work / 'conversion.log'} and rerun", flush=True)


def collect_training(command, args, argv, spec):
    if args.out is None or args.no_images:
        raise ValueError("--format lerobot-v3 requires --out and RGB; remove --no-images")
    if any(getattr(args, flag, False) for flag in ("diagnostics", "save_failures", "additional", "scheduler")):
        raise ValueError("Training-only output cannot include diagnostics, failed trajectories or worker flags")
    out = args.out.resolve()
    work = out.with_name(f".{out.name}.collection")
    settings = dict(
        task=args.task,
        planner=spec.entrypoint,
        sampler=spec.sampler,
        table_randomization=args.table_randomization,
        format="lerobot-v3",
    )
    with batch_lock(work) as lock_fd:
        marker = work / "pipeline.json"
        state = json.loads(marker.read_text()) if marker.exists() else None
        if state is not None and state["settings"] != settings:
            raise ValueError(f"Collection settings differ from {marker}; use a separate output directory")
        if state is None and any(path.name != ".batch.lock" for path in work.iterdir()):
            raise ValueError(f"Unrecognized staging directory: {work}")
        # Recover an interruption between the two directory renames below.
        previous = work / "previous"
        if previous.exists() and not out.exists():
            previous.rename(out)
        base = None
        if out.exists():
            manifest = out / "meta/collection.json"
            if not manifest.is_file():
                raise ValueError(f"{out} is not a training-only collection; use a separate output directory")
            base = json.loads(manifest.read_text())
            if base["settings"] != settings:
                raise ValueError(f"Collection settings differ from {manifest}; use a separate output directory")
            info = json.loads((out / "meta/info.json").read_text())
            if info["total_episodes"] != len(base["seeds"]):
                raise ValueError(f"Episode count differs from {manifest}")
            if state is not None and base["run_id"] == state["run_id"]:
                clean_work(work)  # publication succeeded before cleanup was interrupted
                state = None
        have = len(base["seeds"]) if base else 0
        if have >= args.num:
            print(json.dumps(dict(root=str(out), episodes=have, target_reached=True, new_successes=0)), flush=True)
            return 0
        if state is None:
            state = dict(settings=settings, run_id=uuid.uuid4().hex, base_run_id=base["run_id"] if base else None)
            write_json(marker, state)
        if state["base_run_id"] != (base["run_id"] if base else None):
            raise ValueError("Published dataset changed while collection was pending; use a separate output directory")
        if base and getattr(args, "seeds", None) is not None:
            raise ValueError("Fixed --seeds require a fresh output directory")

        raw = work / "raw"
        raw.mkdir(exist_ok=True)
        prepare_output(
            raw,
            task=args.task,
            planner=spec.entrypoint,
            sampler=spec.sampler,
            images=True,
            table_randomization=args.table_randomization,
            checkpoints=False,
        )
        converted_datasets(raw)  # reject missing/damaged committed shards before starting Isaac
        if base and not (raw / "attempts_wprior.jsonl").exists():
            prior = raw / "attempts_wprior.jsonl"
            temporary = prior.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in base["attempts"]))
            temporary.replace(prior)
        missing = args.num - have
        child = None
        if episode_count(raw) < missing:
            child = [
                sys.executable,
                "-m",
                "src.expert",
                command,
                *argv,
                "--format",
                "raw",
                "--no-checkpoints",
                "--out",
                str(raw),
                "--num",
                str(missing),
                "--background-export",
                "--max-pending-episodes",
                str(args.max_pending_episodes or max(2, 2 * getattr(args, "workers", 1))),
            ]
            if command == "batch":
                child += ["--pipeline-lock-fd", str(lock_fd)]
        code = collect_and_convert(child, raw, work, args.export_workers, lock_fd)
        if episode_files(raw):
            return code or 1  # conversion failed; its HDF5 must be retained
        if episode_count(raw) < missing:
            print(f"[expert] Training target unmet; rerun the same command to resume {work}", flush=True)
            return code or 1
        candidate, stage_code = merge_candidate(raw, out, work, have, lock_fd)
        if stage_code:
            return stage_code
        seeds = (base["seeds"] if base else []) + sorted(success_seeds(raw))
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seeds across published and pending episodes")
        write_json(
            candidate / "meta/collection.json",
            dict(settings=settings, run_id=state["run_id"], seeds=seeds, attempts=collection_attempts(raw, args.task)),
        )
        if base:
            out.rename(previous)
        candidate.rename(out)
        clean_work(work)
        print(
            json.dumps(dict(root=str(out), episodes=len(seeds), target_reached=True, new_successes=len(seeds) - have)),
            flush=True,
        )
        return code


def validate_training(root, *, source=None, baseline=None, datasets=None, expected_episodes=None):
    """Read standard v3 data, compare every action/state and decode all RGB frames."""
    import av
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.expert.export import CAMERAS, FPS

    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == FPS
    assert not list(root.rglob("*.h5")) and not (root / "states").exists()
    assert not (root / "meta/replay.jsonl").exists() and not (root / "meta/argus").exists()

    def table_at(path):
        return pa.concat_tables([pq.read_table(file) for file in sorted((path / "data").rglob("*.parquet"))]).sort_by(
            [("index", "ascending")]
        )

    table = table_at(root)
    dataset = LeRobotDataset(REPO_ID, root=root, video_backend="pyav")
    assert len(dataset) == len(table) == info["total_frames"] > 0
    assert dataset.num_episodes == info["total_episodes"]
    if expected_episodes is not None:
        assert dataset.num_episodes == expected_episodes
    np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(len(table)))
    arrays = {key: np.asarray(table[key].to_pylist(), np.float32) for key in ("action", "observation.state")}
    for array in arrays.values():
        assert array.shape == (len(table), 14) and np.isfinite(array).all()
    if source is not None or datasets is not None:
        expected = {key: [] for key in arrays}
        for dataset_root in datasets or []:
            data = table_at(Path(dataset_root))
            for key in expected:
                expected[key].extend(data[key].to_pylist())
        if baseline is not None:
            old = table_at(Path(baseline))
            for key in expected:
                expected[key].extend(old[key].to_pylist())
        source_files = [Path(source)] if source is not None and Path(source).is_file() else episode_files(source)
        for file in source_files:
            with h5py.File(file, "r") as episode:
                meta = json.loads(episode.attrs["meta"])
                assert meta["success"] and 0 < meta["num_frames"] == meta["steps"] <= meta["step_lim"]
                for key, raw_key in (("action", "action"), ("observation.state", "state")):
                    expected[key].extend(episode[raw_key][()].tolist())
        for key in expected:
            np.testing.assert_allclose(arrays[key], np.asarray(expected[key], np.float32), atol=1e-6, rtol=0)
    episodes = table["episode_index"].to_numpy()
    np.testing.assert_array_equal(np.unique(episodes), np.arange(dataset.num_episodes))
    samples = 0
    for episode_index in range(dataset.num_episodes):
        indices = np.flatnonzero(episodes == episode_index)
        np.testing.assert_array_equal(table["frame_index"].to_numpy()[indices], np.arange(len(indices)))
        np.testing.assert_allclose(table["timestamp"].to_numpy()[indices], np.arange(len(indices)) / FPS, atol=1e-5)
        for index in {int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])}:
            frame = dataset[index]
            for key in arrays:
                np.testing.assert_allclose(frame[key].numpy(), arrays[key][index], atol=1e-6)
            for camera in CAMERAS:
                feature = f"observation.images.{camera}"
                image = frame[feature].numpy()
                assert tuple(image.shape) == tuple(info["features"][feature]["shape"])
                assert np.isfinite(image).all()
            samples += 1
    decoded = {f"observation.images.{camera}": 0 for camera in CAMERAS}
    for video in sorted((root / "videos").rglob("*.mp4")):
        camera = video.relative_to(root / "videos").parts[0]
        channels, height, width = info["features"][camera]["shape"]
        assert channels == 3
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == (width, height)
                decoded[camera] += 1
    assert all(count == len(table) for count in decoded.values()), decoded
    return dict(
        dataset=str(root),
        episodes=dataset.num_episodes,
        frames=len(table),
        lerobot_samples_checked=samples,
        decoded_video_frames=sum(decoded.values()),
        training_only=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--validate", type=Path)
    mode.add_argument("--merge", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--datasets", type=Path, nargs="+")
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args(argv)
    if args.check:
        import av  # noqa: F401
        import h5py  # noqa: F401
        from lerobot.datasets.aggregate import aggregate_datasets  # noqa: F401
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

    elif args.merge:
        from lerobot.datasets.aggregate import aggregate_datasets

        if args.output is None:
            parser.error("--merge requires --output")
        aggregate_datasets(
            repo_ids=[REPO_ID] * len(args.merge),
            aggr_repo_id=REPO_ID,
            roots=args.merge,
            aggr_root=args.output,
            concatenate_videos=False,
        )
    else:
        print(
            json.dumps(
                validate_training(
                    args.validate,
                    source=args.source,
                    baseline=args.baseline,
                    datasets=args.datasets,
                    expected_episodes=args.expected_episodes,
                )
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
