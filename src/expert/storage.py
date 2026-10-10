"""Collection bookkeeping shared by serial collection and batch workers."""

import json
from pathlib import Path

CONVERTED_SCHEMA = "robodojo.expert-converted.v1"


def episode_files(out):
    if out is None:
        return []
    return sorted(path for path in Path(out).glob("episode_*.h5") if not path.name.endswith(".partial.h5"))


def converted_files(out):
    if out is None:
        return []
    return sorted((Path(out) / "converted").glob("episode_*.json"))


def success_seeds(out):
    # Scan raw first: conversion commits its receipt before unlinking HDF5, so
    # this order cannot lose a success during the handoff between formats.
    raw = episode_files(out)
    converted = converted_files(out)
    return {int(path.stem.split("_")[-1]) for path in (*raw, *converted)}


def episode_count(out):
    return len(success_seeds(out))


def episode_exists(out, seed):
    out = Path(out)
    return (out / f"episode_{seed:07d}.h5").is_file() or (out / "converted" / f"episode_{seed:07d}.json").is_file()


def converted_metadata(out, seed):
    path = Path(out) / "converted" / f"episode_{seed:07d}.json"
    meta = json.loads(path.read_text())
    if meta.get("schema") != CONVERTED_SCHEMA or meta.get("seed") != seed or not meta.get("success"):
        raise ValueError(f"Invalid conversion receipt: {path}")
    return meta


def episode_metadata(out, seed):
    import h5py

    try:
        with h5py.File(Path(out) / f"episode_{seed:07d}.h5", "r") as episode:
            return json.loads(episode.attrs["meta"])
    except FileNotFoundError:
        return converted_metadata(out, seed)


def read_attempts(out):
    if out is None:
        return []
    rows = []
    for path in sorted(Path(out).glob("attempts_w*.jsonl")):
        for index, line in enumerate(path.read_text().splitlines(), start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid attempt log {path}:{index}; repair the interrupted line before resuming"
                    ) from exc
    return rows


def attempted_seeds(out):
    return success_seeds(out) | {int(row["seed"]) for row in read_attempts(out)}


def prepare_output(out, *, task, planner, images, table_randomization, sampler=None, checkpoints=True):
    """Reject accidental mixing of tasks/planners in a resumed collection.

    Batch calls this before starting children. Existing v1 directories without
    a manifest are checked against episode metadata and then adopted.
    """
    if out is None:
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "collection.json"
    expected = dict(task=task, planner=planner, images=images, table_randomization=table_randomization)
    if not checkpoints:
        expected["checkpoints"] = False
    if sampler is not None:
        expected["sampler"] = sampler
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f"Collection settings differ from {manifest}; use a separate output directory")
        return
    for row in read_attempts(out):
        if row["task"] != task:
            raise ValueError(f"{out} already contains attempts for {row['task']}")
    files = episode_files(out)
    if files:
        import h5py

        for path in files:
            with h5py.File(path, "r") as episode:
                if json.loads(episode.attrs["meta"])["task"] != task or ("images" in episode) != images:
                    raise ValueError(f"Task/image settings differ from {path}; use a separate output directory")
                if ("checkpoint" in episode) != checkpoints:
                    raise ValueError(f"Checkpoint settings differ from {path}; use a separate output directory")
    manifest.write_text(json.dumps(expected, indent=2) + "\n")


def log_attempt(out, worker, row):
    payload = json.dumps(row)
    print("[expert]", payload, flush=True)
    if out is not None:
        with (Path(out) / f"attempts_w{worker}.jsonl").open("a") as log:
            log.write(payload + "\n")
