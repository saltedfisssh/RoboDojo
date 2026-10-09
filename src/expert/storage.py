"""Collection bookkeeping shared by serial collection and batch workers."""

import json
from pathlib import Path


def episode_files(out):
    if out is None:
        return []
    return sorted(path for path in Path(out).glob("episode_*.h5") if not path.name.endswith(".partial.h5"))


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
    return {int(path.stem.split("_")[-1]) for path in episode_files(out)} | {
        int(row["seed"]) for row in read_attempts(out)
    }


def prepare_output(out, *, task, planner, images, table_randomization, sampler=None):
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
    manifest.write_text(json.dumps(expected, indent=2) + "\n")


def log_attempt(out, worker, row):
    payload = json.dumps(row)
    print("[expert]", payload, flush=True)
    if out is not None:
        with (Path(out) / f"attempts_w{worker}.jsonl").open("a") as log:
            log.write(payload + "\n")
