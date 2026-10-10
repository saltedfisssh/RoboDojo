"""Persistent CPU exporter: validate each atomic HDF5 save before deleting it."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import traceback

from src.expert.arguments import positive_int
from src.expert.diagnostics import write_json
from src.expert.export import export
from src.expert.storage import CONVERTED_SCHEMA, converted_metadata, episode_files
from src.expert.training import REPO_ID, validate_training


def convert_episode(raw, converted, path):
    import h5py

    started = time.monotonic()
    seed = int(path.stem.split("_")[-1])
    target = converted / f"episode_{seed:07d}"
    receipt = raw / "converted" / f"episode_{seed:07d}.json"
    if receipt.exists():
        meta = converted_metadata(raw, seed)
        if (raw.parent / meta["dataset"]).resolve() != target.resolve():
            raise ValueError(f"Conversion receipt points to a different dataset: {receipt}")
        # Recover a crash after receipt publication but before HDF5 deletion.
        validate_training(target, source=path, expected_episodes=1)
    else:
        # Without a receipt these are unpublished artifacts from an interrupted
        # conversion; its HDF5 is still available for rebuilding them.
        for disposable in (target, target.with_name(target.name + ".partial")):
            if disposable.exists():
                shutil.rmtree(disposable)
        export([path], target, REPO_ID, training_only=True)
        report = validate_training(target, source=path, expected_episodes=1)
        with h5py.File(path, "r") as episode:
            meta = json.loads(episode.attrs["meta"])
        meta.update(schema=CONVERTED_SCHEMA, dataset=str(target.relative_to(raw.parent)), validation=report)
        write_json(receipt, meta)
    # The v3 shard and its durable receipt exist before the raw file disappears.
    path.unlink()
    print(json.dumps(dict(event="converted", seed=seed, seconds=round(time.monotonic() - started, 3))), flush=True)
    return meta


def watch(raw, converted, done, ready, *, workers=1):
    # Import once before starting Isaac. Each conversion uses its own dataset
    # writer and paths; no workers concurrently append to shared v3 files.
    from lerobot.datasets.aggregate import aggregate_datasets  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
    import torch

    torch.set_num_threads(2)
    converted.mkdir(parents=True, exist_ok=True)
    (raw / "converted").mkdir(exist_ok=True)
    ready.write_text("ready\n")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while True:
            for path, future in list(active.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception as exc:
                        write_json(raw.parent / "conversion.error", dict(file=str(path), error=str(exc)))
                        raise
                    del active[path]
            files = episode_files(raw)  # excludes the recorder's .partial.h5
            for path in files:
                if path not in active and len(active) < workers:
                    active[path] = pool.submit(convert_episode, raw, converted, path)
            if done.exists() and not files and not active:
                return
            time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--done", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--workers", type=positive_int, default=1)
    args = parser.parse_args(argv)
    try:
        watch(args.raw, args.converted, args.done, args.ready, workers=args.workers)
    except Exception as exc:
        error = args.raw.parent / "conversion.error"
        if not error.exists():
            write_json(error, dict(error=f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
