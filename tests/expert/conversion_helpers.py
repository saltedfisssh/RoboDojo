"""Synthetic episodes and conversion receipts shared by pipeline tests."""

import json

import h5py
import numpy as np

from src.expert.diagnostics import write_json
from src.expert.storage import CONVERTED_SCHEMA


def raw_episode(raw, seed, frames=4):
    path = raw / f"episode_{seed:07d}.h5"
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as episode:
        episode.attrs["meta"] = json.dumps(
            dict(
                task="stack_blocks",
                seed=seed,
                success=True,
                num_frames=frames,
                steps=frames,
                step_lim=550,
                instruction="stack the blocks",
            )
        )
        episode.create_dataset("state", data=rng.normal(size=(frames, 14)).astype(np.float32))
        episode.create_dataset("action", data=rng.normal(size=(frames, 14)).astype(np.float32))
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            episode.create_dataset(f"images/{camera}", data=rng.integers(0, 256, (frames, 48, 64, 3), np.uint8))
    return path


def fake_conversion(raw, path):
    with h5py.File(path, "r") as episode:
        meta = json.loads(episode.attrs["meta"])
    target = raw.parent / "converted" / path.stem
    write_json(target / "meta/info.json", dict(total_episodes=1, total_frames=meta["num_frames"]))
    meta.update(schema=CONVERTED_SCHEMA, dataset=str(target.relative_to(raw.parent)))
    write_json(raw / "converted" / f"{path.stem}.json", meta)
    path.unlink()
