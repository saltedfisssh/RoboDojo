import json
from threading import Event

import h5py
import numpy as np
import pytest

from src.datagen.recording import AsyncEpisodeWriter, EpisodeWriter


def test_queued_frames_own_mutable_buffers_and_publish_only_when_complete(tmp_path, monkeypatch):
    started, release = Event(), Event()
    append = EpisodeWriter.append

    def delayed(self, *args):
        started.set()
        assert release.wait(5), "Test writer was never released"
        append(self, *args)

    monkeypatch.setattr(EpisodeWriter, "append", delayed)
    path = tmp_path / "episode.h5"
    writer = AsyncEpisodeWriter(path, {"fps": 25})
    value = np.ones(14, np.float32)
    pixels = np.ones((8, 8, 3), np.uint8)
    context = {"reward": [1]}
    annotation = {"segment_id": 0, "goal": {"position": [1]}}
    try:
        writer.append(value, value, value, {"cam_high": pixels}, ({"state": value}, context), annotation)
        assert started.wait(5)
        value[:] = 2
        pixels[:] = 2
        context["reward"][0] = 2
        annotation["goal"]["position"][0] = 2
        writer.append(value, value, value, {"cam_high": pixels}, ({"state": value}, context), annotation)
        assert writer.count == 2
        assert not path.exists()
        release.set()
        writer.finish(True, terminal_snapshot=({"state": value}, context))
    finally:
        release.set()
        writer.abort()
    with h5py.File(path) as file:
        assert file.attrs["complete"] and file.attrs["success"]
        for i in range(2):
            for key in (
                "observation/state",
                "observation/measured_state",
                "action",
                "snapshots/state",
                "images/cam_high",
            ):
                np.testing.assert_array_equal(file[key][i], np.full_like(file[key][i], i + 1))
            assert json.loads(file["context"][i])["reward"] == [i + 1]
            assert json.loads(file["annotations"][i])["goal"]["position"] == [i + 1]


def test_background_failure_drains_queue_and_leaves_unpublished_partial(tmp_path):
    path = tmp_path / "episode.h5"
    writer = AsyncEpisodeWriter(path, {"fps": 25})
    state = np.zeros(14, np.float32)
    writer.append(state, state, state, {}, ({"state": state}, {}), {"invalid": float("nan")})
    writer.queue.join()
    with pytest.raises(RuntimeError, match="Background HDF5 write failed"):
        writer.finish(True)
    assert not writer.worker.is_alive()
    assert not path.exists()
    with h5py.File(path.with_suffix(".partial.h5")) as file:
        assert not file.attrs["complete"]
