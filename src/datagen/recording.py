"""Streaming, crash-safe raw episode storage. State/frame i is BEFORE action i."""

from copy import deepcopy
import json
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from time import perf_counter

import h5py
import numpy as np

from src.datagen.state import plain


class EpisodeWriter:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(".partial.h5")
        self.file = h5py.File(self.partial, "x")
        self.file.attrs["metadata"] = json.dumps(plain(metadata), allow_nan=False)
        self.file.attrs["complete"] = False
        self.file.attrs["image_compression"] = "lzf"
        self.count = 0

    def _append(self, name, value):
        value = np.asarray(value)
        if name not in self.file:
            compression = (
                {"compression": "lzf"}
                if name.startswith("images/")
                else {
                    "compression": "gzip",
                    "compression_opts": 1,
                }
            )
            self.file.create_dataset(
                name,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                chunks=(1, *value.shape),
                dtype=value.dtype,
                **compression,
            )
        ds = self.file[name]
        ds.resize(self.count + 1, axis=0)
        ds[self.count] = value

    def _text(self, name, value):
        if name not in self.file:
            self.file.create_dataset(name, (0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True)
        ds = self.file[name]
        ds.resize(self.count + 1, axis=0)
        ds[self.count] = json.dumps(plain(value), allow_nan=False)

    def append(self, state, measured, action, images, snapshot, annotation):
        self._append("observation/state", state)
        self._append("observation/measured_state", measured)
        self._append("action", action)
        for name, image in images.items():
            self._append(f"images/{name}", image)
        arrays, context = snapshot
        for name, value in arrays.items():
            self._append(f"snapshots/{name}", value)
        self._text("context", context)
        self._text("annotations", annotation)
        self.count += 1
        if self.count % 25 == 0:
            self.file.flush()

    def finish(self, success, reason=None, terminal_snapshot=None):
        lengths = []
        self.file.visititems(
            lambda name, obj: lengths.append((name, len(obj))) if isinstance(obj, h5py.Dataset) else None
        )
        if any(n != self.count for _, n in lengths):
            self.abort()
            raise RuntimeError("Partially written frame; left as .partial.h5 for diagnosis")
        if terminal_snapshot is not None:
            arrays, context = terminal_snapshot
            for name, value in arrays.items():
                self.file.create_dataset(f"terminal/{name}", data=value)
            self.file.attrs["terminal_context"] = json.dumps(plain(context), allow_nan=False)
        self.file.attrs["success"] = bool(success)
        self.file.attrs["reason"] = reason or ""
        self.file.attrs["num_frames"] = self.count
        budget = json.loads(self.file.attrs["metadata"]).get("benchmark_action_budget")
        if budget is not None:
            self.file.attrs["within_benchmark_budget"] = self.count <= budget
        self.file.attrs["complete"] = True
        self.file.flush()
        self.file.close()
        self.partial.rename(self.path)

    def abort(self):
        self.file.close()


class AsyncEpisodeWriter:
    """Bounded asynchronous compression; copy mutable simulator buffers before enqueueing."""

    def __init__(self, path, metadata, queue_size=2):
        if queue_size < 1:
            raise ValueError("Queue size must be positive")
        self.writer = EpisodeWriter(path, {**metadata, "recording_mode": "async_hdf5"})
        self.count = 0
        self.queue = Queue(maxsize=queue_size)
        self.error = None
        self.cancelled = False
        self.closed = False
        self.write_seconds = 0.0
        self.worker = Thread(target=self._run, name="datagen-hdf5", daemon=True)
        self.worker.start()

    @property
    def file(self):
        return self.writer.file

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None and not self.cancelled:
                    stamp = perf_counter()
                    self.writer.append(*item)
                    self.write_seconds += perf_counter() - stamp
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def _check(self):
        if self.error is not None:
            raise RuntimeError("Background HDF5 write failed") from self.error
        if self.closed:
            raise RuntimeError("Episode writer is already closed")

    def append(self, state, measured, action, images, snapshot, annotation):
        self._check()
        arrays, context = snapshot
        # Isaac getters can return views into the next physics frame. The queue
        # owns every numeric buffer and annotation, including mutable goal fields.
        frame = (
            np.array(state, copy=True),
            np.array(measured, copy=True),
            np.array(action, copy=True),
            {key: np.array(value, copy=True) for key, value in images.items()},
            ({key: np.array(value, copy=True) for key, value in arrays.items()}, deepcopy(context)),
            deepcopy(annotation),
        )
        while True:
            self._check()
            try:
                self.queue.put(frame, timeout=0.1)
                self.count += 1
                return
            except Full:
                continue

    def _stop(self):
        if self.closed:
            return
        # The worker drains queued items after an error, so join cannot hang on
        # an unacknowledged frame or a producer waiting on a full queue.
        self.queue.put(None)
        self.queue.join()
        self.worker.join()
        self.closed = True

    def finish(self, success, reason=None, terminal_snapshot=None):
        self._stop()
        if self.error is not None:
            self.writer.abort()
            raise RuntimeError("Background HDF5 write failed; episode remains partial") from self.error
        if self.writer.count != self.count:
            self.writer.abort()
            raise RuntimeError("Asynchronous frame count mismatch")
        self.file.attrs["background_write_seconds"] = self.write_seconds
        self.writer.finish(success, reason, terminal_snapshot)

    def abort(self):
        self.cancelled = True
        self._stop()
        if self.file.id.valid:
            self.writer.abort()


def read_snapshot(file, frame):
    if not file.attrs.get("complete", False):
        raise ValueError("Cannot restore an incomplete episode")
    if not 0 <= frame < int(file.attrs["num_frames"]):
        raise IndexError(f"Frame {frame} is outside the episode")
    arrays = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            arrays[name] = obj[frame]

    file["snapshots"].visititems(visit)
    return arrays, json.loads(file["context"][frame])


def select_frame(file, *, frame=None, time_s=None, subtask=None):
    """Time selects the frame at or immediately BEFORE it; never silently clamp."""
    if sum(x is not None for x in (frame, time_s, subtask)) != 1:
        raise ValueError("Select exactly one of frame, time_s, subtask")
    count = int(file.attrs["num_frames"])
    if time_s is not None:
        fps = json.loads(file.attrs["metadata"])["fps"]
        if not np.isfinite(time_s) or time_s < 0 or time_s >= count / fps:
            raise ValueError("Timestamp is outside the episode")
        frame = int(np.floor(time_s * fps + 1e-8))
    if subtask is not None:
        frame = next((i for i, a in enumerate(file["annotations"]) if json.loads(a)["segment_id"] == subtask), None)
        if frame is None:
            raise ValueError(f"Subtask segment {subtask} is absent")
    if not 0 <= frame < count:
        raise IndexError(f"Frame {frame} is outside [0, {count})")
    return frame
