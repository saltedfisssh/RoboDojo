"""Reject false successes and stale evidence before expensive export/restore."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.expert import qualification, recorder
from src.expert.diagnostics import write_json


def recorded_episode(tmp_path, monkeypatch, *, outcome="success", end=True):
    env = SimpleNamespace(success=[True], end_flag=[end], step_lim=400, take_action_cnt=[2])
    monkeypatch.setattr(recorder, "_config", lambda env: {})
    writer = recorder.Recorder(env, "stack_blocks_by_language", 200, {}, images=False, checkpoints=False)
    writer.instruction = "stack in order"
    writer.segments = [{"start_frame": 0, "end_frame": 2, "subtask": 0}]
    writer.subtasks = []
    writer.rows = [
        {
            "state": np.zeros(14),
            "action": np.zeros(14),
            "ee_pose": np.zeros(14),
            "segment": 0,
            "held": ["", ""],
            "objects": {},
        }
        for _ in range(2)
    ]
    path = writer.save(tmp_path / "episode.h5", outcome=outcome)
    with h5py.File(path, "a") as episode:
        group = episode.create_group("checkpoint")
        group.create_group("arrays").create_dataset("command", data=np.zeros((2, 14)))
        group.create_dataset("context", data=np.array(["{}", "{}"], dtype=h5py.string_dtype()))
    return path, writer


@pytest.mark.parametrize("outcome,end", [("fail", True), ("success", False)])
def test_failure_never_becomes_successful_training_episode(tmp_path, monkeypatch, outcome, end):
    path, writer = recorded_episode(tmp_path, monkeypatch, outcome=outcome, end=end)
    with h5py.File(path) as episode:
        assert not json.loads(episode.attrs["meta"])["success"]
    with pytest.raises(ValueError, match="Not a successful"):
        qualification.validate_raw(path, images=False)
    with pytest.raises(FileExistsError):
        writer.save(path)


def test_raw_validation_checks_actual_arrays_and_required_images(tmp_path, monkeypatch):
    path, _ = recorded_episode(tmp_path, monkeypatch)
    assert qualification.validate_raw(path, images=False)["frames"] == 2
    with pytest.raises(KeyError):
        qualification.validate_raw(path, images=True)
    with h5py.File(path, "a") as episode:
        episode["action"][1, 5] = np.nan
    with pytest.raises(ValueError, match="Invalid action"):
        qualification.validate_raw(path, images=False)


@pytest.mark.parametrize("fault", ["seed", "source", "runtime"])
def test_qualification_rejects_wrong_split_stale_source_and_runtime(tmp_path, monkeypatch, fault):
    protocol = {"official_files": {}, "seeds": {"validation": [100], "rgb": [200]}}
    runtime = {"versions": {"torch": "test"}, "environment": {}}
    monkeypatch.setattr(qualification, "load_protocol", lambda task: protocol)
    monkeypatch.setattr(qualification, "official_fingerprint", lambda task: {})
    monkeypatch.setattr(qualification, "source_fingerprint", lambda task: {"sha256": "frozen"})
    monkeypatch.setattr(qualification, "runtime_info", lambda: runtime)
    for split in ("validation", "rgb"):
        manifest = {
            "task": "stack_blocks_by_language",
            "split": split,
            "seeds": protocol["seeds"][split],
            "source": {"sha256": "frozen"},
            "protocol": protocol,
            "runtime": runtime,
        }
        if split == "validation":
            if fault == "seed":
                manifest["seeds"] = [0]
            elif fault == "source":
                manifest["source"] = {"sha256": "stale"}
            else:
                manifest["runtime"] = {"versions": {"torch": "changed"}, "environment": {}}
        write_json(tmp_path / split / "run.json", manifest)
    with pytest.raises(ValueError, match="Wrong task or seed split|Source/protocol changed|Runtime versions"):
        qualification.check_evidence(tmp_path / "validation", tmp_path / "rgb")
