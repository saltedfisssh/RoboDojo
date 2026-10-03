import json

import h5py
import numpy as np
import pytest

from src.datagen.annotations import persistent_rows, segments_from_frames
from src.datagen.recording import EpisodeWriter, read_snapshot, select_frame
from src.datagen.replay import resolve_episode
from src.datagen.state import decode_context, encode_context


def write_fixture(path, success=True, with_images=True):
    metadata = {
        "fps": 25,
        "state_timing": "before_action",
        "task": "stack_blocks",
        "seed": 42,
        "instruction": "Stack the three blocks.",
        "layout": {},
    }
    writer = EpisodeWriter(path, metadata)
    for i in range(6):
        state = np.full(14, i, dtype=np.float32)
        images = {"cam_high": np.full((32, 48, 3), i * 30, np.uint8)} if with_images else {}
        writer.append(
            state,
            state,
            state + 1,
            images,
            ({"scene/articulation/robot0/joint_position": state[None]}, {"frame": i}),
            {
                "segment_id": i // 3,
                "primitive": "lift" if i >= 3 else "grasp",
                "text": "Lift the block." if i >= 3 else "Grasp the block.",
                "source": "planner",
            },
        )
    writer.finish(success)


def test_record_restore_alignment_and_boundaries(tmp_path):
    path = tmp_path / "episode.h5"
    write_fixture(path)
    with h5py.File(path) as f:
        assert select_frame(f, time_s=0.119) == 2
        assert select_frame(f, time_s=0.12) == 3
        assert select_frame(f, subtask=1) == 3
        arrays, context = read_snapshot(f, 3)
        assert context["frame"] == 3
        np.testing.assert_array_equal(
            arrays["scene/articulation/robot0/joint_position"], f["observation/state"][3][None]
        )
        np.testing.assert_array_equal(f["action"][3], f["observation/state"][4])
        for value in [-1, 0.24, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                select_frame(f, time_s=value)
        with pytest.raises(IndexError):
            read_snapshot(f, 6)
        with pytest.raises(ValueError):
            select_frame(f, frame=0, time_s=0)


def test_incomplete_not_restorable_or_overwritten(tmp_path):
    path = tmp_path / "episode.h5"
    writer = EpisodeWriter(path, {"fps": 25})
    writer.abort()
    with h5py.File(writer.partial) as f, pytest.raises(ValueError, match="incomplete"):
        read_snapshot(f, 0)
    with pytest.raises(FileExistsError):
        EpisodeWriter(path, {})


def test_reward_tuple_and_array_roundtrip():
    # Lists mean OR/AND groups; tuples mean callable predicates in RewardManager.
    original = [[("is_lift", {"label": "block_0", "z_threshold": 0.1})], np.arange(3, dtype=np.float32)]
    restored = decode_context(json.loads(json.dumps(encode_context(original))))
    assert isinstance(restored[0], list)
    assert isinstance(restored[0][0], tuple)
    assert restored[0] == original[0]
    np.testing.assert_array_equal(restored[1], original[1])
    assert restored[1].dtype == original[1].dtype


def test_reward_parser_methods_roundtrip_without_arbitrary_callables():
    class Parser:
        def select_label_by_zmin(self, labels):
            return labels[0]

    parser = Parser()
    encoded = encode_context(("is_axis_up", {"label": parser.select_label_by_zmin}), parser)
    restored = decode_context(json.loads(json.dumps(encoded)), parser)
    assert restored[1]["label"](["bowl0", "bowl1"]) == "bowl0"
    with pytest.raises(TypeError):
        encode_context(lambda: None, parser)
    with pytest.raises(ValueError):
        decode_context({"__parser_method__": "__init__"}, parser)
    with pytest.raises(ValueError):
        decode_context({"__parser_method__": "missing"}, parser)


def test_atomic_language_rebased(tmp_path):
    path = tmp_path / "episode.h5"
    write_fixture(path)
    with h5py.File(path) as f:
        segments = segments_from_frames([json.loads(a) for a in f["annotations"]])
    assert [(s["start_frame"], s["end_frame"]) for s in segments] == [(0, 3), (3, 6)]
    rows = persistent_rows(segments, 25, 3, 6)
    assert len(rows) == 1
    assert rows[0]["content"] == "Lift the block."
    assert rows[0]["timestamp"] == 0


def test_existing_official_dataset_cannot_restore(tmp_path):
    with pytest.raises(ValueError, match="no simulator checkpoints"):
        resolve_episode(tmp_path, 0)


def test_manifest_relocation_and_traversal(tmp_path):
    (tmp_path / "meta").mkdir()
    manifest = tmp_path / "meta/replay.jsonl"
    entry = {"episode_index": 2, "state_path": "states/source.h5", "source_start_frame": 40, "length": 20}
    manifest.write_text(json.dumps(entry))
    assert resolve_episode(tmp_path, 2) == (tmp_path / "states/source.h5", 40, 20)
    entry["state_path"] = "../outside.h5"
    manifest.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="inside"):
        resolve_episode(tmp_path, 2)
