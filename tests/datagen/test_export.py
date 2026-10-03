import json

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lerobot")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from test_recording import write_fixture

from src.datagen.export import export_episodes


@pytest.mark.parametrize("atomic", [False, True])
def test_real_lerobot_roundtrip(tmp_path, atomic):
    raw = tmp_path / "source.h5"
    failed = tmp_path / "failed.h5"
    write_fixture(raw)
    write_fixture(failed, success=False)
    root = tmp_path / "export"
    manifest = export_episodes([raw, failed], root, "local/test", split_subtasks=atomic)
    assert len(manifest) == (2 if atomic else 1)
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_frames"] == 6
    assert "language_persistent" in info["features"]
    table = pq.read_table(next((root / "data").rglob("*.parquet")))
    assert table["source_frame_index"].to_pylist() == list(range(6))
    ds = LeRobotDataset("local/test", root=root, video_backend="pyav")
    sample = ds[3]
    assert sample["observation.images.cam_high"].shape == (3, 32, 48)
    np.testing.assert_array_equal(sample["observation.state"], np.full(14, 3))
    np.testing.assert_array_equal(sample["action"], np.full(14, 4))
    rows = sample["language_persistent"]
    assert rows[-1]["content"] == "Lift the block."
    assert rows[-1]["timestamp"] == pytest.approx(0 if atomic else 0.12)
    assert (root / manifest[-1]["state_path"]).is_file()
    assert manifest[-1]["source_start_frame"] == (3 if atomic else 0)


def test_export_rejects_debug_and_failed_episodes(tmp_path):
    failed = tmp_path / "failed.h5"
    write_fixture(failed, success=False)
    with pytest.raises(ValueError, match="No complete successful"):
        export_episodes([failed], tmp_path / "empty", "local/test")
    debug = tmp_path / "debug.h5"
    write_fixture(debug, with_images=False)
    with pytest.raises(ValueError, match="no images"):
        export_episodes([debug], tmp_path / "debug", "local/test")


def test_hook_context_carries_into_atomic_episode(tmp_path, monkeypatch):
    raw = tmp_path / "source.h5"
    write_fixture(raw)

    def provider(context):
        assert context.num_frames == 6
        return {
            "persistent": [
                {
                    "role": "assistant",
                    "style": "memory",
                    "content": "The object is grasped.",
                    "timestamp": 0.04,
                    "camera": None,
                    "tool_calls": None,
                }
            ],
            "events": {
                3: [
                    {
                        "role": "user",
                        "style": "interjection",
                        "content": "Continue.",
                        "camera": None,
                        "tool_calls": None,
                    }
                ]
            },
        }

    monkeypatch.setattr("src.datagen.export.load_provider", lambda spec: provider)
    root = tmp_path / "atomic"
    export_episodes([raw], root, "local/test", split_subtasks=True, annotation_hook="test:provider")
    ds = LeRobotDataset("local/test", root=root, video_backend="pyav")
    sample = ds[3]
    memory = next(r for r in sample["language_persistent"] if r["style"] == "memory")
    assert memory["timestamp"] == 0
    assert sample["language_events"][0]["content"] == "Continue."
    assert ds[4]["language_events"] == []


def test_provider_failure_does_not_publish_partial_dataset(tmp_path, monkeypatch):
    raw = tmp_path / "source.h5"
    write_fixture(raw)

    def broken_provider(context):
        raise RuntimeError("annotation unavailable")

    monkeypatch.setattr("src.datagen.export.load_provider", lambda spec: broken_provider)
    root = tmp_path / "failed_export"
    with pytest.raises(RuntimeError, match="annotation unavailable"):
        export_episodes([raw], root, "local/test", annotation_hook="test:provider")
    assert not root.exists()
    assert root.with_name("failed_export.partial").exists()
