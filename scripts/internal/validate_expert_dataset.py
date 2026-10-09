"""Read an exported expert dataset with LeRobot and decode every video frame."""

import argparse
import json
from pathlib import Path
import time


def validate(root):
    import av
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    info = json.loads((root / "meta/info.json").read_text())
    dataset = LeRobotDataset("local/robodojo_expert", root=root, video_backend="pyav")
    rows = [json.loads(line) for line in (root / "meta/replay.jsonl").read_text().splitlines()]
    assert len(rows) == info["total_episodes"] == dataset.num_episodes
    table = pa.concat_tables([pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))])
    table = table.sort_by([("index", "ascending")])
    assert "language_persistent" in table.column_names
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    frame_total, sample_count = 0, 0
    for row in rows:
        with h5py.File(root / row["state_path"], "r") as state:
            meta = json.loads(state.attrs["meta"])
            count = meta["num_frames"]
            assert meta["success"] and count == row["length"] <= meta["step_lim"]
            assert state["action"].shape == state["state"].shape == (count, 14)
            assert state["checkpoint/context"].shape == (count,)
            assert np.isfinite(state["action"][()]).all()
            assert np.isfinite(state["state"][()]).all()
            np.testing.assert_allclose(actions[frame_total : frame_total + count], state["action"][()], atol=1e-6)
            np.testing.assert_allclose(states[frame_total : frame_total + count], state["state"][()], atol=1e-6)
            segments = json.loads(state.attrs["segments"])
            assert segments and all(0 <= segment["start_frame"] < segment["end_frame"] <= count for segment in segments)
            annotation = json.loads((root / f"meta/argus/episode_{row['episode_index']:06d}.json").read_text())
            assert annotation["labels"]["completion"]["task_completed"] == "success"
            assert annotation["labels"]["timeline"]
            for offset in sorted({0, count // 2, count - 1}):
                frame = dataset[frame_total + offset]
                np.testing.assert_allclose(frame["action"].numpy(), state["action"][offset], atol=1e-6)
                np.testing.assert_allclose(frame["observation.state"].numpy(), state["state"][offset], atol=1e-6)
                for camera in cameras:
                    image = frame[f"observation.images.{camera}"]
                    assert tuple(image.shape) == (3, 480, 640), image.shape
                    assert torch.isfinite(image).all() and float(image.std()) > 0.001
                sample_count += 1
        frame_total += count
    assert frame_total == info["total_frames"] == len(dataset)
    decoded = {}
    for video in sorted((root / "videos").rglob("*.mp4")):
        with av.open(str(video)) as container:
            assert container.streams.video[0].codec_context.name == "h264"
            count = 0
            for frame in container.decode(video=0):
                assert frame.width == 640 and frame.height == 480
                count += 1
        decoded[str(video.relative_to(root))] = count
    assert sum(decoded.values()) == frame_total * len(cameras)
    return {
        "dataset": str(root.resolve()),
        "episodes": len(rows),
        "frames": frame_total,
        "lerobot_samples_checked": sample_count,
        "decoded_video_frames": sum(decoded.values()),
        "videos": decoded,
        "action_and_state_match": True,
        "argus_completion_valid": True,
        "checkpoints_present": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = validate(args.dataset)
    report["validation_seconds"] = time.monotonic() - started
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
