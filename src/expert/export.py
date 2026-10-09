"""Export expert HDF5 episodes to LeRobot v3 in the official RoboDojo layout.

Runs in the LeRobot venv (``bash scripts/expert.sh export``). Per episode it
also writes the Argus-schema label JSON and a state sidecar for restoring any
frame in Isaac.

Output:
  data/, videos/, meta/info.json ...   same features as RoboDojo_lerobot_v30_video
  + subtask_index, primitive_index     per-frame planner ids (int64)
  + language_persistent                LeRobot language rows (subtask / task_aug)
  meta/argus/episode_XXXXXX.json       Argus label file (labels.timeline, key_events, ...)
  meta/subtasks.jsonl                  [start_frame, end_frame) per planner segment
  meta/replay.jsonl                    episode -> states/*.h5 sidecar
  states/episode_XXXXXX.h5             per-frame physical checkpoints (no RGB)
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from src.expert.argus import build as build_argus

JOINTS = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(7)]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FPS = 25
_ROW = [("role", "string"), ("content", "string"), ("style", "string")]


def _struct(with_timestamp):
    import pyarrow as pa

    fields = [(n, pa.string()) for n, _ in _ROW]
    if with_timestamp:
        fields.append(("timestamp", pa.float64()))
    return pa.struct([*fields, ("camera", pa.string()), ("tool_calls", pa.null())])


def timeline_language(argus):
    rows = []
    for e in argus["labels"]["timeline"]:
        if e["contribution"] != "advancing":
            continue
        text = e["action"]
        if e["object"] and e["object"] not in text:
            text = f"{text} the {e['object']}"
        if e["destination"] and e["destination"] not in text and e["action"] not in ("grasp", "approach"):
            text = f"{text} to the {e['destination']}"
        rows.append(
            {
                "role": "assistant",
                "content": f"{e['arm']} arm: {text}",
                "style": "subtask",
                "timestamp": float(e["start_s"]),
                "camera": None,
                "tool_calls": None,
            }
        )
    for v in argus["labels"]["instruction_variants"]:
        rows.append(
            {"role": "user", "content": v, "style": "task_aug", "timestamp": 0.0, "camera": None, "tool_calls": None}
        )
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def export(inputs, out, repo_id, vcodec="h264"):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(out)
    stage = out.with_name(out.name + ".partial")
    if out.exists() or stage.exists():
        raise FileExistsError(f"{out} (or its .partial) already exists")
    files = []
    for path in sorted(inputs):
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["meta"])
            if meta["success"] and "images" in f:
                files.append(path)
    if not files:
        raise SystemExit("no successful episodes with images")
    with h5py.File(files[0], "r") as f:
        shape = f["images/cam_high"].shape[1:]
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [JOINTS]},
        "action": {"dtype": "float32", "shape": (14,), "names": [JOINTS]},
        "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
        "primitive_index": {"dtype": "int64", "shape": (1,), "names": None},
        **{
            f"observation.images.{c}": {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]}
            for c in CAMERAS
        },
    }
    ds = LeRobotDataset.create(
        repo_id,
        FPS,
        features,
        root=stage,
        robot_type="unified_robot",
        image_writer_threads=4,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec=vcodec, crf=20, preset="fast", g=2),
        encoder_threads=2,
    )
    (stage / "states").mkdir()
    (stage / "meta/argus").mkdir(parents=True, exist_ok=True)
    language, replay, spans = {}, [], []
    try:
        for path in files:
            ep = ds.num_episodes
            argus = build_argus(path, episode_index=ep)
            argus["episode_dir"] = f"episode_{ep:06d}"
            (stage / f"meta/argus/episode_{ep:06d}.json").write_text(json.dumps(argus, indent=1))
            language[ep] = timeline_language(argus)
            with h5py.File(path, "r") as f:
                meta = json.loads(f.attrs["meta"])
                segments = json.loads(f.attrs["segments"])
                n = meta["num_frames"]
                sub_of = np.full(n, -1, np.int64)
                prim_of = np.full(n, -1, np.int64)
                for k, s in enumerate(segments):
                    sub_of[s["start_frame"] : s["end_frame"]] = s["subtask"]
                    prim_of[s["start_frame"] : s["end_frame"]] = k
                    spans.append(
                        {
                            "episode_index": ep,
                            **{
                                key: s[key]
                                for key in ("start_frame", "end_frame", "primitive", "text", "arm", "object", "subtask")
                            },
                        }
                    )
                state, action = f["state"][()], f["action"][()]
                imgs = {c: f[f"images/{c}"] for c in CAMERAS}
                for i in range(n):
                    ds.add_frame(
                        {
                            "observation.state": state[i].astype(np.float32),
                            "action": action[i].astype(np.float32),
                            "subtask_index": np.array([sub_of[i]], np.int64),
                            "primitive_index": np.array([prim_of[i]], np.int64),
                            "task": meta["instruction"],
                            **{f"observation.images.{c}": imgs[c][i] for c in CAMERAS},
                        }
                    )
                ds.save_episode()
                side = f"states/episode_{ep:06d}.h5"
                with h5py.File(stage / side, "w") as dst:
                    for k, v in f.attrs.items():
                        dst.attrs[k] = v
                    for k in ("checkpoint", "state", "action", "segment", "held", "objects"):
                        if k in f:
                            f.copy(k, dst)
            replay.append(
                {
                    "episode_index": ep,
                    "state_path": side,
                    "task": meta["task"],
                    "seed": meta["seed"],
                    "length": n,
                    "source": str(Path(path).resolve()),
                }
            )
    finally:
        ds.finalize()
    write_language(stage, language)
    (stage / "meta/replay.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replay))
    (stage / "meta/subtasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in spans))
    stage.rename(out)
    return len(replay)


def write_language(root, language):
    """Add language_persistent / language_events after finalize (LeRobot annotation pipeline)."""
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
    from lerobot.datasets.language import language_feature_info
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        if "language_persistent" in table.column_names:
            table = table.drop_columns(["language_persistent", "language_events"])
        eps = table["episode_index"].to_pylist()
        # tool_calls is an arrow.json extension that pyarrow cannot build from
        # Python; a null-typed field is accepted by the LeRobot reader.
        persistent = pa.array([language[e] for e in eps], type=pa.list_(_struct(True)))
        events = pa.array([[] for _ in eps], type=pa.list_(_struct(False)))
        table = pa.Table.from_arrays(
            [*table.columns, persistent, events], names=[*table.column_names, "language_persistent", "language_events"]
        )
        tmp = path.with_suffix(".parquet.tmp")
        write_table_one_row_group_per_episode(table, tmp)
        tmp.replace(path)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"].update(language_feature_info())
    # Match the official dataset's image metadata convention (channels-first).
    for key, feat in info["features"].items():
        if feat.get("dtype") == "video" and feat["names"][:2] == ["height", "width"]:
            h, w, c = feat["shape"]
            feat["shape"], feat["names"] = [c, h, w], ["channels", "height", "width"]
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, nargs="+", required=True, help="episode .h5 files or directories")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/robodojo_expert")
    ap.add_argument("--vcodec", default="h264", help="h264 (fast) or libsvtav1 (official dataset uses av1)")
    args = ap.parse_args(argv)
    paths = []
    for p in args.input:
        paths.extend(sorted(p.rglob("episode_*.h5")) if p.is_dir() else [p])
    paths = [p for p in paths if not p.name.endswith(".partial.h5")]
    n = export(paths, args.output, args.repo_id, args.vcodec)
    print(json.dumps({"episodes": n, "root": str(args.output)}))


if __name__ == "__main__":
    main()
