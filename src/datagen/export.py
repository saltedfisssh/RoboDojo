"""Convert successful raw demonstrations to LeRobot v3 with language and replay sidecars.

Run with the separate LeRobot venv; never import LeRobot into Isaac's environment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from src.datagen.annotations import (
    AnnotationContext,
    load_provider,
    persistent_rows,
    segments_from_frames,
    slice_persistent,
    validate_additions,
)

JOINT_NAMES = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_open")
]


def export_episodes(paths, root, repo_id, *, split_subtasks=False, annotation_hook=None):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    destination = Path(root)
    root = destination.with_name(destination.name + ".partial")
    if destination.exists() or root.exists():
        raise FileExistsError(f"Export destination and staging directory must be new: {destination}")
    accepted = []
    cameras = None
    for path in sorted({Path(p).resolve() for p in paths}):
        with h5py.File(path, "r") as f:
            if not f.attrs.get("complete") or not f.attrs.get("success") or not f.attrs.get("num_frames"):
                continue
            if "images" not in f:
                raise ValueError(f"Episode {path} has no images; --no-images is debug-only")
            schema = {k: tuple(v.shape[1:]) for k, v in f["images"].items()}
            meta = json.loads(f.attrs["metadata"])
            if meta["fps"] != 25 or meta["state_timing"] != "before_action":
                raise ValueError(f"Incompatible timing in {path}")
            if cameras is not None and cameras != schema:
                raise ValueError("All episodes must have the same cameras and image sizes")
            cameras = schema
            accepted.append(path)
    if not accepted:
        raise ValueError("No complete successful episodes to export")
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "observation.measured_state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
        "source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        **{
            f"observation.images.{k}": {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]}
            for k, shape in cameras.items()
        },
    }
    dataset = LeRobotDataset.create(
        repo_id,
        25,
        features,
        root=root,
        robot_type="dual_x5",
        image_writer_threads=4,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=20, preset="fast", g=2),
        encoder_threads=2,
    )
    (root / "states").mkdir()
    manifest, span_records = [], []
    task_definitions = {}
    timelines = {}
    provider = load_provider(annotation_hook) if annotation_hook else None
    try:
        for path in accepted:
            with h5py.File(path, "r") as f:
                metadata = json.loads(f.attrs["metadata"])
                for revision in {metadata.get("source_snapshot"), metadata.get("planned_source_snapshot")} - {None}:
                    if len(revision) != 64 or any(c not in "0123456789abcdef" for c in revision):
                        raise ValueError("Invalid source snapshot identifier")
                    archive = Path(__file__).resolve().parents[2] / ".cache/datagen-source" / f"{revision}.tar.gz"
                    if not archive.is_file():
                        raise FileNotFoundError(f"Recorded source snapshot is missing: {archive}")
                    (root / "sources").mkdir(exist_ok=True)
                    shutil.copyfile(archive, root / "sources" / archive.name)
                task_definitions[metadata["task"]] = metadata.get("task_definition", "official")
                count = int(f.attrs["num_frames"])
                annotations = [json.loads(a) for a in f["annotations"]]
                segments = segments_from_frames(annotations)
                annotation_context = AnnotationContext(metadata, segments, str(path), 25, count)
                additions = provider(annotation_context) if provider else {}
                validate_additions(additions, annotation_context, {f"observation.images.{k}" for k in cameras})
                # Persist provenance and physical snapshots without duplicating RGB data.
                with Path(path).open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                sidecar = f"states/{digest[:20]}.h5"
                with h5py.File(root / sidecar, "w") as dst:
                    for k, v in f.attrs.items():
                        dst.attrs[k] = v
                    for k in ("snapshots", "context", "annotations", "action", "observation", "terminal"):
                        if k not in f:
                            continue
                        f.copy(k, dst)
                ranges = (
                    [(s["start_frame"], s["end_frame"], s["text"]) for s in segments]
                    if split_subtasks
                    else [(0, count, metadata["instruction"])]
                )
                for start, end, instruction in ranges:
                    episode_id = dataset.num_episodes
                    language = persistent_rows(segments, 25, start, end)
                    language.extend(slice_persistent(additions.get("persistent", []), start / 25, end / 25))
                    timelines[episode_id] = {
                        "persistent": language,
                        "events": additions.get("events", {}),
                        "offset": start,
                    }
                    for i in range(start, end):
                        frame = {
                            "observation.state": f["observation/state"][i].astype(np.float32),
                            "observation.measured_state": f["observation/measured_state"][i].astype(np.float32),
                            "action": f["action"][i].astype(np.float32),
                            "subtask_index": np.array([annotations[i]["segment_id"]], dtype=np.int64),
                            "source_frame_index": np.array([i], dtype=np.int64),
                            "next.done": np.array([i == end - 1], dtype=np.bool_),
                            "task": instruction,
                            **{f"observation.images.{k}": f[f"images/{k}"][i] for k in cameras},
                        }
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    manifest.append(
                        {
                            "episode_index": episode_id,
                            "state_path": sidecar,
                            "source_sha256": digest,
                            "source_episode": str(Path(path).resolve()),
                            "source_start_frame": start,
                            "length": end - start,
                            "task": metadata["task"],
                            "seed": metadata["seed"],
                            "fps": 25,
                        }
                    )
                    span_records.extend(
                        {
                            **s,
                            "episode_index": episode_id,
                            "start_frame": max(start, s["start_frame"]) - start,
                            "end_frame": min(end, s["end_frame"]) - start,
                        }
                        for s in segments
                        if s["start_frame"] < end and s["end_frame"] > start
                    )
    finally:
        dataset.finalize()
    write_language_columns(root, timelines)
    for name, records in (("replay.jsonl", manifest), ("subtasks.jsonl", span_records)):
        (root / "meta" / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "meta/datagen.json").write_text(
        json.dumps(
            {
                "schema": "robodojo.datagen.v1",
                "split_subtasks": split_subtasks,
                "boundary_semantics": "planner stages; half-open [start_frame,end_frame)",
                "success_filter": "recorded task predicate passed",
                "task_definitions": task_definitions,
                "annotation_hook": annotation_hook,
            },
            indent=2,
        )
        + "\n"
    )
    root.rename(destination)
    return manifest


def write_language_columns(root, timelines):
    # Match LeRobot's annotation pipeline: annotate AFTER DatasetWriter.finalize.
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
    from lerobot.datasets.language import language_feature_info, validate_camera_field
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        persistent, events = [], []
        for episode, frame in zip(table["episode_index"].to_pylist(), table["frame_index"].to_pylist(), strict=True):
            timeline = timelines[episode]
            persistent.append(timeline["persistent"])
            source_frame = frame + timeline["offset"]
            rows = timeline["events"].get(str(source_frame), timeline["events"].get(source_frame, []))
            events.append(rows)
            for row in timeline["persistent"] + rows:
                validate_camera_field(row.get("style"), row.get("camera"))
                if row.get("content") is None and row.get("tool_calls") is None:
                    raise ValueError("Language row requires content or tool_calls")
        # The upstream writer uses inferred Arrow types (pa.json_ cannot materialize tool_calls lists).
        table = pa.Table.from_arrays(
            [*table.columns, pa.array(persistent), pa.array(events)],
            names=[*table.column_names, "language_persistent", "language_events"],
        )
        tmp = path.with_suffix(".parquet.tmp")
        write_table_one_row_group_per_episode(table, tmp)
        tmp.replace(path)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"].update(language_feature_info())
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-id", default="local/robodojo_generated")
    p.add_argument("--split-subtasks", action="store_true")
    p.add_argument("--annotation-hook", help="Optional local module:function implementing AnnotationProvider")
    args = p.parse_args()
    paths = list(args.input.rglob("*.h5")) if args.input.is_dir() else [args.input]
    paths = [p for p in paths if not p.name.endswith(".partial.h5")]
    records = export_episodes(
        paths, args.output, args.repo_id, split_subtasks=args.split_subtasks, annotation_hook=args.annotation_hook
    )
    print(json.dumps({"episodes": len(records), "root": str(args.output)}))


if __name__ == "__main__":
    main()
