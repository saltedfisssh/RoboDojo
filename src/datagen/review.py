"""Build an honest 35-task inventory, complete-episode previews and LeRobot exports."""

import argparse
from datetime import UTC, datetime
import fcntl
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import h5py
import numpy as np

from src.datagen.annotations import segments_from_frames


def write_text_atomically(path, text):
    """Keep the last valid report when a shared filesystem rejects a write."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_episode(path):
    with h5py.File(path, "r") as f:
        metadata = json.loads(f.attrs["metadata"])
        count = int(f.attrs.get("num_frames", 0))
        return {
            "path": str(path.resolve()),
            "seed": metadata["seed"],
            "frames": count,
            "duration_s": count / metadata["fps"],
            "success": bool(f.attrs.get("success", False)),
            "complete": bool(f.attrs.get("complete", False)),
            "rgb": "images" in f
            and set(f["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
            and all(len(ds) == count for ds in f["images"].values()),
            "terminal_state": "terminal" in f,
            "within_benchmark_budget": (
                bool(f.attrs["within_benchmark_budget"]) if "within_benchmark_budget" in f.attrs else None
            ),
            "task_definition": metadata.get("task_definition", "official"),
            "generation_mode": metadata.get("generation_mode", "physical_collection"),
            "reason": f.attrs.get("reason", ""),
        }


def preview(source, target):
    """Encode every frame in order: head at full size, both wrists at half size."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.mp4")
    with h5py.File(source) as f:
        count = int(f.attrs["num_frames"])
        head = f["images/cam_high"]
        h, w = head.shape[1:3]
        wrists = [k for k in f["images"] if k != "cam_high"]
        if len(wrists) != 2:
            raise ValueError("Review previews require head and two wrist cameras")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{w + w // 2}x{h}",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(partial),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            try:
                for i in range(count):
                    side = np.concatenate([f[f"images/{key}"][i][::2, ::2, :3] for key in wrists], axis=0)
                    frame = np.concatenate([head[i][..., :3], side], axis=1)
                    process.stdin.write(np.ascontiguousarray(frame).tobytes())
            finally:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"Preview encoding failed: {source}")
        spans = segments_from_frames([json.loads(a) for a in f["annotations"]])
        write_text_atomically(target.with_suffix(".json"), json.dumps(spans, indent=2) + "\n")
    partial.rename(target)


def validate_export(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/robodojo_review", root=root, video_backend="pyav")
    records = [json.loads(line) for line in (root / "meta/replay.jsonl").read_text().splitlines()]
    checked, offset = [], 0
    for record in records:
        with h5py.File(root / record["state_path"]) as f:
            count = record["length"]
            for local in sorted({0, count // 2, count - 1}):
                row = dataset[offset + local]
                np.testing.assert_array_equal(row["action"].numpy(), f["action"][local])
                np.testing.assert_array_equal(row["observation.state"].numpy(), f["observation/state"][local])
                for key in dataset.meta.camera_keys:
                    assert tuple(row[key].shape) == (3, 480, 640), (key, row[key].shape)
                    assert np.isfinite(row[key].numpy()).all()
                checked.append(offset + local)
            offset += count
    assert offset == len(dataset)
    report = {"episodes": len(records), "frames": offset, "reader_rows_checked": checked, "passed": True}
    write_text_atomically(root / "meta/readback.json", json.dumps(report, indent=2) + "\n")
    return report


def export_review_error(destination, valid_episodes, required):
    """A historical reader result must not keep a quarantined source accepted."""
    readback = destination / "meta/readback.json"
    manifest = destination / "meta/replay.jsonl"
    if not readback.exists():
        return "LeRobot reader validation has not completed"
    if not manifest.exists():
        return "The export has no source episode manifest"
    report = json.loads(readback.read_text())
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not report.get("passed") or report.get("episodes", 0) < required or len(records) < required:
        return "The export does not have enough reader-validated episodes"
    current_sources = {str(Path(episode["path"]).resolve()) for episode in valid_episodes}
    if any(str(Path(record["source_episode"]).resolve()) not in current_sources for record in records):
        return "An exported source episode is no longer in the accepted raw collection"
    return None


def build_review(root, required=3, export_ready=False):
    from src.datagen.export import export_episodes

    tasks = json.loads(Path(__file__).with_name("tasks35.json").read_text())
    for task in tasks:
        name = task["task"]
        episodes = []
        for path in sorted((root / "raw").glob(f"**/{name}/episode_*.h5")):
            if not path.name.endswith(".partial.h5"):
                episodes.append(inspect_episode(path))
        accepted = []
        seeds = {}
        for episode in episodes:
            if not all(episode[k] for k in ("complete", "success", "rgb", "terminal_state")):
                continue
            if episode["seed"] in seeds:
                episode["duplicate_seed_of"] = seeds[episode["seed"]]
                continue
            seeds[episode["seed"]] = episode["path"]
            accepted.append(episode)
        task.update(successful_rgb_episodes=len(accepted), requested=required, attempts=episodes)
        debug = []
        for path in sorted((root / "debug").glob(f"{name}_*.h5"), key=lambda p: p.stat().st_mtime):
            if not path.name.endswith(".partial.h5"):
                debug.append(inspect_episode(path))
        task["debug_attempts"] = debug
        task["physical_smoke_passed"] = any(e["success"] and e["complete"] for e in debug) or bool(accepted)
        task["process_failures"] = []
        for path in sorted((root / "raw").glob(f"**/{name}/attempt_*.json")):
            attempt = json.loads(path.read_text())
            if not attempt["success"]:
                task["process_failures"].append(attempt)
        task["generated_ready"] = len(accepted) >= required
        task["ready"] = (
            task["generated_ready"]
            and task["official_task_class_present"]
            and not task.get("dataset_definition_mismatch", False)
        )
        destination = root / "lerobot" / name
        if export_ready and task["generated_ready"]:
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                export_episodes([e["path"] for e in accepted[:required]], destination, f"local/{name}")
            if not (destination / "meta/readback.json").exists():
                validate_export(destination)
        if export_ready:
            for episode in accepted[:required]:
                target = root / "previews" / name / f"episode_{episode['seed']:010d}.mp4"
                if not target.exists():
                    preview(episode["path"], target)
        for episode in accepted[:required]:
            target = root / "previews" / name / f"episode_{episode['seed']:010d}.mp4"
            if target.exists():
                episode["preview"] = str(target.resolve())
        valid_episodes = [e for e in episodes if all(e[k] for k in ("complete", "success", "rgb", "terminal_state"))]
        task["export_review_error"] = (
            export_review_error(destination, valid_episodes, required) if destination.exists() else None
        )
        task["exported"] = destination.exists() and task["export_review_error"] is None
    result = {
        "updated_at": datetime.now(UTC).isoformat(),
        "requested_tasks": 35,
        "episodes_per_task": required,
        "generated_tasks": sum(t["generated_ready"] for t in tasks),
        "ready_tasks": sum(t["ready"] for t in tasks),
        "exported_tasks": sum(t["exported"] for t in tasks),
        "tasks": tasks,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_text_atomically(root / "status.json", json.dumps(result, indent=2) + "\n")
    lines = [
        "# RoboDojo 35 类采集检查",
        "",
        f"目标：每类 {required} 条新生成的完整成功轨迹。当前 {result['generated_tasks']}/35 类达到数量"
        f"（其中 {result['ready_tasks']} 类通过官方任务判据），"
        f"{result['exported_tasks']}/35 类已导出并通过 LeRobot 读取检查。",
        "",
        "状态以实际文件为准；失败、无 RGB、未结束的轨迹不计入。超出官方步数预算的轨迹会标出。",
        f"最近更新：{result['updated_at']}",
        "",
        "| 类别 | 物理验证 | 成功 RGB episode | 已导出 | 检查视频 |",
        "|---|---|---:|---|---|",
    ]
    budget_labels = {True: "", False: "（超预算）", None: "（预算未记录）"}
    for task in tasks:
        links = " · ".join(
            f"[{e['seed']}{budget_labels[e['within_benchmark_budget']]}]({e['preview']})"
            for e in task["attempts"]
            if "preview" in e
        )
        status = "是" if task["exported"] else "否"
        if task.get("export_review_error"):
            status += "；导出需重新验证"
        if not task["official_task_class_present"]:
            status += "；重建定义，未经官方验证"
        elif task.get("dataset_instruction_mismatch"):
            status += "；数据集指令含额外步骤"
        physical = "通过" if task["physical_smoke_passed"] else "待通过"
        lines.append(
            f"| {task['task']} | {physical} | {task['successful_rgb_episodes']}/{required} | {status} | {links} |"
        )
    lines.extend(["", "任务定义差异：", ""])
    lines.extend(f"- `{task['task']}`：{task['definition_note']}" for task in tasks if task.get("definition_note"))
    write_text_atomically(root / "REVIEW.md", "\n".join(lines) + "\n")
    from src.datagen.review_page import render_review_page

    write_text_atomically(root / "review.html", render_review_page(root, result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--export-ready", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = build_review(args.root.resolve(), args.episodes, args.export_ready)
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
