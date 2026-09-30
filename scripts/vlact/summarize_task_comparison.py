"""Audit completed paired rollouts and render camera contact sheets (requires av)."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import av
import numpy as np
from PIL import Image, ImageDraw


def inspect_video(path):
    samples, count, first, last = {}, 0, None, None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_hint = stream.frames
        indices = {0, frame_hint // 3, 2 * frame_hint // 3}
        fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if first is None:
                first = rgb
            if count in indices:
                samples[count] = rgb
            last = rgb
            count += 1
    if count < 2:
        raise ValueError(f"Missing video frames: {path}")
    samples[count - 1] = last
    stats = dict(
        path=str(path),
        frames=count,
        fps=fps,
        shape=list(first.shape),
        first_rgb_mean=float(first.mean()),
        first_rgb_std=float(first.std()),
        last_rgb_mean=float(last.mean()),
        last_rgb_std=float(last.std()),
        first_last_mae=float(np.abs(first.astype(float) - last.astype(float)).mean()),
    )
    return stats, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    status = json.loads((output / "sweep_status.json").read_text())
    summary = dict(
        scope="paired layout 0, one episode/model/task, privileged state oracle", sweep_state=status["state"], runs=[]
    )
    rows = []
    for run in status["runs"]:
        if run["state"] != "complete":
            continue
        directory = Path(run["result_directory"])
        oracle = [json.loads(line) for line in (directory / "oracle_steps.jsonl").read_text().splitlines()]
        resume_path = directory.parent / f"_resume_{directory.name}.json"
        if resume_path.exists():
            resume = json.loads(resume_path.read_text())
        else:
            # main.py removes the resume manifest on normal completion.
            log = Path(run["log"]).read_text(errors="replace")
            counts = re.findall(r"Success nums: (\d+), Fail nums: (\d+), Unstable nums: (\d+)", log)
            if not counts or "(eval completed)" not in log:
                raise ValueError(f"Missing final simulator counts: {run['log']}")
            abandoned = set()
            for ids in re.findall(r"abandon seeds=(\[[^\]]*\])", log):
                abandoned.update(json.loads(ids))
            resume = {"unstable_nums": int(counts[-1][2]), "abandoned_layout_ids": sorted(abandoned)}
        video_stats, sheets = {}, []
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            paths = list(directory.glob(f"episode_*_{camera}_*.mp4"))
            if len(paths) != 1:
                raise ValueError(f"Expected one episode video for {camera}: {directory}")
            stats, samples = inspect_video(paths[0])
            video_stats[camera] = stats
            sheet_path = output / f"{run['task']}_{run['checkpoint_step']}_{camera}.jpg"
            sheet = Image.new("RGB", (320 * len(samples), 265), "white")
            draw = ImageDraw.Draw(sheet)
            for column, (frame, rgb) in enumerate(samples.items()):
                sheet.paste(Image.fromarray(rgb).resize((320, 240)), (column * 320, 25))
                draw.text((column * 320 + 4, 6), f"{run['checkpoint_step']} {camera} frame={frame}", fill="black")
            sheet.save(sheet_path, quality=90)
            sheets.append(str(sheet_path))
        stages = dict(Counter(row["stage"] for row in oracle))
        transitions = [{k: row[k] for k in ("step", "stage", "current_subtask")} for row in oracle if row["transition"]]
        progress = {}
        for key in ("inserted_count", "balls_in_vase", "inserted", "lifted", "held_by_target"):
            values = [row[key] for row in oracle if key in row and not isinstance(row[key], list)]
            if values:
                progress[key] = dict(max=max(values), final=values[-1])
        item = dict(
            **run,
            oracle_steps=len(oracle),
            stages=stages,
            transitions=transitions,
            progress=progress,
            unstable=resume["unstable_nums"],
            abandoned_layout_ids=resume["abandoned_layout_ids"],
            videos=video_stats,
            contact_sheets=sheets,
        )
        summary["runs"].append(item)
        rows.append(
            dict(
                task=run["task"],
                checkpoint_step=run["checkpoint_step"],
                success_rate=run["result"]["success_rate"],
                score=run["result"]["score"],
                eval_time=run["result"]["eval_time"],
                control_steps=len(oracle),
                final_stage=oracle[-1]["stage"],
                transitions=len(transitions),
                unstable=resume["unstable_nums"],
                abandoned=len(resume["abandoned_layout_ids"]),
            )
        )
    (output / "rollout_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (output / "per_task.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
