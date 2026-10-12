#!/usr/bin/env python3
"""Check every published JSON/PNG pair and compare with official sets."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.eval_layouts.storage import DEFAULT_ROOT, KINDS, atomic_json, existing_records, manifest, paths, task_names


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def pose_digest(layout):
    records = []
    for kind in KINDS:
        for category, objects in layout.get(kind, {}).items():
            for item in objects:
                q = np.asarray(item["default_ori"], dtype=float)
                q /= np.linalg.norm(q)
                if next((v for v in q if abs(v) > 1e-8), 1) < 0:
                    q = -q
                records.append(
                    [
                        kind,
                        category,
                        item["category_idx"],
                        item.get("label"),
                        np.round(item["default_pos"], 4).tolist(),
                        np.round(q, 4).tolist(),
                        np.round(item["scale"], 4).tolist(),
                    ]
                )
    return digest(sorted(records, key=lambda r: json.dumps(r)))


def official_summary(root):
    result = {}
    for set_seed in (0, 1, 2):
        directory = root / "RoboDojo/arx_x5" / str(set_seed)
        if not directory.is_dir():
            continue
        tasks = {}
        for task in task_names():
            files = [p for p in directory.glob(f"{task}_*.json") if p.stem[len(task) + 1 :].isdigit()]
            data = [json.loads(p.read_text()) for p in files]
            tasks[task] = {
                "files": len(files),
                "distinct_json": len({digest(d) for d in data}),
                "distinct_poses": len({pose_digest(d) for d in data}),
            }
        result[str(set_seed)] = {"file_count": sum(t["files"] for t in tasks.values()), "tasks": tasks}
    return result


def trajectory_summary(root):
    result = {}
    for seed in (0, 1, 2):
        lengths = []
        directory = root / "RoboDojo/imitate_sorting_sequence" / str(seed)
        for path in directory.glob("*.pkl"):
            data = pickle.loads(path.read_bytes())
            lengths.append(sum(len(chunk) for chunk in data["joint_path"]))
        if lengths:
            result[str(seed)] = {
                "files": len(lengths),
                "min_control_steps": min(lengths),
                "median_control_steps": float(np.median(lengths)),
                "max_control_steps": max(lengths),
            }
    return result


def validate_teacher(root, record, layout):
    path = root / record["trajectory_path"]
    expected = root / "Traj/RoboDojo/imitate_sorting_sequence" / str(record["set_seed"]) / f"{record['index']}.pkl"
    if path != expected:
        raise ValueError(f"Trajectory episode ID differs from layout: {path}")
    data = pickle.loads(path.read_bytes())
    if set(data) != {"joint_path", "target_label", "aim_label", "target_place_tag"}:
        raise ValueError(f"Unexpected native trajectory fields: {path}")
    if sorted(data["target_label"]) != [f"t{i}" for i in range(5)] or sorted(data["aim_label"]) != [
        f"aim{i}" for i in range(5)
    ]:
        raise ValueError(f"Invalid demonstration labels: {path}")
    instances = {
        item["label"]: (category, item["category_idx"]) for category, items in layout["Rigid"].items() for item in items
    }
    for target, aim in zip(data["target_label"], data["aim_label"]):
        if target.removeprefix("t") != aim.removeprefix("aim") or instances[target] != instances[aim]:
            raise ValueError(f"Demonstration objects do not match target models: {path}")
    validation = record["validation"]["support_trajectory"]
    steps = [step for chunk in data["joint_path"] for step in chunk]
    if not 0 < len(steps) <= 16000 or len(steps) != validation["physics_steps"]:
        raise ValueError(f"Incorrect demonstration length: {path}")
    if len(data["target_place_tag"]) != 5 or data["target_label"] != validation["target_label"]:
        raise ValueError(f"Demonstration annotations differ from validation: {path}")
    if (
        not validation["native_replay"]
        or len(validation["aim_in_basket"]) != 5
        or min(validation["aim_in_basket"]) < 0.5
    ):
        raise ValueError(f"Demonstration lacks a successful native terminal check: {path}")
    for key, width in (("support_arm0_joint_state", 7), ("support_ee0_joint_state", 2)):
        for field in ("position", "velocity"):
            values = np.asarray([step[key][field] for step in steps], dtype=float)
            if values.shape != (len(steps), width) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key}.{field} in {path}")


def audit(root=DEFAULT_ROOT, set_seed=1000, count=50):
    snapshot = {task: existing_records(root, task, set_seed, require_contiguous=False) for task in task_names()}
    state = manifest(root, set_seed=set_seed, count=count, records_by_task=snapshot)
    tasks, png_count = {}, 0
    render_profiles = Counter()
    camera_sets = Counter()
    gallery = [
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>RoboDojo 场景布局</title>",
        "<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;background:#f5f5f5;color:#222}"
        "section{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;background:white;padding:10px;border-radius:8px}"
        "img{width:320px;height:240px;object-fit:contain}summary{padding:18px;font-size:18px;cursor:pointer}"
        "a{color:#2455a4}figcaption{padding-top:8px}</style>",
        f"<h1>RoboDojo 集合 {set_seed}</h1><p>真实 Isaac Sim 初始化渲染，每张图片对应同名 JSON。"
        f"已验证 {state['layout_count']} / {len(task_names()) * count} 份布局。</p>",
    ]
    sheet = Image.new("RGB", (6 * 240, 9 * 204), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for task_number, task in enumerate(task_names()):
        records = snapshot[task]
        payloads = []
        gallery.append(f"<details><summary>{html.escape(task)} · {len(records)} / {count}</summary><section>")
        for r in records:
            scene, image, _ = paths(root, task, r["index"], set_seed)
            payloads.append(json.loads(scene.read_text()))
            if task == "imitate_sorting_sequence":
                validate_teacher(root, r, payloads[-1])
            with Image.open(image) as img:
                img.load()
                if img.mode != "RGB" or list(np.asarray(img).shape) != r["validation"]["image_shape"]:
                    raise ValueError(f"Image dimensions or mode differ from validation: {image}")
                if np.asarray(img).std() < 1:
                    raise ValueError(f"Empty render: {image}")
                if r["index"] == 0:
                    thumb = img.copy()
                    thumb.thumbnail((240, 180))
                    x, y = task_number % 6 * 240, task_number // 6 * 204
                    sheet.paste(thumb, (x, y))
                    draw.text((x + 4, y + 184), task, fill="black")
            if r["sampling_seed"] < 1000:
                raise ValueError(f"Source sampling seed precedes 1000: {scene}")
            png_count += 1
            render_profiles[(r["validation"].get("render") or {}).get("profile", "unrecorded")] += 1
            camera_sets[", ".join(r["validation"].get("initialized_cameras", ["unrecorded"]))] += 1
            scene_url, image_url = str(scene.relative_to(root)), str(image.relative_to(root))
            gallery.append(
                f"<figure><a href='{image_url}'><img loading='lazy' src='{image_url}'></a>"
                f"<figcaption>#{r['index']} · 采样种子 {r['sampling_seed']} · "
                f"<a href='{scene_url}'>JSON</a> · <a href='{image_url}'>PNG</a></figcaption></figure>"
            )
        gallery.append("</section></details>")
        scratch = ROOT / "data/eval_layout_generation" / str(set_seed) / task
        attempts = [
            json.loads(line) for journal in scratch.rglob("attempts.jsonl") for line in journal.read_text().splitlines()
        ]
        failures = Counter(a.get("reason") for a in attempts if a.get("accepted") is False)
        tasks[task] = {
            "files": len(records),
            "distinct_json": len({digest(d) for d in payloads}),
            "distinct_poses": len({pose_digest(d) for d in payloads}),
            "duplicate_json_count": len(records) - len({digest(d) for d in payloads}),
            "sampling_seeds": [r["sampling_seed"] for r in records],
            "rejections": dict(failures),
            "reward_initialization_checks": sum(
                r["validation"].get("reward_initialization") == "passed" for r in records
            ),
            "support_replay_checks": sum(
                r["validation"].get("support_trajectory", {}).get("native_replay", False) for r in records
            ),
        }
        teacher_steps = [
            r["validation"]["support_trajectory"]["physics_steps"]
            for r in records
            if r["validation"].get("support_trajectory")
        ]
        if teacher_steps:
            tasks[task]["teacher_control_steps"] = {
                "min": min(teacher_steps),
                "median": float(np.median(teacher_steps)),
                "max": max(teacher_steps),
            }
    result = {
        "complete": state["complete"],
        "set_seed": set_seed,
        "png_verified": png_count,
        "pose_comparison": "Positions/scales and normalized sign-canonical quaternions rounded to 4 decimals",
        "official": official_summary(ROOT / "Assets/Eval_Layout"),
        "official_sorting_trajectories": trajectory_summary(ROOT / "Assets/Traj"),
        "render_profile_counts": dict(render_profiles),
        "initialized_camera_counts": dict(camera_sets),
        "ours": tasks,
    }
    atomic_json(root / "comparison.json", result)
    sheet.save(root / "preview.png")
    (root / "index.html").write_text("\n".join(gallery) + "</html>")
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "json_count": state["layout_count"],
                "png_verified": png_count,
                "preview": str(root / "preview.png"),
                "gallery": str(root / "index.html"),
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--set-seed", type=int, default=1000)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    audit(args.root.resolve(), args.set_seed, args.count)
