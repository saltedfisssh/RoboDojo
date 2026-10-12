"""Official JSON payloads and separate generation/render audit records."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.expert.layout import ROOT

DEFAULT_ROOT = ROOT / "Eval_Layout"
KINDS = ("Rigid", "Geometry", "Articulation", "Dynamic", "Garment", "Fluid")
FIELDS = {
    "category",
    "category_idx",
    "xlim",
    "ylim",
    "zlim",
    "qpos",
    "rotate_deg",
    "rotate_rand",
    "relative_plane",
    "place_tag",
    "margin",
    "check_mode",
    "need_check_stable",
    "label",
    "default_pos",
    "default_ori",
    "scale",
    "physics",
    "visual",
    "group",
    "clutter_idx",
    "type",
    "yaml_path",
    "dynamic_layout",
    "shared_y",
    "symmetric_index",
    "symmetric_total",
}


def task_names():
    from scripts.internal.task_inventory import build_inventory

    return sorted(r["name"] for r in build_inventory()["tasks"] if r["runnable"])


def official_payload(layout):
    """Use the existing official field names, without provenance in the payload."""
    from copy import deepcopy

    result = deepcopy(layout)
    for kind in KINDS:
        for records in result.get(kind, {}).values():
            for inst in records:
                for key in list(inst):
                    if key not in FIELDS or key == "group" and inst[key] is None:
                        del inst[key]
                for key, value in {
                    "place_tag": None,
                    "margin": 0.01,
                    "check_mode": "bbox",
                    "rotate_deg": 0,
                    "rotate_rand": False,
                }.items():
                    inst.setdefault(key, value)
    for kind in ("Room", "Table", "Ground", "Light"):
        if kind in result:
            result[kind].pop("random", None)
    if "Ground" in result:
        result["Ground"].get("materials", {})["random"] = False
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def paths(root, task, index, set_seed=1000):
    name = f"{task}_{index}"
    directory = Path(root) / "RoboDojo/arx_x5" / str(set_seed)
    return (
        directory / (name + ".json"),
        directory / (name + ".png"),
        Path(root) / "metadata" / str(set_seed) / (name + ".json"),
    )


def existing_records(root, task, set_seed=1000, *, require_contiguous=True):
    records = []
    for path in (Path(root) / "metadata" / str(set_seed)).glob(f"{task}_*.json"):
        data = json.loads(path.read_text())
        if data.get("task") != task:
            continue
        scene, image, _ = paths(root, task, data["index"], set_seed)
        if (
            data["validation"]["status"] != "passed"
            or sha256(scene) != data["json_sha256"]
            or sha256(image) != data["image_sha256"]
        ):
            raise ValueError(f"Invalid published layout or image: {path}")
        if data.get("trajectory_path") and sha256(Path(root) / data["trajectory_path"]) != data["trajectory_sha256"]:
            raise ValueError(f"Invalid paired support trajectory: {path}")
        records.append(data)
    records.sort(key=lambda r: r["index"])
    if require_contiguous and [r["index"] for r in records] != list(range(len(records))):
        raise ValueError(f"Non-contiguous layout IDs: {task}")
    return records


def manifest(root=DEFAULT_ROOT, *, set_seed=1000, count=50, records_by_task=None):
    tasks = {}
    for task in task_names():
        records = (
            records_by_task[task]
            if records_by_task is not None
            else existing_records(root, task, set_seed, require_contiguous=False)
        )
        files_complete = [r["index"] for r in records] == list(range(count))
        checks_complete = all(
            r["validation"].get("reward_initialization") == "passed"
            and r["validation"].get("latest_recheck", {}).get("status", "passed") == "passed"
            and (
                task != "imitate_sorting_sequence"
                or r.get("trajectory_path")
                and r["validation"].get("support_trajectory", {}).get("native_replay")
            )
            for r in records
        )
        tasks[task] = {
            "count": len(records),
            "complete": files_complete and checks_complete,
            "files_complete": files_complete,
            "initialization_checks_complete": checks_complete,
            "sampling_seeds": [r["sampling_seed"] for r in records],
            "distinct_json_count": len({r["json_sha256"] for r in records}),
        }
    result = {
        "source": "fresh_task_config_sampling",
        "set_seed": set_seed,
        "target_per_task": count,
        "task_count": len(tasks),
        "layout_count": sum(r["count"] for r in tasks.values()),
        "image_count": sum(r["count"] for r in tasks.values()),
        "complete": all(r["complete"] for r in tasks.values()),
        "tasks": tasks,
    }
    atomic_json(Path(root) / "manifest.json", result)
    return result
