"""Version/config/asset identity needed to rebuild a physical checkpoint."""

import hashlib
import importlib.metadata
from pathlib import Path
import subprocess

from src.datagen.layouts import OBJECTS, ROOT


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_versions():
    return {
        "isaacsim": importlib.metadata.version("isaacsim"),
        "isaaclab_package": importlib.metadata.version("isaaclab"),
        "isaaclab_release": (ROOT / "third_party/IsaacLab/VERSION").read_text().strip(),
        "curobo": importlib.metadata.version("nvidia-curobo"),
    }


def asset_manifest(layout, *, robot_names=("x5",), task=None):
    paths = set()
    for kind in ("Rigid", "Geometry", "Articulation", "Garment", "Fluid", "Dynamic"):
        for category, records in layout.get(kind, {}).items():
            for rec in records:
                folder = OBJECTS / kind / category / f"{rec['category_idx']:05d}"
                paths.add(folder / "metadata.json")
                paths.add(folder / ("object.usdz" if (folder / "object.usdz").exists() else "object.usd"))
    paths.update(ROOT / "Assets/Robots/x5" / name for name in ("ARX.usd", "X5A.urdf", "robot_config.yml", "curobo.yml"))
    for robot in set(robot_names) - {"x5"}:
        folder = ROOT / "Assets/Robots" / robot
        for pattern in ("*.usd", "*.usdz", "*.urdf", "*.yml", "*.yaml"):
            paths.update(folder.rglob(pattern))
    if task in ("play_tic_tac_toe", "make_kong", "imitate_sorting_sequence"):
        paths.update((ROOT / "Assets/Traj/RoboDojo" / task).rglob("*.pkl"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def verify_provenance(metadata):
    actual = runtime_versions()
    if metadata.get("versions") != actual:
        raise ValueError(f"Simulator/planner versions changed: recorded={metadata.get('versions')}, current={actual}")
    for relative, digest in metadata.get("asset_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Recorded asset changed or missing: {relative}")


def code_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_manifest():
    from src.datagen.frozen import SOURCES

    if SOURCES:
        return {name: hashlib.sha256(source).hexdigest() for name, source in SOURCES.items()}
    paths = set((ROOT / "src/datagen").rglob("*.py"))
    paths.update((ROOT / "env/planner_manager").glob("*.py"))
    paths.update((ROOT / "env/reward_manager").glob("*.py"))
    paths.update((ROOT / "task/RoboDojo/tasks").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
