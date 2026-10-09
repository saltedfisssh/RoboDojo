"""Small, JSON-safe episode evidence for planner iteration (no Isaac imports)."""

from dataclasses import asdict
import json
from pathlib import Path
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def progress(path, seed, **values):
    if path is not None:
        write_json(path, dict(seed=seed, updated=time.time(), **values))


def episode_diagnostics(motion):
    """Snapshot real scene state, not a replacement for official success."""
    from src.expert.recorder import _plain
    from src.expert.runtime import failing_predicates

    objects = {}
    for kind in ("Rigid", "Geometry", "Articulation"):
        for label in motion.labels(kind):
            position, quaternion = motion.pose(label)
            objects[label] = dict(position=position.tolist(), quaternion=quaternion.tolist())
    return {
        "step": motion.now,
        "step_limit": int(motion.env.step_lim),
        "last_stage": _plain(asdict(motion.segment)) if motion.segment else None,
        "subtasks": _plain(motion.subtasks),
        "segments": [_plain(asdict(segment)) for segment in motion.segments],
        "held": motion.held.copy(),
        "objects": objects,
        "failing_predicates": failing_predicates(motion.env),
    }
