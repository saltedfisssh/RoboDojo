"""Per-episode HDF5 recorder.

Row ``i`` holds the observation (state + RGB) *before* action ``i`` and a full
physical checkpoint of that instant, so any frame can be restored in Isaac.
"""

import json
from pathlib import Path
import time

import h5py
import numpy as np

SCHEMA = "robodojo.expert-episode.v1"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class Recorder:
    def __init__(self, env, task, seed, layout, *, images=True, checkpoints=True, planner_entrypoint=None):
        self.env, self.task, self.seed, self.layout = env, task, int(seed), layout
        self.images = images
        self.checkpoints = checkpoints
        self.planner_entrypoint = planner_entrypoint
        self.rows = []
        self.adapter = None
        self.t0 = time.time()

    def bind(self, planner):
        if self.checkpoints and self.adapter is None:
            from src.datagen.state import RigidStateAdapter

            self.env.datagen_render_warmup = False
            self.adapter = RigidStateAdapter(self.env, planner)
        self.planner = planner

    def add(self, env, action, segment, segment_index):
        obs = env.observe(images=self.images)
        row = {
            "state": obs["state"],
            "action": np.asarray(action, np.float32),
            "ee_pose": np.concatenate([obs["ee_pose"][k] for k in sorted(obs["ee_pose"])]).astype(np.float32),
            "images": obs["images"],
            "segment": segment_index,
            "held": [h or "" for h in self.planner.held],
            "objects": self._object_poses(),
        }
        if self.adapter is not None:
            row["checkpoint"] = self.adapter.capture()
        self.instruction = obs["instruction"]
        self.rows.append(row)

    def _object_poses(self):
        p = self.planner
        out = {}
        for kind in ("Rigid", "Geometry"):
            for label in p.labels(kind):
                pos, quat = p.pose(label)
                out[label] = np.r_[pos, quat].astype(np.float32)
        return out

    def annotate(self, planner):
        """Segments as half-open [start_frame, end_frame) in recorded frames.

        Row i is recorded right before action i, so step counts are frame ids.
        Background segments (the other arm homing) may overlap foreground ones.
        """
        n = len(self.rows)
        self.segments = []
        for seg in planner.segments:
            start, end = seg.start_step, n if seg.end_step is None else seg.end_step
            if end <= start:
                continue
            self.segments.append(
                {
                    "primitive": seg.primitive,
                    "text": seg.text,
                    "arm": seg.arm,
                    "object": seg.obj,
                    "subtask": seg.subtask,
                    "background": seg.background,
                    "goal": _plain({k: v for k, v in seg.goal.items() if not k.startswith("_")}),
                    "start_frame": start,
                    "end_frame": min(end, n),
                }
            )
        self.subtasks = _plain(planner.subtasks)

    def save(self, path, *, outcome="success"):
        path = Path(path)
        tmp = path.with_suffix(".partial.h5")
        if path.exists() or tmp.exists():
            raise FileExistsError(f"Refusing to overwrite {path} or {tmp}")
        env = self.env
        n = len(self.rows)
        with h5py.File(tmp, "x") as f:
            f.attrs["schema"] = SCHEMA
            meta = {
                "task": self.task,
                "seed": self.seed,
                "planner": self.planner_entrypoint,
                "instruction": str(self.instruction),
                "fps": 25,
                "num_frames": n,
                "step_lim": int(env.step_lim),
                "success": outcome == "success" and bool(env.success[0] and env.end_flag[0]),
                "outcome": outcome,
                "steps": int(env.take_action_cnt[0]),
                "action_semantics": "absolute joints; left j1..6, left gripper open[0,1], right j1..6, right gripper",
                "state_semantics": "measured joints + previously commanded gripper opening (official ObsManager)",
                "collect_seconds": round(time.time() - self.t0, 1),
            }
            f.attrs["meta"] = json.dumps(meta)
            f.attrs["layout"] = json.dumps(self.layout)
            f.attrs["segments"] = json.dumps(self.segments)
            f.attrs["subtasks"] = json.dumps(self.subtasks)
            f.attrs["env_config"] = json.dumps(_plain(_config(env)))
            for key in ("state", "action", "ee_pose"):
                f.create_dataset(key, data=np.stack([r[key] for r in self.rows]))
            f.create_dataset("segment", data=np.asarray([r["segment"] for r in self.rows], np.int32))
            f.create_dataset("held", data=np.asarray([r["held"] for r in self.rows], dtype=h5py.string_dtype()))
            g = f.create_group("objects")
            for label in self.rows[0]["objects"]:
                g.create_dataset(label, data=np.stack([r["objects"][label] for r in self.rows]))
            if self.images and self.rows[0]["images"]:
                g = f.create_group("images")
                for cam in CAMERAS:
                    frames = np.stack([r["images"][cam] for r in self.rows])
                    g.create_dataset(cam, data=frames, chunks=(1, *frames.shape[1:]), compression="lzf")
            if self.adapter is not None:
                g = f.create_group("checkpoint")
                g.attrs["schema"] = self.adapter.schema
                keys = sorted(self.rows[0]["checkpoint"][0])
                arrays = g.create_group("arrays")
                for k in keys:
                    arrays.create_dataset(k, data=np.stack([r["checkpoint"][0][k] for r in self.rows]))
                g.create_dataset(
                    "context",
                    data=np.asarray([json.dumps(r["checkpoint"][1]) for r in self.rows], dtype=h5py.string_dtype()),
                    compression="gzip",
                )
        tmp.rename(path)
        return path


def _config(env):
    from omegaconf import OmegaConf

    return {"robot": OmegaConf.to_container(env.robot_config, resolve=True)}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
