"""CPU-only checks for the expert data generator (no Isaac)."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.expert.layout import LayoutSampler, sample_layout

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["stack_blocks", "stack_bowls", "swap_T", "sort_nesting_dolls_by_size"]


@pytest.mark.parametrize("task", TASKS)
def test_layout_matches_eval_schema(task):
    ref = next((ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0").glob(f"{task}_0.json"))
    ref = json.loads(ref.read_text())
    layout = sample_layout(task, 3)
    assert set(layout) == set(ref)
    ref_labels = sorted(i["label"] for insts in ref["Rigid"].values() for i in insts)
    labels = sorted(i["label"] for insts in layout["Rigid"].values() for i in insts)
    assert labels == ref_labels
    keys = set(next(iter(ref["Rigid"].values()))[0])
    for insts in layout["Rigid"].values():
        for inst in insts:
            assert keys <= set(inst)
            x, y = inst["default_pos"][:2]
            assert inst["xlim"][0] - 1e-9 <= x <= inst["xlim"][1] + 1e-9
            assert inst["ylim"][0] - 1e-9 <= y <= inst["ylim"][1] + 1e-9


def test_layout_reproducible_and_varied():
    a, b, c = sample_layout("stack_blocks", 7), sample_layout("stack_blocks", 7), sample_layout("stack_blocks", 8)
    assert a == b and a != c


def test_prohibited_area_respected():
    sampler = LayoutSampler("stack_blocks")
    for seed in range(100):
        for insts in sampler.sample(seed)["Rigid"].values():
            for inst in insts:
                x, y = inst["default_pos"][:2]
                for x0, y0, x1, y1 in sampler.prohibited:
                    assert not (x0 < x < x1 and y0 < y < y1)


def _planner_module():
    spec = importlib.util.spec_from_file_location("expert_planner", ROOT / "src/expert/planner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retime_respects_limits():
    mod = _planner_module()
    p = mod.Planner.__new__(mod.Planner)
    path = np.linspace(np.zeros(6), [1.2, -0.4, 0.3, 0, 0, 2.0], 400)
    out = p._resample(path, np.zeros(6))
    steps = np.abs(np.diff(np.vstack([np.zeros(6), out]), axis=0)).max(1)
    assert steps.max() <= mod.FREE_VMAX + 1e-6
    assert np.abs(np.diff(steps)).max() <= mod.FREE_AMAX + 1e-6
    np.testing.assert_allclose(out[-1], path[-1], atol=1e-9)


def test_argus_labels_from_synthetic_episode(tmp_path):
    from src.expert.argus import build

    n = 60
    layout = sample_layout("stack_blocks", 0)
    labels = [i["label"] for insts in layout["Rigid"].values() for i in insts]
    objects = {
        lbl: np.tile(np.r_[0.1 * k, -0.15, 0.78, 1, 0, 0, 0], (n, 1)).astype(np.float32) for k, lbl in enumerate(labels)
    }
    objects[labels[1]][40:, :3] = [0.0, -0.15, 0.81]  # placed on block 0 at frame 40
    segs = [
        dict(primitive="approach", start_frame=0, end_frame=10, arm=1, object=labels[1], subtask=0, goal={}),
        dict(primitive="grasp", start_frame=10, end_frame=20, arm=1, object=labels[1], subtask=0, goal={}),
        dict(primitive="transfer", start_frame=20, end_frame=30, arm=1, object=labels[1], subtask=0, goal={}),
        dict(primitive="lower", start_frame=30, end_frame=35, arm=1, object=labels[1], subtask=0, goal={}),
        dict(primitive="release", start_frame=35, end_frame=40, arm=1, object=labels[1], subtask=0, goal={}),
        dict(primitive="home", start_frame=40, end_frame=60, arm=1, object=None, subtask=0, goal={}),
    ]
    path = tmp_path / "ep.h5"
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(
            {
                "task": "stack_blocks",
                "seed": 0,
                "num_frames": n,
                "success": True,
                "instruction": "Stack the three blocks with different textures.",
            }
        )
        f.attrs["layout"] = json.dumps(layout)
        f.attrs["segments"] = json.dumps(segs)
        held = np.full((n, 2), "", dtype=object)
        held[20:40, 1] = labels[1]
        f.create_dataset("held", data=held.astype(h5py.string_dtype()))
        g = f.create_group("objects")
        for lbl, v in objects.items():
            g.create_dataset(lbl, data=v)
    out = build(path)
    lab = out["labels"]
    assert set(lab) >= {
        "scene",
        "timeline",
        "key_events",
        "state_changes",
        "completion",
        "scene_graph",
        "instruction_variants",
    }
    actions = [e["action"] for e in lab["timeline"]]
    assert actions[:3] == ["approach", "grasp", "lift and transport"]
    assert lab["timeline"][-1]["contribution"] == "idle"
    assert any(e["kind"] == "subgoal_complete" and "placed on" in e["label"] for e in lab["key_events"])
    assert lab["completion"]["task_completed"] == "success"
    assert all(e["end_s"] >= e["start_s"] for e in lab["timeline"])


def test_visual_randomization_keeps_object_layout():
    plain = sample_layout("stack_blocks", 11)
    varied = sample_layout("stack_blocks", 11, table_randomization=1.0)
    assert plain["Rigid"] == varied["Rigid"]
    assert plain["Table"]["default"] != varied["Table"]["default"] or plain["Ground"] != varied["Ground"]
