"""Inspect task configurations and report planner/state support without starting Isaac."""

import argparse
import json

import yaml

from src.datagen.layouts import ROOT
from src.datagen.planner import SUPPORTED_TASKS

# These are engineering assessments, not claims of task success.
PRECISION = {
    "deposit_coin",
    "fasten_screws",
    "hang_mugs",
    "insert_key",
    "insert_tubes",
    "plug_in_charger",
    "play_stacking_toy",
}
CONTACT = {"align_blocks", "push_T", "sweep_blocks", "play_Xylophone"}
BIMANUAL = {"fill_pen_holder", "put_bottles_into_dustbin", "store_tools_in_toolbox"}
INTERACTIVE = {"imitate_sorting_sequence", "make_kong", "play_tic_tac_toe"}


def inventory():
    result = []
    for path in sorted((ROOT / "task/RoboDojo/config").glob("*.yml")):
        if path.name.startswith("_"):
            continue
        cfg = yaml.safe_load(path.read_text()) or {}
        task = path.stem
        base = task.removesuffix("_random")
        kinds = [k for k in ("Rigid", "Geometry", "Articulation", "Dynamic", "Garment", "Fluid") if cfg.get(k)]
        if "Garment" in kinds:
            difficulty, gap = "very_high", "Particle/node state, cloth self-contact and a folding policy"
        elif "Fluid" in kinds:
            difficulty, gap = "very_high", "Particle state and emitter/system parameters; pouring feedback"
        elif "Dynamic" in kinds:
            difficulty, gap = "high", "Conveyor body state, clock, spawn schedule and moving-target interception"
        elif base in INTERACTIVE:
            difficulty, gap = "high", "Support-arm script queues, interaction phase and memory state"
        elif "Articulation" in kinds:
            difficulty, gap = "high", "Object joint state, drive targets and task-specific contact skills"
        elif base in PRECISION:
            difficulty, gap = "high", "Functional grasp/insertion frames and constrained Cartesian/contact control"
        elif base in CONTACT:
            difficulty, gap = "high", "Tool grasp and sustained nonprehensile contact; completion predicates"
        elif base in BIMANUAL:
            difficulty, gap = "medium_high", "Coordinated two-arm grasp/handover and obstacle checks"
        else:
            difficulty, gap = "medium", "Asset-specific grasps, goal poses and task ordering"
        supported = task in SUPPORTED_TASKS
        if not supported:
            adapter = "rigid_candidate" if set(kinds) <= {"Rigid", "Geometry"} else "requires_adapter"
        elif base in INTERACTIVE:
            adapter = "interactive_experimental"
        elif any(k in kinds for k in ("Garment", "Fluid", "Dynamic")):
            adapter = "particle_dynamic_experimental"
        elif "Articulation" in kinds:
            adapter = "articulated_experimental"
        else:
            adapter = "rigid"
        result.append(
            {
                "task": task,
                "object_kinds": kinds,
                "planner": "implemented" if supported else "not_implemented",
                "state_adapter": adapter,
                "difficulty": difficulty,
                "remaining_work": gap,
                "validation": "See docs/datagen/VALIDATION.md; implementation alone is not validation",
            }
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    records = inventory()
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for row in records:
            print(f"{row['task']:42} {row['planner']:16} {row['state_adapter']:18} {row['difficulty']}")


if __name__ == "__main__":
    main()
