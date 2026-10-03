"""Training fixtures for the original cloth, fluid and conveyor task definitions."""

import numpy as np

from src.datagen.layouts import metadata

TASKS = ("fold_clothes", "pour_liquid_into_cup", "match_and_pick_from_conveyor")


def sample(task, rng, layout):
    from src.datagen.curriculum import add_object as add

    layout["Rigid"] = {}
    if task == "fold_clothes":
        add(layout, "Top_Long", int(rng.choice([1, 4, 9])), "target", [rng.uniform(-0.01, 0.01), -0.20], kind="Garment")
    elif task == "pour_liquid_into_cup":
        bottle = add(layout, "wuliangye", 0, "bottle", [0.15, -0.27], quaternion=[1, 0, 0, 0])
        # Start with the cup clear of the bottle's neck-grasp approach.
        # The robot physically slides it to the supported pouring position
        # after lifting and staging the bottle. Keep the original asset,
        # fluid transform and reward, and record this training layout.
        add(layout, "mug", 15, "cup", [0.22, -0.18], quaternion=[1, 0, 0, 0])
        # Fill the bottle's lower body, initially below the neck and wholly inside the vessel.
        add(
            layout,
            "wuliangye",
            0,
            "wine",
            [0.15, -0.27],
            kind="Fluid",
            z=bottle["default_pos"][2] - 0.015,
            quaternion=[1, 0, 0, 0],
        )
    elif task == "match_and_pick_from_conveyor":
        add(layout, "conveyor", 0, "conveyor", [-2.08, 0.175], kind="Dynamic", z=0.06, quaternion=[0, 0, 0, 1])
        # Belt surface travels +X at 0.1 m/s. The remembered object leads its matching target.
        targets = [
            ("bottle", 47, "target0", -0.10),
            ("bottle", 47, "target1", -0.85),
            ("camera", 2, "other0", -0.55),
            ("phone", 0, "other1", -1.10),
            ("game_machine", 0, "other2", -1.40),
        ]
        for category, idx, label, x in targets:
            data = metadata(category, idx)
            q = data.get("active", {}).get("place", {}).get("up", {}).get("orientation", [1, 0, 0, 0])
            from transforms3d.quaternions import quat2mat

            vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ quat2mat(q).T
            record = add(layout, category, idx, label, [x, -0.08], z=0.781 - vertices[:, 2].min(), quaternion=q)
            record["need_check_stable"] = False
    else:
        raise ValueError(task)
    from src.datagen.curriculum import jitter_groups

    groups = {
        "pour_liquid_into_cup": [["bottle", "wine"], ["cup"]],
        "match_and_pick_from_conveyor": [[name] for name in ("target0", "target1", "other0", "other1", "other2")],
    }
    jitter_groups(layout, groups.get(task, []), rng)
    return layout
