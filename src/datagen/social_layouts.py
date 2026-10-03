"""Independent layouts for support-arm demonstrations and tool sweeping."""

import numpy as np
from transforms3d.euler import euler2quat

from src.datagen.layouts import metadata

TASKS = ("imitate_sorting_sequence", "make_kong", "sweep_blocks")


def sample(task, rng, layout):
    from src.datagen.curriculum import add_object as add

    layout["Rigid"] = {}
    if task == "imitate_sorting_sequence":
        # The demo basket is 202 mm wide. At X=.30 its left wall overlaps
        # the open Panda fingers while grasping the rightmost figure.
        for i, xy in enumerate(((-0.31, -0.14), (0.42, 0.24))):
            add(layout, "basket", 5, f"basket{i}", xy, kind="Geometry", quaternion=euler2quat(0, 0, np.pi / 2))
        for i, (category, model) in enumerate(
            (("toy_car", 0), ("action_camera", 0), ("phone", 0), ("watch", 0), ("garage", 15))
        ):
            placements = metadata(category, model)["active"]["place"]
            orientation = placements.get("up", next(iter(placements.values())))["orientation"]
            add(layout, category, model, f"t{i}", [-0.25 + 0.12 * i, -0.34], quaternion=orientation)
            add(
                layout,
                category,
                model,
                f"aim{i}",
                [-0.25 + 0.105 * i, 0.20 + rng.uniform(-0.006, 0.006)],
                quaternion=orientation,
            )
    elif task == "make_kong":
        models = rng.choice(34, 7, replace=False)
        for group, count in enumerate((3, 3, 3, 3, 2)):
            for j in range(count):
                add(
                    layout,
                    "mahjong",
                    int(models[group]),
                    f"mahjong{group}_{j}",
                    [-0.325 + 0.046 * (group * 3 + j), -0.15],
                    quaternion=euler2quat(np.pi / 2, 0, 0),
                )
        for i, x in enumerate((0, 0.046, -0.046, -0.092)):
            add(
                layout,
                "mahjong",
                int(models[i]),
                f"mahjong{i + 5}_0",
                [x, 0.05],
                quaternion=metadata("mahjong", int(models[i]))["active"]["place"]["straight_back"]["orientation"],
            )
        for i in range(2):
            lower = add(
                layout,
                "mahjong",
                int(models[5 + i]),
                f"other{i}",
                [-0.4, -0.05 + 0.048 * i],
                quaternion=[0, 0.70710678, 0.70710678, 0],
            )
            label = "mahjong9_0" if i == 0 else "other2"
            add(
                layout,
                "mahjong",
                int(models[4] if i == 0 else models[6]),
                label,
                [-0.4, -0.05 + 0.048 * i],
                z=lower["default_pos"][2] + 0.033,
                quaternion=[0, 0.70710678, 0.70710678, 0],
            )
    elif task == "sweep_blocks":
        add(layout, "broom_shovel", 0, "broom_shovel", [-0.12, -0.18], quaternion=euler2quat(0, 0, np.pi / 2))
        add(layout, "broom", 0, "broom", [-0.30, -0.36], quaternion=euler2quat(0, np.pi / 2, 0))
        for i, (x, y) in enumerate(((0.10, -0.23), (0.15, -0.18), (0.10, -0.13))):
            add(layout, "small_cube", i, f"cube{i}", [x + rng.uniform(-0.005, 0.005), y], quaternion=[1, 0, 0, 0])
    else:
        raise ValueError(task)
    return layout
