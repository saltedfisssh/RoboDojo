"""Independent training layouts for peg, hinge, and desk manipulation."""

import numpy as np
from transforms3d.euler import euler2quat

from src.datagen.layouts import metadata

TASKS = (
    "play_stacking_toy",
    "organize_table",
    "hang_mugs",
    "fill_egg_holder",
    "make_toast",
    "insert_key",
    "store_laptop_and_headphones",
)


def sample(task, rng, layout):
    from src.datagen.curriculum import add_object as add

    layout["Rigid"] = {}
    if task == "play_stacking_toy":
        add(layout, "stack_base", 0, "stack_base", [0, -0.12], kind="Geometry", quaternion=[1, 0, 0, 0])
        positions = [
            (-0.30, -0.12),
            (-0.30, -0.23),
            (-0.30, -0.34),
            (-0.18, -0.34),
            (-0.18, -0.23),
            (-0.07, -0.34),
            (0.07, -0.34),
            (0.18, -0.34),
            (0.30, -0.34),
            (0.30, -0.20),
        ]
        models = rng.permutation(4)
        for i, xy in enumerate(positions):
            group = 0 if i < 4 else 1 if i < 7 else 2 if i < 9 else 3
            add(
                layout,
                "stackingblocks",
                int(models[group]),
                f"block{i}",
                np.asarray(xy) + rng.uniform(-0.003, 0.003, 2),
                quaternion=[1, 0, 0, 0],
            )
    elif task == "organize_table":
        for cat, idx, label, xy in [
            # Its front must clear the extended fingers in the home pose.
            ("drawer", 0, "drawer", (-0.35, -0.08)),
            # The drawer reaches X=-0.163. The old cushion at X=-0.16
            # overlapped its footprint and left no room for the figure/jaws.
            ("cube_cushion", 2, "cube_cushion", (-0.08, -0.10)),
            ("mousemat", 0, "mousemat", (0.30, -0.10)),
            ("frame", 0, "frame", (0.10, 0)),
            ("monitor", 0, "monitor", (0.10, 0.25)),
        ]:
            add(layout, cat, idx, label, xy, kind="Geometry")
        for cat, idx, xy in [
            ("alarm", 0, (-0.25, -0.30)),
            ("garage", 8, (-0.13, -0.32)),
            ("keyboard", 0, (0.10, -0.23)),
            ("mouse", 4, (0.33, -0.34)),
        ]:
            add(layout, cat, idx, cat, np.asarray(xy) + rng.uniform(-0.003, 0.003, 2))
    elif task == "hang_mugs":
        add(layout, "cup_holder", 0, "cup_holder", [0, -0.10], kind="Geometry", quaternion=[1, 0, 0, 0])
        for i, xy in enumerate(((-0.27, -0.25), (0.27, -0.25), (-0.12, -0.36))):
            add(layout, "mug", 2, f"mug{i}", np.asarray(xy) + rng.uniform(-0.003, 0.003, 2), quaternion=[1, 0, 0, 0])
    elif task == "fill_egg_holder":
        basket = add(layout, "egg_basket", 0, "egg_basket", [-0.24, -0.23], kind="Geometry", quaternion=[1, 0, 0, 0])
        add(layout, "egg_holder", 0, "egg_holder", [0.0, -0.22], kind="Articulation", quaternion=[1, 0, 0, 0])
        for i, (dx, dy) in enumerate(((-0.035, -0.035), (0.035, -0.035), (-0.035, 0.035), (0.035, 0.035))):
            bottom = basket["default_pos"][2] - 0.046
            egg_min = np.asarray(metadata("egg", 0)["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
            add(
                layout,
                "egg",
                0,
                f"target{i}",
                [-0.24 + dx, -0.23 + dy],
                z=bottom - egg_min + 0.003,
                quaternion=[1, 0, 0, 0],
            )
    elif task == "make_toast":
        # Keep slices outside the left home gripper's swept volume as well
        # as in separate shelf slots, so settling does not knock them down.
        shelf = add(layout, "bread_shelf", 0, "bread_shelf", [-0.22, -0.33], kind="Geometry", quaternion=[1, 0, 0, 0])
        add(
            layout,
            "toaster",
            0,
            "toaster",
            [-0.07, -0.20],
            kind="Articulation",
            quaternion=euler2quat(0, 0, -np.pi / 2),
        )
        for i, dx in enumerate((0.035, 0.01, -0.015, -0.039)):
            add(
                layout,
                "bread",
                0,
                f"bread_{i}",
                [-0.22 + dx, -0.33 - 0.007],
                quaternion=euler2quat(0, np.pi / 2, 0),
                z=shelf["default_pos"][2] + 0.055,
            )
    elif task == "insert_key":
        add(
            layout, "key", 0, "key", [-0.20 + rng.uniform(-0.005, 0.005), -0.28], quaternion=euler2quat(np.pi / 2, 0, 0)
        )
        add(layout, "key_slot", 0, "slot", [0.17, -0.22], kind="Geometry", quaternion=[1, 0, 0, 0])
    elif task == "store_laptop_and_headphones":
        add(layout, "headset", 2, "headset", [-0.13, -0.32])
        add(layout, "headset_stand", 0, "stand", [-0.23, -0.15], kind="Geometry", quaternion=[1, 0, 0, 0])
        # Bring the side-grasp corridor in front of the fixed camera pole.
        # The rack sits outside the stand's footprint during upright insertion.
        stand = add(layout, "laptop_stand", 1, "laptop_stand", [0.14, -0.11], kind="Geometry", quaternion=[1, 0, 0, 0])
        stand_top = (
            stand["default_pos"][2]
            + np.asarray(metadata("laptop_stand", 1, "Geometry")["geometry"]["aligned_bbox"]["vertices"])[:, 2].max()
        )
        # The body-link mesh extends 3.34 mm below its root. The old fixed
        # 0.815 m spawn intersected the 0.845 m stand top by over 30 mm,
        # leaving the base underneath the support after settling.
        add(
            layout,
            "laptop",
            0,
            "laptop",
            [0.14, -0.19],
            kind="Articulation",
            quaternion=[1, 0, 0, 0],
            z=stand_top + 0.0065,
        )
        add(
            layout,
            "vertical_laptop_storage_rack",
            0,
            "vertical_laptop_storage_rack",
            [0.40, -0.07],
            kind="Geometry",
            quaternion=euler2quat(0, 0, np.pi / 2),
        )
    else:
        raise ValueError(task)
    from src.datagen.curriculum import jitter_groups

    groups = {
        "fill_egg_holder": [["egg_basket", "target0", "target1", "target2", "target3"]],
        "make_toast": [["bread_shelf", "bread_0", "bread_1", "bread_2", "bread_3"], ["toaster"]],
        "store_laptop_and_headphones": [["headset"], ["laptop", "laptop_stand"], ["vertical_laptop_storage_rack"]],
    }
    jitter_groups(layout, groups.get(task, []), rng)
    return layout
