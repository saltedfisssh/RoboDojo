"""Task-specific training layouts, independent of evaluation layouts."""

from copy import deepcopy

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from src.datagen.assembly_layouts import TASKS as ASSEMBLY_TASKS
from src.datagen.layouts import metadata
from src.datagen.particle_layouts import TASKS as PARTICLE_TASKS
from src.datagen.social_layouts import TASKS as SOCIAL_TASKS

TASKS = (
    (
        "arrange_largest_number",
        "classify_objects",
        "cover_blocks",
        "stack_bowls",
        "display_letters",
        "fasten_screws",
        "insert_tubes",
        "plug_in_charger",
        "deposit_coin",
        "push_T",
        "press_by_number",
        "build_tower",
        "put_bottles_into_dustbin",
        "pour_balls_into_vase",
        "pack_objects_into_box",
        "play_Xylophone",
        "swap_blocks",
        "play_tic_tac_toe",
        "fill_pen_holder",
    )
    + ASSEMBLY_TASKS
    + PARTICLE_TASKS
    + SOCIAL_TASKS
)


def add_object(layout, category, index, label, xy, *, kind="Rigid", quaternion=None, z=None, physics=None):
    data = metadata(category, index, kind)
    if quaternion is None:
        quaternion = data.get("active", {}).get("place", {}).get("up", {}).get("orientation", [1, 0, 0, 0])
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ quat2mat(quaternion).T
    record = {
        "category": category,
        "category_idx": int(index),
        "label": label,
        "default_pos": [*map(float, xy), float(0.767 - vertices[:, 2].min() if z is None else z)],
        "default_ori": quaternion.tolist(),
        "scale": [1, 1, 1],
        "physics": {**data.get("physics", {}), **(physics or {}), "type": kind.lower()},
        "visual": deepcopy(data.get("visual", {})),
        "need_check_stable": kind == "Rigid",
    }
    layout.setdefault(kind, {}).setdefault(category, []).append(record)
    return record


def jitter_groups(layout, groups, rng, bound=0.004, xy_mask=(1, 1)):
    """Vary supported fixtures together; never separate an object from its initial stand."""
    records = {
        rec["label"]: rec
        for kind in ("Rigid", "Geometry", "Articulation", "Garment", "Fluid", "Dynamic")
        for items in layout.get(kind, {}).values()
        for rec in items
    }
    for labels in groups:
        delta = rng.uniform(-bound, bound, 2) * xy_mask
        for label in labels:
            record = records[label]
            record["default_pos"][:2] = (np.asarray(record["default_pos"][:2]) + delta).tolist()


def sample(task, rng, layout):
    if task in SOCIAL_TASKS:
        from src.datagen.social_layouts import sample as sample_social

        return sample_social(task, rng, layout)
    if task in PARTICLE_TASKS:
        from src.datagen.particle_layouts import sample as sample_particles

        return sample_particles(task, rng, layout)
    if task in ASSEMBLY_TASKS:
        from src.datagen.assembly_layouts import sample as sample_assembly

        return sample_assembly(task, rng, layout)
    layout["Rigid"] = {}
    if task == "arrange_largest_number":
        values = sorted(rng.choice(np.arange(1, 10), 4, replace=False), reverse=True)
        texture = int(rng.integers(3)) * 10
        order = [1, 0, 3, 2]  # Every object must be moved; each arm orders its own pair.
        for i, j in enumerate(order):
            x = (-0.24, -0.10, 0.10, 0.24)[i]
            add_object(
                layout,
                "number",
                int(values[j]) + texture,
                f"digit_{i}",
                [x + rng.uniform(-0.008, 0.008), -0.35 + rng.uniform(-0.008, 0.008)],
                quaternion=[1, 0, 0, 0],
            )
        for i, x in enumerate((-0.24, -0.10, 0.10, 0.24)):
            add_object(layout, "cube_cushion", 1, f"mat_{i}", [x, -0.18], kind="Geometry")
    elif task == "classify_objects":
        # One item/category is a valid instance of the official 1..3 item sampler.
        choices = [("toy_car", 0), ("action_camera", 0), ("watch", 0)]
        for i, (category, index) in enumerate(choices):
            x = (-0.28, 0, 0.28)[i]
            add_object(layout, "basket", i + 2, f"basket{i}", [x, -0.12], kind="Geometry")
            add_object(
                layout,
                category,
                index,
                f"cat{i}_0",
                [x + rng.uniform(-0.025, 0.025), -0.36 + rng.uniform(-0.012, 0.012)],
            )
    elif task == "cover_blocks":
        colors = rng.permutation([("red", 4), ("green", 6), ("blue", 5)])
        for i, (color, index) in enumerate(colors):
            x = (-0.23, 0, 0.23)[i]
            add_object(layout, "cube", int(index), str(color), [x, -0.22], quaternion=[1, 0, 0, 0])
            add_object(layout, "cup", 1, f"cup{i}", [x + rng.uniform(-0.005, 0.005), -0.37], quaternion=[0, 1, 0, 0])
    elif task == "stack_bowls":
        # Model 1 has a deep rim that the X5 fingers can support. Shallow models
        # 9/10 tilt and destabilize the stack; keep them outside this curriculum.
        index = 1
        for i, x in enumerate((0, -0.24, 0.24)):
            add_object(
                layout,
                "bowl",
                index,
                f"bowl{i}",
                [x + rng.uniform(-0.008, 0.008), -0.28 + rng.uniform(-0.01, 0.01)],
                quaternion=[1, 0, 0, 0],
            )
    elif task == "display_letters":
        letters = ["r", "o", "b", "o", "d", "o", "j", "o"]
        permutation = np.r_[rng.permutation(4), rng.permutation(np.arange(4, 8))]
        for slot, letter_id in enumerate(permutation):
            add_object(
                layout,
                letters[letter_id],
                0,
                f"letter{letter_id}",
                [-0.28 + slot * 0.08 + rng.uniform(-0.003, 0.003), -0.35],
                quaternion=euler2quat(-np.pi / 2, 0, 0),
            )
    elif task == "fasten_screws":
        for i, model in enumerate(rng.choice(5, 3, replace=False)):
            x = (-0.24, 0, 0.24)[i]
            add_object(layout, "factory_nut", model, f"nut{i}", [x, -0.34], quaternion=[1, 0, 0, 0])
            add_object(layout, "factory_bolt", model, f"bolt{i}", [x, -0.17], kind="Geometry", quaternion=[1, 0, 0, 0])
    elif task == "insert_tubes":
        add_object(layout, "test_tube_slot", 0, "slot", [0, -0.12], kind="Geometry")
        for i, x in enumerate((-0.25, 0.08, 0.25)):
            add_object(layout, "test_tube", 2, f"tube{i}", [x, -0.38], quaternion=[1, 0, 0, 0])
    elif task == "plug_in_charger":
        x = float(rng.uniform(-0.25, -0.20))
        # Rest on the broad body face. Balancing on the plug pins toppled
        # during the approach in most seeds before the fingers could arrive.
        add_object(layout, "charger", 0, "charger", [x, -0.35], quaternion=[1, 0, 0, 0])
        add_object(layout, "socket", 0, "socket", [x, -0.15], quaternion=[1, 0, 0, 0])
    elif task == "deposit_coin":
        x = float(rng.uniform(-0.28, -0.23))
        stand = add_object(layout, "vertical_coin_stand", 0, "vertical_coin_stand", [x, -0.35], kind="Geometry")
        add_object(
            layout,
            "coin",
            0,
            "coin0",
            [x, -0.35],
            quaternion=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
            z=stand["default_pos"][2] + 0.0068,
        )
        add_object(layout, "piggy_bank", 0, "piggy_bank", [x, -0.12], kind="Geometry", quaternion=[1, 0, 0, 0])
    elif task == "push_T":
        x = float(rng.uniform(-0.23, -0.18))
        # Push along the stem, with both fingertips against the broad crossbar.
        # A transverse push contacts the recessed stem and protruding crossbar
        # at different times, inducing large rotation even at the nominal COM.
        orientation = euler2quat(0, 0, np.pi / 2)
        add_object(layout, "t", 0, "t", [x, -0.33], quaternion=orientation)
        add_object(layout, "t_cushion", 0, "target_t", [x, -0.16], kind="Geometry", quaternion=orientation)
    elif task == "press_by_number":
        for i, x in enumerate((-0.18, 0, 0.18)):
            add_object(layout, "SpringButton", 2 if i < 2 else 3, f"button{i}", [x, -0.23], kind="Articulation")
            if i < 2:
                add_object(
                    layout,
                    "number_card",
                    int(rng.integers(1, 4)),
                    f"num{i}",
                    [x, -0.08],
                    kind="Geometry",
                    quaternion=[0, 0, 0.42261, -0.90631],
                    z=0.805,
                )
    elif task == "build_tower":
        specs = [
            (0, (0, -0.29)),
            (2, (-0.27, -0.14)),
            (2, (0.27, -0.14)),
            # Keep the two top pieces within X5's vertical-lift workspace.
            # Their former back-row positions admitted grasp IK but not lift.
            (1, (-0.30, -0.23)),
            (3, (0.30, -0.23)),
            (2, (-0.27, -0.31)),
            (2, (0.27, -0.31)),
            # Keep the second wide board away from the front camera rail;
            # its open jaws span the board's narrow Y dimension at approach.
            (4, (0.15, -0.05)),
        ]
        for i, (model, xy) in enumerate(specs):
            add_object(
                layout,
                "block",
                model,
                f"block{i}",
                np.asarray(xy) + rng.uniform(-0.003, 0.003, 2),
                quaternion=euler2quat(np.pi / 2, 0, 0) if i == 4 else [1, 0, 0, 0],
            )
    elif task == "put_bottles_into_dustbin":
        # The tabletop ends at X=-0.70; the opening must be outside its edge.
        add_object(layout, "dustbin", 0, "dustbin", [-0.87, -0.40], kind="Geometry", z=0.35)
        for i, model in enumerate(rng.choice([5, 25, 41, 50, 59, 60], 4, replace=False)):
            # Upright bottles at x=-0.34 intersect the home gripper before
            # frame zero. Use disjoint horizontal bottles below that sweep.
            xy = [(-0.36, -0.12), (-0.12, -0.12), (-0.36, -0.30), (-0.12, -0.30)][i]
            add_object(layout, "bottle", model, f"bottle{i}", xy, quaternion=euler2quat(0, np.pi / 2, 0))
    elif task == "pour_balls_into_vase":
        x = float(rng.uniform(-0.26, -0.22))
        # The former Y=-0.35 slanted grasp folded link6 into link2 during
        # descent. Move the cup forward to keep the whole side approach clear.
        cup_y = -0.25
        cup = add_object(layout, "cup", 6, "cup", [x, cup_y], quaternion=[1, 0, 0, 0])
        # Leave room for both open fingers during the slanted side grasp.
        # At the same X the near side of the wide vase obstructed one jaw.
        add_object(layout, "vase", 2, "vase", [x + 0.15, -0.20], kind="Geometry", quaternion=[1, 0, 0, 0])
        for i in range(7):
            dx, dy = (
                (0.0, 0.0) if i == 0 else (0.015 * np.cos((i - 1) * np.pi / 3), 0.015 * np.sin((i - 1) * np.pi / 3))
            )
            sphere = add_object(
                layout,
                "sphere",
                0,
                f"sphere_{i}",
                [x + dx, cup_y + dy],
                z=cup["default_pos"][2] - 0.022,
                quaternion=[1, 0, 0, 0],
            )
            sphere["need_check_stable"] = False
    elif task == "pack_objects_into_box":
        # Keep the high shoe/hammer transfer within the same continuous IK
        # branch; at y=-0.10 the right elbow reached its joint limit.
        add_object(layout, "box", 0, "box", [0, -0.18], quaternion=euler2quat(0, 0, np.pi / 2))
        for category, model, xy in [
            ("car", 6, (-0.28, -0.32)),
            ("shoe", 4, (0.28, -0.32)),
            ("hammer", 3, (-0.28, -0.10)),
            ("electric_toothbrush", 0, (0.34, -0.10)),
        ]:
            add_object(
                layout,
                category,
                model,
                category,
                np.asarray(xy) + rng.uniform(-0.003, 0.003, 2),
                quaternion=euler2quat(0, -np.pi / 2, 0) if category == "electric_toothbrush" else [1, 0, 0, 0],
            )
    elif task == "play_Xylophone":
        # Leave the left wrist reach margin at the eighth key after the
        # slender mallet rolls within the grasp during earlier strikes.
        add_object(layout, "xylophone", 0, "xylophone", [-0.12, -0.07], kind="Geometry", quaternion=[0, 0, 0, 1])
        stand = add_object(layout, "mallet_stand", 0, "mallet_stand", [-0.30, -0.22])
        add_object(
            layout, "mallet", 0, "mallet", [-0.30, -0.22], quaternion=[0, 0, 0, 1], z=stand["default_pos"][2] + 0.011
        )
    elif task == "swap_blocks":
        mats = []
        for i, x in enumerate((-0.115, 0, 0.115)):
            mats.append(add_object(layout, "cube_cushion", 1, f"mat{i}", [x, -0.20], kind="Geometry"))
        for i, j in enumerate((0, 2)):
            cube = add_object(
                layout,
                "cube",
                3,
                f"target{i}",
                mats[j]["default_pos"][:2],
                quaternion=euler2quat(0, 0, rng.uniform(-0.15, 0.15)),
            )
            cube["default_pos"][2] += 0.006
            cube["relative_plane"] = f"mat{j}"
        button = add_object(layout, "SpringButton", 2, "button0", [0, -0.34], kind="Articulation")
        # Initialize the controller from the asset's spring target, rather than
        # the transient joint position captured during scene construction.
        button["physics"]["initialize_from_authored_drive_targets"] = True
    elif task == "play_tic_tac_toe":
        # Opponent templates are shipped task assets; use their documented fixed fixture coordinates.
        add_object(layout, "checkerboard", 0, "checkerboard", [0, -0.075], kind="Geometry")
        for i, x in enumerate((-0.2, -0.1, 0, 0.1, 0.2)):
            add_object(layout, "chessman", 0, f"player_piece{i}", [x, -0.25])
        for i, x in enumerate((-0.15, -0.05, 0.05, 0.15)):
            add_object(layout, "chessman", 1, f"opponent_piece{i}", [x, 0.2])
    elif task == "fill_pen_holder":
        add_object(layout, "pen_holder", 0, "pen_holder", [0.12, -0.16], quaternion=[1, 0, 0, 0])
        for i, (x, y) in enumerate(((-0.29, -0.36), (-0.10, -0.36), (-0.29, -0.15), (-0.10, -0.15))):
            category = "pen" if i < 2 else "oil_pen"
            add_object(
                layout,
                category,
                0,
                f"target{i}",
                [x, y],
                quaternion=[1, 0, 0, 0] if i < 2 else euler2quat(0, np.pi / 2, 0),
            )
    else:
        raise ValueError(task)
    groups = {
        "insert_tubes": [[f"tube{i}"] for i in range(3)],
        "play_Xylophone": [["mallet", "mallet_stand"]],
        "play_tic_tac_toe": [[f"player_piece{i}"] for i in range(5)],
        "fill_pen_holder": [["pen_holder"], *[[f"target{i}"] for i in range(4)]],
    }
    jitter_groups(layout, groups.get(task, []), rng)
    if task == "press_by_number":
        # Preserve the center button's arm assignment while varying reach distance.
        jitter_groups(layout, [["button0", "num0"], ["button1", "num1"], ["button2"]], rng, xy_mask=(0, 1))
    return layout
