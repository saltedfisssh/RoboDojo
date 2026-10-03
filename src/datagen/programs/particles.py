"""Robot-contact cloth folding, measured pouring, and moving-target interception."""

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.programs.assembly import top_rotation


def cloth_point(p, tag):
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "target"))
    indices = lm.get_instance_metadata(0, label="target")["passive"]["functional"][tag]["id"]
    points = obj.sample_mesh_vertices()[0]
    return np.asarray(points)[indices].mean(axis=0)


def cloth_pinch_target(p, tag, arm, *, jaw_yaw_deg=None, tilted_hem=False):
    point = cloth_point(p, tag)
    rotation = top_rotation(arm)
    lm = p.env.scene_manager.layout_manager
    tags = lm.get_instance_metadata(0, label="target")["passive"]["functional"]
    if f"{tag}_below" in tags:
        below = cloth_point(p, f"{tag}_below")
        # Enclose both layers of the sleeve seam. Gripping just its top vertex
        # lets the lower layer slide out while the stiff cloth peels away.
        delta = below[:2] - point[:2]
        point = (point + below) / 2
        if np.linalg.norm(delta) > 0.002:
            yaw = np.arctan2(delta[1], delta[0])
            rotation = euler2mat(0, 0, yaw) @ np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float)
    if jaw_yaw_deg is not None:
        rotation = euler2mat(0, 0, np.deg2rad(jaw_yaw_deg)) @ np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float)
    if tilted_hem:
        if not tag.endswith("_hem"):
            raise ValueError("Tilted hem grasp requires a hem functional point")
        # Keep the closing direction horizontal and move the wrists outward.
        # The vertical wrist pair intersects at link4 when both hems are held.
        yaw = np.pi / 2 if arm == 0 else -np.pi / 2
        tilt = np.deg2rad(-15 if arm == 0 else 15)
        rotation = euler2mat(0, tilt, 0) @ euler2mat(0, 0, yaw) @ top_rotation(arm)
    quat = mat2quat(rotation)
    # Cloth particles rest several millimeters above the table. Enclose the
    # edge down to 0.7 mm above the table instead of touching only the top of
    # the particles; the X5 collision tips extend 12.57 mm beyond the reference.
    table = p.env.datagen_env_config["scene"]["Table"]
    table_top = float(table["default_pos"][2]) + float(table["scale"][2]) / 2
    wrist = point.copy()
    if tilted_hem:
        wrist -= rotation[:, 0] * p.actions.robots[arm].gripper_bias
        # X5 link7/link8 mesh projection at the two 15-degree poses gives
        # 153.459 / 153.531 mm; round up so neither tip penetrates the table.
        wrist[2] = table_top + 0.0007 + 0.1536
    else:
        wrist[2] = table_top + 0.0007 + p.actions.robots[arm].gripper_bias + 0.01257
    return wrist, quat, point


def pinch_cloth(p, tag, arm, *, jaw_yaw_deg=None, inboard_offset=0.0, tilted_hem=False):
    wrist, quat, point = cloth_pinch_target(p, tag, arm, jaw_yaw_deg=jaw_yaw_deg, tilted_hem=tilted_hem)
    if inboard_offset:
        shoulder = cloth_point(p, f"{tag.split('_')[0]}_shoulder")
        direction = shoulder[:2] - point[:2]
        offset = direction / np.linalg.norm(direction) * inboard_offset
        wrist[:2] += offset
        point[:2] += offset
    p.stage("approach_cloth", f"Approach the {tag} of the garment.", arm=arm, object_label="target")
    p.move(arm, wrist + [0, 0, 0.07], quat)
    p.gripper(arm, 1, frames=6)
    p.linear(arm, wrist, quat, step=0.003)
    p.stage("pinch_cloth", f"Pinch the {tag} between the fingers.", arm=arm, object_label="target")
    for opening in np.linspace(1, 0, 26)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(10)
    return wrist, quat, point


def move_pair(p, targets, quats):
    from src.datagen.planner import PlanningFailure

    paths = []
    for arm in range(2):
        robot = p.actions.robots[arm]
        result = p.rm.planner[robot.robot_name].plan_path(
            p.rm.get_joint(robot)[0], np.r_[targets[arm], quats[arm]], robot.entity_origin_pose
        )
        if result["status"] != "Success":
            raise PlanningFailure("Bimanual cloth path could not be planned")
        trajectory = np.asarray(result["position"])
        indices = list(range(p.actions.ticks - 1, len(trajectory), p.actions.ticks))
        if not indices or indices[-1] != len(trajectory) - 1:
            indices.append(len(trajectory) - 1)
        paths.append(trajectory[indices])
    for i in range(max(map(len, paths))):
        command = p.actions.command.copy()
        for arm, path in enumerate(paths):
            command[arm * 7 : arm * 7 + 6] = path[min(i, len(path) - 1)]
        p.tick(command)
    p.hold(5)


def fold_held_sleeve(p, source, target, arm, quat):
    """Carry the pinched seam across the body inside the arm's reachable height."""
    from src.datagen.planner import PlanningFailure

    destination_point = cloth_point(p, target) + [0, 0, 0.025]
    destination = destination_point + [0, 0, p.actions.robots[arm].gripper_bias]
    p.stage("fold_sleeve", f"Fold the {source} over the {target}.", arm=arm, object_label="target")
    p.annotation["goal"]["garment_points"] = {source: {"position": destination_point.tolist()}}
    # The 60 mm overhead target lies beyond X5 reach when crossing the torso.
    # Slow Cartesian motion at 30 mm clearance preserves the sleeve grasp.
    p.linear(arm, destination + [0, 0, 0.03], quat, step=0.002)
    p.linear(arm, destination, quat, step=0.002)
    if source == "right_sleeve":
        release_supported_sleeve(p, arm)
    else:
        p.stage("release_cloth", f"Release the folded {source}.", arm=arm, object_label="target")
        p.gripper(arm, 1)
        p.linear(arm, destination + [0, 0, 0.03], quat, step=0.002)
        p.home(arm)
    # Match the official sleeve ranges after release. A closed gripper and
    # completed wrist trajectory do not establish that the cloth followed.
    delta = cloth_point(p, source) - cloth_point(p, target)
    x_lower, x_upper = (-0.125, 0.14) if source == "left_sleeve" else (-0.14, 0.125)
    parser = p.env.reward_manager.func_parser
    query = dict(env_idx=0, label="target", point_A=source, point_B=target)
    x_ok = parser.is_garment_pointA_close_to_pointB_by_x_range({**query, "x_lower": x_lower, "x_upper": x_upper})
    y_ok = parser.is_garment_pointA_close_to_pointB_by_y_range({**query, "y_lower": -0.125, "y_upper": 0.14})
    if not (x_ok and y_ok):
        raise PlanningFailure(f"{source} did not remain folded after release: offset={delta.tolist()}")


def release_supported_sleeve(p, arm=1):
    """Lower the sleeve onto the garment before gradually releasing its stored bend."""
    from src.datagen.planner import PlanningFailure

    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    wrist = ee[:3] - [0, 0, 0.018]
    table = p.env.datagen_env_config["scene"]["Table"]
    minimum = float(table["default_pos"][2]) + float(table["scale"][2]) / 2 + 0.0007 + 0.15757
    if wrist[2] < minimum:
        raise PlanningFailure("The sleeve release would put the extended fingers too close to the table")
    p.stage(
        "support_sleeve",
        "Lower the held sleeve close to the folded garment before release.",
        arm=arm,
        object_label="target",
    )
    p.linear(arm, wrist, ee[3:], step=0.001, settle=15)
    p.stage("release_cloth", "Gradually release the supported sleeve.", arm=arm, object_label="target")
    for opening in np.linspace(float(p.actions.command[arm * 7 + 6]), 1, 41)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(10)
    p.linear(arm, wrist + [0, 0, 0.050], ee[3:], step=0.002)
    p.home(arm)
    p.hold(10)


def fold(p):
    from src.datagen.planner import PlanningFailure

    for arm, (source, target) in enumerate((("left_sleeve", "right_chest"), ("right_sleeve", "left_chest"))):
        # Pinch inside both cuffs so both layers stay within the pads.
        # The equivalent 205-degree jaw direction avoids joint6's +pi limit
        # during the cross-body transfer; 25 degrees reaches that boundary.
        wrist, quat, point = pinch_cloth(
            p,
            source,
            arm,
            jaw_yaw_deg=205 if arm == 1 else None,
            inboard_offset=0.020,
        )
        p.stage("lift_cloth", f"Lift the pinched {source} from the table.", arm=arm, object_label="target")
        lift_points = {source: {"min_z": float(point[2] + 0.025)}}
        tags = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="target")["passive"]["functional"]
        if f"{source}_below" in tags:
            lift_points[f"{source}_below"] = {"min_z": float(cloth_point(p, f"{source}_below")[2] + 0.025)}
        p.annotation["goal"]["garment_points"] = lift_points
        p.linear(arm, wrist + [0, 0, 0.08], quat, step=0.003)
        if any(cloth_point(p, tag)[2] < goal["min_z"] for tag, goal in lift_points.items()):
            raise PlanningFailure(f"Cloth pinch did not lift both layers of {source}")
        fold_held_sleeve(p, source, target, arm, quat)
    fold_hem(p, tilted_wrists=True)


def fold_hem(p, *, tilted_wrists=False, inboard_offset=0.0):
    """Complete the bimanual body fold after both sleeves have been folded."""
    from src.datagen.planner import PlanningFailure

    # Put the right wrist above its hem before the left hand pins its corner.
    # This reduces arm overlap during the right-hand approach, which failed
    # to track its target with the left wrist already low in the previous run.
    wrist, quat, _ = cloth_pinch_target(p, "right_hem", 1, tilted_hem=tilted_wrists)
    p.stage("approach_hem", "Position the right hand above the garment hem.", arm=1, object_label="target")
    p.move(1, wrist + [0, 0, 0.07], quat)
    pinches = [
        pinch_cloth(p, f"{side}_hem", arm, tilted_hem=tilted_wrists, inboard_offset=inboard_offset)
        for arm, side in enumerate(("left", "right"))
    ]
    quats = [item[1] for item in pinches]
    p.stage("lift_hem", "Lift both lower garment corners together.", object_label="target")
    p.annotation["goal"]["garment_points"] = {
        f"{side}_hem": {"min_z": float(pinches[arm][2][2] + 0.025)} for arm, side in enumerate(("left", "right"))
    }
    move_pair(p, [item[0] + [0, 0, 0.10] for item in pinches], quats)
    for arm, side in enumerate(("left", "right")):
        if cloth_point(p, f"{side}_hem")[2] < pinches[arm][2][2] + 0.025:
            raise PlanningFailure(f"Cloth pinch did not lift the {side} hem")
    destinations = [
        cloth_point(p, f"{side}_shoulder")
        + [0, 0, 0.035]
        - quat2mat(quats[arm])[:, 0] * p.actions.robots[arm].gripper_bias
        for arm, side in enumerate(("left", "right"))
    ]
    p.stage("fold_body", "Fold the garment hem onto the shoulders with both hands.", object_label="target")
    p.annotation["goal"]["garment_points"] = {
        f"{side}_hem": {
            "position": (destinations[arm] + quat2mat(quats[arm])[:, 0] * p.actions.robots[arm].gripper_bias).tolist()
        }
        for arm, side in enumerate(("left", "right"))
    }
    move_pair(p, [d + [0, 0, 0.07] for d in destinations], quats)
    move_pair(p, destinations, quats)
    p.stage("release_cloth", "Release both corners of the folded garment.", object_label="target")
    command = p.actions.command.copy()
    command[[6, 13]] = 1
    for _ in range(15):
        p.tick(command)
    move_pair(p, [d + [0, 0, 0.10] for d in destinations], quats)
    p.home(0)
    p.home(1)


def pour(p, *, return_bottle=True):
    import json

    from pxr import Usd, UsdGeom

    from src.datagen.ik import plan_approach

    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    transform = UsdGeom.Xformable(fluid.point_instancer.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p.recorder.path.with_suffix(".fluid_transform.json").write_text(
        json.dumps(
            {
                "scope": "Live USD local-to-world transform; original reward remains unchanged",
                "matrix_row_vectors": np.asarray(transform).tolist(),
                "official_init_pos": np.asarray(fluid.init_pos).tolist(),
                "official_init_ori": np.asarray(fluid.init_ori).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
    p.stage("clear_table", "Raise the empty right gripper before approaching the bottle.", arm=1)
    p.move(1, ee[:3] + [0, 0, 0.10], ee[3:])
    start, start_q = p.object_pose("bottle")
    # A horizontal pinch of the narrow neck keeps both pads between the
    # shoulder and collar. Its higher grasp clears the low wrist position
    # that prevented the earlier horizontal body grasp.
    rotation = euler2mat(0, np.pi, 0)
    # The official bottle has zero friction. Its 82 mm body slips from
    # the nearly fully open jaws beyond horizontal. Enclose the narrow
    # neck between the shoulder and collar instead, without changing friction.
    # The rear portions of the pads extend below their tip center. At 71 mm
    # they hit the shoulder during approach, tilting the bottle before closure.
    grasp = start + quat2mat(start_q) @ [0, 0, 0.085]
    robot = p.actions.robots[1]
    approach = grasp - rotation[:, 0] * (robot.gripper_bias + 0.07)
    result = plan_approach(
        p.rm.planner[robot.robot_name],
        p.rm.get_joint(robot)[0],
        np.r_[approach, mat2quat(rotation)],
        robot.entity_origin_pose,
    )
    arm, quat = 1, mat2quat(rotation)
    wrist = grasp - rotation[:, 0] * robot.gripper_bias
    p.stage("preshape", "Open the fingers to enclose the bottle neck.", arm=arm, object_label="bottle")
    p.gripper(arm, 0.75, frames=8)
    p.stage("approach", "Approach the bottle neck from the side.", arm=arm, object_label="bottle")
    p.move(arm, approach, quat, result=result)
    p.stage("align_grasp", "Align both pads between the bottle shoulder and collar.", arm=arm, object_label="bottle")
    p.linear(arm, wrist, quat)
    p.stage("grasp", "Close the fingers around the bottle neck.", arm=arm, object_label="bottle")
    p.gripper(arm, 0)
    lift_held_liquid(p)
    reorient_liquid_neck(p)
    # Re-measure the held transform at each phase. Brace before fluid exits,
    # and use a lower final tilt that keeps the right elbow unfolded.
    shared = dict(tilt_axis=(-1, 1, 0), cup_clearance_m=0.035, feedback_grasp=True, return_bottle=False)
    phases = []
    phases.append(pour_held_liquid(p, arm, mouth_xy_offset=(-0.025, 0.0), transfer_only=True, **shared))
    phases.append(
        pour_held_liquid(p, arm, mouth_xy_offset=(-0.005, 0.015), max_tilt_deg=78.125, final_hold_frames=0, **shared)
    )
    brace_liquid_cup(p)
    phases.append(
        pour_held_liquid(
            p,
            arm,
            transfer_tilt_deg=-78.125,
            mouth_xy_offset=(0.025, 0.023),
            max_tilt_deg=115,
            final_hold_frames=0,
            track_cup=True,
            **shared,
        )
    )
    wait_for_contained_liquid(p)
    if return_bottle:
        phases.append(upright_poured_liquid(p, arm, start, start_q, start_tilt_deg=115, release_cup_before_return=True))
    return {"phases": phases}


def brace_liquid_cup(p):
    """Support the cup on the table with a measured, bounded left-hand grasp."""
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    position, quaternion = p.object_pose("cup")
    rotation = euler2mat(0, np.pi / 4, -np.pi / 12) @ np.diag([1, -1, -1])
    robot = p.actions.robots[0]
    wrist = position + quat2mat(quaternion) @ [-0.015, 0, 0.005] - rotation[:, 0] * robot.gripper_bias
    quat = mat2quat(rotation)
    approach = wrist - rotation[:, 0] * 0.065
    result = plan_approach(
        p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
    )
    p.stage("brace_cup", "Enclose the cup body while keeping its base on the table.", arm=0, object_label="cup")
    p.gripper(0, 1, frames=5)
    p.move(0, approach, quat, result=result)
    p.linear(0, wrist, quat, step=0.002)
    for opening in np.linspace(1, 0.48, 41)[1:]:
        p.gripper(0, float(opening), frames=1)
    p.hold(30)
    after, after_q = p.object_pose("cup")
    drift = float(np.linalg.norm(after - position))
    tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(after_q)[2, 2], -1, 1))))
    if drift > 0.006 or tilt > 5:
        raise PlanningFailure(f"The cup brace displaced or tilted the cup: drift={drift:.5f} m, tilt={tilt:.2f} deg")


def wait_for_contained_liquid(p, max_frames=400):
    from src.datagen.planner import PlanningFailure

    checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    if len(checks) != 1:
        raise PlanningFailure("Expected one official fluid containment predicate")
    p.stage("settle_flow", "Keep the bottle tilted until the poured liquid settles in the cup.", arm=1)
    p.annotation["goal"]["fluid_in_container"] = dict(checks[0][1])
    stable = 0
    for _ in range(max_frames):
        p.hold(1)
        stable = stable + 1 if p.env.reward_manager.check_once(checks[0], 0) else 0
        if stable >= 20:
            return
    raise PlanningFailure("The liquid did not sustain the original containment and residual predicates")


def release_liquid_cup(p):
    p.stage(
        "release_cup", "Release the supported cup before returning the bottle beside it.", arm=0, object_label="cup"
    )
    for opening in np.linspace(float(p.actions.command[6]), 1, 31)[1:]:
        p.gripper(0, float(opening), frames=1)
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    p.stage("withdraw_cup_hand", "Withdraw the left fingers clear of the cup.", arm=0)
    p.linear(0, ee[:3] - quat2mat(ee[3:])[:, 0] * 0.07, ee[3:], step=0.002)
    p.home(0)


def reorient_liquid_neck(p):
    """Support the neck collar at the measured wrist tilt before carrying the bottle."""
    from src.datagen.planner import PlanningFailure

    position, _ = p.object_pose("bottle")
    rotation = euler2mat(0, np.pi - np.deg2rad(38), 0)
    wrist = position + [0, 0, 0.090] - rotation[:, 0] * p.actions.robots[1].gripper_bias
    p.stage(
        "reorient_neck_grasp",
        "Angle the wrist while supporting the bottle beneath its neck collar.",
        arm=1,
        object_label="bottle",
    )
    p.linear(1, wrist, mat2quat(rotation), step=0.002, angular_step=np.deg2rad(1), settle=20)
    actual, _ = p.object_pose("bottle")
    if np.linalg.norm(actual - position) > 0.030:
        raise PlanningFailure("The bottle slipped out of the neck grasp during wrist reorientation")


def lift_held_liquid(p, *, lift_height=0.12, tilt_deg=-10.0):
    """Lift the neck grasp along a continuous path clear of link3/link5."""
    from src.datagen.planner import PlanningFailure

    arm = 1
    position, quaternion = p.object_pose("bottle")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    delta = euler2mat(0, np.deg2rad(tilt_deg), 0)
    root = position + [0, 0, lift_height]
    wrist = root + delta @ (ee[:3] - position)
    quat = mat2quat(delta @ quat2mat(ee[3:]))
    p.stage("lift", "Lift the bottle by its neck with a small controlled tilt.", arm=arm, object_label="bottle")
    p.annotation["goal"]["object_min_z"] = float(position[2] + 0.08)
    p.linear(arm, wrist, quat, step=0.002, angular_step=np.deg2rad(1), settle=10)
    actual, actual_q = p.object_pose("bottle")
    expected_q = mat2quat(delta @ quat2mat(quaternion))
    error = float(np.linalg.norm(actual - root))
    dot = abs(np.dot(actual_q / np.linalg.norm(actual_q), expected_q))
    angle_error = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
    if actual[2] < position[2] + 0.08 or error > 0.03 or angle_error > 20:
        raise PlanningFailure(
            f"Bottle neck lift slipped: dz={actual[2] - position[2]:.4f}, "
            f"position error={error:.4f}, angle error={angle_error:.2f}"
        )
    return {"lift_m": float(actual[2] - position[2]), "position_error_m": error, "angle_error_deg": angle_error}


def pour_held_liquid(
    p,
    arm,
    start=None,
    start_q=None,
    *,
    return_bottle=True,
    transfer_tilt_deg=0.0,
    tilt_axis="y",
    cup_clearance_m=0.010,
    feedback_grasp=False,
    mouth_xy_offset=(0.015, 0.0),
    mouth_height_m=None,
    max_tilt_deg=125.0,
    track_cup=False,
    final_hold_frames=100,
    transfer_only=False,
):
    """Keep the bottle clear of the cup while rotating a measured grasp."""
    from transforms3d.axangles import axangle2mat

    from src.datagen.planner import PlanningFailure

    if isinstance(tilt_axis, str):
        if tilt_axis not in ("x", "y"):
            raise ValueError("Liquid tilt axis must be x, y, or a finite 3-vector")
        axis = np.array([1, 0, 0] if tilt_axis == "x" else [0, 1, 0], dtype=float)
    else:
        axis = np.asarray(tilt_axis, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
            raise ValueError("Liquid tilt axis must be x, y, or a finite nonzero 3-vector")
        axis = axis / np.linalg.norm(axis)
    if return_bottle and (start is None or start_q is None):
        raise ValueError("Returning the bottle requires its original pose")
    if mouth_height_m is None:
        mouth_height_m = cup_clearance_m + 0.020
    if not np.isfinite(mouth_height_m) or mouth_height_m <= 0:
        raise ValueError("Bottle mouth clearance must be finite and positive")
    cup, _ = p.object_pose("cup")
    cup_top = p.object_vertices("cup")[:, 2].max()
    vertices = np.asarray(
        p.env.scene_manager.layout_manager.get_instance_metadata(0, label="bottle")["geometry"]["aligned_bbox"][
            "vertices"
        ]
    )
    mouth_local = np.array([0, 0, 0.153])
    mouth = cup + [*mouth_xy_offset, 0]
    mouth[2] = cup_top + mouth_height_m
    pos, q = p.object_pose("bottle")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    tool_local = quat2mat(q).T @ quat2mat(ee[3:])
    offset = quat2mat(q).T @ (ee[:3] - pos)
    errors = []

    def tilt_to(angle, settle=2):
        nonlocal cup, cup_top, mouth
        if track_cup:
            cup, _ = p.object_pose("cup")
            cup_top = p.object_vertices("cup")[:, 2].max()
            mouth = cup + [*mouth_xy_offset, 0]
            mouth[2] = cup_top + mouth_height_m
        rotation = axangle2mat(axis, angle)
        root = mouth - rotation @ mouth_local
        # An arbitrary root-height clamp let the bottle body enter the cup
        # while still upright. Clear the cup rim with the whole rotated bbox.
        root[2] = max(root[2], cup_top + cup_clearance_m - (vertices @ rotation.T)[:, 2].min())
        quat = mat2quat(rotation @ tool_local)
        p.annotation["goal"].update(object_position=root.tolist(), object_quaternion=mat2quat(rotation).tolist())
        if feedback_grasp:
            from src.datagen.ik import cartesian_waypoints

            initial_pos, initial_q = p.object_pose("bottle")
            # Re-measure the held transform between short object-space
            # waypoints. Keep the original final target and slip thresholds;
            # a sliding bottle must still reach the specified pose.
            for pose in cartesian_waypoints(
                np.r_[initial_pos, initial_q], root, mat2quat(rotation), 0.008, np.deg2rad(3)
            ):
                actual, actual_q = p.object_pose("bottle")
                ee_now = p.rm.get_real_endpose(p.actions.robots[arm])[0]
                delta = quat2mat(pose[3:]) @ quat2mat(actual_q).T
                quat = mat2quat(delta @ quat2mat(ee_now[3:]))
                p.linear(
                    arm,
                    pose[:3] + delta @ (ee_now[:3] - actual),
                    quat,
                    step=0.0015,
                    angular_step=np.deg2rad(1),
                    settle=2,
                )
                reached, reached_q = p.object_pose("bottle")
                dot = abs(np.dot(reached_q / np.linalg.norm(reached_q), pose[3:]))
                if np.linalg.norm(reached - pose[:3]) > 0.03 or 2 * np.arccos(np.clip(dot, 0, 1)) > np.deg2rad(20):
                    raise PlanningFailure("Bottle slipped at a measured transfer/pour waypoint")
            p.hold(settle)
        else:
            p.linear(arm, root + rotation @ offset, quat, step=0.002, angular_step=np.deg2rad(1), settle=settle)
        actual, actual_q = p.object_pose("bottle")
        error = float(np.linalg.norm(actual - root))
        dot = abs(np.dot(actual_q / np.linalg.norm(actual_q), mat2quat(rotation)))
        angle_error = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
        cup_now, _ = p.object_pose("cup")
        errors.append({"tilt_deg": float(np.rad2deg(angle)), "position_error_m": error, "angle_error_deg": angle_error})
        if error > 0.03 or angle_error > 20:
            raise PlanningFailure(f"Bottle slipped during pouring: {error:.4f} m, {angle_error:.2f} degrees")
        if np.linalg.norm(cup_now[:2] - cup[:2]) > 0.006:
            raise PlanningFailure("The cup moved during pouring; the saved mouth target is no longer valid")
        return quat

    p.stage("transfer", "Carry the bottle above the cup rim.", arm=arm, object_label="bottle")
    transfer_angle = np.deg2rad(transfer_tilt_deg)
    tilt_to(transfer_angle, settle=10)
    if transfer_only:
        return {"bottle_tracking": errors}
    pos, q = p.object_pose("bottle")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    tool_local = quat2mat(q).T @ quat2mat(ee[3:])
    offset = quat2mat(q).T @ (ee[:3] - pos)
    p.stage("pour", "Tilt the bottle over the cup and let the liquid flow.", arm=arm, object_label="bottle")
    fluid_checks = [params for name, params in p.terminal_checks if name == "is_A_fluid_in_B"]
    if len(fluid_checks) != 1:
        raise PlanningFailure("Pouring requires one unambiguous official fluid containment predicate")
    p.annotation["goal"]["fluid_in_container"] = dict(fluid_checks[0])
    angles = np.linspace(transfer_angle, -np.deg2rad(max_tilt_deg), 33)
    for angle in angles:
        tilt_to(angle)
    p.hold(final_hold_frames)
    if not return_bottle:
        return {"bottle_tracking": errors}
    p.stage("stop_pouring", "Reduce the bottle tilt to stop pouring.", arm=arm, object_label="bottle")
    for angle in angles[-2::-1]:
        quat = tilt_to(angle)
    p.place("bottle", start, arm, quat, target_quaternion=start_q, contact=True, approach_clearance=0.04)
    p.home(arm)
    return {"bottle_tracking": errors}


def upright_poured_liquid(
    p,
    arm,
    start,
    start_q,
    *,
    start_tilt_deg=125.0,
    wait_before_upright=False,
    mouth_height_m=0.055,
    release_cup_before_return=False,
    place_after_upright=True,
):
    """Return a diagonal neck grasp after the liquid has finished entering the cup."""
    from transforms3d.axangles import axangle2mat

    from src.datagen.planner import PlanningFailure

    if not np.isfinite(mouth_height_m) or mouth_height_m <= 0:
        raise ValueError("Bottle mouth clearance must be finite and positive")
    cup, _ = p.object_pose("cup")
    cup_top = p.object_vertices("cup")[:, 2].max()
    mouth = cup + [0.025, 0.023, 0]
    mouth[2] = cup_top + mouth_height_m
    vertices = np.asarray(
        p.env.scene_manager.layout_manager.get_instance_metadata(0, label="bottle")["geometry"]["aligned_bbox"][
            "vertices"
        ]
    )
    p.stage(
        "stop_pouring",
        "Bring the emptied bottle upright while keeping its mouth above the cup.",
        arm=arm,
        object_label="bottle",
    )
    errors = []
    angles = -np.deg2rad(np.arange(start_tilt_deg, -0.01, -2.5))
    if wait_before_upright:
        angles = np.sort(np.unique(np.r_[angles, -np.deg2rad(32.5), 0.0]))
    fluid_checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    for angle in angles:
        rotation = axangle2mat(np.array([-1, 1, 0]) / np.sqrt(2), angle)
        root = mouth - rotation @ [0, 0, 0.153]
        root[2] = max(root[2], cup_top + 0.035 - (vertices @ rotation.T)[:, 2].min())
        position, quaternion = p.object_pose("bottle")
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        delta = rotation @ quat2mat(quaternion).T
        tool_q = mat2quat(delta @ quat2mat(ee[3:]))
        p.annotation["goal"].update(object_position=root.tolist(), object_quaternion=mat2quat(rotation).tolist())
        p.linear(arm, root + delta @ (ee[:3] - position), tool_q, step=0.002, angular_step=np.deg2rad(1), settle=2)
        actual, actual_q = p.object_pose("bottle")
        error = float(np.linalg.norm(actual - root))
        dot = abs(np.dot(actual_q / np.linalg.norm(actual_q), mat2quat(rotation)))
        angle_error = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
        if error > 0.03 or angle_error > 20:
            raise PlanningFailure(f"Bottle slipped while returning upright: {error:.4f} m, {angle_error:.2f} degrees")
        errors.append({"tilt_deg": float(np.rad2deg(angle)), "position_error_m": error, "angle_error_deg": angle_error})
        if wait_before_upright and np.isclose(angle, -np.deg2rad(32.5)):
            if len(fluid_checks) != 1:
                raise PlanningFailure("Expected one official fluid containment predicate")
            p.stage(
                "settle_before_upright",
                "Wait for the liquid to settle before bringing the bottle fully upright.",
                arm=arm,
                object_label="bottle",
            )
            stable = 0
            for _ in range(180):
                p.hold(1)
                stable = stable + 1 if p.env.reward_manager.check_once(fluid_checks[0], 0) else 0
                if stable >= 20:
                    break
            else:
                raise PlanningFailure("Liquid did not remain contained before the upright transition")
            p.stage("stop_pouring", "Finish bringing the bottle upright after the liquid settles.", arm=arm)
    if release_cup_before_return:
        release_liquid_cup(p)
    if not place_after_upright:
        return {"bottle_tracking": errors, "bottle_held_upright": True}
    p.place(
        "bottle",
        start,
        arm,
        tool_q,
        target_quaternion=start_q,
        contact=True,
        linear_transfer=True,
        precise=True,
        approach_clearance=0.04,
    )
    p.home(arm)
    return {"bottle_tracking": errors}


def intercept(p):
    from src.datagen.ik import solve_nearby
    from src.datagen.planner import PlanningFailure

    arm = 0
    robot = p.actions.robots[arm]
    # Close across the belt instead of along its travel direction.
    quat = mat2quat(euler2mat(0, np.pi / 2, 0))
    p.stage("remember", "Observe the first object on the moving conveyor.")
    p.hold(20)
    target, _ = p.object_pose("target1")
    vertices = p.object_vertices("target1")
    grasp_z = (vertices[:, 2].min() + vertices[:, 2].max()) / 2 + robot.gripper_bias
    p.stage("wait_matching_object", "Wait above the belt for the matching object.", arm=arm, object_label="target1")
    p.move(arm, [-0.38, target[1], grasp_z + 0.10], quat)
    for _ in range(600):
        if p.object_pose("target1")[0][0] >= -0.46:
            break
        p.hold(1)
    else:
        raise PlanningFailure("The conveyor did not bring the matching object into reach")
    initial_height = p.object_pose("target1")[0][2]
    lift_x = None
    phases = (("track_above", 20, 1), ("descend_moving", 25, 1), ("grasp_moving", 12, 0), ("lift_moving", 35, 0))
    for phase, frames, opening in phases:
        p.stage(
            phase,
            {
                "track_above": "Match the object's belt motion while keeping the open fingers above it.",
                "descend_moving": "Lower the open fingers around the matching object while tracking it.",
                "grasp_moving": "Close the fingers while following the matching object.",
                "lift_moving": "Lift the matching object off the moving belt.",
            }[phase],
            arm=arm,
            object_label="target1",
        )
        if phase == "lift_moving":
            p.annotation["goal"]["object_min_z"] = float(initial_height + 0.10)
        for i in range(frames):
            pos, _ = p.object_pose("target1")
            actual = p.rm.get_real_endpose(robot)[0]
            if phase == "track_above":
                z = grasp_z + 0.10
            elif phase == "descend_moving":
                z = grasp_z + 0.10 * (1 - (i + 1) / frames)
            elif phase == "grasp_moving":
                z = grasp_z
            else:
                # The 15 cm wrist target reached the arm's extension limit
                # after the object had already risen 13 cm. A 12.5 cm
                # lift leaves margin over the official 10 cm object test.
                z = grasp_z + 0.125 * (i + 1) / frames
                # A held object no longer follows the conveyor. Continuing
                # its 4 mm/frame feed-forward carries it out of left-arm reach.
                if lift_x is None and pos[2] > initial_height + 0.025:
                    lift_x = float(actual[0])
            desired = np.array([pos[0] + 0.004 if lift_x is None else lift_x, pos[1], z])
            delta = desired - actual[:3]
            desired = actual[:3] + delta * min(1, 0.012 / max(np.linalg.norm(delta), 1e-8))
            joints = p.rm.get_joint(robot)[0]
            solver = p.rm.planner[robot.robot_name]
            pose = np.r_[desired, quat]
            result = solver.solve_ik_to_joint(joints, pose, robot.entity_origin_pose)
            if result["status"] != "Success" or np.max(np.abs(result["joint_value"] - joints)) > 0.35:
                nearby = solve_nearby(solver, joints, pose, robot.entity_origin_pose)
                if nearby is None:
                    raise PlanningFailure("Moving-target interception lost a continuous IK solution")
                result["joint_value"] = nearby
            command = p.actions.command.copy()
            command[:6], command[6] = result["joint_value"], opening
            p.tick(command)
    if p.object_pose("target1")[0][2] < initial_height + 0.10:
        raise PlanningFailure("Matching object was not physically lifted from the belt")


def run(p, task):
    from src.datagen.programs.liquid import pour_with_supported_cup

    programs = {
        "fold_clothes": fold,
        "pour_liquid_into_cup": pour_with_supported_cup,
        "match_and_pick_from_conveyor": intercept,
    }
    programs[task](p)
