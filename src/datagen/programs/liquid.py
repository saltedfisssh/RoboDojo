"""Physical cup relocation, supported pouring and bottle return.

All diagnostic suffixes through bottle return have passed. Full initial-state
collection independently verifies the composed program and all task predicates.
"""

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.programs.sweeping import _require_no_robot_pair_contacts


def liquid_stage_bottle_away_from_cup(p, *, offset=(0.14, 0, 0)):
    """v148 frame247: clear the left-hand cup path with a real right-hand translation."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_stage_bottle_away_from_cup requires pour_liquid_into_cup")
    position, quat = p.object_pose("bottle")
    cup, _ = p.object_pose("cup")
    target = position + np.asarray(offset)
    with contact_report(p):
        p.stage(
            "clear_cup_path",
            "Move the upright held bottle right to clear the cup-hand path.",
            arm=1,
            object_label="bottle",
        )
        move_held(p, "bottle", target, quat, 1, linear=True)
        p.hold(20)
        actual, actual_q = p.object_pose("bottle")
        error = float(np.linalg.norm(actual - target))
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(quat).T @ quat2mat(actual_q)) - 1) / 2, -1, 1))))
        cup_drift = float(np.linalg.norm(p.object_pose("cup")[0] - cup))
        if error > 0.008 or angle > 5 or cup_drift > 0.005:
            raise PlanningFailure(
                f"Bottle staging lost tracking: error={error:.5f}, angle={angle:.2f}, cup={cup_drift:.5f}"
            )
    return {"target": target, "position_error_m": error, "angle_error_deg": angle, "cup_drift_m": cup_drift}


def liquid_reposition_supported_cup(p, *, cup_dx=0.12):
    """v148 frame247: slide the empty cup into position after lifting the bottle."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held
    from src.datagen.programs.particles import brace_liquid_cup, release_liquid_cup

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_reposition_supported_cup requires pour_liquid_into_cup")
    before, quat = p.object_pose("cup")
    bottle, _ = p.object_pose("bottle")
    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    target = before.copy()
    target[:2] = np.asarray(fluid.init_pos)[:2] + [cup_dx, 0]
    with contact_report(p):
        brace_liquid_cup(p)
        p.stage(
            "position_cup",
            "Slide the empty cup beside the lifted bottle while supporting its base.",
            arm=0,
            object_label="cup",
        )
        for alpha in np.linspace(0, 1, 31)[1:]:
            waypoint = before + alpha * (target - before)
            move_held(p, "cup", waypoint, quat, 0, linear=True)
            actual, actual_q = p.object_pose("cup")
            drift = float(np.linalg.norm(actual - waypoint))
            tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(actual_q)[2, 2], -1, 1))))
            if drift > 0.008 or tilt > 5:
                raise PlanningFailure(f"Supported cup slide lost tracking: error={drift:.5f} m, tilt={tilt:.2f} deg")
        release_liquid_cup(p)
        p.hold(20)
        actual, actual_q = p.object_pose("cup")
        error = float(np.linalg.norm(actual - target))
        bottle_drift = float(np.linalg.norm(p.object_pose("bottle")[0] - bottle))
        if error > 0.008 or bottle_drift > 0.015:
            raise PlanningFailure(
                f"Cup positioning disturbed the held objects: cup={error:.5f}, bottle={bottle_drift:.5f}"
            )
    return {
        "cup_start": before,
        "cup_target": target,
        "cup_final": actual,
        "cup_error_m": error,
        "bottle_drift_m": bottle_drift,
    }


def liquid_restage_after_left_cup(p):
    """After moving the cup left, undo the low staging translation with the held bottle."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_restage_after_left_cup requires pour_liquid_into_cup")
    position, quat = p.object_pose("bottle")
    cup, _ = p.object_pose("cup")
    target = position + [-0.12, 0, 0.10]
    with contact_report(p):
        p.stage(
            "raise_staged_bottle",
            "Bring the held bottle back up after the cup has moved clear to the left.",
            arm=1,
            object_label="bottle",
        )
        move_held(p, "bottle", target, quat, 1, linear=True)
        p.hold(20)
        actual, actual_q = p.object_pose("bottle")
        error = float(np.linalg.norm(actual - target))
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(quat).T @ quat2mat(actual_q)) - 1) / 2, -1, 1))))
        cup_drift = float(np.linalg.norm(p.object_pose("cup")[0] - cup))
        if error > 0.008 or angle > 5 or cup_drift > 0.005:
            raise PlanningFailure(
                f"Bottle restaging lost tracking: error={error:.5f}, angle={angle:.2f}, cup={cup_drift:.5f}"
            )
    return {"position_error_m": error, "angle_error_deg": angle, "cup_drift_m": cup_drift}


def liquid_positioned_cup_pretilt(p):
    """After real cup relocation: transfer the held bottle and reach the pre-flow tilt."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_positioned_cup_pretilt requires pour_liquid_into_cup")
    shared = dict(tilt_axis=(-1, 1, 0), cup_clearance_m=0.035, feedback_grasp=True, return_bottle=False)
    with contact_report(p):
        transfer = pour_held_liquid(p, 1, mouth_xy_offset=(-0.025, 0.0), transfer_only=True, **shared)
        tilt = pour_held_liquid(
            p, 1, mouth_xy_offset=(-0.005, 0.015), max_tilt_deg=78.125, final_hold_frames=0, **shared
        )
    return {"transfer": transfer, "pretilt": tilt}


def _require_liquid_contact_clearance(path):
    import json

    from src.datagen.planner import PlanningFailure

    _require_no_robot_pair_contacts(path)
    for row in json.loads(path.read_text())["pairs"]:
        if any("/robot0/" in c for c in row["colliders"]) and any("/wuliangye/" in c for c in row["colliders"]):
            if row["max_impulse"] > 0 or row["min_separation_m"] <= 0:
                raise PlanningFailure("The cup-supporting hand contacted the bottle")


def _liquid_twisted_arc(
    p,
    angles,
    *,
    stage,
    azimuth_deg=-15,
    spin_deg=60,
    mouth_xy=(0.025, 0.023),
    mouth_height=0.055,
):
    from transforms3d.axangles import axangle2mat

    from src.datagen.ik import cartesian_waypoints
    from src.datagen.planner import PlanningFailure

    vertices = np.asarray(
        p.env.scene_manager.layout_manager.get_instance_metadata(
            0,
            label="bottle",
        )["geometry"]["aligned_bbox"]["vertices"]
    )
    axis = np.array([np.cos(np.deg2rad(azimuth_deg)), np.sin(np.deg2rad(azimuth_deg)), 0])
    spin = axangle2mat([0, 0, 1], np.deg2rad(spin_deg))
    cup_start = p.object_pose("cup")[0]
    checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    if len(checks) != 1:
        raise PlanningFailure("Expected one unchanged fluid containment predicate")
    p.stage(
        stage,
        "Keep the bottle mouth over the supported cup while rotating the held bottle.",
        arm=1,
        object_label="bottle",
    )
    if stage == "pour":
        p.annotation["goal"]["fluid_in_container"] = dict(checks[0][1])
    errors = []
    for angle in angles:
        cup, cup_q = p.object_pose("cup")
        cup_tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(cup_q)[2, 2], -1, 1))))
        if np.linalg.norm(cup[:2] - cup_start[:2]) > 0.006 or cup_tilt > 5:
            raise PlanningFailure("The supported cup moved during the twisted pour")
        top = p.object_vertices("cup")[:, 2].max()
        rotation = axangle2mat(axis, np.deg2rad(angle)) @ spin
        mouth = cup + [*mouth_xy, 0]
        mouth[2] = top + mouth_height
        root = mouth - rotation @ [0, 0, 0.153]
        root[2] = max(root[2], top + 0.035 - (vertices @ rotation.T)[:, 2].min())
        p.annotation["goal"].update(object_position=root.tolist(), object_quaternion=mat2quat(rotation).tolist())
        start = np.concatenate(p.object_pose("bottle"))
        for wp in cartesian_waypoints(start, root, mat2quat(rotation), 0.006, np.deg2rad(2)):
            actual, actual_q = p.object_pose("bottle")
            ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
            delta = quat2mat(wp[3:]) @ quat2mat(actual_q).T
            p.linear(
                1,
                wp[:3] + delta @ (ee[:3] - actual),
                mat2quat(delta @ quat2mat(ee[3:])),
                step=0.0015,
                angular_step=np.deg2rad(1),
                settle=2,
            )
            reached, reached_q = p.object_pose("bottle")
            distance = float(np.linalg.norm(reached - wp[:3]))
            angular = float(np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(reached_q, wp[3:])), 0, 1))))
            if distance > 0.030 or angular > 20:
                raise PlanningFailure(f"The twisted bottle grasp slipped: {distance:.5f} m, {angular:.2f} deg")
        effective_mouth = root + rotation @ [0, 0, 0.153]
        measured_mouth = reached + quat2mat(reached_q) @ [0, 0, 0.153]
        errors.append(
            {
                "tilt_deg": float(angle),
                "position_error_m": distance,
                "angle_error_deg": angular,
                "commanded_mouth_world": effective_mouth.tolist(),
                "effective_mouth_clearance_m": float(effective_mouth[2] - top),
                "measured_mouth_clearance_m": float(measured_mouth[2] - p.object_vertices("cup")[:, 2].max()),
            }
        )
    return {
        "bottle_tracking": errors,
        "bottle_spin_deg": spin_deg,
        "tilt_axis_azimuth_deg": azimuth_deg,
        "mouth_xy_offset_m": list(mouth_xy),
        "mouth_height_m": mouth_height,
    }


def liquid_twisted_upright(p, *, mouth_xy=(0.025, 0.023), mouth_height=0.055):
    """Reverse the measured twisted pour while retaining the left cup support."""
    from src.datagen.contact_diagnostics import contact_report

    options = {"mouth_xy": mouth_xy, "mouth_height": mouth_height}
    with contact_report(p):
        tilted = _liquid_twisted_arc(p, np.linspace(125, 78.125, 33), stage="stop_pouring", **options)
        unspin = _liquid_twisted_arc(p, [78.125], stage="untwist_bottle", azimuth_deg=-45, spin_deg=0, **options)
        upright = _liquid_twisted_arc(
            p,
            np.linspace(78.125, 0, 41),
            stage="upright_bottle",
            azimuth_deg=-45,
            spin_deg=0,
            **options,
        )
    _require_liquid_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {"reduce_tilt": tilted, "unspin": unspin, "upright": upright}


def liquid_center_twisted_stream(p):
    """Aim inside the cup before flow, then drain along the demonstrated open-elbow branch."""
    import json

    from pxr import Usd, UsdGeom

    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_center_twisted_stream requires pour_liquid_into_cup")
    include_static_geometry(p)
    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    transform = UsdGeom.Xformable(fluid.point_instancer.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p.recorder.path.with_suffix(".fluid_transform.json").write_text(
        json.dumps(
            {
                "matrix_row_vectors": np.asarray(transform).tolist(),
                "official_init_pos": np.asarray(fluid.init_pos).tolist(),
                "official_init_ori": np.asarray(fluid.init_ori).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    angle = float(np.rad2deg(np.arccos(np.clip(quat2mat(p.object_pose("bottle")[1])[2, 2], -1, 1))))
    with contact_report(p):
        result = _liquid_twisted_arc(
            p,
            np.linspace(angle, 125, 33),
            stage="pour",
            mouth_xy=(0.0, 0.023),
            mouth_height=0.040,
        )
        p.stage("settle_flow", "Keep the centered bottle still while the remaining liquid drains into the cup.", arm=1)
        p.hold(40)
    _require_liquid_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return result


def liquid_park_in_front_of_cup(p):
    """Place the emptied upright bottle on clear table space in front of the cup."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_park_in_front_of_cup requires pour_liquid_into_cup")
    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    park = np.asarray(fluid.init_pos) + [0, -0.12, 0.015]
    ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
    with contact_report(p):
        p.place(
            "bottle",
            park,
            1,
            ee[3:],
            target_quaternion=[1, 0, 0, 0],
            contact=True,
            linear_transfer=True,
            precise=True,
            approach_clearance=0.14,
        )
        p.home(1)
        p.hold(20)
        actual, quat = p.object_pose("bottle")
        error = float(np.linalg.norm(actual - park))
        tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(quat)[2, 2], -1, 1))))
        if error > 0.015 or tilt > 10:
            raise PlanningFailure(f"The returned bottle did not remain upright: error={error:.5f}, tilt={tilt:.2f}")
    return {"park_position": park, "position_error_m": error, "bottle_tilt_deg": tilt}


def liquid_release_centered_cup(p):
    """Release the cup only after the bottle is upright and the original liquid predicate passes."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import release_liquid_cup

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_release_centered_cup requires pour_liquid_into_cup")
    include_static_geometry(p)
    checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    if len(checks) != 1 or not p.env.reward_manager.check_once(checks[0], 0):
        raise PlanningFailure("The original liquid predicate must pass before releasing the cup")
    tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(p.object_pose("bottle")[1])[2, 2], -1, 1))))
    if tilt > 10:
        raise PlanningFailure("The poured bottle must already be upright before releasing the cup")
    before, _ = p.object_pose("cup")
    with contact_report(p):
        release_liquid_cup(p)
        p.stage("verify_cup", "Check that the released cup retains the poured liquid.", arm=0, object_label="cup")
        p.hold(20)
        after, quaternion = p.object_pose("cup")
        drift = float(np.linalg.norm(after - before))
        cup_tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(quaternion)[2, 2], -1, 1))))
        if drift > 0.006 or cup_tilt > 5:
            raise PlanningFailure(f"Cup release lost support: drift={drift:.5f} m, tilt={cup_tilt:.2f} deg")
        if not p.env.reward_manager.check_once(checks[0], 0):
            raise PlanningFailure("The original liquid predicate failed after releasing the cup")
    _require_liquid_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {"cup_released": True, "cup_drift_m": drift, "cup_tilt_deg": cup_tilt}


def liquid_park_centered_bottle(p):
    """Return the physically held upright bottle and validate unintended contact reports."""
    from src.datagen.collision_world import include_static_geometry

    include_static_geometry(p)
    result = liquid_park_in_front_of_cup(p)
    _require_liquid_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return result


def grasp_bottle_neck(p, *, approach_tilt_deg=0.0, continuous_approach=False):
    import json

    from pxr import Usd, UsdGeom

    from src.datagen.ik import plan_approach
    from src.datagen.programs.particles import lift_held_liquid, reorient_liquid_neck

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
    rotation = euler2mat(0, np.pi + np.deg2rad(approach_tilt_deg), 0)
    # The official bottle has zero friction. Its 82 mm body slips from
    # the nearly fully open jaws beyond horizontal. Enclose the narrow
    # neck between the shoulder and collar instead, without changing friction.
    # The rear portions of the pads extend below their tip center. At 71 mm
    # they hit the shoulder during approach, tilting the bottle before closure.
    grasp = start + quat2mat(start_q) @ [0, 0, 0.085]
    robot = p.actions.robots[1]
    approach = grasp - rotation[:, 0] * (robot.gripper_bias + 0.07)
    result = (
        None
        if continuous_approach
        else plan_approach(
            p.rm.planner[robot.robot_name],
            p.rm.get_joint(robot)[0],
            np.r_[approach, mat2quat(rotation)],
            robot.entity_origin_pose,
        )
    )
    arm, quat = 1, mat2quat(rotation)
    wrist = grasp - rotation[:, 0] * robot.gripper_bias
    p.stage("preshape", "Open the fingers to enclose the bottle neck.", arm=arm, object_label="bottle")
    p.gripper(arm, 0.75, frames=8)
    p.stage("approach", "Approach the bottle neck from the side.", arm=arm, object_label="bottle")
    if continuous_approach:
        p.linear(arm, approach, quat, step=0.003, angular_step=np.deg2rad(2))
    else:
        p.move(arm, approach, quat, result=result)
    p.stage("align_grasp", "Align both pads between the bottle shoulder and collar.", arm=arm, object_label="bottle")
    p.linear(arm, wrist, quat)
    p.stage("grasp", "Close the fingers around the bottle neck.", arm=arm, object_label="bottle")
    p.gripper(arm, 0)
    lift_held_liquid(p)
    reorient_liquid_neck(p)


def brace_and_twist_before_flow(p):
    """Brace the cup and rotate the neck grasp before liquid leaves the bottle."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import brace_liquid_cup, pour_held_liquid

    report = p.recorder.path.with_suffix(".contacts.json")
    with contact_report(p):
        brace_liquid_cup(p)
        pour_held_liquid(
            p,
            1,
            transfer_only=True,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            mouth_xy_offset=(0.025, 0.023),
            track_cup=True,
            feedback_grasp=True,
            return_bottle=False,
        )
    p.recorder.path.with_suffix(".brace_align.contacts.json").write_bytes(report.read_bytes())
    _require_liquid_contact_clearance(report)
    with contact_report(p):
        # Preserve the demonstrated rotation prefix. Move to the centered
        # stream before flow, stopping at the last full waypoint below 86 deg.
        result = _liquid_twisted_arc(p, np.linspace(78.125, 125, 33)[:6], stage="prepare_pour")
    _require_liquid_contact_clearance(report)
    return result


def pour_with_supported_cup(p):
    """Execute the real pickup, cup slide, centered pour and return from the initial layout."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import wait_for_contained_liquid

    include_static_geometry(p)
    report = p.recorder.path.with_suffix(".contacts.json")

    def preserve(name):
        p.recorder.path.with_suffix(f".{name}.contacts.json").write_bytes(report.read_bytes())
        _require_liquid_contact_clearance(report)

    with contact_report(p):
        grasp_bottle_neck(p)
    preserve("neck_grasp")
    liquid_stage_bottle_away_from_cup(p, offset=(0.12, 0, -0.10))
    preserve("stage_right")
    liquid_reposition_supported_cup(p, cup_dx=-0.07)
    preserve("reposition_cup")
    liquid_restage_after_left_cup(p)
    preserve("restage_bottle")
    liquid_positioned_cup_pretilt(p)
    preserve("pretilt")
    brace_and_twist_before_flow(p)
    preserve("twist_before_flow")
    liquid_center_twisted_stream(p)
    preserve("centered_pour")
    wait_for_contained_liquid(p)
    liquid_twisted_upright(p, mouth_xy=(0.0, 0.023), mouth_height=0.040)
    preserve("upright")
    liquid_release_centered_cup(p)
    preserve("release_cup")
    liquid_park_centered_bottle(p)
    preserve("park_bottle")
