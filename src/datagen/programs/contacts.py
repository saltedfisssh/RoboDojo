"""Force-producing contact primitives. All changes result from commanded robot motion."""

import numpy as np
from transforms3d.quaternions import mat2quat


def press_button(p, label):
    from src.datagen.planner import PlanningFailure

    pos, _ = p.object_pose(label)
    arm = int(pos[0] >= 0)
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
    metadata = lm.get_instance_metadata(0, label=label)
    joint_name = metadata["passive"]["functional"]["press"]["parent_joint"]
    quat = mat2quat(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float))
    if arm == 1:
        quat = mat2quat(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float))
    contact = pos.copy()
    contact[2] = p.object_vertices(label)[:, 2].max() + p.actions.robots[arm].gripper_bias
    p.stage("approach", f"Position the closed gripper above {label}.", arm=arm, object_label=label)
    p.move(arm, contact + [0, 0, 0.045], quat)
    p.gripper(arm, 0, frames=8)
    p.stage("press", f"Press {label} down.", arm=arm, object_label=label)
    p.annotation["goal"]["articulation_joint"] = {"label": label, "joint": joint_name, "max_ratio": 0.5}

    def depressed():
        current = obj.get_joint_info(joint_name)
        return (current["position"] - current["lower"]) / (current["upper"] - current["lower"]) < 0.35

    # Stop on measured button travel. Driving to a fixed deep wrist target
    # after the 6 mm spring bottoms out can destabilize the robot articulation.
    p.linear(arm, contact + [0, 0, -0.018], quat, step=0.0015, stop_when=depressed)
    info = obj.get_joint_info(joint_name)
    ratio = (info["position"] - info["lower"]) / (info["upper"] - info["lower"])
    if ratio >= 0.5:
        raise PlanningFailure(f"{label} did not depress: joint ratio={ratio:.3f}")
    p.stage("release_button", f"Release {label} and wait for its spring to return.", arm=arm, object_label=label)
    p.annotation["goal"]["articulation_joint"] = {"label": label, "joint": joint_name, "min_ratio": 0.9}
    p.linear(arm, contact + [0, 0, 0.045], quat, step=0.003, settle=12)
    info = obj.get_joint_info(joint_name)
    ratio = (info["position"] - info["lower"]) / (info["upper"] - info["lower"])
    if ratio < 0.9:
        raise PlanningFailure(f"{label} did not spring back: joint ratio={ratio:.3f}")
    p.gripper(arm, 1, frames=8)
    return arm


def play_xylophone(p):
    from transforms3d.quaternions import quat2mat

    # Rotating the home gripper downward at its original height sweeps its
    # fingertips across the lightweight handle before reaching the approach.
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    p.stage("clear_table", "Raise the empty gripper before approaching the mallet.", arm=0)
    p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
    root, orientation = p.object_pose("mallet")
    rotation = quat2mat(orientation)
    # Use the measured handle direction. A rotated 21 cm mallet can exceed
    # the jaw width in an axis-aligned bbox although its handle is graspable.
    grasp = root + rotation @ [0, 0.09, 0]
    # A tilted head's bbox is not the table height beneath the thin handle.
    # Keep the 12.57 mm extended tips above the actual 0.765 m tabletop.
    grasp[2] = max(grasp[2], 0.779)
    handle = rotation[:, 1]
    yaw = np.rad2deg(np.arctan2(handle[1], handle[0])) - 90
    arm, quat = p.pick("mallet", arm=0, grasp_point=grasp, yaw_candidates=(yaw + 180, yaw), linear_approach=True)
    strike_xylophone_keys(p, arm, quat)


def lift_xylophone_mallet(p, arm, quat):
    """Lift relative to the official first-contact reference, which can precede settling."""
    from src.datagen.planner import PlanningFailure

    p.stage("lift", "Lift the mallet before striking the next key.", arm=arm, object_label="mallet")
    pos, _ = p.object_pose("mallet")
    name = p.env.scene_manager.layout_manager.get_instance_name(0, "mallet")
    reference = p.env.reward_manager.func_parser.pre_state[0][name]["pose"]
    minimum_z = max(float(pos[2] + 0.025), float(reference[2] + 0.035))
    p.annotation["goal"]["object_min_z"] = minimum_z
    actual = p.rm.get_real_endpose(p.actions.robots[arm])[0][:3]
    rise = max(0.045, minimum_z - pos[2])
    p.linear(arm, actual + [0, 0, rise], quat, step=0.003, settle=8)
    pending = p.env.reward_manager.check_list[0]
    if pending and pending[0][0][0] == "is_lift":
        raise PlanningFailure("The mallet did not clear the official previous-contact lift threshold")


def strike_xylophone_keys(p, arm, quat, start_key=0):
    """Execute the remaining keys, checking both tool endpoints and official history."""
    from src.datagen.planner import PlanningFailure

    for i in range(start_key, 8):
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        beat = p.functional_pose("mallet", "beat", "active")[:3]
        point = p.functional_pose("xylophone", f"hit_{i}")[:3]
        point[2] += 0.024
        wrist = point + ee[:3] - beat
        p.stage("approach_key", f"Move the mallet above xylophone key {i + 1}.", arm=arm, object_label="mallet")
        p.move(arm, wrist + [0, 0, 0.045], quat)
        # The slender handle can roll in the fingers during transport. Re-read
        # the actual striking point before descending to the next key.
        ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
        beat = p.functional_pose("mallet", "beat", "active")[:3]
        wrist = point + ee[:3] - beat
        p.stage("strike", f"Strike xylophone key {i + 1}.", arm=arm, object_label="mallet")
        p.annotation["goal"]["functional_point"] = {
            "label": "mallet",
            "tag": "beat",
            "type": "active",
            "index": 0,
            "position": point.tolist(),
            "position_tolerance_m": 0.01,
        }
        p.linear(arm, wrist, quat, step=0.003, settle=8)
        # First contact can roll the handle in the fingers. Keep the recorded
        # atomic endpoint at the actual mallet head, not just the wrist target.
        for _ in range(3):
            error = point - p.functional_pose("mallet", "beat", "active")[:3]
            if np.linalg.norm(error) <= 0.006:
                break
            if np.linalg.norm(error) > 0.04:
                raise PlanningFailure("The mallet slipped away from its striking point")
            actual = p.rm.get_real_endpose(p.actions.robots[arm])[0][:3]
            p.linear(arm, actual + error, quat, step=0.002, settle=3)
        if np.linalg.norm(point - p.functional_pose("mallet", "beat", "active")[:3]) > 0.01:
            raise PlanningFailure(f"Mallet head did not settle over xylophone key {i + 1}")
        expected = 14 - 2 * i
        if len(p.env.reward_manager.check_list[0]) != expected:
            raise PlanningFailure(f"Official xylophone sequence did not register key {i + 1}")
        if i < 7:
            lift_xylophone_mallet(p, arm, quat)
