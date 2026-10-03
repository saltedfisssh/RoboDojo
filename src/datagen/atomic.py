"""Observable atomic-skill metrics; a closed gripper alone is not verified grasp success."""

import numpy as np


def evaluate_atomic(env, annotation):
    goal, primitive = annotation.get("goal", {}), annotation["primitive"]
    metrics = {"primitive": primitive, "passed": None, "criterion": "no goal recorded"}
    if not goal:
        return metrics
    rm = env.robot_manager
    arms = [r for r in rm.robot_list if r.type == "target"]
    arm = annotation.get("arm")
    checks = []
    if "opponent_turn_completed" in goal:
        from src.datagen.interaction import turn_metrics

        metrics["opponent"] = turn_metrics(env, goal["opponent_turn_completed"])
        checks.append(metrics["opponent"]["passed"])
    containment_goal = any(key in goal for key in ("contained_objects", "fluid_in_container", "official_object_checks"))
    object_goal = any(
        key in goal
        for key in (
            "object_position",
            "object_xy_position",
            "object_quaternion",
            "object_axis_up",
            "object_min_z",
            "object_height_range",
            "contained_objects",
            "fluid_in_container",
            "articulation_joint",
            "functional_point",
            "garment_points",
            "official_object_checks",
        )
    )
    if "official_object_checks" in goal:
        from src.datagen.state import decode_context

        manager = env.reward_manager
        predicates = decode_context(goal["official_object_checks"], manager.func_parser)
        metrics["official_object_checks"] = bool(manager.check_once(predicates, 0, op="and"))
        checks.append(metrics["official_object_checks"])
    if "garment_points" in goal:
        lm = env.scene_manager.layout_manager
        label = annotation["object"]
        obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
        tags = lm.get_instance_metadata(0, label=label)["passive"]["functional"]
        vertices = np.asarray(obj.sample_mesh_vertices()[0])
        metrics["garment_points"] = {}
        for tag, spec in goal["garment_points"].items():
            point = vertices[tags[tag]["id"]].mean(axis=0)
            measured = {"position": point.tolist()}
            if "min_z" in spec:
                measured["height_passed"] = bool(point[2] >= spec["min_z"])
                checks.append(measured["height_passed"])
            if "position" in spec:
                error = float(np.linalg.norm(point - spec["position"]))
                measured["position_error_m"] = error
                checks.append(error <= spec.get("position_tolerance_m", 0.04))
            metrics["garment_points"][tag] = measured
    if "functional_point" in goal:
        spec = goal["functional_point"]
        lm = env.scene_manager.layout_manager
        points = lm.get_functional_points(
            spec["tag"],
            spec.get("type", "active"),
            config=lm.get_instance_metadata(0, label=spec["label"]),
            obj_name=lm.get_instance_name(0, spec["label"]),
            env_idx=0,
        )
        point = np.asarray(points[spec.get("index", 0)])[:3]
        error = float(np.linalg.norm(point - spec["position"]))
        metrics["functional_point_error_m"] = error
        checks.append(error <= spec.get("position_tolerance_m", 0.01))
    if "articulation_joint" in goal:
        spec = goal["articulation_joint"]
        lm = env.scene_manager.layout_manager
        obj = lm.get_scene_object(0, lm.get_instance_name(0, spec["label"]))
        info = obj.get_joint_info(spec["joint"])
        ratio = (info["position"] - info["lower"]) / (info["upper"] - info["lower"])
        metrics["object_joint_ratio"] = float(ratio)
        checks.append(spec.get("min_ratio", -np.inf) <= ratio <= spec.get("max_ratio", np.inf))
    if "contained_objects" in goal:
        contained = {
            label: bool(
                env.reward_manager.func_parser.is_A_in_B({"env_idx": 0, "label_A": label, "label_B": goal["container"]})
            )
            for label in goal["contained_objects"]
        }
        metrics["objects_in_container"] = contained
        checks.append(all(contained.values()))
    if "object_height_range" in goal:
        # Containment alone accepts an egg resting on a compartment rim.
        # Retain both official bounds and the live support frame for seating.
        metrics["object_height_range"] = bool(
            env.reward_manager.func_parser.is_A_point_above_B_point_by_z_range(
                {**goal["object_height_range"], "env_idx": 0}
            )
        )
        checks.append(metrics["object_height_range"])
    if "fluid_in_container" in goal:
        # Reuse the official particle predicate, including residual/scatter
        # handling. The demonstrator's bottle angle is only a diagnostic.
        metrics["fluid_in_container"] = bool(
            env.reward_manager.func_parser.is_A_fluid_in_B({**goal["fluid_in_container"], "env_idx": 0})
        )
        checks.append(metrics["fluid_in_container"])
    if "ee_position" in goal:
        pose = rm.get_real_endpose(arms[arm])[0]
        metrics["ee_position_error_m"] = float(np.linalg.norm(pose[:3] - goal["ee_position"]))
        target_q = np.asarray(goal["ee_quaternion"])
        dot = abs(np.dot(pose[3:] / np.linalg.norm(pose[3:]), target_q / np.linalg.norm(target_q)))
        metrics["ee_orientation_error_deg"] = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
        # Object outcomes define lift/place/press/pour success. A VLA may reach
        # those outcomes with a different valid wrist pose from the demonstrator.
        if not object_goal:
            checks += [metrics["ee_position_error_m"] <= 0.025, metrics["ee_orientation_error_deg"] <= 15]
    if "joint_position" in goal:
        error = np.max(np.abs(rm.get_joint(arms[arm])[0] - goal["joint_position"]))
        metrics["joint_error_rad"] = float(error)
        checks.append(error <= 0.08)
    if any(
        key in goal
        for key in ("object_position", "object_xy_position", "object_quaternion", "object_axis_up", "object_min_z")
    ):
        lm = env.scene_manager.layout_manager
        pos, quat = lm.get_instance_pose(0, label=annotation["object"])
        if "object_axis_up" in goal:
            from transforms3d.quaternions import quat2mat

            axis = np.asarray(goal["object_axis_up"], dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
                raise ValueError("object_axis_up must be a finite nonzero three-vector")
            actual_axis = quat2mat(np.asarray(quat)) @ (axis / np.linalg.norm(axis))
            angle = float(np.rad2deg(np.arccos(np.clip(actual_axis[2], -1, 1))))
            metrics["object_axis_error_deg"] = angle
            checks.append(angle <= goal.get("axis_tolerance_deg", 25))
        if "object_position" in goal:
            error = np.linalg.norm(np.asarray(pos) - goal["object_position"])
            metrics["object_position_error_m"] = float(error)
            if not containment_goal:
                checks.append(error <= goal.get("position_tolerance_m", 0.025))
        if "object_xy_position" in goal:
            error = np.linalg.norm(np.asarray(pos)[:2] - goal["object_xy_position"])
            metrics["object_xy_error_m"] = float(error)
            if not containment_goal:
                checks.append(error <= goal.get("position_tolerance_m", 0.025))
        if "object_quaternion" in goal:
            target_q = np.asarray(goal["object_quaternion"])
            quat = np.asarray(quat)
            dot = abs(np.dot(quat / np.linalg.norm(quat), target_q / np.linalg.norm(target_q)))
            angle = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
            metrics["object_orientation_error_deg"] = angle
            if not containment_goal:
                checks.append(angle <= goal.get("orientation_tolerance_deg", 15))
        if "object_min_z" in goal:
            metrics["object_height_m"] = float(pos[2])
            checks.append(float(pos[2]) >= goal["object_min_z"])
    if "gripper_open" in goal:
        r = arms[arm]
        value = rm.get_end_effector_real_val(r)[0][0]
        opening = (value - r.gripper_scale[0]) / (r.gripper_scale[1] - r.gripper_scale[0])
        opening = opening if r.gripper_move["sign"] == 1 else 1 - opening
        metrics["measured_gripper_open"] = float(opening)
        if goal["gripper_open"] >= 0.9:
            checks.append(opening >= 0.85)
        else:
            metrics["criterion"] = "grasp requires a subsequent lift/contact check; closure alone is insufficient"
            return metrics
    if primitive == "pour" and not containment_goal:
        metrics["criterion"] = "legacy pouring goal has no content outcome; tool poses alone do not verify pouring"
        return metrics
    metrics["criterion"] = "planner goal with measured physical-state criteria"
    metrics["passed"] = bool(all(checks)) if checks else None
    return metrics
