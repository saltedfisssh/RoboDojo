"""Continuous contact IK using cuRobo v2's collision-checked candidate set."""

import numpy as np

from src.datagen.planning_diagnostics import trace_slow_planning


def cartesian_waypoints(start, target, quaternion, step, angular_step=np.deg2rad(3)):
    """Interpolate translation and shortest-arc WXYZ rotation together."""
    if step <= 0 or angular_step <= 0:
        raise ValueError("Cartesian interpolation step sizes must be positive")
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    q0 = start[3:] / np.linalg.norm(start[3:])
    q1 = np.asarray(quaternion, dtype=float)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    theta = np.arccos(np.clip(dot, 0, 1))
    count = max(1, int(np.ceil(np.linalg.norm(target - start[:3]) / step)), int(np.ceil(2 * theta / angular_step)))
    for alpha in np.linspace(0, 1, count + 1)[1:]:
        if theta < 1e-7:
            quat = (1 - alpha) * q0 + alpha * q1
        else:
            quat = (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)
        yield np.r_[start[:3] + alpha * (target - start[:3]), quat / np.linalg.norm(quat)]


def nearest_valid_solution(positions, success, joint_names, active_names, current, max_delta=0.35):
    """Keep only successful, finite solutions within the actual joint winding."""
    positions = np.asarray(positions)
    positions = positions.reshape(-1, positions.shape[-1])
    success = np.asarray(success, dtype=bool).reshape(-1)
    if len(success) != len(positions) or len(joint_names) != positions.shape[-1]:
        raise ValueError("IK candidate dimensions do not match success flags and joint names")
    indices = [joint_names.index(name) for name in active_names]
    candidates = positions[:, indices]
    distances = np.max(np.abs(candidates - np.asarray(current)), axis=1)
    valid = success & np.isfinite(candidates).all(axis=1) & (distances <= max_delta)
    if not valid.any():
        return None
    distances[~valid] = np.inf
    return candidates[np.argmin(distances)].copy()


@trace_slow_planning
def solve_nearby(solver, joints, target_pose, real_robot_pose, max_delta=0.35):
    # Verified against dev100 IKSolver.solve_pose/_get_result: return_seeds
    # changes result selection, while config.num_seeds keeps graph dimensions.
    from utils.transformer import calculate_target_pose

    transformed = calculate_target_pose(real_robot_pose, solver.robot_origin_pose, np.array(target_pose))
    position, quaternion = solver._trans_from_world_to_base(
        np.asarray(solver.robot_origin_pose, dtype=np.float32), np.asarray(transformed, dtype=np.float32)
    )
    position = np.asarray(position) + np.asarray(solver.frame_bias)
    ik = solver.ik_solver
    current = solver._build_joint_state(joints)
    goal = solver._build_goal_pose(position, quaternion)

    def select(result):
        if result.success is None or result.js_solution is None:
            return None
        return nearest_valid_solution(
            result.js_solution.position.detach().cpu().numpy(),
            result.success.detach().cpu().numpy(),
            list(result.js_solution.joint_names or []),
            solver.active_joints_name,
            joints,
            max_delta,
        )

    candidate = select(ik.solve_pose(goal_tool_poses=goal, current_state=current, return_seeds=ik.config.num_seeds))
    if candidate is not None:
        return candidate
    # dev100 solve_pose normally appends global random seeds to one current
    # state seed. A local set can retain the contact branch when that single
    # seed fails. Keep the captured graph size and all collision/success checks.
    # get_active_js is essential: seed_config uses the optimizer's DOF order.
    active = ik.get_active_js(current).position
    limits = ik.kinematics.get_joint_limits().position
    configs = local_seed_configs(
        active.detach().cpu().numpy().reshape(-1),
        limits.detach().cpu().numpy(),
        ik.config.num_seeds,
    )
    result = ik.solve_pose(
        goal_tool_poses=goal,
        current_state=current,
        seed_config=active.new_tensor(configs).unsqueeze(0),
        return_seeds=ik.config.num_seeds,
    )
    return select(result)


def local_seed_configs(current, limits, count, radius=0.12):
    """Bounded deterministic neighborhoods, without changing joint winding."""
    current = np.asarray(current, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if current.ndim != 1 or limits.shape != (2, len(current)) or count < 1 or radius <= 0:
        raise ValueError("Invalid local IK seed dimensions or radius")
    if not np.isfinite(current).all() or not np.isfinite(limits).all() or np.any(limits[0] > limits[1]):
        raise ValueError("Invalid local IK seed state or limits")
    offsets = [np.zeros_like(current)]
    for scale in (radius / 2, radius):
        for direction in np.eye(len(current)):
            offsets.extend((scale * direction, -scale * direction))
    rng = np.random.default_rng(0)
    offsets.extend(rng.uniform(-radius, radius, (max(0, count - len(offsets)), len(current))))
    return np.clip(current + np.asarray(offsets[:count]), limits[0], limits[1])


def approach_seed_configs(current, limits, count):
    """Explore short joint windings as well as the current, possibly wound, pose."""
    current = np.asarray(current, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if current.ndim != 1 or limits.shape != (2, len(current)) or count < 2:
        raise ValueError("Invalid approach IK seed dimensions")
    if not np.isfinite(current).all() or not np.isfinite(limits).all() or np.any(limits[0] > limits[1]):
        raise ValueError("Invalid approach IK state or limits")
    # X5's first five URDF limits span +/-10 rad. Uniform seeds over that
    # range often choose long windings for simple near-table approaches.
    low = np.clip(-np.pi, limits[0], limits[1])
    high = np.clip(np.pi, limits[0], limits[1])
    configs = np.random.default_rng(17).uniform(low, high, (count, len(current)))
    configs[0] = np.clip(current, limits[0], limits[1])
    configs[1] = np.clip(np.zeros_like(current), limits[0], limits[1])
    return configs


def approach_ik(solver, joints, target_pose, real_robot_pose):
    """Additional cuRobo seed search for free-space goals, retaining collision checks."""
    from utils.transformer import calculate_target_pose

    transformed = calculate_target_pose(real_robot_pose, solver.robot_origin_pose, np.array(target_pose))
    position, quaternion = solver._trans_from_world_to_base(
        np.asarray(solver.robot_origin_pose, dtype=np.float32), np.asarray(transformed, dtype=np.float32)
    )
    ik = solver.ik_solver
    current = solver._build_joint_state(joints)
    goal = solver._build_goal_pose(np.asarray(position) + np.asarray(solver.frame_bias), quaternion)
    active = ik.get_active_js(current).position
    configs = approach_seed_configs(
        active.detach().cpu().numpy().reshape(-1),
        ik.kinematics.get_joint_limits().position.detach().cpu().numpy(),
        ik.config.num_seeds,
    )
    result = ik.solve_pose(
        goal_tool_poses=goal,
        current_state=current,
        seed_config=active.new_tensor(configs).unsqueeze(0),
        return_seeds=ik.config.num_seeds,
    )
    if result.success is None or result.js_solution is None:
        return None
    return nearest_valid_solution(
        result.js_solution.position.detach().cpu().numpy(),
        result.success.detach().cpu().numpy(),
        list(result.js_solution.joint_names or []),
        solver.active_joints_name,
        joints,
        max_delta=np.inf,
    )


def plan_approach(solver, joints, target_pose, real_robot_pose):
    """Try the official pose planner, then a checked joint path to a checked IK goal."""
    result = solver.plan_path(joints, target_pose, real_robot_pose)
    if result["status"] == "Success":
        return result
    goal = approach_ik(solver, joints, target_pose, real_robot_pose)
    if goal is None:
        return result
    # An endpoint solution alone never authorizes motion. The cuRobo joint
    # planner must accept the full path from the measured current state.
    return solver.plan_joint(joints, goal)
