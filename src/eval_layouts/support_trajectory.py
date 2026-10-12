"""Plan and verify native-format demonstrations for new sorting layouts."""

from copy import deepcopy
from pathlib import Path
import pickle

import numpy as np
from transforms3d.quaternions import quat2mat


def compact_commands(steps, stride=2):
    fingers = np.asarray([np.asarray(step["support_ee0_joint_state"]["position"]).reshape(-1) for step in steps])
    moving = np.r_[False, np.any(np.abs(np.diff(fingers, axis=0)) > 1e-7, axis=1)]
    # Preserve physical finger motion and 0.2 s of contact settling. Shortening
    # jaw closure as well as transport can release an incompletely held object.
    protected = np.convolve(moving.astype(int), np.ones(51, dtype=int))[: len(steps)] > 0
    sampled = np.floor(np.arange(0, len(steps), stride)).astype(int)
    indices = sorted(set(sampled) | set(np.flatnonzero(protected)) | {len(steps) - 1})
    selected = deepcopy([steps[i] for i in indices])
    for ordinal, control in enumerate(selected):
        scale = indices[ordinal + 1] - indices[ordinal] if ordinal + 1 < len(indices) else 1
        for state in control.values():
            state["velocity"] = (np.asarray(state["velocity"]) * scale).tolist()
    return selected


def prepare_planner(env):
    """Keep offline CUDA planning in a separate process, like a model server."""
    from src.eval_layouts.teacher_solver import TeacherSolver

    robot = env.robot_manager.get_robot_by_arm_name("support_arm0")
    if robot.robot_name not in env.robot_manager.planner:
        planner = TeacherSolver(robot, env.dt, env.layout_device_id)
        env.robot_manager.planner[robot.robot_name] = planner
        env.robot_manager.ik_solver[robot.robot_name] = planner
    return robot


def basket_drop_target(p, label, basket):
    """Keep the measured footprint clear of the rim, then stack above contents."""
    index = int(label.removeprefix("aim").removeprefix("t"))
    offsets = ((-0.035, -0.075), (0.035, -0.075), (0.035, 0.065), (0, 0), (-0.035, 0.075))
    target, _ = p.object_pose(basket)
    target[:2] += offsets[index]
    position, _ = p.object_pose(label)
    vertices = p.object_vertices(label)
    lower, upper = vertices.min(0) - position, vertices.max(0) - position
    basket_vertices = p.object_vertices(basket)
    basket_min, basket_max = basket_vertices.min(0), basket_vertices.max(0)
    minimum = basket_min[:2] + 0.012 - lower[:2]
    maximum = basket_max[:2] - 0.012 - upper[:2]
    if np.any(minimum > maximum):
        from src.datagen.planner import PlanningFailure

        raise PlanningFailure(f"Measured footprint of {label} does not fit inside the basket rim")
    target[:2] = np.clip(target[:2], minimum, maximum)
    support_top = basket_max[2]
    prefix = "aim" if label.startswith("aim") else "t"
    for other_index in range(5):
        other = f"{prefix}{other_index}"
        if other == label:
            continue
        other_position, _ = p.object_pose(other)
        if not np.all((other_position[:2] > basket_min[:2]) & (other_position[:2] < basket_max[:2])):
            continue
        occupied = p.object_vertices(other)
        if np.all(occupied.max(0)[:2] > target[:2] + lower[:2]) and np.all(
            occupied.min(0)[:2] < target[:2] + upper[:2]
        ):
            support_top = max(support_top, occupied[:, 2].max())
    target[2] = support_top + 0.010 - lower[2]
    return target


def grasp_rotations(vertices):
    """Try jaw directions aligned with the object's measured box edges."""
    from transforms3d.euler import euler2mat

    vertices = np.asarray(vertices)
    angles = {0.0, round(np.pi / 2, 8)}
    for first in vertices:
        for second in vertices:
            edge = (second - first)[:2]
            if np.linalg.norm(edge) > 1e-5:
                angles.add(round(float((np.arctan2(edge[1], edge[0]) - np.pi / 2) % np.pi), 8))
    rotations = [
        euler2mat(0, 0, angle + turn) @ np.diag([1, -1, -1]) for angle in sorted(angles) for turn in (0, np.pi)
    ]
    return sorted(rotations, key=lambda rotation: np.ptp(vertices @ rotation[:, 1]))


def demonstrate(planner):
    """Native teacher motions, with grasp yaw adapted to randomly rotated assets."""
    from shapely.geometry import MultiPoint, Polygon
    from transforms3d.quaternions import mat2quat

    from src.datagen.ik import cartesian_waypoints
    from src.datagen.planner import PlanningFailure

    def checked_line(solver, joint, start, target, quaternion, origin):
        for pose in cartesian_waypoints(np.r_[start, quaternion], target, quaternion, 0.01):
            result = solver.solve_ik_to_joint(joint, pose, origin)
            if result["status"] != "Success":
                raise PlanningFailure("Offline contact feasibility check failed")
            joint = np.asarray(result["joint_value"])
        return joint

    p, rm, robot = planner.p, planner.rm, planner.robot
    solver = rm.planner[robot.robot_name]
    # Reject a footprint that cannot fit before moving any of the other objects.
    for row in p.env.aim_label:
        basket_drop_target(p, row[0], "basket1")
    for row in p.env.aim_label:
        label = row[0]
        p.stage(
            "observe_demo", f"Observe the support arm put {label} into the demonstration basket.", object_label=label
        )
        pos, _ = p.object_pose(label)
        vertices = p.object_vertices(label)
        grasp = (vertices.min(0) + vertices.max(0)) / 2
        grasp[2] += robot.gripper_bias
        widths = []
        obstacles = [
            MultiPoint(p.object_vertices(other)[:, :2]).convex_hull
            for other in [*[f"aim{i}" for i in range(5)], "basket0", "basket1"]
            if other != label
        ]

        def clearance_cost(rotation):
            closing, along = rotation[:2, 1], rotation[:2, 0]
            width = float(np.ptp(vertices @ rotation[:, 1]))
            opening = min(max(robot.gripper_scale), width / 2 + 0.007)
            area = 0.0
            for sign in (-1, 1):
                center = grasp[:2] + sign * opening * closing
                finger = Polygon(
                    [
                        center + x * along + y * closing
                        for x, y in ((-0.012, -0.008), (0.012, -0.008), (0.012, 0.008), (-0.012, 0.008))
                    ]
                )
                area += sum(finger.intersection(obstacle).area for obstacle in obstacles)
            return area, width

        for rotation in sorted(grasp_rotations(vertices), key=clearance_cost):
            width = float(np.ptp(vertices @ rotation[:, 1]))
            if width > 2 * max(robot.gripper_scale) - 0.003:
                continue
            widths.append(width)
            planner.quat = mat2quat(rotation)
            approach = solver.plan_path(
                rm.get_joint(robot)[0], np.r_[grasp + [0, 0, 0.10], planner.quat], robot.entity_origin_pose
            )
            if approach["status"] == "Success":
                try:
                    joint = checked_line(
                        solver,
                        np.asarray(approach["position"])[-1],
                        grasp + [0, 0, 0.10],
                        grasp,
                        planner.quat,
                        robot.entity_origin_pose,
                    )
                    checked_line(solver, joint, grasp, grasp + [0, 0, 0.08], planner.quat, robot.entity_origin_pose)
                    target = basket_drop_target(p, label, "basket1") + grasp - pos
                    basket_approach = solver.plan_path(
                        joint, np.r_[target + [0, 0, 0.10], planner.quat], robot.entity_origin_pose
                    )
                    if basket_approach["status"] != "Success":
                        continue
                    checked_line(
                        solver,
                        np.asarray(basket_approach["position"])[-1],
                        target + [0, 0, 0.10],
                        target,
                        planner.quat,
                        robot.entity_origin_pose,
                    )
                except PlanningFailure:
                    continue
                break
        else:
            raise PlanningFailure(f"No reachable grasp for {label}; feasible jaw widths={widths}")
        opening = min(max(robot.gripper_scale), width / 2 + 0.007)
        planner.gripper((opening - min(robot.gripper_scale)) / np.ptp(robot.gripper_scale))
        planner.execute(rm.plan_ee(0, "support_arm0", result=approach, need_plan=False))
        planner.linear(grasp)
        planner.gripper(0)
        planner.linear(grasp + [0, 0, 0.12], stop_when=lambda: p.object_pose(label)[0][2] >= pos[2] + 0.08)
        actual, _ = p.object_pose(label)
        if actual[2] < pos[2] + 0.07:
            raise PlanningFailure(f"Support arm did not physically lift {label}")
        ee = rm.get_real_endpose(robot)[0]
        target = basket_drop_target(p, label, "basket1")
        offset = ee[:3] - actual
        planner.move(target + offset + [0, 0, 0.10])
        planner.linear(target + offset)
        planner.gripper(1)
        planner.move(target + offset + [0, 0, 0.10])
        p.hold(10)
        if not p.env.reward_manager.check_once(p.env.reward_manager.is_A_in_B(label_A=label, label_B="basket1"), 0):
            raise PlanningFailure(f"Support arm did not leave {label} in the demonstration basket")
    result = solver.plan_joint(rm.get_joint(robot)[0], planner.home)
    planner.execute(rm.plan_ee(0, "support_arm0", result=result, need_plan=False))
    p.env.query_support_times[0] = 1
    p.hold(10)
    if not p.env.success[0]:
        raise PlanningFailure("The generated support-arm demonstration failed the official stability check")


class DemoRecorder:
    def __init__(self, env, *, record=True):
        self.env, self.rm = env, env.robot_manager
        self.record, self.step_count = record, 0
        self.annotation = {"goal": {}}
        self.steps = []
        self.last = deepcopy(
            {
                key: self.rm.control_manager.prev_control[0][key]
                for key in ("support_arm0_joint_state", "support_ee0_joint_state")
            }
        )

    def stage(self, primitive, text, **kwargs):
        self.annotation = {"goal": {}}
        print("SUPPORT_DEMO", text, flush=True)

    def object_pose(self, label):
        pos, quat = self.env.scene_manager.layout_manager.get_instance_pose(0, label=label)
        return np.asarray(pos, dtype=float).copy(), np.asarray(quat, dtype=float).copy()

    def object_vertices(self, label):
        data = self.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
        pos, quat = self.object_pose(label)
        return np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ quat2mat(quat).T + pos

    def hold(self, frames=10):
        from env.environment.task_env import TaskEnv
        from src.datagen.planner import PlanningFailure

        for _ in range(frames * 10):
            if self.step_count >= 20000:
                raise PlanningFailure("Support demonstration exceeds 20000 physics steps")
            pending = self.env.support_arm_action[0]
            control = pending.pop(0) if pending else {}
            if self.record:
                self.last.update(deepcopy(control))
                recorded = {
                    key: {name: np.asarray(value).tolist() for name, value in state.items()}
                    for key, state in self.last.items()
                }
                self.steps.append(recorded)
            self.step_count += 1
            if self.step_count % 250 == 0 and hasattr(self.env, "layout_activity_path"):
                self.env.layout_activity_path.touch()
            if self.step_count % 1000 == 0:
                print("SUPPORT_PHYSICS_PROGRESS", "planning" if self.record else "replay", self.step_count, flush=True)
            self.rm.control_manager.push([0], [[control]])
            TaskEnv.step(self.env, self.rm.control_manager.pop([0]))
            self.env.sim_step(render=False)


def generate_and_verify(env, sampling_seed, destination, index):
    from src.datagen.planner import PlanningFailure
    from src.datagen.support_demo import SupportPlanner

    class TeacherPlanner(SupportPlanner):
        def execute(self, controls, *, settle=5):
            if controls is not None and len(controls) > 150:
                selected = list(controls[::2])
                if (len(controls) - 1) % 2:
                    selected.append(controls[-1])
                controls = deepcopy(selected)
                for control in controls:
                    for state in control.values():
                        if "velocity" in state:
                            state["velocity"] = (np.asarray(state["velocity"]) * 2).tolist()
            return super().execute(controls, settle=settle)

        def linear(self, point, step=0.006, *, stop_when=None):
            try:
                return super().linear(point, step, stop_when=stop_when)
            except PlanningFailure as exc:
                if "continuous IK" in str(exc):
                    # A short contact segment can outrun the physical joint
                    # servos. Retry from the measured pose after settling,
                    # retaining the original IK collision and continuity tests.
                    self.p.hold(10)
                    try:
                        return super().linear(point, step / 2, stop_when=stop_when)
                    except PlanningFailure as retry:
                        if "Cartesian tracking error" not in str(retry):
                            raise
                        exc = retry
                if "Cartesian tracking error" not in str(exc):
                    raise
                before = self.rm.get_real_endpose(self.robot)[0][:3].copy()
                self.p.hold(10)
                actual = self.rm.get_real_endpose(self.robot)[0][:3]
                error = np.linalg.norm(actual - point)
                if error > 0.015:
                    self.p.hold(20)
                    actual = self.rm.get_real_endpose(self.robot)[0][:3]
                    error = np.linalg.norm(actual - point)
                print(
                    "SUPPORT_TRACKING",
                    {
                        "target": np.asarray(point).tolist(),
                        "before": before.tolist(),
                        "after": actual.tolist(),
                        "error": float(error),
                    },
                    flush=True,
                )
                # Contact can stop the fingertip before the nominal free-space
                # pose. Physical lift, basket placement and native replay
                # remain mandatory; this is an offline planning guard only.
                if error > 0.04:
                    raise

    order = np.random.default_rng(sampling_seed + 7041).permutation(5)
    env.target_label = [[f"t{i}"] for i in order]
    env.aim_label = [[f"aim{i}"] for i in order]
    env.target_place_tag = [["up"] for _ in order]
    robot = prepare_planner(env)
    recorder = DemoRecorder(env)
    demonstrate(TeacherPlanner(recorder))
    # Later placements can dislodge earlier contents. TaskEnv.step advances
    # controls without EvalEnv's reward updates, so check the native terminal
    # condition explicitly before spending time on another full replay.
    env.check_support_arm_stable(0)
    live_inside = [
        env.reward_manager.call_func_parser(env.reward_manager.is_A_in_B(label_A=f"aim{i}", label_B="basket1"), 0)
        for i in range(5)
    ]
    if min(live_inside) < 0.5 or env.unstable_envs:
        raise PlanningFailure(f"Live support demonstration failed native terminal check: {live_inside}")
    # The original demonstrations are about 4–6k control steps. Shorten the
    # conservative offline motions before testing the exported commands in
    # the actual task; successful live planning alone is insufficient.
    labels = {
        "target_label": [row[0] for row in env.target_label],
        "aim_label": [row[0] for row in env.aim_label],
        "target_place_tag": ["up"] * 5,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    replay_attempts = []
    for stride in (2, 1.5, 1):
        exported = compact_commands(recorder.steps, stride=stride)
        # The native task allows 1600 actions, each advancing ten physics steps.
        if len(exported) > 16000:
            continue
        data = {"joint_path": [exported], **labels}
        destination.write_bytes(pickle.dumps(data, protocol=4))
        # Read the exported file through the original task, and replay on a
        # freshly reset layout. The planner is not used for verification.
        env.reset(seed=[index])
        env.run_reward()
        env.query_support_arm_traj(0)
        replay = DemoRecorder(env, record=False)
        while env.support_arm_action[0]:
            replay.hold(1)
        replay.hold(20)
        env.check_support_arm_stable(0)
        inside = [
            env.reward_manager.call_func_parser(env.reward_manager.is_A_in_B(label_A=f"aim{i}", label_B="basket1"), 0)
            for i in range(5)
        ]
        passed = min(inside) >= 0.5 and not env.unstable_envs
        replay_attempts.append(
            {"stride": stride, "control_steps": len(exported), "aim_in_basket": inside, "passed": passed}
        )
        print("SUPPORT_REPLAY_RESULT", replay_attempts[-1], flush=True)
        if passed:
            break
    else:
        raise PlanningFailure(f"Exported support demonstration failed native replay: {replay_attempts}")
    return {
        "status": "passed",
        "physics_steps": len(exported),
        "planning_physics_steps": len(recorder.steps),
        "native_replay": True,
        "trajectory_export_stride": stride,
        "replay_export_attempts": replay_attempts,
        "live_terminal_check": True,
        "teacher_placement": "measured_bbox_with_rim_clearance",
        "gripper_timing_preserved": True,
        "target_label": data["target_label"],
        "aim_in_basket": inside,
        "offline_teacher_cuda_graphs": env.robot_manager.planner[robot.robot_name].use_cuda_graph,
        "offline_teacher_process": "isolated cuRobo; native replay in Isaac",
    }
