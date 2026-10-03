"""A freshly planned support-arm demonstration for independent sorting layouts."""

import numpy as np


def training_class(base):
    class TrainingSorting(base):
        def load_support_arm_traj(self):
            order = np.random.default_rng(self.get_env_seed(0) + 7041).permutation(5)
            self.target_label = [[f"t{i}"] for i in order]
            self.aim_label = [[f"aim{i}"] for i in order]
            self.target_place_tag = [["up"] for _ in order]
            self.datagen_support_replay = False

        def query_support_arm_traj(self, env_idx):
            if self.datagen_support_replay and not self.support_arm_action[env_idx]:
                self.query_support_times[env_idx] = 1

    return TrainingSorting


def basket_drop_target(p, label, basket):
    """Use separate basket regions and release above existing physical contents."""
    index = int(label.removeprefix("aim").removeprefix("t"))
    offsets = ((-0.035, -0.075), (0.035, -0.075), (0.035, 0.065), (0, 0), (-0.035, 0.075))
    target, _ = p.object_pose(basket)
    target[:2] += offsets[index]
    pos, _ = p.object_pose(label)
    vertices = p.object_vertices(label)
    lower = vertices.min(axis=0) - pos
    upper = vertices.max(axis=0) - pos
    support_top = p.object_vertices(basket)[:, 2].max()
    prefix = "aim" if label.startswith("aim") else "t"
    for other_index in range(5):
        other = f"{prefix}{other_index}"
        if other == label:
            continue
        other_pos, _ = p.object_pose(other)
        basket_pos, _ = p.object_pose(basket)
        if np.linalg.norm(other_pos[:2] - basket_pos[:2]) > 0.18:
            continue
        occupied = p.object_vertices(other)
        if np.all(occupied.max(0)[:2] > target[:2] + lower[:2]) and np.all(
            occupied.min(0)[:2] < target[:2] + upper[:2]
        ):
            support_top = max(support_top, occupied[:, 2].max())
    target[2] = support_top + 0.010 - lower[2]
    return target


class SupportPlanner:
    def __init__(self, planner):
        from transforms3d.quaternions import mat2quat

        self.p = planner
        self.rm = planner.rm
        self.robot = self.rm.get_robot_by_arm_name("support_arm0")
        self.home = self.rm.get_joint(self.robot)[0].copy()
        # Franka's panda_hand approaches along local +Z, unlike X5's local +X.
        self.quat = mat2quat(np.diag([1, -1, -1]))

    def execute(self, controls, *, settle=5):
        from src.datagen.planner import PlanningFailure

        if controls is None:
            raise PlanningFailure("Support-arm cuRobo plan failed")
        self.p.env.support_arm_action[0] = controls
        while self.p.env.support_arm_action[0]:
            self.p.hold(1)
        self.p.hold(settle)

    def move(self, point):
        solver = self.rm.planner[self.robot.robot_name]
        result = solver.plan_path(
            self.rm.get_joint(self.robot)[0], np.r_[point, self.quat], self.robot.entity_origin_pose
        )
        self.execute(self.rm.plan_ee(0, "support_arm0", result=result, need_plan=False))

    def linear(self, point, step=0.003, *, stop_when=None):
        """Track a straight approach using the same cuRobo IK as the task arms."""
        from src.datagen.ik import cartesian_waypoints, solve_nearby
        from src.datagen.planner import PlanningFailure

        start = self.rm.get_real_endpose(self.robot)[0]
        self.p.annotation["goal"]["support_ee_position"] = np.asarray(point).tolist()
        solver = self.rm.planner[self.robot.robot_name]
        for pose in cartesian_waypoints(start, point, self.quat, step):
            current = self.rm.get_joint(self.robot)[0]
            result = solver.solve_ik_to_joint(current, pose, self.robot.entity_origin_pose)
            if result["status"] != "Success" or np.max(np.abs(result["joint_value"] - current)) > 0.35:
                nearby = solve_nearby(solver, current, pose, self.robot.entity_origin_pose)
                if nearby is None:
                    raise PlanningFailure("Support-arm contact motion lost a continuous IK solution")
                result["joint_value"] = nearby
            positions = np.asarray(
                [current + min((i + 1) / 9, 1) * (result["joint_value"] - current) for i in range(10)]
            )
            controls = self.rm.plan_ee(
                0,
                "support_arm0",
                result={"status": "Success", "position": positions, "velocity": np.zeros_like(positions)},
                need_plan=False,
            )
            self.execute(controls, settle=0)
            if stop_when is not None and stop_when():
                return
        self.p.hold(5)
        actual = self.rm.get_real_endpose(self.robot)[0]
        error = np.linalg.norm(actual[:3] - point)
        if error > 0.015:
            raise PlanningFailure(f"Support-arm Cartesian tracking error {error:.4f} m at {np.asarray(point).tolist()}")

    def gripper(self, opening):
        self.execute(self.rm.plan_endeffector_joint(0, "support_arm0", target_val=opening, control_step_num=100))

    def demonstrate(self):
        from transforms3d.euler import euler2mat
        from transforms3d.quaternions import mat2quat

        from src.datagen.planner import PlanningFailure

        p = self.p
        for label_row in p.env.aim_label:
            label = label_row[0]
            p.stage(
                "observe_demo",
                f"Observe the support arm put {label} into the demonstration basket.",
                object_label=label,
            )
            pos, _ = p.object_pose(label)
            vertices = p.object_vertices(label)
            grasp = (vertices.min(0) + vertices.max(0)) / 2
            grasp[2] += self.robot.gripper_bias
            # A fallen figure/phone is longer than the Panda's jaw span in
            # one direction. Select the narrow direction in its measured
            # pose and verify the overhead approach with the actual solver.
            rotations = [euler2mat(0, 0, angle) @ np.diag([1, -1, -1]) for angle in (0, np.pi / 2)]
            solver = self.rm.planner[self.robot.robot_name]
            for rotation in sorted(rotations, key=lambda r: np.ptp(vertices @ r[:, 1])):
                if np.ptp(vertices @ rotation[:, 1]) > 2 * max(self.robot.gripper_scale) - 0.003:
                    continue
                self.quat = mat2quat(rotation)
                approach = solver.plan_path(
                    self.rm.get_joint(self.robot)[0],
                    np.r_[grasp + [0, 0, 0.10], self.quat],
                    self.robot.entity_origin_pose,
                )
                if approach["status"] == "Success":
                    break
            else:
                raise PlanningFailure(f"Support arm has no reachable grasp within the jaw span for {label}")
            self.gripper(1)
            self.execute(self.rm.plan_ee(0, "support_arm0", result=approach, need_plan=False))
            self.linear(grasp)
            self.gripper(0)
            # Seed10010 aim3 physically clears the table after 80 mm but
            # the extra ascent loses continuous IK. Its checkpoint-to-basket
            # continuation succeeds; use the measured lift as the endpoint.
            self.linear(
                grasp + [0, 0, 0.12],
                stop_when=lambda: p.object_pose(label)[0][2] >= pos[2] + 0.08,
            )
            actual, _ = p.object_pose(label)
            if actual[2] < pos[2] + 0.07:
                raise PlanningFailure(f"Support arm did not physically lift {label}")
            ee = self.rm.get_real_endpose(self.robot)[0]
            target = basket_drop_target(p, label, "basket1")
            offset = ee[:3] - actual
            self.move(target + offset + [0, 0, 0.10])
            self.linear(target + offset)
            self.gripper(1)
            self.move(target + offset + [0, 0, 0.10])
            p.hold(10)
            check = p.env.reward_manager.is_A_in_B(label_A=label, label_B="basket1")
            if not p.env.reward_manager.check_once(check, 0):
                raise PlanningFailure(f"Support arm did not leave {label} in the demonstration basket")
        result = self.rm.planner[self.robot.robot_name].plan_joint(self.rm.get_joint(self.robot)[0], self.home)
        self.execute(self.rm.plan_ee(0, "support_arm0", result=result, need_plan=False))
        p.env.query_support_times[0] = 1
        p.hold(10)
        if not p.env.success[0]:
            raise PlanningFailure("The generated support-arm demonstration failed the official stability check")


def recorded_suffix(file, start_frame):
    """Reconstruct future support commands from recorded pre-action inputs, including pauses."""
    import json

    from src.datagen.state import decode_context

    steps, last_nonempty = [], 0
    for raw in file["context"][start_frame:]:
        context = decode_context(json.loads(raw).get("interaction"))
        controls = context.get("next_controls", []) if context else []
        steps.extend(controls + [{} for _ in range(10 - len(controls))])
        if controls:
            last_nonempty = len(steps)
    return steps[:last_nonempty]
