"""Task programs built from observable approach/grasp/lift/transfer/place/retract skills."""

from time import perf_counter

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.curriculum import TASKS as TABLETOP_TASKS
from src.datagen.state import terminal_checks

SUPPORTED_TASKS = (
    "stack_blocks",
    "general_pickup",
    "stack_blocks_by_language",
    "swap_T",
    "sort_nesting_dolls_by_size",
) + TABLETOP_TASKS


class PlanningFailure(RuntimeError):
    pass


class RigidPlanner:
    def __init__(self, env, actions, recorder, state_adapter, max_frames=1800):
        self.env, self.actions = env, actions
        self.recorder, self.state_adapter = recorder, state_adapter
        self.rm = env.robot_manager
        self.segment_id = -1
        self.segment_start = 0
        self.annotation = None
        self.failure_reason = None
        self.timings = dict(observation=0.0, images=0.0, checkpoint=0.0, write=0.0, physics=0.0)
        self.max_frames = max_frames
        self.terminal_checks = terminal_checks(env.reward_manager)

    def stage(self, primitive, text, *, arm=None, object_label=None):
        self.segment_id += 1
        self.segment_start = self.recorder.count
        self.annotation = {
            "segment_id": self.segment_id,
            "primitive": primitive,
            "text": text,
            "arm": arm,
            "object": object_label,
            "source": "planner",
            "start_frame": self.segment_start,
            "goal": {},
        }
        print(f"[planner] {self.segment_id}: {text}", flush=True)

    def tick(self, command):
        if self.recorder.count >= self.max_frames:
            raise PlanningFailure(f"Exceeded {self.max_frames} frames")
        stamp = perf_counter()
        observed, measured = self.actions.observation_state(), self.actions.measured_state()
        now = perf_counter()
        self.timings["observation"] += now - stamp
        stamp = now
        images = self.actions.images()
        now = perf_counter()
        self.timings["images"] += now - stamp
        stamp = now
        snapshot = self.state_adapter.capture()
        now = perf_counter()
        self.timings["checkpoint"] += now - stamp
        stamp = now
        self.recorder.append(observed, measured, command, images, snapshot, self.annotation)
        now = perf_counter()
        self.timings["write"] += now - stamp
        stamp = now
        self.actions.apply(command)
        self.timings["physics"] += perf_counter() - stamp
        if self.recorder.count % 100 == 0:
            totals = ", ".join(f"{key}={value:.1f}s" for key, value in self.timings.items())
            print(f"[timing] frames={self.recorder.count}: {totals}", flush=True)

    def hold(self, frames=10):
        for _ in range(frames):
            self.tick(self.actions.command.copy())

    def object_pose(self, label):
        lm = self.env.scene_manager.layout_manager
        pos, quat = lm.get_instance_pose(0, label=label)
        return np.asarray(pos, dtype=float).copy(), np.asarray(quat, dtype=float).copy()

    def object_vertices(self, label):
        lm = self.env.scene_manager.layout_manager
        data = lm.get_instance_metadata(0, label=label)
        vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])
        pos, quat = self.object_pose(label)
        return vertices @ quat2mat(quat).T + pos

    def move(self, arm, ee_position, ee_quaternion, settle=5, result=None):
        self.annotation["goal"].update(
            ee_position=np.asarray(ee_position).tolist(), ee_quaternion=np.asarray(ee_quaternion).tolist()
        )
        r = self.actions.robots[arm]
        planner = self.rm.planner[r.robot_name]
        pose = np.r_[ee_position, ee_quaternion]
        if result is None:
            result = planner.plan_path(self.rm.get_joint(r)[0], pose, r.entity_origin_pose)
        if result["status"] != "Success":
            raise PlanningFailure(f"cuRobo could not reach {pose.tolist()} with arm {arm}")
        path = np.asarray(result["position"])
        indices = list(range(self.actions.ticks - 1, len(path), self.actions.ticks))
        if not indices or indices[-1] != len(path) - 1:
            indices.append(len(path) - 1)
        for i in indices:
            command = self.actions.command.copy()
            command[arm * 7 : arm * 7 + 6] = path[i]
            self.tick(command)
        self.hold(settle)
        actual = self.rm.get_real_endpose(r)[0]
        error = np.linalg.norm(actual[:3] - ee_position)
        if error > 0.025:
            raise PlanningFailure(f"End effector tracking error {error:.4f} m")

    def gripper(self, arm, opening, frames=15):
        self.annotation["goal"]["gripper_open"] = opening
        command = self.actions.command.copy()
        command[arm * 7 + 6] = opening
        for _ in range(frames):
            self.tick(command)

    def linear(self, arm, target, quaternion, *, step=0.004, angular_step=np.deg2rad(3), settle=5, stop_when=None):
        """Straight Cartesian contact motion with locally continuous cuRobo IK."""
        from src.datagen.ik import cartesian_waypoints, solve_nearby

        r = self.actions.robots[arm]
        start = self.rm.get_real_endpose(r)[0].copy()
        target = np.asarray(target)
        self.annotation["goal"].update(ee_position=target.tolist(), ee_quaternion=np.asarray(quaternion).tolist())
        solver = self.rm.planner[r.robot_name]
        for pose in cartesian_waypoints(start, target, quaternion, step, angular_step):
            joints = self.rm.get_joint(r)[0]
            result = solver.solve_ik_to_joint(joints, pose, r.entity_origin_pose)
            if result["status"] != "Success" or np.max(np.abs(result["joint_value"] - joints)) > 0.35:
                nearby = solve_nearby(solver, joints, pose, r.entity_origin_pose)
                if nearby is None:
                    if result["status"] != "Success":
                        raise PlanningFailure(f"Contact motion IK failed for arm {arm} at {pose.tolist()}")
                    raise PlanningFailure(
                        f"Contact motion would jump between IK branches for arm {arm} "
                        f"at {pose.tolist()}; "
                        f"current={np.asarray(joints).tolist()}, best={result['joint_value'].tolist()}"
                    )
                result["joint_value"] = nearby
            command = self.actions.command.copy()
            command[arm * 7 : arm * 7 + 6] = result["joint_value"]
            self.tick(command)
            if stop_when is not None and stop_when():
                break
        self.hold(settle)

    def functional_pose(self, label, tag, kind="passive", index=0):
        lm = self.env.scene_manager.layout_manager
        points = lm.get_functional_points(
            tag,
            kind,
            config=lm.get_instance_metadata(0, label=label),
            obj_name=lm.get_instance_name(0, label),
            env_idx=0,
        )
        if len(points) <= index:
            raise PlanningFailure(f"Missing {kind} functional point {label}/{tag}")
        return np.asarray(points[index])

    def support_pose(self, label, tag, index=0):
        lm = self.env.scene_manager.layout_manager
        points, _ = lm.get_support_points(
            tag,
            "passive",
            config=lm.get_instance_metadata(0, label=label),
            obj_name=lm.get_instance_name(0, label),
            env_idx=0,
        )
        if len(points) <= index:
            raise PlanningFailure(f"Missing support point {label}/{tag}")
        return np.asarray(points[index])

    def pick(
        self,
        label,
        arm=None,
        *,
        grasp_point=None,
        yaw_candidates=(180, 90, 270, 0),
        lift_height=0.12,
        linear_approach=False,
        pregrasp_opening=None,
        grasp_opening=None,
        linear_lift=False,
        minimum_lift=0.08,
    ):
        if not 0 < minimum_lift <= lift_height:
            raise ValueError("minimum_lift must be positive and no greater than lift_height")
        pos, _ = self.object_pose(label)
        if arm is None:
            arm = 0 if pos[0] < 0 else 1
        r = self.actions.robots[arm]
        # X5 link6 +X points through the fingers; +Y spans the jaws. Approach vertically.
        rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=float)
        vertices = self.object_vertices(label)
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        grasp = center.copy()
        grasp[2] = max(center[2], min(vertices[:, 2].min() + 0.018, vertices[:, 2].max() - 0.002))
        if grasp_point is not None:
            grasp = np.asarray(grasp_point).copy()
        wrist = grasp - rotation[:, 0] * r.gripper_bias
        above = wrist + [0, 0, 0.08]
        planner = self.rm.planner[r.robot_name]
        approach_plan = None
        for yaw in yaw_candidates:
            candidate_rotation = euler2mat(0, 0, np.deg2rad(yaw)) @ rotation
            if grasp_point is None:
                width = np.ptp(vertices @ candidate_rotation[:, 1])
                if width > 2 * max(r.gripper_scale) - 0.003:
                    continue
            quat = mat2quat(candidate_rotation)
            grasp_ik = planner.solve_ik_to_joint(self.rm.get_joint(r)[0], np.r_[wrist, quat], r.entity_origin_pose)
            if grasp_ik["status"] != "Success":
                continue
            approach_plan = planner.plan_path(self.rm.get_joint(r)[0], np.r_[above, quat], r.entity_origin_pose)
            if approach_plan["status"] == "Success":
                break
        if approach_plan is None or approach_plan["status"] != "Success":
            raise PlanningFailure(f"No feasible top grasp approach for {label}")
        self.stage("approach", f"Move the {['left', 'right'][arm]} gripper above {label}.", arm=arm, object_label=label)
        self.move(arm, above, quat, result=approach_plan)
        if pregrasp_opening is not None:
            self.stage("preshape", f"Narrow the open fingers to approach {label}.", arm=arm, object_label=label)
            self.gripper(arm, pregrasp_opening, frames=8)
        self.stage("descend", f"Lower the open gripper around {label}.", arm=arm, object_label=label)
        if linear_approach:
            self.linear(arm, wrist, quat)
        else:
            self.move(arm, wrist, quat)
        self.stage("grasp", f"Close the gripper on {label}.", arm=arm, object_label=label)
        if grasp_opening is None:
            self.gripper(arm, 0)
        else:
            # A small block resting squarely on a high-friction pad can be
            # ejected by fast, full-force closure. Approach a task-calibrated
            # jaw gap gradually and retain that target while carrying it.
            opening = float(self.actions.command[arm * 7 + 6])
            for target_opening in np.linspace(opening, grasp_opening, 26)[1:]:
                self.gripper(arm, float(target_opening), frames=1)
            self.hold(8)
        self.stage("lift", f"Lift {label} clear of the table.", arm=arm, object_label=label)
        self.annotation["goal"]["object_min_z"] = float(pos[2] + minimum_lift)
        if linear_lift:
            self.linear(arm, wrist + [0, 0, lift_height], quat, settle=10)
        else:
            self.move(arm, wrist + [0, 0, lift_height], quat, settle=10)
        lifted, _ = self.object_pose(label)
        if lifted[2] - pos[2] < minimum_lift:
            raise PlanningFailure(f"{label} did not lift: dz={lifted[2] - pos[2]:.4f}")
        return arm, quat

    def place(
        self,
        label,
        target_position,
        arm,
        quat,
        target_quaternion=None,
        *,
        precise=False,
        contact=False,
        alignment_tolerance_m=None,
        approach_clearance=0.10,
        linear_transfer=False,
        retract_along_tool=False,
        retract_offset=None,
    ):
        if retract_offset is not None:
            retract_offset = np.asarray(retract_offset, dtype=float)
            if retract_offset.shape != (3,) or not np.isfinite(retract_offset).all():
                raise ValueError("Retraction offset must be a finite 3-vector")
        if approach_clearance <= 0:
            raise ValueError("Placement approach clearance must be positive")
        r = self.actions.robots[arm]
        pos, object_quat = self.object_pose(label)
        ee = self.rm.get_real_endpose(r)[0]
        # Closed-loop object-to-wrist offset after contact (includes grasp slip).
        offset = ee[:3] - pos
        if target_quaternion is not None:
            delta = quat2mat(target_quaternion) @ quat2mat(object_quat).T
            offset = delta @ offset
            quat = mat2quat(delta @ quat2mat(ee[3:]))
        target_ee = np.asarray(target_position) + offset
        above = target_ee.copy()
        above[2] = max(above[2] + approach_clearance, ee[2])
        self.stage("transfer", f"Carry {label} above its destination.", arm=arm, object_label=label)
        if linear_transfer:
            # Thin grasp contacts can roll during a fast cuRobo transfer.
            # Interpolate at 1.5 mm / 1 degree per recorded control frame.
            self.linear(arm, above, quat, step=0.0015, angular_step=np.deg2rad(1), settle=10)
        else:
            solver = self.rm.planner[r.robot_name]
            result = solver.plan_path(self.rm.get_joint(r)[0], np.r_[above, quat], r.entity_origin_pose)
            if result["status"] != "Success":
                # A tall destination can put a 10 cm overhead waypoint outside X5 reach.
                # Use at most 5 cm (or the explicit task clearance) and never lower
                # below the current wrist during the horizontal transfer.
                above[2] = max(target_ee[2] + min(approach_clearance, 0.05), ee[2])
                result = solver.plan_path(self.rm.get_joint(r)[0], np.r_[above, quat], r.entity_origin_pose)
            self.move(arm, above, quat, result=result)
        if precise:
            # Re-measure after transport: thin objects can slip within the fingers.
            pos, object_quat = self.object_pose(label)
            ee = self.rm.get_real_endpose(r)[0]
            expected_object = above - offset
            if np.linalg.norm(pos - expected_object) > 0.05:
                raise PlanningFailure(f"{label} left the gripper during transfer")
            delta = np.eye(3) if target_quaternion is None else quat2mat(target_quaternion) @ quat2mat(object_quat).T
            target_ee = np.asarray(target_position) + delta @ (ee[:3] - pos)
            quat = mat2quat(delta @ quat2mat(ee[3:]))
        if alignment_tolerance_m is not None:
            if not precise or not contact or alignment_tolerance_m <= 0:
                raise ValueError("Insertion alignment requires precise contact placement and a positive tolerance")
            self.stage(
                "align_insertion", f"Center {label} over the peg before descending.", arm=arm, object_label=label
            )
            self.annotation["goal"].update(
                object_xy_position=np.asarray(target_position)[:2].tolist(),
                position_tolerance_m=alignment_tolerance_m,
            )
            for _ in range(5):
                pos, _ = self.object_pose(label)
                correction = np.asarray(target_position) - pos
                correction[2] = 0
                if np.linalg.norm(correction) <= alignment_tolerance_m:
                    break
                ee = self.rm.get_real_endpose(r)[0]
                self.linear(arm, ee[:3] + correction, quat, step=0.001, settle=8)
            pos, object_quat = self.object_pose(label)
            if np.linalg.norm(pos[:2] - np.asarray(target_position)[:2]) > alignment_tolerance_m:
                raise PlanningFailure(f"{label} could not center above its insertion target")
            ee = self.rm.get_real_endpose(r)[0]
            delta = np.eye(3) if target_quaternion is None else quat2mat(target_quaternion) @ quat2mat(object_quat).T
            target_ee = np.asarray(target_position) + delta @ (ee[:3] - pos)
            quat = mat2quat(delta @ quat2mat(ee[3:]))
        self.stage("place", f"Lower {label} onto its destination.", arm=arm, object_label=label)
        self.annotation["goal"]["object_position"] = np.asarray(target_position).tolist()
        if target_quaternion is not None:
            self.annotation["goal"]["object_quaternion"] = np.asarray(target_quaternion).tolist()
        if precise:
            self.annotation["goal"].update(position_tolerance_m=0.006, orientation_tolerance_deg=7.0)
        if contact:
            self.linear(arm, target_ee, quat, step=0.001 if alignment_tolerance_m is not None else 0.003, settle=10)
        else:
            self.move(arm, target_ee, quat)
        if precise:
            for _ in range(2):
                actual, _ = self.object_pose(label)
                correction = np.asarray(target_position) - actual
                correction[2] = 0  # Do not push a supported object through its destination.
                if np.linalg.norm(correction) <= 0.006:
                    break
                if np.linalg.norm(correction) > 0.05:
                    raise PlanningFailure(f"{label} slipped out of its precision placement grasp")
                ee = self.rm.get_real_endpose(r)[0]
                self.linear(arm, ee[:3] + correction, ee[3:], step=0.002, settle=8)
        self.stage("release", f"Open the gripper to release {label}.", arm=arm, object_label=label)
        if precise:
            # Thin digits can catch a finger when opened in one step and slide
            # off the pad. Release gradually after the object is supported.
            for opening in np.linspace(float(self.actions.command[arm * 7 + 6]), 1, 21)[1:]:
                self.gripper(arm, float(opening), frames=1)
            self.hold(5)
        else:
            self.gripper(arm, 1)
        if retract_along_tool or retract_offset is not None:
            self.stage("retract", f"Withdraw the fingers clear of {label}.", arm=arm, object_label=label)
            ee = self.rm.get_real_endpose(r)[0]
            offset = -quat2mat(ee[3:])[:, 0] * 0.08 if retract_offset is None else retract_offset
            self.linear(arm, ee[:3] + offset, ee[3:], step=0.003)
        else:
            self.stage("retract", f"Raise the gripper away from {label}.", arm=arm, object_label=label)
            self.move(arm, above, quat)

    def pick_side(
        self,
        label,
        grasp_point,
        rotation,
        arm,
        lift_height=0.10,
        *,
        pregrasp_opening=None,
        approach_result=None,
        linear_lift=False,
        grasp_ramp_frames=0,
    ):
        """A calibrated side grasp, keeping a held container opening unobstructed."""
        r = self.actions.robots[arm]
        rotation = np.asarray(rotation, dtype=float)
        quat = mat2quat(rotation)
        wrist = np.asarray(grasp_point) - rotation[:, 0] * r.gripper_bias
        approach = wrist - rotation[:, 0] * 0.07
        initial, _ = self.object_pose(label)
        if pregrasp_opening is not None:
            self.stage("preshape", f"Narrow the fingers for the side grasp of {label}.", arm=arm, object_label=label)
            self.gripper(arm, pregrasp_opening, frames=8)
        self.stage("approach", f"Approach the side of {label}.", arm=arm, object_label=label)
        self.move(arm, approach, quat, result=approach_result)
        self.stage("align_grasp", f"Align the open fingers around {label}.", arm=arm, object_label=label)
        self.linear(arm, wrist, quat)
        self.stage("grasp", f"Close the gripper around {label}.", arm=arm, object_label=label)
        if grasp_ramp_frames:
            opening = float(self.actions.command[arm * 7 + 6])
            for value in np.linspace(opening, 0, grasp_ramp_frames + 1)[1:]:
                self.gripper(arm, float(value), frames=1)
            self.hold(5)
        else:
            self.gripper(arm, 0)
        self.stage("lift", f"Lift and support {label} from the side.", arm=arm, object_label=label)
        self.annotation["goal"]["object_min_z"] = float(initial[2] + 0.08)
        if linear_lift:
            self.linear(arm, wrist + [0, 0, lift_height], quat, step=0.002)
        else:
            self.move(arm, wrist + [0, 0, lift_height], quat)
        lifted, _ = self.object_pose(label)
        if lifted[2] < initial[2] + 0.08:
            raise PlanningFailure(f"Side grasp failed to lift {label}")
        return arm, quat

    def home(self, arm):
        r = self.actions.robots[arm]
        self.stage("return_home", f"Return the {['left', 'right'][arm]} arm to its initial pose.", arm=arm)
        self.annotation["goal"]["joint_position"] = self.actions.home[arm * 7 : arm * 7 + 6].tolist()
        result = self.rm.planner[r.robot_name].plan_joint(
            self.rm.get_joint(r)[0], self.actions.home[arm * 7 : arm * 7 + 6]
        )
        if result["status"] != "Success":
            raise PlanningFailure("Return-to-home plan failed")
        path = result["position"]
        for i in list(range(self.actions.ticks - 1, len(path), self.actions.ticks)) + [len(path) - 1]:
            command = self.actions.command.copy()
            command[arm * 7 : arm * 7 + 6] = path[i]
            self.tick(command)
        self.hold(15)

    def relocate(
        self,
        label,
        target,
        target_quaternion=None,
        *,
        precise=False,
        grasp_point=None,
        yaw_candidates=(180, 90, 270, 0),
        approach_clearance=0.10,
        grasp_opening=None,
        linear_lift=False,
    ):
        """Transfer across arm workspaces through a reachable table buffer."""
        pos, _ = self.object_pose(label)
        target_arm = 0 if target[0] < 0 else 1
        source_arm = (0 if pos[0] < 0 else 1) if abs(pos[0]) > 0.03 else target_arm
        arm, quat = self.pick(
            label,
            arm=source_arm,
            grasp_point=grasp_point,
            yaw_candidates=yaw_candidates,
            grasp_opening=grasp_opening,
            linear_lift=linear_lift,
        )
        if source_arm != target_arm and abs(target[0]) > 0.03:
            lifted, _ = self.object_pose(label)
            buffer = np.array([0.0, -0.27, 0.767 + lifted[2] - self.object_vertices(label)[:, 2].min()])
            self.place(label, buffer, arm, quat, target_quaternion=target_quaternion)
            self.home(arm)
            arm, quat = self.pick(label, arm=target_arm, grasp_opening=grasp_opening, linear_lift=linear_lift)
        self.place(
            label,
            target,
            arm,
            quat,
            target_quaternion=target_quaternion,
            precise=precise,
            approach_clearance=approach_clearance,
        )
        self.home(arm)

    def run(self, task):
        if task in TABLETOP_TASKS:
            from src.datagen.programs.tabletop import run

            run(self, task)
        elif task == "general_pickup":
            self.pick("target")
        elif task in ("stack_blocks", "stack_blocks_by_language"):
            base = "block_0"
            for label in ("block_1", "block_2"):
                arm, quat = self.pick(label)
                base_pos, _ = self.object_pose(base)
                obj_pos, _ = self.object_pose(label)
                bottom_offset = obj_pos[2] - self.object_vertices(label)[:, 2].min()
                target = base_pos.copy()
                target[2] = self.object_vertices(base)[:, 2].max() + bottom_offset + 0.002
                self.place(label, target, arm, quat)
                self.home(arm)
                base = label
        elif task == "swap_T":
            targets = {label: self.object_pose(label) for label in ("t0", "t1")}
            buffer = targets["t0"][0].copy()
            buffer[:2] = [0, -0.37]
            for label, target, target_q in (
                ("t0", buffer, targets["t0"][1]),
                ("t1", *targets["t0"]),
                ("t0", *targets["t1"]),
            ):
                self.relocate(label, target, target_quaternion=target_q)
        elif task == "sort_nesting_dolls_by_size":
            lm = self.env.scene_manager.layout_manager
            labels = sorted(
                (f"doll{i}" for i in range(5)),
                key=lambda label: lm.get_instance_metadata(0, label=label)["model_id"] % 5,
                reverse=True,
            )
            # The sampler shuffles the two pairs within each arm workspace;
            # swap each inverted pair via a free table position on that side.
            for arm, pair in enumerate(((0, 1), (3, 4))):
                first, second = (f"doll{i}" for i in pair)
                if labels[pair[0]] == first:
                    continue
                targets = {label: self.object_pose(label) for label in (first, second)}
                buffer = targets[first][0].copy()
                buffer[:2] = [-0.27 if arm == 0 else 0.27, -0.36]
                for label, target in ((first, buffer), (second, targets[first][0]), (first, targets[second][0])):
                    side, quat = self.pick(label, arm=arm)
                    # Different sizes have different USD origins above the table.
                    pos, _ = self.object_pose(label)
                    target = np.asarray(target).copy()
                    target[2] = 0.767 + pos[2] - self.object_vertices(label)[:, 2].min()
                    self.place(label, target, side, quat, target_quaternion=[1, 0, 0, 0])
                    self.home(side)
        else:
            raise ValueError(f"No task planner for {task}")
        return self.finish_task()

    def finish_task(self):
        """Recheck original terminal predicates after a full program or restored suffix."""
        self.stage("settle", "Wait for the final physical state to settle.")
        stable = []
        failures = []
        for _ in range(15):
            self.hold(1)
            # The task RewardManager latches successes. Re-evaluate the ORIGINAL
            # final stage predicates so a later drop fails, while earlier
            # sequential stages (cover/uncover, press/release) need not co-exist.
            failures = [check for check in self.terminal_checks if not self.env.reward_manager.check_once(check, 0)]
            stable.append(not failures)
        passed = all(stable[-10:]) and bool(self.env.reward_manager.get_reward()[0]) and all(self.env.success)
        if not passed:
            remaining = self.env.reward_manager.check_list[0]
            if failures:
                self.failure_reason = f"Final predicates failed: {failures!r}"
            elif remaining:
                self.failure_reason = f"{len(remaining)} sequential stages remain; next: {remaining[0]!r}"
            else:
                self.failure_reason = "Final stability, triggered task history, or support-arm execution did not pass"
            print(f"[planner] {self.failure_reason}", flush=True)
        return passed
