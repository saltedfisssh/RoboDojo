"""Local task definitions absent from the checkout. These are NOT official validators."""

import numpy as np
from transforms3d.euler import euler2quat

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class DisplayLetters(TaskEnv):
    """Instruction-only reconstruction of dataset task 5: spell RoboDojo in a row."""

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 2400  # Local collection limit; no official action budget is known.

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        rm = self.reward_manager
        checks = []
        for i in range(8):
            checks.extend(
                [
                    rm.is_A_xy_distance_close_to_pos(
                        label=f"letter{i}", pos=[-0.28 + i * 0.08, -0.18], dis_threshold=0.012
                    ),
                    rm.is_qpos_close(
                        label_A=f"letter{i}", qpos=euler2quat(-np.pi / 2, 0, 0).tolist(), dis_threshold=15
                    ),
                    rm.is_not_lift(label=f"letter{i}", z_threshold=0.008),
                ]
            )
        rm.check([*checks, rm.all_robot_back_to_origin(), rm.is_all_gripper_open()])

    def gen_instruction(self, env_idx):
        return ['Arrange the letters to spell "RoboDojo" in a row.']
