import tempfile
from types import SimpleNamespace
import unittest

from src.eval_client.subtask_oracle import PushTOracle, PushTState


class PushTOracleTest(unittest.TestCase):
    def test_alignment_release_and_recovery_require_stable_current_state(self):
        state = PushTState()
        for step in range(3):
            state.update(step, True, False)
        self.assertEqual((state.stage, state.generation), ("release", 1))
        for step in range(3, 6):
            state.update(step, True, True)
        self.assertEqual((state.stage, state.generation), ("retract", 2))
        state.update(6, False, True)
        state.update(7, True, True)
        self.assertEqual(state.stage, "retract")
        for step in range(8, 11):
            state.update(step, False, True)
        self.assertEqual((state.stage, state.generation), ("push", 3))

    def test_duplicate_observations_and_alignment_flicker_do_not_advance(self):
        state = PushTState()
        for _ in range(10):
            state.update(0, True, True)
        self.assertEqual(state.stage, "push")
        state.update(1, False, True)
        state.update(2, True, True)
        state.update(3, True, True)
        self.assertEqual(state.stage, "push")

    def test_provider_requires_position_and_orientation_and_resets_layout(self):
        class Rewards:
            position = False

            def is_qpos_close(self, **kwargs):
                return True

            def is_AB_xy_distance_within_threshold(self, **kwargs):
                return self.position

            def is_all_gripper_open(self, **kwargs):
                return True

            def check_once(self, conditions, env_idx, op="or"):
                return all(conditions) if op == "and" else any(conditions)

        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(
                task_name="push_T",
                save_dir=directory,
                take_action_cnt=[0, 0],
                current_env_seed_map={0: 0, 1: 1},
                reward_manager=Rewards(),
            )
            oracle = PushTOracle(env)
            for step in range(5):
                env.take_action_cnt[0] = step
                oracle.annotate(0, {})
            self.assertEqual(oracle.states[0].stage, "push")
            env.reward_manager.position = True
            for step in range(5, 8):
                env.take_action_cnt[0] = step
                oracle.annotate(0, {})
            self.assertEqual(oracle.states[0].stage, "release")
            oracle.annotate(1, {})
            self.assertEqual(oracle.states[1].stage, "push")
            observation = {}
            env.current_env_seed_map[0] = 2
            oracle.annotate(0, observation)
            self.assertEqual(oracle.states[0].stage, "push")
            self.assertTrue(observation["subtask_id"].startswith("1:"))


if __name__ == "__main__":
    unittest.main()
