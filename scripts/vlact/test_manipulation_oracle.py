import tempfile
from types import SimpleNamespace
import unittest

from src.eval_client.manipulation_oracle import (
    ManipulationOracle,
    StageState,
    charger_stage,
    pour_stage,
    tubes_stage,
)


class ManipulationOracleTest(unittest.TestCase):
    def test_charger_grasp_point_offset_and_handover_release(self):
        class Rewards:
            def is_all_gripper_open(self, **kwargs):
                return False

            def is_lift(self, **kwargs):
                return True

            def is_A_depth_in_B(self, **kwargs):
                return False

            def is_A_in_B(self, **kwargs):
                return True

            def is_axis_up(self, **kwargs):
                return True

            def check_once(self, conditions, env_idx, op="or"):
                return all(conditions) if op == "and" else any(conditions)

        robots = [
            SimpleNamespace(type="target", arm_name=f"{arm}_arm", gripper_scale=[0, 1], gripper_bias=0.145)
            for arm in ("left", "right")
        ]
        openings = {"left_arm": 0.2, "right_arm": 0.2}
        manager = SimpleNamespace(
            robot_list=robots,
            get_real_endpose=lambda robot, **kwargs: {0: [0.055, 0, 0.9, 1, 0, 0, 0]},
            get_end_effector_real_val=lambda robot, **kwargs: {0: openings[robot.arm_name]},
        )
        layout = SimpleNamespace(get_instance_pose=lambda **kwargs: ([0.2, 0, 0.9], [1, 0, 0, 0]))
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(
                task_name="plug_in_charger",
                save_dir=directory,
                reward_manager=Rewards(),
                robot_manager=manager,
                scene_manager=SimpleNamespace(layout_manager=layout),
            )
            oracle = ManipulationOracle(env)
            state = StageState("handover", context={"source_arm": "left", "target_arm": "right"})
            facts = oracle.facts(0, state)
            self.assertFalse(facts["inserted"])
            self.assertFalse(facts["held_by_target"])
            self.assertAlmostEqual(facts["gripper_distances"]["right"], 0)
            openings["left_arm"] = 0.9
            self.assertTrue(oracle.facts(0, state)["held_by_target"])

    def test_debounce_duplicate_flicker_and_recovery(self):
        state = StageState("pick")
        for _ in range(5):
            state.update(0, "pour")
        state.update(1, "pick")
        state.update(2, "pour")
        state.update(3, "pour")
        self.assertEqual(state.stage, "pick")
        state.update(4, "pour")
        self.assertEqual((state.stage, state.generation), ("pour", 1))
        for step in range(5, 8):
            state.update(step, "pick")
        self.assertEqual((state.stage, state.generation), ("pick", 2))

    def test_charger_handover_insert_and_drop(self):
        facts = dict(
            inserted=False,
            grippers_open=False,
            lifted=True,
            held_by_source=True,
            held_by_target=False,
            handover_needed=True,
        )
        self.assertEqual(charger_stage("pick", facts), "handover")
        facts.update(held_by_source=False, held_by_target=True)
        self.assertEqual(charger_stage("handover", facts), "insert")
        facts.update(lifted=False)
        self.assertEqual(charger_stage("insert", facts), "insert")
        facts.update(held_by_target=False)
        self.assertEqual(charger_stage("insert", facts), "pick")
        facts.update(inserted=True)
        self.assertEqual(charger_stage("pick", facts), "release")
        facts.update(grippers_open=True)
        self.assertEqual(charger_stage("release", facts), "retract")

    def test_tubes_release_gate_and_current_count_recovery(self):
        facts = dict(inserted_count=1, grippers_open=False)
        self.assertEqual(tubes_stage("first", facts), "first")
        facts["grippers_open"] = True
        self.assertEqual(tubes_stage("first", facts), "second")
        facts["inserted_count"] = 0
        self.assertEqual(tubes_stage("second", facts), "first")
        facts.update(inserted_count=3, grippers_open=False)
        self.assertEqual(tubes_stage("third", facts), "release")
        facts["grippers_open"] = True
        self.assertEqual(tubes_stage("release", facts), "retract")

    def test_pour_does_not_finish_on_partial_contents_or_dropped_cup(self):
        facts = dict(balls_in_vase=6, cup_on_table=True, grippers_open=True, lifted=False)
        self.assertEqual(pour_stage("pour", facts), "pick")
        facts.update(lifted=True, grippers_open=False)
        self.assertEqual(pour_stage("pick", facts), "pour")
        facts.update(balls_in_vase=7, cup_on_table=False)
        self.assertEqual(pour_stage("pour", facts), "place")
        facts.update(cup_on_table=True)
        self.assertEqual(pour_stage("place", facts), "place")
        facts.update(grippers_open=True)
        self.assertEqual(pour_stage("place", facts), "retract")
        facts.update(balls_in_vase=6)
        self.assertEqual(pour_stage("retract", facts), "pour")

    def test_tube_provider_requires_all_predicates_and_resets_per_env(self):
        class Rewards:
            def is_all_gripper_open(self, **kwargs):
                return True

            def check_once(self, conditions, env_idx, op="or"):
                return all(conditions) if op == "and" else any(conditions)

        checks = [[True, True, False], [False, False, False], [False, False, False]]
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(
                task_name="insert_tubes",
                save_dir=directory,
                take_action_cnt=[0, 0],
                current_env_seed_map={0: 0, 1: 1},
                reward_manager=Rewards(),
                _tube_checks=lambda i: checks[i],
            )
            oracle = ManipulationOracle(env)
            for step in range(3):
                env.take_action_cnt[0] = step
                oracle.annotate(0, {})
            self.assertEqual(oracle.states[0].stage, "first")
            checks[0][-1] = True
            for step in range(3, 6):
                env.take_action_cnt[0] = step
                oracle.annotate(0, {})
            self.assertEqual(oracle.states[0].stage, "second")
            oracle.annotate(1, {})
            self.assertEqual(oracle.states[1].stage, "first")
            env.take_action_cnt[0] = 0
            obs = {}
            oracle.annotate(0, obs)
            self.assertEqual(oracle.states[0].stage, "first")
            self.assertTrue(obs["subtask_id"].startswith("1:"))


if __name__ == "__main__":
    unittest.main()
