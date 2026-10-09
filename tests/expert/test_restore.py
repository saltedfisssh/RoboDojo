from types import SimpleNamespace

import numpy as np
import pytest

from src.expert import restore


@pytest.mark.parametrize("budget,finish,expected_frame", [(100, 4, 3), (3, 4, 3), (100, 2, 2)])
def test_replay_drift_compares_the_same_pre_action_frame(monkeypatch, budget, finish, expected_frame):
    env = SimpleNamespace(take_action_cnt=[1], end_flag=[False])
    actions = []
    observations = []

    def take_action(action):
        actions.append(action)
        env.take_action_cnt[0] += 1
        env.end_flag[0] = env.take_action_cnt[0] >= finish

    task_env = SimpleNamespace(take_action=take_action, get_obs=lambda: observations.append(True))
    # A perfect deterministic replay has zero error at matching checkpoints.
    adapter = SimpleNamespace(rollout_errors=lambda state: {"error": abs(env.take_action_cnt[0] - state)})
    monkeypatch.setattr(restore, "checkpoint", lambda f, index: (index, {}))
    f = {"action": np.arange(4 * 14).reshape(4, 14)}
    report = restore.replay_recording(task_env, env, f, adapter, frame=1, budget=budget, video=True)
    assert report == {"replay_drift_frame": expected_frame, "replay_drift": {"error": 0}}
    assert len(actions) == len(observations) == min(budget, finish) - 1
    np.testing.assert_array_equal(actions[0]["left_arm_joint_state"], f["action"][1, :6])
    np.testing.assert_array_equal(actions[0]["right_ee_joint_state"], f["action"][1, 13:14])
