import json
from types import SimpleNamespace

import pytest

from src.datagen.interaction import capture, restore
from src.datagen.state import decode_context, encode_context


def test_opponent_mid_turn_restore_keeps_only_unexecuted_actions():
    manager = SimpleNamespace(
        plan_ee=lambda *a, **kw: [{"arm": i} for i in range(6)],
        plan_endeffector_joint=lambda *a, **kw: [{"gripper": i + 10} for i in range(6)],
    )
    original = SimpleNamespace(
        interact=True,
        datagen_task="play_tic_tac_toe",
        query_support_times=[1],
        pending_support_check=[(0, 2)],
        support_arm_action=[[{}, {}]],
    )
    saved = decode_context(json.loads(json.dumps(encode_context(capture(original)))))
    loaded = SimpleNamespace(
        interact=True,
        datagen_task="play_tic_tac_toe",
        robot_manager=manager,
        traj=[[{}, {}, {"arm": None, "eef": None}]],
    )
    restore(loaded, saved)
    assert loaded.query_support_times == [1]
    assert loaded.pending_support_check == [(0, 2)]
    assert loaded.support_arm_action == [[{"arm": 4, "gripper": 14}, {"arm": 5, "gripper": 15}]]
    saved["remaining_controls"] = 7
    with pytest.raises(ValueError, match="cursor"):
        restore(loaded, saved)


def test_unknown_or_missing_interaction_state_is_rejected():
    with pytest.raises(ValueError, match="not implemented"):
        capture(SimpleNamespace(interact=True, datagen_task="unknown_interactive_task"))
    with pytest.raises(ValueError, match="missing"):
        restore(SimpleNamespace(interact=True), None)
