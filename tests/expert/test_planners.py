"""Task intent and recovery branches with deterministic scene/motion doubles.

These check goal selection and command ordering; physical feasibility still
requires Isaac. All successful place calls update the live scene.
"""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from src.expert.contracts import PlanningError, TaskContext
from src.expert.planner import TABLE_TOP
from src.expert.planners.sort_nesting_dolls import SortNestingDollsPlanner
from src.expert.planners.stack_blocks import StackBlocksPlanner
from src.expert.planners.stack_bowls import StackBowlsPlanner
from src.expert.planners.swap_t import SwapTPlanner
from src.expert.task_plugins.stack_blocks_by_language.planner import Planner as OrderedStackPlanner


class Scene:
    def __init__(self, positions, sizes=None):
        self.positions = {label: np.array(position, dtype=float) for label, position in positions.items()}
        self.sizes = sizes or {label: index for index, label in enumerate(positions)}
        self.queue = {0: None, 1: None}
        self.events = []
        self.env = object()
        # Compatibility with the pre-refactor task programs for trace comparison.
        self.lm = SimpleNamespace(get_instance_metadata=lambda _, label: self.metadata(label))

    def pose(self, label):
        return self.positions[label].copy(), np.array([1.0, 0, 0, 0])

    def metadata(self, label):
        return {"model_id": self.sizes[label]}

    def labels(self, kind):
        return list(self.positions) if kind == "Rigid" else []

    def local_box(self, label):
        return np.full(3, -0.02), np.full(3, 0.02)

    def world_vertices(self, label):
        return np.asarray(list(product((-0.02, 0.02), repeat=3))) + self.positions[label]

    def top_z(self, label):
        return self.positions[label][2] + 0.02

    def bottom_z(self, label):
        return self.positions[label][2] - 0.02

    def arm_for(self, xy):
        return 0 if xy[0] < 0 else 1

    def begin_subtask(self, text, **kwargs):
        self.events.append(("subtask", text, kwargs))

    def drain(self):
        self.queue = {0: None, 1: None}


class Manipulation:
    def __init__(self, scene, *, single_arm=False, feasible=True):
        self.scene, self.single_arm, self.feasible = scene, single_arm, feasible
        self.placements = []

    def mesh_grasps(self, label):
        return [(self.scene.pose(label)[0], [0.0])]

    def pick_place_grasp(self, label, tcp, yaws, arms, target, **kwargs):
        if not self.feasible:
            return None
        arm = 0 if self.single_arm else arms[0]
        if arm not in arms:
            return None
        return SimpleNamespace(arm=arm, tcp=tcp(0.0) if callable(tcp) else tcp, rotation=np.eye(3))

    def pick(self, label, grasp, **kwargs):
        self.scene.events.append(("pick", label, grasp.arm))

    def place(self, label, arm, target, **kwargs):
        self.placements.append((label, arm, np.asarray(target).copy(), kwargs))
        self.scene.positions[label] = np.asarray(target).copy()
        self.scene.events.append(("place", label, arm))

    def hold_aside(self, label, arm, xy, height, **kwargs):
        self.scene.positions[label][:2] = xy
        self.scene.positions[label][2] += height
        self.scene.events.append(("hold", label, arm))

    def home(self, arms, **kwargs):
        self.scene.events.append(("home", list(arms), kwargs))


def context(task, positions, **kwargs):
    scene = Scene(positions)
    skills = Manipulation(scene, **kwargs)
    return TaskContext(task, 42, {}, scene.env, scene, skills)


@pytest.mark.parametrize(
    "planner,task,labels",
    [
        (StackBlocksPlanner, "stack_blocks", ("block_0", "block_1", "block_2")),
        (StackBowlsPlanner, "stack_bowls", ("bowl0", "bowl1", "bowl2")),
    ],
)
def test_stacking_uses_live_support_and_finishes_at_home(planner, task, labels):
    positions = {label: [x, -0.15, TABLE_TOP + 0.02] for label, x in zip(labels, (-0.12, 0.0, 0.12), strict=True)}
    ctx = context(task, positions)
    planner(ctx).run()
    assert len(ctx.skills.placements) == 2
    for position in ctx.motion.positions.values():
        np.testing.assert_allclose(position[:2], [0.0, -0.15])
    first, second = ctx.skills.placements
    assert second[2][2] > first[2][2]  # second target uses the newly placed support
    assert ctx.motion.events[-1] == ("home", [0, 1], {})
    if task == "stack_bowls":
        assert all(move[3]["contact_lag"] == 0.004 for move in ctx.skills.placements)


def test_stacking_fails_without_executing_infeasible_grasp():
    ctx = context("stack_blocks", {f"block_{i}": [0.1 * i, -0.15, 0.8] for i in range(3)}, feasible=False)
    with pytest.raises(PlanningError, match="no base/arm assignment"):
        StackBlocksPlanner(ctx).run()
    assert ctx.skills.placements == []


def test_language_stack_preserves_requested_order_despite_cheaper_other_base():
    # block_1 is central and would be the shortest unordered stacking base.
    positions = {f"block_{i}": [x, -0.15, TABLE_TOP + 0.02] for i, x in enumerate((-0.12, 0.0, 0.12))}
    ctx = context("stack_blocks_by_language", positions)
    OrderedStackPlanner(ctx).run()
    assert [move[0] for move in ctx.skills.placements] == ["block_1", "block_2"]
    heights = [ctx.motion.pose(f"block_{i}")[0][2] for i in range(3)]
    assert heights[0] < heights[1] < heights[2]
    assert ctx.motion.events[-1] == ("home", [0, 1], {})


def test_language_stack_relocates_only_the_base_before_ordered_stacking():
    positions = {f"block_{i}": [x, -0.15, TABLE_TOP + 0.02] for i, x in enumerate((0.30, -0.35, -0.12))}
    ctx = context("stack_blocks_by_language", positions)
    OrderedStackPlanner(ctx).run()
    assert [move[0] for move in ctx.skills.placements] == ["block_0", "block_1", "block_2"]
    assert abs(ctx.motion.pose("block_0")[0][0]) <= 0.05
    heights = [ctx.motion.pose(f"block_{i}")[0][2] for i in range(3)]
    assert heights[0] < heights[1] < heights[2]
    assert all(move[3]["precise"] for move in ctx.skills.placements)


@pytest.mark.parametrize("single_arm,expected_moves", [(False, 2), (True, 3)])
def test_swap_preserves_original_goals_across_dual_arm_and_buffer(single_arm, expected_moves):
    positions = {"t0": [-0.1, -0.15, 0.78], "t1": [0.1, -0.15, 0.78]}
    ctx = context("swap_T", positions, single_arm=single_arm)
    SwapTPlanner(ctx).run()
    assert len(ctx.skills.placements) == expected_moves
    np.testing.assert_allclose(ctx.motion.positions["t0"], positions["t1"])
    np.testing.assert_allclose(ctx.motion.positions["t1"], positions["t0"])
    assert ctx.motion.events[-1] == ("home", [0, 1], {})
    assert all(move[3]["precise"] for move in ctx.skills.placements)
    if single_arm:
        assert ctx.skills.placements[0][0] == ctx.skills.placements[-1][0]
    else:
        assert any(event[0] == "hold" for event in ctx.motion.events)


def test_swap_fails_cleanly_when_neither_strategy_has_a_grasp():
    ctx = context("swap_T", {"t0": [-0.1, -0.15, 0.78], "t1": [0.1, -0.15, 0.78]}, feasible=False)
    with pytest.raises(PlanningError, match="even via a table buffer"):
        SwapTPlanner(ctx).run()
    assert ctx.skills.placements == []


def test_dolls_use_official_model_size_order_and_nonoverlapping_slots():
    positions = {f"doll{i}": [-0.25 + 0.1 * i, -0.32, 0.8] for i in range(5)}
    ctx = context("sort_nesting_dolls_by_size", positions)
    SortNestingDollsPlanner(ctx).run()
    ordered = sorted(ctx.motion.positions, key=lambda label: ctx.motion.positions[label][0])
    assert ordered == ["doll4", "doll3", "doll2", "doll1", "doll0"]
    xy = np.array([ctx.motion.positions[label][:2] for label in ordered])
    np.testing.assert_allclose(xy[:, 1], xy[0, 1])
    assert np.all(np.diff(xy[:, 0]) >= 0.04 + 0.018 - 1e-9)
    assert ctx.motion.events[-1] == ("home", [0, 1], {})


def test_rearrangement_exhausts_strategies_without_looping_forever():
    ctx = context(
        "sort_nesting_dolls_by_size",
        {f"doll{i}": [-0.25 + 0.1 * i, -0.32, 0.8] for i in range(5)},
        feasible=False,
    )
    with pytest.raises(PlanningError, match="rearrangement stuck"):
        SortNestingDollsPlanner(ctx).run()
