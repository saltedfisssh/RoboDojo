"""Shared stack planning: choose a base, gather if needed, execute live grasps."""

from dataclasses import dataclass

import numpy as np

from src.expert.contracts import PlanningError, TaskPlanner
from src.expert.planner import TABLE_TOP
from src.expert.planners.common.objects import StackProfile, arm_order


@dataclass(frozen=True)
class StackMove:
    label: str
    arm: int


@dataclass(frozen=True)
class StackPlan:
    base: str
    moves: tuple[StackMove, ...]
    relocation: np.ndarray | None = None


class StackingPlanner(TaskPlanner):
    """Subclasses supply labels and an object profile; sequencing is shared."""

    labels: tuple[str, ...]
    profile_type: type[StackProfile]
    center_spots = ((0.0, -0.15), (0.0, -0.20), (-0.05, -0.15), (0.05, -0.15))
    gather_spots = ((0.14, -0.15), (0.14, -0.21), (0.10, -0.10), (0.20, -0.12), (0.08, -0.20))

    def run(self):
        self.profile = self.profile_type(self.motion)
        plan = self.select_plan()
        if plan is None and self.gather():
            self.motion.drain()
            plan = self.select_plan()
        if plan is None:
            raise PlanningError(f"no base/arm assignment reaches every {self.profile.noun}")
        self.execute(plan)

    def _feasible_moves(self, base_xy, base_top, movers):
        motion, skills, profile = self.motion, self.skills, self.profile
        moves, top = [], base_top
        for level, label in enumerate(movers, start=1):
            if np.linalg.norm(motion.pose(label)[0][:2] - base_xy) > profile.max_carry:
                return None
            grasp = skills.pick_place_grasp(
                label,
                *profile.grasp(label),
                arm_order(motion, motion.pose(label)[0]),
                profile.stack_target(label, base_xy, top, level),
                lift=profile.lift,
            )
            if grasp is None:
                return None
            moves.append(StackMove(label, grasp.arm))
            top += profile.added_height(label)
        return tuple(moves)

    def select_plan(self):
        """Try bases in carry-distance order, charging 0.25 m per relocation."""
        motion, profile = self.motion, self.profile
        candidates = []
        for base in self.base_candidates():
            position = motion.pose(base)[0]
            movers = self.movers_for(base)
            for xy in self.base_locations(base):
                if xy is None:
                    distance = sum(np.linalg.norm(motion.pose(label)[0][:2] - position[:2]) for label in movers)
                    candidates.append((distance, base, None, movers))
                    continue
                center = np.asarray(xy)
                distance = np.linalg.norm(position[:2] - center) + 0.25
                distance += sum(np.linalg.norm(motion.pose(label)[0][:2] - center) for label in movers)
                candidates.append((distance, base, center, movers))
        for _, base, center, movers in sorted(candidates, key=lambda item: item[0]):
            position = motion.pose(base)[0]
            if center is None:
                moves = self._feasible_moves(position[:2], motion.top_z(base), movers)
                if moves is not None:
                    return StackPlan(base, moves)
                continue
            if np.linalg.norm(position[:2] - center) > profile.max_carry:
                continue
            relocation = np.r_[center, position[2]]
            grasp = self.skills.pick_place_grasp(
                base,
                *profile.grasp(base),
                arm_order(motion, position),
                relocation,
                lift=profile.lift,
            )
            if grasp is None:
                continue
            moves = self._feasible_moves(center, motion.top_z(base), movers)
            if moves is not None:
                return StackPlan(base, moves, relocation)
        return None

    def base_candidates(self):
        """Tasks with an ordered stack can constrain the bottom object."""
        return self.labels

    def base_locations(self, base):
        """None keeps the base in place; other entries relocate it by grasping."""
        return (None, *self.center_spots)

    def movers_for(self, base):
        """Default unordered stacking minimizes horizontal carry distance."""
        return sorted(
            (label for label in self.labels if label != base),
            key=lambda label: abs(self.motion.pose(label)[0][0] - self.motion.pose(base)[0][0]),
        )

    def gather(self):
        """Bring outliers into free central spots before retrying stack search."""
        motion, skills, profile = self.motion, self.skills, self.profile
        moved = False
        radii = {label: float(np.max(np.ptp(motion.world_vertices(label)[:, :2], axis=0))) / 2 for label in self.labels}
        for label in sorted(self.labels, key=lambda item: -abs(motion.pose(item)[0][0])):
            position = motion.pose(label)[0]
            if abs(position[0]) < 0.22:
                continue
            side = np.sign(position[0])
            arm = 0 if side < 0 else 1
            for x, y in self.gather_spots:
                xy = np.array([side * x, y])
                if any(
                    np.linalg.norm(motion.pose(other)[0][:2] - xy) < radii[label] + radii[other] + 0.02
                    for other in self.labels
                    if other != label
                ):
                    continue
                target = np.r_[xy, TABLE_TOP + profile.rest_height[label] + 0.003]
                grasp = skills.pick_place_grasp(label, *profile.grasp(label), (arm,), target, lift=profile.lift)
                if grasp is None:
                    continue
                motion.begin_subtask(
                    f"move the {profile.noun} {label} closer to the centre",
                    object=label,
                    target=xy.tolist(),
                )
                motion.drain()
                self._pick(label, grasp, target, (arm,))
                skills.place(
                    label, arm, target, text=profile.noun, dest="the middle of the table", **profile.place_options
                )
                skills.home([arm], background=True)
                moved = True
                break
        return moved

    def _pick(self, label, grasp, target, arms):
        profile = self.profile
        self.skills.pick(
            label,
            grasp,
            lift=profile.lift,
            text=profile.noun,
            replan=lambda: self.skills.pick_place_grasp(
                label,
                *profile.grasp(label),
                arms,
                target,
                lift=profile.lift,
            ),
        )

    def execute(self, plan):
        motion, skills, profile = self.motion, self.skills, self.profile
        moves = list(plan.moves)
        if plan.relocation is not None:
            moves.insert(0, StackMove(plan.base, arm_order(motion, motion.pose(plan.base)[0])[0]))
        top_label, level = plan.base, 0
        for index, move in enumerate(moves):
            label, arm = move.label, move.arm
            relocating = index == 0 and plan.relocation is not None
            if relocating:
                motion.begin_subtask(
                    f"move the {profile.noun} {plan.base} to the centre of the table",
                    object=plan.base,
                    target="table centre",
                )
                target, destination = plan.relocation, "the centre of the table"
            else:
                level += 1
                motion.begin_subtask(f"stack {label} on {top_label}", object=label, target=top_label)
                target = profile.stack_target(label, motion.pose(plan.base)[0][:2], motion.top_z(top_label), level)
                destination = f"the {top_label}"
            arms = (arm, 1 - arm) if not motion.queue.get(1 - arm) else (arm,)
            grasp = skills.pick_place_grasp(label, *profile.grasp(label), arms, target, lift=profile.lift)
            if grasp is None:
                raise PlanningError(f"no grasp of {label} reaches {destination}")
            self._pick(label, grasp, target, (grasp.arm,))
            skills.place(label, grasp.arm, target, text=profile.noun, dest=destination, **profile.place_options)
            next_arm = moves[index + 1].arm if index + 1 < len(moves) else None
            if next_arm is not None and next_arm != grasp.arm:
                skills.home([grasp.arm], background=True)
            if not relocating:
                top_label = label
        motion.drain()
        skills.home([0, 1])
