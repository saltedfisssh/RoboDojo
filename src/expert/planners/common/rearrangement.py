"""Rearrange rigid objects using direct moves, relays and obstacle clearing."""

import numpy as np

from src.expert.contracts import PlanningError
from src.expert.planner import TABLE_TOP
from src.expert.planners.common.objects import arm_order


class Rearranger:
    def __init__(self, context, profile, labels, *, max_moves=12):
        self.motion, self.skills = context.motion, context.skills
        self.profile, self.labels = profile, tuple(labels)
        self.max_moves, self.moves, self.last_arm = max_moves, 0, None
        self.radius = {
            label: float(np.max(np.ptp(self.motion.world_vertices(label)[:, :2], axis=0))) / 2 for label in labels
        }

    def xy(self, label):
        return self.motion.pose(label)[0][:2]

    def free(self, label, xy, *, ignore=(), margin=0.012):
        return all(
            np.linalg.norm(self.xy(other) - xy) >= self.radius[label] + self.radius.get(other, 0.04) + margin
            for other in self.motion.labels("Rigid")
            if other not in (label, *ignore)
        )

    def target(self, label, xy):
        return np.r_[xy, TABLE_TOP + self.profile.rest_height[label] + 0.003]

    def grasp_to(self, label, xy, arms):
        return self.skills.pick_place_grasp(
            label,
            *self.profile.grasp(label),
            arms,
            self.target(label, xy),
            lift=self.profile.lift,
        )

    def move(self, label, xy, grasp, text):
        self.moves += 1
        if self.moves > self.max_moves:
            raise PlanningError("too many rearrangement moves")
        self.motion.begin_subtask(text, object=label, target=[float(value) for value in xy])
        if self.motion.queue.get(grasp.arm):
            self.motion.drain()
        self.skills.pick(
            label,
            grasp,
            lift=self.profile.lift,
            text=self.profile.noun,
            replan=lambda: self.grasp_to(label, xy, (grasp.arm,)),
        )
        region = "left" if xy[0] < -0.12 else "right" if xy[0] > 0.12 else "center"
        self.skills.place(
            label,
            grasp.arm,
            self.target(label, xy),
            text=self.profile.noun,
            dest=f"the {region} of the table",
            **self.profile.place_options,
        )
        self.last_arm = grasp.arm

    def spots(self, label, near):
        candidates = [np.array([x, y]) for x in np.arange(-0.36, 0.37, 0.06) for y in np.arange(-0.24, 0.01, 0.06)]
        candidates.sort(key=lambda xy: np.linalg.norm(xy - near))
        return [xy for xy in candidates if self.free(label, xy)]

    def _direct_move(self, pending, goals, describe):
        for label in sorted(pending, key=lambda item: np.linalg.norm(self.xy(item) - goals[item])):
            if not self.free(label, goals[label]):
                continue
            arms = arm_order(self.motion, self.xy(label))
            if self.last_arm is not None and self.motion.queue.get(1 - self.last_arm):
                arms = (self.last_arm,)
            grasp = self.grasp_to(label, goals[label], arms)
            if grasp is None:
                continue
            if self.last_arm is not None and grasp.arm != self.last_arm:
                self.skills.home([self.last_arm], background=True)
            self.move(label, goals[label], grasp, describe(label))
            return True
        return False

    def _relay(self, pending, goals):
        for label in pending:
            if not self.free(label, goals[label]):
                continue
            source_arm = arm_order(self.motion, self.xy(label))[0]
            for spot in self.spots(label, (self.xy(label) + goals[label]) / 2):
                grasp = self.grasp_to(label, spot, (source_arm,))
                if grasp is not None:
                    self.move(label, spot, grasp, f"move the {self.profile.noun} toward the other arm")
                    return True
        return False

    def _clear_blocker(self, pending, goals):
        for label in pending:
            blockers = [
                other
                for other in self.labels
                if other != label
                and np.linalg.norm(self.xy(other) - goals[label]) < self.radius[label] + self.radius[other] + 0.012
            ]
            for blocker in blockers:
                for spot in self.spots(blocker, self.xy(blocker)):
                    if any(
                        np.linalg.norm(spot - goals[other]) < self.radius[blocker] + self.radius[other] + 0.012
                        for other in goals
                        if other != blocker
                    ):
                        continue
                    grasp = self.grasp_to(blocker, spot, arm_order(self.motion, self.xy(blocker)))
                    if grasp is not None:
                        self.move(blocker, spot, grasp, f"move the {self.profile.noun} out of the way")
                        return True
        return False

    def run(self, goals, describe):
        """Move each label to its goal xy; re-evaluate the live scene after every move."""
        while pending := [label for label in goals if np.linalg.norm(self.xy(label) - goals[label]) > 0.015]:
            if self._direct_move(pending, goals, describe):
                continue
            if self._relay(pending, goals):
                continue
            if self._clear_blocker(pending, goals):
                continue
            raise PlanningError("rearrangement stuck: no feasible move")
        self.motion.drain()
        self.skills.home([0, 1])
