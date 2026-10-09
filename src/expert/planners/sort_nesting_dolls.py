"""Sort nesting dolls by the official size ordering (experimental planner)."""

import numpy as np

from src.expert.contracts import TaskPlanner
from src.expert.planners.common.objects import ObjectProfile
from src.expert.planners.common.rearrangement import Rearranger


class DollProfile(ObjectProfile):
    noun = "nesting doll"
    lift = 0.06

    def grasp(self, label):
        position, _ = self.motion.pose(label)
        top = self.motion.top_z(label)
        height = top - self.motion.bottom_z(label)
        tcp = np.r_[position[:2], top - min(0.03, 0.3 * height)]
        return tcp, [np.deg2rad(angle) for angle in range(-180, 180, 45)]


class SortNestingDollsPlanner(TaskPlanner):
    labels = tuple(f"doll{index}" for index in range(5))
    size_words = ("largest", "second largest", "middle", "second smallest", "smallest")

    def run(self):
        sizes = {label: self.motion.metadata(label)["model_id"] % 5 for label in self.labels}
        # Official order: model_id % 5 descending, smallest first from the left.
        order = sorted(self.labels, key=sizes.__getitem__, reverse=True)
        rearranger = Rearranger(self.context, DollProfile(self.motion), self.labels)
        goals = self.choose_row(rearranger, order)
        rearranger.run(
            goals, lambda label: f"place the {self.size_words[sizes[label]]} nesting doll in its slot in the row"
        )

    def choose_row(self, rearranger, order):
        """Choose a row requiring few occupied goal slots to be cleared."""
        widths = [2 * rearranger.radius[label] for label in order]
        gap = 0.018
        total = sum(widths) + gap * (len(order) - 1)
        best_score, best_goals = None, None
        for y in (-0.15, -0.12, -0.18, -0.10, -0.20):
            for x0 in (-total / 2, -total / 2 - 0.03, -total / 2 + 0.03):
                goals, x = {}, x0
                for label, width in zip(order, widths, strict=True):
                    goals[label] = np.array([x + width / 2, y])
                    x += width + gap
                conflicts = sum(
                    1
                    for label in self.labels
                    for goal in goals.values()
                    if np.linalg.norm(rearranger.xy(label) - goal) < rearranger.radius[label] + 0.05
                    and np.linalg.norm(rearranger.xy(label) - goals[label]) > 0.015
                )
                done = sum(np.linalg.norm(rearranger.xy(label) - goals[label]) < 0.015 for label in self.labels)
                score = conflicts - 2 * done
                if best_score is None or score < best_score:
                    best_score, best_goals = score, goals
        return best_goals
