"""Compatibility for old TASKS[name](motion, skills) callers.

New planners implement TaskPlanner in planners/ and register in registry.py.
"""

from collections.abc import Mapping

from src.expert.contracts import TaskContext
from src.expert.registry import REGISTRY, load_planner


class _LegacyTasks(Mapping):
    def __iter__(self):
        return iter(REGISTRY)

    def __len__(self):
        return len(REGISTRY)

    def __getitem__(self, task):
        spec = REGISTRY[task]

        def run(motion, skills):
            context = TaskContext(task, 0, {}, motion.env, motion, skills)
            load_planner(spec)(context).run()
            return True

        return run


TASKS = _LegacyTasks()
