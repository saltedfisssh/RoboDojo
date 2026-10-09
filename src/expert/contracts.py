"""Task planner extension contract. Safe to import without Isaac or cuRobo."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.expert.planner import Planner
    from src.expert.skills import Skills


@dataclass(frozen=True)
class TaskContext:
    """Episode-local services; planners must not retain them between episodes.

    Use ``motion`` and ``skills`` for actions so recording, annotations and
    official termination stay on the shared control path. ``env`` is available
    for task-specific observations and interactions. Layout is the sampled
    input; query motion/env for live state after objects have moved.
    """

    task: str
    seed: int
    layout: dict
    env: Any
    motion: Planner
    skills: Skills


class TaskPlanner(ABC):
    """One instance per episode; success is decided by the official environment.

    ``run`` returns normally when the program finishes, raises PlanningError
    when no feasible continuation exists, and lets EpisodeEnd propagate.
    Constructors must not move the robot; put episode work in ``run``.
    """

    def __init__(self, context: TaskContext):
        self.context = context
        self.motion = context.motion
        self.skills = context.skills

    @abstractmethod
    def run(self) -> None:
        """Execute this task using the episode's live scene state."""


class PlannerFactory(Protocol):
    def __call__(self, context: TaskContext) -> TaskPlanner: ...


class PlanningError(RuntimeError):
    """No feasible continuation; discard the episode."""


class EpisodeEnd(Exception):
    """The official episode-end logic fired (success or step limit)."""
