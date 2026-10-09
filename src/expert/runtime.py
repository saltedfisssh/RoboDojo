"""Episode lifecycle, independent of task-specific planning and CLI parsing."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import traceback

from src.expert.contracts import EpisodeEnd, PlannerFactory, PlanningError, TaskContext
from src.expert.planner import Planner
from src.expert.skills import Skills


@contextmanager
def simulation_session(task: str, *, images: bool, exit_code=None, device_id=0):
    # Isaac imports must follow launch_app, including on error paths.
    from src.expert.sim import close_app, launch_app, make_env

    app = launch_app(cameras=images, device_id=device_id)
    code = 0
    try:
        env = make_env(task, app, cameras=images, device_id=device_id)
        try:
            yield env
        finally:
            env.close()
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        if isinstance(exc, SystemExit) and isinstance(exc.code, int):
            code = exc.code
        traceback.print_exc()  # close_app can terminate before Python unwinds
        raise
    finally:
        close_app(app, code or (exit_code() if exit_code is not None else 0))


@dataclass(frozen=True)
class EpisodeResult:
    result: str
    reason: str | None = None
    steps: int = 0
    primitive_steps: dict[str, int] = field(default_factory=dict)

    @property
    def success(self):
        return self.result == "success"


def failing_predicates(env):
    """Best-effort diagnostics; never substitute these for official termination."""
    failures = []
    try:
        from src.datagen.state import terminal_checks

        manager = env.reward_manager
        for check in terminal_checks(manager):
            if not manager.check_once(check, 0):
                name, args = check if isinstance(check, tuple) else (str(check), {})
                brief = {
                    key: value
                    for key, value in (args or {}).items()
                    if key in ("label", "label_A", "label_B", "label_list", "threshold")
                }
                failures.append(f"{name}{brief}")
    except Exception as exc:
        failures.append(f"<unavailable: {exc}>")
    return "; ".join(failures) or "none failing now (sequential stage pending)"


def run_episode(env, factory: PlannerFactory, *, task: str, seed: int, layout: dict, recorder=None):
    """Reset, execute and annotate one episode. Saving belongs to collection.

    Planner return values cannot declare success. Only official end_flag AND
    success can admit a trajectory, and PlanningError always rejects it.
    """
    if not env.reset_layout(seed, layout):
        return EpisodeResult("unstable")
    motion = Planner(env, recorder)
    context = TaskContext(task, seed, layout, env, motion, Skills(motion))
    reason = None
    try:
        factory(context).run()
        # Allow the official reward's settling stages to finish.
        for _ in range(40):
            motion.step(motion.command.copy())
        reason = "program finished but task not successful: " + failing_predicates(env)
    except EpisodeEnd:
        if not (env.end_flag[0] and env.success[0]):
            reason = f"step limit {env.step_lim} reached"
    except PlanningError as exc:
        reason = f"planning: {exc}"
    success = bool(env.success[0] and env.end_flag[0] and reason is None)
    durations = {}
    for segment in motion.segments:
        if segment.background:
            continue
        end = segment.end_step if segment.end_step is not None else int(env.take_action_cnt[0])
        durations[segment.primitive] = durations.get(segment.primitive, 0) + end - segment.start_step
    if recorder is not None:
        recorder.annotate(motion)
    return EpisodeResult("success" if success else "fail", reason, int(env.take_action_cnt[0]), durations)
