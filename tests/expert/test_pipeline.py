"""Collection contracts that must hold without an Isaac/GPU runtime."""

from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.expert import collect, runtime
from src.expert.batch import run_workers, worker_commands
from src.expert.contracts import EpisodeEnd, PlanningError
from src.expert.layout import LayoutError
from src.expert.registry import planner_spec
from src.expert.storage import attempted_seeds, prepare_output, read_attempts


class Motion:
    def __init__(self, env, recorder):
        self.env = env
        self.command = np.zeros(14)
        self.segments = []

    def step(self, action):
        self.env.take_action_cnt[0] += 1
        if self.env.take_action_cnt[0] >= self.env.finish_after:
            self.env.end_flag[0] = True
            raise EpisodeEnd


@pytest.fixture
def environment(monkeypatch):
    env = SimpleNamespace(
        success=[True],
        end_flag=[False],
        step_lim=40,
        take_action_cnt=[0],
        finish_after=1,
        reset_layout=lambda seed, layout: True,
    )
    monkeypatch.setattr(runtime, "Planner", Motion)
    monkeypatch.setattr(runtime, "Skills", lambda motion: object())
    monkeypatch.setattr(runtime, "failing_predicates", lambda env: "unfinished")
    return env


def test_only_official_success_can_admit_episode(environment):
    env = environment

    def factory(context):
        return SimpleNamespace(run=lambda: True)

    env.finish_after = 100
    result = runtime.run_episode(env, factory, task="stack_blocks", seed=1, layout={})
    assert result.result == "fail"  # returning True never declares success
    assert result.steps == 40
    env.take_action_cnt[0] = 0
    env.finish_after = 1
    env.success[0] = False
    result = runtime.run_episode(env, factory, task="stack_blocks", seed=2, layout={})
    assert not result.success and "step limit" in result.reason
    env.success[0] = True
    result = runtime.run_episode(env, factory, task="stack_blocks", seed=3, layout={})
    assert result.success


def test_failed_reset_does_not_construct_task_planner(environment):
    environment.reset_layout = lambda seed, layout: False

    def forbidden(context):
        pytest.fail("Unstable scene must not execute a planner")

    result = runtime.run_episode(environment, forbidden, task="stack_blocks", seed=1, layout={})
    assert result.result == "unstable"


def test_planning_failure_and_fresh_episode_context(environment):
    contexts = []

    def factory(context):
        contexts.append(context)

        def run():
            raise PlanningError("no grasp")

        return SimpleNamespace(run=run)

    for seed in (1, 2):
        result = runtime.run_episode(environment, factory, task="stack_blocks", seed=seed, layout={"seed": seed})
        assert result.result == "fail" and result.reason == "planning: no grasp"
    assert contexts[0].motion is not contexts[1].motion
    assert [context.seed for context in contexts] == [1, 2]
    assert contexts[1].layout == {"seed": 2}


@pytest.mark.parametrize("fail_build", [False, True])
def test_simulation_cleanup_on_failure(monkeypatch, fail_build):
    from src.expert import sim

    closed = []
    monkeypatch.setattr(sim, "launch_app", lambda **kwargs: SimpleNamespace(close=lambda: closed.append("app")))
    codes = []

    def close(app, code):
        codes.append(code)
        app.close()

    monkeypatch.setattr(sim, "close_app", close)

    def build(*args, **kwargs):
        if fail_build:
            raise RuntimeError("build failed")
        return SimpleNamespace(close=lambda: closed.append("env"))

    monkeypatch.setattr(sim, "make_env", build)
    with pytest.raises(RuntimeError):
        with runtime.simulation_session("stack_blocks", images=False):
            raise RuntimeError("episode failed")
    assert closed == (["app"] if fail_build else ["env", "app"])
    assert codes == [1]


def test_collection_logs_layout_errors_saves_only_success_and_resumes(tmp_path, monkeypatch):
    from src.expert import layout, recorder

    sessions = []

    @contextmanager
    def session(*args, **kwargs):
        sessions.append(True)
        yield object()

    def sample(seed):
        if seed == 0:
            raise LayoutError("crowded")
        return {"seed": seed}

    class Recorder:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path):
            path.write_bytes(b"synthetic success")
            return path

    monkeypatch.setattr(runtime, "simulation_session", session)
    monkeypatch.setattr(layout, "LayoutSampler", lambda *args, **kwargs: SimpleNamespace(sample=sample))
    monkeypatch.setattr(recorder, "Recorder", Recorder)
    monkeypatch.setattr(
        runtime,
        "run_episode",
        lambda *args, **kwargs: runtime.EpisodeResult("fail" if kwargs["seed"] == 1 else "success"),
    )
    args = collect.parse(["--task", "stack_blocks", "--num", "1", "--out", str(tmp_path), "--seed-start", "0"])
    spec = planner_spec(args.task)
    summary = collect.collect(args, spec, object())
    assert summary["target_reached"] and summary["collected"] == 1
    assert [row["result"] for row in read_attempts(tmp_path)] == ["layout_error", "fail", "success"]
    assert [path.name for path in tmp_path.glob("*.h5")] == ["episode_0000002.h5"]
    assert attempted_seeds(tmp_path) == {0, 1, 2}
    # Already complete: no second Isaac session.
    assert collect.collect(args, spec, object())["target_reached"]
    assert len(sessions) == 1
    args.num = 2
    assert collect.collect(args, spec, object())["target_reached"]
    assert [row["seed"] for row in read_attempts(tmp_path)] == [0, 1, 2, 3]


@pytest.mark.parametrize("code", [0, 1])
def test_session_passes_result_to_kit_before_close(monkeypatch, code):
    from src.expert import sim

    received = []
    app = object()
    monkeypatch.setattr(sim, "launch_app", lambda **kwargs: app)
    monkeypatch.setattr(sim, "make_env", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sim, "close_app", lambda actual_app, result: received.append((actual_app, result)))
    with runtime.simulation_session("stack_blocks", images=False, exit_code=lambda: code):
        pass
    assert received == [(app, code)]


@pytest.mark.parametrize(
    "have,target,workers,gpu_ids",
    [
        (0, 10, 3, [1, 3, 5]),
        (8, 10, 4, [1, 3, 5]),
        (10, 10, 3, [1, 3, 5]),
        (17, 20, 2, [1, 3, 5]),
        (0, 32, 8, list(range(8))),
        (0, 32, 16, list(range(8))),
        (0, 32, 32, list(range(8))),
        (19, 100, 32, list(range(8))),
    ],
)
def test_batch_quotas_are_exact_and_seed_streams_disjoint(tmp_path, have, target, workers, gpu_ids):
    args = SimpleNamespace(
        task="stack_blocks",
        num=target,
        workers=workers,
        out=tmp_path,
        seed_start=100,
        attempts_per_episode=4,
        table_randomization=0.3,
        planner="example:Planner",
        no_images=True,
        gpu=0,
        gpu_ids=gpu_ids,
        scheduling="static",
    )
    commands = worker_commands(args, have)
    quotas, seeds = [], set()
    for worker, command in commands:
        parsed = collect.parse(command[4:])
        assert parsed.additional and parsed.planner == "example:Planner" and parsed.no_images
        assert parsed.gpu == gpu_ids[worker % len(gpu_ids)]
        quotas.append(parsed.num)
        stream = {parsed.seed_start + index * parsed.seed_stride for index in range(30)}
        assert not stream & seeds
        seeds |= stream
    assert sum(quotas) == max(0, target - have)


def test_batch_cleans_started_workers_if_spawn_fails(tmp_path, monkeypatch):
    from src.expert import batch

    (tmp_path / "logs").mkdir()
    stopped = []
    process = SimpleNamespace(
        poll=lambda: 0 if stopped else None,
        terminate=lambda: stopped.append(True),
        wait=lambda **kwargs: 0,
    )

    def spawn(*args, **kwargs):
        if spawn.called:
            raise OSError("spawn failed")
        spawn.called = True
        return process

    spawn.called = False
    monkeypatch.setattr(batch.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="spawn failed"):
        run_workers(SimpleNamespace(out=tmp_path, startup_delay=0), [(0, ["worker"]), (1, ["worker"])])
    assert stopped


def test_output_rejects_mixed_task_or_planner(tmp_path):
    settings = dict(task="stack_blocks", planner="example:Planner", images=True, table_randomization=0.0)
    prepare_output(tmp_path, **settings)
    prepare_output(tmp_path, **settings)
    with pytest.raises(ValueError, match="settings differ"):
        prepare_output(tmp_path, **{**settings, "task": "stack_bowls"})
    assert json.loads((tmp_path / "collection.json").read_text()) == settings


def test_restore_closes_file_and_session_on_failure(tmp_path, monkeypatch):
    import h5py

    from src.expert import restore

    path = tmp_path / "episode.h5"
    with h5py.File(path, "w") as episode:
        episode.attrs["meta"] = json.dumps(dict(task="stack_blocks", num_frames=2, fps=25))
        episode.attrs["segments"] = "[]"
        episode.attrs["subtasks"] = "[]"
    closed, files = [], []

    @contextmanager
    def session(*args, **kwargs):
        try:
            yield object()
        finally:
            closed.append(True)

    def fail(args, env, path, episode, *metadata):
        files.append(episode)
        raise RuntimeError("restore failed")

    monkeypatch.setattr(runtime, "simulation_session", session)
    monkeypatch.setattr(restore, "restore_episode", fail)
    with pytest.raises(RuntimeError, match="restore failed"):
        restore.main(["--episode", str(path), "--frame", "0", "--no-images"])
    assert closed == [True] and not files[0].id.valid
    # Invalid selection must fail before launching another session.
    with pytest.raises(SystemExit, match="outside"):
        restore.main(["--episode", str(path), "--frame", "9"])
    assert len(closed) == 1
