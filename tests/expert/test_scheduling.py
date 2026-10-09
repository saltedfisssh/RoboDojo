"""Exercise the shared target across real process/socket boundaries."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import h5py
import pytest

from src.expert import batch
from src.expert.scheduling import WorkQueue, batch_lock, serve_queue
from src.expert.storage import log_attempt, read_attempts


def queue_args(out, target=5, workers=2, attempts=4):
    return SimpleNamespace(
        out=out,
        num=target,
        workers=workers,
        task="stack_blocks",
        planner=None,
        seed_start=0,
        attempts_per_episode=attempts,
        startup_delay=0,
        scheduling="dynamic",
        gpu_ids=list(range(workers)),
        gpu=0,
        no_images=True,
        table_randomization=0.0,
    )


def finish(queue, worker, seed, result="success"):
    row = dict(seed=seed, task="stack_blocks", result=result, seconds=0, steps=1)
    if result == "success":
        (queue.out / f"episode_{seed:07d}.h5").touch()
    log_attempt(queue.out, worker, row)
    queue.complete(worker, row)


# Worker 0 waits while worker 1 finishes three episodes. A fixed quota of two
# for worker 1 would deadlock this test; a shared queue must keep it productive.
FAKE_WORKER = r"""
import os
from pathlib import Path
import sys
import time
from src.expert.scheduling import WorkerClient
from src.expert.storage import log_attempt
worker, address, directory, mode = sys.argv[1:]
worker, directory = int(worker), Path(directory)
def wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.01)
if worker == 1:
    wait_for(directory / 'slow_started')
    if mode == 'late':
        time.sleep(30)  # emulate an Isaac process still initializing
with WorkerClient(address, worker) as jobs:
    previous, count = None, 0
    while (seed := jobs.acquire(previous)) is not None:
        if worker == 0 and count == 0:
            (directory / 'slow_started').touch()
            if mode == 'crash':
                os._exit(7)
            if mode == 'unbalanced':
                wait_for(directory / 'fast_completed_three')
        (directory / f'episode_{seed:07d}.h5').touch()
        previous = dict(seed=seed, task='stack_blocks', result='success', steps=1, seconds=0)
        log_attempt(directory, worker, previous)
        count += 1
        if worker == 1 and count == 3:
            (directory / 'fast_completed_three').touch()
"""


@pytest.mark.parametrize("mode", ["unbalanced", "crash"])
def test_fast_process_takes_more_work_and_survives_peer_exit(tmp_path, mode):
    args = queue_args(tmp_path)
    (tmp_path / "logs").mkdir()
    queue = WorkQueue(args, {0: 0, 1: 1}, "test:Planner")
    with serve_queue(queue) as address:
        commands = [
            (worker, [sys.executable, "-c", FAKE_WORKER, str(worker), address, str(tmp_path), mode])
            for worker in range(2)
        ]
        codes = batch.run_workers(args, commands, queue)
    assert codes == ([0, 0] if mode == "unbalanced" else [7, 0])
    assert queue.successes == 5 and not queue.inflight
    assert len(list(tmp_path.glob("episode_*.h5"))) == 5
    fast = [json.loads(line) for line in (tmp_path / "attempts_w1.jsonl").read_text().splitlines()]
    assert len(fast) >= 3
    rows = read_attempts(tmp_path)
    assert len({row["seed"] for row in rows}) == len(rows)
    if mode == "crash":
        assert queue.assigned == 6 and sum(row["result"] == "error" for row in rows) == 1


def test_inflight_success_slots_prevent_overshoot_and_failure_unblocks_waiter(tmp_path):
    queue = WorkQueue(queue_args(tmp_path, target=2), {0: 0, 1: 1, 2: 2}, "test:Planner")
    assert queue.acquire(0)["seed"] == 0
    assert queue.acquire(1)["seed"] == 1
    waiting = threading.Event()

    def request():
        waiting.set()
        return queue.acquire(2)

    with ThreadPoolExecutor() as pool:
        future = pool.submit(request)
        assert waiting.wait(timeout=1)
        assert not future.done()
        finish(queue, 0, 0, "fail")
        assert future.result(timeout=2)["seed"] == 2
        finish(queue, 1, 1)
        finish(queue, 2, 2)
    assert queue.successes == 2
    assert queue.acquire(0) == {"seed": None, "target_reached": True}


def test_completed_goal_does_not_wait_for_uninitialized_process(tmp_path):
    args = queue_args(tmp_path)
    (tmp_path / "logs").mkdir()
    queue = WorkQueue(args, {0: 0, 1: 1}, "test:Planner")
    with serve_queue(queue) as address:
        commands = [
            (worker, [sys.executable, "-c", FAKE_WORKER, str(worker), address, str(tmp_path), "late"])
            for worker in range(2)
        ]
        codes = batch.run_workers(args, commands, queue)
    assert codes == [0, -15]
    assert queue.successes == 5 and queue.stopped_unused == {1}
    assert queue.ever_assigned == {0}


def test_global_budget_counts_failures_without_deadlock(tmp_path):
    queue = WorkQueue(queue_args(tmp_path, target=2, attempts=1), {0: 0}, "test:Planner")
    for seed in range(2):
        assert queue.acquire(0)["seed"] == seed
        finish(queue, 0, seed, "fail")
    assert queue.acquire(0) == {"seed": None, "target_reached": False}
    assert queue.assigned == queue.completed == queue.budget == 2


def test_resume_skips_failed_saved_and_interrupted_seeds(tmp_path):
    args = queue_args(tmp_path, target=3)
    queue = WorkQueue(args, {0: 0}, "test:Planner")
    assert queue.acquire(0)["seed"] == 0
    finish(queue, 0, 0, "fail")
    assert queue.acquire(0)["seed"] == 1
    finish(queue, 0, 1)
    assert queue.acquire(0)["seed"] == 2  # interrupted before recording
    resumed = WorkQueue(args, {0: 0}, "test:Planner")
    assert resumed.successes == 1 and resumed.budget == 8
    assert resumed.acquire(0)["seed"] == 3


@pytest.mark.parametrize("attempt_logged", [False, True])
def test_crash_after_save_recovers_success_exactly_once(tmp_path, attempt_logged):
    queue = WorkQueue(queue_args(tmp_path, target=1), {0: 0}, "test:Planner")
    seed = queue.acquire(0)["seed"]
    path = tmp_path / f"episode_{seed:07d}.h5"
    with h5py.File(path, "w") as episode:
        episode.attrs["meta"] = json.dumps(dict(seed=seed, success=True, num_frames=3))
    if attempt_logged:
        log_attempt(tmp_path, 0, dict(seed=seed, task="stack_blocks", result="success", steps=3, seconds=0))
    queue.worker_exited(0, 7)
    queue.worker_exited(0, 7)
    queue.complete(0, read_attempts(tmp_path)[0])  # delayed acknowledgement after exit recovery
    assert queue.successes == queue.completed == 1
    assert len(read_attempts(tmp_path)) == 1
    assert queue.acquire(0)["seed"] is None


def test_exited_worker_cannot_receive_a_late_assignment(tmp_path):
    queue = WorkQueue(queue_args(tmp_path), {0: 0}, "test:Planner")
    queue.worker_exited(0, 1)
    assert queue.acquire(0)["seed"] is None
    assert queue.assigned == 0


def test_collect_uses_shared_target_in_one_session(tmp_path, monkeypatch):
    from src.expert import collect, layout, recorder, runtime
    from src.expert.registry import planner_spec

    sessions = []

    @contextmanager
    def session(*args, **kwargs):
        sessions.append(True)
        yield object()

    class Recorder:
        def __init__(self, *args, **kwargs):
            pass

        def save(self, path):
            path.touch()
            return path

    monkeypatch.setattr(runtime, "simulation_session", session)
    monkeypatch.setattr(layout, "LayoutSampler", lambda *args, **kwargs: SimpleNamespace(sample=lambda seed: {}))
    monkeypatch.setattr(recorder, "Recorder", Recorder)
    monkeypatch.setattr(
        runtime,
        "run_episode",
        lambda *args, **kwargs: runtime.EpisodeResult("fail" if kwargs["seed"] == 0 else "success"),
    )
    queue = WorkQueue(queue_args(tmp_path, target=2), {0: 0}, "test:Planner")
    with serve_queue(queue) as address:
        args = collect.parse(
            ["--task", "stack_blocks", "--out", str(tmp_path), "--num", "100", "--additional", "--scheduler", address]
        )
        result = collect.collect(args, planner_spec(args.task), object())
    assert result["target_reached"] and result["collected"] == 2 and result["fail"] == 1
    assert len(sessions) == 1 and queue.successes == 2


def test_batch_output_has_one_coordinator(tmp_path):
    with batch_lock(tmp_path):
        with pytest.raises(ValueError, match="Another batch"):
            with batch_lock(tmp_path):
                pytest.fail("second coordinator admitted")
    with batch_lock(tmp_path):
        pass


def test_output_lock_survives_coordinator_death(tmp_path):
    parent = r"""
import os, subprocess, sys
from pathlib import Path
from src.expert.scheduling import batch_lock
child = ("import time,sys; from pathlib import Path; end=time.monotonic()+10\n"
         "while not Path(sys.argv[1]).exists() and time.monotonic()<end: time.sleep(.01)")
with batch_lock(Path(sys.argv[1])) as fd:
    subprocess.Popen([sys.executable, '-c', child, sys.argv[2]], pass_fds=(fd,),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os._exit(0)
"""
    marker = tmp_path / "child_can_exit"
    subprocess.run([sys.executable, "-c", parent, str(tmp_path), str(marker)], check=True)
    try:
        with pytest.raises(ValueError, match="Another batch"):
            with batch_lock(tmp_path):
                pytest.fail("orphan writer must still own the lock")
    finally:
        marker.touch()
    deadline = time.monotonic() + 5
    while True:
        try:
            with batch_lock(tmp_path):
                break
        except ValueError:
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_already_complete_batch_does_not_start_workers(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("completed target must not start Isaac")

    monkeypatch.setattr(batch, "run_workers", lambda args, commands: [] if not commands else forbidden())
    batch.prepare_output(
        tmp_path,
        task="stack_blocks",
        planner="src.expert.planners.stack_blocks:StackBlocksPlanner",
        images=True,
        table_randomization=0.0,
    )
    (tmp_path / "episode_0000000.h5").touch()
    assert batch.main(["--task", "stack_blocks", "--num", "1", "--gpu-ids", "0", "1", "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["worker_gpus"] == {}


def test_dynamic_slots_default_to_one_per_gpu(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(batch, "run_batch", lambda args, spec: captured.append(args) or 0)
    assert batch.main(["--task", "stack_blocks", "--gpu-ids", "0", "2", "5", "--out", str(tmp_path)]) == 0
    assert captured[0].workers == 3 and captured[0].scheduling == "dynamic"
    args = captured[0]
    args.num = 100
    commands = batch.worker_commands(args, 98)
    assert len(commands) == 2  # do not launch excess Isaac processes near target
    assert [int(command[command.index("--gpu") + 1]) for _, command in commands] == [0, 2]
