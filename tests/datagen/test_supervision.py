import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from src.datagen import batch


def test_native_graph_stall_releases_only_its_worker_group(tmp_path, monkeypatch):
    log = tmp_path / "worker.log"
    log.write_text("[datagen] Acquired the Isaac GPU lease.\n")
    clock, signals = [0.0], []
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout):
            if signals:
                return -15
            clock[0] += 60
            with log.open("a") as stream:
                stream.write('Timeout (0:01:00)!\n  File "/env/torch/cuda/graphs.py", line 88 in replay\n')
            raise subprocess.TimeoutExpired("collector", timeout)

    code, reason = batch.supervise(Process(), log, 14400)
    assert code == 124 and "five stack dumps" in reason
    assert clock[0] == 300
    assert signals == [(12345, batch.signal.SIGTERM)]


def test_graph_stack_monitor_handles_split_logs_and_resets_on_progress():
    monitor = batch.PlannerStallMonitor()
    for stamp in range(0, 1000, 60):
        assert not monitor.feed(b'  File "/env/torch/cuda/gra', stamp)
        assert not monitor.feed(b'phs.py", line 88 in replay\n', stamp)
        assert not monitor.feed(b"[timing] frames=100: physics=10s\n", stamp)
    assert monitor.replays == 0
    # A single read of old stack dumps must not terminate a newly attached monitor.
    assert not monitor.feed(b'File "/env/torch/cuda/graphs.py", line 88 in replay\n' * 8, 1100)
    assert not monitor.feed(b"[planner] 3: Close the gripper.\n", 1400)


def test_gpu_queue_does_not_spend_execution_timeout(tmp_path, monkeypatch):
    log = tmp_path / "worker.log"
    log.write_text("[datagen] Waiting for the Isaac GPU lease.\n")
    clock = [0.0]
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])

    class Process:
        def __init__(self):
            self.waits = 0

        def poll(self):
            return None

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                clock[0] = 1000
            elif self.waits == 2:
                log.write_text(log.read_text() + "[datagen] Acquired the Isaac GPU lease.\n")
            else:
                return 0
            raise subprocess.TimeoutExpired("collector", timeout)

    assert batch.supervise(Process(), log, 10) == (0, None)


def test_active_execution_still_times_out(tmp_path, monkeypatch):
    log = tmp_path / "worker.log"
    log.write_text("[datagen] Acquired the Isaac GPU lease.\n")
    clock = [0.0]
    signals = []
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout):
            if signals:
                return -15
            clock[0] += 11
            raise subprocess.TimeoutExpired("collector", timeout)

    assert batch.supervise(Process(), log, 10) == (124, "Attempt active execution timeout")
    assert signals == [(12345, batch.signal.SIGTERM)]


def test_batch_stops_when_another_shard_completes_remaining_budget(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    output = root / "worker"
    launches = []

    def accepted(path, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as file:
            file.attrs.update(complete=True, success=True, num_frames=1, metadata=json.dumps({"seed": seed}))
            file.create_group("terminal")
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                file.create_dataset(f"images/{camera}", data=np.zeros((1, 2, 2, 3), np.uint8))

    def spawn(command, **kwargs):
        launches.append(command)
        accepted(Path(command[command.index("--output") + 1]), 10)
        # Another producer finishes a distinct episode during this collection.
        accepted(root / "other/stack_blocks/episode_0000000020.h5", 20)
        return object()

    monkeypatch.setattr(batch.subprocess, "Popen", spawn)
    monkeypatch.setattr(batch, "supervise", lambda *args: (0, None))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "--tasks",
            "stack_blocks",
            "--episodes",
            "2",
            "--max-attempts",
            "4",
            "--seed-start",
            "10",
            "--output",
            str(output),
            "--reference-root",
            str(root),
        ],
    )
    assert batch.main() == 0
    assert len(launches) == 1


def test_follow_dispatch_revisits_fixed_task_and_preserves_prior_attempts(tmp_path, monkeypatch):
    from src.datagen import dispatch

    debug = tmp_path / "debug"
    debug.mkdir()
    old = debug / "stack_blocks_v1.h5"
    old.write_bytes(b"failed simulation")
    current = [old]
    launches = []
    sleeps = []
    accepted = {}
    monkeypatch.setattr(dispatch, "latest_smoke", lambda *args: current[0])
    monkeypatch.setattr(dispatch, "smoke_passed", lambda path: path != old)
    monkeypatch.setattr(dispatch, "accepted_by_seed", lambda paths: dict(accepted))

    def wait(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 4:
            raise AssertionError("Dispatcher did not finish")
        # One unchanged failed revision must not trigger a GPU batch.
        if len(sleeps) == 2:
            current[0] = debug / "stack_blocks_v2.h5"
            current[0].write_bytes(b"successful simulation")
        elif len(sleeps) == 3:
            current[0] = debug / "stack_blocks_v3.h5"
            current[0].write_bytes(b"corrected simulation")

    def collect(command, **kwargs):
        launches.append(command)
        if len(launches) == 2:
            accepted.update({i: Path(str(i)) for i in range(3)})

    monkeypatch.setattr(dispatch.time, "sleep", wait)
    monkeypatch.setattr(dispatch.subprocess, "run", collect)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dispatch", "--root", str(tmp_path), "--tasks", "stack_blocks", "--follow"],
    )
    dispatch.main()
    assert len(launches) == 2
    outputs = [Path(cmd[cmd.index("--output") + 1]).name for cmd in launches]
    assert outputs == ["stack_blocks_v2", "stack_blocks_v3"]
    report = json.loads((tmp_path / "dispatch_all.json").read_text())
    assert report["stack_blocks"]["ready"]


def test_failed_attempt_shutdown_is_bounded_after_artifacts_are_finalized(tmp_path, monkeypatch):
    log = tmp_path / "worker.log"
    log.write_text(
        "[datagen] Acquired the Isaac GPU lease.\n[datagen] Attempt failed; artifacts finalized; closing Isaac.\n"
    )
    clock = [0.0]
    signals = []
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout):
            if signals:
                return -15
            clock[0] += 20
            raise subprocess.TimeoutExpired("collector", timeout)

    assert batch.supervise(Process(), log, 14400) == (124, "Failed attempt stalled during Isaac shutdown")
    assert signals == [(12345, batch.signal.SIGTERM)]
    assert clock[0] == 60


def test_physx_cuda_failure_releases_lease_without_waiting_full_job_timeout(tmp_path, monkeypatch):
    log = tmp_path / "worker.log"
    log.write_text(
        "[datagen] Acquired the Isaac GPU lease.\n"
        "PhysX Internal CUDA error. Simulation cannot continue! Error code 700!\n"
    )
    clock = [0.0]
    signals = []
    monkeypatch.setattr(batch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(batch.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout):
            if signals:
                return -15
            clock[0] += 20
            raise subprocess.TimeoutExpired("collector", timeout)

    assert batch.supervise(Process(), log, 14400) == (124, "Isaac CUDA failure did not exit within 60 seconds")
    assert signals == [(12345, batch.signal.SIGTERM)]
    assert clock[0] == 60
