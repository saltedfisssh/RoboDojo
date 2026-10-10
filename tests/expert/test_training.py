"""Overlap collection with validated conversion, preserving resumable successes."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.expert import collect, training
from src.expert.batch import worker_commands
from src.expert.registry import planner_spec
from src.expert.scheduling import batch_lock
from src.expert.storage import (
    attempted_seeds,
    episode_count,
    episode_files,
    log_attempt,
    prepare_output,
)
from tests.expert.conversion_helpers import fake_conversion, raw_episode


def request(tmp_path, num=2):
    argv = [
        "--task",
        "stack_blocks",
        "--num",
        str(num),
        "--seed-start",
        "0",
        "--format",
        "lerobot-v3",
        "--out",
        str(tmp_path / "dataset"),
    ]
    args = collect.parse(argv)
    return args, argv, planner_spec(args.task)


def intercept_stages(monkeypatch, *, fault=None):
    calls = []

    def collect_and_convert(child, raw, work, workers, lock_fd):
        calls.append(child)
        if child is not None:
            args = collect.parse(child[4:])
            assert args.format == "raw" and args.no_checkpoints and args.background_export
            assert args.max_pending_episodes >= 2 and workers == 1
            seed = 0
            used = attempted_seeds(args.out)
            count = args.num - episode_count(args.out)
            for _ in range(count if fault != "collection" else 1):
                while seed in used:
                    seed += 1
                raw_episode(args.out, seed)
                log_attempt(args.out, 0, dict(seed=seed, task="stack_blocks", result="success", steps=4))
                used.add(seed)
        if fault == "conversion":
            return 7
        for path in episode_files(raw):
            fake_conversion(raw, path)
        return 1 if fault == "collection" else 0

    def run(command, lock_fd):
        calls.append(command)
        if "--merge" in command:
            if fault == "merge":
                return 9
            out = Path(command[command.index("--output") + 1])
            roots = command[command.index("--merge") + 1 : command.index("--output")]
            count = sum(json.loads((Path(root) / "meta/info.json").read_text())["total_episodes"] for root in roots)
            (out / "meta").mkdir(parents=True)
            (out / "meta/info.json").write_text(json.dumps(dict(total_episodes=count)))
        if "--validate" in command and fault == "validation":
            return 8
        return 0

    monkeypatch.setattr(training, "run_stage", run)
    monkeypatch.setattr(training, "collect_and_convert", collect_and_convert)
    return calls


@pytest.mark.parametrize("fault", ["collection", "conversion", "merge", "validation"])
def test_failed_pipeline_keeps_successes_for_resume(tmp_path, monkeypatch, fault):
    args, argv, spec = request(tmp_path)
    intercept_stages(monkeypatch, fault=fault)
    assert training.collect_training("collect", args, argv, spec) != 0
    assert not args.out.exists()
    work = tmp_path / ".dataset.collection"
    assert episode_count(work / "raw") > 0
    assert bool(episode_files(work / "raw")) == (fault == "conversion")
    if fault != "conversion":
        assert training.converted_datasets(work / "raw")  # successes survive without HDF5
    calls = intercept_stages(monkeypatch)
    assert training.collect_training("collect", args, argv, spec) == 0
    if fault != "collection":
        assert calls[0] is None  # all successes already saved, no new Isaac work
    assert (args.out / "meta/collection.json").exists()
    assert not (work / "raw").exists()
    assert not list(args.out.rglob("*.h5"))


def test_completed_pipeline_skips_all_processes_and_rejects_changed_settings(tmp_path, monkeypatch):
    args, argv, spec = request(tmp_path)
    intercept_stages(monkeypatch)
    assert training.collect_training("collect", args, argv, spec) == 0
    monkeypatch.setattr(training, "run_stage", lambda *args: pytest.fail("completed output must not start a process"))
    monkeypatch.setattr(training, "collect_and_convert", lambda *args: pytest.fail("completed output must not convert"))
    assert training.collect_training("collect", args, argv, spec) == 0
    args.table_randomization = 1.0
    with pytest.raises(ValueError, match="settings differ"):
        training.collect_training("collect", args, argv, spec)


@pytest.mark.parametrize("option", ["--no-images", "--diagnostics", "--save-failures"])
def test_training_incompatible_options_fail_before_processes(tmp_path, monkeypatch, option):
    _, argv, spec = request(tmp_path)
    args = collect.parse([*argv, option])
    monkeypatch.setattr(training, "run_stage", lambda *args: pytest.fail("invalid request must not start a process"))
    with pytest.raises(ValueError):
        training.collect_training("collect", args, argv, spec)
    assert not args.out.exists()


def test_checkpoint_setting_is_validated_on_resume(tmp_path):
    settings = dict(task="stack_blocks", planner="test:Planner", images=True, table_randomization=0.0)
    prepare_output(tmp_path, **settings, checkpoints=False)
    prepare_output(tmp_path, **settings, checkpoints=False)
    with pytest.raises(ValueError, match="settings differ"):
        prepare_output(tmp_path, **settings, checkpoints=True)


@pytest.mark.parametrize("converted", [False, True])
def test_saved_episode_without_attempt_log_preserves_steps_for_batch_resume(tmp_path, converted):
    from src.expert.batch import summarize

    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw_episode(raw, seed=0)
    if converted:
        fake_conversion(raw, path)
    log_attempt(raw, 0, dict(seed=1, task="stack_blocks", result="fail", steps=3))
    (raw / "scheduling_test.jsonl").write_text(
        "".join(json.dumps(dict(event="assigned", seed=seed)) + "\n" for seed in (0, 2))
    )
    attempts = training.collection_attempts(raw, "stack_blocks")
    assert [(row["seed"], row["result"]) for row in attempts] == [(0, "success"), (1, "fail"), (2, "interrupted")]
    assert attempts[0]["steps"] == 4
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    (resumed / "attempts_wprior.jsonl").write_text("".join(json.dumps(row) + "\n" for row in attempts))
    assert summarize(resumed)["median_steps"] == 4
    assert attempted_seeds(resumed) == {0, 1, 2}


def test_interrupted_publication_restores_old_dataset_and_reuses_pending_success(tmp_path, monkeypatch):
    args, argv, spec = request(tmp_path)
    intercept_stages(monkeypatch)
    assert training.collect_training("collect", args, argv, spec) == 0
    args, argv, spec = request(tmp_path, num=3)
    original = Path.rename

    def interrupted(path, destination):
        if path.name == "merged":
            raise OSError("interrupted publication")
        return original(path, destination)

    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError, match="interrupted publication"):
        training.collect_training("collect", args, argv, spec)
    assert not args.out.exists()
    assert (tmp_path / ".dataset.collection/previous/meta/info.json").exists()
    assert not episode_files(tmp_path / ".dataset.collection/raw")
    monkeypatch.setattr(Path, "rename", original)
    calls = intercept_stages(monkeypatch)
    assert training.collect_training("collect", args, argv, spec) == 0
    assert calls[0] is None
    assert json.loads((args.out / "meta/collection.json").read_text())["seeds"] == [0, 1, 2]


def test_pipeline_lock_remains_with_workers_after_batch_parent_exits(tmp_path):
    parent = r"""
import os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from src.expert import batch
from src.expert.scheduling import batch_lock
work, marker = Path(sys.argv[1]), sys.argv[2]
raw = work / 'raw'
(raw / 'logs').mkdir(parents=True)
child = ("import time,sys; from pathlib import Path; end=time.monotonic()+10\n"
         "while not Path(sys.argv[1]).exists() and time.monotonic()<end: time.sleep(.01)")
spawn = subprocess.Popen
def spawn_then_exit(*args, **kwargs):
    spawn(*args, **kwargs)
    os._exit(0)
with batch_lock(work) as pipeline_fd, batch_lock(raw) as raw_fd:
    args = SimpleNamespace(out=raw, startup_delay=0, batch_lock_fd=raw_fd, pipeline_lock_fd=pipeline_fd)
    batch.subprocess.Popen = spawn_then_exit
    batch.run_workers(args, [(0, [sys.executable, '-c', child, marker])])
"""
    work = tmp_path / "work"
    marker = tmp_path / "worker_can_exit"
    subprocess.run([sys.executable, "-c", parent, str(work), str(marker)], check=True)
    try:
        with pytest.raises(ValueError, match="Another batch"):
            with batch_lock(work):
                pytest.fail("old workers must still hold the pipeline lock")
    finally:
        marker.touch()
    deadline = time.monotonic() + 5
    while True:
        try:
            with batch_lock(work):
                break
        except ValueError:
            assert time.monotonic() < deadline
            time.sleep(0.01)


def test_checkpoint_flag_reaches_every_batch_worker(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        out=tmp_path,
        num=3,
        workers=3,
        scheduling="dynamic",
        task="stack_blocks",
        gpu_ids=[0, 1, 2],
        gpu=0,
        seed_start=0,
        attempts_per_episode=4,
        table_randomization=0.0,
        planner=None,
        no_images=False,
        no_checkpoints=True,
        background_export=True,
        max_pending_episodes=4,
    )
    commands = worker_commands(args, 0)
    assert len(commands) == 3
    assert all(collect.parse(command[4:]).no_checkpoints for _, command in commands)
    assert all(collect.parse(command[4:]).max_pending_episodes == 4 for _, command in commands)
    assert all(collect.parse(command[4:]).background_export for _, command in commands)


# The collector starts planning its next trajectory immediately after saving.
# It waits for conversion/deletion before saving the next one, so an exporter
# that waits for the whole collection to finish cannot pass this test.
FAKE_COLLECTOR = r"""
import json, sys, time
from pathlib import Path
import h5py
import numpy as np
from src.expert.storage import attempted_seeds, episode_count, log_attempt
raw, goal = Path(sys.argv[1]), int(sys.argv[2])
used, seed, first = attempted_seeds(raw), 0, True
while episode_count(raw) < goal:
    while seed in used:
        seed += 1
    path = raw / f'episode_{seed:07d}.h5'
    tmp = path.with_suffix('.partial.h5')
    rng = np.random.default_rng(seed)
    with h5py.File(tmp, 'w') as f:
        f.attrs['meta'] = json.dumps(dict(task='stack_blocks', seed=seed, success=True,
            num_frames=4, steps=4, step_lim=550, instruction='stack the blocks'))
        for key in ('state', 'action'):
            f.create_dataset(key, data=rng.normal(size=(4,14)).astype(np.float32))
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            f.create_dataset(f'images/{camera}', data=rng.integers(0,256,(4,48,64,3),np.uint8))
    tmp.rename(path)
    log_attempt(raw, 0, dict(seed=seed, task='stack_blocks', result='success', steps=4))
    used.add(seed)
    if first:
        (raw.parent / 'next_planning').touch()
        deadline = time.monotonic() + 60
        while path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('conversion must finish while collection is still running')
            time.sleep(.01)
        receipt = raw / 'converted' / f'episode_{seed:07d}.json'
        assert receipt.is_file()
        (raw.parent / 'overlap.json').write_text(json.dumps(dict(
            deleted_while_collecting=True, planning_started_before_deletion=True, seed=seed)))
        first = False
"""


def test_real_training_export_merge_and_cleanup(tmp_path, monkeypatch):
    if not (training.ROOT / ".venv-lerobot/bin/python").exists():
        pytest.skip("requires the local LeRobot export environment")
    args, argv, spec = request(tmp_path, num=3)
    args.export_workers = 2
    calls = []
    start = training.start_process

    def start_with_fake_isaac(command, lock_fd, **kwargs):
        calls.append(command)
        if command[1:4] == ["-m", "src.expert", "collect"]:
            child = collect.parse(command[4:])
            assert child.no_checkpoints and child.background_export
            command = [sys.executable, "-c", FAKE_COLLECTOR, str(child.out), str(child.num)]
        return start(command, lock_fd, **kwargs)

    monkeypatch.setattr(training, "start_process", start_with_fake_isaac)
    assert training.collect_training("collect", args, argv, spec) == 0
    overlap = json.loads((tmp_path / ".dataset.collection/overlap.json").read_text())
    assert overlap["deleted_while_collecting"] and overlap["planning_started_before_deletion"]
    assert not list(args.out.rglob("*.h5"))
    assert not (args.out / "states").exists()
    assert not (args.out / "meta/argus").exists()
    info = json.loads((args.out / "meta/info.json").read_text())
    assert info["total_episodes"] == 3 and info["total_frames"] == 12
    assert "subtask_index" not in info["features"] and "language_persistent" not in info["features"]
    args, argv, spec = request(tmp_path, num=4)
    assert training.collect_training("collect", args, argv, spec) == 0
    info = json.loads((args.out / "meta/info.json").read_text())
    receipt = json.loads((args.out / "meta/collection.json").read_text())
    assert info["total_episodes"] == 4 and info["total_frames"] == 16
    assert receipt["seeds"] == [0, 1, 2, 3]
    assert not (tmp_path / ".dataset.collection/raw").exists()
    before = len(calls)
    assert training.collect_training("collect", args, argv, spec) == 0
    assert len(calls) == before
