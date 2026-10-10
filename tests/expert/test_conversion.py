"""Raw deletion follows validation and a recoverable conversion receipt."""

import json
from pathlib import Path
import sys
import time

import pytest

from src.expert import collect, conversion, training
from src.expert.scheduling import batch_lock
from src.expert.storage import converted_metadata, episode_count, episode_files, success_seeds
from tests.expert.conversion_helpers import fake_conversion, raw_episode


@pytest.fixture
def converter(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw_episode(raw, seed=5)
    converted = tmp_path / "converted"
    events = []

    def export(files, target, repo_id, *, training_only):
        assert files == [path] and path.exists() and training_only
        events.append("export")
        (target / "meta").mkdir(parents=True)
        (target / "meta/info.json").write_text(json.dumps(dict(total_episodes=1, total_frames=4)))

    def validate(target, *, source, expected_episodes):
        assert source == path and path.exists() and expected_episodes == 1
        assert (target / "meta/info.json").is_file()
        events.append("validate")
        return dict(episodes=1, frames=4)

    monkeypatch.setattr(conversion, "export", export)
    monkeypatch.setattr(conversion, "validate_training", validate)
    return raw, converted, path, events


def test_delete_only_after_validation_and_receipt(converter, monkeypatch):
    raw, converted, path, events = converter
    unlink = Path.unlink

    def checked_unlink(file, *args, **kwargs):
        if file == path:
            assert events == ["export", "validate"]
            assert converted_metadata(raw, 5)["validation"] == dict(episodes=1, frames=4)
            assert training.converted_datasets(raw) == [converted / path.stem]
            events.append("delete")
        return unlink(file, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", checked_unlink)
    conversion.convert_episode(raw, converted, path)
    assert events == ["export", "validate", "delete"] and not path.exists()
    assert episode_count(raw) == 1 and success_seeds(raw) == {5}


@pytest.mark.parametrize("stage", ["export", "validate_training"])
def test_conversion_error_retains_hdf5_without_completion_receipt(converter, monkeypatch, stage):
    raw, converted, path, _ = converter

    def fail(*args, **kwargs):
        raise ValueError("failed conversion")

    monkeypatch.setattr(conversion, stage, fail)
    with pytest.raises(ValueError, match="failed conversion"):
        conversion.convert_episode(raw, converted, path)
    assert path.exists() and not (raw / "converted").exists()


def test_crash_after_receipt_before_delete_revalidates_and_reuses_shard(converter, monkeypatch):
    raw, converted, path, events = converter
    unlink = Path.unlink

    def interrupted(file, *args, **kwargs):
        if file == path:
            raise OSError("interrupted deletion")
        return unlink(file, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError, match="interrupted deletion"):
        conversion.convert_episode(raw, converted, path)
    assert path.exists() and converted_metadata(raw, 5)["success"]
    assert episode_count(raw) == 1  # HDF5 plus receipt must not count twice
    monkeypatch.setattr(Path, "unlink", unlink)
    conversion.convert_episode(raw, converted, path)
    assert events == ["export", "validate", "validate"] and not path.exists()


def test_backpressure_waits_only_for_finalized_hdf5(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw_episode(raw, 0)
    raw_episode(raw, 1)
    (raw / "episode_0000002.partial.h5").touch()
    args = collect.parse(
        ["--task", "stack_blocks", "--out", str(raw), "--background-export", "--max-pending-episodes", "2"]
    )
    waits = []

    def convert_while_waiting(seconds):
        waits.append(seconds)
        fake_conversion(raw, first)

    monkeypatch.setattr(collect.time, "sleep", convert_while_waiting)
    collect.wait_for_conversion(args)
    assert len(waits) == 1 and len(episode_files(raw)) == 1 and episode_count(raw) == 2
    collect.wait_for_conversion(args)
    assert len(waits) == 1  # completed receipt and .partial do not block a new attempt


@pytest.mark.parametrize("report_error", [False, True])
@pytest.mark.parametrize("stage", ["collecting", "draining", "resume"])
def test_converter_exit_stops_pipeline_and_preserves_pending_file(tmp_path, monkeypatch, report_error, stage):
    raw = tmp_path / "raw"
    raw.mkdir()
    fake_converter = r"""
import json, sys, time
from pathlib import Path
work, report_error = Path(sys.argv[1]), int(sys.argv[2])
(work / 'conversion.ready').touch()
deadline = time.monotonic() + 10
while not (work / 'raw/episode_0000000.h5').exists():
    if time.monotonic() > deadline: raise TimeoutError('collector failed to start')
    time.sleep(.01)
if report_error:
    (work / 'conversion.error').write_text(json.dumps(dict(error='forced conversion error')))
    time.sleep(60)  # emulate another conversion thread stuck during shutdown
raise SystemExit(7)
"""
    fake_collector = (
        "import sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_bytes(b'pending raw data'); time.sleep(int(sys.argv[2]))"
    )
    processes = []
    start = training.start_process

    def start_with_fake_converter(command, lock_fd, **kwargs):
        if "src.expert.conversion" in command:
            command = [sys.executable, "-c", fake_converter, str(tmp_path), str(int(report_error))]
        process = start(command, lock_fd, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(training, "start_process", start_with_fake_converter)
    path = raw / "episode_0000000.h5"
    if stage == "resume":
        path.write_bytes(b"pending raw data")
        child = None
    else:
        child = [sys.executable, "-c", fake_collector, str(path), "60" if stage == "collecting" else "0"]
    started = time.monotonic()
    with batch_lock(tmp_path) as lock_fd:
        code = training.collect_and_convert(child, raw, tmp_path, 1, lock_fd)
    assert code == (1 if report_error else 7) and time.monotonic() - started < 15
    assert path.read_bytes() == b"pending raw data"
    if stage == "collecting":
        assert processes[1].returncode == -15
    elif stage == "draining":
        assert processes[1].returncode == 0
    assert all(process.poll() is not None for process in processes)
    assert (tmp_path / "conversion.log").is_file()


def test_interrupted_uncommitted_export_is_rebuilt(converter):
    raw, converted, path, events = converter
    target = converted / path.stem
    partial = target.with_name(target.name + ".partial")
    partial.mkdir(parents=True)
    (partial / "broken.mp4").write_bytes(b"incomplete")
    conversion.convert_episode(raw, converted, path)
    assert events == ["export", "validate"] and not partial.exists() and not path.exists()
