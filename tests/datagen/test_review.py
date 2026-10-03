import errno
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from src.datagen.batch import accepted_by_seed
from src.datagen.recording import EpisodeWriter
from src.datagen.review import build_review, write_text_atomically


def test_report_write_failure_preserves_last_valid_report(tmp_path, monkeypatch):
    report = tmp_path / "status.json"
    report.write_text('{"ready_tasks": 17}\n')
    write = Path.write_text

    def interrupted_write(path, text, *args, **kwargs):
        write(path, text[:5], *args, **kwargs)
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", interrupted_write)
        with pytest.raises(OSError, match="No space left"):
            write_text_atomically(report, '{"ready_tasks": 18}\n')
    assert json.loads(report.read_text()) == {"ready_tasks": 17}
    assert list(tmp_path.iterdir()) == [report]
    write_text_atomically(report, '{"ready_tasks": 18}\n')
    assert json.loads(report.read_text()) == {"ready_tasks": 18}


def write_episode(root, seed, *, success=True, rgb=True, terminal=True):
    target = root / "raw" / "worker0" / "stack_blocks" / f"episode_{seed:010d}.h5"
    writer = EpisodeWriter(
        target,
        {
            "task": "stack_blocks",
            "seed": seed,
            "fps": 25,
            "benchmark_action_budget": 10,
        },
    )
    snapshot = ({"command": np.zeros(14)}, {})
    writer.append(
        np.zeros(14),
        np.zeros(14),
        np.zeros(14),
        {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
        if rgb
        else {},
        snapshot,
        {"segment_id": 0, "text": "Test", "primitive": "hold"},
    )
    writer.finish(success, terminal_snapshot=snapshot if terminal else None)
    return target


def test_review_requires_success_rgb_terminal_and_full_episode(tmp_path):
    write_episode(tmp_path, 1)
    write_episode(tmp_path, 2, success=False)
    write_episode(tmp_path, 3, rgb=False)
    write_episode(tmp_path, 4, terminal=False)
    partial = write_episode(tmp_path, 5)
    partial.rename(partial.with_suffix(".partial.h5"))
    report = build_review(tmp_path, required=1)
    stack = next(t for t in report["tasks"] if t["task"] == "stack_blocks")
    assert stack["successful_rgb_episodes"] == 1
    assert report["ready_tasks"] == 1
    assert report["exported_tasks"] == 0
    assert len(report["tasks"]) == 35
    assert not next(t for t in report["tasks"] if t["task"] == "display_letters")["ready"]
    assert json.loads((tmp_path / "status.json").read_text())["ready_tasks"] == 1


def test_review_preserves_preview_links_in_status_only_refresh(tmp_path):
    write_episode(tmp_path, 1)
    preview = tmp_path / "previews/stack_blocks/episode_0000000001.mp4"
    preview.parent.mkdir(parents=True)
    preview.touch()
    report = build_review(tmp_path, required=1)
    stack = next(t for t in report["tasks"] if t["task"] == "stack_blocks")
    assert stack["attempts"][0]["preview"] == str(preview)
    with h5py.File(stack["attempts"][0]["path"]) as f:
        assert f.attrs["success"]


def test_review_does_not_count_re_rendered_seed_as_an_independent_episode(tmp_path):
    original = write_episode(tmp_path, 1)
    duplicate = tmp_path / "raw/replayed/stack_blocks/episode_0000000001.h5"
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(original, duplicate)
    report = build_review(tmp_path, required=2)
    stack = next(task for task in report["tasks"] if task["task"] == "stack_blocks")
    assert stack["successful_rgb_episodes"] == 1
    assert not stack["ready"]
    assert sum("duplicate_seed_of" in episode for episode in stack["attempts"]) == 1
    assert len(accepted_by_seed((tmp_path / "raw").glob("**/stack_blocks/episode_*.h5"))) == 1


def test_promote_only_selects_successful_unrepresented_physical_seeds(tmp_path):
    from src.datagen.promote import candidates

    write_episode(tmp_path, 1)
    debug = tmp_path / "debug"
    debug.mkdir()
    for seed, success in ((1, True), (2, True), (3, False), (4, True)):
        source_root = tmp_path / f"source_{seed}"
        source = write_episode(source_root, seed, success=success, rgb=False)
        shutil.copyfile(source, debug / f"stack_blocks_{seed}.h5")
    shutil.copyfile(debug / "stack_blocks_2.h5", debug / "stack_blocks_2_again.h5")
    selected = list(candidates(tmp_path, 3))
    assert sorted(seed for _, _, seed in selected) == [2, 4]
    assert list(candidates(tmp_path, 1)) == []


def test_quarantined_source_invalidates_historical_reader_success(tmp_path):
    source = write_episode(tmp_path, 1)
    destination = tmp_path / "lerobot/stack_blocks/meta"
    destination.mkdir(parents=True)
    (destination / "readback.json").write_text(json.dumps({"passed": True, "episodes": 1}))
    (destination / "replay.jsonl").write_text(json.dumps({"source_episode": str(source.resolve())}) + "\n")
    report = build_review(tmp_path, required=1)
    assert report["exported_tasks"] == 1
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    source.rename(quarantine / source.name)
    report = build_review(tmp_path, required=1)
    stack = next(task for task in report["tasks"] if task["task"] == "stack_blocks")
    assert report["exported_tasks"] == 0
    assert not stack["exported"]
    assert "no longer" in stack["export_review_error"]
    assert (destination / "readback.json").exists()  # Preserve the historical result.


def test_verified_archive_symlink_preserves_export_membership(tmp_path):
    source = write_episode(tmp_path, 1)
    destination = tmp_path / "lerobot/stack_blocks/meta"
    destination.mkdir(parents=True)
    (destination / "readback.json").write_text(json.dumps({"passed": True, "episodes": 1}))
    (destination / "replay.jsonl").write_text(json.dumps({"source_episode": str(source)}) + "\n")
    archive = tmp_path / "local_archive" / source.name
    archive.parent.mkdir()
    source.rename(archive)
    source.symlink_to(archive)
    assert build_review(tmp_path, required=1)["exported_tasks"] == 1
    # Keeping a broken alias must not preserve acceptance after quarantine.
    archive.rename(archive.with_suffix(".quarantine"))
    from src.datagen.review import export_review_error

    assert "no longer" in export_review_error(destination.parent, [], 1)
