"""Development workflow contracts without Isaac or GPU execution."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.expert import collect, dev, development, registry
from src.expert.diagnostics import write_json
from src.expert.layout_audit import audit_layout


def test_task_plugins_are_discovered_without_importing_code(tmp_path, monkeypatch):
    directory = tmp_path / "classify_objects"
    directory.mkdir()
    write_json(
        directory / "planner.json",
        {
            "schema": 1,
            "task": "classify_objects",
            "entrypoint": "src.expert.task_plugins.classify_objects.planner:Planner",
            "status": "experimental",
            "sampler": "src.expert.task_plugins.classify_objects.layout:Sampler",
        },
    )
    monkeypatch.setattr(registry, "PLUGIN_ROOT", tmp_path)
    spec = registry.planner_spec("classify_objects")
    assert spec.sampler.endswith("layout:Sampler")
    assert spec.status == "experimental"
    # The module doesn't exist, but inventory and manifest discovery still work.
    assert any(row["task"] == "classify_objects" and row["planner"] for row in registry.task_inventory())


def test_task_plugin_cannot_silently_replace_builtin(tmp_path, monkeypatch):
    directory = tmp_path / "stack_blocks"
    directory.mkdir()
    write_json(directory / "planner.json", {"schema": 1, "task": "stack_blocks", "status": "experimental"})
    monkeypatch.setattr(registry, "PLUGIN_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Duplicate planner"):
        registry.registered_planners()


@pytest.mark.parametrize(
    "task,expected",
    [
        ("stack_blocks_by_language", None),
        ("stack_blocks_random", "Clutter"),
        ("pour_balls_into_vase", "object-relative"),
        ("arrange_largest_number", "linked_count"),
        ("match_and_pick_from_conveyor", "scene_config"),
        ("cover_blocks", "place_tag"),
    ],
)
def test_layout_preflight_reports_real_unsupported_semantics(task, expected):
    report = audit_layout(task)
    if expected:
        assert expected in " ".join(report["issues"])
        assert not report["default_sampler_compatible"]
    else:
        assert report["default_sampler_compatible"]


def test_handoff_creation_is_task_local_and_never_overwrites(tmp_path, monkeypatch):
    (tmp_path / "src/expert/task_plugins").mkdir(parents=True)
    monkeypatch.setattr(development, "ROOT", tmp_path)
    monkeypatch.setattr(development, "official_fingerprint", lambda task: {"official": "hash"})
    monkeypatch.setattr(registry, "PLUGIN_ROOT", tmp_path / "empty_registry")
    folder = development.create_plugin("classify_objects")
    assert folder == tmp_path / "src/expert/task_plugins/classify_objects"
    data = json.loads((folder / "planner.json").read_text())
    assert data["status"] == "experimental"
    assert "PlanningError" in (folder / "planner.py").read_text()
    protocol = development.load_protocol("classify_objects")
    assert len(protocol["seeds"]["validation"]) == 10
    before = (folder / "planner.py").read_text()
    with pytest.raises(FileExistsError):
        development.create_plugin("classify_objects")
    assert (folder / "planner.py").read_text() == before


def test_fixed_seeds_attempt_failures_and_do_not_stop_at_success_quota(tmp_path, monkeypatch):
    from src.expert import runtime

    @contextmanager
    def session(*args, **kwargs):
        yield SimpleNamespace()

    seen = []

    def attempt(args, spec, factory, env, sampler, seed):
        seen.append(seed)
        return {"seed": seed, "result": "success" if seed == 7 else "unstable"}

    monkeypatch.setattr(runtime, "simulation_session", session)
    monkeypatch.setattr(collect, "load_sampler", lambda spec: lambda *args, **kwargs: object())
    monkeypatch.setattr(collect, "collect_attempt", attempt)
    args = collect.parse(["--task", "stack_blocks", "--num", "1", "--seeds", "7", "9", "11", "--out", str(tmp_path)])
    collect.collect(args, registry.planner_spec("stack_blocks"), object())
    assert seen == [7, 9, 11]


def make_run(tmp_path, monkeypatch, rows, seeds=(0, 1, 2)):
    monkeypatch.setattr(dev, "source_fingerprint", lambda task: {"sha256": "frozen"})
    write_json(
        tmp_path / "run.json",
        {
            "task": "stack_blocks",
            "split": "development",
            "seeds": list(seeds),
            "source": {"sha256": "frozen"},
            "minimum_success_rate": 0.8,
        },
    )
    write_json(tmp_path / "execution.json", {"exit_code": 1, "timed_out": False})
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/attempts_w0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_report_counts_layout_rejections_and_missing_seeds(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, [{"seed": 0, "result": "success"}, {"seed": 1, "result": "unstable"}])
    report = dev.report_run(run)
    assert report["success_rate_all_seeds"] == 1 / 3
    assert report["missing_seeds"] == [2]
    assert not report["meets_threshold"]


def test_report_rejects_duplicate_attempts_and_source_changes(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, [{"seed": 0, "result": "success"}] * 2, seeds=(0,))
    assert not dev.report_run(run)["run_complete"]
    assert dev.report_run(run)["success_rate_all_seeds"] == 1.0
    monkeypatch.setattr(dev, "source_fingerprint", lambda task: {"sha256": "changed"})
    assert not dev.report_run(run)["source_unchanged"]


def test_supervisor_times_out_actual_subprocess_and_keeps_log(tmp_path):
    result = development.supervise(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"],
        tmp_path / "child.log",
        timeout=0.3,
    )
    assert result["timed_out"] and result["exit_code"] != 0
    assert "started" in (tmp_path / "child.log").read_text()


def test_gpu_lock_is_shared_across_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERT_DEV_LOCK_DIR", str(tmp_path))
    with development.gpu_lock(0, timeout=0.1):
        result = subprocess.run(
            [sys.executable, "-c", "from src.expert.development import gpu_lock; gpu_lock(0, 0.1).__enter__()"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0 and "slot busy" in result.stderr
    with development.gpu_lock(0, timeout=0.1):
        pass


def test_protocol_rejects_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "ROOT", tmp_path)
    write_json(
        tmp_path / "src/expert/task_plugins/stack_blocks/validation.json",
        {
            "schema": 1,
            "task": "stack_blocks",
            "seeds": {"development": [0], "validation": [0], "rgb": [2]},
            "minimum_success_rate": 0.8,
            "minimum_rgb_successes": 1,
        },
    )
    with pytest.raises(ValueError, match="overlapping"):
        development.load_protocol("stack_blocks")


def test_success_receipt_without_file_cannot_qualify(tmp_path, monkeypatch):
    run = make_run(tmp_path, monkeypatch, [{"seed": 0, "result": "success"}], seeds=(0,))
    result = dev.report_run(run)
    assert result["success_rate_all_seeds"] == 1.0
    assert result["missing_success_files"] == [0]
    assert not result["meets_threshold"]


def test_rgb_run_uses_its_own_protocol_threshold(tmp_path, monkeypatch):
    run = make_run(
        tmp_path,
        monkeypatch,
        [
            {"seed": 0, "result": "success"},
            {"seed": 1, "result": "fail"},
            {"seed": 2, "result": "fail"},
        ],
    )
    (run / "raw/episode_0000000.h5").touch()
    manifest = json.loads((run / "run.json").read_text())
    manifest.update(split="rgb", protocol={"minimum_rgb_successes": 1})
    write_json(run / "run.json", manifest)
    result = dev.report_run(run)
    assert result["run_complete"] and result["meets_threshold"]
    assert result["success_rate_all_seeds"] == 1 / 3


def test_sampler_change_cannot_mix_resumed_collection(tmp_path):
    from src.expert.storage import prepare_output

    settings = dict(task="stack_blocks", planner="example:Planner", images=False, table_randomization=0.0)
    prepare_output(tmp_path, **settings, sampler="example:Sampler")
    with pytest.raises(ValueError, match="settings differ"):
        prepare_output(tmp_path, **settings, sampler="example:OtherSampler")


def test_timeout_removes_descendant_that_ignores_termination(tmp_path):
    pid_file = tmp_path / "pid"
    descendant = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {descendant!r}]); time.sleep(60)"
    result = development.supervise([sys.executable, "-c", parent], tmp_path / "parent.log", timeout=1)
    assert result["timed_out"]
    pid = int(pid_file.read_text())
    # A killed orphan can briefly remain as a zombie until PID 1 reaps it.
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        import time

        for _ in range(50):
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            os.kill(pid, 9)
            pytest.fail("Supervisor left its descendant running")


def test_task_source_fingerprint_isolates_other_contributors(tmp_path, monkeypatch):
    monkeypatch.setattr(development, "ROOT", tmp_path)
    monkeypatch.setattr(development, "official_fingerprint", lambda task: {})
    for name in (
        "src/expert/task_plugins/first/planner.py",
        "src/expert/task_plugins/second/planner.py",
        "src/expert/object_names.json",
        "scripts/generate_data.sh",
        "scripts/run-lerobot.sh",
        "scripts/internal/uv_env.sh",
        "scripts/requirements-lerobot.txt",
        "scripts/internal/validate_expert_dataset.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    before = development.source_fingerprint("first")
    (tmp_path / "src/expert/task_plugins/second/planner.py").write_text("other contributor")
    assert development.source_fingerprint("first") == before
    requirements = tmp_path / "scripts/requirements-lerobot.txt"
    requirements.write_text("changed export environment")
    assert development.source_fingerprint("first")["sha256"] != before["sha256"]
    requirements.write_text("original")
    (tmp_path / "src/expert/task_plugins/first/planner.py").write_text("iteration")
    assert development.source_fingerprint("first")["sha256"] != before["sha256"]
    with pytest.raises(ValueError, match="Source changed"):
        development.archive_sources(tmp_path / "source.zip", before)
