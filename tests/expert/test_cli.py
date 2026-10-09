"""The public CLI and planner discovery must work before simulator bootstrap."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.expert.contracts import TaskPlanner
from src.expert.registry import BUILTINS, load_planner, official_tasks, planner_spec, registered_planners

ROOT = Path(__file__).resolve().parents[2]


def test_task_inventory_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "src.expert", "tasks", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(result.stdout)
    assert {row["task"] for row in rows} == set(official_tasks())
    assert sum(row["planner"] is not None for row in rows) == len(registered_planners())
    assert next(row for row in rows if row["task"] == "sort_nesting_dolls_by_size")["status"] == "experimental"
    for row in rows:
        if row["task"] not in registered_planners():
            assert row["status"] == "unimplemented"


@pytest.mark.parametrize(
    "command", [[], ["--help"], ["collect", "--help"], ["batch", "--help"], ["probe", "--help"], ["dev", "--help"]]
)
def test_help_does_not_need_simulator(command):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "src.expert", *command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--task", "not_a_task"], "Unknown official task"),
        (["--task", "stack_blocks", "--num", "0"], "positive integer"),
        (["--task", "stack_blocks", "--table-randomization", "1.1"], "between 0 and 1"),
    ],
)
def test_invalid_requests_fail_before_simulator(arguments, message):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "src.expert", "collect", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and message in result.stderr
    assert "isaac" not in result.stderr.lower()


def test_missing_planner_fails_before_simulator(tmp_path):
    # Contributors may have implemented any official task in the real checkout.
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; from pathlib import Path; from src.expert import registry; "
            "registry.PLUGIN_ROOT = Path(sys.argv[1]); from src.expert.cli import main; "
            "raise SystemExit(main(['collect', '--task', 'classify_objects']))",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "No expert planner" in result.stderr
    assert "isaac" not in result.stderr.lower()


def test_builtin_factories_and_legacy_mapping():
    from src.expert.tasks import TASKS

    assert set(TASKS) == {spec.task for spec in BUILTINS}
    for spec in BUILTINS:
        assert issubclass(load_planner(spec), TaskPlanner)
        assert callable(TASKS[spec.task])


def test_custom_planner_for_unimplemented_official_task():
    spec = planner_spec("classify_objects", "src.expert.planners.stack_blocks:StackBlocksPlanner")
    assert spec.status == "custom" and issubclass(load_planner(spec), TaskPlanner)
    with pytest.raises(ValueError, match="module:PlannerClass"):
        planner_spec("classify_objects", "not.an.entrypoint")


@pytest.mark.parametrize("entrypoint", ["builtins:dict", "src.expert.contracts:TaskPlanner"])
def test_invalid_planner_contract_fails_before_bootstrap(entrypoint):
    with pytest.raises(TypeError, match="concrete TaskPlanner"):
        load_planner(planner_spec("classify_objects", entrypoint))
