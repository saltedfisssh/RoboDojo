"""Source fingerprints, handoff scaffolds and bounded subprocess supervision."""

from contextlib import contextmanager, suppress
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import zipfile

from src.expert.diagnostics import write_json

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_fingerprint(task):
    files = [ROOT / f"task/RoboDojo/config/{task}.yml", ROOT / f"task/RoboDojo/tasks/{task}.py"]
    files += list((ROOT / "env/reward_manager").glob("*.py"))
    files += list((ROOT / "env_cfg").rglob("*.yml"))
    files.append(ROOT / "task/RoboDojo/config/_task.yml")
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def source_fingerprint(task):
    files = []
    for directory in ("src/expert", "src/datagen", "env", "utils", "task/RoboDojo/tasks"):
        for path in (ROOT / directory).rglob("*.py"):
            if "task_plugins" in path.parts and path.parent.name != "task_plugins" and task not in path.parts:
                continue
            files.append(path)
    files += list((ROOT / f"src/expert/task_plugins/{task}").glob("*.json"))
    files += [
        ROOT / name
        for name in (
            "scripts/generate_data.sh",
            "scripts/run-lerobot.sh",
            "scripts/internal/uv_env.sh",
            "scripts/requirements-lerobot.txt",
            "scripts/internal/validate_expert_dataset.py",
        )
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}
    names = ROOT / "src/expert/object_names.json"
    hashes[str(names.relative_to(ROOT))] = digest(names)
    hashes.update(official_fingerprint(task))
    return {"sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "files": hashes}


def archive_sources(path, fingerprint):
    """Keep the reviewable source even when a later iteration changes the checkout."""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in fingerprint["files"].items():
            source = (ROOT / name).read_bytes()
            if hashlib.sha256(source).hexdigest() != expected:
                raise ValueError(f"Source changed while archiving: {name}")
            archive.writestr(name, source)


def asset_fingerprint(samples):
    """Hash referenced object files and robot configs; external textures remain dependencies."""
    files = set()
    for sample in samples:
        for obj in sample["objects"]:
            folder = ROOT / "Assets/Object/RoboDojo" / obj["kind"] / obj["category"] / f"{obj['model']:05d}"
            files.add(folder / "metadata.json")
            asset = folder / "object.usdz"
            files.add(asset if asset.exists() else folder / "object.usd")
    files.update((ROOT / "Assets/Robots/x5").glob("*.yml"))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}


def runtime_info():
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in ("isaacsim", "isaaclab", "nvidia-curobo", "torch", "numpy", "h5py"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "versions": versions,
        "environment": {
            key: os.environ.get(key)
            for key in ("EXPERT_PHYSX_GPU", "EXPERT_CPU_THREADS", "EXPERT_PHYSX_THREADS", "CUDA_VISIBLE_DEVICES")
        },
    }


def load_protocol(task):
    path = ROOT / f"src/expert/task_plugins/{task}/validation.json"
    protocol = json.loads(path.read_text())
    if protocol.get("schema") != 1 or protocol.get("task") != task:
        raise ValueError(f"Invalid validation protocol: {path}")
    seen = set()
    for split in ("development", "validation", "rgb"):
        seeds = protocol["seeds"][split]
        if not seeds or any(type(seed) is not int or seed < 0 or seed in seen for seed in seeds):
            raise ValueError(f"Invalid/overlapping seed split: {split}")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seeds in {split}")
        seen.update(seeds)
    if not 0 < protocol["minimum_success_rate"] <= 1 or protocol["minimum_rgb_successes"] < 1:
        raise ValueError("Invalid qualification thresholds")
    return protocol


def create_plugin(task):
    from src.expert.registry import official_tasks, registered_planners

    if task not in official_tasks():
        raise ValueError(f"Unknown official task: {task}")
    if task in registered_planners():
        raise ValueError(f"Planner already registered for {task}")
    directory = ROOT / f"src/expert/task_plugins/{task}"
    directory.mkdir()  # never overwrite another contributor's work
    (directory / "__init__.py").write_text(f'"""Task plugin for {task}."""\n')
    (directory / "planner.py").write_text(
        '"""Implement this task against live simulator state."""\n\n'
        "from src.expert.contracts import PlanningError, TaskPlanner\n\n\n"
        "class Planner(TaskPlanner):\n"
        "    def run(self):\n"
        '        raise PlanningError("Planner not implemented; read HANDOFF.md")\n'
    )
    write_json(
        directory / "planner.json",
        {
            "schema": 1,
            "task": task,
            "entrypoint": f"src.expert.task_plugins.{task}.planner:Planner",
            "sampler": None,
            "status": "experimental",
            "note": "Not yet physically validated",
        },
    )
    write_json(directory / "validation.json", default_protocol(task))
    write_handoff(task, directory)
    return directory


def default_protocol(task):
    return {
        "schema": 1,
        "task": task,
        "seeds": {"development": [0, 1, 2], "validation": list(range(100, 110)), "rgb": [200, 201, 202]},
        "minimum_success_rate": 0.8,
        "minimum_rgb_successes": 1,
        "official_files": official_fingerprint(task),
    }


def write_handoff(task, directory):
    (directory / "HANDOFF.md").write_text(f"""# {task}: planner handoff

Implement this task in this directory. Read `docs/expert/DEVELOPMENT.md` first.
Official task: `task/RoboDojo/tasks/{task}.py`; layout: `task/RoboDojo/config/{task}.yml`.

The planner is discovered from `planner.json`; do not edit the central registry.
Own this directory and `tests/expert/test_{task}.py`; use a separate git worktree
per contributor. Shared-core fixes should be separate changes with regression tests.

```bash
bash scripts/generate_data.sh dev inspect --task {task}
bash scripts/generate_data.sh dev run --task {task} --split development --out data/dev/{task}/v1 --gpu 0
# Read report.json, raw/diagnostics/, worker.log; change the planner, then use a NEW run directory.
bash scripts/generate_data.sh dev run --task {task} --split validation --out data/dev/{task}/validation-v1 --gpu 0
bash scripts/generate_data.sh dev run --task {task} --split rgb --out data/dev/{task}/rgb-v1 --gpu 0
bash scripts/generate_data.sh dev qualify \\
  --validation data/dev/{task}/validation-v1 --rgb data/dev/{task}/rgb-v1 \\
  --out data/dev/{task}/qualification-v1 --gpu 0
```

Read the official instruction, reward, query/failure conditions and step limit.
Do not change them, teleport objects during execution, narrow the official asset
or layout distribution, remove inconvenient seeds, or treat a planner return value
as success. The expert may use simulator truth; this is not evidence of VLA perception.
Use motion/skills for every action; let EpisodeEnd propagate. Log semantic stages
with begin_subtask/stage. Query live poses after movement. Add semantic CPU tests
for target selection/order and recovery; CPU tests are not physical evidence.

If inspect reports unsupported layout semantics, implement a task-local sampler
in `layout.py` and reference its `module:Class` in `planner.json` (`sampler`).
Preserve the official YAML semantics and add layout tests. Do not silently ignore
relative frames, place_tag, clutter, physics, or dynamic initialization.

Use development seeds to iterate. Freeze code before running validation and RGB
splits. If those results lead to another code change, record the validation reuse;
the reused seeds are regression evidence, not a fresh held-out estimate. Never
claim a broad success rate from one successful rollout. Keep the plugin experimental
until a reviewer has checked qualification.json and the videos.

Deliver planner files, semantic tests, exact commands, all attempted seeds,
successes/failures, code fingerprint, export/restore evidence and known limitations
in VALIDATION.md. No GPU access or missing assets must be reported as untested.
""")


@contextmanager
def gpu_lock(gpu, timeout=600):
    """Cooperative lock across worktrees on this host; never kills other users."""
    directory = Path(os.environ.get("EXPERT_DEV_LOCK_DIR", "/tmp/robodojo-expert-dev"))
    directory.mkdir(exist_ok=True)
    path = directory / f"{socket.gethostname()}-gpu{gpu}.lock"
    with path.open("a+") as stream:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"GPU {gpu} development slot busy: {path}")
                time.sleep(0.2)
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def supervise(command, log, *, timeout, progress_path=None, lock_fd=None, env=None):
    """Bound startup and each episode, killing only this invocation's process group."""
    started = time.monotonic()
    current_seed, since = None, started
    timed_out = False
    with Path(log).open("x") as output:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_fd,) if lock_fd is not None else (),
        )
        try:
            while child.poll() is None:
                if progress_path and progress_path.exists():
                    try:
                        seed = json.loads(progress_path.read_text()).get("seed")
                        if seed != current_seed:
                            current_seed, since = seed, time.monotonic()
                    except (ValueError, OSError):
                        pass  # progress is diagnostic, never a success receipt
                if time.monotonic() - since > timeout:
                    timed_out = True
                    break
                time.sleep(0.2)
        finally:
            # A parent can exit on TERM while a grandchild ignores it and keeps
            # the inherited GPU lock. Always clean the owned process group.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return {
        "exit_code": child.returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
        "last_seed": current_seed,
        "command": command,
        "log": str(log),
    }


def utc_now():
    return datetime.now(UTC).isoformat()
