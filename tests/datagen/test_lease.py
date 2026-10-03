import fcntl
import json
import os
import subprocess
import sys
import time

from src.datagen.lease import next_ticket, process_identity


def test_unreadable_ticket_does_not_crash_other_waiting_workers(tmp_path):
    record = {
        "pid": os.getpid(),
        "process_start": process_identity(os.getpid()),
        "priority": 0,
        "created_ns": time.time_ns(),
    }
    (tmp_path / "broken.json").write_text("")
    (tmp_path / "valid.json").write_text(json.dumps(record))
    assert next_ticket(tmp_path) == "valid.json"
    (tmp_path / "broken.json").write_text(json.dumps({**record, "priority": -4}))
    assert next_ticket(tmp_path) == "broken.json"


def test_dead_or_reused_pid_cannot_block_the_gpu_queue(tmp_path):
    live = {"pid": os.getpid(), "process_start": process_identity(os.getpid()), "priority": 10, "created_ns": 2}
    (tmp_path / "live.json").write_text(json.dumps(live))
    (tmp_path / "stale.json").write_text(json.dumps({**live, "priority": -1, "process_start": "obsolete"}))
    assert next_ticket(tmp_path) == "live.json"
    assert not (tmp_path / "stale.json").exists()


def test_old_rgb_ticket_eventually_precedes_new_smoke_and_frozen_workers_agree(tmp_path):
    now = time.time_ns()
    live = {"pid": os.getpid(), "process_start": process_identity(os.getpid())}
    (tmp_path / "rgb.json").write_text(json.dumps({**live, "priority": 10, "created_ns": now - 1900 * 10**9}))
    (tmp_path / "smoke.json").write_text(json.dumps({**live, "priority": 0, "created_ns": now}))
    assert next_ticket(tmp_path) == "rgb.json"
    # Old frozen workers only sort persisted priority and creation time.
    tickets = [(p.name, json.loads(p.read_text())) for p in tmp_path.glob("*.json")]
    assert min(tickets, key=lambda item: (item[1]["priority"], item[1]["created_ns"]))[0] == "rgb.json"
    (tmp_path / "rgb.json").unlink()
    assert next_ticket(tmp_path) == "smoke.json"


def test_smoke_precedes_rgb_and_still_takes_the_same_exclusive_lock(tmp_path):
    cache = tmp_path / ".cache"
    cache.mkdir()
    stream = (cache / "datagen-isaac.lock").open("a")
    fcntl.flock(stream, fcntl.LOCK_EX)
    script = """
from pathlib import Path
import sys
from src.datagen import lease
lease.ROOT = Path(sys.argv[1])
with lease.acquire(priority=int(sys.argv[2])):
    with (lease.ROOT / 'order').open('a') as output:
        output.write(sys.argv[2] + '\\n')
"""
    children = []
    try:
        for priority in (10, 0):
            process = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path), str(priority)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            children.append(process)
            assert "Waiting for the Isaac GPU lease" in process.stdout.readline()
        assert not (tmp_path / "order").exists()
        stream.close()
        for process in children:
            output, error = process.communicate(timeout=20)
            assert process.returncode == 0, output + error
        assert (tmp_path / "order").read_text().splitlines() == ["0", "10"]
        assert not list((cache / "datagen-isaac-queue").glob("*.json"))
    finally:
        stream.close()
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_rgb_wait_limit_survives_an_older_promoted_smoke_and_respects_explicit_priority(tmp_path):
    now = time.time_ns()
    live = {"pid": os.getpid(), "process_start": process_identity(os.getpid())}
    (tmp_path / "smoke.json").write_text(
        json.dumps({**live, "priority": -1, "original_priority": 0, "created_ns": now - 7200 * 10**9})
    )
    (tmp_path / "rgb.json").write_text(json.dumps({**live, "priority": 10, "created_ns": now - 1900 * 10**9}))
    (tmp_path / "explicit.json").write_text(json.dumps({**live, "priority": -2, "created_ns": now}))
    assert next_ticket(tmp_path) == "explicit.json"
    assert json.loads((tmp_path / "rgb.json").read_text())["priority"] == 10
    (tmp_path / "explicit.json").unlink()
    assert next_ticket(tmp_path) == "rgb.json"
    tickets = [(p.name, json.loads(p.read_text())) for p in tmp_path.glob("*.json")]
    assert min(tickets, key=lambda item: (item[1]["priority"], item[1]["created_ns"]))[0] == "rgb.json"
    (tmp_path / "rgb.json").unlink()
    assert next_ticket(tmp_path) == "smoke.json"
