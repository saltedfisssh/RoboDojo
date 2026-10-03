"""Serialize Isaac processes on the shared GPU; CPU export remains concurrent."""

import fcntl
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from src.datagen.layouts import ROOT


def process_identity(pid):
    try:
        # Field 22 is the kernel process start time; PID reuse must not keep
        # an abandoned high-priority ticket alive indefinitely.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def next_ticket(folder):
    waiting = []
    for path in folder.glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            # A storage outage or an older non-atomic queue editor must not
            # crash every unrelated waiting simulation worker.
            continue
        if process_identity(record["pid"]) != record["process_start"]:
            path.unlink(missing_ok=True)
        else:
            waiting.append((path, record))
    # Promoting the oldest ticket of any kind still left RGB waiting behind
    # hours of older smoke jobs. Protect aged RGB explicitly, including when
    # an old frozen worker has already promoted a smoke ticket to -1.
    # Explicit user/probe priorities remain respected. Persist the selection
    # so frozen workers agree without interrupting the active lease holder.
    now = time.time_ns()
    explicit_priority = any(record["priority"] < 0 and "original_priority" not in record for _, record in waiting)
    aged_rgb = [
        (path, record)
        for path, record in waiting
        if record.get("original_priority", record["priority"]) >= 10
        and now - record["created_ns"] >= 1800 * 1_000_000_000
    ]
    promote = None
    if aged_rgb and not explicit_priority:
        path, record = min(aged_rgb, key=lambda item: item[1]["created_ns"])
        promote = (path, record, -2)
    elif waiting and not any(record["priority"] < 0 for _, record in waiting):
        path, record = min(waiting, key=lambda item: item[1]["created_ns"])
        if now - record["created_ns"] >= 1800 * 1_000_000_000:
            promote = (path, record, -1)
    if promote is not None:
        path, record, priority = promote
        if record["priority"] != priority:
            promoted = {
                **record,
                "original_priority": record.get("original_priority", record["priority"]),
                "priority": priority,
            }
            temporary = path.with_suffix(f".{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(promoted))
            if path.exists() and process_identity(record["pid"]) == record["process_start"]:
                temporary.replace(path)
                record.update(promoted)
            else:
                temporary.unlink(missing_ok=True)
    return (
        min((record["priority"], record["created_ns"], path.name) for path, record in waiting)[2] if waiting else None
    )


def acquire(priority=10):
    folder = ROOT / ".cache"
    folder.mkdir(exist_ok=True)
    stream = (folder / "datagen-isaac.lock").open("a")
    queue = folder / "datagen-isaac-queue"
    queue.mkdir(exist_ok=True)
    ticket = queue / f"{os.getpid()}-{uuid4().hex}.json"
    temporary = ticket.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "process_start": process_identity(os.getpid()),
                "priority": priority,
                "created_ns": time.time_ns(),
            }
        )
    )
    temporary.replace(ticket)
    print("[datagen] Waiting for the Isaac GPU lease.", flush=True)
    eligible_since = None
    try:
        while True:
            if next_ticket(queue) == ticket.name:
                # Let a continuing smoke campaign enqueue its next task before
                # starting another long RGB job. The same kernel lock also
                # serializes workers launched before this priority queue existed.
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if eligible_since is None:
                        eligible_since = time.monotonic()
                    if priority <= 0 or time.monotonic() - eligible_since >= 2:
                        break
                    fcntl.flock(stream, fcntl.LOCK_UN)
                except BlockingIOError:
                    eligible_since = None
            else:
                eligible_since = None
            time.sleep(0.25)
    except BaseException:
        stream.close()
        raise
    finally:
        ticket.unlink(missing_ok=True)
    print("[datagen] Acquired the Isaac GPU lease.", flush=True)
    return stream
