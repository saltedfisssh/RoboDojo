"""Stop only our own worker if Kit loses its GPU or makes no progress."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def start(journal, log):
    pid = os.getpid()
    offset = log.stat().st_size if log.exists() else 0
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), str(pid), identity(pid), str(journal), str(log), str(offset)],
        stdin=subprocess.DEVNULL,
    )
    # A remote NFS stat may report a stale offset. Ignore old crash messages
    # until this run's marker has arrived through the SSH log stream.
    print("EVAL_WATCHDOG_READY", pid, flush=True)
    return process


def monitor(pid, token, journal, log, offset):
    last_activity = time.time()
    marker = f"EVAL_WATCHDOG_READY {pid}".encode()
    active, tail = False, b""
    fatal = (
        b"GPU crash is detected",
        b"VkResult: ERROR_DEVICE_LOST",
        b"Scene state is corrupted. Simulation cannot continue",
        b"Invalid PhysX transform detected for /World/envs/env_0/robot",
    )
    while identity(pid) == token:
        time.sleep(10)
        reason = None
        if journal.exists():
            last_activity = max(last_activity, journal.stat().st_mtime)
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if not active:
                chunk = tail + chunk
                position = chunk.find(marker)
                if position >= 0:
                    active, chunk = True, chunk[position + len(marker) :]
                else:
                    tail, chunk = chunk[-len(marker) :], b""
            if active and any(message in chunk for message in fatal):
                reason = "Kit GPU device lost"
        if time.time() - last_activity > 900:
            reason = "No completed candidate or new attempt for 900 seconds"
        if reason and identity(pid) == token:
            print(f"EVAL_WORKER_WATCHDOG pid={pid}: {reason}; controller will resume", flush=True)
            os.kill(pid, signal.SIGKILL)
            return


if __name__ == "__main__":
    monitor(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]))
