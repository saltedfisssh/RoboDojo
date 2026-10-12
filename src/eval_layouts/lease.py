"""A task lease using atomic directories on the shared filesystem.

The cluster mount's flock is local to each server, so it cannot protect a
layout task from simultaneous local and SSH writers.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


class TaskLock:
    def __init__(self, directory):
        self.path = Path(directory) / ".generation.lease"
        self.token = uuid4().hex
        self.heartbeat = None

    def acquire(self, *, wait=True):
        while True:
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                heartbeat = self.path / "heartbeat"
                observed = heartbeat if heartbeat.exists() else self.path
                try:
                    age = time.time() - observed.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > 120:
                    reaper = self.path.with_name(".generation.reaping")
                    try:
                        reaper.mkdir()
                    except FileExistsError:
                        if time.time() - reaper.stat().st_mtime > 120:
                            reaper.rmdir()
                    else:
                        try:
                            if self.path.exists() and time.time() - observed.stat().st_mtime > 120:
                                stale = self.path.with_name(f".generation.stale-{uuid4().hex}")
                                self.path.rename(stale)
                                shutil.rmtree(stale)
                        except FileNotFoundError:
                            pass
                        finally:
                            reaper.rmdir()
                    continue
                if not wait:
                    raise BlockingIOError(f"Layout generation already owns {self.path.parent}")
                time.sleep(1)
        (self.path / "owner.json").write_text(json.dumps({"token": self.token, "pid": os.getpid()}))
        (self.path / "heartbeat").touch()
        # A small independent process keeps the lease alive even when a Kit
        # native call holds Python's GIL, and releases it if this worker dies.
        self.heartbeat = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(self.path),
                self.token,
                str(os.getpid()),
                process_identity(os.getpid()),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self

    def ensure_owned(self):
        owner = json.loads((self.path / "owner.json").read_text())
        if owner["token"] != self.token:
            raise RuntimeError(f"Lost layout task lease: {self.path.parent}")
        (self.path / "heartbeat").touch()

    def release(self):
        if self.heartbeat is not None:
            self.heartbeat.terminate()
            self.heartbeat.wait(timeout=10)
        try:
            if json.loads((self.path / "owner.json").read_text())["token"] == self.token:
                shutil.rmtree(self.path)
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self.acquire(wait=True)

    def __exit__(self, *_):
        self.release()


if __name__ == "__main__":
    path, token, pid, identity = sys.argv[1:]
    path, pid = Path(path), int(pid)
    try:
        while json.loads((path / "owner.json").read_text())["token"] == token:
            if process_identity(pid) != identity:
                shutil.rmtree(path)
                break
            (path / "heartbeat").touch()
            time.sleep(5)
    except FileNotFoundError:
        pass
