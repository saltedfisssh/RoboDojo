"""Local work queue for persistent, GPU-bound collection processes.

Only the batch parent allocates seeds and success slots. Workers request their
next episode when ready, so a slow scene cannot hold up another worker's queue.
"""

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import json
from pathlib import Path
import socket
import socketserver
import tempfile
import threading
import time
import uuid

from src.expert.storage import attempted_seeds, episode_count, episode_exists, episode_metadata, log_attempt


@contextmanager
def batch_lock(out):
    """Prevent two coordinators from allocating into the same output directory."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another batch is already collecting into {out}") from exc
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reserved_seeds(out):
    """Also skip interrupted assignments when resuming; never reuse a lease."""
    seeds = attempted_seeds(out)
    for path in out.glob("scheduling_*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "assigned":
                seeds.add(event["seed"])
    return seeds


class WorkQueue:
    def __init__(self, args, worker_gpus, planner):
        self.out = args.out
        self.task = args.task
        self.planner = planner
        self.target = args.num
        self.successes = episode_count(args.out)
        self.budget = max(0, self.target - self.successes) * args.attempts_per_episode
        self.next_seed = args.seed_start
        self.used = reserved_seeds(args.out)
        self.worker_gpus = worker_gpus
        self.inflight = {}
        self.receipts = set()
        self.exited = set()
        self.ever_assigned = set()
        self.stopped_unused = set()
        self.assigned = 0
        self.completed = 0
        self.cancelled = False
        self.condition = threading.Condition()
        self.events = args.out / f"scheduling_{uuid.uuid4().hex}.jsonl"
        self.started = time.monotonic()

    def event(self, kind, **values):
        row = dict(event=kind, utc=datetime.now(UTC).isoformat(), elapsed=round(time.monotonic() - self.started, 3))
        with self.events.open("a") as stream:
            stream.write(json.dumps({**row, **values}) + "\n")

    def acquire(self, worker):
        with self.condition:
            if worker not in self.worker_gpus or worker in self.inflight:
                raise ValueError(f"Invalid or duplicate worker request: {worker}")
            while True:
                if (
                    self.cancelled
                    or worker in self.exited
                    or self.successes >= self.target
                    or self.assigned >= self.budget
                ):
                    return {"seed": None, "target_reached": self.successes >= self.target}
                # Reserve one possible success per running episode. Near the
                # target, idle workers wait for a failure or global completion.
                if self.successes + len(self.inflight) < self.target:
                    while self.next_seed in self.used:
                        self.next_seed += 1
                    seed = self.next_seed
                    self.used.add(seed)
                    self.next_seed += 1
                    self.inflight[worker] = seed
                    self.ever_assigned.add(worker)
                    self.assigned += 1
                    self.event("assigned", worker=worker, gpu=self.worker_gpus[worker], seed=seed)
                    return {"seed": seed}
                self.condition.wait()

    def complete(self, worker, row):
        with self.condition:
            if (worker, row["seed"]) in self.receipts:
                return  # acknowledgement raced with recovery after process exit
            if self.inflight.get(worker) != row["seed"]:
                raise ValueError(f"Unexpected completion from worker {worker}: {row['seed']}")
            if row["result"] == "success":
                path = self.out / f"episode_{row['seed']:07d}.h5"
                if not episode_exists(self.out, row["seed"]):
                    raise ValueError(f"Worker reported success before saving {path}")
                self.successes += 1
            del self.inflight[worker]
            self.receipts.add((worker, row["seed"]))
            self.completed += 1
            self.event("completed", worker=worker, seed=row["seed"], result=row["result"], successes=self.successes)
            self.condition.notify_all()

    def worker_exited(self, worker, code):
        """Release a reservation only after the process can no longer write."""
        with self.condition:
            self.exited.add(worker)
            seed = self.inflight.get(worker)
            if seed is not None:
                # A crash can happen after save, background conversion or attempt
                # logging, but before the worker acknowledges completion.
                log = self.out / f"attempts_w{worker}.jsonl"
                rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
                rows = [row for row in rows if row["seed"] == seed]
                if rows:
                    row = rows[-1]
                else:
                    row = dict(seed=seed, task=self.task, planner=self.planner, gpu=self.worker_gpus[worker])
                    path = self.out / f"episode_{seed:07d}.h5"
                    if episode_exists(self.out, seed):
                        meta = episode_metadata(self.out, seed)
                        if not meta["success"] or meta["seed"] != seed:
                            raise ValueError(f"Invalid recovered episode: {path}")
                        row.update(result="success", steps=meta["num_frames"], file=str(path), seconds=0)
                    else:
                        row.update(result="error", reason=f"worker exited before reporting (code {code})", seconds=0)
                    log_attempt(self.out, worker, row)
                self.complete(worker, row)
            self.event("worker_exit", worker=worker, code=code)
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def stop_unused(self, worker):
        """The goal must not wait for a process still stuck in initialization."""
        with self.condition:
            if self.successes < self.target or worker in self.ever_assigned or worker in self.stopped_unused:
                return False
            self.stopped_unused.add(worker)
            self.event("stop_unused", worker=worker, reason="target reached before first assignment")
            return True


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


@contextmanager
def serve_queue(queue):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            worker = None
            while line := self.rfile.readline():
                request = json.loads(line)
                if worker is None:
                    worker = request["worker"]
                if worker != request["worker"]:
                    raise ValueError("A connection cannot change worker id")
                if request.get("completed") is not None:
                    queue.complete(worker, request["completed"])
                response = queue.acquire(worker)
                self.wfile.write((json.dumps(response) + "\n").encode())
                self.wfile.flush()
                if response["seed"] is None:
                    return

    # Unix sockets belong on a local filesystem, not the shared dataset mount.
    # TemporaryDirectory is owner-only; no network port or pickle is involved.
    with tempfile.TemporaryDirectory(prefix="expert-queue-") as directory:
        address = str(Path(directory) / "work.sock")
        with _Server(address, Handler) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            thread.start()
            try:
                yield address
            finally:
                queue.cancel()
                server.shutdown()
                thread.join()


class WorkerClient:
    def __init__(self, address, worker):
        self.worker = worker
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(address)
        self.stream = self.socket.makefile("rwb")
        self.target_reached = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        self.socket.close()

    def acquire(self, completed=None):
        self.stream.write((json.dumps({"worker": self.worker, "completed": completed}) + "\n").encode())
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise RuntimeError("Batch coordinator disconnected")
        response = json.loads(line)
        self.target_reached = response.get("target_reached", False)
        return response["seed"]
