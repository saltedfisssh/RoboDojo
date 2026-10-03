"""Dump Python stacks when a native planner call stops making progress."""

from contextlib import contextmanager
import faulthandler
from functools import wraps


@contextmanager
def trace_after(seconds):
    # The native watchdog can report a thread blocked inside a CUDA call;
    # no ptrace permission, signal delivery or simulator mutation is needed.
    faulthandler.dump_traceback_later(seconds, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def trace_slow_planning(method):
    @wraps(method)
    def traced(*args, **kwargs):
        with trace_after(60):
            return method(*args, **kwargs)

    return traced
