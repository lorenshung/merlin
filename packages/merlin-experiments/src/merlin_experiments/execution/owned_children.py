"""Linux direct/adopted child ownership for an invocation's sole subreaper.

The caller must become a subreaper before creating any work. ECHILD, after actual
termination and waitpid reaping, closes that descendant tree. A /proc roster is
only used to obtain pidfds for the owner's unreaped children, never as a proof
that unrelated or delegated processes were discovered.
"""
from __future__ import annotations

import ctypes
import os
import signal
import time
from pathlib import Path

from . import _protocol as protocol


def become_child_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    flag = ctypes.c_int()
    if libc.prctl(36, 1, 0, 0, 0) != 0 or libc.prctl(37, ctypes.byref(flag), 0, 0, 0) != 0 or flag.value != 1:
        raise RuntimeError("Linux child subreaper unavailable")


def reap_owned_children(process, budget):
    deadline = time.monotonic() + budget
    escalation = time.monotonic() + min(1.0, budget / 2)
    owned, reaped, complete, error = {}, 0, False, None
    try:
        while time.monotonic() < deadline:
            children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
            for child in map(int, children):
                if child not in owned:
                    owned[child] = protocol.pidfd_open(child)
            for child, fd in list(owned.items()):
                try:
                    protocol.send_signal(fd, signal.SIGKILL if time.monotonic() >= escalation else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                waited, status = os.waitpid(child, os.WNOHANG)
                if waited:
                    if process is not None and child == process.pid:
                        process.returncode = os.waitstatus_to_exitcode(status)
                    os.close(owned.pop(child))
                    reaped += 1
            if not owned:
                try:
                    os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                except ChildProcessError:
                    complete = True
                    break
            time.sleep(0.02)
    except BaseException as exc:
        error = type(exc).__name__
    finally:
        protocol.close_fds(owned.values())
    return {"complete": complete, "reaped_children": reaped, "error": error}
