"""Which elaborated-RTL simulator certifies a capsule — and why that one.

The cert tier is a FIDELITY, not a simulator. VCS, GSIM and Verilator all run the elaborated design and
all produce an ``elaborated_rtl`` verdict; which one answers is an availability and cost decision, not a
statement about how trustworthy the result is. Binding a tier index to one binary (``L3 = verilator``)
made that decision invisible and unchangeable, and it put two different fidelities on the same rung
across targets.

The order is COST, since the fidelity is equal:

* ``vcs`` — the reference commercial simulator; used when a license and the resources are actually free.
* ``gsim`` — the fast FIRRTL simulator, and the default working choice. Measured at corpus scale on the
  SIMT target: 25 capsules certified in 48 min, mean 115 s/capsule, against ~45 min/capsule on Verilator
  — ~23x, i.e. the same sweep is ~19 h on Verilator. That is the difference between a cert tier that
  runs per-capsule and one affordable only once per run.
* ``verilator`` — last resort, for a target with no GSIM adapter yet. Not a fidelity compromise; it is
  simply the slow one.

Selection is by AVAILABILITY in that order, and every engine passed over is recorded with the reason it
was passed over. A tier that cannot run must come back as unavailable with that record — never silently
downgraded to a model tier, which is how a functional result gets read as an RTL certification.

WHAT "EQUAL FIDELITY" RESTS ON, and what it does not cover. The equality above is established by an
equivalence certificate, and those certificates carry ``evidence: output_bytes``: the engines produced
identical output for identical ELFs. Halting behaviour is NOT in that evidence, and the engines differ
there. Measured on this target: a program that violates a design ``assert`` makes Verilator ``$stop``
and exit non-zero, while the GSIM model prints the same assertion and runs to completion with
``exit_code=0`` and a full console. A caller that decided on the exit code alone therefore treated a
refused program as a clean run — so the backend refuses on the assertion text itself, and an engine
adopted here on an output-bytes certificate alone has been adopted on a narrower claim than this
ranking implies. Assertion enforcement, trap reporting and timeout semantics each need their own
evidence before an engine is ranked as equal.
"""

from __future__ import annotations

import fcntl
import os
import stat
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Declared once, with the rationale above. Cost order among engines of EQUAL fidelity.
ENGINE_PRIORITY: tuple[str, ...] = ("vcs", "gsim", "verilator")

# Bound per-suite fan-out by simulator cost, even when a caller explicitly asks
# for more workers. Host-load sizing is a snapshot: two independent grades can
# both observe an idle host and otherwise each start a full-width pool. The
# asynchronous broker and direct/scheduled graders must use the same caps.
CAPSULE_WORKER_CAP: dict[str, int] = {"verilator": 4, "gsim": 5, "spike": 8}
GSIM_RUNTIME_SLOT_PROTOCOL = "reentrant_per_thread_v1"

# Host runners and selected backends may both guard the same synchronous native
# call. Reuse only this thread's ownership; independent threads still need slots.
_GSIM_LOCAL = threading.local()
_GSIM_FDS: set[int] = set()
_GSIM_FD_LOCK = threading.Lock()


def _gsim_after_fork_child() -> None:
    global _GSIM_LOCAL
    # flock ownership follows the open file description across fork. Close the
    # child's copies without unlocking the parent's reservations.
    for fd in _GSIM_FDS:
        os.close(fd)
    _GSIM_FDS.clear()
    _GSIM_LOCAL = threading.local()
    _GSIM_FD_LOCK.release()


os.register_at_fork(
    before=_GSIM_FD_LOCK.acquire,
    after_in_parent=_GSIM_FD_LOCK.release,
    after_in_child=_gsim_after_fork_child,
)


def _close_gsim_fd(fd: int) -> None:
    with _GSIM_FD_LOCK:
        _GSIM_FDS.discard(fd)
        os.close(fd)


def _verified_slot_file(path: Path) -> int:
    """Open a private lock inode and keep fork cleanup aware of its descriptor."""
    with _GSIM_FD_LOCK:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        _GSIM_FDS.add(fd)
    try:
        info = os.fstat(fd)
    except BaseException:
        _close_gsim_fd(fd)
        raise
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        _close_gsim_fd(fd)
        raise RuntimeError(f"GSim slot file is not regular and owned by this user: {path}")
    return fd


def _read_process_evidence(path: Path, member: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, ProcessLookupError) as exc:
        try:
            member.stat()
        except (FileNotFoundError, ProcessLookupError):
            return None  # The process exited during the census.
        except OSError as stat_exc:
            raise RuntimeError(f"cannot verify native process {member.name}: {stat_exc}") from stat_exc
        raise RuntimeError(f"cannot verify native process {member.name}: missing {path.name}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot verify native process {member.name}: {exc}") from exc


def _status_rows(status: bytes, field: bytes) -> list[list[bytes]]:
    rows = []
    for line in status.splitlines():
        name, separator, value = line.partition(b":")
        if separator and name == field:
            rows.append(value.split())
    return rows


def _verified_uids(status: bytes, member: Path) -> tuple[int, ...]:
    uid_rows = _status_rows(status, b"Uid")
    if len(uid_rows) != 1 or len(uid_rows[0]) != 4:
        raise RuntimeError(f"cannot verify native process {member.name}: malformed Uid evidence")
    try:
        credentials = tuple(int(value) for value in uid_rows[0])
    except ValueError as exc:
        raise RuntimeError(f"cannot verify native process {member.name}: malformed Uid evidence") from exc
    if any(value < 0 for value in credentials):
        raise RuntimeError(f"cannot verify native process {member.name}: malformed Uid evidence")
    return credentials


def _native_gsim_count(*, proc_root: Path = Path("/proc")) -> int:
    """Bound same-user native plusarg load without assuming an emulator name or path.

    The kernel-reported UID is checked before reading an argv. Missing or unreadable
    evidence refuses admission; a partial native plusarg signature and a verified live
    process with persistently empty argv each count as one potential native. This is an
    upper bound, not an exact process census. A process appearing after this snapshot
    remains an external race.
    """
    try:
        members = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot verify native process census: {exc}") from exc
    current_uid = os.getuid()
    count = 0
    for member in members:
        if not member.name.isdecimal():
            continue
        status = _read_process_evidence(member / "status", member)
        if status is None:
            continue
        credentials = _verified_uids(status, member)
        if current_uid not in credentials:
            continue
        cmdline = _read_process_evidence(member / "cmdline", member)
        if cmdline is None:
            continue
        if not cmdline:
            # A fork/exec or exit can leave a live status with an empty argv for a
            # moment. Wait briefly for complete evidence, then conservatively count
            # a verified same-user live process as one potential native simulator.
            for attempt in range(5):
                current_status = _read_process_evidence(member / "status", member)
                if current_status is None:
                    break
                state_rows = _status_rows(current_status, b"State")
                if (
                    len(state_rows) != 1
                    or not state_rows[0]
                    or len(state_rows[0][0]) != 1
                    or not state_rows[0][0].isalpha()
                ):
                    raise RuntimeError(f"cannot verify native process {member.name}: malformed State evidence")
                if state_rows[0][0] in {b"Z", b"X"}:
                    break
                if current_uid not in _verified_uids(current_status, member):
                    raise RuntimeError(f"cannot verify native process {member.name}: changed Uid evidence")
                if attempt == 4:
                    count += 1
                    break
                time.sleep(0.01)
                cmdline = _read_process_evidence(member / "cmdline", member)
                if cmdline is None or cmdline:
                    break
            if not cmdline:
                continue
        if not cmdline.endswith(b"\0"):
            raise RuntimeError(f"cannot verify native process {member.name}: incomplete argv")
        argv = cmdline[:-1].split(b"\0")
        if any(arg.startswith((b"+loadmem=", b"+max-cycles=")) for arg in argv):
            count += 1
    return count


def _locked_slot_count(root: Path, *, own_index: int) -> int:
    """Include this pending reservation and all old/new kernel file-lock holders."""
    count = 1
    for index in range(CAPSULE_WORKER_CAP["gsim"]):
        if index == own_index:
            continue
        fd = _verified_slot_file(root / f"slot_{index}.lock")
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                count += 1
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            _close_gsim_fd(fd)
    return count


@contextmanager
def _admission_mutex(root: Path, deadline: float | None):
    fd = _verified_slot_file(root / "admission.lock")
    locked = False
    try:
        while not locked:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("all five GSim slots stayed busy until the capsule wait deadline")
                time.sleep(0.01)
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        _close_gsim_fd(fd)


def capsule_worker_cap(engine: str) -> int:
    """Conservative parallel capsule limit for one simulator engine."""
    return CAPSULE_WORKER_CAP.get(engine, 8)


@contextmanager
def gsim_runtime_slot(*, wait_timeout_s: float | None = None, slot_root: Path | None = None):
    """Hold one of five same-user GSim slots for the entire native simulation.

    Per-suite worker bounds do not protect the machine when two independent grades run at once.
    Advisory file locks survive abrupt worker exit without stale PID reclamation. A private,
    fixed per-user directory makes every Merlin capsule process share the same limit. An
    admission mutex joins held/pending reservations with a same-user native /proc census;
    uncoordinated launches after that snapshot cannot be prevented by this protocol.
    Nested synchronous calls in the same thread reuse its slot. Other threads and
    forked children acquire their own; do not launch concurrent children inside one slot.
    """
    root = slot_root or Path("/tmp") / f"merlin_gsim_slots_{os.getuid()}"
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"GSim slot directory is not private and owned by this user: {root}")
    key = str(root.resolve())
    held = getattr(_GSIM_LOCAL, "held", None)
    if held is None:
        held = _GSIM_LOCAL.held = {}
    if key in held:
        yield
        return
    owner_pid = os.getpid()
    deadline = None if wait_timeout_s is None else time.monotonic() + max(0.0, wait_timeout_s)
    fd = None
    try:
        while fd is None:
            with _admission_mutex(root, deadline):
                for index in range(CAPSULE_WORKER_CAP["gsim"]):
                    opened = _verified_slot_file(root / f"slot_{index}.lock")
                    try:
                        fcntl.flock(opened, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        _close_gsim_fd(opened)
                        continue
                    except BaseException:
                        _close_gsim_fd(opened)
                        raise
                    try:
                        # The census includes verified live empty-argv PIDs as potential
                        # natives, so this sum is a conservative admission upper bound.
                        admitted_load = _native_gsim_count() + _locked_slot_count(root, own_index=index)
                        if admitted_load <= CAPSULE_WORKER_CAP["gsim"]:
                            fd = opened
                            break
                    finally:
                        if fd is None:
                            fcntl.flock(opened, fcntl.LOCK_UN)
                            _close_gsim_fd(opened)
                    break
            if fd is None:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("all five GSim slots stayed busy until the capsule wait deadline")
                time.sleep(0.1)
        held[key] = fd
        yield
    finally:
        if fd is not None and os.getpid() == owner_pid:
            held.pop(key, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
            _close_gsim_fd(fd)


# Every engine here answers at this fidelity; the tier records it rather than inferring from the name.
ELABORATED_RTL = "elaborated_rtl"


class UnrecordedSelection(RuntimeError):
    """An engine reported itself available without saying why. Refused: a tier that resolved to an engine
    for no recorded reason cannot be audited afterwards, and reads as if it were the declared one."""

    def __init__(self, target: str, engine: str):
        self.target, self.engine = target, engine
        super().__init__(
            f"{target}: engine {engine!r} reported available with no reason recorded; a "
            f"selection that cannot be explained afterwards is refused, not defaulted"
        )


class NoEngineAvailable(RuntimeError):
    """No elaborated-RTL engine can run for this target. Carries the per-engine reasons."""

    def __init__(self, target: str, considered: list[dict[str, Any]]):
        self.target, self.considered = target, considered
        detail = "; ".join(f"{c['engine']}: {c['reason']}" for c in considered) or "none registered"
        super().__init__(f"{target}: no elaborated-RTL engine available ({detail})")


def _ordered(engines: dict[str, Any]) -> list[str]:
    """Registered engines in priority order; anything unknown to the policy sorts last, alphabetically,
    so a newly added engine is USED rather than silently dropped before anyone declares its priority."""
    known = [e for e in ENGINE_PRIORITY if e in engines]
    return known + sorted(e for e in engines if e not in ENGINE_PRIORITY)


def select(target: str, engines: dict[str, Callable[[], tuple[bool, str]]]) -> dict[str, Any]:
    """Choose the elaborated-RTL engine for ``target``.

    ``engines`` maps an engine name to a probe returning ``(available, reason)``. Probes are called in
    priority order and STOP at the first available one, so an expensive probe for a lower-priority
    engine is never paid. Returns the selection record; raises :class:`NoEngineAvailable` when none can
    run (fail closed — the caller reports the tier unavailable, it does not substitute a lesser tier).
    """
    considered: list[dict[str, Any]] = []
    for name in _ordered(engines):
        try:
            ok, reason = engines[name]()
        except Exception as exc:  # noqa: BLE001 - a broken probe is not availability
            ok, reason = False, f"probe raised {type(exc).__name__}: {exc}"
        considered.append({"engine": name, "available": bool(ok), "reason": reason})
        if ok and not str(reason or "").strip():
            # An engine that resolved for NO RECORDED REASON is the silent-degradation shape: the tier
            # answers with a different engine than the capsule asked for, the numbers look right, and the
            # result gets cited. Refuse it rather than defaulting.
            raise UnrecordedSelection(target, name)
        if ok:
            return {
                "engine": name,
                "fidelity": ELABORATED_RTL,
                "reason": reason,
                "considered": considered,
                "passed_over": [c["engine"] for c in considered[:-1]],
            }
    raise NoEngineAvailable(target, considered)


def describe(selection: dict[str, Any]) -> str:
    """One line for a report: what ran, and what it was chosen over."""
    over = selection.get("passed_over") or []
    tail = f" (over {', '.join(over)})" if over else ""
    return f"{selection['engine']} [{selection['fidelity']}]{tail}"
