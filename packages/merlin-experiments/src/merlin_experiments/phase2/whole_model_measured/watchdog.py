"""Keep a measured run going across launcher exits -- without ever signalling a live one.

When a run's launcher EXITS, the watchdog decides from the run's own records whether the next run
may start, and if so prepares it with :func:`.runs.resume` (the SAME method and prohibited roles, the
latest workspace as the seed, the store kept) and starts it through the declared launch.  It only
ever acts on a launcher that is already gone.

It does NOT relaunch:

* a run that stopped on EVIDENCE (a plateau, a certified best at the bar) -- that is the answer;
* a run an operator asked to stop (:func:`.sessions.request_stop`), whether or not its launcher got to
  record the stop before it exited;
* a run that lasted under ``min_seconds`` -- a crash loop is reported, not repeated;
* past ``max_relaunches``.

A run the host's memory guard stopped is relaunched only after memory has stayed above
``recovered_gib`` for three consecutive checks, bounded by ``memory_wait_seconds``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import runs as RUNS
from . import sessions as SES

#: Written by the session loop (:func:`.sessions.run_sessions`) into the run's stage directory.
SESSIONS_RECORD = Path("stage") / "sessions.json"
#: Written by the launcher when the host's memory guard stopped it.
RESOURCE_RECORD = Path("stage") / "host_resource_telemetry.json"
EVIDENCE_KINDS = ("plateau", "bar")


@dataclass
class WatchPolicy:
    max_relaunches: int = 6
    min_seconds: float = 900.0
    recovered_gib: float = 28.0
    memory_wait_seconds: float = 4 * 3600.0
    poll_seconds: float = 30.0


def alive(pid: int) -> bool:
    return Path(f"/proc/{int(pid)}").is_dir()


def memory_available_gib() -> float:
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
    return int(fields["MemAvailable"].split()[0]) / 2**20


def stop_reason(run_dir: Path) -> dict[str, Any] | None:
    """Why the run's own records say it must not be relaunched, or None."""
    requested = SES.operator_stop(Path(run_dir) / SES.OPERATOR_STOP_FILE)
    if requested is not None:
        # AN OPERATOR'S REQUEST OUTRANKS AN EXIT: a launcher that died before reaching its next session
        # boundary never recorded the stop, and relaunching it would undo the request.
        return {"reason": requested["reason"], "kind": SES.OPERATOR_STOP}
    try:
        document = json.loads((Path(run_dir) / SESSIONS_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stopped = document.get("stopped") or {}
    if stopped.get("kind") in EVIDENCE_KINDS:
        return {"reason": f"the run stopped on evidence: {stopped.get('reason')}", "kind": "evidence"}
    if stopped.get("kind"):
        # THE LAUNCHER DECIDED TO STOP, and wrote why: a spent budget, a crash loop, a circuit breaker,
        # an exhausted account, another model.  Its exit is not a crash, and a relaunch would only
        # re-confirm the same stop -- after spending a board job on it (measured on the old line: a
        # plateaued run relaunched by exit alone dispatched one board job re-confirming its best).
        return {"reason": f"the run stopped itself ({stopped['kind']}): {stopped.get('reason')}", "kind": "recorded"}
    return None


def memory_guard_stopped(run_dir: Path) -> bool:
    try:
        return json.loads((Path(run_dir) / RESOURCE_RECORD).read_text()).get("status") == "resource_limit"
    except (OSError, ValueError):
        return False


def watch(
    run_dir: Path,
    pid: int,
    *,
    launch: Callable[[RUNS.PreparedRun], int],
    why: str,
    policy: WatchPolicy | None = None,
    resume: Callable[..., RUNS.PreparedRun] = RUNS.resume,
    is_alive: Callable[[int], bool] = alive,
    memory: Callable[[], float] = memory_available_gib,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Watch ``pid`` (the launcher of ``run_dir``); on its exit, relaunch per the policy.  Returns the
    watch's own record: every relaunch and why it stopped."""
    policy = policy or WatchPolicy()
    run_dir = Path(run_dir)
    relaunches: list[dict[str, Any]] = []
    while True:
        started = clock()
        log(f"watching {run_dir.name} launcher {pid}")
        while is_alive(pid):
            sleep(policy.poll_seconds)
        lasted = clock() - started
        evidence = stop_reason(run_dir)
        final = evidence is not None and evidence["kind"] in ("evidence", SES.OPERATOR_STOP)
        if evidence is not None and (final or not memory_guard_stopped(run_dir)):
            return {"stopped": evidence, "relaunches": relaunches}
        if len(relaunches) >= policy.max_relaunches:
            return {"stopped": {"kind": "budget", "reason": "the relaunch budget is spent"}, "relaunches": relaunches}
        if memory_guard_stopped(run_dir):
            calm, waited = 0, 0.0
            while calm < 3 and waited < policy.memory_wait_seconds:
                calm = calm + 1 if memory() >= policy.recovered_gib else 0
                sleep(60.0)
                waited += 60.0
            if calm < 3:
                return {"stopped": {"kind": "memory", "reason": "memory did not recover"}, "relaunches": relaunches}
        elif lasted < policy.min_seconds:
            return {
                "stopped": {
                    "kind": "crash_loop",
                    "reason": f"the run lasted {lasted:.0f} s, under {policy.min_seconds:g} s",
                },
                "relaunches": relaunches,
            }
        prepared = resume(run_dir, why=why)
        pid = launch(prepared)
        relaunches.append({"from": str(run_dir), "to": str(prepared.run_dir), "method": prepared.method, "pid": pid})
        log(f"relaunched {prepared.run_dir} ({prepared.method}) pid {pid}")
        run_dir = prepared.run_dir


__all__ = ["WatchPolicy", "memory_guard_stopped", "stop_reason", "watch"]
