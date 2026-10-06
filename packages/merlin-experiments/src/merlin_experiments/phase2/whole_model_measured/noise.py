"""How noisy ONE machine is, from its own repeated solo readings of identical programs.

A noise margin decides what counts as an improvement; a drift tolerance decides whether a batch's
control still measures what it measured alone.  Neither is a property of the experiment -- each is a
property of the MACHINE.  Measured on two boards of one target: an identical ELF read alone five days
apart moved 2.4% on one and 0.36% on the other.  One constant for both crowns noise on the first and
hides a broken batch on the second.

So both are derived here, per machine (by the device's own identity, ``device.binary_sha256``), from
the stores' SOLO readings (a result run alone, or a batch of one) of the SAME program (``elf_sha256``):

* ``same_day`` -- the largest relative spread between two solo readings of one program taken the same
  UTC day: the run-to-run noise a margin must clear;
* ``cross_day`` -- the same between readings taken on different days: the drift a stale solo reference
  carries against today's batch.

A machine with fewer than two same-day solo readings of any one program has NO established noise:
that is flagged (``established: false`` and ``flag``), never filled in with a guess, and every caller
falls back to its declared bound and says so.

Every reading is a ``MEASURED`` result; a program carried from another job (``same_program_as``) is
that job's reading and is counted once.  Nothing here names a target or a machine.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import jobs as J

SCHEMA = "merlin.phase2.whole_model_measured.machine_noise.v1"
#: How many same-day solo readings of one program establish a machine's noise.
MIN_SAME_DAY_REPEATS = 2
NOT_ESTABLISHED = (
    f"fewer than {MIN_SAME_DAY_REPEATS} same-day solo repeats of an identical program on this machine: "
    "its noise is not established"
)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def reading_of(document: Mapping[str, Any] | None, *, source: str = "") -> dict[str, Any] | None:
    """One SOLO reading from a result document, or None (not measured, batched, carried, unidentified)."""
    if not isinstance(document, Mapping) or document.get("timing_status") != V.TIMING_MEASURED:
        return None
    if document.get("same_program_as") or document.get("measured_by"):
        return None
    batch = document.get("batch")
    if isinstance(batch, Mapping) and int(batch.get("size") or 1) > 1:
        return None
    cycles = (document.get("verdict") or {}).get("whole_window_cycles")
    elf = (document.get("build") or {}).get("elf_sha256")
    device = (document.get("device") or {}).get("binary_sha256")
    finished = str(document.get("finished_at") or "")
    if not isinstance(cycles, int) or cycles <= 0 or not elf or not device or len(finished) < 8:
        return None
    return {"elf_sha256": str(elf), "device": str(device), "day": finished[:8], "cycles": cycles, "source": source}


def result_paths(root: Path) -> list[Path]:
    """Every result a store holds, every attempt of every job included (a reading is a reading)."""
    root = Path(root)
    if not root.is_dir():
        return []
    paths = list(root.glob(f"*/{J.RESULT_FILE}"))
    paths += list(root.glob(f"*/{J.ATTEMPTS_DIR}/*/{J.RESULT_FILE}"))
    paths += [p for p in root.glob(f"*/*/{J.RESULT_FILE}") if p.parent.name.startswith(J.ARCHIVED_ATTEMPT_PREFIXES)]
    return sorted(set(paths))


def solo_readings(roots: Iterable[Path] = (), *, extra: Iterable[Path] = ()) -> list[dict[str, Any]]:
    """The solo readings of ``roots`` (stores) and of ``extra`` (single result files, e.g. a control's)."""
    seen: set[str] = set()
    readings = []
    for path in [*(p for root in roots for p in result_paths(Path(root))), *(Path(p) for p in extra)]:
        key = str(Path(path).resolve())
        if key in seen:
            continue
        seen.add(key)
        reading = reading_of(_read(path), source=str(path))
        if reading is not None:
            readings.append(reading)
    return readings


def _spread(a: int, b: int) -> float:
    return abs(int(a) - int(b)) / min(int(a), int(b))


def _summary(spreads: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(spreads)
    return {
        "pairs": len(ordered),
        "max": round(ordered[-1], 6) if ordered else None,
        "median": round(ordered[len(ordered) // 2], 6) if ordered else None,
    }


def machine_noise(readings: Sequence[Mapping[str, Any]], *, device: str | None) -> dict[str, Any]:
    """``device``'s noise from ``readings`` (see the module doc).  ``device`` None: nothing is known."""
    mine = [r for r in readings if device and r.get("device") == device]
    by_program: dict[str, list[Mapping[str, Any]]] = {}
    for reading in mine:
        by_program.setdefault(str(reading["elf_sha256"]), []).append(reading)
    same_day, cross_day = [], []
    for rows in by_program.values():
        for a, b in combinations(rows, 2):
            (same_day if a["day"] == b["day"] else cross_day).append(_spread(a["cycles"], b["cycles"]))
    established = len(same_day) >= 1  # one same-day pair is two same-day readings of one program
    return {
        "schema": SCHEMA,
        "device": device,
        "solo_readings": len(mine),
        "programs_repeated": sum(1 for rows in by_program.values() if len(rows) > 1),
        "same_day": _summary(same_day),
        "cross_day": _summary(cross_day),
        "established": established,
        "flag": None if established else NOT_ESTABLISHED,
        "basis": "relative spread |a-b|/min(a,b) between solo readings of one identical program (elf_sha256) on "
        "this device (binary_sha256)",
    }


def margin(noise: Mapping[str, Any] | None, *, floor: float, batched_vs_solo: float | None = None) -> dict[str, Any]:
    """The improvement margin on this machine: the largest of the floor, the same-day solo spread and the
    batched-vs-solo repeat spread, each where measured -- with which one set it."""
    candidates = {"floor": float(floor)}
    same_day = ((noise or {}).get("same_day") or {}).get("max")
    if isinstance(same_day, (int, float)):
        candidates["same_day_solo_spread"] = float(same_day)
    if isinstance(batched_vs_solo, (int, float)):
        candidates["batched_vs_solo_median"] = float(batched_vs_solo)
    basis = max(candidates, key=lambda k: candidates[k])
    return {
        "margin": candidates[basis],
        "basis": basis,
        "candidates": candidates,
        "established": bool((noise or {}).get("established")),
        "flag": (noise or {}).get("flag") if noise else NOT_ESTABLISHED,
    }


def drift_tolerance(noise: Mapping[str, Any] | None, *, declared: float, same_day: bool | None) -> dict[str, Any]:
    """The batch control's drift tolerance on this machine: the declared bound, widened to the machine's
    own measured spread over the same timescale (``same_day``: is the control's solo reading from the
    batch's own day) when that spread is established -- a noisy machine's own run-to-run movement is not
    a broken batch.  Never narrower than declared; unestablished noise leaves the declared bound, flagged."""
    declared = float(declared)
    if not noise or not noise.get("established"):
        return {
            "tolerance": declared,
            "basis": "declared",
            "declared": declared,
            "flag": (noise or {}).get("flag") or NOT_ESTABLISHED,
        }
    scale = "same_day" if same_day else "cross_day"
    observed = (noise.get(scale) or {}).get("max")
    if not isinstance(observed, (int, float)):
        scale, observed = "same_day", (noise.get("same_day") or {}).get("max")
    tolerance = max(declared, float(observed or 0.0))
    return {
        "tolerance": tolerance,
        "basis": f"{scale}_solo_spread" if tolerance > declared else "declared",
        "declared": declared,
        "observed": observed,
        "scale": scale,
        "flag": None,
    }


__all__ = [
    "MIN_SAME_DAY_REPEATS",
    "NOT_ESTABLISHED",
    "SCHEMA",
    "drift_tolerance",
    "machine_noise",
    "margin",
    "reading_of",
    "result_paths",
    "solo_readings",
]
