"""Tier A of the phase-2 fast feedback: the whole-model program on the FUNCTIONAL simulator, as a screen.

    screen = structure_screen(elf, groups=..., target=..., out=...)   # ~1.5 min for a ResNet
    calibration = fit_calibration(collect_pairs([store]))             # refit from every measured pair

The functional simulator runs the SAME ELF the board runs, in about a minute and a half, and every
group's own local check runs with it. So it answers "is every group still correct" at once. It also
prints a per-group cycle count, and that number is where this module is careful: measured on 142 group
pairs (the same ELFs on the functional simulator and on the board), the functional simulator ranks
whole models correctly but is BLIND TO DATA MOVEMENT -- the board/simulator ratio is ~4x for a
convolution, ~6x for a matmul (up to 122x), ~89x for a residual add, and of the groups whose board
cycles moved by more than 5% it agreed on direction for barely a third.

So the screen's cycles are labelled a STRUCTURE SCREEN and never feed an objective:

* **Correctness** per group is the program's own local line (``GM_LOCAL`` for an exact group,
  ``GM_BOUND`` for a tolerance group); a group with neither is ``absent``, and a console missing any
  expected group line is refused outright.
* **Calibration** per kind is REFIT from every (functional, board) pair the measurement store has
  accumulated -- never a constant: :func:`collect_pairs` walks the store for jobs that carry both a
  board result and a screen of the same ELF, and :func:`fit_calibration` states, per kind and per
  group, the median board/simulator ratio, its spread, and the pair count.
* **``spike_blind``** marks a group whose kind (or whose own measured ratio) sits above
  ``blind_ratio`` -- a memory-bound group, whose functional cycles say nothing about its board cycles.
  No per-group board estimate is stated: a group's own ratio depends on what it spends its time
  on (an operand gather on the core runs at ~1.3x, the kernel beside it at ~5x), so the class's
  ratio and spread are given instead, for a reader to weigh.

Nothing here names a target: the simulator is reached through the target's backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "LABEL",
    "SCHEMA",
    "ScreenRefusal",
    "collect_pairs",
    "fit_calibration",
    "pairs_from_consoles",
    "routes_of",
    "screen_console",
    "screen_dir",
    "structure_screen",
]

SCHEMA = "whole_model_structure_screen_v1"
LABEL = "STRUCTURE SCREEN (functional simulator): correctness and compute structure only; never an objective"
SCREEN_FILE = "structure_screen.json"
#: Above this board/simulator ratio a kind's (or group's) functional cycles are blind to its board cost.
BLIND_RATIO = 20.0


class ScreenRefusal(ValueError):
    """The console cannot be read as a whole-model screen, and the message says why."""


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------------------------------- the screen


def screen_console(
    text: str,
    groups: Mapping[str, str],
    *,
    templates: Mapping[str, str] | None = None,
    calibration: Mapping[str, Any] | None = None,
    blind_ratio: float = BLIND_RATIO,
    routes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read a functional-simulator console against the groups the build states (``{group: compare}``).

    ``routes`` (``{group: who answered it}``, :func:`routes_of`) selects the calibration of the
    group's kind UNDER ITS ROUTE when the fit has one; the kind alone is the fallback.

    Refused (``status: refused``) when any expected group line is missing: a partial console has no
    per-group reading. Otherwise every group gets its simulator cycles, its local correctness and, from
    ``calibration``, whether the simulator is blind to it.
    """
    from . import whole_model_verdict as V

    try:
        parsed = V.parse_log(text, templates)
    except V.VerdictRefusal as why:
        return {"schema": SCHEMA, "label": LABEL, "status": "refused", "refusal": str(why)}
    expected, printed = set(map(str, groups)), set(parsed.groups)
    if printed != expected:
        return {
            "schema": SCHEMA,
            "label": LABEL,
            "status": "refused",
            "refusal": (
                f"the console printed {len(printed & expected)} of the {len(expected)} expected group lines "
                f"(missing {sorted(expected - printed, key=int)[:12]}); a partial run is not read"
            ),
        }
    kinds = (calibration or {}).get("kinds") or {}
    per_group = (calibration or {}).get("groups") or {}
    rows = []
    for group in sorted(expected, key=int):
        line = parsed.groups[group]
        if groups[group] in ("bounded", "bounded_int"):
            got = parsed.bounds.get(group)
            state = "absent" if got is None else ("correct" if got[1] == 0 else "wrong")
            detail = None if got is None else {"max_abs": got[0], "over": got[1], "bound": got[2]}
        else:
            got = parsed.local.get(group)
            state = "absent" if got is None else ("correct" if got[0] == 0 else "wrong")
            detail = None if got is None else {"mismatches": got[0], "of": got[1]}
        row: dict[str, Any] = {"group": int(group), "kind": line.kind, "spike_cycles": line.cycles, "local": state}
        if routes:
            row["on"] = routes.get(group)
        if detail and state != "correct":
            row["failure"] = detail
        on = (routes or {}).get(group)
        classes = (calibration or {}).get("classes") or {}
        fit_kind = classes.get(_class(line.kind, on)) if on else None
        basis = f"{line.kind!r} answered by {on}" if fit_kind else f"kind {line.kind!r}"
        fit_kind = fit_kind or kinds.get(line.kind)
        fit_group = (per_group.get(_class(group, on)) if on else None) or per_group.get(group)
        reasons = []
        if fit_kind and fit_kind.get("spike_blind"):
            reasons.append(f"{basis}: board/simulator median {fit_kind['median_ratio']}x")
        if fit_group and fit_group["median_ratio"] > blind_ratio:
            reasons.append(f"this group: board/simulator median {fit_group['median_ratio']}x")
        row["spike_blind"] = bool(reasons) if (fit_kind or fit_group) else None
        if reasons:
            row["blind_because"] = reasons
        elif fit_kind:
            # The class's ratio and spread, NOT a per-group estimate: measured on the same program, a
            # group whose time is an operand gather on the core sits near 1.3x while its class sits
            # near 4.6x, so a multiplied-out "board estimate" would be a number nobody measured.
            row["class_ratio"] = {k: fit_kind[k] for k in ("median_ratio", "p10", "p90", "n")}
        rows.append(row)
    wrong = [r["group"] for r in rows if r["local"] != "correct"]
    return {
        "schema": SCHEMA,
        "label": LABEL,
        "feeds_objective": False,
        "status": "screened",
        "all_groups_correct": not wrong,
        "groups_not_correct": wrong,
        "whole_window_spike_cycles": parsed.whole_window_cycles,
        "argmax": list(parsed.argmax) if parsed.argmax else None,
        # A model graded on its output tensor states ``(within, of)`` instead of a class.
        "output": list(parsed.output) if parsed.output else None,
        # The whole output's digest and every dispatch's result digest: what a reference arm compares.
        "output_digest": list(parsed.output_digest) if parsed.output_digest else None,
        "words": {g: list(v) for g, v in parsed.words.items()},
        "calibration": {
            "pairs": (calibration or {}).get("pairs"),
            "sources": (calibration or {}).get("sources"),
            "blind_ratio": blind_ratio,
        }
        if calibration
        else None,
        "groups": rows,
    }


def structure_screen(
    elf: str | Path,
    *,
    groups: Mapping[str, str],
    target: str,
    out: str | Path,
    elf_sha256: str | None = None,
    templates: Mapping[str, str] | None = None,
    simulator: str = "spike",
    timeout: int = 4 * 3600,
    calibration: Mapping[str, Any] | None = None,
    routes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``elf`` on the target's functional simulator and screen it (see :func:`screen_console`).

    The console and the screen are written to ``out`` (``console.txt``, ``structure_screen.json``); put
    ``out`` beside a board result of the same ELF and :func:`collect_pairs` will learn from it.
    """
    from merlin.runtime.backends import base as backends

    elf, out = Path(elf), Path(out)
    observed = _sha256(elf)
    if elf_sha256 and observed != elf_sha256:
        raise ScreenRefusal(f"{elf} is {observed[:12]}, not the {elf_sha256[:12]} named for the screen")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    console = backends.get_backend(target).run_elf(elf, simulator=simulator, timeout=timeout)
    wall = round(time.monotonic() - started, 1)
    (out / "console.txt").write_text(console, encoding="utf-8")
    screen = screen_console(console, groups, templates=templates, calibration=calibration, routes=routes)
    screen.update({"elf_sha256": observed, "simulator": simulator, "wall_s": wall, "target": target})
    (out / SCREEN_FILE).write_text(json.dumps(screen, indent=1) + "\n", encoding="utf-8")
    return screen


# ------------------------------------------------------------------------------------- calibration


def _group_cycles(text: str, templates: Mapping[str, str] | None = None) -> dict[str, tuple[str, int]]:
    from . import whole_model_verdict as V

    parsed = V.parse_log(text, templates)
    return {g: (line.kind, line.cycles) for g, line in parsed.groups.items()}


#: Older service records spell the package's side ``submission``; it is the same side.
_SIDE = {"submission": "package"}


def routes_of(record: Mapping[str, Any]) -> dict[str, str]:
    """``{group: route}`` from a build record (the whole-model build's or the service's).

    A route is who answered the group and, for a kernel, the command shape it lowered to
    (``package:resident_matmul``): a kernel that hands its loop to the hardware sequencer costs the
    functional simulator almost nothing, so two kernels of one kind can differ by 100x in how blind
    the simulator is to them, and the shape is what tells them apart.
    """
    rows = (record.get("attribution") or {}).get("per_group") or record.get("groups") or ()
    routes = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("group") is None:
            continue
        side = _SIDE.get(str(row.get("on")), str(row.get("on")))
        shape = row.get("lowering") or row.get("shape")
        routes[str(row["group"])] = f"{side}:{shape}" if shape and side == "package" else side
    return routes


def pairs_from_consoles(
    spike_console: str,
    board_console: str,
    *,
    source: str,
    templates: Mapping[str, str] | None = None,
    routes: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-group (kind, route, simulator cycles, board cycles) for two consoles of the SAME ELF."""
    spike, board = _group_cycles(spike_console, templates), _group_cycles(board_console, templates)
    if set(spike) != set(board):
        raise ScreenRefusal(f"{source}: the two consoles print different group sets")
    return [
        {
            "source": source,
            "group": g,
            "kind": spike[g][0],
            "on": (routes or {}).get(g),
            "spike": spike[g][1],
            "board": board[g][1],
        }
        for g in sorted(spike, key=int)
        if spike[g][1] > 0 and board[g][1] > 0
    ]


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def screen_dir(store_base: str | Path, elf_sha256: str) -> Path:
    """Where the store keeps the screen of one ELF: by the program's content, beside the jobs."""
    return Path(store_base) / "_structure_screens" / str(elf_sha256)


def collect_pairs(store_bases: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every (simulator, board) group pair the measurement store holds for one ELF measured on both.

    A board result (a job's ``result.json`` whose device is on the board rung and whose run completed)
    counts when the store also holds a screen of the ELF it names, under
    ``<base>/_structure_screens/<elf sha256>/`` (:func:`screen_dir`) -- keyed by the program's bytes, so
    a screen is written once per ELF and never into another run's job directory. Each ELF counts once,
    however many jobs measured it.
    """
    seen: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for base in store_bases:
        for result_path in sorted(Path(base).glob("*/*/result.json")):
            job = result_path.parent
            elf_named = str((_read(result_path).get("build") or {}).get("elf_sha256") or "")
            if not elf_named:
                continue
            screen_path = screen_dir(base, elf_named) / SCREEN_FILE
            if not screen_path.is_file():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                screen = json.loads(screen_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            elf = (result.get("build") or {}).get("elf_sha256")
            run = result.get("run") or {}
            if not elf or elf != screen.get("elf_sha256") or elf in seen or not run.get("completed"):
                continue
            if (result.get("device") or {}).get("rung") != "fpga_firesim":
                continue
            uart = Path(str(run.get("uart_log") or ""))
            console = screen_path.parent / "console.txt"
            if not uart.is_file() or not console.is_file():
                continue
            try:
                pairs += pairs_from_consoles(
                    console.read_text(encoding="utf-8", errors="replace"),
                    uart.read_text(encoding="utf-8", errors="replace"),
                    source=f"{job.parent.name}/{job.name[:16]} elf {elf[:12]}",
                    routes=routes_of(result.get("build") or {}),
                )
            except (ScreenRefusal, ValueError):
                continue
            seen.add(elf)
    return pairs


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fit_calibration(pairs: Sequence[Mapping[str, Any]], *, blind_ratio: float = BLIND_RATIO) -> dict[str, Any]:
    """Per kind and per group: the board/simulator ratio's median, spread and pair count; the blind flag.

    Also the log-log correlation of the two instruments over every pair, which says how much a
    simulator cycle count tells about a board one at all.
    """
    by_kind: dict[str, list[float]] = {}
    by_class: dict[str, list[float]] = {}
    by_group: dict[str, list[float]] = {}
    xs, ys = [], []
    for pair in pairs:
        ratio = float(pair["board"]) / float(pair["spike"])
        by_kind.setdefault(str(pair["kind"]), []).append(ratio)
        if pair.get("on"):
            by_class.setdefault(_class(pair["kind"], pair["on"]), []).append(ratio)
            by_group.setdefault(_class(pair["group"], pair["on"]), []).append(ratio)
        by_group.setdefault(str(pair["group"]), []).append(ratio)
        xs.append(math.log(float(pair["spike"])))
        ys.append(math.log(float(pair["board"])))

    def summary(ratios: list[float]) -> dict[str, Any]:
        median = statistics.median(ratios)
        return {
            "n": len(ratios),
            "median_ratio": round(median, 3),
            "p10": round(_quantile(ratios, 0.1), 3),
            "p90": round(_quantile(ratios, 0.9), 3),
            "max_ratio": round(max(ratios), 3),
            "spike_blind": median > blind_ratio,
        }

    correlation = None
    if len(xs) >= 3:
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
        sxx = sum((a - mx) ** 2 for a in xs) ** 0.5
        syy = sum((b - my) ** 2 for b in ys) ** 0.5
        correlation = round(sxy / (sxx * syy), 4) if sxx and syy else None
    return {
        "pairs": len(pairs),
        "sources": sorted({str(p.get("source")) for p in pairs}),
        "blind_ratio": blind_ratio,
        "log_log_r": correlation,
        "kinds": {kind: summary(r) for kind, r in sorted(by_kind.items())},
        # THE ROUTE IS PART OF THE KEY. A group the library answers with a sequenced loop is almost
        # free on the functional simulator (the sequencer's work is not an instruction it counts),
        # so one kind can sit at 5x under one route and at 2000x under another; a per-kind ratio
        # averaged over both describes neither.
        "classes": {name: summary(r) for name, r in sorted(by_class.items())},
        "groups": {group: summary(r) for group, r in sorted(by_group.items())},
    }


def _class(kind_or_group: Any, on: Any) -> str:
    return f"{kind_or_group}@{on}"
