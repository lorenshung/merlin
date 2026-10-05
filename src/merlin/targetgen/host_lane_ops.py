"""The host operations real captures contain, per ``(family, dtype)``, with their frontend identity.

The host-lane requirement says WHICH ``(family, dtype)`` work a target must leave on the host. A probe
capsule for that pair has to be built from an operation that work actually consists of: a probe
written with a stock op of the same family (``gelu`` for an elementwise pair, ``reduce_sum`` for a
reduction) exercises a program no captured model contains, and it falls outside reviewed host
declarations that name only the operations the captures carry. So the candidates are read from the
captures themselves -- each region's model2MLIR op (``prov.op``, the same vocabulary as Merlin's op
names) and its frontend operator (``prov.aten``) -- and a probe is chosen only from those.

Nothing here names a target or an operation; it counts what the captured regions say they are.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def pair_key(family: str, dtype: str) -> str:
    return f"{family}/{dtype}"


def observed_host_ops(captures: Mapping[str, str | Path], pairs: Iterable[tuple[str, str]]) -> dict[str, list]:
    """``"family/dtype" -> [{op, frontend_op, n_regions}]`` for the requested pairs, most frequent first.

    A region whose provenance names no op is not counted: an anonymous region cannot be reproduced by
    a probe, and guessing its op would invent the very stock program this exists to avoid. An
    unreadable capture contributes nothing (it is reported by the pair census that reads it too).
    """
    from merlin.targetgen import model_coverage as mc
    from merlin.targetgen.conformance import capsule_dtype

    wanted = {(str(family), str(dtype)) for family, dtype in pairs}
    counts: dict[tuple[str, str], Counter] = {pair: Counter() for pair in wanted}
    for _label, path in sorted((captures or {}).items()):
        try:
            sources = list(mc.region_sources(mc.load_module(path)))
        except Exception:  # noqa: BLE001 -- an unreadable capture is evidence neither way
            continue
        for family, dtype, op, frontend in sources:
            if not family or not dtype or not op:
                continue
            try:
                dtype = capsule_dtype(str(dtype))
            except Exception:  # noqa: BLE001 -- an unmappable token stays as spelled
                dtype = str(dtype)
            key = (str(family), str(dtype))
            if key in counts:
                counts[key][(str(op), frontend)] += 1
    return {
        pair_key(*pair): [
            {"op": op, "frontend_op": frontend, "n_regions": n}
            for (op, frontend), n in sorted(counter.items(), key=lambda item: (-item[1], item[0][0]))
        ]
        for pair, counter in sorted(counts.items())
    }


def choose(observed: Iterable[Mapping[str, Any]] | None, writable: Iterable[str]) -> dict | None:
    """The most frequent observed host op a writer can express, or ``None`` when none can.

    ``None`` is reported by the caller as an uncovered pair; substituting a writable op that the
    captures do not contain is exactly the failure this module replaces.
    """
    allowed = set(writable)
    for row in observed or ():
        if row.get("op") in allowed:
            return dict(row)
    return None
