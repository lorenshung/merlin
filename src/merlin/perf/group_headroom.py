"""Per-group HEADROOM: measured cycles against the same-machine reference, and two DERIVED bounds --
never a suggested fix, only where a group sits relative to what the target's own facts say is
achievable.

Two bounds, both roofline-simple by design (this is a per-group DIAGNOSTIC, not the tiling-search
achievable bound :mod:`merlin.perf.derived_bound` computes for a whole workload search):

* **compute bound** -- ``macs / mesh_macs_per_cycle``, where the mesh's rate is the array geometry
  :func:`merlin.perf.derived_bound.machine_from_facts` already derives from the target's own RTL
  facts (never a nameplate, never a literal here).
* **data-movement bound** -- ``moved_bytes / dram_bytes_per_cycle``, the same machine's own derived
  DMA rate. ``moved_bytes`` is the group's OWN tensors (image, weight, result) -- never the im2col
  matrix a convolution's unit never materializes, which would over-charge a k>1 tap ninefold (see
  :class:`merlin.perf.schedule_proxy.Contraction`'s own note on the same trap).

Every rate is read off :class:`merlin.perf.derived_bound.Machine`; a term whose fact is UNKNOWN drops
out rather than being defaulted, and the group's bound is then reported over whatever terms resolved
-- never silently zero, never a guess. An op this module has no MAC-bearing formula for (a residual
add, a window mean) gets no bound at all: fabricating one would be worse than omitting it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decompose import is_unknown
from .derived_bound import Machine, _width_bits, machine_from_facts

__all__ = [
    "SCHEMA",
    "group_bound",
    "group_headroom",
    "macs_and_bytes",
    "machine_for",
]

SCHEMA = "merlin_group_headroom_v1"

#: Term names, matching :mod:`merlin.perf.derived_bound`'s vocabulary so a reader of both never sees
#: the same idea spelled two ways.
COMPUTE_TERM = "compute"
TRAFFIC_TERM = "dram_traffic"


def _dtype_bytes(dtype: Any) -> int | None:
    bits = _width_bits(dtype)
    return None if bits is None else -(-bits // 8)  # ceil: a sub-byte width still occupies one byte


def macs_and_bytes(op: Any, facts: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(macs, moved_bytes)`` for one group's shape facts, or ``None`` when ``op`` states no
    MAC-bearing contraction this estimate covers, or a needed extent/dtype is missing."""
    operand_bytes = _dtype_bytes(facts.get("operand_dtype"))
    output_bytes = _dtype_bytes(facts.get("output_dtype")) or operand_bytes
    if operand_bytes is None:
        return None
    if op == "matmul":
        try:
            m, k, n = int(facts["M"]), int(facts["K"]), int(facts["N"])
        except (KeyError, TypeError, ValueError):
            return None
        if min(m, k, n) <= 0:
            return None
        moved = m * k * operand_bytes + k * n * operand_bytes + m * n * (output_bytes or operand_bytes)
        return m * k * n, moved
    if op == "conv2d":
        try:
            co, ci = int(facts["N"]), int(facts["ci"])
            himg, wimg = int(facts["Himg"]), int(facts["Wimg"])
            kh, kw = int(facts["kh"]), int(facts["kw"])
            stride = list(facts.get("stride") or (1, 1))
            padding = list(facts.get("padding") or (0, 0, 0, 0))
            sh, sw = int(stride[0]), int(stride[1])
            pad_h, pad_w = int(padding[0]) + int(padding[2]), int(padding[1]) + int(padding[3])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if min(co, ci, himg, wimg, kh, kw, sh, sw) <= 0:
            return None
        hout = (himg - kh + pad_h) // sh + 1
        wout = (wimg - kw + pad_w) // sw + 1
        if hout <= 0 or wout <= 0:
            return None
        macs = hout * wout * co * ci * kh * kw
        # The unit's OWN tensors: the image and the weight, never the im2col matrix a native
        # convolution unit forms in its own address stream and no buffer ever holds whole.
        moved = himg * wimg * ci * operand_bytes + co * ci * kh * kw * operand_bytes
        moved += hout * wout * co * (output_bytes or operand_bytes)
        return macs, moved
    return None


def machine_for(target: str) -> Machine:
    """The target's derived machine facts, without consulting the circuit for a fill/drain depth --
    this diagnostic's two terms (mesh rate, DMA rate) do not need one, and deriving it is the
    expensive half of :func:`merlin.perf.derived_bound.machine_from_facts`."""
    return machine_from_facts(target, measure_fill=False)


def group_bound(macs: int, moved_bytes: int, machine: Machine) -> dict[str, Any]:
    """The two derived bounds for one group's ``(macs, moved_bytes)``, each dropped when its own
    machine fact is UNKNOWN rather than defaulted. ``bound_cycles`` is the max of what resolved (the
    binding constraint), or ``None`` when neither did."""
    rows, cols, muls = machine.array_rows, machine.array_cols, machine.muls_per_element
    mesh_macs_per_cycle = (
        int(rows) * int(cols) * int(muls) if not (is_unknown(rows) or is_unknown(cols) or is_unknown(muls)) else None
    )
    compute_bound_cycles = macs / mesh_macs_per_cycle if mesh_macs_per_cycle else None
    dram_rate = machine.dram_bytes_per_cycle
    dram_bytes_per_cycle = None if is_unknown(dram_rate) or not dram_rate else dram_rate
    traffic_bound_cycles = moved_bytes / dram_bytes_per_cycle if dram_bytes_per_cycle else None
    terms = {
        name: cycles
        for name, cycles in ((COMPUTE_TERM, compute_bound_cycles), (TRAFFIC_TERM, traffic_bound_cycles))
        if cycles is not None
    }
    return {
        "macs": macs,
        "moved_bytes": moved_bytes,
        "mesh_macs_per_cycle": mesh_macs_per_cycle,
        "dram_bytes_per_cycle": dram_bytes_per_cycle,
        "compute_bound_cycles": compute_bound_cycles,
        "traffic_bound_cycles": traffic_bound_cycles,
        "bound_cycles": max(terms.values()) if terms else None,
        "limiter": max(terms, key=terms.get) if terms else None,
    }


def group_headroom(
    *, op: Any, shape_facts: Mapping[str, Any] | None, ours_cycles: Any, reference_cycles: Any, machine: Machine
) -> dict[str, Any] | None:
    """One group's full headroom document, or ``None`` when its op/shape facts admit no MAC-bearing
    estimate. ``headroom_cycles`` is ``ours - reference`` (never derived, the measurement already
    states both); ``over_bound`` is ``ours / bound_cycles`` when the bound resolved."""
    if not isinstance(shape_facts, Mapping) or not isinstance(ours_cycles, int):
        return None
    counted = macs_and_bytes(op, shape_facts)
    if counted is None:
        return None
    macs, moved_bytes = counted
    bound = group_bound(macs, moved_bytes, machine)
    document: dict[str, Any] = {"schema": SCHEMA, **bound}
    if isinstance(reference_cycles, int):
        document["headroom_cycles"] = ours_cycles - reference_cycles
    if bound["bound_cycles"]:
        document["over_bound"] = round(ours_cycles / bound["bound_cycles"], 4)
    return document
