"""Operation/form coverage of explicit public capsule metadata.

Coverage is a declared capability obligation, not numerical qualification. The caller
selects the admitted public cohort; this module never discovers corpora or goldens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from merlin.common.paths import data_path
from merlin.targetgen import op_form_regimes as FR
from merlin.targetgen import semantic_families as sf

HOMES = ("accelerator", "host", "structural", "undetermined", "unknown")
ACCELERATOR, HOST, STRUCTURAL, UNDETERMINED, UNKNOWN = HOMES
EPILOGUE_AXIS = "epilogue"


@dataclass(frozen=True)
class OpHome:
    """Where one op runs on one target, and the evidence for the answer."""

    op: str
    home: str
    family: str | None = None
    reason: str = ""
    #: The :data:`eligibility.REFUSALS` tag behind a ``host`` verdict; ``None`` otherwise. ``reason``
    #: is for a reader, this is for a consumer that must act on the cause.
    refusal: str | None = None
    #: The hardware admits this op only IN COMPOSITION with a producer it declares (a ``composed_with``
    #: capability). Load-bearing for whoever writes the missing capsule: a standalone capsule for such
    #: an op is the WRONG capsule -- the eligibility oracle refuses one as a false fallback.
    fused_only: bool = False
    #: The declared dtypes the verdict was reached over. Empty means the target declares none for this
    #: family, which is itself the refusal.
    dtypes: tuple[str, ...] = ()
    #: Which merlin facilities can mint a capsule demanding this op. EMPTY IS A REAL FINDING: an op the
    #: hardware admits that no builder and no torch template can express is an obligation nobody can
    #: discharge, and reporting it as a plain gap would send an author to write an impossible capsule.
    expressible_by: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        out: dict = {"op": self.op, "home": self.home}
        if self.family:
            out["family"] = self.family
        if self.reason:
            out["reason"] = self.reason
        if self.refusal:
            out["refusal"] = self.refusal
        if self.fused_only:
            out["fused_only"] = True
        if self.dtypes:
            out["dtypes"] = list(self.dtypes)
        out["expressible_by"] = list(self.expressible_by)
        return out


@dataclass
class OpDemand:
    """How one graded cohort demands an op: as a capsule's payload, or as a capsule's epilogue stage."""

    as_operation: int = 0
    as_epilogue: int = 0
    capsules: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return self.as_operation + self.as_epilogue

    def to_dict(self) -> dict:
        return {
            "as_operation": self.as_operation,
            "as_epilogue": self.as_epilogue,
            "n": self.total,
            "capsules": list(self.capsules),
        }


def _declared_dtypes(family: str, cap_map: dict) -> tuple[str, ...]:
    """Every dtype the target declares for ``family``, or for all of a composite's primitives."""
    seen: set[str] = set()
    fams = [family] if family in cap_map else list(sf.primitives_of(family))
    for fam in fams:
        cap = cap_map.get(fam)
        if cap is not None:
            seen.update(str(d) for d in (cap.dtypes or ()))
    return tuple(sorted(seen))


def home_of(
    op: str,
    *,
    cap_map: dict,
    undetermined=None,
    expressible: dict[str, tuple[str, ...]] | None = None,
    family: str | None = None,
) -> OpHome:
    """Where ``op`` runs on the target whose declared capabilities are ``cap_map``.

    The decision, in order, and every step reads a declaration rather than a literal:

    1. the op is DECLARED structural (:data:`semantic_families.OP_STRUCTURAL`) -> ``structural``;
    2. it resolves to no family -> ``unknown``, which the gate treats as an error. Not ``host``:
       "the host carries it" is a placement claim, and nobody made one;
    3. the eligibility oracle admits it at some dtype the target declares for its family, standalone
       or composed with a producer the target also declares -> ``accelerator``;
    4. the oracle is UNDETERMINED -> ``undetermined``, which is the target's OWN declaration that no
       evidence source could decide the family. Reported as unmeasured and scored neither way -- calling
       it host would shrink the accelerator obligation to flatter it, and calling it a missing home
       would accuse a target of an omission it explicitly recorded;
    5. otherwise -> ``host``, carrying the oracle's own refusal tag.

    Step 3 asks the composed question with ``fused_with`` = every family the target declares, i.e. "is
    there ANY graph context in which this hardware admits this op?". That is the right question for a
    coverage obligation: a capability the manifest marks reachable only behind a contraction is still a
    capability, and an op admitted only that way needs a FUSED capsule -- which is why the verdict
    records ``fused_only`` instead of collapsing the two cases.

    ``family`` overrides step 2 for a caller that ALREADY resolved it from better evidence -- the
    importer's own ``prov.family`` tag, say, which is the frontend saying what the region is rather than
    this function inferring it from a name. Without it, a hint the name table does not carry
    (``abs``, ``mean``, ``rsqrt``) was refused as having no home while its tag said ``elementwise``
    all along: the "nobody translated the word" refusal, arriving through this function instead of
    through the taxonomy.
    """
    from merlin.targetgen import eligibility as EL

    why = sf.structural_reason(op)
    express = tuple((expressible or {}).get(op, ()))
    if why is not None:
        return OpHome(op=op, home=STRUCTURAL, reason=why, expressible_by=express)

    family = family or sf.from_op(op)
    if family is None:
        return OpHome(
            op=op,
            home=UNKNOWN,
            reason=(
                "no declared home: the op resolves to no semantic family and is not declared structural. "
                "Classify it in semantic_families._OP_FAMILY, or declare it in OP_STRUCTURAL with the "
                "reason it names no computation"
            ),
            expressible_by=express,
        )

    dtypes = _declared_dtypes(family, cap_map)
    if family in cap_map and not dtypes:
        return OpHome(op=op, home=UNKNOWN, family=family, reason="declared capability has no operand formats")
    # No declared dtype for the family is not a missing probe -- an empty dtype declaration NARROWS
    # (eligibility.empty_declaration_is_narrowing("dtypes")), so asking once with None still reaches the
    # oracle's own refusal and the answer carries its tag.
    probes: tuple[str | None, ...] = dtypes or (None,)
    producers = frozenset(str(f) for f in cap_map)
    verdict = None
    fused_only = False
    for dtype in probes:
        # The family is passed explicitly, not left to the oracle's own name resolution: this function
        # may have been handed one from better evidence, and `RegionDescriptor.resolved_family` returns
        # an already-canonical family untouched, so the two agree where both can answer.
        region = EL.RegionDescriptor(op=op, family=family, in_dtype=dtype)
        standalone = EL.is_eligible(region, cap_map, undetermined=undetermined)
        if standalone.eligible:
            return OpHome(
                op=op, home=ACCELERATOR, family=family, reason=standalone.reason, dtypes=dtypes, expressible_by=express
            )
        composed = EL.is_eligible(region, cap_map, undetermined=undetermined, fused_with=producers)
        if composed.eligible:
            return OpHome(
                op=op,
                home=ACCELERATOR,
                family=family,
                reason=f"admitted only in composition with {sorted(producers)}: {composed.reason}",
                fused_only=True,
                dtypes=dtypes,
                expressible_by=express,
            )
        # Keep the FIRST refusal, and prefer an undetermined one: undetermined outranks a refusal
        # because it is the only verdict that must not be scored, and one dtype reaching it is enough.
        if composed.undetermined or standalone.undetermined:
            verdict = composed if composed.undetermined else standalone
            fused_only = False
            break
        if verdict is None:
            verdict = standalone
            fused_only = standalone.refusal == "fused_only"
    if verdict is None:  # unreachable while `probes` is non-empty; fail closed rather than assume
        return OpHome(
            op=op, home=UNKNOWN, family=family, reason="no eligibility verdict was reached", expressible_by=express
        )
    if verdict.undetermined:
        return OpHome(
            op=op,
            home=UNDETERMINED,
            family=family,
            reason=verdict.reason,
            refusal=verdict.refusal,
            dtypes=dtypes,
            expressible_by=express,
        )
    return OpHome(
        op=op,
        home=HOST,
        family=family,
        reason=verdict.reason,
        refusal=verdict.refusal,
        fused_only=fused_only,
        dtypes=dtypes,
        expressible_by=express,
    )


@dataclass(frozen=True)
class OpCell:
    """One coverage requirement at OP resolution: this op, on this axis, in this regime.

    PER AXIS, NOT A FULL CROSS PRODUCT, and that is a deliberate narrowing. Crossing six form axes
    against each other would demand thousands of cells no real program presents -- the same mistake
    ``conformance.uncovered`` records for its composition axis, which is reported beside the cells and
    never crossed with them. A capsule covers every axis it can be placed on at once, so the axes are
    independent questions about the same corpus rather than a grid to fill.
    """

    op: str
    axis: str
    regime: str

    def key(self) -> str:
        return f"{self.op}/{self.axis}={self.regime}"


def frontend_opset() -> tuple[str, ...]:
    """Read the installed capsule vocabulary; malformed/empty universes refuse."""
    path = data_path("contract", "schemas", "capsule.schema.json")
    doc = json.loads(path.read_text())
    values = doc["properties"]["operation"]["properties"]["op"]["enum"]
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("operation vocabulary must be a nonempty string enum")
    if len(set(values)) != len(values):
        raise ValueError("operation vocabulary contains duplicates")
    return tuple(sorted(values))


def homes_for_capabilities(cap_map: dict, *, ops=None, undetermined=()) -> dict[str, OpHome]:
    """Classify declarations without importing a compiler or inferring support."""
    if not isinstance(cap_map, dict):
        raise ValueError("explicit capability map required")
    return {
        op: home_of(op, cap_map=cap_map, undetermined=undetermined) for op in (frontend_opset() if ops is None else ops)
    }


def required_op_cells(
    homes: dict[str, OpHome], bounds: FR.FormBounds, *, epilogue_stages=(), windowed_ops=()
) -> dict[str, dict]:
    """Union of independent axis obligations; never a Cartesian product of axes.

    Windowed operations and epilogue stages must come from the selected source /
    readout contract. No inference from names, measured outputs or unseen members.
    """
    cells = {}
    for op, home in sorted(homes.items()):
        if home.home != ACCELERATOR or not home.family:
            continue
        regimes = [(r.axis, r.regime) for r in FR.regimes_for_family(home.family, bounds)]
        if op in windowed_ops:
            regimes += [(axis, value) for axis in FR.WINDOW_AXES for value in FR.AXIS_REGIMES[axis]]
        if "contraction" in sf.primitives_of(home.family):
            regimes += [(EPILOGUE_AXIS, value) for value in epilogue_stages]
        for axis, regime in regimes:
            cell = OpCell(op, axis, regime)
            cells[cell.key()] = {"op": op, "family": home.family, "axis": axis, "regime": regime}
    return cells


def public_cohort(capsules, *, excluded=()) -> tuple[dict, ...]:
    """Require explicitly public metadata and unique names, before exclusions.

    Callers must use public readers: passing hidden metadata here is a refusal,
    not permission to read hidden answers. No source pool is silently graded.
    """
    selected, names = [], set()
    excluded = set(excluded)
    for cap in capsules:
        if not isinstance(cap, dict) or cap.get("label") != "public":
            raise ValueError("coverage accepts explicitly public capsule metadata only")
        name = cap.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("capsule names must be nonempty and unique")
        operation = cap.get("operation")
        if not isinstance(operation, dict) or not isinstance(operation.get("op"), str) or not operation["op"]:
            raise ValueError("capsule operation must be explicit")
        stages = (operation.get("attributes") or {}).get("epilogue", ())
        if not isinstance(stages, (list, tuple)) or any(not isinstance(v, str) or not v for v in stages):
            raise ValueError("epilogue must be a sequence of stage names")
        names.add(name)
        if name not in excluded:
            selected.append(cap)
    if excluded - names:
        raise ValueError("exclusions name absent public members")
    return tuple(selected)


def demanded_ops(capsules, *, excluded=()) -> dict[str, OpDemand]:
    demand = {}
    for cap in public_cohort(capsules, excluded=excluded):
        operation = cap.get("operation") or {}
        stages = (operation.get("attributes") or {}).get("epilogue") or ()
        for op, epilogue in [(operation.get("op"), False)] + [(s, True) for s in stages]:
            if not isinstance(op, str) or not op:
                raise ValueError("operation and epilogue names must be explicit")
            record = demand.setdefault(op, OpDemand())
            record.as_epilogue += int(epilogue)
            record.as_operation += int(not epilogue)
            if cap["name"] not in record.capsules:
                record.capsules += (cap["name"],)
    return demand


def capsule_cells(cap: dict, bounds: FR.FormBounds) -> set[str]:
    operation = cap.get("operation") or {}
    op = operation.get("op")
    axes = FR.classify_capsule(cap, bounds)
    cells = {OpCell(op, axis, regime).key() for axis, regime in axes.items() if regime != FR.UNDERIVABLE}
    cells.update(
        OpCell(op, EPILOGUE_AXIS, stage).key() for stage in (operation.get("attributes") or {}).get("epilogue") or ()
    )
    return cells


def cell_coverage(
    homes: dict[str, OpHome],
    bounds: FR.FormBounds,
    capsules,
    *,
    capability_map: dict,
    excluded=(),
    epilogue_stages=(),
    windowed_ops=(),
) -> dict:
    selected = public_cohort(capsules, excluded=excluded)
    cells = required_op_cells(homes, bounds, epilogue_stages=epilogue_stages, windowed_ops=windowed_ops)
    cover = {key: [] for key in cells}
    points = {}
    refused = {}
    for cap in selected:
        refusal = capsule_admission(cap, capability_map, bounds)
        if refusal is not None:
            refused[cap["name"]] = refusal
            continue
        landed = capsule_cells(cap, bounds)
        points[cap["name"]] = sorted(landed)
        for key in landed & cells.keys():
            cover[key].append(cap["name"])
    # Duplicate cell signatures are diagnostics only; they never authorize deleting
    # whole-model or independently meaningful numerical tests.
    signatures = {}
    for name, point in points.items():
        if point:
            signatures.setdefault(tuple(point), []).append(name)
    return {
        "cells": cells,
        "refused_members": refused,
        "cover": cover,
        "under": sorted(k for k, v in cover.items() if not v),
        "n_required_cells": len(cells),
        "points": points,
        "duplicate_signatures": [v for v in signatures.values() if len(v) > 1],
        "unknown_homes": sorted(op for op, h in homes.items() if h.home == UNKNOWN),
        "undetermined_homes": sorted(op for op, h in homes.items() if h.home == UNDETERMINED),
        "qualification": "metadata_coverage_only",
    }


def problems(report: dict, *, ratchet=()) -> list[str]:
    """Refuse uncovered/unknown cells and stale debt. Never bootstrap a ratchet."""
    actual = {f"cell:{k}" for k in report["under"]}
    actual.update(f"home:{op}" for op in report["unknown_homes"])
    actual.update(f"member:{name}" for name in report.get("refused_members", {}))
    debt = set(ratchet)
    return sorted(actual - debt) + [f"stale:{key}" for key in sorted(debt - actual)]


def capsule_admission(cap: dict, cap_map: dict, bounds: FR.FormBounds) -> str | None:
    """Check built metadata against the selected capability, before coverage credit.

    A general family is not proof of a particular shape or conversion. Unknown
    layout/form or compound semantics refuse here rather than inventing a route.
    """
    from merlin.targetgen.eligibility import RegionDescriptor, is_eligible

    operation = cap.get("operation") or {}
    op = operation.get("op")
    family = sf.from_op(op)
    if family not in sf.PRIMITIVES:
        return "compound or structural region needs an explicit lowering admission"
    operands = [t for t in cap.get("inputs", ()) if t.get("role") in ("input", "weight")]
    if not operands or any(not t.get("dtype") or not t.get("shape") for t in operands):
        return "operand shape/format unavailable"
    attrs = operation.get("attributes") or {}
    m, k, n = FR._mkn(cap)
    capability = cap_map.get(family)
    if capability is None:
        return "no declared primitive capability"
    if capability.layouts and not attrs.get("layout"):
        return "layout evidence unavailable"
    if not capability.arbitrary_mnk:
        if not all((m, k, n, bounds.tile_edge)):
            return "fixed shape capability needs exact matrix geometry"
        if any(d % bounds.tile_edge for d in (m, k, n)):
            return "shape exceeds declared whole-tile capability"
    if family == "contraction" and not all((m, k, n)):
        return "contraction indexing not proved"
    lhs = next((t for t in operands if t.get("name") == attrs.get("lhs")), None)
    lhs = lhs or next((t for t in operands if t.get("role") == "input"), None)
    rhs = next((t for t in operands if t.get("name") == attrs.get("weight")), None)
    rhs = rhs or next((t for t in operands if t.get("role") == "weight"), None)
    if lhs is None or (family == "contraction" and rhs is None):
        return "operand roles unavailable"
    region = RegionDescriptor(
        op=op,
        family=family,
        in_dtype=lhs["dtype"],
        weight_dtype=rhs["dtype"] if family == "contraction" else None,
        rank=len(lhs["shape"]),
        m=m,
        k=k,
        n=n,
        layout=attrs.get("layout"),
        form=sf.operation_form(op),
        out_dtype=attrs.get("dtype"),
    )
    verdict = is_eligible(region, cap_map)
    return None if verdict.eligible else verdict.reason
