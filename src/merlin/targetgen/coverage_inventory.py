"""One row per required coverage item: what a target owes, who witnesses it, and whether that held.

The conformance requirement (:mod:`merlin.targetgen.conformance`) and its gap readers answer a
per-AXIS question -- how many cells, composition shapes, host-lane pairs ... no capsule exercises. A
Phase 1 coverage claim needs the per-ITEM join those readers leave implicit: for each required item,
which placement the target's own manifest demands for it, which capsules witness it, what those
capsules were observed to do when graded, and therefore whether the item is covered. This module is
that join. It composes existing derivations and never restates them:

* the requirement items are read from the target's conformance spec (``cells``, ``host_lane``,
  ``host_only``, ``composition``, ``epilogue``, ``shape_geometry``);
* the REQUIRED placement of each item comes from the eligibility oracle
  (:func:`merlin.targetgen.eligibility.is_eligible`) over the target's capability contract -- never
  from the spec's own say-so, so a spec that has drifted from its manifest surfaces as ``UNKNOWN``
  instead of being trusted;
* witnesses come from the readers the conformance gate already uses
  (:func:`merlin.targetgen.contract.materialize.capsule_cell_rows`,
  :mod:`merlin.targetgen.boundary`, the conformance gap readers), so "this capsule exercises that item"
  means exactly what it means in the gate;
* observed placement and verdict come from per-capsule grading results when they are supplied.

FAIL CLOSED. An item is ``covered`` only when its required placement is decided, at least one
witness was graded ``pass``, and that witness's observed placement matches the requirement. An
``UNKNOWN`` placement, a witness that was ``not_run``, or a pass whose placement was never measured is
NEVER covered -- each is reported as ``unknown`` with its reason, because a coverage figure that counts
"nobody looked" as success is the failure every gate in this repo exists to prevent.

DEVELOPMENT WORKLOADS ONLY. The held-out claim models (``merlin/contract/claim_models.yaml``, read only
through :mod:`merlin.targetgen.claim_models`) may not decide what a coverage suite contains. Supplying
one as a workload, or supplying grading results for a capsule cut from one, raises
:class:`HoldoutInputError`. Held-out evidence already baked into a tracked spec or corpus is
SEPARATED, not trusted: it is listed on each row and on the inventory, and an item whose requirement
rests only on held-out evidence is ``unknown``.

Target-agnostic by construction: the target is a parameter and every fact is read from its contract,
spec, corpus and results. The pure core is :func:`build_inventory` (inputs in, rows out);
:func:`collect_witnesses` and :func:`inventory_for_target` are the thin readers around it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes
from merlin.perf import lowering_coverage as LC
from merlin.targetgen import boundary as BD
from merlin.targetgen import claim_models as CM
from merlin.targetgen import eligibility as EL
from merlin.targetgen import semantic_families as SF

SCHEMA = "merlin.targetgen.coverage_inventory.v1"

# --- vocabularies --------------------------------------------------------------------------------

#: Requirement axes read from the conformance spec, in report order.
KINDS: tuple[str, ...] = ("cell", "host_lane", "host_only", "composition", "epilogue", "shape_geometry")

#: Required placement. ``MIXED`` is a composition that crosses the host/accelerator seam by definition.
MUST_ACCELERATE = "must_accelerate"
HOST = "host"
MIXED = "mixed"
UNKNOWN = "UNKNOWN"
REQUIRED_PLACEMENTS: tuple[str, ...] = (MUST_ACCELERATE, HOST, MIXED, UNKNOWN)

#: Observed placement of one graded witness. ``SILENT_FALLBACK`` is an eligible region the compile left
#: on the host without a reason; it never satisfies any requirement.
OBSERVED_ACCELERATOR = LC.ACCELERATOR
OBSERVED_HOST = LC.HOST
OBSERVED_MIXED = MIXED
SILENT_FALLBACK = "silent_host_fallback"
NOT_RUN = "not_run"

#: Which observed placement satisfies which required placement. UNKNOWN satisfies nothing.
_SATISFIES: dict[str, str] = {
    MUST_ACCELERATE: OBSERVED_ACCELERATOR,
    HOST: OBSERVED_HOST,
    MIXED: OBSERVED_MIXED,
}

COVERED = "covered"
MISSING = "missing"
STATUS_UNKNOWN = "unknown"
STATUSES: tuple[str, ...] = (COVERED, MISSING, STATUS_UNKNOWN)

#: The grading verdict that is a pass. Every other result status -- and its absence -- is not.
_PASS = "pass"

#: Eligibility refusals that do not DECIDE the question; anything else is a definitive "host".
_UNDECIDED_REFUSALS = frozenset({"unrecognized_family", "undetermined_family", "scale_granularity_unknown"})


class HoldoutInputError(ValueError):
    """A held-out claim model was supplied as an input to a development-only inventory."""


# --- rows ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WitnessObservation:
    """One witnessing capsule and what grading observed of it (``not_run`` when no result exists)."""

    capsule: str
    verdict: str = NOT_RUN
    placement: str = UNKNOWN

    def to_json(self) -> dict:
        return {"capsule": self.capsule, "verdict": self.verdict, "placement": self.placement}


@dataclass(frozen=True)
class CoverageRow:
    """One required coverage item and its join."""

    kind: str
    key: str
    signature_identity: str
    family: str | None
    dtype: str | None
    shape: str | None
    boundary: str
    required_placement: str
    placement_basis: str
    development_evidence: tuple[str, ...]
    holdout_evidence_excluded: tuple[str, ...]
    witnesses: tuple[str, ...]
    observations: tuple[WitnessObservation, ...]
    verdict: str
    observed_placement: str
    status: str
    reasons: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "key": self.key,
            "signature_identity": self.signature_identity,
            "family": self.family,
            "dtype": self.dtype,
            "shape": self.shape,
            "boundary": self.boundary,
            "required_placement": self.required_placement,
            "placement_basis": self.placement_basis,
            "development_evidence": list(self.development_evidence),
            "holdout_evidence_excluded": list(self.holdout_evidence_excluded),
            "witnesses": list(self.witnesses),
            "observations": [o.to_json() for o in self.observations],
            "verdict": self.verdict,
            "observed_placement": self.observed_placement,
            "status": self.status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CoverageInventory:
    """Every row for one target, with the digest of exactly the inputs that produced them."""

    target: str
    rows: tuple[CoverageRow, ...]
    inputs_digest: str
    input_digests: Mapping[str, str] = field(default_factory=dict)
    holdout_witnesses_excluded: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def counts(self) -> dict[str, dict[str, int]]:
        """``{kind: {status: n}}`` plus a ``total`` row; every status is present, zero or not."""
        out: dict[str, dict[str, int]] = {}
        for kind in (*KINDS, "total"):
            out[kind] = {status: 0 for status in STATUSES}
        for row in self.rows:
            out[row.kind][row.status] += 1
            out["total"][row.status] += 1
        return out

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "target": self.target,
            "inputs_digest": self.inputs_digest,
            "input_digests": dict(sorted(self.input_digests.items())),
            "counts": self.counts(),
            "holdout_witnesses_excluded": list(self.holdout_witnesses_excluded),
            "notes": list(self.notes),
            "rows": [row.to_json() for row in self.rows],
        }


# --- helpers -------------------------------------------------------------------------------------


def _canonical(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()


def _digest(document: Any) -> str:
    return sha256_bytes(_canonical(document))


def _holdout_of(name: str) -> str | None:
    """The claim model a bundle or capsule name carries, by either naming shape."""
    return CM.model_of(name) or CM.mentioned_claim_model(name)


def _caps_for(family: str, cap_map: Mapping[str, Any]) -> list[Any]:
    if family in cap_map:
        return [cap_map[family]]
    return [cap_map[p] for p in SF.primitives_of(family) if p in cap_map]


def required_placement(
    family: str | None,
    dtype: str | None,
    cap_map: Mapping[str, Any],
    *,
    undetermined: frozenset[str] = frozenset(),
    providers: Mapping | None = None,
) -> tuple[str, str, str]:
    """``(placement, boundary, basis)`` for one family/dtype, from the eligibility oracle alone.

    Asked standalone first. A capability the manifest declares only FUSED is then asked again with
    its own declared producers present, and the boundary records that it is reachable only that way.
    A refusal that does not decide (unrecognised or undetermined family) is ``UNKNOWN``, never host.
    """
    if not family:
        return UNKNOWN, BD.UNKNOWN, "the requirement item names no semantic family"
    region = EL.RegionDescriptor(family=family, in_dtype=dtype)
    verdict = EL.is_eligible(region, dict(cap_map), undetermined=undetermined, providers=dict(providers or {}))
    if verdict.eligible:
        return MUST_ACCELERATE, BD.A, verdict.reason
    if verdict.refusal == "fused_only":
        producers = tuple(sorted({p for cap in _caps_for(family, cap_map) for p in cap.composed_with}))
        fused = EL.is_eligible(
            region, dict(cap_map), undetermined=undetermined, providers=dict(providers or {}), fused_with=producers
        )
        if fused.eligible:
            return MUST_ACCELERATE, "fused_with:" + "+".join(producers), fused.reason
        verdict = fused
    if verdict.undetermined or verdict.refusal in _UNDECIDED_REFUSALS:
        return UNKNOWN, BD.UNKNOWN, verdict.reason
    return HOST, BD.HOST_ONLY, verdict.reason


def composition_placement(kind: str, cap_map: Mapping[str, Any]) -> tuple[str, str]:
    """``(placement, basis)`` for a composition shape, parsed from its own tokens.

    The shape vocabulary is :mod:`merlin.targetgen.boundary`'s; a shape is split on its ``->`` seams and
    each segment compared to that module's accelerator/host tokens. A shape needing an accelerator
    segment on a target whose manifest admits no family at all is ``UNKNOWN``, not satisfiable.
    """
    if kind == BD.ROUTING:
        lanes = {BD.ACCEL, BD.HOST}
    else:
        tokens = [t.strip() for t in str(kind).split("->")]
        if not tokens or any(t not in (BD.ACCEL, BD.HOST) for t in tokens):
            return UNKNOWN, f"composition shape {kind!r} is not in the boundary vocabulary"
        lanes = set(tokens)
    if BD.ACCEL in lanes and not cap_map:
        return UNKNOWN, "the shape needs an accelerator region and the manifest admits no family"
    if lanes == {BD.ACCEL}:
        return MUST_ACCELERATE, "every segment of the shape is accelerator work"
    if lanes == {BD.HOST}:
        return HOST, "the shape has no accelerator region"
    return MIXED, "the shape crosses the host/accelerator seam"


def observed_placement(result: Mapping[str, Any] | None) -> str:
    """Where one graded capsule's work landed, read from its whole-module placement census.

    Only a census that DECIDED its population counts (``status == offloaded`` with a silent-fallback
    ledger); anything else -- no census, an undecided one, a malformed one -- is ``UNKNOWN``. An eligible
    region left on the host without a reason is :data:`SILENT_FALLBACK`, whatever else ran.
    """
    if not isinstance(result, Mapping):
        return UNKNOWN
    census = result.get("placement_census")
    if not isinstance(census, Mapping):
        return UNKNOWN
    silent = census.get("silent_fallbacks")
    if census.get("silent_fallbacks_status") != LC.STATUS_OFFLOADED or not isinstance(silent, list):
        return UNKNOWN
    if silent:
        return SILENT_FALLBACK
    coverage = census.get("coverage") if isinstance(census.get("coverage"), Mapping) else {}
    on_unit, on_host = coverage.get("on_accelerator"), coverage.get("on_host")
    if type(on_unit) is not int or type(on_host) is not int:
        return UNKNOWN
    if on_unit and on_host:
        return OBSERVED_MIXED
    if on_unit:
        return OBSERVED_ACCELERATOR
    if on_host:
        return OBSERVED_HOST
    return UNKNOWN


def _verdict(result: Mapping[str, Any] | None) -> str:
    if not isinstance(result, Mapping):
        return NOT_RUN
    status = result.get("status")
    return str(status) if isinstance(status, str) and status else NOT_RUN


def _split_evidence(
    labels: Iterable[str] | None, workloads: frozenset[str] | None
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """``(development, held_out, had_evidence)`` for one item's ``observed_in``-style list."""
    if labels is None:
        return (), (), False
    names = sorted({str(x) for x in labels})
    held = tuple(n for n in names if _holdout_of(n))
    dev = [n for n in names if not _holdout_of(n)]
    if workloads is not None:
        dev = [n for n in dev if n in workloads]
    return tuple(dev), held, bool(names)


def _status(
    required: str, observations: tuple[WitnessObservation, ...], unknown: list[str]
) -> tuple[str, str, str, list[str]]:
    """``(status, verdict, observed_placement, reasons)``. The fail-closed decision, in one place."""
    reasons = list(unknown)
    want = _SATISFIES.get(required)
    passing = [o for o in observations if o.verdict == _PASS]
    good = [o for o in passing if want is not None and o.placement == want]
    if good:
        verdict, placed = _PASS, want
    elif passing:
        verdict, placed = _PASS, sorted({o.placement for o in passing})[0]
    elif observations and all(o.verdict == NOT_RUN for o in observations):
        verdict, placed = NOT_RUN, UNKNOWN
    elif observations:
        ran = sorted({o.verdict for o in observations if o.verdict != NOT_RUN})
        verdict, placed = ran[0], UNKNOWN
    else:
        verdict, placed = "no_witness", UNKNOWN
    if reasons:
        return STATUS_UNKNOWN, verdict, placed, reasons
    if not observations:
        return MISSING, verdict, placed, ["no capsule in the graded corpus witnesses this item"]
    if good:
        return COVERED, verdict, placed, []
    if all(o.verdict == NOT_RUN for o in observations):
        return STATUS_UNKNOWN, verdict, placed, ["witnesses exist but none was graded (not_run is not a pass)"]
    if any(o.placement == UNKNOWN for o in passing):
        return (
            STATUS_UNKNOWN,
            verdict,
            placed,
            ["a witness passed but its placement was not measured; a pass cannot prove where the work ran"],
        )
    if passing:
        return (
            MISSING,
            verdict,
            placed,
            [f"passing witnesses placed the work as {sorted({o.placement for o in passing})}, required {required}"],
        )
    return MISSING, verdict, placed, ["every graded witness failed or was withheld"]


# --- requirement items ---------------------------------------------------------------------------


def _items(spec_doc: Mapping[str, Any]) -> list[dict]:
    """Every requirement item the spec declares, normalised; nothing about placement yet."""
    items: list[dict] = []
    for cell in spec_doc.get("cells") or ():
        items.append(
            {
                "kind": "cell",
                "key": str(cell.get("cell")),
                "family": cell.get("family"),
                "dtype": cell.get("dtype"),
                "shape": cell.get("alignment"),
                "evidence": cell.get("observed_in"),
                "declared": MUST_ACCELERATE,
            }
        )
    for pair in (spec_doc.get("host_lane") or {}).get("required") or ():
        items.append(
            {
                "kind": "host_lane",
                "key": f"{pair.get('family')}/{pair.get('dtype')}",
                "family": pair.get("family"),
                "dtype": pair.get("dtype"),
                "shape": None,
                "evidence": None,
                "declared": HOST,
            }
        )
    host_only = spec_doc.get("host_only") or {}
    for family in host_only.get("families") or ():
        items.append(
            {
                "kind": "host_only",
                "key": str(family),
                "family": family,
                "dtype": (host_only.get("dtypes") or {}).get(family),
                "shape": None,
                "evidence": None,
                "declared": HOST,
            }
        )
    for kind, models in ((spec_doc.get("composition") or {}).get("required") or {}).items():
        items.append(
            {
                "kind": "composition",
                "key": str(kind),
                "family": None,
                "dtype": None,
                "shape": None,
                "evidence": models or [],
                "declared": None,
            }
        )
    for stage in (spec_doc.get("epilogue") or {}).get("required") or ():
        items.append(
            {
                "kind": "epilogue",
                "key": str(stage.get("stage")),
                "family": stage.get("family"),
                "dtype": None,
                "shape": None,
                "evidence": None,
                "declared": MUST_ACCELERATE,
            }
        )
    for geom in (spec_doc.get("shape_geometry") or {}).get("required") or ():
        items.append(
            {
                "kind": "shape_geometry",
                "key": str(geom.get("class")),
                "family": geom.get("family"),
                "dtype": None,
                "shape": geom.get("class"),
                "evidence": geom.get("observed_in"),
                "declared": MUST_ACCELERATE,
                "unreachable": geom.get("unreachable"),
            }
        )
    return items


# --- the pure join -------------------------------------------------------------------------------


def build_inventory(
    target: str,
    spec_doc: Mapping[str, Any],
    contract: Mapping[str, Any],
    witnesses: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    results: Mapping[str, Mapping[str, Any]] | None = None,
    workloads: Iterable[str] | None = None,
) -> CoverageInventory:
    """Join one target's requirement, manifest, witnesses and (optional) grading results. Pure.

    ``witnesses`` is ``{kind: {item key: [capsule names]}}`` (see :func:`collect_witnesses`).
    ``results`` is ``{capsule name: capsule_result document}``; absent means nothing was graded.
    ``workloads`` restricts which development workloads may evidence a requirement; absent means
    every non-held-out workload the spec names. Raises :class:`HoldoutInputError` for a held-out
    workload or result, and ``ValueError`` for a spec that belongs to another target.
    """
    if not isinstance(spec_doc, Mapping) or spec_doc.get("target") != target:
        raise ValueError(
            f"conformance spec is for target {spec_doc.get('target') if isinstance(spec_doc, Mapping) else None!r}, "
            f"not {target!r}"
        )
    chosen = None
    if workloads is not None:
        chosen = frozenset(str(w) for w in workloads)
        held = sorted(w for w in chosen if _holdout_of(w))
        if held:
            raise HoldoutInputError(
                f"held-out claim model(s) supplied as development workloads: {held}; "
                f"{CM.exclusion_rule() or 'claim models may not enter coverage selection'}"
            )
    graded = dict(results or {})
    held_results = sorted(name for name in graded if _holdout_of(str(name)))
    if held_results:
        raise HoldoutInputError(
            f"grading results supplied for held-out claim-model capsule(s): {held_results}; claim results "
            f"are a forbidden source for development coverage"
        )

    cap_map = EL.capability_map_from_contract(dict(contract))
    providers = EL.providers_from_contract(dict(contract))
    undetermined = EL.undetermined_families_from_contract(dict(contract))

    held_witnesses: set[str] = set()
    clean: dict[str, dict[str, tuple[str, ...]]] = {}
    for kind, by_key in (witnesses or {}).items():
        for key, names in (by_key or {}).items():
            keep = []
            for name in names or ():
                if _holdout_of(str(name)):
                    held_witnesses.add(str(name))
                else:
                    keep.append(str(name))
            clean.setdefault(str(kind), {})[str(key)] = tuple(sorted(set(keep)))

    rows: list[CoverageRow] = []
    for item in _items(spec_doc):
        kind, key = item["kind"], item["key"]
        unknown: list[str] = []
        if kind == "composition":
            placement, basis = composition_placement(key, cap_map)
            boundary = key
        else:
            placement, boundary, basis = required_placement(
                item["family"], item["dtype"], cap_map, undetermined=undetermined, providers=providers
            )
        if placement == UNKNOWN:
            unknown.append(f"required placement is undecided: {basis}")
        declared = item.get("declared")
        if declared is not None and placement not in (UNKNOWN, declared):
            unknown.append(
                f"the conformance spec requires this item as {declared} but the manifest's eligibility "
                f"places it {placement} ({basis}); the spec and the manifest disagree"
            )
        if item.get("unreachable"):
            unknown.append(f"the requirement declares this item unreachable: {item['unreachable']}")
        dev, held, had = _split_evidence(item.get("evidence"), chosen)
        if had and not dev:
            unknown.append("the requirement is evidenced only by held-out or unselected workloads")
        names = clean.get(kind, {}).get(key, ())
        observations = tuple(
            WitnessObservation(
                capsule=name,
                verdict=_verdict(graded.get(name)),
                placement=observed_placement(graded.get(name)) if name in graded else UNKNOWN,
            )
            for name in names
        )
        status, verdict, placed, reasons = _status(placement, observations, unknown)
        identity = _digest(
            {
                "kind": kind,
                "key": key,
                "family": item["family"],
                "dtype": item["dtype"],
                "shape": item["shape"],
            }
        )
        rows.append(
            CoverageRow(
                kind=kind,
                key=key,
                signature_identity=identity,
                family=item["family"],
                dtype=item["dtype"],
                shape=item["shape"],
                boundary=boundary,
                required_placement=placement,
                placement_basis=basis,
                development_evidence=dev,
                holdout_evidence_excluded=held,
                witnesses=names,
                observations=observations,
                verdict=verdict,
                observed_placement=placed,
                status=status,
                reasons=tuple(reasons),
            )
        )
    order = {k: i for i, k in enumerate(KINDS)}
    rows.sort(key=lambda r: (order.get(r.kind, len(order)), r.key))

    digests = {
        "spec": _digest(dict(spec_doc)),
        "contract": _digest(dict(contract)),
        "witnesses": _digest(clean),
        "results": _digest(graded),
        "workloads": _digest(sorted(chosen) if chosen is not None else None),
        "claim_models": _digest(list(CM.claim_models())),
    }
    notes = []
    if not graded:
        notes.append("no grading results were supplied, so no item can be covered; witnessed items are unknown")
    return CoverageInventory(
        target=target,
        rows=tuple(rows),
        inputs_digest=_digest({"schema": SCHEMA, "target": target, **digests}),
        input_digests=digests,
        holdout_witnesses_excluded=tuple(sorted(held_witnesses)),
        notes=tuple(notes),
    )


# --- readers around the pure join ----------------------------------------------------------------


def collect_witnesses(
    spec_doc: Mapping[str, Any],
    corpus_roots,
    *,
    labels: Iterable[str] = ("public", "dev"),
    exclude: Iterable[str] = (),
    capability_contract: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """``{kind: {item key: [capsule names]}}`` from the readers the conformance gate already uses."""
    from merlin.targetgen import conformance as CF
    from merlin.targetgen.contract.materialize import capsule_cell_rows

    labelset, skip = set(labels), set(exclude)
    contract = dict(capability_contract) if capability_contract is not None else None
    tile = (spec_doc.get("boundaries") or {}).get("tile_edge")
    out: dict[str, dict[str, list[str]]] = {kind: {} for kind in KINDS}
    for row in capsule_cell_rows(corpus_roots, labels=labelset, tile_dim=tile, exclude=skip):
        for cell in row["cells"]:
            out["cell"].setdefault("/".join(x for x in cell if x), []).append(str(row["name"]))
    if (spec_doc.get("host_lane") or {}).get("required"):
        lane = BD.host_lane_coverage(
            dict(spec_doc), corpus_roots, labels=labelset, exclude=skip, capability_contract=contract
        )
        out["host_lane"] = {k: list(v) for k, v in (lane.get("covered_by") or {}).items()}
    if (spec_doc.get("host_only") or {}).get("families"):
        only = BD.host_only_coverage(
            dict(spec_doc), corpus_roots, labels=labelset, exclude=skip, capability_contract=contract
        )
        out["host_only"] = {k: list(v) for k, v in (only.get("covered_by") or {}).items()}
    if (spec_doc.get("composition") or {}).get("required"):
        corpus = BD.corpus_boundaries(corpus_roots, str(spec_doc.get("target") or ""), labels=labelset, exclude=skip)
        out["composition"] = {k: list(v) for k, v in (corpus.get("by_kind") or {}).items()}
    epilogue = (spec_doc.get("epilogue") or {}).get("required")
    if epilogue:
        gap = CF._epilogue_gap(epilogue, corpus_roots, labels=labelset, exclude=skip)
        for source in ("fused_by", "standalone_by"):
            for k, v in (gap.get(source) or {}).items():
                out["epilogue"].setdefault(k, []).extend(v)
    geometry = (spec_doc.get("shape_geometry") or {}).get("required")
    if geometry:
        gap = CF._geometry_gap(geometry, corpus_roots, labels=labelset, exclude=skip)
        out["shape_geometry"] = {k: list(v) for k, v in (gap.get("covered_by") or {}).items()}
    return {kind: {k: sorted(set(v)) for k, v in sorted(by_key.items())} for kind, by_key in out.items()}


def results_from_dir(root: str | Path) -> dict[str, dict]:
    """``{capsule: capsule_result}`` from every ``capsule_result.json`` under ``root``.

    Two results naming one capsule is an ambiguous grade and raises rather than picking one.
    """
    out: dict[str, dict] = {}
    for path in sorted(Path(root).rglob("capsule_result.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        name = document.get("capsule") if isinstance(document, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: capsule result names no capsule")
        if name in out:
            raise ValueError(f"{path}: a second result for capsule {name!r}; the grade is ambiguous")
        out[name] = document
    return out


def inventory_for_target(
    target: str,
    *,
    results: Mapping[str, Mapping[str, Any]] | None = None,
    workloads: Iterable[str] | None = None,
    labels: Iterable[str] = ("public", "dev"),
    exclude: Iterable[str] | None = None,
    spec_path: str | Path | None = None,
) -> CoverageInventory:
    """The inventory for a registered target from its tracked spec, contract and graded corpus.

    Reads the reference conformance spec, the target's capability contract, and its descriptor's
    graded corpus roots and grading exclusions (plus ``exclude``). Every missing input raises.
    """
    import yaml

    from merlin.targetgen import target_registry as TR
    from merlin.targetgen.corpora import conformance_reference
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    path = Path(spec_path) if spec_path is not None else conformance_reference(target)
    spec_doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract = TR.load_contract(target)
    descriptor = descriptor_for(target)
    if descriptor is None:
        raise FileNotFoundError(f"no target experiment descriptor for {target!r}; its graded corpus is unknown")
    te = load_target_experiment(descriptor)
    skip = set(te.graded_exclude or ()) | set(exclude or ())
    witnesses = collect_witnesses(
        spec_doc, list(te.graded_roots()), labels=labels, exclude=skip, capability_contract=contract
    )
    return build_inventory(target, spec_doc, contract, witnesses, results=results, workloads=workloads)
