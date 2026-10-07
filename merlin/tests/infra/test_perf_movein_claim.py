"""The PJ family must FAIL a hoisting compiler, PASS a just-in-time one, and demand NEITHER of a shape
where the lever is measured not to pay.

Two demonstrations. The first is the both-directions proof: a nest standing its whole operand slab in
front of itself fails, one placing each transfer beside the position that reads it passes, and a
single-position control passes BOTH. The second is the regime conditioning: the demand is waived on
mesh-critical members, the waiver is recorded rather than silent, and a cohort carrying no waived
member is REFUSED. The contract below mirrors the shape the shared template declares (both regime
edges and an admitted staging depth); the phase-0 stream-family test decides the template's own members
through this analyzer. Target-agnostic throughout.
"""

from __future__ import annotations

import pytest

from merlin.perf import movein_claim as MIC
from merlin.perf import movein_placement as MP

ROWS = 16


@pytest.fixture(scope="module")
def contract() -> dict:
    """A frozen PJ acceptance: the two measured regime edges and the declared staging depth."""
    return {
        "schema_version": 1,
        "analyzer": MIC.ANALYZER,
        "property": {"admitted_staging_positions": 2},
        "regime": {"load_critical_at_or_below": 4, "mesh_critical_at_or_above": 7},
        "evidence": {
            "correctness_simulator": "l2_sim",
            "correctness_tier": "L2",
            "structural_instrument": "merlin.targetgen.rocc.decode",
            "structural_tier": "L1",
        },
    }


def _mvin(slot: int, arg: int, offset: int, rows: int = ROWS) -> dict:
    return {
        "class": "MVIN",
        "funct": 2,
        "decoded": {
            "spad_addr": slot,
            "rows": rows,
            "dram": {"kind": "argbase", "arg_index": arg, "offset": offset, "raw": None},
        },
    }


def _stage(slot: int, acc: int) -> dict:
    return {"class": "PRELOAD", "funct": 6, "decoded": {"weight_spad": slot, "c_addr": acc}}


def _compute(slot: int) -> dict:
    return {"class": "COMPUTE_PRELOADED", "funct": 4, "decoded": {"a_spad": slot}}


def _nest(*, hoist: bool, m_blocks: int, k_blocks: int, rows: int = ROWS) -> dict:
    """The SAME transfers to the same addresses with the same widths; only the placement differs."""
    activation = lambda i, j: 1000 + i * k_blocks + j  # noqa: E731
    weight = lambda j: 2000 + j  # noqa: E731
    stream: list[dict] = []
    if hoist:
        stream += [
            _mvin(activation(i, j), 0, (i * k_blocks + j) * rows * rows)
            for i in range(m_blocks)
            for j in range(k_blocks)
        ]
        stream += [_mvin(weight(j), 1, j * rows * rows) for j in range(k_blocks)]
    moved: set[int] = set()
    for j in range(k_blocks):
        for i in range(m_blocks):
            if not hoist:
                if activation(i, j) not in moved:
                    stream.append(_mvin(activation(i, j), 0, (i * k_blocks + j) * rows * rows))
                    moved.add(activation(i, j))
                if weight(j) not in moved:
                    stream.append(_mvin(weight(j), 1, j * rows * rows))
                    moved.add(weight(j))
            stream.append(_stage(weight(j), 3000 + i))
            stream.append(_compute(activation(i, j)))
    return {"instructions": stream}


def _verdict(contract: dict, **kwargs) -> dict:
    return MP.placement_verdict(_nest(**kwargs), admitted_positions=MIC.admitted_positions(contract))


# --------------------------------------------------------------------------------------------
# Fail and pass, in both directions.
# --------------------------------------------------------------------------------------------


def test_the_contract_declares_the_depth_and_both_edges(contract):
    assert MIC.admitted_positions(contract) == 2
    low, high = MIC.regime_bracket(contract)
    assert low < high, "two edges and an undecided band, not a boundary inside the measured interval"


def test_the_hoisting_nest_fails(contract):
    verdict = _verdict(contract, hoist=True, m_blocks=4, k_blocks=2)
    assert verdict["verdict"] == MP.FAIL
    # 8 activation tiles + 2 weight tiles stand in front of a first position that reads 2 of them.
    assert verdict["exposed_prefix"] == 10 and verdict["per_position_transfers"] == 2
    assert verdict["excess"] == 10 - 2 * 2


def test_the_just_in_time_nest_passes(contract):
    verdict = _verdict(contract, hoist=False, m_blocks=4, k_blocks=2)
    assert verdict["verdict"] == MP.PASS and verdict["exposed_prefix"] <= verdict["admitted_prefix"]


def test_the_two_compilers_emit_the_same_transfers(contract):
    hoisted = _nest(hoist=True, m_blocks=4, k_blocks=2)
    jit = _nest(hoist=False, m_blocks=4, k_blocks=2)

    def key(stream):
        return sorted(
            (i["decoded"]["spad_addr"], i["decoded"]["dram"]["arg_index"], i["decoded"]["dram"]["offset"])
            for i in stream["instructions"]
            if i["class"] == "MVIN"
        )

    assert key(hoisted) == key(jit)


@pytest.mark.parametrize("hoist", [True, False])
def test_the_control_shape_passes_both_compilers(contract, hoist):
    assert _verdict(contract, hoist=hoist, m_blocks=1, k_blocks=1)["verdict"] == MP.PASS


def test_a_retaining_position_without_a_sentinel_is_refused_not_guessed(contract):
    """A staging command naming a value no transfer produced is UNKNOWN unless the target says it retains."""
    out = MP.placement_verdict({"instructions": [_stage(7, 0), _compute(9)]}, admitted_positions=2)
    assert out["verdict"] == MP.REFUSED and "no observed producer" in out["reason"]


def test_a_contract_declaring_no_admitted_depth_is_refused():
    out = MP.placement_verdict(_nest(hoist=True, m_blocks=2, k_blocks=2), admitted_positions=None)
    assert out["verdict"] == MP.REFUSED and "UNDECLARED" in out["reason"]


def test_a_trace_with_no_compute_is_refused_not_passed():
    assert MP.placement_verdict({"instructions": [_mvin(0, 0, 0)]}, admitted_positions=2)["verdict"] == MP.REFUSED


def test_the_undisclosed_load_window_is_surfaced_and_not_demanded_on(contract):
    verdict = _verdict(contract, hoist=True, m_blocks=4, k_blocks=2)
    assert verdict["mid_stream_window"] == MP.UNKNOWN_WINDOW and verdict["peak_mid_stream_run"] >= 1


# --------------------------------------------------------------------------------------------
# The regime conditioning: the part that keeps this demand from being wrong.
# --------------------------------------------------------------------------------------------


def _member(name: str, m: int, k: int, contract: dict) -> dict:
    return {
        "name": name,
        "performance": {
            "family": "PJ",
            "claim": "EMITS",
            "shape_geometry": {"M": m, "K": k, "N": ROWS, "row_block": ROWS},
            "acceptance": contract,
        },
    }


def _cohort(contract: dict) -> list[dict]:
    return [
        _member(f"PJ{i:02d}", m * ROWS, k * ROWS, contract)
        for i, (m, k) in enumerate((m, k) for m in (1, 4, 8) for k in (1, 2))
    ]


def test_the_cohort_carries_all_three_buckets(contract):
    bracket = MIC.regime_bracket(contract)
    roles = {m["name"]: MIC.classify(m, bracket=bracket) for m in _cohort(contract)}
    assert roles["PJ00"] == MIC.CONTROL and roles["PJ01"] == MIC.LEVER_BEARING
    assert roles["PJ02"] == roles["PJ03"] == MIC.LEVER_BEARING
    assert roles["PJ04"] == roles["PJ05"] == MIC.EXEMPT


def test_a_member_inside_the_undecided_band_is_not_assigned_a_lever(contract):
    low, high = MIC.regime_bracket(contract)
    member = {"performance": {"shape_geometry": {"M": (low + 1) * ROWS, "K": 2 * ROWS, "N": ROWS}}}
    assert MIC.classify(member, row_block=ROWS, bracket=(low, high)) is None


def test_a_contract_with_no_regime_bracket_is_refused(contract):
    stripped = {k: v for k, v in contract.items() if k != "regime"}
    out = MIC.preflight_movein_evidence(_cohort(stripped), replicates=["r000"])
    assert out["status"] == MIC.REFUSED and any("regime bracket" in r for r in out["refusal_reasons"])


def test_a_cohort_with_no_exempt_member_is_refused(contract):
    bracket = MIC.regime_bracket(contract)
    kept = [m for m in _cohort(contract) if MIC.classify(m, bracket=bracket) != MIC.EXEMPT]
    out = MIC.preflight_movein_evidence(kept, replicates=["r000"])
    assert out["status"] == MIC.REFUSED and any("exempt" in r for r in out["refusal_reasons"])


def test_the_whole_cohort_is_ready(contract):
    out = MIC.preflight_movein_evidence(_cohort(contract), replicates=["r000"])
    assert out["status"] == MIC.READY, out["refusal_reasons"]
    assert out["cohort"][MIC.EXEMPT]


def _rows(contract: dict, *, exempt: str, lever: str) -> list[dict]:
    bracket = MIC.regime_bracket(contract)
    rows = []
    for member in _cohort(contract):
        role = MIC.classify(member, bracket=bracket)
        verdict = exempt if role == MIC.EXEMPT else (lever if role == MIC.LEVER_BEARING else MP.PASS)
        rows.append({"capsule": member["name"], "row_block": ROWS, "verdict": {"verdict": verdict, "excess": 0}})
    return rows


def test_a_hoisting_mesh_critical_member_does_not_refute_the_family(contract):
    """THE waiver: the exempt member FAILS the property and the family still establishes, on the record."""
    out = MIC.analyze_movein_claim(_cohort(contract), _rows(contract, exempt=MP.FAIL, lever=MP.PASS))
    assert out["verdict"] == MIC.ESTABLISHED
    assert out["waived"] and MP.FAIL in out["waived_verdicts"].values()


def test_a_hoisting_load_critical_member_refutes_the_family(contract):
    out = MIC.analyze_movein_claim(_cohort(contract), _rows(contract, exempt=MP.PASS, lever=MP.FAIL))
    assert out["verdict"] == MIC.REFUTED and out["failed"]


def test_a_failing_control_refuses_the_cohort_rather_than_scoring_it(contract):
    rows = [
        {"capsule": m["name"], "row_block": ROWS, "verdict": {"verdict": MP.FAIL, "excess": 1}}
        for m in _cohort(contract)
    ]
    out = MIC.analyze_movein_claim(_cohort(contract), rows)
    assert out["verdict"] == MIC.REFUSED and out["controls_failed"]


def test_the_family_decides_from_traces_when_no_verdict_is_supplied(contract):
    bracket = MIC.regime_bracket(contract)
    rows = []
    for member in _cohort(contract):
        role = MIC.classify(member, bracket=bracket)
        trace = _nest(hoist=role == MIC.LEVER_BEARING, m_blocks=4, k_blocks=2)
        rows.append({"capsule": member["name"], "row_block": ROWS, "trace": trace})
    assert MIC.analyze_movein_claim(_cohort(contract), rows)["verdict"] == MIC.REFUTED
