"""The PA family must FAIL a compiler that re-presents the array's operand, and PASS one that does not.

The defective compiler modelled here is a REALISTIC one: it emits exactly the instruction count the
ISA forces, with the same addresses and widths, and differs from the good one only in which staging
commands NAME a block. A second defective compiler -- a nest ordered so a different block lands at
every position -- has no consecutive repeats, passes a run-length demand outright and pays the same
cost; both must fail. The control shape must pass for every compiler.

Target-agnostic: the staging vocabulary is the emitted ABI's own def-use model, the retain sentinel and
the accumulator geometry come from a synthetic selected support and synthetic RTL facts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.perf import stationary_claim as STC
from merlin.perf import stationary_residency as SR
from merlin.runtime.backends import base
from merlin.targetgen.rocc import decode

#: The value this synthetic ABI accepts in place of a staged address: "keep what the array holds".
SENTINEL = (1 << 32) - 1
ROWS = 16
#: Accumulator tiles the synthetic machine holds -- enough for every committed output below.
ACCUMULATOR_TILES = 32


def _vocabulary(sentinel=SENTINEL) -> SR.Vocabulary:
    structural = SR.stationary_vocabulary()
    assert structural is not None
    return SR.Vocabulary(
        stage_classes=structural.stage_classes,
        compute_classes=structural.compute_classes,
        staged_fields=structural.staged_fields,
        width_field=structural.width_field,
        sentinel=sentinel,
        retain_form=structural.retain_form,
    )


def _mvin(slot: int, arg: int, offset: int, rows: int) -> dict:
    return {
        "class": "MVIN",
        "funct": 2,
        "decoded": {
            "spad_addr": slot,
            "rows": rows,
            "dram": {"kind": "argbase", "arg_index": arg, "offset": offset, "raw": None},
        },
    }


def _stage(slot: int | None, acc: int) -> dict:
    """One staging command. ``None`` means the retain sentinel -- keep what the array holds."""
    return {
        "class": "PRELOAD",
        "funct": 6,
        "decoded": {"weight_spad": SENTINEL if slot is None else slot, "c_addr": acc},
    }


def _compute(slot: int, *, fresh: bool) -> dict:
    return {
        "class": "COMPUTE_PRELOADED" if fresh else "COMPUTE_ACCUMULATE",
        "funct": 4 if fresh else 5,
        "decoded": {"a_spad": slot, "bd": SENTINEL},
    }


def _nest(*, order: str, peel: bool, m_blocks: int, k_blocks: int, rows: int = ROWS) -> dict:
    """One tiled contraction from one of three compilers; the instruction COUNT is identical in all."""
    activation = lambda i, j: 1000 + i * k_blocks + j  # noqa: E731
    weight = lambda j: 2000 + j  # noqa: E731
    stream = [
        _mvin(activation(i, j), 0, (i * k_blocks + j) * rows * rows, rows)
        for i in range(m_blocks)
        for j in range(k_blocks)
    ]
    stream += [_mvin(weight(j), 1, j * rows * rows, rows) for j in range(k_blocks)]
    if order == "m_inner":
        positions = [(j, i) for j in range(k_blocks) for i in range(m_blocks)]
    else:
        positions = [(j, i) for i in range(m_blocks) for j in range(k_blocks)]
    held: object = None
    for j, i in positions:
        fresh = (not peel) or weight(j) != held
        stream.append(_stage(weight(j) if fresh else None, 3000 + i))
        stream.append(_compute(activation(i, j), fresh=fresh))
        if fresh:
            held = weight(j)
    return {"instructions": stream}


def _verdict(trace: dict, *, tiles=ACCUMULATOR_TILES, vocabulary=None) -> dict:
    return SR.residency_verdict(trace, target=None, vocabulary=vocabulary or _vocabulary(), accumulator_tiles=tiles)


# --------------------------------------------------------------------------------------------
# The facts the demand rests on are DERIVED.
# --------------------------------------------------------------------------------------------


def test_the_retain_form_is_read_from_the_emitted_abi():
    vocabulary = SR.stationary_vocabulary()
    assert vocabulary is not None and vocabulary.retain_form is True
    assert len(vocabulary.compute_classes) > 1
    # no target named: the sentinel is the TARGET's and is not invented
    assert vocabulary.sentinel is None


def test_the_retain_sentinel_comes_from_the_selected_support(monkeypatch):
    semantics = SimpleNamespace(isa_constants=lambda target: {"RETAIN_SENTINEL": 77}, GARBAGE=SENTINEL)
    monkeypatch.setattr(decode, "_semantics", lambda target: semantics)
    assert decode.retain_sentinel("synthetic") == 77  # the published fact wins
    semantics.isa_constants = lambda target: {}
    assert decode.retain_sentinel("synthetic") == SENTINEL  # the decoder's own comparand
    assert SR.stationary_vocabulary("synthetic").sentinel == SENTINEL
    monkeypatch.setattr(decode, "_semantics", lambda target: SimpleNamespace(isa_constants=lambda t: {}))
    assert decode.retain_sentinel("synthetic") is None


def test_without_a_sentinel_the_verdict_refuses():
    verdict = _verdict(_nest(order="m_inner", peel=True, m_blocks=4, k_blocks=2), vocabulary=_vocabulary(None))
    assert verdict["verdict"] == SR.REFUSED and "retain sentinel" in verdict["reason"]


def test_the_accumulator_bound_is_the_targets_depth_over_its_row_count(monkeypatch):
    facts = {
        "arrays": [{"name": "mesh", "rows": ROWS, "cols": ROWS}],
        "datapaths": [{"name": "acc_store"}],
        "memories": [{"name": "acc_store", "depth": 512}, {"name": "operand_store", "depth": 4096}],
    }
    monkeypatch.setattr(SR, "_facts", lambda target: facts)
    assert SR.array_row_block("synthetic") == ROWS
    assert SR.accumulator_tile_capacity("synthetic") == 512 // ROWS


def test_an_underivable_accumulator_falls_back_to_the_weaker_floor_and_says_so(monkeypatch):
    monkeypatch.setattr(SR, "_facts", lambda target: None)
    verdict = SR.residency_verdict(
        _nest(order="m_inner", peel=False, m_blocks=4, k_blocks=2), target="synthetic", vocabulary=_vocabulary()
    )
    assert verdict["floor_kind"] == SR.RUN_LENGTH_FLOOR and verdict["accumulator_tiles"] is None


# --------------------------------------------------------------------------------------------
# Fail and pass, in both directions, against two realistic defective compilers.
# --------------------------------------------------------------------------------------------


def test_the_naive_nest_fails():
    verdict = _verdict(_nest(order="m_inner", peel=False, m_blocks=4, k_blocks=2))
    assert verdict["verdict"] == SR.FAIL
    assert verdict["named_count"] == 8 and verdict["excess"] == 6


def test_the_peeled_nest_passes():
    verdict = _verdict(_nest(order="m_inner", peel=True, m_blocks=4, k_blocks=2))
    assert verdict["verdict"] == SR.PASS and verdict["retained"] == 6


def test_the_same_instruction_count_passes_and_fails():
    naive = _nest(order="m_inner", peel=False, m_blocks=4, k_blocks=2)
    peeled = _nest(order="m_inner", peel=True, m_blocks=4, k_blocks=2)
    assert len(naive["instructions"]) == len(peeled["instructions"])
    assert _verdict(naive)["verdict"] == SR.FAIL and _verdict(peeled)["verdict"] == SR.PASS


def test_the_reordered_nest_that_dodges_the_run_length_floor_still_fails():
    verdict = _verdict(_nest(order="m_outer", peel=False, m_blocks=4, k_blocks=2))
    assert verdict["runs"] == verdict["named_count"], "the dodge is real: no run repeats a tile"
    assert verdict["floor_kind"] == SR.REUSE_ORDERED_FLOOR
    assert verdict["verdict"] == SR.FAIL and verdict["excess"] == 6


def test_peeling_a_scattered_nest_is_not_enough():
    assert _verdict(_nest(order="m_outer", peel=True, m_blocks=4, k_blocks=2))["verdict"] == SR.FAIL


@pytest.mark.parametrize("order", ["m_inner", "m_outer"])
@pytest.mark.parametrize("peel", [False, True])
def test_the_control_shape_passes_every_compiler_the_lever_member_fails(order, peel):
    verdict = _verdict(_nest(order=order, peel=peel, m_blocks=1, k_blocks=4))
    assert verdict["verdict"] == SR.PASS and verdict["excess"] == 0


# --------------------------------------------------------------------------------------------
# Fail closed: every underivable input refuses rather than passing.
# --------------------------------------------------------------------------------------------


def test_a_trace_presenting_nothing_is_refused_not_passed():
    assert _verdict({"instructions": []})["verdict"] == SR.REFUSED


def test_a_presented_tile_with_no_observed_producer_is_refused():
    verdict = _verdict({"instructions": [_mvin(5, 0, 0, 1), _stage(77, 0), _compute(1, fresh=True)]})
    assert verdict["verdict"] == SR.REFUSED and "no observed producer" in verdict["reason"]


def test_a_producer_whose_off_chip_source_is_unresolved_is_refused():
    opaque = {"class": "MVIN", "funct": 2, "decoded": {"spad_addr": 0, "rows": 1, "dram": {"kind": "unknown"}}}
    verdict = _verdict({"instructions": [opaque, _stage(0, 0)]})
    assert verdict["verdict"] == SR.REFUSED and "does not resolve" in verdict["reason"]


def test_the_tile_identity_is_the_off_chip_source_and_not_the_staging_slot():
    first, again = _mvin(0, 0, 0, 1), _mvin(99, 0, 0, 1)  # same off-chip bytes, different slot
    verdict = _verdict({"instructions": [first, _stage(0, 0), again, _stage(99, 1)]})
    assert verdict["distinct_tiles"] == 1 and verdict["verdict"] == SR.FAIL


# --------------------------------------------------------------------------------------------
# The cohort, and the refusals that keep it honest.
# --------------------------------------------------------------------------------------------


def _member(name: str, m: int, k: int) -> dict:
    return {
        "name": name,
        "performance": {
            "family": "PA",
            "claim": "EMITS",
            "shape_geometry": {"M": m, "K": k, "N": ROWS, "row_block": ROWS},
            "acceptance": {
                "analyzer": STC.ANALYZER,
                "evidence": {
                    "correctness_simulator": "l2_sim",
                    "correctness_tier": "L2",
                    "structural_instrument": "merlin.targetgen.rocc.decode",
                    "structural_tier": "L1",
                },
            },
        },
    }


def _cohort() -> list[dict]:
    return [_member("PA00", ROWS, ROWS), _member("PA01", ROWS, 2 * ROWS), _member("PA02", 4 * ROWS, ROWS),
            _member("PA03", 4 * ROWS, 2 * ROWS)]  # fmt: skip


def test_the_cohort_is_classified_from_the_recorded_row_block():
    roles = {m["name"]: STC.classify(m) for m in _cohort()}
    assert roles == {"PA00": STC.CONTROL, "PA01": STC.CONTROL, "PA02": STC.LEVER_BEARING, "PA03": STC.LEVER_BEARING}


def test_a_cohort_with_no_control_or_no_lever_is_refused():
    lever = [m for m in _cohort() if STC.classify(m) == STC.LEVER_BEARING]
    controls = [m for m in _cohort() if STC.classify(m) == STC.CONTROL]
    assert any("control" in r for r in STC.preflight_stationary_evidence(lever, replicates=["r000"])["refusal_reasons"])
    out = STC.preflight_stationary_evidence(controls, replicates=["r000"])
    assert out["status"] == STC.REFUSED and any("cannot fail" in r for r in out["refusal_reasons"])


def test_the_whole_cohort_is_ready():
    out = STC.preflight_stationary_evidence(_cohort(), replicates=["r000"])
    assert out["status"] == STC.READY, out["refusal_reasons"]


def test_a_named_target_without_a_sentinel_is_refused_at_preflight(monkeypatch):
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(rocc_semantics=None))
    out = STC.preflight_stationary_evidence(_cohort(), replicates=["r000"], target="synthetic")
    assert out["status"] == STC.REFUSED and "retain sentinel" in out["refusal_reasons"][0]


def _rows(verdict_for) -> list[dict]:
    return [{"capsule": m["name"], "row_block": ROWS, "verdict": verdict_for(STC.classify(m))} for m in _cohort()]


def test_a_lever_bearing_member_passing_only_at_the_weak_floor_is_undecided_not_established():
    def verdict(role):
        floor = SR.RUN_LENGTH_FLOOR if role == STC.LEVER_BEARING else SR.REUSE_ORDERED_FLOOR
        return {"verdict": SR.PASS, "floor_kind": floor, "excess": 0}

    out = STC.analyze_stationary_claim(_cohort(), _rows(verdict))
    assert out["verdict"] == STC.REFUSED and out["weak_floor"]


def test_the_analyzer_establishes_when_every_member_meets_the_admitted_floor():
    rows = _rows(lambda role: {"verdict": SR.PASS, "floor_kind": SR.REUSE_ORDERED_FLOOR, "excess": 0})
    assert STC.analyze_stationary_claim(_cohort(), rows)["verdict"] == STC.ESTABLISHED


def test_the_analyzer_refutes_when_a_lever_bearing_member_re_presents():
    def verdict(role):
        lever = role == STC.LEVER_BEARING
        return {"verdict": SR.FAIL if lever else SR.PASS, "floor_kind": SR.REUSE_ORDERED_FLOOR, "excess": 6 * lever}

    out = STC.analyze_stationary_claim(_cohort(), _rows(verdict))
    assert out["verdict"] == STC.REFUTED and out["failed"] == ["PA02", "PA03"]


def test_a_failing_control_refuses_the_cohort_rather_than_scoring_it():
    rows = _rows(lambda role: {"verdict": SR.FAIL, "floor_kind": SR.REUSE_ORDERED_FLOOR, "excess": 1})
    out = STC.analyze_stationary_claim(_cohort(), rows)
    assert out["verdict"] == STC.REFUSED and out["controls_failed"]


def test_rows_disagreeing_about_the_machine_are_refused():
    rows = _rows(lambda role: {"verdict": SR.PASS, "floor_kind": SR.REUSE_ORDERED_FLOOR, "excess": 0})
    rows[0]["row_block"] = ROWS * 2
    assert STC.analyze_stationary_claim(_cohort(), rows)["verdict"] == STC.REFUSED


def test_the_declared_mode_reports_through_trace_check(monkeypatch):
    from merlin.targetgen import trace_check

    vocabulary = _vocabulary()
    monkeypatch.setattr(SR, "stationary_vocabulary", lambda target=None: vocabulary)
    monkeypatch.setattr(SR, "accumulator_tile_capacity", lambda target: ACCUMULATOR_TILES)
    expected = {"instruction_classes": [], "modes": {"stationary_resident": True}}
    naive = trace_check.check(_nest(order="m_inner", peel=False, m_blocks=4, k_blocks=2), expected, target="t")
    peeled = trace_check.check(_nest(order="m_inner", peel=True, m_blocks=4, k_blocks=2), expected, target="t")
    assert any("stationary-operand residency (fail)" in v for v in naive["violations"])
    assert not any("stationary-operand residency" in v for v in peeled["violations"])
