"""The PD family must FAIL a compiler that does not keep its load configurations, and PASS one that does.

A performance demand is harder than a correctness one because a slow-but-correct program still
computes the right answer, so the only thing separating a real demand from a decorative one is showing
it move in BOTH directions. Every property here is asserted on a stream that does not use the lever
and on one that does; the control members are asserted a third way -- they must pass for both,
because on a shape whose operands share one movement configuration there is no reissue to avoid.

Target-agnostic: the selector, the capacity and the class table come from a synthetic selected support,
exactly the seam a real target's support fills.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.perf import load_state_claim as LSC
from merlin.perf import load_state_residency as LSR
from merlin.runtime.backends import base

#: Two operands of a tiled contraction whose row pitches differ: the second word of the load
#: configuration is the row stride, so these are two DIFFERENT configurations by construction.
PITCH_A = 32
PITCH_B = 64
SELECTOR = {"offset": 3, "width": 2, "capacity": 2}


class _Support:
    """A selected support publishing a load-configuration layout and two inbound-movement classes."""

    def __init__(self, *, layout=True, movement=("MVIN", "MVIN2")):
        fields = {"state_id": {"offset": 3, "width": 2}, "other": {"offset": 8, "width": 4}}
        self.layout = {"fields": fields} if layout else None
        self.movement = movement

    def isa_constants(self, target):
        table = {0: "CONFIG", **{i + 2: name for i, name in enumerate(self.movement)}, 7: "MVOUT"}
        isa = {"CUSTOM_OPCODE": 0x2B, "FUNCT3": 3, "FUNCT_CLASS": table}
        if self.layout is not None:
            isa["CONFIG_LD_LAYOUT"] = self.layout
        return isa

    def decode_instruction(self, funct, rs1, rs2, isa):  # pragma: no cover - not exercised here
        return "UNKNOWN", {}

    def instruction_funct(self, name, rs1, isa):  # pragma: no cover - not exercised here
        raise ValueError(name)


@pytest.fixture
def support(monkeypatch):
    chosen = {"synthetic": SimpleNamespace(rocc_semantics=_Support())}
    monkeypatch.setattr(base, "get_backend", lambda target: chosen[target])
    return chosen


def _config(index: int, *, state: int, pitch: int, selector: dict = SELECTOR) -> dict:
    return {
        "index": index,
        "class": LSR.CONFIG_LOAD_CLASS,
        "funct": 0,
        "rs1": {"kind": "const", "raw": state << int(selector["offset"])},
        "rs2": {"kind": "const", "raw": pitch},
        "decoded": {"subtype": "LD", "stride": pitch},
    }


def _move(index: int) -> dict:
    return {"index": index, "class": "MVIN", "funct": 2, "decoded": {}}


def _trace(instructions: list[dict]) -> dict:
    return {"source": "synthetic", "abi": {}, "instructions": instructions}


def single_slot_stream(*, blocks: int, pitches: tuple[int, ...]) -> dict:
    """The DESCRIBED defect: one selector value and ONE memo slot, re-issued only on a change.

    On a nest whose operands share a pitch that memo HITS and the configuration is issued once; on
    one whose pitches differ it never hits, because the two alternate.
    """
    out: list[dict] = []
    held: int | None = None
    for _ in range(blocks):
        for pitch in pitches:
            if pitch != held:
                out.append(_config(len(out), state=0, pitch=pitch))
                held = pitch
            out.append(_move(len(out)))
    return _trace(out)


def no_memo_stream(*, blocks: int, pitches: tuple[int, ...]) -> dict:
    out: list[dict] = []
    for _ in range(blocks):
        for pitch in pitches:
            out.append(_config(len(out), state=0, pitch=pitch))
            out.append(_move(len(out)))
    return _trace(out)


def resident_stream(*, blocks: int, pitches: tuple[int, ...]) -> dict:
    out: list[dict] = []
    for state, pitch in enumerate(dict.fromkeys(pitches)):
        out.append(_config(len(out), state=state, pitch=pitch))
    for _ in range(blocks):
        for _pitch in pitches:
            out.append(_move(len(out)))
    return _trace(out)


# --------------------------------------------------------------------------------------------
# The derivation: every number the demand uses comes from the selected support, or it refuses.
# --------------------------------------------------------------------------------------------


def test_the_selector_and_its_narrowed_capacity_come_from_the_selected_support(support):
    selector = LSR.load_state_selector("synthetic")
    # two bits name four states, but the table declares two ways to ask for one
    assert selector == {"offset": 3, "width": 2, "capacity": 2}
    assert LSR.movement_in_classes_for("synthetic") == frozenset({"MVIN", "MVIN2"})
    assert LSR.load_state_capacity("synthetic") == 2


def test_a_support_publishing_no_layout_has_no_selector(monkeypatch):
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(rocc_semantics=_Support(layout=False)))
    assert LSR.load_state_selector("synthetic") is None


def test_an_unresolvable_target_has_no_selector_rather_than_a_default(monkeypatch):
    def missing(target):
        raise KeyError(target)

    monkeypatch.setattr(base, "get_backend", missing)
    assert LSR.load_state_selector("synthetic") is None


def test_a_selector_without_a_derived_capacity_falls_back_to_its_own_span():
    assert LSR.capacity_from_selector({"offset": 3, "width": 2}) == 4
    assert LSR.capacity_from_selector({"offset": 3, "width": 2, "capacity": 2}) == 2
    assert LSR.capacity_from_selector({"offset": 3}) is None
    assert LSR.capacity_from_selector(None) is None


def test_an_underivable_selector_refuses_rather_than_passing():
    stream = resident_stream(blocks=4, pitches=(PITCH_A, PITCH_B))
    verdict = LSR.residency_verdict(stream, selector=None)
    assert verdict["verdict"] == LSR.REFUSED
    assert LSR.residency_findings(stream, selector=None), "a refusal must not produce silence"


def test_a_trace_with_no_load_configuration_is_refused_not_passed():
    assert LSR.residency_verdict(_trace([_move(0)]), selector=SELECTOR)["verdict"] == LSR.REFUSED


def test_an_unresolvable_configuration_word_is_refused():
    stream = resident_stream(blocks=2, pitches=(PITCH_A, PITCH_B))
    stream["instructions"][0]["rs1"] = {"kind": "argbase", "arg_index": 0, "offset": 0}
    assert LSR.residency_verdict(stream, selector=SELECTOR)["verdict"] == LSR.REFUSED


# --------------------------------------------------------------------------------------------
# BOTH DIRECTIONS, on the shape where the lever exists, and the control that discriminates.
# --------------------------------------------------------------------------------------------


def test_the_single_slot_nest_fails():
    verdict = LSR.residency_verdict(single_slot_stream(blocks=8, pitches=(PITCH_A, PITCH_B)), selector=SELECTOR)
    assert verdict["verdict"] == LSR.FAIL
    assert (verdict["config_count"], verdict["distinct_configurations"], verdict["excess"]) == (16, 2, 14)
    assert verdict["redundant"] and verdict["states_used"] == [0]


def test_the_resident_nest_passes():
    stream = resident_stream(blocks=8, pitches=(PITCH_A, PITCH_B))
    verdict = LSR.residency_verdict(stream, selector=SELECTOR)
    assert verdict["verdict"] == LSR.PASS and verdict["excess"] == 0
    assert LSR.residency_findings(stream, selector=SELECTOR) == []


def test_the_demand_names_a_property_and_not_a_mechanism():
    """Staging both operands under ONE configuration also passes: the selector is not demanded."""
    one = _trace([_config(0, state=0, pitch=PITCH_A)] + [_move(i + 1) for i in range(16)])
    assert LSR.residency_verdict(one, selector=SELECTOR)["verdict"] == LSR.PASS


def test_a_program_needing_more_configurations_than_the_machine_holds_is_refused():
    pitches = tuple(range(16, 16 * (SELECTOR["capacity"] + 2), 16))
    assert LSR.residency_verdict(no_memo_stream(blocks=2, pitches=pitches), selector=SELECTOR)["verdict"] == (
        LSR.REFUSED
    )


@pytest.mark.parametrize("build", [single_slot_stream, resident_stream])
def test_the_control_shape_passes_the_very_compiler_the_lever_member_fails(build):
    verdict = LSR.residency_verdict(build(blocks=8, pitches=(PITCH_A, PITCH_A)), selector=SELECTOR)
    assert verdict["verdict"] == LSR.PASS, verdict["reason"]
    assert verdict["distinct_configurations"] == 1


def test_a_nest_that_reconfigures_before_every_transfer_fails_even_the_control_shape():
    verdict = LSR.residency_verdict(no_memo_stream(blocks=8, pitches=(PITCH_A, PITCH_A)), selector=SELECTOR)
    assert verdict["verdict"] == LSR.FAIL and verdict["excess"] == 15


# --------------------------------------------------------------------------------------------
# The declared mode, through the channel an author already reads.
# --------------------------------------------------------------------------------------------


def _bracketed(stream: dict) -> dict:
    body = [dict(i, index=n + 1) for n, i in enumerate(stream["instructions"])]
    fence = {"class": "FENCE"}
    return {**stream, "instructions": [dict(fence, index=0), *body, dict(fence, index=len(body) + 1)]}


def _mode_findings(stream: dict, target) -> list[str]:
    from merlin.targetgen import trace_check

    result = trace_check.check(
        _bracketed(stream), {"instruction_classes": [], "modes": {"load_state_resident": True}}, target=target
    )
    return [v for v in result["violations"] if "load-state residency" in v]


def test_the_declared_mode_fires_on_the_single_slot_nest(support):
    found = _mode_findings(single_slot_stream(blocks=8, pitches=(PITCH_A, PITCH_B)), "synthetic")
    assert found and "16 load configurations are issued for 2" in found[0]


def test_the_declared_mode_is_silent_on_the_resident_nest(support):
    assert _mode_findings(resident_stream(blocks=8, pitches=(PITCH_A, PITCH_B)), "synthetic") == []


def test_the_declared_mode_refuses_in_words_without_a_target():
    assert _mode_findings(resident_stream(blocks=8, pitches=(PITCH_A, PITCH_B)), None)


# --------------------------------------------------------------------------------------------
# The family verdict over a cohort.
# --------------------------------------------------------------------------------------------


def _member(name: str, n: int, k: int) -> dict:
    return {
        "name": name,
        "inputs": [{"name": "A", "role": "input", "shape": [16, k]}, {"name": "W", "role": "weight", "shape": [k, n]}],
        "performance": {
            "family": "PD",
            "claim": "EMITS",
            "acceptance": {
                "analyzer": LSC.ANALYZER,
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
    return [_member("PD00", 32, 32), _member("PD01", 32, 64), _member("PD02", 64, 32), _member("PD03", 64, 64)]


def test_the_cohort_is_classified_from_its_declared_operand_pitches():
    assert {c["name"]: LSC.classify(c) for c in _cohort()} == {
        "PD00": LSC.CONTROL,
        "PD01": LSC.LEVER_BEARING,
        "PD02": LSC.LEVER_BEARING,
        "PD03": LSC.CONTROL,
    }


_PASSING = {"verdict": LSR.PASS, "config_count": 2, "distinct_configurations": 2, "excess": 0}
_FAILING = {"verdict": LSR.FAIL, "config_count": 16, "distinct_configurations": 2, "excess": 14}


def _rows(verdicts) -> list[dict]:
    return [{"capsule": c["name"], "verdict": verdicts[LSC.classify(c)]} for c in _cohort()]


def test_the_family_is_refuted_when_the_refutable_members_reissue():
    decided = LSC.analyze_load_state_claim(_cohort(), _rows({LSC.LEVER_BEARING: _FAILING, LSC.CONTROL: _PASSING}))
    assert decided["verdict"] == LSC.REFUTED and decided["failed"] == ["PD01", "PD02"]


def test_the_family_is_established_when_they_do_not():
    decided = LSC.analyze_load_state_claim(_cohort(), _rows({LSC.LEVER_BEARING: _PASSING, LSC.CONTROL: _PASSING}))
    assert decided["verdict"] == LSC.ESTABLISHED


def test_a_failing_control_refuses_the_cohort_instead_of_scoring_it():
    decided = LSC.analyze_load_state_claim(_cohort(), _rows({LSC.LEVER_BEARING: _PASSING, LSC.CONTROL: _FAILING}))
    assert decided["verdict"] == LSC.REFUSED and decided["controls_failed"]


def test_a_cohort_of_controls_alone_is_refused_because_it_could_not_fail():
    controls = [c for c in _cohort() if LSC.classify(c) == LSC.CONTROL]
    rows = [{"capsule": c["name"], "verdict": _PASSING} for c in controls]
    assert LSC.analyze_load_state_claim(controls, rows)["verdict"] == LSC.REFUSED


def test_an_undecided_member_leaves_the_family_undecided_rather_than_established():
    refused = {"verdict": LSR.REFUSED, "reason": "no selector"}
    rows = _rows({LSC.LEVER_BEARING: refused, LSC.CONTROL: _PASSING})
    assert LSC.analyze_load_state_claim(_cohort(), rows)["verdict"] == LSC.REFUSED


def test_the_family_decides_from_traces_when_no_verdict_is_supplied():
    streams = {
        LSC.LEVER_BEARING: single_slot_stream(blocks=8, pitches=(PITCH_A, PITCH_B)),
        LSC.CONTROL: single_slot_stream(blocks=8, pitches=(PITCH_A, PITCH_A)),
    }
    rows = [{"capsule": c["name"], "trace": streams[LSC.classify(c)], "selector": SELECTOR} for c in _cohort()]
    assert LSC.analyze_load_state_claim(_cohort(), rows)["verdict"] == LSC.REFUTED
    streams[LSC.LEVER_BEARING] = resident_stream(blocks=8, pitches=(PITCH_A, PITCH_B))
    rows = [{"capsule": c["name"], "trace": streams[LSC.classify(c)], "selector": SELECTOR} for c in _cohort()]
    assert LSC.analyze_load_state_claim(_cohort(), rows)["verdict"] == LSC.ESTABLISHED


def test_the_preflight_refuses_a_cohort_that_could_never_refute_the_claim():
    controls = [c for c in _cohort() if LSC.classify(c) == LSC.CONTROL]
    refused = LSC.preflight_load_state_evidence(controls, replicates=["r000"])
    assert refused["status"] == LSC.REFUSED and "cannot fail" in refused["refusal_reasons"][0]
    ready = LSC.preflight_load_state_evidence(_cohort(), replicates=["r000"])
    assert ready["status"] == "READY"
    assert {row["capsule"] for row in ready["expected_identities"]} == {c["name"] for c in _cohort()}
    assert all(row["simulator"] and row["tier"] for row in ready["expected_identities"])


def test_the_preflight_refuses_a_target_whose_selector_is_underivable(monkeypatch):
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(rocc_semantics=_Support(layout=False)))
    decided = LSC.preflight_load_state_evidence(_cohort(), replicates=["r000"], target="synthetic")
    assert decided["status"] == LSC.REFUSED and decided["refusal_reasons"]


# --------------------------------------------------------------------------------------------
# The class vocabulary is the shared one, never a re-listed copy.
# --------------------------------------------------------------------------------------------


def test_the_inbound_classes_are_read_from_the_shared_vocabulary(support, monkeypatch):
    """A class the shared table declares as inbound movement is counted with no edit to this module,
    and an outbound one never is."""
    from merlin.targetgen import semantic_families as SF

    monkeypatch.setitem(SF._ISA_CLASS_FAMILY, "XFER_IN", "movement")
    monkeypatch.setitem(SF._ISA_CLASS_DIRECTION, "XFER_IN", "in")
    support["synthetic"] = SimpleNamespace(rocc_semantics=_Support(movement=("MVIN", "XFER_IN", "MVOUT")))
    assert LSR.movement_in_classes_for("synthetic") == frozenset({"MVIN", "XFER_IN"})
    assert LSR.load_state_capacity("synthetic") == 2


def test_the_load_configuration_class_is_the_shared_inbound_configuration():
    from merlin.targetgen import semantic_families as SF

    assert SF.configuration_classes("in") == frozenset({LSR.CONFIG_LOAD_CLASS})
    assert LSR.CONFIG_LOAD_CLASS not in SF.configuration_classes("out")
