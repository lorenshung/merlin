"""The declared ladder, held against the vocabularies it claims to bind to.

The contract's own argument for existing is that three spellings of this ladder already live in code
and a fourth would drift. That is only true if something checks — so the first class here reads the
tracked declaration and requires every tier, engine and capsule tier it names to be one those modules
actually have.

The second class tests the rule the ladder exists to enforce, in both directions: the adjudicating
rung is quotable, every other rung is not, and "we could not tell" is neither.
"""

from __future__ import annotations

import pytest

from merlin.perf import measurement_ladder as ML


@pytest.fixture(scope="module")
def ladder() -> ML.Ladder:
    return ML.load_ladder()


def _pins(tmp_path, body: str):
    p = tmp_path / "ladder.yaml"
    p.write_text(body, encoding="utf-8")
    return p


#: A minimal well-formed ladder. Each malformed test below is this with ONE thing changed, so the
#: failure a test provokes is the thing it named and not an unrelated omission.
_MINIMAL = """
version: 1
claim_kinds:
  ranking: {means: an order}
  cycle_count: {means: the device's number}
rungs:
  cheap:
    tier: static
    establishes: [ranking]
    refuses: [cycle_count]
    adjudicates_cycle_counts: false
    evidence: {kind: rank_correlation}
  real:
    tier: fpga
    engines: [somesim]
    establishes: [ranking, cycle_count]
    adjudicates_cycle_counts: true
    evidence: {kind: the_device}
targets:
  t_one:
    rungs: [cheap, real]
    cycle_adjudicator: real
"""


class TestTheDeclarationBindsToTheCodeItNamesRatherThanDrifting:
    def test_every_rung_tier_is_a_tier_the_measurement_module_has(self, ladder):
        from merlin.kernels.measurement import TIER_ORDER

        for rung in ladder.rungs.values():
            assert rung.tier in TIER_ORDER, (
                f"rung {rung.name!r} declares tier {rung.tier!r}, which kernels.measurement.TIER_ORDER "
                f"does not have ({list(TIER_ORDER)}); the ladder would be a fourth spelling"
            )

    def test_every_declared_engine_is_one_the_engine_policy_ranks_or_the_adjudicator_itself(self, ladder):
        from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY

        elaborated = [r for r in ladder.rungs.values() if r.tier == "rtl"]
        assert elaborated, "the ladder declares no elaborated-RTL rung"
        for rung in elaborated:
            for engine in rung.engines:
                assert engine in ENGINE_PRIORITY, (
                    f"rung {rung.name!r} names engine {engine!r}, which rtl_engine_policy does not "
                    f"rank ({list(ENGINE_PRIORITY)})"
                )

    def test_every_capsule_tier_is_one_the_capsule_runner_maps(self, ladder):
        from merlin.targetgen.runner_config import conventional_tier_sim

        _TIER_SIM = conventional_tier_sim()

        for rung in ladder.rungs.values():
            if rung.capsule_tier:
                assert rung.capsule_tier in _TIER_SIM, (
                    f"rung {rung.name!r} declares capsule tier {rung.capsule_tier!r}, which "
                    f"the capsule runner's tier map does not map ({sorted(_TIER_SIM)})"
                )

    def test_the_rungs_are_ordered_by_ascending_tier(self, ladder):
        # Cheapest first, and the ladder's own claim is that authority tracks cost. A rung whose tier
        # sits below a cheaper rung's would make `establishes` and `tier` tell different stories.
        from merlin.kernels.measurement import TIER_ORDER

        seen = [TIER_ORDER.index(r.tier) for r in ladder.rungs.values()]
        assert seen == sorted(seen), "rungs are declared out of tier order"

    def test_exactly_the_adjudicating_rung_establishes_a_cycle_count(self, ladder):
        establishing = {n for n, r in ladder.rungs.items() if r.supports(ML.CYCLE_COUNT)}
        adjudicating = {n for n, r in ladder.rungs.items() if r.adjudicates_cycle_counts}
        assert establishing == adjudicating

    def test_the_elaborated_rtl_rung_is_declared_a_search_signal(self, ladder):
        """The finding this whole contract came out of, asserted rather than left to prose.

        Its certificate compares output bytes; the `cycle_accurate: true` beside them is a literal
        the producer writes. So the rung establishes output equivalence and refuses cycle counts, and
        it must say why in a field a reader can find.
        """
        elaborated = [r for r in ladder.rungs.values() if r.tier == "rtl"]
        for rung in elaborated:
            assert not rung.adjudicates_cycle_counts
            assert ML.CYCLE_COUNT in rung.refuses
            assert "output_equivalence" in rung.establishes
            assert rung.evidence.get("kind") == "output_bytes"
            assert rung.why_not_cycles.strip(), f"{rung.name!r} refuses cycle counts without saying why"

    def test_a_target_naming_a_non_adjudicating_rung_as_its_adjudicator_is_refused(self, tmp_path):
        bad = _MINIMAL.replace("cycle_adjudicator: real", "cycle_adjudicator: cheap")
        with pytest.raises(ML.LadderError, match="adjudicates_cycle_counts: false"):
            ML.load_ladder(_pins(tmp_path, bad))

    def test_a_rung_whose_two_statements_disagree_is_refused(self, tmp_path):
        # `adjudicates_cycle_counts: true` while not establishing cycle_count: the authority would
        # depend on which field a caller read, which is how the typed constant survived.
        bad = _MINIMAL.replace(
            "    establishes: [ranking]\n    refuses: [cycle_count]\n    adjudicates_cycle_counts: false",
            "    establishes: [ranking]\n    refuses: [cycle_count]\n    adjudicates_cycle_counts: true",
        )
        with pytest.raises(ML.LadderError, match="must agree"):
            ML.load_ladder(_pins(tmp_path, bad))

    def test_a_rung_with_no_stated_evidence_is_refused(self, tmp_path):
        bad = _MINIMAL.replace("    evidence: {kind: the_device}\n", "")
        with pytest.raises(ML.LadderError, match="evidence.kind"):
            ML.load_ladder(_pins(tmp_path, bad))

    def test_an_undeclared_claim_kind_is_refused(self, tmp_path):
        bad = _MINIMAL.replace("establishes: [ranking]", "establishes: [ranking, vibes]")
        with pytest.raises(ML.LadderError, match="vibes"):
            ML.load_ladder(_pins(tmp_path, bad))

    def test_an_adjudicator_the_target_does_not_list_is_refused(self, tmp_path):
        bad = _MINIMAL.replace("rungs: [cheap, real]", "rungs: [cheap]")
        with pytest.raises(ML.LadderError, match="does not exist for it"):
            ML.load_ladder(_pins(tmp_path, bad))

    def test_the_minimal_ladder_itself_loads(self, tmp_path):
        # Otherwise every refusal above could be passing for the wrong reason.
        got = ML.load_ladder(_pins(tmp_path, _MINIMAL))
        assert got.for_target("t_one").cycle_adjudicator == "real"


class TestOnlyTheAdjudicatingRungMayBeQuoted:
    def test_the_adjudicating_rung_is_quotable_and_every_other_rung_is_not(self, ladder):
        for target_name in sorted(ladder.targets):
            tl = ladder.targets[target_name]
            if not tl.adjudicated:
                continue
            for rung in tl.rungs:
                got = ML.adjudicate(target_name, rung, ladder=ladder)
                assert got.quotable is (rung == tl.cycle_adjudicator), (
                    f"{target_name}/{rung}: quotable={got.quotable} but the declared adjudicator is "
                    f"{tl.cycle_adjudicator!r}"
                )

    def test_an_engine_resolves_through_its_rung_rather_than_by_spelling(self, ladder):
        adjudicators = [t for t in ladder.targets.values() if t.adjudicated]
        if not adjudicators:
            pytest.skip("no adjudicated target declared")
        tl = adjudicators[0]
        rung = ladder.rungs[tl.cycle_adjudicator]
        for engine in rung.engines:
            got = ML.adjudicate(tl.target, engine, ladder=ladder)
            assert got.quotable
            assert got.instrument == rung.name  # reported as the RUNG, not as the engine string

    def test_a_search_signal_rung_refuses_cycles_and_still_establishes_what_it_is_good_for(self, ladder):
        for target_name in sorted(ladder.targets):
            tl = ladder.targets[target_name]
            for rung_name in tl.rungs:
                rung = ladder.rungs[rung_name]
                if rung.adjudicates_cycle_counts:
                    continue
                assert not ML.adjudicate(target_name, rung_name, ladder=ladder).quotable
                for kind in rung.establishes:
                    got = ML.adjudicate(target_name, rung_name, claim_kind=kind, ladder=ladder)
                    assert got.state == ML.ADJUDICATED, (
                        f"{rung_name!r} declares it establishes {kind!r} but the ladder refuses it; a "
                        "rung that is good for nothing is not a rung"
                    )

    def test_an_unstated_instrument_is_unknown_not_a_refusal(self, ladder):
        adjudicated = [t for t in sorted(ladder.targets) if ladder.targets[t].adjudicated]
        if not adjudicated:
            pytest.skip("no adjudicated target declared")
        for spelling in ("", "   ", ML.UNKNOWN):
            got = ML.adjudicate(adjudicated[0], spelling, ladder=ladder)
            assert got.state == ML.UNKNOWN
            assert not got.quotable

    def test_an_undeclared_target_is_unknown_not_unadjudicated(self, ladder):
        # Both refuse the quote, and they send a reader to different places: one to build an FPGA
        # image, the other to fix the contract. Collapsing them loses that.
        got = ML.adjudicate("a_target_the_contract_has_never_heard_of", "firesim", ladder=ladder)
        assert got.state == ML.UNKNOWN
        assert not got.quotable

    def test_a_declared_target_with_no_adjudicator_is_unadjudicated(self, ladder):
        undeclared = [t for t in sorted(ladder.targets) if not ladder.targets[t].adjudicated]
        if not undeclared:
            pytest.skip("every declared target has a cycle adjudicator")
        tl = ladder.targets[undeclared[0]]
        got = ML.adjudicate(tl.target, tl.rungs[-1], ladder=ladder)
        assert got.state == ML.UNADJUDICATED
        assert not got.quotable

    def test_require_adjudicated_raises_on_everything_but_adjudicated(self, ladder):
        adjudicated = [t for t in sorted(ladder.targets) if ladder.targets[t].adjudicated]
        if not adjudicated:
            pytest.skip("no adjudicated target declared")
        target = adjudicated[0]
        tl = ladder.targets[target]
        assert ML.require_adjudicated(target, tl.cycle_adjudicator, ladder=ladder).quotable
        for rung in tl.rungs:
            if rung == tl.cycle_adjudicator:
                continue
            with pytest.raises(ML.Unadjudicated):
                ML.require_adjudicated(target, rung, ladder=ladder)
        with pytest.raises(ML.Unadjudicated):
            ML.require_adjudicated(target, ML.UNKNOWN, ladder=ladder)
