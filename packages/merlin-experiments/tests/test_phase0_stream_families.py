"""Phase 0 selects the emitted-stream families (PD, PA, PJ), and what it mints is decidable.

The shared template used to record these three as blocked because their analyzers were missing. They
are live sweeps now, so this pins the whole path a target's derivation takes: the template loads them
as EMITS families, gate-admitted sweeps materialize members onto the direct corpus path with their
declared modes and correctness-tier cap, the members the corpus builder writes classify into the
buckets each analyzer requires, and phase-2 dispatch resolves each family to its own procedure.
A family whose gate a target fails is skipped with evidence, never minted.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import profiles, writer
from merlin_experiments.phase0 import sweeps as SWEEPS
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import prompt as PP
from merlin_experiments.phase2.claims import dispatch as CD
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.corpus import PerformanceCapsule

from merlin.common.paths import repo_root
from merlin.perf import member_geometry
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import target_experiment

STREAM = ("PD", "PA", "PJ")
TILE = 16
#: The synthetic target's own PJ measurement. A regime bracket is device data, so the shared template
#: carries none and every target that wants a decidable PJ cohort declares one in its recipe.
SYNTHETIC_PJ = {"load_critical_at_or_below": 4, "mesh_critical_at_or_above": 7, "basis": "synthetic bracket"}


def _profile(tmp_path, regimes: dict | None = None) -> dict:
    profile: dict = {"capsules": []}
    if regimes:
        profile["performance_regimes"] = copy.deepcopy(regimes)
    profiles._merge_shared_perf(
        profile,
        source=tmp_path / "recipe.yaml",
        performance_template=repo_root() / "experiments/templates/phase0/performance.yaml",
    )
    profile["sweeps"] = [s for s in profile["sweeps"] if s["id"] in STREAM]
    return profile


def _facts(**overrides) -> dict:
    names = {t for s in _profile_sweeps() for t in s["base"]["performance"]["gate"]["traits"]}
    values = {name: overrides.get(name, True) for name in names}
    return {
        "traits": {
            name: {"satisfied": value, "tier": "test_tier", "evidence": f"{name}={value}", "missing": []}
            for name, value in values.items()
        }
    }


def _profile_sweeps() -> list[dict]:
    import yaml

    document = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    return [s for s in document["sweeps"] if s["id"] in STREAM]


@pytest.fixture
def binding(monkeypatch):
    contract = {"runner": {"tier_sim": {"L2": "functional_sim", "L3": "rtl_sim"}}}
    monkeypatch.setattr(
        target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(contract=contract)
    )
    monkeypatch.setattr(member_geometry, "census_classes", lambda target: {})
    return CS.CorpusBinding(
        target="synthetic",
        tile_dim=TILE,
        operand_dtype="int8",
        accum_dtype="int32",
        integer=True,
        tiers=["L2", "L3"],
        compare="exact",
    )


def _members(tmp_path, binding, regimes=None, **trait_overrides) -> tuple[list[dict], list[dict]]:
    skipped: list[dict] = []
    profile = _profile(tmp_path, {"PJ": SYNTHETIC_PJ} if regimes is None else regimes)
    entries = SWEEPS.expand_sweeps(profile, binding, trait_facts=_facts(**trait_overrides), skipped=skipped)
    return entries, skipped


def _capsule(entry: dict, binding) -> dict:
    """What the phase-0 writer persists for one member: the built capsule plus its declared blocks."""
    cap, _mlir = CS.build(copy.deepcopy(entry), binding)
    cap["name"] = entry["name"]
    writer._carry_declared_blocks(entry, cap)
    assert writer._stamp_member_geometry(cap, binding)
    return cap


def test_the_template_loads_the_stream_families_as_live_emits_sweeps(tmp_path):
    profile = _profile(tmp_path)
    families = {row["family"]: row for row in profile["_performance_template"]["families"]}
    assert all(families[f]["claim"] == "EMITS" for f in STREAM)
    blocked = {row["family"] for row in profile["_performance_template"]["blocked_unimplemented"]}
    assert blocked.isdisjoint(STREAM)


def test_an_emits_block_without_an_analyzer_is_refused_at_load():
    performance = copy.deepcopy(_profile_sweeps()[0]["base"]["performance"])
    del performance["acceptance"]
    with pytest.raises(ValueError, match="acceptance.analyzer is required"):
        profiles._validate_performance_block(performance, owner="PD")


def test_the_phase0_selection_mints_every_stream_family(tmp_path, binding):
    entries, skipped = _members(tmp_path, binding)
    by_family: dict[str, list[dict]] = {}
    for entry in entries:
        by_family.setdefault(entry["performance"]["family"], []).append(entry)
    assert {f: len(rows) for f, rows in by_family.items()} == {"PD": 4, "PA": 4, "PJ": 6}
    assert not skipped
    for entry in entries:
        evidence = entry["performance"]["acceptance"]["evidence"]
        assert evidence["correctness_simulator"] == "functional_sim"  # resolved from the target route
        assert "timing_simulator" not in evidence
        assert entry["max_oracle_tier"] == "L2"
        assert entry["performance"]["cost"]["projected_cycles"] == "not_a_cycle_observable"
    assert all(e["modes"]["load_state_resident"] for e in by_family["PD"])
    assert all(e["modes"]["stationary_resident"] for e in by_family["PA"])


def test_a_family_whose_gate_the_target_fails_is_skipped_with_evidence(tmp_path, binding):
    entries, skipped = _members(tmp_path, binding, persistent_configuration_state=False)
    assert "PD" not in {e["performance"]["family"] for e in entries}
    (row,) = [s for s in skipped if s["family"] == "PD"]
    assert row["status"] == "skipped_inapplicable"


def test_the_minted_cohorts_are_decidable_by_their_own_analyzers(tmp_path, binding):
    entries, _ = _members(tmp_path, binding)
    cohorts: dict[str, list[dict]] = {}
    for entry in entries:
        cohorts.setdefault(entry["performance"]["family"], []).append(_capsule(entry, binding))
    for family, descriptors in cohorts.items():
        resolved = CD.resolve(descriptors)
        assert resolved.identity.declared == descriptors[0]["performance"]["acceptance"]["analyzer"]
        preflight = resolved.preflight(descriptors, replicates=["r000"])
        assert preflight["status"] == "READY", (family, preflight["refusal_reasons"])
        roles = {role for role, names in preflight["cohort"].items() if names}
        assert roles == set(next(s for s in _profile_sweeps() if s["id"] == family)["comparison_roles"])
    assert all(c["expected"]["modes"]["load_state_resident"] for c in cohorts["PD"])
    assert all(c["expected"]["modes"]["stationary_resident"] for c in cohorts["PA"])
    assert all(c["performance"]["shape_geometry"]["row_block"] == TILE for c in cohorts["PJ"])


def test_dispatch_registers_every_stream_analyzer_and_refuses_a_mixed_cohort(tmp_path, binding):
    table = CD._registry()
    for sweep in _profile_sweeps():
        assert sweep["base"]["performance"]["acceptance"]["analyzer"] in table
    entries, _ = _members(tmp_path, binding)
    capsules = [_capsule(e, binding) for e in entries]
    with pytest.raises(CD.DispatchError, match="mixed cohort"):
        CD.resolve(capsules)
    decided = CD.analyze(
        [c for c in capsules if c["performance"]["family"] == "PD"],
        [
            {"capsule": c["name"], "verdict": {"verdict": "PASS"}}
            for c in capsules
            if c["performance"]["family"] == "PD"
        ],
    )
    assert decided["verdict"] == "ESTABLISHED" and decided["declared_analyzer"].endswith("load_state_claim/v1")


def test_phase2_admits_the_claim_and_refuses_to_measure_it_as_cycles(tmp_path, binding):
    PP.PerfFamily("PD", "EMITS", "control", "stream", "{}").validate()
    entries, _ = _members(tmp_path, binding)
    capsules = tuple(
        PerformanceCapsule("PD", e["name"], tmp_path, f"_perf/{e['name']}", _capsule(e, binding), "0" * 64, 3, 300)
        for e in entries
        if e["performance"]["family"] == "PD"
    )
    with pytest.raises(StageGateError, match="decided from the candidate's emitted stream"):
        RECORD.prepare_formal_claim(capsules)


# ------------------------------------------------------------------ the PJ bracket is the target's data
def _pj_regime(profile: dict) -> dict:
    (sweep,) = [s for s in profile["sweeps"] if s["id"] == "PJ"]
    return sweep["base"]["performance"]["acceptance"]["regime"]


def test_the_shared_template_carries_no_pj_bracket_edges():
    """Where PJ's regimes split is measured on one device; the shared template must not hold it."""
    (sweep,) = [s for s in _profile_sweeps() if s["id"] == "PJ"]
    regime = sweep["base"]["performance"]["acceptance"]["regime"]
    assert not set(profiles._REGIME_EDGES) & set(regime)
    assert regime["bracket"] == regime["basis"] == profiles.TARGET_DECLARED


def test_the_selected_targets_bracket_is_merged_where_the_template_marks_it(tmp_path):
    regime = _pj_regime(_profile(tmp_path, {"PJ": SYNTHETIC_PJ}))
    assert list(regime) == [
        "discriminant",
        "load_critical_at_or_below",
        "mesh_critical_at_or_above",
        "undecided_band",
        "basis",
        "waiver",
    ]
    assert (regime["load_critical_at_or_below"], regime["mesh_critical_at_or_above"]) == (4, 7)
    assert regime["basis"] == "synthetic bracket"


def test_the_template_digest_is_not_changed_by_a_targets_bracket(tmp_path):
    with_bracket = _profile(tmp_path, {"PJ": SYNTHETIC_PJ})["_performance_template"]["sha256"]
    without = _profile(tmp_path)["_performance_template"]["sha256"]
    assert with_bracket == without


def test_a_target_declaring_no_bracket_has_its_pj_cohort_refused(tmp_path, binding):
    """No edge is defaulted: the contract records UNDECLARED and the analyzer refuses, saying so."""
    regime = _pj_regime(_profile(tmp_path))
    assert regime["bracket"] == regime["basis"] == profiles.TARGET_UNDECLARED
    assert not set(profiles._REGIME_EDGES) & set(regime)
    entries, skipped = _members(tmp_path, binding, regimes={})
    assert not skipped
    cohort = [_capsule(e, binding) for e in entries if e["performance"]["family"] == "PJ"]
    assert cohort
    preflight = CD.resolve(cohort).preflight(cohort, replicates=["r000"])
    assert preflight["status"] == "REFUSED"
    (reason,) = preflight["refusal_reasons"]
    assert "no regime bracket" in reason and profiles.TARGET_UNDECLARED in reason


@pytest.mark.parametrize(
    "regimes, message",
    [
        ({"PD": SYNTHETIC_PJ}, "declares no target-supplied regime bracket"),
        ({"PJ": {**SYNTHETIC_PJ, "mesh_critical_at_or_above": 4}}, "lower edge must be below"),
        ({"PJ": {"load_critical_at_or_below": 4, "mesh_critical_at_or_above": 7}}, "needs exactly"),
        ({"PJ": {**SYNTHETIC_PJ, "basis": " "}}, "must state the measurement"),
    ],
)
def test_a_malformed_or_misplaced_bracket_is_refused_at_load(tmp_path, regimes, message):
    with pytest.raises(ValueError, match=message):
        _profile(tmp_path, regimes)


def test_the_gemmini_recipe_reproduces_the_bracket_the_template_used_to_hold(tmp_path):
    """Target-specific, selected explicitly: the move kept gemmini's frozen PJ contract identical."""
    import yaml

    recipe = yaml.safe_load((repo_root() / "examples/gemmini/phase0/recipe.yaml").read_text())
    regime = _pj_regime(_profile(tmp_path, recipe["performance_regimes"]))
    assert regime == {
        "discriminant": "reload_multiplicity_output_row_blocks_spanned",
        "load_critical_at_or_below": 4,
        "mesh_critical_at_or_above": 7,
        "undecided_band": "refuse_the_cohort_rather_than_assign_a_lever_by_coin_toss",
        "basis": (
            "four shapes on pinned elaborated RTL, each arm bit-exact against the vendor kernel in the same "
            "binary: 3136x64x64 (mult 196, burst->jit -3.2%), 49x512x1024 at a 7-row geometry (mult 7, "
            "-0.6%), 64x512x1024 (mult 4, -22.3%), 64x256x1024 (mult 4, -12.7%)"
        ),
        "waiver": regime["waiver"],
    }
    assert list(regime)[1:3] == list(profiles._REGIME_EDGES)
