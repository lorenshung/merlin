"""One selected measured timing record for installed and native Phase 1 readers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import timing
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import certification

from merlin.common.digest import sha256_file


def test_selected_timing_is_bound_to_target_config_and_simulator_bytes(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    target = "synthetic_device"
    config = "SyntheticConfig"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic_device\n")
    simulator = tmp_path / "chipyard/sims/verilator" / f"simulator-chipyard.harness-{config}"
    simulator.parent.mkdir(parents=True)
    simulator.write_bytes(b"observed simulator bytes")
    selected = timing.timing_path(tmp_path, target)
    selected.write_text(
        json.dumps(
            {
                "target": target,
                "config": config,
                "verilator_per_capsule_s": 1500.0,
                "simulator_sha256": sha256_file(simulator),
                "measured_by": "fixture observation",
            }
        )
    )
    import merlin.common.paths as paths
    import merlin.targetgen.target_experiment as experiments

    monkeypatch.setattr(paths, "ext_path", lambda name: tmp_path / "chipyard")
    monkeypatch.setattr(
        experiments, "load_target_experiment", lambda path: SimpleNamespace(target=target, sim_via="chipyard")
    )
    monkeypatch.setattr(
        experiments, "declared_vs_resolved_contract", lambda selected: (None, tmp_path / "contract.yaml", "agree")
    )
    monkeypatch.setattr(
        experiments,
        "load_capability_manifest",
        lambda name, **kw: SimpleNamespace(contract={"runtime": {"rtl_sim_config": config}}),
    )
    context = SimpleNamespace(target=target, descriptor=descriptor, experiment=tmp_path)
    record = timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    assert record["verilator_per_capsule_s"] == 1500
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 3000
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == 3000

    with pytest.raises(ValueError, match="not bound to target"):
        timing.read_verified_timing(selected, descriptor=descriptor, target="other_device")
    alias = tmp_path / "alias.json"
    alias.symlink_to(selected)
    with pytest.raises(ValueError, match="symlink"):
        timing.read_verified_timing(alias, descriptor=descriptor, target=target)
    simulator.write_bytes(b"different simulator bytes")
    with pytest.raises(ValueError, match="simulator bytes changed"):
        timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 2400
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == simjob._CERT_TIMEOUT_FALLBACK_S


def test_vcs_timing_is_bound_to_simv_and_daidir_bytes(tmp_path, monkeypatch):
    target = "synthetic_device"
    config = "SyntheticConfig"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic_device\n")
    simv = tmp_path / "vcs" / "simv"
    (tmp_path / "vcs" / "simv.daidir").mkdir(parents=True)
    simv.write_bytes(b"observed simv loader")
    (tmp_path / "vcs" / "simv.daidir" / "_1_archive_1.so").write_bytes(b"compiled design")
    monkeypatch.setenv("MERLIN_SYNTHETIC_DEVICE_SIMV", str(simv))
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "vcs")
    selected = timing.timing_path(tmp_path, target)
    record = {
        "target": target,
        "config": config,
        "engine": "vcs",
        "per_capsule_s": 600.0,
        **timing.simulator_identity("vcs", simv),
        "measured_by": "fixture observation",
    }
    selected.write_text(json.dumps(record))
    import merlin.common.paths as paths
    import merlin.targetgen.target_experiment as experiments

    monkeypatch.setattr(paths, "ext_path", lambda name: tmp_path / "chipyard")
    monkeypatch.setattr(
        experiments, "load_target_experiment", lambda path: SimpleNamespace(target=target, sim_via="chipyard")
    )
    monkeypatch.setattr(
        experiments, "declared_vs_resolved_contract", lambda selected: (None, tmp_path / "contract.yaml", "agree")
    )
    monkeypatch.setattr(
        experiments,
        "load_capability_manifest",
        lambda name, **kw: SimpleNamespace(contract={"runtime": {"rtl_sim_config": config}}),
    )
    context = SimpleNamespace(target=target, descriptor=descriptor, experiment=tmp_path)
    assert timing.read_verified_timing(selected, descriptor=descriptor, target=target)["per_capsule_s"] == 600
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 1200
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == max(
        simjob._CERT_TIMEOUT_FLOOR_S, 1200
    )

    # A VCS-pinned run refuses a Verilator observation, and a VCS record without a daidir identity.
    legacy = dict(record, engine=None, verilator_per_capsule_s=600.0)
    del legacy["engine"]
    selected.write_text(json.dumps(legacy))
    with pytest.raises(ValueError):
        timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    selected.write_text(json.dumps({k: v for k, v in record.items() if k != "simulator_daidir_sha256"}))
    with pytest.raises(ValueError, match="daidir identity"):
        timing.read_verified_timing(selected, descriptor=descriptor, target=target)
    # The design lives in the daidir: changing it (not the loader) must invalidate the observation.
    selected.write_text(json.dumps(record))
    (tmp_path / "vcs" / "simv.daidir" / "_1_archive_1.so").write_bytes(b"another design")
    with pytest.raises(ValueError, match="daidir bytes changed"):
        timing.read_verified_timing(selected, descriptor=descriptor, target=target)


def test_non_chipyard_simulator_keeps_legacy_timeout_without_chipyard_claim(tmp_path, monkeypatch):
    import merlin.targetgen.target_experiment as experiments

    target = "arc_device"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: arc_device\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    selected = scripts / f".oracle_timing.{target}.json"
    selected.write_text(json.dumps({"verilator_per_capsule_s": 1500}))
    context = SimpleNamespace(target=target, descriptor=descriptor, experiment=tmp_path)
    monkeypatch.setattr(
        experiments, "load_target_experiment", lambda path: SimpleNamespace(target=target, sim_via="mlc_arc")
    )
    monkeypatch.setattr(
        timing, "read_verified_timing", lambda *args, **kwargs: pytest.fail("Chipyard verifier used for arc")
    )
    assert not timing.requires_chipyard_timing(descriptor)
    assert certification._verilator_per_capsule_timeout(context, timing_file=selected) == 3000
    monkeypatch.setattr(simjob, "_cert_budget_s", lambda target: (None, "no fit"))
    assert simjob._per_capsule_timeout(0, context=context, timing_file=selected)[0] == 3000


@pytest.fixture
def functional_corpus(tmp_path, monkeypatch):
    import merlin.targetgen.target_experiment as experiments

    public = tmp_path / "corpus/public"
    public.mkdir(parents=True)
    (public / "capsule.yaml").write_text("required_oracle_tiers: [L2]\n")
    selected = SimpleNamespace(sim_via="chipyard", capsule_corpus=public, graded_roots=lambda: [public])
    monkeypatch.setattr(experiments, "load_target_experiment", lambda path: selected)
    return tmp_path / "descriptor.yaml", public


def test_functional_corpus_does_not_require_an_unexecuted_rtl_measurement(functional_corpus):
    descriptor, public = functional_corpus
    assert not timing.requires_chipyard_timing(descriptor)
    (public / "capsule.yaml").write_text("required_oracle_tiers: [L0, L1, L2]\n")
    assert not timing.requires_chipyard_timing(descriptor)


def test_operator_tier_floor_requires_timing_only_for_scored_capsules(functional_corpus, monkeypatch):
    descriptor, public = functional_corpus
    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "L3")
    assert timing.requires_chipyard_timing(descriptor)
    (public / "capsule.yaml").write_text("required_oracle_tiers: [L2]\nscored: false\n")
    assert not timing.requires_chipyard_timing(descriptor)
    (public / "capsule.yaml").write_text("required_oracle_tiers: [L2]\n")
    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "L3-ish")
    assert timing.requires_chipyard_timing(descriptor)  # an invalid floor fails closed


@pytest.mark.parametrize("location", ["public", "hidden", "override"])
def test_rtl_requirement_anywhere_retains_measured_timing_gate(functional_corpus, location):
    descriptor, public = functional_corpus
    root = public if location == "public" else public.parent / location
    root.mkdir(exist_ok=True)
    (root / "capsule.yaml").write_text("required_oracle_tiers: [L2, L3]\n")
    kwargs = {"public_root": root} if location == "override" else {}
    assert timing.requires_chipyard_timing(descriptor, **kwargs)


@pytest.mark.parametrize(
    "document",
    [
        "{}",
        "[]",
        "required_oracle_tiers: []",
        "required_oracle_tiers: L2",
        "required_oracle_tiers: [unknown]",
        "required_oracle_tiers: [true]",
        "broken: [",
    ],
)
def test_unknown_or_malformed_corpus_still_requires_timing(functional_corpus, document):
    descriptor, public = functional_corpus
    (public / "capsule.yaml").write_text(document)
    assert timing.requires_chipyard_timing(descriptor)


def test_absent_or_indirect_corpus_retains_timing_requirement(functional_corpus):
    descriptor, public = functional_corpus
    capsule = public / "capsule.yaml"
    contents = capsule.read_text()
    capsule.unlink()
    assert timing.requires_chipyard_timing(descriptor)
    alternate = public.parent / "external.yaml"
    alternate.write_text(contents)
    capsule.symlink_to(alternate)
    assert timing.requires_chipyard_timing(descriptor)
