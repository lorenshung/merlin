"""Normal Phase 0 writes independent synthetic contracts through to frozen Phase 2.

No generated capsules/goldens are authored in these tests and no simulator runs.
The hardware facts are explicit synthetic declarations, not qualified RTL facts.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import component_generation as G
from merlin_experiments.phase0 import generation
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2.contracts import StageGateError

from merlin.runtime.backends import base
from merlin.targetgen import phase_policy as PP
from merlin.targetgen import readout_facet, target_registry
from merlin.targetgen.rtl import facts


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if path.suffix == ".json" else yaml.safe_dump(value))
    return path


@pytest.fixture
def independent(tmp_path, monkeypatch):
    provider = tmp_path / "support"
    contract_document = {
        "name": "fixture",
        "capabilities": {"mesh": {"rows": 2, "cols": 2}},
        "encoding": {
            "semantic_class": {"a": "MVIN", "b": "COMPUTE", "c": "MVOUT"},
            "corpus_issue_order": ["MVIN", "COMPUTE", "MVOUT"],
        },
        "compute_units": [
            {
                "name": "array",
                "kind": "systolic",
                "dtypes": ["int8"],
                "accumulate": [{"acc": "i32"}],
                "ops": ["matmul", "movement"],
                "semantic_capabilities": [
                    {"family": "contraction", "dtypes": ["int8"], "ranks": [2]},
                    {
                        "family": "movement",
                        "dtypes": ["int8"],
                        "result_dtypes": ["i32"],
                        "forms": ["copy"],
                        "layouts": ["row_major_contiguous"],
                    },
                ],
            }
        ],
    }
    contract = write(provider / "contracts/target_contract.yaml", contract_document)
    (provider / "backend.py").write_text("# Synthetic selected source owner, no executable backend.\n")
    raw = write(
        tmp_path / "facts.json",
        {"facts": {"arrays": [{"name": "mesh", "rows": 2, "cols": 2}], "memories": [{"name": "scratch"}]}},
    )
    info = target_registry.TargetInfo(
        name="fixture",
        kind="external",
        base=provider,
        contract_path=contract,
        dialect_plan_path=provider / "contracts/dialect_plan.yaml",
        facts_path=raw,
        backend="fixture",
    )
    monkeypatch.setattr(target_registry, "resolve", lambda target: info)
    monkeypatch.setattr(facts, "find_facts", lambda target, explicit=None: raw)
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **k: pytest.fail("test must not extract RTL"))
    monkeypatch.setattr(readout_facet, "capture_inputs", lambda *a, **k: {"scalar_abi": None, "readouts": None})
    monkeypatch.setattr(base, "execution_capability_facts", lambda target: {})
    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda descriptor: None)
    # The hardware identity uses the same canonical selected-contract format as
    # EvidenceSelection. All other generation/evidence bodies execute normally.
    hardware = {
        "contract_sha256": hashlib.sha256(
            (json.dumps(contract_document, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest(),
        "raw_facts_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }
    software_document = {
        "schema": "merlin.software_spec.v1",
        "target": "fixture",
        "status": "reviewed",
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
            "overflow": "wrap_internal_mac",
        },
        "operations": {
            "movement": {"placement": "accelerator", "dtypes": ["int8"], "layouts": ["row_major_contiguous"]},
            "contraction": {"placement": "accelerator", "dtypes": ["int8"], "ranks": [2]},
        },
        "component_performance": {
            "schema": G.DECLARATION,
            "status": "reviewed",
            "hardware": hardware,
            "objectives": [
                {
                    "family": "transfer",
                    "operations": ["movement"],
                    "objective": {
                        "metric": "complete_component_cycles",
                        "unit": "cycles",
                        "direction": "min",
                        "basis": "all input preparation, load/store, readout and output publication",
                    },
                }
            ],
        },
    }
    software = write(tmp_path / "software.yaml", software_document)
    recipe = write(
        tmp_path / "recipe.yaml", {"software_spec": str(software), "datapath": {"required_oracle_tiers": ["L0"]}}
    )
    performance = {
        "level": "L1_separation_floor",
        "family": "transfer",
        "lever": "transfer scheduling",
        "member_class": "OBJECTIVE",
        "claim": "DIFFERENTIAL",
        "comparand": {
            "kind": "candidate",
            "against": "same generated input",
            "cancels": "input bytes",
            "demand_equal": "full output",
        },
        "falsifier": {
            "observation": "complete component cycles",
            "fires_when": "candidate fails declared comparison",
            "negative_control": "unchanged compiler",
        },
        "gate": {
            "traits": ["managed_scratchpad"],
            "instrument": "component counter",
            "capacity": "declared scratch",
            "on_missing": "skip_with_evidence",
        },
        "regime": {"separation": "same boundaries", "layout": "contiguous"},
        "emitter": {"status": "existing", "entry": "movement", "knobs": {}},
        "cost": {
            "tier": "analytic_or_component_rtl",
            "runs": 1,
            "projected_cycles": "unbounded",
            "basis": "unmeasured declaration",
        },
    }
    template = write(
        tmp_path / "performance.yaml",
        {
            "sweeps": [
                {
                    "id": "transfer",
                    "name": "transfer_{M}_{N}",
                    "axes": {"M": [2, 3], "N": [2, 5]},
                    "fit_axes": ["M", "N"],
                    "base": {
                        "kind": "isa",
                        "cat": "_perf",
                        "op": "movement",
                        "label": "dev",
                        "performance": performance,
                    },
                }
            ]
        },
    )
    baseline = tmp_path / "baseline/isa"
    baseline.mkdir(parents=True)
    descriptor = write(
        tmp_path / "target.yaml",
        {"target": "fixture", "backend_package_dir": str(provider), "capsule_corpus": str(baseline)},
    )
    return {
        "descriptor": descriptor,
        "recipe": recipe,
        "performance_template": template,
        "software_spec": software,
        "capability_contract": contract,
        "rtl_facts": raw,
        "output_root": tmp_path / "generated",
        "evidence_mode": "diagnostic",
        "component_only": True,
    }


def live(options):
    return C.discover_performance_corpus(
        SimpleNamespace(target="fixture", capsule_corpus=options["output_root"] / "isa", graded_roots=lambda: [])
    )


def test_normal_generator_freeze_and_zero_mac_phase_admission(independent, tmp_path):
    written = generation.generate_target("fixture", **independent)
    assert len(written) == 4
    assert all(path.parent.name == "_perf" for path in written)
    emitted = yaml.safe_load((written[0] / "golden.yaml").read_bytes())
    assert emitted["golden_source"] == "merlin_tensor_int" and emitted["outputs"]
    corpus = live(independent)
    objectives = C.performance_objectives(corpus)
    assert set(objectives) == {member.capsule for member in corpus.capsules}
    for member in corpus.capsules:
        assert PP.declared_macs(member.descriptor)[0] == 0
        assert PP.priceable(member.descriptor).value == PP.NO
        assert PP.priceable(member.descriptor, performance_objective=objectives[member.capsule]).value == PP.YES
        assert (
            PP.phase_of(
                member.descriptor,
                target="fixture",
                cycle_accurate_available=None,
                performance_objective=objectives[member.capsule],
            ).cert.value
            == PP.UNKNOWN
        )
    frozen = C.freeze_performance_corpus(corpus, tmp_path / "frozen")
    loaded = C.load_frozen_performance_corpus(
        frozen.root,
        manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256,
        expected_target="fixture",
    )
    assert C.performance_objectives(loaded) == objectives
    descriptors = [member.descriptor for member in loaded.capsules]
    unknown_split = PP.split_report(
        descriptors,
        target="fixture",
        cycle_accurate_available=None,
        performance_objectives=C.performance_objectives(loaded),
    )
    assert unknown_split["counts"][PP.UNDETERMINED] == 4
    assert all(
        verdict.price.value == PP.YES and verdict.cert.value == PP.UNKNOWN for verdict in unknown_split["verdicts"]
    )
    # No certification history is supplied by a training corpus. The same
    # explicit objective map makes all four obligations visible, unanchored.
    anchors = PP.anchors(
        descriptors,
        target="fixture",
        cycle_accurate_available=None,
        roots=[],
        performance_objectives=C.performance_objectives(loaded),
    )
    assert anchors["n_orphaned"] == 4 and anchors["n_paired"] == anchors["n_self_certified"] == 0
    assert PP.anchors(descriptors, target="fixture", cycle_accurate_available=None, roots=[])["n_orphaned"] == 0
    from merlin_experiments.phase2 import broker_policy as BP

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler").write_text("# synthetic host-owned compiler\n")
    write(
        candidate / "manifest.yaml",
        {"entrypoints": {"tool": "compiler"}, "commands": {"parse": {"argv": ["{tool}", "{input}", "{output}"]}}},
    )
    policy = BP.select_workflow(
        BP.COMPONENT_ONLY_V1,
        candidate=candidate,
        target_experiment=SimpleNamespace(target="fixture", path=independent["descriptor"]),
        receipt_path=tmp_path / "receipts.jsonl",
        component_corpus=loaded,
    )
    assert policy.workflow_id == BP.COMPONENT_ONLY_V1
    assert not any(action.name.startswith("whole-model") for action in policy.build_registry())
    identity = corpus.performance_generation["component_generation"]
    assert identity["whole_coverage_verified"] is False and identity["timing_verified"] is False
    assert identity["evidence_status"] != "verified"
    guards = json.loads((independent["output_root"] / "_evidence/coverage/phase2-functional-guards.json").read_bytes())
    assert guards["status"] == "not_established" and guards["guards"] == []
    assert identity["recipe"]["sha256"] == hashlib.sha256(independent["recipe"].read_bytes()).hexdigest()
    assert all(Path(pin["path"]).is_file() for pin in identity["generator_sources"])
    # Every current owner source is present, including this admission module.
    assert any(Path(pin["path"]).name == "component_generation.py" for pin in identity["generator_sources"])


@pytest.mark.parametrize(
    "field", ["conformance_spec", "synth_profile", "smt_profile", "hidden_profile", "evidence_input"]
)
def test_model_and_hidden_inputs_refuse_before_read(independent, monkeypatch, field):
    forbidden = Path("/unreadable/no-model-input")
    monkeypatch.setattr(generation, "load_profile", lambda *a, **k: pytest.fail("must refuse before profile loading"))
    with pytest.raises(ValueError, match="refuses capture"):
        generation.generate_target("fixture", **{**independent, field: forbidden})
    assert not independent["output_root"].exists()


@pytest.mark.parametrize("field", ["workload_spec", "claim_boundary", "grading"])
def test_nonindependent_descriptor_refuses(independent, field):
    document = yaml.safe_load(independent["descriptor"].read_bytes())
    document[field] = {"not_training_input": True}
    write(independent["descriptor"], document)
    with pytest.raises(ValueError, match="independent descriptor"):
        generation.generate_target("fixture", **independent)
    assert not independent["output_root"].exists()


@pytest.mark.parametrize(
    "change",
    ["hardware", "unreviewed", "unknown_owner", "wrong_owner", "duplicate", "authored_provenance", "no_objective"],
)
def test_invalid_objective_declarations_fail_closed(independent, change):
    document = yaml.safe_load(independent["software_spec"].read_bytes())
    declaration = document["component_performance"]
    if change == "hardware":
        declaration["hardware"]["raw_facts_sha256"] = "0" * 64
    elif change == "unreviewed":
        declaration["status"] = "unreviewed"
    elif change == "unknown_owner":
        declaration["objectives"][0]["operations"] = ["missing"]
    elif change == "wrong_owner":
        declaration["objectives"][0]["operations"] = ["contraction"]
    elif change == "duplicate":
        declaration["objectives"].append(deepcopy(declaration["objectives"][0]))
    elif change == "authored_provenance":
        declaration["objectives"][0]["objective"]["provenance"] = ["not a generated proof"]
    else:
        declaration["objectives"] = []
    write(independent["software_spec"], document)
    reasons = {
        "hardware": "differs from selected hardware",
        "unreviewed": "explicit reviewed component_performance",
        "unknown_owner": "unique reviewed software operation",
        "wrong_owner": "owner does not declare the generated operation",
        "duplicate": "unknown or duplicate generated family",
        "authored_provenance": "provenance is derived",
        "no_objective": "failed to generate",
    }
    with pytest.raises((ValueError, RuntimeError), match=reasons[change]):
        generation.generate_target("fixture", **independent)
    if change == "no_objective":
        with pytest.raises(StageGateError):
            live(independent)


def test_unknown_work_is_not_repaired_by_an_objective():
    document = {
        "software_screen": {"status": "admitted"},
        "operation": {"op": "not_a_priced_family"},
        "performance": {
            "objective": PP.PerformanceObjective("cycles", "cycles", "min", "complete scope", ("explicit",)).to_dict()
        },
    }
    with pytest.raises(ValueError, match="work admission refused"):
        G.require_written(document)


def test_previous_history_corpus_cannot_be_a_generation_input(independent, monkeypatch):
    independent["output_root"].mkdir()
    monkeypatch.setattr(generation, "load_profile", lambda *a, **k: pytest.fail("must not load previous history"))
    with pytest.raises(ValueError, match="fresh output_root"):
        generation.generate_target("fixture", **independent)


def test_generated_objective_source_and_binding_tamper_refuse(independent):
    generation.generate_target("fixture", **independent)
    corpus = live(independent)
    first = corpus.capsules[0]
    first.descriptor["performance"]["objective"]["basis"] = "changed in-memory binding"
    with pytest.raises(StageGateError, match="descriptor changed"):
        C.performance_objectives(corpus)
    first.descriptor["performance"]["objective"]["basis"] = (
        "all input preparation, load/store, readout and output publication"
    )
    objective = first.source_dir / "capsule.yaml"
    source = yaml.safe_load(objective.read_bytes())
    source["performance"]["component_generation_sha256"] = "0" * 64
    write(objective, source)
    with pytest.raises(StageGateError, match="identity"):
        live(independent)


def test_cli_and_normal_adapter_forward_explicit_component_option(independent, monkeypatch):
    from merlin_experiments import adapters
    from merlin_experiments.phase0 import __main__ as cli

    calls = []
    monkeypatch.setattr(cli, "generate_target", lambda target, **kwargs: calls.append((target, kwargs)) or [])
    argv = ["--target", "fixture", "--component-only"]
    for key, value in independent.items():
        if key != "component_only":
            argv += ["--" + key.replace("_", "-"), str(value)]
    assert cli.main(argv) == 0
    assert calls == [("fixture", independent)]
    config = {key: str(value) for key, value in independent.items() if key not in {"output_root", "component_only"}}
    config["component_only"] = True
    adapters.ADAPTERS["capsule_derivation"].validate(config)
    with pytest.raises(adapters.SpecError, match="hidden inputs"):
        adapters.ADAPTERS["capsule_derivation"].validate({**config, "hidden_profile": "/do-not-read"})
