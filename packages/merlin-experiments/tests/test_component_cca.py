"""Host-selected CCA broker round trips; no agent, compiler or hardware launched."""

import copy
import json
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from component_baseline_fixture import unissued_runtime
from merlin_experiments.phase2 import broker
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import component_cca as CC
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from test_component_workflow import context

from merlin.benchharness import hash_tree
from merlin.kernels.cca import CCA, CommunicationFacet, ComputeFacet, CoverageFacet
from merlin.kernels.cca_contract import FACET_CLASSES

pytestmark = pytest.mark.usefixtures("synthetic_component_controller")


def observations(*, baseline, candidate, corpus, target_descriptor, timeout_s):
    assert timeout_s > 0
    root = candidate.parent / "host-private-artifacts"
    root.mkdir(exist_ok=True)
    result = {}
    for member in corpus.capsules:
        pair = []
        for arm, compiler, resident in (("baseline", baseline, True), ("candidate", candidate, False)):
            artifact = root / (member.capsule + "-" + arm + ".asm")
            artifact.write_text("synthetic observed artifact " + arm)
            cca = CCA(
                "contraction",
                ["engine"],
                compute=ComputeFacet(op="contraction", accumulator_resident=resident),
                provenance={"level": "asm", "source": "/private/reference-program.asm"},
            )
            pair.append(
                CC.ComponentCCAObservation(
                    cca,
                    artifact,
                    sha256_file(artifact),
                    hash_tree(compiler)["sha256"],
                    member.source_sha256,
                    corpus.capsules_sha256,
                    sha256_file(target_descriptor),
                )
            )
        result[member.family, member.capsule] = tuple(pair)
    return result


def selected(tmp_path, callback=observations):
    inputs = context(tmp_path)
    baseline = inputs["baseline_admission"].baseline
    runtime = unissued_runtime(target_descriptor=inputs["target_experiment"].path, cca_provider=callback)
    inputs["independent_runtime"] = runtime
    source = Path(__file__)
    provider = CC.ComponentCCAProvider(callback, source, sha256_file(source), baseline,
                                      hash_tree(baseline)["sha256"], inputs["baseline_admission"], runtime)
    return inputs, provider


def execute(inputs, provider):
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **inputs, component_cca=provider)
    engine = broker.Broker(
        SimpleNamespace(),
        inputs["target_experiment"],
        inputs["candidate"],
        policy.build_registry(),
        inputs["receipt_path"],
        workflow=policy,
        deadline=time.monotonic() + 30,
        max_calls=5,
        max_tool_seconds=10,
    )
    return policy, engine.execute({"action": CC.ACTION})


def test_real_broker_all_facets_unknowns_and_private_receipt_roundtrip(tmp_path):
    inputs, provider = selected(tmp_path)
    policy, result = execute(inputs, provider)
    assert result["returncode"] == 0, result
    document = json.loads(result["stdout"])
    row = document["evidence"][0]
    report = row["report"]
    assert document["tier"] == CC.TIER
    assert document["promotion"] == "NO_FINAL_ACCEPTANCE"
    assert {gap["axis"] for gap in report["divergences"]} == {"compute.accumulator_resident"}
    assert set(report["reflected_axes"]) == {
        f"{name}.{field.name}" for name, cls in FACET_CLASSES.items() for field in fields(cls)
    }
    assert report["facets"]["communication"]["status"] == "UNKNOWN"
    assert report["facets"]["coverage"]["status"] == "UNKNOWN"
    assert "communication.host_device_bytes" in report["common_missing_axes"]
    assert report["application_coverage"]["status"] == report["cycle_cost"]["status"] == "UNKNOWN"
    assert "/private/reference" not in result["stdout"]
    assert str(provider.baseline) not in result["stdout"]
    assert str(inputs["component_corpus"].root) not in result["stdout"]
    receipts = [json.loads(line) for line in inputs["receipt_path"].read_text().splitlines()]
    audit = {
        "broker_invocations": [
            {"action": item["action"], "bindings_sha256": item["bindings_command_sha256"]} for item in receipts
        ]
    }
    actions = tuple(action for action in policy.build_registry() if not action.required)
    qualified = policy.verify_receipts(
        inputs["receipt_path"], actions, audit, candidate_sha256=hash_tree(inputs["candidate"])["sha256"]
    )
    assert qualified["feedback_successes"] == 1 and qualified["final_acceptance"] == "NOT_ESTABLISHED"


def test_unavailable_cca_is_advertised_without_granting_authoring(tmp_path):
    inputs, _provider = selected(tmp_path)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **inputs)
    action = next(action for action in policy.build_registry() if action.name == CC.ACTION)
    assert not action.available and "CCA" in action.unavailable_reason
    assert not policy.admission(CC.ACTION).reserved


def projection(scope="kernel", *, communication=None, coverage=None, register=None):
    return CCA(
        "contraction",
        ["engine"],
        ComputeFacet(register_block=register),
        communication=communication,
        coverage=coverage,
        scope=scope,
        provenance={
            "level": "asm",
            "source_sha256": "a" * 64,
            "producer_sha256": "b" * 64,
            "private_metadata_sha256": "c" * 64,
        },
    ).to_dict()


def test_one_sided_unknown_and_register_hint_are_both_visible():
    a = projection(register=(4, 8), communication=CommunicationFacet(host_device_bytes=0))
    b = projection()
    report = CC.complete_report(a, b)
    assert {"axis": "communication", "missing": "candidate"} in report["uncomparable_axes"]
    assert {"axis": "compute.register_block", "missing": "candidate"} in report["uncomparable_axes"]
    assert any(gap["axis"] == "compute.register_block" for gap in report["divergences"])
    assert report["facets"]["communication"]["axes"][0]["status"] == "UNKNOWN"


def test_communication_and_coverage_divergences_use_full_comparator():
    a = projection(
        "program",
        communication=CommunicationFacet(host_device_bytes=0, fences=0),
        coverage=CoverageFacet(claimed_mac_fraction=1.0),
    )
    b = projection(
        "program",
        communication=CommunicationFacet(host_device_bytes=100, fences=1),
        coverage=CoverageFacet(claimed_mac_fraction=0.5),
    )
    report = CC.complete_report(a, b)
    assert {gap["axis"] for gap in report["divergences"]} == {
        "communication.host_device_bytes",
        "communication.fences",
        "coverage.claimed_mac_fraction",
    }
    assert report["application_coverage"]["status"] == "UNKNOWN"


def test_every_populated_reflected_axis_reaches_the_core_comparator():
    def value(annotation, arm):
        args = get_args(annotation)
        if type(None) in args:
            return value(next(kind for kind in args if kind is not type(None)), arm)
        if annotation is bool:
            return bool(arm)
        if annotation in (int, float):
            return annotation(arm + 1)
        if annotation is str:
            return ("baseline", "candidate")[arm]
        if annotation is tuple:
            return (arm + 1, arm + 2)
        assert args and args[1] is Ellipsis
        return (value(args[0], arm),)

    pair = []
    for arm in range(2):
        facets = {
            name: cls(**{key: value(kind, arm) for key, kind in get_type_hints(cls).items()})
            for name, cls in FACET_CLASSES.items()
        }
        pair.append(
            CCA("contraction", ["engine"], **facets, scope="program", provenance=projection()["provenance"]).to_dict()
        )
    report = CC.complete_report(*pair)
    assert {row["axis"] for row in report["divergences"]} == set(report["reflected_axes"])
    assert not report["common_missing_axes"] and not report["uncomparable_axes"]
    assert all(facet["status"] == "COMPARABLE" for facet in report["facets"].values())


def test_exact_multiple_generated_members_are_bound_without_reference_inputs(tmp_path):
    inputs, provider = selected(tmp_path)
    generated = tmp_path / "generated"
    member = generated / "_tuning" / "tail"
    member.mkdir()
    descriptor = json.loads((generated / "_tuning" / "member" / "capsule.yaml").read_text())
    descriptor["name"] = "tail"
    (member / "capsule.yaml").write_text(json.dumps(descriptor))
    manifest = json.loads((generated / "MANIFEST.yaml").read_text())
    manifest["generated"].append("_tuning/tail")
    (generated / "MANIFEST.yaml").write_text(json.dumps(manifest))
    live = C.discover_performance_corpus(
        SimpleNamespace(
            target="fixture", capsule_corpus=generated / "public", graded_roots=lambda: [generated / "public"]
        )
    )
    inputs["component_corpus"] = C.freeze_performance_corpus(live, tmp_path / "frozen-pair")
    # Re-freeze the synthetic baseline's domain before constructing this policy.
    inputs["baseline_admission"] = replace(inputs["baseline_admission"], corpus=inputs["component_corpus"])
    provider = replace(provider, baseline_admission=inputs["baseline_admission"])
    _policy, result = execute(inputs, provider)
    assert result["returncode"] == 0, result
    rows = json.loads(result["stdout"])["evidence"]
    assert {(row["family"], row["capsule"], row["member_sha256"]) for row in rows} == {
        (member.family, member.capsule, member.source_sha256) for member in inputs["component_corpus"].capsules
    }
    assert len(rows) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "scope",
        "invalid-scope",
        "coverage-at-kernel",
        "op",
        "backend",
        "facet-schema",
        "nonfinite",
        "bool-as-count",
        "private-provenance",
        "source-as-facet",
    ],
)
def test_comparison_refuses_scope_identity_schema_and_private_source(failure):
    a, b = projection(), projection()
    if failure == "scope":
        b["scope"] = "dispatch"
    elif failure == "invalid-scope":
        b["scope"] = "unknown"
    elif failure == "coverage-at-kernel":
        b["coverage"] = (
            CoverageFacet().to_dict()
            if hasattr(CoverageFacet(), "to_dict")
            else {field.name: getattr(CoverageFacet(), field.name) for field in fields(CoverageFacet)}
        )
    elif failure == "op":
        b["op"] = "other"
    elif failure == "backend":
        b["backend"] = ["other-engine"]
    elif failure == "facet-schema":
        b["compute"].pop("epilogue")
    elif failure == "nonfinite":
        b["communication"] = {
            field.name: getattr(CommunicationFacet(), field.name) for field in fields(CommunicationFacet)
        }
        b["communication"]["host_device_bytes"] = float("nan")
    elif failure == "bool-as-count":
        b["communication"] = {
            field.name: getattr(CommunicationFacet(), field.name) for field in fields(CommunicationFacet)
        }
        b["communication"]["fences"] = True
    elif failure == "private-provenance":
        b["provenance"]["source"] = "/secret"
    elif failure == "source-as-facet":
        b["compute"]["epilogue"] = "void private_program() { return; }"
    with pytest.raises(StageGateError):
        CC.complete_report(a, b)


def test_missing_populated_axis_from_core_comparator_refuses(monkeypatch):
    a = projection(communication=CommunicationFacet(fences=1))
    b = projection(communication=CommunicationFacet(fences=2))
    monkeypatch.setattr(CC, "compare", lambda *_args, **_kwargs: [])
    with pytest.raises(StageGateError, match="omitted"):
        CC.complete_report(a, b)


def missing(**kwargs):
    return {}


def stale_member(**kwargs):
    raw = observations(**kwargs)
    key = next(iter(raw))
    raw[key] = (replace(raw[key][0], member_sha256="d" * 64), raw[key][1])
    return raw


def stale_artifact(**kwargs):
    raw = observations(**kwargs)
    key = next(iter(raw))
    raw[key][0].artifact.write_text("changed emitted artifact")
    return raw


def mutate_baseline(**kwargs):
    raw = observations(**kwargs)
    (kwargs["baseline"] / "compiler").write_text("changed baseline")
    return raw


def mutate_candidate(**kwargs):
    raw = observations(**kwargs)
    (kwargs["candidate"] / "compiler").write_text("changed candidate")
    return raw


def mutate_corpus(**kwargs):
    raw = observations(**kwargs)
    kwargs["corpus"].manifest_path.write_text("changed corpus")
    return raw


def mutate_target(**kwargs):
    raw = observations(**kwargs)
    kwargs["target_descriptor"].write_text("changed target")
    return raw


@pytest.mark.parametrize(
    "callback", [missing, stale_member, stale_artifact, mutate_baseline, mutate_candidate, mutate_corpus, mutate_target]
)
def test_actual_broker_membership_and_provider_effects_fail_closed(tmp_path, callback):
    inputs, provider = selected(tmp_path, callback)
    _policy, result = execute(inputs, provider)
    assert result["returncode"] == 125 and not result["stdout"]
    assert "/private" not in result["stderr"]


def test_same_file_changed_executing_callable_refuses(tmp_path):
    inputs, provider = selected(tmp_path)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **inputs, component_cca=provider)
    object.__setattr__(provider, "evaluate", missing)
    with pytest.raises(StageGateError, match="callable binding"):
        policy.execute({}, CC.ACTION, {}, 0, 1, time.monotonic())


def test_same_callable_changed_executing_code_refuses(tmp_path):
    inputs, provider = selected(tmp_path)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **inputs, component_cca=provider)
    original = observations.__code__
    try:
        observations.__code__ = missing.__code__
        with pytest.raises(StageGateError, match="callable binding"):
            policy.execute({}, CC.ACTION, {}, 0, 1, time.monotonic())
    finally:
        observations.__code__ = original


def test_component_cca_refuses_untyped_provider_and_overlapping_compiler_arms(tmp_path):
    inputs, provider = selected(tmp_path)
    with pytest.raises(StageGateError, match="typed component CCA"):
        BP.select_workflow(BP.COMPONENT_ONLY_V1, **inputs, component_cca=observations)
    overlapping = replace(
        provider, baseline=inputs["candidate"], baseline_sha256=hash_tree(inputs["candidate"])["sha256"]
    )
    with pytest.raises(StageGateError, match="different baseline"):
        execute(inputs, overlapping)


@pytest.mark.parametrize(
    "failure",
    [
        "unknown-facet-omitted",
        "unknown-axis-omitted",
        "divergence-omitted",
        "scope",
        "corpus",
        "manifest",
        "candidate",
        "configuration",
        "provider",
    ],
)
def test_serialized_feedback_replays_all_facets_and_input_bindings(tmp_path, failure):
    inputs, provider = selected(tmp_path)
    _policy, result = execute(inputs, provider)
    assert result["returncode"] == 0, result
    document = copy.deepcopy(json.loads(result["stdout"]))
    row = document["evidence"][0]
    if failure == "unknown-facet-omitted":
        row["report"]["facets"].pop("coverage")
    elif failure == "unknown-axis-omitted":
        row["report"]["common_missing_axes"].pop()
    elif failure == "divergence-omitted":
        row["report"]["divergences"] = []
    elif failure == "scope":
        row["report"]["scope"] = "program"
    elif failure == "corpus":
        document["corpus_sha256"] = "d" * 64
    elif failure == "manifest":
        document["manifest_sha256"] = "d" * 64
    elif failure == "candidate":
        document["candidate_sha256"] = "d" * 64
    elif failure == "configuration":
        document["configuration_sha256"] = "d" * 64
    elif failure == "provider":
        document["provider_sha256"] = "d" * 64
    with pytest.raises(StageGateError):
        CW.validate_component_feedback(document)


def test_correspondence_reports_users_effects_lifetimes_and_editable_owners():
    a = CC.ComponentCorrespondence(users=("producer.consumer",), effects=("read.immutable",),
                                   lifetimes=(("buffer", 0, 3),),
                                   edit_owners=(("packing", "merlin.llvmlower"),))
    b = replace(a, effects=("read.epoch_mutation",), lifetimes=None)
    report = CC.complete_report(projection(), projection(), a, b)
    assert report["correspondence"]["effects"]["status"] == "DIVERGENCE"
    assert report["correspondence"]["lifetimes"]["status"] == "UNKNOWN"
    assert report["correspondence"]["edit_owners"]["status"] == "SAME"
    assert report["correspondence"]["resource_pressure"]["missing"] == ["baseline", "candidate"]
    with pytest.raises(StageGateError):
        CC.complete_report(projection(), projection(), replace(a, users=("/private/answers",)), b)
