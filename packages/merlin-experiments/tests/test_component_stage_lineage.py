"""The stage carries observed lineage into requalification; fixtures grant no authority."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_launch as CL
from merlin_experiments.phase2.contracts import StageGateError


def test_stage_requalifies_only_the_observed_descendant(tmp_path, monkeypatch):
    candidate, stage = tmp_path / "candidate", tmp_path / "stage"
    candidate.mkdir()
    stage.mkdir()
    (candidate / "compiler.py").write_text("# synthetic transport fixture\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    target = SimpleNamespace(path=descriptor, target="synthetic")
    corpus = SimpleNamespace(manifest_sha256="1" * 64, capsules_sha256="2" * 64, capsules=())
    origin = SimpleNamespace(receipt_sha256="3" * 64)
    runtime_authority = object()
    qualification = SimpleNamespace(
        corpus_root=tmp_path / "corpus",
        contract_root=tmp_path / "contract",
        source_root=tmp_path,
        runtime=(),
        compiler_origin=origin,
        runtime_authority=runtime_authority,
    )

    def policy(**kwargs):
        return SimpleNamespace(**kwargs, workflow_id="synthetic-transport", build_registry=lambda: (),
                               verify_receipts=lambda *a, **kw: {})

    initial = policy(
        candidate=candidate,
        target_experiment=target,
        receipt_path=stage / "probe/receipts.jsonl",
        component_corpus=corpus,
        component_analytical=object(),
        component_cca=None,
        component_rtl=None,
        services=None,
    )
    inputs = CL.ComponentLaunchInputs(
        SimpleNamespace(root=tmp_path / "view"),
        candidate,
        initial,
        object(),
        qualification,
        (),
        (SimpleNamespace(destination="/usr/bin/bwrap", source=Path("/usr/bin/bwrap"), verify=lambda: None),),
        (),
        Path("/usr/bin/client"),
        "/usr/bin/client",
        tmp_path / "private_auth",
        stage,
        tmp_path / "prices.json",
    )
    launch = CL.QualifiedComponentLaunch(inputs, tmp_path / "qualification.json", "4" * 64, "fixture", "low", object())
    monkeypatch.setattr(CL.QualifiedComponentLaunch, "verify", lambda self: None)
    monkeypatch.setattr(CL.ComponentLaunchInputs, "verify", lambda self, **kwargs: None)
    monkeypatch.setattr(CL, "ComponentOnlyPolicy", policy)
    monkeypatch.setattr(CL, "strict_tool_policy", lambda *args, **kwargs: ())
    monkeypatch.setattr(CL.SI, "prepare_component_prompt_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(CL.SP, "render_component_prompt", lambda *args: "synthetic transport control")
    monkeypatch.setattr(CL.B, "stage_broker_shim", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        CL.B,
        "Broker",
        lambda *args, **kwargs: SimpleNamespace(token="fixture", serving=lambda **kwargs: contextlib.nullcontext()),
    )
    monkeypatch.setattr(CL.TEL, "prepare", lambda **kwargs: {})
    monkeypatch.setattr(CL.TEL, "collect_round", lambda *args, **kwargs: {})
    monkeypatch.setattr(CL.TEL, "finalize", lambda *args, **kwargs: {})
    monkeypatch.setattr(CL, "audit_codex_transcript", lambda *args: {"clean": True})
    observed, lineage = [], SimpleNamespace(receipt_sha256="5" * 64)

    def author(actual_launch, authored_inputs, **kwargs):
        assert actual_launch is launch and authored_inputs.qualification is qualification
        assert authored_inputs.policy.receipt_path == stage / "control-authoring/receipts.jsonl"
        transcript = stage / "synthetic_transcript.json"
        transcript.write_text("{}\n")
        observed.append("author")
        return 0, transcript, lineage

    def requalify(actual_candidate, **kwargs):
        assert actual_candidate == candidate
        assert kwargs["compiler_origin"] is origin
        assert kwargs["compiler_lineage"] is lineage
        assert kwargs["runtime_authority"] is runtime_authority
        observed.append("qualify")
        return SimpleNamespace(receipt_sha256="6" * 64, verify=lambda: observed.append("verify"))

    monkeypatch.setattr(CL, "run_component_origin_round", author)
    monkeypatch.setattr(CL, "qualify_component_compiler", requalify)
    record = CL.run_component_stage(
        launch,
        model="fixture",
        effort="low",
        wall_budget_seconds=10,
        max_tool_calls=2,
        tool_timeout_seconds=10,
        suite="synthetic-transport",
    )
    document = CL.C.mapping_file(record)
    assert observed == ["author", "qualify", "verify"]
    assert document["phase1_origin_sha256"] == origin.receipt_sha256
    assert document["phase2_lineage_sha256"] == lineage.receipt_sha256
    assert document["admission"]["consumable"] is True
    assert document["final_acceptance"] == "NOT_ESTABLISHED"

    def refused_author(*args, **kwargs):
        raise StageGateError("observed descendant refused")

    monkeypatch.setattr(CL, "run_component_origin_round", refused_author)
    with pytest.raises(StageGateError, match="descendant refused"):
        CL.run_component_stage(
            launch,
            model="fixture",
            effort="low",
            wall_budget_seconds=10,
            max_tool_calls=2,
            tool_timeout_seconds=10,
            suite="synthetic-transport",
        )
    assert observed == ["author", "qualify", "verify"]
