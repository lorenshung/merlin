"""Qualification facets with real native records and isolated synthetic authorities.

Native compilation here tests dependency reopening, not a target runtime role.
The fresh origin, numerical grader and physical stage facets remain synthetic.
"""

import contextlib
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from merlin_experiments.phase1 import component_qualification as Q
from merlin_experiments.phase1 import component_witness as W
from merlin_experiments.phase1 import source_inputs
from merlin_experiments.phase2.component_experiment import RuntimeGrant
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common import invocation_record


@pytest.fixture
def domain(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_grade, capsule_runner

    candidate, contract, corpus = tmp_path / "candidate", tmp_path / "contract", tmp_path / "corpus"
    candidate.mkdir()
    contract.mkdir()
    corpus.mkdir()
    (candidate / "compiler.py").write_text("def lower(): return 'program'\n")
    (candidate / "compiler.c").write_text("int main(void) { return CONTROL_RESULT; }\n")
    (tmp_path / "control.h").write_text("#define CONTROL_RESULT 0\n")
    (contract / "policy.json").write_text('{"quality_budget":"original"}')
    target = tmp_path / "target.yaml"
    target.write_text("target: synthetic\n")
    runtime = tmp_path / "runtime"
    runtime.write_text("selected independent execution runtime\n")
    grant = RuntimeGrant(runtime, "/usr/bin/runtime", hashlib.sha256(runtime.read_bytes()).hexdigest())
    report = {
        "sha256": "1" * 64,
        "obligations": [
            {
                "id": cohort,
                "mandatory": True,
                "cohort": cohort,
                "expectation": "admitted_program",
                "members": [{"name": cohort, "member": "cases/" + cohort}],
            }
            for cohort in ("functional_guard", "withheld_transfer")
        ],
    }
    for row in report["obligations"]:
        member = row["members"][0]
        directory = corpus / member["member"]
        directory.mkdir(parents=True)
        source = directory / "source.mlir"
        source.write_text("module {}\n")
        (directory / "capsule.yaml").write_text(
            json.dumps(
                {
                    "name": member["name"],
                    "interface_mlir": "source.mlir",
                    "component_coverage": {"generated_effects": []},
                }
            )
        )
        member.update(
            program_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), sha256="3" * 64, output_roster=["output"]
        )

    def stage_verifier(**kwargs):
        directory = kwargs["result_path"].parent
        source = kwargs["capsule_root"] / "source.mlir"
        stages = [W.ComponentStageFile("source", source, hashlib.sha256(source.read_bytes()).hexdigest(), ())]
        parents = [
            ("prepared_ir", ("source",)),
            ("partition", ("prepared_ir",)),
            ("emitted_host", ("partition",)),
            ("emitted_device", ("partition",)),
            ("object", ("emitted_host", "emitted_device")),
            ("elf", ("object",)),
            ("execution", ("elf",)),
            ("outputs", ("execution",)),
            ("effects", ("execution",)),
        ]
        for stage, owners in parents:
            path = directory / (stage + ".witness")
            path.write_text("evaluated synthetic stage " + stage)
            index = {row.stage: row for row in stages}
            stages.append(
                W.ComponentStageFile(
                    stage,
                    path,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    tuple((parent, index[parent].sha256) for parent in owners),
                )
            )
        return W.ComponentStageWitness(
            kwargs["member"]["program_sha256"],
            kwargs["candidate_sha256"],
            kwargs["member"]["sha256"],
            hashlib.sha256(target.read_bytes()).hexdigest(),
            "mlir",
            tuple(stages),
            ("output",),
            kwargs["required_effects"],
            "original_independent_golden",
        )

    owner = ModuleType("merlin_experiments.phase0.component_coverage")
    owner.verify_report = lambda root: report
    owner.build_guard_link = lambda value: {"report_sha256": value["sha256"]}
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setattr(source_inputs, "record", lambda **kw: {"version": 1, "synthetic_source_closure": "closed"})
    monkeypatch.setattr(source_inputs, "verify", lambda *a, **kw: None)
    monkeypatch.setattr(Q, "_component_sources", lambda: {"synthetic_owned_sources": "frozen"})
    monkeypatch.setattr(Q, "_evaluator_distribution", lambda: {"synthetic_installed_evaluator": "frozen"})
    monkeypatch.setattr(Q, "verify_component_view", lambda _view: None)
    monkeypatch.setattr(Q, "qualified_package_execution", lambda **_kwargs: contextlib.nullcontext())
    # This suite isolates grader facets with synthetic observations. It does not
    # issue fresh-author authority or prove admission of an actual experiment.
    monkeypatch.setattr(
        Q, "_verify_origin", lambda *_args: {"phase1_origin_sha256": "4" * 64, "phase2_lineage_sha256": None}
    )
    monkeypatch.setattr(Q, "_verify_runtime", lambda *_args: "5" * 64)
    monkeypatch.setattr(Q, "selected_compiler_transport", lambda *_args, **_kwargs: None)
    # Static-role admission is a separate isolated facet here. These synthetic
    # numerical/witness controls never issue an experiment's source-only proof.
    monkeypatch.setattr(Q, "qualification_compile_roles", lambda *_args, **_kwargs: ({"unit_facet": "isolated"}, []))
    support = SimpleNamespace(
        grader=lambda *args, **kwargs: capsule_grade.grade(*args, **kwargs),
        stage_verifier=stage_verifier,
        verify=lambda **kwargs: None,
    )
    monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *a: {"selected": object()})
    values = dict(
        candidate=candidate,
        corpus_root=corpus,
        target_experiment=SimpleNamespace(path=target, target="synthetic", sim_via="selected"),
        source_root=tmp_path,
        contract_root=contract,
        evidence_root=tmp_path / "evidence",
        runtime=(grant,),
        view=Q.ComponentView(tmp_path / "synthetic_public_view", "1" * 64, "2" * 64, "3" * 64),
        runtime_authority=support,
    )
    return values, capsule_grade, report


def _grade(domain, monkeypatch, *, numeric="pass", cert_verdict=None, write_records=True, observe=True):
    values, engine, report = domain
    calls = []

    def execute(package, **kwargs):
        calls.append(kwargs)
        assert package != values["candidate"]
        expected = [values["corpus_root"] / row["members"][0]["member"] for row in report["obligations"]]
        assert kwargs["capsules_root"] == expected and kwargs["labels"] == {"public", "hidden"}
        rows = [
            {"capsule": row["cohort"], "status": "pass", "numeric": numeric, "cert_verdict": cert_verdict}
            for row in report["obligations"]
        ]
        if write_records:
            for row in rows:
                path = kwargs["runs_root"] / row["capsule"]
                path.mkdir(parents=True)
                (path / "capsule_result.json").write_text(json.dumps(row))
                if observe:
                    compiler = shutil.which("cc")
                    if compiler is None:
                        pytest.skip("native control needs an installed C compiler")
                    source, header = package / "compiler.c", values["source_root"] / "control.h"
                    executable = path / "executed.elf"
                    invocation_record.run(
                        [compiler, str(source), "-include", str(header), "-o", str(executable)],
                        directory=path,
                        stage="private_native_control_compile",
                        inputs=(source, header),
                        outputs=(executable,),
                        dependencies=(package / "compiler.py",),
                        check=True,
                        capture_output=True,
                        timeout=20,
                    )
                    invocation_record.run(
                        [str(executable)],
                        directory=path,
                        stage="private_native_control_execute",
                        inputs=(executable,),
                        check=True,
                        capture_output=True,
                        timeout=20,
                    )
                (path / "full-output.json").write_text('{"every_output":[1,2,3]}')
        return {"per_capsule": rows, "integrity_status": "clean"}

    monkeypatch.setattr(engine, "grade", execute)
    return calls


def test_actual_full_domain_grade_qualifies_and_output_mutation_refuses(domain, monkeypatch):
    calls = _grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "qualified" and len(calls) == 1
    qualification.verify()
    output = next((qualification.receipt.parent / "grade").rglob("full-output.json"))
    output.write_text('{"only_one_output":[1]}')
    with pytest.raises(StageGateError, match="output evidence"):
        qualification.verify()


def test_numerical_domain_cannot_bypass_missing_source_only_evaluation(domain, monkeypatch):
    from merlin_experiments.phase1.component_compile_admission import qualification_compile_roles

    calls = _grade(domain, monkeypatch)
    monkeypatch.setattr(Q, "qualification_compile_roles", qualification_compile_roles)
    with pytest.raises(StageGateError, match="evaluated original source-only roles"):
        Q.qualify_component_compiler(**domain[0])
    assert calls == [] and not domain[0]["evidence_root"].exists()


def test_numerical_pass_preserves_unresolved_original_static_denominator(domain, monkeypatch):
    calls = _grade(domain, monkeypatch)
    failures = ["required source-only role unresolved: unseen-large:resource_legality"]
    binding = {"static_denominator": {"required": 4, "proved": 0, "unknown": 4}}
    monkeypatch.setattr(Q, "qualification_compile_roles", lambda *_args, **_kwargs: (binding, failures))
    qualification = Q.qualify_component_compiler(**domain[0])
    document = Q.C.mapping_file(qualification.receipt)
    assert len(calls) == 1 and qualification.status == "refused"
    assert document["failures"] == failures and document["compile_roles"] == binding
    with pytest.raises(StageGateError, match="no evaluated domain qualification"):
        qualification.verify()


def test_private_compiler_drift_refuses_with_original_candidate_and_grade_unchanged(domain, monkeypatch):
    _grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    qualification.verify()
    evidence = qualification.receipt.parent
    candidate_before = Q.C.exact_tree_record(qualification.candidate)["sha256"]
    grade_before = Q.C.exact_tree_record(evidence / "grade")["sha256"]
    clone = evidence / "compiler" / "compiler.c"
    clone.write_text("int main(void) { return 1; }\n")
    assert Q.C.exact_tree_record(qualification.candidate)["sha256"] == candidate_before
    assert Q.C.exact_tree_record(evidence / "grade")["sha256"] == grade_before
    record = next(
        path
        for path in (evidence / "grade").rglob("invocation.json")
        if json.loads(path.read_bytes())["stage"] == "private_native_control_compile"
    )
    with pytest.raises(ValueError, match="dependency or product changed"):
        invocation_record.verify(record)
    with pytest.raises(StageGateError, match="compiler snapshot changed"):
        qualification.verify()


def test_external_actual_invocation_input_drift_is_reopened(domain, monkeypatch):
    _grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    qualification.verify()
    grade_before = Q.C.exact_tree_record(qualification.receipt.parent / "grade")["sha256"]
    (domain[0]["source_root"] / "control.h").write_text("#define CONTROL_RESULT 1\n")
    assert Q.C.exact_tree_record(qualification.receipt.parent / "grade")["sha256"] == grade_before
    with pytest.raises(StageGateError, match="actual execution evidence changed"):
        qualification.verify()


def test_numeric_pass_without_actual_invocation_records_refuses(domain, monkeypatch):
    _grade(domain, monkeypatch, observe=False)
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "refused"
    assert any("lacks actual invocation" in row for row in Q.C.mapping_file(qualification.receipt)["failures"])


def test_every_original_member_requires_actual_invocations(domain, monkeypatch):
    _grade(domain, monkeypatch)
    original = domain[1].grade

    def incomplete(package, **kwargs):
        score = original(package, **kwargs)
        shutil.rmtree(kwargs["runs_root"] / "withheld_transfer" / "invocations")
        return score

    monkeypatch.setattr(domain[1], "grade", incomplete)
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "refused"
    assert any(
        "withheld_transfer" in row and "invocation" in row
        for row in Q.C.mapping_file(qualification.receipt)["failures"]
    )


def test_saved_stage_witness_replays_current_original_member_and_effects(domain, monkeypatch):
    _grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    qualification.verify()
    member = domain[2]["obligations"][0]["members"][0]
    capsule = domain[0]["corpus_root"] / member["member"] / "capsule.yaml"
    original = capsule.read_bytes()
    declaration = json.loads(original)
    declaration["component_coverage"]["generated_effects"] = ["new_independent_effect"]
    capsule.write_text(json.dumps(declaration))
    with pytest.raises(StageGateError, match="omits declared effects"):
        qualification.verify()
    capsule.write_bytes(original)
    member["sha256"] = "6" * 64
    with pytest.raises(StageGateError, match="authority differs"):
        qualification.verify()


@pytest.mark.parametrize(
    "numeric,certification,records",
    [("fail", None, True), ("pass", {"would_be_status": "incomplete"}, True), ("pass", None, False)],
)
def test_numeric_missing_execution_or_uncertified_pass_cannot_qualify(
    domain, monkeypatch, numeric, certification, records
):
    _grade(domain, monkeypatch, numeric=numeric, cert_verdict=certification, write_records=records)
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "refused"
    with pytest.raises(StageGateError):
        replace(qualification, status="qualified").verify()


def test_qualification_refuses_missing_withheld_domain_before_execution(domain, monkeypatch):
    domain[2]["obligations"].pop()
    _grade(domain, monkeypatch)
    with pytest.raises(StageGateError, match="withheld transfer"):
        Q.qualify_component_compiler(**domain[0])


def test_numerical_success_without_import_link_effect_witnesses_refuses(domain, monkeypatch):
    _grade(domain, monkeypatch)
    monkeypatch.setattr(
        domain[0]["runtime_authority"],
        "stage_verifier",
        lambda **kwargs: (_ for _ in ()).throw(StageGateError("missing source-to-executable producer")),
    )
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "refused"
    assert any("witnesses UNKNOWN" in reason for reason in json.loads(qualification.receipt.read_bytes())["failures"])


def test_added_or_changed_ordinary_implementation_source_invalidates_qualification(domain, monkeypatch):
    _grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    monkeypatch.setattr(Q, "_component_sources", lambda: {"synthetic_owned_sources": "changed"})
    with pytest.raises(StageGateError, match="source membership changed"):
        qualification.verify()


def test_declared_unsupported_case_consumes_actual_refusal_record_not_missing_score_plane(domain, monkeypatch):
    domain[2]["obligations"][0]["expectation"] = "unsupported_program"
    _grade(domain, monkeypatch)
    grade = domain[1].grade

    def actual_refusal(package, **kwargs):
        score = grade(package, **kwargs)
        row = score["per_capsule"][0]
        row.update(status="declined", numeric="skipped")
        assert "failure_plane" not in row  # ordinary projected score schema
        path = kwargs["runs_root"] / row["capsule"] / "capsule_result.json"
        path.write_text(
            json.dumps(
                {
                    "capsule": row["capsule"],
                    "status": "declined",
                    "failure": {"plane": "backend_declined"},
                    "declined": {"reason": "explicit independently declared unsupported source"},
                }
            )
        )
        return score

    monkeypatch.setattr(domain[1], "grade", actual_refusal)
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "qualified"


def test_nonzero_native_observation_cannot_be_relabelled_as_an_original_refusal(domain, monkeypatch):
    domain[2]["obligations"][0]["expectation"] = "unsupported_program"
    _grade(domain, monkeypatch)
    grade = domain[1].grade

    def relabelled_crash(package, **kwargs):
        score = grade(package, **kwargs)
        row = score["per_capsule"][0]
        owner = kwargs["runs_root"] / row["capsule"]
        result = invocation_record.run(
            [sys.executable, "-I", "-B", "-c", "raise SystemExit(1)"],
            directory=owner,
            stage="private_observed_failure",
            dependencies=(package / "compiler.py",),
            capture_output=True,
            timeout=20,
        )
        assert result.returncode != 0
        row.update(status="declined", numeric="skipped")
        (owner / "capsule_result.json").write_text(
            json.dumps(
                {
                    "capsule": row["capsule"],
                    "status": "declined",
                    "failure": {"plane": "backend_declined"},
                    "declined": {"reason": "relabelled failed tool"},
                }
            )
        )
        return score

    monkeypatch.setattr(domain[1], "grade", relabelled_crash)
    qualification = Q.qualify_component_compiler(**domain[0])
    assert qualification.status == "refused"
    assert any("invocation did not complete" in row for row in Q.C.mapping_file(qualification.receipt)["failures"])
