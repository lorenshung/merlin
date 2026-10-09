"""Independent control admission and actual process-evidence failure checks."""
import time
from dataclasses import fields
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_runtime_qualification as Q
from merlin_experiments.phase2.contracts import StageGateError
from test_component_observer import actual_records


def test_private_control_roster_covers_each_positive_and_negative_mechanism():
    assert len(Q.CONTROL_CASES) == 14
    assert set(Q.CONTROL_CASES) == {
        mechanism + "." + direction
        for mechanism in Q.CONTROL_MECHANISMS for direction in ("positive", "negative")
    }
    assert "instruction_audit.negative" in Q.CONTROL_CASES
    assert "original_output_roster.negative" in Q.CONTROL_CASES


def test_saved_control_report_or_dataclass_cannot_issue_runtime_authority(tmp_path):
    values = {field.name: None for field in fields(Q.IndependentRuntimeQualification)}
    with pytest.raises(StageGateError, match="live executed controls"):
        Q.IndependentRuntimeQualification(**values).verify()
    with pytest.raises(StageGateError, match="runtime|independently prepared"):
        Q.qualify_independent_component_runtime(context=SimpleNamespace(status="qualified"),
                                                evidence_root=tmp_path / "private")


@pytest.mark.parametrize("declared", [None, 0, -1, 601, True, 1.5])
def test_control_processes_cannot_run_without_bounded_budget(declared):
    with pytest.raises(StageGateError, match="bounded remaining budget"):
        Q._bounded_arguments({"timeout": declared}, "timeout", started=time.monotonic(), timeout_s=600)


def test_budget_clamps_actual_call_without_rewriting_prepared_fixture():
    arguments = {"timeout": 600, "inputs": "unchanged"}
    actual = Q._bounded_arguments(arguments, "timeout", started=time.monotonic() - 10, timeout_s=600)
    assert 0 < actual["timeout"] <= 590 and arguments["timeout"] == 600
    assert actual["inputs"] == arguments["inputs"]


def test_control_arguments_bind_actual_values_and_reject_executable_objects(tmp_path):
    original = {"timeout": 600, "input": tmp_path / "original", "labels": {"dev", "public"}}
    before = Q.document_sha256(Q._argument_binding(original))
    original["input"] = tmp_path / "different"
    assert Q.document_sha256(Q._argument_binding(original)) != before
    with pytest.raises(StageGateError, match="unbound executable"):
        Q._argument_binding({"callback": lambda: None})


def test_actual_native_invocations_are_required_and_reopened(tmp_path):
    _selected, source, root, _paths = actual_records(tmp_path)
    fixture = SimpleNamespace(evidence_root=root, required_invocation_stages=("native_object_link",))
    records = Q._invocations(fixture)
    assert len(records) == 2
    fixture.required_invocation_stages = ("unobserved_device_execution",)
    with pytest.raises(StageGateError, match="omitted normal invoked stages"):
        Q._invocations(fixture)
    fixture.required_invocation_stages = ("native_object_link",)
    source.write_text("int main(void) { return 2; }\n")
    with pytest.raises(ValueError, match="changed"):
        Q._invocations(fixture)
