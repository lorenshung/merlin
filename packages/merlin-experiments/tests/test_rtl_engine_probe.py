"""Actual harmless native controls test the generic observer, never target RTL."""
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.rtl_engine_probe import probe_rtl_engine
from merlin_experiments.phase2.rtl_engine_protocol import (
    RtlProbeControl,
    RtlProbeStep,
    check_original_outputs,
    prepare_rtl_engine_selection,
)


def selection(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("actual native C compiler unavailable")
    support = tmp_path / "independent-support.c"
    support.write_text("/* Diagnostic support selection, not RTL/runtime qualification. */\n")
    tool = Path(compiler).resolve()
    return prepare_rtl_engine_selection(tools=(tool,), source_files=(support,)), tool


def native_control(compiler, *, id="clock_reset", contract="clock_reset", expected=3, source=None):
    source = source or b'#include <stdio.h>\nint main(void) { puts("a,b\\n3,4"); return 0; }\n'
    return RtlProbeControl(id, contract, (("control.c", source),), (
        RtlProbeStep("compile", (str(compiler), "@WORK@/control.c", "-o", "@WORK@/control"),
                     ("control.c",), ("control",)),
        RtlProbeStep("observe", ("@WORK@/control",), ("control",), ()),
    ), ("a", "b"), ((expected, 4),))


def run(tmp_path, selected, *controls):
    return probe_rtl_engine(selection=selected, controls=controls,
                            evidence_root=tmp_path / "private-probe", environment={"PATH": "/usr/bin:/bin"})


def test_actual_native_outputs_join_to_original_sources_without_runtime_claim(tmp_path):
    selected, compiler = selection(tmp_path)
    control = native_control(compiler)
    admission = run(tmp_path, selected, control)
    assert admission.require_controls((control.id,)) == admission.sha256
    report = admission.readiness()
    assert report["contracts"]["clock_reset"]["status"] == "PASS"
    assert report["contracts"]["memory_semantics"]["status"] == "UNKNOWN"
    assert report["contracts"]["program_loading"]["status"] == "UNKNOWN"
    assert report["target_runtime"] == report["target_timer"] == report["hardware_equivalence"] == "UNKNOWN"
    assert str(tmp_path) not in str(report)
    assert len(admission.outcomes[0].invocations) == 2
    assert admission.outcomes[0].original_rows == ((3, 4),)
    with pytest.raises(StageGateError, match="issued by actual"):
        replace(admission).verify()


def test_complete_output_mismatch_refuses_memory_readiness(tmp_path):
    selected, compiler = selection(tmp_path)
    admission = run(tmp_path, selected, native_control(compiler, contract="memory_semantics", expected=2))
    row = admission.outcomes[0]
    assert row.status == "REFUSED" and row.failed_step == "observe"
    assert row.reason == "complete_original_outputs_mismatch"
    with pytest.raises(StageGateError, match="memory_semantics"):
        admission.require_controls((row.control_id,))


def test_actual_input_grammar_failure_stops_before_expensive_followup(tmp_path):
    selected, compiler = selection(tmp_path)
    control = native_control(compiler, contract="input_grammar", source=b"not valid C\n")
    admission = run(tmp_path, selected, control)
    assert admission.outcomes[0].failed_step == "compile"
    assert len(admission.outcomes[0].invocations) == 1
    assert admission.readiness()["contracts"]["input_grammar"]["status"] == "REFUSED"
    assert not (admission.evidence_root / control.id / "control").exists()


@pytest.mark.parametrize("payload", [b"PASS\n", b"a\n3\n", b"a,b\n3,4\n5,6\n", b"a,b\n3,5\n", b"b,a\n4,3\n"])
def test_supplied_pass_label_or_partial_outputs_cannot_replace_original_gate(payload):
    control = native_control(Path("/unused/compiler"))
    with pytest.raises(StageGateError, match="original output"):
        check_original_outputs(payload, control)


@pytest.mark.parametrize("change", ["tool_source", "control_source", "receipt", "stdout", "new_member"])
def test_every_source_and_observation_reopens_and_membership_is_closed(tmp_path, change):
    selected, compiler = selection(tmp_path)
    admission = run(tmp_path, selected, native_control(compiler))
    if change == "tool_source":
        selected.source_files[0].write_text("changed\n")
    elif change == "control_source":
        (admission.evidence_root / "clock_reset" / "control.c").write_text("changed\n")
    elif change == "receipt":
        admission.receipt.write_text('{"status":"PASS"}\n')
    elif change == "stdout":
        rows = admission.outcomes[0].invocations
        (rows[-1].parent / "stdout.bin").write_bytes(b"a,b\n3,4\n")
        (rows[-1].parent / "stderr.bin").write_bytes(b"foreign observation\n")
    else:
        (admission.evidence_root / "new-foreign-member").write_text("unowned\n")
    with pytest.raises(StageGateError, match="changed"):
        admission.verify()


def test_unselected_command_and_source_escape_refuse_before_execution(tmp_path):
    selected, compiler = selection(tmp_path)
    control = native_control(compiler)
    foreign = replace(control, steps=(replace(control.steps[0], argv=("/bin/false",)),))
    with pytest.raises(StageGateError, match="selected or actually produced"):
        run(tmp_path, selected, foreign)
    with pytest.raises(StageGateError, match="escapes"):
        replace(control, files=(("../foreign-source", b"x"),)).verify()


def test_semantic_control_cannot_omit_the_original_numeric_roster(tmp_path):
    _, compiler = selection(tmp_path)
    control = native_control(compiler)
    with pytest.raises(StageGateError, match="complete original numeric"):
        replace(control, output_columns=(), expected_rows=()).verify()


def test_explicit_generated_directory_membership_joins_actual_native_execution(tmp_path):
    selected, compiler = selection(tmp_path)
    original = native_control(compiler)
    compile = RtlProbeStep("compile", (str(compiler), "@WORK@/control.c", "-o", "@WORK@/products/control"),
                           ("control.c",), ("products",), output_directories=("products",))
    execute = RtlProbeStep("observe", ("@WORK@/products/control",), ("products",), ())
    control = replace(original, steps=(compile, execute))
    admission = run(tmp_path, selected, control)
    assert admission.require_controls((control.id,)) == admission.sha256
    assert admission.outcomes[0].original_rows == original.expected_rows


def test_bounded_hanging_control_retains_actual_partial_output_and_never_passes(tmp_path):
    selected, compiler = selection(tmp_path)
    source = b'#include <stdio.h>\nint main(void){puts("a,b\\n3,4");fflush(stdout);for(;;){}}\n'
    original = native_control(compiler, source=source)
    control = replace(original, steps=(original.steps[0], replace(original.steps[1], timeout_seconds=0.05)))
    admission = run(tmp_path, selected, control)
    row = admission.outcomes[0]
    assert row.status == "REFUSED" and row.reason == "bounded_process_timeout"
    assert (row.invocations[-1].parent / "partial_stdout.bin").read_bytes() == b"a,b\n3,4\n"
    with pytest.raises(StageGateError, match="readiness refuses"):
        admission.require_controls((control.id,))
