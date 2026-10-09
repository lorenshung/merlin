"""Execute public native RTL tools on original small private controls.

Explicit tool selections are diagnostics, never an accelerator/runtime grant.
The produced function implements all state, memory and clock semantics.
"""

import importlib.util
import json
import os
import sys
import sysconfig
from pathlib import Path

import pytest
from merlin_experiments.phase2 import rtl_state_control
from merlin_experiments.phase2.rtl_engine_probe import probe_rtl_engine
from merlin_experiments.phase2.rtl_engine_protocol import (
    RtlProbeControl,
    RtlProbeStep,
    prepare_rtl_engine_selection,
)

from merlin.common import invocation_record

_fixture_path = Path(__file__).with_name("rtl_native_control_cases.py")
_fixture_spec = importlib.util.spec_from_file_location("private_native_rtl_controls", _fixture_path)
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
pipelined_read, rw_memory, separate_memory, two_clocks = (
    _fixture.pipelined_read,
    _fixture.rw_memory,
    _fixture.separate_memory,
    _fixture.two_clocks,
)


def tools():
    selections = [
        os.environ.get(name)
        for name in (
            "MERLIN_TEST_RTL_FIRTOOL",
            "MERLIN_TEST_RTL_SIMULATOR",
            "MERLIN_TEST_RTL_CC",
            "MERLIN_TEST_RTL_OPT",
            "MERLIN_TEST_RTL_TRANSLATE",
        )
    ]
    if not all(selections):
        pytest.skip("explicit public native RTL tools are required")
    return tuple(Path(value).resolve() for value in selections)


def execute(tmp_path, case, *, converted=False, expected_override=None):
    firtool, simulator, compiler, optimizer, translator = tools()
    python = Path(sys.executable).resolve()
    source, original_stimuli, expected = case()
    model = "NativeClock" if case is two_clocks else "NativeMemory"
    files = [("original.fir", source), ("stimuli.json", original_stimuli)]
    steps = []
    selected = "original.fir"
    if converted:
        files.append(("convert.anno.json", b'[{"class":"sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"}]\n'))
        steps.extend(
            (
                RtlProbeStep(
                    "parse_annotated",
                    (
                        str(firtool),
                        "@WORK@/original.fir",
                        "--parse-only",
                        "--annotation-file=@WORK@/convert.anno.json",
                        "-o",
                        "@WORK@/parsed.mlir",
                    ),
                    ("original.fir", "convert.anno.json"),
                    ("parsed.mlir",),
                ),
                RtlProbeStep(
                    "native_memory_conversion",
                    (str(optimizer), "@WORK@/parsed.mlir", "--firrtl-mem-to-reg-of-vec", "-o", "@WORK@/converted.mlir"),
                    ("parsed.mlir",),
                    ("converted.mlir",),
                ),
            )
        )
        selected = "converted.mlir"
    steps.extend(
        (
            RtlProbeStep(
                "native_hw",
                (str(firtool), "@WORK@/" + selected, "--ir-hw", "-o", "@WORK@/hw.mlir"),
                (selected,),
                ("hw.mlir",),
            ),
            RtlProbeStep(
                "native_model",
                (str(simulator), "@WORK@/hw.mlir", "--state-file=@WORK@/state.json", "-o", "@WORK@/model.ll"),
                ("hw.mlir",),
                ("model.ll", "state.json"),
            ),
            RtlProbeStep(
                "original_driver",
                (
                    str(python),
                    "-m",
                    rtl_state_control.__name__,
                    "--layout",
                    "@WORK@/state.json",
                    "--stimuli",
                    "@WORK@/stimuli.json",
                    "--model",
                    model,
                    "--entrypoint",
                    model + "_eval",
                    "--output",
                    "@WORK@/driver.c",
                ),
                ("state.json", "stimuli.json"),
                ("driver.c",),
            ),
            RtlProbeStep(
                "native_link",
                (str(compiler), "-O1", "@WORK@/model.ll", "@WORK@/driver.c", "-o", "@WORK@/control"),
                ("model.ll", "driver.c"),
                ("control",),
            ),
            RtlProbeStep("original_observation", ("@WORK@/control",), ("control",), ()),
        )
    )
    selected_sources = (
        Path(rtl_state_control.__file__).resolve(),
        Path(__file__).resolve(),
        Path(__file__).with_name("rtl_native_control_cases.py").resolve(),
    )
    selection = prepare_rtl_engine_selection(tools=(*tools(), python), source_files=selected_sources)
    declaration = json.loads(original_stimuli)
    control = RtlProbeControl(
        "native_control",
        "clock_reset" if case is two_clocks else "memory_semantics",
        tuple(files),
        tuple(steps),
        ("sample", *declaration["outputs"]),
        expected if expected_override is None else expected_override,
    )
    import_roots = os.pathsep.join(
        dict.fromkeys(
            (
                *(str(Path(value).resolve()) for value in os.environ["PYTHONPATH"].split(os.pathsep) if value),
                sysconfig.get_path("purelib"),
                sysconfig.get_path("platlib"),
            )
        )
    )
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "PYTHONPATH": import_roots}
    if value := os.environ.get("MERLIN_TEST_RTL_LD_LIBRARY_PATH"):
        environment["LD_LIBRARY_PATH"] = value
    admission = probe_rtl_engine(
        selection=selection, controls=(control,), evidence_root=tmp_path / "probe", environment=environment
    )
    return admission, control


@pytest.mark.parametrize("case", [rw_memory, separate_memory, pipelined_read, two_clocks])
def test_actual_public_native_rtl_memory_and_clock_controls(tmp_path, case):
    admission, control = execute(tmp_path, case)
    row = admission.outcomes[0]
    if row.status != "PASS":
        detail = row.invocations[-1].parent.joinpath("stderr.bin").read_bytes()
        pytest.fail(f"{row.failed_step}: {row.reason}: {detail.decode(errors='replace')[-3000:]}")
    assert admission.require_controls((control.id,)) == admission.sha256
    assert row.original_rows == control.expected_rows and len(row.invocations) == 5
    readiness = admission.readiness()
    assert readiness["target_runtime"] == readiness["hardware_equivalence"] == readiness["target_timer"] == "UNKNOWN"
    assert readiness["contracts"]["program_loading"]["status"] == "UNKNOWN"
    for record in row.invocations:
        document = invocation_record.verify(record)
        assert document["inputs_unchanged"] and document["dependencies_unchanged"]


def test_real_native_memory_observation_keeps_the_original_complete_numeric_gate(tmp_path):
    source, original, expected = rw_memory()
    del source, original
    # A single changed original readback must refuse despite actual compilation.
    bad = ((expected[0][0], expected[0][1] + 1), *expected[1:])
    admission, _ = execute(tmp_path, rw_memory, expected_override=bad)
    row = admission.outcomes[0]
    assert row.status == "REFUSED" and row.reason == "complete_original_outputs_mismatch"
    assert row.failed_step == "original_observation" and len(row.invocations) == 5


def test_actual_stock_register_vector_conversion_cannot_hide_read_latency_defect(tmp_path):
    # This is a real upstream transform, not a hand-edited memory model. The
    # original latency/edge roster rejects it on the selected stock tool version.
    admission, _ = execute(tmp_path, rw_memory, converted=True)
    row = admission.outcomes[0]
    assert len(row.invocations) == 7
    assert row.status == "REFUSED" and row.failed_step == "original_observation"
    assert row.reason == "complete_original_outputs_mismatch"
    output = row.invocations[-1].parent.joinpath("stdout.bin").read_text()
    assert "10,68\n" in output  # Original no-edge sample must still read 51.
