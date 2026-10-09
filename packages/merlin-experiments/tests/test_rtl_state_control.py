"""Transport controls do not supply simulation or qualification semantics."""

import json
import shutil

import pytest
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.rtl_state_control import render_opaque_state_control


def declarations():
    layout = [
        {
            "name": "Control",
            "numStateBytes": 16,
            "states": [
                {"name": "input_value", "offset": 0, "numBits": 16, "type": "input"},
                {"name": "result", "offset": 8, "numBits": 16, "type": "output"},
            ],
        }
    ]
    samples = {
        "byte_order": "little",
        "inputs": ["input_value"],
        "outputs": ["result"],
        "samples": [{"id": 0, "values": [0xABCD], "evaluations": 1, "observe": True}],
    }
    return layout, samples


def render(layout, samples, **kwargs):
    return render_opaque_state_control(
        layout_bytes=json.dumps(layout).encode(),
        stimuli_bytes=json.dumps(samples).encode(),
        model="Control",
        entrypoint=kwargs.get("entrypoint", "evaluate"),
    )


@pytest.mark.parametrize("byte_order", ["little", "big"])
def test_actual_native_driver_only_transports_explicit_io_bytes(tmp_path, byte_order):
    from merlin.common import invocation_record

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    layout, samples = declarations()
    samples["byte_order"] = byte_order
    source = tmp_path / "driver.c"
    source.write_bytes(render(layout, samples))
    # A harmless byte copy tests the transport, not RTL/physical semantics.
    function = tmp_path / "function.c"
    function.write_text("#include <stdint.h>\nvoid evaluate(void *p) { uint8_t *b=p;b[8]=b[0];b[9]=b[1]; }\n")
    executable = tmp_path / "control"
    result = invocation_record.run(
        [compiler, str(source), str(function), "-o", str(executable)],
        directory=tmp_path / "evidence",
        stage="native_transport_build",
        inputs=(source, function),
        outputs=(executable,),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    result = invocation_record.run(
        [str(executable)],
        directory=tmp_path / "evidence",
        stage="native_transport_run",
        inputs=(executable,),
        capture_output=True,
    )
    assert result.returncode == 0 and result.stdout == b"sample,result\n0,43981\n"
    for record in (tmp_path / "evidence/invocations").glob("*/invocation.json"):
        invocation_record.verify(record)


@pytest.mark.parametrize(
    "change",
    [
        "extent",
        "overlap",
        "wide",
        "empty",
        "ambiguous",
        "extra",
        "missing_input",
        "missing_output",
        "partial_sample",
        "boolean_value",
        "excess_value",
        "unknown_order",
        "unbounded_evals",
        "no_outputs",
        "injection",
    ],
)
def test_unsupported_or_incomplete_native_layout_and_stimuli_refuse(change):
    layout, samples = declarations()
    if change == "extent":
        layout[0]["states"][1]["offset"] = 15
    elif change == "overlap":
        layout[0]["states"][1]["offset"] = 1
    elif change == "wide":
        layout[0]["states"][1]["numBits"] = 65
    elif change == "empty":
        layout[0]["states"] = []
    elif change == "ambiguous":
        layout.append(layout[0])
    elif change == "extra":
        layout[0]["states"][0]["qualified"] = True
    elif change == "missing_input":
        samples["inputs"] = []
    elif change == "missing_output":
        samples["outputs"] = []
    elif change == "partial_sample":
        samples["samples"][0]["values"] = []
    elif change == "boolean_value":
        samples["samples"][0]["values"] = [True]
    elif change == "excess_value":
        samples["samples"][0]["values"] = [65536]
    elif change == "unknown_order":
        samples["byte_order"] = "native"
    elif change == "unbounded_evals":
        samples["samples"][0]["evaluations"] = 33
    elif change == "no_outputs":
        samples["samples"][0]["observe"] = False
    else:
        with pytest.raises(StageGateError, match="identifier"):
            render(layout, samples, entrypoint='evaluate); system("foreign");')
        return
    with pytest.raises(StageGateError):
        render(layout, samples)


def test_supplied_layout_and_correctness_labels_do_not_issue_any_authority():
    layout, samples = declarations()
    source = render(layout, samples)
    assert type(source) is bytes
    assert b"calloc(16, 1)" in source
    assert b"clock" not in source and b"cycles" not in source


def test_foreign_lifecycle_abi_and_duplicate_layout_fields_refuse():
    layout, samples = declarations()
    # A produced layout that additionally requires lifecycle/runtime context is
    # outside this opaque-state contract, even if its public scalar I/O matches.
    layout[0]["initialFnSym"] = ""
    layout[0]["finalFnSym"] = ""
    with pytest.raises(StageGateError, match="unsupported model"):
        render(layout, samples)
    with pytest.raises(StageGateError, match="JSON data"):
        render_opaque_state_control(
            layout_bytes=b'[{"name":"a","name":"b"}]',
            stimuli_bytes=json.dumps(samples).encode(),
            model="Control",
            entrypoint="evaluate",
        )
