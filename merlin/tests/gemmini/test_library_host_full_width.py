"""A full-width matmul the library answers on the HOST is computed correctly, on every machine.

The library's host matmul (``matmul_cpu``) writes ``elem_t``: it does not implement a full-width
(int32) result, and ``tiled_matmul`` says so only under its debug checks. With loop-descriptor
instructions prohibited, a declined full-width matmul falls to exactly that path -- measured on
ResNet-50's classifier (g71) built for a full-width machine: 1000 of 1000 elements wrong, argmax 199
against 21. The driver now computes such a group with its own full-width routine (``hr_acc_matmul``,
the one a machine without the readout already uses).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.perf import whole_model_build as W
from merlin.runtime.backends import base as backends

pytestmark = pytest.mark.target("gemmini")

_TARGET = "gemmini"
P = backends.whole_model_driver(_TARGET).program
#: The library's loop-free paths when the parameter header states no output-stationary dataflow: both
#: kinds on the host (the case the defect lived in), read from the driver's own derivation.
_HOST_LIBRARY = P.library_path_without_loops("")
_ACCELERATED_LIBRARY = P.library_path_without_loops("#define GEMMINI_OS_DATAFLOW 1\n")


def _model():
    steps = [
        {"kind": "matmul", "group": 1, "in": "IMAGE", "relu": False, "bias": "BIAS_g1", "scale": 0.05,
         "weight": "W_g1", "out": "B_g1", "m": 4, "k": 8, "n": 16},
        {"kind": "matmul", "group": 2, "in": "B_g1", "relu": False, "bias": "BIAS_g2", "scale": None,
         "weight": "W_g2", "out": "B_g2", "m": 4, "k": 16, "n": 32, "dequantize": 0.01},
    ]  # fmt: skip
    arrays = {f"W_g{s['group']}": np.ones((s["k"], s["n"]), np.int8) for s in steps}
    arrays.update({f"BIAS_g{s['group']}": np.zeros(s["n"], np.int32) for s in steps})
    arrays["IMAGE_DATA"] = np.ones((4, 8), np.int8)
    arrays["GOLDEN"] = np.zeros(128, np.float32)
    buffers = [
        {"name": "B_g1", "elements": 64, "ctype": "elem_t"},
        {"name": "B_g2", "elements": 128, "ctype": "acc_t"},
    ]
    return {"steps": steps, "buffers": buffers, "arrays": arrays, "classes": 128, "groups": 3, "device_groups": 2}


def test_the_derivation_puts_the_library_matmul_on_the_host_without_an_os_dataflow():
    assert _HOST_LIBRARY["matmul"]["host"] is True
    assert _ACCELERATED_LIBRARY["matmul"]["host"] is False


def test_a_full_width_library_host_matmul_uses_the_drivers_routine():
    source = P.render(_model(), verify="local", library=_HOST_LIBRARY)
    assert "hr_acc_matmul(B_g1, (const elem_t *)W_g2, BIAS_g2, B_g2, 4, 16, 32, 0);" in source
    assert "static void hr_acc_matmul(" in source  # the routine is emitted for it
    assert "tiled_matmul_auto(4, 32, 16, B_g1" not in source  # never the library's matmul_cpu
    # The requantizing group keeps the library's host call: matmul_cpu computes an elem_t result right.
    assert "tiled_matmul_auto(4, 16, 8, IMAGE" in source


def test_without_the_host_path_nothing_changes():
    """MUTATION THIS CATCHES: route every full-width group through the driver's routine, and a program
    whose library runs it on the accelerator (with its full-width readout) loses that call."""
    for library in (None, _ACCELERATED_LIBRARY):
        source = P.render(_model(), verify="local", library=library)
        assert "hr_acc_matmul(" not in source and "tiled_matmul_auto(4, 32, 16, B_g1" in source


def test_a_package_answered_full_width_group_keeps_the_package_kernel():
    kernels = {"calls": {2: "gemmini_kernel_g2(B_g1, W_g2, BIAS_g2, B_g2);"}, "definitions": ""}
    source = P.render(_model(), verify="local", library=_HOST_LIBRARY, sched_kernels=kernels)
    assert "gemmini_kernel_g2(B_g1, W_g2, BIAS_g2, B_g2);" in source and "hr_acc_matmul(B_g1" not in source


def test_a_declined_residual_add_takes_the_librarys_host_code_not_its_loop_stream():
    """The library's accelerated residual add is a loop-descriptor stream; with the role prohibited its
    default call is trapped, and the trap (an ``exit``) made every later group dead code -- measured, the
    next group's output buffer vanished from the linked program. It now takes the library's host code."""
    step = {"kind": "sum", "group": 6, "rows": 4, "cols": 16, "lhs_load": 0.5, "rhs_load": 1.0,
            "readout": 2.0, "lhs": "B_g4", "rhs": "B_g5", "out": "B_g6", "relu": True}  # fmt: skip
    assert _HOST_LIBRARY["sum"]["host"] is True
    assert P._call(step, _HOST_LIBRARY).rstrip(";").endswith(f"{P.LIBRARY_HOST})")
    assert P._call(step).rstrip(";").endswith(f"{P.LIBRARY_DEFAULT})")  # unchanged when loops are allowed


# ------------------------------------------------------------- the real classifier, on spike (slow)

_CAPSULE = merlin_dir() / "contract" / "capsules" / "model" / "SY_model_resnet50"
_HEADER = (
    merlin_dir()
    / "experiments/capsule_bench/targets/gemmini/contracts/harness_curated/gemmini-rocc-tests/include/gemmini_params.h"
)


def _ready() -> str:
    if not (_CAPSULE / "capsule.weights.safetensors").is_file():
        return "the model capsule's gitignored weights are not present"
    if not backends.get_backend(_TARGET).available("spike"):
        return "spike-gemmini is unavailable"
    return ""


@pytest.mark.slow
@pytest.mark.timeout(7200)
@pytest.mark.skipif(bool(_ready()), reason=_ready() or "ready")
def test_the_library_program_classifies_on_a_full_width_machine_with_loops_prohibited(monkeypatch, tmp_path):
    """A package that declines every group: each is the library's, loop-free, on a full-width machine --
    the classifier (the model's full-width group) included. Before the fix its 1000 elements were all
    wrong and the argmax was 199."""
    import negotiating_stub_package as STUB

    STUB.install(monkeypatch, tmp_path / "package", "decline", target=_TARGET)
    record = W.build(
        tmp_path / "package", _CAPSULE, target=_TARGET, machine="gemmini_gsim_emulator", header=_HEADER,
        out=tmp_path / "b", verify="local", prohibited_roles=["loop_descriptor"],
    )  # fmt: skip
    assert record["attribution"]["counts"][W.ON_PACKAGE] == 0
    oracle = json.loads((tmp_path / "b" / "oracle.json").read_text(encoding="utf-8"))
    console = backends.get_backend(_TARGET).run_elf(record["elf"], simulator="spike", timeout=7000)
    verdict = W.grade(console, oracle)
    assert not verdict["disagree"], verdict["disagree"][:4]
    assert verdict["argmax"] is not None and verdict["argmax"]["got"] == verdict["argmax"]["want"]
