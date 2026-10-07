"""Actual lowering/native checks for exact scalar pointwise scheduling hints."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_unroll import FEATURE, MARKER, _edit_pipeline
from merlin.llvmlower.toolchain import clang, m2m_python


def source(n, *, alias=False, fmas=4, division=True, fastmath=False, dtype="f32"):
    typ = f"tensor<{n}x{dtype}>"
    body = []
    for i in range(fmas):
        prior = "%x" if i == 0 else f"%p{i - 1}"
        flag = " fastmath<contract>" if fastmath else ""
        body.append(f"%p{i} = math.fma {prior}, %y, %x{flag} : {dtype}")
    value = f"%p{fmas - 1}"
    if division:
        flag = " fastmath<arcp>" if fastmath else ""
        body.append(f"%d = arith.divf {value}, %y{flag} : {dtype}")
        value = "%d"
    body.append(f"linalg.yield {value} : {dtype}")
    initial = "%a" if alias else "%c"
    args = f"%a: {typ}, %b: {typ}" + ("" if alias else f", %c: {typ}")
    return f"""module {{ func.func @forward({args}) -> {typ} attributes {{llvm.emit_c_interface}} {{
      %r = linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]}}
        ins(%a,%b:{typ},{typ}) outs({initial}:{typ}) {{
        ^bb0(%x:{dtype},%y:{dtype},%z:{dtype}):
          {" ".join(body)}
      }} -> {typ}
      return %r:{typ}
    }} }}"""


def lower(tmp_path, src, selected):
    features = {"lower_fma_to_intrinsic"} | ({FEATURE} if selected else set())
    work = tmp_path / ("selected" if selected else "control")
    work.mkdir()
    return work, lower_to_llvm_ir(src, workdir=work, features=features)


def test_default_pipeline_is_byte_unchanged_and_marker_precedes_cf():
    baseline = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({FEATURE}))
    assert selected.replace(MARKER + ",", "") == baseline
    assert selected.index(MARKER) < selected.index("convert-scf-to-cf")
    with pytest.raises(ValueError):
        _edit_pipeline([])
    with pytest.raises(ValueError):
        _edit_pipeline([MARKER, "convert-scf-to-cf"])


@pytest.mark.parametrize("n,alias", [(16, False), (17, False), (17, True)])
def test_actual_native_tail_existing_fma_and_tensor_alias_semantics(tmp_path, n, alias):
    a = np.linspace(-3, 3, n, dtype=np.float32)
    b = np.linspace(0.25, 2, n, dtype=np.float32)
    # Exercise special values and cancellation while retaining the original FMA.
    a[:5] = [0.0, -0.0, np.inf, -np.inf, 1.0000001192092896]
    b[:5] = [0.0, -0.0, 1.0, 1.0, -1.0000001192092896]
    outputs = []
    for selected in (False, True):
        work, llvm = lower(tmp_path, source(n, alias=alias), selected)
        assert ("llvm.loop.unroll.count" in llvm) == selected
        assert "@llvm.fma.f32" in llvm
        assert "fdiv float" in llvm
        assert " contract " not in llvm
        ll = work / "model.ll"
        ll.write_text(llvm)
        so = work / ("model_" + hashlib.sha256(llvm.encode()).hexdigest()[:16] + ".so")
        subprocess.run(
            [
                str(clang()),
                "-O3",
                "-fPIC",
                "-shared",
                "-ffp-contract=off",
                str(ll),
                str(mlir_runtime_c()),
                "-lm",
                "-o",
                str(so),
            ],
            check=True,
            capture_output=True,
        )
        original = a.copy()
        inputs = [a, b] if alias else [a, b, np.zeros(n, dtype=np.float32)]
        output = np.empty(n, dtype=np.float32)
        HostModel.load(str(so))([(v.ctypes.data, v.shape) for v in [*inputs, output]])
        np.testing.assert_array_equal(a.view(np.uint32), original.view(np.uint32))
        outputs.append(output)
    np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))


@pytest.mark.parametrize("change", ["short_chain", "no_division", "f64", "tiny", "fastmath", "dynamic"])
def test_unrecognized_or_nonprofitable_loops_keep_baseline(tmp_path, change):
    options = {
        "short_chain": {"fmas": 3},
        "no_division": {"division": False},
        "f64": {"dtype": "f64"},
        "tiny": {},
        "fastmath": {"fastmath": True},
        "dynamic": {},
    }
    n = 1 if change == "tiny" else "?" if change == "dynamic" else 17
    _, llvm = lower(tmp_path, source(n, **options[change]), True)
    assert "llvm.loop.unroll.count" not in llvm


@pytest.mark.parametrize("change", ["iter_args", "existing_annotation", "strictfp", "external_call", "nested_region"])
def test_structural_refusals(tmp_path, change):
    from merlin.llvmlower.scalar_pointwise_unroll import RUNNER_PRELUDE

    text = """module { func.func @forward(%a:memref<17xf32>,%b:memref<17xf32>) {
      %c0 = arith.constant 0:index
      %c1 = arith.constant 1:index
      %c17 = arith.constant 17:index
      scf.for %i = %c0 to %c17 step %c1 {
        %x = memref.load %a[%i]:memref<17xf32>
        %y = memref.load %b[%i]:memref<17xf32>
        %p0 = math.fma %x,%y,%x:f32
        %p1 = math.fma %p0,%y,%x:f32
        %p2 = math.fma %p1,%y,%x:f32
        %p3 = math.fma %p2,%y,%x:f32
        %d = arith.divf %p3,%y:f32
        memref.store %d,%b[%i]:memref<17xf32>
      }
      return
    } }"""
    if change == "iter_args":
        text = text.replace(
            "scf.for %i = %c0 to %c17 step %c1 {", "%r = scf.for %i = %c0 to %c17 step %c1 iter_args(%v=%c0) -> index {"
        )
        text = text.replace(
            "memref.store %d,%b[%i]:memref<17xf32>", "memref.store %d,%b[%i]:memref<17xf32>\nscf.yield %v:index"
        )
    elif change == "existing_annotation":
        text = text.replace(
            "      }\n      return",
            "      } {loop_annotation=#llvm.loop_annotation<unroll=<disable=true>>}\n      return",
        )
    elif change == "strictfp":
        text = text.replace("%b:memref<17xf32>) {", '%b:memref<17xf32>) attributes {passthrough=["strictfp"]} {')
    elif change == "external_call":
        text = text.replace("module {", "module { func.func private @opaque(f32)")
        text = text.replace("%d = arith.divf", "func.call @opaque(%p3):(f32)->()\n%d = arith.divf")
    else:
        text = text.replace("%d = arith.divf", "scf.for %j=%c0 to %c1 step %c1 { }\n%d = arith.divf")
    src, script = tmp_path / "input.mlir", tmp_path / "run.py"
    src.write_text(text)
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + RUNNER_PRELUDE.split("_PU_MARKER =", 1)[0]
        + "ctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
        + "print(_annotate_pointwise_unroll(ctx,module))\n"
    )
    proc = subprocess.run([str(m2m_python()), str(script), str(src)], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "0"
