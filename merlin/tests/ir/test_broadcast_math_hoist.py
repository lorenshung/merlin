"""Typed source invariance and compiled numerical tests for broadcast math hoisting."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.broadcast_math_hoist import FEATURE, MARKER, apply_for_test
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang


def source(rows, columns, *, cast=False, other_use=False, init_read=False, permute=False):
    tensor = f"tensor<{columns}x{rows}xf32>" if permute else f"tensor<{rows}x{columns}xf32>"
    output_map = "(d1,d0)" if permute else "(d0,d1)"
    scalar = "f64" if cast else "f32"
    widening = "%wide=arith.extf %row:f32 to f64" if cast else ""
    narrow = "%root=arith.truncf %inv:f64 to f32" if cast else ""
    add_other = "%plus=arith.addf %product,%mean32:f32" if other_use else ""
    mean32 = (
        "%mean32=arith.truncf %mean:f64 to f32"
        if cast and other_use
        else "%mean32=arith.addf %mean,%zero:f32"
        if other_use
        else ""
    )
    value = "%plus" if other_use else "%product"
    add_init = f"%final=arith.addf {value},%old:f32" if init_read else ""
    if init_read:
        value = "%final"
    return f"""module {{ func.func @forward(%a:{tensor},%r:tensor<{rows}x1xf32>,%c:{tensor})->{tensor}
      attributes {{llvm.emit_c_interface}} {{
      %divisor=arith.constant {float(max(1, columns))}:{scalar}
      %epsilon=arith.constant 0.00001:{scalar}
      %zero=arith.constant 0.0:f32
      %out=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->{output_map}>,affine_map<(d0,d1)->(d0,0)>,affine_map<(d0,d1)->{output_map}>],iterator_types=["parallel","parallel"]}}
      ins(%a,%r:{tensor},tensor<{rows}x1xf32>) outs(%c:{tensor}) {{
       ^bb0(%x:f32,%row:f32,%old:f32):
       {widening}
       %mean=arith.divf {"%wide" if cast else "%row"},%divisor:{scalar}
       %arg=arith.addf %mean,%epsilon:{scalar}
       %inv=math.rsqrt %arg:{scalar}
       {narrow}
       %product=arith.mulf %x,{"%root" if cast else "%inv"}:f32
       {mean32}
       {add_other}
       {add_init}
       linalg.yield {value}:f32
      }} -> {tensor}
      return %out:{tensor}
    }} }}"""


def compiled(tmp_path, text, arguments, shape, features):
    key = hashlib.sha256((text + str(sorted(features))).encode()).hexdigest()[:16]
    work = tmp_path / key
    work.mkdir()
    llvm = lower_to_llvm_ir(text, workdir=work, features=features)
    ll, library = work / "model.ll", work / f"model_{key}.so"
    ll.write_text(llvm)
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
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    output = np.empty(shape, dtype="f4")
    HostModel.load(str(library))(
        [(value.ctypes.data, value.shape) for value in arguments] + [(output.ctypes.data, output.shape)]
    )
    return output, llvm


@pytest.mark.parametrize(
    "rows,columns,options",
    [
        (3, 7, {}),
        (5, 13, {"cast": True}),
        (2, 17, {"other_use": True}),
        (7, 3, {"permute": True, "init_read": True}),
        (3, 11, {"cast": True, "other_use": True, "init_read": True}),
    ],
)
def test_actual_native_unequal_rows_tails_casts_and_live_uses(tmp_path, rows, columns, options):
    text = source(rows, columns, **options)
    transformed, count = apply_for_test(text)
    assert count == 1 and f"tensor<{rows}xf" in transformed
    rng = np.random.default_rng(432)
    shape = (columns, rows) if options.get("permute") else (rows, columns)
    a = rng.normal(size=shape).astype("f4")
    row = np.geomspace(0.001, 63, rows).astype("f4").reshape(rows, 1)
    initial = rng.normal(size=shape).astype("f4")
    arguments = [a, row, initial]
    control_arguments = [value.copy() for value in arguments]
    candidate_arguments = [value.copy() for value in arguments]
    control, _ = compiled(tmp_path, text, control_arguments, shape, set())
    candidate, llvm = compiled(tmp_path, text, candidate_arguments, shape, {FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    # This ABI permits bufferization to reuse the explicit output-init buffer.
    # Both arms get fresh identical inputs, including that live initial value.
    for value, old in zip(candidate_arguments, control_arguments):
        np.testing.assert_array_equal(value.view("u4"), old.view("u4"))
    assert "@rsqrt" in llvm


def test_actual_native_special_values_preserve_source_math_bits(tmp_path):
    text = source(9, 5)
    a = np.tile(np.array([0.0, -0.0, 1.0, -2.0, 3.0], "f4"), (9, 1))
    row = np.array([0.0, -0.0, 1.0, np.inf, -np.inf, np.nan, -1.0, 1e-38, 1e30], "f4").reshape(9, 1)
    initial = np.full_like(a, 17.0)
    control, _ = compiled(tmp_path, text, [a, row, initial], a.shape, set())
    candidate, _ = compiled(tmp_path, text, [a, row, initial], a.shape, {FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))


@pytest.mark.parametrize("rows,columns", [(0, 7), (3, 0), (3, 1)])
def test_empty_or_no_repetition_retains_source_and_values(tmp_path, rows, columns):
    text = source(rows, columns)
    _, count = apply_for_test(text)
    assert count == 0
    a = np.ones((rows, columns), "f4")
    row = np.ones((rows, 1), "f4")
    initial = np.full_like(a, -42.0)
    control, _ = compiled(tmp_path, text, [a, row, initial], a.shape, set())
    candidate, _ = compiled(tmp_path, text, [a, row, initial], a.shape, {FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))


@pytest.mark.parametrize(
    "change", ["dynamic", "reduction", "varying", "call", "index", "strict", "fastmath", "init_dependence"]
)
def test_refused_effects_or_unproved_invariance(change):
    text = source(3, 7)
    if change == "dynamic":
        text = text.replace("tensor<3x7xf32>", "tensor<3x?xf32>")
    elif change == "reduction":
        text = text.replace('iterator_types=["parallel","parallel"]', 'iterator_types=["parallel","reduction"]')
    elif change == "varying":
        text = text.replace("%mean=arith.divf %row", "%mean=arith.divf %x")
    elif change == "call":
        text = text.replace("module {", "module {func.func private @observe(f32)->f32\n", 1)
        text = text.replace("%mean=", "%observed=func.call @observe(%x):(f32)->f32\n %mean=", 1)
    elif change == "index":
        text = text.replace("%mean=", "%index=linalg.index 1:index\n %mean=", 1)
    elif change == "strict":
        text = text.replace("llvm.emit_c_interface}", "llvm.emit_c_interface,strictfp}")
    elif change == "fastmath":
        text = text.replace("math.rsqrt %arg:f32", "math.rsqrt %arg fastmath<afn>:f32")
    else:
        text = text.replace("%mean=arith.divf %row", "%mean=arith.divf %old")
    _, count = apply_for_test(text)
    assert count == 0


def test_default_pipeline_identity_and_after_fusion_placement():
    default = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({FEATURE}))
    assert selected.replace(MARKER + ",", "") == default
    assert selected.index("linalg-fuse-elementwise-ops") < selected.index(MARKER) < selected.index("one-shot-bufferize")


@pytest.mark.parametrize("scope", ["generic", "scalar"])
def test_nested_strict_scope_refuses(scope):
    text = source(3, 7)
    if scope == "generic":
        text = text.replace("outs(%c:tensor<3x7xf32>) {", "outs(%c:tensor<3x7xf32>) attrs = {strictfp} {")
    else:
        text = text.replace("math.rsqrt %arg:f32", "math.rsqrt %arg {strictfp}:f32")
    _, count = apply_for_test(text)
    assert count == 0
