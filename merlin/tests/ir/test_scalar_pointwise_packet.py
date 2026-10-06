"""Exact tensor lane scheduling through actual upstream lowering and native code."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import (
    BROADCAST_FEATURE,
    BROADCAST_MARKER,
    FEATURE,
    FOUR_FEATURE,
    FOUR_MARKER,
    MARKER,
    MULTIPLY_FOUR_FEATURE,
    MULTIPLY_FOUR_MARKER,
    TWO_MULTIPLY_FOUR_FEATURE,
    TWO_MULTIPLY_FOUR_MARKER,
    apply_for_test,
)
from merlin.llvmlower.toolchain import clang


def source(n, *, alias=False, init_read=False, fastmath=False):
    flags = " fastmath<contract>" if fastmath else ""
    initial = "%a" if alias else "%c"
    declaration = "" if alias else f", %c:tensor<2x{n}xf32>"
    return f"""module {{ func.func @forward(%a:tensor<2x{n}xf32>,%b:tensor<{n}xf32>,%s:tensor<f32>{declaration}) -> tensor<2x{n}xf32> attributes {{llvm.emit_c_interface}} {{
    %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d1)>,affine_map<(d0,d1)->()>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}}
      ins(%a,%b,%s:tensor<2x{n}xf32>,tensor<{n}xf32>,tensor<f32>) outs({initial}:tensor<2x{n}xf32>) {{
      ^bb0(%x:f32,%y:f32,%s0:f32,%z:f32):
       %p=math.fma %x,%y,%s0{flags}:f32
       %q=math.fma %p,%y,%x{flags}:f32
       %u=math.fma %q,%y,%s0{flags}:f32
       %v=math.fma %u,%y,%x{flags}:f32
       %w=arith.divf %v,%s0:f32
       {"%r0=arith.addf %w,%z:f32" if init_read else ""}
       linalg.yield {"%r0" if init_read else "%w"}:f32
      }} -> tensor<2x{n}xf32>
      return %r:tensor<2x{n}xf32>
    }} }}"""


def run(tmp_path, text, args, shape, features):
    key = hashlib.sha256((text + str(sorted(features))).encode()).hexdigest()[:16]
    work = tmp_path / key
    work.mkdir()
    llvm = lower_to_llvm_ir(text, workdir=work, features=features | {"lower_fma_to_intrinsic"})
    ll = work / "model.ll"
    ll.write_text(llvm)
    so = work / "model.so"
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
    out = np.empty(shape, dtype=np.float32)
    HostModel.load(str(so))([(x.ctypes.data, x.shape) for x in args] + [(out.ctypes.data, out.shape)])
    return out, llvm


@pytest.mark.parametrize("feature", [FEATURE, FOUR_FEATURE])
@pytest.mark.parametrize("n,alias", [(1, False), (5, False), (8, True), (11, True)])
def test_actual_native_lane_tail_broadcast_and_live_alias(tmp_path, feature, n, alias):
    rng = np.random.default_rng(83)
    a = rng.normal(size=(2, n)).astype("f4")
    b = rng.normal(size=n).astype("f4")
    s = np.array(3.0, dtype="f4")
    c = np.full((2, n), 17.0, dtype="f4")
    old = a.copy()
    args = [a, b, s] + ([] if alias else [c])
    text = source(n, alias=alias)
    transformed, count = apply_for_test(text, lanes=2 if feature == FEATURE else 4)
    assert count == 1 and "linalg.generic" not in transformed
    control, _ = run(tmp_path, text, args, (2, n), set())
    candidate, ll = run(tmp_path, text, args, (2, n), {feature})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    np.testing.assert_array_equal(a.view("u4"), old.view("u4"))
    assert "@llvm.fma.f32" in ll and "fdiv float" in ll


@pytest.mark.parametrize("option", [{"init_read": True}, {"fastmath": True}])
def test_unsupported_source_keeps_ir_identical(tmp_path, option):
    text = source(7, **option)
    control = lower_to_llvm_ir(text, workdir=tmp_path / "c")
    selected = lower_to_llvm_ir(text, workdir=tmp_path / "s", features={FEATURE})
    assert selected == control


def test_default_and_feature_constraints():
    default = _upstream_pipeline(frozenset())
    for feature, marker in [(FEATURE, MARKER), (FOUR_FEATURE, FOUR_MARKER), (BROADCAST_FEATURE, BROADCAST_MARKER)]:
        selected = _upstream_pipeline(frozenset({feature}))
        assert selected.replace(marker + ",", "") == default
        assert selected.index(marker) < selected.index("one-shot-bufferize")
    with pytest.raises(ValueError, match="alternatives"):
        _upstream_pipeline(frozenset({FEATURE, FOUR_FEATURE}))
    with pytest.raises(ValueError, match="alternatives"):
        _upstream_pipeline(frozenset({FEATURE, BROADCAST_FEATURE}))


@pytest.mark.parametrize("change", ["dynamic", "reduction", "index", "constant_map", "call"])
def test_structural_refusal_preserves_source(change):
    text = source(8)
    if change == "dynamic":
        text = text.replace("tensor<2x8xf32>", "tensor<2x?xf32>").replace("tensor<8xf32>", "tensor<?xf32>")
    elif change == "reduction":
        text = text.replace('iterator_types=["parallel","parallel"]', 'iterator_types=["parallel","reduction"]')
    elif change == "index":
        text = text.replace("%p=math.fma", "%idx=linalg.index 0:index\n       %p=math.fma")
    elif change == "constant_map":
        text = text.replace("affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(0,d1)>", 1)
    else:
        text = text.replace("module {", "module { func.func private @observe(f32)->f32\n", 1).replace(
            "%p=math.fma", "%observed=func.call @observe(%x):(f32)->f32\n       %p=math.fma"
        )
    transformed, count = apply_for_test(text)
    assert count == 0 and "linalg.generic" in transformed


def test_strict_function_attribute_refuses_lane_reordering():
    text = source(8).replace("attributes {llvm.emit_c_interface}", "attributes {llvm.emit_c_interface, strictfp}")
    transformed, count = apply_for_test(text)
    assert count == 0 and "linalg.generic" in transformed


@pytest.mark.parametrize("n,alias", [(1, False), (5, False), (8, True)])
def test_broadcast_axis_shares_loads_and_preserves_actual_native_tail_and_live_alias(tmp_path, n, alias):
    text = source(n, alias=alias).replace("tensor<2x", "tensor<3x")
    rng = np.random.default_rng(713)
    a = rng.normal(size=(3, n)).astype("f4")
    b = rng.normal(size=n).astype("f4")
    s = np.array(3.0, dtype="f4")
    c = np.full((3, n), 17.0, dtype="f4")
    old = a.copy()
    args = [a, b, s] + ([] if alias else [c])
    transformed, count = apply_for_test(text, broadcast=True)
    assert count == 1 and "scalar_pointwise_broadcast_packet_2" in transformed
    # Full packet has two varying extracts and one shared channel/scalar extract;
    # the odd-row tail has one of each. This validates emitted memory work.
    assert transformed.count("tensor.extract") == 7
    control, _ = run(tmp_path, text, args, (3, n), set())
    candidate, _ = run(tmp_path, text, args, (3, n), {BROADCAST_FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    np.testing.assert_array_equal(a.view("u4"), old.view("u4"))


def test_broadcast_axis_requires_a_used_nonscalar_shared_input():
    text = source(8).replace("%b:tensor<8xf32>", "%b:tensor<2x8xf32>")
    text = text.replace("tensor<8xf32>,tensor<f32>", "tensor<2x8xf32>,tensor<f32>")
    text = text.replace("affine_map<(d0,d1)->(d1)>", "affine_map<(d0,d1)->(d0,d1)>")
    transformed, count = apply_for_test(text, broadcast=True)
    assert count == 0 and "linalg.generic" in transformed
    # A syntactically projected but unused operand grants no sharing opportunity.
    unused = (
        source(8)
        .replace("%x,%y,%s0", "%x,%s0,%s0")
        .replace("%p,%y,%x", "%p,%s0,%x")
        .replace("%q,%y,%s0", "%q,%s0,%s0")
        .replace("%u,%y,%x", "%u,%s0,%x")
    )
    _, count = apply_for_test(unused, broadcast=True)
    assert count == 0


def test_broadcast_axis_handles_permuted_output_and_multiple_outer_dimensions(tmp_path):
    text = source(7).replace("tensor<2x7xf32>", "tensor<3x5x7xf32>")
    text = text.replace("(d0,d1)->(d0,d1)", "(d0,d1,d2)->(d0,d1,d2)")
    text = text.replace("(d0,d1)->(d1)", "(d0,d1,d2)->(d2)")
    text = text.replace("(d0,d1)->()", "(d0,d1,d2)->()")
    text = text.replace('iterator_types=["parallel","parallel"]', 'iterator_types=["parallel","parallel","parallel"]')
    # Only the destination/result uses the independent output permutation.
    text = text.replace("%c:tensor<3x5x7xf32>", "%c:tensor<7x3x5xf32>")
    text = text.replace("-> tensor<3x5x7xf32>", "-> tensor<7x3x5xf32>")
    text = text.replace("outs(%c:tensor<3x5x7xf32>)", "outs(%c:tensor<7x3x5xf32>)")
    text = text.replace("return %r:tensor<3x5x7xf32>", "return %r:tensor<7x3x5xf32>")
    needle = "affine_map<(d0,d1,d2)->(d0,d1,d2)>]"
    text = text.replace(needle, "affine_map<(d0,d1,d2)->(d2,d0,d1)>]")
    rng = np.random.default_rng(935)
    args = [
        rng.normal(size=(3, 5, 7)).astype("f4"),
        rng.normal(size=7).astype("f4"),
        np.array(3.0, dtype="f4"),
        np.full((7, 3, 5), 17.0, dtype="f4"),
    ]
    control, _ = run(tmp_path, text, args, (7, 3, 5), set())
    candidate, _ = run(tmp_path, text, args, (7, 3, 5), {BROADCAST_FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))


@pytest.mark.parametrize("option", [{"init_read": True}, {"fastmath": True}])
def test_broadcast_unsupported_source_keeps_llvm_identical(tmp_path, option):
    text = source(7, **option)
    control = lower_to_llvm_ir(text, workdir=tmp_path / "control")
    candidate = lower_to_llvm_ir(text, workdir=tmp_path / "candidate", features={BROADCAST_FEATURE})
    assert candidate == control


def test_broadcast_strict_function_and_invalid_width_refuse():
    text = source(8).replace("attributes {llvm.emit_c_interface}", "attributes {llvm.emit_c_interface, strictfp}")
    _, count = apply_for_test(text, broadcast=True)
    assert count == 0
    with pytest.raises(ValueError, match="two lanes"):
        apply_for_test(source(8), broadcast=True, lanes=4)


def multiplication_source(rows, width, *, unit_axis=True, alias=False):
    row_type = f"tensor<{rows}x1xf32>" if unit_axis else f"tensor<{rows}xf32>"
    row_map = "(d0,0)" if unit_axis else "(d0)"
    init = "%a" if alias else "%init"
    declaration = "" if alias else f", %init:tensor<{rows}x{width}xf32>"
    return f"""module {{ func.func @forward(%a:tensor<{rows}x{width}xf32>,%w:tensor<{width}xf32>,%row:{row_type}{declaration}) -> tensor<{rows}x{width}xf32> attributes {{llvm.emit_c_interface}} {{
      %scale=arith.constant 1.015625:f32
      %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d1)>,affine_map<(d0,d1)->{row_map}>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}}
      ins(%a,%w,%row:tensor<{rows}x{width}xf32>,tensor<{width}xf32>,{row_type}) outs({init}:tensor<{rows}x{width}xf32>) {{
       ^bb0(%x:f32,%weight:f32,%inv:f32,%unused:f32):
         %u=arith.mulf %x,%inv:f32
         %v=arith.mulf %weight,%u:f32
         %q=arith.mulf %v,%scale:f32
         linalg.yield %q:f32
      }} -> tensor<{rows}x{width}xf32>
      return %r:tensor<{rows}x{width}xf32>
    }} }}"""


@pytest.mark.parametrize("rows,width,alias", [(1, 1, False), (3, 5, False), (2, 8, True), (5, 11, True)])
def test_multiplication_packet_actual_source_order_and_tails(tmp_path, rows, width, alias):
    rng = np.random.default_rng(561)
    a = rng.normal(size=(rows, width)).astype("f4")
    w = rng.normal(size=width).astype("f4")
    row = np.linspace(0.12345, 2.56789, rows, dtype="f4").reshape(rows, 1)
    init = np.full(a.shape, 17.0, dtype="f4")
    old = a.copy()
    args = [a, w, row] + ([] if alias else [init])
    text = multiplication_source(rows, width, alias=alias)
    transformed, count = apply_for_test(text, lanes=4, multiplication=True)
    assert count == 1 and "linalg.generic" not in transformed
    assert "scalar_pointwise_multiplication_packet_4" in transformed
    control, _ = run(tmp_path, text, args, a.shape, set())
    candidate, ll = run(tmp_path, text, args, a.shape, {MULTIPLY_FOUR_FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    np.testing.assert_array_equal(a.view("u4"), old.view("u4"))
    assert "fmul float" in ll and "fdiv float" not in ll and "@llvm.fma.f32" not in ll


@pytest.mark.parametrize(
    "change",
    [
        "fma",
        "divide",
        "precision",
        "init_read",
        "constant_coordinate",
        "strict",
        "inner_strict",
        "fastmath",
        "call",
        "empty",
    ],
)
def test_multiplication_selector_refuses_unsupported_arithmetic(change):
    text = multiplication_source(3, 7)
    if change == "fma":
        text = text.replace("%q=arith.mulf %v,%scale:f32", "%q=math.fma %v,%scale,%x:f32")
    elif change == "divide":
        text = text.replace("%q=arith.mulf %v,%scale:f32", "%q=arith.divf %v,%scale:f32")
    elif change == "precision":
        text = text.replace("f32", "f64")
    elif change == "init_read":
        text = text.replace("%q=arith.mulf %v,%scale:f32", "%q=arith.mulf %v,%unused:f32")
    elif change == "constant_coordinate":
        text = text.replace("(d0,0)", "(d0,1)").replace("tensor<3x1xf32>", "tensor<3x2xf32>")
    elif change == "strict":
        text = text.replace("llvm.emit_c_interface", "llvm.emit_c_interface, strictfp")
    elif change == "inner_strict":
        text = text.replace("%x,%inv:f32", "%x,%inv {strictfp}:f32")
    elif change == "fastmath":
        text = text.replace("%x,%inv:f32", "%x,%inv fastmath<contract>:f32")
    elif change == "call":
        text = text.replace("module {", "module {func.func private @observe(f32)->f32", 1).replace(
            "%u=arith.mulf", "%z=func.call @observe(%x):(f32)->f32\n%u=arith.mulf"
        )
    else:
        text = text.replace("tensor<3x7xf32>", "tensor<3x0xf32>").replace("tensor<7xf32>", "tensor<0xf32>")
    transformed, count = apply_for_test(text, lanes=4, multiplication=True)
    assert count == 0 and "linalg.generic" in transformed


def test_multiplication_schedule_is_explicit_and_composes_disjoint_selector(tmp_path):
    default = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({MULTIPLY_FOUR_FEATURE}))
    assert selected.replace(MULTIPLY_FOUR_MARKER + ",", "") == default
    together = _upstream_pipeline(frozenset({FEATURE, MULTIPLY_FOUR_FEATURE}))
    assert MARKER in together and MULTIPLY_FOUR_MARKER in together
    text = multiplication_source(2, 5)
    control = lower_to_llvm_ir(text, workdir=tmp_path / "c")
    old_selected = lower_to_llvm_ir(text, workdir=tmp_path / "old", features={FEATURE})
    assert control == old_selected
    mixed = lower_to_llvm_ir(text, workdir=tmp_path / "mixed", features={FEATURE, MULTIPLY_FOUR_FEATURE})
    multiply = lower_to_llvm_ir(text, workdir=tmp_path / "multiply", features={MULTIPLY_FOUR_FEATURE})
    assert mixed == multiply


def two_multiplication_source(rows, width, *, cast=False, alias=False, residual=False):
    text = multiplication_source(rows, width, alias=alias)
    text = text.replace("%q=arith.mulf %v,%scale:f32", "%q=arith.addf %v,%scale:f32" if residual else "")
    if not residual:
        text = text.replace("linalg.yield %q:f32", "linalg.yield %v:f32")
    if cast:
        assert not alias
        text = text.replace(f"%a:tensor<{rows}x{width}xf32>", f"%a:tensor<{rows}x{width}xi32>")
        text = text.replace(f"ins(%a,%w,%row:tensor<{rows}x{width}xf32>", f"ins(%a,%w,%row:tensor<{rows}x{width}xi32>")
        text = text.replace("^bb0(%x:f32", "^bb0(%x:i32")
        text = text.replace(
            "%u=arith.mulf %x,%inv:f32", "%float=arith.sitofp %x:i32 to f32\n%u=arith.mulf %float,%inv:f32"
        )
    return text


@pytest.mark.parametrize(
    "rows,width,cast,alias,residual",
    [
        (1, 1, False, False, False),
        (3, 5, True, False, False),
        (2, 8, False, True, True),
        (5, 11, True, False, True),
        (3, 7, False, True, False),
        (2, 13, False, False, True),
    ],
)
def test_two_multiplications_source_order_cast_tails_live_inputs(tmp_path, rows, width, cast, alias, residual):
    words = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x3F800001,
            0xBF800001,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x7F812345,
            0x00800000,
            0x7F7FFFFF,
        ],
        dtype="u4",
    )
    if cast:
        a = np.resize(
            np.array([0, -1, 2147483647, -2147483648, 16777217, -16777217], dtype="i4"), rows * width
        ).reshape(rows, width)
    else:
        a = np.resize(words, rows * width).view("f4").reshape(rows, width)
    weight = np.linspace(-0.1234567, 1.9876543, width, dtype="f4")
    row = np.linspace(0.12345, 2.56789, rows, dtype="f4").reshape(rows, 1)
    init = np.full((rows, width), 17.0, dtype="f4")
    args = [a, weight, row] + ([] if alias else [init])
    original = [v.copy() for v in args]
    text = two_multiplication_source(rows, width, cast=cast, alias=alias, residual=residual)
    transformed, count = apply_for_test(text, lanes=4, multiplication=True, two_products=True)
    assert count == 1 and "scalar_pointwise_two_multiplications_packet_4" in transformed
    control, _ = run(tmp_path, text, args, (rows, width), set())
    candidate, ll = run(tmp_path, text, args, (rows, width), {TWO_MULTIPLY_FOUR_FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    for value, saved in zip(args, original):
        np.testing.assert_array_equal(value.view("u4"), saved.view("u4"))
    assert "fdiv float" not in ll and "@llvm.fma.f32" not in ll


@pytest.mark.parametrize(
    "change", ["one", "three", "divide", "precision", "init_read", "strict", "fastmath", "subtraction", "empty"]
)
def test_two_multiplications_refuses_other_numeric_and_effect_contracts(change):
    text = two_multiplication_source(3, 7, residual=True)
    if change == "one":
        text = text.replace("%v=arith.mulf %weight,%u:f32", "%v=arith.addf %weight,%u:f32")
    elif change == "three":
        text = text.replace("%q=arith.addf", "%q=arith.mulf")
    elif change == "divide":
        text = text.replace("%q=arith.addf", "%q=arith.divf")
    elif change == "precision":
        text = text.replace("f32", "f64")
    elif change == "init_read":
        text = text.replace("%v,%scale:f32", "%v,%unused:f32")
    elif change == "strict":
        text = text.replace("llvm.emit_c_interface", "llvm.emit_c_interface, strictfp")
    elif change == "fastmath":
        text = text.replace("%x,%inv:f32", "%x,%inv fastmath<contract>:f32")
    elif change == "subtraction":
        text = text.replace("%q=arith.addf", "%q=arith.subf")
    else:
        text = text.replace("tensor<3x7xf32>", "tensor<3x0xf32>").replace("tensor<7xf32>", "tensor<0xf32>")
    transformed, count = apply_for_test(text, lanes=4, multiplication=True, two_products=True)
    assert count == 0 and "linalg.generic" in transformed


def test_two_product_policy_is_separate_and_does_not_broaden_existing_selection(tmp_path):
    default = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({TWO_MULTIPLY_FOUR_FEATURE}))
    assert selected.replace(TWO_MULTIPLY_FOUR_MARKER + ",", "") == default
    together = _upstream_pipeline(frozenset({FEATURE, MULTIPLY_FOUR_FEATURE, TWO_MULTIPLY_FOUR_FEATURE}))
    assert all(marker in together for marker in [MARKER, MULTIPLY_FOUR_MARKER, TWO_MULTIPLY_FOUR_MARKER])
    text = two_multiplication_source(3, 5, cast=True, residual=True)
    control = lower_to_llvm_ir(text, workdir=tmp_path / "control")
    old = lower_to_llvm_ir(text, workdir=tmp_path / "old", features={MULTIPLY_FOUR_FEATURE})
    assert old == control
    old_text = multiplication_source(3, 5)
    old_plain = lower_to_llvm_ir(old_text, workdir=tmp_path / "old_plain")
    new_only = lower_to_llvm_ir(old_text, workdir=tmp_path / "new_only", features={TWO_MULTIPLY_FOUR_FEATURE})
    assert new_only == old_plain
    old_selected = lower_to_llvm_ir(old_text, workdir=tmp_path / "old_selected", features={MULTIPLY_FOUR_FEATURE})
    both = lower_to_llvm_ir(
        old_text, workdir=tmp_path / "both", features={MULTIPLY_FOUR_FEATURE, TWO_MULTIPLY_FOUR_FEATURE}
    )
    assert both == old_selected
    with pytest.raises(ValueError, match="explicit multiplication"):
        apply_for_test(text, two_products=True)
