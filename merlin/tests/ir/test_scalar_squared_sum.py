"""Actual source-order squared reductions and tensor alias ownership."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.scalar_squared_sum import FEATURE, MARKER, apply_for_test, edit_pipeline
from merlin.llvmlower.toolchain import clang


def source(rows, width, *, transpose=False, reverse=False, live=False):
    inp = f"tensor<{width}x{rows}xf32>" if transpose else f"tensor<{rows}x{width}xf32>"
    out = f"tensor<{rows}xf32>"
    input_map = "(d1,d0)" if transpose else "(d0,d1)"
    return f"""module {{func.func @forward(%a:{inp},%seed:{out}) -> {out} attributes {{llvm.emit_c_interface}} {{
      %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->{input_map}>,affine_map<(d0,d1)->(d0)>],iterator_types=["parallel","reduction"]}}
      ins(%a:{inp}) outs(%seed:{out}) {{
       ^bb0(%x:f32,%z:f32):
        %p=arith.mulf %x,%x:f32
        %v=arith.addf {"%z,%p" if reverse else "%p,%z"}:f32
        linalg.yield %v:f32
      }} -> {out}
      {'%again=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]} ins(%r,%seed:' + out + "," + out + ") outs(%seed:" + out + ") {^bb0(%x:f32,%y:f32,%unused:f32):%v=arith.addf %x,%y:f32 linalg.yield %v:f32} -> " + out if live else ""}
      return {"%again" if live else "%r"}:{out}
    }} }}"""


def run(tmp_path, src, args, shape, features):
    key = hashlib.sha256((src + str(sorted(features))).encode()).hexdigest()[:16]
    root = tmp_path / key
    root.mkdir(parents=True)
    text = lower_to_llvm_ir(src, workdir=root, features=features)
    path = root / "model.ll"
    path.write_text(text)
    subprocess.run(
        [
            str(clang()),
            "-O3",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            str(path),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(root / "model.so"),
        ],
        check=True,
        capture_output=True,
    )
    out = np.full(shape, -999.0, dtype="f4")
    HostModel.load(str(root / "model.so"))([(v.ctypes.data, v.shape) for v in [*args, out]])
    return out, text


@pytest.mark.parametrize(
    "rows,width,transpose,reverse,live",
    [
        (1, 1, False, False, False),
        (3, 7, False, True, True),
        (5, 11, True, False, True),
        (2, 0, False, True, False),
        (0, 5, True, False, False),
    ],
)
def test_native_nonzero_initial_tail_maps_empty_and_live_seed(tmp_path, rows, width, transpose, reverse, live):
    rng = np.random.default_rng(673)
    a = rng.normal(size=(rows, width)).astype("f4")
    seed = rng.normal(size=rows).astype("f4")
    if rows and width:
        a[0, 0] = np.float32(1.0000001192092896)
        seed[0] = np.float32(-1.000000238418579)
    expected = seed.copy()
    for k in range(width):
        product = np.float32(a[:, k] * a[:, k])
        expected = np.float32(expected + product) if reverse else np.float32(product + expected)
    if live:
        expected = np.float32(expected + seed)
    storage = np.ascontiguousarray(a.T) if transpose else a
    text = source(rows, width, transpose=transpose, reverse=reverse, live=live)
    changed, count = apply_for_test(text)
    assert count == 1 and "scalar_squared_sum_accumulator" in changed and "scf.for" in changed
    originals = [v.copy() for v in [storage, seed]]
    control_args = [storage.copy(), seed.copy()]
    selected_args = [storage.copy(), seed.copy()]
    before, _ = run(tmp_path, text, control_args, seed.shape, set())
    after, llvm = run(tmp_path, text, selected_args, seed.shape, {FEATURE})
    np.testing.assert_array_equal(after.view("u4"), before.view("u4"))
    np.testing.assert_array_equal(after.view("u4"), expected.view("u4"))
    assert "llvm.fma" not in llvm and "fmul contract" not in llvm
    # Upstream may forward a writable seed to the tensor result. Compare both
    # actual boundary effects; a live seed additionally retains its input value.
    for before_arg, after_arg in zip(control_args, selected_args):
        np.testing.assert_array_equal(before_arg.view("u4"), after_arg.view("u4"))
    np.testing.assert_array_equal(selected_args[0].view("u4"), originals[0].view("u4"))
    if live:
        np.testing.assert_array_equal(selected_args[1].view("u4"), originals[1].view("u4"))


def test_runtime_argument_alias_preserves_both_tensor_values(tmp_path):
    base = np.array([1.0000001192092896, -0.0, 3.0, -2.0, 1e-20, 7.0], dtype="f4")
    a = base.reshape(3, 2)
    seed = base[:3]
    old = base.copy()
    text = source(3, 2, live=True)
    before, _ = run(tmp_path, text, [a, seed], seed.shape, set())
    after, _ = run(tmp_path, text, [a, seed], seed.shape, {FEATURE})
    np.testing.assert_array_equal(after.view("u4"), before.view("u4"))
    np.testing.assert_array_equal(base.view("u4"), old.view("u4"))


def test_raw_signed_zero_subnormal_infinity_and_nan_inputs(tmp_path):
    raw = np.array(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x3F800001,
            0xBF800001,
            0x7F800000,
            0xFF800000,
            0x7FC00123,
            0x7F800123,
            0x7F7FFFFF,
            0xFF7FFFFF,
        ],
        dtype="u4",
    )
    a = np.tile(raw.view("f4"), (3, 1))
    seed = np.array([-0.0, -1.000000238418579, 2.0], dtype="f4")
    text = source(3, 12)
    before, _ = run(tmp_path, text, [a.copy(), seed.copy()], seed.shape, set())
    after, _ = run(tmp_path, text, [a.copy(), seed.copy()], seed.shape, {FEATURE})
    np.testing.assert_array_equal(after.view("u4"), before.view("u4"))


@pytest.mark.parametrize(
    "change",
    ["dynamic", "fastmath", "extra", "other_product", "constant_map", "strictfp", "strictfp_parent", "library_call"],
)
def test_refuses_unproved_source_semantics(change):
    text = source(3, 5)
    if change == "dynamic":
        text = text.replace("3x5xf32", "3x?xf32")
    elif change == "fastmath":
        text = text.replace("mulf %x,%x:f32", "mulf %x,%x fastmath<contract>:f32")
    elif change == "extra":
        text = text.replace("linalg.yield %v:f32", "%q=math.absf %v:f32 linalg.yield %q:f32")
    elif change == "other_product":
        text = text.replace("mulf %x,%x:f32", "mulf %x,%z:f32")
    elif change == "constant_map":
        text = text.replace("tensor<3x5xf32>", "tensor<1x5xf32>").replace("->(d0,d1)>", "->(0,d1)>")
    elif change == "strictfp":
        text = text.replace(
            'iterator_types=["parallel","reduction"]', 'iterator_types=["parallel","reduction"],test.strictfp'
        )
    elif change == "strictfp_parent":
        text = text.replace("attributes {llvm.emit_c_interface}", "attributes {llvm.emit_c_interface,test.strictfp}")
    else:
        text = text.replace(
            'iterator_types=["parallel","reduction"]', 'iterator_types=["parallel","reduction"],library_call="unproven"'
        )
    unchanged, count = apply_for_test(text)
    assert count == 0 and "linalg.generic" in unchanged and "scalar_squared_sum_accumulator" not in unchanged


def test_default_and_explicit_pipeline_marker():
    baseline = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({FEATURE}))
    assert selected.replace(MARKER + ",", "") == baseline
    assert selected.index(MARKER) < selected.index("one-shot-bufferize")
    for stages in [[], ["one-shot-bufferize", "one-shot-bufferize"], [MARKER, "one-shot-bufferize"]]:
        original = list(stages)
        with pytest.raises(ValueError):
            edit_pipeline(stages)
        assert stages == original


def test_default_actual_lowering_is_byte_identical(tmp_path):
    text = source(2, 3)
    _, first = run(tmp_path / "one", text, [np.ones((2, 3), dtype="f4"), np.zeros(2, dtype="f4")], (2,), set())
    _, second = run(tmp_path / "two", text, [np.ones((2, 3), dtype="f4"), np.zeros(2, dtype="f4")], (2,), frozenset())
    assert first == second


def test_actual_batched_permuted_output_map(tmp_path):
    text = """module {func.func @forward(%a:tensor<2x3x7xf32>,%seed:tensor<3x2xf32>) -> tensor<3x2xf32> attributes {llvm.emit_c_interface} {
    %r=linalg.generic {indexing_maps=[affine_map<(d0,d1,d2)->(d0,d1,d2)>,affine_map<(d0,d1,d2)->(d1,d0)>],iterator_types=["parallel","parallel","reduction"]}
    ins(%a:tensor<2x3x7xf32>) outs(%seed:tensor<3x2xf32>) {^bb0(%x:f32,%z:f32):
    %p=arith.mulf %x,%x:f32 %v=arith.addf %z,%p:f32 linalg.yield %v:f32} -> tensor<3x2xf32>
    return %r:tensor<3x2xf32>}}"""
    rng = np.random.default_rng(987)
    a = rng.normal(size=(2, 3, 7)).astype("f4")
    seed = rng.normal(size=(3, 2)).astype("f4")
    expected = seed.T.copy()
    for k in range(7):
        expected = np.float32(expected + np.float32(a[:, :, k] * a[:, :, k]))
    before, _ = run(tmp_path, text, [a.copy(), seed.copy()], seed.shape, set())
    after, _ = run(tmp_path, text, [a.copy(), seed.copy()], seed.shape, {FEATURE})
    np.testing.assert_array_equal(after.view("u4"), before.view("u4"))
    np.testing.assert_array_equal(after.view("u4"), expected.T.copy().view("u4"))


@pytest.mark.parametrize("width", [0, 5])
def test_rank_zero_result_preserves_source_initialization(tmp_path, width):
    text = f"""module {{func.func @forward(%a:tensor<{width}xf32>,%seed:tensor<f32>) -> tensor<f32> attributes {{llvm.emit_c_interface}} {{
    %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}}
    ins(%a:tensor<{width}xf32>) outs(%seed:tensor<f32>) {{^bb0(%x:f32,%z:f32):
    %p=arith.mulf %x,%x:f32 %v=arith.addf %p,%z:f32 linalg.yield %v:f32}} -> tensor<f32>
    return %r:tensor<f32>}}}}"""
    a = np.arange(width, dtype="f4") / np.float32(7)
    seed = np.array(-0.0, dtype="f4")
    before, _ = run(tmp_path, text, [a.copy(), seed.copy()], (), set())
    after, _ = run(tmp_path, text, [a.copy(), seed.copy()], (), {FEATURE})
    np.testing.assert_array_equal(after.view("u4"), before.view("u4"))
