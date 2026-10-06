"""Actual alias/ownership behavior at the upstream result-conversion seam."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.bufferized_result_identity import FEATURE, edit_pipeline
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import MULTIPLY_FOUR_FEATURE
from merlin.llvmlower.toolchain import clang


def source(rows, width, *, live_input=False, repeated_input=False):
    extra = (
        """%sum = linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}
      ins(%r,%a:TYPE,TYPE) outs(%empty:TYPE) {
       ^bb0(%x:f32,%y:f32,%out:f32):
        %z=arith.addf %x,%y:f32
        linalg.yield %z:f32
      } -> TYPE"""
        if live_input
        else ""
    )
    tensor = f"tensor<{rows}x{width}xf32>"
    extra = extra.replace("TYPE", tensor)
    return f"""module {{func.func @forward(%a:{tensor},%b:tensor<{width}xf32>) -> {tensor} attributes {{llvm.emit_c_interface}} {{
      %c=arith.constant 0.375:f32
      %empty=tensor.empty():{tensor}
      %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d1)>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}}
       ins(%a,%b:{tensor},tensor<{width}xf32>) outs(%a:{tensor}) {{
        ^bb0(%x:f32,%y:f32,%unused:f32):
         %u=arith.mulf %x,%c:f32
         %v=arith.mulf %u,{"%x" if repeated_input else "%y"}:f32
         %w=arith.mulf %v,%c:f32
         linalg.yield %w:f32
       }} -> {tensor}
      {extra}
      return {"%sum" if live_input else "%r"}:{tensor}
    }} }}"""


def run(tmp_path, text, args, shape, features):
    key = hashlib.sha256((text + str(sorted(features))).encode()).hexdigest()[:16]
    root = tmp_path / key
    root.mkdir()
    ll = lower_to_llvm_ir(text, workdir=root, features=features)
    (root / "model.ll").write_text(ll)
    subprocess.run(
        [
            str(clang()),
            "-O3",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            str(root / "model.ll"),
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
    return out, ll


@pytest.mark.parametrize(
    "rows,width,live,repeated",
    [
        (1, 1, False, False),
        (3, 7, False, True),
        (2, 8, True, False),
        (5, 11, True, True),
        (2, 0, True, False),
    ],
)
def test_actual_source_live_alias_repeated_input_empty_and_tails(tmp_path, rows, width, live, repeated):
    rng = np.random.default_rng(1429)
    a = rng.normal(size=(rows, width)).astype("f4")
    b = rng.normal(size=width).astype("f4")
    saved = [v.copy() for v in [a, b]]
    text = source(rows, width, live_input=live, repeated_input=repeated)
    control, _ = run(tmp_path, text, [a, b], a.shape, {MULTIPLY_FOUR_FEATURE})
    candidate, _ = run(tmp_path, text, [a, b], a.shape, {MULTIPLY_FOUR_FEATURE, FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    for value, original in zip([a, b], saved):
        np.testing.assert_array_equal(value.view("u4"), original.view("u4"))


def test_complete_loop_result_forwards_output_without_heap_copy(tmp_path):
    # A result rooted in a fresh tensor is the actual problematic seam. Input
    # arguments remain live and separate; upstream proves its allocation can
    # become the public output instead of a temporary followed by a copy.
    text = source(3, 9).replace("outs(%a:", "outs(%empty:")
    rng = np.random.default_rng(61)
    args = [rng.normal(size=(3, 9)).astype("f4"), rng.normal(size=9).astype("f4")]
    control, before = run(tmp_path, text, args, (3, 9), {MULTIPLY_FOUR_FEATURE})
    candidate, after = run(tmp_path, text, args, (3, 9), {MULTIPLY_FOUR_FEATURE, FEATURE})
    np.testing.assert_array_equal(candidate.view("u4"), control.view("u4"))
    assert "call ptr @malloc" in before and "call void @llvm.memcpy." in before
    assert "call ptr @malloc" not in after and "call void @llvm.memcpy." not in after


def test_multiple_results_keep_live_original_tensor_and_ownership(tmp_path):
    text = source(3, 7, live_input=True)
    tensor = "tensor<3x7xf32>"
    text = text.replace(f"-> {tensor} attributes", f"-> ({tensor},{tensor}) attributes", 1)
    text = text.replace(f"return %sum:{tensor}", f"return %sum,%a:{tensor},{tensor}")
    rng = np.random.default_rng(932)
    inputs = [rng.normal(size=(3, 7)).astype("f4"), rng.normal(size=7).astype("f4")]
    original = [v.copy() for v in inputs]
    results = []
    for name, features in [
        ("control", {MULTIPLY_FOUR_FEATURE}),
        ("selected", {MULTIPLY_FOUR_FEATURE, FEATURE}),
    ]:
        root = tmp_path / name
        root.mkdir()
        ll = lower_to_llvm_ir(text, workdir=root, features=features)
        (root / "model.ll").write_text(ll)
        subprocess.run(
            [
                str(clang()),
                "-O3",
                "-shared",
                "-fPIC",
                "-ffp-contract=off",
                str(root / "model.ll"),
                str(mlir_runtime_c()),
                "-lm",
                "-o",
                str(root / "model.so"),
            ],
            check=True,
            capture_output=True,
        )
        outputs = [np.full((3, 7), -999.0, dtype="f4") for _ in range(2)]
        HostModel.load(str(root / "model.so"))([(v.ctypes.data, v.shape) for v in [*inputs, *outputs]])
        results.append(outputs)
        np.testing.assert_array_equal(outputs[1].view("u4"), original[0].view("u4"))
        for value, saved in zip(inputs, original):
            np.testing.assert_array_equal(value.view("u4"), saved.view("u4"))
    for before, after in zip(*results):
        np.testing.assert_array_equal(before.view("u4"), after.view("u4"))


def test_default_pipeline_identity_and_exact_insertion():
    baseline = _upstream_pipeline(frozenset()).split(",")
    selected = _upstream_pipeline(frozenset({FEATURE})).split(",")
    i = next(i for i, p in enumerate(selected) if p.startswith("buffer-results-to-out-params"))
    assert selected[i - 1] == "canonicalize"
    assert selected[: i - 1] + selected[i:] == baseline
    assert edit_pipeline(selected) == selected


@pytest.mark.parametrize(
    "passes",
    [
        [],
        ["one-shot-bufferize"],
        ["buffer-results-to-out-params"],
        ["buffer-results-to-out-params", "one-shot-bufferize"],
        ["one-shot-bufferize", "one-shot-bufferize", "buffer-results-to-out-params"],
        ["one-shot-bufferize", "buffer-results-to-out-params", "buffer-results-to-out-params"],
    ],
)
def test_ambiguous_or_missing_anchor_refuses_without_mutation(passes):
    original = list(passes)
    with pytest.raises(ValueError, match="one ordered"):
        edit_pipeline(passes)
    assert passes == original


def test_ordinary_unselected_source_preserves_actual_default_ir(tmp_path):
    text = source(2, 7)
    control = lower_to_llvm_ir(text, workdir=tmp_path / "control")
    explicit = lower_to_llvm_ir(text, workdir=tmp_path / "explicit", features={FEATURE})
    assert explicit == control
