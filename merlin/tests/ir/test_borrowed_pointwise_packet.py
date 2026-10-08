"""Explicit borrowed writer lanes through the shipped ordinary compiler feature."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects.linalg.ops import GenericOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.scalar_pointwise_packet import (
    BORROWED_FEATURE,
    BORROWED_MARKER,
    FEATURE,
    BorrowedPointwiseEffects,
    apply_for_test,
    bind_borrowed_pointwise_packet,
)
from merlin.llvmlower.toolchain import clang

EFFECTS = BorrowedPointwiseEffects(True, True, True, True, True)


def source(rows, columns, *, input_pitch=None, output_pitch=None, init_read=False, fastmath=False):
    input_pitch = input_pitch or columns + 3
    output_pitch = output_pitch or columns + 5
    flags = " fastmath<contract>" if fastmath else ""
    text = f"""module {{ func.func @writer(
      %a:memref<{rows}x{columns}xf32,strided<[{input_pitch},1]>>,
      %b:memref<{columns}xf32>,
      %out:memref<{rows}x{columns}xf32,strided<[{output_pitch},1]>>) {{
    linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
      affine_map<(d0,d1)->(d1)>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}}
      ins(%a,%b:memref<{rows}x{columns}xf32,strided<[{input_pitch},1]>>,memref<{columns}xf32>)
      outs(%out:memref<{rows}x{columns}xf32,strided<[{output_pitch},1]>>) {{
      ^bb0(%x:f32,%y:f32,%z:f32):
       %p=math.fma %x,%y,%x{flags}:f32
       %q=math.fma %p,%y,%x{flags}:f32
       %u=math.fma %q,%y,%x{flags}:f32
       %v=math.fma %u,%y,%x{flags}:f32
       %w=arith.divf %v,%y:f32
       {"%r0=arith.addf %w,%z:f32" if init_read else ""}
       linalg.yield {"%r0" if init_read else "%w"}:f32
      }}
      return
    }} }}"""
    return text


def bind(text, effects=EFFECTS):
    module = parse_mlir_text(text)
    op = next(op for op in module.walk() if isinstance(op, GenericOp))
    bind_borrowed_pointwise_packet(op, effects=effects)
    return str(module) + "\n"


def lower(tmp_path, text, selected):
    features = frozenset([BORROWED_FEATURE]) if selected else frozenset()
    pipeline = (
        _upstream_pipeline(features)
        .replace("convert-func-to-llvm", "convert-func-to-llvm{use-bare-ptr-memref-call-conv}")
        .replace("convert-math-to-libm,convert-math-to-llvm", "convert-math-to-llvm,convert-math-to-libm")
    )
    return lower_to_llvm_ir(text, workdir=tmp_path, pipeline=pipeline, features=features)


def compiled(tmp_path, text, selected):
    ll = lower(tmp_path, text, selected)
    ir = tmp_path / "writer.ll"
    ir.write_text(ll)
    so = tmp_path / "writer.so"
    subprocess.run(
        [str(clang()), "-O3", "-fPIC", "-ffp-contract=off", "-shared", str(ir), "-lm", "-o", str(so)],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.writer.argtypes = [ctypes.c_void_p] * 3
    return lib, ll


@pytest.mark.parametrize("rows,columns,alias", [(1, 1, False), (3, 5, False), (2, 8, True), (5, 11, True)])
def test_native_original_words_pitches_tails_readonly_alias_modes(tmp_path, rows, columns, alias):
    text = bind(source(rows, columns))
    original, control = compiled(tmp_path / "c", text, False)
    candidate, selected = compiled(tmp_path / "s", text, True)
    if columns > 1:
        assert selected.count("@llvm.fma.f32(") > control.count("@llvm.fma.f32(")
    rng = np.random.default_rng(8297)
    a = rng.normal(size=(rows, columns + 3)).astype("f4")
    b = a[0, :columns] if alias else rng.uniform(0.5, 1.5, columns).astype("f4")
    saved_a, saved_b = a.tobytes(), b.tobytes()
    fenv = ctypes.CDLL(None)
    try:
        for mode in [0, 0x400, 0x800, 0xC00]:
            assert fenv.fesetround(mode) == 0
            outputs = []
            for lib in [original, candidate]:
                out = np.full(64 + rows * (columns + 5) + 64, np.float32(73), "f4")
                lib.writer(a.ctypes.data, b.ctypes.data, out.ctypes.data + 64 * 4)
                assert fenv.fegetround() == mode
                assert np.all(out[:64] == 73) and np.all(out[-64:] == 73)
                parent = out[64:-64].reshape(rows, columns + 5)
                assert np.all(parent[:, columns:] == 73)
                outputs.append(out.view("u4").copy())
            np.testing.assert_array_equal(outputs[0], outputs[1])
            assert a.tobytes() == saved_a and b.tobytes() == saved_b
    finally:
        fenv.fesetround(0)


@pytest.mark.parametrize(
    "change",
    [
        "missing_contract",
        "init_read",
        "fastmath",
        "strict_scope",
        "overlapping_output",
        "dynamic_stride",
        "unknown_call",
        "strict_generic",
    ],
)
def test_incomplete_or_unsupported_writer_keeps_default_ir(tmp_path, change):
    text = source(
        3,
        7,
        init_read=change == "init_read",
        fastmath=change == "fastmath",
        output_pitch=1 if change == "overlapping_output" else None,
    )
    if change != "missing_contract":
        text = bind(text)
    if change == "strict_scope":
        text = text.replace("func.func @writer", "func.func @writer").replace(">>) {", ">>) attributes {strictfp} {", 1)
    elif change == "dynamic_stride":
        text = text.replace("strided<[10, 1]>", "strided<[?, 1]>")
    elif change == "strict_generic":
        text = text.replace("merlin.borrowed_pointwise_effects =", "strictfp, merlin.borrowed_pointwise_effects =", 1)
        assert "strictfp," in text
    elif change == "unknown_call":
        text = text.replace("builtin.module {", "builtin.module { func.func private @opaque(f32)->f32\n", 1).replace(
            "%p = math.fma", "%unknown = func.call @opaque(%x) : (f32)->f32\n      %p = math.fma"
        )
    unchanged, count = apply_for_test(text, borrowed=True)
    assert count == 0 and "linalg.generic" in unchanged
    if change == "dynamic_stride":
        # The ordinary bare-pointer ABI cannot lower a dynamic stride. This is
        # a structural refusal test, not an upstream ABI readiness claim.
        return
    control = lower(tmp_path / "c", text, False)
    selected = lower(tmp_path / "s", text, True)
    assert selected == control


@pytest.mark.parametrize("permission", list(vars(EFFECTS)))
def test_explicit_missing_permission_refuses_before_mutation(permission):
    module = parse_mlir_text(source(3, 5))
    op = next(op for op in module.walk() if isinstance(op, GenericOp))
    before = str(module)
    with pytest.raises(ValueError, match="effect proof"):
        bind_borrowed_pointwise_packet(op, effects=replace(EFFECTS, **{permission: False}))
    assert str(module) == before


def test_feature_is_normal_optional_and_composes_with_tensor_packets():
    default = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset([BORROWED_FEATURE]))
    assert selected.replace(BORROWED_MARKER + ",", "") == default
    together = _upstream_pipeline(frozenset([FEATURE, BORROWED_FEATURE]))
    assert together.count(BORROWED_MARKER) == 1


def test_original_output_coordinate_permutation_is_preserved(tmp_path):
    text = source(3, 5)
    text = text.replace("memref<3x5xf32,strided<[10,1]>>", "memref<5x3xf32,strided<[8,1]>>")
    text = text.replace("affine_map<(d0,d1)->(d0,d1)>],iterator_types", "affine_map<(d0,d1)->(d1,d0)>],iterator_types")
    text = bind(text)
    original, _ = compiled(tmp_path / "c", text, False)
    candidate, _ = compiled(tmp_path / "s", text, True)
    rng = np.random.default_rng(593)
    a = rng.normal(size=(3, 8)).astype("f4")
    b = rng.uniform(0.5, 1.5, 5).astype("f4")
    inputs = a.tobytes(), b.tobytes()
    outputs = []
    for lib in [original, candidate]:
        out = np.full(64 + 5 * 8 + 64, np.float32(73), "f4")
        lib.writer(a.ctypes.data, b.ctypes.data, out.ctypes.data + 64 * 4)
        assert np.all(out[:64] == 73) and np.all(out[-64:] == 73)
        assert np.all(out[64:-64].reshape(5, 8)[:, 3:] == 73)
        outputs.append(out.view("u4"))
    np.testing.assert_array_equal(*outputs)
    assert (a.tobytes(), b.tobytes()) == inputs


def test_unknown_affine_layout_refuses_without_throwing():
    text = bind(source(3, 7))
    text = text.replace("strided<[10, 1]>", "affine_map<(d0,d1)->(d0 floordiv 2, d1, d0 mod 2)>")
    unchanged, count = apply_for_test(text, borrowed=True)
    assert count == 0 and "linalg.generic" in unchanged


@pytest.mark.parametrize("rows,columns,rank_one,replacements", [(3, 5, False, 1), (1, 1, True, 1), (1, 3, True, 2)])
def test_enclosing_source_trace_survives_outer_loop_and_scalar_tail(tmp_path, rows, columns, rank_one, replacements):
    from xdsl.dialects.builtin import StringAttr

    text = source(rows, columns)
    if rank_one:
        for pitch in [columns + 3, columns + 5]:
            text = text.replace(f"{rows}x{columns}xf32,strided<[{pitch},1]>", f"{columns}xf32")
        text = text.replace("(d0,d1)->(d0,d1)", "(d0)->(d0)").replace("(d0,d1)->(d1)", "(d0)->(d0)")
        text = text.replace('["parallel","parallel"]', '["parallel"]')
    module = parse_mlir_text(text)
    writer = next(op for op in module.walk() if isinstance(op, GenericOp))
    writer.attributes.update(
        {
            "prov.srcid": StringAttr("source.writer"),
            "prov.role": StringAttr("independent_output"),
            "prov.transforms": StringAttr("prior_transform"),
        }
    )
    bind_borrowed_pointwise_packet(writer, effects=EFFECTS)
    transformed, count = apply_for_test(str(module), borrowed=True)
    assert count == 1 and "merlin.borrowed_pointwise_effects" not in transformed
    # Upstream's custom scf.for printer places arbitrary attributes after the
    # region; the current xDSL custom parser cannot consume that spelling.
    # Reparse with the actual upstream parser and use its generic serialization
    # for the structural source-trace assertion.
    from merlin.llvmlower.toolchain import m2m_python

    serialized = tmp_path / "transformed.mlir"
    serialized.write_text(transformed)
    generic = subprocess.run(
        [
            str(m2m_python()),
            "-c",
            "from torch_mlir import ir; import sys; c=ir.Context(); "
            "m=ir.Module.parse(open(sys.argv[1]).read(),c); m.operation.verify(); "
            "print(m.operation.get_asm(print_generic_op_form=True))",
            str(serialized),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = parse_mlir_text(generic)
    traced = [op for op in result.walk() if "prov.srcid" in op.attributes]
    assert len(traced) == replacements
    assert all(op.name in {"scf.for", "memref.store"} for op in traced)
    for op in traced:
        assert op.attributes["prov.srcid"] == StringAttr("source.writer")
        assert op.attributes["prov.role"] == StringAttr("independent_output")
        assert op.attributes["prov.transforms"] == StringAttr("prior_transform,borrowed_scalar_pointwise_packet_2")
