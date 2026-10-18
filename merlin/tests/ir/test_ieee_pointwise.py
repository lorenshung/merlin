# ruff: noqa: E501 -- static MLIR fixtures retain one operation per line
"""IEEE classification covers raw words; finite arithmetic is absent from readout."""

import math
import random
import struct

import pytest
from xdsl.dialects.builtin import bf16, f16, f32, f64
from xdsl.ir import Block

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.ieee_pointwise import special


@pytest.mark.parametrize("typ", [f16, bf16, f32, f64])
def test_classification_does_not_quiet_or_round_inputs(typ):
    block = Block(arg_types=[typ])

    def emit(op):
        block.add_op(op)
        return op.results[0]

    result = special(block.args[0], emit)
    width = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}[str(typ)]
    rng = random.Random(842)
    samples = range(1 << 16) if width == 16 else [rng.getrandbits(width) for _ in range(4096)]
    for raw in samples:
        values = {block.args[0]: raw}
        for op in block.ops:
            args = [values[v] for v in op.operands]
            if op.name == "arith.constant":
                value = op.value.value.data
            elif op.name == "arith.bitcast":
                value = args[0]
            elif op.name == "arith.andi":
                value = args[0] & args[1]
            elif op.name == "arith.cmpi":
                value = args[0] == args[1]
            else:
                raise AssertionError(op.name)
            values[op.results[0]] = value
        data = (raw << 16).to_bytes(4, "little") if typ == bf16 else raw.to_bytes(width // 8, "little")
        value = struct.unpack("<f" if typ == bf16 else {16: "<e", 32: "<f", 64: "<d"}[width], data)[0]
        assert bool(values[result]) == (not math.isfinite(value))


@pytest.mark.parametrize("operation", ["addf", "subf", "mulf"])
def test_floating_special_arithmetic_occurs_only_in_the_affected_branch(operation):
    source = parse_mlir_text(f"""module {{
      func.func @forward(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {{
        %e=tensor.empty() : tensor<2xf32>
        %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
          ins(%x,%y : tensor<2xf32>,tensor<2xf32>) outs(%e : tensor<2xf32>) {{
        ^bb0(%a:f32,%b:f32,%unused:f32):
          %v=arith.{operation} %a,%b : f32
          linalg.yield %v : f32
        }} -> tensor<2xf32>
        return %r : tensor<2xf32>
      }} }}""")
    _, proofs = rewrite_floating_components(source, [(0, 1), (0, 1)], ieee_pointwise=True)
    assert proofs[0]["ieee_special_readout"] == "affected_elements_only"
    branches = [op for op in source.walk() if op.name == "scf.if"]
    assert len(branches) == 1
    chosen, retained = branches[0].regions
    assert not any(op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"} for op in chosen.walk())
    assert [op.name for op in retained.walk()] == ["scf.yield"]
    source.verify()


@pytest.mark.parametrize("typ", [f16, bf16, f32, f64])
@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_explicit_special_words_match_native_cpu(typ, operation):
    torch = pytest.importorskip("torch")
    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.llvmlower.ieee_pointwise import special_arithmetic

    fraction, exponent, bias = FORMAT[str(typ)]
    width = 1 + fraction + exponent
    mask, sign = (1 << width) - 1, 1 << (fraction + exponent)
    infinity = ((1 << exponent) - 1) << fraction
    nan = infinity | 1 << (fraction - 1)
    words = [0, sign, bias << fraction, (bias << fraction) | sign, infinity, infinity | sign, nan, nan | sign]
    block = Block(arg_types=[typ, typ])

    def emit(op):
        block.add_op(op)
        return op.results[0]

    result = special_arithmetic("arith." + operation + "f", block.args, emit)
    dtype = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}[str(typ)]
    storage = {16: torch.int16, 32: torch.int32, 64: torch.int64}[width]
    for a in words:
        for b in words:
            if max(a & ~sign, b & ~sign) < infinity:
                continue
            if a & ~sign > infinity and b & ~sign > infinity and a != b:
                continue  # The independent admission audit refuses competing payloads.
            values = dict(zip(block.args, [a, b]))
            for op in block.ops:
                args = [values[v] for v in op.operands]
                if op.name == "arith.constant":
                    value = op.value.value.data
                elif op.name == "arith.bitcast":
                    value = args[0]
                elif op.name == "arith.andi":
                    value = args[0] & args[1]
                elif op.name == "arith.ori":
                    value = args[0] | args[1]
                elif op.name == "arith.xori":
                    value = args[0] ^ args[1]
                elif op.name == "arith.select":
                    value = args[1] if args[0] else args[2]
                elif op.name == "arith.cmpi":
                    value = {0: args[0] == args[1], 1: args[0] != args[1], 6: args[0] < args[1], 8: args[0] > args[1]}[
                        op.predicate.value.data
                    ]
                else:
                    raise AssertionError(op.name)
                t = op.results[0].type
                if hasattr(t, "width"):
                    value &= (1 << t.width.data) - 1
                values[op.results[0]] = value
            pair = torch.tensor([v if v < sign else v - (1 << width) for v in [a, b]], dtype=storage).view(dtype)
            expected = int(getattr(torch, operation)(pair[0], pair[1]).view(storage)) & mask
            assert values[result] == expected


def test_special_readout_does_not_taint_an_unread_shape_operand():
    from merlin.llvmlower.ieee_pointwise import dependence_proof, overlay

    module = parse_mlir_text('''module {
      func.func @forward(%x: tensor<2xf32>, %y: tensor<2xf32>, %shape: tensor<2xf32>) -> tensor<2xf32> {
        %e=tensor.empty() : tensor<2xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y,%shape : tensor<2xf32>,tensor<2xf32>,tensor<2xf32>) outs(%e : tensor<2xf32>) {
        ^bb0(%a:f32,%b:f32,%unread:f32,%unused:f32):
          %v=arith.addf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<2xf32>
        return %r : tensor<2xf32>
      }}''')
    source = next(op for op in module.walk() if op.name == "linalg.generic")
    proof = dependence_proof(source)
    assert [d["operand"] for d in proof["input_dependencies"]] == [0, 1]
    readout = Block(arg_types=[f32] * 4)

    def emit(op):
        readout.add_op(op)
        return op.results[0]

    overlay(source, readout.args[:3], readout.args[3], emit)
    assert not readout.args[2].uses
    assert readout.args[0].uses and readout.args[1].uses


def test_overlay_dependence_follows_permuted_and_broadcast_input_maps():
    import numpy as np

    from merlin.llvmlower.ieee_pointwise import dependence_proof

    source = parse_mlir_text('''module {
      func.func @forward(%x: tensor<3x2xf32>, %y: tensor<3xf32>, %unread: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %e=tensor.empty() : tensor<2x3xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d1,d0)>,affine_map<(d0,d1)->(d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}
          ins(%x,%y,%unread : tensor<3x2xf32>,tensor<3xf32>,tensor<2x3xf32>) outs(%e : tensor<2x3xf32>) {
        ^bb0(%a:f32,%b:f32,%ignored:f32,%unused:f32):
          %v=arith.addf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<2x3xf32>
        return %r : tensor<2x3xf32>
      }}''')
    generic = next(op for op in source.walk() if op.name == "linalg.generic")
    proof = dependence_proof(generic)
    assert proof["output_shape"] == [2, 3]
    assert [d["operand"] for d in proof["input_dependencies"]] == [0, 1]
    arrays = [np.array([[0., np.inf], [np.nan, 1.], [-0., 2.]]),
              np.array([0., 3., -np.inf]), np.full((2, 3), np.nan)]
    maps = [m.data for m in generic.get_indexing_maps()]
    affected = np.zeros((2, 3), dtype=bool)
    for index in np.ndindex(affected.shape):
        affected[index] = any(not np.isfinite(arrays[d["operand"]][maps[d["operand"]].eval(index, ())])
                              for d in proof["input_dependencies"])
    assert np.array_equal(affected, ~np.isfinite(arrays[0].T) | ~np.isfinite(arrays[1][None, :]))
    assert not affected[0, 0]  # Unread NaN storage must never taint this finite output.


def test_overlay_refuses_reduction_dependence_without_a_closed_proof():
    from merlin.llvmlower.ieee_pointwise import dependence_proof

    module = parse_mlir_text('''module {
      func.func @forward(%x: tensor<3xf32>,%seed: tensor<f32>) -> tensor<f32> {
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}
          ins(%x : tensor<3xf32>) outs(%seed : tensor<f32>) {
          ^bb0(%a:f32,%b:f32):
            %v=arith.addf %a,%b : f32
            linalg.yield %v : f32
        } -> tensor<f32>
        return %r : tensor<f32>
      }}''')
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    with pytest.raises(ValueError, match="closed parallel output index map"):
        dependence_proof(generic)
