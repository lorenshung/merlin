"""Reachability tests for invariant-weight preprocessing.

Registration alone is not reachability: this lever was accepted by the feature registry while the
whole-model preparation seam never forwarded it to the pass that actually rewrites the IR.
"""

from __future__ import annotations

import inspect
import json
import struct

import numpy as np

from merlin.llvmlower import c_runtime, quant_hoist
from merlin.runtime.backends import zephyr_model


class _Seq:
    def __init__(self, data):
        self.data = data


class _Named:
    def __init__(self, name):
        self.name = name


class _Block:
    def __init__(self, names):
        self.ops = [_Named(name) for name in names]


class _Region:
    def __init__(self, names):
        self.blocks = [_Block(names)]


class _Generic:
    def __init__(self, body):
        self.name = "linalg.generic"
        self.properties = {"iterator_types": _Seq(["parallel"])}
        self.regions = [_Region([*body, "linalg.yield"])]
        self.operands = [object(), object()]


def test_quant_hoist_recognizes_the_current_guarded_scale_and_inverse_multiply():
    """The consumer must track both canonical quant formulas the producer legitimately emits."""
    guarded_scale = _Generic(
        [
            "arith.divf",
            "arith.maximumf",
            "arith.cmpf",
            "arith.select",
        ]
    )
    inverse_multiply = _Generic(
        [
            "arith.mulf",
            "math.roundeven",
            "arith.minimumf",
            "arith.maximumf",
            "arith.fptosi",
        ]
    )

    assert quant_hoist._is_scale(guarded_scale)
    assert quant_hoist._quantize_formula(inverse_multiply) == "multiply_inverse"


def test_prepare_for_lowering_forwards_quant_hoist_and_the_bundle_authority():
    src = inspect.getsource(zephyr_model.prepare_for_lowering)

    assert "hoist_weight_invariant_quantize=" in src
    assert "from ...llvmlower.quant_hoist import FEATURE" in src
    assert "_HOIST_WEIGHT_INVARIANT_QUANTIZE in _closed" in src
    assert "bundle_dir=" in src


def test_c_runtime_appends_the_quant_hoist_plan_to_the_forward_abi(tmp_path):
    """Rewriting @forward without binding its new arguments would run on garbage bytes."""
    model, prepared, out = (tmp_path / "capture", tmp_path / "prepared", tmp_path / "generated")
    model.mkdir()
    (model / "model.mlir").write_text("""builtin.module {
  func.func @forward(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    func.return %arg0 : tensor<4xf32>
  }
}
""")
    header = json.dumps({}).encode()
    (model / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "x"}}))
    (model / "input_order.json").write_text(json.dumps({"x": 0}))
    np.savez(model / "inputs.npz", in0=np.ones(4, np.float32))
    prepared.mkdir()
    lifted = np.array([3.0, 1.0], np.float32)
    quant_hoist.write_plan(prepared, [quant_hoist.HoistedArg("folded", (2,), "f32")])
    quant_hoist.write_values(prepared, {"folded": lifted})

    info = c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)

    assert info["n_quant_hoist"] == 1
    rows = [line for line in (out / "model_gen.h").read_text().splitlines() if line.strip().startswith("{MERLIN_")]
    assert len(rows) == 3 and "MERLIN_WEIGHT" in rows[1]
    offset = int(rows[1].split(",")[1].strip().rstrip("L"))
    got = np.frombuffer((out / "weights.bin").read_bytes(), dtype=np.float32, count=2, offset=offset)
    assert np.array_equal(got, lifted)


def test_stored_integer_weight_transpose_is_exact_and_activation_stays_runtime():
    from merlin.frontends.linalg_mlir import parse_mlir_text

    source = """module {
      func.func @forward(%w: tensor<2x3xi8>, %x: tensor<1xf32>) -> tensor<3x2xi8> {
        %empty = tensor.empty() : tensor<3x2xi8>
        %t = linalg.transpose ins(%w : tensor<2x3xi8>)
            outs(%empty : tensor<3x2xi8>) permutation = [1, 0]
        return %t : tensor<3x2xi8>
      }
    }"""
    weight = np.array([[-128, 0, 127], [9, -2, 5]], dtype=np.int8)
    module = parse_mlir_text(source)
    args, values, count = quant_hoist.apply(module, frozenset({1}), lambda i: weight)
    module.verify()
    assert count == 1
    assert args[0].shape == (3, 2)
    np.testing.assert_array_equal(values[args[0].key], weight.T)
    assert values[args[0].key].flags.c_contiguous
    assert not any(op.name == "linalg.transpose" for op in module.walk())
    for activation_args, loader in [(frozenset({0}), lambda i: weight), (frozenset({1}), lambda i: None)]:
        module = parse_mlir_text(source)
        assert quant_hoist.apply(module, activation_args, loader) == ([], {}, 0)
        assert any(op.name == "linalg.transpose" for op in module.walk())


def _nested_layout_source(extra_return=False):
    output = "tensor<2x3x2xi8>, tensor<3x4xi8>" if extra_return else "tensor<2x3x2xi8>"
    ret = "%final, %first : tensor<2x3x2xi8>, tensor<3x4xi8>" if extra_return else "%final : tensor<2x3x2xi8>"
    return f"""module {{
      func.func @forward(%w: tensor<4x3xi8>, %x: tensor<1xf32>) -> ({output}) {{
        %e0 = tensor.empty() : tensor<3x4xi8>
        %first = linalg.transpose ins(%w : tensor<4x3xi8>)
          outs(%e0 : tensor<3x4xi8>) permutation = [1, 0]
        %flat = tensor.collapse_shape %first [[0, 1]] : tensor<3x4xi8> into tensor<12xi8>
        %view = tensor.expand_shape %flat [[0, 1, 2]] output_shape [3, 2, 2]
          : tensor<12xi8> into tensor<3x2x2xi8>
        %e1 = tensor.empty() : tensor<2x3x2xi8>
        %final = linalg.transpose ins(%view : tensor<3x2x2xi8>)
          outs(%e1 : tensor<2x3x2xi8>) permutation = [1, 0, 2]
        return {ret}
      }}
    }}"""


def test_nested_stored_layout_preserves_order_and_lifts_only_live_frontier():
    from merlin.frontends.linalg_mlir import parse_mlir_text

    weight = np.arange(-6, 6, dtype=np.int8).reshape(4, 3)
    module = parse_mlir_text(_nested_layout_source())
    args, values, count = quant_hoist.apply(module, frozenset({1}), lambda i: weight)
    module.verify()
    assert count == len(args) == 1
    expected = weight.T.reshape(3, 2, 2).transpose(1, 0, 2)
    np.testing.assert_array_equal(values[args[0].key], expected)
    assert not np.array_equal(expected, weight.reshape(3, 2, 2).transpose(1, 0, 2))
    assert not any(op.name in {"linalg.transpose", *quant_hoist.VIEW_OPS} for op in module.walk())


def test_nested_stored_layout_preserves_independent_ancestor_consumer():
    from merlin.frontends.linalg_mlir import parse_mlir_text

    weight = np.arange(-6, 6, dtype=np.int8).reshape(4, 3)
    module = parse_mlir_text(_nested_layout_source(extra_return=True))
    args, values, count = quant_hoist.apply(module, frozenset({1}), lambda i: weight)
    module.verify()
    assert count == len(args) == 2
    np.testing.assert_array_equal(values[args[0].key], weight.T)
    np.testing.assert_array_equal(values[args[1].key], weight.T.reshape(3, 2, 2).transpose(1, 0, 2))


def test_nested_stored_layout_leaves_activation_or_unstored_input_unchanged():
    from merlin.frontends.linalg_mlir import parse_mlir_text

    for indices, loader in [(frozenset({0}), lambda i: np.ones((4, 3), np.int8)), (frozenset({1}), lambda i: None)]:
        module = parse_mlir_text(_nested_layout_source())
        assert quant_hoist.apply(module, indices, loader) == ([], {}, 0)
        assert sum(op.name == "linalg.transpose" for op in module.walk()) == 2


def test_nested_stored_layout_rejects_mismatched_payload_before_mutation():
    import pytest

    from merlin.frontends.linalg_mlir import parse_mlir_text

    for bad in [np.ones((4, 3), np.int32), np.ones((3, 4), np.int8)]:
        module = parse_mlir_text(_nested_layout_source())
        with pytest.raises(quant_hoist.QuantHoistRefused, match="tensor type"):
            quant_hoist.apply(module, frozenset({1}), lambda i: bad)
        assert sum(op.name == "linalg.transpose" for op in module.walk()) == 2
        assert len(quant_hoist._forward_func(module).regions[0].blocks[0].args) == 2
