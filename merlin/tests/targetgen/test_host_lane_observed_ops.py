"""Host probes are built from the host operations real captures contain, never from stock ops."""

from __future__ import annotations

from merlin.targetgen import corpus_synth as CS
from merlin.targetgen import host_lane_ops
from merlin.targetgen import micro_model as MM

_CAPTURE = """"builtin.module"() ({
  "func.func"() <{function_type = (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>, sym_name = "forward"}> ({
  ^bb0(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>):
    %e = "tensor.empty"() : () -> tensor<4x8xf32>
    %m = "linalg.generic"(%a, %b, %e) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%x: f32, %y: f32, %z: f32):
      %p = "arith.mulf"(%x, %y) : (f32, f32) -> f32
      "linalg.yield"(%p) : (f32) -> ()
    }) {prov.op = "mul", prov.aten = "aten.mul.Tensor", prov.family = "elementwise"} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    "func.return"(%m) : (tensor<4x8xf32>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def test_observed_host_ops_name_the_captured_operation(tmp_path):
    path = tmp_path / "model.mlir"
    path.write_text(_CAPTURE)
    observed = host_lane_ops.observed_host_ops({"app": path}, [("elementwise_map", "f32"), ("reduction", "f32")])
    assert observed["elementwise_map/f32"] == [{"op": "mul", "frontend_op": "aten.mul.Tensor", "n_regions": 1}]
    # A pair with no captured region is empty, never filled with a stock op.
    assert observed["reduction/f32"] == []


def test_a_probe_is_an_observed_operation_or_nothing():
    pool = CS.available_ops()
    rows = [
        {"op": "add", "frontend_op": "aten.add.Tensor", "n_regions": 9},  # has an accelerator builder
        {"op": "mul", "frontend_op": "aten.mul.Tensor", "n_regions": 3},
    ]
    op, row = CS._host_probe_op({"observed_ops": rows}, "elementwise_map", "f32", pool)
    assert (op, row["frontend_op"]) == ("mul", "aten.mul.Tensor")
    for family, rows, expected in (
        (
            "normalization",
            [{"op": "layer_norm", "frontend_op": "aten.native_layer_norm.default", "n_regions": 1}],
            "layer_norm",
        ),
        ("reduction", [{"op": "reduce_mean", "frontend_op": "aten.mean.dim", "n_regions": 1}], "reduce_mean"),
        ("contraction", [{"op": "batch_matmul", "frontend_op": "aten.bmm.default", "n_regions": 1}], "batch_matmul"),
        ("movement", [{"op": "permute", "frontend_op": "aten.permute.default", "n_regions": 1}], "permute"),
    ):
        assert CS._host_probe_op({"observed_ops": rows}, family, "f32", pool)[0] == expected
    # Observed work no writer can express is reported, not replaced by a stock op of the family.
    unwritable = [{"op": "minmax", "frontend_op": "aten.relu.default", "n_regions": 4}]
    assert CS._host_probe_op({"observed_ops": unwritable}, "elementwise_map", "f32", pool) == (None, None)
    assert CS._host_probe_op({"observed_ops": []}, "elementwise_map", "f32", pool) == (None, None)
    # A historical requirement without the key keeps its family-level choice.
    assert CS._host_probe_op({}, "elementwise_map", "f32", pool)[1] is None


def test_micro_model_host_layers_use_observed_operations():
    rows = [{"op": "reduce_mean", "frontend_op": "aten.mean.dim", "n_regions": 3}]
    op, (_init, forward) = MM.statement_for("reduction", rows)
    assert op == "reduce_mean" and "mean(" in forward
    assert (
        MM.statement_for("elementwise_map", [{"op": "mul", "frontend_op": "aten.mul.Tensor", "n_regions": 1}])[0]
        == "mul"
    )
    try:
        MM.statement_for("elementwise_map", [{"op": "minmax", "frontend_op": "aten.relu.default", "n_regions": 1}])
    except MM.UnwritableLayer:
        pass
    else:
        raise AssertionError("an unexpressible observed host operation must not fall back to a stock op")


def test_observed_operation_writers_render_runnable_loaders():
    from merlin.targetgen.capsule_source import build_loader_src

    for op in ("layer_norm", "reduce_mean", "batch_matmul", "permute"):
        source = build_loader_src({"op": op, "M": 4, "K": 8, "N": 4, "dtype": "f32"})
        compile(source, f"<{op}>", "exec")
