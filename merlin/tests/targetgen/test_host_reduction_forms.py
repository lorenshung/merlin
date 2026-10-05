"""A host region's trailing-window sum is stated in the device's window form, read from the IR."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import group_capsule_entries as G

_MEAN = """"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x8x4x4xf32>) -> tensor<1x8xf32>, sym_name = "forward"}> ({
  ^bb0(%x: tensor<1x8x4x4xf32>):
    %c0 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %e = "tensor.empty"() : () -> tensor<1x8xf32>
    %z = "linalg.fill"(%c0, %e) <{operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb1(%a: f32, %b: f32):
      "linalg.yield"(%a) : (f32) -> ()
    }) : (f32, tensor<1x8xf32>) -> tensor<1x8xf32>
    %s = "linalg.reduce"(%x, %z) <{dimensions = array<i64: 2, 3>}> ({
    ^bb2(%p: f32, %q: f32):
      %r = "arith.addf"(%p, %q) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%r) : (f32) -> ()
    }) : (tensor<1x8x4x4xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
    %n = "arith.constant"() <{value = 1.600000e+01 : f32}> : () -> f32
    %ns = "tensor.splat"(%n) : (f32) -> tensor<1x8xf32>
    %e2 = "tensor.empty"() : () -> tensor<1x8xf32>
    %m = "linalg.generic"(%s, %ns, %e2) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb3(%u: f32, %v: f32, %w: f32):
      %d = "arith.divf"(%u, %v) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%d) : (f32) -> ()
    }) : (tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
    "func.return"(%m) : (tensor<1x8xf32>) -> ()
  }) : () -> ()
}) : () -> ()
"""


@pytest.fixture
def ops(tmp_path):
    path = tmp_path / "mean.mlir"
    path.write_text(_MEAN, encoding="utf-8")
    module = mq.parse(str(path))
    return [op for op in module.walk() if mq.op_name(op) in ("linalg.reduce", "linalg.generic", "tensor.splat")]


def _host(members, index=3):
    return SimpleNamespace(placement="host", members=list(members), index=index, in_dtype="fp32")


def test_a_divided_trailing_sum_is_a_window_mean(ops):
    (form,) = G.host_reduction_forms([_host(ops)])
    assert (form["kind"], form["rows"], form["window"], form["divisor"]) == ("window_mean", 8, 16, 16.0)
    assert (form["input_rank"], form["reduced_dims"]) == (4, 2)
    entry = G.reduction_entry(form, operand_dtype="int8")
    assert (entry["op"], entry["M"], entry["K"], entry["N"]) == ("matmul", 8, 16, 1)
    assert entry["epilogue"] == ["acc_scale"] and entry["acc_scale"] == pytest.approx(1 / 16)


def test_an_undivided_sum_is_a_row_sum_and_device_groups_are_ignored(ops):
    (form,) = G.host_reduction_forms([_host(ops[:1])])
    assert form["kind"] == "row_sum" and form["divisor"] is None
    assert G.reduction_entry(form, operand_dtype="int8")["epilogue"] == []
    device = SimpleNamespace(placement="systolic", members=list(ops), index=4, in_dtype="int8")
    assert G.host_reduction_forms([device]) == []
