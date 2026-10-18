# ruff: noqa: E501 -- the captured module is printed one op per line
"""A capture in the integer_readout numerics is admitted with its readout ON THE DEVICE.

WHY THIS EXISTS. The core-aten int8 pipeline read every contraction's int32 accumulator back to the
host, cast it to f32 and multiplied twice. That host cast is exact only for |acc| <= 2^24, which bounded K
to 1024 at int8 (TinyLlama's K = 2048/5632 and addmm's K = 3072 were refused), and the f32 bias stayed a
host add. The frontend's integer_readout numerics instead seed the accumulator with an int32 bias and
read it out once, fused, as int8 (``prov.role = "readout"`` with ``prov.quant.*``). This plan admits that
form: an accumulator-seed load, the contraction, the fused acc_scale/ReLU stage and an int8 readout
crossing -- and no cast bound.
"""

from __future__ import annotations

import copy

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.kernels.shapes import accumulator_init, observe_contractions, zero_initialised
from merlin.targetgen.core_aten_spec_int8 import admission_plan, device_readout
from merlin.targetgen.host_capabilities import validate_host_capabilities

pytestmark = pytest.mark.target("gemmini")
_BASE = repo_root() / "examples/gemmini/target"

SRC = """
module {
  func.func @forward(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>, %bias: tensor<16xi32>) -> tensor<4x16xi8> {
    %e = tensor.empty() : tensor<4x16xi32>
    %seed = "linalg.broadcast"(%bias, %e) <{dimensions = array<i64: 0>}> ({
    ^bb0(%i: i32, %j: i32):
      linalg.yield %i : i32
    }) {prov.role = "acc_seed"} : (tensor<16xi32>, tensor<4x16xi32>) -> tensor<4x16xi32>
    %acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<4x8xi8>, tensor<8x16xi8>) outs(%seed : tensor<4x16xi32>) attrs = {prov.accumulator_init = "int32_seed", prov.aten = "aten.addmm.default"} {
    ^bb0(%x: i8, %y: i8, %z: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %z, %m : i32
      linalg.yield %s : i32
    } -> tensor<4x16xi32>
    %o = tensor.empty() : tensor<4x16xi8>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%acc : tensor<4x16xi32>) outs(%o : tensor<4x16xi8>) attrs = {prov.role = "readout", prov.quant.scale = 0.0125 : f32, prov.quant.zero_point = 0 : i64, prov.quant.quant_min = -128 : i64, prov.quant.quant_max = 127 : i64, prov.quant.relu = "true"} {
    ^bb0(%v: i32, %w: i8):
      %f = arith.sitofp %v : i32 to f32
      %c = arith.constant 0.0125 : f32
      %p = arith.mulf %f, %c : f32
      %t = arith.fptosi %p : f32 to i8
      linalg.yield %t : i8
    } -> tensor<4x16xi8>
    return %r : tensor<4x16xi8>
  }
}
"""


@pytest.fixture(scope="module")
def selection():
    return {
        "spec": fixture_resolved(_BASE, "gemmini"),
        "host": validate_host_capabilities(load_yaml(_BASE / "host-capabilities.yaml")),
    }


def _plan(selection, source=SRC):
    with IR_LOCK:
        module = mq.parse(source)
        ((op, _shape),) = observe_contractions(module)
        return op, admission_plan(module, selection, captured=True)


def test_the_seed_and_the_fused_readout_are_read_structurally():
    with IR_LOCK:
        ((op, _shape),) = observe_contractions(mq.parse(SRC))
        assert accumulator_init(op) == "int32_seed" and not zero_initialised(op)
        readout = device_readout(op)
    assert readout["relu"] and readout["quant_min"] == -128 and readout["granularity"] == "tensor"
    assert readout["scales"] == [pytest.approx(0.0125)]


def test_the_device_readout_plan_replaces_the_host_cast_and_its_k_bound(selection):
    _op, plan = _plan(selection)
    (site,) = plan["sites"]
    assert site["readout"] == "device_int8" and site["accumulator_init"] == "int32_seed"
    steps = {(row["step"], row.get("declaration") or row.get("op")) for row in site["steps"]}
    assert (2, "transfer_contracts.accumulator_seed_load") in steps
    assert (4, "transfer_contracts.int8_readout") in steps and (4, "elementwise_map") in steps
    assert not any(step for step, _ in steps if step in (5, 6)) and site["cast_exact_range_obligation"] is None
    fused = next(row for row in site["steps"] if row.get("op") == "elementwise_map")
    assert fused["software"]["status"] == "admitted"
    assert fused["device_readout"]["relu"] is True


def test_the_unreviewed_crossings_are_the_only_refusals_and_reviewing_them_admits(selection):
    _op, plan = _plan(selection)
    reasons = plan["sites"][0]["reasons"]
    rows = {row["id"]: row for row in selection["spec"]["transfer_contracts"]["declarations"]}
    unreviewed = [
        key for key in ("accumulator_seed_load", "int8_readout") if rows[key]["evidence"].get("status") != "reviewed"
    ]
    if not unreviewed:
        assert plan["sites"][0]["status"] == "admitted", reasons
        return
    assert all("accumulator_seed_load" in r or "int8_readout" in r for r in reasons), reasons
    reviewed = copy.deepcopy(selection)
    for row in reviewed["spec"]["transfer_contracts"]["declarations"]:
        if row["id"] in unreviewed:
            row["evidence"]["status"] = "reviewed"
    _op, plan = _plan(reviewed)
    assert plan["sites"][0]["status"] == "admitted", plan["sites"][0]["reasons"]


def test_a_readout_relu_with_a_nonpositive_scale_is_refused(selection):
    negative = SRC.replace("prov.quant.scale = 0.0125 : f32", "prov.quant.scale = -0.0125 : f32")
    _op, plan = _plan(selection, negative)
    assert any("ReLU" in reason for reason in plan["sites"][0]["reasons"])


def test_unknown_readout_numerics_are_refused_before_any_frontend_import():
    from merlin.capture.integerization import integerize_static_pt2e

    with pytest.raises(ValueError, match="numerics"):
        integerize_static_pt2e(None, (), numerics="fixed_point")
