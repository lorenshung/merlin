"""Adjacent provenance is diagnostic, never authoritative frontend semantics."""

import pytest
from merlin.common import mlir_query as mq
from merlin.targetgen.application_inventory import _reduction_provenance_hint


_SHARED_INITIALIZER = '''"builtin.module"() ({
"func.func"() <{function_type = (tensor<4x4xf32>) -> (tensor<4xf32>, tensor<4xf32>), sym_name = "forward"}> ({
^bb0(%x: tensor<4x4xf32>):
%c0 = "arith.constant"() <{value = 0.0 : f32}> : () -> f32
%e = "tensor.empty"() : () -> tensor<4xf32>
%z = "linalg.fill"(%c0,%e) <{operandSegmentSizes = array<i32: 1, 1>}> ({
  ^bb1(%a:f32,%b:f32): "linalg.yield"(%a) : (f32)->()
}) {prov.region_id = "R", prov.aten = "aten.sum"} : (f32,tensor<4xf32>)->tensor<4xf32>
%a = "linalg.reduce"(%x,%z) <{dimensions = array<i64: 1>}> ({
  ^bb2(%p:f32,%q:f32): %r = "arith.addf"(%p,%q) : (f32,f32)->f32 "linalg.yield"(%r) : (f32)->()
}) : (tensor<4x4xf32>,tensor<4xf32>)->tensor<4xf32>
%b = "linalg.reduce"(%x,%z) <{dimensions = array<i64: 1>}> ({
  ^bb3(%p:f32,%q:f32): %r = "arith.addf"(%p,%q) : (f32,f32)->f32 "linalg.yield"(%r) : (f32)->()
}) : (tensor<4x4xf32>,tensor<4xf32>)->tensor<4xf32>
%u = "tensor.cast"(%a) {prov.region_id = "R", prov.aten = "aten.sum"} : (tensor<4xf32>)->tensor<4xf32>
%v = "tensor.cast"(%b) {prov.region_id = "R", prov.aten = "aten.sum"} : (tensor<4xf32>)->tensor<4xf32>
"func.return"(%u,%v) : (tensor<4xf32>,tensor<4xf32>)->()
}) : ()->()
}) : ()->()'''


def test_shared_tagged_initializer_cannot_label_two_untagged_reductions():
    module = mq.parse(_SHARED_INITIALIZER)
    reductions = [op for op in mq.walk(module) if mq.op_name(op) == "linalg.reduce"]
    assert len(reductions) == 2
    assert len(tuple(reductions[0].operands[-1].uses)) == 2
    for reduction in reductions:
        provenance, inference = _reduction_provenance_hint(reduction, mq)
        assert not provenance.get("prov.aten")
        assert inference is None


@pytest.mark.parametrize("body", ["arith.addf", "arith.maximumf"])
def test_exclusive_initializer_with_agreeing_consumers_only_suggests_a_tag(body):
    begin = _SHARED_INITIALIZER.index('%b = "linalg.reduce"')
    end = _SHARED_INITIALIZER.index('%u = "tensor.cast"', begin)
    single = (_SHARED_INITIALIZER[:begin]
              + '%b = "tensor.cast"(%a) {prov.region_id = "R", prov.aten = "aten.sum"} '
                ': (tensor<4xf32>)->tensor<4xf32>\n'
              + _SHARED_INITIALIZER[end:])
    module = mq.parse(single.replace("arith.addf", body))
    reduction = next(op for op in mq.walk(module) if mq.op_name(op) == "linalg.reduce")
    assert len(tuple(reduction.operands[-1].uses)) == 1
    provenance, inference = _reduction_provenance_hint(reduction, mq)
    assert not provenance.get("prov.aten")
    assert inference["status"] == "unverified"
    assert inference["suggested_frontend_op"] == "aten.sum"
    assert inference["method"] == "reduction_initializer_and_consumers_agree_v1"
