"""Only uniform or identity behavior over the complete Boolean domain is admitted."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.boolean_domain import fold_boolean_domain


def source(operator, constant, extension="extui"):
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4xi1>) -> tensor<4xi64> {{
      %c=arith.constant {constant} : i64
      %s=tensor.splat %c : tensor<4xi64>
      %e=tensor.empty() : tensor<4xi64>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,
        affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x,%s:tensor<4xi1>,tensor<4xi64>) outs(%e:tensor<4xi64>) {{
        ^bb0(%a:i1,%b:i64,%unused:i64):
          %wide=arith.{extension} %a : i1 to i64
          %v=arith.{operator} %wide,%b : i64
          linalg.yield %v : i64
        }} -> tensor<4xi64>
      func.return %r : tensor<4xi64>
    }} }}""")


@pytest.mark.parametrize(
    "operator,constant,kind,expected",
    [
        ("andi", 3, "identity", None),
        ("andi", 0, "fill", 0),
        ("ori", 3, "fill", 3),
        ("ori", 0, "identity", None),
        ("xori", 0, "identity", None),
        ("andi", -1, "identity", None),
    ],
)
def test_literal_integer_operations_fold_to_identity_widening_or_uniform_fill(operator, constant, kind, expected):
    module = source(operator, constant)
    assert fold_boolean_domain(module) == 1
    module.verify()
    assert not any(op.name == "arith." + operator for op in module.walk())
    if kind == "identity":
        assert any(op.name == "arith.extui" for op in module.walk())
    else:
        returned = next(op for op in module.walk() if op.name == "func.return").operands[0]
        assert returned.owner.name == "tensor.splat"
        assert returned.owner.operands[0].owner.value.value.data == expected
    assert fold_boolean_domain(module) == 0


@pytest.mark.parametrize(
    "operator,constant,extension", [("xori", 1, "extui"), ("andi", 3, "extsi"), ("ori", 2, "extui")]
)
def test_nonuniform_nonidentity_table_is_not_synthesized(operator, constant, extension):
    module = source(operator, constant, extension)
    original = str(module)
    assert fold_boolean_domain(module) == 0
    assert str(module) == original


def test_runtime_self_product_or_self_or_cannot_hide_their_construction_class():
    for operator in ("muli", "ori", "andi", "xori"):
        module = source(operator, 3)
        generic = next(op for op in module.walk() if op.name == "linalg.generic")
        arithmetic = next(op for op in generic.body.block.ops if op.name == "arith." + operator)
        arithmetic.operands = [arithmetic.operands[0], arithmetic.operands[0]]
        original = str(module)
        assert fold_boolean_domain(module) == 0
        assert str(module) == original


def test_unknown_input_carrier_cannot_be_erased_by_a_uniform_boolean_result():
    from xdsl.dialects.builtin import UnregisteredOp

    module = source("ori", 3)
    function = next(op for op in module.walk() if op.name == "func.func")
    original_input = function.body.block.args[0]
    unknown = UnregisteredOp.with_name("opaque.payload").create(
        operands=[original_input], result_types=[original_input.type]
    )
    function.body.block.insert_op_before(unknown, function.body.block.first_op)
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    generic.operands[0] = unknown.results[0]
    original = str(module)
    assert fold_boolean_domain(module) == 0
    assert str(module) == original


@pytest.mark.parametrize("operator,divisor", [("floordivsi", 2), ("divsi", -2)])
def test_complete_boolean_integer_quotient_is_a_shape_only_zero_fill(operator, divisor):
    module = source(operator, divisor)
    assert fold_boolean_domain(module) == 1
    module.verify()


def test_literal_zero_divisor_keeps_its_undefined_operation():
    module = source("floordivsi", 0)
    original = str(module)
    assert fold_boolean_domain(module) == 0
    assert str(module) == original


@pytest.mark.parametrize("condition,folded", [(0, False), (1, True)])
def test_assertions_are_removed_only_when_true_for_the_entire_domain(condition, folded):
    from xdsl.dialects import arith, cf
    from xdsl.dialects.builtin import IntegerAttr, i1

    module = source("floordivsi", 2)
    block = next(op for op in module.walk() if op.name == "linalg.generic").body.block
    constant = arith.ConstantOp(IntegerAttr(condition, i1))
    assertion = cf.AssertOp(constant.result, "division defined")
    block.insert_ops_before([constant, assertion], block.first_op)
    original = str(module)
    assert bool(fold_boolean_domain(module)) == folded
    if not folded:
        assert str(module) == original
