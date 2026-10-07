"""Prepared source lifetimes require actual effects, not purity labels."""

import pytest
from xdsl.dialects import builtin, func, memref
from xdsl.ir import Block, Region

from merlin.llvmlower.prepared_operand_effects import validate_prepared_source_lifetime, validate_pure_tensor_function
from merlin.llvmlower.tensor_preparation_identity import TensorPreparationRequest, find_tensor_preparation_opportunities


def fixture():
    typ = builtin.TensorType(builtin.f32, [3, 17])
    body = Block(arg_types=[typ])
    body.add_op(func.ReturnOp(body.args[0]))
    reader = func.FuncOp("read", ([typ], [typ]), Region(body), visibility="private")
    block = Block(arg_types=[typ])
    calls = [func.CallOp("read", [block.args[0]], [typ]) for _ in range(4)]
    block.add_ops([*calls, func.ReturnOp(calls[-1].results[0])])
    owner = func.FuncOp("owner", ([typ], [typ]), Region(block))
    module = builtin.ModuleOp([reader, owner])
    (opportunity,) = find_tensor_preparation_opportunities(
        [TensorPreparationRequest(block.args[0], call, "1" * 64) for call in calls]
    )
    module.verify()
    return module, reader, owner, calls, opportunity


def test_complete_source_bodies_are_checked():
    module, _, _, _, opportunity = fixture()
    validate_prepared_source_lifetime(opportunity, module)


@pytest.mark.parametrize("mode", ["external", "recursive", "allocation"])
def test_opaque_recursive_or_buffer_body_refuses(mode):
    module, reader, _, _, opportunity = fixture()
    body = reader.body.block
    terminator = body.last_op
    if mode == "external":
        body.insert_op_before(func.CallOp("opaque", [body.args[0]], []), terminator)
    elif mode == "recursive":
        body.insert_op_before(func.CallOp("read", [body.args[0]], [body.args[0].type]), terminator)
    else:
        allocation = memref.AllocOp.get(builtin.f32, shape=[17])
        body.insert_op_before(allocation, terminator)
    # Re-discover after mutation: refusal is from live effect inspection,
    # independent of the source snapshot mismatch check.
    with pytest.raises(ValueError):
        validate_pure_tensor_function(reader, {"read": reader})


def test_unknown_effect_between_borrows_refuses():
    module, _, owner, calls, opportunity = fixture()
    owner.body.block.insert_op_before(func.CallOp("unknown_writer", [], []), calls[2])
    with pytest.raises(ValueError, match="unknown external effects"):
        validate_prepared_source_lifetime(opportunity, module)


def test_complete_source_declaration_is_not_a_purity_certificate():
    declaration = func.FuncOp("read", ([], []), Region(), visibility="private")
    with pytest.raises(ValueError, match="complete"):
        validate_pure_tensor_function(declaration, {"read": declaration})
