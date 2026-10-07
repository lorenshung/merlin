"""Conservative typed source immutability checks for prepared operand lifetimes.

This proves properties of the original tensor program. An external physical
producer/consumer still needs its independently validated borrowed-buffer proof;
function names or proof hashes alone do not supply that permission.
"""

from __future__ import annotations

from xdsl.dialects import builtin, func
from xdsl.dialects.linalg.ops import GenericOp, ReduceOp, YieldOp
from xdsl.traits import get_effects

from .tensor_preparation_identity import validate_tensor_preparation_opportunity


def validate_pure_tensor_function(function, functions, *, active=()):
    """Require a complete nonrecursive tensor/scalar body with no opaque calls.

    Tensor linalg operations have value semantics when every shaped operand is a
    tensor and their scalar regions are pure. Buffer linalg, unknown effects,
    address conversion, external functions, recursion and captured buffers refuse.
    """
    if not isinstance(function, func.FuncOp) or len(function.body.blocks) != 1:
        raise ValueError("prepared source requires a complete single-block tensor function")
    if function in active:
        raise ValueError("recursive prepared source effects are unknown")
    for value in (*function.body.block.args, *function.function_type.outputs):
        _value_type(value.type if hasattr(value, "type") else value)
    for operation in function.body.block.ops:
        _operation(operation, functions, (*active, function))


def _value_type(typ):
    if isinstance(typ, builtin.TensorType):
        if not isinstance(typ.encoding, builtin.NoneAttr):
            raise ValueError("prepared source tensor encoding has unknown effects")
        _value_type(typ.element_type)
    elif not isinstance(typ, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)):
        raise ValueError("prepared source may not expose buffer or address values")


def _operation(operation, functions, active):
    for value in (*operation.operands, *operation.results):
        _value_type(value.type)
    if isinstance(operation, func.CallOp):
        callee = functions.get(operation.callee.root_reference.data)
        if callee is None:
            raise ValueError("prepared source call has unknown external effects")
        validate_pure_tensor_function(callee, functions, active=active)
        return
    if isinstance(operation, (func.ReturnOp, YieldOp)):
        return
    if isinstance(operation, (GenericOp, ReduceOp)):
        # The only admitted effect-free structured special case: immutable
        # tensor values with pure scalar regions. No buffer outputs allowed.
        if not operation.results or any(not isinstance(v.type, builtin.TensorType) for v in operation.results):
            raise ValueError("prepared source structured operation requires tensor results")
    elif get_effects(operation) != set():
        raise ValueError("prepared source operation has unknown or nonempty effects")
    for region in operation.regions:
        for block in region.blocks:
            for argument in block.args:
                _value_type(argument.type)
            for nested in block.ops:
                _operation(nested, functions, active)


def validate_prepared_source_lifetime(opportunity, module):
    """Check every operation from first borrow through the last source consumer.

    This is deliberately conservative: even an unrelated unknown memory effect
    refuses. The result grants no physical cached-storage or external ABI proof.
    """
    validate_tensor_preparation_opportunity(opportunity)
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    operations = tuple(opportunity.owner_block.ops)
    positions = [operations.index(call) for call in opportunity.consumers]
    if positions != sorted(set(positions)):
        raise ValueError("prepared source consumers are not in unique source order")
    for operation in operations[positions[0] : positions[-1] + 1]:
        _operation(operation, functions, ())
