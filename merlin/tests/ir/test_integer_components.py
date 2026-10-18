"""Execute generated host limb arithmetic at native wraparound boundaries."""

import json

import numpy as np
import pytest
from xdsl.dialects.builtin import IntegerType

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.boolean_components import rewrite_boolean_components
from merlin.llvmlower.integer_components import rewrite_integer_components


def module(dtype, expression):
    return parse_mlir_text(
        """builtin.module {
    func.func @forward(%x: tensor<4xTYPE>, %y: tensor<4xTYPE>) -> tensor<4xTYPE> {
      %empty = tensor.empty() : tensor<4xTYPE>
      %result = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]
      } ins(%x, %y : tensor<4xTYPE>, tensor<4xTYPE>) outs(%empty : tensor<4xTYPE>) {
      ^bb0(%a: TYPE, %b: TYPE, %c: TYPE):
        EXPR
        linalg.yield %v : TYPE
      } -> tensor<4xTYPE>
      func.return %result : tensor<4xTYPE>
    }}""".replace("TYPE", dtype).replace("EXPR", expression)
    )


def evaluate(module, arrays):
    """Small integer SSA interpreter; all generated tensors have identity maps."""
    function = next(op for op in module.walk() if op.name == "func.func")
    values = dict(zip(function.body.block.args, [np.array(a, dtype=object) for a in arrays]))

    def normalize(value, typ):
        if isinstance(typ, IntegerType):
            bits = typ.width.data
            value = int(value) % (1 << bits)
            return value - (1 << bits) if value >= (1 << (bits - 1)) else value
        return value

    def scalar(op, env):
        a = [env[v] for v in op.operands]
        name = op.name
        if name == "arith.constant":
            result = op.value.value.data
        elif name in {"arith.extsi", "arith.trunci"}:
            result = a[0]
        elif name == "arith.extui":
            result = a[0] % (1 << op.operands[0].type.width.data)
        elif name == "arith.addi":
            result = a[0] + a[1]
        elif name == "arith.subi":
            result = a[0] - a[1]
        elif name == "arith.muli":
            result = a[0] * a[1]
        elif name == "arith.shrsi":
            result = a[0] >> a[1]
        elif name == "arith.shli":
            assert 0 <= a[1] < op.results[0].type.width.data
            result = a[0] << a[1]
        elif name == "arith.cmpi":
            assert op.predicate.value.data == 1
            result = int(a[0] != a[1])
        else:
            raise AssertionError(name)
        env[op.results[0]] = normalize(result, op.results[0].type)

    for op in function.body.block.ops:
        if op.name == "func.return":
            return values[op.operands[0]].tolist()
        if op.name == "arith.constant":
            scalar(op, values)
        elif op.name == "tensor.empty":
            values[op.results[0]] = np.zeros(op.results[0].type.get_shape(), object)
        elif op.name == "tensor.splat":
            values[op.results[0]] = np.full(op.results[0].type.get_shape(), values[op.operands[0]], object)
        elif op.name == "linalg.generic":
            output = np.zeros(op.results[0].type.get_shape(), object)
            for index in np.ndindex(output.shape):
                env = {arg: values[source][index] for arg, source in zip(op.body.block.args, op.operands)}
                for child in op.body.block.ops:
                    if child.name == "linalg.yield":
                        output[index] = env[child.operands[0]]
                    else:
                        scalar(child, env)
            values[op.results[0]] = output
        else:
            raise AssertionError(op.name)
    raise AssertionError("no return")


@pytest.mark.parametrize("bits", [8, 32, 64])
@pytest.mark.parametrize("coefficient", [-3, 1, 7])
def test_linear_components_recombine_with_native_wraparound(bits, coefficient):
    source = module(
        f"i{bits}",
        f"%s = arith.constant {coefficient} : i{bits}\n"
        f"%p = arith.muli %a, %s : i{bits}\n%v = arith.addi %p, %b : i{bits}",
    )
    limit = 1 << (bits - 1)
    x, y = [-limit, limit - 1, -129 % limit, 128 % limit], [1, -1, 127, -128]
    plans = rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[limit, 128])
    source.verify()
    assert plans and all(p["radix"] == 256 for p in plans)
    assert all(
        json.loads(op.attributes["test.limb"].data)["family"] == "F2"
        for op in source.walk()
        if "test.limb" in op.attributes
    )
    expected = [((coefficient * a + b + limit) % (2 * limit)) - limit for a, b in zip(x, y)]
    assert evaluate(source, [x, y]) == expected


def test_boolean_or_uses_unsigned_truth_codes_and_exact_host_predicate():
    source = module("i1", "%v = arith.ori %a, %b : i1")
    bounds, count = rewrite_boolean_components(source)
    assert count == 1
    assert rewrite_integer_components(source, limb_attribute="test.limb", input_bounds=[1, 1], known_bounds=bounds)
    assert evaluate(source, [[0, 0, 1, 1], [0, 1, 0, 1]]) == [0, -1, -1, -1]
