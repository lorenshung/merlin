"""Retain source fill/copy semantics before a scalar power decomposition.

ATen's literal-zero and literal-one powers dispatch to fill_ and copy_. Keeping
that source distinction preserves signaling NaNs and complex zero components.
Only static scalar arguments and ABI types participate in this proof.
"""

import hashlib
import json
import math


def specialize(directory, report):
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        FloatAttr,
        IntegerAttr,
        IntegerType,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.frontends.linalg_mlir import parse_mlir_file

    records = [r for r in report.get("cases", ()) if r.get("status") == "bundled"]
    if len(records) != 1:
        return None
    record = records[0]
    case = record["case"]
    if case.get("overload") != "aten.pow.Tensor_Scalar":
        return None
    args = case.get("arguments", {})
    items = args.get("args", {}).get("items", [])
    exponent = items[1] if len(items) > 1 else args.get("kwargs", {}).get("exponent")
    if isinstance(exponent, dict) and exponent.get("kind") == "float":
        exponent = float(exponent["value"])
    if not isinstance(exponent, (int, float)) or not math.isfinite(exponent):
        return None
    original = (directory / "model.mlir").read_bytes()
    module = parse_mlir_file(directory / "model.mlir")
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1 or functions[0].sym_name.data != "forward":
        return None
    function = functions[0]
    if len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    origin = {getattr(op.attributes.get("prov.aten"), "data", None) for op in function.walk()} - {None}
    if origin != {case["overload"]}:
        return None
    for op in function.body.block.walk():
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if not (
            name.startswith(("arith.", "math.", "tensor.", "linalg."))
            or name in {"scf.for", "scf.if", "scf.yield", "func.return"}
        ):
            return None
    source_type = function.body.block.args[0].type
    if not isinstance(source_type, TensorType):
        return None
    boolean = str(source_type.element_type) == "i1"
    if exponent not in (0, 1) and not (boolean and exponent > 0):
        return None
    indices = record.get("output_indices", ())
    posts = record.get("semantic_boundary", {}).get("post_indices", ())
    outputs = list(function.function_type.outputs)
    if len(indices) != 1 or len(posts) != 1 or set((*indices, *posts)) != set(range(len(outputs))):
        return None
    result_type = outputs[indices[0]]
    if not isinstance(result_type, TensorType) or any(n < 0 for n in result_type.get_shape()):
        return None
    if exponent != 0 and source_type.get_shape() != result_type.get_shape():
        return None
    block = Block(arg_types=[source_type])
    operations = []
    result = None
    if exponent == 0:
        element = result_type.element_type
        complex_result = record["output_abi"][0]["dtype"].startswith("complex<")
        if complex_result:
            from math import prod

            if result_type.get_shape()[-1] != 2:
                return None
            values = [1.0, 0.0] * (prod(result_type.get_shape()) // 2)
            literal = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(result_type, values))
            operations.append(literal)
            result = literal.result
        else:
            attr = IntegerAttr(1, element) if isinstance(element, IntegerType) else FloatAttr(1.0, element)
            literal = arith.ConstantOp(attr)
            filled = tensor.SplatOp(literal.result, [], result_type)
            operations.extend([literal, filled])
            result = filled.result
        family = "uniform_literal_fill"
    else:
        source_element, element = source_type.element_type, result_type.element_type
        destination = tensor.EmptyOp([], result_type)
        scalar = Block(arg_types=[source_element, element])
        value = scalar.args[0]
        if source_element != element:
            if isinstance(source_element, IntegerType) and not isinstance(element, IntegerType):
                conversion = (arith.UIToFPOp if boolean else arith.SIToFPOp)(value, element)
            elif isinstance(source_element, IntegerType) and isinstance(element, IntegerType) and boolean:
                conversion = arith.ExtUIOp(value, element)
            else:
                return None
            scalar.add_op(conversion)
            value = conversion.result
        scalar.add_op(L.YieldOp(value))
        identity = AffineMapAttr(AffineMap.identity(len(result_type.get_shape())))
        copied = L.GenericOp(
            inputs=[block.args[0]],
            outputs=[destination.tensor],
            body=Region(scalar),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(result_type.get_shape())),
            result_types=[result_type],
        )
        operations.extend([destination, copied])
        result = copied.results[0]
        family = "storage_copy_or_declared_conversion"
    for op in operations:
        op.attributes["prov.aten"] = StringAttr(case["overload"])
    returned = [None] * len(outputs)
    returned[indices[0]] = result
    returned[posts[0]] = block.args[0]
    if any(value is None or value.type != typ for value, typ in zip(returned, outputs, strict=True)):
        return None
    block.add_ops([*operations, func.ReturnOp(*returned)])
    replacement = function.clone_without_regions()
    replacement.body.add_block(block)
    Rewriter.replace_op(function, replacement)
    module.verify()
    prepared = str(module) + "\n"
    (directory / "model.mlir").write_text(prepared)
    proof = dict(
        method="source_literal_power_fill_copy_dispatch_or_complete_boolean_domain",
        family=family,
        exponent=exponent,
        source_dtype=str(source_type.element_type),
        source_sha256=hashlib.sha256(original).hexdigest(),
        prepared_sha256=hashlib.sha256(prepared.encode()).hexdigest(),
        parameters="schema scalar and declared ABI only; no tensor payload read",
    )
    (directory / "native-source-specialization.json").write_text(json.dumps(proof, indent=2) + "\n")
    record["source_specialization"] = proof
    return proof
