"""Retain deterministic mask endpoints before the frontend RNG expansion.

The source call's literal probability decides a uniform mask for every input.
Private capsule RNG state has no observable dependence at either endpoint.
Unknown effects, external state, source compositions and partial ABIs refuse.
"""

import hashlib
import json


def closed_private_state(module, function):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.source_primitive_effects import pure_tensor_source

    def name(op):
        return getattr(getattr(op, "op_name", None), "data", op.name)

    globals_ = {}
    for op in module.body.block.ops:
        if op is function:
            continue
        if name(op) != "ml_program.global" or getattr(op.properties.get("sym_visibility"), "data", None) != "private":
            return False
        symbol = getattr(op.properties.get("sym_name"), "data", None)
        if not symbol or symbol in globals_ or "value" not in op.properties:
            return False
        globals_[symbol] = op
    clone = function.clone()
    for op in list(clone.walk()):
        if name(op) not in {"ml_program.global_load", "ml_program.global_store"}:
            continue
        ref = op.properties.get("global")
        symbol = getattr(getattr(ref, "root_reference", None), "data", None)
        if symbol not in globals_ or ref.nested_references:
            return False
        declared = globals_[symbol].properties
        if name(op) == "ml_program.global_load":
            if op.operands or len(op.results) != 1 or op.results[0].type != declared["type"]:
                return False
            Rewriter.replace_op(op, arith.ConstantOp(declared["value"]))
        else:
            if op.results or len(op.operands) != 1 or op.operands[0].type != declared["type"]:
                return False
            Rewriter.erase_op(op)
    return pure_tensor_source(clone)


def specialize(directory, report):
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, StringAttr, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.frontends.linalg_mlir import parse_mlir_file

    records = [r for r in report.get("cases", ()) if r.get("status") == "bundled"]
    if len(records) != 1:
        return None
    record = records[0]
    case = record.get("case") or {}
    if case.get("overload") != "aten.native_dropout.default":
        return None
    args = case.get("arguments", {})
    items = args.get("args", {}).get("items", ())
    kwargs = args.get("kwargs", {})
    p = items[1] if len(items) > 1 else kwargs.get("p")
    train = items[2] if len(items) > 2 else kwargs.get("train")
    if train not in (True, False) or (train and p not in (0, 1)):
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
    if origin != {case["overload"]} or not closed_private_state(module, function):
        return None
    source = function.body.block.args[0]
    indices = record.get("output_indices", ())
    posts = record.get("semantic_boundary", {}).get("post_indices", ())
    types = list(function.function_type.outputs)
    if len(indices) != 2 or len(posts) != 1 or set((*indices, *posts)) != set(range(len(types))):
        return None
    result, mask = (types[i] for i in indices)
    if (
        result != source.type
        or not isinstance(result, TensorType)
        or not isinstance(mask, TensorType)
        or str(result.element_type) not in {"f16", "bf16", "f32", "f64"}
        or str(mask.element_type) != "i1"
        or any(n < 0 for n in (*result.get_shape(), *mask.get_shape()))
    ):
        return None
    kept = not train or p == 0
    block = Block(arg_types=[source.type])
    operations = []
    output = block.args[0]
    if train:
        empty = tensor.EmptyOp([], result)
        body = Block(arg_types=[result.element_type] * 2)
        factor = arith.ConstantOp(FloatAttr(float(kept), result.element_type))
        product = arith.MulfOp(body.args[0], factor.result)
        body.add_ops([factor, product, L.YieldOp(product.result)])
        identity = AffineMapAttr(AffineMap.identity(len(result.get_shape())))
        scaled = L.GenericOp(
            inputs=[block.args[0]],
            outputs=[empty.tensor],
            body=Region(body),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(result.get_shape())),
            result_types=[result],
        )
        operations.extend([empty, scaled])
        output = scaled.results[0]
    bit = arith.ConstantOp(IntegerAttr(int(kept), mask.element_type))
    filled = tensor.SplatOp(bit.result, [], mask)
    operations.extend([bit, filled])
    for op in operations:
        op.attributes["prov.aten"] = StringAttr(case["overload"])
    returned = [None] * len(types)
    returned[indices[0]], returned[indices[1]], returned[posts[0]] = output, filled.result, block.args[0]
    if any(v is None or v.type != typ for v, typ in zip(returned, types, strict=True)):
        return None
    block.add_ops([*operations, func.ReturnOp(*returned)])
    replacement = function.clone_without_regions()
    replacement.body.add_block(block)
    Rewriter.replace_op(function, replacement)
    for op in list(module.body.block.ops):
        if op is not replacement:
            Rewriter.erase_op(op)
    module.verify()
    prepared = str(module) + "\n"
    (directory / "model.mlir").write_text(prepared)
    proof = dict(
        method="source_constant_probability_endpoint",
        probability=p,
        training=train,
        mask=int(kept),
        state_scope="capsule_private; no observable output or future endpoint output depends on RNG state",
        retained_arithmetic="source literal scaling with its typed IEEE boundary",
        source_sha256=hashlib.sha256(original).hexdigest(),
        prepared_sha256=hashlib.sha256(prepared.encode()).hexdigest(),
        parameters="source literals and ABI only; evaluation tensors never read",
    )
    record["source_specialization"] = proof
    (directory / "native-source-specialization.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof
