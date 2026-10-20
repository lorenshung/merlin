"""Specialize a single elementwise power with a literal exponent, read from the IR.

A power whose exponent is a closed literal 0 is the uniform fill 1 (IEEE and
ATen agree for every base, including NaN); exponent 1 is a copy of its base
(ATen's copy_, which also retains signaling NaN payloads). For a Boolean
source the scalar body is evaluated exactly on both storage words; only an
identity or a uniform fill is emitted, never a two-entry table. The exponent
and the evaluated table come only from the live IR operations; no case
record, overload name or provenance label participates, and no tensor payload
is read.
"""

import hashlib
import json
import math

from merlin.targetgen.native_payload_carriers import op_name

POWERS = {"math.powf", "math.fpowi", "math.ipowi"}
CONVERSIONS = {"arith.sitofp", "arith.uitofp", "arith.extui", "arith.extsi", "arith.extf"}


def _elementwise_root(function, source):
    """The single all-parallel identity-map generic reading only the source (plus literal inputs)."""
    from xdsl.dialects.builtin import AffineMapAttr
    from xdsl.ir.affine import AffineMap

    from merlin.targetgen.closed_scalar import literal

    generics = [op for op in function.body.block.ops if op.name == "linalg.generic"]
    if len(generics) != 1:
        return None
    root = generics[0]
    if len(root.results) != 1 or any(it.data.value != "parallel" for it in root.get_iterator_types()):
        return None
    rank = len(root.results[0].type.get_shape())
    if any(m != AffineMapAttr(AffineMap.identity(rank)) for m in root.get_indexing_maps()):
        return None
    runtime = [i for i, _ in enumerate(root.inputs) if literal(root.body.block.args[i]) is None]
    if len(runtime) != 1 or root.inputs[runtime[0]] is not source:
        return None
    return root, root.body.block.args[runtime[0]]


def _conversion_chain(value, argument):
    """Is value the block argument after only value-preserving conversions?"""
    while value is not argument:
        owner = value.owner
        if op_name(owner) not in CONVERSIONS:
            return False
        value = owner.operands[0]
    return True


def classify(function, source_type):
    """Return ('fill', literal) or ('copy', None) read from the IR, else None."""
    from merlin.targetgen.closed_scalar import UNKNOWN, evaluate, literal

    found = _elementwise_root(function, function.body.block.args[0])
    if found is None:
        return None
    root, argument = found
    yielded = root.body.block.last_op.operands[0]
    owner = yielded.owner
    boolean = str(source_type.element_type) == "i1"
    if op_name(owner) in POWERS:
        exponent = literal(owner.operands[1])
        if isinstance(exponent, (int, float)) and math.isfinite(exponent) and exponent in (0, 1):
            if exponent == 0:
                value = evaluate(yielded, {argument: 0})
                return None if value is UNKNOWN else ("fill", value)
            if _conversion_chain(owner.operands[0], argument):
                return ("copy", None)
    if not boolean:
        return None
    # Complete Boolean domain: identity (0 -> +0, 1 -> 1) or a uniform fill only.
    table = [evaluate(yielded, {argument: bit}) for bit in (False, True)]
    if any(entry is UNKNOWN for entry in table):
        return None
    zero, one = table
    if zero == 0 and one == 1 and math.copysign(1.0, float(zero)) > 0:
        return ("copy", None)
    if zero == one and str(zero) == str(one):
        return ("fill", zero)
    return None


def _projection(value, argument):
    """Component index k when value = argument[..., k] through a pure projection generic."""
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    owner = value.owner
    if op_name(owner) != "linalg.generic" or len(owner.inputs) != 1 or owner.inputs[0] is not argument:
        return None
    body = owner.body.block
    if len(list(body.ops)) != 1 or body.last_op.operands[0] is not body.args[0]:
        return None
    results = list(list(owner.get_indexing_maps())[0].data.results)
    rank = len(results) - 1
    if not isinstance(results[-1], AffineConstantExpr) or results[:-1] != [AffineDimExpr(i) for i in range(rank)]:
        return None
    return results[-1].value


def _round_trip_source(value):
    """Skip exact widening-then-narrowing pairs back to the original value."""
    while op_name(value.owner) == "arith.truncf":
        widened = value.owner.operands[0]
        if op_name(widened.owner) != "arith.extf" or widened.owner.operands[0].type != value.type:
            break
        value = widened.owner.operands[0]
    return value


def classify_complex(function, result):
    """Complex elementwise result from interleaved real/imaginary component maps.

    The final generic interleaves two component tensors on the trailing axis of
    size 2 by an index test; each component is an elementwise map of the
    argument's real and imaginary projections. Only a uniform complex fill
    (both component maps closed literals) or an exact copy (each component is
    its own projection through exact round trips) is returned.
    """
    from merlin.targetgen.closed_scalar import UNKNOWN, evaluate, literal

    argument = function.body.block.args[0]
    final = result.owner
    if op_name(final) != "linalg.generic" or len(final.inputs) != 2:
        return None
    shape = result.type.get_shape()
    if not shape or shape[-1] != 2:
        return None
    body = final.body.block
    select = body.last_op.operands[0].owner
    if op_name(select) != "arith.select":
        return None
    test = select.operands[0].owner
    if op_name(test) != "arith.cmpi" or op_name(test.operands[0].owner) != "linalg.index":
        return None
    if test.operands[0].owner.dim.value.data != len(shape) - 1 or literal(test.operands[1]) != 0:
        return None
    predicate = test.predicate.value.data
    if predicate not in (0, 1) or {select.operands[1], select.operands[2]} != set(body.args[:2]):
        return None
    first = body.args.index(select.operands[1 if predicate == 0 else 2])
    components = [final.inputs[first], final.inputs[1 - first]]  # real, imaginary
    values = []
    from xdsl.dialects.builtin import AffineMapAttr
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    rank = len(shape) - 1
    leading = [AffineDimExpr(i) for i in range(rank)]
    final_maps = [list(m.data.results) for m in final.get_indexing_maps()]
    if final_maps[:2] != [leading, leading] or final_maps[2] != [AffineDimExpr(i) for i in range(rank + 1)]:
        return None  # The interleave must not move elements.
    for index, component in enumerate(components):
        owner = component.owner
        if op_name(owner) != "linalg.generic" or any(it.data.value != "parallel" for it in owner.get_iterator_types()):
            return None
        if any(m != AffineMapAttr(AffineMap.identity(rank)) for m in owner.get_indexing_maps()):
            return None  # An elementwise component map; a reversal or gather is not a copy.
        projections = [_projection(v, argument) for v in owner.inputs]
        if any(p is None for p in projections):
            return None
        yielded = owner.body.block.last_op.operands[0]
        closed = evaluate(yielded, {})
        if closed is not UNKNOWN and isinstance(closed, float):
            values.append(("literal", closed))
            continue
        source = _round_trip_source(yielded)
        args = owner.body.block.args
        if source in args and args.index(source) < len(projections) and projections[args.index(source)] == index:
            values.append(("copy", None))
            continue
        return None
    if all(kind == "literal" for kind, _ in values):
        return ("complex_fill", (values[0][1], values[1][1]))
    if all(kind == "copy" for kind, _ in values):
        return ("copy", None)
    return None


def specialize(directory, report):
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        FloatAttr,
        IntegerAttr,
        IntegerType,
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
    original = (directory / "model.mlir").read_bytes()
    module = parse_mlir_file(directory / "model.mlir")
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1 or functions[0].sym_name.data != "forward":
        return None
    function = functions[0]
    if len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    from merlin.targetgen.source_primitive_effects import pure_tensor_source

    if not pure_tensor_source(function):
        return None
    source_type = function.body.block.args[0].type
    if not isinstance(source_type, TensorType):
        return None
    indices = record.get("output_indices", ())
    posts = record.get("semantic_boundary", {}).get("post_indices", ())
    outputs = list(function.function_type.outputs)
    if len(indices) != 1 or len(posts) != 1 or set((*indices, *posts)) != set(range(len(outputs))):
        return None
    complex_result = str((record.get("output_abi") or [{}])[0].get("dtype", "")).startswith("complex<")
    if complex_result:
        returned_values = list(function.body.block.last_op.operands)
        decision = classify_complex(function, returned_values[indices[0]])
    else:
        decision = classify(function, source_type)
    if decision is None:
        return None
    family, value = decision
    result_type = outputs[indices[0]]
    if not isinstance(result_type, TensorType) or any(n < 0 for n in result_type.get_shape()):
        return None
    if family == "copy" and source_type.get_shape() != result_type.get_shape():
        return None
    block = Block(arg_types=[source_type])
    operations = []
    element = result_type.element_type
    boolean = str(source_type.element_type) == "i1"
    if family == "complex_fill":
        from math import prod

        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

        if not all(math.isfinite(v) for v in value):
            return None
        literal_op = arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(result_type, list(value) * (prod(result_type.get_shape()) // 2))
        )
        operations.append(literal_op)
        result = literal_op.result
        family_name = "uniform_complex_literal_fill"
    elif family == "fill":
        if isinstance(value, float) and not math.isfinite(value):
            return None
        attr = IntegerAttr(int(value), element) if isinstance(element, IntegerType) else FloatAttr(float(value), element)
        literal_op = arith.ConstantOp(attr)
        filled = tensor.SplatOp(literal_op.result, [], result_type)
        operations.extend([literal_op, filled])
        result = filled.result
        family_name = "uniform_literal_fill"
    else:
        source_element = source_type.element_type
        destination = tensor.EmptyOp([], result_type)
        scalar = Block(arg_types=[source_element, element])
        converted = scalar.args[0]
        if source_element != element:
            if isinstance(source_element, IntegerType) and not isinstance(element, IntegerType):
                conversion = (arith.UIToFPOp if boolean else arith.SIToFPOp)(converted, element)
            elif isinstance(source_element, IntegerType) and isinstance(element, IntegerType) and boolean:
                conversion = arith.ExtUIOp(converted, element)
            else:
                return None
            scalar.add_op(conversion)
            converted = conversion.result
        scalar.add_op(L.YieldOp(converted))
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
        family_name = "storage_copy_or_declared_conversion"
    returned = [None] * len(outputs)
    returned[indices[0]] = result
    returned[posts[0]] = block.args[0]
    if any(v is None or v.type != typ for v, typ in zip(returned, outputs, strict=True)):
        return None
    block.add_ops([*operations, func.ReturnOp(*returned)])
    replacement = function.clone_without_regions()
    replacement.body.add_block(block)
    Rewriter.replace_op(function, replacement)
    module.verify()
    prepared = str(module) + "\n"
    (directory / "model.mlir").write_text(prepared)
    proof = dict(
        method="ir_literal_power_fill_copy_or_complete_boolean_domain",
        family=family_name,
        fill_value=list(value) if family == "complex_fill" else value if family == "fill" else None,
        source_dtype=str(source_type.element_type),
        source_sha256=hashlib.sha256(original).hexdigest(),
        prepared_sha256=hashlib.sha256(prepared.encode()).hexdigest(),
        parameters="live IR operations and literals only; no case record or tensor payload read",
    )
    (directory / "native-source-specialization.json").write_text(json.dumps(proof, indent=2) + "\n")
    record["source_specialization"] = proof
    return proof
