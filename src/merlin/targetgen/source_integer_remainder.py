"""Prove a uniform rational zero from a finite integral floating image."""

import hashlib
import json
import math
from fractions import Fraction


def specialize(directory, report, software_spec):
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import FloatAttr, IntegerType, StringAttr, TensorType
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter

    from merlin.common.yaml import load_yaml
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.quantized_native import parameters
    from merlin.targetgen.exact_arithmetic import round_binary
    from merlin.targetgen.source_primitive_effects import pure_tensor_source

    if software_spec is None:
        return None
    records = [r for r in report.get("cases", ()) if r.get("status") == "bundled"]
    if len(records) != 1:
        return None
    record = records[0]
    case = record.get("case") or {}
    overload = case.get("overload")
    if overload not in {"aten.fmod.Scalar", "aten.remainder.Scalar"}:
        return None
    arguments = case.get("arguments", {})
    items = arguments.get("args", {}).get("items", [])
    divisor = items[1] if len(items) > 1 else arguments.get("kwargs", {}).get("other")
    if not isinstance(divisor, (int, float)) or not divisor or not math.isfinite(divisor):
        return None
    original = (directory / "model.mlir").read_bytes()
    module = parse_mlir_file(directory / "model.mlir")
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1:
        return None
    function = functions[0]
    if function.sym_name.data != "forward" or len(function.body.blocks) != 1 or len(function.body.block.args) != 1:
        return None
    if {getattr(op.attributes.get("prov.aten"), "data", None) for op in function.walk()} - {None} != {overload}:
        return None
    if not pure_tensor_source(function):
        return None
    source = function.body.block.args[0].type
    if not isinstance(source, TensorType) or not isinstance(source.element_type, IntegerType):
        return None
    bits = source.element_type.width.data
    if bits not in {1, 8, 16, 32, 64}:
        return None
    indices = record.get("output_indices", ())
    posts = record.get("semantic_boundary", {}).get("post_indices", ())
    types = list(function.function_type.outputs)
    if len(indices) != 1 or len(posts) != 1 or set((*indices, *posts)) != set(range(len(types))):
        return None
    result = types[indices[0]]
    if not isinstance(result, TensorType) or str(result.element_type) not in {"f32", "f64"}:
        return None
    # Both conversion formats contain every rounded integer image as a finite
    # integral value. Scalar promotion rounds the divisor into that opmath.
    try:
        divisor = round_binary(Fraction(divisor), 32 if str(result.element_type) == "f32" else 64)
    except (ValueError, OverflowError):
        return None
    if not divisor or not math.isfinite(divisor) or (1 / abs(Fraction(divisor))).denominator != 1:
        return None
    if result.get_shape() != source.get_shape() or types[posts[0]] != source:
        return None
    contract = parameters(load_yaml(software_spec))
    block = Block(arg_types=[source])
    zero = arith.ConstantOp(FloatAttr(0.0, result.element_type))
    filled = tensor.SplatOp(zero.result, [], result)
    filled.attributes["prov.aten"] = StringAttr(overload)
    values = [None] * len(types)
    values[indices[0]] = filled.result
    values[posts[0]] = block.args[0]
    block.add_ops([zero, filled, func.ReturnOp(*values)])
    replacement = function.clone_without_regions()
    replacement.body.add_block(block)
    Rewriter.replace_op(function, replacement)
    module.verify()
    prepared = str(module) + "\n"
    (directory / "model.mlir").write_text(prepared)
    proof = dict(
        method="finite_integral_float_image_mod_exact_reciprocal",
        input_signed_storage_bits=bits,
        opmath=str(result.element_type),
        divisor=divisor,
        zero_sign="canonical positive at quantized output boundary",
        parameters="schema scalar and ABI only; no tensor payload read",
        source_sha256=hashlib.sha256(original).hexdigest(),
        prepared_sha256=hashlib.sha256(prepared.encode()).hexdigest(),
        quantized_contract=contract,
    )
    (directory / "native-source-remainder.json").write_text(json.dumps(proof, indent=2) + "\n")
    record["source_specialization"] = proof
    return proof, original
