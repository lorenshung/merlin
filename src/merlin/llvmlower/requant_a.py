"""Emit the shared requant-A boundary after validating the exact source body."""

import hashlib
import json
import struct
from fractions import Fraction
from pathlib import Path

from merlin.runtime.requant_a import plan, validate

from .exact_readout_source import Scalar


def frozen_bias(op, bias):
    """Return exact f32 constants, plus a bit guard for file-bound parameters.

    A parameter file supplies candidates, never permission to ignore a runtime
    operand. Its bitwise guard preserves the ordinary callable ABI if a caller
    changes a weight after capture. Literal SSA constants need no guard.
    """
    if bias is None:
        return None
    from xdsl.ir import BlockArgument

    from merlin.capture.safetensors import load_header

    owner = bias.owner
    shape = tuple(bias.type.get_shape())
    if len(shape) > 1:
        return None
    count = shape[0] if shape else 1
    values = None
    binding = dict(source="ssa_constant", guarded=False)
    if getattr(owner, "name", None) == "arith.constant":
        attr = owner.properties["value"]
        values = list(attr.get_values())
    elif (
        getattr(owner, "name", None) == "tensor.splat"
        and getattr(owner.operands[0].owner, "name", None) == "arith.constant"
    ):
        values = [owner.operands[0].owner.value.value.data] * count
    elif isinstance(bias, BlockArgument):
        function = bias.owner.parent_op()
        if function is None or function.name != "func.func":
            return None
        module = function
        while module.parent_op() is not None:
            module = module.parent_op()
        path = getattr(module.attributes.get("prov.weights_file"), "data", None)
        if not path:
            return None
        path = Path(path)
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not path.is_file() or not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get(str(bias.index), {})
        if entry.get("kind") != "param" or entry.get("dtype") != "float32" or tuple(entry.get("shape", [])) != shape:
            return None
        header, start = load_header(path)
        item = header.get(entry.get("weight"), {})
        if item.get("dtype") != "F32" or tuple(item.get("shape", [])) != shape:
            raise ValueError("frozen bias payload differs from its declared ABI")
        low, high = item["data_offsets"]
        if low < 0 or high - low != 4 * count:
            raise ValueError("invalid frozen bias byte interval")
        with path.open("rb") as source:
            source.seek(start + low)
            data = source.read(high - low)
        if len(data) != high - low:
            raise ValueError("truncated frozen bias payload")
        values = struct.unpack("<" + "f" * count, data)
        binding = dict(
            source="frozen_parameter_candidate",
            argument=bias.index,
            weight=entry["weight"],
            guarded=True,
            payload_sha256=hashlib.sha256(data).hexdigest(),
        )
    if values is None or len(values) != count:
        return None
    import math

    if any(not math.isfinite(v) for v in values):
        return None
    fractions = [Fraction(v) for v in values]
    bits = [int.from_bytes(struct.pack("<f", v), "little") for v in values]
    return fractions, bits, binding


def records(scales, offsets, params):
    """Per-channel normalization keeps negative coefficients as integer weights."""
    if len(set(scales)) == 1:
        scales = scales[:1]
    if len(set(offsets)) == 1:
        offsets = offsets[:1]
    count = max(len(scales), len(offsets))
    if len(scales) not in (1, count) or len(offsets) not in (1, count):
        raise ValueError("requant-A channel parameter extents disagree")
    result = []
    for channel in range(count):
        ratio = scales[channel % len(scales)]
        weight = 1 if ratio > 0 else -1 if ratio < 0 else 0
        lo, hi = (-(1 << 31), (1 << 31) - 1) if weight == 1 else (-(1 << 31) + 1, 1 << 31)
        if weight == 0:
            lo = hi = 0
        rec = plan(
            abs(ratio) if ratio else Fraction(1),
            offsets[channel % len(offsets)],
            params["zero_point"],
            params["quant_min"],
            params["quant_max"],
            params["relu"],
            lo,
            hi,
        )
        validate(rec)
        rec.update(
            channel=channel,
            normal_form=[dict(tile="accumulator", weight=str(weight))],
            normal_form_width=64,
            accumulator_bounds=[
                dict(
                    tile="accumulator",
                    accumulator_min=str(-(1 << 31)),
                    accumulator_max=str((1 << 31) - 1),
                    bound_source="full_int32_storage",
                )
            ],
        )
        result.append(rec)
    return result


def emit(b, value, certificates, index=None):
    """No per-element rational arithmetic: i64 affine map or threshold search."""
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IndexType, IntegerType, TensorType, i8, i64
    from xdsl.ir import Block, Region

    from .exact_readout import table_lookup

    for rec in certificates:
        validate(rec)
    value = b.cast(value, i64)
    weights = [int(r["normal_form"][0]["weight"]) for r in certificates]
    if len(set(weights)) == 1:
        if weights[0] < 0:
            value = b.neg(value)
        elif weights[0] == 0:
            value = b.c(0, i64)
    else:
        weight = table_lookup(b, weights, index, i64)
        value = b.mul(value, weight)
    fast = [r["algorithm"] == "clamp_mul_add_shift" for r in certificates]

    def affine(builder):
        def parameter(key):
            values = [int(r["constants"][key]) if f else 0 for r, f in zip(certificates, fast, strict=True)]
            return table_lookup(builder, values, index, i64)

        v = builder.emit(arith.MaxSIOp(value, parameter("L")))
        v = builder.emit(arith.MinSIOp(v, parameter("H")))
        q = builder.add(builder.mul(v, parameter("M")), parameter("B"))
        q = builder.emit(arith.ShRSIOp(q, parameter("S")))
        if any(r.get("constants", {}).get("out_clamp") for r in certificates):
            qlo = [r.get("constants", {}).get("qlo", r["quant_min"]) for r in certificates]
            qhi = [r.get("constants", {}).get("qhi", r["quant_max"]) for r in certificates]
            q = builder.emit(arith.MaxSIOp(q, table_lookup(builder, qlo, index, i64)))
            q = builder.emit(arith.MinSIOp(q, table_lookup(builder, qhi, index, i64)))
        return builder.cast(q, i8)

    def threshold(builder):
        width = certificates[0]["quant_max"] - certificates[0]["quant_min"]
        rows = [
            list(map(int, r["thresholds"])) if not f else [int(r["domain"]["max"]) + 1] * width
            for r, f in zip(certificates, fast, strict=True)
        ]
        # A sentinel makes the final inactive lookup safe even at lo==hi.
        rows = [[*row, int(r["domain"]["max"]) + 1] for row, r in zip(rows, certificates, strict=True)]
        table = builder.emit(
            arith.ConstantOp(
                DenseIntOrFPElementsAttr.from_list(
                    TensorType(i64, [len(rows), width + 1]), [v for row in rows for v in row]
                )
            )
        )
        channel = index if index is not None else builder.c(0, IndexType())
        lo, hi, one = builder.c(0, i64), builder.c(width, i64), builder.c(1, i64)
        for _ in range(width.bit_length()):
            mid = builder.shr(builder.add(lo, hi), one)
            item = builder.emit(
                tensor.ExtractOp(table, [channel, builder.emit(arith.IndexCastOp(mid, IndexType()))], i64)
            )
            above = builder.band(builder.cmp(value, item, "sge"), builder.cmp(lo, hi, "slt"))
            lo, hi = builder.select(above, builder.add(mid, one), lo), builder.select(above, hi, mid)
        return builder.cast(builder.add(lo, builder.c(certificates[0]["quant_min"], i64)), i8)

    if all(fast):
        return affine(b)
    if not any(fast):
        return threshold(b)
    flag = table_lookup(b, list(map(int, fast)), index, IntegerType(1))
    regions = []
    for build in (affine, threshold):
        block = Block()
        inner = Scalar()
        result = build(inner)
        block.add_ops([*inner.ops, scf.YieldOp(result)])
        regions.append(Region(block))
    return b.emit(scf.IfOp(flag, [i8], regions[0], regions[1]))


def quantized_body(op, ratios, params, frozen):
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block

    offsets = (
        [Fraction(0)] if frozen is None else [v * Fraction(params["beta"]) / params["output_scale"] for v in frozen[0]]
    )
    certs = records(ratios, offsets, params)
    block = Block(arg_types=[v.type for v in op.body.block.args])
    b = Scalar()
    index = b.emit(linalg.IndexOp(len(op.results[0].type.get_shape()) - 1)) if len(certs) > 1 else None
    result = emit(b, block.args[0], certs, index)
    block.add_ops([*b.ops, linalg.YieldOp(result)])
    proof = dict(method="requant_A", schema="requant-A-v1", channels=certs, axis=-1)
    if frozen is not None:
        proof["frozen_bias"] = {**frozen[2], "bit_patterns": list(map(str, frozen[1]))}
        if frozen[2]["guarded"]:
            block = guarded_body(op, block, frozen[1])
    return block, proof


def guarded_body(op, fast_block, bits):
    """An immutable candidate cannot silently specialize a callable parameter."""
    from xdsl.dialects import arith, scf
    from xdsl.dialects.builtin import i32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region

    from .exact_readout import table_lookup

    block = Block(arg_types=[v.type for v in op.body.block.args])
    b = Scalar()
    index = b.emit(linalg.IndexOp(len(op.results[0].type.get_shape()) - 1)) if len(bits) > 1 else None
    raw = b.emit(arith.BitcastOp(block.args[1], i32))
    equal = b.cmp(raw, table_lookup(b, bits, index, i32), "eq")
    regions = []
    for source in (fast_block, op.body.block):
        values = dict(zip(source.args, block.args, strict=True))
        inner = Block()
        for old in source.ops:
            if old.name == "linalg.yield":
                inner.add_op(scf.YieldOp(values[old.operands[0]]))
            elif old.name == "linalg.index":
                # linalg.index must stay directly under linalg.generic.
                values[old.results[0]] = index if index is not None else b.emit(old.clone())
            else:
                inner.add_op(old.clone(value_mapper=values))
        regions.append(Region(inner))
    fast_block.drop_all_references()
    result = b.emit(scf.IfOp(equal, [op.results[0].type.element_type], regions[0], regions[1]))
    block.add_ops([*b.ops, linalg.YieldOp(result)])
    return block


def frozen_float_body(op, scales, params, frozen):
    """Retain an observed IEEE boundary, using its actual frozen bias grid.

    IEEE outputs have many more than 255 steps and cannot claim the byte
    requant-A contract. This exact specialization is recorded distinctly.
    """
    from xdsl.dialects.builtin import IntegerType
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block

    from .exact_readout import emit_dyadic_float, floating_body, table_lookup

    offsets = [v * Fraction(params["beta"]) for v in frozen[0]]
    if all(v == 0 for v in offsets):
        block, proof = floating_body(op, scales, {**params, "bias": None})
    else:
        count = max(len(scales), len(offsets))
        if len(scales) not in (1, count) or len(offsets) not in (1, count):
            raise ValueError("frozen floating bias channel extent differs")
        base = min(1 - v.denominator.bit_length() for v in [*scales, *offsets])
        coeffs = [int(scales[i % len(scales)] / Fraction(2) ** base) for i in range(count)]
        constants = [int(offsets[i % len(offsets)] / Fraction(2) ** base) for i in range(count)]
        bound = max(abs(c) * (1 << 31) + abs(k) for c, k in zip(coeffs, constants, strict=True))
        width = max(64, ((bound.bit_length() + 1 + 63) // 64) * 64)
        typ = IntegerType(width)
        block = Block(arg_types=[v.type for v in op.body.block.args])
        b = Scalar()
        index = b.emit(linalg.IndexOp(len(op.results[0].type.get_shape()) - 1)) if count > 1 else None
        value = b.add(
            b.mul(b.cast(block.args[0], typ), table_lookup(b, coeffs, index, typ)),
            table_lookup(b, constants, index, typ),
        )
        result = emit_dyadic_float(b, value, base, op.results[0].type.element_type)
        block.add_ops([*b.ops, linalg.YieldOp(result)])
        proof = dict(
            method="exact_frozen_dyadic_ieee",
            integer_width=width,
            exponent=base,
            numerator_bound=str(bound),
            proof="exact frozen rational bias plus full int32 product; one IEEE rounding",
        )
    proof["frozen_bias"] = {**frozen[2], "bit_patterns": list(map(str, frozen[1]))}
    if frozen[2]["guarded"]:
        block = guarded_body(op, block, frozen[1])
    return block, proof


def validate_lowered(op, scales, ratios, params, proof):
    """Rebuild imported A IR from the exact source, including guard and constants.

    A marker or a self-consistent certificate alone cannot authenticate the
    emitted arithmetic. Rechecking the complete body also binds each channel's
    constants to its original ratio, offset, zero point and accumulator domain.
    """
    from .exact_readout_source import build_readout

    if proof.get("schema") != "requant-A-v1" or params["output_scale"] is None:
        raise ValueError("invalid requant-A source binding")
    for certificate in proof["channels"]:
        validate(certificate)
    frozen = None
    binding = proof.get("frozen_bias")
    if binding is not None:
        bits = [int(v) for v in binding["bit_patterns"]]
        if any(not 0 <= v < 1 << 32 for v in bits):
            raise ValueError("invalid frozen bias bit pattern")
        values = [struct.unpack("<f", v.to_bytes(4, "little"))[0] for v in bits]
        import math

        if not all(math.isfinite(v) for v in values):
            raise ValueError("invalid frozen bias value")
        frozen = ([Fraction(v) for v in values], bits, {k: v for k, v in binding.items() if k != "bit_patterns"})
        if not binding.get("guarded"):
            literal = frozen_bias(op, params["bias"])
            if literal is None or literal[2]["guarded"] or literal[1] != bits:
                raise ValueError("unguarded bias is not an SSA constant")
    elif params["bias"] is not None:
        raise ValueError("missing frozen bias binding")
    operations, result = build_readout(op.inputs[0], scales, **params)
    rebuilt, expected_proof = quantized_body(result.owner, ratios, params, frozen)
    compared = op.body.clone()
    for inner in compared.walk():
        for key in ("prov.source_node_ids", "prov.origin_node_ids", "prov.trace_role"):
            inner.attributes.pop(key, None)
    matches = (
        proof == expected_proof
        and compared.block.is_structurally_equivalent(rebuilt)
        and op.indexing_maps == result.owner.indexing_maps
        and op.iterator_types == result.owner.iterator_types
    )
    compared.drop_all_references()
    rebuilt.drop_all_references()
    for item in reversed(operations):
        item.drop_all_references()
    if not matches:
        raise ValueError("requant-A proof or emitted body differs from exact source")
