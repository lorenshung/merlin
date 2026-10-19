"""Certified exact integer readout specialization before upstream LLVM lowering.

The source body is checked structurally against the frontend arithmetic, never
accepted on provenance alone. Bounds use the complete signed int32 domain.
Unsupported expressions retain their original exact implementation.
"""

import json
from fractions import Fraction
from functools import lru_cache

from merlin.llvmlower.exact_readout_source import Scalar, build_readout


@lru_cache(maxsize=4096)
def prove_scale(numerator, denominator, low=-128, high=127, zero_point=0):
    """Prove a u64*u64->u128 quotient estimate and exact remainder correction.

    M=ceil(num*2^s/den), E=M*den-num*2^s. For 0<=a<=2^31,
    a*M/2^s = a*num/den + a*E/(den*2^s).
    A*E < den*2^s proves floor(a*M/2^s) is q or q+1.
    Comparing estimate*den to a*num recovers q EXACTLY, then the exact
    remainder decides all half ties. This extends the provider reciprocal
    certificate to avoid an overflowing intermediate a*num in signed i64.
    """
    num, den, a = abs(numerator), denominator, 1 << 31
    fallback = dict(method="source_exact", reason="rational exceeds certified u128 intermediates")
    if not num or den <= 0 or den >= 1 << 127 or a * num >= 1 << 127:
        return fallback
    # A quotient and its estimate must fit i64; products and residual fit i128.
    if (a * num + den - 1) // den >= (1 << 63) - 1:
        return fallback
    candidate = None
    # Prefer a proved exact floor when it still fits the u64 multiplier.
    # Otherwise the relaxed estimate is corrected with the exact numerator.
    for exact_floor in (True, False):
        for shift in range(128):
            m = (num * (1 << shift) + den - 1) // den
            error = m * den - num * (1 << shift)
            limit = (1 if exact_floor else den) * (1 << shift)
            if m < 1 << 64 and a * error < limit:
                candidate = m, shift, error, exact_floor
                break
        if candidate is not None:
            break
    if candidate is None:
        return dict(method="source_exact", reason="no certified u64 multiplier")
    m, shift, error, exact_floor = candidate
    max_estimate = a * m >> shift
    if max_estimate * den >= 1 << 127:
        return fallback

    def fast(value):
        magnitude = abs(value)
        q = magnitude * m >> shift
        q -= q * den > magnitude * num
        r = magnitude * num - q * den
        q += r > den - r or (r == den - r and q & 1)
        return -q if (value < 0) != (numerator < 0) else q

    points = {-(1 << 31), (1 << 31) - 1, -1, 0, 1}
    for k in range(low - zero_point - 1, high - zero_point + 2):
        edge = (2 * k + 1) * den // (2 * num)
        for v in (edge, -edge):
            points.update(range(v - 2, v + 3))
    points = sorted(v for v in points if -(1 << 31) <= v < 1 << 31)
    for v in points:
        if fast(v) != round(Fraction(v * numerator, den)):
            return dict(method="source_exact", reason="boundary verification failed")
    return dict(
        method="exact_multiply_shift",
        numerator=str(numerator),
        denominator=str(den),
        accumulator_min=-(1 << 31),
        accumulator_max=(1 << 31) - 1,
        bound_source="full_int32_storage",
        multiplier=str(m),
        shift=shift,
        exact_floor=exact_floor,
        error=str(error),
        error_product=str(a * error),
        error_limit=str((1 if exact_floor else den) * (1 << shift)),
        quotient_proof="A*E<D*2^s => estimate=q or q+1; subtract iff estimate*D>a*abs(num)",
        rounding_proof="exact r; increment iff r>D-r or (r=D-r and q odd); signed symmetry",
        verification="formal_full_domain_and_boundary_tests",
        tested_values=len(points),
    )


def table_lookup(b, values, index, typ):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType

    if len(set(values)) == 1:
        return b.c(values[0], typ)
    if typ.width.data > 64:
        from xdsl.dialects.builtin import i64

        result = b.c(0, typ)
        for offset in range(0, typ.width.data, 64):
            words = [(v >> offset) & ((1 << 64) - 1) for v in values]
            word = b.cast(table_lookup(b, words, index, i64), typ, unsigned=True)
            result = b.bor(result, b.shl(word, b.c(offset, typ)))
        return result
    table = b.emit(arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(typ, [len(values)]), values)))
    return b.emit(tensor.ExtractOp(table, [index], typ))


def emit_scale(b, accumulator, proofs, low, high, zp, relu, index=None):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerType, i8, i64

    proofs = proofs if isinstance(proofs, list) else [proofs]

    def parameters(key):
        return [int(p[key]) for p in proofs]

    # Use native arithmetic whenever the full-domain proof permits it.
    residual_type = (
        i64
        if all((1 << 31) * abs(int(p["numerator"])) + int(p["denominator"]) < 1 << 63 for p in proofs)
        else IntegerType(128)
    )
    product_type = (
        i64 if all((1 << 31) * int(p["multiplier"]) < 1 << 64 and p["shift"] < 64 for p in proofs) else IntegerType(128)
    )

    def parameter(key, typ, absolute=False):
        values = parameters(key)
        if absolute:
            values = [abs(v) for v in values]
        return table_lookup(b, values, index, typ)

    value = b.cast(accumulator, i64)
    negative = b.cmp(value, b.c(0, i64), "slt")
    magnitude = b.select(negative, b.neg(value), value)
    product = b.mul(b.cast(magnitude, product_type, unsigned=True), parameter("multiplier", product_type))
    estimate = b.cast(b.shr(product, parameter("shift", product_type)), i64, unsigned=True)
    numerator = b.mul(b.cast(magnitude, residual_type, unsigned=True), parameter("numerator", residual_type, True))
    den = parameter("denominator", residual_type)
    if all(p["exact_floor"] for p in proofs):
        q = estimate
    else:
        excess = b.cmp(b.mul(b.cast(estimate, residual_type, unsigned=True), den), numerator, "ugt")
        q = b.sub(estimate, b.cast(excess, i64, unsigned=True))
    remainder = b.sub(numerator, b.mul(b.cast(q, residual_type, unsigned=True), den))
    complement = b.sub(den, remainder)
    odd = b.cmp(b.band(q, b.c(1, i64)), b.c(0, i64), "ne")
    up = b.bor(b.cmp(remainder, complement, "ugt"), b.band(b.cmp(remainder, complement, "eq"), odd))
    q = b.add(q, b.cast(up, i64, unsigned=True))
    sign = table_lookup(b, [int(p["numerator"]) < 0 for p in proofs], index, IntegerType(1))
    negative = b.emit(arith.XOrIOp(negative, sign))
    q = b.select(negative, b.neg(q), q)
    low = max(low, min(high, zp)) if relu else low
    q = b.emit(arith.MaxSIOp(q, b.c(low - zp, i64)))
    q = b.emit(arith.MinSIOp(q, b.c(high - zp, i64)))
    return b.cast(b.add(q, b.c(zp, i64)), i8)


def _parameters(op):
    from xdsl.dialects.builtin import f32, i32

    a = op.attributes
    if (
        op.name != "linalg.generic"
        or a.get("merlin.exact_readout_proof") is not None
        or getattr(a.get("prov.role"), "data", None) != "readout"
        or getattr(a.get("prov.quant.arithmetic"), "data", None) != "exact"
        or len(op.inputs) not in (1, 2)
        or op.inputs[0].type.element_type != i32
    ):
        return None

    def integer(key, default):
        return a[key].value.data if key in a else default

    def boolean(key):
        v = a.get(key)
        return getattr(v, "data", None) == "true" or getattr(getattr(v, "value", None), "data", 0) != 0

    nums, dens = a.get("prov.quant.scale_numerator"), a.get("prov.quant.scale_denominator")
    if nums is None or dens is None or len(nums) != len(dens):
        return None
    ratios = [Fraction(int(n.data), int(d.data)) for n, d in zip(nums, dens)]
    output_scale = a.get("prov.quant.output_scale")
    output_scale = Fraction(output_scale.value.data) if output_scale is not None else None
    quantized = str(op.results[0].type.element_type) == "i8"
    if quantized and output_scale is None:
        output_scale = Fraction(1)
    if not quantized and op.results[0].type.element_type != f32:
        return None
    beta = a.get("prov.quant.beta")
    params = dict(
        bias=op.inputs[1] if len(op.inputs) == 2 else None,
        beta=beta.value.data if beta is not None else 1.0,
        output_scale=output_scale,
        zero_point=integer("prov.quant.zero_point", 0),
        quant_min=integer("prov.quant.quant_min", -128),
        quant_max=integer("prov.quant.quant_max", 127),
        relu=boolean("prov.quant.relu"),
        unsigned=getattr(a.get("prov.quant.storage_dtype"), "data", None) == "uint8",
    )
    scales = [r * (output_scale if output_scale is not None else 1) for r in ratios]
    return scales, ratios, params


def lower_exact_readouts(module):
    from xdsl.dialects.builtin import StringAttr
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region

    count = 0
    for op in list(module.walk()):
        parameters = _parameters(op)
        if parameters is None:
            continue
        scales, ratios, params = parameters
        # The frontend integer-grid builder is defined for binary source
        # coefficients. Arbitrary non-dyadic provenance cannot authenticate it.
        if any(s.denominator & (s.denominator - 1) for s in scales):
            op.attributes["merlin.exact_readout_proof"] = StringAttr(
                json.dumps(dict(method="source_exact", reason="source coefficient is not dyadic"))
            )
            continue
        source_ops, source = build_readout(op.inputs[0], scales, **params)
        expected = source.owner
        matches = (
            op.body.block.is_structurally_equivalent(expected.body.block)
            and op.indexing_maps == expected.indexing_maps
            and op.iterator_types == expected.iterator_types
        )
        # Delete detached validation operations so they do not leave ghost SSA uses.
        for item in reversed(source_ops):
            item.drop_all_references()
        if not matches:
            proof = dict(method="source_exact", reason="source scalar body differs from exact readout specification")
        elif params["output_scale"] is None:
            block, proof = floating_body(op, scales, params)
            old = op.detach_region(op.body)
            op.add_region(Region(block))
            old.drop_all_references()
            count += 1
        elif params["bias"] is not None:
            proof = dict(method="source_exact", reason="unbounded runtime float bias needs the exact wide path")
        else:
            proofs = [
                dict(
                    prove_scale(
                        r.numerator, r.denominator, params["quant_min"], params["quant_max"], params["zero_point"]
                    )
                )
                for r in ratios
            ]
            proof = dict(method="exact_multiply_shift", scales=proofs)
            if any(p["method"] != "exact_multiply_shift" for p in proofs):
                proof = dict(method="source_exact", scales=proofs, reason="scale certificate failed")
            else:
                block = Block(arg_types=[v.type for v in op.body.block.args])
                b = Scalar()
                index = None
                if len(proofs) > 1:
                    index = b.emit(linalg.IndexOp(len(op.results[0].type.get_shape()) - 1))
                result = emit_scale(
                    b,
                    block.args[0],
                    proofs,
                    params["quant_min"],
                    params["quant_max"],
                    params["zero_point"],
                    params["relu"],
                    index,
                )
                block.add_ops([*b.ops, linalg.YieldOp(result)])
                old = op.detach_region(op.body)
                op.add_region(Region(block))
                old.drop_all_references()
                count += 1
        op.attributes["merlin.exact_readout_proof"] = StringAttr(json.dumps(proof, sort_keys=True))
    return count


def emit_dyadic_float(b, value, exponent, out_type):
    """One RNE to f32 using native limb leading-bit searches; every alignment/rounding bit is exact.

    Wide values are needed only for arbitrary finite runtime bias. The leading
    search and mantissa/exponent arithmetic themselves stay native i64.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import i32, i64

    width = value.type.width.data
    typ = value.type
    z, one = b.c(0, i64), b.c(1, i64)
    negative = b.cmp(value, b.c(0, typ), "slt")
    mag = b.select(negative, b.neg(value), value)
    length = z
    for offset in range(0, width, 64):
        word = b.cast(b.shr(mag, b.c(offset, typ)), i64, unsigned=True) if offset else b.cast(mag, i64, unsigned=True)
        probe, bits = word, z
        for step in (32, 16, 8, 4, 2, 1):
            large = b.cmp(probe, b.c(1 << step, i64), "uge")
            probe = b.select(large, b.shr(probe, b.c(step, i64)), probe)
            bits = b.add(bits, b.select(large, b.c(step, i64), z))
        bits = b.add(bits, b.c(offset + 1, i64))
        length = b.select(b.cmp(word, z, "ne"), bits, length)
    drop = b.emit(arith.MaxSIOp(b.sub(length, b.c(24, i64)), b.c(-149 - exponent, i64)))
    right = b.emit(arith.MinSIOp(b.emit(arith.MaxSIOp(drop, z)), b.c(width - 1, i64)))
    # A negative drop is at least -23 (nonzero integers have length >=1).
    left = b.emit(arith.MinSIOp(b.emit(arith.MaxSIOp(b.neg(drop), z)), b.c(23, i64)))
    q = b.cast(b.shr(mag, b.cast(right, typ, unsigned=True)), i64, unsigned=True)
    q = b.shl(q, left)
    mask = b.sub(b.shl(b.c(1, typ), b.cast(right, typ, unsigned=True)), b.c(1, typ))
    rem = b.band(mag, mask)
    half_shift = b.emit(arith.MaxSIOp(b.sub(right, one), z))
    half = b.shl(b.c(1, typ), b.cast(half_shift, typ, unsigned=True))
    up = b.bor(b.cmp(rem, half, "ugt"), b.band(b.cmp(rem, half, "eq"), b.cmp(b.band(q, one), z, "ne")))
    up = b.band(up, b.cmp(drop, z, "sgt"))
    rounded = b.add(q, b.cast(up, i64, unsigned=True))
    rounded = b.select(b.cmp(drop, b.c(width, i64), "sge"), z, rounded)
    carry = b.cmp(rounded, b.c(1 << 24, i64), "uge")
    rounded = b.select(carry, b.shr(rounded, one), rounded)
    unit = b.add(b.c(exponent, i64), b.add(drop, b.cast(carry, i64, unsigned=True)))
    normal = b.cmp(rounded, b.c(1 << 23, i64), "uge")
    exp_field = b.add(unit, b.c(150, i64))
    bits = b.select(normal, b.add(b.shl(exp_field, b.c(23, i64)), b.sub(rounded, b.c(1 << 23, i64))), rounded)
    overflow = b.band(normal, b.cmp(exp_field, b.c(254, i64), "sgt"))
    bits = b.select(overflow, b.c(0x7F800000, i64), bits)
    bits = b.bor(bits, b.select(negative, b.c(0x80000000, i64), z))
    return b.emit(arith.BitcastOp(b.cast(bits, i32, unsigned=True), out_type))


def floating_body(op, scales, params):
    from xdsl.dialects.builtin import IntegerType, i64
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block

    from merlin.llvmlower.exact_readout_source import readout_plan

    bias = params["bias"]
    base, coeffs, beta_coeff, bias_unit, _, width = readout_plan(
        scales, params["beta"], None if bias is None else bias.type.element_type, None
    )
    # Exact values need a sign bit; no normalization padding is necessary.
    bound = max(abs(c) for c in coeffs) * (1 << 31) + abs(beta_coeff) * (1 << 277)
    width = max(64, bound.bit_length() + 1)
    width = (width + 63) // 64 * 64
    typ = IntegerType(width)
    block = Block(arg_types=[v.type for v in op.body.block.args])
    b = Scalar()
    index = None
    if len(set(coeffs)) > 1:
        index = b.emit(linalg.IndexOp(len(op.results[0].type.get_shape()) - 1))
    coefficient = table_lookup(b, coeffs, index, typ)
    value = b.mul(b.cast(block.args[0], typ), coefficient)
    if bias is not None:
        bm, be, finite = b.decode(block.args[1])
        b.assert_(finite, "exact readout requires finite bias")
        shift = b.cast(b.sub(be, b.c(bias_unit, i64)), typ)
        fixed = b.shl(b.cast(bm, typ), shift)
        value = b.add(value, b.mul(fixed, b.c(beta_coeff, typ)))
    result = emit_dyadic_float(b, value, base, op.results[0].type.element_type)
    block.add_ops([*b.ops, linalg.YieldOp(result)])
    return block, dict(
        method="exact_dyadic_limb_normalization",
        integer_width=width,
        exponent=base,
        accumulator_min=-(1 << 31),
        accumulator_max=(1 << 31) - 1,
        numerator_bound=str(bound),
        coefficients=[str(c) for c in coeffs],
        proof=(
            "full i32 product plus every finite f32 bias fits signed width; "
            "exact leading bits and sticky remainder; one IEEE RNE"
        ),
    )
