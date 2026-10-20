"""Payload-independent byte boundaries for exact component arithmetic.

This is a numerical contract on floating source families, not a claim about
native IEEE accumulation. Only entry/output quantization rounds. Component
formation and full-i32 readout retain the exact rational operation.
"""

import math
from fractions import Fraction


def parameters(spec):
    numeric = spec["numerical_semantics"]
    rule = numeric.get("quantized_native_fallback", {})
    domain = numeric["calibration_domain"]
    if rule.get("admission") != "after_native_exact_failure" or domain.get("evaluation_inputs") != "never_read":
        raise ValueError("missing input-independent quantized fallback contract")
    if domain["scale_policy"] != "enclosing_power_of_two" or not domain["minimum"] < domain["maximum"]:
        raise ValueError("unsupported declared observer domain")
    extent = max(abs(Fraction(domain["minimum"])), abs(Fraction(domain["maximum"])))
    scale = Fraction(2) ** math.ceil(math.log2(extent / Fraction(127)))
    if numeric["quantization_nonfinite"] != "byte_storage_zero_nan_saturating_infinity":
        raise ValueError("unsupported nonfinite quantizer contract")
    return dict(
        scale_numerator=scale.numerator,
        scale_denominator=scale.denominator,
        exponent=scale.numerator.bit_length() - scale.denominator.bit_length(),
        quant_min=-128,
        quant_max=127,
        zero_point=0,
        nonfinite=numeric["quantization_nonfinite"],
        calibration_domain=domain,
        schema="merlin.quantized_native.v1",
    )


def quantize_entries(module, contract):
    from xdsl.dialects import arith
    from xdsl.dialects import math as maths
    from xdsl.dialects.builtin import FloatAttr, f64, i32

    from merlin.llvmlower.floating_component_ir import point

    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    known = {}
    for value in function.body.block.args:
        typ = value.type.element_type
        if str(typ) not in {"f16", "bf16", "f32"}:
            continue
        uses = list(value.uses)
        operations = []

        def scalar(xs, emit):
            x = emit(arith.ExtFOp(xs[0], f64))

            def c(v):
                return emit(arith.ConstantOp(FloatAttr(float(v), f64)))

            isnan = emit(arith.CmpfOp(x, x, "uno"))
            x = emit(arith.SelectOp(isnan, c(0), x))
            x = emit(arith.MaximumfOp(x, c(-128 * 2.0 ** contract["exponent"])))
            x = emit(arith.MinimumfOp(x, c(127 * 2.0 ** contract["exponent"])))
            # Exact f32->f64 extension and power-of-two product in this band.
            scaled = emit(arith.MulfOp(x, c(2.0 ** -contract["exponent"])))
            rounded = emit(maths.RoundEvenOp(scaled))
            code = emit(arith.FPToSIOp(rounded, i32))
            f = emit(arith.SIToFPOp(code, typ))
            return emit(arith.MulfOp(f, emit(arith.ConstantOp(FloatAttr(2.0 ** contract["exponent"], typ)))))

        result = point([value], value.type, scalar, operations)
        function.body.block.insert_ops_before(operations, function.body.block.first_op)
        for use in uses:
            if use.operation.name != "func.return":
                use.operation.operands[use.index] = result
        known[result] = (contract["exponent"], 128)
    return known


def output_boundary(integer, exponent, bound, denominator, typ, operations, contract):
    """Eight exact integer comparisons; no rational runtime multiply/divide."""
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, IndexType, IntegerAttr, TensorType, i32, i64

    from merlin.llvmlower.floating_component_ir import point

    ratio = Fraction(2) ** (exponent - contract["exponent"]) / denominator
    thresholds = []
    for q in range(-127, 128):
        b = Fraction(2 * q - 1, 2) / ratio
        t = -(-b.numerator // b.denominator)
        if b.denominator == 1 and q % 2:
            t += 1
        thresholds.append(min(bound + 1, max(-bound, t)))
    thresholds.append(bound + 1)
    # Limb tables avoid unsupported >64-bit vector element legalization.
    width = integer.type.element_type.width.data

    def scalar(xs, emit):
        def c(v, t=i64):
            return emit(arith.ConstantOp(IntegerAttr(v, t)))

        lo, hi, one = c(0), c(255), c(1)
        for _ in range(8):
            mid = emit(arith.ShRUIOp(emit(arith.AddiOp(lo, hi)), one))
            index = emit(arith.IndexCastOp(mid, IndexType()))
            threshold = c(0, integer.type.element_type)
            for offset in range(0, width, 64):
                words = [(x >> offset) & ((1 << 64) - 1) for x in thresholds]
                table = emit(arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [256]), words)))
                word = emit(tensor.ExtractOp(table, [index], i64))
                limb = emit(arith.ExtUIOp(word, integer.type.element_type))
                if offset:
                    limb = emit(arith.ShLIOp(limb, c(offset, integer.type.element_type)))
                threshold = emit(arith.OrIOp(threshold, limb))
            above = emit(arith.CmpiOp(xs[0], threshold, "sge"))
            active = emit(arith.CmpiOp(lo, hi, "slt"))
            above = emit(arith.AndIOp(above, active))
            lo = emit(arith.SelectOp(above, emit(arith.AddiOp(mid, one)), lo))
            hi = emit(arith.SelectOp(above, hi, mid))
        code = emit(arith.AddiOp(lo, c(-128)))
        code = emit(arith.TruncIOp(code, i32))
        fp = emit(arith.SIToFPOp(code, typ.element_type))
        return emit(arith.MulfOp(fp, emit(arith.ConstantOp(FloatAttr(2.0 ** contract["exponent"], typ.element_type)))))

    return point([integer], typ, scalar, operations)


def finite_source(module):
    """Remove NaN word overlays under the declared finite rational body.

    The source's numerical families must still lower completely. No division
    by a runtime value or literal zero is licensed by this transformation.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i1
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.entry_weight_projection import _pure_dce
    from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals

    for op in list(module.walk()):
        if op.name == "arith.constant":
            attr = op.properties.get("value", op.attributes.get("value"))
            values = (
                list(attr.get_values())
                if hasattr(attr, "get_values")
                else [getattr(getattr(attr, "value", None), "data", 0)]
            )
            if any(isinstance(v, float) and not math.isfinite(v) for v in values):
                raise ValueError("nonfinite literal arithmetic is not a finite quantized family")
        if op.name == "arith.cmpf" and op.predicate.value.data in (7, 14):
            # Ordered/unordered differ only in their nonfinite classification.
            c = arith.ConstantOp(IntegerAttr(op.predicate.value.data == 7, i1))
            Rewriter.replace_op(op, [c], [c.result])
    # Fold selectors of these proven finite classifications, then prune words.
    for op in list(module.walk()):
        if op.name == "arith.select":
            owner = op.operands[0].owner
            if getattr(owner, "name", None) == "arith.constant" and isinstance(owner.value, IntegerAttr):
                Rewriter.replace_op(op, [], [op.operands[1 if owner.value.value.data else 2]])
    for function in module.walk():
        if function.name == "func.func":
            _pure_dce(function)
    fold_uniform_numeric_literals(module)
