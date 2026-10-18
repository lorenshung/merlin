"""Exact rational RNE and IEEE signed-zero epilogues over full integer readout."""

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import Rewriter

from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer


def component_readout(
    *,
    point,
    integer_type,
    contractions,
    grids,
    raw_forms,
    components,
    proofs,
    packing_limit,
    wide_packing,
    ieee_pointwise=False,
):
    """Bind one program's numerical proof state; return its exact readout emitter."""
    i128 = integer_type

    def publish(op, integer, exponent, integer_bound, operations, compute_type=None, denominator=1):
        compute_type = compute_type or op.results[0].type.element_type
        output_type = op.results[0].type.element_type

        def convert(values, emit):
            rounded = encode_fixed_integer(values[0], exponent, compute_type, emit, denominator=denominator)
            if output_type != compute_type:
                rounded = emit(arith.TruncFOp(rounded, output_type))
            return rounded

        converted = point([integer], op.results[0].type, convert, operations)
        # Exact zero has a source-defined sign, independent of the discarded
        # floating computation. Underflow of a nonzero integer keeps the RNE sign.
        identity = AffineMapAttr(AffineMap.identity(len(op.results[0].type.get_shape())))
        maps = list(op.get_indexing_maps())
        empty = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[output_type, i128] + [v.type.element_type for v in op.inputs] + [output_type])

        def emit(operation):
            block.add_op(operation)
            return operation.results[0]

        def sign(value):
            from xdsl.dialects.builtin import IntegerType

            f, e, _ = FORMAT[str(value.type)]
            typ = IntegerType(f + e + 1)
            raw = emit(arith.BitcastOp(value, typ))
            mask = emit(arith.ConstantOp(IntegerAttr(1 << (f + e), typ)))
            masked = emit(arith.AndIOp(raw, mask))
            zero = emit(arith.ConstantOp(IntegerAttr(0, typ)))
            return emit(arith.CmpiOp(masked, zero, "ne"))

        scalar_ops = [child for child in op.body.block.ops if child.name != "linalg.yield"]
        negative = None
        if len(scalar_ops) == 1:
            scalar = scalar_ops[0]
            if all(value in op.body.block.args[: len(op.inputs)] for value in scalar.operands):
                operands = [block.args[2 + op.body.block.args.index(value)] for value in scalar.operands]
                if scalar.name in {"arith.extf", "arith.truncf", "arith.negf"}:
                    negative = sign(operands[0])
                    if scalar.name == "arith.negf":
                        from xdsl.dialects.builtin import i1

                        one = emit(arith.ConstantOp(IntegerAttr(1, i1)))
                        negative = emit(arith.XOrIOp(negative, one))
                elif scalar.name in {"arith.mulf", "arith.divf"}:
                    negative = emit(arith.XOrIOp(sign(operands[0]), sign(operands[1])))
                elif scalar.name in {"arith.addf", "arith.subf"}:
                    signs = [sign(value) for value in operands]
                    if scalar.name == "arith.subf":
                        from xdsl.dialects.builtin import i1

                        one = emit(arith.ConstantOp(IntegerAttr(1, i1)))
                        signs[1] = emit(arith.XOrIOp(signs[1], one))
                    negative = emit(arith.AndIOp(*signs))
                    for value in operands:
                        zero = emit(arith.ConstantOp(FloatAttr(0.0, value.type)))
                        is_zero = emit(arith.CmpfOp(value, zero, "oeq"))
                        negative = emit(arith.AndIOp(negative, is_zero))
        positive_zero = emit(arith.ConstantOp(FloatAttr(0.0, output_type)))
        signed_zero = positive_zero
        if negative is not None:
            negative_zero = emit(arith.ConstantOp(FloatAttr(-0.0, output_type)))
            signed_zero = emit(arith.SelectOp(negative, negative_zero, positive_zero))
        integer_zero = emit(arith.ConstantOp(IntegerAttr(0, i128)))
        exact_zero = emit(arith.CmpiOp(block.args[1], integer_zero, "eq"))
        selected = emit(arith.SelectOp(exact_zero, signed_zero, block.args[0]))
        if ieee_pointwise:
            from merlin.llvmlower.ieee_pointwise import overlay

            selected = overlay(op, block.args[2 : 2 + len(op.inputs)], selected, emit)
        block.add_op(L.YieldOp(selected))
        if (
            op in contractions
            or any(it.data == L.IteratorType.REDUCTION for it in op.get_iterator_types())
            or any(child.name == "scf.for" for child in op.body.walk())
        ):
            # A positive-zero seeded contraction rounds exact cancellation to +0.
            result = converted
        else:
            final = L.GenericOp(
                inputs=[converted, integer, *op.inputs],
                outputs=[empty.tensor],
                body=Region(block),
                indexing_maps=ArrayAttr([identity, identity, *maps[:-1], identity]),
                iterator_types=op.get_iterator_types(),
                result_types=[op.results[0].type],
            )
            operations.extend([empty, final])
            result = final.results[0]
        max_exponent = (2 * integer_bound).bit_length() - 1 + exponent if integer_bound else 0
        grids[result] = (
            (exponent, 2 * integer_bound)
            if denominator == 1
            and (2 * integer_bound).bit_length() <= packing_limit
            and max_exponent < FORMAT[str(output_type)][2]
            else None
        )
        raw_forms[result] = (integer, exponent, integer_bound, denominator)
        tag = getattr(integer.owner.attributes.get("merlin.exact_component"), "data", None)
        if tag:
            components[result] = tag
        proofs.append(
            dict(
                source_operation=op.name,
                source_data_arithmetic=any(
                    child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf"}
                    for child in op.body.walk()
                ),
                common_exponent=exponent,
                integer_absolute_bound=integer_bound,
                reconstruction=f"bounded_i{i128.width.data}",
                packing="exact_ieee_bit_fields" if wide_packing else "exact_binary64_power_of_two",
                rounding="integer_quotient_remainder_rne",
                exact_zero_sign="source_ieee_sign_bits",
                ieee_special_readout="affected_elements_only" if ieee_pointwise else "none",
            )
        )
        Rewriter.replace_op(op, operations, [result])

    return publish
