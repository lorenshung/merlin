"""Exact pointwise IEEE products using dtype-bounded significand limb GEMMs.

Exponents remain runtime representation metadata. Full significands (including
every subnormal) are device operands; their exact product receives one integer
RNE readout. No exponent span or limb count is inferred from tensor data.
"""


def rewrite_significand_products(module):
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        IntegerAttr,
        TensorType,
        UnitAttr,
        i64,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.llvmlower.floating_component_ir import point
    from merlin.llvmlower.ieee_dyadic_ir import decode, encode
    from merlin.llvmlower.ieee_pointwise import overlay
    from merlin.llvmlower.uniform_numeric_literals import compile_time_operand

    bounds, proofs = {}, []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        token = str(op.results[0].type.element_type)
        if token not in {"f16", "bf16", "f32"} or any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        if (
            len(body) != 2
            or body[0].name != "arith.mulf"
            or list(body[0].operands) != list(op.body.block.args[:2])
            or list(body[1].operands) != list(body[0].results)
        ):
            continue
        if not any(compile_time_operand(value) for value in op.inputs):
            continue
        shape = op.results[0].type.get_shape()
        maps = list(op.get_indexing_maps())
        identity = AffineMapAttr(AffineMap.identity(len(shape)))
        if maps[-1] != identity:
            continue
        operations, expanded = [], []
        magnitude = (1 << (FORMAT[token][0] + 1)) - 1
        operand_bounds = []
        for source, amap in zip(op.inputs, maps[:-1]):
            from math import isfinite

            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            literal = _constant_of(source)
            bound = magnitude
            if literal is not None and isfinite(literal):
                fraction, exponent, _ = FORMAT[token]
                if token == "bf16":
                    import struct

                    from merlin.targetgen.exact_arithmetic import round_binary

                    rounded = round_binary(literal, 16, bfloat16=True)
                    word = struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16
                else:
                    word = int.from_bytes(source.type.element_type.pack([float(literal)]), "little")
                field = word >> fraction & ((1 << exponent) - 1)
                coefficient = word & ((1 << fraction) - 1)
                coefficient |= 1 << fraction if field else 0
                if word >> (fraction + exponent):
                    coefficient = -coefficient
                bound = abs(coefficient)
                constant = arith.ConstantOp(IntegerAttr(coefficient, i64))
                splat = tensor.SplatOp(constant.result, [], TensorType(i64, source.type.get_shape()))
                operations.extend([constant, splat])
                packed = splat.result
            else:
                packed = point(
                    [source],
                    TensorType(i64, source.type.get_shape()),
                    lambda args, emit: decode(args[0], emit)[0],
                    operations,
                )
            bounds[packed] = bound
            operand_bounds.append(bound)
            empty = tensor.EmptyOp([], TensorType(i64, shape))
            block = Block(arg_types=[i64, i64])
            block.add_op(L.YieldOp(block.args[0]))
            broadcast = L.GenericOp(
                inputs=[packed],
                outputs=[empty.tensor],
                body=Region(block),
                indexing_maps=ArrayAttr([amap, identity]),
                iterator_types=op.get_iterator_types(),
                result_types=[empty.tensor.type],
            )
            operations.extend([empty, broadcast])
            literal = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [3]), [prod(shape), 1, 1]))
            reshaped = tensor.ReshapeOp(broadcast.results[0], literal.result, TensorType(i64, [prod(shape), 1, 1]))
            operations.extend([literal, reshaped])
            bounds[broadcast.results[0]] = bounds[reshaped.result] = bound
            expanded.append(reshaped.result)
        constant = arith.ConstantOp(IntegerAttr(0, i64))
        seed = tensor.SplatOp(constant.result, [], TensorType(i64, [prod(shape), 1, 1]))
        block = Block(arg_types=[i64, i64, i64])
        product = arith.MuliOp(block.args[0], block.args[1])
        added = arith.AddiOp(product.result, block.args[2])
        block.add_ops([product, added, L.YieldOp(added.result)])
        product_maps = ArrayAttr(
            [
                AffineMapAttr(AffineMap(4, 0, tuple(AffineDimExpr(i) for i in axes)))
                for axes in [(0, 1, 3), (0, 3, 2), (0, 1, 2)]
            ]
        )
        integer = L.GenericOp(
            inputs=expanded,
            outputs=[seed.result],
            body=Region(block),
            indexing_maps=product_maps,
            iterator_types=ArrayAttr(
                [L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 3 + [L.IteratorTypeAttr(L.IteratorType.REDUCTION)]
            ),
            result_types=[seed.result.type],
        )
        operations.extend([constant, seed, integer])
        bounds[seed.result] = 0
        product_bound = prod(operand_bounds)
        bounds[integer.results[0]] = product_bound
        literal = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), list(shape)))
        reshaped = tensor.ReshapeOp(integer.results[0], literal.result, TensorType(i64, shape))
        operations.extend([literal, reshaped])
        bounds[reshaped.result] = product_bound
        empty = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[i64, *[v.type.element_type for v in op.inputs], op.results[0].type.element_type])

        def emit(child):
            block.add_op(child)
            return child.results[0]

        a, b = [decode(v, emit) for v in block.args[1:3]]
        exponent = emit(arith.AddiOp(a[1], b[1]))
        negative_zero = emit(arith.XOrIOp(a[2], b[2]))
        rounded = encode(block.args[0], exponent, op.results[0].type.element_type, emit, zero_negative=negative_zero)
        rounded = overlay(op, list(block.args[1:3]), rounded, emit)
        if token == "bf16" and "merlin.cpu_storage_profile" in op.attributes:
            import json

            from merlin.llvmlower.ieee_pointwise import canonical_bf16_storage

            rounded = canonical_bf16_storage(
                rounded, json.loads(op.attributes["merlin.cpu_storage_profile"].data), emit
            )
        block.add_op(L.YieldOp(rounded))
        readout = L.GenericOp(
            inputs=[reshaped.result, *op.inputs],
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, *maps]),
            iterator_types=op.get_iterator_types(),
            result_types=[empty.tensor.type],
        )
        readout.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        operations.extend([empty, readout])
        Rewriter.replace_op(op, operations, [readout.results[0]])
        proofs.append(
            dict(
                form="typed_ieee_significand_product",
                eligibility="at_least_one_compile_time_operand",
                dtype=token,
                significand_absolute_bound=magnitude,
                product_absolute_bound=product_bound,
                exponent="runtime_ieee_exponent_sum",
                source_data_arithmetic=True,
                rounding="single_integer_RNE_with_gradual_underflow_and_overflow",
            )
        )
    module.verify()
    return bounds, proofs
