"""Native sign and closed compare/select epilogues over exact integer words.

Every payload bit is preserved, including signed zeros and NaN payloads. The
linear data term is an exact byte component producer; the sign-bit offset is
explicit modular host recombination. No floating conversion is involved.
"""


def rewrite_native_sign_components(
    module, *, canonical_bf16_nan=False, defer_floating_negation=False, cpu_profiles=None
):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, IntegerType, TensorType, UnitAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    widths = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
    bounds = {}
    proofs = []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = list(op.body.block.ops)
        argument = op.body.block.args[0]
        negate = len(body) == 2 and body[0].name == "arith.negf" and body[0].operands[0] is argument
        literals = set()
        literal_casts = {"arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf"}
        for child in body[:-3]:
            if child.name == "arith.constant" or (
                child.name in literal_casts and all(value in literals for value in child.operands)
            ):
                literals.update(child.results)
            else:
                break
        clamp = False
        if len(body) >= 4:
            compare, select, terminator = body[-3:]
            choices = set(compare.operands)
            clamp = (
                sum(len(child.results) for child in body[:-3]) == len(literals)
                and compare.name in {"arith.cmpi", "arith.cmpf"}
                and argument in choices
                and len(choices) == 2
                and (choices - {argument}) <= literals
                and select.name == "arith.select"
                and select.operands[0] is compare.results[0]
                and set(select.operands[1:]) == choices
                and terminator.name == "linalg.yield"
                and list(terminator.operands) == list(select.results)
            )
        if not (negate or clamp):
            continue
        source = op.inputs[0]
        bits = widths.get(str(source.type.element_type))
        floating = bits is not None
        if negate and floating and defer_floating_negation:
            # Keep the finite dyadic grid visible to following arithmetic.
            # The floating component pass lowers negation with its zero sign.
            continue
        if isinstance(source.type.element_type, IntegerType) and source.type.element_type.width.data in {8, 16, 32, 64}:
            bits = source.type.element_type.width.data
        if bits is None or source.type.element_type != op.results[0].type.element_type:
            continue
        cpu_profile = (
            (cpu_profiles or {}).get(source) if floating and str(source.type.element_type) in {"f16", "bf16"} else None
        )
        if floating and str(source.type.element_type) in {"f16", "bf16"} and cpu_profiles is not None:
            if cpu_profile is None:
                continue
        element = IntegerType(bits)
        shape = source.type.get_shape() if clamp else op.results[0].type.get_shape()
        identity = AffineMapAttr(AffineMap.identity(len(source.type.get_shape())))
        iterators = ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(source.type.get_shape()))
        empty = tensor.EmptyOp([], TensorType(element, source.type.get_shape()))
        block = Block(arg_types=[source.type.element_type, element])
        if floating:
            cast = arith.BitcastOp(block.args[0], element)
            block.add_ops([cast, L.YieldOp(cast.result)])
        else:
            block.add_op(L.YieldOp(block.args[0]))
        packed = L.GenericOp(
            inputs=[source],
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=iterators,
            result_types=[empty.tensor.type],
        )
        bounds[packed.results[0]] = 1 << (bits - 1)
        destination = tensor.EmptyOp([], TensorType(element, shape))
        block = Block(arg_types=[element, element])
        offset = arith.ConstantOp(IntegerAttr(-(1 << (bits - 1)) if negate else 0, element))
        addition = arith.AddiOp(block.args[0], offset.result)
        block.add_ops([offset, addition, L.YieldOp(addition.result)])
        changed = L.GenericOp(
            inputs=[packed.results[0]],
            outputs=[destination.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, identity]) if clamp else op.get_indexing_maps(),
            iterator_types=iterators if clamp else op.get_iterator_types(),
            result_types=[destination.tensor.type],
        )
        bounds[changed.results[0]] = 1 << (bits - 1)
        out = tensor.EmptyOp([], TensorType(source.type.element_type, shape))
        block = Block(arg_types=[element, source.type.element_type])
        if floating:
            raw = block.args[0]
            if cpu_profile is not None and str(source.type.element_type) in {"f16", "bf16"}:
                from merlin.llvmlower.native_cpu_layout import vector_predicate

                def put(child):
                    block.add_op(child)
                    return child.results[0]

                def c(v):
                    return put(arith.ConstantOp(IntegerAttr(v, element)))

                fraction = 7 if str(source.type.element_type) == "bf16" else 10
                infinity = 0x7F80 if fraction == 7 else 0x7C00
                magnitude = put(arith.AndIOp(raw, c(0x7FFF)))
                nan = put(arith.CmpiOp(magnitude, c(infinity), "ugt"))
                vector = vector_predicate(cpu_profile, put)
                corrected = (
                    c(0x7FC0 if negate else 0xFFFF) if fraction == 7 else put(arith.OrIOp(raw, c(1 << (fraction - 1))))
                )
                # Sign change's scalar C10 path converts; threshold's vector
                # path converts. The other path retains the original word.
                if negate:
                    from xdsl.dialects.builtin import i1

                    vector = put(arith.XOrIOp(vector, put(arith.ConstantOp(IntegerAttr(1, i1)))))
                flag = put(arith.AndIOp(nan, vector))
                raw = put(arith.SelectOp(flag, corrected, raw))
            if negate and canonical_bf16_nan and str(source.type.element_type) == "bf16":
                mask = arith.ConstantOp(IntegerAttr(0x7FFF, element))
                infinity = arith.ConstantOp(IntegerAttr(0x7F80, element))
                canonical = arith.ConstantOp(IntegerAttr(0x7FC0, element))
                magnitude = arith.AndIOp(raw, mask.result)
                nan = arith.CmpiOp(magnitude.result, infinity.result, "ugt")
                selected = arith.SelectOp(nan.result, canonical.result, raw)
                block.add_ops([mask, infinity, canonical, magnitude, nan, selected])
                raw = selected.result
            cast = arith.BitcastOp(raw, source.type.element_type)
            block.add_ops([cast, L.YieldOp(cast.result)])
        else:
            block.add_op(L.YieldOp(block.args[0]))
        out_map = AffineMapAttr(AffineMap.identity(len(shape)))
        result = L.GenericOp(
            inputs=[changed.results[0]],
            outputs=[out.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([out_map, out_map]),
            iterator_types=iterators if clamp else op.get_iterator_types(),
            result_types=[out.tensor.type],
        )
        operations = [empty, packed, destination, changed, out, result]
        if clamp:
            retained = op.clone(value_mapper={source: result.results[0]})
            operations.append(retained)
            result = retained
        # The closed sign/clamp proof licenses this host readout, including
        # BF16 NaN canonicalization and the original literal compare/select.
        result.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        Rewriter.replace_op(op, operations, [result.results[0]])
        proofs.append(
            dict(
                source="arith.negf" if negate else "closed_constant_compare_select",
                representation="ieee_storage_bits" if floating else "native_signed_integer",
                width=bits,
                arithmetic="sign_bit_offset_modulo_storage_width" if negate else "source_compare_select_retained_host",
                nan_payload=(
                    "exhaustive_CPU_scalar_vector_storage_forms"
                    if cpu_profile is not None
                    else "canonical_positive_quiet"
                    if negate and canonical_bf16_nan and str(source.type.element_type) == "bf16"
                    else "preserved"
                    if negate
                    else "source_predicate_semantics"
                ),
                **({"cpu_storage_profile": cpu_profile} if cpu_profile is not None else {}),
            )
        )
    module.verify()
    return bounds, proofs
