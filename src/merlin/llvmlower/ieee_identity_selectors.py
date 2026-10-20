"""Keep the quieting arithmetic branch when an ordered selector is redundant."""


def fold_ieee_identity_selectors(module):
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    payloads = LiteralPayloads()

    def parallel(op):
        if getattr(op, "name", None) != "linalg.generic" or len(op.results) != 1:
            return False
        identity = AffineMap.identity(len(op.results[0].type.get_shape()))
        return all(it.data.value == "parallel" for it in op.get_iterator_types()) and all(
            m.data == identity for m in op.get_indexing_maps()
        )

    def product_is_identity(value, data):
        product = value.owner
        if not parallel(product) or value.type != data.type:
            return False
        body = list(product.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.mulf" or body[1].name != "linalg.yield":
            return False
        if body[0].fastmath.data or list(body[1].operands) != list(body[0].results):
            return False
        operands = []
        for operand in body[0].operands:
            if operand not in product.body.block.args[: len(product.inputs)]:
                return False
            operands.append(product.inputs[operand.index])
        return any(operands[i] is data and _constant_of(operands[1 - i]) == 1.0 for i in (0, 1))

    count = 0
    for op in list(module.walk()):
        if not parallel(op) or len(op.inputs) != 3 or any(payloads.classify(v) & UNKNOWN for v in op.inputs):
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.select" or body[1].name != "linalg.yield":
            continue
        if list(body[0].operands) != list(op.body.block.args[:3]) or list(body[1].operands) != list(body[0].results):
            continue
        predicate = op.inputs[0].owner
        if not parallel(predicate):
            continue
        comparison = list(predicate.body.block.ops)
        if len(comparison) != 2 or comparison[0].name != "arith.cmpf" or comparison[0].fastmath.data:
            continue
        if list(comparison[1].operands) != list(comparison[0].results):
            continue
        if list(comparison[0].operands) != list(predicate.body.block.args[:2]):
            continue
        if len(predicate.inputs) != 2:
            continue
        p = comparison[0].predicate.value.data
        # On NaN retain the original multiplication, never replace quieting
        # arithmetic by the source word. On all non-NaNs x*1 is exactly x.
        for data_slot in (1, 2):
            data, product = op.inputs[data_slot], op.inputs[3 - data_slot]
            unordered_selects_product = p in (range(1, 8) if data_slot == 1 else range(8, 15))
            if not unordered_selects_product or not product_is_identity(product, data):
                continue
            if not any(
                predicate.inputs[i] is data and _constant_of(predicate.inputs[1 - i]) is not None for i in (0, 1)
            ):
                continue
            Rewriter.replace_op(op, [], [product])
            count += 1
            break
    if count:
        from merlin.llvmlower.entry_weight_projection import _pure_dce

        for function in module.walk():
            if function.name == "func.func":
                _pure_dce(function)
    if count:
        module.verify()
    return count


def fold_zero_sign_selectors(module):
    """select(x == 0, copysign(0,x), x) preserves every IEEE storage word."""
    from xdsl.dialects import arith
    from xdsl.ir import BlockArgument
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    def literal(v):
        if isinstance(v, BlockArgument):
            parent = v.owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and v.index < len(parent.inputs):
                v = parent.inputs[v.index]
        return _constant_of(v)

    count = 0
    for op in list(module.walk()):
        if op.name != "arith.select" or str(op.results[0].type) not in FORMAT:
            continue
        predicate = op.operands[0].owner
        if getattr(predicate, "name", None) != "arith.cmpf" or predicate.fastmath.data:
            continue
        if predicate.predicate.value.data not in {1, 13}:
            continue
        zero_slot = 1 if predicate.predicate.value.data == 1 else 2
        data = op.operands[3 - zero_slot]
        if not any(predicate.operands[i] is data and literal(predicate.operands[1 - i]) == 0 for i in (0, 1)):
            continue
        cast = op.operands[zero_slot].owner
        if getattr(cast, "name", None) != "arith.bitcast":
            continue
        mask = cast.operands[0].owner
        if getattr(mask, "name", None) != "arith.andi":
            continue
        width = sum(FORMAT[str(data.type)][:2]) + 1
        for slot in (0, 1):
            raw = mask.operands[slot].owner
            bits = literal(mask.operands[1 - slot])
            if (
                getattr(raw, "name", None) == "arith.bitcast"
                and raw.operands[0] is data
                and raw.results[0].type == mask.results[0].type
                and getattr(getattr(raw.results[0].type, "width", None), "data", None) == width
                and isinstance(bits, int)
                and bits % (1 << width) == 1 << (width - 1)
            ):
                block = op.parent_block()
                Rewriter.replace_op(op, [], [data])
                # Only remove the proved, now-dead scalar wiring. Unknown
                # effects and other region operations retain their identity.
                for dead in (cast, mask, raw, predicate):
                    if all(not value.uses for value in dead.results):
                        Rewriter.erase_op(dead)
                for dead in list(block.ops):
                    if isinstance(dead, arith.ConstantOp) and all(not value.uses for value in dead.results):
                        Rewriter.erase_op(dead)
                count += 1
                break
    return count
