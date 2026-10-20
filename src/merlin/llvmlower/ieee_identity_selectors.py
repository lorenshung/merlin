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
