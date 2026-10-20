"""An iteration dimension of extent one has a compile-time zero coordinate."""


def fold_static_unit_indices(module):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType
    from xdsl.rewriter import Rewriter

    changed = 0
    for op in list(module.walk()):
        if op.name == "linalg.index":
            parent = op.parent_op()
            if getattr(parent, "name", None) != "linalg.generic":
                continue
            try:
                ranges = parent.get_static_loop_ranges()
            except (ValueError, AttributeError):
                continue
            axis = op.dim.value.data
            if not 0 <= axis < len(ranges) or ranges[axis] != 1:
                continue
            constant = arith.ConstantOp(IntegerAttr(0, IndexType()))
        elif op.name in {"arith.index_cast", "arith.index_castui"}:
            source = op.operands[0].owner
            attr = source.properties.get("value") if getattr(source, "name", None) == "arith.constant" else None
            if (
                not isinstance(attr, IntegerAttr)
                or attr.value.data != 0
                or not isinstance(op.results[0].type, IntegerType)
            ):
                continue
            constant = arith.ConstantOp(IntegerAttr(0, op.results[0].type))
        else:
            continue
        Rewriter.replace_op(op, [constant], [constant.result])
        changed += 1
    if changed:
        from xdsl.transforms.dead_code_elimination import region_dce

        from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals

        region_dce(module.body)
        changed += fold_uniform_numeric_literals(module)
    return changed
