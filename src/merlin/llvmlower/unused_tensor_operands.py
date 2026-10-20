"""Remove tensor operands whose scalar argument is unused by a generic region."""


def prune_unused_tensor_inputs(module):
    from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, i32
    from xdsl.rewriter import Rewriter

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.body.blocks) != 1:
            continue
        n = len(op.inputs)
        if len(op.body.block.args) != n + len(op.outputs):
            continue
        unused = [i for i in range(n) if not op.body.block.args[i].uses]
        if not unused:
            continue
        kept = [i for i in range(n) if i not in unused]
        clone = op.clone()
        for index in reversed(unused):
            clone.body.block.erase_arg(clone.body.block.args[index])
        clone.operands = [op.inputs[i] for i in kept] + list(op.outputs)
        maps = list(op.get_indexing_maps())
        clone.properties["indexing_maps"] = ArrayAttr([maps[i] for i in kept] + maps[n:])
        clone.properties["operandSegmentSizes"] = DenseArrayBase.from_list(i32, [len(kept), len(op.outputs)])
        Rewriter.replace_op(op, clone)
        changed += 1
    module.verify()
    return changed
