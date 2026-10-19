"""Scalar dataflow patterns excluded from native primitive construction.

Follow typed tensor payloads across region boundaries; addresses and unchanged
literal compare/select clamps are not predicate implementations. No source
operator names or provenance labels participate in classification.
"""

CASTS = {
    "arith.extf",
    "arith.truncf",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.sitofp",
    "arith.uitofp",
    "tensor.cast",
    "tensor.reshape",
    "tensor.splat",
}
ADDITIONS = {"arith.addi", "arith.addf", "arith.subi", "arith.subf"}
EXTREMA = {
    "arith.maximumf",
    "arith.minimumf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minsi",
    "arith.minui",
}
ROUNDING = {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}


def dataflow(root, memo, expand, combine=any):
    """Evaluate a scalar SSA DAG iteratively, including deeply split programs."""
    stack, active, dependencies = [root], set(), {}
    while stack:
        value = stack[-1]
        if value in memo:
            stack.pop()
            active.discard(value)
            continue
        if value not in dependencies:
            dependencies[value] = expand(value)
        children = dependencies[value]
        if isinstance(children, bool):
            memo[value] = children
            continue
        pending = next((v for v in children if v not in memo), None)
        if pending is None:
            memo[value] = combine(memo[v] for v in children)
            continue
        active.add(value)
        if pending in active:
            raise ValueError("cyclic payload dataflow has no native construction proof")
        stack.append(pending)
    return memo[root]


class Payloads:
    """Resolve scalar payload definitions, without expanding shared DAGs."""

    def __init__(self, module):
        self.operations = set(module.walk())
        self._resolve_memo = {}
        self._contraction_memo = {}
        self._readout_memo = {}

    def resolve(self, value):
        path = []
        while value not in self._resolve_memo:
            path.append(value)
            child = self.alias(value)
            if child is None:
                break
            value = child
        root = self._resolve_memo.get(value, value)
        self._resolve_memo.update((v, root) for v in path)
        return root

    def alias(self, value):
        owner = value.owner
        name = getattr(owner, "name", None)
        if name in CASTS:
            return owner.operands[0]
        if name == "linalg.generic":
            return owner.body.block.last_op.operands[value.index]
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                return parent.inputs[value.index]
        return None

    def operation(self, value):
        return self.resolve(value).owner

    def constant(self, value):
        op = self.operation(value)
        if getattr(op, "name", None) != "arith.constant":
            return None
        attr = op.properties.get("value", op.attributes.get("value"))
        if hasattr(attr, "value"):
            return attr.value.data
        return None

    def clamp(self, value):
        op = self.operation(value)
        if getattr(op, "name", None) in EXTREMA:
            return op
        if getattr(op, "name", None) != "arith.select":
            return None
        compare = self.operation(op.operands[0])
        if getattr(compare, "name", None) not in {"arith.cmpi", "arith.cmpf"}:
            return None
        if {self.resolve(v) for v in compare.operands} == {self.resolve(v) for v in op.operands[1:]}:
            return op
        return None

    def contraction_dependent(self, value):
        return dataflow(value, self._contraction_memo, self.contraction_dependencies)

    def contraction_dependencies(self, value):
        owner = value.owner
        name = getattr(owner, "name", None)
        if name == "linalg.generic":
            if any(it.data.value == "reduction" for it in owner.get_iterator_types()):
                return True
            return (owner.body.block.last_op.operands[value.index],)
        if name in {"linalg.matmul", "linalg.batch_matmul", "linalg.reduce", "scf.for"}:
            return True
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                return (parent.inputs[value.index],)
            return False
        return tuple(getattr(owner, "operands", ()))

    def reduction_product(self, op):
        """Prove this product contributes to a local loop-carried reduction."""

        def depends(value, target):
            def expand(v):
                return True if v is target else tuple(getattr(v.owner, "operands", ()))

            return dataflow(value, {}, expand)

        parent = op.parent_op()
        while parent is not None:
            if parent.name == "scf.for":
                block = parent.body.block
                for carried, yielded in zip(block.args[1:], block.last_op.operands, strict=True):
                    if depends(yielded, op.results[0]) and depends(yielded, carried):
                        return True
            parent = parent.parent_op()
        return False

    def readout(self, value):
        """Does this observation reach a result through storage conversions only?

        Balanced digit extraction feeds subsequent arithmetic producers. A
        rounding readout publishes the narrowed value itself. Tensor shapes
        and region wrappers do not distinguish those mathematical roles.
        """
        return dataflow(value, self._readout_memo, self.readout_dependencies)

    def readout_dependencies(self, value):
        children = []
        for use in value.uses:
            consumer = use.operation
            if consumer not in self.operations:
                continue
            if consumer.name == "func.return":
                return True
            movement = {
                "tensor.extract",
                "tensor.extract_slice",
                "tensor.insert_slice",
                "tensor.collapse_shape",
                "tensor.expand_shape",
                "tensor.concat",
            }
            boolean_chain = consumer.name in {"arith.ori", "arith.andi", "arith.xori"} and str(value.type) == "i1"
            if consumer.name in CASTS | movement | {"arith.fptosi", "arith.fptoui"} or boolean_chain:
                children.extend(consumer.results)
            if consumer.name == "arith.select" and use.index == 0:
                if all(self.constant(v) is not None for v in consumer.operands[1:]):
                    children.extend(consumer.results)
            if consumer.name == "scf.yield":
                parent = consumer.parent_op()
                if use.index < len(parent.results):
                    children.append(parent.results[use.index])
            if consumer.name == "linalg.yield":
                parent = consumer.parent_op()
                if parent.name == "linalg.generic" and use.index < len(parent.results):
                    children.append(parent.results[use.index])
            if consumer.name == "linalg.generic" and use.index < len(consumer.inputs):
                children.append(consumer.body.block.args[use.index])
        return tuple(children)

    def patterns(self, op):
        reasons = []
        name = op.name
        # Boolean union has no admitted native arithmetic primitive. An add and
        # truth test is an identity even when hidden inside a reduction.
        if name == "arith.ori" and str(op.results[0].type) == "i1" and self.readout(op.results[0]):
            parent = op.parent_op()
            if (
                parent is not None
                and parent.name == "linalg.generic"
                and any(str(v.type.element_type) == "i1" for v in parent.results)
            ):
                reasons.append("multi_op_identity")
        if name in {"arith.cmpi", "arith.cmpf"}:
            if self.readout(op.results[0]) and any(self.clamp(v) is not None for v in op.operands):
                reasons.append("saturation_comparators")
            # Widened Boolean addition followed by a truth predicate, including
            # separated host/device generics. Integer RNE's internal predicates
            # operate on integer payloads rather than Boolean source leaves.
            for operand in op.operands:
                producer = self.operation(operand)
                if getattr(producer, "name", None) in ADDITIONS and all(
                    str(self.resolve(v).type) in {"i1"}
                    or str(getattr(self.resolve(v).type, "element_type", "")) == "i1"
                    for v in producer.operands
                ):
                    reasons.append("multi_op_identity")
        # Extrema reconstructed as rectifier(x-y)+y (or the subtract variant).
        if name in ADDITIONS:
            for operand in op.operands:
                clamp = self.clamp(operand)
                if clamp is None:
                    continue
                for candidate in clamp.operands:
                    difference = self.operation(candidate)
                    if getattr(difference, "name", None) not in {"arith.subf", "arith.subi"}:
                        continue
                    if any(
                        self.resolve(v) == self.resolve(other)
                        for v in difference.operands
                        for other in op.operands
                        if other is not operand
                    ):
                        reasons.append("multi_op_identity")
        # Bias followed by a rounding/narrowing observation. The exact IEEE
        # encoder uses quotient/remainder and a conditional ties-even increment,
        # rather than an unconditional literal offset in a floating readout.
        if name in ROUNDING | {
            "arith.fptosi",
            "arith.fptoui",
            "arith.shrsi",
            "arith.shrui",
            "arith.divsi",
            "arith.divui",
        }:
            producer = self.operation(op.operands[0])
            if getattr(producer, "name", None) in ADDITIONS:
                offsets = [self.constant(v) for v in producer.operands]
                if any(v is not None and v != 0 for v in offsets):
                    floating = name in ROUNDING | {"arith.fptosi", "arith.fptoui"}
                    if floating or self.readout(op.results[0]):
                        reasons.append("readout_offset_rounding")
        # Floating magic-bias rounding: (x + C) - C, C nonzero.
        if name == "arith.subf":
            addition = self.operation(op.operands[0])
            offset = self.constant(op.operands[1])
            if offset is not None and offset != 0 and getattr(addition, "name", None) == "arith.addf":
                if any(self.constant(v) == offset for v in addition.operands):
                    reasons.append("readout_offset_rounding")
        return sorted(set(reasons))
